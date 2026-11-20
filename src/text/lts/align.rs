//! Letter/phone alignment for training (spec §7.5 step 1).
//!
//! Each letter maps to zero, one or two phones, restricted to a hand-written table of
//! allowable pairs. The best alignment maximizes the product of pair probabilities; the
//! probabilities are re-estimated from the alignments and the process repeats.
//! Ordered maps throughout: sums are taken in a fixed order, so training is reproducible.

use std::collections::BTreeMap;

pub const ALLOWABLES: &str = include_str!("../data/lts-allowables.txt");

/// `P(label | letter)` in log domain; absent pairs are not allowed.
pub struct Table {
    logp: Vec<BTreeMap<String, f32>>,
}

impl Table {
    /// Uniform over the allowable pairs.
    pub fn from_allowables(text: &str) -> Table {
        let mut logp = vec![BTreeMap::new(); 26];
        for line in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let mut it = line.split_whitespace();
            let Some(letter) = it
                .next()
                .and_then(|l| l.bytes().next())
                .filter(|b| b.is_ascii_lowercase())
            else {
                continue;
            };
            let labels: Vec<&str> = it.collect();
            let p = (1.0 / labels.len().max(1) as f32).ln();
            for l in labels {
                logp[(letter - b'a') as usize].insert(l.to_string(), p);
            }
        }
        Table { logp }
    }

    fn get(&self, letter: u8, label: &str) -> Option<f32> {
        self.logp[(letter - b'a') as usize].get(label).copied()
    }

    /// Re-estimates the probabilities from aligned words (labels without stress), with a small floor
    /// so allowable pairs never disappear.
    pub fn reestimate(&mut self, aligned: &[(Vec<u8>, Vec<String>)]) {
        let mut counts: Vec<BTreeMap<String, f32>> = self
            .logp
            .iter()
            .map(|m| m.keys().map(|k| (k.clone(), 0.1f32)).collect())
            .collect();
        for (letters, labels) in aligned {
            for (l, lab) in letters.iter().zip(labels) {
                if let Some(c) = counts[(l - b'a') as usize].get_mut(lab) {
                    *c += 1.0;
                }
            }
        }
        for (i, m) in counts.into_iter().enumerate() {
            let total: f32 = m.values().sum();
            self.logp[i] = m.into_iter().map(|(k, c)| (k, (c / total).ln())).collect();
        }
    }
}

/// Best alignment of `letters` (a–z) to `phones` (no stress): one label per letter.
/// `None` when the word cannot be aligned with the allowable pairs.
pub fn align(table: &Table, letters: &[u8], phones: &[&str]) -> Option<Vec<String>> {
    let (n, m) = (letters.len(), phones.len());
    const NEG: f32 = f32::NEG_INFINITY;
    // best[i][j]: letters[..i] produce phones[..j]. back: how many phones the last letter took.
    let mut best = vec![vec![NEG; m + 1]; n + 1];
    let mut back = vec![vec![0u8; m + 1]; n + 1];
    best[0][0] = 0.0;
    let mut double = String::new();
    for i in 1..=n {
        let letter = letters[i - 1];
        for j in 0..=m {
            let mut top = NEG;
            let mut take = 0u8;
            if best[i - 1][j] > NEG
                && let Some(p) = table.get(letter, "_")
            {
                top = best[i - 1][j] + p;
            }
            if j >= 1
                && best[i - 1][j - 1] > NEG
                && let Some(p) = table.get(letter, phones[j - 1])
                && best[i - 1][j - 1] + p > top
            {
                top = best[i - 1][j - 1] + p;
                take = 1;
            }
            if j >= 2 && best[i - 1][j - 2] > NEG {
                double.clear();
                double.push_str(phones[j - 2]);
                double.push('-');
                double.push_str(phones[j - 1]);
                if let Some(p) = table.get(letter, &double)
                    && best[i - 1][j - 2] + p > top
                {
                    top = best[i - 1][j - 2] + p;
                    take = 2;
                }
            }
            best[i][j] = top;
            back[i][j] = take;
        }
    }
    if best[n][m] == NEG {
        return None;
    }
    let mut labels = vec![String::new(); n];
    let mut j = m;
    for i in (1..=n).rev() {
        labels[i - 1] = match back[i][j] {
            0 => "_".to_string(),
            1 => phones[j - 1].to_string(),
            _ => format!("{}-{}", phones[j - 2], phones[j - 1]),
        };
        j -= back[i][j] as usize;
    }
    Some(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(word: &str, phones: &str) -> Option<String> {
        let t = Table::from_allowables(ALLOWABLES);
        let p: Vec<&str> = phones.split(' ').collect();
        align(&t, word.as_bytes(), &p).map(|l| l.join(" "))
    }

    #[test]
    fn aligns_epsilons_and_doubles() {
        assert_eq!(a("box", "B AA K S").unwrap(), "B AA K-S");
        assert_eq!(a("cute", "K Y UW T").unwrap(), "K Y-UW T _");
        let though = a("though", "DH OW").unwrap();
        assert_eq!(though.split(' ').filter(|l| *l == "_").count(), 4);
        assert!(though.contains("DH") && though.contains("OW"));
        let knight = a("knight", "N AY T").unwrap();
        assert!(knight.starts_with("_ N AY"), "{knight}");
        assert!(knight.ends_with("T"));
        assert_eq!(a("table", "T EY B AH L").unwrap(), "T EY B AH-L _");
        assert_eq!(a("one", "W AH N").unwrap(), "W-AH N _");
    }

    #[test]
    fn impossible_words_fail() {
        assert!(a("cat", "D AO G Z IY").is_none());
        assert!(a("b", "K").is_none());
    }

    #[test]
    fn reestimation_prefers_what_it_saw() {
        let mut t = Table::from_allowables(ALLOWABLES);
        let before = t.get(b'c', "K").unwrap();
        let aligned = vec![
            (
                b"cat".to_vec(),
                vec!["K".to_string(), "AE".to_string(), "T".to_string()]
            );
            50
        ];
        t.reestimate(&aligned);
        assert!(t.get(b'c', "K").unwrap() > before);
        assert!(t.get(b'c', "S").is_some(), "allowable pairs keep a floor");
    }
}
