//! Word error rate after our own normalization (spec §16.5).

use crate::text::EnglishNormalizer;
use crate::text::lts::edit_distance;
use crate::traits::{Normalizer, Token};

/// Lowercase words of a text, numbers and symbols expanded by our normalizer.
pub fn words(text: &str) -> Vec<String> {
    let n = EnglishNormalizer::new(Box::new(|_| true));
    n.normalize(text)
        .into_iter()
        .filter_map(|t| match t {
            Token::Word { text, .. } => Some(text.to_lowercase().replace('\'', "")),
            _ => None,
        })
        .filter(|w| !w.is_empty())
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct Wer {
    pub reference_words: usize,
    pub errors: usize,
    /// `(reference word or "", hypothesis word or "")` for each substitution, deletion, insertion.
    pub confusions: Vec<(String, String)>,
}

impl Wer {
    pub fn rate(&self) -> f32 {
        self.errors as f32 / self.reference_words.max(1) as f32
    }

    pub fn add(&mut self, other: &Wer) {
        self.reference_words += other.reference_words;
        self.errors += other.errors;
        self.confusions.extend(other.confusions.iter().cloned());
    }
}

/// WER of one hypothesis against its reference, with the aligned confusions.
pub fn wer(reference: &str, hypothesis: &str) -> Wer {
    let (r, h) = (words(reference), words(hypothesis));
    // Full DP table for the backtrace.
    let mut d = vec![vec![0usize; h.len() + 1]; r.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=r.len() {
        for j in 1..=h.len() {
            let sub = d[i - 1][j - 1] + usize::from(r[i - 1] != h[j - 1]);
            d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    debug_assert_eq!(d[r.len()][h.len()], edit_distance(&r, &h));
    let mut confusions = Vec::new();
    let (mut i, mut j) = (r.len(), h.len());
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + usize::from(r[i - 1] != h[j - 1]) {
            if r[i - 1] != h[j - 1] {
                confusions.push((r[i - 1].clone(), h[j - 1].clone()));
            }
            i -= 1;
            j -= 1;
        } else if i > 0 && d[i][j] == d[i - 1][j] + 1 {
            confusions.push((r[i - 1].clone(), String::new()));
            i -= 1;
        } else {
            confusions.push((String::new(), h[j - 1].clone()));
            j -= 1;
        }
    }
    confusions.reverse();
    Wer {
        reference_words: r.len(),
        errors: d[r.len()][h.len()],
        confusions,
    }
}

/// Line-by-line WER of two texts.
pub fn wer_lines(reference: &str, hypothesis: &str) -> Wer {
    let mut total = Wer::default();
    let mut hyp = hypothesis.lines();
    for r in reference.lines().filter(|l| !l.trim().is_empty()) {
        total.add(&wer(r, hyp.next().unwrap_or("")));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_errors() {
        assert_eq!(wer("the quick brown fox", "The quick brown fox.").errors, 0);
        let w = wer("the quick brown fox", "the twin grown fox jumps");
        assert_eq!(w.errors, 3);
        assert_eq!(w.reference_words, 4);
        assert!(w.confusions.contains(&("quick".into(), "twin".into())));
        assert_eq!(
            wer("she paid $3 today", "she paid three dollars today").errors,
            0
        );
        assert_eq!(wer("one two", "").errors, 2);
    }
}
