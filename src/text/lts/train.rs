//! Training of the letter-to-sound trees (spec §7.5 steps 1–5). Deterministic for a given input.

use std::collections::BTreeMap;

use crate::text::lts::align::{ALLOWABLES, Table, align};
use crate::text::lts::cart::{
    ANY_CONSONANT, ANY_VOWEL, CONTEXT, Cart, Node, WORD_FEATURES, answers, letter_at,
};

/// One dictionary entry: lowercase a–z word and its phones with stress (`AE1`).
pub struct Entry {
    pub word: String,
    pub phones: Vec<String>,
}

/// Parses `cmudict.dict`: first variant only, no digits, apostrophes or periods in the word.
pub fn parse_dict(text: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split(" #").next().unwrap_or("");
        let mut it = line.split_whitespace();
        let Some(word) = it.next() else { continue };
        if !word.bytes().all(|b| b.is_ascii_lowercase()) || word.len() < 2 {
            continue;
        }
        let phones: Vec<String> = it.map(str::to_string).collect();
        if !phones.is_empty() {
            out.push(Entry {
                word: word.to_string(),
                phones,
            });
        }
    }
    out
}

fn strip(phone: &str) -> &str {
    phone.trim_end_matches(|c: char| c.is_ascii_digit())
}

/// A word aligned to labels that carry stress.
pub struct Aligned {
    pub word: Vec<u8>,
    pub labels: Vec<String>,
}

pub struct AlignmentResult {
    pub aligned: Vec<Aligned>,
    pub failed: usize,
}

/// Iterative alignment: uniform start, then `rounds` of re-estimation.
pub fn align_all(entries: &[Entry], rounds: usize) -> AlignmentResult {
    let mut table = Table::from_allowables(ALLOWABLES);
    let bare: Vec<Vec<&str>> = entries
        .iter()
        .map(|e| e.phones.iter().map(|p| strip(p)).collect())
        .collect();
    let mut last: Vec<Option<Vec<String>>> = Vec::new();
    for _ in 0..rounds.max(1) {
        last = entries
            .iter()
            .zip(&bare)
            .map(|(e, p)| align(&table, e.word.as_bytes(), p))
            .collect();
        let pairs: Vec<(Vec<u8>, Vec<String>)> = entries
            .iter()
            .zip(&last)
            .filter_map(|(e, l)| l.clone().map(|l| (e.word.as_bytes().to_vec(), l)))
            .collect();
        table.reestimate(&pairs);
    }
    let mut aligned = Vec::new();
    let mut failed = 0;
    for (e, labels) in entries.iter().zip(last) {
        let Some(labels) = labels else {
            failed += 1;
            continue;
        };
        // Put the stress digits back: walk the phones in order.
        let mut next = e.phones.iter();
        let with_stress: Vec<String> = labels
            .iter()
            .map(|l| {
                if l == "_" {
                    return l.clone();
                }
                l.split('-')
                    .map(|_| next.next().cloned().unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join("-")
            })
            .collect();
        aligned.push(Aligned {
            word: e.word.as_bytes().to_vec(),
            labels: with_stress,
        });
    }
    AlignmentResult { aligned, failed }
}

/// Letter window plus the word-level features.
const WIDTH: usize = 2 * CONTEXT as usize + 1 + WORD_FEATURES.len();

/// Feature code of slot `k`.
fn offset_of(k: usize) -> i8 {
    let window = 2 * CONTEXT as usize + 1;
    if k < window {
        k as i8 - CONTEXT
    } else {
        WORD_FEATURES[k - window]
    }
}

struct Example {
    /// Letters at offsets −CONTEXT…+CONTEXT (index = offset + CONTEXT).
    ctx: [u8; WIDTH],
    label: u32,
}

fn entropy_sum(counts: &BTreeMap<u32, u32>, total: u32) -> f64 {
    // total × entropy, in nats.
    let t = total as f64;
    counts
        .values()
        .filter(|c| **c > 0)
        .map(|c| -(*c as f64) * ((*c as f64) / t).ln())
        .sum()
}

fn grow(examples: &mut [Example], stop: usize, nodes: &mut Vec<Node>) -> u32 {
    let index = nodes.len() as u32;
    let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
    for e in examples.iter() {
        *counts.entry(e.label).or_default() += 1;
    }
    // Majority label; ties go to the smaller label index for determinism.
    let majority = counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(l, _)| *l)
        .unwrap_or(0);
    if examples.len() < stop || counts.len() <= 1 {
        nodes.push(Node::leaf(majority));
        return index;
    }
    let total = examples.len() as u32;
    let base = entropy_sum(&counts, total);
    // For every question (offset, letter): class counts of the "yes" side.
    let mut best: Option<(f64, usize, u8)> = None;
    for k in 0..WIDTH {
        if k == CONTEXT as usize {
            continue;
        }
        let mut by_letter: BTreeMap<u8, BTreeMap<u32, u32>> = BTreeMap::new();
        for e in examples.iter() {
            *by_letter
                .entry(e.ctx[k])
                .or_default()
                .entry(e.label)
                .or_default() += 1;
        }
        if by_letter.len() < 2 {
            continue;
        }
        // Class questions: the union of the vowel letters, and of the consonant letters.
        for class in [ANY_VOWEL, ANY_CONSONANT] {
            let mut merged: BTreeMap<u32, u32> = BTreeMap::new();
            for (letter, m) in &by_letter {
                if answers(*letter, class) {
                    for (l, c) in m {
                        *merged.entry(*l).or_default() += c;
                    }
                }
            }
            if !merged.is_empty() {
                by_letter.insert(class, merged);
            }
        }
        for (letter, yes) in &by_letter {
            let n_yes: u32 = yes.values().sum();
            let n_no = total - n_yes;
            if n_yes == 0 || n_no == 0 {
                continue;
            }
            let mut no = counts.clone();
            for (l, c) in yes {
                if let Some(x) = no.get_mut(l) {
                    *x -= c;
                }
            }
            let gain = base - entropy_sum(yes, n_yes) - entropy_sum(&no, n_no);
            let better = match best {
                None => gain > 1e-9,
                Some((g, bk, bl)) => {
                    gain > g + 1e-9 || ((gain - g).abs() <= 1e-9 && (k, *letter) < (bk, bl))
                }
            };
            if better {
                best = Some((gain, k, *letter));
            }
        }
    }
    let Some((_, k, letter)) = best else {
        nodes.push(Node::leaf(majority));
        return index;
    };
    nodes.push(Node {
        offset: offset_of(k),
        letter,
        yes: 0,
        no: 0,
    });
    // Partition in place: "yes" examples first.
    let mut split = 0;
    for i in 0..examples.len() {
        if answers(examples[i].ctx[k], letter) {
            examples.swap(i, split);
            split += 1;
        }
    }
    let (yes_part, no_part) = examples.split_at_mut(split);
    let yes = grow(yes_part, stop, nodes);
    let no = grow(no_part, stop, nodes);
    nodes[index as usize].yes = yes;
    nodes[index as usize].no = no;
    index
}

/// Grows one tree per letter. `stop`: a node with fewer examples becomes a leaf.
pub fn train(aligned: &[&Aligned], stop: usize) -> Cart {
    let mut label_ids: BTreeMap<&str, u32> = BTreeMap::new();
    for a in aligned {
        for l in &a.labels {
            let n = label_ids.len() as u32;
            label_ids.entry(l.as_str()).or_insert(n);
        }
    }
    // Stable ids: sorted by label text.
    let labels: Vec<String> = label_ids.keys().map(|k| k.to_string()).collect();
    let id: BTreeMap<&str, u32> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i as u32))
        .collect();
    let mut per_letter: Vec<Vec<Example>> = (0..26).map(|_| Vec::new()).collect();
    for a in aligned {
        for (i, l) in a.labels.iter().enumerate() {
            let mut ctx = [b'#'; WIDTH];
            for (k, c) in ctx.iter_mut().enumerate() {
                *c = letter_at(&a.word, i, offset_of(k));
            }
            per_letter[(a.word[i] - b'a') as usize].push(Example {
                ctx,
                label: id[l.as_str()],
            });
        }
    }
    let mut trees = Vec::with_capacity(26);
    for mut examples in per_letter {
        let mut nodes = Vec::new();
        if !examples.is_empty() {
            grow(&mut examples, stop, &mut nodes);
        }
        trees.push(nodes);
    }
    Cart { trees, labels }
}

#[derive(Debug, Default)]
pub struct Score {
    pub words: usize,
    pub words_correct: usize,
    /// Words correct when stress is ignored.
    pub words_correct_no_stress: usize,
    pub letters: usize,
    pub letters_correct: usize,
}

/// Letters and words correct on aligned words the tree has not seen.
pub fn score(cart: &Cart, held_out: &[&Aligned]) -> Score {
    let mut s = Score::default();
    for a in held_out {
        let mut all = true;
        let mut all_bare = true;
        for (i, gold) in a.labels.iter().enumerate() {
            let got = cart.label(&a.word, i);
            s.letters += 1;
            if got == gold {
                s.letters_correct += 1;
            } else {
                all = false;
                let bare = |l: &str| {
                    l.chars()
                        .filter(|c| !c.is_ascii_digit())
                        .collect::<String>()
                };
                if bare(got) != bare(gold) {
                    all_bare = false;
                }
            }
        }
        s.words += 1;
        s.words_correct += usize::from(all);
        s.words_correct_no_stress += usize::from(all_bare);
    }
    s
}

/// Every 10th word is held out, before any training.
pub fn split(aligned: &[Aligned]) -> (Vec<&Aligned>, Vec<&Aligned>) {
    let mut train = Vec::new();
    let mut held = Vec::new();
    for (i, a) in aligned.iter().enumerate() {
        if i % 10 == 9 {
            held.push(a)
        } else {
            train.push(a)
        }
    }
    (train, held)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DICT: &str = "cat K AE1 T\nchat CH AE1 T\ncot K AA1 T\nchin CH IH1 N\nbox B AA1 K S\ncity S IH1 T IY0\ncent S EH1 N T\ncell S EH1 L\ncan K AE1 N\ncan(2) K AH0 N\nit's IH1 T S\ncut K AH1 T # comment\nchip CH IH1 P\ncab K AE1 B\ncit S IH1 T\n";

    #[test]
    fn dictionary_parsing() {
        let d = parse_dict(DICT);
        assert!(d.iter().all(|e| e.word != "it's" && !e.word.contains('(')));
        assert_eq!(d.iter().filter(|e| e.word == "can").count(), 1);
        assert_eq!(
            d.iter().find(|e| e.word == "cut").unwrap().phones,
            ["K", "AH1", "T"]
        );
    }

    #[test]
    fn learns_soft_and_hard_c_and_is_deterministic() {
        let entries = parse_dict(DICT);
        let res = align_all(&entries, 4);
        assert_eq!(res.failed, 0);
        let refs: Vec<&Aligned> = res.aligned.iter().collect();
        let cart = train(&refs, 1);
        let say = |w: &str| {
            cart.predict(w)
                .iter()
                .map(|p| p.0.name())
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(say("cat"), "K AE T");
        assert_eq!(say("chat"), "CH AE T");
        assert_eq!(say("cit"), "S IH T");
        assert_eq!(say("cob"), "K AA B");
        assert_eq!(say("chab"), "CH AE B");
        let again = train(&res.aligned.iter().collect::<Vec<_>>(), 1);
        assert_eq!(cart.to_bytes(), again.to_bytes());
        let s = score(&cart, &refs);
        assert_eq!(s.letters, s.letters_correct);
        assert_eq!(s.words, s.words_correct);
    }
}
