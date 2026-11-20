//! Letter-to-sound (spec §7.5).

pub mod align;
pub mod cart;
pub mod rules;
pub mod train;

pub use cart::CartLts;
pub use rules::RuleLts;

use crate::phone::{Phone, parse_phone_token};
use crate::traits::LetterToSound;

/// Levenshtein distance between two phone strings.
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, y) in b.iter().enumerate() {
            cur.push(
                (prev[j] + usize::from(x != y))
                    .min(prev[j + 1] + 1)
                    .min(cur[j] + 1),
            );
        }
        prev = cur;
    }
    prev[b.len()]
}

pub struct Evaluation {
    pub words: usize,
    pub words_correct: usize,
    pub ref_phones: usize,
    pub errors: usize,
    /// `(word, reference, hypothesis)` for every wrong word.
    pub misses: Vec<(String, String, String)>,
}

impl Evaluation {
    /// Phoneme error rate: edit distance ÷ reference length, stress ignored.
    pub fn per(&self) -> f32 {
        self.errors as f32 / self.ref_phones.max(1) as f32
    }
}

/// Evaluates on a `word PH PH ...` list (`#` comments).
pub fn evaluate(lts: &dyn LetterToSound, list: &str) -> Evaluation {
    let mut ev = Evaluation {
        words: 0,
        words_correct: 0,
        ref_phones: 0,
        errors: 0,
        misses: Vec::new(),
    };
    for line in list.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(word) = it.next() else { continue };
        let reference: Vec<Phone> = it
            .filter_map(|t| parse_phone_token(t).ok())
            .map(|p| p.0)
            .collect();
        let hyp: Vec<Phone> = lts.pronounce(word).iter().map(|s| s.phone).collect();
        let d = edit_distance(&reference, &hyp);
        ev.words += 1;
        ev.ref_phones += reference.len();
        ev.errors += d;
        if d == 0 {
            ev.words_correct += 1;
        } else {
            let show = |p: &[Phone]| p.iter().map(|p| p.name()).collect::<Vec<_>>().join(" ");
            ev.misses
                .push((word.to_string(), show(&reference), show(&hyp)));
        }
    }
    ev
}
