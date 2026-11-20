//! The built-in exception lexicon, compiled into the binary.

use std::collections::HashMap;

use crate::text::lexicon::{Pron, parse_pron};

pub const SOURCE: &str = include_str!("../data/builtin-lexicon.txt");

/// Returns (words, letter names). Lines that do not parse are skipped; a test asserts none do.
pub fn parse() -> (HashMap<&'static str, Pron>, HashMap<char, Pron>) {
    let mut words = HashMap::with_capacity(700);
    let mut letters = HashMap::with_capacity(26);
    for (word, pron) in entries() {
        match word.strip_prefix('=') {
            Some(l) => {
                if let Some(c) = l.chars().next() {
                    letters.insert(c, pron);
                }
            }
            None => {
                words.entry(word).or_insert(pron);
            }
        }
    }
    (words, letters)
}

fn entries() -> impl Iterator<Item = (&'static str, Pron)> {
    SOURCE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let (word, rest) = l.split_once(' ')?;
            Some((word, parse_pron(rest)?))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_line_parses() {
        let lines = SOURCE
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .count();
        assert_eq!(entries().count(), lines);
        let (words, letters) = parse();
        assert_eq!(letters.len(), 26);
        assert!(words.len() >= 500, "{}", words.len());
        // Every entry has at least one vowel, and at most one primary stress.
        for (w, p) in &words {
            assert!(p.iter().any(|x| x.0.is_vowel()), "{w}");
            assert!(
                p.iter()
                    .filter(|x| x.1 == crate::phone::Stress::Primary)
                    .count()
                    <= 1,
                "{w}"
            );
        }
    }
}
