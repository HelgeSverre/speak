//! `RuleLts`: hand-written context rules in the NRL scheme (Elovitz et al. 1976; spec §7.5).
//!
//! Each rule is `left [graphemes] right = phones`. The word is scanned left to right; at each
//! position the first rule whose bracketed text and contexts match is applied.

use std::sync::OnceLock;

use crate::chunk::Segment;
use crate::phone::{Phone, Stress};
use crate::traits::LetterToSound;

const RULES: &str = include_str!("../data/lts-rules.txt");

#[derive(Clone, Copy, Debug, PartialEq)]
enum Sym {
    Lit(u8),
    /// `#` one or more vowels.
    Vowels,
    /// `:` zero or more consonants.
    Consonants,
    /// `^` one consonant.
    Consonant,
    /// `.` a voiced consonant.
    Voiced,
    /// `%` a suffix: e, er, es, ed, ing, ely.
    Suffix,
    /// `+` a front vowel.
    Front,
    /// `$` word boundary.
    Boundary,
}

#[derive(Clone, Debug)]
struct Rule {
    left: Vec<Sym>,
    text: Vec<u8>,
    right: Vec<Sym>,
    phones: Vec<Phone>,
}

pub struct RuleLts {
    by_letter: Vec<Vec<Rule>>,
}

fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

fn is_consonant(c: u8) -> bool {
    c.is_ascii_lowercase() && !is_vowel(c)
}

fn symbols(s: &str) -> Vec<Sym> {
    s.bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .map(|b| match b {
            b'#' => Sym::Vowels,
            b':' => Sym::Consonants,
            b'^' => Sym::Consonant,
            b'.' => Sym::Voiced,
            b'%' => Sym::Suffix,
            b'+' => Sym::Front,
            b'$' => Sym::Boundary,
            other => Sym::Lit(other.to_ascii_lowercase()),
        })
        .collect()
}

/// Parses the rule file. Malformed lines are reported, not skipped silently.
fn parse(text: &str) -> Result<Vec<Vec<Rule>>, String> {
    let mut by_letter: Vec<Vec<Rule>> = vec![Vec::new(); 26];
    for (n, raw) in text.lines().enumerate() {
        // Comments start with `//` (`#` is a context symbol).
        let line = raw.split("//").next().unwrap_or("");
        if line.trim().is_empty() {
            continue;
        }
        let err = |why: &str| format!("lts-rules.txt line {}: {why}: {raw}", n + 1);
        let (lhs, rhs) = line.split_once('=').ok_or_else(|| err("missing ="))?;
        let open = lhs.find('[').ok_or_else(|| err("missing ["))?;
        let close = lhs.find(']').ok_or_else(|| err("missing ]"))?;
        let text: Vec<u8> = lhs[open + 1..close]
            .trim()
            .bytes()
            .map(|b| b.to_ascii_lowercase())
            .collect();
        if text.is_empty() || !text.iter().all(|b| b.is_ascii_lowercase() || *b == b'\'') {
            return Err(err("bad bracket text"));
        }
        let mut phones = Vec::new();
        for tok in rhs.split_whitespace() {
            phones.push(
                Phone::from_name(tok)
                    .filter(|p| *p != Phone::SIL)
                    .ok_or_else(|| err("unknown phone"))?,
            );
        }
        let first = text[0];
        if !first.is_ascii_lowercase() {
            return Err(err("bracket must start with a letter"));
        }
        by_letter[(first - b'a') as usize].push(Rule {
            left: symbols(&lhs[..open]),
            text,
            right: symbols(&lhs[close + 1..]),
            phones,
        });
    }
    Ok(by_letter)
}

const SUFFIXES: [&[u8]; 6] = [b"ing", b"ely", b"er", b"es", b"ed", b"e"];

/// Matches `syms` against `w` starting at `pos`, moving right.
fn match_right(syms: &[Sym], w: &[u8], pos: usize) -> bool {
    let Some((first, rest)) = syms.split_first() else {
        return true;
    };
    let c = w.get(pos).copied();
    match first {
        Sym::Lit(l) => c == Some(*l) && match_right(rest, w, pos + 1),
        Sym::Vowels => {
            let mut p = pos;
            while w.get(p).is_some_and(|c| is_vowel(*c)) {
                p += 1;
                if match_right(rest, w, p) {
                    return true;
                }
            }
            false
        }
        Sym::Consonants => {
            let mut p = pos;
            loop {
                if match_right(rest, w, p) {
                    return true;
                }
                if w.get(p).is_some_and(|c| is_consonant(*c)) {
                    p += 1;
                } else {
                    return false;
                }
            }
        }
        Sym::Consonant => c.is_some_and(is_consonant) && match_right(rest, w, pos + 1),
        Sym::Voiced => {
            c.is_some_and(|c| b"bdvgjlmnrwz".contains(&c)) && match_right(rest, w, pos + 1)
        }
        Sym::Front => {
            c.is_some_and(|c| matches!(c, b'e' | b'i' | b'y')) && match_right(rest, w, pos + 1)
        }
        Sym::Suffix => SUFFIXES.iter().any(|s| {
            w[pos.min(w.len())..].starts_with(s)
                && pos + s.len() == w.len()
                && match_right(rest, w, w.len())
        }),
        Sym::Boundary => pos >= w.len() && match_right(rest, w, pos),
    }
}

/// Matches `syms` (written left to right) against the text ending just before `end`, moving left.
fn match_left(syms: &[Sym], w: &[u8], end: usize) -> bool {
    let Some((last, rest)) = syms.split_last() else {
        return true;
    };
    let c = end.checked_sub(1).map(|i| w[i]);
    match last {
        Sym::Lit(l) => c == Some(*l) && match_left(rest, w, end - 1),
        Sym::Vowels => {
            let mut e = end;
            while e > 0 && is_vowel(w[e - 1]) {
                e -= 1;
                if match_left(rest, w, e) {
                    return true;
                }
            }
            false
        }
        Sym::Consonants => {
            let mut e = end;
            loop {
                if match_left(rest, w, e) {
                    return true;
                }
                if e > 0 && is_consonant(w[e - 1]) {
                    e -= 1;
                } else {
                    return false;
                }
            }
        }
        Sym::Consonant => c.is_some_and(is_consonant) && match_left(rest, w, end - 1),
        Sym::Voiced => {
            c.is_some_and(|c| b"bdvgjlmnrwz".contains(&c)) && match_left(rest, w, end - 1)
        }
        Sym::Front => {
            c.is_some_and(|c| matches!(c, b'e' | b'i' | b'y')) && match_left(rest, w, end - 1)
        }
        Sym::Suffix => false,
        Sym::Boundary => end == 0 && match_left(rest, w, end),
    }
}

const PENULT_SUFFIXES: &[&str] = &[
    "tion", "sion", "cian", "tions", "sions", "cious", "tious", "cial", "tial", "ic", "ics", "ish",
    "ient", "ience", "ual",
];
const PREFIXES: &[&str] = &["be", "de", "re", "un", "in", "ex", "con", "pre", "a"];

fn is_long(p: Phone) -> bool {
    matches!(
        p,
        Phone::EY | Phone::AY | Phone::OW | Phone::AW | Phone::OY | Phone::IY | Phone::UW
    )
}

/// Stress for rule output (the NRL rules give none), then reduction of unstressed vowels.
pub fn assign_stress(word: &str, phones: &[Phone]) -> Vec<Segment> {
    let vowels: Vec<usize> = (0..phones.len())
        .filter(|i| phones[*i].is_vowel())
        .collect();
    let n = vowels.len();
    let mut primary = 0usize;
    let prefixed = PREFIXES.iter().any(|p| {
        let b = word.as_bytes();
        let after = b.get(p.len()).copied();
        let then = b.get(p.len() + 1).copied();
        word.len() > p.len() + 2
            && word.starts_with(p)
            && match *p {
                // Open prefixes are a syllable of their own only before a single consonant
                // and a vowel (about, begin, decide, return; not apple, better).
                "a" | "be" | "de" | "re" => {
                    after.is_some_and(is_consonant) && then.is_some_and(is_vowel)
                }
                "ex" => true,
                _ => after.is_some_and(is_consonant),
            }
    });
    if n == 2 {
        if prefixed && !phones[vowels[1]].is_reduced() {
            primary = 1;
        }
    } else if n >= 3 {
        let penult = n - 2;
        let heavy = {
            let v = vowels[penult];
            let consonants_after = vowels[penult + 1] - v - 1;
            is_long(phones[v]) || consonants_after >= 2
        };
        primary = if PENULT_SUFFIXES.iter().any(|s| word.ends_with(s)) || heavy {
            penult
        } else {
            n - 3
        };
        if primary == 0 && prefixed && n == 3 && !heavy {
            primary = 1;
        }
    }
    // A reduced vowel cannot carry the stress: move to the nearest full vowel.
    if n > 0 && phones[vowels[primary]].is_reduced() {
        let full = |i: &usize| !phones[vowels[*i]].is_reduced();
        if let Some(i) = (primary + 1..n)
            .find(full)
            .or_else(|| (0..primary).rev().find(full))
        {
            primary = i;
        }
    }
    let mut out = Vec::with_capacity(phones.len());
    let mut vi = 0;
    for p in phones {
        let mut seg = Segment::new(*p, Stress::None, 0);
        if p.is_vowel() {
            if vi == primary {
                seg.stress = Stress::Primary;
            } else {
                seg.phone = match *p {
                    Phone::AH => Phone::AX,
                    Phone::ER => Phone::AXR,
                    Phone::IH => Phone::IX,
                    other => other,
                };
            }
            vi += 1;
        }
        out.push(seg);
    }
    out
}

impl RuleLts {
    pub fn new() -> RuleLts {
        static PARSED: OnceLock<Vec<Vec<Rule>>> = OnceLock::new();
        let by_letter = PARSED
            .get_or_init(|| match parse(RULES) {
                Ok(r) => r,
                // The file is compiled in; a malformed rule is a build defect caught by tests.
                Err(_) => vec![Vec::new(); 26],
            })
            .clone();
        RuleLts { by_letter }
    }

    pub fn rule_count(&self) -> usize {
        self.by_letter.iter().map(Vec::len).sum()
    }

    /// Phones without stress.
    pub fn phones(&self, word: &str) -> Vec<Phone> {
        let w: Vec<u8> = word
            .bytes()
            .map(|b| b.to_ascii_lowercase())
            .filter(|b| b.is_ascii_lowercase())
            .collect();
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < w.len() {
            let rules = &self.by_letter[(w[pos] - b'a') as usize];
            let hit = rules.iter().find(|r| {
                w[pos..].starts_with(&r.text)
                    && match_left(&r.left, &w, pos)
                    && match_right(&r.right, &w, pos + r.text.len())
            });
            match hit {
                Some(r) => {
                    out.extend_from_slice(&r.phones);
                    pos += r.text.len();
                }
                None => pos += 1,
            }
        }
        // Doubled consonant phones collapse (the rules map each letter of "tt" separately).
        out.dedup_by(|a, b| a == b && !a.is_vowel());
        out
    }
}

impl Default for RuleLts {
    fn default() -> Self {
        RuleLts::new()
    }
}

impl LetterToSound for RuleLts {
    fn pronounce(&self, word: &str) -> Vec<Segment> {
        let lower: String = word
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        let mut phones = self.phones(&lower);
        if phones.is_empty() && !lower.is_empty() {
            // Every letter list ends with a default, so this is unreachable with a sane rule file.
            phones.push(Phone::AX);
        }
        assign_stress(&lower, &phones)
    }
}

/// Validates the compiled-in rule file.
pub fn check_rules() -> Result<usize, String> {
    let by_letter = parse(RULES)?;
    for (i, rules) in by_letter.iter().enumerate() {
        let letter = (b'a' + i as u8) as char;
        let last = rules.last().ok_or(format!("no rules for '{letter}'"))?;
        if !(last.left.is_empty() && last.right.is_empty() && last.text.len() == 1) {
            return Err(format!(
                "the list for '{letter}' must end with a context-free default"
            ));
        }
    }
    Ok(by_letter.iter().map(Vec::len).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Unit;
    use crate::phone::print_arpabet;

    fn say(word: &str) -> String {
        let segs = RuleLts::new().pronounce(word);
        print_arpabet(&segs.into_iter().map(Unit::Seg).collect::<Vec<_>>())
    }

    #[test]
    fn rule_file_is_well_formed() {
        let n = check_rules().unwrap();
        assert!(n >= 300, "{n} rules");
    }

    #[test]
    fn context_symbols() {
        let w = b"making";
        assert!(match_right(&symbols("^%"), w, 2));
        assert!(!match_right(&symbols("^%"), b"camera", 2));
        assert!(match_right(&symbols("#"), b"boat", 1));
        assert!(match_right(&symbols(":$"), b"cat", 3));
        assert!(match_right(&symbols(":$"), b"cats", 2));
        assert!(match_left(&symbols("$:"), b"strap", 3));
        assert!(match_left(&symbols("#:"), b"total", 3));
        assert!(!match_left(&symbols("$"), b"ab", 1));
        assert!(match_right(&symbols("+"), b"ce", 1));
        assert!(match_right(&symbols("."), b"ab", 1));
    }

    #[test]
    fn some_regular_words() {
        assert_eq!(say("cat"), "K AE1 T");
        assert_eq!(say("make"), "M EY1 K");
        assert_eq!(say("nation"), "N EY1 SH AX N");
        assert_eq!(say("light"), "L AY1 T");
        assert_eq!(say("quick"), "K W IH1 K");
        assert_eq!(say("ship"), "SH IH1 P");
        assert_eq!(say("butter"), "B AH1 T AXR");
        assert_eq!(say("city"), "S IH1 T IY");
    }

    #[test]
    fn never_empty() {
        let lts = RuleLts::new();
        for w in [
            "qzxwv",
            "blorptastic",
            "x",
            "zzz",
            "a",
            "bcdfg",
            "rhythm",
            "I'm",
        ] {
            assert!(!lts.pronounce(w).is_empty(), "{w}");
        }
    }
}
