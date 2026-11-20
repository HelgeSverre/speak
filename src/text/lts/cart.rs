//! `CartLts`: per-letter decision trees we train from CMUdict (spec §7.5; Black, Lenzo & Pagel 1998).
//!
//! Features: the letters at offsets −3…+3. A node asks "is the letter at offset k equal to c".
//! Leaf labels are phones with stress (`AE1`), `_` for a silent letter, or two phones (`K-S`).

use crate::chunk::Segment;
use crate::phone::{Phone, Stress, parse_phone_token};
use crate::traits::LetterToSound;

pub const MAGIC: &[u8; 4] = b"SLC1";
pub const CONTEXT: i8 = 4;
const LEAF: i8 = 127;

/// The tree compiled into the binary; empty until `speak-lab lts train` has produced it.
pub const EMBEDDED: &[u8] = include_bytes!("../../../data/lts-en.cart");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    /// Letter offset within ±`CONTEXT`, a word-level feature code, or `LEAF`.
    pub offset: i8,
    /// Letter asked about (`#` = outside the word, `V`/`C` = any vowel/consonant letter). Unused in a leaf.
    pub letter: u8,
    /// Child when the answer is yes; in a leaf: the label index.
    pub yes: u32,
    pub no: u32,
}

impl Node {
    pub fn leaf(label: u32) -> Node {
        Node {
            offset: LEAF,
            letter: 0,
            yes: label,
            no: 0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.offset == LEAF
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cart {
    /// One tree per letter a–z; node 0 is the root. An empty tree means "no data for this letter".
    pub trees: Vec<Vec<Node>>,
    pub labels: Vec<String>,
}

/// Class questions use these codes in `Node::letter`: any vowel letter, any consonant letter.
pub const ANY_VOWEL: u8 = b'V';
pub const ANY_CONSONANT: u8 = b'C';

/// Does the letter `c` answer the question `q` (a letter, `#`, or a class code)?
#[inline]
pub fn answers(c: u8, q: u8) -> bool {
    match q {
        ANY_VOWEL => is_vowel_letter(c),
        ANY_CONSONANT => c.is_ascii_lowercase() && !is_vowel_letter(c),
        _ => c == q,
    }
}

/// Word-level features, encoded as offsets beyond the letter window. Their value is a digit.
/// Number of vowel-letter groups in the word (a syllable estimate), capped at 6.
pub const SYLLABLES: i8 = 100;
/// Which vowel group, counted from the end of the word, the letter is in or follows (capped at 5).
pub const FROM_END: i8 = 101;
pub const WORD_FEATURES: [i8; 2] = [SYLLABLES, FROM_END];

fn is_vowel_letter(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

/// Value of feature `offset` for letter `i`: the letter at `i + offset` (`#` outside the word),
/// or a digit for the word-level features.
#[inline]
pub fn letter_at(word: &[u8], i: usize, offset: i8) -> u8 {
    match offset {
        SYLLABLES => {
            let groups = (0..word.len())
                .filter(|k| {
                    is_vowel_letter(word[*k]) && (*k == 0 || !is_vowel_letter(word[*k - 1]))
                })
                .count();
            b'0' + groups.min(6) as u8
        }
        FROM_END => {
            // Vowel groups that start after position i.
            let after = (i + 1..word.len())
                .filter(|k| is_vowel_letter(word[*k]) && !is_vowel_letter(word[*k - 1]))
                .count();
            b'0' + after.min(5) as u8
        }
        _ => {
            let p = i as isize + offset as isize;
            if p < 0 || p >= word.len() as isize {
                b'#'
            } else {
                word[p as usize]
            }
        }
    }
}

impl Cart {
    pub fn node_count(&self) -> usize {
        self.trees.iter().map(Vec::len).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.node_count() * 10 + 4096);
        out.extend_from_slice(MAGIC);
        for tree in &self.trees {
            out.extend_from_slice(&(tree.len() as u32).to_le_bytes());
            for n in tree {
                out.push(n.offset as u8);
                out.push(n.letter);
                out.extend_from_slice(&n.yes.to_le_bytes());
                out.extend_from_slice(&n.no.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.labels.len() as u16).to_le_bytes());
        for l in &self.labels {
            out.push(l.len() as u8);
            out.extend_from_slice(l.as_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Cart, String> {
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = b.get(at..at + n).ok_or("truncated")?;
            at += n;
            Ok(s)
        };
        if take(4)? != MAGIC {
            return Err("bad magic".into());
        }
        let mut trees = Vec::with_capacity(26);
        for _ in 0..26 {
            let n = u32::from_le_bytes(take(4)?.try_into().map_err(|_| "truncated")?) as usize;
            let mut tree = Vec::with_capacity(n);
            for _ in 0..n {
                let r = take(10)?;
                tree.push(Node {
                    offset: r[0] as i8,
                    letter: r[1],
                    yes: u32::from_le_bytes([r[2], r[3], r[4], r[5]]),
                    no: u32::from_le_bytes([r[6], r[7], r[8], r[9]]),
                });
            }
            trees.push(tree);
        }
        let n_labels = u16::from_le_bytes(take(2)?.try_into().map_err(|_| "truncated")?) as usize;
        let mut labels = Vec::with_capacity(n_labels);
        for _ in 0..n_labels {
            let len = take(1)?[0] as usize;
            labels.push(String::from_utf8(take(len)?.to_vec()).map_err(|_| "bad label")?);
        }
        // Every child and label index must be in range: prediction then cannot go out of bounds.
        for tree in &trees {
            for n in tree {
                let ok = if n.is_leaf() {
                    (n.yes as usize) < labels.len()
                } else {
                    (n.yes as usize) < tree.len() && (n.no as usize) < tree.len()
                };
                if !ok {
                    return Err("index out of range".into());
                }
            }
        }
        Ok(Cart { trees, labels })
    }

    /// Label for letter `i` of `word` (lowercase a–z).
    pub fn label(&self, word: &[u8], i: usize) -> &str {
        let Some(tree) = word[i]
            .checked_sub(b'a')
            .and_then(|l| self.trees.get(l as usize))
        else {
            return "_";
        };
        if tree.is_empty() {
            return "_";
        }
        let mut n = 0usize;
        // A well-formed tree is at most as deep as it has nodes.
        for _ in 0..tree.len() {
            let node = tree[n];
            if node.is_leaf() {
                return &self.labels[node.yes as usize];
            }
            n = if answers(letter_at(word, i, node.offset), node.letter) {
                node.yes as usize
            } else {
                node.no as usize
            };
        }
        "_"
    }

    /// Phones with stress, as CMUdict would write them.
    pub fn predict(&self, word: &str) -> Vec<(Phone, Stress)> {
        let w: Vec<u8> = word
            .bytes()
            .map(|b| b.to_ascii_lowercase())
            .filter(|b| b.is_ascii_lowercase())
            .collect();
        let mut out = Vec::new();
        for i in 0..w.len() {
            for tok in self.label(&w, i).split('-').filter(|t| *t != "_") {
                if let Ok(p) = parse_phone_token(tok) {
                    out.push(p);
                }
            }
        }
        out
    }
}

/// CMUdict's unstressed vowels map to our reduced phones (spec §6.1).
pub fn reduce(phone: Phone, stress: Stress) -> Phone {
    match (phone, stress) {
        (Phone::AH, Stress::None) => Phone::AX,
        (Phone::ER, Stress::None) => Phone::AXR,
        (Phone::IH, Stress::None) => Phone::IX,
        (p, _) => p,
    }
}

pub struct CartLts {
    cart: Cart,
}

impl CartLts {
    pub fn new(cart: Cart) -> CartLts {
        CartLts { cart }
    }

    /// The compiled-in tree, if this build has one.
    pub fn embedded() -> Option<CartLts> {
        if EMBEDDED.is_empty() {
            return None;
        }
        Cart::from_bytes(EMBEDDED).ok().map(CartLts::new)
    }
}

impl LetterToSound for CartLts {
    fn pronounce(&self, word: &str) -> Vec<Segment> {
        let mut pron = self.cart.predict(word);
        if pron.is_empty() && word.chars().any(|c| c.is_ascii_alphabetic()) {
            pron.push((Phone::AH, Stress::Primary));
        }
        // Exactly one primary stress: keep the first, promote if there is none.
        let mut seen = false;
        for p in pron.iter_mut().filter(|p| p.0.is_vowel()) {
            if p.1 == Stress::Primary {
                if seen {
                    p.1 = Stress::Secondary;
                }
                seen = true;
            }
        }
        if !seen
            && let Some(p) = pron
                .iter_mut()
                .find(|p| p.0.is_vowel() && p.1 == Stress::Secondary)
                .or(None)
        {
            p.1 = Stress::Primary;
            seen = true;
        }
        if !seen && let Some(p) = pron.iter_mut().find(|p| p.0.is_vowel()) {
            p.1 = Stress::Primary;
        }
        pron.into_iter()
            .map(|(p, s)| Segment::new(reduce(p, s), s, 0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Cart {
        // Letter c: "is the next letter h?" → CH, else K. Letter a: AE1. Letter h: silent. Letter t: T.
        let mut trees = vec![Vec::new(); 26];
        trees[2] = vec![
            Node {
                offset: 1,
                letter: b'h',
                yes: 1,
                no: 2,
            },
            Node::leaf(0),
            Node::leaf(1),
        ];
        trees[0] = vec![Node::leaf(2)];
        trees[7] = vec![Node::leaf(3)];
        trees[19] = vec![Node::leaf(4)];
        Cart {
            trees,
            labels: vec![
                "CH".into(),
                "K".into(),
                "AE1".into(),
                "_".into(),
                "T".into(),
            ],
        }
    }

    #[test]
    fn predicts_and_round_trips() {
        let cart = tiny();
        let say = |w: &str| {
            cart.predict(w)
                .iter()
                .map(|p| p.0.name())
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(say("cat"), "K AE T");
        assert_eq!(say("chat"), "CH AE T");
        let bytes = cart.to_bytes();
        assert_eq!(Cart::from_bytes(&bytes).unwrap(), cart);
        assert!(Cart::from_bytes(&bytes[..bytes.len() - 3]).is_err());
        assert!(Cart::from_bytes(b"XXXX").is_err());
        let mut bad = bytes.clone();
        bad[4 + 4 + 2] = 99; // root of letter a points at a label that does not exist
        assert!(Cart::from_bytes(&bad).is_err());
    }

    #[test]
    fn lts_always_returns_a_phone_and_one_primary() {
        let lts = CartLts::new(tiny());
        assert!(!lts.pronounce("zzz").is_empty());
        let segs = lts.pronounce("catcat");
        assert_eq!(
            segs.iter().filter(|s| s.stress == Stress::Primary).count(),
            1
        );
    }
}
