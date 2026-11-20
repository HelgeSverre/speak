//! `LexiconG2p`: words → phones (spec §7.4, §7.5). No word is ever dropped.

use std::sync::Arc;

use crate::chunk::{Chunk, Item, Segment, Unit, Word, WordSource};
use crate::error::Result;
use crate::phone::{Phone, Stress, parse_arpabet};
use crate::text::lexicon::{Lexicons, Pron, homograph};
use crate::traits::{G2p, LetterToSound};

pub struct LexiconG2p {
    lex: Arc<Lexicons>,
    lts: Box<dyn LetterToSound>,
    /// What the letter-to-sound stage is reported as.
    lts_source: WordSource,
}

fn segs(pron: &Pron) -> Vec<Segment> {
    pron.iter().map(|(p, s)| Segment::new(*p, *s, 0)).collect()
}

fn is_vowel_letter(c: char) -> bool {
    matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn is_sibilant(p: Phone) -> bool {
    matches!(
        p,
        Phone::S | Phone::Z | Phone::SH | Phone::ZH | Phone::CH | Phone::JH
    )
}

/// Suffix phones with voicing agreement (spec §7.4 row 4).
fn attach(stem: &Pron, suffix: &str) -> Pron {
    let mut out = stem.clone();
    let last = stem.last().map(|x| x.0).unwrap_or(Phone::AX);
    let n = Stress::None;
    match suffix {
        "s" => {
            if is_sibilant(last) {
                out.extend([(Phone::IX, n), (Phone::Z, n)]);
            } else if last.is_voiced() {
                out.push((Phone::Z, n));
            } else {
                out.push((Phone::S, n));
            }
        }
        "ed" => {
            if matches!(last, Phone::T | Phone::D) {
                out.extend([(Phone::IX, n), (Phone::D, n)]);
            } else if last.is_voiced() {
                out.push((Phone::D, n));
            } else {
                out.push((Phone::T, n));
            }
        }
        "ing" => out.extend([(Phone::IX, n), (Phone::NG, n)]),
        "ly" => out.extend([(Phone::L, n), (Phone::IY, n)]),
        "er" => out.push((Phone::AXR, n)),
        "est" => out.extend([(Phone::IX, n), (Phone::S, n), (Phone::T, n)]),
        _ => {}
    }
    out
}

impl LexiconG2p {
    pub fn new(
        lex: Arc<Lexicons>,
        lts: Box<dyn LetterToSound>,
        lts_source: WordSource,
    ) -> LexiconG2p {
        LexiconG2p {
            lex,
            lts,
            lts_source,
        }
    }

    fn spell(&self, text: &str) -> Vec<Segment> {
        let mut out = Vec::new();
        for c in text.chars().filter(|c| c.is_ascii_alphabetic()) {
            if let Some(p) = self.lex.letter(c) {
                out.extend(segs(p));
            }
        }
        out
    }

    /// Strips a suffix and looks the stem up (spec §7.4 row 4).
    fn morphology(&self, w: &str) -> Option<(Pron, WordSource)> {
        let try_stems = |stems: &[String], suffix: &str| -> Option<(Pron, WordSource)> {
            stems
                .iter()
                .filter(|s| s.len() >= 2)
                .find_map(|s| self.lex.lookup(s))
                .map(|(pron, source)| (attach(&pron, suffix), source))
        };
        let cut = |n: usize| w[..w.len() - n].to_string();
        let undouble = |s: &str| {
            let b = s.as_bytes();
            if b.len() >= 3 && b[b.len() - 1] == b[b.len() - 2] {
                s[..s.len() - 1].to_string()
            } else {
                String::new()
            }
        };
        if let Some(stem) = w.strip_suffix("'s") {
            return try_stems(&[stem.to_string()], "s");
        }
        if w.len() < 4 {
            return None;
        }
        if w.ends_with("ies")
            && let Some(hit) = try_stems(&[format!("{}y", cut(3))], "s")
        {
            return Some(hit);
        }
        if w.ends_with("es")
            && let Some(hit) = try_stems(&[cut(2)], "s")
        {
            return Some(hit);
        }
        if w.ends_with('s')
            && !w.ends_with("ss")
            && let Some(hit) = try_stems(&[cut(1)], "s")
        {
            return Some(hit);
        }
        for (ending, suffix) in [("ed", "ed"), ("ing", "ing"), ("er", "er"), ("est", "est")] {
            if !w.ends_with(ending) {
                continue;
            }
            let base = cut(ending.len());
            let mut stems = vec![base.clone(), format!("{base}e"), undouble(&base)];
            if let Some(y) = base.strip_suffix('i') {
                stems.push(format!("{y}y"));
            }
            if let Some(hit) = try_stems(&stems, suffix) {
                return Some(hit);
            }
        }
        if w.ends_with("ly") {
            let base = cut(2);
            let mut stems = vec![base.clone()];
            if let Some(y) = base.strip_suffix('i') {
                stems.push(format!("{y}y"));
            }
            return try_stems(&stems, "ly");
        }
        None
    }

    /// A compound whose two halves are both lexicon words of at least 3 letters.
    fn compound(&self, w: &str) -> Option<(Pron, WordSource)> {
        if !w.is_ascii() {
            return None;
        }
        for i in 3..=w.len().saturating_sub(3) {
            if let (Some((a, sa)), Some((b, _))) =
                (self.lex.lookup(&w[..i]), self.lex.lookup(&w[i..]))
            {
                let mut out = a;
                // The second half keeps a secondary stress.
                out.extend(b.into_iter().map(|(p, s)| {
                    (
                        p,
                        if s == Stress::Primary {
                            Stress::Secondary
                        } else {
                            s
                        },
                    )
                }));
                return Some((out, sa));
            }
        }
        None
    }

    pub fn pronounce(&self, text: &str, spell: bool) -> (Vec<Segment>, WordSource) {
        if spell {
            return (self.spell(text), WordSource::Spelled);
        }
        let trimmed = text.trim_matches('\'');
        let lower = trimmed.to_lowercase();
        if let Some((p, source)) = self.lex.lookup(&lower) {
            return (segs(&p), source);
        }
        if let Some((p, source)) = self.morphology(&lower) {
            return (segs(&p), source);
        }
        let letters: String = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if letters != lower
            && let Some((p, source)) = self.lex.lookup(&letters)
        {
            return (segs(&p), source);
        }
        let n = letters.chars().count();
        let all_caps = trimmed.chars().all(|c| c.is_ascii_uppercase());
        if n > 0 && (!letters.chars().any(is_vowel_letter) || (all_caps && (2..=5).contains(&n))) {
            return (self.spell(&letters), WordSource::Spelled);
        }
        if let Some((p, source)) = self.compound(&letters) {
            return (segs(&p), source);
        }
        let out = self.lts.pronounce(&letters);
        if out.is_empty() {
            // Nothing pronounceable is left (the normalizer only passes letters): spell what there is.
            return (self.spell(&letters), WordSource::Spelled);
        }
        (out, self.lts_source)
    }
}

impl G2p for LexiconG2p {
    fn phonemize(&self, chunk: &mut Chunk) -> Result<()> {
        let mut units: Vec<Unit> = Vec::with_capacity(chunk.items.len() * 5);
        let items = std::mem::take(&mut chunk.items);
        let lowered: Vec<String> = chunk.words.iter().map(|w| w.text.to_lowercase()).collect();
        for item in &items {
            match item {
                Item::Word { index, spell } => {
                    let lower = chunk.words[*index].text.to_lowercase();
                    let prev = index.checked_sub(1).map(|i| lowered[i].as_str());
                    // Homographs are decided by their neighbours, unless the user lexicon has the word.
                    let user = self
                        .lex
                        .lookup(&lower)
                        .is_some_and(|l| l.1 == WordSource::User);
                    let homograph = if *spell || user {
                        None
                    } else {
                        homograph::pronounce(&lower, prev, &lowered)
                    };
                    let (mut segs, source) = match homograph {
                        Some(p) => (segs(&p), WordSource::Builtin),
                        None => self.pronounce(&chunk.words[*index].text, *spell),
                    };
                    chunk.words[*index].source = source;
                    if matches!(units.last(), Some(Unit::Seg(_))) {
                        units.push(Unit::WordBreak);
                    }
                    for s in &mut segs {
                        s.word = *index as u32;
                    }
                    units.extend(segs.into_iter().map(Unit::Seg));
                }
                Item::Punct(c) => units.push(Unit::Punct(*c)),
                Item::Silence(ms) => units.push(Unit::Silence {
                    ms: *ms / chunk.controls.rate.max(0.1),
                }),
                Item::Phonemes(p) => {
                    // Malformed inline phonemes are skipped, never fatal.
                    if let Ok(parsed) = parse_arpabet(p) {
                        chunk.words.push(Word {
                            text: format!("[[{p}]]"),
                            span: (0, 0),
                            source: WordSource::User,
                        });
                        let index = (chunk.words.len() - 1) as u32;
                        if matches!(units.last(), Some(Unit::Seg(_))) {
                            units.push(Unit::WordBreak);
                        }
                        for mut u in parsed {
                            if let Unit::Seg(s) = &mut u {
                                s.word = index;
                            }
                            units.push(u);
                        }
                    }
                }
            }
        }
        chunk.items = items;
        chunk.units = units;
        Ok(())
    }
}
