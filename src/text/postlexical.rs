//! Postlexical rules, applied to the phone string of a whole chunk (spec §7.6).

use crate::chunk::{Chunk, Segment, Unit};
use crate::phone::{Phone, Stress};
use crate::text::funcwords::is_function_word;

fn seg(u: &Unit) -> Option<&Segment> {
    u.seg()
}

/// Index of the next segment, crossing word breaks only.
fn next_seg(units: &[Unit], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while let Some(u) = units.get(j) {
        match u {
            Unit::Seg(_) => return Some(j),
            Unit::WordBreak => j += 1,
            _ => return None,
        }
    }
    None
}

fn prev_seg(units: &[Unit], i: usize) -> Option<usize> {
    let mut j = i;
    while j > 0 {
        j -= 1;
        match &units[j] {
            Unit::Seg(_) => return Some(j),
            Unit::WordBreak => {}
            _ => return None,
        }
    }
    None
}

fn word_of(words: &[String], s: &Segment) -> String {
    words.get(s.word as usize).cloned().unwrap_or_default()
}

fn word_text(chunk: &Chunk, s: &Segment) -> String {
    chunk
        .words
        .get(s.word as usize)
        .map(|w| w.text.to_lowercase())
        .unwrap_or_default()
}

/// Phonemes the user wrote (`[[...]]`) are taken as given.
fn is_literal(text: &str) -> bool {
    text.starts_with("[[")
}

pub fn postlexical(chunk: &mut Chunk) {
    let texts: Vec<String> = chunk
        .units
        .iter()
        .map(|u| seg(u).map(|s| word_text(chunk, s)).unwrap_or_default())
        .collect();
    let n_words = {
        let mut ws: Vec<u32> = chunk.segments().map(|s| s.word).collect();
        ws.dedup();
        ws.len()
    };
    let words: Vec<String> = chunk.words.iter().map(|w| w.text.clone()).collect();
    let units = &mut chunk.units;

    // Function-word destressing: the vowel is kept.
    if n_words > 1 {
        for (i, u) in units.iter_mut().enumerate() {
            if let Unit::Seg(s) = u
                && s.stress == Stress::Primary
                && is_function_word(&texts[i])
            {
                s.stress = Stress::None;
            }
        }
    }

    // the / a / to depend on the next word.
    for i in 0..units.len() {
        let Some(s) = seg(&units[i]) else { continue };
        if !s.phone.is_vowel() {
            continue;
        }
        let word = s.word;
        let next_word_phone = (i + 1..units.len())
            .take_while(|j| !matches!(units[*j], Unit::Punct(_) | Unit::Silence { .. }))
            .find_map(|j| seg(&units[j]).filter(|n| n.word != word).map(|n| n.phone));
        let before_vowel = next_word_phone.is_some_and(|p| p.is_vowel());
        let replacement = match texts[i].as_str() {
            "the" if before_vowel => Some((Phone::IY, Stress::None)),
            "a" if n_words == 1 => Some((Phone::EY, Stress::Primary)),
            "to" if next_word_phone.is_some() && !before_vowel => Some((Phone::AX, Stress::None)),
            "to" if before_vowel => Some((Phone::UW, Stress::None)),
            _ => None,
        };
        if let (Some((phone, stress)), Unit::Seg(s)) = (replacement, &mut units[i]) {
            s.phone = phone;
            s.stress = stress;
        }
    }

    // Flapping: T or D between a vowel (or R) and an unstressed vowel; across a word break only
    // when the T or D ends its word.
    for i in 0..units.len() {
        let Some(s) = seg(&units[i]) else { continue };
        if !matches!(s.phone, Phone::T | Phone::D) || is_literal(&texts[i]) {
            continue;
        }
        let (Some(p), Some(n)) = (prev_seg(units, i), next_seg(units, i)) else {
            continue;
        };
        let (ps, ns) = (seg(&units[p]), seg(&units[n]));
        let (Some(ps), Some(ns)) = (ps, ns) else {
            continue;
        };
        let left_ok = ps.word == s.word && (ps.phone.is_vowel() || ps.phone == Phone::R);
        let same_word = ns.word == s.word;
        let right_ok = ns.phone.is_vowel()
            && if same_word {
                ns.stress == Stress::None
            } else {
                true
            };
        if left_ok
            && right_ok
            && let Unit::Seg(s) = &mut units[i]
        {
            s.phone = Phone::DX;
        }
    }

    // Geminate reduction: identical consonants across a word break → one, lengthened.
    let mut i = 0;
    while i < units.len() {
        if let (Some(a), Some(n)) = (seg(&units[i]), next_seg(units, i)) {
            let b = seg(&units[n]);
            if let Some(b) = b
                && a.phone == b.phone
                && a.phone.is_consonant()
                && a.word != b.word
                && !is_literal(&word_of(&words, a))
                && !is_literal(&word_of(&words, b))
            {
                if let Unit::Seg(s) = &mut units[n] {
                    s.dur_scale *= 1.3;
                }
                units.remove(i);
                continue;
            }
        }
        i += 1;
    }

    let literal: Vec<bool> = units
        .iter()
        .map(|u| seg(u).is_some_and(|s| is_literal(&word_of(&words, s))))
        .collect();

    // Syllabic reduction: AX before L, N or M, after a consonant: keep the schwa, mark it short.
    for i in 0..units.len() {
        let Some(s) = seg(&units[i]) else { continue };
        if s.phone != Phone::AX || s.stress != Stress::None || literal[i] {
            continue;
        }
        let word = s.word;
        let after = units
            .get(i + 1)
            .and_then(seg)
            .filter(|n| n.word == word)
            .map(|n| n.phone);
        let before = i
            .checked_sub(1)
            .and_then(|j| seg(&units[j]))
            .filter(|p| p.word == word)
            .map(|p| p.phone);
        if matches!(after, Some(Phone::L | Phone::N | Phone::M))
            && before.is_some_and(|p| p.is_consonant())
            && let Unit::Seg(s) = &mut units[i]
        {
            s.dur_scale *= 0.6;
        }
    }

    // Glottal onset: a chunk-initial stressed vowel gets a preceding Q.
    if let Some(first) = units.iter().position(|u| matches!(u, Unit::Seg(_)))
        && let Some(s) = seg(&units[first])
        && s.phone.is_vowel()
        && s.stress != Stress::None
        && !is_literal(&word_of(&words, s))
    {
        let q = Segment::new(Phone::Q, Stress::None, s.word);
        units.insert(first, Unit::Seg(q));
    }
}
