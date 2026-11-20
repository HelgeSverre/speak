//! `RuleProsody`: Klatt's duration model and a declination-plus-tune F0 model (spec §7.7).

use crate::chunk::{Chunk, Terminator, Unit};
use crate::klatt::tables::row;
use crate::phone::{Class, Phone, Stress};
use crate::traits::{Prosody, ProsodyParams};

/// Duration rule factors (Klatt 1979 via secondary sources; starting values).
pub mod factor {
    pub const CLAUSE_FINAL: f32 = 1.4;
    pub const NON_PHRASE_FINAL: f32 = 0.6;
    pub const NON_WORD_FINAL: f32 = 0.85;
    pub const POLYSYLLABIC: f32 = 0.8;
    pub const NON_INITIAL_CONSONANT: f32 = 0.85;
    pub const UNSTRESSED_MEDIAL_VOWEL: f32 = 0.5;
    pub const UNSTRESSED_VOWEL: f32 = 0.7;
    pub const UNSTRESSED_PREVOCALIC_SONORANT: f32 = 0.1;
    pub const UNSTRESSED_CONSONANT: f32 = 0.7;
    pub const BEFORE_VOICELESS_STOP: f32 = 0.7;
    pub const BEFORE_VOICED_FRICATIVE: f32 = 1.6;
    pub const BEFORE_VOICED_STOP: f32 = 1.2;
    pub const BEFORE_NASAL: f32 = 0.85;
    pub const WORD_FINAL_OPEN: f32 = 1.2;
    pub const VOWEL_BEFORE_VOWEL: f32 = 1.2;
    pub const VOWEL_AFTER_VOWEL: f32 = 0.7;
    pub const CONSONANT_BETWEEN: f32 = 0.5;
    pub const CONSONANT_IN_CLUSTER: f32 = 0.7;
    pub const ASPIRATION_MS: f32 = 25.0;
    /// Pause for a bare `_` in phoneme input.
    pub const DEFAULT_SILENCE_MS: f32 = 100.0;
}

/// Tune values in semitones relative to the base F0.
pub mod tune {
    pub const START: f32 = 2.0;
    pub const END: f32 = -3.0;
    pub const FIRST_ACCENT: f32 = 4.0;
    pub const ACCENT: f32 = 3.0;
    pub const SECONDARY: f32 = 1.0;
    pub const PEAK_POSITION: f32 = 0.3;
    pub const NUCLEUS: f32 = 3.0;
    pub const NUCLEUS_QUESTION: f32 = -2.0;
    pub const NUCLEUS_EXCLAIM: f32 = 6.0;
    pub const FALL: f32 = -6.0;
    pub const RISE: f32 = 7.0;
    pub const CONTINUATION_LOW: f32 = -1.0;
    pub const CONTINUATION_HIGH: f32 = 1.0;
    pub const MICRO: f32 = 0.05;
    pub const MICRO_MS: f32 = 50.0;
}

const WH_WORDS: &[&str] = &["who", "what", "when", "where", "why", "how", "which"];

pub fn clause_pause_ms(c: char) -> f32 {
    Terminator::from_char(c).pause_ms()
}

#[derive(Clone, Debug)]
struct Info {
    unit: usize,
    phone: Phone,
    word: usize,
    phrase: usize,
    syl: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Syl {
    stress: Stress,
    word: usize,
    phrase: usize,
    /// Index into `infos` of the first segment and of the vowel (if any).
    first: usize,
    vowel: Option<usize>,
    last: usize,
}

pub struct RuleProsody;

impl Prosody for RuleProsody {
    fn apply(&self, chunk: &mut Chunk, voice: &ProsodyParams) {
        if chunk.timed {
            return;
        }
        let rate = (chunk.controls.rate * voice.rate).max(0.1);
        let pause_rate = rate / voice.pause.max(0.1);
        chunk.pause_after_ms *= chunk.controls.rate.max(0.1) / pause_rate;
        insert_pauses(chunk, pause_rate);
        let (infos, syls, phrase_marks) = analyse(chunk);
        durations(chunk, &infos, &syls, rate);
        intonation(chunk, &infos, &syls, &phrase_marks, voice);
    }
}

/// Clause punctuation inside the chunk becomes a pause; bare silences get a default length.
fn insert_pauses(chunk: &mut Chunk, rate: f32) {
    let last_seg = chunk.units.iter().rposition(|u| matches!(u, Unit::Seg(_)));
    let mut out = Vec::with_capacity(chunk.units.len() + 4);
    let n = chunk.units.len();
    for i in 0..n {
        let u = chunk.units[i].clone();
        match u {
            Unit::Punct(c) => {
                out.push(Unit::Punct(c));
                let internal = last_seg.is_some_and(|l| i < l);
                let next_is_silence = matches!(chunk.units.get(i + 1), Some(Unit::Silence { .. }));
                if internal && !next_is_silence {
                    out.push(Unit::Silence {
                        ms: clause_pause_ms(c) / rate,
                    });
                }
            }
            Unit::Silence { ms } if ms <= 0.0 => out.push(Unit::Silence {
                ms: factor::DEFAULT_SILENCE_MS / rate,
            }),
            other => out.push(other),
        }
    }
    chunk.units = out;
}

/// Words, phrases and syllables. Returns the punctuation mark that closed each phrase.
fn analyse(chunk: &Chunk) -> (Vec<Info>, Vec<Syl>, Vec<Option<char>>) {
    let mut infos: Vec<Info> = Vec::new();
    let mut marks: Vec<Option<char>> = Vec::new();
    let (mut word, mut phrase) = (0usize, 0usize);
    let mut phrase_has_seg = false;
    let mut pending_mark: Option<char> = None;
    for (ui, u) in chunk.units.iter().enumerate() {
        match u {
            Unit::Seg(s) => {
                infos.push(Info {
                    unit: ui,
                    phone: s.phone,
                    word,
                    phrase,
                    syl: 0,
                });
                phrase_has_seg = true;
            }
            Unit::WordBreak => word += 1,
            Unit::Punct(c) => {
                pending_mark = Some(*c);
                word += 1;
            }
            Unit::Silence { .. } => {
                word += 1;
                if phrase_has_seg {
                    marks.push(pending_mark.take().or(Some(',')));
                    phrase += 1;
                    phrase_has_seg = false;
                }
            }
        }
    }
    if phrase_has_seg {
        marks.push(None);
    }

    // Syllables: a vowel plus the consonants before it back to the previous vowel within the
    // word; trailing consonants join the last syllable.
    let mut syls: Vec<Syl> = Vec::new();
    let mut i = 0;
    while i < infos.len() {
        let w = infos[i].word;
        let mut j = i;
        while j < infos.len() && infos[j].word == w {
            j += 1;
        }
        let first_syl = syls.len();
        let mut start = i;
        for k in i..j {
            if infos[k].phone.is_vowel() {
                let stress = chunk.units[infos[k].unit]
                    .seg()
                    .map(|s| s.stress)
                    .unwrap_or_default();
                syls.push(Syl {
                    stress,
                    word: w,
                    phrase: infos[i].phrase,
                    first: start,
                    vowel: Some(k),
                    last: k,
                });
                start = k + 1;
            }
        }
        if syls.len() == first_syl {
            syls.push(Syl {
                stress: Stress::None,
                word: w,
                phrase: infos[i].phrase,
                first: i,
                vowel: None,
                last: j - 1,
            });
        } else if let Some(s) = syls.last_mut() {
            s.last = j - 1;
        }
        for (si, s) in syls.iter().enumerate().skip(first_syl) {
            for info in &mut infos[s.first..=s.last] {
                info.syl = si;
            }
        }
        i = j;
    }
    (infos, syls, marks)
}

fn durations(chunk: &mut Chunk, infos: &[Info], syls: &[Syl], rate: f32) {
    use factor::*;
    let last_syl_of_phrase = |phrase: usize| syls.iter().rposition(|s| s.phrase == phrase);
    for (i, info) in infos.iter().enumerate() {
        let r = row(info.phone);
        let syl = &syls[info.syl];
        let is_v = info.phone.is_vowel();
        let prev = i
            .checked_sub(1)
            .map(|p| &infos[p])
            .filter(|p| p.phrase == info.phrase);
        let next = infos.get(i + 1).filter(|n| n.phrase == info.phrase);
        let next_in_word = next.filter(|n| n.word == info.word);
        let prev_in_word = prev.filter(|p| p.word == info.word);
        let phrase_final_syl = last_syl_of_phrase(info.phrase) == Some(info.syl);
        let word_syls: Vec<usize> = (0..syls.len())
            .filter(|s| syls[*s].word == info.word)
            .collect();
        let word_final_syl = word_syls.last() == Some(&info.syl);
        let word_first_syl = word_syls.first() == Some(&info.syl);
        let word_initial = prev_in_word.is_none();
        let unstressed = syl.stress == Stress::None;

        let mut prcnt = 1.0f32;
        let mut mindur = r.min;
        // 2. Clause-final lengthening.
        if phrase_final_syl && (is_v || syl.vowel.is_some_and(|v| i > v)) {
            prcnt *= CLAUSE_FINAL;
        }
        if is_v {
            // 3–5.
            if !phrase_final_syl {
                prcnt *= NON_PHRASE_FINAL;
            }
            if !word_final_syl {
                prcnt *= NON_WORD_FINAL;
            }
            if word_syls.len() > 1 {
                prcnt *= POLYSYLLABIC;
            }
        } else if !word_initial {
            // 6.
            prcnt *= NON_INITIAL_CONSONANT;
        }
        // 7. Unstressed shortening.
        if unstressed {
            mindur /= 2.0;
            prcnt *= if is_v {
                if !word_first_syl && !word_final_syl {
                    UNSTRESSED_MEDIAL_VOWEL
                } else {
                    UNSTRESSED_VOWEL
                }
            } else if matches!(info.phone.class(), Class::Liquid | Class::Glide)
                && next_in_word.is_some_and(|n| n.phone.is_vowel())
            {
                UNSTRESSED_PREVOCALIC_SONORANT
            } else {
                UNSTRESSED_CONSONANT
            };
        }
        // 8. Postvocalic context.
        if is_v {
            let f = match next_in_word.map(|n| n.phone) {
                None => WORD_FINAL_OPEN,
                Some(p) if p.is_voiceless_stop() => BEFORE_VOICELESS_STOP,
                Some(p) if p.class() == Class::Fricative && p.is_voiced() => {
                    BEFORE_VOICED_FRICATIVE
                }
                Some(p) if p.is_voiced_stop() => BEFORE_VOICED_STOP,
                Some(p) if p.class() == Class::Nasal => BEFORE_NASAL,
                _ => 1.0,
            };
            prcnt *= if phrase_final_syl { f } else { 0.7 + 0.3 * f };
        }
        // 9. Clusters.
        let pv = prev.map(|p| p.phone.is_vowel());
        let nv = next.map(|n| n.phone.is_vowel());
        if is_v {
            if nv == Some(true) {
                prcnt *= VOWEL_BEFORE_VOWEL;
            }
            if pv == Some(true) {
                prcnt *= VOWEL_AFTER_VOWEL;
            }
        } else if pv == Some(false) && nv == Some(false) {
            prcnt *= CONSONANT_BETWEEN;
        } else if pv == Some(false) || nv == Some(false) {
            prcnt *= CONSONANT_IN_CLUSTER;
        }
        let mut dur = mindur + (r.inh - mindur) * prcnt;
        // 10. Aspiration lengthening.
        let after_voiceless_stop = prev_in_word.is_some_and(|p| p.phone.is_voiceless_stop());
        let lengthens =
            (is_v && !unstressed) || matches!(info.phone.class(), Class::Liquid | Class::Glide);
        if after_voiceless_stop && lengthens {
            dur += ASPIRATION_MS;
        }
        if let Unit::Seg(seg) = &mut chunk.units[info.unit] {
            dur *= seg.dur_scale;
            seg.dur_ms = (dur / rate).max(r.min / 2.0);
        }
    }
}

fn interpolate(points: &[(f32, f32)], t: f32) -> f32 {
    match points.iter().position(|p| p.0 >= t) {
        None => points.last().map(|p| p.1).unwrap_or(0.0),
        Some(0) => points[0].1,
        Some(i) => {
            let (a, b) = (points[i - 1], points[i]);
            if b.0 - a.0 <= 1e-6 {
                b.1
            } else {
                a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0)
            }
        }
    }
}

fn intonation(
    chunk: &mut Chunk,
    infos: &[Info],
    syls: &[Syl],
    marks: &[Option<char>],
    voice: &ProsodyParams,
) {
    use tune::*;
    // Start time of every unit.
    let mut starts = Vec::with_capacity(chunk.units.len());
    let mut ends = Vec::with_capacity(chunk.units.len());
    let mut t = 0.0f32;
    for u in &chunk.units {
        starts.push(t);
        t += u.dur_ms();
        ends.push(t);
    }
    let total = t.max(1.0);
    let t0 = |i: usize| starts[infos[i].unit];
    let t1 = |i: usize| ends[infos[i].unit];
    let base_hz = chunk.controls.pitch_base_hz.unwrap_or(voice.base_hz)
        * 2f32.powf(chunk.controls.pitch_shift_st / 12.0);

    if voice.flat {
        for info in infos {
            if let Unit::Seg(s) = &mut chunk.units[info.unit] {
                s.f0 = if s.phone.is_voiced() {
                    vec![(0.0, base_hz)]
                } else {
                    Vec::new()
                };
            }
        }
        return;
    }

    let baseline = |t: f32| START + (END - START) * t / total;
    let wh_question = chunk
        .words
        .first()
        .is_some_and(|w| WH_WORDS.contains(&w.text.to_lowercase().as_str()));
    let mut contour: Vec<(f32, f32)> = Vec::new();
    let push = |c: &mut Vec<(f32, f32)>, t: f32, st: f32| match c.last() {
        Some(last) if t <= last.0 + 0.5 => {}
        _ => c.push((t, st)),
    };
    let mut first_accent = true;
    for (phrase, mark) in marks.iter().enumerate() {
        let ps: Vec<usize> = (0..syls.len())
            .filter(|s| syls[*s].phrase == phrase)
            .collect();
        let Some(&last_syl) = ps.last() else { continue };
        let phrase_start = t0(syls[ps[0]].first);
        // The tune ends on the last voiced segment; a final voiceless consonant carries no pitch.
        let phrase_end = (syls[ps[0]].first..=syls[last_syl].last)
            .rev()
            .find(|i| infos[*i].phone.is_voiced())
            .map(t1)
            .unwrap_or_else(|| t1(syls[last_syl].last));
        let terminator = match mark {
            Some(c) => Terminator::from_char(*c),
            None => chunk.terminator,
        };
        let nucleus = ps
            .iter()
            .rev()
            .find(|s| syls[**s].stress == Stress::Primary && syls[**s].vowel.is_some())
            .or_else(|| ps.iter().rev().find(|s| syls[**s].vowel.is_some()))
            .copied();
        push(&mut contour, phrase_start, baseline(phrase_start));
        for &s in &ps {
            let syl = &syls[s];
            let Some(v) = syl.vowel else { continue };
            if Some(s) == nucleus {
                break;
            }
            let amp = match syl.stress {
                Stress::Primary => {
                    let a = if first_accent { FIRST_ACCENT } else { ACCENT };
                    first_accent = false;
                    a
                }
                Stress::Secondary => SECONDARY,
                Stress::None => continue,
            };
            let onset = t0(syl.first);
            let peak = t0(v) + PEAK_POSITION * (t1(v) - t0(v));
            push(&mut contour, onset, baseline(onset));
            push(&mut contour, peak, baseline(peak) + amp);
            push(&mut contour, t1(v), baseline(t1(v)));
        }
        let Some(nuc) = nucleus else { continue };
        let syl = &syls[nuc];
        let v = syl.vowel.unwrap_or(syl.first);
        let onset = t0(syl.first);
        let peak = t0(v) + PEAK_POSITION * (t1(v) - t0(v));
        first_accent = false;
        let (nucleus_st, tail): (f32, &[f32]) = match terminator {
            Terminator::Question if !wh_question => (NUCLEUS_QUESTION, &[RISE]),
            Terminator::Exclaim => (NUCLEUS_EXCLAIM, &[FALL]),
            Terminator::Comma | Terminator::Semicolon | Terminator::Colon | Terminator::Dash => {
                (NUCLEUS, &[CONTINUATION_LOW, CONTINUATION_HIGH])
            }
            _ => (NUCLEUS, &[FALL]),
        };
        push(&mut contour, onset, baseline(onset));
        push(&mut contour, peak, baseline(peak) + nucleus_st);
        if tail.len() == 2 {
            let last_onset = t0(syls[last_syl].first);
            let low = if last_onset > peak + 20.0 {
                last_onset
            } else {
                (peak + phrase_end) / 2.0
            };
            push(&mut contour, low, tail[0]);
            push(&mut contour, phrase_end, tail[1]);
        } else {
            push(&mut contour, phrase_end, tail[0]);
        }
    }
    if contour.is_empty() {
        contour.push((0.0, 0.0));
    }

    let hz = |st: f32| base_hz * 2f32.powf(st * voice.range / 12.0);
    for (i, info) in infos.iter().enumerate() {
        let (a, b) = (t0(i), t1(i));
        let dur = (b - a).max(1e-3);
        let micro = info.phone.is_vowel()
            && i > 0
            && infos[i - 1].phrase == info.phrase
            && infos[i - 1].phone.is_obstruent()
            && !infos[i - 1].phone.is_voiced();
        let Unit::Seg(seg) = &mut chunk.units[info.unit] else {
            continue;
        };
        seg.f0.clear();
        if !seg.phone.is_voiced() {
            continue;
        }
        let mut times = vec![a];
        times.extend(
            contour
                .iter()
                .map(|p| p.0)
                .filter(|t| *t > a + 0.5 && *t < b - 0.5),
        );
        if micro && MICRO_MS < dur - 0.5 {
            times.push(a + MICRO_MS);
        }
        times.push(b);
        times.sort_by(f32::total_cmp);
        times.dedup_by(|x, y| (*x - *y).abs() < 0.5);
        for t in times {
            let mut f = hz(interpolate(&contour, t));
            if micro {
                f *= 1.0 + MICRO * (1.0 - (t - a) / MICRO_MS).max(0.0);
            }
            seg.f0.push((((t - a) / dur).clamp(0.0, 1.0), f));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phone::parse_arpabet;

    fn chunk(text: &str) -> Chunk {
        let units = parse_arpabet(text).unwrap();
        let terminator = match units.last() {
            Some(Unit::Punct(c)) => Terminator::from_char(*c),
            _ => Terminator::None,
        };
        let mut c = Chunk {
            units,
            terminator,
            ..Chunk::default()
        };
        RuleProsody.apply(&mut c, &ProsodyParams::default());
        c
    }

    fn dur(c: &Chunk, n: usize) -> f32 {
        c.segments().nth(n).unwrap().dur_ms
    }

    #[test]
    fn duration_formula_worked_example() {
        // AA: INH 240, MIN 100, rules 3, 4, 5 → 100 + 140 × 0.6 × 0.85 × 0.8 = 157 ms.
        let c = chunk("F AA1 L OW0 IY0 / M AE1 N .");
        assert!((dur(&c, 1) - 157.1).abs() < 0.5, "{}", dur(&c, 1));
    }

    #[test]
    fn final_voicing_contrast_is_in_the_vowel() {
        let bed = chunk("B EH1 D .");
        let bet = chunk("B EH1 T .");
        assert!(
            dur(&bed, 1) > dur(&bet, 1) * 1.3,
            "{} vs {}",
            dur(&bed, 1),
            dur(&bet, 1)
        );
    }

    #[test]
    fn every_segment_gets_a_duration_and_voiced_ones_f0() {
        let c = chunk("DH AX / K W IH1 K / B R AW1 N / F AA1 K S , / JH AH1 M P S .");
        for s in c.segments() {
            assert!(
                s.dur_ms >= row(s.phone).min / 2.0 - 1e-3,
                "{:?} {}",
                s.phone,
                s.dur_ms
            );
            assert_eq!(s.f0.is_empty(), !s.phone.is_voiced(), "{:?}", s.phone);
            for (pos, hz) in &s.f0 {
                assert!((0.0..=1.0).contains(pos) && *hz > 50.0 && *hz < 300.0);
            }
        }
        // The comma became a pause.
        assert!(
            c.units
                .iter()
                .any(|u| matches!(u, Unit::Silence { ms } if (*ms - 150.0).abs() < 1.0))
        );
    }

    #[test]
    fn rate_shortens() {
        let mut fast = Chunk {
            units: parse_arpabet("HH AX L OW1 .").unwrap(),
            ..Chunk::default()
        };
        fast.controls.rate = 2.0;
        RuleProsody.apply(&mut fast, &ProsodyParams::default());
        let slow = chunk("HH AX L OW1 .");
        assert!(fast.dur_ms() < slow.dur_ms() * 0.65);
    }

    fn last_f0(c: &Chunk) -> f32 {
        c.segments()
            .filter(|s| !s.f0.is_empty())
            .last()
            .unwrap()
            .f0
            .last()
            .unwrap()
            .1
    }

    #[test]
    fn tunes() {
        let base = 110.0;
        assert!(last_f0(&chunk("IH1 T / IH1 Z / R EH1 D .")) < base * 0.8);
        assert!(last_f0(&chunk("IH1 Z / IH1 T / R EH1 D ?")) > base * 1.3);
        let mut wh = Chunk {
            units: parse_arpabet("W AH1 T / IH1 Z / IH1 T ?").unwrap(),
            terminator: Terminator::Question,
            words: vec![crate::chunk::Word {
                text: "What".into(),
                span: (0, 4),
                source: crate::chunk::WordSource::Builtin,
            }],
            ..Chunk::default()
        };
        RuleProsody.apply(&mut wh, &ProsodyParams::default());
        assert!(
            last_f0(&wh) < base * 0.8,
            "{:?}",
            wh.segments()
                .map(|s| (s.phone, s.f0.clone()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn flat_voice_is_flat() {
        let mut c = Chunk {
            units: parse_arpabet("HH AX L OW1 .").unwrap(),
            ..Chunk::default()
        };
        RuleProsody.apply(
            &mut c,
            &ProsodyParams {
                base_hz: 100.0,
                flat: true,
                ..ProsodyParams::default()
            },
        );
        assert!(c.segments().flat_map(|s| s.f0.iter()).all(|p| p.1 == 100.0));
    }

    #[test]
    fn pitch_controls() {
        let mut c = Chunk {
            units: parse_arpabet("AA1").unwrap(),
            ..Chunk::default()
        };
        c.controls.pitch_base_hz = Some(200.0);
        RuleProsody.apply(&mut c, &ProsodyParams::default());
        let first = c.segments().next().unwrap().f0[0].1;
        assert!(first > 200.0 && first < 300.0, "{first}");
    }
}
