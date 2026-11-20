//! Singing: syllables fitted onto notes (a `Prosody` that replaces the speech rules).
//!
//! Each syllable takes one note. The vowel starts on the beat; onset consonants are taken from
//! the end of the previous note, coda consonants from the end of their own. Pitch glides
//! between notes, and held notes get a delayed vibrato.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::chunk::{Chunk, Unit};
use crate::error::{Error, Result};
use crate::klatt::tables::row;
use crate::phone::Stress;
use crate::traits::{Prosody, ProsodyParams};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    /// MIDI note number; `None` is a rest.
    pub midi: Option<f32>,
    pub beats: f32,
}

pub struct Tune {
    pub name: &'static str,
    pub title: &'static str,
    /// Beats per minute.
    pub tempo: f32,
    pub melody: &'static str,
    pub lyrics: &'static str,
}

/// Public-domain tunes.
pub const TUNES: &[Tune] = &[
    Tune {
        name: "daisy",
        title: "Daisy Bell (Harry Dacre, 1892)",
        tempo: 168.0,
        melody: "D5:3 B4:3 G4:3 D4:3 E4 F#4 G4 E4:2 G4 D4:5 R \
                 A4:3 D5:3 B4:3 G4:3 E4 F#4 G4 A4:2 B4 A4:5 R \
                 B4 C5 B4 A4:2 D5 B4:2 A4 G4:5 R \
                 A4:2 B4 G4:2 E4 G4:2 E4 D4:5 R \
                 D4 G4:2 B4 A4:2 D4 G4:2 B4 A4:2 \
                 B4 C5 D5 B4 G4 A4:2 D4 G4:6",
        lyrics: "Daisy, Daisy, give me your answer do. I'm half crazy, all for the love of you. \
                 It won't be a stylish marriage, I can't afford a carriage. \
                 But you'll look sweet upon the seat of a bicycle built for two.",
    },
    Tune {
        name: "twinkle",
        title: "Twinkle, Twinkle, Little Star (traditional)",
        tempo: 120.0,
        melody: "C4 C4 G4 G4 A4 A4 G4:2 F4 F4 E4 E4 D4 D4 C4:2 \
                 G4 G4 F4 F4 E4 E4 D4:2 G4 G4 F4 F4 E4 E4 D4:2 \
                 C4 C4 G4 G4 A4 A4 G4:2 F4 F4 E4 E4 D4 D4 C4:2",
        lyrics: "Twinkle, twinkle, little star, how I wonder what you are. \
                 Up above the world so high, like a diamond in the sky. \
                 Twinkle, twinkle, little star, how I wonder what you are.",
    },
    Tune {
        name: "ode",
        title: "Ode to Joy (Beethoven, 1824; words Henry van Dyke, 1907)",
        tempo: 132.0,
        melody: "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4:1.5 D4:0.5 D4:2 \
                 E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 D4:1.5 C4:0.5 C4:2",
        lyrics: "Joyful, joyful, we adore thee, God of glory, Lord of love. \
                 Hearts unfold like flowers before thee, opening to the sun above.",
    },
];

pub fn tune(name: &str) -> Option<&'static Tune> {
    TUNES.iter().find(|t| t.name == name)
}

/// Parses `C4 D#4:2 Bb3:0.5 R:1`. A note without `:beats` lasts one beat.
pub fn parse_melody(text: &str) -> Result<Vec<Note>> {
    let mut notes = Vec::new();
    for tok in text.split_whitespace() {
        let bad = || {
            Error::usage(format!(
                "melody: bad note '{tok}' (expected e.g. C4, F#3:2, Bb4:0.5, R)"
            ))
        };
        let (name, beats) = match tok.split_once(':') {
            Some((n, b)) => (
                n,
                b.parse::<f32>()
                    .ok()
                    .filter(|b| b.is_finite() && *b > 0.0 && *b <= 64.0)
                    .ok_or_else(bad)?,
            ),
            None => (tok, 1.0),
        };
        if name.eq_ignore_ascii_case("r") || name == "_" {
            notes.push(Note { midi: None, beats });
            continue;
        }
        let mut chars = name.chars();
        let letter = chars.next().ok_or_else(bad)?.to_ascii_uppercase();
        let mut semitone = match letter {
            'C' => 0,
            'D' => 2,
            'E' => 4,
            'F' => 5,
            'G' => 7,
            'A' => 9,
            'B' => 11,
            _ => return Err(bad()),
        };
        let rest: String = chars.collect();
        let octave_text = match rest.chars().next() {
            Some('#') => {
                semitone += 1;
                &rest[1..]
            }
            Some('b') => {
                semitone -= 1;
                &rest[1..]
            }
            _ => rest.as_str(),
        };
        let octave: i32 = octave_text
            .parse()
            .ok()
            .filter(|o| (0..=8).contains(o))
            .ok_or_else(bad)?;
        notes.push(Note {
            midi: Some((12 * (octave + 1) + semitone) as f32),
            beats,
        });
    }
    if !notes.iter().any(|n| n.midi.is_some()) {
        return Err(Error::usage("melody: no notes"));
    }
    Ok(notes)
}

fn midi_hz(midi: f32) -> f32 {
    440.0 * 2f32.powf((midi - 69.0) / 12.0)
}

pub mod style {
    /// Pitch glide between notes, centred a little after the beat.
    pub const GLIDE_BEFORE_MS: f32 = 30.0;
    pub const GLIDE_AFTER_MS: f32 = 45.0;
    /// Vibrato (rate after Prame 1994): starts late, fades in.
    pub const VIBRATO_HZ: f32 = 5.8;
    pub const VIBRATO_ST: f32 = 0.45;
    pub const VIBRATO_DELAY_MS: f32 = 200.0;
    pub const VIBRATO_RAMP_MS: f32 = 250.0;
    /// The tune is moved so its mean sits this far above the speaking pitch.
    pub const TESSITURA: f32 = 1.5;
    pub const MIN_VOWEL_MS: f32 = 70.0;
    pub const F0_STEP_MS: f32 = 12.0;
}

pub struct SingProsody {
    notes: Vec<Note>,
    beat_ms: f32,
    /// Extra transposition in semitones (`--key`).
    key: f32,
    /// Next note; the melody loops and carries on across chunks.
    cursor: AtomicUsize,
}

struct Syl {
    onset: Vec<usize>,
    vowel: usize,
    coda: Vec<usize>,
}

impl SingProsody {
    pub fn new(notes: Vec<Note>, tempo_bpm: f32, key: f32) -> SingProsody {
        SingProsody {
            notes,
            beat_ms: 60_000.0 / tempo_bpm.clamp(30.0, 400.0),
            key,
            cursor: AtomicUsize::new(0),
        }
    }

    /// Semitones that bring the tune into the voice's singing range.
    fn transpose(&self, base_hz: f32) -> f32 {
        let pitches: Vec<f32> = self.notes.iter().filter_map(|n| n.midi).collect();
        let mean = pitches.iter().sum::<f32>() / pitches.len().max(1) as f32;
        let target = 69.0 + 12.0 * (base_hz * style::TESSITURA / 440.0).log2();
        (target - mean).round() + self.key
    }
}

fn syllables(chunk: &Chunk) -> Vec<Syl> {
    // Segment unit indices grouped by word.
    let mut words: Vec<Vec<usize>> = vec![Vec::new()];
    for (i, u) in chunk.units.iter().enumerate() {
        match u {
            Unit::Seg(_) => words.last_mut().unwrap_or(&mut Vec::new()).push(i),
            _ => words.push(Vec::new()),
        }
    }
    let phone = |i: usize| chunk.units[i].seg().map(|s| s.phone);
    let mut out: Vec<Syl> = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    for w in words.iter().filter(|w| !w.is_empty()) {
        let first = out.len();
        for &i in w {
            if phone(i).is_some_and(|p| p.is_vowel()) {
                out.push(Syl {
                    onset: std::mem::take(&mut pending),
                    vowel: i,
                    coda: Vec::new(),
                });
            } else {
                pending.push(i);
            }
        }
        // Consonants after the word's last vowel are its coda; a vowel-less word joins the next onset.
        if out.len() > first
            && let Some(last) = out.last_mut()
        {
            last.coda = std::mem::take(&mut pending);
        }
    }
    if let Some(last) = out.last_mut() {
        last.coda.append(&mut pending);
    }
    out
}

impl Prosody for SingProsody {
    fn apply(&self, chunk: &mut Chunk, voice: &ProsodyParams) {
        use style::*;
        if chunk.timed {
            return;
        }
        let syls = syllables(chunk);
        chunk.pause_after_ms = 0.0;
        if syls.is_empty() {
            return;
        }
        let base_hz = chunk.controls.pitch_base_hz.unwrap_or(voice.base_hz)
            * 2f32.powf(chunk.controls.pitch_shift_st / 12.0);
        let shift = self.transpose(base_hz);
        let rate = chunk.controls.rate.max(0.1);
        let beat = self.beat_ms / rate;

        // Sung consonants are short and do not stretch with the note.
        let cons_ms = |i: usize| -> f32 {
            let p = chunk.units[i]
                .seg()
                .map(|s| s.phone)
                .unwrap_or(crate::phone::Phone::SIL);
            let r = row(p);
            (0.6 * r.inh).clamp(r.min * 0.8, 90.0)
        };
        let sum = |v: &[usize]| v.iter().map(|i| cons_ms(*i)).sum::<f32>();

        // One note per syllable; rests before it become silence.
        let mut plan: Vec<(f32, f32, f32)> = Vec::new(); // (rest before ms, note ms, hz)
        for _ in &syls {
            let mut rest = 0.0;
            loop {
                let n = self.notes[self.cursor.fetch_add(1, Ordering::Relaxed) % self.notes.len()];
                match n.midi {
                    None => rest += n.beats * beat,
                    Some(m) => {
                        plan.push((rest, n.beats * beat, midi_hz(m + shift)));
                        break;
                    }
                }
            }
        }

        let mut dur = vec![0.0f32; chunk.units.len()];
        let mut silence_before: Vec<(usize, f32)> = Vec::new();
        // (start ms, end ms, hz) of each note's voiced span, for the pitch track.
        let mut spans: Vec<(f32, f32)> = Vec::new();
        let mut t = 0.0f32;
        for (k, syl) in syls.iter().enumerate() {
            let (rest, note_ms, _) = plan[k];
            let onset = sum(&syl.onset);
            if k == 0 {
                // The first onset leads in; the first note pays for it.
                t += onset;
            } else if rest > 0.0 {
                let gap = (rest - onset).max(0.0);
                let first = syl.onset.first().copied().unwrap_or(syl.vowel);
                silence_before.push((first, gap));
                t += gap + onset;
            }
            for &i in &syl.onset {
                dur[i] = cons_ms(i);
            }
            let coda = sum(&syl.coda);
            let next_rest = plan.get(k + 1).map(|p| p.0).unwrap_or(0.0);
            let next_onset = syls
                .get(k + 1)
                .filter(|_| next_rest <= 0.0)
                .map(|s| sum(&s.onset))
                .unwrap_or(0.0);
            let lead = if k == 0 {
                onset.min(note_ms * 0.3)
            } else {
                0.0
            };
            let budget = note_ms - lead;
            let mut scale = 1.0;
            let floor = MIN_VOWEL_MS.max(0.35 * budget);
            if budget - coda - next_onset < floor && coda + next_onset > 0.0 {
                scale = ((budget - floor) / (coda + next_onset)).clamp(0.3, 1.0);
            }
            let vowel = (budget - (coda + next_onset) * scale).max(40.0);
            dur[syl.vowel] = vowel;
            for &i in &syl.coda {
                dur[i] = cons_ms(i) * scale;
            }
            if scale < 1.0
                && let Some(next) = syls.get(k + 1)
            {
                // The next onset is squeezed by the same factor (applied when its turn comes).
                for &i in &next.onset {
                    dur[i] = -scale;
                }
            }
            spans.push((t, t + vowel + coda * scale));
            t += vowel + coda * scale + next_onset * scale;
        }
        // Squeezed onsets were marked with a negative factor.
        for syl in &syls {
            for &i in &syl.onset {
                if dur[i] < 0.0 {
                    dur[i] = cons_ms(i) * -dur[i];
                } else if dur[i] == 0.0 {
                    dur[i] = cons_ms(i);
                }
            }
        }

        // Write durations, insert rests.
        let mut units = Vec::with_capacity(chunk.units.len() + silence_before.len());
        for (i, u) in chunk.units.iter().enumerate() {
            if let Some((_, ms)) = silence_before.iter().find(|s| s.0 == i)
                && *ms > 0.0
            {
                units.push(Unit::Silence { ms: *ms });
            }
            let mut u = u.clone();
            if let Unit::Seg(s) = &mut u {
                s.dur_ms = dur[i].max(15.0);
                s.f0.clear();
                if s.phone.is_vowel() {
                    s.stress = Stress::Primary;
                }
            }
            units.push(u);
        }
        chunk.units = units;

        // Note starts on the final timeline (vowel onsets).
        let mut starts: Vec<f32> = Vec::with_capacity(syls.len());
        {
            let mut t = 0.0f32;
            let mut vowels_seen = 0;
            for u in &chunk.units {
                if let Unit::Seg(s) = u
                    && s.phone.is_vowel()
                    && vowels_seen < syls.len()
                {
                    starts.push(t);
                    vowels_seen += 1;
                }
                t += u.dur_ms();
            }
        }
        let pitch = |t: f32| -> f32 {
            // Which note is sounding, with a glide into it.
            let k = starts
                .iter()
                .rposition(|s| *s - GLIDE_BEFORE_MS <= t)
                .unwrap_or(0);
            let hz = plan[k.min(plan.len() - 1)].2;
            let since = t - starts[k];
            let mut f = hz;
            if k > 0 && since < GLIDE_AFTER_MS {
                let prev = plan[k - 1].2;
                let x = ((since + GLIDE_BEFORE_MS) / (GLIDE_BEFORE_MS + GLIDE_AFTER_MS))
                    .clamp(0.0, 1.0);
                let x = 0.5 - 0.5 * (std::f32::consts::PI * x).cos();
                f = prev * (hz / prev).powf(x);
            }
            if !voice.flat && since > VIBRATO_DELAY_MS {
                let depth = VIBRATO_ST * ((since - VIBRATO_DELAY_MS) / VIBRATO_RAMP_MS).min(1.0);
                f *= 2f32.powf(
                    depth * (2.0 * std::f32::consts::PI * VIBRATO_HZ * since / 1000.0).sin() / 12.0,
                );
            }
            f
        };
        let mut t = 0.0f32;
        for u in &mut chunk.units {
            let d = u.dur_ms();
            if let Unit::Seg(s) = u
                && s.phone.is_voiced()
            {
                let steps = (d / F0_STEP_MS).ceil().max(1.0) as usize;
                s.f0 = (0..=steps)
                    .map(|j| {
                        (
                            j as f32 / steps as f32,
                            pitch(t + d * j as f32 / steps as f32),
                        )
                    })
                    .collect();
            }
            t += d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phone::parse_arpabet;

    #[test]
    fn melody_parsing() {
        let n = parse_melody("C4 F#3:2 Bb4:0.5 R A4").unwrap();
        assert_eq!(
            n[0],
            Note {
                midi: Some(60.0),
                beats: 1.0
            }
        );
        assert_eq!(
            n[1],
            Note {
                midi: Some(54.0),
                beats: 2.0
            }
        );
        assert_eq!(
            n[2],
            Note {
                midi: Some(70.0),
                beats: 0.5
            }
        );
        assert_eq!(n[3].midi, None);
        assert!((midi_hz(69.0) - 440.0).abs() < 1e-3);
        assert!(parse_melody("H4").is_err());
        assert!(parse_melody("C4:x").is_err());
        assert!(parse_melody("R R").is_err());
        for t in TUNES {
            assert!(parse_melody(t.melody).unwrap().len() > 10, "{}", t.name);
        }
    }

    fn sing(text: &str, melody: &str, tempo: f32) -> Chunk {
        let mut c = Chunk {
            units: parse_arpabet(text).unwrap(),
            ..Chunk::default()
        };
        SingProsody::new(parse_melody(melody).unwrap(), tempo, 0.0)
            .apply(&mut c, &ProsodyParams::default());
        c
    }

    #[test]
    fn vowels_start_on_the_beat() {
        // 120 bpm: 500 ms per beat.
        let c = sing("L AA1 / L AA1 / L AA1 / L AA1", "C4 D4 E4 C4:2", 120.0);
        let mut t = 0.0;
        let mut onsets = Vec::new();
        for u in &c.units {
            if u.seg().is_some_and(|s| s.phone.is_vowel()) {
                onsets.push(t);
            }
            t += u.dur_ms();
        }
        // The first onset leads in and the first note pays for it; after that the beat is exact.
        assert!(onsets[0] < 80.0);
        for (k, o) in onsets.iter().enumerate().skip(1) {
            assert!((o - 500.0 * k as f32).abs() < 2.0, "{onsets:?}");
        }
        // The last note is two beats long.
        assert!((t - onsets[3] - 1000.0).abs() < 60.0, "{t} {onsets:?}");
    }

    #[test]
    fn pitch_follows_the_notes_with_octave_fitting() {
        let c = sing("L AA1 / L AA1", "C4:2 G4:2", 120.0);
        let vowels: Vec<_> = c.segments().filter(|s| s.phone.is_vowel()).collect();
        let (a, b) = (vowels[0].f0[2].1, vowels[1].f0.last().unwrap().1);
        // A fifth apart (ratio 1.5), within the vibrato depth.
        assert!((b / a - 1.5).abs() < 0.06, "{a} {b}");
        // Fitted near 1.5 × the 110 Hz speaking pitch.
        assert!(a > 110.0 && a < 220.0, "{a}");
        // Vibrato on the held note: the pitch moves.
        let held: Vec<f32> = vowels[1].f0.iter().map(|p| p.1).collect();
        let (lo, hi) = held
            .iter()
            .fold((f32::MAX, 0.0f32), |(l, h), v| (l.min(*v), h.max(*v)));
        assert!(hi / lo > 1.02, "no vibrato: {lo}..{hi}");
    }

    #[test]
    fn rests_and_looping() {
        let c = sing("L AA1 / L AA1 / L AA1", "C4 R:2 D4", 120.0);
        assert!(
            c.units
                .iter()
                .any(|u| matches!(u, Unit::Silence { ms } if *ms > 800.0))
        );
        // Three syllables on a two-note melody: the tune loops.
        assert_eq!(
            c.segments()
                .filter(|s| s.phone.is_vowel() && !s.f0.is_empty())
                .count(),
            3
        );
    }
}
