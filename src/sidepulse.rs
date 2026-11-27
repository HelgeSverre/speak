//! SidePulse mouth: the voice as a mouth on a strip of 8 RGB LEDs, seen from the front. Optional:
//! nothing else depends on a strip being present.
//!
//! On each syllable the mouth opens from the centre by the vowel's openness (F1), then closes to
//! a dim slit, or goes dark when the lips shut. The colour runs from a centre hue set by F2 to
//! warmer lips, and the brightness follows stress. `cli::lights` drives a SidePulse strip with
//! these shapes; the browser build draws them.

use crate::chunk::{Chunk, Unit};
use crate::klatt::tables::row;
use crate::phone::{Place, Stress};

pub const LEDS: usize = 8;

pub fn hsv(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let f = |u: f32| ((u + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (f(r), f(g), f(b))
}

pub type Rgb = (u8, u8, u8);
/// One colour per LED, left to right.
pub type Shape = [Rgb; LEDS];

pub const DARK: Shape = [(0, 0, 0); LEDS];

/// One syllable: when its vowel sounds and what the mouth looks like on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Syllable {
    pub start_ms: f32,
    pub dur_ms: f32,
    /// Jaw opening from the vowel's F1: 0 = close (IY, UW), 1 = open (AA, AE).
    pub open: f32,
    /// Hue at the centre of the mouth from F2: back vowels warm, front vowels cool.
    pub hue: f32,
    /// Brightness from stress.
    pub level: f32,
    /// The consonant after the vowel closes the lips (P, B, M, F, V, W).
    pub lips_shut: bool,
}

/// Syllables of a timed chunk; times are relative to the chunk start.
pub fn syllables(chunk: &Chunk) -> Vec<Syllable> {
    let mut out: Vec<Syllable> = Vec::new();
    let mut t = 0.0f32;
    let mut after_vowel = false;
    for u in &chunk.units {
        if let Unit::Seg(s) = u
            && s.dur_ms > 0.0
        {
            if s.phone.is_vowel() {
                let r = row(s.phone);
                let (f1, f2) = (0.5 * (r.f[0].0 + r.f[0].1), 0.5 * (r.f[1].0 + r.f[1].1));
                out.push(Syllable {
                    start_ms: t,
                    dur_ms: s.dur_ms,
                    open: ((f1 - 280.0) / (700.0 - 280.0)).clamp(0.0, 1.0),
                    hue: 10.0 + ((f2 - 900.0) / (2100.0 - 900.0)).clamp(0.0, 1.0) * 200.0,
                    level: match s.stress {
                        Stress::Primary => 1.0,
                        Stress::Secondary => 0.7,
                        Stress::None => 0.45,
                    },
                    lips_shut: false,
                });
                after_vowel = true;
            } else if after_vowel {
                if let Some(last) = out.last_mut() {
                    last.lips_shut = s.phone.place() == Place::Labial;
                }
                after_vowel = false;
            }
        }
        t += u.dur_ms();
    }
    out
}

/// The mouth as a lit band centred on the strip: `open` sets its half-width, the lips (outer
/// LEDs) are dimmer than the middle, and the hue drifts from the centre outwards.
pub fn mouth(open: f32, hue: f32, level: f32) -> Shape {
    let half = 0.7 + open.clamp(0.0, 1.0) * 3.3;
    let centre = (LEDS as f32 - 1.0) / 2.0;
    let mut shape = DARK;
    for (i, led) in shape.iter_mut().enumerate() {
        let d = (i as f32 - centre).abs();
        let cover = (half - d + 0.5).clamp(0.0, 1.0);
        let lip = 1.0 - 0.6 * d / (centre + 0.5);
        let v = level * cover * lip;
        if v > 0.02 {
            *led = hsv(hue + 50.0 * d / centre, 0.9, v);
        }
    }
    shape
}

/// The resting mouth between syllables: a dim slit, or dark when the lips are shut.
pub fn rest(s: &Syllable) -> Shape {
    if s.lips_shut {
        DARK
    } else {
        mouth(0.0, s.hue, 0.25 * s.level)
    }
}

/// The mouth reaches `shape` at `ms`, moving there smoothly from the previous key.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    pub ms: u32,
    pub shape: Shape,
}

/// Keyframes for `syls` (absolute times in ms), starting dark. Each syllable opens the mouth
/// towards its vowel and closes it again before the next one; before a pause the mouth goes dark.
/// Two keys with the same shape are a hold.
pub fn keyframes(syls: &[Syllable]) -> Vec<Key> {
    let mut keys: Vec<Key> = Vec::new();
    for (i, s) in syls.iter().enumerate() {
        let (cursor, held) = keys.last().map_or((0, DARK), |k| (k.ms, k.shape));
        let begin = ((s.start_ms - 40.0).max(0.0) as u32).max(cursor);
        if keys.is_empty() || begin > cursor {
            keys.push(Key {
                ms: begin,
                shape: held,
            });
        }
        let peak = ((s.start_ms + 0.45 * s.dur_ms) as u32).max(begin + 20);
        let vowel_end = s.start_ms + s.dur_ms;
        let next = syls.get(i + 1).map(|n| (n.start_ms - 40.0).max(0.0) as u32);
        let (target, end) = match next {
            Some(n) if (n as f32) - vowel_end < 250.0 => (rest(s), n),
            _ => (DARK, (vowel_end + 120.0) as u32),
        };
        keys.push(Key {
            ms: peak,
            shape: mouth(s.open, s.hue, s.level),
        });
        keys.push(Key {
            ms: end.max(peak + 20),
            shape: target,
        });
    }
    keys
}
