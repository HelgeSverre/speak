//! Recording prompts (spec §8.2.6): one nonsense word per diphone, with a guide rendering
//! whose timing is known exactly.

use std::collections::HashMap;

use crate::chunk::{Chunk, Segment, Unit};
use crate::diphone::build::{BUILD_F0, EDGE_MS, build_ms};
use crate::diphone::inventory;
use crate::phone::{Phone, Stress};

/// Carrier phones around the target pair: `_ T AA A B AA _`.
const CARRIER_C: Phone = Phone::T;
const CARRIER_V: Phone = Phone::AA;

#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    pub id: String,
    pub a: Phone,
    pub b: Phone,
    /// The phones to say, silence included at both ends.
    pub phones: Vec<Phone>,
}

impl Prompt {
    /// Index of A within `phones`.
    fn target(&self) -> usize {
        if self.a == Phone::SIL {
            0
        } else if self.b == Phone::SIL {
            self.phones.len() - 2
        } else {
            3
        }
    }

    /// Start time of each phone, plus the end, with durations scaled per phone by `scale`.
    pub fn times(&self, scale: &dyn Fn(usize) -> f32) -> Vec<f32> {
        let mut t = vec![0.0f32];
        for (i, p) in self.phones.iter().enumerate() {
            let last = t[t.len() - 1];
            t.push(last + build_ms(*p) * scale(i));
        }
        t
    }

    /// Middle of A, the A/B boundary and the middle of B, in ms, for the given phone times.
    pub fn landmarks(&self, times: &[f32]) -> (f32, f32, f32) {
        let i = self.target();
        (
            0.5 * (times[i] + times[i + 1]),
            times[i + 1],
            0.5 * (times[i + 1] + times[i + 2]),
        )
    }

    /// Timed phones at a flat pitch. `scale(i)` stretches phone `i`.
    pub fn chunk(&self, f0: f32, scale: &dyn Fn(usize) -> f32) -> Chunk {
        let units = self
            .phones
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let ms = build_ms(*p) * scale(i);
                if *p == Phone::SIL {
                    Unit::Silence { ms }
                } else {
                    let stress = if p.is_vowel() && !p.is_reduced() {
                        Stress::Primary
                    } else {
                        Stress::None
                    };
                    let mut s = Segment::new(*p, stress, 0);
                    s.dur_ms = ms;
                    if p.is_voiced() {
                        s.f0 = vec![(0.0, f0)];
                    }
                    Unit::Seg(s)
                }
            })
            .collect();
        Chunk {
            units,
            timed: true,
            ..Chunk::default()
        }
    }

    pub fn text(&self) -> String {
        self.phones
            .iter()
            .map(|p| p.text())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn phones_for(a: Phone, b: Phone) -> Vec<Phone> {
    use Phone::SIL;
    if a == SIL {
        // Silence-adjacent diphones use `_ B AA _` and `_ AA A _`.
        vec![SIL, b, CARRIER_V, SIL]
    } else if b == SIL {
        vec![SIL, CARRIER_V, a, SIL]
    } else {
        vec![SIL, CARRIER_C, CARRIER_V, a, b, CARRIER_V, SIL]
    }
}

/// Text the frequency order is taken from (ours; any ordinary English would do).
const FREQUENCY_TEXT: &str = "The quick brown fox jumps over the lazy dog near the quiet river bank. \
    A formant synthesizer turns a list of phonemes into sound with a handful of resonators. \
    Please call me back before five, because the report is due tomorrow morning. \
    She said that they would be there when the weather was better and the roads were open. \
    I think we should have another look at what the people in this house are doing with their time. \
    There is nothing more to say about it now, so let us go home and get some sleep. \
    How many of these things do you want me to bring for you and your friends this evening? \
    He could not remember which of the three small boxes was the one with the letters inside. \
    Every year thousands of visitors come to see the old city and its famous market by the water.";

/// How often each diphone occurs in ordinary text, by running our own frontend.
pub fn frequencies() -> HashMap<(Phone, Phone), usize> {
    let mut counts = HashMap::new();
    let Ok(fe) = crate::text::Frontend::new(None, None, None) else {
        return counts;
    };
    let Ok(chunks) = fe.chunks(FREQUENCY_TEXT, crate::chunk::Controls::default()) else {
        return counts;
    };
    for c in chunks {
        let mut seq = vec![Phone::SIL];
        seq.extend(c.segments().map(|s| s.phone));
        seq.push(Phone::SIL);
        for w in seq.windows(2) {
            *counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
    }
    counts
}

/// One prompt per inventory diphone, the most useful ones first.
pub fn all() -> Vec<Prompt> {
    let freq = frequencies();
    let mut pairs = inventory::all();
    // Stable: frequency descending, then inventory order.
    let order: HashMap<(Phone, Phone), usize> =
        pairs.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    pairs.sort_by(|x, y| {
        freq.get(y)
            .unwrap_or(&0)
            .cmp(freq.get(x).unwrap_or(&0))
            .then(order[x].cmp(&order[y]))
    });
    pairs
        .into_iter()
        .enumerate()
        .map(|(i, (a, b))| Prompt {
            id: format!("p{:04}", i + 1),
            a,
            b,
            phones: phones_for(a, b),
        })
        .collect()
}

/// `prompts.tsv`: id, phone sequence, target diphone.
pub fn tsv(prompts: &[Prompt]) -> String {
    prompts
        .iter()
        .map(|p| format!("{}\t{}\t{}\n", p.id, p.text(), inventory::name(p.a, p.b)))
        .collect()
}

pub fn parse_tsv(text: &str) -> Vec<Prompt> {
    text.lines()
        .filter_map(|l| {
            let mut it = l.split('\t');
            let (id, phones, target) = (it.next()?, it.next()?, it.next()?);
            let (a, b) = target.split_once('-')?;
            let phones: Option<Vec<Phone>> = phones.split(' ').map(Phone::from_name).collect();
            Some(Prompt {
                id: id.to_string(),
                a: Phone::from_name(a)?,
                b: Phone::from_name(b)?,
                phones: phones?,
            })
        })
        .collect()
}

/// The guide: flat pitch, inherent durations.
pub fn guide(p: &Prompt) -> Chunk {
    p.chunk(BUILD_F0, &|_| 1.0)
}

pub const GUIDE_EDGE_MS: f32 = EDGE_MS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_diphone_exactly_once_and_frequent_ones_first() {
        let prompts = all();
        let inv = inventory::all();
        assert_eq!(prompts.len(), inv.len());
        let mut seen: Vec<(Phone, Phone)> = prompts.iter().map(|p| (p.a, p.b)).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), inv.len());
        let freq = frequencies();
        assert!(
            freq.len() > 200,
            "{} distinct diphones in the frequency text",
            freq.len()
        );
        assert!(
            freq.get(&(prompts[0].a, prompts[0].b))
                .copied()
                .unwrap_or(0)
                >= freq
                    .get(&(prompts[300].a, prompts[300].b))
                    .copied()
                    .unwrap_or(0)
        );
        // The target pair is adjacent in the phone sequence.
        for p in &prompts {
            assert!(
                p.phones.windows(2).any(|w| w[0] == p.a && w[1] == p.b),
                "{}",
                p.text()
            );
        }
        assert_eq!(parse_tsv(&tsv(&prompts)), prompts);
    }

    #[test]
    fn landmarks_follow_the_durations() {
        let p = Prompt {
            id: "x".into(),
            a: Phone::S,
            b: Phone::IY,
            phones: phones_for(Phone::S, Phone::IY),
        };
        assert_eq!(p.text(), "_ T AA S IY AA _");
        let times = p.times(&|_| 1.0);
        // _ 100, T 75, AA 240, S 105, IY 155, AA 240, _ 100.
        assert_eq!(
            times,
            [0.0, 100.0, 175.0, 415.0, 520.0, 675.0, 915.0, 1015.0]
        );
        assert_eq!(p.landmarks(&times), (467.5, 520.0, 597.5));
        let guide = guide(&p);
        assert!((guide.dur_ms() - 1015.0).abs() < 1e-3);
        let edge = Prompt {
            id: "y".into(),
            a: Phone::SIL,
            b: Phone::HH,
            phones: phones_for(Phone::SIL, Phone::HH),
        };
        assert_eq!(edge.landmarks(&edge.times(&|_| 1.0)), (50.0, 100.0, 140.0));
    }
}
