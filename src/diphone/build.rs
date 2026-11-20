//! Voice builders. The bootstrap voice is analysed from our own formant synthesizer, which
//! gives exact boundaries, F0 and voicing to test the pipeline against (spec §8.2.4).

use crate::chunk::{Chunk, Segment, Unit};
use crate::diphone::format::{Diphone, VoiceFile};
use crate::diphone::{FRAME_MS, inventory};
use crate::dsp::lpc::{self, LpcFrame};
use crate::error::Result;
use crate::klatt::tables::row;
use crate::klatt::{KlattSynth, Voice};
use crate::phone::{Phone, Stress};
use crate::traits::Synthesizer;

/// Silence around the pair, and the pitch everything is rendered at.
pub const EDGE_MS: f32 = 100.0;
pub const BUILD_F0: f32 = 110.0;

/// Duration a phone is rendered with: its inherent duration, with a floor so the halves hold
/// a few frames.
pub fn build_ms(p: Phone) -> f32 {
    if p == Phone::SIL {
        EDGE_MS
    } else {
        row(p).inh.max(40.0)
    }
}

/// The timed phones `_ A B _` (silence phones are not doubled).
pub fn prompt(a: Phone, b: Phone) -> Chunk {
    let mut units = Vec::new();
    let mut push = |p: Phone| {
        if p == Phone::SIL {
            units.push(Unit::Silence { ms: EDGE_MS });
        } else {
            let stress = if p.is_vowel() && !p.is_reduced() {
                Stress::Primary
            } else {
                Stress::None
            };
            let mut s = Segment::new(p, stress, 0);
            s.dur_ms = build_ms(p);
            if p.is_voiced() {
                s.f0 = vec![(0.0, BUILD_F0)];
            }
            units.push(Unit::Seg(s));
        }
    };
    if a != Phone::SIL {
        push(Phone::SIL);
    }
    push(a);
    push(b);
    if b != Phone::SIL {
        push(Phone::SIL);
    }
    Chunk {
        units,
        timed: true,
        ..Chunk::default()
    }
}

/// Times (ms) of the middle of A, the A/B boundary and the middle of B in `prompt(a, b)`.
pub fn landmarks(a: Phone, b: Phone) -> (f32, f32, f32) {
    let lead = if a == Phone::SIL { 0.0 } else { EDGE_MS };
    let (da, db) = (build_ms(a), build_ms(b));
    (lead + da / 2.0, lead + da, lead + da + db / 2.0)
}

/// Cuts the diphone out of the analysis of its prompt.
pub fn cut(a: Phone, b: Phone, frames: &[LpcFrame]) -> Diphone {
    let (mid_a, boundary, mid_b) = landmarks(a, b);
    let index = |ms: f32| ((ms / FRAME_MS).round() as usize).min(frames.len());
    let (start, end) = (
        index(mid_a),
        index(mid_b).max(index(mid_a) + 1).min(frames.len()),
    );
    Diphone {
        a,
        b,
        frames: frames[start..end].to_vec(),
        boundary: index(boundary).clamp(start, end) - start,
        residual: Vec::new(),
        epochs: Vec::new(),
    }
}

/// The voice the bootstrap inventory is rendered with: `klatt-male` without any perturbation.
pub fn source_voice() -> Voice {
    let base = Voice::find("klatt-male").unwrap_or_else(|| crate::klatt::voice::VOICES[0].clone());
    Voice {
        flutter: 0.0,
        jitter: 0.0,
        shimmer: 0.0,
        ..base
    }
}

/// Residual samples are stored as i16 with this scale.
pub const RESIDUAL_SCALE: f32 = 2.0 * 32767.0;

/// LPC residual of `pcm`: the inverse lattice with the same per-sample interpolated
/// coefficients the synthesizer uses.
pub fn residual(pcm: &[f32], frames: &[LpcFrame]) -> Vec<f32> {
    let pre = lpc::pre_emphasis(pcm, lpc::PRE_EMPHASIS);
    let mut inv = crate::dsp::lattice::InverseLattice::new(lpc::ORDER);
    let last = frames.len().saturating_sub(1);
    pre.iter()
        .enumerate()
        .map(|(n, x)| {
            if frames.is_empty() {
                return 0.0;
            }
            let pos = n as f32 / crate::diphone::HOP as f32;
            let i = (pos as usize).min(last);
            let f = frames[i].lerp(&frames[(i + 1).min(last)], pos.fract());
            inv.process(*x, &f.k)
        })
        .collect()
}

/// Renders and analyses one diphone, with its residual and exact pitch marks.
/// A fresh synthesizer per prompt: no state leaks between them.
pub fn analyse(a: Phone, b: Phone, voice: &Voice) -> Result<Diphone> {
    let mut synth = KlattSynth::new(voice.clone());
    synth.renderer_mut().epochs = Some(Vec::new());
    let mut pcm = Vec::new();
    synth.synthesize(&prompt(a, b), &mut pcm)?;
    let marks = synth.renderer_mut().epochs.take().unwrap_or_default();
    let frames = lpc::analyze(&pcm, crate::diphone::SR);
    let mut d = cut(a, b, &frames);
    // The same cut, in samples.
    let (mid_a, _, _) = landmarks(a, b);
    let start = ((mid_a / FRAME_MS).round() as usize).min(frames.len()) * crate::diphone::HOP;
    let len = d.frames.len() * crate::diphone::HOP;
    let res = residual(&pcm, &frames);
    d.residual = (start..start + len)
        .map(|n| {
            (res.get(n).copied().unwrap_or(0.0) * RESIDUAL_SCALE)
                .round()
                .clamp(-32767.0, 32767.0) as i16
        })
        .collect();
    d.epochs = marks
        .iter()
        .map(|m| *m as usize)
        .filter(|m| *m >= start && *m < start + len)
        .map(|m| (m - start) as u32)
        .collect();
    Ok(d)
}

/// A voice holding just the given pairs (tests, experiments).
pub fn subset(pairs: &[(Phone, Phone)]) -> Result<VoiceFile> {
    let voice = source_voice();
    let diphones = pairs
        .iter()
        .map(|(a, b)| analyse(*a, *b, &voice))
        .collect::<Result<Vec<_>>>()?;
    Ok(VoiceFile::new(
        "name=subset; source=klatt-male".into(),
        diphones,
    ))
}

/// Renders and analyses every diphone of the inventory with `klatt-male`.
pub fn bootstrap(name: &str, mut progress: impl FnMut(usize, usize)) -> Result<VoiceFile> {
    let voice = source_voice();
    let all = inventory::all();
    let mut diphones = Vec::with_capacity(all.len());
    for (i, (a, b)) in all.iter().enumerate() {
        diphones.push(analyse(*a, *b, &voice)?);
        progress(i + 1, all.len());
    }
    let description = format!(
        "name={name}; speaker=klatt-male; source=klatt-male; LPC diphones analysed from klatt-male; licence=MIT OR Apache-2.0"
    );
    Ok(VoiceFile::new(description, diphones))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landmarks_and_cut() {
        let (mid_a, boundary, mid_b) = landmarks(Phone::AA, Phone::B);
        assert_eq!(
            (mid_a, boundary, mid_b),
            (100.0 + 120.0, 100.0 + 240.0, 100.0 + 240.0 + 42.5)
        );
        let (mid_a, boundary, _) = landmarks(Phone::SIL, Phone::HH);
        assert_eq!((mid_a, boundary), (50.0, 100.0));
        let frames = vec![LpcFrame::SILENT; 200];
        let d = cut(Phone::AA, Phone::B, &frames);
        assert_eq!(d.frames.len(), 33);
        assert_eq!(d.boundary, 24);
        assert_eq!(prompt(Phone::SIL, Phone::HH).units.len(), 3);
        assert_eq!(prompt(Phone::AA, Phone::B).units.len(), 4);
    }
}
