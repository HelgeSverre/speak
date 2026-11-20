//! LPC synthesis: excitation → all-pole lattice → de-emphasis (spec §8.2.3).

use crate::diphone::format::VoiceFile;
use crate::diphone::select::Plan;
use crate::diphone::{HOP, SR};
use crate::dsp::lattice::Lattice;
use crate::dsp::lpc::{LpcFrame, ORDER, PRE_EMPHASIS};
use crate::dsp::source::Klglott88;
use crate::dsp::{Noise, soft_limit};
use crate::error::{Error, Result};
use crate::traits::{Flow, GlottalSource, PcmWriter, VoiceQuality};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Excitation {
    /// One impulse per period: the classic LPC buzz.
    Pulse,
    /// KLGLOTT88, shared with the formant synthesizer: softer.
    Glottal,
    /// The stored residual, re-pitched with PSOLA (needs a voice file with residuals).
    Residual,
}

impl Excitation {
    pub fn parse(s: &str) -> Result<Excitation> {
        match s {
            "pulse" => Ok(Excitation::Pulse),
            "glottal" => Ok(Excitation::Glottal),
            "residual" => Ok(Excitation::Residual),
            other => Err(Error::usage(format!(
                "-P excitation={other}: expected pulse, glottal or residual"
            ))),
        }
    }
}

/// Samples between coefficient updates (1 ms).
const UPDATE: usize = 16;
/// 20 ms blocks.
const BLOCK_FRAMES: usize = 4;
/// Open quotient of the glottal excitation.
const OQ: f32 = 0.5;

pub struct LpcSynth {
    excitation: Excitation,
    lattice: Lattice,
    glottal: Klglott88,
    noise: Noise,
    /// Phase of the pulse train in periods.
    phase: f32,
    de: f32,
    block: Vec<f32>,
}

impl LpcSynth {
    pub fn new(excitation: Excitation) -> LpcSynth {
        LpcSynth {
            excitation,
            lattice: Lattice::new(ORDER),
            glottal: Klglott88::new(SR as f32),
            noise: Noise::new(0x51DE_CAFE),
            phase: 0.0,
            de: 0.0,
            block: Vec::with_capacity(BLOCK_FRAMES * HOP),
        }
    }

    /// One sample of voiced excitation at unit RMS per period.
    #[inline]
    fn voiced(&mut self, f0: f32) -> f32 {
        match self.excitation {
            Excitation::Pulse | Excitation::Residual => {
                self.phase += f0 / SR as f32;
                if self.phase >= 1.0 {
                    self.phase -= 1.0;
                    // One sample of amplitude sqrt(T0) has unit mean square over the period.
                    (SR as f32 / f0).sqrt()
                } else {
                    0.0
                }
            }
            Excitation::Glottal => {
                let q = VoiceQuality {
                    oq: OQ,
                    ..VoiceQuality::default()
                };
                // RMS of 2x − 3x² over the open phase is sqrt(2/15); over the period sqrt(2·OQ/15).
                self.glottal.next(f0, &q) / (2.0 * OQ / 15.0).sqrt()
            }
        }
    }

    /// Renders frames (5 ms each) in 20 ms blocks.
    pub fn render(
        &mut self,
        frames: &[LpcFrame],
        volume: f32,
        out: &mut dyn PcmWriter,
    ) -> Result<Flow> {
        for (bi, block) in frames.chunks(BLOCK_FRAMES).enumerate() {
            self.block.clear();
            for (fi, frame) in block.iter().enumerate() {
                let next = frames.get(bi * BLOCK_FRAMES + fi + 1).unwrap_or(frame);
                for step in 0..HOP / UPDATE {
                    let f = frame.lerp(next, step as f32 * UPDATE as f32 / HOP as f32);
                    // Voicing below 0.3 is treated as 0, above 0.7 as 1.
                    let v = ((f.voicing - 0.3) / 0.4).clamp(0.0, 1.0);
                    let f0 = f.f0.clamp(40.0, 600.0);
                    for _ in 0..UPDATE {
                        let voiced = if v > 0.0 { self.voiced(f0) } else { 0.0 };
                        // Uniform noise in [−1, 1] has RMS 1/√3.
                        let unvoiced = self.noise.sample() * 3f32.sqrt();
                        let e = (v * voiced + (1.0 - v) * unvoiced) * f.gain;
                        let y = self.lattice.process(e, &f.k);
                        self.de = y + PRE_EMPHASIS * self.de;
                        self.block.push(soft_limit(self.de * volume));
                    }
                }
            }
            if self.block.iter().any(|s| !s.is_finite()) {
                // A filter that left the rails is reset rather than allowed to poison the rest.
                self.lattice.reset();
                self.de = 0.0;
                return Err(Error::Synthesis(
                    "non-finite sample in the LPC synthesizer".into(),
                ));
            }
            if out.write(&self.block)? == Flow::Stop {
                return Ok(Flow::Stop);
            }
        }
        Ok(Flow::Continue)
    }
}

impl LpcSynth {
    /// Renders frames with a ready-made excitation signal (one sample per output sample).
    pub fn render_excited(
        &mut self,
        frames: &[LpcFrame],
        excitation: &[f32],
        volume: f32,
        out: &mut dyn PcmWriter,
    ) -> Result<Flow> {
        for (bi, block) in frames.chunks(BLOCK_FRAMES).enumerate() {
            self.block.clear();
            for (fi, frame) in block.iter().enumerate() {
                let index = bi * BLOCK_FRAMES + fi;
                let next = frames.get(index + 1).unwrap_or(frame);
                for step in 0..HOP / UPDATE {
                    let f = frame.lerp(next, step as f32 * UPDATE as f32 / HOP as f32);
                    for k in 0..UPDATE {
                        let e = excitation
                            .get(index * HOP + step * UPDATE + k)
                            .copied()
                            .unwrap_or(0.0);
                        let y = self.lattice.process(e, &f.k);
                        self.de = y + PRE_EMPHASIS * self.de;
                        self.block.push(soft_limit(self.de * volume));
                    }
                }
            }
            if self.block.iter().any(|s| !s.is_finite()) {
                self.lattice.reset();
                self.de = 0.0;
                return Err(Error::Synthesis(
                    "non-finite sample in the LPC synthesizer".into(),
                ));
            }
            if out.write(&self.block)? == Flow::Stop {
                return Ok(Flow::Stop);
            }
        }
        Ok(Flow::Continue)
    }
}

/// Excitation from the stored residuals (spec §8.2.3, §9.3): in voiced stretches, two-period
/// Hann grains of the residual around its pitch marks are overlap-added at the target period
/// (PSOLA on the residual). Unvoiced stretches use noise at the stored gain.
pub fn residual_excitation(plan: &Plan, voice: &VoiceFile) -> Vec<f32> {
    let len = plan.frames.len() * HOP;
    let mut out = vec![0.0f32; len];
    let mut noise = Noise::new(0x51DE_CAFE);
    let default_period = SR as f32 / 110.0;
    let mut pos = 0.0f32;
    let mut in_voiced = false;
    while (pos as usize) < len {
        let j = pos as usize / HOP;
        let f = &plan.frames[j];
        let source = plan.map[j].and_then(|(id, at)| {
            voice
                .diphones
                .get(id)
                .filter(|d| d.epochs.len() >= 2 && !d.residual.is_empty())
                .map(|d| (d, at))
        });
        let Some((d, at)) = source.filter(|_| f.voicing > 0.5 && f.f0 > 0.0) else {
            // Unvoiced or no residual: noise for the rest of this frame.
            let end = (j + 1) * HOP;
            for o in out.iter_mut().take(end.min(len)).skip(pos as usize) {
                *o += noise.sample() * 3f32.sqrt() * f.gain;
            }
            pos = end as f32;
            in_voiced = false;
            continue;
        };
        // Source sample this output position maps to.
        let s = at * HOP as f32 + (pos - (j * HOP) as f32);
        let nearest = (0..d.epochs.len())
            .min_by(|a, b| {
                (d.epochs[*a] as f32 - s)
                    .abs()
                    .total_cmp(&(d.epochs[*b] as f32 - s).abs())
            })
            .unwrap_or(0);
        let mark = d.epochs[nearest] as f32;
        if !in_voiced {
            // First mark of a voiced stretch: put it where the source mark maps to.
            let snapped = pos + (mark - s);
            if snapped > pos && (snapped as usize) < len {
                pos = snapped;
            }
            in_voiced = true;
        }
        let period = if nearest + 1 < d.epochs.len() {
            (d.epochs[nearest + 1] - d.epochs[nearest]) as f32
        } else if nearest > 0 {
            (d.epochs[nearest] - d.epochs[nearest - 1]) as f32
        } else {
            default_period
        }
        .clamp(20.0, 400.0);
        let p = period.round() as isize;
        let centre = pos.round() as isize;
        for k in -p..=p {
            let (src, dst) = (mark as isize + k, centre + k);
            if src < 0 || src >= d.residual.len() as isize || dst < 0 || dst >= len as isize {
                continue;
            }
            let w = 0.5 + 0.5 * (std::f32::consts::PI * k as f32 / p as f32).cos();
            out[dst as usize] +=
                d.residual[src as usize] as f32 / crate::diphone::build::RESIDUAL_SCALE * w;
        }
        pos += (SR as f32 / f.f0.clamp(40.0, 600.0)).max(8.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(voicing: f32, f0: f32, n: usize) -> Vec<LpcFrame> {
        let mut f = LpcFrame::SILENT;
        f.gain = 0.05;
        f.voicing = voicing;
        f.f0 = f0;
        f.k[0] = -0.5;
        vec![f; n]
    }

    #[test]
    fn loudness_does_not_depend_on_pitch_or_excitation() {
        let level = |e: Excitation, f0: f32, voicing: f32| {
            let mut s = LpcSynth::new(e);
            let mut out = Vec::new();
            s.render(&frames(voicing, f0, 100), 1.0, &mut out).unwrap();
            assert_eq!(out.len(), 100 * HOP);
            crate::dsp::rms_dbfs(&out[1600..])
        };
        // Same excitation, different pitch: the level holds (pulse amplitude grows with sqrt(T0)).
        for e in [Excitation::Pulse, Excitation::Glottal] {
            let (low, high) = (level(e, 110.0, 1.0), level(e, 220.0, 1.0));
            assert!(
                (low - high).abs() < 3.5,
                "{e:?}: {low} dB at 110 Hz, {high} dB at 220 Hz"
            );
        }
        // Noise and pulses of unit RMS give the same level; the glottal pulse has its energy at
        // low frequencies, which this low-pass test filter favours.
        let (pulse, noise, glottal) = (
            level(Excitation::Pulse, 110.0, 1.0),
            level(Excitation::Pulse, 110.0, 0.0),
            level(Excitation::Glottal, 110.0, 1.0),
        );
        assert!(
            (pulse - noise).abs() < 3.5,
            "pulse {pulse} dB, noise {noise} dB"
        );
        assert!(
            (pulse - glottal).abs() < 9.0,
            "pulse {pulse} dB, glottal {glottal} dB"
        );
    }

    #[test]
    fn silence_in_silence_out() {
        let mut s = LpcSynth::new(Excitation::Glottal);
        let mut out = Vec::new();
        s.render(&vec![LpcFrame::SILENT; 20], 1.0, &mut out)
            .unwrap();
        assert!(out.iter().all(|v| *v == 0.0));
    }
}
