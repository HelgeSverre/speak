//! `--resynth`: WAV → analysis → vocoder → output (spec §9.2). The tool for comparing
//! analysis/synthesis techniques on real audio.

use std::path::Path;

use crate::audio::StreamResampler;
use crate::cli::analyze::read_wav;
use crate::diphone::synth::{Excitation, LpcSynth};
use crate::dsp::griffinlim::{self, HOP, ITERATIONS, MOMENTUM, N_FFT, N_MELS};
use crate::dsp::mel::MelBank;
use crate::dsp::stft::Stft;
use crate::dsp::{lpc, pitch, psola};
use crate::error::{Error, Result};
use crate::traits::{Analyzer, Features, Flow, Mods, PcmWriter, Vocoder};

pub const SR: u32 = 16000;
pub const VOCODERS: &[&str] = &["lpc", "psola", "griffinlim"];

fn wrong_features(name: &str) -> Error {
    Error::Synthesis(format!("{name}: features from another analyzer"))
}

fn write_blocks(pcm: &[f32], out: &mut dyn PcmWriter) -> Result<()> {
    for block in pcm.chunks(320) {
        if out.write(block)? == Flow::Stop {
            break;
        }
    }
    Ok(())
}

/// LPC frames + F0 + voicing, resynthesized with our excitation.
pub struct LpcVocoder {
    pub excitation: Excitation,
}

impl Analyzer for LpcVocoder {
    fn analyze(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Features> {
        Ok(Features::Lpc {
            frames: lpc::analyze(pcm, sample_rate),
        })
    }
}

impl Vocoder for LpcVocoder {
    fn decode(&mut self, features: &Features, mods: &Mods, out: &mut dyn PcmWriter) -> Result<()> {
        let Features::Lpc { frames } = features else {
            return Err(wrong_features("lpc"));
        };
        if frames.is_empty() {
            return Ok(());
        }
        // Time scale: resample the frame sequence. Pitch scale: multiply F0.
        let n = ((frames.len() as f32 * mods.time).round() as usize).max(1);
        let scaled: Vec<lpc::LpcFrame> = (0..n)
            .map(|j| {
                let pos = (j as f32 / mods.time).min((frames.len() - 1) as f32);
                let i = pos as usize;
                let mut f = frames[i].lerp(&frames[(i + 1).min(frames.len() - 1)], pos.fract());
                f.f0 *= mods.pitch;
                f
            })
            .collect();
        LpcSynth::new(self.excitation)
            .render(&scaled, 1.0, out)
            .map(|_| ())
    }
}

/// Pitch marks + TD-PSOLA.
pub struct PsolaVocoder;

impl Analyzer for PsolaVocoder {
    fn analyze(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Features> {
        let track = pitch::track(pcm, sample_rate);
        let (marks, voiced) = psola::epochs(pcm, sample_rate, &track);
        Ok(Features::Marks {
            pcm: pcm.to_vec(),
            marks,
            voiced,
        })
    }
}

impl Vocoder for PsolaVocoder {
    fn decode(&mut self, features: &Features, mods: &Mods, out: &mut dyn PcmWriter) -> Result<()> {
        let Features::Marks { pcm, marks, voiced } = features else {
            return Err(wrong_features("psola"));
        };
        write_blocks(
            &psola::psola(pcm, marks, voiced, mods.time, mods.pitch)?,
            out,
        )
    }
}

/// Log-mel analysis, filterbank pseudo-inverse and Fast Griffin-Lim.
pub struct GriffinLimVocoder {
    stft: Stft,
    bank: MelBank,
}

impl GriffinLimVocoder {
    pub fn new() -> GriffinLimVocoder {
        GriffinLimVocoder {
            stft: Stft::new(N_FFT, HOP),
            bank: MelBank::new(N_MELS, N_FFT, SR, 0.0, 8000.0, false),
        }
    }
}

impl Default for GriffinLimVocoder {
    fn default() -> Self {
        GriffinLimVocoder::new()
    }
}

impl Analyzer for GriffinLimVocoder {
    fn analyze(&mut self, pcm: &[f32], _sample_rate: u32) -> Result<Features> {
        Ok(Features::Mel {
            frames: griffinlim::mel_spectrogram(pcm, &self.stft, &self.bank),
            len: pcm.len(),
        })
    }
}

impl Vocoder for GriffinLimVocoder {
    fn decode(&mut self, features: &Features, mods: &Mods, out: &mut dyn PcmWriter) -> Result<()> {
        let Features::Mel { frames, len } = features else {
            return Err(wrong_features("griffinlim"));
        };
        if (mods.pitch - 1.0).abs() > 1e-3 {
            return Err(Error::unsupported("griffinlim cannot change pitch"));
        }
        if frames.is_empty() {
            return Ok(());
        }
        // Time scale: resample the mel frames in time.
        let n = ((frames.len() as f32 * mods.time).round() as usize).max(1);
        let scaled: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                let pos = (j as f32 / mods.time).min((frames.len() - 1) as f32);
                let (i, t) = (pos as usize, pos.fract());
                let next = &frames[(i + 1).min(frames.len() - 1)];
                frames[i]
                    .iter()
                    .zip(next)
                    .map(|(a, b)| a + (b - a) * t)
                    .collect()
            })
            .collect();
        let out_len = (*len as f32 * mods.time).round() as usize;
        let linear = griffinlim::mel_to_linear(
            &scaled,
            &griffinlim::pseudo_inverse(&self.bank),
            self.bank.bins,
        );
        write_blocks(
            &griffinlim::griffin_lim(&linear, &self.stft, out_len, ITERATIONS, MOMENTUM),
            out,
        )
    }
}

/// Reads a WAV file, mixes to mono and resamples to 16 kHz.
pub fn load(path: &Path) -> Result<Vec<f32>> {
    let (pcm, rate) = read_wav(path)?;
    if rate == SR {
        return Ok(pcm);
    }
    let mut out = Vec::with_capacity(pcm.len() * SR as usize / rate.max(1) as usize + 16);
    let mut r = StreamResampler::new(rate, SR)?;
    let mut emit = |s: &[f32]| {
        out.extend_from_slice(s);
        Ok(true)
    };
    for block in pcm.chunks(4096) {
        r.process(block, &mut emit)?;
    }
    r.flush(&mut emit)?;
    Ok(out)
}

/// Analysis and resynthesis with the named vocoder.
pub fn resynth(
    pcm: &[f32],
    vocoder: &str,
    excitation: Excitation,
    mods: &Mods,
    out: &mut dyn PcmWriter,
) -> Result<()> {
    let run = |v: &mut dyn AnalyzerVocoder, out: &mut dyn PcmWriter| -> Result<()> {
        let features = v.analyze(pcm, SR)?;
        v.decode(&features, mods, out)
    };
    match vocoder {
        "lpc" => {
            if excitation == Excitation::Residual {
                return Err(Error::unsupported(
                    "--resynth --vocoder lpc: excitation must be pulse or glottal",
                ));
            }
            run(&mut LpcVocoder { excitation }, out)
        }
        "psola" => {
            psola::check_factors(mods.time, mods.pitch)?;
            run(&mut PsolaVocoder, out)
        }
        "griffinlim" => {
            if (mods.pitch - 1.0).abs() > 1e-3 {
                return Err(Error::unsupported("griffinlim cannot change pitch"));
            }
            run(&mut GriffinLimVocoder::new(), out)
        }
        other => Err(Error::usage(format!(
            "--vocoder {other}: expected one of {}",
            VOCODERS.join(", ")
        ))),
    }
}

/// Both halves of a vocoder.
trait AnalyzerVocoder: Analyzer + Vocoder {}
impl<T: Analyzer + Vocoder> AnalyzerVocoder for T {}

#[cfg(test)]
mod tests {
    use super::*;

    fn vowel() -> Vec<f32> {
        let mut s = crate::klatt::KlattSynth::new(crate::klatt::Voice {
            flutter: 0.0,
            ..crate::klatt::Voice::find("klatt-male").unwrap()
        });
        let mut pcm = Vec::new();
        let chunk = crate::pho::parse_pho("AA1 500 0 120 100 120\n")
            .unwrap()
            .remove(0);
        crate::traits::Synthesizer::synthesize(&mut s, &chunk, &mut pcm).unwrap();
        pcm
    }

    fn median_f0(pcm: &[f32]) -> f32 {
        let mut v: Vec<f32> = pitch::track(pcm, SR)
            .iter()
            .map(|p| p.f0)
            .filter(|f| *f > 0.0)
            .collect();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    }

    #[test]
    fn every_vocoder_keeps_duration_and_pitch_at_factor_one() {
        let x = vowel();
        for v in VOCODERS {
            let mut y = Vec::new();
            resynth(&x, v, Excitation::Glottal, &Mods::default(), &mut y).unwrap();
            assert!(
                (y.len() as f32 / x.len() as f32 - 1.0).abs() < 0.03,
                "{v}: {} vs {}",
                y.len(),
                x.len()
            );
            assert!(y.iter().all(|s| s.is_finite()));
            assert!(
                (median_f0(&y) / 120.0 - 1.0).abs() < 0.05,
                "{v}: F0 {}",
                median_f0(&y)
            );
        }
    }

    #[test]
    fn modifications() {
        let x = vowel();
        let st4 = 2f32.powf(4.0 / 12.0);
        for v in ["lpc", "psola"] {
            let mut up = Vec::new();
            resynth(
                &x,
                v,
                Excitation::Pulse,
                &Mods {
                    time: 1.0,
                    pitch: st4,
                },
                &mut up,
            )
            .unwrap();
            assert!(
                (median_f0(&up) / 120.0 / st4 - 1.0).abs() < 0.04,
                "{v}: {}",
                median_f0(&up)
            );
            assert!((up.len() as f32 / x.len() as f32 - 1.0).abs() < 0.02);
            let mut fast = Vec::new();
            resynth(
                &x,
                v,
                Excitation::Pulse,
                &Mods {
                    time: 175.0 / 260.0,
                    pitch: 1.0,
                },
                &mut fast,
            )
            .unwrap();
            assert!(
                (fast.len() as f32 / x.len() as f32 - 175.0 / 260.0).abs() < 0.03,
                "{v}"
            );
        }
        let mut slow = Vec::new();
        resynth(
            &x,
            "griffinlim",
            Excitation::Pulse,
            &Mods {
                time: 1.5,
                pitch: 1.0,
            },
            &mut slow,
        )
        .unwrap();
        assert!((slow.len() as f32 / x.len() as f32 - 1.5).abs() < 0.03);
        let e = resynth(
            &x,
            "griffinlim",
            Excitation::Pulse,
            &Mods {
                time: 1.0,
                pitch: 1.1,
            },
            &mut Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            (e.to_string().as_str(), e.exit_code()),
            ("griffinlim cannot change pitch", 2)
        );
        assert_eq!(
            resynth(
                &x,
                "nope",
                Excitation::Pulse,
                &Mods::default(),
                &mut Vec::new()
            )
            .unwrap_err()
            .exit_code(),
            2
        );
        assert_eq!(
            resynth(
                &x,
                "psola",
                Excitation::Pulse,
                &Mods {
                    time: 3.0,
                    pitch: 1.0
                },
                &mut Vec::new()
            )
            .unwrap_err()
            .exit_code(),
            2
        );
    }
}
