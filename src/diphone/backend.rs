//! `Backend` and `Synthesizer` for diphone voices.

use std::path::PathBuf;

use crate::chunk::Chunk;
use crate::diphone::format::{self, VoiceFile};
use crate::diphone::select::{self, plan};
use crate::diphone::synth::{Excitation, LpcSynth, residual_excitation};
use crate::diphone::{FRAME_MS, HOP, SR};
use crate::error::{Error, Result};
use crate::traits::{
    Backend, DataDirs, Flow, PcmWriter, ProsodyParams, SynthOptions, SynthSpec, SynthStats,
    Synthesizer, VoiceInfo, VoiceSpec, VoiceState,
};

pub const DEFAULT_VOICE: &str = "boot";
pub const BOOTSTRAP_HINT: &str = "speak-lab diphones build --bootstrap";

pub fn voices_dir(data: &DataDirs) -> PathBuf {
    data.data.join("voices")
}

pub fn voice_path(data: &DataDirs, name: &str) -> PathBuf {
    voices_dir(data).join(format!("{name}.sdv"))
}

/// A `key=value; key=value` description field.
pub fn description_field(description: &str, key: &str) -> Option<String> {
    description.split(';').map(str::trim).find_map(|kv| {
        kv.strip_prefix(key)
            .and_then(|v| v.strip_prefix('='))
            .map(|v| v.trim().to_string())
    })
}

pub struct DiphoneBackend;

impl Backend for DiphoneBackend {
    fn id(&self) -> &'static str {
        "diphone"
    }

    fn technique(&self) -> &'static str {
        "diphone concatenation, LPC lattice resynthesis"
    }

    fn sample_rate(&self) -> u32 {
        SR
    }

    fn default_voice(&self) -> &'static str {
        DEFAULT_VOICE
    }

    fn param_keys(&self) -> &'static [&'static str] {
        &["excitation"]
    }

    fn voices(&self, data: &DataDirs) -> Vec<VoiceInfo> {
        let mut out = Vec::new();
        if let Ok(dir) = std::fs::read_dir(voices_dir(data)) {
            for entry in dir.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("sdv") {
                    continue;
                }
                let Some(name) = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                else {
                    continue;
                };
                let (state, description) = match format::read_header(&path) {
                    Ok((d, n, _)) => {
                        let text = d
                            .split(';')
                            .map(str::trim)
                            .find(|p| !p.contains('='))
                            .map(str::to_string)
                            .unwrap_or_else(|| format!("LPC diphones ({n})"));
                        (VoiceState::Installed, text)
                    }
                    Err(_) => (VoiceState::Missing, "unreadable voice file".to_string()),
                };
                out.push(VoiceInfo {
                    name,
                    locale: "en_US".into(),
                    backend: "diphone",
                    state,
                    description,
                });
            }
        }
        if !out.iter().any(|v| v.name == DEFAULT_VOICE) {
            out.push(VoiceInfo {
                name: DEFAULT_VOICE.into(),
                locale: "en_US".into(),
                backend: "diphone",
                state: VoiceState::Missing,
                description: format!("not built; run: {BOOTSTRAP_HINT}"),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    fn open(
        &self,
        voice: &VoiceSpec,
        opts: &SynthOptions,
        data: &DataDirs,
    ) -> Result<Box<dyn Synthesizer>> {
        let path = voice_path(data, &voice.name);
        if !path.is_file() {
            let hint = if voice.name == DEFAULT_VOICE {
                BOOTSTRAP_HINT.to_string()
            } else {
                format!("speak-lab diphones build --dir DIR --name {}", voice.name)
            };
            return Err(Error::VoiceMissing {
                backend: "diphone".into(),
                name: voice.name.clone(),
                hint,
            });
        }
        let file = VoiceFile::read(&path)?;
        let excitation = Excitation::parse(
            opts.params
                .get("excitation")
                .map(String::as_str)
                .unwrap_or("glottal"),
        )?;
        if excitation == Excitation::Residual && !file.has_residual() {
            return Err(Error::unsupported(format!(
                "voice '{}' has no residuals; rebuild it, or use -P excitation=pulse|glottal",
                voice.name
            )));
        }
        Ok(Box::new(DiphoneSynth::new(file, excitation)))
    }
}

pub struct DiphoneSynth {
    voice: VoiceFile,
    synth: LpcSynth,
    excitation: Excitation,
    base_hz: f32,
}

impl DiphoneSynth {
    pub fn new(voice: VoiceFile, excitation: Excitation) -> DiphoneSynth {
        // A recorded voice states its speaking pitch; the bootstrap voice was rendered at 110 Hz.
        let base_hz = description_field(&voice.description, "f0")
            .and_then(|v| v.parse().ok())
            .filter(|v: &f32| (50.0..=400.0).contains(v))
            .unwrap_or(110.0);
        DiphoneSynth {
            voice,
            synth: LpcSynth::new(excitation),
            excitation,
            base_hz,
        }
    }

    pub fn voice(&self) -> &VoiceFile {
        &self.voice
    }
}

impl Synthesizer for DiphoneSynth {
    fn spec(&self) -> SynthSpec {
        SynthSpec {
            sample_rate: SR,
            prosody: ProsodyParams {
                base_hz: self.base_hz,
                ..ProsodyParams::default()
            },
        }
    }

    fn synthesize(&mut self, chunk: &Chunk, out: &mut dyn PcmWriter) -> Result<SynthStats> {
        let p = plan(chunk, &self.voice, self.base_hz);
        let flow = if self.excitation == Excitation::Residual {
            let e = residual_excitation(&p, &self.voice);
            self.synth
                .render_excited(&p.frames, &e, chunk.controls.volume, out)?
        } else {
            self.synth.render(&p.frames, chunk.controls.volume, out)?
        };
        Ok(SynthStats {
            samples: (p.frames.len() * HOP) as u64,
            stopped: flow == Flow::Stop,
        })
    }

    fn dump(&mut self, chunk: &Chunk, stage: &str) -> Result<Option<String>> {
        let p = plan(chunk, &self.voice, self.base_hz);
        match stage {
            "units" => Ok(Some(select::dump_units(&p.units))),
            "frames" => {
                let mut s = String::new();
                for (i, f) in p.frames.iter().enumerate() {
                    s.push_str(&format!(
                        "{:.0}\t{:.1}\t{:.2}\t{:.5}",
                        i as f32 * FRAME_MS,
                        f.f0,
                        f.voicing,
                        f.gain
                    ));
                    for k in &f.k {
                        s.push_str(&format!("\t{k:.4}"));
                    }
                    s.push('\n');
                }
                Ok(Some(s))
            }
            _ => Ok(None),
        }
    }
}
