//! `Backend` and `Synthesizer` for the formant synthesizer.

use crate::chunk::Chunk;
use crate::error::{Error, Result};
use crate::klatt::voice::{DEFAULT_VOICE, PARAM_KEYS, VOICES, Voice};
use crate::klatt::{FRAME_MS, Frame, Renderer, SR, compile};
use crate::traits::{
    Backend, DataDirs, Flow, PcmWriter, SynthOptions, SynthSpec, SynthStats, Synthesizer,
    VoiceInfo, VoiceSpec, VoiceState,
};

pub struct KlattBackend;

impl Backend for KlattBackend {
    fn id(&self) -> &'static str {
        "klatt"
    }

    fn technique(&self) -> &'static str {
        "cascade/parallel formant synthesis by rule"
    }

    fn sample_rate(&self) -> u32 {
        SR
    }

    fn default_voice(&self) -> &'static str {
        DEFAULT_VOICE
    }

    fn param_keys(&self) -> &'static [&'static str] {
        PARAM_KEYS
    }

    fn voices(&self, _data: &DataDirs) -> Vec<VoiceInfo> {
        VOICES
            .iter()
            .map(|v| VoiceInfo {
                name: v.name.to_string(),
                locale: "en_US".into(),
                backend: "klatt",
                state: VoiceState::BuiltIn,
                description: v.description.to_string(),
            })
            .collect()
    }

    fn open(
        &self,
        voice: &VoiceSpec,
        opts: &SynthOptions,
        _data: &DataDirs,
    ) -> Result<Box<dyn Synthesizer>> {
        let v = Voice::find(&voice.name)
            .ok_or_else(|| {
                Error::usage(format!("unknown voice '{}'. Run speak -v '?'", voice.name))
            })?
            .with_options(opts)?;
        Ok(Box::new(KlattSynth::new(v)))
    }
}

pub struct KlattSynth {
    voice: Voice,
    renderer: Renderer,
}

impl KlattSynth {
    pub fn new(voice: Voice) -> KlattSynth {
        KlattSynth {
            renderer: Renderer::new(&voice),
            voice,
        }
    }

    pub fn voice(&self) -> &Voice {
        &self.voice
    }

    pub fn renderer_mut(&mut self) -> &mut Renderer {
        &mut self.renderer
    }
}

impl Synthesizer for KlattSynth {
    fn spec(&self) -> SynthSpec {
        SynthSpec {
            sample_rate: SR,
            prosody: self.voice.prosody(),
        }
    }

    fn synthesize(&mut self, chunk: &Chunk, out: &mut dyn PcmWriter) -> Result<SynthStats> {
        let frames = compile::compile(chunk, &self.voice);
        let flow = self.renderer.render(&frames, chunk.controls.volume, out)?;
        Ok(SynthStats {
            samples: (frames.len() * crate::klatt::FRAME_SAMPLES) as u64,
            stopped: flow == Flow::Stop,
        })
    }

    fn dump(&mut self, chunk: &Chunk, stage: &str) -> Result<Option<String>> {
        if stage != "frames" {
            return Ok(None);
        }
        let mut s = String::new();
        for (i, f) in compile::compile(chunk, &self.voice).iter().enumerate() {
            s.push_str(&f.tsv(i as f32 * FRAME_MS));
            s.push('\n');
        }
        Ok(Some(s))
    }
}

pub fn frames_header() -> &'static str {
    Frame::TSV_HEADER
}
