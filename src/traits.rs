//! Stage traits (spec §4.2).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::chunk::{Chunk, Segment};
use crate::error::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// The consumer is gone or the run was interrupted.
    Stop,
}

pub trait PcmWriter {
    fn write(&mut self, samples: &[f32]) -> Result<Flow>;
}

impl PcmWriter for Vec<f32> {
    fn write(&mut self, samples: &[f32]) -> Result<Flow> {
        self.extend_from_slice(samples);
        Ok(Flow::Continue)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SinkStats {
    /// Samples accepted, at the rate given to `begin`.
    pub samples: u64,
    pub underruns: u64,
    /// When the first sample was accepted.
    pub first_sample: Option<Instant>,
}

pub trait AudioSink: PcmWriter + Send {
    fn begin(&mut self, sample_rate: u32) -> Result<()>;
    /// Blocks until all audio has been played or written.
    fn finish(&mut self) -> Result<SinkStats>;
    /// Seconds of audio that have left the device, for sinks that play in real time.
    fn clock(&self) -> Option<std::sync::Arc<dyn Fn() -> f64 + Send + Sync>> {
        None
    }
}

/// A token in the normalizer's output (spec §7.1).
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Word {
        text: String,
        span: (usize, usize),
        /// The word is a letter name (`p m`, `F B I`), not the word spelled that way.
        spell: bool,
    },
    Punct(char),
    Command(EmbeddedCommand),
    PhonemeLiteral(String),
    ParagraphBreak,
    /// A single newline in the input; the chunker decides what it means.
    Newline,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmbeddedCommand {
    Silence(f32),
    Rate(f32),
    Volume(f32),
    PitchBase(f32),
}

pub trait Normalizer: Send {
    fn normalize(&self, text: &str) -> Vec<Token>;
}

pub trait LetterToSound: Send + Sync {
    /// Always returns at least one phone for a word with at least one letter.
    fn pronounce(&self, word: &str) -> Vec<Segment>;
}

pub trait G2p: Send {
    /// Fills `chunk.units`. Must never drop a word.
    fn phonemize(&self, chunk: &mut Chunk) -> Result<()>;
}

/// Prosody parameters of a voice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProsodyParams {
    pub base_hz: f32,
    /// Multiplier on all pitch excursions.
    pub range: f32,
    /// Flat contour (`klatt-robot`).
    pub flat: bool,
    /// The voice's own pace: multiplies the speaking rate.
    pub rate: f32,
    /// Multiplier on pauses.
    pub pause: f32,
}

impl Default for ProsodyParams {
    fn default() -> Self {
        ProsodyParams {
            base_hz: 110.0,
            range: 1.0,
            flat: false,
            rate: 1.0,
            pause: 1.0,
        }
    }
}

pub trait Prosody: Send {
    /// Sets `dur_ms` and `f0` on every segment and pause.
    fn apply(&self, chunk: &mut Chunk, voice: &ProsodyParams);
}

#[derive(Clone, Debug)]
pub struct DataDirs {
    pub data: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceState {
    BuiltIn,
    Installed,
    Missing,
}

impl VoiceState {
    pub fn label(self) -> &'static str {
        match self {
            VoiceState::BuiltIn => "built-in",
            VoiceState::Installed => "installed",
            VoiceState::Missing => "missing",
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoiceInfo {
    pub name: String,
    pub locale: String,
    pub backend: &'static str,
    pub state: VoiceState,
    pub description: String,
}

#[derive(Clone, Debug)]
pub struct VoiceSpec {
    pub name: String,
}

/// Backend-independent synthesis options: `-P key=value` pairs.
#[derive(Clone, Debug, Default)]
pub struct SynthOptions {
    pub params: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug)]
pub struct SynthSpec {
    pub sample_rate: u32,
    pub prosody: ProsodyParams,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SynthStats {
    pub samples: u64,
    pub stopped: bool,
}

pub trait Backend {
    fn id(&self) -> &'static str;
    fn technique(&self) -> &'static str;
    fn sample_rate(&self) -> u32;
    fn default_voice(&self) -> &'static str;
    /// `-P` keys this backend understands.
    fn param_keys(&self) -> &'static [&'static str];
    fn voices(&self, data: &DataDirs) -> Vec<VoiceInfo>;
    fn open(
        &self,
        voice: &VoiceSpec,
        opts: &SynthOptions,
        data: &DataDirs,
    ) -> Result<Box<dyn Synthesizer>>;
}

pub trait Synthesizer: Send {
    fn spec(&self) -> SynthSpec;
    /// Writes mono f32 PCM. May call `out.write` many times.
    fn synthesize(&mut self, chunk: &Chunk, out: &mut dyn PcmWriter) -> Result<SynthStats>;
    /// Text for `--dump frames` / `--dump units`; `None` when the stage does not exist.
    fn dump(&mut self, _chunk: &Chunk, _stage: &str) -> Result<Option<String>> {
        Ok(None)
    }
}

/// Voice-quality parameters of a glottal source (spec §8.1.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceQuality {
    /// Open quotient, 0.3..0.8.
    pub oq: f32,
    /// Spectral tilt in dB at 3 kHz. Used by `lf` for the return phase.
    pub tl: f32,
    /// Speed quotient (LF only).
    pub sq: f32,
    /// Diplophonia in percent.
    pub di: f32,
}

impl Default for VoiceQuality {
    fn default() -> Self {
        VoiceQuality {
            oq: 0.5,
            tl: 0.0,
            sq: 2.0,
            di: 0.0,
        }
    }
}

pub trait GlottalSource: Send {
    fn reset(&mut self);
    /// One sample of glottal flow derivative at the source's sample rate. `f0 <= 0` is unvoiced.
    fn next(&mut self, f0: f32, q: &VoiceQuality) -> f32;
    /// True on the sample where a new period starts (pitch mark).
    fn epoch(&self) -> bool;
}

/// What an analyzer extracts, and what a vocoder turns back into sound (spec §9.2).
#[derive(Clone, Debug)]
pub enum Features {
    /// LPC frames with F0 and voicing, one per 5 ms.
    Lpc {
        frames: Vec<crate::dsp::lpc::LpcFrame>,
    },
    /// The waveform with its pitch marks.
    Marks {
        pcm: Vec<f32>,
        marks: Vec<usize>,
        voiced: Vec<bool>,
    },
    /// Mel energies per STFT frame, and the length of the analysed signal in samples.
    Mel { frames: Vec<Vec<f32>>, len: usize },
}

/// Modifications a vocoder applies while decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mods {
    /// Output duration ÷ input duration.
    pub time: f32,
    /// Output F0 ÷ input F0.
    pub pitch: f32,
}

impl Default for Mods {
    fn default() -> Self {
        Mods {
            time: 1.0,
            pitch: 1.0,
        }
    }
}

pub trait Analyzer: Send {
    fn analyze(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Features>;
}

pub trait Vocoder: Send {
    /// A vocoder may reject modifications it cannot do.
    fn decode(&mut self, features: &Features, mods: &Mods, out: &mut dyn PcmWriter) -> Result<()>;
}
