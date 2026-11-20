//! Diphone concatenation with LPC source-filter resynthesis (spec §8.2).
//!
//! The filter is measured from audio (here: from our own formant synthesizer's output, or from
//! recordings), stored per diphone as reflection coefficients, and driven by our own excitation.

pub mod autolabel;
pub mod backend;
pub mod build;
pub mod format;
pub mod inventory;
pub mod prompts;
pub mod select;
pub mod synth;

pub use backend::DiphoneBackend;
pub use format::{Diphone, VoiceFile};

/// Sample rate of diphone voices.
pub const SR: u32 = 16000;
/// Samples per analysis frame (5 ms).
pub const HOP: usize = 80;
pub const FRAME_MS: f32 = 5.0;
