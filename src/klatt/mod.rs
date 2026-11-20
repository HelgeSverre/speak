//! Klatt cascade/parallel formant synthesizer (spec §8.1), written from Klatt 1980 and
//! Klatt & Klatt 1990. Two layers: a rule compiler (timed phones → frames) and a renderer
//! (frames → samples).

pub mod backend;
pub mod compile;
pub mod context;
pub mod frame;
pub mod gains;
pub mod render;
pub mod tables;
pub mod voice;

pub use backend::{KlattBackend, KlattSynth};
pub use frame::Frame;
pub use render::Renderer;
pub use voice::{SourceKind, Voice};

/// Output sample rate.
pub const SR: u32 = 16000;
/// Frame period in milliseconds.
pub const FRAME_MS: f32 = 5.0;
/// Samples per frame.
pub const FRAME_SAMPLES: usize = 80;
/// Samples between coefficient updates (1 ms).
pub const UPDATE_SAMPLES: usize = 16;
/// Source oversampling factor.
pub const OVERSAMPLE: usize = 4;

#[cfg(test)]
pub mod tests;
