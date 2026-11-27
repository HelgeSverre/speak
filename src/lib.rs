//! `speak`: offline text-to-speech. One crate; modules follow the pipeline stages.
//!
//! Shared types live at the root (`phone`, `chunk`, `traits`, `error`, `timing`, `pho`);
//! `dsp`, `klatt`, `audio` and `cli` build on them.

pub mod chunk;
pub mod error;
pub mod pho;
pub mod phone;
pub mod timing;
pub mod traits;

pub mod audio;
#[cfg(feature = "cli")]
pub mod cli;
pub mod diphone;
pub mod dsp;
pub mod klatt;
pub mod sidepulse;
pub mod text;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use chunk::{Chunk, Controls, Item, Segment, Terminator, Unit, Word, WordSource};
pub use error::{Error, Result};
pub use phone::{Class, Phone, Place, Stress};
pub use traits::*;

/// Words per minute that corresponds to `Controls::rate == 1.0`.
pub const DEFAULT_WPM: f32 = 175.0;
