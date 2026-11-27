//! Command-line front ends.

pub mod analyze;
pub mod args;
pub mod config;
pub mod interactive;
pub mod lab;
pub mod lights;
pub mod listing;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod pipeline;
pub mod resynth;
pub mod wer;

use crate::diphone::DiphoneBackend;
use crate::klatt::KlattBackend;
use crate::traits::Backend;

/// All backends of this build, in listing order.
pub fn backends() -> Vec<Box<dyn Backend>> {
    vec![Box::new(KlattBackend), Box::new(DiphoneBackend)]
}
