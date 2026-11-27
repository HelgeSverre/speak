//! Audio output: playback through cpal, files, and a null sink (spec §10).

#[cfg(feature = "cli")]
pub mod capture;
pub mod file;
pub mod format;
pub mod null;
#[cfg(feature = "cli")]
pub mod playback;
pub mod resample;

pub use file::FileSink;
pub use format::{DataFormat, FileFormat};
pub use null::NullSink;
#[cfg(feature = "cli")]
pub use playback::Playback;
pub use resample::StreamResampler;
