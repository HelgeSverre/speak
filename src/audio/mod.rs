//! Audio output: playback through cpal, files, and a null sink (spec §10).

pub mod capture;
pub mod file;
pub mod format;
pub mod null;
pub mod playback;
pub mod resample;

pub use file::FileSink;
pub use format::{DataFormat, FileFormat};
pub use null::NullSink;
pub use playback::Playback;
pub use resample::StreamResampler;
