use std::time::Instant;

use crate::error::Result;
use crate::traits::{AudioSink, Flow, PcmWriter, SinkStats};

/// Discards audio; counts samples. Used by `--bench` and `--dump`.
#[derive(Default)]
pub struct NullSink {
    stats: SinkStats,
}

impl NullSink {
    pub fn new() -> NullSink {
        NullSink::default()
    }
}

impl PcmWriter for NullSink {
    fn write(&mut self, samples: &[f32]) -> Result<Flow> {
        if self.stats.first_sample.is_none() && !samples.is_empty() {
            self.stats.first_sample = Some(Instant::now());
        }
        self.stats.samples += samples.len() as u64;
        Ok(Flow::Continue)
    }
}

impl AudioSink for NullSink {
    fn begin(&mut self, _sample_rate: u32) -> Result<()> {
        self.stats = SinkStats::default();
        Ok(())
    }

    fn finish(&mut self) -> Result<SinkStats> {
        Ok(self.stats)
    }
}
