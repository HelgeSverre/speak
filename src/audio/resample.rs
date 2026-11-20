//! Fixed-ratio streaming resampler on rubato's FFT resampler.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::error::{Error, Result};

pub struct StreamResampler {
    inner: Fft<f32>,
    ratio: f64,
    pending: Vec<f32>,
    out: Vec<f32>,
    /// Leading output frames still to drop (the resampler's delay).
    skip: usize,
    fed: u64,
    produced: u64,
}

impl StreamResampler {
    pub fn new(from: u32, to: u32) -> Result<StreamResampler> {
        // 20 ms input chunks keep the added latency small.
        let chunk = (from as usize / 50).max(64);
        let inner = Fft::<f32>::new(from as usize, to as usize, chunk, 1, FixedSync::Input)
            .map_err(|e| Error::audio(format!("resampler {from} → {to} Hz: {e}")))?;
        let skip = inner.output_delay();
        let out = vec![0.0; inner.output_frames_max()];
        Ok(StreamResampler {
            inner,
            ratio: to as f64 / from as f64,
            pending: Vec::new(),
            out,
            skip,
            fed: 0,
            produced: 0,
        })
    }

    fn run_chunk(
        &mut self,
        emit: &mut dyn FnMut(&[f32]) -> Result<bool>,
        limit: Option<u64>,
    ) -> Result<bool> {
        let need = self.inner.input_frames_next();
        let err = |e: String| Error::audio(format!("resampler: {e}"));
        let n_out = {
            let input = InterleavedSlice::new(&self.pending[..need], 1, need)
                .map_err(|e| err(e.to_string()))?;
            let cap = self.out.len();
            let mut output = InterleavedSlice::new_mut(&mut self.out[..], 1, cap)
                .map_err(|e| err(e.to_string()))?;
            let (_, n) = self
                .inner
                .process_into_buffer(&input, &mut output, None)
                .map_err(|e| err(e.to_string()))?;
            n
        };
        self.pending.drain(..need);
        let drop = self.skip.min(n_out);
        self.skip -= drop;
        let mut slice = &self.out[drop..n_out];
        if let Some(limit) = limit {
            let left = limit.saturating_sub(self.produced) as usize;
            slice = &slice[..slice.len().min(left)];
        }
        self.produced += slice.len() as u64;
        if slice.is_empty() {
            Ok(true)
        } else {
            emit(slice)
        }
    }

    /// Feeds samples; calls `emit` with resampled audio. `emit` returns false to stop.
    pub fn process(
        &mut self,
        input: &[f32],
        emit: &mut dyn FnMut(&[f32]) -> Result<bool>,
    ) -> Result<bool> {
        self.pending.extend_from_slice(input);
        self.fed += input.len() as u64;
        while self.pending.len() >= self.inner.input_frames_next() {
            if !self.run_chunk(emit, None)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Flushes the tail so the total output is `ceil(input × ratio)` frames.
    pub fn flush(&mut self, emit: &mut dyn FnMut(&[f32]) -> Result<bool>) -> Result<bool> {
        let total = (self.fed as f64 * self.ratio).ceil() as u64;
        let mut guard = 0;
        while self.produced < total && guard < 64 {
            let need = self.inner.input_frames_next();
            if self.pending.len() < need {
                self.pending.resize(need, 0.0);
            }
            if !self.run_chunk(emit, Some(total))? {
                return Ok(false);
            }
            guard += 1;
        }
        self.pending.clear();
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_and_tone_survive() {
        for to in [44100u32, 48000, 22050, 8000] {
            let mut r = StreamResampler::new(16000, to).unwrap();
            let input: Vec<f32> = (0..16000)
                .map(|n| (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 16000.0).sin() * 0.5)
                .collect();
            let mut out = Vec::new();
            let mut emit = |s: &[f32]| {
                out.extend_from_slice(s);
                Ok(true)
            };
            for block in input.chunks(320) {
                r.process(block, &mut emit).unwrap();
            }
            r.flush(&mut emit).unwrap();
            assert_eq!(out.len(), to as usize);
            let mid = &out[out.len() / 4..out.len() * 3 / 4];
            let rms = crate::dsp::rms(mid);
            assert!((rms - 0.5 / 2f32.sqrt()).abs() < 0.01, "rms {rms} at {to}");
            // Zero crossings give the frequency.
            let crossings = mid.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count() as f32;
            let freq = crossings / (mid.len() as f32 / to as f32);
            assert!((freq - 440.0).abs() < 3.0, "{freq} Hz at {to}");
        }
    }
}
