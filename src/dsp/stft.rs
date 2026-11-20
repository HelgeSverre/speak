//! Short-time Fourier transform and its inverse, on `realfft` (spec §9.1).

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::dsp::window::hann;

pub type Spectrum = Vec<Complex<f32>>;

pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Stft {
        let mut planner = RealFftPlanner::<f32>::new();
        Stft {
            n_fft,
            hop,
            window: hann(n_fft),
            forward: planner.plan_fft_forward(n_fft),
            inverse: planner.plan_fft_inverse(n_fft),
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Number of frames for a signal of `len` samples (centred frames).
    pub fn frames_for(&self, len: usize) -> usize {
        len / self.hop + 1
    }

    /// Centred frames with reflect padding: frame `i` is centred at sample `i × hop`.
    pub fn forward(&self, pcm: &[f32]) -> Vec<Spectrum> {
        let half = (self.n_fft / 2) as isize;
        let n = pcm.len() as isize;
        let at = |i: isize| -> f32 {
            if n == 0 {
                return 0.0;
            }
            // Reflect without repeating the edge sample.
            let mut j = i;
            if j < 0 {
                j = -j;
            }
            if j >= n {
                j = 2 * (n - 1) - j;
            }
            pcm[j.clamp(0, n - 1) as usize]
        };
        let mut buf = self.forward.make_input_vec();
        let mut out = Vec::with_capacity(self.frames_for(pcm.len()));
        for f in 0..self.frames_for(pcm.len()) {
            let centre = (f * self.hop) as isize;
            for (k, b) in buf.iter_mut().enumerate() {
                *b = at(centre - half + k as isize) * self.window[k];
            }
            let mut spec = self.forward.make_output_vec();
            // The buffers come from the plan, so the lengths always match.
            if self.forward.process(&mut buf, &mut spec).is_err() {
                spec.fill(Complex::new(0.0, 0.0));
            }
            out.push(spec);
        }
        out
    }

    /// One frame back to `n_fft` windowed samples.
    fn inverse_frame(&self, spec: &Spectrum, out: &mut [f32]) {
        let mut s = spec.clone();
        // realfft refuses a spectrum whose DC or Nyquist bin has an imaginary part.
        if let Some(first) = s.first_mut() {
            first.im = 0.0;
        }
        if let Some(last) = s.last_mut() {
            last.im = 0.0;
        }
        if self.inverse.process(&mut s, out).is_err() {
            out.fill(0.0);
        }
        // realfft is unnormalized.
        let scale = 1.0 / self.n_fft as f32;
        for (o, w) in out.iter_mut().zip(&self.window) {
            *o *= scale * w;
        }
    }

    /// Overlap-add with window-sum-square normalization; returns `len` samples.
    pub fn inverse(&self, frames: &[Spectrum], len: usize) -> Vec<f32> {
        let half = self.n_fft / 2;
        let total = len + self.n_fft;
        let mut acc = vec![0.0f32; total];
        let mut norm = vec![0.0f32; total];
        let mut buf = self.inverse.make_output_vec();
        for (f, spec) in frames.iter().enumerate() {
            let start = f * self.hop;
            if start + self.n_fft > total {
                break;
            }
            self.inverse_frame(spec, &mut buf);
            for k in 0..self.n_fft {
                acc[start + k] += buf[k];
                norm[start + k] += self.window[k] * self.window[k];
            }
        }
        (0..len)
            .map(|i| {
                if norm[i + half] > 1e-8 {
                    acc[i + half] / norm[i + half]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Streaming overlap-add: feed one spectrum, get `hop` samples; keeps an `n_fft − hop` tail.
pub struct OverlapAdd {
    stft: Stft,
    tail: Vec<f32>,
    norm: Vec<f32>,
    buf: Vec<f32>,
}

impl OverlapAdd {
    pub fn new(n_fft: usize, hop: usize) -> OverlapAdd {
        let stft = Stft::new(n_fft, hop);
        let buf = stft.inverse.make_output_vec();
        OverlapAdd {
            stft,
            tail: vec![0.0; n_fft],
            norm: vec![0.0; n_fft],
            buf,
        }
    }

    pub fn push(&mut self, spec: &Spectrum, out: &mut Vec<f32>) {
        let (n, hop) = (self.stft.n_fft, self.stft.hop);
        self.stft.inverse_frame(spec, &mut self.buf);
        for k in 0..n {
            self.tail[k] += self.buf[k];
            self.norm[k] += self.stft.window[k] * self.stft.window[k];
        }
        for k in 0..hop {
            out.push(if self.norm[k] > 1e-8 {
                self.tail[k] / self.norm[k]
            } else {
                0.0
            });
        }
        self.tail.copy_within(hop.., 0);
        self.norm.copy_within(hop.., 0);
        self.tail[n - hop..].fill(0.0);
        self.norm[n - hop..].fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Noise;

    #[test]
    fn round_trip() {
        let mut noise = Noise::new(4);
        let x: Vec<f32> = (0..16000)
            .map(|i| 0.4 * noise.sample() + 0.3 * (i as f32 * 0.05).sin())
            .collect();
        for (n_fft, hop) in [(1024, 256), (512, 128), (400, 100)] {
            let stft = Stft::new(n_fft, hop);
            let frames = stft.forward(&x);
            assert_eq!(frames.len(), stft.frames_for(x.len()));
            assert_eq!(frames[0].len(), n_fft / 2 + 1);
            let y = stft.inverse(&frames, x.len());
            let worst = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst <= 1e-5, "{n_fft}/{hop}: error {worst}");
        }
    }

    #[test]
    fn a_sine_lands_in_its_bin() {
        let x: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin())
            .collect();
        let stft = Stft::new(1024, 256);
        let frame = &stft.forward(&x)[10];
        let peak = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
            .unwrap()
            .0;
        assert_eq!(peak, 64);
    }

    #[test]
    fn streaming_matches_batch_after_the_startup() {
        let mut noise = Noise::new(8);
        let x: Vec<f32> = (0..8000).map(|_| noise.sample()).collect();
        let stft = Stft::new(512, 128);
        let frames = stft.forward(&x);
        let mut ola = OverlapAdd::new(512, 128);
        let mut y = Vec::new();
        for f in &frames {
            ola.push(f, &mut y);
        }
        // The streaming output is delayed by half a window.
        for i in 1000..6000 {
            assert!((y[i + 256] - x[i]).abs() < 1e-4, "sample {i}");
        }
    }
}
