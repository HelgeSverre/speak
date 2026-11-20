//! Decimating FIR low-pass for the oversampled glottal source (spec §8.1.1).

use std::f32::consts::PI;

/// Hann-windowed sinc, `taps` long, cutoff as a fraction of the input sample rate, unity DC gain.
pub fn windowed_sinc(taps: usize, cutoff: f32) -> Vec<f32> {
    let m = (taps - 1) as f32 / 2.0;
    let mut h: Vec<f32> = (0..taps)
        .map(|n| {
            let x = n as f32 - m;
            let sinc = if x.abs() < 1e-6 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * x).sin() / (PI * x)
            };
            let w = 0.5 - 0.5 * (2.0 * PI * (n as f32 + 0.5) / taps as f32).cos();
            sinc * w
        })
        .collect();
    let sum: f32 = h.iter().sum();
    for v in &mut h {
        *v /= sum;
    }
    h
}

/// Integer-factor decimator with a ring-buffer delay line.
#[derive(Clone, Debug)]
pub struct Decimator {
    h: Vec<f32>,
    buf: Vec<f32>,
    pos: usize,
    factor: usize,
    phase: usize,
}

impl Decimator {
    /// `cutoff_hz` relative to the output rate `out_rate`; the input rate is `factor × out_rate`.
    pub fn new(factor: usize, taps: usize, cutoff_hz: f32, out_rate: f32) -> Decimator {
        let h = windowed_sinc(taps, cutoff_hz / (out_rate * factor as f32));
        Decimator {
            buf: vec![0.0; taps],
            h,
            pos: 0,
            factor,
            phase: 0,
        }
    }

    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.phase = 0;
    }

    /// Feeds one input sample; returns an output sample every `factor` inputs.
    #[inline]
    pub fn push(&mut self, x: f32) -> Option<f32> {
        let n = self.buf.len();
        self.buf[self.pos] = x;
        self.pos = (self.pos + 1) % n;
        self.phase += 1;
        if self.phase < self.factor {
            return None;
        }
        self.phase = 0;
        let (tail, head) = self.buf.split_at(self.pos);
        let mut acc = 0.0;
        // Oldest sample is at `pos`; the filter is symmetric, so order does not matter.
        for (c, s) in self.h.iter().zip(head.iter().chain(tail.iter())) {
            acc += c * s;
        }
        Some(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gain_at(freq: f32) -> f32 {
        let mut d = Decimator::new(4, 32, 7200.0, 16000.0);
        let mut out = Vec::new();
        for n in 0..64000 {
            if let Some(y) = d.push((2.0 * PI * freq * n as f32 / 64000.0).sin()) {
                out.push(y);
            }
        }
        crate::dsp::rms(&out[100..]) * 2f32.sqrt()
    }

    #[test]
    fn passes_low_and_rejects_high() {
        assert!((gain_at(1000.0) - 1.0).abs() < 0.02);
        assert!(gain_at(4000.0) > 0.8);
        assert!(gain_at(16000.0) < 0.02);
        assert!(gain_at(24000.0) < 0.02);
    }

    #[test]
    fn output_rate() {
        let mut d = Decimator::new(4, 32, 7200.0, 16000.0);
        let n = (0..640).filter(|_| d.push(1.0).is_some()).count();
        assert_eq!(n, 160);
    }
}
