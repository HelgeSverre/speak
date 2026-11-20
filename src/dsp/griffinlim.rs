//! Mel spectrogram → waveform: filterbank pseudo-inverse, then Fast Griffin-Lim (spec §9.2).

use realfft::num_complex::Complex;

use crate::dsp::Noise;
use crate::dsp::mel::MelBank;
use crate::dsp::stft::{Spectrum, Stft};

pub const N_FFT: usize = 1024;
pub const HOP: usize = 256;
pub const N_MELS: usize = 80;
pub const ITERATIONS: usize = 32;
pub const MOMENTUM: f32 = 0.99;

/// Solves `A·X = B` for symmetric positive-definite `A` (n × n) by Gaussian elimination with
/// partial pivoting. `B` has `m` columns. Returns `None` when `A` is singular.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize, m: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot =
            (col..n).max_by(|x, y| a[x * n + col].abs().total_cmp(&a[y * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            for k in 0..m {
                b.swap(col * m + k, pivot * m + k);
            }
        }
        let d = a[col * n + col];
        for row in col + 1..n {
            let f = a[row * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            for k in 0..m {
                b[row * m + k] -= f * b[col * m + k];
            }
        }
    }
    for col in (0..n).rev() {
        let d = a[col * n + col];
        for k in 0..m {
            b[col * m + k] /= d;
        }
        for row in 0..col {
            let f = a[row * n + col];
            if f != 0.0 {
                for k in 0..m {
                    b[row * m + k] -= f * b[col * m + k];
                }
            }
        }
    }
    Some(b)
}

/// Pseudo-inverse of the filterbank, `Mᵀ(MMᵀ)⁻¹`: `bins × n_mels`, row-major.
pub fn pseudo_inverse(bank: &MelBank) -> Vec<f32> {
    let (n, bins) = (bank.n_mels, bank.bins);
    let mut mmt = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v: f64 = bank
                .row(i)
                .iter()
                .zip(bank.row(j))
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            mmt[i * n + j] = v;
            mmt[j * n + i] = v;
        }
        // A little regularization keeps empty or tiny filters from blowing up.
        mmt[i * n + i] += 1e-9;
    }
    // Solve (MMᵀ)·X = M, then the pseudo-inverse is Xᵀ.
    let m: Vec<f64> = bank.weights.iter().map(|v| *v as f64).collect();
    let x = solve(mmt, m, n, bins).unwrap_or_else(|| vec![0.0; n * bins]);
    let mut out = vec![0.0f32; bins * n];
    for i in 0..n {
        for k in 0..bins {
            out[k * n + i] = x[i * bins + k] as f32;
        }
    }
    out
}

/// Linear magnitudes from mel energies; the pseudo-inverse can go negative, so clamp.
pub fn mel_to_linear(mel: &[Vec<f32>], pinv: &[f32], bins: usize) -> Vec<Vec<f32>> {
    mel.iter()
        .map(|frame| {
            let n = frame.len();
            (0..bins)
                .map(|k| {
                    (0..n)
                        .map(|i| pinv[k * n + i] * frame[i])
                        .sum::<f32>()
                        .max(1e-5)
                })
                .collect()
        })
        .collect()
}

/// Fast Griffin-Lim (Perraudin et al. 2013): iterates projections with momentum.
pub fn griffin_lim(
    magnitude: &[Vec<f32>],
    stft: &Stft,
    len: usize,
    iterations: usize,
    momentum: f32,
) -> Vec<f32> {
    if magnitude.is_empty() {
        return vec![0.0; len];
    }
    let mut noise = Noise::new(0x6121_FF1A);
    // Random initial phases, deterministic.
    let mut angles: Vec<Spectrum> = magnitude
        .iter()
        .map(|f| {
            f.iter()
                .map(|_| {
                    let phi = std::f32::consts::PI * noise.sample();
                    Complex::new(phi.cos(), phi.sin())
                })
                .collect()
        })
        .collect();
    let apply = |angles: &[Spectrum]| -> Vec<Spectrum> {
        magnitude
            .iter()
            .zip(angles)
            .map(|(m, a)| m.iter().zip(a).map(|(m, a)| a * *m).collect())
            .collect()
    };
    let mut previous: Option<Vec<Spectrum>> = None;
    for _ in 0..iterations {
        let pcm = stft.inverse(&apply(&angles), len);
        let mut rebuilt = stft.forward(&pcm);
        rebuilt.truncate(magnitude.len());
        while rebuilt.len() < magnitude.len() {
            rebuilt.push(vec![Complex::new(0.0, 0.0); stft.bins()]);
        }
        for (f, frame) in rebuilt.iter().enumerate() {
            for (k, c) in frame.iter().enumerate() {
                let mut t = *c;
                if let Some(prev) = &previous {
                    t += (*c - prev[f][k]) * momentum;
                }
                let norm = t.norm();
                angles[f][k] = if norm > 1e-12 {
                    t / norm
                } else {
                    Complex::new(1.0, 0.0)
                };
            }
        }
        previous = Some(rebuilt);
    }
    stft.inverse(&apply(&angles), len)
}

/// Magnitude spectrogram and the mel energies of a signal.
pub fn mel_spectrogram(pcm: &[f32], stft: &Stft, bank: &MelBank) -> Vec<Vec<f32>> {
    stft.forward(pcm)
        .iter()
        .map(|f| bank.apply(&f.iter().map(|c| c.norm()).collect::<Vec<_>>()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_inverse_recovers_a_smooth_spectrum() {
        let bank = MelBank::new(N_MELS, N_FFT, 16000, 0.0, 8000.0, false);
        let pinv = pseudo_inverse(&bank);
        // M · M⁺ · mel == mel for any mel vector in the range.
        let spec: Vec<f32> = (0..bank.bins)
            .map(|k| 1.0 + (k as f32 * 0.02).sin())
            .collect();
        let mel = bank.apply(&spec);
        let lin = mel_to_linear(std::slice::from_ref(&mel), &pinv, bank.bins);
        let back = bank.apply(&lin[0]);
        for (a, b) in mel.iter().zip(&back) {
            // The clamp to non-negative magnitudes costs a little at the narrow low filters.
            assert!((a - b).abs() / a.max(1e-6) < 0.12, "{a} vs {b}");
        }
    }

    #[test]
    fn a_sine_survives_mel_and_griffin_lim() {
        let x: Vec<f32> = (0..16000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let stft = Stft::new(N_FFT, HOP);
        let bank = MelBank::new(N_MELS, N_FFT, 16000, 0.0, 8000.0, false);
        let mel = mel_spectrogram(&x, &stft, &bank);
        let lin = mel_to_linear(&mel, &pseudo_inverse(&bank), bank.bins);
        let y = griffin_lim(&lin, &stft, x.len(), ITERATIONS, MOMENTUM);
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
        let frame = &stft.forward(&y)[20];
        let peak = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
            .unwrap()
            .0;
        // 440 Hz is bin 28.16 of 1024 at 16 kHz.
        assert!((peak as i32 - 28).abs() <= 1, "peak in bin {peak}");
    }
}
