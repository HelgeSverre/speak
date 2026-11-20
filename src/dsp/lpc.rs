//! Linear prediction: autocorrelation, Levinson-Durbin, LPC spectrum, formant peaks (spec §8.2.2).

use std::f32::consts::PI;

/// Result of analysing one frame.
#[derive(Clone, Debug)]
pub struct Lpc {
    /// Predictor polynomial `A(z) = 1 + a[1]z⁻¹ + … + a[p]z⁻ᵖ`; `a[0] == 1`.
    pub a: Vec<f32>,
    /// Reflection coefficients `k[1..=p]` stored at indices `0..p`.
    pub k: Vec<f32>,
    /// Prediction error energy.
    pub error: f32,
}

pub fn pre_emphasis(x: &[f32], coef: f32) -> Vec<f32> {
    let mut prev = 0.0;
    x.iter()
        .map(|v| {
            let y = v - coef * prev;
            prev = *v;
            y
        })
        .collect()
}

pub fn hamming(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (n.max(2) - 1) as f32).cos())
        .collect()
}

pub fn autocorrelation(x: &[f32], order: usize) -> Vec<f64> {
    (0..=order)
        .map(|lag| {
            x.iter()
                .zip(x.iter().skip(lag))
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum()
        })
        .collect()
}

/// Levinson-Durbin recursion on autocorrelation values `r[0..=p]`.
pub fn levinson(r: &[f64]) -> Lpc {
    let p = r.len() - 1;
    let mut a = vec![0.0f64; p + 1];
    a[0] = 1.0;
    let mut k = vec![0.0f32; p];
    let mut e = r[0];
    if e <= 0.0 {
        return Lpc {
            a: a.iter().map(|v| *v as f32).collect(),
            k,
            error: 0.0,
        };
    }
    for i in 1..=p {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let ki = (-acc / e).clamp(-0.999_999, 0.999_999);
        k[i - 1] = ki as f32;
        let prev = a.clone();
        for j in 1..i {
            a[j] = prev[j] + ki * prev[i - j];
        }
        a[i] = ki;
        e *= 1.0 - ki * ki;
        if e <= 0.0 {
            break;
        }
    }
    Lpc {
        a: a.iter().map(|v| *v as f32).collect(),
        k,
        error: e as f32,
    }
}

/// Windowed autocorrelation LPC of one frame (no pre-emphasis; apply it first if wanted).
pub fn analyze_frame(frame: &[f32], order: usize) -> Lpc {
    let w = hamming(frame.len());
    let x: Vec<f32> = frame.iter().zip(&w).map(|(a, b)| a * b).collect();
    let mut r = autocorrelation(&x, order);
    // A trace of white noise keeps the recursion well conditioned on synthetic signals.
    r[0] *= 1.0 + 1e-9;
    levinson(&r)
}

/// Magnitude of `1 / A(e^{jω})` in dB at `freq` Hz.
pub fn spectrum_db(a: &[f32], freq: f32, sample_rate: f32) -> f32 {
    let w = 2.0 * PI * freq / sample_rate;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (n, c) in a.iter().enumerate() {
        re += c * (w * n as f32).cos();
        im -= c * (w * n as f32).sin();
    }
    -10.0 * (re * re + im * im).max(1e-20).log10()
}

/// Peaks of the LPC spectrum between 90 Hz and 0.47 × sample rate, ascending, with parabolic
/// refinement. Returns `(frequency Hz, level dB)`.
pub fn spectral_peaks(a: &[f32], sample_rate: f32) -> Vec<(f32, f32)> {
    let step = 5.0;
    let n = (sample_rate * 0.47 / step) as usize;
    let s: Vec<f32> = (0..n)
        .map(|i| spectrum_db(a, i as f32 * step, sample_rate))
        .collect();
    let mut peaks = Vec::new();
    for i in 1..n - 1 {
        let f = i as f32 * step;
        if f >= 90.0 && s[i] > s[i - 1] && s[i] >= s[i + 1] {
            let d = s[i - 1] - 2.0 * s[i] + s[i + 1];
            let off = if d.abs() > 1e-9 {
                0.5 * (s[i - 1] - s[i + 1]) / d
            } else {
                0.0
            };
            peaks.push((f + off * step, s[i]));
        }
    }
    peaks
}

/// Formant estimate of a frame: pre-emphasis 0.97, Hamming, LPC of `order`, spectral peaks.
pub fn formants(frame: &[f32], sample_rate: f32, order: usize) -> Vec<f32> {
    let x = pre_emphasis(frame, 0.97);
    let lpc = analyze_frame(&x, order);
    spectral_peaks(&lpc.a, sample_rate)
        .into_iter()
        .map(|(f, _)| f)
        .collect()
}

/// Converts reflection coefficients to the predictor polynomial.
pub fn reflection_to_predictor(k: &[f32]) -> Vec<f32> {
    let mut a = vec![1.0f32];
    for (i, ki) in k.iter().enumerate() {
        let prev = a.clone();
        a.push(*ki);
        for j in 1..=i {
            a[j] = prev[j] + ki * prev[i + 1 - j];
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Noise;

    #[test]
    fn recovers_all_pole_filter() {
        let k_true = [-0.8f32, 0.6, -0.3, 0.2, -0.1, 0.05];
        let a = reflection_to_predictor(&k_true);
        let mut noise = Noise::new(7);
        let mut y = vec![0.0f32; 200_000];
        for n in 0..y.len() {
            let mut v = noise.sample();
            for j in 1..a.len() {
                if n >= j {
                    v -= a[j] * y[n - j];
                }
            }
            y[n] = v;
        }
        // Plain (unwindowed) autocorrelation over a long signal.
        let lpc = levinson(&autocorrelation(&y, k_true.len()));
        for (est, truth) in lpc.k.iter().zip(k_true) {
            assert!((est - truth).abs() < 2e-2, "{est} vs {truth}");
        }
        // Exact autocorrelation of the model gives the coefficients back within 1e-3:
        // compute it from a long impulse response.
        let mut h = vec![0.0f32; 4000];
        for n in 0..h.len() {
            let mut v = if n == 0 { 1.0 } else { 0.0 };
            for j in 1..a.len() {
                if n >= j {
                    v -= a[j] * h[n - j];
                }
            }
            h[n] = v;
        }
        let exact = levinson(&autocorrelation(&h, k_true.len()));
        for (est, truth) in exact.k.iter().zip(k_true) {
            assert!((est - truth).abs() < 1e-3, "{est} vs {truth}");
        }
    }

    #[test]
    fn predictor_round_trip() {
        let k = [0.5f32, -0.4, 0.3];
        let a = reflection_to_predictor(&k);
        assert_eq!(a.len(), 4);
        assert!((a[3] - 0.3).abs() < 1e-6);
    }
}

/// Analysis settings of spec §8.2.2.
pub const ORDER: usize = 18;
pub const PRE_EMPHASIS: f32 = 0.97;
pub const HOP_MS: f32 = 5.0;
pub const WINDOW_MS: f32 = 25.0;

/// One analysis frame: reflection coefficients, residual RMS, F0 and voicing.
#[derive(Clone, Debug, PartialEq)]
pub struct LpcFrame {
    pub k: [f32; ORDER],
    pub gain: f32,
    pub f0: f32,
    pub voicing: f32,
}

impl LpcFrame {
    pub const SILENT: LpcFrame = LpcFrame {
        k: [0.0; ORDER],
        gain: 0.0,
        f0: 0.0,
        voicing: 0.0,
    };

    /// Linear interpolation; reflection coefficients stay stable.
    pub fn lerp(&self, other: &LpcFrame, t: f32) -> LpcFrame {
        let mut out = self.clone();
        for i in 0..ORDER {
            out.k[i] = self.k[i] + (other.k[i] - self.k[i]) * t;
        }
        out.gain = self.gain + (other.gain - self.gain) * t;
        out.voicing = self.voicing + (other.voicing - self.voicing) * t;
        out.f0 = if self.f0 > 0.0 && other.f0 > 0.0 {
            self.f0 + (other.f0 - self.f0) * t
        } else {
            self.f0.max(other.f0)
        };
        out
    }
}

/// Frame-by-frame LPC analysis: pre-emphasis, 25 ms Hamming window every 5 ms, order 18.
/// Frame `i` is centred at `i × 5 ms`.
pub fn analyze(pcm: &[f32], sample_rate: u32) -> Vec<LpcFrame> {
    let sr = sample_rate as f32;
    let hop = (sr * HOP_MS / 1000.0) as usize;
    let win = (sr * WINDOW_MS / 1000.0) as usize;
    let x = pre_emphasis(pcm, PRE_EMPHASIS);
    let w = hamming(win);
    let w_energy: f32 = w.iter().map(|v| v * v).sum();
    let pitch = crate::dsp::pitch::track(pcm, sample_rate);
    let frames = pcm.len() / hop.max(1);
    let mut out = Vec::with_capacity(frames);
    let mut buf = vec![0.0f32; win];
    for i in 0..frames {
        let centre = (i * hop) as isize;
        for (j, b) in buf.iter_mut().enumerate() {
            let at = centre - (win / 2) as isize + j as isize;
            *b = if at >= 0 && (at as usize) < x.len() {
                x[at as usize] * w[j]
            } else {
                0.0
            };
        }
        let mut r = autocorrelation(&buf, ORDER);
        let mut frame = LpcFrame::SILENT;
        if r[0] > 1e-9 {
            // A trace of white noise keeps the recursion well conditioned.
            r[0] *= 1.0 + 1e-6;
            let lpc = levinson(&r);
            for (dst, src) in frame.k.iter_mut().zip(&lpc.k) {
                *dst = src.clamp(-0.999, 0.999);
            }
            // Residual RMS: prediction error energy over the window energy.
            frame.gain = (lpc.error.max(0.0) / w_energy).sqrt();
        }
        if let Some(p) = pitch.get(i) {
            frame.f0 = p.f0;
            frame.voicing = p.voicing;
        }
        out.push(frame);
    }
    out
}

#[cfg(test)]
mod analysis_tests {
    use super::*;
    use crate::dsp::Noise;
    use crate::dsp::lattice::{InverseLattice, Lattice};

    #[test]
    fn residual_reconstructs_the_input() {
        // Analysis → inverse filter → synthesis with the true residual gives the input back.
        let mut noise = Noise::new(9);
        let k_true = [-0.7f32, 0.5, -0.2, 0.1];
        let mut shaper = Lattice::new(4);
        let x: Vec<f32> = (0..16000)
            .map(|_| shaper.process(noise.sample() * 0.1, &k_true))
            .collect();
        let frames = analyze(&x, 16000);
        assert_eq!(frames.len(), 200);
        let pre = pre_emphasis(&x, PRE_EMPHASIS);
        let (mut inv, mut syn) = (InverseLattice::new(ORDER), Lattice::new(ORDER));
        let mut y = Vec::with_capacity(x.len());
        let mut de = 0.0f32;
        for (n, s) in pre.iter().enumerate() {
            let pos = n as f32 / 80.0;
            let i = (pos as usize).min(frames.len() - 1);
            let f = frames[i].lerp(&frames[(i + 1).min(frames.len() - 1)], pos.fract());
            let e = inv.process(*s, &f.k);
            let v = syn.process(e, &f.k);
            de = v + PRE_EMPHASIS * de;
            y.push(de);
        }
        let signal: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
        let err: f64 = x
            .iter()
            .zip(&y)
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        let snr = 10.0 * (signal / err.max(1e-30)).log10();
        assert!(snr >= 60.0, "SNR {snr} dB");
    }

    #[test]
    fn gain_is_the_residual_rms() {
        // White noise is its own residual: gain ≈ its RMS (after pre-emphasis).
        let mut noise = Noise::new(2);
        let x: Vec<f32> = (0..16000).map(|_| noise.sample() * 0.2).collect();
        let frames = analyze(&x, 16000);
        let rms = crate::dsp::rms(&pre_emphasis(&x, PRE_EMPHASIS));
        let mean: f32 = frames[20..180].iter().map(|f| f.gain).sum::<f32>() / 160.0;
        assert!((mean / rms - 1.0).abs() < 0.35, "gain {mean} vs rms {rms}");
        assert!(frames[20..180].iter().all(|f| f.voicing < 0.6));
    }
}
