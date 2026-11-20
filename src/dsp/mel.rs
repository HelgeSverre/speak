//! Mel filterbank, log-mel and MFCC (spec §9.1).

use std::f32::consts::PI;

/// Slaney's mel scale: linear below 1 kHz, logarithmic above.
pub fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

pub fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * F_SP
    }
}

/// HTK's mel scale, as an option.
pub fn hz_to_mel_htk(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

pub fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Triangular filters over the bins of an `n_fft` transform. `htk = false`: Slaney scale with
/// area normalization.
pub struct MelBank {
    pub n_mels: usize,
    pub bins: usize,
    /// `n_mels × bins`, row-major.
    pub weights: Vec<f32>,
}

impl MelBank {
    pub fn new(
        n_mels: usize,
        n_fft: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: f32,
        htk: bool,
    ) -> MelBank {
        let bins = n_fft / 2 + 1;
        type Scale = fn(f32) -> f32;
        let (to_mel, to_hz): (Scale, Scale) = if htk {
            (hz_to_mel_htk, mel_to_hz_htk)
        } else {
            (hz_to_mel, mel_to_hz)
        };
        let (lo, hi) = (to_mel(fmin), to_mel(fmax));
        let edges: Vec<f32> = (0..n_mels + 2)
            .map(|i| to_hz(lo + (hi - lo) * i as f32 / (n_mels + 1) as f32))
            .collect();
        let mut weights = vec![0.0f32; n_mels * bins];
        for m in 0..n_mels {
            let (left, centre, right) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = if htk { 1.0 } else { 2.0 / (right - left) };
            for k in 0..bins {
                let f = k as f32 * sample_rate as f32 / n_fft as f32;
                let up = (f - left) / (centre - left);
                let down = (right - f) / (right - centre);
                weights[m * bins + k] = up.min(down).max(0.0) * norm;
            }
        }
        MelBank {
            n_mels,
            bins,
            weights,
        }
    }

    pub fn row(&self, m: usize) -> &[f32] {
        &self.weights[m * self.bins..(m + 1) * self.bins]
    }

    /// Mel energies of one magnitude (or power) spectrum.
    pub fn apply(&self, spectrum: &[f32]) -> Vec<f32> {
        (0..self.n_mels)
            .map(|m| self.row(m).iter().zip(spectrum).map(|(w, s)| w * s).sum())
            .collect()
    }
}

pub fn log_mel(mel: &[f32]) -> Vec<f32> {
    mel.iter().map(|v| v.max(1e-10).ln()).collect()
}

/// DCT-II of a log-mel vector: the first `n` cepstral coefficients (orthonormal scaling).
pub fn mfcc(log_mel: &[f32], n: usize) -> Vec<f32> {
    let m = log_mel.len() as f32;
    (0..n)
        .map(|k| {
            let scale = if k == 0 {
                (1.0 / m).sqrt()
            } else {
                (2.0 / m).sqrt()
            };
            scale
                * log_mel
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * (PI * k as f32 * (i as f32 + 0.5) / m).cos())
                    .sum::<f32>()
        })
        .collect()
}

/// First-order regression deltas over ±2 frames.
pub fn deltas(frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = frames.len();
    (0..n)
        .map(|t| {
            let at = |d: isize| &frames[(t as isize + d).clamp(0, n as isize - 1) as usize];
            (0..frames[t].len())
                .map(|c| (1.0 * (at(1)[c] - at(-1)[c]) + 2.0 * (at(2)[c] - at(-2)[c])) / 10.0)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_invert() {
        for hz in [0.0, 200.0, 999.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 0.5);
            assert!((mel_to_hz_htk(hz_to_mel_htk(hz)) - hz).abs() < 0.5);
        }
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-3);
    }

    #[test]
    fn filterbank_rows_peak_in_order_and_cover_the_band() {
        let bank = MelBank::new(80, 1024, 16000, 0.0, 8000.0, false);
        let mut last_peak = 0;
        for m in 0..80 {
            let row = bank.row(m);
            assert!(row.iter().all(|w| *w >= 0.0));
            let peak = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert!(peak >= last_peak, "row {m}");
            last_peak = peak;
            // Slaney normalization: each triangle has (about) unit area in Hz.
            let area: f32 = row.iter().sum::<f32>() * 16000.0 / 1024.0;
            assert!((area - 1.0).abs() < 0.35, "row {m}: area {area}");
        }
        let htk = MelBank::new(40, 512, 16000, 0.0, 8000.0, true);
        assert!((0..40).all(|m| htk.row(m).iter().cloned().fold(0.0f32, f32::max) <= 1.0 + 1e-6));
    }

    #[test]
    fn mfcc_of_a_flat_spectrum_is_only_c0() {
        let c = mfcc(&[2.0; 40], 13);
        assert!((c[0] - 2.0 * 40f32.sqrt()).abs() < 1e-3);
        assert!(c[1..].iter().all(|v| v.abs() < 1e-4));
        // A pure tone gives a peaked mel vector, so higher coefficients are not zero.
        let bank = MelBank::new(40, 512, 16000, 0.0, 8000.0, false);
        let mut spec = vec![1e-6f32; 257];
        spec[32] = 1.0;
        let tone = mfcc(&log_mel(&bank.apply(&spec)), 13);
        assert!(tone[1..].iter().any(|v| v.abs() > 0.5));
    }

    #[test]
    fn deltas_of_a_ramp_are_constant() {
        let frames: Vec<Vec<f32>> = (0..10).map(|t| vec![t as f32 * 2.0]).collect();
        let d = deltas(&frames);
        assert!(d[3..7].iter().all(|v| (v[0] - 2.0).abs() < 1e-5));
    }
}
