//! F0 and voicing by normalized autocorrelation (spec §8.2.2).

use crate::dsp::fir::windowed_sinc;

pub const HOP_MS: f32 = 5.0;
const WINDOW_MS: f32 = 40.0;
const F0_MIN: f32 = 60.0;
const F0_MAX: f32 = 400.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pitch {
    /// Hz; 0 when unvoiced.
    pub f0: f32,
    /// Normalized autocorrelation peak, 0..1.
    pub voicing: f32,
}

fn lowpass(x: &[f32], cutoff_hz: f32, sample_rate: f32) -> Vec<f32> {
    let h = windowed_sinc(65, cutoff_hz / sample_rate);
    let half = h.len() / 2;
    (0..x.len())
        .map(|n| {
            let mut acc = 0.0;
            for (k, c) in h.iter().enumerate() {
                let i = n as isize + k as isize - half as isize;
                if i >= 0 && (i as usize) < x.len() {
                    acc += c * x[i as usize];
                }
            }
            acc
        })
        .collect()
}

fn median5(v: &[f32], i: usize) -> f32 {
    let lo = i.saturating_sub(2);
    let hi = (i + 3).min(v.len());
    let mut w: Vec<f32> = v[lo..hi].iter().copied().filter(|x| *x > 0.0).collect();
    if w.is_empty() {
        return 0.0;
    }
    w.sort_by(f32::total_cmp);
    w[w.len() / 2]
}

/// One value per 5 ms hop; frame `i` is centred at `i × 5 ms`.
pub fn track(pcm: &[f32], sample_rate: u32) -> Vec<Pitch> {
    let sr = sample_rate as f32;
    let x = lowpass(pcm, 900.0, sr);
    let hop = (sr * HOP_MS / 1000.0) as usize;
    let win = (sr * WINDOW_MS / 1000.0) as usize;
    let (lag_min, lag_max) = ((sr / F0_MAX) as usize, (sr / F0_MIN) as usize);
    let frames = pcm.len() / hop.max(1);
    let level = crate::dsp::rms(&x).max(1e-6);
    let mut raw = Vec::with_capacity(frames);
    for i in 0..frames {
        let centre = i * hop;
        let start = centre.saturating_sub(win / 2);
        let end = (start + win).min(x.len());
        let frame = &x[start..end];
        if frame.len() <= lag_max + 2 || crate::dsp::rms(frame) < 0.02 * level {
            raw.push(Pitch {
                f0: 0.0,
                voicing: 0.0,
            });
            continue;
        }
        let r = |lag: usize| -> f32 {
            let n = frame.len() - lag;
            let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
            for k in 0..n {
                let (a, b) = (frame[k] as f64, frame[k + lag] as f64);
                ab += a * b;
                aa += a * a;
                bb += b * b;
            }
            if aa <= 0.0 || bb <= 0.0 {
                0.0
            } else {
                (ab / (aa * bb).sqrt()) as f32
            }
        };
        let values: Vec<f32> = (lag_min - 1..=lag_max + 1).map(r).collect();
        // The first strong peak, not the global one: avoids octave-down errors.
        let global = values[1..values.len() - 1]
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max);
        let mut best = 1;
        for k in 1..values.len() - 1 {
            if values[k] >= values[k - 1] && values[k] >= values[k + 1] && values[k] >= 0.9 * global
            {
                best = k;
                break;
            }
        }
        let (a, b, c) = (values[best - 1], values[best], values[best + 1]);
        let d = a - 2.0 * b + c;
        let offset = if d.abs() > 1e-9 {
            0.5 * (a - c) / d
        } else {
            0.0
        };
        let lag = (lag_min - 1 + best) as f32 + offset.clamp(-1.0, 1.0);
        let voicing = b.clamp(0.0, 1.0);
        raw.push(Pitch {
            f0: if voicing > 0.5 { sr / lag } else { 0.0 },
            voicing,
        });
    }
    let f0s: Vec<f32> = raw.iter().map(|p| p.f0).collect();
    raw.iter()
        .enumerate()
        .map(|(i, p)| Pitch {
            f0: if p.f0 > 0.0 { median5(&f0s, i) } else { 0.0 },
            voicing: p.voicing,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_a_pulse_train_and_rejects_noise() {
        let sr = 16000u32;
        for f0 in [90.0f32, 140.0, 250.0] {
            let period = sr as f32 / f0;
            let x: Vec<f32> = (0..sr as usize)
                .map(|n| {
                    let ph = (n as f32 % period) / period;
                    (1.0 - ph) * (2.0 * std::f32::consts::PI * 3.0 * ph).sin()
                })
                .collect();
            let t = track(&x, sr);
            let mid = &t[20..t.len() - 20];
            let ok = mid.iter().filter(|p| (p.f0 - f0).abs() / f0 < 0.02).count();
            assert!(
                ok as f32 >= 0.95 * mid.len() as f32,
                "{f0}: {ok}/{}",
                mid.len()
            );
        }
        let mut noise = crate::dsp::Noise::new(3);
        let n: Vec<f32> = (0..16000).map(|_| noise.sample()).collect();
        let t = track(&n, sr);
        assert!(t.iter().filter(|p| p.f0 > 0.0).count() < t.len() / 10);
    }
}
