//! Pitch marks and TD-PSOLA (spec §9.3): two-period Hann grains at the pitch marks,
//! overlap-added at new spacings.

use crate::dsp::pitch::{self, Pitch};
use crate::error::{Error, Result};

/// Marks in unvoiced regions.
const UNVOICED_STEP_MS: f32 = 5.0;

/// One pitch mark per glottal period in voiced regions (from the F0 track, refined to the
/// nearest local energy peak), and one every 5 ms elsewhere. Returns (sample positions, voiced flags).
pub fn epochs(pcm: &[f32], sample_rate: u32, track: &[Pitch]) -> (Vec<usize>, Vec<bool>) {
    let sr = sample_rate as f32;
    let hop = (sr * pitch::HOP_MS / 1000.0) as usize;
    let f0_at = |sample: usize| track.get(sample / hop.max(1)).map(|p| p.f0).unwrap_or(0.0);
    let step = (sr * UNVOICED_STEP_MS / 1000.0) as usize;
    // Smoothed energy: its peaks sit at the main excitation of each period.
    let half = (sr * 0.001) as usize;
    let energy: Vec<f32> = (0..pcm.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(pcm.len()));
            pcm[a..b].iter().map(|v| v * v).sum::<f32>()
        })
        .collect();
    let peak_in = |a: usize, b: usize| -> usize {
        let (a, b) = (a.min(pcm.len().saturating_sub(1)), b.min(pcm.len()));
        (a..b.max(a + 1))
            .max_by(|x, y| energy[*x].total_cmp(&energy[*y]))
            .unwrap_or(a)
    };
    let (mut marks, mut voiced) = (Vec::new(), Vec::new());
    let mut pos = 0usize;
    let mut last_voiced: Option<usize> = None;
    while pos < pcm.len() {
        let f0 = f0_at(pos);
        if f0 > 0.0 {
            let period = (sr / f0) as usize;
            let mark = match last_voiced {
                // Expect the next mark one period on; search ±20 % around it.
                Some(prev) => {
                    let expect = prev + period;
                    let slack = (period / 5).max(1);
                    peak_in(expect.saturating_sub(slack), expect + slack + 1)
                }
                // First mark of a voiced stretch: the strongest point of the first period.
                None => peak_in(pos, pos + period),
            };
            if mark >= pcm.len() {
                break;
            }
            marks.push(mark);
            voiced.push(true);
            last_voiced = Some(mark);
            pos = mark + period / 2;
        } else {
            marks.push(pos);
            voiced.push(false);
            last_voiced = None;
            pos += step.max(1);
        }
    }
    (marks, voiced)
}

/// Checks the factors PSOLA can handle.
pub fn check_factors(time: f32, pitch: f32) -> Result<()> {
    for (name, f) in [("time", time), ("pitch", pitch)] {
        if !(0.5..=2.0).contains(&f) {
            return Err(Error::unsupported(format!(
                "psola: {name} factor {f:.2} is outside 0.5–2.0"
            )));
        }
    }
    Ok(())
}

/// TD-PSOLA. `time`: output duration ÷ input duration. `pitch`: output F0 ÷ input F0.
pub fn psola(
    pcm: &[f32],
    marks: &[usize],
    voiced: &[bool],
    time: f32,
    pitch: f32,
) -> Result<Vec<f32>> {
    check_factors(time, pitch)?;
    let out_len = (pcm.len() as f32 * time).round() as usize;
    let mut out = vec![0.0f32; out_len];
    if marks.len() < 2 {
        return Ok(out);
    }
    // Local period at each mark.
    let period = |i: usize| -> usize {
        let next = if i + 1 < marks.len() {
            marks[i + 1] - marks[i]
        } else {
            marks[i] - marks[i - 1]
        };
        next.max(1)
    };
    // Start on the first mark's own position, so unchanged factors reproduce the input.
    let mut t_out = marks[0] as f32 * time;
    let mut src = 0usize;
    let mut last_src = usize::MAX;
    let mut repeats = 0usize;
    while (t_out as usize) < out_len {
        // The source mark nearest to this point of the time-scaled axis.
        let t_src = t_out / time;
        while src + 1 < marks.len()
            && (marks[src + 1] as f32 - t_src).abs() <= (marks[src] as f32 - t_src).abs()
        {
            src += 1;
        }
        let p = period(src);
        let centre = marks[src];
        if src == last_src {
            repeats += 1;
        } else {
            repeats = 0;
            last_src = src;
        }
        // Repeated unvoiced grains sound tonal: reverse every other one.
        let reverse = !voiced[src] && repeats % 2 == 1;
        let at = t_out.round() as isize;
        for k in -(p as isize)..=(p as isize) {
            let s = if reverse {
                centre as isize - k
            } else {
                centre as isize + k
            };
            let o = at + k;
            if s < 0 || s >= pcm.len() as isize || o < 0 || o >= out_len as isize {
                continue;
            }
            // Two-period Hann window centred on the mark.
            let w = 0.5 + 0.5 * (std::f32::consts::PI * k as f32 / p as f32).cos();
            out[o as usize] += pcm[s as usize] * w;
        }
        // Voiced: the period shrinks or grows with the pitch factor. Unvoiced: marks keep their spacing.
        t_out += if voiced[src] {
            p as f32 / pitch
        } else {
            p as f32
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pulses(f0: f32, n: usize) -> Vec<f32> {
        let period = 16000.0 / f0;
        (0..n)
            .map(|i| {
                let ph = (i as f32 % period) / period;
                (-8.0 * ph).exp() * (2.0 * std::f32::consts::PI * 5.0 * ph).sin()
            })
            .collect()
    }

    fn median_f0(pcm: &[f32]) -> f32 {
        let mut v: Vec<f32> = pitch::track(pcm, 16000)
            .iter()
            .map(|p| p.f0)
            .filter(|f| *f > 0.0)
            .collect();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    }

    #[test]
    fn marks_are_one_per_period() {
        let x = pulses(125.0, 16000);
        let track = pitch::track(&x, 16000);
        let (marks, voiced) = epochs(&x, 16000, &track);
        let v: Vec<usize> = marks
            .iter()
            .zip(&voiced)
            .filter(|(_, v)| **v)
            .map(|(m, _)| *m)
            .collect();
        assert!((v.len() as i32 - 125).abs() <= 4, "{} marks", v.len());
        let consistent = v
            .windows(2)
            .filter(|w| ((w[1] - w[0]) as i32 - 128).abs() <= 2)
            .count();
        assert!(consistent as f32 >= 0.95 * (v.len() - 1) as f32);
    }

    #[test]
    fn pitch_and_time_change_independently() {
        let x = pulses(120.0, 16000);
        let track = pitch::track(&x, 16000);
        let (marks, voiced) = epochs(&x, 16000, &track);
        // +4 semitones: F0 × 1.26, same duration.
        let up = psola(&x, &marks, &voiced, 1.0, 2f32.powf(4.0 / 12.0)).unwrap();
        assert!((up.len() as f32 / x.len() as f32 - 1.0).abs() < 0.02);
        let ratio = median_f0(&up) / median_f0(&x);
        assert!((ratio - 1.26).abs() < 0.04, "pitch ratio {ratio}");
        // 1.5 × longer: same F0.
        let slow = psola(&x, &marks, &voiced, 1.5, 1.0).unwrap();
        assert!((slow.len() as f32 / x.len() as f32 - 1.5).abs() < 0.03);
        assert!((median_f0(&slow) / median_f0(&x) - 1.0).abs() < 0.02);
        // Identity keeps the signal.
        let same = psola(&x, &marks, &voiced, 1.0, 1.0).unwrap();
        let err: f32 = x[2000..14000]
            .iter()
            .zip(&same[2000..14000])
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let sig: f32 = x[2000..14000].iter().map(|a| a * a).sum();
        assert!(10.0 * (sig / err.max(1e-12)).log10() > 20.0);
    }

    #[test]
    fn out_of_range_factors_are_refused() {
        assert!(psola(&[0.0; 100], &[10, 50], &[false, false], 2.5, 1.0).is_err());
        assert!(psola(&[0.0; 100], &[10, 50], &[false, false], 1.0, 0.3).is_err());
        assert_eq!(check_factors(3.0, 1.0).unwrap_err().exit_code(), 2);
    }
}
