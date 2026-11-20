//! Dynamic time warping with a band constraint (spec §9.1).

/// Alignment of sequence `a` (length n) to `b` (length m).
pub struct Warp {
    /// Pairs `(i, j)` from `(0, 0)` to `(n−1, m−1)`.
    pub path: Vec<(usize, usize)>,
    /// Mean local cost along the path.
    pub cost: f32,
}

fn distance(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt()
}

/// Euclidean local cost, symmetric steps, Sakoe-Chiba band of `band` frames around the
/// diagonal (which is stretched when the lengths differ). `None` when a sequence is empty.
pub fn dtw(a: &[Vec<f32>], b: &[Vec<f32>], band: usize) -> Option<Warp> {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return None;
    }
    let slope = m as f32 / n as f32;
    // The band must at least cover the length difference.
    let band = band.max(1) as isize;
    let lo = |i: usize| ((i as f32 * slope) as isize - band).clamp(0, m as isize - 1) as usize;
    let hi = |i: usize| ((i as f32 * slope) as isize + band).clamp(0, m as isize - 1) as usize;
    const INF: f32 = f32::INFINITY;
    let mut d = vec![INF; n * m];
    // 0 = diagonal, 1 = from (i−1, j), 2 = from (i, j−1).
    let mut step = vec![0u8; n * m];
    for i in 0..n {
        for j in lo(i)..=hi(i) {
            let c = distance(&a[i], &b[j]);
            let (best, s) = if i == 0 && j == 0 {
                (0.0, 0)
            } else {
                let diag = if i > 0 && j > 0 {
                    d[(i - 1) * m + j - 1]
                } else {
                    INF
                };
                let up = if i > 0 { d[(i - 1) * m + j] } else { INF };
                let left = if j > 0 { d[i * m + j - 1] } else { INF };
                if diag <= up && diag <= left {
                    (diag, 0)
                } else if up <= left {
                    (up, 1)
                } else {
                    (left, 2)
                }
            };
            d[i * m + j] = best + c;
            step[i * m + j] = s;
        }
    }
    if !d[n * m - 1].is_finite() {
        return None;
    }
    let mut path = vec![(n - 1, m - 1)];
    let (mut i, mut j) = (n - 1, m - 1);
    while i > 0 || j > 0 {
        match step[i * m + j] {
            0 if i > 0 && j > 0 => {
                i -= 1;
                j -= 1;
            }
            1 if i > 0 => i -= 1,
            _ if j > 0 => j -= 1,
            _ => i -= 1,
        }
        path.push((i, j));
    }
    path.reverse();
    let cost = d[n * m - 1] / path.len() as f32;
    Some(Warp { path, cost })
}

impl Warp {
    /// Position in `b` that frame `i` of `a` maps to: the mean of its matches.
    pub fn map(&self, i: usize) -> f32 {
        let js: Vec<usize> = self.path.iter().filter(|p| p.0 == i).map(|p| p.1).collect();
        if js.is_empty() {
            return self.path.last().map(|p| p.1 as f32).unwrap_or(0.0);
        }
        js.iter().sum::<usize>() as f32 / js.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(n: usize, stretch: f32) -> Vec<Vec<f32>> {
        // A feature that changes non-uniformly in time.
        (0..n)
            .map(|i| {
                let t = i as f32 / stretch;
                vec![
                    (t * 0.11).sin() * 3.0,
                    (t * 0.037).cos() * 2.0,
                    (t * 0.005).powi(2),
                ]
            })
            .collect()
    }

    #[test]
    fn recovers_a_linear_stretch() {
        let a = sequence(200, 1.0);
        let b = sequence(260, 1.3);
        let w = dtw(&a, &b, 60).unwrap();
        assert_eq!(w.path.first(), Some(&(0, 0)));
        assert_eq!(w.path.last(), Some(&(199, 259)));
        for i in (10..190).step_by(10) {
            assert!(
                (w.map(i) - i as f32 * 1.3).abs() <= 2.0,
                "frame {i} → {}",
                w.map(i)
            );
        }
        assert!(w.cost < 0.2);
        // Monotonic.
        assert!(
            w.path
                .windows(2)
                .all(|p| p[1].0 >= p[0].0 && p[1].1 >= p[0].1)
        );
    }

    #[test]
    fn identity_and_degenerate_inputs() {
        let a = sequence(50, 1.0);
        let w = dtw(&a, &a, 10).unwrap();
        assert!(w.path.iter().all(|(i, j)| i == j));
        assert!(w.cost < 1e-6);
        assert!(dtw(&a, &[], 10).is_none());
        assert!(dtw(&[], &a, 10).is_none());
    }
}
