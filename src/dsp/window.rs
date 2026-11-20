//! Window functions. Periodic forms (divide by N, not N − 1): their overlap-add sums to a constant.

use std::f32::consts::PI;

pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

pub fn hamming(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_hann_overlap_adds_to_a_constant() {
        let (n, hop) = (1024, 256);
        let w = hann(n);
        for start in 0..hop {
            let sum: f32 = (0..n / hop).map(|k| w[start + k * hop]).sum();
            assert!((sum - 2.0).abs() < 1e-4, "{sum}");
        }
        assert_eq!(hann(8)[0], 0.0);
        assert!((hamming(8)[0] - 0.08).abs() < 1e-6);
    }
}
