//! Lattice filters driven by reflection coefficients (spec §8.2.3).
//!
//! Any set with |k| < 1 is a stable all-pole filter, so coefficients can be interpolated
//! freely between frames. The analysis (FIR) and synthesis (IIR) lattices are exact inverses
//! sample by sample, even while the coefficients change.

pub const MAX_K: f32 = 0.999;

/// All-pole synthesis lattice: excitation in, speech out.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Backward prediction errors b[0..p].
    b: Vec<f32>,
}

impl Lattice {
    pub fn new(order: usize) -> Lattice {
        Lattice {
            b: vec![0.0; order + 1],
        }
    }

    pub fn reset(&mut self) {
        self.b.fill(0.0);
    }

    /// One sample. `k[i]` is reflection coefficient i+1.
    #[inline]
    pub fn process(&mut self, excitation: f32, k: &[f32]) -> f32 {
        let p = k.len().min(self.b.len() - 1);
        let mut f = excitation;
        for i in (0..p).rev() {
            let ki = k[i].clamp(-MAX_K, MAX_K);
            f -= ki * self.b[i];
            self.b[i + 1] = self.b[i] + ki * f;
        }
        self.b[0] = f;
        f
    }
}

/// Analysis lattice (inverse filter): speech in, residual out.
#[derive(Clone, Debug)]
pub struct InverseLattice {
    b: Vec<f32>,
}

impl InverseLattice {
    pub fn new(order: usize) -> InverseLattice {
        InverseLattice {
            b: vec![0.0; order + 1],
        }
    }

    pub fn reset(&mut self) {
        self.b.fill(0.0);
    }

    #[inline]
    pub fn process(&mut self, x: f32, k: &[f32]) -> f32 {
        let p = k.len().min(self.b.len() - 1);
        let mut f = x;
        let mut b_prev = self.b[0];
        self.b[0] = x;
        for (i, ki) in k.iter().enumerate().take(p) {
            let ki = ki.clamp(-MAX_K, MAX_K);
            let f_next = f + ki * b_prev;
            let b_next = b_prev + ki * f;
            b_prev = self.b[i + 1];
            self.b[i + 1] = b_next;
            f = f_next;
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Noise;
    use crate::dsp::lpc::reflection_to_predictor;

    #[test]
    fn synthesis_lattice_equals_the_direct_form_filter() {
        let k = [-0.8f32, 0.6, -0.3, 0.2, -0.1, 0.05];
        let a = reflection_to_predictor(&k);
        let mut lat = Lattice::new(k.len());
        let mut noise = Noise::new(11);
        let mut y = vec![0.0f32; 2000];
        for n in 0..y.len() {
            let e = noise.sample();
            let mut v = e;
            for j in 1..a.len() {
                if n >= j {
                    v -= a[j] * y[n - j];
                }
            }
            y[n] = v;
            let got = lat.process(e, &k);
            assert!((got - v).abs() < 2e-3, "sample {n}: {got} vs {v}");
        }
    }

    #[test]
    fn inverse_then_synthesis_is_identity_with_moving_coefficients() {
        let mut noise = Noise::new(5);
        let (mut inv, mut syn) = (InverseLattice::new(8), Lattice::new(8));
        for n in 0..4000 {
            // Coefficients drift every sample.
            let t = n as f32 / 4000.0;
            let k: Vec<f32> = (0..8)
                .map(|i| 0.9 * ((i as f32 + 1.0) * (1.0 + t)).sin() * (1.0 - 0.1 * i as f32))
                .collect();
            let x = noise.sample();
            let y = syn.process(inv.process(x, &k), &k);
            assert!((y - x).abs() < 1e-4, "sample {n}: {y} vs {x}");
        }
    }

    #[test]
    fn extreme_coefficients_stay_finite() {
        let mut lat = Lattice::new(4);
        let k = [1.5f32, -1.5, 1.0, -1.0];
        assert!((0..10000).all(|n| {
            lat.process(if n % 100 == 0 { 1.0 } else { 0.0 }, &k)
                .is_finite()
        }));
    }
}
