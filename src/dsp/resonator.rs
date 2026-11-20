//! Second-order resonator and antiresonator (Klatt 1980 eq. 1–2; spec §8.1.2).

use std::f32::consts::PI;

/// Resonators at or above this fraction of the sample rate are bypassed.
pub const BYPASS_FRACTION: f32 = 0.47;

#[derive(Clone, Copy, Debug)]
struct Coef {
    a: f64,
    b: f64,
    c: f64,
    bypass: bool,
}

fn coef(f: f32, bw: f32, sample_rate: f32) -> Coef {
    if f >= BYPASS_FRACTION * sample_rate {
        return Coef {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            bypass: true,
        };
    }
    // Coefficients and state are f64: A is tiny for low F and narrow BW, and 1/A in the
    // antiresonator amplifies f32 rounding.
    let pi = PI as f64;
    let (f, bw, t) = (f as f64, bw as f64, 1.0 / sample_rate as f64);
    let c = -(-2.0 * pi * bw * t).exp();
    let b = 2.0 * (-pi * bw * t).exp() * (2.0 * pi * f * t).cos();
    Coef {
        a: 1.0 - b - c,
        b,
        c,
        bypass: false,
    }
}

/// `y[n] = A·x[n] + B·y[n−1] + C·y[n−2]`, unity gain at DC.
#[derive(Clone, Copy, Debug)]
pub struct Resonator {
    k: Coef,
    y1: f64,
    y2: f64,
}

impl Default for Resonator {
    fn default() -> Self {
        Resonator {
            k: Coef {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                bypass: true,
            },
            y1: 0.0,
            y2: 0.0,
        }
    }
}

impl Resonator {
    pub fn new(f: f32, bw: f32, sample_rate: f32) -> Resonator {
        let mut r = Resonator::default();
        r.set(f, bw, sample_rate);
        r
    }

    /// Changes the coefficients; the state is kept.
    pub fn set(&mut self, f: f32, bw: f32, sample_rate: f32) {
        self.k = coef(f, bw, sample_rate);
    }

    pub fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        if self.k.bypass {
            return x;
        }
        let y = self.k.a * x as f64 + self.k.b * self.y1 + self.k.c * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y as f32
    }
}

/// The inverse of a resonator: `y[n] = A'·x[n] + B'·x[n−1] + C'·x[n−2]`.
#[derive(Clone, Copy, Debug)]
pub struct AntiResonator {
    a: f64,
    b: f64,
    c: f64,
    bypass: bool,
    x1: f64,
    x2: f64,
}

impl Default for AntiResonator {
    fn default() -> Self {
        AntiResonator {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            bypass: true,
            x1: 0.0,
            x2: 0.0,
        }
    }
}

impl AntiResonator {
    pub fn new(f: f32, bw: f32, sample_rate: f32) -> AntiResonator {
        let mut r = AntiResonator::default();
        r.set(f, bw, sample_rate);
        r
    }

    pub fn set(&mut self, f: f32, bw: f32, sample_rate: f32) {
        let k = coef(f, bw, sample_rate);
        self.bypass = k.bypass;
        if !k.bypass {
            self.a = 1.0 / k.a;
            self.b = -k.b / k.a;
            self.c = -k.c / k.a;
        }
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        if self.bypass {
            return x;
        }
        let x = x as f64;
        let y = self.a * x + self.b * self.x1 + self.c * self.x2;
        self.x2 = self.x1;
        self.x1 = x;
        y as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 16000.0;

    /// Magnitude response at `f` Hz, measured from the impulse response.
    fn magnitude(h: &[f32], f: f32) -> f32 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (n, v) in h.iter().enumerate() {
            let w = 2.0 * std::f64::consts::PI * f as f64 * n as f64 / SR as f64;
            re += *v as f64 * w.cos();
            im -= *v as f64 * w.sin();
        }
        (re * re + im * im).sqrt() as f32
    }

    fn impulse_response(f: f32, bw: f32) -> Vec<f32> {
        let mut r = Resonator::new(f, bw, SR);
        (0..8000)
            .map(|n| r.process(if n == 0 { 1.0 } else { 0.0 }))
            .collect()
    }

    #[test]
    fn peak_bandwidth_and_dc_gain() {
        for (f, bw) in [
            (500.0, 60.0),
            (1500.0, 90.0),
            (2500.0, 150.0),
            (3300.0, 250.0),
        ] {
            let h = impulse_response(f, bw);
            let mut best = (0.0, 0.0);
            let mut x = f * 0.8;
            while x < f * 1.2 {
                let m = magnitude(&h, x);
                if m > best.1 {
                    best = (x, m);
                }
                x += 1.0;
            }
            assert!((best.0 - f).abs() / f < 0.01, "peak {} for F {}", best.0, f);
            let half = best.1 / 2f32.sqrt();
            let mut lo = best.0;
            while magnitude(&h, lo) > half {
                lo -= 0.5;
            }
            let mut hi = best.0;
            while magnitude(&h, hi) > half {
                hi += 0.5;
            }
            let width = hi - lo;
            assert!((width - bw).abs() / bw < 0.10, "width {width} for BW {bw}");
            assert!((magnitude(&h, 0.0) - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn antiresonator_inverts_resonator() {
        let mut r = Resonator::new(450.0, 100.0, SR);
        let mut z = AntiResonator::new(450.0, 100.0, SR);
        let mut noise = crate::dsp::Noise::new(1);
        for _ in 0..4000 {
            let x = noise.sample();
            let y = z.process(r.process(x));
            assert!((y - x).abs() < 1e-5, "{y} vs {x}");
        }
    }

    #[test]
    fn bypass_near_nyquist() {
        let mut r = Resonator::new(7600.0, 600.0, SR);
        assert_eq!(r.process(0.25), 0.25);
    }
}
