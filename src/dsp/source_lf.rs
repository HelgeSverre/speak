//! Liljencrants-Fant model of the glottal flow derivative (spec §8.1.4).
//!
//! Open phase `E(t) = E0·e^{αt}·sin(ωg·t)` up to the excitation instant `te`, then an
//! exponential return to zero. Times are in fractions of the period. `ε` and `α` are solved
//! once per parameter set and cached.

use std::f32::consts::PI;

use crate::dsp::source::PeriodClock;
use crate::traits::{GlottalSource, VoiceQuality};

/// Return-phase time as a fraction of the period at TL = 0.
const RETURN_FRACTION: f64 = 0.004;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Shape {
    oq: f32,
    sq: f32,
    tl: f32,
    te: f32,
    ta: f32,
    wg: f32,
    alpha: f32,
    epsilon: f32,
    e0: f32,
}

/// Solves the LF parameters for one period of length 1. `None` when the iteration fails.
fn solve(oq: f32, sq: f32, tl: f32) -> Option<Shape> {
    let oq = oq.clamp(0.3, 0.8);
    let sq = sq.clamp(1.2, 4.0);
    let te = oq as f64;
    let tc = 1.0f64;
    let tb = tc - te;
    // Return phase. Spectral tilt is applied after the source, by the same filter for every
    // source, so the return phase here is short and only grows mildly with TL: with 2 % of the
    // period the source alone is 8 dB darker than KLGLOTT88 between 0.5 and 3 kHz.
    let ta = (RETURN_FRACTION * (1.0 + tl.clamp(0.0, 34.0) as f64 / 34.0)).min(0.6 * tb);
    // Flow peak from the speed quotient (opening time ÷ closing time).
    let tp = te * sq as f64 / (1.0 + sq as f64);
    let wg = std::f64::consts::PI / tp;

    // ε·ta = 1 − exp(−ε·tb), by Newton iteration.
    let mut eps = 1.0 / ta;
    for _ in 0..10 {
        let f = eps * ta - 1.0 + (-eps * tb).exp();
        let df = ta - tb * (-eps * tb).exp();
        if df.abs() < 1e-12 {
            break;
        }
        eps -= f / df;
    }
    if !eps.is_finite() || eps <= 0.0 {
        return None;
    }
    // Area of the return phase with Ee = 1 (negative flow derivative).
    let a2 = -(1.0 / (eps * ta)) * ((1.0 - (-eps * tb).exp()) / eps - tb * (-eps * tb).exp());
    // Net flow over the period as a function of α, with E(te) = −1.
    let net = |alpha: f64| -> f64 {
        let s = (wg * te).sin();
        let c = (wg * te).cos();
        let e0 = -1.0 / ((alpha * te).exp() * s);
        let open = ((alpha * te).exp() * (alpha * s - wg * c) + wg) / (alpha * alpha + wg * wg);
        e0 * open + a2
    };
    // Newton with a numerical derivative, at most 10 steps; then check the residual.
    let mut alpha = 0.0f64;
    for _ in 0..10 {
        let f = net(alpha);
        let h = 1e-4 * alpha.abs().max(1.0);
        let df = (net(alpha + h) - net(alpha - h)) / (2.0 * h);
        if df.abs() < 1e-12 {
            break;
        }
        let step = (f / df).clamp(-10.0, 10.0);
        alpha -= step;
        if step.abs() < 1e-9 {
            break;
        }
    }
    if !alpha.is_finite() || net(alpha).abs() > 1e-4 {
        return None;
    }
    let e0 = -1.0 / ((alpha * te).exp() * (wg * te).sin());
    Some(Shape {
        oq,
        sq,
        tl,
        te: te as f32,
        ta: ta as f32,
        wg: wg as f32,
        alpha: alpha as f32,
        epsilon: eps as f32,
        e0: e0 as f32,
    })
}

impl Shape {
    /// Flow derivative at phase `x` (0..1 of the period), −1 at the excitation instant.
    fn at(&self, x: f32) -> f32 {
        if x <= self.te {
            self.e0 * (self.alpha * x).exp() * (self.wg * x).sin()
        } else {
            let tb = 1.0 - self.te;
            -(1.0 / (self.epsilon * self.ta))
                * ((-self.epsilon * (x - self.te)).exp() - (-self.epsilon * tb).exp())
        }
    }
}

pub struct Lf {
    clock: PeriodClock,
    shape: Shape,
}

impl Lf {
    pub fn new(sample_rate: f32) -> Lf {
        // The default parameters always solve; the fallback keeps the constructor total.
        let shape = solve(0.5, 2.0, 0.0).unwrap_or(Shape {
            oq: 0.5,
            sq: 2.0,
            tl: 0.0,
            te: 0.5,
            ta: 0.02,
            wg: PI / 0.33,
            alpha: 0.0,
            epsilon: 50.0,
            e0: 1.0,
        });
        Lf {
            clock: PeriodClock::new(sample_rate),
            shape,
        }
    }
}

impl GlottalSource for Lf {
    fn reset(&mut self) {
        self.clock.reset();
    }

    #[inline]
    fn next(&mut self, f0: f32, q: &VoiceQuality) -> f32 {
        if !self.clock.tick(f0, q.di) {
            return 0.0;
        }
        // Re-solve at a period start when the parameters moved; keep the old solution on failure.
        if self.clock.epoch() {
            let (oq, sq) = (q.oq.clamp(0.3, 0.8), q.sq.clamp(1.2, 4.0));
            if ((oq - self.shape.oq).abs() > 1e-3
                || (sq - self.shape.sq).abs() > 1e-3
                || (q.tl - self.shape.tl).abs() > 0.25)
                && let Some(s) = solve(oq, sq, q.tl)
            {
                self.shape = s;
            }
        }
        self.clock.amp() * self.shape.at(self.clock.phase() / self.clock.t0())
    }

    fn epoch(&self) -> bool {
        self.clock.epoch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_across_the_parameter_range_with_zero_net_flow() {
        for oq in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
            for sq in [1.2, 2.0, 3.0, 4.0] {
                for tl in [0.0, 6.0, 20.0, 34.0] {
                    let s = solve(oq, sq, tl)
                        .unwrap_or_else(|| panic!("no solution for oq {oq} sq {sq} tl {tl}"));
                    let n = 20000;
                    let sum: f32 =
                        (0..n).map(|i| s.at(i as f32 / n as f32)).sum::<f32>() / n as f32;
                    assert!(
                        sum.abs() < 2e-3,
                        "net flow {sum} for oq {oq} sq {sq} tl {tl}"
                    );
                    assert!((s.at(s.te) + 1.0).abs() < 1e-3, "E(te) = {}", s.at(s.te));
                    assert!((0..n).all(|i| s.at(i as f32 / n as f32).is_finite()));
                }
            }
        }
    }

    #[test]
    fn period_and_epochs() {
        let mut s = Lf::new(64000.0);
        let q = VoiceQuality::default();
        let mut epochs = Vec::new();
        let mut min = 0.0f32;
        for i in 0..64000 {
            min = min.min(s.next(150.0, &q));
            if s.epoch() {
                epochs.push(i);
            }
        }
        assert!((epochs.len() as i32 - 150).abs() <= 1);
        let mean = (epochs[epochs.len() - 1] - epochs[0]) as f32 / (epochs.len() - 1) as f32;
        assert!((64000.0 / mean - 150.0).abs() / 150.0 < 0.005);
        assert!((min + 1.0).abs() < 0.03, "negative peak {min}");
    }
}
