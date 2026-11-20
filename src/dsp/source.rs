//! Voicing sources (spec §8.1.4). All run at the oversampled source rate and produce the
//! glottal flow derivative, normalized to −1 at closure.

use std::f32::consts::PI;

use crate::dsp::resonator::{AntiResonator, Resonator};
use crate::traits::{GlottalSource, VoiceQuality};

/// Flutter (Klatt & Klatt 1990 eq. 1): slow deterministic F0 wobble in Hz.
/// `fl` in percent, `t` in seconds.
pub fn flutter_hz(fl: f32, f0: f32, t: f32) -> f32 {
    (fl / 50.0)
        * (f0 / 100.0)
        * ((2.0 * PI * 12.7 * t).sin() + (2.0 * PI * 7.1 * t).sin() + (2.0 * PI * 4.7 * t).sin())
}

/// Period bookkeeping shared by the sources: fractional phase in samples.
#[derive(Clone, Debug)]
pub struct PeriodClock {
    sample_rate: f32,
    /// Samples since the current period started.
    phase: f32,
    /// Length of the current period in samples; 0 when unvoiced.
    t0: f32,
    /// Base period before the diplophonia shift.
    epoch: bool,
    /// Counts periods for diplophonia (odd ones are delayed and attenuated).
    count: u32,
    /// Amplitude of the current period (diplophonia).
    amp: f32,
}

impl PeriodClock {
    pub fn new(sample_rate: f32) -> PeriodClock {
        PeriodClock {
            sample_rate,
            phase: 0.0,
            t0: 0.0,
            epoch: false,
            count: 0,
            amp: 1.0,
        }
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.t0 = 0.0;
        self.epoch = false;
        self.count = 0;
        self.amp = 1.0;
    }

    fn start_period(&mut self, f0: f32, di: f32) {
        let base = self.sample_rate / f0;
        // Diplophonia: every second period is delayed by DI/100 × T0/2 and attenuated by DI/100.
        // Delaying a pulse lengthens the period before it and shortens the one after it.
        let d = (di / 100.0).clamp(0.0, 0.9);
        let shift = d * base / 2.0;
        if self.count.is_multiple_of(2) {
            self.t0 = base + shift;
            self.amp = 1.0;
        } else {
            self.t0 = base - shift;
            self.amp = 1.0 - d;
        }
        self.count = self.count.wrapping_add(1);
        self.epoch = true;
    }

    /// Advances one sample. Returns false when unvoiced.
    #[inline]
    pub fn tick(&mut self, f0: f32, di: f32) -> bool {
        self.epoch = false;
        if f0 <= 0.0 {
            self.t0 = 0.0;
            self.phase = 0.0;
            self.count = 0;
            return false;
        }
        if self.t0 <= 0.0 {
            self.phase = 0.0;
            self.start_period(f0, di);
        } else {
            self.phase += 1.0;
            if self.phase >= self.t0 {
                self.phase -= self.t0;
                self.start_period(f0, di);
            }
        }
        true
    }

    #[inline]
    pub fn phase(&self) -> f32 {
        self.phase
    }

    #[inline]
    pub fn t0(&self) -> f32 {
        self.t0
    }

    #[inline]
    pub fn amp(&self) -> f32 {
        self.amp
    }

    #[inline]
    pub fn epoch(&self) -> bool {
        self.epoch
    }
}

/// KLGLOTT88: derivative of Rosenberg's `a·t² − b·t³` flow during the open phase.
#[derive(Clone, Debug)]
pub struct Klglott88 {
    clock: PeriodClock,
}

impl Klglott88 {
    pub fn new(sample_rate: f32) -> Klglott88 {
        Klglott88 {
            clock: PeriodClock::new(sample_rate),
        }
    }
}

impl GlottalSource for Klglott88 {
    fn reset(&mut self) {
        self.clock.reset();
    }

    #[inline]
    fn next(&mut self, f0: f32, q: &VoiceQuality) -> f32 {
        if !self.clock.tick(f0, q.di) {
            return 0.0;
        }
        let open = q.oq.clamp(0.1, 0.95) * self.clock.t0();
        let x = self.clock.phase() / open;
        if x < 1.0 {
            self.clock.amp() * (2.0 * x - 3.0 * x * x)
        } else {
            0.0
        }
    }

    fn epoch(&self) -> bool {
        self.clock.epoch()
    }
}

/// Klatt 1980's impulse source: impulse → RGP (F 0, BW 100) → RGZ (F 1500, BW 6000) → difference.
#[derive(Clone, Debug)]
pub struct Impulse {
    clock: PeriodClock,
    rgp: Resonator,
    rgz: AntiResonator,
    /// Part of a fractionally placed impulse that falls on the next sample.
    carry: f32,
    last: f32,
    gain: f32,
}

impl Impulse {
    pub fn new(sample_rate: f32) -> Impulse {
        Impulse {
            clock: PeriodClock::new(sample_rate),
            rgp: Resonator::new(0.0, 100.0, sample_rate),
            rgz: AntiResonator::new(1500.0, 6000.0, sample_rate),
            carry: 0.0,
            last: 0.0,
            // Fitted so a vowel has the same level as with KLGLOTT88 (see klatt tests).
            gain: sample_rate / 64000.0 * 560.0,
        }
    }
}

impl GlottalSource for Impulse {
    fn reset(&mut self) {
        self.clock.reset();
        self.rgp.reset();
        self.rgz.reset();
        self.carry = 0.0;
        self.last = 0.0;
    }

    #[inline]
    fn next(&mut self, f0: f32, q: &VoiceQuality) -> f32 {
        let voiced = self.clock.tick(f0, q.di);
        let mut x = self.carry;
        self.carry = 0.0;
        if voiced && self.clock.epoch() {
            // The period started `frac` samples ago: split the impulse over two samples so the
            // period is not quantized to whole samples.
            let frac = self.clock.phase().clamp(0.0, 1.0);
            x += self.clock.amp() * (1.0 - frac);
            self.carry = self.clock.amp() * frac;
        }
        let y = self.rgz.process(self.rgp.process(x));
        let d = y - self.last;
        self.last = y;
        -d * self.gain * 64.0
    }

    fn epoch(&self) -> bool {
        self.clock.epoch()
    }
}

/// One-pole low-pass whose attenuation at 3 kHz is `TL` dB (spec §8.1.4).
#[derive(Clone, Copy, Debug)]
pub struct Tilt {
    sample_rate: f32,
    a: f32,
    y1: f32,
    tl: f32,
}

impl Tilt {
    pub fn new(sample_rate: f32) -> Tilt {
        Tilt {
            sample_rate,
            a: 0.0,
            y1: 0.0,
            tl: 0.0,
        }
    }

    pub fn set(&mut self, tl_db: f32) {
        let tl = tl_db.clamp(0.0, 40.0);
        if (tl - self.tl).abs() < 1e-3 {
            return;
        }
        self.tl = tl;
        if tl <= 0.0 {
            self.a = 0.0;
            return;
        }
        let g = 10f32.powf(-tl / 10.0);
        let c = (2.0 * PI * 3000.0 / self.sample_rate).cos();
        let p = 1.0 - g * c;
        self.a = (p - (p * p - (1.0 - g) * (1.0 - g)).max(0.0).sqrt()) / (1.0 - g);
    }

    pub fn reset(&mut self) {
        self.y1 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = (1.0 - self.a) * x + self.a * self.y1;
        self.y1 = y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 64000.0;

    fn run(
        src: &mut dyn GlottalSource,
        f0: f32,
        q: &VoiceQuality,
        n: usize,
    ) -> (Vec<f32>, Vec<usize>) {
        let mut out = Vec::with_capacity(n);
        let mut epochs = Vec::new();
        for i in 0..n {
            out.push(src.next(f0, q));
            if src.epoch() {
                epochs.push(i);
            }
        }
        (out, epochs)
    }

    #[test]
    fn period_matches_f0() {
        let q = VoiceQuality::default();
        for f0 in [100.0, 150.0, 220.0] {
            let sources: [Box<dyn GlottalSource>; 2] =
                [Box::new(Klglott88::new(SR)), Box::new(Impulse::new(SR))];
            for mut s in sources {
                let (_, epochs) = run(s.as_mut(), f0, &q, SR as usize);
                let periods = (epochs.len() - 1) as f32;
                let mean = (epochs[epochs.len() - 1] - epochs[0]) as f32 / periods;
                let measured = SR / mean;
                assert!((measured - f0).abs() / f0 < 0.005, "{measured} vs {f0}");
                // One epoch per period over one second.
                assert!((epochs.len() as f32 - f0).abs() <= 1.0);
            }
        }
    }

    #[test]
    fn open_quotient() {
        for oq in [0.4, 0.5, 0.7] {
            let q = VoiceQuality {
                oq,
                ..VoiceQuality::default()
            };
            let mut s = Klglott88::new(SR);
            let (out, _) = run(&mut s, 100.0, &q, SR as usize);
            let open = out.iter().filter(|v| **v != 0.0).count() as f32 / out.len() as f32;
            assert!((open - oq).abs() < 0.01, "{open} vs {oq}");
            let min = out.iter().cloned().fold(0.0f32, f32::min);
            assert!((min + 1.0).abs() < 0.02, "closure peak {min}");
        }
    }

    #[test]
    fn unvoiced_is_silent_and_restarts() {
        let q = VoiceQuality::default();
        let mut s = Klglott88::new(SR);
        assert_eq!(s.next(0.0, &q), 0.0);
        assert!(!s.epoch());
        s.next(100.0, &q);
        assert!(s.epoch());
    }

    #[test]
    fn diplophonia_alternates_periods() {
        let q = VoiceQuality {
            di: 25.0,
            ..VoiceQuality::default()
        };
        let mut s = Klglott88::new(SR);
        let (_, epochs) = run(&mut s, 100.0, &q, SR as usize);
        let a = (epochs[1] - epochs[0]) as f32;
        let b = (epochs[2] - epochs[1]) as f32;
        let ratio = a.max(b) / a.min(b);
        assert!(ratio > 1.2 && ratio < 1.35, "{ratio}");
    }

    #[test]
    fn tilt_attenuation_at_3khz() {
        for tl in [6.0, 12.0, 24.0] {
            let mut t = Tilt::new(SR);
            t.set(tl);
            let x: Vec<f32> = (0..SR as usize)
                .map(|n| (2.0 * PI * 3000.0 * n as f32 / SR).sin())
                .collect();
            let y: Vec<f32> = x.iter().map(|v| t.process(*v)).collect();
            let db = 20.0 * (crate::dsp::rms(&y[1000..]) / crate::dsp::rms(&x[1000..])).log10();
            assert!((db + tl).abs() < 0.5, "{db} dB for TL {tl}");
        }
    }

    #[test]
    fn flutter_is_small_and_deterministic() {
        let a = flutter_hz(25.0, 110.0, 0.37);
        assert_eq!(a, flutter_hz(25.0, 110.0, 0.37));
        assert!((0..1000).all(|i| flutter_hz(25.0, 110.0, i as f32 / 100.0).abs() < 1.7));
    }
}
