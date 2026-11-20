//! Signal processing primitives. All of it is our code.

pub mod dtw;
pub mod fir;
pub mod griffinlim;
pub mod lattice;
pub mod level;
pub mod lpc;
pub mod mel;
pub mod pitch;
pub mod psola;
pub mod resonator;
pub mod source;
pub mod source_lf;
pub mod stft;
pub mod window;

pub use fir::Decimator;
pub use level::{peak, rms, rms_dbfs};
pub use resonator::{AntiResonator, Resonator};
pub use source::{Impulse, Klglott88, Tilt, flutter_hz};

/// Deterministic white noise, uniform in [-1, 1] (xorshift32).
#[derive(Clone, Debug)]
pub struct Noise(u32);

impl Noise {
    pub fn new(seed: u32) -> Noise {
        Noise(if seed == 0 { 0x9E37_79B9 } else { seed })
    }

    #[inline]
    pub fn sample(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Identity below |x| = 0.8, smooth saturation to 1.0 above (spec §8.1.5).
#[inline]
pub fn soft_limit(x: f32) -> f32 {
    let a = x.abs();
    if a <= 0.8 {
        x
    } else {
        x.signum() * (0.8 + 0.2 * ((a - 0.8) / 0.2).tanh())
    }
}
