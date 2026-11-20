//! One scalar gain per signal path (spec §8.1.5), fitted to the level tests of spec §16.2.

/// dB → linear with 60 dB as unity; 0 dB and below is off.
#[inline]
pub fn lin(db: f32) -> f32 {
    if db <= 0.0 {
        0.0
    } else {
        10f32.powf((db - 60.0) / 20.0)
    }
}

pub const K_AV: f32 = 1.0;
pub const K_AH: f32 = 0.0914;
pub const K_AF: f32 = 1.0;
/// Parallel formant gains A2…A6.
pub const K_A: [f32; 5] = [0.193, 0.193, 0.193, 0.193, 0.1756];
pub const K_AB: f32 = 0.1406;
pub const K_OUT: f32 = 0.1547;
