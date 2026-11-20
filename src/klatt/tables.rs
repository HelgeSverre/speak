//! Target tables (spec §8.1.9, §8.1.7, §7.7). One row per phone; no numbers elsewhere.
//! Vowels: Klatt 1980 Table II. Consonants: Klatt 1980 Table III (front-vowel context).

use crate::phone::{Class, Phone, Place};

/// `(start, end)` target; equal for a steady phone.
pub type Glide = (f32, f32);

#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub phone: Phone,
    /// F1–F3 targets in Hz.
    pub f: [Glide; 3],
    /// B1–B3 in Hz.
    pub b: [Glide; 3],
    /// Parallel gains A2–A6 in dB.
    pub a: [f32; 5],
    /// Bypass gain in dB.
    pub ab: f32,
    /// Inherent and minimum duration in ms.
    pub inh: f32,
    pub min: f32,
}

const fn s(x: f32) -> Glide {
    (x, x)
}

#[allow(clippy::too_many_arguments)]
const fn son(
    phone: Phone,
    f1: Glide,
    f2: Glide,
    f3: Glide,
    b1: Glide,
    b2: Glide,
    b3: Glide,
    inh: f32,
    min: f32,
) -> Row {
    Row {
        phone,
        f: [f1, f2, f3],
        b: [b1, b2, b3],
        a: [0.0; 5],
        ab: 0.0,
        inh,
        min,
    }
}

#[allow(clippy::too_many_arguments)]
const fn obs(
    phone: Phone,
    f: [f32; 3],
    b: [f32; 3],
    a: [f32; 5],
    ab: f32,
    inh: f32,
    min: f32,
) -> Row {
    Row {
        phone,
        f: [s(f[0]), s(f[1]), s(f[2])],
        b: [s(b[0]), s(b[1]), s(b[2])],
        a,
        ab,
        inh,
        min,
    }
}

use Phone::*;

/// Indexed by `Phone::index()`.
pub const ROWS: [Row; Phone::COUNT] = [
    son(
        IY,
        (310.0, 290.0),
        (2020.0, 2070.0),
        s(2960.0),
        (45.0, 60.0),
        s(200.0),
        s(400.0),
        155.0,
        55.0,
    ),
    son(
        IH,
        (400.0, 470.0),
        (1800.0, 1600.0),
        (2570.0, 2600.0),
        s(50.0),
        s(100.0),
        s(140.0),
        135.0,
        40.0,
    ),
    son(
        EY,
        (480.0, 330.0),
        (1720.0, 2020.0),
        (2520.0, 2600.0),
        (70.0, 55.0),
        s(100.0),
        s(200.0),
        190.0,
        100.0,
    ),
    son(
        EH,
        (530.0, 620.0),
        (1680.0, 1530.0),
        (2500.0, 2530.0),
        s(60.0),
        s(90.0),
        s(200.0),
        150.0,
        70.0,
    ),
    son(
        AE,
        (620.0, 650.0),
        (1660.0, 1490.0),
        (2430.0, 2470.0),
        s(70.0),
        (150.0, 100.0),
        s(320.0),
        230.0,
        80.0,
    ),
    son(
        AA,
        s(700.0),
        s(1220.0),
        s(2600.0),
        s(130.0),
        s(70.0),
        s(160.0),
        240.0,
        100.0,
    ),
    son(
        AO,
        (600.0, 630.0),
        (990.0, 1040.0),
        (2570.0, 2600.0),
        s(90.0),
        s(100.0),
        s(80.0),
        240.0,
        100.0,
    ),
    son(
        AH,
        s(620.0),
        s(1220.0),
        s(2550.0),
        s(80.0),
        s(50.0),
        s(140.0),
        140.0,
        60.0,
    ),
    son(
        AX,
        s(500.0),
        s(1400.0),
        s(2500.0),
        s(80.0),
        s(90.0),
        s(150.0),
        120.0,
        60.0,
    ),
    son(
        IX,
        s(420.0),
        s(1700.0),
        s(2550.0),
        s(60.0),
        s(100.0),
        s(150.0),
        110.0,
        40.0,
    ),
    son(
        OW,
        (540.0, 450.0),
        (1100.0, 900.0),
        s(2300.0),
        s(80.0),
        s(70.0),
        s(70.0),
        220.0,
        80.0,
    ),
    son(
        UH,
        (450.0, 500.0),
        (1100.0, 1180.0),
        (2350.0, 2390.0),
        s(80.0),
        s(100.0),
        s(80.0),
        160.0,
        60.0,
    ),
    son(
        UW,
        (350.0, 320.0),
        (1250.0, 900.0),
        s(2200.0),
        s(65.0),
        s(110.0),
        s(140.0),
        210.0,
        70.0,
    ),
    son(
        ER,
        (470.0, 420.0),
        (1270.0, 1310.0),
        s(1540.0),
        s(100.0),
        s(60.0),
        s(110.0),
        180.0,
        80.0,
    ),
    son(
        AXR,
        (470.0, 420.0),
        (1270.0, 1310.0),
        s(1540.0),
        s(100.0),
        s(60.0),
        s(110.0),
        130.0,
        60.0,
    ),
    son(
        AY,
        (660.0, 400.0),
        (1200.0, 1880.0),
        (2550.0, 2500.0),
        (100.0, 70.0),
        (70.0, 100.0),
        s(200.0),
        250.0,
        150.0,
    ),
    son(
        AW,
        (640.0, 420.0),
        (1230.0, 940.0),
        (2550.0, 2350.0),
        s(80.0),
        s(70.0),
        (140.0, 80.0),
        260.0,
        100.0,
    ),
    son(
        OY,
        (550.0, 360.0),
        (960.0, 1820.0),
        (2400.0, 2450.0),
        (80.0, 60.0),
        s(50.0),
        (130.0, 160.0),
        280.0,
        150.0,
    ),
    obs(
        P,
        [400.0, 1100.0, 2150.0],
        [300.0, 150.0, 220.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        63.0,
        90.0,
        50.0,
    ),
    obs(
        B,
        [200.0, 1100.0, 2150.0],
        [60.0, 110.0, 130.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        63.0,
        85.0,
        60.0,
    ),
    obs(
        T,
        [400.0, 1600.0, 2600.0],
        [300.0, 120.0, 250.0],
        [0.0, 30.0, 45.0, 57.0, 63.0],
        0.0,
        75.0,
        50.0,
    ),
    obs(
        D,
        [200.0, 1600.0, 2600.0],
        [60.0, 100.0, 170.0],
        [0.0, 47.0, 60.0, 62.0, 60.0],
        0.0,
        75.0,
        50.0,
    ),
    obs(
        K,
        [300.0, 1990.0, 2850.0],
        [250.0, 160.0, 330.0],
        [0.0, 53.0, 43.0, 45.0, 45.0],
        0.0,
        80.0,
        60.0,
    ),
    obs(
        G,
        [200.0, 1990.0, 2850.0],
        [60.0, 150.0, 280.0],
        [0.0, 53.0, 43.0, 45.0, 45.0],
        0.0,
        80.0,
        60.0,
    ),
    obs(
        CH,
        [350.0, 1800.0, 2820.0],
        [200.0, 90.0, 300.0],
        [0.0, 44.0, 60.0, 53.0, 53.0],
        0.0,
        120.0,
        80.0,
    ),
    obs(
        JH,
        [260.0, 1800.0, 2820.0],
        [60.0, 80.0, 270.0],
        [0.0, 44.0, 60.0, 53.0, 53.0],
        0.0,
        110.0,
        70.0,
    ),
    obs(
        F,
        [340.0, 1100.0, 2080.0],
        [200.0, 120.0, 150.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        57.0,
        100.0,
        80.0,
    ),
    obs(
        V,
        [220.0, 1100.0, 2080.0],
        [60.0, 90.0, 120.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        57.0,
        60.0,
        40.0,
    ),
    obs(
        TH,
        [320.0, 1290.0, 2540.0],
        [200.0, 90.0, 200.0],
        [0.0, 0.0, 0.0, 0.0, 28.0],
        48.0,
        90.0,
        60.0,
    ),
    obs(
        DH,
        [270.0, 1290.0, 2540.0],
        [60.0, 80.0, 170.0],
        [0.0, 0.0, 0.0, 0.0, 28.0],
        48.0,
        50.0,
        30.0,
    ),
    obs(
        S,
        [320.0, 1390.0, 2530.0],
        [200.0, 80.0, 200.0],
        [0.0, 0.0, 0.0, 0.0, 52.0],
        0.0,
        105.0,
        60.0,
    ),
    obs(
        Z,
        [240.0, 1390.0, 2530.0],
        [70.0, 60.0, 180.0],
        [0.0, 0.0, 0.0, 0.0, 52.0],
        0.0,
        75.0,
        40.0,
    ),
    obs(
        SH,
        [300.0, 1840.0, 2750.0],
        [200.0, 100.0, 300.0],
        [0.0, 57.0, 48.0, 48.0, 46.0],
        0.0,
        105.0,
        80.0,
    ),
    obs(
        ZH,
        [240.0, 1840.0, 2750.0],
        [70.0, 80.0, 270.0],
        [0.0, 57.0, 48.0, 48.0, 46.0],
        0.0,
        70.0,
        40.0,
    ),
    // HH takes its formants from the neighbouring vowel; the row is a neutral fallback.
    obs(
        HH,
        [500.0, 1500.0, 2500.0],
        [300.0, 150.0, 200.0],
        [0.0; 5],
        0.0,
        80.0,
        20.0,
    ),
    son(
        M,
        s(480.0),
        s(1270.0),
        s(2130.0),
        s(40.0),
        s(200.0),
        s(200.0),
        70.0,
        60.0,
    ),
    son(
        N,
        s(480.0),
        s(1340.0),
        s(2470.0),
        s(40.0),
        s(300.0),
        s(300.0),
        60.0,
        50.0,
    ),
    son(
        NG,
        s(480.0),
        s(2000.0),
        s(2900.0),
        s(40.0),
        s(300.0),
        s(300.0),
        95.0,
        60.0,
    ),
    son(
        L,
        s(310.0),
        s(1050.0),
        s(2880.0),
        s(50.0),
        s(100.0),
        s(280.0),
        80.0,
        40.0,
    ),
    son(
        R,
        s(310.0),
        s(1060.0),
        s(1380.0),
        s(70.0),
        s(100.0),
        s(120.0),
        80.0,
        30.0,
    ),
    son(
        W,
        s(290.0),
        s(610.0),
        s(2150.0),
        s(50.0),
        s(80.0),
        s(60.0),
        80.0,
        60.0,
    ),
    son(
        Y,
        s(260.0),
        s(2070.0),
        s(3020.0),
        s(40.0),
        s(250.0),
        s(500.0),
        80.0,
        40.0,
    ),
    son(
        DX,
        s(300.0),
        s(1600.0),
        s(2600.0),
        s(60.0),
        s(100.0),
        s(170.0),
        20.0,
        20.0,
    ),
    son(
        Q,
        s(500.0),
        s(1500.0),
        s(2500.0),
        s(60.0),
        s(90.0),
        s(150.0),
        30.0,
        20.0,
    ),
    son(
        SIL,
        s(500.0),
        s(1500.0),
        s(2500.0),
        s(60.0),
        s(90.0),
        s(150.0),
        0.0,
        0.0,
    ),
];

pub fn row(p: Phone) -> &'static Row {
    &ROWS[p.index()]
}

/// Transition behaviour of a class (spec §8.1.7 step 2).
#[derive(Clone, Copy, Debug)]
pub struct Transition {
    pub rank: u8,
    /// Coarticulation coefficient for F1, and for F2/F3.
    pub k1: f32,
    pub k23: f32,
    /// Transition time inside the phone itself and inside the neighbour, ms.
    pub t_int: f32,
    pub t_ext: f32,
}

pub fn transition(p: Phone) -> Transition {
    let t = |rank, k1, k23, t_int, t_ext| Transition {
        rank,
        k1,
        k23,
        t_int,
        t_ext,
    };
    match p.class() {
        Class::Stop | Class::Affricate => match p.place() {
            Place::Labial => t(9, 0.1, 0.5, 10.0, 40.0),
            Place::Velar => t(9, 0.1, 0.5, 10.0, 60.0),
            _ => t(9, 0.1, 0.5, 10.0, 50.0),
        },
        Class::Nasal => t(8, 0.1, 0.5, 15.0, 40.0),
        Class::Fricative => t(7, 0.2, 0.5, 20.0, 45.0),
        Class::Flap => t(6, 0.2, 0.5, 5.0, 30.0),
        Class::Liquid => t(5, 0.3, 0.25, 35.0, 70.0),
        Class::Glide => t(4, 0.3, 0.25, 40.0, 80.0),
        Class::Vowel => t(2, 0.5, 0.5, 40.0, 40.0),
        Class::Aspirate | Class::Glottal | Class::Silence => t(0, 0.5, 0.5, 0.0, 0.0),
    }
}

/// Burst length in ms (spec §8.1.7 step 1).
pub fn burst_ms(p: Phone) -> f32 {
    match p {
        P => 5.0,
        T => 10.0,
        K => 15.0,
        B => 5.0,
        D => 8.0,
        G => 12.0,
        _ => 0.0,
    }
}

/// Source amplitudes and timing constants of the sub-segment rules (spec §8.1.7 step 1).
pub mod amp {
    pub const VOWEL_PRIMARY: f32 = 60.0;
    pub const VOWEL_SECONDARY: f32 = 58.0;
    pub const VOWEL_UNSTRESSED: f32 = 57.0;
    pub const LIQUID_GLIDE: f32 = 50.0;
    pub const NASAL: f32 = 52.0;
    pub const NASAL_FNZ: f32 = 450.0;
    pub const NASALIZE_MS: f32 = 40.0;
    pub const NASALIZED_B1: f32 = 100.0;
    pub const FRIC_AF: f32 = 60.0;
    pub const VFRIC_AV: f32 = 47.0;
    pub const VFRIC_TL: f32 = 15.0;
    pub const VFRIC_AF: f32 = 50.0;
    pub const BURST_AF: f32 = 60.0;
    pub const ASP_AH: f32 = 55.0;
    pub const ASP_B1: f32 = 250.0;
    pub const ASP_STRESSED_MS: f32 = 45.0;
    pub const ASP_OTHER_MS: f32 = 15.0;
    pub const VOICEBAR_AV: f32 = 40.0;
    pub const VOICEBAR_TL: f32 = 20.0;
    pub const VOICEBAR_F1: f32 = 180.0;
    pub const VOICEBAR_B1: f32 = 60.0;
    pub const WEAK_BURST_DB: f32 = 6.0;
    pub const AFFRICATE_CLOSURE: f32 = 0.4;
    pub const HH_AH: f32 = 55.0;
    pub const HH_B1: f32 = 300.0;
    pub const FLAP_AV: f32 = 50.0;
    pub const Q_F0_DROP: f32 = 0.15;
    pub const Q_SPAN_MS: f32 = 30.0;
    /// Diphthong glide runs from 25 % to 85 % of the vowel.
    pub const GLIDE_START: f32 = 0.25;
    pub const GLIDE_END: f32 = 0.85;
    /// Source amplitude ramps are 10 ms wide, centred on the boundary.
    pub const RAMP_MS: f32 = 10.0;
    /// AV falls by 6 dB over the last 100 ms before a pause.
    pub const FINAL_FALL_DB: f32 = 6.0;
    pub const FINAL_FALL_MS: f32 = 100.0;
    pub const F4: f32 = 3300.0;
    /// Breathy offset (Klatt & Klatt 1990): over the last 150 ms before a pause OQ, TL and AH rise.
    pub const OFFSET_MS: f32 = 150.0;
    pub const OFFSET_OQ: f32 = 0.1;
    pub const OFFSET_TL: f32 = 6.0;
    pub const OFFSET_AH: f32 = 6.0;
    /// AH the offset rises from when the voice has no breath of its own.
    pub const OFFSET_AH_FLOOR: f32 = 38.0;
    /// Hard onset: the first two periods after a pause or glottal stop use a smaller OQ.
    pub const ONSET_MS: f32 = 20.0;
    pub const ONSET_OQ: f32 = 0.1;
    /// Phrase-final creak (voices with `fry`): span, pitch drop and open-quotient drop at full strength.
    pub const FRY_MS: f32 = 180.0;
    pub const FRY_F0_DROP: f32 = 0.45;
    pub const FRY_OQ: f32 = 0.12;
}

/// Context-dependent values (spec §8.1.8). Columns: front, back unrounded, back rounded vowels.
/// Starting values are ours, inferred from Klatt 1980/1987 and general acoustic phonetics.
pub mod context {
    pub const LABIAL_F2: [f32; 3] = [1100.0, 900.0, 800.0];
    pub const ALVEOLAR_F2: [f32; 3] = [1800.0, 1700.0, 1600.0];
    pub const VELAR_F2: [f32; 3] = [2300.0, 1700.0, 1300.0];
    pub const VELAR_F3: [f32; 3] = [2900.0, 2400.0, 2100.0];
    /// Velar burst before back vowels: energy at F2, the higher formants cut.
    pub const VELAR_BACK_A2: f32 = 60.0;
    pub const VELAR_BACK_CUT_DB: f32 = 15.0;
    /// Dark `L` (not before a vowel in its word).
    pub const DARK_L_F2: f32 = 850.0;
    pub const DARK_L_F3: f32 = 2800.0;
    /// A vowel before `R`: F3 glides here over the last 40 %.
    pub const R_COLOUR_F3: f32 = 1700.0;
    pub const R_COLOUR_FROM: f32 = 0.6;
    /// A vowel shorter than this fraction of its inherent duration moves toward schwa.
    pub const REDUCE_BELOW: f32 = 0.6;
    pub const REDUCE_AMOUNT: f32 = 0.3;
    /// `L R W Y` after a voiceless stop: the first half is devoiced.
    pub const DEVOICE_AH: f32 = 50.0;
    pub const DEVOICE_FRACTION: f32 = 0.5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_in_phone_order() {
        for (i, r) in ROWS.iter().enumerate() {
            assert_eq!(r.phone.index(), i);
            assert!(r.min <= r.inh);
        }
    }
}
