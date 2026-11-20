//! Voice presets (spec §8.1.6).

use crate::error::{Error, Result};
use crate::traits::{ProsodyParams, SynthOptions};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Klglott88,
    Lf,
    Impulse,
}

impl SourceKind {
    pub fn parse(s: &str) -> Option<SourceKind> {
        match s {
            "klglott88" => Some(SourceKind::Klglott88),
            "lf" => Some(SourceKind::Lf),
            "impulse" => Some(SourceKind::Impulse),
            _ => None,
        }
    }
}

/// Parameters that are fixed for a voice.
#[derive(Clone, Debug, PartialEq)]
pub struct Voice {
    pub name: &'static str,
    pub description: &'static str,
    pub base_hz: f32,
    /// Flat F0 contour.
    pub flat: bool,
    /// Multiplies F1–F8.
    pub formant_scale: f32,
    pub oq: f32,
    pub tl: f32,
    /// AH in dB added in vowels; 0 = none.
    pub breath: f32,
    pub source: SourceKind,
    /// Flutter in percent.
    pub flutter: f32,
    /// Diplophonia in percent.
    pub di: f32,
    /// LF speed quotient.
    pub sq: f32,
    /// Noise only: AV → 0, AH = former AV − 5.
    pub whisper: bool,
    /// Random period perturbation in percent (cycle to cycle).
    pub jitter: f32,
    /// Random amplitude perturbation in percent (cycle to cycle).
    pub shimmer: f32,
    /// Creak at the end of falling phrases, 0..1: pitch drops and the glottis tightens.
    pub fry: f32,
    /// Speaking style: pitch range, pace and pause multipliers.
    pub range: f32,
    pub pace: f32,
    pub pause: f32,
    pub fnp: f32,
    pub bnp: f32,
    pub bnz: f32,
    pub f4: f32,
    pub b4: f32,
    /// F5–F8 and B5–B8.
    pub f_high: [f32; 4],
    pub b_high: [f32; 4],
}

const BASE: Voice = Voice {
    name: "klatt-male",
    description: "Formant synthesizer, male",
    base_hz: 110.0,
    flat: false,
    formant_scale: 1.0,
    oq: 0.5,
    tl: 6.0,
    breath: 0.0,
    source: SourceKind::Klglott88,
    flutter: 25.0,
    di: 0.0,
    sq: 2.0,
    whisper: false,
    jitter: 0.0,
    shimmer: 0.0,
    fry: 0.0,
    range: 1.0,
    pace: 1.0,
    pause: 1.0,
    fnp: 270.0,
    bnp: 100.0,
    bnz: 100.0,
    f4: 3300.0,
    b4: 250.0,
    f_high: [3750.0, 4900.0, 6500.0, 7500.0],
    b_high: [200.0, 1000.0, 500.0, 600.0],
};

pub const VOICES: &[Voice] = &[
    BASE,
    Voice {
        name: "klatt-female",
        description: "Formant synthesizer, female",
        base_hz: 200.0,
        formant_scale: 1.17,
        oq: 0.6,
        tl: 10.0,
        breath: 40.0,
        ..BASE
    },
    Voice {
        name: "klatt-child",
        description: "Formant synthesizer, child",
        base_hz: 280.0,
        formant_scale: 1.35,
        oq: 0.6,
        tl: 8.0,
        breath: 30.0,
        ..BASE
    },
    Voice {
        name: "klatt-breathy",
        description: "Formant synthesizer, breathy male",
        oq: 0.7,
        tl: 18.0,
        breath: 50.0,
        ..BASE
    },
    Voice {
        name: "klatt-creaky",
        description: "Formant synthesizer, creaky male",
        base_hz: 90.0,
        oq: 0.35,
        tl: 2.0,
        di: 25.0,
        flutter: 40.0,
        ..BASE
    },
    Voice {
        name: "klatt-robot",
        description: "Formant synthesizer, monotone impulse source",
        base_hz: 100.0,
        flat: true,
        tl: 0.0,
        source: SourceKind::Impulse,
        flutter: 0.0,
        ..BASE
    },
    Voice {
        name: "klatt-pip",
        description: "Bright and curious: quick, wide pitch range",
        base_hz: 150.0,
        formant_scale: 1.10,
        oq: 0.55,
        tl: 7.0,
        breath: 25.0,
        flutter: 30.0,
        jitter: 0.5,
        shimmer: 3.0,
        range: 1.5,
        pace: 1.08,
        pause: 0.85,
        ..BASE
    },
    Voice {
        name: "klatt-calm",
        description: "Calm and soft: slow, narrow pitch range, long pauses",
        base_hz: 100.0,
        formant_scale: 0.97,
        oq: 0.62,
        tl: 12.0,
        breath: 35.0,
        jitter: 0.3,
        shimmer: 2.0,
        range: 0.6,
        pace: 0.88,
        pause: 1.35,
        ..BASE
    },
    Voice {
        name: "klatt-gravel",
        description: "Low and gravelly: rough source, creak at phrase ends",
        base_hz: 84.0,
        formant_scale: 0.94,
        oq: 0.4,
        tl: 3.0,
        di: 10.0,
        jitter: 1.6,
        shimmer: 8.0,
        fry: 1.0,
        range: 0.9,
        pace: 0.95,
        ..BASE
    },
    Voice {
        name: "klatt-drama",
        description: "Theatrical: very wide pitch range, deliberate pace",
        base_hz: 120.0,
        oq: 0.5,
        tl: 5.0,
        jitter: 0.4,
        shimmer: 3.0,
        fry: 0.5,
        range: 1.9,
        pace: 0.9,
        pause: 1.25,
        ..BASE
    },
    Voice {
        name: "klatt-whisper",
        description: "Formant synthesizer, whispered",
        whisper: true,
        ..BASE
    },
];

pub const DEFAULT_VOICE: &str = "klatt-male";

/// `-P` keys of the Klatt backend.
pub const PARAM_KEYS: &[&str] = &[
    "source",
    "oq",
    "tl",
    "flutter",
    "di",
    "formant_scale",
    "breath",
    "sq",
];

impl Voice {
    pub fn find(name: &str) -> Option<Voice> {
        VOICES.iter().find(|v| v.name == name).cloned()
    }

    pub fn prosody(&self) -> ProsodyParams {
        ProsodyParams {
            base_hz: self.base_hz,
            range: self.range,
            flat: self.flat,
            rate: self.pace,
            pause: self.pause,
        }
    }

    /// Applies `-P key=value` overrides.
    pub fn with_options(mut self, opts: &SynthOptions) -> Result<Voice> {
        for (key, value) in &opts.params {
            let num = |lo: f32, hi: f32| -> Result<f32> {
                value
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= lo && *v <= hi)
                    .ok_or_else(|| Error::usage(format!("-P {key}={value}: expected {lo}..{hi}")))
            };
            match key.as_str() {
                "source" => {
                    self.source = SourceKind::parse(value).ok_or_else(|| {
                        Error::usage(format!(
                            "-P source={value}: expected klglott88, lf or impulse"
                        ))
                    })?;
                }
                "oq" => self.oq = num(0.3, 0.8)?,
                "tl" => self.tl = num(0.0, 34.0)?,
                "flutter" => self.flutter = num(0.0, 100.0)?,
                "di" => self.di = num(0.0, 80.0)?,
                "formant_scale" => self.formant_scale = num(0.7, 1.6)?,
                "breath" => self.breath = num(0.0, 70.0)?,
                "sq" => self.sq = num(1.2, 4.0)?,
                "jitter" => self.jitter = num(0.0, 10.0)?,
                "shimmer" => self.shimmer = num(0.0, 40.0)?,
                "fry" => self.fry = num(0.0, 1.0)?,
                "range" => self.range = num(0.0, 3.0)?,
                "pace" => self.pace = num(0.5, 2.0)?,
                "pause" => self.pause = num(0.2, 3.0)?,
                _ => {}
            }
        }
        Ok(self)
    }
}
