//! Klatt backend: utterance-level tests and golden audio (spec §16.2).

use std::path::PathBuf;

use speak::chunk::Chunk;
use speak::cli::pipeline::arpabet_chunks;
use speak::dsp::rms_dbfs;
use speak::klatt::{KlattSynth, SR, Voice};
use speak::pho::{parse_pho, print_pho};
use speak::text::RuleProsody;
use speak::traits::{Prosody, Synthesizer};
use speak::{Controls, Unit};

fn synth_chunks(voice: &str, chunks: &mut [Chunk]) -> Vec<f32> {
    let v = Voice::find(voice).unwrap();
    let mut s = KlattSynth::new(v);
    let spec = s.spec();
    let mut out = Vec::new();
    for c in chunks.iter_mut() {
        RuleProsody.apply(c, &spec.prosody);
        s.synthesize(c, &mut out).unwrap();
    }
    out
}

fn speak_arpabet(voice: &str, text: &str) -> (Vec<Chunk>, Vec<f32>) {
    let mut chunks = arpabet_chunks(text, Controls::default()).unwrap();
    let pcm = synth_chunks(voice, &mut chunks);
    (chunks, pcm)
}

#[test]
fn audio_length_equals_durations_and_is_clean() {
    for voice in ["klatt-male", "klatt-female", "klatt-robot", "klatt-whisper"] {
        let (chunks, pcm) = speak_arpabet(
            voice,
            "DH AX / K W IH1 K / B R AW1 N / F AA1 K S , / JH AH1 M P S / OW1 V AXR / DH AX / L EY1 Z IY0 / D AO1 G . W AY1 ?",
        );
        let want_ms: f32 = chunks.iter().map(|c| c.dur_ms() + c.pause_after_ms).sum();
        let got_ms = pcm.len() as f32 * 1000.0 / SR as f32;
        assert!(
            (want_ms - got_ms).abs() <= 5.0 * chunks.len() as f32,
            "{voice}: {want_ms} vs {got_ms}"
        );
        assert!(
            pcm.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
            "{voice}"
        );
        assert!(
            rms_dbfs(&pcm) > -45.0,
            "{voice} is too quiet: {}",
            rms_dbfs(&pcm)
        );
    }
}

#[test]
fn pho_round_trip_gives_the_same_audio_length() {
    let (chunks, pcm) = speak_arpabet("klatt-male", "HH AX L OW1 / W ER1 L D .");
    let mut text = String::new();
    print_pho(&chunks[0], &mut text);
    let mut again = parse_pho(&text).unwrap();
    let pcm2 = synth_chunks("klatt-male", &mut again);
    let diff_ms = (pcm.len() as f32 - pcm2.len() as f32).abs() * 1000.0 / SR as f32;
    assert!(diff_ms <= 10.0, "{diff_ms} ms");
    // The printed durations sum to the audio length.
    let sum: f32 = text
        .lines()
        .filter(|l| !l.starts_with(';'))
        .filter_map(|l| l.split_whitespace().nth(1)?.parse::<f32>().ok())
        .sum();
    assert!((sum - pcm.len() as f32 * 1000.0 / SR as f32).abs() <= 5.0);
}

/// Lag of the autocorrelation peak in a window, in samples (with parabolic refinement).
fn period(x: &[f32]) -> f32 {
    let (lo, hi) = (SR as usize / 400, SR as usize / 60);
    let r = |lag: usize| -> f32 {
        x.iter().zip(&x[lag..]).map(|(a, b)| a * b).sum::<f32>() / (x.len() - lag) as f32
    };
    let best = (lo..hi).max_by(|a, b| r(*a).total_cmp(&r(*b))).unwrap();
    let (a, b, c) = (r(best - 1), r(best), r(best + 1));
    best as f32 + 0.5 * (a - c) / (a - 2.0 * b + c)
}

#[test]
fn robot_has_a_constant_period() {
    let (chunks, pcm) = speak_arpabet("klatt-robot", "AA1 / IY1 / UW1 / AA1 / IY1 .");
    let voiced_ms: f32 = chunks[0].dur_ms();
    let n = (voiced_ms * SR as f32 / 1000.0) as usize;
    let mut periods = Vec::new();
    let mut at = 1600;
    while at + 1024 < n - 1600 {
        periods.push(period(&pcm[at..at + 1024]));
        at += 800;
    }
    assert!(periods.len() >= 5);
    for p in &periods {
        assert!(
            (p - 160.0).abs() / 160.0 < 0.01,
            "period {p} (all {periods:?})"
        );
    }
}

#[test]
fn robot_is_as_loud_as_the_default_voice() {
    let level = |v: &str| {
        let (_, pcm) = speak_arpabet(v, "AA1");
        rms_dbfs(&pcm[pcm.len() / 4..pcm.len() / 2])
    };
    let diff = level("klatt-robot") - level("klatt-male");
    assert!(diff.abs() < 6.0, "robot is {diff} dB from male");
}

#[test]
fn rate_and_pitch_controls() {
    let dur = |rate: f32| {
        let mut c = arpabet_chunks(
            "HH AX L OW1 / W ER1 L D .",
            Controls {
                rate,
                ..Controls::default()
            },
        )
        .unwrap();
        synth_chunks("klatt-male", &mut c).len()
    };
    assert!(dur(300.0 / 175.0) as f32 <= 0.7 * dur(1.0) as f32);
    let mut c = arpabet_chunks(
        "AA1",
        Controls {
            pitch_base_hz: Some(200.0),
            ..Controls::default()
        },
    )
    .unwrap();
    let pcm = synth_chunks("klatt-male", &mut c);
    let p = period(&pcm[1600..1600 + 1024]);
    let f0 = SR as f32 / p;
    assert!(f0 > 180.0 && f0 < 240.0, "{f0}");
}

const GOLDEN: &[(&str, &str, &str)] = &[
    ("hello", "klatt-male", "HH AX L OW1 / W ER1 L D ."),
    (
        "fox",
        "klatt-male",
        "DH AX / K W IH1 K / B R AW1 N / F AA1 K S / JH AH1 M P S / OW1 V AXR / DH AX / L EY1 Z IY0 / D AO1 G .",
    ),
    ("question", "klatt-male", "AA1 R / Y UW1 / DH EH1 R ?"),
    (
        "clause",
        "klatt-male",
        "W EH1 L , / AY1 / TH IH1 NG K / S OW1 !",
    ),
    (
        "nasals",
        "klatt-male",
        "M AE1 N IY0 / M EH1 N / S IH1 NG / S AO1 NG Z .",
    ),
    (
        "fricatives",
        "klatt-male",
        "SH IY1 / S EH1 L Z / S IY1 SH EH2 L Z / V EH1 R IY0 / F AE1 S T .",
    ),
    (
        "stops",
        "klatt-male",
        "P IY1 T AXR / B AO1 T / AX / B IH1 G / K AY1 T .",
    ),
    ("female", "klatt-female", "HH AX L OW1 / W ER1 L D ."),
    ("robot", "klatt-robot", "AY1 / AE1 M / AX / R OW1 B AA2 T ."),
    (
        "whisper",
        "klatt-whisper",
        "K IY1 P / IH1 T / S IY1 K R IX0 T .",
    ),
];

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/klatt")
}

fn snr_db(reference: &[f32], test: &[f32]) -> f32 {
    let signal: f64 = reference.iter().map(|v| (*v as f64).powi(2)).sum();
    let noise: f64 = reference
        .iter()
        .zip(test)
        .map(|(a, b)| ((*a - *b) as f64).powi(2))
        .sum();
    if noise <= 0.0 {
        200.0
    } else {
        (10.0 * (signal / noise).log10()) as f32
    }
}

#[test]
fn golden_audio() {
    let update = std::env::var_os("UPDATE_GOLDEN").is_some();
    let dir = fixtures();
    for (name, voice, phones) in GOLDEN {
        let pho_path = dir.join(format!("{name}.pho"));
        let wav_path = dir.join(format!("{name}.wav"));
        if update {
            let (chunks, _) = speak_arpabet(voice, phones);
            let mut text = String::new();
            for c in &chunks {
                print_pho(c, &mut text);
            }
            std::fs::write(&pho_path, text).unwrap();
        }
        // The fixture is the .pho file: the renderer is tested on fixed timed phones.
        let mut chunks = parse_pho(&std::fs::read_to_string(&pho_path).unwrap()).unwrap();
        assert!(
            chunks
                .iter()
                .any(|c| c.units.iter().any(|u| matches!(u, Unit::Seg(_))))
        );
        let pcm = synth_chunks(voice, &mut chunks);
        // Compare what a 16-bit file holds.
        let quantized: Vec<i16> = pcm
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .collect();
        if update {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: SR,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&wav_path, spec).unwrap();
            for s in &quantized {
                w.write_sample(*s).unwrap();
            }
            w.finalize().unwrap();
            continue;
        }
        let stored: Vec<i16> = hound::WavReader::open(&wav_path)
            .unwrap()
            .samples::<i16>()
            .map(|s| s.unwrap())
            .collect();
        assert_eq!(stored.len(), quantized.len(), "{name}: length");
        let to_f = |v: &[i16]| v.iter().map(|s| *s as f32 / 32767.0).collect::<Vec<f32>>();
        let snr = snr_db(&to_f(&stored), &to_f(&quantized));
        let floor = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            60.0
        } else {
            40.0
        };
        assert!(snr >= floor, "{name}: SNR {snr} dB");
    }
}

#[test]
fn rate_calibration_175_words_per_minute() {
    // `-r 175` means 175 words per minute on the benchmark text, pauses included (spec §8.3).
    use speak::text::Frontend;
    let text = include_str!("fixtures/bench.txt");
    let words = text.split_whitespace().count() as f32;
    let fe = Frontend::new(None, None, None).unwrap();
    let wpm = |rate: f32| {
        let mut chunks = fe
            .chunks(
                text,
                Controls {
                    rate,
                    ..Controls::default()
                },
            )
            .unwrap();
        let pcm = synth_chunks("klatt-male", &mut chunks);
        words / (pcm.len() as f32 / SR as f32 / 60.0)
    };
    let native = wpm(1.0);
    assert!((native - 175.0).abs() <= 15.0, "{native} wpm at -r 175");
    // The rate control is roughly proportional.
    let fast = wpm(260.0 / 175.0);
    assert!((fast - 260.0).abs() <= 35.0, "{fast} wpm at -r 260");
}
