//! Sources, context rules and voice quality (spec §16.2).

use speak::Controls;
use speak::cli::pipeline::arpabet_chunks;
use speak::dsp::lpc;
use speak::dsp::source::Klglott88;
use speak::dsp::source_lf::Lf;
use speak::klatt::{KlattSynth, SR, Voice};
use speak::text::RuleProsody;
use speak::traits::{GlottalSource, Prosody, SynthOptions, Synthesizer, VoiceQuality};

/// Level in dB of the harmonics of a 100 Hz source between `lo` and `hi` Hz.
fn band_db(src: &mut dyn GlottalSource, q: &VoiceQuality, lo: f32, hi: f32) -> f32 {
    let sr = 64000.0f32;
    let x: Vec<f32> = (0..64000).map(|_| src.next(100.0, q)).collect();
    let mut power = 0.0f64;
    let mut f = 100.0f32;
    while f <= hi {
        if f >= lo {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (n, v) in x.iter().enumerate() {
                let w = 2.0 * std::f64::consts::PI * f as f64 * n as f64 / sr as f64;
                re += *v as f64 * w.cos();
                im += *v as f64 * w.sin();
            }
            power += re * re + im * im;
        }
        f += 100.0;
    }
    10.0 * power.log10() as f32
}

#[test]
fn lf_and_klglott88_have_similar_tilt() {
    let q = VoiceQuality::default();
    let tilt =
        |s: &mut dyn GlottalSource| band_db(s, &q, 500.0, 1000.0) - band_db(s, &q, 2000.0, 3000.0);
    let (k, l) = (
        tilt(&mut Klglott88::new(64000.0)),
        tilt(&mut Lf::new(64000.0)),
    );
    println!("tilt 0.5–1 kHz over 2–3 kHz: klglott88 {k:.1} dB, lf {l:.1} dB");
    assert!((k - l).abs() <= 3.0, "klglott88 {k} dB vs lf {l} dB");
}

fn speak(voice: &str, opts: &[(&str, &str)], phones: &str) -> Vec<f32> {
    let mut o = SynthOptions::default();
    for (k, v) in opts {
        o.params.insert(k.to_string(), v.to_string());
    }
    let mut s = KlattSynth::new(Voice::find(voice).unwrap().with_options(&o).unwrap());
    let spec = s.spec();
    let mut out = Vec::new();
    for mut c in arpabet_chunks(phones, Controls::default()).unwrap() {
        RuleProsody.apply(&mut c, &spec.prosody);
        s.synthesize(&c, &mut out).unwrap();
    }
    out
}

#[test]
fn all_three_sources_speak() {
    for source in ["klglott88", "lf", "impulse"] {
        let pcm = speak("klatt-male", &[("source", source)], "T EH1 S T IX NG .");
        assert!(pcm.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
        let level = speak::dsp::rms_dbfs(&pcm[pcm.len() / 4..pcm.len() / 2]);
        assert!(level > -40.0 && level < -8.0, "{source}: {level} dBFS");
    }
}

/// F2 just after the release of an 80 ms stop (timed input, so the release time is known).
fn f2_at_onset(vowel: &str) -> f32 {
    let pho = format!("G 80\n{vowel}1 220 0 110 100 100\n");
    let mut chunks = speak::pho::parse_pho(&pho).unwrap();
    let mut s = KlattSynth::new(Voice {
        flutter: 0.0,
        ..Voice::find("klatt-male").unwrap()
    });
    let mut pcm = Vec::new();
    s.synthesize(&chunks.remove(0), &mut pcm).unwrap();
    let at = (0.100 * SR as f32) as usize;
    let f = lpc::formants(&pcm[at - 200..at + 200], SR as f32, 18);
    // F1 is low at a stop release; F2 is the next peak above 700 Hz.
    f.into_iter().find(|x| *x > 700.0).unwrap()
}

#[test]
fn velar_onset_depends_on_the_vowel() {
    let (front, back) = (f2_at_onset("IY"), f2_at_onset("UW"));
    assert!(
        front - back >= 600.0,
        "F2 after G: {front} before IY, {back} before UW"
    );
}

#[test]
fn creaky_voice_alternates_period_lengths() {
    // Timed input at a flat 90 Hz, and the renderer's own pitch marks: exact period lengths.
    let mut chunks = speak::pho::parse_pho("AA1 400 0 90 100 90\n").unwrap();
    let mut s = KlattSynth::new(Voice {
        flutter: 0.0,
        jitter: 0.0,
        ..Voice::find("klatt-creaky").unwrap()
    });
    s.renderer_mut().epochs = Some(Vec::new());
    let mut pcm = Vec::new();
    s.synthesize(&chunks.remove(0), &mut pcm).unwrap();
    let epochs = s.renderer_mut().epochs.take().unwrap();
    let periods: Vec<f32> = epochs.windows(2).map(|w| (w[1] - w[0]) as f32).collect();
    assert!(periods.len() > 20);
    let nominal = SR as f32 / 90.0;
    // Skip the onset and the creaky ending; every period is 10–15 % off nominal, alternating.
    let body = &periods[2..periods.len() * 2 / 3];
    for pair in body.as_chunks::<2>().0 {
        for p in pair {
            let off = (p - nominal).abs() / nominal;
            assert!(
                (0.09..=0.16).contains(&off),
                "period {p} vs nominal {nominal}: {periods:?}"
            );
        }
        assert!(
            (pair[0] - nominal) * (pair[1] - nominal) < 0.0,
            "not alternating: {pair:?}"
        );
    }
}

#[test]
fn analyze_reports_table_formants() {
    let pcm = speak("klatt-male", &[("flutter", "0")], "AA1");
    let tsv = speak::cli::analyze::analyze(&pcm, SR, false, true, false);
    let rows: Vec<Vec<f32>> = tsv
        .lines()
        .skip(1)
        .map(|l| l.split('\t').filter_map(|v| v.parse().ok()).collect())
        .collect();
    let mid = &rows[rows.len() / 3];
    assert!((mid[1] - 700.0).abs() / 700.0 < 0.08, "F1 {}", mid[1]);
    assert!((mid[2] - 1220.0).abs() / 1220.0 < 0.08, "F2 {}", mid[2]);
}
