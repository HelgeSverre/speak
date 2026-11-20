//! Renderer tests: formants and levels (spec §16.2).

use crate::dsp::{lpc, peak, rms_dbfs};
use crate::klatt::tables::row;
use crate::klatt::voice::Voice;
use crate::klatt::{Frame, Renderer, SR};
use crate::phone::Phone;

/// Steady frames for `ms` milliseconds, built by `f`.
pub fn steady(ms: f32, f: impl Fn(&mut Frame)) -> Vec<Frame> {
    let mut frame = Frame::SILENCE;
    f(&mut frame);
    vec![frame; (ms / 5.0) as usize]
}

pub fn render(voice: &Voice, frames: &[Frame]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut r = Renderer::new(voice);
    r.render(frames, 1.0, &mut out).unwrap();
    out
}

fn male_flat() -> Voice {
    Voice {
        flutter: 0.0,
        ..Voice::find("klatt-male").unwrap()
    }
}

pub fn vowel_frames(p: Phone, ms: f32, f0: f32, voice: &Voice) -> Vec<Frame> {
    let r = row(p);
    steady(ms, |f| {
        for k in 0..3 {
            f.f[k] = r.f[k].0;
            f.b[k] = r.b[k].0;
        }
        f.av = 60.0;
        f.f0 = f0;
        f.oq = voice.oq;
        f.tl = voice.tl;
    })
}

fn obstruent_frames(p: Phone, ms: f32) -> Vec<Frame> {
    let r = row(p);
    steady(ms, |f| {
        for k in 0..3 {
            f.f[k] = r.f[k].0;
            f.b[k] = r.b[k].0;
        }
        f.af = 60.0;
        f.a = r.a;
        f.ab = r.ab;
    })
}

fn middle(x: &[f32]) -> &[f32] {
    &x[x.len() / 4..x.len() * 3 / 4]
}

#[test]
fn monophthong_formants_match_table() {
    let voice = male_flat();
    for p in [
        Phone::IY,
        Phone::IH,
        Phone::EH,
        Phone::AE,
        Phone::AA,
        Phone::AO,
        Phone::AH,
        Phone::UH,
        Phone::UW,
        Phone::AX,
        Phone::IX,
    ] {
        let pcm = render(&voice, &vowel_frames(p, 300.0, 110.0, &voice));
        let mid = pcm.len() / 2;
        let frame = &pcm[mid - 200..mid + 200];
        let est = lpc::formants(frame, SR as f32, 18);
        let r = row(p);
        for (k, target) in [r.f[0].0, r.f[1].0].into_iter().enumerate() {
            let best = est
                .iter()
                .cloned()
                .min_by(|a, b| (a - target).abs().total_cmp(&(b - target).abs()))
                .unwrap();
            assert!(
                (best - target).abs() / target < 0.08,
                "{:?} F{} target {target}, estimated {best} (all {est:?})",
                p,
                k + 1
            );
        }
    }
}

#[test]
fn levels() {
    let voice = male_flat();
    let aa = rms_dbfs(middle(&render(
        &voice,
        &vowel_frames(Phone::AA, 300.0, 110.0, &voice),
    )));
    assert!((aa + 20.0).abs() <= 1.0, "AA {aa} dBFS");
    for (p, rel, tol) in [
        (Phone::SH, -9.0, 3.0),
        (Phone::S, -14.0, 3.0),
        (Phone::F, -21.0, 4.0),
    ] {
        let db = rms_dbfs(middle(&render(&voice, &obstruent_frames(p, 300.0)))) - aa;
        assert!(
            (db - rel).abs() <= tol,
            "{p:?} {db} dB re AA, want {rel} ± {tol}"
        );
    }
    // HH: aspiration through the cascade with AA formants.
    let r = row(Phone::AA);
    let hh = steady(300.0, |f| {
        for k in 0..3 {
            f.f[k] = r.f[k].0;
            f.b[k] = r.b[k].0;
        }
        f.b[0] = 300.0;
        f.ah = 55.0;
    });
    let db = rms_dbfs(middle(&render(&voice, &hh))) - aa;
    assert!((db + 16.0).abs() <= 4.0, "HH {db} dB re AA");
}

#[test]
fn no_clipping_and_deterministic() {
    let voice = Voice::find("klatt-male").unwrap();
    let frames = vowel_frames(Phone::AA, 200.0, 110.0, &voice);
    let a = render(&voice, &frames);
    let b = render(&voice, &frames);
    assert_eq!(a, b);
    assert_eq!(a.len(), 200 * 16);
    assert!(peak(&a) <= 1.0);
}

#[test]
#[ignore]
fn print_levels() {
    let voice = male_flat();
    let aa = rms_dbfs(middle(&render(
        &voice,
        &vowel_frames(Phone::AA, 300.0, 110.0, &voice),
    )));
    println!("AA {aa:.2} dBFS");
    for p in [
        Phone::SH,
        Phone::S,
        Phone::F,
        Phone::TH,
        Phone::T,
        Phone::K,
        Phone::P,
        Phone::CH,
    ] {
        println!(
            "{p:?} {:.2} re AA",
            rms_dbfs(middle(&render(&voice, &obstruent_frames(p, 300.0)))) - aa
        );
    }
    let r = row(Phone::AA);
    let hh = steady(300.0, |f| {
        for k in 0..3 {
            f.f[k] = r.f[k].0;
            f.b[k] = r.b[k].0;
        }
        f.b[0] = 300.0;
        f.ah = 55.0;
    });
    println!(
        "HH {:.2} re AA",
        rms_dbfs(middle(&render(&voice, &hh))) - aa
    );
    for p in [Phone::IY, Phone::UW, Phone::AE] {
        println!(
            "{p:?} {:.2} dBFS",
            rms_dbfs(middle(&render(
                &voice,
                &vowel_frames(p, 300.0, 110.0, &voice)
            )))
        );
    }
}

#[test]
#[ignore]
fn print_formants() {
    let voice = male_flat();
    for p in [Phone::AO, Phone::AA, Phone::UW, Phone::IY, Phone::UH] {
        let pcm = render(&voice, &vowel_frames(p, 300.0, 110.0, &voice));
        let mid = pcm.len() / 2;
        for (w, o) in [
            (200, 18),
            (320, 18),
            (400, 16),
            (400, 20),
            (600, 18),
            (300, 22),
        ] {
            let est = lpc::formants(&pcm[mid - w..mid + w], SR as f32, o);
            println!("{p:?} w{} o{o}: {:?}", 2 * w, &est[..est.len().min(4)]);
        }
    }
}
