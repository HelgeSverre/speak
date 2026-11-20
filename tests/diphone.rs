//! Diphone LPC backend (spec §16.4).

use speak::diphone::backend::DiphoneSynth;
use speak::diphone::build;
use speak::diphone::synth::Excitation;
use speak::dsp::{lpc, pitch};
use speak::klatt::tables::row;
use speak::klatt::{KlattSynth, Voice};
use speak::pho::parse_pho;
use speak::traits::Synthesizer;
use speak::{Phone, Unit};

const SR: u32 = 16000;

/// Every diphone a phone string needs, with silence at both ends.
fn pairs_for(phones: &[Phone]) -> Vec<(Phone, Phone)> {
    let mut seq = vec![Phone::SIL];
    seq.extend_from_slice(phones);
    seq.push(Phone::SIL);
    let mut out: Vec<(Phone, Phone)> = seq.windows(2).map(|w| (w[0], w[1])).collect();
    out.sort();
    out.dedup();
    out
}

const HELLO: &str = "HH 62\nAX 48 0 118\nL 71 0 120\nOW1 182 0 124 40 131 100 105\n_ 60\nW 66 0 108\nER1 200 0 112 30 118 100 95\nL 70 0 95 100 88\nD 70 0 88 100 85\n_ 200\n";

fn hello_voice() -> speak::diphone::VoiceFile {
    use Phone::*;
    let mut pairs = pairs_for(&[HH, AX, L, OW]);
    pairs.extend(pairs_for(&[W, ER, L, D]));
    pairs.sort();
    pairs.dedup();
    build::subset(&pairs).unwrap()
}

fn render_diphone(voice: speak::diphone::VoiceFile, excitation: Excitation, pho: &str) -> Vec<f32> {
    let mut s = DiphoneSynth::new(voice, excitation);
    let mut out = Vec::new();
    for c in parse_pho(pho).unwrap() {
        s.synthesize(&c, &mut out).unwrap();
    }
    out
}

#[test]
fn measured_filter_matches_the_ruled_one() {
    // The AA-AA diphone midpoint has the formants the Klatt table asked for.
    let voice = build::subset(&[(Phone::AA, Phone::AA)]).unwrap();
    let d = voice.get(Phone::AA, Phone::AA).unwrap();
    let mid = &d.frames[d.frames.len() / 2];
    let a = lpc::reflection_to_predictor(&mid.k);
    let peaks: Vec<f32> = lpc::spectral_peaks(&a, SR as f32)
        .into_iter()
        .map(|p| p.0)
        .collect();
    let r = row(Phone::AA);
    for target in [r.f[0].0, r.f[1].0] {
        let best = peaks
            .iter()
            .cloned()
            .min_by(|x, y| (x - target).abs().total_cmp(&(y - target).abs()))
            .unwrap();
        assert!(
            (best - target).abs() / target < 0.10,
            "target {target}, measured {best} ({peaks:?})"
        );
    }
    assert!(
        mid.voicing > 0.8 && (mid.f0 - 110.0).abs() < 3.0,
        "voicing {} f0 {}",
        mid.voicing,
        mid.f0
    );
}

#[test]
fn voicing_detector_agrees_with_the_known_truth() {
    // In the bootstrap voice the truth is known: vowels are voiced, S is not.
    let voice = build::subset(&[
        (Phone::AA, Phone::S),
        (Phone::S, Phone::IY),
        (Phone::SIL, Phone::S),
    ])
    .unwrap();
    let aa_s = voice.get(Phone::AA, Phone::S).unwrap();
    assert!(
        aa_s.frames[..aa_s.boundary - 2]
            .iter()
            .all(|f| f.voicing > 0.7)
    );
    assert!(
        aa_s.frames[aa_s.boundary + 3..]
            .iter()
            .all(|f| f.voicing < 0.5),
        "{:?}",
        aa_s.frames.iter().map(|f| f.voicing).collect::<Vec<_>>()
    );
    let s_iy = voice.get(Phone::S, Phone::IY).unwrap();
    assert!(
        s_iy.frames[s_iy.boundary + 4..]
            .iter()
            .all(|f| f.voicing > 0.7)
    );
}

#[test]
fn same_pho_through_both_backends() {
    let mut klatt = Vec::new();
    let mut k = KlattSynth::new(Voice::find("klatt-male").unwrap());
    for c in parse_pho(HELLO).unwrap() {
        k.synthesize(&c, &mut klatt).unwrap();
    }
    let diphone = render_diphone(hello_voice(), Excitation::Glottal, HELLO);
    let diff_ms = (klatt.len() as f32 - diphone.len() as f32).abs() / 16.0;
    assert!(diff_ms <= 10.0, "durations differ by {diff_ms} ms");
    assert!(diphone.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    assert!(speak::dsp::rms_dbfs(&diphone) > -40.0);

    // F0 of the diphone output follows the targets within 5 % on voiced frames.
    let chunk = parse_pho(HELLO).unwrap().remove(0);
    let mut targets: Vec<(f32, f32)> = Vec::new();
    let mut t = 0.0;
    for u in &chunk.units {
        if let Unit::Seg(s) = u {
            targets.extend(s.f0.iter().map(|(pos, hz)| (t + pos * s.dur_ms, *hz)));
        }
        t += u.dur_ms();
    }
    let target_at = |ms: f32| {
        let i = targets
            .iter()
            .position(|p| p.0 >= ms)
            .unwrap_or(targets.len() - 1)
            .max(1);
        let (a, b) = (targets[i - 1], targets[i]);
        if ms <= a.0 {
            a.1
        } else if ms >= b.0 {
            b.1
        } else {
            a.1 + (b.1 - a.1) * (ms - a.0) / (b.0 - a.0)
        }
    };
    let tracked = pitch::track(&diphone, SR);
    let (mut ok, mut voiced) = (0, 0);
    for (i, p) in tracked.iter().enumerate() {
        let ms = i as f32 * 5.0;
        // Vowels and sonorants away from the edges.
        if p.f0 > 0.0 && p.voicing > 0.85 && ms > 80.0 && ms < 600.0 {
            voiced += 1;
            if (p.f0 - target_at(ms)).abs() / target_at(ms) < 0.05 {
                ok += 1;
            }
        }
    }
    assert!(voiced > 30, "only {voiced} voiced frames");
    assert!(
        ok as f32 >= 0.9 * voiced as f32,
        "{ok}/{voiced} voiced frames within 5 % of the F0 targets"
    );
}

#[test]
fn joins_are_smooth_and_excitations_work() {
    for e in [Excitation::Pulse, Excitation::Glottal] {
        let pcm = render_diphone(hello_voice(), e, HELLO);
        let worst = pcm
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.5, "{e:?}: sample-to-sample jump {worst}");
    }
    assert!(Excitation::parse("nonsense").is_err());
}

#[test]
fn f0_tracker_on_klatt_output() {
    for f0 in [90.0f32, 130.0, 180.0, 250.0] {
        let pho = format!("AA1 400 0 {f0} 100 {f0}\n");
        let mut s = KlattSynth::new(Voice {
            flutter: 0.0,
            ..Voice::find("klatt-male").unwrap()
        });
        let mut pcm = Vec::new();
        s.synthesize(&parse_pho(&pho).unwrap().remove(0), &mut pcm)
            .unwrap();
        let t = pitch::track(&pcm, SR);
        let voiced: Vec<_> = t[8..t.len() - 12].iter().collect();
        let ok = voiced
            .iter()
            .filter(|p| (p.f0 - f0).abs() / f0 < 0.02)
            .count();
        assert!(
            ok as f32 >= 0.95 * voiced.len() as f32,
            "{f0} Hz: {ok}/{}",
            voiced.len()
        );
    }
}

#[test]
fn installed_bootstrap_voice_covers_the_inventory() {
    let data = speak::cli::pipeline::data_dirs();
    let path = speak::diphone::backend::voice_path(&data, "boot");
    if !path.is_file() {
        println!("skipped: boot.sdv is not built");
        return;
    }
    let v = speak::diphone::VoiceFile::read(&path).unwrap();
    for (a, b) in speak::diphone::inventory::all() {
        assert!(
            v.get(a, b).is_some_and(|d| !d.frames.is_empty()),
            "missing {a:?}-{b:?}"
        );
    }
}

#[test]
fn residual_excitation_reconstructs_the_source_rendering() {
    use speak::diphone::select::Plan;
    use speak::diphone::synth::{LpcSynth, residual_excitation};
    // The AA-AA diphone at its original pitch and timing gives back the Klatt rendering it came from.
    let voice = build::subset(&[(Phone::AA, Phone::AA)]).unwrap();
    let d = voice.get(Phone::AA, Phone::AA).unwrap().clone();
    assert_eq!(d.residual.len(), d.frames.len() * 80);
    assert!(d.epochs.len() >= 20, "{} pitch marks", d.epochs.len());
    let frames: Vec<_> = d
        .frames
        .iter()
        .map(|f| speak::dsp::lpc::LpcFrame {
            f0: 110.0,
            voicing: 1.0,
            ..f.clone()
        })
        .collect();
    let plan = Plan {
        map: (0..frames.len()).map(|j| Some((0, j as f32))).collect(),
        frames,
        units: Vec::new(),
    };
    let excitation = residual_excitation(&plan, &voice);
    let mut got = Vec::new();
    LpcSynth::new(Excitation::Residual)
        .render_excited(&plan.frames, &excitation, 1.0, &mut got)
        .unwrap();

    // The source: the same prompt, the same cut.
    let mut klatt = KlattSynth::new(build::source_voice());
    let mut source = Vec::new();
    klatt
        .synthesize(&build::prompt(Phone::AA, Phone::AA), &mut source)
        .unwrap();
    let start = (build::landmarks(Phone::AA, Phone::AA).0 / 5.0).round() as usize * 80;
    let reference = &source[start..start + got.len()];
    // Skip the filter's warm-up and the last period.
    let (a, b) = (1200, got.len() - 400);
    let signal: f64 = reference[a..b].iter().map(|v| (*v as f64).powi(2)).sum();
    let noise: f64 = reference[a..b]
        .iter()
        .zip(&got[a..b])
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum();
    let snr = 10.0 * (signal / noise.max(1e-20)).log10();
    assert!(snr >= 20.0, "SNR {snr:.1} dB");
}

#[test]
fn autolabeller_finds_the_boundaries_of_a_simulated_speaker() {
    use speak::diphone::{autolabel, prompts};
    // Prompts, a different voice at a different speed with jittered durations, and the truth.
    let dir = std::env::temp_dir().join(format!("speak-autolabel-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let list: Vec<_> = prompts::all().into_iter().take(60).collect();
    std::fs::write(dir.join("prompts.tsv"), prompts::tsv(&list)).unwrap();
    assert_eq!(
        autolabel::simulate(&dir, "klatt-female", 0.9, 0.1, 1, None).unwrap(),
        60
    );
    let (voice, report) = autolabel::build_from_dir(&dir, "t", |_, _| {}).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(report.recorded, 60);
    assert!(report.rejected.len() <= 3, "rejected {:?}", report.rejected);
    let (median, p90) = (
        report.percentile(0.5).unwrap(),
        report.percentile(0.9).unwrap(),
    );
    assert!(
        median <= 15.0 && p90 <= 40.0,
        "boundary error median {median} ms, p90 {p90} ms"
    );
    assert!(
        (report.median_f0 - 200.0).abs() < 15.0,
        "speaking pitch {}",
        report.median_f0
    );
    assert!(voice.description.contains("source=recorded") && voice.description.contains("f0="));
    for d in &voice.diphones {
        assert!(!d.frames.is_empty() && d.boundary <= d.frames.len());
        assert_eq!(d.residual.len(), d.frames.len() * 80);
    }
}

#[test]
fn guide_length_matches_the_prompt_durations() {
    use speak::diphone::{autolabel, prompts};
    for p in prompts::all().iter().step_by(400) {
        let pcm = autolabel::render_guide(p).unwrap();
        let want = prompts::guide(p).dur_ms();
        assert!(
            (pcm.len() as f32 / 16.0 - want).abs() <= 5.0,
            "{}: {} vs {want}",
            p.id,
            pcm.len() as f32 / 16.0
        );
    }
}

#[test]
fn diphone_rate_matches_the_formant_backend() {
    // Both backends share the duration model: the same text gives the same length, so the
    // 175 wpm calibration of the formant backend holds here too.
    use speak::text::{Frontend, RuleProsody};
    use speak::traits::Prosody;
    let data = speak::cli::pipeline::data_dirs();
    let path = speak::diphone::backend::voice_path(&data, "boot");
    if !path.is_file() {
        println!("skipped: boot.sdv is not built");
        return;
    }
    let text = include_str!("fixtures/bench.txt");
    let fe = Frontend::new(None, None, None).unwrap();
    let mut d = DiphoneSynth::new(
        speak::diphone::VoiceFile::read(&path).unwrap(),
        Excitation::Glottal,
    );
    let mut k = KlattSynth::new(Voice::find("klatt-male").unwrap());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for mut c in fe.chunks(text, speak::Controls::default()).unwrap() {
        RuleProsody.apply(&mut c, &k.spec().prosody);
        k.synthesize(&c, &mut a).unwrap();
        d.synthesize(&c, &mut b).unwrap();
    }
    assert_eq!(a.len(), b.len());
    let wpm = text.split_whitespace().count() as f32 / (b.len() as f32 / 16000.0 / 60.0);
    assert!((wpm - 175.0).abs() <= 15.0, "{wpm} wpm");
}
