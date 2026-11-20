//! Intelligibility against macOS `say` (spec §16.5). Whisper and `say` are measuring
//! instruments only. Run with: cargo test -- --ignored intelligibility --nocapture

use std::path::{Path, PathBuf};
use std::process::Command;

use speak::Controls;
use speak::audio::FileSink;
use speak::cli::backends;
use speak::cli::pipeline::data_dirs;
use speak::cli::wer::{Wer, wer};
use speak::text::{Frontend, RuleProsody};
use speak::traits::{AudioSink, Prosody, SynthOptions, VoiceSpec};

const SENTENCES: &str = include_str!("fixtures/sentences.txt");

fn have(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn synth(backend: &str, voice: &str, lts: &str, text: &str, path: &Path) -> bool {
    let all = backends();
    let Some(b) = all.iter().find(|b| b.id() == backend) else {
        return false;
    };
    let Ok(mut s) = b.open(
        &VoiceSpec { name: voice.into() },
        &SynthOptions::default(),
        &data_dirs(),
    ) else {
        return false;
    };
    // CMUdict is used when the user has built it (SPEAK_DATA_DIR or the default data directory).
    let fe = Frontend::new(None, Some(&data_dirs().data), Some(lts)).unwrap();
    let mut sink = FileSink::create(path).unwrap();
    let spec = s.spec();
    sink.begin(spec.sample_rate).unwrap();
    for mut c in fe.chunks(text, Controls::default()).unwrap() {
        RuleProsody.apply(&mut c, &spec.prosody);
        s.synthesize(&c, &mut sink).unwrap();
    }
    sink.finish().unwrap();
    true
}

/// Transcribes all files with one whisper run; returns one transcript per file.
fn transcribe(files: &[PathBuf], dir: &Path) -> Vec<String> {
    let out = dir.join("w");
    let status = Command::new("whisper")
        .args(files)
        .args([
            "--model",
            "base.en",
            "--language",
            "en",
            "--fp16",
            "False",
            "--output_format",
            "txt",
            "--output_dir",
        ])
        .arg(&out)
        .output()
        .expect("whisper runs");
    assert!(
        status.status.success(),
        "whisper failed: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    files
        .iter()
        .map(|f| {
            std::fs::read_to_string(out.join(f.file_stem().unwrap()).with_extension("txt"))
                .unwrap_or_default()
                .replace('\n', " ")
        })
        .collect()
}

fn score(name: &str, files: &[PathBuf], dir: &Path, sentences: &[&str]) -> Wer {
    let mut total = Wer::default();
    for (reference, hyp) in sentences.iter().zip(transcribe(files, dir)) {
        let w = wer(reference, &hyp);
        if w.errors > 0 {
            println!(
                "  [{name}] {reference}\n  {:>width$} {}",
                "->",
                hyp.trim(),
                width = name.len() + 4
            );
        }
        total.add(&w);
    }
    let mut pairs = total.confusions.clone();
    pairs.sort();
    println!(
        "{name} WER {:.1} %  ({} errors in {} words)",
        total.rate() * 100.0,
        total.errors,
        total.reference_words
    );
    println!(
        "  confusions: {}",
        pairs
            .iter()
            .map(|(r, h)| format!(
                "{}>{}",
                if r.is_empty() { "+" } else { r },
                if h.is_empty() { "-" } else { h }
            ))
            .collect::<Vec<_>>()
            .join(" ")
    );
    total
}

#[test]
#[ignore]
fn intelligibility() {
    if !have("whisper") {
        println!("skipped: whisper is not on PATH");
        return;
    }
    let sentences: Vec<&str> = SENTENCES.lines().filter(|l| !l.trim().is_empty()).collect();
    let dir = std::env::temp_dir().join(format!("speak-intel-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    if have("say") && have("ffmpeg") {
        let mut files = Vec::new();
        for (i, s) in sentences.iter().enumerate() {
            let aiff = dir.join(format!("say{i:02}.aiff"));
            let wav = dir.join(format!("say{i:02}.wav"));
            assert!(
                Command::new("say")
                    .arg("-o")
                    .arg(&aiff)
                    .arg(s)
                    .status()
                    .unwrap()
                    .success()
            );
            assert!(
                Command::new("ffmpeg")
                    .args(["-v", "error", "-y", "-i"])
                    .arg(&aiff)
                    .arg(&wav)
                    .status()
                    .unwrap()
                    .success()
            );
            files.push(wav);
        }
        score("say", &files, &dir, &sentences);
    } else {
        println!("say baseline skipped: say or ffmpeg missing");
    }

    // (label, backend, voice, lts, WER gate)
    let systems: &[(&str, &str, &str, &str, f32)] = &[
        ("klatt (rules)", "klatt", "klatt-male", "rules", 0.35),
        ("klatt (cart)", "klatt", "klatt-male", "cart", 0.20),
        ("diphone (boot)", "diphone", "boot", "cart", 0.35),
        ("diphone (simtest)", "diphone", "simtest", "cart", 0.40),
    ];
    for (label, backend, voice, lts, gate) in systems {
        let mut files = Vec::new();
        for (i, s) in sentences.iter().enumerate() {
            let wav = dir.join(format!("{}{i:02}.wav", label.replace([' ', '(', ')'], "")));
            if !synth(backend, voice, lts, s, &wav) {
                println!("{label} skipped: voice not available");
                files.clear();
                break;
            }
            files.push(wav);
        }
        if files.is_empty() {
            continue;
        }
        let w = score(label, &files, &dir, &sentences);
        assert!(
            w.rate() <= *gate,
            "{label} WER {:.1} % is above the gate {:.0} %",
            w.rate() * 100.0,
            gate * 100.0
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
