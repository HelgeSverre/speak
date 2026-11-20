//! CLI behaviour (spec §3).

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn speak(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_speak"))
        .args(args)
        .env("SPEAK_LIGHTS", "0")
        .env("SPEAK_DATA_DIR", "/nonexistent")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn speak_stdin(args: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_speak"))
        .args(args)
        .env("SPEAK_LIGHTS", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("speak-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn dump_stages_are_stable() {
    let o = speak(&[
        "--dump",
        "text",
        "Dr. Smith paid $3.50 on 12/03/2024 at 5pm.",
    ]);
    assert_eq!(
        out(&o).trim(),
        "doctor Smith paid three dollars fifty on december third twenty twenty four at five p m."
    );
    assert_eq!(
        out(&speak(&["--dump", "phonemes", "Hello world."])).trim(),
        "HH AX L OW1 / W ER1 L D ."
    );
    assert_eq!(
        out(&speak(&["--dump", "phonemes", "butter"])).trim(),
        "B AH1 DX AXR"
    );
    let chunks = out(&speak(&[
        "--dump",
        "chunks",
        "First sentence. Second one, with a clause! Third?",
    ]));
    let marks: Vec<&str> = chunks
        .lines()
        .map(|l| l.split('\t').nth(1).unwrap())
        .collect();
    assert_eq!(marks, [".", "!", "?"]);
    let pho = out(&speak(&["--dump", "pho", "Hello."]));
    assert!(pho.starts_with("; chunk 0"));
    assert!(pho.lines().count() >= 5);
    let frames = out(&speak(&["--dump", "frames", "Hi."]));
    assert!(frames.starts_with("t_ms\tf0\tav"));
    assert!(speak(&["--dump", "nonsense", "x"]).status.code() == Some(2));
}

#[test]
fn stdin_and_files() {
    let o = speak_stdin(&["--dump", "phonemes"], "qzxwv blorptastic 1969\n");
    assert!(o.status.success());
    let text = out(&o);
    assert!(
        !text
            .chars()
            .any(|c| c.is_ascii_digit() && !"012".contains(c)),
        "{text}"
    );
    assert!(text.split('/').count() >= 5);

    let f = tmp("in.txt");
    std::fs::write(&f, "Line one.\nLine two.\n").unwrap();
    let wav = tmp("s.wav");
    assert!(
        speak(&["-f", f.to_str().unwrap(), "-o", wav.to_str().unwrap()])
            .status
            .success()
    );
    let reader = hound::WavReader::open(&wav).unwrap();
    assert_eq!(reader.spec().sample_rate, 16000);
    assert!(reader.duration() > 16000, "longer than a second");
}

#[test]
fn usage_errors_exit_2() {
    let f = tmp("both.txt");
    std::fs::write(&f, "x").unwrap();
    let o = speak(&["-f", f.to_str().unwrap(), "positional"]);
    assert_eq!(o.status.code(), Some(2), "{}", err(&o));
    assert!(err(&o).starts_with("speak: "));
    assert_eq!(
        speak(&["-v", "nosuchvoice", "--dump", "text", "x"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        speak(&["-b", "nosuchbackend", "--dump", "text", "x"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        speak(&["-P", "bogus=1", "--dump", "text", "x"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        speak(&["-P", "oq=7", "--dump", "text", "x"]).status.code(),
        Some(2)
    );
    assert_eq!(
        speak(&["--pitch", "loud", "--dump", "text", "x"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        speak(&["--phonemes", "-o", tmp("x.wav").to_str().unwrap(), "XX"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        speak(&["--melody", "H9", "--dump", "pho", "la"])
            .status
            .code(),
        Some(2)
    );
}

#[test]
fn empty_input_is_success_without_audio() {
    let wav = tmp("empty.wav");
    let o = speak_stdin(&["-o", wav.to_str().unwrap()], "");
    assert!(o.status.success(), "{}", err(&o));
    assert_eq!(hound::WavReader::open(&wav).unwrap().duration(), 0);
}

#[test]
fn listings() {
    let voices = out(&speak(&["-v", "?"]));
    assert!(voices.lines().any(|l| l.starts_with("klatt-male")
        && l.contains("en_US")
        && l.contains("built-in")
        && l.contains("# ")));
    assert_eq!(out(&speak(&["--list-voices"])), voices);
    let backends = out(&speak(&["--list-backends"]));
    assert!(backends.contains("klatt") && backends.contains("16000"));
    assert!(out(&speak(&["--sing=?"])).contains("daisy"));
}

#[test]
fn timing_and_verbose_go_to_stderr() {
    let wav = tmp("t.wav");
    let o = speak(&[
        "-o",
        wav.to_str().unwrap(),
        "--timing",
        "-V",
        "The cat sat.",
    ]);
    assert!(o.status.success());
    let e = err(&o);
    for key in [
        "backend",
        "voice",
        "sample_rate",
        "chunks",
        "startup_ms",
        "frontend_init_ms",
        "voice_open_ms",
        "frontend_ms",
        "ttfa_ms",
        "synth_total_ms",
        "audio_s",
        "rtf",
        "underruns",
        "lexicon_hits",
    ] {
        assert!(e.lines().any(|l| l.starts_with(key)), "missing {key}");
    }
    assert!(
        e.contains("from builtin") && (e.contains("from rules") || e.contains("from cart")),
        "{e}"
    );
    assert!(out(&o).is_empty());
    let o = speak(&["-o", wav.to_str().unwrap(), "--timing=json", "Hi."]);
    let e = err(&o);
    assert!(e.trim().starts_with('{') && e.contains("\"rtf\":"));
}

#[test]
fn phoneme_input_and_controls() {
    let wav = tmp("p.wav");
    let fast = tmp("fast.wav");
    assert!(
        speak(&[
            "--phonemes",
            "-o",
            wav.to_str().unwrap(),
            "HH AX L OW1 / W ER1 L D ."
        ])
        .status
        .success()
    );
    assert!(
        speak(&[
            "--phonemes",
            "-r",
            "300",
            "-o",
            fast.to_str().unwrap(),
            "HH AX L OW1 / W ER1 L D ."
        ])
        .status
        .success()
    );
    let d = |p: &std::path::Path| hound::WavReader::open(p).unwrap().duration() as f32;
    assert!(d(&fast) <= 0.7 * d(&wav));
    let pho = out(&speak(&[
        "--pitch",
        "200",
        "--phonemes",
        "--dump",
        "pho",
        "AA1",
    ]));
    assert!(
        pho.split_whitespace()
            .filter_map(|t| t.parse::<f32>().ok())
            .any(|v| (200.0..300.0).contains(&v))
    );
    // Singing: one note per syllable, in time.
    let sung = tmp("sung.wav");
    assert!(
        speak(&[
            "--melody",
            "C4 D4 E4:2",
            "--tempo",
            "120",
            "-o",
            sung.to_str().unwrap(),
            "la la la"
        ])
        .status
        .success()
    );
    let secs = d(&sung) / 16000.0;
    assert!((secs - 2.0).abs() < 0.15, "{secs}");
}
