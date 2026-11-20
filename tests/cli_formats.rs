//! File formats, config, embedded commands, exit codes and signals through the binary (spec §3, §13).

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn home() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("speak-home-{}", std::process::id()));
    std::fs::create_dir_all(dir.join(".config/speak")).unwrap();
    dir
}

fn cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_speak"));
    c.env("SPEAK_LIGHTS", "0")
        .env("HOME", home())
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("SPEAK_CONFIG")
        .env_remove("SPEAK_VOICE")
        .env_remove("SPEAK_BACKEND")
        .env("SPEAK_DATA_DIR", home().join("data"))
        .stdin(Stdio::null());
    c
}

fn speak(args: &[&str]) -> Output {
    cmd().args(args).output().unwrap()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn tmp(name: &str) -> PathBuf {
    home().join(name)
}

fn wav_seconds(path: &PathBuf) -> f32 {
    let r = hound::WavReader::open(path).unwrap();
    r.duration() as f32 / r.spec().sample_rate as f32
}

#[test]
fn containers_and_extension_inference() {
    let aiff = tmp("a.aiff");
    assert!(
        speak(&["-o", aiff.to_str().unwrap(), "Hello world."])
            .status
            .success()
    );
    let b = std::fs::read(&aiff).unwrap();
    assert_eq!(&b[..4], b"FORM");
    assert_eq!(&b[8..12], b"AIFF");
    // FORM size and SSND size are patched; frame count matches.
    let form = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    assert_eq!(form + 8, b.len());
    let frames = u32::from_be_bytes([b[22], b[23], b[24], b[25]]) as usize;
    assert_eq!(frames * 2, b.len() - 54);
    assert_eq!(
        &b[28..38],
        &[0x40, 0x0C, 0xFA, 0, 0, 0, 0, 0, 0, 0],
        "16000 Hz as 80-bit extended"
    );

    let noext = tmp("noext");
    assert!(
        speak(&["-o", noext.to_str().unwrap(), "Hello world."])
            .status
            .success()
    );
    assert_eq!(
        &std::fs::read(&noext).unwrap()[..4],
        b"FORM",
        "AIFF is say's default"
    );

    let wav = tmp("b.wav");
    assert!(
        speak(&[
            "-o",
            wav.to_str().unwrap(),
            "--data-format=LEF32@48000",
            "--channels",
            "2",
            "Hello world."
        ])
        .status
        .success()
    );
    let r = hound::WavReader::open(&wav).unwrap();
    assert_eq!(
        (
            r.spec().sample_rate,
            r.spec().channels,
            r.spec().sample_format
        ),
        (48000, 2, hound::SampleFormat::Float)
    );

    let raw = tmp("c.raw");
    assert!(
        speak(&["-o", raw.to_str().unwrap(), "Hello world."])
            .status
            .success()
    );
    let plain = tmp("c.wav");
    assert!(
        speak(&["-o", plain.to_str().unwrap(), "Hello world."])
            .status
            .success()
    );
    assert_eq!(
        std::fs::read(&raw).unwrap().len(),
        std::fs::read(&plain).unwrap().len() - 44
    );

    let forced = tmp("forced.dat");
    assert!(
        speak(&[
            "-o",
            forced.to_str().unwrap(),
            "--file-format",
            "WAVE",
            "Hi."
        ])
        .status
        .success()
    );
    assert_eq!(&std::fs::read(&forced).unwrap()[..4], b"RIFF");
}

#[test]
fn wave_to_stdout_has_unknown_length_sizes() {
    let o = speak(&["-o", "-", "Hello world."]);
    assert!(o.status.success());
    let b = &o.stdout;
    assert_eq!(&b[..4], b"RIFF");
    assert_eq!(&b[4..8], &[0xFF; 4]);
    assert_eq!(&b[40..44], &[0xFF; 4]);
    assert!(b.len() > 16000);
    assert!(o.stderr.is_empty() || !err(&o).contains("error"));
}

#[test]
fn invalid_formats_and_unsupported_flags_exit_2() {
    let out = tmp("x.wav");
    let o = out.to_str().unwrap();
    for args in [
        vec!["-o", o, "--data-format=BEI16", "x"],
        vec!["-o", o, "--data-format=aac", "x"],
        vec!["-o", o, "--data-format=LEI8", "x"],
        vec!["-o", o, "--file-format=m4af", "x"],
        vec!["-o", o, "--channels", "3", "x"],
        vec!["-n", "host:1", "x"],
        vec!["--bit-rate", "64000", "-o", o, "x"],
        vec!["--quality", "3", "-o", o, "x"],
        vec!["--data-format=LEI16", "x"],
    ] {
        let r = speak(&args);
        assert_eq!(r.status.code(), Some(2), "{args:?}: {}", err(&r));
        assert!(err(&r).starts_with("speak: "));
    }
    assert!(err(&speak(&["-n", "host:1", "x"])).contains("-n is not supported"));
    for q in ["--file-format=?", "--data-format=?"] {
        let r = speak(&[q]);
        assert!(r.status.success() && !r.stdout.is_empty());
    }
}

#[test]
fn embedded_commands() {
    let (a, b) = (tmp("plain.wav"), tmp("slnc.wav"));
    assert!(
        speak(&["-o", a.to_str().unwrap(), "One. Two."])
            .status
            .success()
    );
    assert!(
        speak(&["-o", b.to_str().unwrap(), "One.[[slnc 500]] Two."])
            .status
            .success()
    );
    let added = wav_seconds(&b) - wav_seconds(&a);
    assert!((added - 0.5).abs() <= 0.012, "{added}");

    let (slow, fast) = (tmp("slow.wav"), tmp("fast.wav"));
    let text = "The first part is slow. The second part is the one that changes.";
    assert!(
        speak(&["-o", slow.to_str().unwrap(), text])
            .status
            .success()
    );
    assert!(
        speak(&[
            "-o",
            fast.to_str().unwrap(),
            "The first part is slow. [[rate 350]] The second part is the one that changes."
        ])
        .status
        .success()
    );
    assert!(wav_seconds(&fast) < wav_seconds(&slow) * 0.85);

    let (p1, p2) = (tmp("p1.wav"), tmp("p2.wav"));
    assert!(
        speak(&[
            "-o",
            p1.to_str().unwrap(),
            "[[inpt PHON]] HH AX L OW1 [[inpt TEXT]]"
        ])
        .status
        .success()
    );
    assert!(
        speak(&["-o", p2.to_str().unwrap(), "--phonemes", "HH AX L OW1"])
            .status
            .success()
    );
    assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
}

#[test]
fn config_precedence_alias_and_user_lexicon() {
    let cfg = tmp("c.toml");
    std::fs::write(&cfg, "rate = 100\n[klatt]\nvoice = \"klatt-female\"\n[voices.bot]\nbackend = \"klatt\"\nvoice = \"klatt-robot\"\nrate = 300\nparams = { tl = 3 }\n").unwrap();
    let c = cfg.to_str().unwrap();
    let dur = |extra: &[&str], env: Option<(&str, &str)>| {
        let out = tmp(&format!(
            "cfg{}.wav",
            extra.join("").replace(['/', ':'], "_")
        ));
        let mut command = cmd();
        command
            .args(["--config", c, "-o", out.to_str().unwrap()])
            .args(extra)
            .arg("Testing the configuration file.");
        if let Some((k, v)) = env {
            command.env(k, v);
        }
        let o = command.output().unwrap();
        assert!(o.status.success(), "{}", err(&o));
        (wav_seconds(&out), err(&o))
    };
    // File: rate 100 is slower than the default 175.
    let (from_file, timing) = dur(&["--timing"], None);
    assert!(timing.contains("klatt-female"), "config voice: {timing}");
    let plain = tmp("plain2.wav");
    assert!(
        speak(&[
            "-o",
            plain.to_str().unwrap(),
            "Testing the configuration file."
        ])
        .status
        .success()
    );
    assert!(from_file > wav_seconds(&plain) * 1.2);
    // Flag beats file.
    let (from_flag, _) = dur(&["-r", "300"], None);
    assert!(from_flag < from_file * 0.6);
    // Environment beats file; flag beats environment.
    let (_, t) = dur(&["--timing"], Some(("SPEAK_VOICE", "klatt-whisper")));
    assert!(t.contains("klatt-whisper"), "{t}");
    let (_, t) = dur(
        &["--timing", "-v", "klatt-male"],
        Some(("SPEAK_VOICE", "klatt-whisper")),
    );
    assert!(t.contains("klatt-male"), "{t}");
    // SPEAK_CONFIG names the file too.
    let o = cmd()
        .env("SPEAK_CONFIG", c)
        .args(["--timing", "-o", tmp("envcfg.wav").to_str().unwrap(), "Hi."])
        .output()
        .unwrap();
    assert!(err(&o).contains("klatt-female"));
    // Alias: voice and rate from the alias.
    let (alias, t) = dur(&["-v", "bot", "--timing"], None);
    assert!(t.contains("klatt-robot") && alias < from_file * 0.6, "{t}");

    // Unknown key: names the key, exit 2. A missing explicit file is an error too.
    let bad = tmp("bad.toml");
    std::fs::write(&bad, "rate = 100\nnonsense = 1\n").unwrap();
    let o = speak(&["--config", bad.to_str().unwrap(), "--dump", "text", "x"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(
        err(&o).contains("nonsense") && err(&o).contains("line 2"),
        "{}",
        err(&o)
    );
    assert_eq!(
        speak(&["--config", "/nonexistent/c.toml", "--dump", "text", "x"])
            .status
            .code(),
        Some(2)
    );

    // User lexicon overrides the built-in entry.
    let before = String::from_utf8_lossy(&speak(&["--dump", "phonemes", "hello"]).stdout)
        .trim()
        .to_string();
    assert_eq!(before, "HH AX L OW1");
    std::fs::write(
        home().join(".config/speak/lexicon.txt"),
        "hello Y OW1  # override\n",
    )
    .unwrap();
    let o = speak(&["-V", "--dump", "phonemes", "hello"]);
    std::fs::remove_file(home().join(".config/speak/lexicon.txt")).unwrap();
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "Y OW1");
    assert!(err(&o).contains("from user"));
}

#[cfg(unix)]
#[test]
fn sigint_exits_130_and_leaves_a_valid_file() {
    let long = "This is sentence number one. ".repeat(20000);
    let input = tmp("long.txt");
    std::fs::write(&input, long).unwrap();
    for name in ["int.wav", "int.aiff"] {
        let out = tmp(name);
        let mut child = cmd()
            .args(["-f", input.to_str().unwrap(), "-o", out.to_str().unwrap()])
            .spawn()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            child.try_wait().unwrap().is_none(),
            "finished before the signal: raise the repeat count"
        );
        assert!(
            Command::new("kill")
                .args(["-INT", &child.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        let status = child.wait().unwrap();
        assert_eq!(status.code(), Some(130));
        let b = std::fs::read(&out).unwrap();
        if name.ends_with(".wav") {
            let r = hound::WavReader::open(&out).unwrap();
            assert!(r.duration() > 16000);
            assert_eq!(r.duration() as usize * 2, b.len() - 44);
        } else {
            let form = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
            assert_eq!(form + 8, b.len());
            assert!(u32::from_be_bytes([b[22], b[23], b[24], b[25]]) > 16000);
        }
    }
}
