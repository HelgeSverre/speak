//! `speak-lab`: the workshop (spec §3.6).

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};

use crate::audio::FileSink;
use crate::cli::analyze;
use crate::cli::pipeline::data_dirs;
use crate::cli::wer::wer_lines;
use crate::diphone::{VoiceFile, autolabel, backend, build, inventory, prompts};
use crate::error::{Error, Result};
use crate::text::lexicon::compiled;
use crate::text::lts::{CartLts, RuleLts, cart, evaluate, train};
use crate::traits::LetterToSound;
use crate::traits::{AudioSink, PcmWriter};

#[derive(Parser, Debug)]
#[command(
    name = "speak-lab",
    about = "Workshop for speak: build voices and lexicons, analyse audio",
    version
)]
pub struct LabArgs {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Word error rate of a hypothesis against a reference, line by line, after our normalization
    Wer {
        reference: PathBuf,
        hypothesis: PathBuf,
    },
    /// Pronunciation lexicon tools
    Lexicon {
        #[command(subcommand)]
        command: LexiconCommand,
    },
    /// Letter-to-sound tools
    Lts {
        #[command(subcommand)]
        command: LtsCommand,
    },
    /// TSV tracks of a WAV file, one row per 5 ms
    Analyze {
        file: PathBuf,
        /// F0 and voicing
        #[arg(long)]
        f0: bool,
        /// F1–F4 from LPC
        #[arg(long)]
        formants: bool,
        /// Reflection coefficients and gain
        #[arg(long)]
        lpc: bool,
    },
    /// Diphone voice tools
    Diphones {
        #[command(subcommand)]
        command: DiphonesCommand,
    },
    /// Greyscale spectrogram as a PGM image
    Spectrogram { file: PathBuf, out: PathBuf },
}

#[derive(Subcommand, Debug)]
pub enum DiphonesCommand {
    /// Build a diphone voice: --bootstrap analyses our own formant synthesizer
    Build {
        /// Analyse Klatt renderings of every diphone
        #[arg(long)]
        bootstrap: bool,
        /// Voice name (default: boot)
        #[arg(long)]
        name: Option<String>,
        /// Directory with recordings made by `diphones record`
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Inventory, coverage and sizes of a voice
    Info { voice: String },
    /// Write the prompt list and one guide WAV per prompt
    Prompts {
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
        /// Only the first N prompts (the most frequent diphones come first)
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Record the prompts interactively: guide plays, you repeat it in a monotone
    Record {
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,
        /// Start at this prompt id (default: the first one without a recording)
        #[arg(long, value_name = "ID")]
        from: Option<String>,
    },
    /// Test helper: write "recordings" rendered by another voice with perturbed timing
    #[command(hide = true)]
    Simulate {
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,
        #[arg(long, default_value = "klatt-female")]
        voice: String,
        #[arg(long, default_value_t = 0.9)]
        rate: f32,
        #[arg(long, default_value_t = 0.1)]
        jitter: f32,
        #[arg(long, default_value_t = 1)]
        seed: u32,
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[derive(Subcommand, Debug)]
pub enum LexiconCommand {
    /// Compile CMUdict into our lookup format (default output: <data>/lexicon/en-us.lex)
    Build {
        dict: PathBuf,
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
pub enum LtsCommand {
    /// Train the letter-to-sound decision tree from CMUdict
    Train {
        dict: PathBuf,
        /// Output file (default: data/lts-en.cart)
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
        /// Smallest node that is still split; raised automatically until the file fits
        #[arg(long, default_value_t = 3)]
        stop: usize,
        /// Size limit in bytes
        #[arg(long, default_value_t = 2_000_000)]
        max_bytes: usize,
    },
    /// Held-out accuracy on CMUdict, and phoneme error rate on a word list
    Eval {
        dict: PathBuf,
        #[arg(long, default_value = "cart")]
        method: String,
        /// `word PH PH ...` list for the phoneme error rate
        #[arg(long, value_name = "FILE")]
        words: Option<PathBuf>,
    },
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })
}

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, bytes).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })
}

fn percent(a: usize, b: usize) -> f64 {
    100.0 * a as f64 / b.max(1) as f64
}

fn lts_train(dict: &Path, out: Option<PathBuf>, stop: usize, max_bytes: usize) -> Result<()> {
    let t = Instant::now();
    let entries = train::parse_dict(&read(dict)?);
    if entries.len() < 100 {
        return Err(Error::invalid(
            dict.display().to_string(),
            "not a pronunciation dictionary (fewer than 100 usable entries)",
        ));
    }
    let res = train::align_all(&entries, 4);
    println!(
        "entries {}  aligned {}  alignment failures {} ({:.2} %)",
        entries.len(),
        res.aligned.len(),
        res.failed,
        percent(res.failed, entries.len())
    );
    // Every 10th word is held out before training.
    let (train_set, held) = train::split(&res.aligned);
    let mut stop = stop.max(1);
    let (cart, bytes) = loop {
        let cart = train::train(&train_set, stop);
        let bytes = cart.to_bytes();
        println!(
            "stop {stop}: nodes {}  labels {}  size {} bytes",
            cart.node_count(),
            cart.labels.len(),
            bytes.len()
        );
        if bytes.len() <= max_bytes {
            break (cart, bytes);
        }
        stop += if stop < 10 { 1 } else { stop / 4 };
    };
    let s = train::score(&cart, &held);
    println!(
        "held out: words {:.1} %  (ignoring stress {:.1} %)  letters {:.1} %  on {} words",
        percent(s.words_correct, s.words),
        percent(s.words_correct_no_stress, s.words),
        percent(s.letters_correct, s.letters),
        s.words
    );
    let out = out.unwrap_or_else(|| PathBuf::from("data/lts-en.cart"));
    write(&out, &bytes)?;
    println!(
        "wrote {} ({} bytes) in {:.1} s",
        out.display(),
        bytes.len(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn lts_eval(dict: &Path, method: &str, words: Option<PathBuf>) -> Result<()> {
    let entries = train::parse_dict(&read(dict)?);
    let res = train::align_all(&entries, 4);
    let (_, held) = train::split(&res.aligned);
    let lts: Box<dyn LetterToSound> = match method {
        "cart" => Box::new(CartLts::embedded().ok_or_else(|| {
            Error::usage("this build has no tree; run speak-lab lts train and rebuild")
        })?),
        "rules" => Box::new(RuleLts::new()),
        other => {
            return Err(Error::usage(format!(
                "--method {other}: expected rules or cart"
            )));
        }
    };
    // Compare predicted phones with the dictionary on held-out words (stress ignored, reduced vowels mapped).
    let (mut words_ok, mut phones, mut errors) = (0usize, 0usize, 0usize);
    for a in &held {
        let gold: Vec<crate::phone::Phone> = a
            .labels
            .iter()
            .flat_map(|l| l.split('-'))
            .filter(|t| *t != "_")
            .filter_map(|t| crate::phone::parse_phone_token(t).ok())
            .map(|(p, s)| cart::reduce(p, s))
            .collect();
        let got: Vec<crate::phone::Phone> = lts
            .pronounce(&String::from_utf8_lossy(&a.word))
            .iter()
            .map(|s| s.phone)
            .collect();
        let d = crate::text::lts::edit_distance(&gold, &got);
        phones += gold.len();
        errors += d;
        words_ok += usize::from(d == 0);
    }
    println!(
        "{method}: held-out CMUdict ({} words): words correct {:.1} %  phoneme error rate {:.1} %",
        held.len(),
        percent(words_ok, held.len()),
        percent(errors, phones)
    );
    if method == "cart" {
        // Label-level score, the measure the spec gates on.
        let cart = cart::Cart::from_bytes(cart::EMBEDDED)
            .map_err(|e| Error::invalid("data/lts-en.cart", e))?;
        let s = train::score(&cart, &held);
        println!(
            "cart: words {:.1} %  (ignoring stress {:.1} %)  letters {:.1} %",
            percent(s.words_correct, s.words),
            percent(s.words_correct_no_stress, s.words),
            percent(s.letters_correct, s.letters)
        );
    }
    if let Some(list) = words {
        let ev = evaluate(lts.as_ref(), &read(&list)?);
        println!(
            "{method}: {}: PER {:.1} %  words correct {}/{}",
            list.display(),
            ev.per() * 100.0,
            ev.words_correct,
            ev.words
        );
    }
    Ok(())
}

fn diphones(command: DiphonesCommand) -> Result<()> {
    let data = data_dirs();
    match command {
        DiphonesCommand::Build {
            bootstrap: true,
            name,
            ..
        } => {
            let name = name.unwrap_or_else(|| backend::DEFAULT_VOICE.to_string());
            let t = Instant::now();
            let voice = build::bootstrap(&name, |done, total| {
                if done % 200 == 0 || done == total {
                    eprint!("\rspeak-lab: analysed {done}/{total} diphones");
                }
            })?;
            eprintln!();
            let path = backend::voice_path(&data, &name);
            voice.write(&path)?;
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!(
                "wrote {} ({} diphones, {} frames, {:.1} MB) in {:.1} s",
                path.display(),
                voice.diphones.len(),
                voice.frame_count(),
                size as f64 / 1e6,
                t.elapsed().as_secs_f64()
            );
            Ok(())
        }
        DiphonesCommand::Build {
            dir: Some(dir),
            name,
            ..
        } => {
            let name =
                name.ok_or_else(|| Error::usage("diphones build --dir DIR needs --name NAME"))?;
            let t = Instant::now();
            let (voice, report) = autolabel::build_from_dir(&dir, &name, |done, total| {
                if done % 100 == 0 || done == total {
                    eprint!("\rspeak-lab: labelled {done}/{total} prompts");
                }
            })?;
            eprintln!();
            let path = backend::voice_path(&data, &name);
            voice.write(&path)?;
            println!(
                "prompts {}  recorded {}  built {}  rejected {} ({:.1} %)",
                report.prompts,
                report.recorded,
                report.built,
                report.rejected.len(),
                percent(report.rejected.len(), report.recorded)
            );
            if let (Some(median), Some(p90)) = (report.percentile(0.5), report.percentile(0.9)) {
                println!(
                    "boundary error vs truth: median {median:.1} ms, p90 {p90:.1} ms over {} prompts",
                    report.boundary_errors.len()
                );
            }
            println!("speaking pitch {:.0} Hz", report.median_f0);
            if !report.rejected.is_empty() {
                println!(
                    "re-record: {}",
                    report
                        .rejected
                        .iter()
                        .take(40)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            println!(
                "wrote {} in {:.1} s",
                path.display(),
                t.elapsed().as_secs_f64()
            );
            Ok(())
        }
        DiphonesCommand::Build { .. } => Err(Error::usage(
            "diphones build: give --bootstrap, or --dir DIR --name NAME",
        )),
        DiphonesCommand::Prompts { out, limit } => {
            let mut list = prompts::all();
            if let Some(n) = limit {
                list.truncate(n);
            }
            std::fs::create_dir_all(out.join("guide"))?;
            write(&out.join("prompts.tsv"), prompts::tsv(&list).as_bytes())?;
            for p in &list {
                let pcm = autolabel::render_guide(p)?;
                let mut sink = FileSink::create(&out.join("guide").join(format!("{}.wav", p.id)))?;
                sink.begin(crate::diphone::SR)?;
                sink.write(&pcm)?;
                sink.finish()?;
            }
            let seconds: f32 = list
                .iter()
                .map(|p| prompts::guide(p).dur_ms() / 1000.0 + 0.5)
                .sum();
            println!(
                "wrote {} prompts and guides to {} (about {:.0} minutes of recording)",
                list.len(),
                out.display(),
                2.0 * seconds / 60.0
            );
            Ok(())
        }
        DiphonesCommand::Simulate {
            dir,
            voice,
            rate,
            jitter,
            seed,
            limit,
        } => {
            let n = autolabel::simulate(&dir, &voice, rate, jitter, seed, limit)?;
            println!(
                "wrote {n} simulated recordings and truth.tsv to {}",
                dir.display()
            );
            Ok(())
        }
        DiphonesCommand::Record { dir, from } => record(&dir, from.as_deref()),
        DiphonesCommand::Info { voice } => {
            let path = if voice.contains('/') || voice.ends_with(".sdv") {
                PathBuf::from(&voice)
            } else {
                backend::voice_path(&data, &voice)
            };
            if !path.is_file() {
                return Err(Error::VoiceMissing {
                    backend: "diphone".into(),
                    name: voice,
                    hint: backend::BOOTSTRAP_HINT.into(),
                });
            }
            let v = VoiceFile::read(&path)?;
            let inventory = inventory::all();
            let missing: Vec<String> = inventory
                .iter()
                .filter(|(a, b)| v.get(*a, *b).is_none())
                .map(|(a, b)| inventory::name(*a, *b))
                .collect();
            println!("file         {}", path.display());
            println!("description  {}", v.description);
            println!("sample rate  {} Hz", crate::diphone::SR);
            println!("LPC order    {}", crate::dsp::lpc::ORDER);
            println!(
                "diphones     {} of {} in the inventory",
                v.diphones.len(),
                inventory.len()
            );
            println!(
                "frames       {} ({:.1} s of speech)",
                v.frame_count(),
                v.frame_count() as f32 * 0.005
            );
            println!(
                "residuals    {}",
                if v.has_residual() { "yes" } else { "no" }
            );
            if !missing.is_empty() {
                println!(
                    "missing      {} ({}{})",
                    missing.len(),
                    missing
                        .iter()
                        .take(12)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" "),
                    if missing.len() > 12 { " …" } else { "" }
                );
            }
            Ok(())
        }
    }
}

/// Interactive recording: play the guide, record, accept / redo / quit.
fn record(dir: &Path, from: Option<&str>) -> Result<()> {
    use crate::audio::{Playback, capture};
    use std::io::BufRead;
    let tsv = read(&dir.join("prompts.tsv"))?;
    let list = prompts::parse_tsv(&tsv);
    std::fs::create_dir_all(dir.join("rec"))?;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Fail early, before the first prompt, when there is no microphone.
    capture::record(0.05, crate::diphone::SR, &stop)?;
    let start = match from {
        Some(id) => list
            .iter()
            .position(|p| p.id == id)
            .ok_or_else(|| Error::usage(format!("--from {id}: no such prompt")))?,
        None => list
            .iter()
            .position(|p| !dir.join("rec").join(format!("{}.wav", p.id)).is_file())
            .unwrap_or(list.len()),
    };
    println!("Repeat each prompt after the guide, in a steady monotone, with the guide's timing.");
    println!("Enter = keep, r = record again, q = stop (progress is saved).");
    let stdin = std::io::stdin();
    let mut i = start;
    while i < list.len() {
        let p = &list[i];
        println!(
            "\n{}  ({} of {})  target {}\n    {}",
            p.id,
            i + 1,
            list.len(),
            crate::diphone::inventory::name(p.a, p.b),
            p.text()
        );
        let guide = autolabel::render_guide(p)?;
        let mut play = Playback::open(None, crate::diphone::SR, stop.clone())?;
        play.begin(crate::diphone::SR)?;
        play.write(&guide)?;
        play.finish()?;
        println!("    recording…");
        let pcm = capture::record(
            guide.len() as f32 / crate::diphone::SR as f32 + 0.5,
            crate::diphone::SR,
            &stop,
        )?;
        if capture::is_all_zero(&pcm) {
            println!(
                "    the recording is completely silent: the terminal probably lacks microphone permission"
            );
            println!("    (System Settings > Privacy & Security > Microphone)");
        }
        print!("    keep? [Enter / r / q] ");
        std::io::Write::flush(&mut std::io::stdout())?;
        let mut answer = String::new();
        if stdin.lock().read_line(&mut answer)? == 0 {
            break;
        }
        match answer.trim() {
            "q" => break,
            "r" => continue,
            _ => {
                let mut sink = FileSink::create(&dir.join("rec").join(format!("{}.wav", p.id)))?;
                sink.begin(crate::diphone::SR)?;
                sink.write(&pcm)?;
                sink.finish()?;
                i += 1;
            }
        }
    }
    println!(
        "\n{} of {} prompts recorded. Build with: speak-lab diphones build --dir {} --name NAME",
        i,
        list.len(),
        dir.display()
    );
    Ok(())
}

pub fn run(args: LabArgs) -> Result<()> {
    match args.command {
        Command::Wer {
            reference,
            hypothesis,
        } => {
            let w = wer_lines(&read(&reference)?, &read(&hypothesis)?);
            println!(
                "words {}  errors {}  WER {:.1} %",
                w.reference_words,
                w.errors,
                w.rate() * 100.0
            );
            for (r, h) in &w.confusions {
                println!(
                    "  {:<16} -> {}",
                    if r.is_empty() { "(inserted)" } else { r },
                    if h.is_empty() { "(deleted)" } else { h }
                );
            }
            Ok(())
        }
        Command::Lexicon {
            command: LexiconCommand::Build { dict, out },
        } => {
            let (text, n) = compiled::build(&read(&dict)?);
            let out =
                out.unwrap_or_else(|| data_dirs().data.join("lexicon").join(compiled::FILE_NAME));
            write(&out, text.as_bytes())?;
            println!(
                "wrote {} ({n} entries, {} bytes)",
                out.display(),
                text.len()
            );
            Ok(())
        }
        Command::Analyze {
            file,
            f0,
            formants,
            lpc,
        } => {
            let (pcm, rate) = analyze::read_wav(&file)?;
            // With no selection, print everything.
            let all = !(f0 || formants || lpc);
            print!(
                "{}",
                analyze::analyze(&pcm, rate, f0 || all, formants || all, lpc || all)
            );
            Ok(())
        }
        Command::Diphones { command } => diphones(command),
        Command::Spectrogram { file, out } => {
            let (pcm, rate) = analyze::read_wav(&file)?;
            write(&out, &analyze::spectrogram_pgm(&pcm, rate))
        }
        Command::Lts {
            command:
                LtsCommand::Train {
                    dict,
                    out,
                    stop,
                    max_bytes,
                },
        } => lts_train(&dict, out, stop, max_bytes),
        Command::Lts {
            command:
                LtsCommand::Eval {
                    dict,
                    method,
                    words,
                },
        } => lts_eval(&dict, &method, words),
    }
}
