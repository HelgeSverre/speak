//! The run loop (spec §4.3): frontend thread → synth thread → sink.

use std::io::{BufRead, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::{Duration, Instant};

use crate::audio::format::{self, DataFormat, FileFormat};
use crate::audio::{FileSink, NullSink, Playback, playback};
use crate::chunk::{Chunk, Controls, Terminator, Unit, WordSource};
use crate::cli::args::Args;
use crate::cli::config::{self, Config};
use crate::cli::{backends, lights, listing};
use crate::error::{Error, Result};
use crate::pho::{parse_pho, print_pho};
use crate::phone::{parse_arpabet, print_arpabet};
use crate::text::sing::{self, SingProsody};
use crate::text::{Frontend, RuleProsody};
use crate::timing::TimingReport;
use crate::traits::{
    AudioSink, Backend, DataDirs, Flow, PcmWriter, Prosody, SynthOptions, VoiceSpec,
};

const DUMP_STAGES: &[&str] = &["text", "chunks", "phonemes", "pho", "frames", "units"];
/// `-P` keys that belong to the frontend rather than a backend.
const FRONTEND_KEYS: &[&str] = &["lts"];

pub fn data_dirs() -> DataDirs {
    let data = std::env::var_os("SPEAK_DATA_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default();
            if cfg!(target_os = "macos") {
                home.join("Library/Application Support/speak")
            } else {
                std::env::var_os("XDG_DATA_HOME")
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| home.join(".local/share"))
                    .join("speak")
            }
        });
    DataDirs { data }
}

/// `~/.config/speak/lexicon.txt` (spec §7.4 row 1).
pub fn user_lexicon_path() -> PathBuf {
    config::config_dir().join("lexicon.txt")
}

/// What `-v`, `-b`, the environment and the config file resolve to (spec §3.3).
struct Resolved {
    backend: usize,
    voice: String,
    /// Rate and params contributed by a config alias.
    alias_rate: Option<f32>,
    alias_params: Vec<(String, String)>,
}

fn resolve(
    all: &[Box<dyn Backend>],
    args: &Args,
    cfg: &Config,
    data: &DataDirs,
) -> Result<Resolved> {
    let find_backend = |id: &str| {
        all.iter().position(|b| b.id() == id).ok_or_else(|| {
            let ids: Vec<_> = all.iter().map(|b| b.id()).collect();
            Error::usage(format!(
                "unknown backend '{id}'. Available: {}",
                ids.join(", ")
            ))
        })
    };
    let default_voice = |bi: usize| {
        let from_cfg = match all[bi].id() {
            "klatt" => cfg.klatt.voice.clone(),
            "diphone" => cfg.diphone.voice.clone(),
            _ => None,
        };
        from_cfg.unwrap_or_else(|| all[bi].default_voice().to_string())
    };
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let plain = |backend: usize, voice: String| Resolved {
        backend,
        voice,
        alias_rate: None,
        alias_params: Vec::new(),
    };
    let voice = args.voice.clone().or_else(|| env("SPEAK_VOICE"));
    let Some(v) = voice else {
        let id = args
            .backend
            .clone()
            .or_else(|| env("SPEAK_BACKEND"))
            .or_else(|| cfg.backend.clone())
            .unwrap_or_else(|| "klatt".into());
        let bi = find_backend(&id)?;
        return Ok(plain(bi, default_voice(bi)));
    };
    if let Some((b, name)) = v.split_once(':') {
        return Ok(plain(find_backend(b)?, name.to_string()));
    }
    if let Some(alias) = cfg.voices.get(&v) {
        let id = alias
            .backend
            .clone()
            .or_else(|| args.backend.clone())
            .or_else(|| cfg.backend.clone())
            .unwrap_or_else(|| "klatt".into());
        let bi = find_backend(&id)?;
        return Ok(Resolved {
            backend: bi,
            voice: alias.voice.clone().unwrap_or_else(|| default_voice(bi)),
            alias_rate: alias.rate,
            alias_params: alias.params(),
        });
    }
    let hits: Vec<usize> = (0..all.len())
        .filter(|i| args.backend.as_deref().is_none_or(|id| id == all[*i].id()))
        .filter(|i| all[*i].voices(data).iter().any(|x| x.name == v))
        .collect();
    match hits.as_slice() {
        [one] => Ok(plain(*one, v)),
        [] => Err(Error::usage(format!(
            "unknown voice '{v}'. Run speak -v '?'"
        ))),
        many => {
            let names: Vec<String> = many
                .iter()
                .map(|i| format!("{}:{v}", all[*i].id()))
                .collect();
            Err(Error::usage(format!(
                "voice '{v}' is ambiguous: {}",
                names.join(", ")
            )))
        }
    }
}

fn parse_params(
    all: &[Box<dyn Backend>],
    raw: &[String],
    alias: &[(String, String)],
    cfg: &Config,
) -> Result<SynthOptions> {
    let mut opts = SynthOptions::default();
    let known =
        |k: &str| FRONTEND_KEYS.contains(&k) || all.iter().any(|b| b.param_keys().contains(&k));
    // Lowest precedence first: config file, alias, then -P.
    if let Some(lts) = &cfg.lts {
        opts.params.insert("lts".into(), lts.clone());
    }
    if let Some(e) = &cfg.diphone.excitation {
        opts.params.insert("excitation".into(), e.clone());
    }
    for (k, v) in alias {
        if !known(k) {
            return Err(Error::Config(format!(
                "config: unknown voice parameter '{k}'"
            )));
        }
        opts.params.insert(k.clone(), v.clone());
    }
    for p in raw {
        let (k, v) = p
            .split_once('=')
            .ok_or_else(|| Error::usage(format!("-P {p}: expected key=value")))?;
        if !known(k) {
            return Err(Error::usage(format!("-P {k}: unknown key")));
        }
        opts.params.insert(k.to_string(), v.to_string());
    }
    Ok(opts)
}

pub fn parse_pitch(s: &str, controls: &mut Controls) -> Result<()> {
    let bad = || {
        Error::usage(format!(
            "--pitch {s}: expected Hz (110) or semitones (+2st)"
        ))
    };
    if let Some(st) = s.strip_suffix("st") {
        let v: f32 = st.parse().map_err(|_| bad())?;
        if !(-24.0..=24.0).contains(&v) {
            return Err(bad());
        }
        controls.pitch_shift_st = v;
    } else {
        let v: f32 = s.parse().map_err(|_| bad())?;
        if !(40.0..=500.0).contains(&v) {
            return Err(bad());
        }
        controls.pitch_base_hz = Some(v);
    }
    Ok(())
}

fn controls(args: &Args, cfg: &Config, alias_rate: Option<f32>) -> Result<Controls> {
    let mut c = Controls::default();
    if let Some(r) = args.rate.or(alias_rate).or(cfg.rate) {
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::usage(format!("-r {r}: expected words per minute")));
        }
        c.rate = r.clamp(80.0, 500.0) / crate::DEFAULT_WPM;
    }
    if let Some(p) = args.pitch.as_deref() {
        parse_pitch(p, &mut c)?;
    }
    if let Some(v) = args.volume.or(cfg.volume) {
        if !(0.0..=2.0).contains(&v) {
            return Err(Error::usage(format!("--volume {v}: expected 0.0–2.0")));
        }
        c.volume = v;
    }
    Ok(c)
}

/// Where the text comes from.
enum Input {
    Text(String),
    /// TTY stdin: speak each line as it is entered.
    Lines,
}

fn read_input(args: &Args) -> Result<Input> {
    if args.input_file.is_some() && !args.text.is_empty() {
        return Err(Error::usage("give either -f file or text, not both"));
    }
    if let Some(path) = &args.input_file {
        if path.as_os_str() == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            return Ok(Input::Text(s));
        }
        return std::fs::read_to_string(path).map(Input::Text).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", path.display()),
            ))
        });
    }
    if !args.text.is_empty() {
        return Ok(Input::Text(args.text.join(" ")));
    }
    if std::io::stdin().is_terminal() {
        return Ok(Input::Lines);
    }
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(Input::Text(s))
}

/// Chunks from ARPAbet text: one per sentence mark.
pub fn arpabet_chunks(text: &str, controls: Controls) -> Result<Vec<Chunk>> {
    let units = parse_arpabet(text)?;
    let mut chunks = Vec::new();
    let mut cur: Vec<Unit> = Vec::new();
    let mut flush = |cur: &mut Vec<Unit>, terminator: Terminator| {
        while matches!(cur.first(), Some(Unit::WordBreak)) {
            cur.remove(0);
        }
        if cur
            .iter()
            .any(|u| matches!(u, Unit::Seg(_) | Unit::Silence { .. }))
        {
            chunks.push(Chunk {
                index: chunks.len(),
                text: print_arpabet(cur),
                units: std::mem::take(cur),
                terminator,
                pause_after_ms: terminator.pause_ms() / controls.rate,
                controls,
                ..Chunk::default()
            });
        }
        cur.clear();
    };
    for u in units {
        let mark = if let Unit::Punct(c) = u {
            Some(c)
        } else {
            None
        };
        cur.push(u);
        if let Some(c @ ('.' | '?' | '!')) = mark {
            flush(&mut cur, Terminator::from_char(c));
        }
    }
    let last = match cur.last() {
        Some(Unit::Punct(c)) => Terminator::from_char(*c),
        _ => Terminator::None,
    };
    flush(&mut cur, last);
    Ok(chunks)
}

fn singer(args: &mut Args) -> Result<Option<SingProsody>> {
    let tune =
        match args.sing.as_deref() {
            Some(name) => Some(sing::tune(name).ok_or_else(|| {
                Error::usage(format!("unknown tune '{name}'. Run speak --sing='?'"))
            })?),
            None => None,
        };
    let (melody, default_tempo) = match (&args.melody, tune) {
        (Some(m), _) => {
            // `@file` reads the notes from a file; lines starting with # are comments.
            let notes = match m.strip_prefix('@') {
                Some(path) => std::fs::read_to_string(path)
                    .map_err(|e| Error::Io(std::io::Error::new(e.kind(), format!("{path}: {e}"))))?
                    .lines()
                    .filter(|l| !l.trim_start().starts_with('#'))
                    .collect::<Vec<_>>()
                    .join(" "),
                None => m.clone(),
            };
            (
                sing::parse_melody(&notes)?,
                tune.map(|t| t.tempo).unwrap_or(120.0),
            )
        }
        (None, Some(t)) => (sing::parse_melody(t.melody)?, t.tempo),
        (None, None) => return Ok(None),
    };
    // A built-in tune without text sings its own words.
    if let Some(t) = tune
        && args.text.is_empty()
        && args.input_file.is_none()
        && args.melody.is_none()
    {
        args.text = vec![t.lyrics.to_string()];
    }
    Ok(Some(SingProsody::new(
        melody,
        args.tempo.unwrap_or(default_tempo),
        args.key.unwrap_or(0.0),
    )))
}

/// Measures time spent inside the sink so RTF counts synthesis only.
struct TimedWriter<'a> {
    inner: &'a mut dyn PcmWriter,
    blocked: Duration,
}

impl PcmWriter for TimedWriter<'_> {
    fn write(&mut self, samples: &[f32]) -> Result<Flow> {
        let t = Instant::now();
        let r = self.inner.write(samples);
        self.blocked += t.elapsed();
        r
    }
}

/// Listings and flags that end the run before any synthesis. `Ok(true)` = done.
fn early_exit(args: &Args, all: &[Box<dyn Backend>], data: &DataDirs) -> Result<bool> {
    for (flag, given) in [
        ("-n", args.network_send.is_some()),
        ("--bit-rate", args.bit_rate.is_some()),
        ("--quality", args.quality.is_some()),
    ] {
        if given {
            return Err(Error::usage(format!("{flag} is not supported")));
        }
    }
    if args.list_backends {
        print!("{}", listing::backend_table(all, data));
        return Ok(true);
    }
    if args.list_voices || args.voice.as_deref() == Some("?") {
        print!("{}", listing::voices(all, args.backend.as_deref(), data));
        return Ok(true);
    }
    if args.file_format.as_deref() == Some("?") {
        print!("{}", format::file_format_listing());
        return Ok(true);
    }
    if args.data_format.as_deref() == Some("?") {
        print!("{}", format::data_format_listing());
        return Ok(true);
    }
    if args.audio_device.as_deref() == Some("?") {
        for d in playback::list_devices()? {
            println!(
                "{:>3}  {}{}",
                d.index,
                d.name,
                if d.is_default { "  (default)" } else { "" }
            );
        }
        return Ok(true);
    }
    if args.sing.as_deref() == Some("?") {
        for t in sing::TUNES {
            println!("{:<10} {}", t.name, t.title);
        }
        return Ok(true);
    }
    Ok(false)
}

/// Container and sample format for `-o` (spec §3.1).
/// Path (`None` = stdout), container, sample format, channels.
type Output = (Option<PathBuf>, FileFormat, DataFormat, u16);

fn output_format(args: &Args) -> Result<Option<Output>> {
    let channels = args.channels.unwrap_or(1);
    if !(1..=2).contains(&channels) {
        return Err(Error::usage(format!(
            "--channels {channels}: expected 1 or 2"
        )));
    }
    let Some(path) = &args.output else {
        if args.file_format.is_some() || args.data_format.is_some() {
            return Err(Error::usage("--file-format and --data-format need -o"));
        }
        return Ok(None);
    };
    let stdout = path.as_os_str() == "-";
    let file_format = match args.file_format.as_deref() {
        Some(f) => FileFormat::parse(f)?,
        None if stdout => FileFormat::Wave,
        None => FileFormat::from_path(path),
    };
    let data = match args.data_format.as_deref() {
        Some(d) => DataFormat::parse(d, file_format.default_data())?,
        None => file_format.default_data(),
    };
    file_format.check(&data)?;
    Ok(Some((
        if stdout { None } else { Some(path.clone()) },
        file_format,
        data,
        channels,
    )))
}

pub fn run(args: Args, start: Instant) -> Result<()> {
    let all = backends();
    let data = data_dirs();
    if early_exit(&args, &all, &data)? {
        return Ok(());
    }
    if let Some(stage) = args.dump.as_deref()
        && !DUMP_STAGES.contains(&stage)
    {
        return Err(Error::usage(format!(
            "--dump {stage}: expected one of {}",
            DUMP_STAGES.join(", ")
        )));
    }
    let timing_fmt = match args.timing.as_deref() {
        None => None,
        Some(f @ ("text" | "json")) => Some(f.to_string()),
        Some(other) => {
            return Err(Error::usage(format!(
                "--timing={other}: expected text or json"
            )));
        }
    };
    let cfg = config::load(args.config.as_deref())?;
    if args.resynth.is_some() {
        return run_resynth(&args, &cfg, start);
    }
    if args.vocoder.is_some() {
        return Err(Error::usage("--vocoder needs --resynth"));
    }
    if let Some(n) = args.bench {
        return run_bench(&args, &cfg, n.clamp(1, 1000));
    }
    // Singing: a tune replaces the speech prosody.
    let mut args = args;
    let singer = singer(&mut args)?;
    let resolved = resolve(&all, &args, &cfg, &data)?;
    let backend = &all[resolved.backend];
    let voice_name = resolved.voice.clone();
    let opts = parse_params(&all, &args.params, &resolved.alias_params, &cfg)?;
    let controls = controls(&args, &cfg, resolved.alias_rate)?;
    let output = output_format(&args)?;
    let input = read_input(&args)?;

    // SIGINT: stop audio, finalize the output file, exit 130. A second one exits at once.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        let _ = ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
        });
    }
    let startup_ms = start.elapsed().as_secs_f64() * 1e3;

    // Three startup tasks run concurrently: frontend, voice open, audio open.
    let dumping = args.dump.is_some();
    let sink_thread = {
        let (rate, stop) = (backend.sample_rate(), stop.clone());
        let device = args
            .audio_device
            .clone()
            .or_else(|| cfg.audio.device.clone());
        std::thread::spawn(move || -> Result<Box<dyn AudioSink>> {
            Ok(match output {
                _ if dumping => Box::new(NullSink::new()),
                Some((path, file_format, data_format, channels)) => Box::new(
                    FileSink::with_format(path.as_deref(), file_format, data_format, channels)?
                        .with_stop(stop),
                ),
                None => Box::new(Playback::open(device.as_deref(), rate, stop)?),
            })
        })
    };

    // Chunks arrive with the total count when it is known (not in line mode).
    let (tx, rx) = sync_channel::<(Chunk, Option<usize>)>(8);
    let frontend = {
        let phonemes = args.phonemes.clone();
        let lts = opts.params.get("lts").cloned();
        let stop = stop.clone();
        std::thread::spawn(move || frontend_thread(input, phonemes, lts, controls, tx, stop))
    };

    let t_voice = Instant::now();
    let mut synth = backend.open(
        &VoiceSpec {
            name: voice_name.clone(),
        },
        &opts,
        &data,
    )?;
    let voice_open_ms = t_voice.elapsed().as_secs_f64() * 1e3;
    let spec = synth.spec();
    let mut sink = sink_thread
        .join()
        .map_err(|_| Error::audio("audio thread panicked"))??;
    sink.begin(spec.sample_rate)?;
    // A connected SidePulse strip follows the voice while audio is played.
    let lights_on = !args.no_lights && cfg.lights.unwrap_or(true);
    let lights = match (sink.clock(), lights_on, lights::detect()) {
        (Some(clock), true, Some(file)) => Some(lights::Lights::start(file, clock, stop.clone())),
        _ => None,
    };
    let highlighter = args
        .interactive
        .then(|| crate::cli::interactive::Highlighter::start(sink.clock(), stop.clone()));

    let prosody: Box<dyn Prosody> = match singer {
        Some(s) => Box::new(s),
        None => Box::new(RuleProsody),
    };
    let mut n_chunks = 0usize;
    let mut hits = [0usize; 6];
    let mut samples_sent = 0u64;
    let mut synth_time = Duration::ZERO;
    let mut buffered: Vec<f32> = Vec::new();
    let stdout = std::io::stdout();
    let mut printed_header = false;
    for (mut chunk, total) in rx {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        n_chunks += 1;
        for w in &chunk.words {
            if let Some(i) = WordSource::ALL.iter().position(|s| *s == w.source) {
                hits[i] += 1;
            }
            if args.verbose {
                eprintln!("speak: word '{}' from {}", w.text, w.source.name());
            }
        }
        let t = Instant::now();
        prosody.apply(&mut chunk, &spec.prosody);
        if let Some(stage) = args.dump.as_deref() {
            let mut out = stdout.lock();
            match stage {
                "phonemes" => writeln!(out, "{}", print_arpabet(&chunk.units))?,
                "pho" => {
                    let mut s = String::new();
                    print_pho(&chunk, &mut s);
                    out.write_all(s.as_bytes())?;
                }
                "text" => writeln!(out, "{}", chunk.text)?,
                "chunks" => writeln!(
                    out,
                    "{}\t{}\t{}",
                    chunk.index,
                    chunk.terminator.symbol(),
                    chunk.text
                )?,
                _ => {
                    let text = synth.dump(&chunk, stage)?.ok_or_else(|| {
                        Error::usage(format!(
                            "--dump {stage}: not available for the {} backend",
                            backend.id()
                        ))
                    })?;
                    if stage == "frames" && backend.id() == "klatt" && !printed_header {
                        writeln!(out, "{}", crate::klatt::backend::frames_header())?;
                        printed_header = true;
                    }
                    out.write_all(text.as_bytes())?;
                }
            }
            continue;
        }
        if let Some(l) = &lights {
            l.chunk(&chunk, samples_sent as f64 / spec.sample_rate as f64);
        }
        if let Some(h) = &highlighter {
            h.chunk(&chunk, samples_sent as f64 / spec.sample_rate as f64);
        }
        let stats = if args.no_stream {
            let s = synth.synthesize(&chunk, &mut buffered)?;
            synth_time += t.elapsed();
            s
        } else {
            let mut timed = TimedWriter {
                inner: sink.as_mut(),
                blocked: Duration::ZERO,
            };
            let s = synth.synthesize(&chunk, &mut timed)?;
            synth_time += t.elapsed().saturating_sub(timed.blocked);
            s
        };
        samples_sent += stats.samples;
        if args.progress {
            let audio_s = samples_sent as f64 / spec.sample_rate as f64;
            let total = total.map(|n| n.to_string()).unwrap_or_else(|| "?".into());
            eprint!(
                "\rspeak: chunk {n_chunks}/{total}  {audio_s:.1} s audio  rtf {:.4} ",
                synth_time.as_secs_f64() / audio_s.max(1e-9)
            );
        }
        if stats.stopped {
            break;
        }
    }
    if args.progress && n_chunks > 0 {
        eprintln!();
    }
    for block in buffered.chunks(320) {
        if sink.write(block)? == Flow::Stop {
            break;
        }
    }
    // The frontend may be blocked on a full channel or on stdin; it ends when the receiver is gone.
    let front = if stop.load(Ordering::Relaxed) && !frontend.is_finished() {
        None
    } else {
        Some(frontend.join())
    };
    let stats = sink.finish()?;
    if let Some(l) = lights {
        l.finish();
    }
    if let Some(h) = highlighter {
        h.finish();
    }
    if stop.load(Ordering::Relaxed) {
        return Err(Error::Interrupted);
    }
    let front = match front {
        Some(joined) => {
            joined.map_err(|_| Error::Synthesis("frontend thread panicked".into()))??
        }
        None => FrontStats::default(),
    };
    if args.verbose && front.dropped > 0 {
        eprintln!("speak: dropped {} unsupported characters", front.dropped);
    }

    if let Some(fmt) = timing_fmt {
        let report = TimingReport {
            backend: backend.id().into(),
            voice: voice_name,
            sample_rate: spec.sample_rate,
            chunks: n_chunks,
            startup_ms,
            frontend_init_ms: front.init_ms,
            voice_open_ms,
            frontend_ms: front.work_ms,
            ttfa_ms: stats
                .first_sample
                .map(|t| (t - start).as_secs_f64() * 1e3)
                .unwrap_or(0.0),
            synth_total_ms: synth_time.as_secs_f64() * 1e3,
            audio_s: stats.samples as f64 / spec.sample_rate as f64,
            underruns: stats.underruns,
            lexicon_hits: hits,
        };
        if fmt == "json" {
            eprintln!("{}", report.to_json())
        } else {
            eprint!("{}", report.to_text())
        }
    }
    Ok(())
}

/// `--resynth`: no text; `-r N` is a speed factor N/175, `--pitch` takes only the `±Nst` form.
fn run_resynth(args: &Args, cfg: &Config, start: Instant) -> Result<()> {
    use crate::cli::resynth;
    use crate::diphone::synth::Excitation;
    use crate::traits::Mods;
    let path = args
        .resynth
        .as_deref()
        .ok_or_else(|| Error::usage("--resynth needs a file"))?;
    if !args.text.is_empty() || args.input_file.is_some() || args.phonemes.is_some() {
        return Err(Error::usage("--resynth takes no text input"));
    }
    let mut mods = Mods::default();
    if let Some(r) = args.rate {
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::usage(format!("-r {r}: expected words per minute")));
        }
        mods.time = crate::DEFAULT_WPM / r;
    }
    if let Some(p) = args.pitch.as_deref() {
        let st: f32 = p
            .strip_suffix("st")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| {
                Error::usage(format!(
                    "--pitch {p}: with --resynth, give semitones (+4st)"
                ))
            })?;
        mods.pitch = 2f32.powf(st / 12.0);
    }
    let mut excitation = Excitation::Glottal;
    for p in &args.params {
        match p.split_once('=') {
            Some(("excitation", v)) => excitation = Excitation::parse(v)?,
            _ => {
                return Err(Error::usage(format!(
                    "-P {p}: --resynth understands only excitation=pulse|glottal"
                )));
            }
        }
    }
    let vocoder = args.vocoder.as_deref().unwrap_or("lpc");
    let output = output_format(args)?;
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        let _ = ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
        });
    }
    let pcm = resynth::load(path)?;
    let mut sink: Box<dyn AudioSink> = match output {
        Some((p, file_format, data_format, channels)) => Box::new(
            FileSink::with_format(p.as_deref(), file_format, data_format, channels)?
                .with_stop(stop.clone()),
        ),
        None => Box::new(Playback::open(
            args.audio_device.as_deref().or(cfg.audio.device.as_deref()),
            resynth::SR,
            stop.clone(),
        )?),
    };
    sink.begin(resynth::SR)?;
    let t = Instant::now();
    let result = resynth::resynth(&pcm, vocoder, excitation, &mods, sink.as_mut());
    let synth_ms = t.elapsed().as_secs_f64() * 1e3;
    let stats = sink.finish()?;
    result?;
    if stop.load(Ordering::Relaxed) {
        return Err(Error::Interrupted);
    }
    if args.timing.is_some() {
        eprintln!(
            "vocoder      {vocoder}\ninput_s      {:.2}\naudio_s      {:.2}\ntotal_ms     {synth_ms:.1}\nstartup_ms   {:.1}",
            pcm.len() as f64 / resynth::SR as f64,
            stats.samples as f64 / resynth::SR as f64,
            start.elapsed().as_secs_f64() * 1e3 - synth_ms
        );
    }
    Ok(())
}

/// `--bench[=N]`: one warm-up, then N full runs (frontend, voice open, synthesis) into a null
/// sink. TTFA here is run start → first sample, without an audio device.
fn run_bench(args: &Args, cfg: &Config, n: usize) -> Result<()> {
    let all = backends();
    let data = data_dirs();
    let resolved = resolve(&all, args, cfg, &data)?;
    let backend = &all[resolved.backend];
    let opts = parse_params(&all, &args.params, &resolved.alias_params, cfg)?;
    let controls = controls(args, cfg, resolved.alias_rate)?;
    let text = match read_input(args)? {
        Input::Text(t) => t,
        Input::Lines => {
            return Err(Error::usage(
                "--bench needs text: give it as arguments, -f file or on stdin",
            ));
        }
    };
    let once = || -> Result<(f64, f64, f64)> {
        let t0 = Instant::now();
        let chunks = match args.phonemes.as_deref() {
            None => Frontend::new(
                Some(&user_lexicon_path()),
                Some(&data.data),
                opts.params.get("lts").map(String::as_str),
            )?
            .chunks(&text, controls)?,
            Some("pho") => parse_pho(&text)?,
            Some(_) => arpabet_chunks(&text, controls)?,
        };
        let mut synth = backend.open(
            &VoiceSpec {
                name: resolved.voice.clone(),
            },
            &opts,
            &data,
        )?;
        let spec = synth.spec();
        let mut sink = NullSink::new();
        sink.begin(spec.sample_rate)?;
        let mut synth_time = Duration::ZERO;
        for mut c in chunks {
            let t = Instant::now();
            RuleProsody.apply(&mut c, &spec.prosody);
            synth.synthesize(&c, &mut sink)?;
            synth_time += t.elapsed();
        }
        let stats = sink.finish()?;
        let audio_s = stats.samples as f64 / spec.sample_rate as f64;
        let ttfa = stats
            .first_sample
            .map(|t| (t - t0).as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        Ok((ttfa, synth_time.as_secs_f64() / audio_s.max(1e-9), audio_s))
    };
    once()?;
    let mut runs = Vec::with_capacity(n);
    for _ in 0..n {
        runs.push(once()?);
    }
    let stat = |pick: fn(&(f64, f64, f64)) -> f64| -> (f64, f64, f64) {
        let mut v: Vec<f64> = runs.iter().map(pick).collect();
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (ttfa, rtf) = (stat(|r| r.0), stat(|r| r.1));
    println!(
        "backend  {}\nvoice    {}\nruns     {n}\naudio_s  {:.2}",
        backend.id(),
        resolved.voice,
        runs[0].2
    );
    println!(
        "ttfa_ms  median {:.2}  min {:.2}  max {:.2}",
        ttfa.0, ttfa.1, ttfa.2
    );
    println!(
        "rtf      median {:.5}  min {:.5}  max {:.5}",
        rtf.0, rtf.1, rtf.2
    );
    Ok(())
}

#[derive(Default)]
struct FrontStats {
    init_ms: f64,
    work_ms: f64,
    dropped: usize,
}

/// normalize → chunk → G2P → postlexical, or the phoneme parsers. Runs on its own thread.
fn frontend_thread(
    input: Input,
    phonemes: Option<String>,
    lts: Option<String>,
    controls: Controls,
    tx: SyncSender<(Chunk, Option<usize>)>,
    stop: Arc<AtomicBool>,
) -> Result<FrontStats> {
    let mut stats = FrontStats::default();
    let t_init = Instant::now();
    // Rule and lexicon files are parsed here, on the frontend thread.
    let fe = match phonemes.as_deref() {
        None => Some(Frontend::new(
            Some(&user_lexicon_path()),
            Some(&data_dirs().data),
            lts.as_deref(),
        )?),
        Some("arpabet" | "pho") => None,
        Some(other) => {
            return Err(Error::usage(format!(
                "--phonemes={other}: expected arpabet or pho"
            )));
        }
    };
    stats.init_ms = t_init.elapsed().as_secs_f64() * 1e3;
    let mut index = 0usize;
    let mut controls = controls;
    let mut process = |text: &str, known_total: bool, stats: &mut FrontStats| -> Result<bool> {
        let t = Instant::now();
        let chunks = match (&fe, phonemes.as_deref()) {
            (Some(fe), _) => fe.chunks(text, controls)?,
            (None, Some("pho")) => {
                let mut cs = parse_pho(text)?;
                for c in &mut cs {
                    c.controls.volume = controls.volume;
                }
                cs
            }
            (None, _) => arpabet_chunks(text, controls)?,
        };
        stats.work_ms += t.elapsed().as_secs_f64() * 1e3;
        let total = known_total.then_some(chunks.len());
        // Embedded commands change the controls for everything after them.
        if let Some(last) = chunks.last() {
            controls = last.controls;
        }
        for mut c in chunks {
            c.index = index;
            index += 1;
            if stop.load(Ordering::Relaxed) || tx.send((c, total)).is_err() {
                return Ok(false);
            }
        }
        Ok(true)
    };
    match input {
        Input::Text(text) => {
            // Long input is processed piece by piece so the first sentence is not kept waiting.
            let pieces = split_pieces(&text);
            let single = pieces.len() == 1;
            for piece in pieces {
                if !process(piece, single, &mut stats)? {
                    break;
                }
            }
        }
        Input::Lines => {
            // TTY: speak each line as it is entered, until EOF.
            for line in std::io::stdin().lock().lines() {
                let line = line?;
                if !process(&line, false, &mut stats)? {
                    break;
                }
            }
        }
    }
    stats.dropped = fe.as_ref().map(|f| f.dropped()).unwrap_or(0);
    Ok(stats)
}

/// Splits long text at paragraph, line and (failing those) plain sentence boundaries into
/// pieces of a few kilobytes. A piece never ends inside an embedded command.
fn split_pieces(text: &str) -> Vec<&str> {
    const TARGET: usize = 2000;
    let bytes = text.as_bytes();
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut open_command = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'[' if bytes.get(i + 1) == Some(&b'[') => open_command = true,
            b']' if bytes.get(i + 1) == Some(&b']') => open_command = false,
            _ => {}
        }
        let size = i - start;
        if size >= TARGET && !open_command {
            let line_end = bytes[i] == b'\n';
            // ". " + capital, after a lowercase word that is not an abbreviation.
            let sentence_end = size >= 2 * TARGET
                && matches!(bytes[i], b'.' | b'!' | b'?')
                && bytes.get(i + 1) == Some(&b' ')
                && bytes.get(i + 2).is_some_and(|c| c.is_ascii_uppercase())
                && {
                    let w = bytes[..i]
                        .iter()
                        .rposition(|c| !c.is_ascii_lowercase())
                        .map(|p| p + 1)
                        .unwrap_or(0);
                    i - w >= 2
                        && crate::text::abbrev::abbreviation(&format!("{}.", &text[w..i])).is_none()
                };
            if line_end || sentence_end {
                pieces.push(&text[start..=i]);
                start = i + 1;
            }
        }
        i += 1;
    }
    if start < text.len() || pieces.is_empty() {
        pieces.push(&text[start..]);
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_cover_the_text_and_split_at_safe_points() {
        assert_eq!(split_pieces("short text"), ["short text"]);
        assert_eq!(split_pieces(""), [""]);
        let long = "This is sentence number one. ".repeat(1000);
        let pieces = split_pieces(&long);
        assert!(pieces.len() > 4);
        assert_eq!(pieces.concat(), long);
        assert!(pieces.iter().all(|p| p.len() < 6000));
        assert!(pieces[0].ends_with("one."));
        let lines = "A line of text without a full stop\n".repeat(500);
        let pieces = split_pieces(&lines);
        assert_eq!(pieces.concat(), lines);
        assert!(pieces.iter().all(|p| p.ends_with('\n')));
        // Never inside an embedded command.
        let cmd = format!(
            "{}[[inpt PHON]] {} [[inpt TEXT]] end.",
            "word ".repeat(450),
            "AA1 . B ".repeat(900)
        );
        assert_eq!(split_pieces(&cmd).len(), 1);
    }
}
