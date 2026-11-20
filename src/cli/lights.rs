//! SidePulse mouth: an 8-LED strip that follows the voice.
//!
//! The device mounts as a small drive; writing an animation program to `LEDS.LED` changes the
//! lights (≤ 512 bytes, ≤ 20 lines). Each syllable becomes one pulse: the mouth opens from the
//! centre by the vowel's openness (F1), the hue follows F2, the brightness follows stress.
//! Programs are written in step with the playback clock, a second or so of speech at a time.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::chunk::{Chunk, Unit};
use crate::klatt::tables::row;
use crate::phone::Stress;

const LEDS: usize = 8;
const MAX_BYTES: usize = 480;
const MAX_LINES: usize = 18;
/// Time a write takes to reach the LEDs (measured 4–9 ms).
const LEAD_S: f64 = 0.012;

/// Seconds of audio that have left the speaker.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

struct Batch {
    at_s: f64,
    end_s: f64,
    program: String,
}

pub struct Lights {
    tx: Option<Sender<Batch>>,
    worker: Option<JoinHandle<()>>,
}

/// The mounted device, unless disabled with `SPEAK_LIGHTS=0`. `SPEAK_LIGHTS=/path` overrides.
pub fn detect() -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = match std::env::var("SPEAK_LIGHTS") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => return None,
        Ok(v) if v.starts_with('/') => vec![PathBuf::from(v)],
        _ => [
            "/Volumes/SidePulse",
            "/Volumes/SidePulsePro",
            "/Volumes/SidePulseDot",
        ]
        .iter()
        .map(PathBuf::from)
        .collect(),
    };
    candidates
        .into_iter()
        .map(|d| d.join("LEDS.LED"))
        .find(|f| f.is_file())
}

fn write_program(file: &Path, program: &str) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(file)?;
    f.write_all(program.as_bytes())?;
    f.sync_all()
}

fn hsv(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let f = |u: f32| ((u + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (f(r), f(g), f(b))
}

/// One syllable: when, how long, and what the mouth looks like.
#[derive(Clone, Debug, PartialEq)]
pub struct Syllable {
    pub start_ms: f32,
    pub dur_ms: f32,
    /// LEDs lit, centred: 2, 4, 6 or 8.
    pub width: usize,
    pub rgb: (u8, u8, u8),
}

/// Syllables of a timed chunk; times are relative to the chunk start.
pub fn syllables(chunk: &Chunk) -> Vec<Syllable> {
    let mut out = Vec::new();
    let mut t = 0.0f32;
    for u in &chunk.units {
        if let Unit::Seg(s) = u
            && s.phone.is_vowel()
            && s.dur_ms > 0.0
        {
            let r = row(s.phone);
            let (f1, f2) = (0.5 * (r.f[0].0 + r.f[0].1), 0.5 * (r.f[1].0 + r.f[1].1));
            let open = ((f1 - 280.0) / (700.0 - 280.0)).clamp(0.0, 1.0);
            let width = 2 + 2 * (open * 3.0).round() as usize;
            // Back vowels are warm, front vowels are cool.
            let hue = 15.0 + ((f2 - 900.0) / (2100.0 - 900.0)).clamp(0.0, 1.0) * 195.0;
            let value = match s.stress {
                Stress::Primary => 1.0,
                Stress::Secondary => 0.7,
                Stress::None => 0.4,
            };
            out.push(Syllable {
                start_ms: t,
                dur_ms: s.dur_ms,
                width: width.min(LEDS),
                rgb: hsv(hue, 0.95, value),
            });
        }
        t += u.dur_ms();
    }
    out
}

fn line(s: &Syllable, dur_ms: f32) -> String {
    let (r, g, b) = s.rgb;
    let color = format!("#{r:02x}{g:02x}{b:02x}");
    let dark = (LEDS - s.width) / 2;
    let mut out = String::new();
    for i in 0..LEDS {
        out.push_str(if i < dark || i >= LEDS - dark {
            "off"
        } else {
            &color
        });
        out.push(' ');
    }
    out.push_str(&format!("{}ms pulse\n", dur_ms.round().max(17.0) as u32));
    out
}

/// Programs covering `syls` (absolute times in ms), each within the device's size limits.
fn batches(syls: &[Syllable]) -> Vec<Batch> {
    let mut out: Vec<Batch> = Vec::new();
    let mut cur = String::new();
    let (mut lines, mut start, mut cursor) = (0usize, 0.0f32, 0.0f32);
    let flush = |cur: &mut String, start: f32, end: f32, out: &mut Vec<Batch>| {
        if !cur.is_empty() {
            out.push(Batch {
                at_s: start as f64 / 1000.0,
                end_s: end as f64 / 1000.0,
                program: std::mem::take(cur),
            });
        }
    };
    for (i, s) in syls.iter().enumerate() {
        // The pulse leads the vowel a little and runs to the next syllable or 60 ms past the vowel.
        let begin = (s.start_ms - 30.0).max(cursor);
        let natural_end = s.start_ms + s.dur_ms + 60.0;
        let end = syls
            .get(i + 1)
            .map(|n| (n.start_ms - 30.0).min(natural_end))
            .unwrap_or(natural_end)
            .max(begin + 30.0);
        let gap = begin - cursor;
        let mut text = String::new();
        if !cur.is_empty() && gap >= 50.0 {
            text.push_str(&format!("off {}ms none\n", gap.round() as u32));
        }
        text.push_str(&line(s, end - begin));
        let new_lines = text.lines().count();
        if cur.is_empty()
            || cur.len() + text.len() > MAX_BYTES
            || lines + new_lines > MAX_LINES
            || gap > 600.0
        {
            flush(&mut cur, start, cursor, &mut out);
            // A program starts dark so pulses return to dark.
            cur.push_str("off\n");
            lines = 1;
            start = begin;
            text = line(s, end - begin);
        }
        lines += text.lines().count();
        cur.push_str(&text);
        cursor = end;
    }
    flush(&mut cur, start, cursor, &mut out);
    out
}

fn run(file: PathBuf, clock: Clock, stop: Arc<AtomicBool>, rx: Receiver<Batch>) {
    // Borrow the LEDs: whatever was showing comes back afterwards.
    let saved = std::fs::read_to_string(&file).ok();
    let mut end_s = 0.0f64;
    for batch in rx {
        while clock() + LEAD_S < batch.at_s && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(2));
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
        // Late by more than the batch itself (a stall): skip it.
        if clock() < batch.end_s && write_program(&file, &batch.program).is_err() {
            break;
        }
        end_s = batch.end_s;
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while clock() < end_s && !stop.load(Ordering::Relaxed) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    let _ = write_program(&file, saved.as_deref().unwrap_or("off\n"));
}

impl Lights {
    pub fn start(file: PathBuf, clock: Clock, stop: Arc<AtomicBool>) -> Lights {
        let (tx, rx) = channel();
        let worker = std::thread::spawn(move || run(file, clock, stop, rx));
        Lights {
            tx: Some(tx),
            worker: Some(worker),
        }
    }

    /// Queues the lights for a timed chunk whose audio starts at `chunk_start_s` in the stream.
    pub fn chunk(&self, chunk: &Chunk, chunk_start_s: f64) {
        let offset = (chunk_start_s * 1000.0) as f32;
        let syls: Vec<Syllable> = syllables(chunk)
            .into_iter()
            .map(|s| Syllable {
                start_ms: s.start_ms + offset,
                ..s
            })
            .collect();
        if let Some(tx) = &self.tx {
            for b in batches(&syls) {
                let _ = tx.send(b);
            }
        }
    }

    /// Waits for the last program to play out and restores the previous LED state.
    pub fn finish(mut self) {
        self.tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Terminator;
    use crate::phone::parse_arpabet;
    use crate::text::RuleProsody;
    use crate::traits::{Prosody, ProsodyParams};

    fn timed(text: &str) -> Chunk {
        let mut c = Chunk {
            units: parse_arpabet(text).unwrap(),
            terminator: Terminator::Period,
            ..Chunk::default()
        };
        RuleProsody.apply(&mut c, &ProsodyParams::default());
        c
    }

    #[test]
    fn one_pulse_per_syllable_and_open_vowels_are_wide() {
        let c = timed("B AA1 B / B IY1 B / B AX B .");
        let s = syllables(&c);
        assert_eq!(s.len(), 3);
        assert!(s[0].width > s[1].width, "AA is more open than IY");
        assert!(s[0].start_ms < s[1].start_ms && s[1].start_ms < s[2].start_ms);
        // Stressed syllables are brighter.
        let lum = |x: &Syllable| x.rgb.0 as u32 + x.rgb.1 as u32 + x.rgb.2 as u32;
        assert!(lum(&s[0]) > lum(&s[2]));
    }

    #[test]
    fn programs_respect_the_device_limits() {
        let c = timed(&"DH AX / K W IH1 K / B R AW1 N / F AA1 K S / JH AH1 M P S / OW1 V AXR / DH AX / L EY1 Z IY0 / D AO1 G , ".repeat(4));
        let bs = batches(&syllables(&c));
        assert!(bs.len() >= 4);
        let mut last = -1.0;
        for b in &bs {
            assert!(b.program.len() <= 512, "{} bytes", b.program.len());
            assert!(b.program.lines().count() <= 20);
            assert!(b.at_s > last && b.end_s > b.at_s);
            last = b.at_s;
            for l in b.program.lines().skip(1) {
                assert!(l.ends_with("ms pulse") || l.ends_with("ms none"), "{l}");
            }
        }
    }

    #[test]
    fn writes_in_step_with_the_clock_and_restores() {
        let dir = std::env::temp_dir().join(format!("speak-lights-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("LEDS.LED");
        std::fs::write(&file, "#00ff00\n").unwrap();
        let t0 = std::time::Instant::now();
        let clock: Clock = Arc::new(move || t0.elapsed().as_secs_f64());
        let lights = Lights::start(file.clone(), clock, Arc::new(AtomicBool::new(false)));
        lights.chunk(&timed("HH AX L OW1 ."), 0.05);
        std::thread::sleep(Duration::from_millis(150));
        let during = std::fs::read_to_string(&file).unwrap();
        assert!(during.contains("pulse"), "{during}");
        lights.finish();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "#00ff00\n");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
