//! SidePulse strip: the mouth of `crate::sidepulse`, played in step with the audio.
//!
//! The device mounts as a small drive; writing an animation program to `LEDS.LED` changes the
//! lights (≤ 512 bytes, ≤ 20 lines). The strip is a mouth seen from the front: on each syllable
//! it opens from the centre by the vowel's openness (F1), then closes to a dim slit, or goes dark
//! when the lips shut or the voice pauses. The colour runs from a centre hue set by F2 to warmer
//! lips, and the brightness follows stress.
//! Programs are written in step with the playback clock, a second or so of speech at a time.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::chunk::Chunk;
use crate::sidepulse::{DARK, Key, Shape, Syllable, keyframes, syllables};
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
    /// Set once the audio has finished: the clock no longer advances, so nothing waits on it.
    ended: Arc<AtomicBool>,
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

/// A transition to `shape` over `ms`, as one program line.
fn line(shape: &Shape, ms: u32) -> String {
    let mut out = String::new();
    for &(r, g, b) in shape {
        out.push_str(&format!("#{r:02x}{g:02x}{b:02x} "));
    }
    out.push_str(&format!("{}ms cosine\n", ms.max(17)));
    out
}

/// Programs playing `keys`, each within the device's size limits. A long hold ends a program;
/// the next one is written when its first key comes up.
fn batches(keys: &[Key]) -> Vec<Batch> {
    let mut out: Vec<Batch> = Vec::new();
    let mut cur = String::from("off\n");
    let (mut lines, mut start, mut end) = (1usize, keys.first().map_or(0, |k| k.ms), 0u32);
    for w in keys.windows(2) {
        let (from, to) = (&w[0], &w[1]);
        let ms = to.ms - from.ms;
        let text = if to.shape != from.shape {
            line(&to.shape, ms)
        } else if ms > 600 {
            // A pause: end this program and start the next one at the following key.
            flush(&mut cur, start, end, &mut out);
            (lines, start) = (0, to.ms);
            continue;
        } else if to.shape == DARK {
            format!("off {ms}ms none\n")
        } else {
            line(&to.shape, ms)
        };
        if cur.len() + text.len() > MAX_BYTES || lines + 1 > MAX_LINES {
            flush(&mut cur, start, end, &mut out);
            (lines, start) = (0, from.ms);
        }
        cur.push_str(&text);
        lines += 1;
        end = to.ms;
    }
    flush(&mut cur, start, end, &mut out);
    out
}

fn flush(cur: &mut String, start: u32, end: u32, out: &mut Vec<Batch>) {
    if !cur.is_empty() && end > start {
        out.push(Batch {
            at_s: start as f64 / 1000.0,
            end_s: end as f64 / 1000.0,
            program: std::mem::take(cur),
        });
    }
    cur.clear();
}

fn run(
    file: PathBuf,
    clock: Clock,
    stop: Arc<AtomicBool>,
    ended: Arc<AtomicBool>,
    rx: Receiver<Batch>,
) {
    // Borrow the LEDs: whatever was showing comes back afterwards.
    let saved = std::fs::read_to_string(&file).ok();
    let done = || stop.load(Ordering::Relaxed) || ended.load(Ordering::Relaxed);
    let mut end_s = 0.0f64;
    for batch in rx {
        while clock() + LEAD_S < batch.at_s && !done() {
            std::thread::sleep(Duration::from_millis(2));
        }
        if done() {
            break;
        }
        // Late by more than the batch itself (a stall): skip it.
        if clock() < batch.end_s && write_program(&file, &batch.program).is_err() {
            break;
        }
        end_s = batch.end_s;
    }
    while clock() < end_s && !done() {
        std::thread::sleep(Duration::from_millis(5));
    }
    let _ = write_program(&file, saved.as_deref().unwrap_or("off\n"));
}

impl Lights {
    pub fn start(file: PathBuf, clock: Clock, stop: Arc<AtomicBool>) -> Lights {
        let (tx, rx) = channel();
        let ended = Arc::new(AtomicBool::new(false));
        let worker = {
            let ended = ended.clone();
            std::thread::spawn(move || run(file, clock, stop, ended, rx))
        };
        Lights {
            tx: Some(tx),
            worker: Some(worker),
            ended,
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
            for b in batches(&keyframes(&syls)) {
                let _ = tx.send(b);
            }
        }
    }

    /// Call once the audio has finished playing: restores the previous LED state.
    pub fn finish(mut self) {
        self.ended.store(true, Ordering::Relaxed);
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
    use crate::sidepulse::{LEDS, Rgb, mouth};
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
    fn one_shape_per_syllable_and_open_vowels_are_wide() {
        let c = timed("B AA1 B / B IY1 B / B AX B .");
        let s = syllables(&c);
        assert_eq!(s.len(), 3);
        assert!(s[0].start_ms < s[1].start_ms && s[1].start_ms < s[2].start_ms);
        assert!(s.iter().all(|x| x.lips_shut), "B closes the lips");
        let lit = |x: &Syllable| {
            mouth(x.open, x.hue, x.level)
                .iter()
                .filter(|c| **c != (0, 0, 0))
                .count()
        };
        assert!(lit(&s[0]) > lit(&s[1]), "AA is more open than IY");
        // Stressed syllables are brighter.
        let lum = |x: &Syllable| -> u32 {
            mouth(x.open, x.hue, x.level)
                .iter()
                .map(|c| c.0 as u32 + c.1 as u32 + c.2 as u32)
                .sum()
        };
        assert!(lum(&s[0]) > lum(&s[2]));
    }

    #[test]
    fn the_mouth_is_symmetric_with_a_gradient() {
        let m = mouth(1.0, 200.0, 1.0);
        assert!(
            m.iter().all(|c| *c != (0, 0, 0)),
            "fully open lights every LED"
        );
        let lum = |c: Rgb| c.0 as u32 + c.1 as u32 + c.2 as u32;
        for i in 0..LEDS / 2 {
            assert_eq!(m[i], m[LEDS - 1 - i]);
        }
        assert!(lum(m[3]) > lum(m[0]), "lips are dimmer than the middle");
        assert_ne!(m[3], m[0], "colour changes towards the lips");
    }

    #[test]
    fn programs_respect_the_device_limits_and_syntax() {
        let c = timed(&"DH AX / K W IH1 K / B R AW1 N / F AA1 K S / JH AH1 M P S / OW1 V AXR / DH AX / L EY1 Z IY0 / D AO1 G , ".repeat(4));
        let bs = batches(&keyframes(&syllables(&c)));
        assert!(bs.len() >= 4);
        let mut last = -1.0;
        for b in &bs {
            assert!(b.program.len() <= 512, "{} bytes", b.program.len());
            assert!(b.program.lines().count() <= 20);
            assert!(b.at_s > last && b.end_s > b.at_s);
            last = b.at_s;
            for l in b.program.lines() {
                let t: Vec<&str> = l.split(' ').collect();
                let ok = match t.as_slice() {
                    ["off"] => true,
                    ["off", ms, "none"] => ms.ends_with("ms"),
                    _ => {
                        t.len() == LEDS + 2
                            && t[..LEDS].iter().all(|c| c.len() == 7 && c.starts_with('#'))
                            && t[LEDS].ends_with("ms")
                            && t[LEDS + 1] == "cosine"
                    }
                };
                assert!(ok, "not a valid program line: {l}");
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
        assert!(during.contains("cosine"), "{during}");
        lights.finish();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "#00ff00\n");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
