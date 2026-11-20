//! `-i`: print the text and highlight each word as it is spoken (spec §3.1).

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::chunk::{Chunk, Item, Unit};
use crate::cli::lights::Clock;

/// One state of the display: at `at_s`, show `line`.
struct Event {
    at_s: f64,
    line: String,
    /// Finish the line (the chunk is over).
    end: bool,
}

pub struct Highlighter {
    tx: Option<Sender<Event>>,
    worker: Option<JoinHandle<()>>,
}

const REVERSE: &str = "\x1b[7m";
const RESET: &str = "\x1b[0m";

/// The chunk's text with word `current` in reverse video (plain when `color` is off).
pub fn render(chunk: &Chunk, current: Option<usize>, color: bool) -> String {
    let mut s = String::new();
    for item in &chunk.items {
        match item {
            Item::Word { index, .. } => {
                if !s.is_empty() {
                    s.push(' ');
                }
                let text = &chunk.words[*index].text;
                if color && current == Some(*index) {
                    s.push_str(&format!("{REVERSE}{text}{RESET}"));
                } else {
                    s.push_str(text);
                }
            }
            Item::Punct(c) => {
                if *c == '—' {
                    s.push(' ');
                }
                s.push(*c);
            }
            Item::Phonemes(p) => {
                if !s.is_empty() {
                    s.push(' ');
                }
                s.push_str(&format!("[[{p}]]"));
            }
            Item::Silence(_) => {}
        }
    }
    s
}

/// Start time (ms from the chunk start) of each word that has phones.
pub fn word_starts(chunk: &Chunk) -> Vec<(usize, f32)> {
    let mut out: Vec<(usize, f32)> = Vec::new();
    let mut t = 0.0f32;
    for u in &chunk.units {
        if let Unit::Seg(s) = u
            && out.last().is_none_or(|l| l.0 != s.word as usize)
        {
            out.push((s.word as usize, t));
        }
        t += u.dur_ms();
    }
    out
}

fn run(clock: Clock, stop: Arc<AtomicBool>, rx: Receiver<Event>) {
    let mut out = std::io::stdout();
    for ev in rx {
        while clock() < ev.at_s && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(3));
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
        // Redraw the line in place.
        let _ = write!(
            out,
            "\r\x1b[2K{}{}",
            ev.line,
            if ev.end { "\n" } else { "" }
        );
        let _ = out.flush();
    }
    // Leave the terminal clean whatever happened.
    let _ = write!(out, "{RESET}");
    let _ = out.flush();
}

impl Highlighter {
    /// `None` when stdout is not a terminal: the text is then printed plainly by `chunk`.
    pub fn start(clock: Option<Clock>, stop: Arc<AtomicBool>) -> Highlighter {
        match clock.filter(|_| std::io::stdout().is_terminal()) {
            Some(clock) => {
                let (tx, rx) = channel();
                let worker = std::thread::spawn(move || run(clock, stop, rx));
                Highlighter {
                    tx: Some(tx),
                    worker: Some(worker),
                }
            }
            None => Highlighter {
                tx: None,
                worker: None,
            },
        }
    }

    /// Queues the display of a timed chunk whose audio starts at `start_s`.
    pub fn chunk(&self, chunk: &Chunk, start_s: f64) {
        let Some(tx) = &self.tx else {
            // No terminal or no real-time clock: just print the text.
            println!("{}", render(chunk, None, false));
            return;
        };
        let _ = tx.send(Event {
            at_s: start_s,
            line: render(chunk, None, true),
            end: false,
        });
        for (word, ms) in word_starts(chunk) {
            let _ = tx.send(Event {
                at_s: start_s + ms as f64 / 1000.0,
                line: render(chunk, Some(word), true),
                end: false,
            });
        }
        let end = start_s + chunk.dur_ms() as f64 / 1000.0;
        let _ = tx.send(Event {
            at_s: end,
            line: render(chunk, None, true),
            end: true,
        });
    }

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
    use crate::text::{Frontend, RuleProsody};
    use crate::traits::{Prosody, ProsodyParams};

    #[test]
    fn highlights_one_word_and_times_are_increasing() {
        let fe = Frontend::new(None, None, None).unwrap();
        let mut c = fe
            .chunks(
                "Each word lights up, as it is spoken.",
                crate::chunk::Controls::default(),
            )
            .unwrap()
            .remove(0);
        RuleProsody.apply(&mut c, &ProsodyParams::default());
        assert_eq!(
            render(&c, None, true),
            "Each word lights up, as it is spoken."
        );
        let line = render(&c, Some(2), true);
        assert_eq!(
            line,
            format!("Each word {REVERSE}lights{RESET} up, as it is spoken.")
        );
        assert_eq!(
            render(&c, Some(2), false),
            "Each word lights up, as it is spoken."
        );
        let starts = word_starts(&c);
        assert_eq!(starts.len(), 8);
        assert!(
            starts
                .windows(2)
                .all(|w| w[1].1 > w[0].1 && w[1].0 == w[0].0 + 1)
        );
        assert_eq!(starts[0], (0, 0.0));
    }
}
