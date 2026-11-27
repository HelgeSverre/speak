//! Browser bindings: text in, PCM plus word timings and SidePulse mouth keyframes out.
//!
//! Built with `wasm-pack build --target web --no-default-features --features wasm`. Only the
//! built-in Klatt voices are available: there is no file system for CMUdict, user lexicons or
//! diphone voices, so words outside the built-in lexicon go through the letter-to-sound tree.

use std::fmt::Write as _;

use wasm_bindgen::prelude::*;

use crate::chunk::{Chunk, Controls, Unit};
use crate::klatt::KlattBackend;
use crate::pho::print_pho;
use crate::phone::print_arpabet;
use crate::sidepulse;
use crate::text::sing::{self, SingProsody};
use crate::text::{Frontend, RuleProsody};
use crate::traits::{Backend, DataDirs, Prosody, SynthOptions, VoiceSpec};

/// Silence kept after the last chunk (as in the CLI).
const FINAL_TAIL_MS: f32 = 60.0;

fn js_err(e: crate::Error) -> JsError {
    JsError::new(&e.to_string())
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The built-in voices as JSON: `[{"name": …, "description": …}]`.
#[wasm_bindgen]
pub fn voices() -> String {
    let data = DataDirs {
        data: Default::default(),
    };
    let items: Vec<String> = KlattBackend
        .voices(&data)
        .iter()
        .map(|v| {
            format!(
                "{{\"name\":{},\"description\":{}}}",
                json_str(&v.name),
                json_str(&v.description)
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// The built-in tunes as JSON: `[{"name": …, "tempo": …, "melody": …, "lyrics": …}]`.
#[wasm_bindgen]
pub fn tunes() -> String {
    let items: Vec<String> = sing::TUNES
        .iter()
        .map(|t| {
            format!(
                "{{\"name\":{},\"tempo\":{},\"melody\":{},\"lyrics\":{}}}",
                json_str(t.name),
                t.tempo,
                json_str(&t.melody.split_whitespace().collect::<Vec<_>>().join(" ")),
                json_str(t.lyrics)
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// The default voice name.
#[wasm_bindgen]
pub fn default_voice() -> String {
    KlattBackend.default_voice().into()
}

/// What to say and how. Empty strings and zeros mean "default", as an omitted flag would.
#[wasm_bindgen]
#[derive(Default)]
pub struct Request {
    text: String,
    voice: String,
    rate: f32,
    pitch: String,
    volume: f32,
    params: String,
    phonemes: String,
    sing: String,
    melody: String,
    tempo: f32,
    key: f32,
}

#[wasm_bindgen]
impl Request {
    #[wasm_bindgen(constructor)]
    pub fn new(text: String) -> Request {
        Request {
            text,
            volume: 1.0,
            ..Request::default()
        }
    }
    /// `-v`
    #[wasm_bindgen(setter)]
    pub fn set_voice(&mut self, v: String) {
        self.voice = v;
    }
    /// `-r`, words per minute; 0 keeps the default.
    #[wasm_bindgen(setter)]
    pub fn set_rate(&mut self, v: f32) {
        self.rate = v;
    }
    /// `--pitch`: `110` or `+2st`.
    #[wasm_bindgen(setter)]
    pub fn set_pitch(&mut self, v: String) {
        self.pitch = v;
    }
    /// `--volume`, 0.0–2.0.
    #[wasm_bindgen(setter)]
    pub fn set_volume(&mut self, v: f32) {
        self.volume = v;
    }
    /// `-P` pairs separated by whitespace: `oq=0.6 breath=0.2`.
    #[wasm_bindgen(setter)]
    pub fn set_params(&mut self, v: String) {
        self.params = v;
    }
    /// `--phonemes`: `arpabet` or `pho`; empty for text.
    #[wasm_bindgen(setter)]
    pub fn set_phonemes(&mut self, v: String) {
        self.phonemes = v;
    }
    /// `--sing`: a built-in tune name.
    #[wasm_bindgen(setter)]
    pub fn set_sing(&mut self, v: String) {
        self.sing = v;
    }
    /// `--melody`: `C4 D4:2 R`.
    #[wasm_bindgen(setter)]
    pub fn set_melody(&mut self, v: String) {
        self.melody = v;
    }
    /// `--tempo` in BPM; 0 keeps the tune's own.
    #[wasm_bindgen(setter)]
    pub fn set_tempo(&mut self, v: f32) {
        self.tempo = v;
    }
    /// `--key` in semitones.
    #[wasm_bindgen(setter)]
    pub fn set_key(&mut self, v: f32) {
        self.key = v;
    }
}

/// A synthesized utterance.
#[wasm_bindgen]
pub struct Utterance {
    samples: Vec<f32>,
    sample_rate: u32,
    sing_shift: f32,
    words: String,
    sidepulse: String,
    text: String,
    phonemes: String,
    pho: String,
}

#[wasm_bindgen]
impl Utterance {
    /// Mono PCM, −1..1.
    #[wasm_bindgen(getter)]
    pub fn samples(&self) -> Vec<f32> {
        self.samples.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Semitones the engine moved the written melody by to fit the voice; 0 when speaking.
    #[wasm_bindgen(getter)]
    pub fn sing_shift(&self) -> f32 {
        self.sing_shift
    }
    /// JSON `[{"text": …, "start": s, "end": s}]`, one entry per spoken word.
    #[wasm_bindgen(getter)]
    pub fn words(&self) -> String {
        self.words.clone()
    }
    /// JSON `[[ms, "#rrggbb", … 8 colours], …]`: the SidePulse mouth reaches each shape at `ms`.
    #[wasm_bindgen(getter)]
    pub fn sidepulse(&self) -> String {
        self.sidepulse.clone()
    }
    /// The normalized text, one chunk per line (`--dump text`).
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }
    /// ARPAbet, one chunk per line (`--dump phonemes`).
    #[wasm_bindgen(getter)]
    pub fn phonemes(&self) -> String {
        self.phonemes.clone()
    }
    /// MBROLA-style phone table with durations and pitch (`--dump pho`).
    #[wasm_bindgen(getter)]
    pub fn pho(&self) -> String {
        self.pho.clone()
    }
}

/// Holds the frontend (lexicon and letter-to-sound tree) so it is parsed once.
#[wasm_bindgen]
pub struct Speaker {
    frontend: Frontend,
}

#[wasm_bindgen]
impl Speaker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Speaker, JsError> {
        Ok(Speaker {
            frontend: Frontend::new(None, None, None).map_err(js_err)?,
        })
    }

    pub fn speak(&self, req: &Request) -> Result<Utterance, JsError> {
        self.run(req).map_err(js_err)
    }
}

impl Speaker {
    fn run(&self, req: &Request) -> crate::Result<Utterance> {
        let mut controls = Controls::default();
        if req.rate > 0.0 {
            controls.rate = req.rate.clamp(80.0, 500.0) / crate::DEFAULT_WPM;
        }
        if !req.pitch.trim().is_empty() {
            controls.set_pitch(req.pitch.trim())?;
        }
        controls.volume = req.volume.clamp(0.0, 2.0);

        let mut opts = SynthOptions::default();
        for p in req.params.split_whitespace() {
            let (k, v) = p
                .split_once('=')
                .ok_or_else(|| crate::Error::usage(format!("-P {p}: expected key=value")))?;
            opts.params.insert(k.into(), v.into());
        }
        let backend = KlattBackend;
        let voice = if req.voice.is_empty() {
            backend.default_voice().to_string()
        } else {
            req.voice.clone()
        };
        let data = DataDirs {
            data: Default::default(),
        };
        let mut synth = backend.open(&VoiceSpec { name: voice }, &opts, &data)?;
        let spec = synth.spec();

        let base_hz = controls.pitch_base_hz.unwrap_or(spec.prosody.base_hz)
            * 2f32.powf(controls.pitch_shift_st / 12.0);
        let (prosody, sing_shift): (Box<dyn Prosody>, f32) = match self.singer(req)? {
            Some(s) => {
                let shift = s.transpose(base_hz);
                (Box::new(s), shift)
            }
            None => (Box::new(RuleProsody), 0.0),
        };
        let mut text = req.text.clone();
        if text.trim().is_empty()
            && let Some(t) = sing::tune(&req.sing)
        {
            text = t.lyrics.to_string();
        }
        let mut chunks: Vec<Chunk> = match req.phonemes.as_str() {
            "" => self.frontend.chunks(&text, controls)?,
            "arpabet" => crate::pho::arpabet_chunks(&text, controls)?,
            "pho" => {
                let mut cs = crate::pho::parse_pho(&text)?;
                for c in &mut cs {
                    c.controls.volume = controls.volume;
                }
                cs
            }
            other => {
                return Err(crate::Error::usage(format!(
                    "--phonemes={other}: expected arpabet or pho"
                )));
            }
        };

        let rate = spec.sample_rate as f64;
        let mut samples: Vec<f32> = Vec::new();
        let (mut words, mut phonemes, mut pho) = (Vec::new(), String::new(), String::new());
        let mut normalized = String::new();
        let mut syls = Vec::new();
        let n = chunks.len();
        for (i, chunk) in chunks.iter_mut().enumerate() {
            chunk.index = i;
            prosody.apply(chunk, &spec.prosody);
            if i + 1 == n {
                chunk.pause_after_ms = chunk.pause_after_ms.min(FINAL_TAIL_MS);
            }
            let start_s = samples.len() as f64 / rate;
            word_times(chunk, start_s, &mut words);
            let offset = (start_s * 1000.0) as f32;
            syls.extend(
                sidepulse::syllables(chunk)
                    .into_iter()
                    .map(|s| sidepulse::Syllable {
                        start_ms: s.start_ms + offset,
                        ..s
                    }),
            );
            normalized.push_str(&chunk.text);
            normalized.push('\n');
            phonemes.push_str(&print_arpabet(&chunk.units));
            phonemes.push('\n');
            print_pho(chunk, &mut pho);
            synth.synthesize(chunk, &mut samples)?;
        }

        let mut mouth_json = String::from("[");
        for (i, k) in sidepulse::keyframes(&syls).iter().enumerate() {
            if i > 0 {
                mouth_json.push(',');
            }
            let _ = write!(mouth_json, "[{}", k.ms);
            for (r, g, b) in k.shape {
                let _ = write!(mouth_json, ",\"#{r:02x}{g:02x}{b:02x}\"");
            }
            mouth_json.push(']');
        }
        mouth_json.push(']');

        Ok(Utterance {
            samples,
            sample_rate: spec.sample_rate,
            sing_shift,
            words: format!("[{}]", words.join(",")),
            sidepulse: mouth_json,
            text: normalized,
            phonemes,
            pho,
        })
    }

    fn singer(&self, req: &Request) -> crate::Result<Option<SingProsody>> {
        let tune = match req.sing.as_str() {
            "" => None,
            name => Some(
                sing::tune(name)
                    .ok_or_else(|| crate::Error::usage(format!("unknown tune '{name}'")))?,
            ),
        };
        let (notes, tempo) = match (req.melody.trim(), tune) {
            ("", None) => return Ok(None),
            ("", Some(t)) => (sing::parse_melody(t.melody)?, t.tempo),
            (m, t) => (sing::parse_melody(m)?, t.map_or(120.0, |t| t.tempo)),
        };
        let tempo = if req.tempo > 0.0 { req.tempo } else { tempo };
        Ok(Some(SingProsody::new(notes, tempo, req.key)))
    }
}

/// Start and end of each word in `chunk`, as JSON objects, from the segment durations.
fn word_times(chunk: &Chunk, start_s: f64, out: &mut Vec<String>) {
    let mut spans: Vec<Option<(f64, f64)>> = vec![None; chunk.words.len()];
    let mut t = start_s;
    for u in &chunk.units {
        let d = u.dur_ms() as f64 / 1000.0;
        if let Unit::Seg(s) = u
            && let Some(span) = spans.get_mut(s.word as usize)
        {
            let e = span.get_or_insert((t, t));
            e.1 = t + d;
        }
        t += d;
    }
    for (w, span) in chunk.words.iter().zip(spans) {
        if let Some((a, b)) = span {
            out.push(format!(
                "{{\"text\":{},\"start\":{a:.3},\"end\":{b:.3}}}",
                json_str(&w.text)
            ));
        }
    }
}
