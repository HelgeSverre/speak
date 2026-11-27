//! `speak --mcp`: a Model Context Protocol server, so coding agents can talk.
//!
//! Stdio transport: one JSON-RPC 2.0 message per line on stdin and on stdout, and nothing
//! else on stdout. Speech runs on a worker thread, one call at a time in arrival order, so
//! the reader stays free to act on `notifications/cancelled` while a sentence is playing.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::cli::args::Args;
use crate::cli::{backends, listing, pipeline};
use crate::error::{Error, Result};
use crate::text::sing;

/// Newest first. A client asking for another version is offered the newest.
const PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const PARSE_ERROR: i64 = -32700;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// A request, a notification, or a response (which we never ask for, so it is ignored).
#[derive(Deserialize)]
struct Message {
    id: Option<Value>,
    method: Option<String>,
    #[serde(default)]
    params: Value,
}

#[derive(Deserialize)]
struct ToolCall {
    name: String,
    #[serde(default)]
    arguments: Map<String, Value>,
}

/// Arguments of the `speak` tool.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeakArgs {
    #[serde(default)]
    text: String,
    voice: Option<String>,
    rate: Option<f32>,
    pitch: Option<String>,
    sing: Option<String>,
    file: Option<PathBuf>,
}

impl SpeakArgs {
    /// The command line this call stands for. Rejects what would read stdin or print to stdout.
    fn into_args(self) -> Result<Args> {
        let text = self.text.trim();
        if text.is_empty() && self.sing.is_none() {
            return Err(Error::usage("give text to say, or a tune to sing"));
        }
        if self.voice.as_deref() == Some("?") {
            return Err(Error::usage("call list_voices to see the voices"));
        }
        if let Some(name) = self.sing.as_deref()
            && sing::tune(name).is_none()
        {
            return Err(Error::usage(format!("unknown tune '{name}'")));
        }
        if self.file.as_ref().is_some_and(|f| f.as_os_str() == "-") {
            return Err(Error::usage(
                "file: give a path; stdout carries the protocol",
            ));
        }
        let mut args = Args::default();
        if !text.is_empty() {
            args.text = vec![text.to_string()];
        }
        args.voice = self.voice;
        args.rate = self.rate;
        args.pitch = self.pitch;
        args.sing = self.sing;
        args.output = self.file;
        Ok(args)
    }
}

/// A validated `speak` call waiting for the worker.
struct Job {
    id: Value,
    args: Args,
    stop: Arc<AtomicBool>,
}

/// Stop flags of the `speak` calls that are queued or playing, by request id.
#[derive(Clone, Default)]
struct Pending(Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>);

impl Pending {
    fn lock(&self) -> MutexGuard<'_, HashMap<String, Arc<AtomicBool>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn add(&self, id: &Value) -> Arc<AtomicBool> {
        self.lock().entry(id.to_string()).or_default().clone()
    }

    fn remove(&self, id: &Value) {
        self.lock().remove(&id.to_string());
    }

    fn stop(&self, id: &Value) {
        if let Some(flag) = self.lock().get(&id.to_string()) {
            flag.store(true, Ordering::SeqCst);
        }
    }

    fn stop_all(&self) {
        for flag in self.lock().values() {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

/// Serves until the client closes stdin, then stops any speech and returns.
pub fn serve() -> Result<()> {
    let pending = Pending::default();
    let (jobs, queue) = mpsc::channel();
    let worker = {
        let pending = pending.clone();
        std::thread::spawn(move || speak_worker(queue, pending))
    };
    let server = Server {
        jobs,
        pending: pending.clone(),
    };
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Message>(&line) {
            Ok(msg) => server.handle(msg),
            Err(e) => Some(error(Value::Null, PARSE_ERROR, &e.to_string())),
        };
        if let Some(reply) = reply {
            send(&reply)?;
        }
    }
    pending.stop_all();
    drop(server);
    worker
        .join()
        .map_err(|_| Error::Synthesis("speech worker panicked".into()))
}

struct Server {
    jobs: mpsc::Sender<Job>,
    pending: Pending,
}

impl Server {
    /// The reply to send now, if any. `speak` replies later, from the worker.
    fn handle(&self, msg: Message) -> Option<Value> {
        let method = msg.method?;
        let Some(id) = msg.id else {
            if method == "notifications/cancelled"
                && let Some(id) = msg.params.get("requestId")
            {
                self.pending.stop(id);
            }
            return None;
        };
        let result = match method.as_str() {
            "initialize" => initialize(&msg.params),
            "ping" => json!({}),
            "tools/list" => json!({ "tools": tools() }),
            "tools/call" => return self.call(id, msg.params),
            _ => {
                return Some(error(
                    id,
                    METHOD_NOT_FOUND,
                    &format!("unknown method {method}"),
                ));
            }
        };
        Some(response(id, result))
    }

    fn call(&self, id: Value, params: Value) -> Option<Value> {
        let call: ToolCall = match serde_json::from_value(params) {
            Ok(call) => call,
            Err(e) => return Some(error(id, INVALID_PARAMS, &e.to_string())),
        };
        let result = match call.name.as_str() {
            "list_voices" => Ok(listing::voices(&backends(), None, &pipeline::data_dirs())),
            "speak" => {
                let Err(e) = self.enqueue(&id, call.arguments) else {
                    return None;
                };
                Err(e)
            }
            other => {
                return Some(error(id, INVALID_PARAMS, &format!("unknown tool {other}")));
            }
        };
        Some(response(id, tool_result(result)))
    }

    /// Validates a `speak` call and queues it. The worker replies once it has been spoken.
    fn enqueue(&self, id: &Value, arguments: Map<String, Value>) -> Result<()> {
        let args = serde_json::from_value::<SpeakArgs>(Value::Object(arguments))
            .map_err(|e| Error::usage(e.to_string()))?
            .into_args()?;
        let stop = self.pending.add(id);
        // The worker ends only after the reader, so the queue is still open.
        let _ = self.jobs.send(Job {
            id: id.clone(),
            args,
            stop,
        });
        Ok(())
    }
}

fn speak_worker(queue: mpsc::Receiver<Job>, pending: Pending) {
    for job in queue {
        let result = if job.stop.load(Ordering::SeqCst) {
            Err(Error::Interrupted)
        } else {
            speak(job.args, job.stop)
        };
        pending.remove(&job.id);
        // A cancelled request gets no response; a failed write means the client is gone.
        if !matches!(result, Err(Error::Interrupted)) {
            let _ = send(&response(job.id, tool_result(result)));
        }
    }
}

fn speak(args: Args, stop: Arc<AtomicBool>) -> Result<String> {
    let file = args.output.clone();
    pipeline::run_until(args, Instant::now(), Some(stop))?;
    Ok(match file {
        Some(path) => {
            let path = std::fs::canonicalize(&path).unwrap_or(path);
            format!("Wrote {}", path.display())
        }
        None => "Spoken.".into(),
    })
}

fn initialize(params: &Value) -> Value {
    let asked = params.get("protocolVersion").and_then(Value::as_str);
    let version = PROTOCOL_VERSIONS
        .iter()
        .find(|v| Some(**v) == asked)
        .unwrap_or(&PROTOCOL_VERSIONS[0]);
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "speak", "version": env!("CARGO_PKG_VERSION") },
        "instructions": "Speaks English aloud through the user's speakers with an offline \
            formant synthesizer. Keep spoken messages short: the user hears them while working.",
    })
}

fn tools() -> Value {
    let tunes: Vec<&str> = sing::TUNES.iter().map(|t| t.name).collect();
    let titles: Vec<String> = sing::TUNES
        .iter()
        .map(|t| format!("{} ({})", t.name, t.title))
        .collect();
    json!([
        {
            "name": "speak",
            "description": "Say text aloud through the user's speakers; returns once it has \
                been spoken. A 1980s-style robot voice, good for short notices such as a \
                finished task or a question that needs the user. With `file`, writes the audio \
                there instead of playing it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What to say, in English. Numbers, dates, money, units \
                            and URLs are read out. May be left out with `sing`: the tune then \
                            sings its own words.",
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice name from list_voices, e.g. klatt-female. \
                            Default: the user's configured voice.",
                    },
                    "rate": {
                        "type": "number",
                        "minimum": 80,
                        "maximum": 500,
                        "description": "Words per minute (default 175).",
                    },
                    "pitch": {
                        "type": "string",
                        "description": "Base pitch in Hz (\"110\") or a shift in semitones \
                            (\"+2st\", \"-3st\").",
                    },
                    "sing": {
                        "type": "string",
                        "enum": tunes,
                        "description": format!(
                            "Sing the text to a built-in tune, one syllable per note: {}.",
                            titles.join(", ")
                        ),
                    },
                    "file": {
                        "type": "string",
                        "description": "Write the audio to this path instead of playing it. \
                            The extension picks the format: .wav, .aiff, .raw.",
                    },
                },
                "additionalProperties": false,
            },
        },
        {
            "name": "list_voices",
            "description": "List the voices `speak` can use: name, locale, backend, whether \
                it is ready, and what it sounds like.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false },
            "annotations": { "readOnlyHint": true },
        },
    ])
}

fn tool_result(result: Result<String>) -> Value {
    let (text, is_error) = match result {
        Ok(text) => (text, false),
        Err(e) => (e.to_string(), true),
    };
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

fn response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// One message per line. The lock keeps the reader's and the worker's lines whole.
fn send(msg: &Value) -> std::io::Result<()> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{msg}")?;
    out.flush()
}
