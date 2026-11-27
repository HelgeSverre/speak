//! `speak` arguments (spec §3).

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone, Default)]
#[command(
    name = "speak",
    about = "Offline text-to-speech with the command-line surface of macOS say",
    version,
    disable_version_flag = true,
    override_usage = "speak [-v voice] [-r rate] [-b backend] [-o outfile] [-f file | string ...] [options]"
)]
pub struct Args {
    /// Text to speak; multiple arguments are joined with spaces
    pub text: Vec<String>,

    /// Read text from this file; - is stdin
    #[arg(short = 'f', long = "input-file", value_name = "file")]
    pub input_file: Option<PathBuf>,

    /// Voice name, backend:voice, or a config alias; '?' lists voices
    #[arg(short = 'v', long, value_name = "voice")]
    pub voice: Option<String>,

    /// Speech rate in words per minute (80–500, default 175)
    #[arg(short = 'r', long, value_name = "wpm")]
    pub rate: Option<f32>,

    /// Write audio to this file instead of playing; - is stdout
    #[arg(short = 'o', long = "output-file", value_name = "file")]
    pub output: Option<PathBuf>,

    /// Container: WAVE, AIFF, raw; '?' lists them. Default: from the extension, else AIFF
    #[arg(long, value_name = "fmt")]
    pub file_format: Option<String>,

    /// Sample format: [BE|LE][I|F][16|24|32][@rate], e.g. LEI16@22050; '?' lists them
    #[arg(long, value_name = "fmt")]
    pub data_format: Option<String>,

    /// Output channels: 1 (default) or 2 (mono duplicated)
    #[arg(long, value_name = "n")]
    pub channels: Option<u16>,

    /// Output device by name (substring) or index; '?' lists devices
    #[arg(short = 'a', long = "audio-device", value_name = "device")]
    pub audio_device: Option<String>,

    /// Progress line on stderr
    #[arg(long)]
    pub progress: bool,

    /// Config file path
    #[arg(long, value_name = "path")]
    pub config: Option<PathBuf>,

    /// Not supported (say compatibility)
    #[arg(
        short = 'n',
        long = "network-send",
        value_name = "name:port",
        hide = true
    )]
    pub network_send: Option<String>,

    /// Not supported (say compatibility)
    #[arg(long, value_name = "rate", hide = true)]
    pub bit_rate: Option<String>,

    /// Not supported (say compatibility)
    #[arg(long, value_name = "q", hide = true)]
    pub quality: Option<String>,

    /// Synthesis backend: klatt (default), diphone
    #[arg(short = 'b', long, value_name = "name")]
    pub backend: Option<String>,

    /// Table of backends
    #[arg(long)]
    pub list_backends: bool,

    /// List voices (same as -v '?'); combine with -b to filter
    #[arg(long)]
    pub list_voices: bool,

    /// Input is phonemes, not text: arpabet (default) or pho
    #[arg(long, value_name = "fmt", num_args = 0..=1, require_equals = true, default_missing_value = "arpabet")]
    pub phonemes: Option<String>,

    /// Print an intermediate representation and exit: text, chunks, phonemes, pho, frames, units
    #[arg(long, value_name = "stage")]
    pub dump: Option<String>,

    /// Base pitch in Hz (110) or a shift in semitones (+2st, -3st)
    #[arg(long, value_name = "value", allow_hyphen_values = true)]
    pub pitch: Option<String>,

    /// Linear output gain, 0.0–2.0
    #[arg(long, value_name = "x")]
    pub volume: Option<f32>,

    /// Sing the text to a built-in tune (daisy, twinkle, ode; '?' lists them). Without text: the tune's own words
    #[arg(long, value_name = "tune", num_args = 0..=1, require_equals = true, default_missing_value = "daisy")]
    pub sing: Option<String>,

    /// Sing the text to these notes, one syllable each: "C4 D4:2 F#3:0.5 R" (note:beats)
    #[arg(long, value_name = "notes")]
    pub melody: Option<String>,

    /// Tempo for --sing / --melody in beats per minute
    #[arg(long, value_name = "bpm")]
    pub tempo: Option<f32>,

    /// Transpose the tune by semitones (it is first moved by whole octaves into the voice's range)
    #[arg(long, value_name = "semitones", allow_hyphen_values = true)]
    pub key: Option<f32>,

    /// Stage parameter key=value; repeatable
    #[arg(short = 'P', value_name = "key=value")]
    pub params: Vec<String>,

    /// Analysis/resynthesis of a WAV file instead of text; -r and --pitch modify the result
    #[arg(long, value_name = "file.wav")]
    pub resynth: Option<PathBuf>,

    /// Vocoder for --resynth: lpc (default), psola, griffinlim
    #[arg(long, value_name = "name")]
    pub vocoder: Option<String>,

    /// Per-stage timing on stderr after the run: text (default) or json
    #[arg(long, value_name = "fmt", num_args = 0..=1, require_equals = true, default_missing_value = "text")]
    pub timing: Option<String>,

    /// Synthesize everything before playing
    #[arg(long)]
    pub no_stream: bool,

    /// Print the text and highlight each word as it is spoken
    #[arg(short = 'i', long)]
    pub interactive: bool,

    /// Synthesize the input N times (default 5) to a null sink after one warm-up; print TTFA and RTF
    #[arg(long, value_name = "N", num_args = 0..=1, require_equals = true, default_missing_value = "5")]
    pub bench: Option<usize>,

    /// Do not drive a connected SidePulse LED strip
    #[arg(long)]
    pub no_lights: bool,

    /// Debug log on stderr
    #[arg(short = 'V', long)]
    pub verbose: bool,

    /// Serve the Model Context Protocol on stdin/stdout so coding agents can speak.
    /// Defaults come from the config file and SPEAK_VOICE / SPEAK_BACKEND
    #[cfg(feature = "mcp")]
    #[arg(long, exclusive = true)]
    pub mcp: bool,

    /// Print the version
    #[arg(long, action = clap::ArgAction::Version)]
    version: Option<bool>,
}
