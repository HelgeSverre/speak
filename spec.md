# `speak` — specification

Status: v2 design, 2026-11-20. Research basis: `docs/research/README.md` (decisions) and reports A–D in `docs/research/`. Citations look like `A §2.1.4`.
Implementation order: `PLAN.md`.

## 1. Summary

`speak` is a personal, offline text-to-speech CLI in Rust with the command-line surface of macOS `say`. **Every synthesis stage is our own code.** No pretrained models, no inference runtime, no third-party phonemizer. The point is to build the techniques ourselves and compare them on the same input.

| Backend | Technique | Voice data | Role |
|---|---|---|---|
| `klatt` | Cascade/parallel formant synthesizer driven by rules | none (tables in source) | the main product |
| `diphone` | Diphone concatenation with LPC source-filter resynthesis | a diphone inventory we build: bootstrapped from our own Klatt output, or from recordings we make | second technique for comparison |
| `neural` (stretch) | Small acoustic model we train from scratch in Rust on public-domain data | weights we produce | optional, last milestone |

What we use that is not ours: general-purpose libraries (argument parsing, audio I/O, FFT, resampling) and, optionally, CMUdict as a pronunciation *data* file.

Target platforms: macOS on Apple Silicon (primary), Linux x86-64/aarch64 (secondary).

## 2. Goals and non-goals

### Goals

1. `speak "text"` plays audio; `speak -o out.wav "text"` writes a file. Flags `-v -r -o -f -a` behave like `say`.
2. Works with no network and no data files (Klatt backend with built-in rules).
3. Every pipeline stage is a trait with at least two of our own implementations where that is instructive: two letter-to-sound methods (hand rules, a decision tree we train), two synthesis backends, three glottal sources, three vocoders for resynthesis.
4. Time to first audio (TTFA, process start → first sample handed to the audio device) ≤ 100 ms and RTF (synthesis time ÷ audio duration) ≤ 0.02 for both backends on the M2 Max.
5. Every stage is inspectable: `--dump` prints the intermediate representation, `--timing` prints per-stage times, `speak-lab` analyses audio.
6. Intelligibility is measured, with macOS `say` as the baseline (§16.5).

### Non-goals

- Running anyone else's model: no Piper, Kokoro, Vocos or other pretrained weights; no ONNX Runtime, candle or burn in the default build; no espeak-ng, misaki or other third-party G2P.
- Matching neural TTS quality. The ceiling is DECtalk/eSpeak-class speech (A §2.1.8).
- Languages other than English. SSML. A daemon. Compressed output formats. `say -n`.

## 3. Command-line interface

Two binaries:

- `speak` — the `say` replacement. No subcommands.
- `speak-lab` — the workshop: builds voices and lexicons, trains the letter-to-sound tree, analyses audio (§3.6).

```
speak [-v voice] [-r rate] [-b backend] [-o outfile [--file-format=fmt] [--data-format=fmt]]
      [-a device] [-f file | string ...] [options]
```

Long options take `--opt=value` or `--opt value`.

### 3.1 Flags shared with `say`

| Flag | Behaviour |
|---|---|
| `string ...` | Text to speak. Multiple arguments are joined with single spaces. |
| `-f file`, `--input-file=file` | Read text from `file`; `-` is stdin. Giving both `-f` and positional text is a usage error. |
| (no text, no `-f`) | Read stdin. If stdin is a TTY: speak each line as it is entered until EOF. If not: read to EOF, then speak. |
| `-v voice`, `--voice=voice` | Voice name, `backend:voice`, or a config alias. `-v '?'` lists voices and exits. |
| `-r rate`, `--rate=rate` | Speech rate in words per minute. Default 175. Clamped to 80–500. |
| `-o file`, `--output-file=file` | Write audio to `file` instead of playing. `-` is stdout. |
| `--file-format=fmt` | `WAVE`, `AIFF`, `raw`. `?` lists them. Default: from the extension (`.wav` → WAVE, `.aiff`/`.aif` → AIFF, `.raw`/`.pcm` → raw); no or unknown extension → AIFF (as `say`); `-o -` → WAVE with unknown-length header. |
| `--data-format=fmt` | Byte order `BE` or `LE`, type `I` or `F`, bits `16`, `24` or `32`, optional `@rate`; e.g. `LEI16@22050`, `LEF32`. `?` lists valid combinations. Defaults: WAVE `LEI16`, AIFF `BEI16`, raw `LEI16`, rate 16000. WAVE accepts `LEI16 LEI24 LEI32 LEF32`; AIFF accepts `BEI16 BEI24 BEI32`; raw accepts all. |
| `--channels=n` | 1 (default) or 2 (mono duplicated). |
| `-a device`, `--audio-device=device` | Output device by name (case-insensitive substring) or index. `?` lists devices. |
| `--progress` | Progress line on stderr: chunks done / total, audio seconds, RTF. |
| `-i`, `--interactive` | Print the text and highlight each word as it is spoken. |

Not supported; exit 2 with `speak: <flag> is not supported`: `-n/--network-send`, `--bit-rate`, `--quality`, compressed `--data-format` values (`aac`, `alac`).

Deliberate deviation: with `-o` and TTY stdin, `say` keeps only the last line; `speak` reads to EOF and writes everything.

### 3.2 `speak`-specific flags

| Flag | Behaviour |
|---|---|
| `-b name`, `--backend=name` | `klatt` (default), `diphone`. (`neural` only in a build with the stretch feature.) |
| `--list-backends` | Table of backends: id, technique, sample rate, number of voices. Exit 0. |
| `--list-voices` | Same as `-v '?'`. Combine with `-b` to filter. |
| `--phonemes[=fmt]` | Input is phonemes, not text. `fmt`: `arpabet` (default, §6.2) or `pho` (timed, §6.4). |
| `--dump=stage` | Print an intermediate representation to stdout and exit without audio. Stages: `text` (normalized), `chunks`, `phonemes`, `pho` (timed phones), `frames` (Klatt parameter frames or LPC frames, TSV), `units` (diphone backend: selected diphones). |
| `--pitch=value` | `110` (base F0 in Hz) or `+2st` / `-3st` (shift). |
| `--volume=x` | Linear output gain, 0.0–2.0, default 1.0. |
| `-P key=value` | Stage parameter, repeatable. Keys in §7 and §8 (`lts`, `source`, `oq`, `tl`, `excitation`, …). Unknown key → exit 2. |
| `--resynth=file.wav` | Analysis/resynthesis: WAV → analysis → vocoder → output. No text input. `-r` and `--pitch` modify the result where the vocoder allows (§9). |
| `--vocoder=name` | For `--resynth`: `lpc` (default), `psola`, `griffinlim`. |
| `--timing[=text\|json]` | Per-stage timing on stderr after the run (§11.3). |
| `--bench[=N]` | Synthesize the input `N` times (default 5) to a null sink after one warm-up; print median and min/max TTFA and RTF. No audio. |
| `--no-stream` | Synthesize everything before playing. |
| `--config=path` | Config file path. |
| `-V`, `--verbose` | Debug log on stderr (which lexicon or rule produced each word, dropped symbols, underruns). |
| `--version`, `-h`/`--help` | |

### 3.3 Voice and backend resolution

1. `-v backend:voice` → that backend and voice.
2. `-v name` → config alias (`[voices.name]`), else the unique backend that has a voice with that name. Ambiguous → exit 2 listing candidates.
3. `-b name` without `-v` → that backend's default voice from config.
4. Neither: config `backend` (default `klatt`).

A diphone voice file that is missing: exit 3 with the exact `speak-lab` command that builds it.

### 3.4 Listings

`-v '?'`:

```
klatt-male        en_US  klatt    built-in   # Formant synthesizer, male
klatt-female      en_US  klatt    built-in   # Formant synthesizer, female
boot              en_US  diphone  installed  # LPC diphones analysed from klatt-male
helge             en_US  diphone  installed  # LPC diphones, recorded 2026-12-01
```

Columns: name, locale, backend, state (`built-in`, `installed`, `missing`), `#` description (from the voice file header). Stable order: backend, then name.

### 3.5 Exit codes and streams

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Runtime failure (synthesis, I/O, audio stream) |
| 2 | Usage error |
| 3 | Voice or data file missing or invalid |
| 4 | No usable audio device |
| 130 | Interrupted (SIGINT): stop audio within 50 ms, finalize any output file header |

Audio or `--dump` data on stdout only. Diagnostics, progress and timing on stderr, prefixed `speak: `.

### 3.6 `speak-lab`

```
speak-lab lexicon build <cmudict.dict> [--out FILE]      # compile CMUdict into our lookup format (§7.4)
speak-lab lts train <cmudict.dict> [--out FILE]          # train the letter-to-sound decision tree (§7.5)
speak-lab lts eval  <cmudict.dict> [--method rules|cart] # held-out accuracy
speak-lab diphones build --bootstrap [--name boot]       # analyse Klatt renderings into a diphone voice (§8.2.4)
speak-lab diphones prompts --out DIR                     # prompt list + guide audio for recording (§8.2.6)
speak-lab diphones record --dir DIR [--from ID]          # record prompts interactively
speak-lab diphones build --dir DIR --name NAME           # autolabel recordings and build a voice
speak-lab diphones info <voice>                          # inventory, coverage, sizes
speak-lab diphones simulate --dir DIR --voice V ...     # test helper: synthetic "recordings" with known boundaries (§8.2.6)
speak-lab analyze <file.wav> [--f0] [--formants] [--lpc] # TSV tracks, one row per 5 ms
speak-lab spectrogram <file.wav> <out.pgm>               # greyscale image, hand-written PGM
speak-lab wer <reference.txt> <hypothesis.txt>           # word error rate after our normalization
```

Exit codes as §3.5.

## 4. Architecture

### 4.1 Pipeline

```
text ─► Normalizer ─► Chunker ─► G2P ─► Postlexical ─► Prosody ─► Synthesizer ─────────────► AudioSink
        (words)      (chunks)   lexicon   (flaps,       (durations,  ├ klatt:   rules → frames → formant filters
                                + LTS     reductions)    F0)         └ diphone: unit lookup → LPC frames → lattice filter
```

| Stage | Trait | Our implementations |
|---|---|---|
| Normalization | `Normalizer` | `EnglishNormalizer` |
| Chunking | (function) | `chunk()` |
| G2P | `G2p`, `LetterToSound` | `LexiconG2p` over `RuleLts` (hand rules) or `CartLts` (tree we train) |
| Postlexical rules | (function) | `postlexical()` |
| Prosody | `Prosody` | `RuleProsody` (Klatt durations + tune-based F0) |
| Synthesis | `Backend`, `Synthesizer` | `Klatt`, `Diphone` |
| Excitation | `GlottalSource` | `Klglott88`, `Lf`, `Impulse` |
| Analysis / vocoder | `Analyzer`, `Vocoder` | `LpcVocoder`, `Psola`, `GriffinLim` |
| Output | `AudioSink` | `Playback` (cpal), `FileSink` (WAVE/AIFF/raw), `NullSink` |

Both backends consume the same timed phone stream, so they can be compared on identical phonemes, durations and F0.

### 4.2 Traits (`speak-core`)

Signatures are normative in shape; adjust lifetimes and generics as needed.

```rust
pub trait Normalizer: Send {
    fn normalize(&self, text: &str) -> Vec<Token>;
}

pub trait LetterToSound: Send {
    /// Always returns at least one phone for a word with at least one letter.
    fn pronounce(&self, word: &str) -> Vec<Segment>;
}

pub trait G2p: Send {
    /// Fills `chunk.units`. Must never drop a word.
    fn phonemize(&self, chunk: &mut Chunk) -> Result<()>;
}

pub trait Prosody: Send {
    /// Sets `dur_ms` and `f0` on every segment and pause.
    fn apply(&self, chunk: &mut Chunk, voice: &ProsodyParams);
}

pub trait Backend {
    fn id(&self) -> &'static str;
    fn voices(&self, data: &DataDirs) -> Vec<VoiceInfo>;
    fn open(&self, voice: &VoiceSpec, opts: &SynthOptions, data: &DataDirs)
        -> Result<Box<dyn Synthesizer>>;
}

pub trait Synthesizer: Send {
    fn spec(&self) -> SynthSpec;      // sample_rate, prosody params of the voice
    /// Writes mono f32 PCM. May call `out.write` many times.
    fn synthesize(&mut self, chunk: &Chunk, out: &mut dyn PcmWriter) -> Result<SynthStats>;
}

pub trait PcmWriter {
    /// `Flow::Stop` when the consumer is gone or the run was interrupted.
    fn write(&mut self, samples: &[f32]) -> Result<Flow>;
}

pub trait GlottalSource: Send {
    fn reset(&mut self);
    /// One sample of glottal flow derivative at the source's sample rate.
    fn next(&mut self, f0: f32, q: &VoiceQuality) -> f32;
    /// True on the sample where a new period starts (pitch mark).
    fn epoch(&self) -> bool;
}

pub trait Analyzer: Send {
    fn analyze(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Features>;
}

pub trait Vocoder: Send {
    /// `mods`: time-scale and pitch-scale factors; a vocoder may reject what it cannot do.
    fn decode(&mut self, features: &Features, mods: &Mods, out: &mut dyn PcmWriter) -> Result<()>;
}

pub trait AudioSink: PcmWriter {
    fn begin(&mut self, sample_rate: u32) -> Result<()>;
    /// Blocks until all audio has been played or written.
    fn finish(&mut self) -> Result<SinkStats>;
}
```

### 4.3 Threads

std threads and bounded `std::sync::mpsc::sync_channel`s. No async runtime.

```
main ──┬─ frontend thread: normalize → chunk → G2P → postlexical → prosody ─┐
       ├─ voice open (diphone: read the voice file)                          ├─► synth thread ─► resampler ─► ring buffer ─► cpal callback
       └─ audio open: device query, stream build                            ─┘                                (rtrb, 2 s)
```

- The three startup tasks run concurrently right after argument parsing.
- Channel `frontend → synth`: capacity 8 chunks.
- The cpal callback only reads from the ring buffer and writes zeros on underrun (counted). No locks, allocation or I/O in the callback.
- SIGINT sets an `AtomicBool`; `PcmWriter::write` returns `Flow::Stop`; the sink drops queued audio.

## 5. Crate layout

One crate, `speak`, with one module per stage. No workspace. Names such as `speak-dsp` or `speak-text` elsewhere in this document and in `PLAN.md` mean the module of that name (`src/dsp/`, `src/text/`); `speak-core` means the shared modules at the crate root; `crates/<name>/tests/x.rs` means `tests/<module>_x.rs` or unit tests inside the module.

```
speak/
├── Cargo.toml                  # single package, two binaries, lints
├── deny.toml                   # licence allowlist (§14)
├── NOTICE
├── spec.md  PLAN.md
├── docs/research/
├── src/
│   ├── lib.rs
│   ├── phone.rs chunk.rs traits.rs error.rs timing.rs pho.rs     # shared types ("speak-core")
│   ├── dsp/                    # resonators, FIR, windows, LPC, pitch detection, STFT/iSTFT, mel, MFCC, DTW,
│   │                           # Griffin-Lim, PSOLA, glottal sources. deps: realfft
│   ├── text/                   # normalizer, numbers, embedded commands, chunker, lexicon, rule LTS, CART LTS,
│   │                           # postlexical rules, prosody rules; data files in src/text/data/
│   ├── klatt/                  # formant renderer, rule compiler, target tables, voice presets
│   ├── diphone/                # voice file format, unit lookup and joining, LPC synthesis, voice builder, autolabeller
│   ├── audio/                  # cpal playback and capture, ring buffer, resampling, WAVE/AIFF/raw. deps: cpal, rtrb, rubato, hound
│   ├── cli/                    # argument parsing, pipeline, config, listings. deps: clap, toml, dirs, ctrlc
│   ├── neural/                 # STRETCH (§17). Behind the cargo feature `neural`, off by default. deps: burn
│   └── bin/speak.rs  bin/speak-lab.rs
├── data/                       # small generated artefacts that are committed (§12)
├── tests/                      # integration tests; tests/fixtures/
└── scripts/                    # fetch-cmudict.sh, bench-vs-say.sh, check-deps.sh
```

Allowed third-party crates (libraries, not models): `clap 4.6`, `cpal 0.18`, `rtrb`, `rubato 5.1`, `hound 3.5`, `realfft 3.5` (+ `rustfft`), `serde 1`, `toml`, `dirs 7`, `thiserror 2`, `ctrlc`. Feature `neural` only: `burn`. Anything else needs a reason written in the milestone summary. Not allowed: `ort`, `tract`, `candle` in the default build, `misaki-rs`, `espeak-*`, `num2words` (number expansion is ours), any crate that embeds a lexicon or weights. Commit `Cargo.lock`.

## 6. Data model (`speak-core`)

### 6.1 Phones

ARPAbet, the CMUdict inventory plus the reduced vowels and allophones our rules need.

| Phone | IPA | Example | Class | | Phone | IPA | Example | Class |
|---|---|---|---|---|---|---|---|---|
| `IY` | i | b**ea**t | vowel | | `P` `B` | p b | | stop, labial |
| `IH` | ɪ | b**i**t | vowel | | `T` `D` | t d | | stop, alveolar |
| `EY` | eɪ | b**ai**t | diphthong | | `K` `G` | k ɡ | | stop, velar |
| `EH` | ɛ | b**e**t | vowel | | `CH` `JH` | tʃ dʒ | | affricate |
| `AE` | æ | b**a**t | vowel | | `F` `V` | f v | | fricative |
| `AA` | ɑ | f**a**ther | vowel | | `TH` `DH` | θ ð | | fricative |
| `AO` | ɔ | b**ough**t | vowel | | `S` `Z` | s z | | fricative |
| `AH` | ʌ | b**u**t | vowel | | `SH` `ZH` | ʃ ʒ | | fricative |
| `AX` | ə | **a**bout | reduced vowel | | `HH` | h | | aspirate |
| `IX` | ɨ | ros**e**s | reduced vowel | | `M` `N` `NG` | m n ŋ | | nasal |
| `OW` | oʊ | b**oa**t | diphthong | | `L` `R` | l ɹ | | liquid |
| `UH` | ʊ | b**oo**k | vowel | | `W` `Y` | w j | | glide |
| `UW` | u | b**oo**t | vowel | | `DX` | ɾ | bu**tt**er | flap |
| `ER` | ɝ | b**ir**d | r-coloured vowel | | `Q` | ʔ | uh**-**oh | glottal stop |
| `AXR` | ɚ | butt**er** | reduced r-coloured vowel | | `_` | | | silence |
| `AY` | aɪ | b**i**te | diphthong | | | | | |
| `AW` | aʊ | b**ou**t | diphthong | | | | | |
| `OY` | ɔɪ | b**oy** | diphthong | | | | | |

44 phones plus silence. Vowels carry stress `0` (none), `1` (primary), `2` (secondary).

CMUdict entries map on load: `AH0 → AX`, `ER0 → AXR`, `IH0 → IX`; everything else unchanged. `DX` and `Q` arise only from postlexical rules (§7.6) or phoneme input.

### 6.2 Phoneme text format

`--dump phonemes` output and `--phonemes` input. Whitespace-separated tokens: ARPAbet phones (vowels with an optional stress digit, default 0), `/` for a word break, punctuation `. , ? ! ; :` as their own tokens.

```
HH AX L OW1 / W ER1 L D .
```

`parse(print(x)) == x` is a property test.

### 6.3 Chunk

```rust
pub struct Chunk {
    pub index: usize,
    pub text: String,                 // normalized text of this chunk
    pub words: Vec<Word>,             // text, byte span in the original input, source: User|Builtin|Cmudict|Cart|Rules|Spelled
    pub units: Vec<Unit>,
    pub terminator: Terminator,       // Period | Question | Exclaim | Comma | Semicolon | Colon | Dash | Paragraph | None
    pub pause_after_ms: f32,
    pub controls: Controls,
}

pub enum Unit {
    Seg(Segment),
    WordBreak,
    Punct(char),
    Silence { ms: f32 },
}

pub struct Segment {
    pub phone: Phone,
    pub stress: Stress,               // None | Secondary | Primary
    pub word: u32,                    // index into `words`
    pub dur_ms: f32,                  // 0.0 until Prosody runs
    pub f0: Vec<(f32, f32)>,          // (position 0..=1 within the segment, Hz); empty until Prosody runs
}

pub struct Controls {
    pub rate: f32,                    // 1.0 = 175 wpm
    pub pitch_base_hz: Option<f32>,
    pub pitch_shift_st: f32,
    pub volume: f32,
}
```

### 6.4 `.pho` text format

`--dump pho` output and `--phonemes=pho` input. One unit per line, the MBROLA interface idea (A §2.3), and the seam between the frontend and either backend:

```
; chunk 0  "hello world."
HH   62
AX   48   0 118
L    71
OW1  182  0 124  40 131  100 112
_    60
W    66
```

`<phone><stress> <duration ms> [<position %> <F0 Hz>]...`; `_` is silence; `;` starts a comment. Feeding the same `.pho` file to both backends is the standard A/B comparison.

## 7. Text frontend (`speak-text`)

All of it is our code. No external phonemizer, tagger or number library.

### 7.1 Normalizer

Input: raw UTF-8. Output: tokens `Word(String) | Punct(char) | Command(EmbeddedCommand) | PhonemeLiteral(String) | ParagraphBreak`.

All number and symbol expansion happens here, so every G2P sees only words. Words from the input keep their case (the G2P uses it for proper nouns and `I`); words produced by an expansion are lowercase. Steps, in order:

1. Unicode NFKC; map curly quotes and dashes to ASCII equivalents except `—` and `…`; strip control characters.
2. Extract embedded commands (§7.2).
3. Tokenize on whitespace, then split punctuation from words. Keep apostrophes inside words (`don't`).
4. Classify and expand each token (first match wins):

| Class | Example | Output |
|---|---|---|
| Currency | `$3.50`, `€20`, `£1,000`, `$5 million` | `three dollars fifty`, `twenty euros`, `one thousand pounds`, `five million dollars` |
| Time | `5pm`, `17:30`, `9:05 a.m.` | `five p m`, `seventeen thirty`, `nine oh five a m` |
| Date | `2024-12-03`, `12/03/2024`, `Dec 3, 2024` | `december third twenty twenty four` (US month/day order for `/`) |
| Year | 4-digit 1100–2099 not followed by a unit | `nineteen eighty four`, `two thousand five`, `twenty twenty six` |
| Ordinal | `3rd`, `21st` | `third`, `twenty first` |
| Decimal / cardinal | `3.14`, `1,234`, `-7`, `1969` in non-year context | `three point one four`, `one thousand two hundred thirty four`, `minus seven` |
| Percent | `45%` | `forty five percent` |
| Unit | `5km`, `10 kg`, `3ms`, `2GB`, `60Hz` | `five kilometers`, `ten kilograms`, `three milliseconds`, `two gigabytes`, `sixty hertz` (table of ~40 units; singular when the number is 1) |
| Range / fraction | `3-5`, `1/2`, `3/4` | `three to five`, `one half`, `three quarters` |
| Phone-like digit strings | `555-1234`, digits ≥ 7 with separators | digit by digit, pause at separators |
| Abbreviation | `Dr.`, `Mr.`, `Mrs.`, `St.`, `vs.`, `e.g.`, `i.e.`, `etc.`, `No.` | `doctor`, `mister`, `missus`, `street`/`saint` (saint before a capitalized word), `versus`, `for example`, `that is`, `et cetera`, `number` (table of ~60) |
| URL / e-mail | `https://example.com/a`, `a.b@c.io` | `example dot com slash a`, `a dot b at c dot i o` (scheme dropped) |
| Identifier | `camelCase`, `snake_case`, `foo-bar2` | `camel case`, `snake case`, `foo bar two` |
| Acronym | all-caps 2–5 letters not in the lexicon (`FBI`, `GPU`) | letter names; lexicon hits (`NASA`) stay words |
| Symbol | `&`, `+`, `=`, `@`, `#`, `°`, `~`, `/` between words | `and`, `plus`, `equals`, `at`, `number`/`hash`, `degrees`, `approximately`, `slash` |
| Unknown character | emoji, CJK | dropped; counted; reported with `-V` |

Number expansion is our own code (`numbers.rs`: cardinals to 10¹⁵, ordinals, years, decimals); no number library. Sentence-final periods must survive abbreviation expansion (`... at 5 p.m.` ends the sentence).

A number must never be dropped: if no rule matches a digit string, read it digit by digit (the failure sherpa-onnx has, D §2.3).

### 7.2 Embedded commands

`[[cmd arg]]` inline, as in `say` (A §2.10, D §2.5).

| Command | Effect |
|---|---|
| `[[slnc N]]` | Silence of N ms |
| `[[rate N]]` | Rate N wpm from here on |
| `[[volm X]]` | Volume 0.0–1.0 from here on |
| `[[pbas N]]` | Base pitch N Hz from here on. Deviation: Apple's unit differs. |
| `[[inpt PHON]]` … `[[inpt TEXT]]` | Text in between is ARPAbet phonemes (§6.2). Deviation: Apple uses its own phoneme codes. |
| `[[ … ]]` with any other content | Inline ARPAbet phonemes |

Commands split chunks: a `rate`/`volm`/`pbas` change starts a new chunk with updated `Controls`.

### 7.3 Chunker

Splits the token stream into chunks, one sentence per chunk. Both backends stream within a chunk, so chunks exist for prosody (one intonation phrase each) and bounded memory, not for latency.

`const MAX_WORDS: usize = 60;`

Rules:

1. Sentence boundaries: `.` `!` `?` `…` followed by whitespace or end of input, and not part of an abbreviation, decimal or initial. Newline: a blank line is a paragraph break; a single newline ends a sentence only if the line ends without punctuation and the next line starts with a capital, otherwise it is a space.
2. A sentence longer than `MAX_WORDS`: split at the last `; : —` before the cap, else the last `,`, else at the cap.
3. A clause boundary inside a sentence (`, ; : —`) does not split the chunk; it becomes a `Punct` unit and prosody inserts a short pause and a continuation tune there (§7.7).
4. Embedded commands that change controls split the chunk (§7.2).
5. Pause after a chunk, before rate scaling: `. ! ? …` 350 ms, `; : —` 250 ms, `,` 150 ms, paragraph 700 ms, forced split 0 ms. Scale by `1 / controls.rate`.

### 7.4 Lexicons

Lookup order for a word (case-folded; original case kept for the acronym and proper-noun rules):

| # | Source | What | Where |
|---|---|---|---|
| 1 | User lexicon | one `word PH PH ...` per line; overrides everything | `~/.config/speak/lexicon.txt` |
| 2 | Built-in exceptions | ~500 entries written by us: function words, numbers' words, letter names, the most frequent irregular words (`one`, `two`, `of`, `was`, `said`, `does`, `women`, …) | compiled in |
| 3 | CMUdict (optional data) | 135,166 entries, American English | `<data>/lexicon/en-us.lex`, if present |
| 4 | Morphology | strip `-s -es -ed -ing -ly -er -est -'s`, look up the stem in 1–3, re-attach the suffix phones with voicing agreement (`Z`/`S`/`IX Z`, `D`/`T`/`IX D`) | code |
| 5 | Letter-to-sound | §7.5 | code or tree |

CMUdict is **data, not a model**: a hand-curated pronunciation list from Carnegie Mellon University under a BSD-2-Clause-style licence (A §2.8). It is optional. `speak` is complete without it; with it, common-word accuracy goes up. `scripts/fetch-cmudict.sh` downloads the pinned file with `curl` and checks its hash; `speak-lab lexicon build` compiles it. The product has no network code.

| CMUdict pin | Value |
|---|---|
| URL | `https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict` |
| Size | 3,618,488 bytes, 135,166 lines |
| SHA-256 | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` |
| Licence file | same revision, `LICENSE`, SHA-256 `bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a` |

Source format: `word PH PH ...`, variants as `word(2)`, comments after ` #`. Compiled format `en-us.lex`: sorted `word\tphones\n` lines (first variant only, phones already mapped per §6.1), read into memory in one call and binary-searched on the raw bytes; no parse at startup. Budget: ≤ 5 ms to open.

Homographs: a table of ~30 words with two pronunciations and a context rule written by us (`read`: past after `have/has/had/was/were/already/yesterday`-type cues, else present; `record`, `present`, `object`, `project`: verb after `to`/a pronoun/a modal, else noun; `live`, `lead`, `wind`, `tear`, `close`, `use`, `the` before a vowel → `DH IY`). No POS tagger.

### 7.5 Letter-to-sound

Two implementations of `LetterToSound`, selectable with `-P lts=rules|cart`. Before either: a word with no vowel letters or an all-caps word of ≤ 5 letters is spelled letter by letter; a compound whose two halves are both lexicon words (each ≥ 3 letters) is concatenated.

#### `RuleLts` — hand-written context rules (default until the tree exists)

The NRL scheme (Elovitz et al. 1976; A §2.8): an ordered rule list per letter, each rule `left [graphemes] right = phones`; scan the word left to right, apply the first rule whose bracketed text and contexts match, advance past the bracketed text.

Context symbols:

| Symbol | Matches |
|---|---|
| `#` | one or more vowels (`a e i o u y`) |
| `:` | zero or more consonants |
| `^` | one consonant |
| `.` | a voiced consonant (`b d v g j l m n r w z`) |
| `%` | a suffix: `e`, `er`, `es`, `ed`, `ing`, `ely` |
| `+` | a front vowel (`e i y`) |
| `$` | word boundary |
| a literal letter | itself |

Examples of the form (write the full set, about 330 rules; this is the size of the published NRL set):

```
$ [a] $       = AX
  [ar] #      = EH R
^ [as] #      = EY S
  [a] ^+#     = EY
  [ch] ^      = K
  [ch]        = CH
# [ed] $      = D
  [tion]      = SH AX N
  [igh]       = AY
  [qu]        = K W
  [c] +       = S
  [c]         = K
```

The NRL report is a US Navy publication and may be used as a reference for the rule set. Do not take rules from derived source code of unclear licence. Rules live in a text file compiled into the binary (`speak-text/data/lts-rules.txt`), parsed once.

Stress (the NRL rules give none): primary on the first syllable for words of ≤ 2 syllables; otherwise on the antepenult, except penult before `-tion -sion -ic -ity`-type suffixes; prefixes `a- be- de- re- un- in- ex- con- pre-` are skipped. Unstressed `AH → AX`, `ER → AXR`, `IH → IX`.

#### `CartLts` — a decision tree we train (A §2.8, Black, Lenzo & Pagel 1998)

Trained by `speak-lab lts train` from CMUdict. It is our model, produced by our code from a data file.

1. **Alignment.** Each letter maps to zero, one or two phones. Start from a hand-written table of allowable letter→phone pairs (including `_` epsilon and double phones such as `x → K-S`, `u → Y-UW`); find the alignment per word by dynamic programming that maximizes the product of pair probabilities; re-estimate the probabilities and repeat (3–5 rounds). Words that cannot be aligned (~1–2 %) are dropped from training.
2. **Features** for predicting the phone of letter `i`: the letters at `i−4 … i+4` (with `#` padding), plus two word-level features: the number of vowel-letter groups in the word (a syllable estimate) and how many vowel groups follow letter `i`. (Implementation finding: with letters `i−3 … i+3` alone, held-out words correct stayed at 47–50 % because stress depends on syllable position; the two word-level features bring it to 52 %.)
3. **Trees.** One classification tree per letter, grown greedily on information gain with questions "is the feature at offset `k` equal to `c`", plus the class questions "is that letter a vowel letter" and "a consonant letter"; stop when a node has fewer than 3 examples or is pure. Labels carry stress on vowels (`AE1`, `AE0`), so stress is predicted with the phone.
4. **Evaluation.** Hold out every 10th word. Report letters correct and words correct. The paper reports 58 % words correct on held-out CMUdict; the gate is ≥ 50 % words and ≥ 88 % letters.
5. **Artefact.** `data/lts-en.cart`: our binary format (per letter: node array of `{offset: i8, letter: u8, yes: u32, no: u32}` and leaf labels). If it is ≤ 2 MB it is committed and compiled into the binary; raise the stop size until it fits. It is derived from CMUdict, so `NOTICE` carries the CMUdict copyright notice.

`-P lts=cart` is the default once `data/lts-en.cart` exists. `RuleLts` stays as the comparison and as the fallback in builds without the tree.

### 7.6 Postlexical rules

Applied to the phone string of a whole chunk, after lookup:

| Rule | Effect |
|---|---|
| Flapping | `T` or `D` between a vowel (or `R`) and an unstressed vowel, within a word or across a word break → `DX` |
| `the` | `DH AX` → `DH IY0` before a vowel-initial word |
| `a` / `an`, `to` | `a` → `AX` (`EY1` only when alone); `to` → `T AX` before a consonant, `T UW0` before a vowel |
| Glottal onset | chunk-initial stressed vowel gets a preceding `Q` |
| Geminate reduction | identical consonants across a word break → one, lengthened ×1.3 |
| Function-word destressing | primary stress of words in the function-word list → none, vowel kept |
| Syllabic reduction | `AX L`, `AX N`, `AX M` after a consonant in an unstressed syllable keep the schwa but mark it short (duration ×0.6) |

### 7.7 Prosody (`RuleProsody`)

Sets `dur_ms` and `f0` on each segment. Both backends use it.

#### Durations: Klatt's model (A §2.1.7)

```
DUR = MINDUR + (INHDUR − MINDUR) × PRCNT / 100
```

`PRCNT` starts at 100 and is multiplied by each applicable rule. Rule percentages are from Klatt 1979 via secondary sources; treat them as starting values.

| # | Rule | Applies to | Factor |
|---|---|---|---|
| 1 | Pause insertion | chunk boundaries: pauses from §7.3. Clause punctuation inside a chunk: `,` 150 ms, `; : —` 250 ms | |
| 2 | Clause-final lengthening | vowel of the last syllable before a pause, and consonants after it | ×1.4 |
| 3 | Non-phrase-final shortening | vowels not in the last syllable of the chunk | ×0.6 |
| 4 | Non-word-final shortening | vowels not in a word-final syllable | ×0.85 |
| 5 | Polysyllabic shortening | vowels in words of more than one syllable | ×0.8 |
| 6 | Non-initial consonant shortening | consonants not word-initial | ×0.85 |
| 7 | Unstressed shortening | unstressed segments: `MINDUR ← MINDUR / 2`; then vowel word-medial ×0.5, other vowel ×0.7, prevocalic liquid/glide ×0.1, other consonant ×0.7 | |
| 8 | Postvocalic context | vowel followed in the same word by: voiceless stop ×0.7; voiced fricative ×1.6; voiced stop ×1.2; nasal ×0.85; nothing (word-final open syllable) ×1.2. Outside a phrase-final syllable use `0.7 + 0.3 × factor`. | |
| 9 | Cluster shortening | vowel before vowel ×1.2; vowel after vowel ×0.7; consonant between consonants ×0.5; consonant after consonant ×0.7; consonant before consonant ×0.7 | |
| 10 | Aspiration lengthening | stressed vowel or sonorant after a voiceless stop | +25 ms after the formula |

Then apply rate: `DUR ← max(MINDUR/2, DUR / controls.rate)`.

Inherent and minimum durations (ms; Klatt 1979 values as commonly reproduced; verify by ear):

| Phone | INH | MIN | | Phone | INH | MIN | | Phone | INH | MIN |
|---|---|---|---|---|---|---|---|---|---|---|
| `IY` | 155 | 55 | | `P` | 90 | 50 | | `M` | 70 | 60 |
| `IH` | 135 | 40 | | `B` | 85 | 60 | | `N` | 60 | 50 |
| `EY` | 190 | 100 | | `T` | 75 | 50 | | `NG` | 95 | 60 |
| `EH` | 150 | 70 | | `D` | 75 | 50 | | `L` | 80 | 40 |
| `AE` | 230 | 80 | | `K` | 80 | 60 | | `R` | 80 | 30 |
| `AA` | 240 | 100 | | `G` | 80 | 60 | | `W` | 80 | 60 |
| `AO` | 240 | 100 | | `CH` | 70 | 50 | | `Y` | 80 | 40 |
| `AH` | 140 | 60 | | `JH` | 70 | 50 | | `HH` | 80 | 20 |
| `AX` | 120 | 60 | | `F` | 100 | 80 | | `DX` | 20 | 20 |
| `IX` | 110 | 40 | | `V` | 60 | 40 | | `Q` | 30 | 20 |
| `OW` | 220 | 80 | | `TH` | 90 | 60 | | | | |
| `UH` | 160 | 60 | | `DH` | 50 | 30 | | | | |
| `UW` | 210 | 70 | | `S` | 105 | 60 | | | | |
| `ER` | 180 | 80 | | `Z` | 75 | 40 | | | | |
| `AXR` | 130 | 60 | | `SH` | 105 | 80 | | | | |
| `AY` | 250 | 150 | | `ZH` | 70 | 40 | | | | |
| `AW` | 260 | 100 | | | | | | | | |
| `OY` | 280 | 150 | | | | | | | | |

#### F0: declination plus tunes (A §2.1.7, eSpeak's head/nucleus/tail idea)

All values are semitones relative to the voice's base F0 (`ProsodyParams.base_hz`, shifted by `--pitch`), multiplied by `ProsodyParams.range` (default 1.0).

1. Baseline: linear in time from +2 st at chunk start to −3 st at chunk end.
2. Accents: each primary-stressed vowel of a content word adds a peak at 30 % of the vowel: +4 st for the first accent in the chunk, +3 st for later ones; secondary stress +1 st. Function words (closed list of ~70: articles, prepositions, auxiliaries, pronouns, conjunctions) get no accent. The rise starts at the onset of the syllable; the fall ends at the end of the vowel.
3. An intonation phrase is a chunk, or a stretch between clause punctuation inside it. Nucleus = last accented vowel in the phrase. Its shape and the end of the phrase depend on the terminator:

| Terminator | Nucleus peak | End of chunk |
|---|---|---|
| `.` `…` paragraph, none | +3 st | fall to −6 st |
| `?` when the chunk starts with a wh-word (`who what when where why how which`) | +3 st | fall to −6 st |
| `?` otherwise | −2 st (low) | rise to +7 st |
| `!` | +6 st | fall to −6 st |
| `,` `;` `:` `—` | +3 st | fall to −1 st, then rise to +1 st over the last syllable |

4. Microprosody: +5 % F0 at the onset of a vowel after a voiceless obstruent, decaying linearly over 50 ms.
5. Targets are placed on voiced segments as `(position, Hz)` points; the renderer interpolates linearly in time and holds the last value across unvoiced segments.

`klatt-robot` uses a flat contour (steps 1–4 off).

## 8. Backends

### 8.1 Klatt (`speak-klatt`)

Written from Klatt 1980 and Klatt & Klatt 1990 (`docs/research/papers/`), A §2.1. Do not read or port GPL implementations (§14). Two layers: a **rule compiler** (timed phones → parameter frames) and a **renderer** (frames → samples). `--dump frames` exposes the seam.

#### 8.1.1 Constants

| Constant | Value |
|---|---|
| Sample rate `SR` | 16000 Hz |
| Frame period | 5 ms (80 samples) |
| Coefficient update | every 1 ms (16 samples), from linearly interpolated frame parameters |
| Source oversampling | 4× (64 kHz), decimated by a 32-tap Hann-windowed sinc FIR, cutoff 0.45 × SR |
| Cascade formants | 8: F1–F4 variable, F5–F8 fixed per voice |

16 kHz keeps /s/ and /ʃ/ energy that 10 kHz loses. A 17 cm tract has one formant per kHz, so 8 formants below the Nyquist frequency (Klatt 1980 §NFC). F7/F8 are an extension, not in the 1980 paper.

#### 8.1.2 Filters (`speak-dsp`)

Resonator (Klatt 1980 eq. 1–2), `T = 1/SR`:

```
y[n] = A·x[n] + B·y[n−1] + C·y[n−2]
C = −exp(−2π·BW·T)
B =  2·exp(−π·BW·T)·cos(2π·F·T)
A =  1 − B − C
```

Antiresonator (zero) with resonator coefficients A, B, C for the same F, BW:

```
y[n] = A'·x[n] + B'·x[n−1] + C'·x[n−2]      A' = 1/A   B' = −B/A   C' = −C/A
```

A resonator whose F is ≥ 0.47 × SR is bypassed. State is kept across coefficient updates.

#### 8.1.3 Signal graph

Radiation (the +6 dB/octave lip characteristic) is folded into the sources: the voicing source is the glottal flow *derivative* and the noise sources are white. There is no output differentiator.

```
voicing ──► tilt LP ──×AV──┐
                            ├─► RNP ─► RNZ ─► R1 ─► R2 ─► R3 ─► R4 ─► R5 ─► R6 ─► R7 ─► R8 ──┐
noise ──×AH (modulated) ───┘                    (cascade)                                      ├─► ×gain ─► soft limit ─► out
                                                                                               │
noise ──×AF (modulated) ──► parallel: R2p…R6p with A2…A6, bypass AB, alternating signs ───────┘
```

Parallel branch, with `n` the frication noise sample and `Rkp` using the same F and BW as the cascade resonator `k` (Klatt 1980 Fig. 6; alternating signs avoid spectral zeros between formants):

```
p = R6p(A6·n)
p = R5p(A5·n) − p
p = R4p(A4·n) − p
p = R3p(A3·n) − p
p = R2p(A2·n) − p
p = AB·n − p
```

Noise: one xorshift32 generator, uniform in [−1, 1], fixed seed per run (deterministic output). While AV > 0, both noise sources are multiplied by 0.5 during the second half of each glottal period (A §2.1.5).

#### 8.1.4 Voicing sources (`trait GlottalSource`)

Period `T0 = 1/F0` and phase are tracked in fractional samples at the oversampled rate; a new period starts when phase ≥ `T0`. F0 used by the source includes flutter (Klatt & Klatt 1990 eq. 1), `FL` in percent, `t` in seconds:

```
ΔF0 = (FL/50)·(F0/100)·[sin(2π·12.7·t) + sin(2π·7.1·t) + sin(2π·4.7·t)]   Hz
```

| Source | Definition |
|---|---|
| `klglott88` (default) | Open phase `x = phase/(OQ·T0)`, `0 ≤ x < 1`: `s = 2x − 3x²` (derivative of Rosenberg's `a·t² − b·t³` flow, normalized to −1 at closure). Closed phase: `s = 0`. |
| `lf` | Liljencrants-Fant model of the flow derivative (A §2.1.4), parameterized by OQ, speed quotient SQ (default 2.0) and return-phase time `ta = 0.02·T0` scaled by TL. Open phase `E(t) = E0·exp(α·t)·sin(ωg·t)` up to the excitation instant `te`, then the exponential return `−(Ee/(ε·ta))·(exp(−ε·(t−te)) − exp(−ε·(tc−te)))`. Solve ε from `ε·ta = 1 − exp(−ε·(tc−te))` and α from the zero-net-flow condition by Newton iteration (≤ 10 steps) once per period; cache while parameters are unchanged. Normalized to −1 at `te`. |
| `impulse` | Unit impulse at each period start → resonator (F = 0, BW = 100 Hz) → antiresonator (F = 1500, BW = 6000) → first difference. Klatt 1980's buzzy source; used by `klatt-robot`. |

Tilt: one-pole low-pass `y = (1−a)·x + a·y[n−1]` at the oversampled rate, with `a` chosen so the attenuation at 3 kHz is `TL` dB. With `g = 10^(−TL/10)` and `c = cos(2π·3000/(4·SR))`:

```
a = [ (1 − g·c) − sqrt((1 − g·c)² − (1 − g)²) ] / (1 − g)        (a = 0 when TL = 0)
```

All sources implement `GlottalSource` (§4.2) and live in `speak-dsp` so the diphone backend can use them.

#### 8.1.5 Frame parameters

```rust
pub struct Frame {
    pub f0: f32,                         // Hz; 0 = unvoiced
    pub av: f32, pub ah: f32, pub af: f32,   // dB, 0..80; 0 = off
    pub oq: f32,                         // 0.3..0.8
    pub tl: f32,                         // dB, 0..34
    pub f: [f32; 4], pub b: [f32; 3],    // F1–F4 Hz, B1–B3 Hz
    pub fnz: f32,                        // nasal zero Hz; equal to FNP = no nasalization
    pub a: [f32; 5], pub ab: f32,        // A2–A6 and bypass, dB
}
```

Fixed per voice: `FNP 270, BNP 100, BNZ 100, B4 250, F5 3750, B5 200, F6 4900, B6 1000, F7 6500, B7 500, F8 7500, B8 600` (Klatt 1980 Table I for F5/F6/B4–B6 and the nasal bandwidths; F7/F8 are starting values to tune).

Amplitudes: `lin(dB) = 0 if dB ≤ 0 else 10^((dB − 60)/20)`. Each path has one scalar gain constant in `gains.rs` (`K_AV, K_AH, K_AF, K_A2…K_A6, K_AB, K_OUT`), fitted once so the level tests in §16.2 pass. There is no AVS parameter: voice bars and voiced fricatives use AV with a high TL instead.

Output: `K_OUT`, then a soft limiter (identity below |x| = 0.8, smooth saturation to 1.0 above).

#### 8.1.6 Voices and voice quality

| Voice | Base F0 | Formant scale (F1–F8) | OQ | TL | Breath (AH added in vowels) | Source | Flutter |
|---|---|---|---|---|---|---|---|
| `klatt-male` (default) | 110 Hz | 1.00 | 0.50 | 6 | 0 | klglott88 | 25 % |
| `klatt-female` | 200 Hz | 1.17 | 0.60 | 10 | 40 dB | klglott88 | 25 % |
| `klatt-child` | 280 Hz | 1.35 | 0.60 | 8 | 30 dB | klglott88 | 25 % |
| `klatt-breathy` | 110 Hz | 1.00 | 0.70 | 18 | 50 dB | klglott88 | 25 % |
| `klatt-creaky` | 90 Hz | 1.00 | 0.35 | 2 | 0 | klglott88, DI 25 % | 40 % |
| `klatt-robot` | 100 Hz, flat | 1.00 | — | 0 | 0 | impulse | 0 |
| `klatt-whisper` | — | 1.00 | — | — | AV → 0, AH = former AV − 5 | noise only | — |

`-P` keys: `source=klglott88|lf|impulse`, `oq`, `tl`, `flutter`, `di` (diplophonia, %: every second period delayed by `DI/100 × T0/2` and attenuated by `DI/100`, Klatt & Klatt 1990), `formant_scale`, `breath`, `sq` (LF speed quotient).

Utterance-level voice quality rules (Klatt & Klatt 1990): in the last 150 ms before a pause, OQ rises by 0.1, TL by 6 dB and AH by 6 dB (breathy offset); the first two periods after a pause or `Q` use OQ − 0.1 (hard onset).

#### 8.1.7 Rule compiler

Input: `Unit`s with `dur_ms` and `f0`. Output: `Vec<Frame>`.

**Step 1 — sub-segments.** Each phone expands to one or more sub-segments, each with a duration and a full target (formants, bandwidths, source amplitudes, parallel gains):

| Class | Sub-segments |
|---|---|
| Vowel | one; diphthongs glide linearly from the start target (at 25 % of the duration) to the end target (at 85 %). AV 60 with primary stress, 58 secondary, 57 unstressed. |
| Liquid, glide | one; AV 50 |
| Nasal | one; AV 52, FNZ 450, formants from the table. The adjacent 40 ms of each neighbouring vowel is nasalized: FNZ ramps between 450 and 270. |
| Voiceless fricative | one; AV 0, AF 60, parallel gains from the table |
| Voiced fricative | one; AV 47 with TL 15, AF 50 |
| Voiceless stop | closure (silence) for `dur − burst`; burst (AF 60, table gains): `P` 5 ms, `T` 10 ms, `K` 15 ms. Then aspiration **inside the following sonorant**: its first N ms have AV 0, AH 55, B1 250 while its formant transitions already run. N = 45 ms before a stressed vowel in a syllable onset, 15 ms otherwise, 0 after `S`. |
| Voiced stop | closure with a voice bar (AV 40, TL 20, F1 180, B1 60) then burst: `B` 5 ms, `D` 8 ms, `G` 12 ms, with AV 40 |
| Stop before another stop or pause | first stop of a cluster: no burst. Before a pause: burst 6 dB weaker. |
| Affricate | closure for 40 % of the duration, then frication with the table values |
| `HH` | AV 0, AH 55, B1 300; F1–F3 copied from the next vowel (else the previous one) |
| `DX` | AV 50, F1 300, F2 1600, F3 2600; no closure |
| `Q` | AV 0 for the duration; F0 of the neighbouring 30 ms lowered by 15 % |
| Silence | all amplitudes 0; formants hold |

**Step 2 — formant tracks** (Holmes-style ranks with locus-type boundaries, A §2.1.6). For each boundary between adjacent sub-segments X and Y, per formant:

- The higher-ranked one dominates: `boundary = T_dom + k·(T_other − T_dom)`. Equal rank: the midpoint.
- The dominant phone sets the transition times: `t_int` inside itself, `t_ext` inside the neighbour.

| Class | Rank | k for F1 | k for F2, F3 | `t_int` / `t_ext` (ms) |
|---|---|---|---|---|
| Stop (and affricate closure) | 9 | 0.1 | 0.5 | labial 10/40, alveolar 10/50, velar 10/60 |
| Nasal | 8 | 0.1 | 0.5 | 15/40 |
| Fricative | 7 | 0.2 | 0.5 | 20/45 |
| Flap | 6 | 0.2 | 0.5 | 5/30 |
| Liquid | 5 | 0.3 | 0.25 | 35/70 |
| Glide | 4 | 0.3 | 0.25 | 40/80 |
| Vowel | 2 | 0.5 | 0.5 | 40/40 |
| `HH`, `Q`, silence | 0 | (no own targets: takes the neighbour's values, no transition) | | |

The track of a phone of duration `D`, with own target `T(t)` (constant, or the diphthong glide), left boundary value `L` over time `tl` and right boundary value `R` over time `tr`:

```
v(t) = T(t) + (L − T(0))·w(t/tl) + (R − T(D))·w((D − t)/tr)
w(u) = 0.5·(1 + cos(π·u)) for 0 ≤ u < 1, else 0
```

The two terms add, so a phone shorter than its transitions undershoots its target (Klatt 1987 Fig. 19). Bandwidths B1–B3 follow the same scheme. F4 is constant at 3300 × formant scale.

**Step 3 — source tracks.** AV, AH, AF, A2–A6, AB are steps per sub-segment with a linear ramp of 10 ms centred on each boundary; closure → burst is abrupt. In the last 100 ms before a pause AV falls by 6 dB. F0 comes from the segments' `(position, Hz)` points.

**Step 4 — sample every 5 ms** into `Frame`s; apply the voice's formant scale, voice quality rules (§8.1.6) and `Controls.volume`.

#### 8.1.8 Context-dependent rules (the deepening beyond Klatt 1980 Table III)

Klatt's consonant table is valid before front vowels only, and he notes pure locus theory fails for velars (A §2.1.6). These rules are part of the compiler, driven by table columns, not by special cases in code.

Vowel context classes: front (`IY IH EY EH AE IX`), back unrounded (`AA AH AX AY AW`), back rounded (`AO OW UH UW OY`), rhotic (`ER AXR`).

| Rule | Detail |
|---|---|
| Stop and nasal loci by context | Each stop/nasal row has F2 and F3 loci for the three non-rhotic classes. Starting values: labial F2 1100 / 900 / 800; alveolar F2 1800 / 1700 / 1600; velar F2 2300 / 1700 / 1300 with F3 2900 / 2400 / 2100 (velar pinch: F2 and F3 start close together before front vowels). |
| Burst spectrum by context | Velar burst: before back vowels A2 60 and A3–A6 reduced by 15 dB (Klatt 1980, text below Table III). Alveolar and labial bursts unchanged. |
| Fricative coarticulation | `S` and `SH` before rounded vowels: A-gains shifted down one formant (A6 → A5 etc.) to lower the noise centre. |
| Dark and light `L` | `L` before a vowel in the same syllable: table values. Otherwise dark: F2 850, F3 2800. |
| `R` colouring | Vowel before `R` in the same syllable: F3 of the last 40 % glides to 1700. |
| Vowel reduction by duration | A vowel shorter than 60 % of its inherent duration moves its targets 30 % toward the `AX` row. |
| Nasalized vowels | Covered by the nasal row in step 1; additionally B1 of the nasalized span is widened to 100. |
| Aspiration and devoicing | `L R W Y` after a voiceless stop in the same syllable: first half devoiced (AV 0, AH 50). |
| Final lengthening interaction | Phrase-final vowels keep full targets regardless of the reduction rule. |

All starting values in this table are ours (inferred from A §2.1.6 and general acoustic phonetics); tune them with `speak-lab analyze` against the formant tests and the WER gate.

#### 8.1.9 Target tables

Vowels (Klatt 1980 Table II; `a → b` = start and end target of a diphthongized vowel):

| Phone | F1 | F2 | F3 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|
| `IY` | 310 → 290 | 2020 → 2070 | 2960 | 45 → 60 | 200 | 400 |
| `IH` | 400 → 470 | 1800 → 1600 | 2570 → 2600 | 50 | 100 | 140 |
| `EY` | 480 → 330 | 1720 → 2020 | 2520 → 2600 | 70 → 55 | 100 | 200 |
| `EH` | 530 → 620 | 1680 → 1530 | 2500 → 2530 | 60 | 90 | 200 |
| `AE` | 620 → 650 | 1660 → 1490 | 2430 → 2470 | 70 | 150 → 100 | 320 |
| `AA` | 700 | 1220 | 2600 | 130 | 70 | 160 |
| `AO` | 600 → 630 | 990 → 1040 | 2570 → 2600 | 90 | 100 | 80 |
| `AH` | 620 | 1220 | 2550 | 80 | 50 | 140 |
| `OW` | 540 → 450 | 1100 → 900 | 2300 | 80 | 70 | 70 |
| `UH` | 450 → 500 | 1100 → 1180 | 2350 → 2390 | 80 | 100 | 80 |
| `UW` | 350 → 320 | 1250 → 900 | 2200 | 65 | 110 | 140 |
| `ER` `AXR` | 470 → 420 | 1270 → 1310 | 1540 | 100 | 60 | 110 |
| `AY` | 660 → 400 | 1200 → 1880 | 2550 → 2500 | 100 → 70 | 70 → 100 | 200 |
| `AW` | 640 → 420 | 1230 → 940 | 2550 → 2350 | 80 | 70 | 140 → 80 |
| `OY` | 550 → 360 | 960 → 1820 | 2400 → 2450 | 80 → 60 | 50 | 130 → 160 |
| `AX` (ours) | 500 | 1400 | 2500 | 80 | 90 | 150 |
| `IX` (ours) | 420 | 1700 | 2550 | 60 | 100 | 150 |

Sonorant consonants (Klatt 1980 Table III; nasals with FNP 270, FNZ 450):

| Phone | F1 | F2 | F3 | B1 | B2 | B3 |
|---|---|---|---|---|---|---|
| `W` | 290 | 610 | 2150 | 50 | 80 | 60 |
| `Y` | 260 | 2070 | 3020 | 40 | 250 | 500 |
| `R` | 310 | 1060 | 1380 | 70 | 100 | 120 |
| `L` | 310 | 1050 | 2880 | 50 | 100 | 280 |
| `M` | 480 | 1270 | 2130 | 40 | 200 | 200 |
| `N` | 480 | 1340 | 2470 | 40 | 300 | 300 |
| `NG` (ours) | 480 | 2000 | 2900 | 40 | 300 | 300 |

Obstruents (Klatt 1980 Table III; front-vowel context; A-values in dB):

| Phone | F1 | F2 | F3 | B1 | B2 | B3 | A2 | A3 | A4 | A5 | A6 | AB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `F` | 340 | 1100 | 2080 | 200 | 120 | 150 | 0 | 0 | 0 | 0 | 0 | 57 |
| `V` | 220 | 1100 | 2080 | 60 | 90 | 120 | 0 | 0 | 0 | 0 | 0 | 57 |
| `TH` | 320 | 1290 | 2540 | 200 | 90 | 200 | 0 | 0 | 0 | 0 | 28 | 48 |
| `DH` | 270 | 1290 | 2540 | 60 | 80 | 170 | 0 | 0 | 0 | 0 | 28 | 48 |
| `S` | 320 | 1390 | 2530 | 200 | 80 | 200 | 0 | 0 | 0 | 0 | 52 | 0 |
| `Z` | 240 | 1390 | 2530 | 70 | 60 | 180 | 0 | 0 | 0 | 0 | 52 | 0 |
| `SH` | 300 | 1840 | 2750 | 200 | 100 | 300 | 0 | 57 | 48 | 48 | 46 | 0 |
| `ZH` (as `SH`, voiced) | 240 | 1840 | 2750 | 70 | 80 | 270 | 0 | 57 | 48 | 48 | 46 | 0 |
| `CH` | 350 | 1800 | 2820 | 200 | 90 | 300 | 0 | 44 | 60 | 53 | 53 | 0 |
| `JH` | 260 | 1800 | 2820 | 60 | 80 | 270 | 0 | 44 | 60 | 53 | 53 | 0 |
| `P` | 400 | 1100 | 2150 | 300 | 150 | 220 | 0 | 0 | 0 | 0 | 0 | 63 |
| `B` | 200 | 1100 | 2150 | 60 | 110 | 130 | 0 | 0 | 0 | 0 | 0 | 63 |
| `T` | 400 | 1600 | 2600 | 300 | 120 | 250 | 0 | 30 | 45 | 57 | 63 | 0 |
| `D` | 200 | 1600 | 2600 | 60 | 100 | 170 | 0 | 47 | 60 | 62 | 60 | 0 |
| `K` | 300 | 1990 | 2850 | 250 | 160 | 330 | 0 | 53 | 43 | 45 | 45 | 0 |
| `G` | 200 | 1990 | 2850 | 60 | 150 | 280 | 0 | 53 | 43 | 45 | 45 | 0 |

All tables live in one data file (`speak-klatt/src/tables.rs`), one row per phone, so tuning never touches logic.

### 8.2 Diphone LPC (`speak-diphone`)

#### 8.2.1 Why this technique

The second backend should teach the most about the other half of classic synthesis (measured filters instead of ruled ones, and concatenation) for the least effort. Options from report A:

| Option | Voice data needed | New concepts | Effort | Verdict |
|---|---|---|---|---|
| LPC analysis/resynthesis only (A §2.2) | none | linear prediction | low | not a TTS backend by itself |
| Diphones + TD-PSOLA (A §2.3) | recordings with reliable pitch marks | concatenation, PSOLA | medium; pitch-marking recordings is the fragile part | later, as an excitation mode |
| **Diphones + LPC** (A §2.2, §2.3; what Festival's `kal` voices do with residual-excited LPC) | diphone inventory | linear prediction, lattice filters, unit inventory, parametric joins, excitation models | medium; no pitch marks needed for the basic mode | **chosen** |
| Unit selection (A §2.4) | hours of labelled speech | cost functions, Viterbi | high | no |
| HMM/statistical parametric (A §2.5) | corpus + full-context labeller + training | too many | high | no |

Diphone LPC wins because: the filter is *measured* from audio, which is the conceptual opposite of Klatt's ruled filter; joins are smooth because we interpolate filter parameters instead of splicing waveforms; pitch and duration are free (the excitation is ours), so the prosody stage is shared with Klatt unchanged; and the inventory can be **bootstrapped from our own Klatt output**, which makes the whole backend buildable and testable without a microphone. Recording our own voice is then a data swap, not new synthesis code.

#### 8.2.2 LPC analysis (`speak-dsp::lpc`)

| Parameter | Value |
|---|---|
| Sample rate | 16000 Hz |
| Pre-emphasis | `y[n] = x[n] − 0.97·x[n−1]` |
| Frame hop / window | 5 ms / 25 ms Hamming |
| Order `p` | 18 |
| Method | autocorrelation + Levinson-Durbin → reflection coefficients `k[1..p]`, prediction error energy `E` |
| Gain | `G = sqrt(E / window energy normalization)`: RMS of the residual |
| F0 and voicing | normalized autocorrelation of the 900 Hz low-passed frame (40 ms window) over lags for 60–400 Hz; `f0` = lag of the peak (parabolic interpolation); `voicing` = peak value clamped to 0–1; median filter of 5 frames on `f0` |

Frame record: `k: [f32; 18], gain: f32, f0: f32, voicing: f32`. Reflection coefficients are stored because any interpolation between two stable sets is stable (|k| < 1).

#### 8.2.3 LPC synthesis

Per output sample:

1. Interpolate `k`, `gain`, `voicing` linearly between frames (update every 1 ms).
2. Excitation `e = voicing·v + (1 − voicing)·u`, with `u` white noise of unit RMS and `v` the voiced source at the target F0 scaled to unit RMS per period (pulse amplitude grows with `sqrt(T0)`), then `× gain`. `voicing` below 0.3 is treated as 0, above 0.7 as 1.
3. All-pole lattice filter with the reflection coefficients.
4. De-emphasis `y[n] = x[n] + 0.97·y[n−1]`; `Controls.volume`; soft limiter.

Excitation modes (`-P excitation=`):

| Mode | Source | Character |
|---|---|---|
| `pulse` | one impulse per period | classic LPC buzz (Speak & Spell) |
| `glottal` (default) | `Klglott88` from §8.1.4, shared code | softer |
| `residual` | the stored LPC residual, re-pitched and re-timed with PSOLA on its pitch marks (§9.3) | closest to the source audio; needs a voice file with residuals |

#### 8.2.4 Diphone inventory and bootstrap voice

A diphone `A-B` runs from the midpoint of phone A to the midpoint of phone B (A §2.3). Inventory: all ordered pairs over the 44 phones plus silence, 45 × 45 = 2025, minus pairs listed as impossible in `inventory.rs` (e.g. `NG` after silence).

Bootstrap voice `boot`, built by `speak-lab diphones build --bootstrap`:

1. For each diphone, compile and render the phone sequence `_ A B _` with `klatt-male`, inherent durations, flat F0 110 Hz, flutter 0.
2. LPC-analyse the rendering (§8.2.2).
3. Cut the frames from mid-A to mid-B; the timings are known exactly from the compiler. Record the frame index of the A/B boundary.
4. Keep the residual and the glottal epochs (exact, from `GlottalSource::epoch`) for the `residual` mode.

The result sounds like LPC-coded Klatt. That is intended: it proves the analysis, storage, lookup, joining and synthesis code against a source with known ground truth (exact boundaries, exact F0, exact formants).

Voice file `<data>/voices/<name>.sdv` (our format, little-endian):

```
magic "SDV1" | u32 sample_rate | u16 lpc_order | u16 frame_us | u32 n_diphones | u32 flags (bit 0: has residual)
256-byte UTF-8 description: name, speaker, date, source ("klatt-male" or "recorded"), licence
index: n × { u8 phone_a, u8 phone_b, u16 n_frames, u16 boundary_frame, u32 frame_offset, u32 residual_offset, u32 n_epochs }
frames: f32 × (lpc_order + 3) per frame
residuals: i16 samples, then u32 epoch positions, per diphone
```

Expected size without residuals: about 6 MB. Voice files are built locally and are not committed.

#### 8.2.5 Unit lookup and joining

For a chunk's timed phones `p0 … pn` (silence added at both ends):

1. Each phone `pi` is covered by the second half of diphone `p(i−1)-pi` and the first half of `pi-p(i+1)`.
2. Time mapping: each half is resampled in time (linear interpolation between frames) to half of the target `dur_ms`. For stops and affricates the burst frames (the 20 ms after the boundary) keep their original length and only the closure stretches.
3. Join at each phone midpoint: cross-interpolate `k` and `gain` over ± 3 frames (15 ms).
4. F0 comes from the prosody targets, not from the stored `f0`. `voicing` comes from the store.
5. Missing diphone `A-B`: use `A-_` then `_-B` with a 10 ms gap removed; log under `-V`.

`--dump units` prints the selected diphones with source and target durations.

#### 8.2.6 Our own recorded voice

Workflow in `speak-lab diphones`:

1. `prompts --out DIR`: writes `prompts.tsv` (`id`, phone sequence, target diphone) and one guide WAV per prompt, rendered by Klatt at flat F0 and fixed timing. Prompt phone sequence: `_ T AA A B AA _` (carrier syllables around the target pair; silence-adjacent diphones use `_ A AA _` and `_ AA B _`). About 2,000 prompts of ~1.5 s: roughly 90 minutes of recording. A partial set is usable (rule 5 above covers gaps).
2. `record --dir DIR`: for each prompt, play the guide, then record from the default input (cpal) for the guide's length + 0.5 s; Enter accepts, `r` repeats, `q` saves progress. The speaker imitates the guide's timing and holds a monotone pitch (A §2.3: diphones are recorded once in monotone nonsense words).
3. `build --dir DIR --name NAME`: per prompt,
   - compute 13 MFCCs + deltas at a 5 ms hop for guide and recording (§9.1);
   - align them with dynamic time warping (Euclidean local cost, symmetric steps, Sakoe-Chiba band of ± 300 ms);
   - map the guide's known phone boundaries and midpoints through the warp path onto the recording;
   - LPC-analyse the recording, cut the diphone, estimate epochs for the residual mode (peak picking on the low-passed residual guided by `f0`);
   - reject prompts whose warp cost is above a threshold and list them for re-recording.

Recordings are made by us, so the voice carries whatever licence we give it; the description field records that.

Headless test of this pipeline (no microphone): the "speaker" is `klatt-female` rendering the same prompts at 90 % rate with ± 10 % random duration jitter (seeded). The true boundaries are known, so the autolabeller's error is measurable (§16.4).

### 8.3 Rate calibration

`-r` is words per minute. Each backend has a `native_wpm` constant: the rate it produces at `Controls.rate = 1.0` on the benchmark text (§16.6). Both backends share the duration model, so the constants should agree; the calibration test asserts `-r 175` yields 175 ± 15 wpm on each.

## 9. DSP and vocoder stage (`speak-dsp`)

All of it is our code on top of `realfft`. `--resynth` exercises it without the text frontend: it is the tool for comparing analysis/synthesis techniques on real audio, including `say` output.

### 9.1 Primitives (C §2.14)

- Periodic Hann and Hamming windows.
- STFT / iSTFT: plan once; scale the inverse by `1/N`; **zero the imaginary parts of the DC and Nyquist bins before every inverse** (realfft returns an error otherwise); overlap-add with window-sum-square normalization; `center` reflect padding; a streaming overlap-add struct with an `n_fft − hop` tail. Round-trip error ≤ 1e-5.
- Mel filterbank (Slaney scale and normalization; HTK as an option), log-mel, and MFCC by DCT-II of the log-mel.
- LPC (§8.2.2), LPC spectrum, formant estimation by root finding or peak picking on the LPC spectrum.
- F0 tracker (§8.2.2) and epoch (pitch mark) detection.
- DTW with a band constraint, returning the path.

### 9.2 Vocoders for `--resynth`

| `--vocoder` | Analysis | Synthesis | Supports `-r` (time scale) | Supports `--pitch` |
|---|---|---|---|---|
| `lpc` (default) | LPC frames + F0 + voicing | §8.2.3 with `-P excitation=pulse\|glottal` | yes | yes |
| `psola` | epochs | TD-PSOLA (§9.3) | yes | yes |
| `griffinlim` | log-mel (16 kHz, n_fft 1024, hop 256, 80 bands, 0–8000 Hz) | mel → linear magnitude by the filterbank pseudo-inverse (`Mᵀ(MMᵀ)⁻¹`, clamp ≥ 1e-5), then Fast Griffin-Lim: 32 iterations, momentum 0.99 (C §2.1) | yes (resample the mel in time) | no (error) |

Input: any WAV; mixed to mono and resampled to 16 kHz with `rubato`. For `--resynth`, `-r N` means time-scale factor `N / 175` and `--pitch` takes only the `±Nst` form.

### 9.3 TD-PSOLA (A §2.3)

1. Epochs: one mark per glottal period in voiced regions (from the F0 track, refined to the nearest local energy peak); unvoiced regions get marks every 5 ms.
2. Grains: two-period Hann-windowed segments centred on each mark.
3. Synthesis marks: spaced by the target period (source period ÷ pitch factor) along the time-scaled axis; each takes the grain of the nearest source mark; overlap-add.
4. Usable range: factors 0.5–2.0. Outside: error.

Used three ways: `--vocoder psola`; the diphone backend's `residual` excitation (PSOLA on the LPC residual, then the lattice filter: residual-excited LPC); pitch/rate experiments on recordings.

## 10. Audio output (`speak-audio`)

### 10.1 Playback

- `cpal` output stream on the default device or `-a`. Prefer a stream config at the backend's sample rate; otherwise use the device's default rate and resample with `rubato` (sinc, fixed ratio) on the synth thread before the ring buffer.
- Ring buffer: `rtrb`, capacity 2 s at the device rate. `write` blocks (parks in 2 ms steps) when full; that back-pressure is what bounds memory.
- Mono is copied to every output channel.
- Playback starts with the first written sample; no pre-buffering.
- `finish()`: wait until the ring buffer is empty, then for the stream's reported latency plus 50 ms, then drop the stream.
- Underruns: callback outputs zeros and increments a counter reported in `--timing`.

### 10.2 Files

- WAVE through `hound`, streamed; the header is finalized on `finish()` and on SIGINT.
- AIFF hand-written: `FORM`/`AIFF` with `COMM` (channels, frame count, bits, sample rate as 80-bit IEEE extended) and `SSND`; big-endian integers; sizes patched on `finish()`.
- raw: samples only.
- `-o -` WAVE: RIFF and data sizes written as `0xFFFFFFFF`.
- `@rate` in `--data-format` resamples with `rubato`.
- Float → integer: clamp to [−1, 1], scale, round; no dither.

## 11. Streaming and latency

### 11.1 Mechanism

1. Chunking (§7.3): one sentence per chunk.
2. The synth thread works ahead of playback; the ring buffer (2 s) and chunk channel (8) bound how far.
3. Both backends write 20 ms blocks as they are rendered, so audio starts before the first sentence is finished.
4. Explicit silence between chunks (§7.3 rule 5).

### 11.2 TTFA budget (cold start, M2 Max)

| Step | klatt | diphone |
|---|---|---|
| Process start, argument parsing, config | 5 ms | 5 ms |
| Lexicon open (parallel with the next two rows) | 5 ms | 5 ms |
| Voice file read | 0 | 10 ms (6 MB) |
| Audio device open | 30 ms | 30 ms |
| Frontend for chunk 1 | 5 ms | 5 ms |
| First 20 ms block | 1 ms | 1 ms |
| **TTFA** | **~45 ms** | **~50 ms** |
| Target (§2) | 100 ms | 100 ms |

Reference: `say -o x.aiff "hello world"` takes 0.55–0.65 s wall time on this machine (D §2.1, re-measured).

Startup rules: no work before argument parsing except what clap needs; lexicon and voice files are read whole and never parsed entry by entry; rule files compiled into the binary are parsed lazily on the frontend thread.

### 11.3 `--timing` output

Text form, on stderr:

```
backend      klatt
voice        klatt-male
sample_rate  16000
chunks       2
startup_ms        3.2
frontend_init_ms  4.8
voice_open_ms     0.0
frontend_ms       1.1
ttfa_ms          41.7
synth_total_ms   18.4
audio_s           4.21
rtf               0.004
underruns         0
lexicon_hits      user 0, builtin 4, cmudict 5, cart 1, rules 0, spelled 0
```

`ttfa_ms` = process start → first sample accepted by the sink. `rtf` = `synth_total_ms / 1000 / audio_s`. `--timing=json` prints one JSON object with the same keys. `--bench` prints `median`, `min`, `max` over N runs.

## 12. Data files

There is no model registry and no download code. Everything `speak` reads at run time is either compiled in, built locally by `speak-lab`, or (CMUdict only) fetched by a script.

| File | Origin | Location | In git |
|---|---|---|---|
| Klatt tables, duration tables, function words, built-in lexicon, LTS rules | written by us | source / `crates/*/data/` | yes |
| `lts-en.cart` | trained by `speak-lab lts train` from CMUdict | `data/lts-en.cart`, compiled in | yes, if ≤ 2 MB |
| `en-us.lex` | compiled by `speak-lab lexicon build` from CMUdict | `<data>/lexicon/` | no |
| `boot.sdv` | built by `speak-lab diphones build --bootstrap` | `<data>/voices/` | no |
| `<name>.sdv` + recordings | recorded and built by us | `<data>/voices/`, recordings wherever `--dir` points | no |
| Stretch: LJSpeech corpus, our trained weights | downloaded by a script / trained by us | `<data>/neural/` | no |

| Directory | macOS | Linux | Override |
|---|---|---|---|
| `<data>` | `~/Library/Application Support/speak/` | `$XDG_DATA_HOME/speak/` (default `~/.local/share/speak/`) | `SPEAK_DATA_DIR` |
| Config | `~/.config/speak/config.toml` | `$XDG_CONFIG_HOME/speak/config.toml` | `SPEAK_CONFIG`, `--config` |

## 13. Configuration

`config.toml`, all keys optional. Precedence: command line > environment > config file > built-in default.

```toml
backend = "klatt"         # klatt | diphone
rate = 175
volume = 1.0
lts = "cart"              # cart | rules

[klatt]
voice = "klatt-male"

[diphone]
voice = "boot"
excitation = "glottal"    # pulse | glottal | residual

[audio]
device = ""               # empty = system default

[voices.narrator]         # alias usable as -v narrator
backend = "klatt"
voice = "klatt-female"
rate = 165
params = { tl = 12, oq = 0.65 }
```

Environment: `SPEAK_CONFIG`, `SPEAK_DATA_DIR`, `SPEAK_BACKEND`, `SPEAK_VOICE`. An unknown key in the file is an error (exit 2) naming the key and line.

## 14. Licensing and provenance

1. `speak` source: MIT OR Apache-2.0. All synthesis, analysis and text-processing code is written for this project.
2. Dependencies are general-purpose libraries under MIT, Apache-2.0, BSD, ISC, Zlib or Unicode licences (§5). `deny.toml` holds that allowlist; `cargo deny check licenses` must pass.
3. `scripts/check-deps.sh` fails if the default build's `cargo tree` contains any of: `ort`, `onnx`, `tract`, `candle`, `burn`, `tch`, `espeak`, `misaki`, `piper`, `kokoro`, `num2words`, `hf-hub`, `reqwest`, `ureq`.
4. No pretrained weights or third-party voice data anywhere: not in the repository, not downloaded, not loaded.
5. Sources that must not be read into or ported to this codebase: espeak-ng (including `klatt.c` and the pure-Rust port), rhdunn/klatt, jh4xsy/klatt80, Praat KlattGrid, svara, Festival/Flite/MBROLA code and voices, DECtalk. Allowed references: the papers in `docs/research/papers/`, the NRL letter-to-sound report, and the MIT-licensed chdh/klatt-syn and klattsch-rs for cross-checking behaviour (not as dependencies, not as code to translate).
6. CMUdict: data file, © Carnegie Mellon University, BSD-2-Clause-style licence (redistribution allowed with the copyright notice). Optional at run time. The trained `lts-en.cart` is derived from it; `NOTICE` reproduces the CMUdict notice.
7. Voices we record belong to us. The stretch corpus LJSpeech is public domain.
8. macOS `say` and `whisper` are **test instruments only** (§16.5): `say` is the comparison baseline, whisper transcribes our output to measure word error rate. Neither is linked, shipped, called by the product, or used as a source of training or voice data.

## 15. Error handling

One error enum in `speak-core` (`thiserror`), mapped to exit codes in `speak-cli`:

| Variant | Exit | Message shape |
|---|---|---|
| `Usage` | 2 | `speak: unknown voice 'x'. Run speak -v '?'` |
| `VoiceMissing { name, hint }` | 3 | `speak: diphone voice 'boot' is not built. Run: speak-lab diphones build --bootstrap` |
| `DataInvalid { file, why }` | 3 | bad magic, wrong sample rate or order, truncated file |
| `Unsupported { what }` | 2 | e.g. `griffinlim cannot change pitch` |
| `Audio { cause }` | 4 | device list hint |
| `Io`, `Synthesis`, `Config` | 1 / 1 / 2 | |

Rules:

- Libraries return `Result`; no `unwrap`/`expect` on data that depends on input or files. Panics are bugs.
- The text frontend never fails on content: unknown characters are skipped, unknown words go through letter-to-sound, unmatched digit strings are read digit by digit. Each is counted and visible under `-V`.
- Filters must not blow up: clamp reflection coefficients to |k| ≤ 0.999, bypass resonators near the Nyquist frequency, and check for non-finite samples per block in debug builds.
- SIGINT: exit 130, output file left valid.

## 16. Testing

### 16.1 Layers

| Layer | Where | Runs in `cargo test` |
|---|---|---|
| Unit tests | each crate | yes |
| Golden audio | `tests/fixtures/{klatt,diphone}/` | yes (diphone goldens only when `boot.sdv` exists; otherwise the test prints `skipped` and passes) |
| CLI tests | `speak-cli/tests` | yes |
| Intelligibility (whisper, test instrument) | `speak-cli/tests/intelligibility.rs` | `cargo test -- --ignored intelligibility`; needs `whisper` on `PATH` |
| Baseline comparison (`say`, test instrument) | `scripts/bench-vs-say.sh`, intelligibility test | manual; macOS only |

### 16.2 Klatt

- Resonator: impulse response peak frequency within 1 % of F; −3 dB width within 10 % of BW; unity DC gain. Antiresonator cascaded with its resonator = identity within 1e-5.
- Sources: period length matches F0 within 0.5 % over 1 s at 100, 150, 220 Hz; open phase fraction matches OQ; `lf` and `klglott88` at the same settings have spectral tilt within 3 dB of each other between 500 Hz and 3 kHz; `epoch()` fires once per period.
- Formant test: render each monophthong for 300 ms at F0 110 Hz; estimate F1 and F2 by LPC (§9.1) at the midpoint; each within 8 % of the table target.
- Level tests (fit the gain constants to these): `AA` at AV 60 → RMS −20 ± 1 dBFS. Relative to it: `SH` −9 ± 3 dB, `S` −14 ± 3 dB, `HH` −16 ± 4 dB, `F` −21 ± 4 dB.
- Rule compiler: duration of the audio equals the sum of `dur_ms` within one frame; no formant outside Klatt 1980 Table I ranges scaled by the voice; no NaN; no sample above 1.0.
- Context rules (§8.1.8): F2 at vowel onset after `G` is higher before `IY` than before `UW` by ≥ 600 Hz; a 50 ms `AA` has F1 below its 240 ms F1 (undershoot); `T` in `B AH1 T ER0` becomes a flap after postlexical rules.
- Golden audio: 10 fixtures. Compare with the stored 16-bit WAV: SNR ≥ 60 dB on the same OS/arch, ≥ 40 dB otherwise. `UPDATE_GOLDEN=1` rewrites them; regenerate only on purpose, in a commit of their own.

### 16.3 Frontend

- Normalizer: table test with ≥ 120 cases covering every class in §7.1. Number expansion: 0–20, tens, hundreds, 10⁳–10¹², negatives, decimals, ordinals, years.
- Chunker: abbreviations, decimals, quotes, newlines.
- Lexicon: built-in entries parse; with `en-us.lex` present, lookup of 20 fixed words equals stored phones; open time ≤ 5 ms.
- `RuleLts`: on `tests/fixtures/lts_words.txt` (500 words with reference phones, hand-checked, not copied wholesale from CMUdict into the repo beyond this list), phoneme error rate (edit distance ÷ reference length, stress ignored) ≤ 35 %.
- `CartLts`: held-out CMUdict accuracy ≥ 50 % words, ≥ 88 % letters (`speak-lab lts eval`, needs the CMUdict file); on `lts_words.txt` PER lower than `RuleLts`.
- Postlexical: one test per rule in §7.6. Homographs: `I read it yesterday` vs `I will read it`.
- No digit and no unknown-phone token in any `--dump phonemes` output over the test corpus.

### 16.4 Diphone and DSP

- LPC: analysing an all-pole synthetic signal recovers its reflection coefficients within 1e-3; analysis → synthesis with the true residual reconstructs the input (SNR ≥ 60 dB).
- F0 tracker on Klatt output at 90–250 Hz: within 2 % on ≥ 95 % of voiced frames. Epoch detector vs the true `epoch()` marks: ≥ 95 % within 1 ms.
- STFT round trip; mel filterbank rows sum and peak positions; MFCC of a pure tone; DTW of a sequence against its 1.3× stretched copy recovers the mapping within 2 frames.
- Bootstrap voice: all inventory diphones present; formants of the `AA-AA` diphone midpoint within 10 % of the Klatt table (the measured filter matches the ruled one).
- Same `.pho` through both backends: equal duration within 10 ms; F0 contour of the diphone output (our tracker) within 5 % of the targets on voiced frames.
- PSOLA: +4 st on a Klatt vowel changes measured F0 by a factor 1.26 ± 3 % and duration by < 2 %; time scale 1.5 changes duration by 1.5 ± 2 % and F0 by < 2 %.
- Griffin-Lim on the mel of a 440 Hz sine: output spectrum peaks within one bin of 440 Hz.
- Autolabeller with the synthetic speaker (§8.2.6): median boundary error ≤ 15 ms, 90th percentile ≤ 40 ms over 200 prompts.

### 16.5 Intelligibility, with `say` as the baseline

Whisper and `say` are measuring instruments here, nothing else (§14 item 8).

Test set: `tests/fixtures/sentences.txt`, 20 sentences (10 Harvard sentences, 5 with numbers/dates/currency, 5 with names and technical terms). For each system: synthesize each sentence to WAV, transcribe with `whisper <wav> --model base.en --language en --fp16 False --output_format txt`, normalize both sides with our normalizer, compute word error rate over the set (`speak-lab wer`).

| System | WER |
|---|---|
| `say` (default voice) — baseline | reported, expected < 5 % |
| klatt, rules LTS, after milestone 3 | ≤ 35 % |
| klatt, with CMUdict + CART, after milestone 5 | ≤ 25 % |
| klatt after milestone 6 (context rules) | ≤ 20 % |
| diphone `boot` | ≤ 35 % |
| diphone voice built from the synthetic speaker (milestone 9) | ≤ 40 % |

The report always prints our WER next to the `say` WER and the per-word confusions, which is the tuning input. Whisper measures intelligibility only; naturalness is judged by ear.

### 16.6 Benchmarks

Benchmark text: `tests/fixtures/bench.txt` (three sentences, 45 words).

- `speak -b <backend> --bench=5 -f tests/fixtures/bench.txt` → TTFA and RTF against §2.
- `scripts/bench-vs-say.sh`: wall time to file for `say -o x.aiff` and `speak -b <each> -o x.wav` on the benchmark text and on `hello world`, 5 runs each, medians in a table; plus binary size.
- Words per minute at `-r 175` for §8.3; `say -r 175` duration on the same text for reference.

## 17. Stretch: a neural model we train ourselves (`speak-neural`)

Optional. Last milestone. Not part of the default build or of any earlier acceptance criterion. **No pretrained checkpoint is used at any point**: no pretrained encoder, aligner, vocoder or embedding.

| Aspect | Decision |
|---|---|
| Framework | `burn` (training and inference in Rust; Metal through wgpu). Chosen over candle because training loops, optimizers and dataset loaders are first-class in burn (D §2.1). |
| Data | LJSpeech 1.1: 13,100 clips, ~24 h, one speaker, public domain (B §2.1). Downloaded by `scripts/fetch-ljspeech.sh` (data, not a model). Resampled to 16 kHz. |
| Text side | Our own frontend produces the phones for every transcript. |
| Durations | From our own aligner: DTW between each clip's MFCCs and the Klatt rendering of its phones (§8.2.6), so no external forced aligner. Clips with high warp cost are dropped. |
| Model | FastSpeech-2-shaped but small (B §2.4): phone + stress embedding (128) → 3 convolutional/self-attention encoder blocks → duration predictor and F0 predictor (2-layer conv each) → length regulator → 3 decoder blocks → 80-band log-mel at 16 kHz, hop 256. Target ~3–5 M parameters. Loss: L1 on mel + MSE on log-duration and log-F0. |
| Waveform | Our Griffin-Lim (§9.2). A vocoder we train ourselves is out of scope. |
| Budget | Fits a laptop: stop at 48 h of training or when validation mel loss plateaus. |
| Exit criteria | `speak -b neural "…"` works; WER ≤ 30 % on the test set; RTF ≤ 0.5. If training does not converge within the budget, write down the loss curves and stop. |
| Artefacts | Our weights in `<data>/neural/`; not committed. |

## 18. Open questions

Each has a default the implementation takes without asking.

| # | Question | Default |
|---|---|---|
| 1 | Klatt at 16 kHz with 8 cascade formants vs the paper's 10 kHz with 5. | 16 kHz / 8. If vowels sound thin or metallic, fall back to 5 cascade formants and compare. |
| 2 | Commit the trained `lts-en.cart` (derived from CMUdict) or build it locally? | Commit if ≤ 2 MB, with the CMUdict notice. |
| 3 | Ship `en-us.lex` compiled in instead of optional? | No: optional file, per the "data as lookup, rules as fallback" decision. |
| 4 | How much of the diphone inventory is worth recording by hand? | Tooling supports partial sets; start with the ~600 most frequent diphones (frequency list from running our frontend over `bench.txt`-like text) and let rule 5 of §8.2.5 fill gaps. |
| 5 | LPC order 18 and 25 ms window for recorded voices. | Keep; revisit with `speak-lab analyze` on real recordings. |
| 6 | Embedded `[[pbas]]` and `[[inpt PHON]]` semantics differ from Apple's. | Keep the documented deviations. |
| 7 | Whether to attempt the neural stretch at all. | Only after milestone 10 and only if Helge asks for it. |
| 8 | A third classic backend (HMM/statistical parametric, A §2.5). | Not planned. |
