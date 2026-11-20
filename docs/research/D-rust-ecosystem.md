# D: Rust ecosystem and existing tools

Research date: 2026-11-20. Versions and dates come from the crates.io API and the GitHub API on that day. Measurements were taken on an Apple M2 Max with rustc 1.99.0; the throwaway probe crate lives outside the project (scratchpad) and is not part of `speak`.

Confidence markers: 🟢 primary source (repo, docs, crates.io, or measured here), 🟡 secondary (blog, README benchmark, issue comment), 🔴 inferred or unverified.

---

## 1. Overview 🟢

A Rust `say` clone needs six pieces: argument parsing, a text frontend (normalization and G2P), an inference runtime for neural models, a DSP toolkit for a hand-written synth, audio output, and model download and caching. Rust covers all six, but the pieces mature at different rates:

| Layer | State in Nov 2026 |
|---|---|
| Audio out (cpal/rodio/hound/rubato) | Mature, many users, active |
| ONNX inference through `ort` | Mature and active, still on `2.0.0-rc.*` tags; the only runtime that runs Piper and Kokoro today |
| Pure-Rust inference (tract, candle-onnx, burn-onnx) | Active, but **none runs Piper today**; tract cannot run Kokoro; burn runs Kokoro with measurable numeric drift |
| Text frontend | Fast-moving: a wave of new GPL-free G2P crates appeared in 2025–26 after sherpa-onnx dropped espeak-ng (Oct 2026). Most are weeks old with 0–50 stars |
| Rust TTS wrappers | Many small single-maintainer crates; Kokoros (842★) is the largest; sherpa-onnx ships an official Rust crate |

Timeline of the events that shape the choices:

- 2023: Piper (rhasspy) becomes the default local TTS; `piper-rs`/`sonata` port it to Rust on ONNX Runtime.
- Jan 2025: Kokoro-82M ONNX exports appear; Kokoros (Rust) ships within days.
- 2025: `ort` 2.0 release candidates add **alternative backends** (`ort-tract`, `ort-candle`) behind the same API (rc.12).
- Aug/Oct 2025: `rhasspy/piper` archived; development continues as `OHF-Voice/piper1-gpl` (GPL-3.0, espeak-ng embedded).
- Mar 2026: `thewh1teagle/sherpa-rs` archived in favour of the official `sherpa-onnx` Rust crate.
- Oct 2026: sherpa-onnx 2.0 **removes espeak-ng** to stay Apache-2.0 and switches to lexicon files; ORT 1.31 starts auto-selecting CoreML on macOS, which crashes Piper models under `ort` rc.14/rc.15 defaults.

---

## 2. Techniques and components

### 2.1 Inference runtimes

#### `ort` (ONNX Runtime bindings) 🟢

- **Version:** `2.0.0-rc.15` (2026-10-19), wraps ONNX Runtime 1.31. 26.9 M downloads; repo `pykeio/ort` 2.6k★, pushed 2026-11-18. MIT OR Apache-2.0. Still a release candidate: pin it with `=`, because rc bumps break the API. For example, rc.15's `Session::builder` takes a `&Environment`, unlike the `Session::builder()` that `piper-rs` uses on rc.12.
- **Linking:**
  - The default features (`download-binaries`, `copy-dylibs`, `tls-native`, `api-31`) download a **statically linked** ONNX Runtime built by pyke into `~/Library/Caches/ort.pyke.io/`. The builds carry SLSA attestations, and `dist.tsv` holds their SHA-256 hashes. Linux builds use clang and libc++, and x86-64 builds need x86-64-v3 (Haswell or newer).
  - `load-dynamic` uses `dlopen` on a `libonnxruntime` found at runtime (`ORT_DYLIB_PATH` or `ort::init_from`).
  - `ORT_LIB_PATH` points the build at your own ONNX Runtime build, static or dynamic (`ORT_PREFER_DYNAMIC_LINK=1`).
- **Binary size, measured:** a stripped release binary (thin LTO) with static ORT 1.31, serde_json and hound comes to **19.3 MB**. It links only system frameworks (Foundation, CoreML, CoreFoundation, libc++, libobjc), with no extra dylibs to ship. 🟢
- **Build gotcha, measured:** with rustc 1.99 on macOS, the static build failed to link with `Undefined symbols: ___isPlatformVersionAtLeast` (from ORT's CoreML EP, ort issue #360). Fix: `rustflags = ["-Clink-arg=-fapple-link-rtlib"]` in `.cargo/config.toml`. The troubleshooting page that ort #360 links to no longer contains this entry. 🟢
- **CoreML EP:** every pyke macOS build includes the CoreML EP (feature `coreml`). **Trap:** since rc.12, `SessionBuilder::new` silently sets `AutoDevicePolicy::MaxEfficiency`, and with ORT 1.31 that hands macOS sessions to CoreML/BNNS (ort #662, #666). With that default, the Piper model **crashed the process** after a 4.5 s load (reproduced here, matching #666). Registering the CPU EP does not help (#662). Call `.with_auto_device(AutoDevicePolicy::PreferCPU)`. 🟢 Even when CoreML works, it is not a win: kokoro-cli measured CoreML *slower* than CPU on an M5 (load 1.7 s vs 0.5 s, RTF 0.21 vs 0.17), and the Kokoros README says the same. 🟡
- **Alternative backends:** with `features = ["alternative-backend"]` and `ort::set_api(ort_tract::api())`, the same `ort` code runs on tract (`ort-tract 0.5.0+0.23`) or candle (`ort-candle 0.5.0+0.11.0`). This is an easy way to A/B runtimes behind one API, but it inherits each backend's op gaps (below). 🟢
- **Measured with Piper `en_US-lessac-medium` (63 MB, 22.05 kHz), CPU, default threads:**
  - Session load about **0.8 s**: 0.81–0.87 s at optimization levels 1–3, 1.18 s with optimizations off, 0.71 s for a pre-optimized model saved with `with_optimized_model_path`.
  - Inference **80–90 ms for 3.7–4.1 s of audio, RTF ≈ 0.021**.
  - Output checked with whisper.cpp large-v3-turbo: "This is a test of the speech synthesizer running locally on a laptop." 🟢
- **Measured with Kokoro v1.0 fp32 ONNX (326 MB, 24 kHz), CPU:** load **0.42 s**; inference **750–800 ms for 4.55 s of audio, RTF ≈ 0.17**. Whisper transcript correct. That RTF matches kokoro-cli's figure of 0.17 on an M5. 🟢
- Takeaway: on this machine, **model load costs more than synthesis** for a short Piper sentence (0.8 s vs 0.09 s). For comparison, `say -o x.aiff "hello world"` takes 0.65 s wall time in total. 🟢

#### `tract` (Sonos) 🟢

- **Version:** `tract-onnx 0.23.10` (2026-11-05), 3.1k★, active, MIT OR Apache-2.0. Pure Rust, no C++ toolchain.
- **Binary size, measured:** **17.9 MB** stripped, the same probe as above. Only 1.4 MB smaller than static ORT, so tract's size advantage over ORT is small. 🟢
- **Piper (VITS) fails, measured:**
  - With symbolic shapes, `into_optimized` fails at `/enc_p/encoder/attn_layers.0/Reshape_10` ("Reshape volume mismatch" in symbolic algebra).
  - With concrete input shapes and `with_ignore_value_info(true)`, the plan builds, but `run` fails at `/Cast_3`: "Undetermined symbol in expression: range".
  - This is the open issue sonos/tract#2931: runtime symbols from `NonZero`/`Range` inside the VITS duration-predictor flows. A contributor has been peeling off fixes since Sep 2026 (#2929, #2932, #2959), but MMS VITS still does not run end to end, and a design decision from the maintainer is pending. 🟢
- **Kokoro fails, measured:** `/decoder/decoder/Concat` gives "Inconsistent concat … `(1+2*range)/2`", again data-dependent lengths, in the iSTFTNet decoder. sonos/tract#1634, "Kokoros onnx support", has been open since Feb 2025. 🟢
- Verdict: not usable for Piper or Kokoro today. Worth re-testing later; the `ort-tract` route makes that a one-line switch.

#### `candle` (Hugging Face) 🟢

- **Version:** `candle-core 0.11.1` (2026-10-31), 21k★, MIT OR Apache-2.0. Metal and Accelerate on Apple, CUDA elsewhere.
- **candle-onnx:** cannot load Piper. PR huggingface/candle#3792, "add Softplus, NonZero, GatherND, ConvTranspose, RandomNormalLike and Pad constant mode", is open (Jul 2026); it says these are exactly the ops VITS/Piper needs. Its Kokoro status is unknown (🔴), and the same ops (ConvTranspose, STFT-like ops) are likely missing.
- **Native candle models:** candle-transformers has TTS examples for parler-tts, metavoice, encodec, mimi, csm and orpheus, but **no Kokoro or VITS**. A Kokoro PR (#3420) was closed unmerged. Third-party native Kokoro ports on candle exist:
  - `any-tts 0.2.0`: Kokoro, Qwen3-TTS, VibeVoice and others behind one trait; CPU/CUDA/Metal.
  - `omp-voice-kokoro 0.1.0`: Kokoro on candle with Metal.
  - Lucas Jin's `Crane`, which Kokoros now points to.

  These run from the original `.pth`/safetensors weights, not ONNX. 🟢 (that they exist) / 🔴 (quality and speed not verified)
- Verdict: the right tool for a later "rewrite the model natively in Rust" experiment, not for loading ONNX files.

#### `burn` + `burn-onnx` (Tracel) 🟢

- **Version:** `burn 0.22.1` (2026-11-09), 16k★. ONNX import moved to `burn-onnx`, which generates Rust code and weights at build time; `burn-import` is legacy.
- **Kokoro** is listed as working in burn-onnx's model tracker (#18), but burn-onnx#371 (open) documents audio that **diverges from ORT**. The cause is Burn's FFT/STFT versus ORT's, amplified by a hand-written atan2 pattern in the Kokoro graph. The best variant in Nov 2026 reaches a Pearson correlation of 0.75–0.83 with ORT's output (#679). 🟢
- **Piper/VITS:** "converts on main, generated code does not compile" (#533, #534). 🟢
- Verdict: interesting for a pure-Rust, GPU-portable (wgpu/Metal) build later; not ready.

### 2.2 Audio output, files and DSP 🟢

| Crate | Version / date | Licence | Notes |
|---|---|---|---|
| `cpal` | 0.18.2 / 2026-08-16 | Apache-2.0 | Low-level. macOS CoreAudio; Linux ALSA by default, with `pulseaudio`, `pipewire` and `jack` features. ALSA dev headers (`libasound2-dev`) are needed at build time even with PipeWire. Callback model; we supply f32 frames. |
| `rodio` | 0.23.0 / 2026-10-11 | MIT OR Apache-2.0 | On cpal. Default features pull playback, recording and flac/mp3/mp4/vorbis/wav decoders plus pulseaudio. Use `default-features = false, features = ["playback"]` for a lean build. `wav_output` uses hound. |
| `hound` | 3.5.1 / 2023-09-25 | Apache-2.0 | WAV read/write. Stable and feature-complete (repo pushed 2026-08). |
| `rubato` | 5.1.0 / 2026-11-12 | MIT | Sync and async resampling (sinc, FFT). Needed because Piper produces 22.05 kHz and Kokoro 24 kHz, while output devices usually run at 44.1/48 kHz. |
| `rustfft` | 6.4.1 / 2025-09-18 | MIT OR Apache-2.0 | Pure Rust, NEON/AVX. |
| `realfft` | 3.5.1 / 2026-10-26 | MIT | Real-to-complex wrapper over rustfft, for STFT/iSTFT, LPC and spectrograms. |
| `symphonia` | 0.6.1 | MPL-2.0 | Decoding only. Not needed unless we read audio. |

`say` writes AIFF/CAF/M4A/WAVE through CoreAudio's `AudioFile` and converters. Rust has no AIFF/AAC encoder that is as complete; hound covers WAV. AIFF is a simple container we could write by hand (🔴); AAC/ALAC would need the macOS `AudioToolbox` FFI or ffmpeg.

### 2.3 Text frontend in Rust

What the models expect:

- **Piper** voices are trained on **espeak-ng IPA**. `phoneme_type: "espeak"`, the `espeak.voice` (e.g. `en-us`) and a `phoneme_id_map` of single IPA characters all live in `<voice>.onnx.json`. Input format: `^`, then each phoneme followed by the pad `_`, then `$`; plus `input_lengths` and `scales` = [noise 0.667, length 1.0, noise_w 0.8] (measured). 🟢
- **Kokoro** expects **misaki** phonemes (IPA plus capital-letter diphthongs such as `I`, `O`, `A`), 115 vocabulary symbols from `tokenizer.json`. Input is `input_ids` padded with 0 at both ends, `style` = `voice[n_tokens]` (a 510×256 f32 table per voice) and `speed`. 🟢

#### espeak-ng in Rust 🟢

| Crate | Version | What it is | Licence reality |
|---|---|---|---|
| `espeak-rs` / `espeak-rs-sys` (thewh1teagle, in the piper-rs repo) | 0.2.0 / 2026-05 | Builds the espeak-ng C sources (git submodule) with CMake and links them statically; `text_to_phonemes`. Used by piper-rs, Kokoros, misaki-rs (default feature), kokoro-cli | Crate says MIT, but the linked binary contains **GPL-3.0** espeak-ng |
| `espeakng-sys` (Better-Player) | 0.3.0 / 2025-05 | Raw FFI to a system libespeak-ng | **Archived** |
| `espeakng` (GnomedDev) | 0.2.0 / 2023-08 | Safe wrapper over espeakng-sys | Stale |
| `espeak-ng` (eugenehp/espeak-ng-rs) | 0.2.0 / 2026-09-05 | **Pure-Rust port** of espeak-ng: dictionary, rules, `text_to_ipa`, and a harmonic/formant synthesizer; tested bit-identical against the C library (319 tests, 27/27 IPA oracle cases). Data from `espeak-ng-data-*` crates (`bundled-data-en` embeds one language) or the runtime `ESPEAK_DATA_PATH` | **GPL-3.0-or-later**, being a port. 27★, single maintainer, a few months old |

The GPL-3 implications 🟡 (general GPL reading, not legal advice):

- Linking espeak-ng statically or dynamically makes the distributed `speak` binary a combined work under GPL-3. The source of the whole program must then be offered under GPL-3-compatible terms. MIT code may be combined in, but the result ships as GPL.
- A pure-Rust *port* is a derivative of espeak-ng, so it is also GPL.
- Running a separate `espeak-ng` executable via `std::process::Command` and reading its IPA output is generally treated as "aggregate", not linking. This is how Piper used to be wrapped by many MIT tools. 🔴 (common practice, legally untested)
- espeak-ng's **output** (phoneme strings, and lexicons generated with it, as misaki-rs did) is generally not covered by the program's GPL. 🔴
- For a personal tool that is never distributed, the GPL imposes nothing. It matters only if `speak` is published under MIT and ships binaries.

Industry signal: sherpa-onnx removed espeak-ng entirely (PR #4055, 2026-10-20) "to keep sherpa-onnx fully compatible with Apache-2.0". Piper voices now get a `lexicon.txt`, plus an optional external phonemizer whose phonemes are passed in as `phoneme_codepoints`/`tokens`. The cost is that out-of-vocabulary words and numbers are **dropped silently** (sherpa-onnx#4139: "The budget is 1969 dollars" is spoken as "The budget is dollars"). 🟢

#### GPL-free G2P in Rust 🟢 (existence and claims) / 🔴 (quality not verified unless stated)

| Crate | Version / date | Licence | What it does |
|---|---|---|---|
| `misaki-rs` | 0.11.0 / 2026-11-19 | MIT (data from misaki, Apache-2.0) | Port of misaki 0.9.4 English (US/GB) with lexicons, spaCy-compatible tokenizer, POS tagger (heteronyms such as *record*), suffix stemming and number words (`num2words`). Everything embedded at compile time. Parity tests against Python misaki on 1,700+ sentences. The default feature `espeak` uses espeak-rs (GPL) for OOV words; with `default-features = false` an OOV word becomes `❓`. 105k downloads; repo 10★ |
| `loqui-g2p` | 0.4.0 / 2026-10-20 | MIT | Misaki port plus an **embedded BART neural OOV fallback** run in pure Rust (safetensors). Claims 0.01–0.78% phoneme error vs the reference. Repo 1★, brand new |
| `piper-plus-g2p` | 0.5.4 / 2026-11-13 | MIT | "eSpeak-ng free" multilingual G2P (8 languages) from the piper-plus fork (240★). Whether it reproduces espeak IPA closely enough for stock rhasspy Piper voices is **unknown** 🔴; piper-plus trains its own voices |
| `floravox-g2p` | 0.8.13 / 2026-11-19 | Apache-2.0 OR MIT | Memory-mapped FST lexicons with a pluggable OOV fallback (by willwade, who is also driving the tract VITS fixes) |
| `voicegarden-lexicons` | 0.3.2 / 2026-11-17 | Apache-2.0 OR MIT | Fetches permissively licensed pronunciation lexicons ("the espeak-ng replacement") |
| `g2p2-core` | 0.3.0 / 2026-11-07 | MIT OR Apache-2.0 | n-gram plus lexicon G2P for 100 languages, zero dependencies (Gladia) |
| `phonetisaurus-g2p` | 0.1.1 / 2025-04 | MIT | Runs Phonetisaurus-trained FSTs |
| `cmudict-fast` | 0.8.0 / 2021-02 | MIT/Apache | CMUdict (ARPAbet) lookup. Old but works; the CMUdict data is BSD-style 🟡 |
| `open-english-pronouncing-dictionary` | 0.1.0 / 2026-05 | **CC-BY-SA-4.0** | About 280k-word fused US IPA lexicon. Share-alike applies to the data |
| `grapheme_to_phoneme`, `arpabet` | 2020 | BSD-4 / MIT | Stale |

#### Text normalization 🟢

- `num2words` 1.2.0 (MIT OR Apache-2.0): numbers, ordinals, years and currency to words, in several languages. misaki-rs already depends on it.
- `text-processing-rs` 0.3.4 (Apache-2.0, FluidInference, 49★): a port of **NVIDIA NeMo Text Processing**, TN and ITN. With the `fst-engine` feature it runs NeMo's own compiled grammars and matches NeMo TN byte for byte in en/de/es/fr/hi/it/ja/zh. Costs about 12 MB of binary and 1–25 ms per sentence. Without it, rule-based taggers run in microseconds. Handles money, dates, times, units and ordinals.
- `text2num` 2.8.0 does the opposite direction (words to numbers). Not needed.
- Unicode: `unicode-normalization` (NFKC, as misaki does it).

### 2.4 Existing Rust TTS projects 🟢

| Project | Version / activity | Stack | Structure and lessons |
|---|---|---|---|
| **piper-rs** (thewh1teagle) | 0.2.0, 2026-05; 67★ | `ort =2.0.0-rc.12` + `espeak-rs` (static C) + serde | One `Piper` struct: `new(model, config)`, `create(text, is_phonemes, speaker_id, length_scale, noise_scale, noise_w) -> (Vec<f32>, rate)`. The config JSON maps straight to a serde struct. The README's "pure Rust implementation" claim is wrong: it is ORT plus C espeak-ng, and it ships under an MIT label. Based on mush42's `sonata` |
| **Kokoros** (lucasjinreal) | no crates.io release; repo 842★, pushed 2026-10 | `ort rc.11`, `espeak-rs`, hound, mp3lame, opus, tokio, reqwest | Workspace: `kokoros` (lib: `onn/ort_koko.rs` session wrapper, `tts/{normalize,phonemizer,tokenize,vocab}.rs`, `utils/{wav,mp3,opus}.rs`), `koko` (CLI), `kokoros-openai` (server). CLI subcommands `text`, `file`, `stream` (stdin lines in, WAV out), `openai`; flags `-o`, `--timestamps` (TSV of word timings with the timestamped ONNX), style mixing `af_sky.4+af_nicole.5`. Caches voices in `~/.cache/kokoros/`. Measured TTFA about 1.4 s on M2 for the server path |
| **kokoro-cli** (ghchinoy) | 0.2.0, 2026-07 | misaki-rs (with espeak fallback, needs CMake) + ort | XDG data dir, a `setup` command that downloads and hash-checks models, a `bench` subcommand, inline phoneme overrides. Published CPU vs CoreML numbers (see above) |
| **loqui** (Awakened-Labs) | 0.4.0, 2026-10 | `loqui-g2p` (GPL-free) + `loqui-kokoro` on ORT, whisper.cpp | `loqui speak "…" -o x.wav`. Models download on first use from HF **pinned to commit revisions with SHA-256 checks**; `--offline` and `loqui fetch`. Voice blends `af_bella(2)+af_sky(1)`, named blends in `~/.config/loqui/voices.toml`. A good model for our download and cache design |
| **sherpa-onnx** official Rust crate | 2.0.2, 2026-11-19; repo 16k★ | C++ library (static by default; build script downloads prebuilt `-lib` archives from GitHub releases; `SHERPA_ONNX_LIB_DIR` override; `shared` feature) | Supports VITS/Piper, Kokoro, Matcha, KittenTTS, Pocket TTS, Supertonic and ZipVoice. Since 2.0 it has no espeak-ng and uses lexicon-only frontends. Useful as a **reference backend** for A/B comparison, not as the core |
| sherpa-rs (thewh1teagle) | 0.6.8 | — | **Archived 2026-03**; points to the official crate |
| **tts** (tts-rs, ndarilek) | 0.26.3 (crate 2024-07; repo active 2026-08) | Wraps OS engines: AVFoundation (macOS/iOS), Speech Dispatcher (Linux), WinRT/SAPI/NVDA, Android, Web | Enables a cheap "system" backend: `speak --engine system` would use the same voices as `say`. Not offline-neural |
| natural-tts | 0.3.1, 2025-12; 26★ | pyo3 to Parler/Coqui/MetaVoice, gTTS, MSEdge, tts-rs | Largely Python and cloud. Not useful |
| `kokoro-tts` (mzdk100) | 0.3.3, 2026-06 | ORT | Chinese-language docs; cross-compiles to Android and iOS |
| `kokoro-en`, `kokoro-micro`, `kokoroxide`, `kittentts`, `any-tts`, `omp-voice-kokoro` | 2025–26 | ORT or candle | Many small single-maintainer ports. They confirm the pattern but none is a dependency to rely on |
| **klattsch-rs** (`klattsch-core`/`-text`/`-wav`) | 0.1.2, 2026-08; MIT | Pure Rust | Parallel-formant **Klatt 1980** synth. `FormantSynth::process` is realtime-safe (no allocation, locks or I/O); `.klatt` phoneme strings in ARPAbet (`"HH AH L OW"`) are compiled to a schedule; a live-parameter event channel; golden-WAV parity tests against the JS original. The best existing reference for our formant backend |
| `klatt` (TTWNO) | 0.0.0, 2025-02; Apache-2.0 | `no_std` | Klatt algorithms from the reference C implementation; not a TTS engine |
| `espeak-ng` (Rust port) | 0.2.0 | Pure Rust, GPL | Contains a 3-formant cascade synth driven by IPA, plus espeak's own harmonic synth. Read for ideas; do not copy into an MIT codebase |
| `svara` | 2.0.0, GPL-3.0-only | — | Formant and glottal-source synthesis; GPL |

Common structure across these projects, and the lessons for `speak`:

1. Every working neural project separates three steps: **phonemize** (text to phoneme string), **tokenize** (phoneme string to IDs from the model config) and **infer** (IDs plus style or speaker to f32 PCM). The API is `fn synth(&mut self, phonemes) -> (Vec<f32>, sample_rate)`. 🟢
2. Accepting **raw phonemes as input** is standard: Piper's `[[ bˈætmæn ]]`, piper-rs's `is_phonemes`, kokoro-cli's inline overrides, sherpa's `phoneme_codepoints`. This lets the frontend and the backend be tested separately. 🟢
3. Streaming is done by **sentence chunking** and synthesizing chunk by chunk (Kokoros `stream`, sherpa's `tts-max-num-sentences`, Piper `--output-raw`). None of the Rust projects streams inside a sentence. 🟢
4. Model management ranges from manual `curl` scripts (Kokoros, sherpa) to pinned, hashed downloads on first use (loqui, kokoro-cli). 🟢

### 2.5 Comparable CLIs (UX reference)

#### macOS `say` 🟢 (from `man say` and running it locally)

```
say [-v voice] [-r rate] [-o outfile [audio format options] | -n name:port | -a device] [-f file | string ...]
```

| Flag | Meaning |
|---|---|
| `string ...` | Text to speak; multiple args are joined with spaces |
| `-f file`, `--input-file=file` | Read text from a file; `-` or no text means stdin |
| `-v voice`, `--voice=voice` | Voice; `-v '?'` lists voices (177 lines here: `Name  locale  # sample sentence`) |
| `-r rate`, `--rate=rate` | Words per minute |
| `-o out`, `--output-file=file` | Write a file (AIFF by default; format inferred from the extension) |
| `--file-format=fmt` | AIFF, AIFC, caff, m4af, WAVE, BW64, RF64, W64f, NeXT, Ogg (flac/opus), 3gpp, Sd2f…; `?` lists them |
| `--data-format=fmt` | `aac`, `alac`, or LPCM spec `[BE\|LE][F\|I\|UI][8\|16\|24\|32\|64][@rate][/flags]`, e.g. `LEF32@8000`; `?` lists them |
| `--channels=n`, `--bit-rate=r`, `--quality=0..127` | Encoder options |
| `-a ID\|name`, `--audio-device=` | Output device; `?` lists them |
| `-n name:port`, `--network-send=` | AUNetSend (legacy) |
| `--progress` | Progress meter |
| `-i`, `--interactive[=markup]` | Print the text and highlight each word as it is spoken (terminfo capability, colour, or `fg/bg`) |

Behaviour worth copying:

- When the input is a TTY, text is spoken **line by line**, and an output file then contains only the last line.
- Exit code 0 on success; diagnostics go to stderr.
- Default output is 22,050 Hz, 16-bit, mono AIFF (measured).
- Embedded commands inline such as `[[slnc 200]]` (from the man page example). Others (`[[rate]]`, `[[volm]]`, `[[pbas]]`, `[[inpt PHON]]`) come from Apple's Speech Synthesis Programming Guide 🟡.
- The default rate in wpm is not printed by `say` 🔴; commonly cited as about 175–200 wpm.

#### Piper (`piper1-gpl`, Python CLI) 🟢

`python3 -m piper -m MODEL [-c CONFIG] [-i FILE ...] [-f OUT.wav | -d DIR [--output-dir-naming timestamp|text] | --output-raw] [-s SPEAKER_ID] [--length-scale F] [--noise-scale F] [--noise-w-scale F] [--sentence-silence SECS] [--volume F] [--no-normalize] [--data-dir DIR ...] [--cuda] [--debug] -- TEXT`.

- Voices download with `python3 -m piper.download_voices [VOICE] [--data-dir DIR]` into the **current directory** by default.
- Without `-f`, the audio plays through `ffplay`.
- Raw espeak phonemes go inline as `[[ … ]]`.
- The old C++ binary (v1.2.0) reads text from stdin: `echo hi | piper -m voice.onnx -f out.wav` or `--output-raw`.
- The docs warn that the CLI "can be slow… it needs to load the model each time".

#### sherpa-onnx `sherpa-onnx-offline-tts` 🟢

- VITS/Piper: `--vits-model --vits-tokens --vits-lexicon --vits-data-dir --vits-dict-dir --vits-noise-scale --vits-noise-scale-w --vits-length-scale`
- Kokoro: `--kokoro-model --kokoro-voices --kokoro-tokens --kokoro-lang --kokoro-data-dir --kokoro-dict-dir --kokoro-lexicon --kokoro-length-scale`
- General: `--num-threads --provider --debug`, `--tts-rule-fsts --tts-rule-fars` (normalization FSTs), `--tts-max-num-sentences`, `--tts-silence-scale`, `--sid`, `--speed`, `--output-filename`, then the text as a positional argument
- Also Matcha, Kitten, Pocket TTS, Supertonic and ZipVoice flag families
- `data-dir` (espeak) is deprecated and ignored since 2.0

#### espeak-ng 🟢 (`src/espeak-ng.1.ronn`)

- Input: `-f FILE`, `--stdin`
- Voice and prosody: `-v VOICE`, `-s WPM` (speed), `-p 0-99` (pitch), `-P` (pitch range), `-a 0-200` (amplitude), `-g UNITS` (word gap in 10 ms), `-k` (capital emphasis), `-l` (line length)
- Output: `-w FILE.wav`, `--stdout`, `-d DEVICE`, `--split=MIN`, `-q` (quiet, no audio)
- Encoding and markup: `-b` (encoding), `-m` (SSML/markup), `--ssml-break=%`
- Phonemes: `-x` (phoneme mnemonics), `-X`, `--ipa[=n]`, `--tie`, `--sep`, `--pho`, `--phonout=FILE`
- Other: `-z`, `-D`, `--punct="chars"`, `--path`, `--voices[=lang]`, `--compile*`

`espeak-ng -q --ipa -v en-us "text"` is the standard way to get Piper-compatible phonemes.

#### Festival 🟡 (from memory and docs, not run locally)

- `festival --tts file.txt` (or stdin) speaks the text.
- `text2wave [-o out.wav] [-otype riff] [-eval '(voice_kal_diphone)'] file.txt` writes a file.
- `festival -b '(SayText "hi")'` runs in batch mode.
- An interactive Scheme REPL; voices are selected with `(voice_NAME)`.

#### Kokoros `koko` 🟢

- Subcommands: `koko text "…" [-o out.wav] [--timestamps]`, `koko file in.txt -o "out_{line}.wav"`, `koko stream` (stdin lines in, WAV out), `koko openai [--instances N]`.
- Style mixing `--style af_sky.4+af_nicole.5`.

**UX conclusion for the spec:** mirror `say`'s surface exactly: positional text, `-f`, `-v` (with `?`), `-r` wpm, `-o` (format from extension), `--file-format`/`--data-format` (subset: WAVE and AIFF LPCM), `-a`, `--progress`, `-i`. Add backend-specific options as long flags, as sherpa does: `--engine formant|piper|kokoro|system`, `--length-scale`, `--noise-scale`, `--speaker`, and phoneme input via `--phonemes` or inline `[[ ]]`. Keep `say`'s TTY line-by-line behaviour, which doubles as natural sentence chunking for low latency. 🔴 (design recommendation)

### 2.6 Model distribution 🟢

**Piper voices:**

- Hosted at **`huggingface.co/rhasspy/piper-voices`**. Repo licence tag: MIT (still updated 2026-11-19). Layout: `<lang>/<locale>/<name>/<quality>/<locale>-<name>-<quality>.onnx`, plus `.onnx.json` and `MODEL_CARD`. Download URL: `https://huggingface.co/rhasspy/piper-voices/resolve/main/<path>`.
- `voices.json` is a machine-readable index: **183 voices, 58 locales, 53 languages**, with per-file `size_bytes` and `md5_digest`.
- Sizes: x_low 20.6–28 MB, low 28–70 MB, medium 63–79 MB, high 63–137 MB. `en_US-lessac-medium` is 63.2 MB.
- **Licences are per voice and come from the dataset.** Piper's LICENSING.md: the project "makes no license grant of its own over voice models". `en_US-lessac-*` comes from the Blizzard 2013 Lessac dataset, which has its own licence. COMMERCIAL_USE.md lists voices whose whole fine-tune chain is permissive. English ones: `en_US-ljspeech-high/medium`, `en_US-kristin-medium`, `en_US-john-medium`, `en_US-norman-medium`, `en_GB-cori-high` (public domain), `en_US-libritts-high`, `en_US-hifi_tts_92-medium`, `en_GB-alba-medium`, `en_GB-cori-medium` (CC BY 4.0), `en_US-joe-medium`, `en_US-mike-medium`, `en_US-kathleen-medium` (CC0), and `en_GB-jenny_dioco-medium` (custom licence: commercial use allowed, the voice must be called "Jenny"). `en_US-amy-medium` and `en_US-hfc_female-medium` (CC BY-NC-SA) are not on the list.
- sherpa-onnx repackages Piper voices as `vits-piper-*.tar.bz2` (plus int8/fp16 variants and a `tokens.txt`) on its `tts-models` GitHub release and as `csukuangfj/vits-piper-*` on HF.

**Kokoro-82M:**

| Source | Files | Size |
|---|---|---|
| `huggingface.co/hexgrad/Kokoro-82M` (Apache-2.0) | `kokoro-v1_0.pth` | 327 MB |
| | `voices/*.pt` | — |
| `huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX` (Apache-2.0) | `onnx/model.onnx` (fp32) | 325.5 MB |
| | `model_fp16` | 163 MB |
| | `model_quantized` (int8) | 92 MB |
| | `model_q8f16` | 86 MB |
| | `model_q4f16` | 155 MB |
| | `model_uint8` | 178 MB |
| | `voices/<name>.bin` (55 files of 510×256 f32) | 512 KB each |
| | `tokenizer.json` | — |
| `onnx-community/Kokoro-82M-v1.0-ONNX-timestamped` | Variant with word-timing outputs (used by Kokoros `--timestamps`) | — |
| GitHub `thewh1teagle/kokoro-onnx` release `model-files-v1.0` | `kokoro-v1.0.onnx` | 325 MB |
| | `kokoro-v1.0.fp16.onnx` | 177 MB |
| | `kokoro-v1.0.int8.onnx` | 92 MB |
| | `voices-v1.0.bin` (all voices in one npz-style file) | 28 MB |

These are the files most Rust ports download. sherpa-onnx ships `kokoro-multi-lang-v1_0` and `kokoro-int8-multi-lang-v1_0` tarballs.

**How tools cache:**

| Tool | Location |
|---|---|
| Piper | CWD, or `--data-dir` |
| Kokoros | `~/.cache/kokoros/`; models in `./checkpoints` |
| kokoro-cli | XDG data dir plus a `setup` command with hashes |
| loqui | Pinned HF revisions, SHA-256, `--offline`, `fetch` |
| sherpa | Manual |
| `ort` | Its own runtime binaries in `~/Library/Caches/ort.pyke.io/` |

- `hf-hub` 1.0.0 uses the standard `~/.cache/huggingface/hub` layout, but 1.0 pulls tokio, reqwest and hf-xet, which is heavy for a CLI.
- `dirs` 7.0.0 / `directories` 6.0.0 give `~/Library/Caches/<app>` (macOS `cache_dir`), `~/Library/Application Support/<app>` (`data_dir`) and `$XDG_CACHE_HOME`/`~/.cache` and `$XDG_DATA_HOME` on Linux.

---

## 3. Comparison tables

### 3.1 Inference runtimes for `speak` (Apple M2 Max, measured unless marked)

| | `ort` 2.0.0-rc.15 (ORT 1.31) | `tract` 0.23.10 | `candle` 0.11 (candle-onnx) | `burn` 0.22 + burn-onnx |
|---|---|---|---|---|
| Runs Piper (VITS) ONNX | **Yes** 🟢 | **No**: Range/NonZero runtime symbols (#2931) 🟢 | No: missing ops (PR #3792 open) 🟢 | No: generated code doesn't compile 🟢 |
| Runs Kokoro ONNX | **Yes** 🟢 | **No**: decoder Concat symbolic dims (#1634) 🟢 | Unknown 🔴 (native ports exist) | Runs, output diverges (r ≈ 0.75–0.83) 🟢 |
| Piper RTF / load | 0.021 / 0.8 s 🟢 | — | — | — |
| Kokoro fp32 RTF / load | 0.17 / 0.42 s 🟢 | — | — | — |
| Stripped binary (probe) | 19.3 MB static 🟢 | 17.9 MB 🟢 | ? | ? |
| Build needs | Prebuilt download (or C++ build); macOS link flag | Pure Rust | Pure Rust | Pure Rust, build-time codegen |
| Apple acceleration | CoreML EP (slower for Kokoro 🟡; crashes Piper via auto policy 🟢) | NEON CPU only | Metal, Accelerate | Metal/wgpu |
| Maturity | rc tags, very active, single core maintainer | Active, Sonos-backed | Active, HF-backed | Active, Tracel-backed |

### 3.2 Frontend options

| Option | Piper-compatible | Kokoro-compatible | Licence of shipped binary | Offline OOV handling |
|---|---|---|---|---|
| espeak-rs (C, static) | Exact 🟢 | Via IPA mapping (Kokoros does this) 🟢 | GPL-3 | Rules, all words |
| `espeak-ng` pure-Rust port | Exact IPA (oracle-tested) 🟢 | Via mapping | GPL-3 | Rules |
| `espeak-ng` executable via subprocess | Exact | Via mapping | Ours stays MIT (aggregate) 🔴 | Rules; needs install |
| misaki-rs `--no-default-features` | No (different symbol set) 🔴 | Exact (parity-tested) 🟢 | MIT/Apache | Drops OOV words (`❓`) |
| loqui-g2p | No 🔴 | Claimed exact plus neural OOV 🟡 | MIT | Neural BART fallback |
| Lexicon (sherpa style, CMUdict/OpenEPD) | Only with mapped lexicon | Partly | Depends on data (OpenEPD is CC-BY-SA) | None |
| piper-plus-g2p | Unknown for stock voices 🔴 | — | MIT | Own rules |

---

## 4. What it means for `speak`

1. **Use `ort` for the neural backends, pinned exactly to `=2.0.0-rc.15`, with static linking (the default download).** It is the only Rust runtime that runs Piper and Kokoro today, and it was verified here end to end with correct whisper transcripts. 🟢
   - Always call `.with_auto_device(AutoDevicePolicy::PreferCPU)` on macOS. Without it, Piper crashes in CoreML/BNNS.
   - Add `-Clink-arg=-fapple-link-rtlib` to `.cargo/config.toml` for aarch64-apple-darwin.
   - Treat CoreML as an opt-in experiment (`--provider coreml`), not a default.
2. **Hide the runtime behind our own `Backend`/`Synth` trait** (input: phoneme IDs plus a voice handle; output: f32 PCM at the model rate), not behind `ort` types. Then a tract, burn or candle backend can be added later. `ort`'s `alternative-backend` plus `ort-tract` is a cheap way to re-test tract once sonos/tract#2931 lands. 🔴
3. **Startup latency comes from model load, not synthesis.** Piper takes 0.8 s to load and 0.09 s to speak a sentence. Options for the spec:
   - Cache the ORT-optimized model next to the download; this measured 0.71 s vs 0.81 s.
   - Try ORT format (`.ort`), memory-mapped weights and fewer intra-op threads (unmeasured 🔴).
   - Keep the formant backend as the instant-start default.
   - Add an optional resident daemon (`speak --serve`) later.
4. **Make the frontend pluggable and decide the licence explicitly.**
   - For an MIT-licensed `speak` with Kokoro: misaki-rs (`default-features = false`) or loqui-g2p, plus text-processing-rs or num2words for normalization.
   - For Piper voices, which need espeak IPA, choose between:
     - (a) a cargo feature `espeak` that links espeak-rs and makes such builds GPL-3;
     - (b) shelling out to an installed `espeak-ng -q --ipa`;
     - (c) the pure-Rust `espeak-ng` crate (GPL).

   For a personal, never-distributed tool, (a) is simplest. 🔴
5. **Accept raw phonemes** (`--phonemes` or inline `[[ … ]]`), as every reference tool does. Frontend and backend can then be tested and compared on their own, and the formant synth can share the same phoneme input. 🟢 (pattern) / 🔴 (recommendation)
6. **Audio:**
   - Use `cpal` directly with a lock-free ring buffer (`rtrb`), or `rodio` with `default-features = false, features = ["playback"]`, so playback can start on the first sentence chunk.
   - Use `rubato` to resample 22.05/24 kHz to the device rate.
   - Use `hound` for `-o *.wav`; hand-write AIFF if we want `say`-compatible default output.
7. **Formant backend:**
   - Write our own Klatt synth on `realfft`/`rustfft`.
   - Use `klattsch-core` (MIT) as the reference and golden-WAV oracle: an ARPAbet schedule in, a realtime-safe `process(&mut [f32])` out.
   - Look at, but do not copy, espeak-ng-rs's synth (GPL).
8. **Models:**
   - Download over plain HTTPS (`ureq` 3) from HF `resolve/<commit-sha>/…` URLs pinned in code.
   - Verify the size and the MD5 from Piper's `voices.json`, or our own SHA-256.
   - Store in `dirs::cache_dir()/speak` (`~/Library/Caches/speak`, `~/.cache/speak`).
   - Provide `speak --download VOICE` and an `--offline` refusal mode, as loqui does.
   - Default to commercially clean voices (e.g. `en_US-ljspeech-medium` or `en_US-kristin-medium` for Piper, `af_heart` for Kokoro) and show each voice's licence in `-v '?'`.
9. **Reference, don't depend:**
   - sherpa-onnx's Rust crate is a good A/B oracle (Piper, Kokoro, Matcha, Kitten from one API), but it bundles its own ORT and lexicon-only frontends.
   - Kokoros, piper-rs and kokoro-cli are short enough to read in an hour and copy patterns from (all MIT/Apache).
   - The `tts` crate offers an optional `--engine system` backend on AVSpeechSynthesizer.

### Recommended crate stack

| Purpose | Crate | Version | Notes |
|---|---|---|---|
| CLI | `clap` (derive) | 4.6.8 | Mirror `say` flags |
| Neural runtime | `ort` | `=2.0.0-rc.15` | Static, `PreferCPU`, `ndarray` feature optional |
| Runtime experiments (later) | `ort-tract` / `tract-onnx`, `candle-core`, `burn` | 0.5.0+0.23 / 0.23.10, 0.11.1, 0.22.1 | Behind a feature, for comparison only |
| Audio out | `cpal` (+ `rtrb`), or `rodio` without default features | 0.18.2 / 0.23.0 | Streaming playback |
| Resampling | `rubato` | 5.1.0 | 22.05/24 kHz to device rate |
| WAV | `hound` | 3.5.1 | `-o out.wav` |
| FFT / DSP | `realfft` + `rustfft` | 3.5.1 / 6.4.1 | Formant synth, analysis |
| G2P for Kokoro | `misaki-rs` (`default-features = false`), alternatively `loqui-g2p` | 0.11.0 / 0.4.0 | GPL-free |
| G2P for Piper | `espeak-rs` behind a GPL `espeak` feature, or an `espeak-ng` subprocess | 0.2.0 | Decide the licence |
| Normalization | `num2words` (start) and `text-processing-rs` (NeMo TN, later) | 1.2.0 / 0.3.4 | Numbers, dates, money |
| Unicode | `unicode-normalization` | 0.1 | NFKC |
| Model config | `serde` + `serde_json` | 1 | Piper `.onnx.json`, Kokoro `tokenizer.json` |
| Download and cache | `ureq` + `sha2` + `dirs` | 3.5.2 / 0.10+ / 7.0.0 | Avoid `hf-hub` 1.0 (tokio/reqwest/xet) |
| Formant reference (dev-dependency or oracle) | `klattsch-core`, `klattsch-text` | 0.1.2 | MIT |
| System voice backend (optional) | `tts` | 0.26.3 | AVSpeechSynthesizer / speech-dispatcher |

---

## 5. Sources

### Measured here (🟢)

Probe crate (scratchpad, deleted after use): `ort =2.0.0-rc.15` (default features) and `tract-onnx 0.23.10` loading `en_US-lessac-medium.onnx` and onnx-community Kokoro `model.onnx` with `af_heart.bin`; outputs transcribed with whisper.cpp large-v3-turbo. `say`, `man say`, `say -v '?'`, `say --file-format='?'`, `afinfo` run locally.

### crates.io API (`https://crates.io/api/v1/crates/<name>`)

ort, ort-sys, ort-tract, ort-candle, candle-core, candle-onnx, burn, burn-import, tract-onnx, cpal, rodio, hound, rubato, rustfft, realfft, symphonia, espeakng-sys, espeak-rs, espeakng, espeak-ng, espeak-ng-data-phonemes, piper-rs, kokoro-tts, kokorox, kokoro-en, kokoro-micro, kokoro-cli, kokoroxide, kittentts, any-tts, omp-voice-kokoro, loqui, loqui-g2p, loqui-kokoro, sherpa-rs, sherpa-onnx, sherpa-onnx-sys, tts, natural-tts, cmudict, cmudict-fast, open-english-pronouncing-dictionary, num2words, text2num, text-processing-rs, misaki-rs, piper-plus-g2p, floravox-g2p, voicegarden-lexicons, g2p2-core, phonetisaurus-g2p, grapheme_to_phoneme, klatt, klattsch-core, svara, phoneme-synth, hf-hub, dirs, directories, clap, ureq, ttspico.

### Repositories and docs

- https://github.com/pykeio/ort: `Cargo.toml` features; docs `setup/linking.mdx`, `misc/prebuilt-binaries.mdx`, `backends/index.mdx`, `backends/tract.mdx`
- https://github.com/pykeio/ort/issues/662, https://github.com/pykeio/ort/issues/666 (CoreML auto-policy), https://github.com/pykeio/ort/issues/360 (`___isPlatformVersionAtLeast`)
- https://github.com/sonos/tract, https://github.com/sonos/tract/issues/1634 (Kokoro), https://github.com/sonos/tract/issues/2928, https://github.com/sonos/tract/issues/2931 (MMS VITS)
- https://github.com/huggingface/candle, https://github.com/huggingface/candle/pull/3792 (VITS ops), https://github.com/huggingface/candle/pull/3420 (closed Kokoro PR)
- https://github.com/tracel-ai/burn, https://github.com/tracel-ai/burn-onnx/issues/18, https://github.com/tracel-ai/burn-onnx/issues/371, https://github.com/tracel-ai/burn-onnx/pull/679
- https://github.com/RustAudio/cpal, https://github.com/RustAudio/rodio, https://github.com/ruuda/hound, https://github.com/HEnquist/rubato, https://github.com/HEnquist/realfft, https://github.com/ejmahler/RustFFT
- https://github.com/thewh1teagle/piper-rs (`Cargo.toml`, `src/lib.rs`, `crates/espeak-rs-sys/build.rs`)
- https://github.com/lucasjinreal/Kokoros (README, tree, `kokoros/Cargo.toml`)
- https://github.com/ghchinoy/kokoro-rs (kokoro-cli README, CoreML vs CPU numbers)
- https://github.com/Awakened-Labs/loqui (README, `crates/loqui-g2p/Cargo.toml`)
- https://github.com/MicheleYin/misaki-rs (README, `Cargo.toml`)
- https://github.com/eugenehp/espeak-ng-rs (README, `Cargo.toml`)
- https://github.com/tgies/klattsch-rs, https://github.com/TTWNO/klatt
- https://github.com/ndarilek/tts-rs, https://github.com/CodersCreative/natural-tts, https://github.com/mzdk100/kokoro, https://github.com/TM9657/any-tts
- https://github.com/FluidInference/text-processing-rs, https://github.com/ayutaz/piper-plus
- https://github.com/k2-fsa/sherpa-onnx: `sherpa-onnx/rust/sherpa-onnx/README.md`, `sherpa-onnx-sys/Cargo.toml` and `build.rs`, `csrc/sherpa-onnx-offline-tts.cc`, `offline-tts-*-model-config.cc`; PR https://github.com/k2-fsa/sherpa-onnx/pull/4055, issues https://github.com/k2-fsa/sherpa-onnx/issues/3731 and https://github.com/k2-fsa/sherpa-onnx/issues/4139; release `tts-models`
- https://github.com/thewh1teagle/sherpa-rs (deprecation notice)
- https://github.com/rhasspy/piper (archived), https://github.com/OHF-Voice/piper1-gpl (`docs/CLI.md`, `docs/LICENSING.md`, `docs/COMMERCIAL_USE.md`, `src/piper/__main__.py`)
- https://github.com/espeak-ng/espeak-ng (`src/espeak-ng.1.ronn`), https://github.com/hexgrad/kokoro, https://github.com/hexgrad/misaki
- https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

### Hugging Face

- https://huggingface.co/rhasspy/piper-voices (`voices.json`, `en/en_US/lessac/*`, MODEL_CARD)
- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX (`onnx/`, `voices/`, `tokenizer.json`)
- https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped
- https://huggingface.co/willwade/mms-tts-multilingual-models-onnx (tract VITS test case)

### Papers

Area D is about software, and none of the tools above (ort, tract, candle, burn, Piper, Kokoro, sherpa-onnx) has its own paper or technical report, so no new PDFs were added. The papers behind the models we run are already in `papers/`, downloaded by areas A–C:

- `2021-vits.pdf` (Piper's architecture)
- `2023-styletts2.pdf` and `2022-istftnet.pdf` (Kokoro's architecture and decoder; the iSTFT/STFT ops are where tract and burn break)
- `1980-klatt-cascade-parallel.pdf` (klattsch and the formant backend)
