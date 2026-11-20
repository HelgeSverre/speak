# Research synthesis for `speak`

Input: `00-brief.md` and the four area reports (A–D), researched 2026-11-20 on an Apple M2 Max.
Output: the decisions below. They are binding for `/spec.md` and `/PLAN.md` in the repo root.

Citations are `A §2.1.4` = report A, section 2.1.4.

## 0. Direction

`speak` is our own implementation. We write the synthesis, the DSP and the text frontend ourselves and do not run pretrained models. The reports surveyed neural systems (areas B, C, most of D) to answer the brief; under this direction they are background reading and a source of design ideas, not a parts list. Report A is the working reference.

| Allowed | Not allowed |
|---|---|
| Our own Rust code for every stage | Pretrained weights of any kind (Piper, Kokoro, Kitten, Vocos, HiFi-GAN, G2P models) |
| General-purpose libraries: argument parsing, audio I/O, FFT, resampling | Inference runtimes in the default build (`ort`, tract, candle, burn) |
| CMUdict as a pronunciation **data** file, optional, lookup only | Third-party phonemizers and frontends (espeak-ng, misaki, num2words-style libraries) |
| Voice data we produce: our Klatt output, our own recordings | Third-party voice databases (MBROLA, Festival voices) |
| Stretch: a small model we train from scratch on public-domain LJSpeech | Fine-tuning or distilling from any existing checkpoint |
| `say` and whisper as test instruments | `say` or whisper inside the product or as data sources |

## 1. Decisions for the eight design questions

| # | Question | Decision |
|---|---|---|
| 1 | Pipeline shape | Seven stages, each a trait with our own implementations: normalize → chunk → G2P (lexicon + letter-to-sound) → postlexical rules → prosody (durations, F0) → synthesizer → audio sink. The seam between frontend and backends is a timed phone stream (`.pho`). |
| 2 | First backend | Hand-written Klatt formant synthesizer; it is the main product. Second: diphone concatenation with LPC resynthesis, with an inventory bootstrapped from our own Klatt output and replaceable by our own recordings. |
| 3 | Inference runtime | None. The stretch milestone uses `burn`, because it trains as well as runs, for a model we train ourselves. |
| 4 | Text frontend | All ours: rule-based normalizer, built-in exception lexicon, NRL-style letter-to-sound rules, and a CART letter-to-sound tree we train from CMUdict. CMUdict is optional lookup data (BSD-2-Clause-style licence). |
| 5 | Latency / streaming | Both backends render far faster than real time and stream 20 ms blocks through a ring buffer. One sentence per chunk. Target: first audio within 100 ms of process start. |
| 6 | Voices and control | Klatt voice = parameter preset (F0, formant scale, open quotient, tilt, breathiness, source model). Diphone voice = an inventory file we build. Rate, pitch and volume work on both because prosody and excitation are ours. `say`-style `[[...]]` commands; no SSML. |
| 7 | Quality versus size | Accepted: DECtalk/eSpeak-class speech, zero model files, a few MB of locally built voice data. Neural-quality speech is out of reach by design. Quality is tracked as word error rate against `say`. |
| 8 | Licensing | Code MIT OR Apache-2.0, all written here. No GPL code read or ported. CMUdict data is BSD-style. No weights to license. Voices we record are ours. |

### 1.1 Pipeline shape

Every TTS system has normalization, G2P and prosody in front of an acoustic stage (A §1, B §1). End-to-end neural models absorb prosody, acoustics and vocoder into one network; classic systems keep them as separate, inspectable stages, which is what we want for experiments. Following A §4.1 the interface between frontend and backends is the MBROLA `.pho` idea: phone, duration, F0 targets. Both of our backends consume exactly that, so the same phones, timing and pitch can be sent to either.

### 1.2 Backends

Klatt (A §2.1): published algorithms, a few hundred lines of DSP, no data, every parameter explicit. Most of the work and most of the learning is in the rule program (A §2.1.6).

Second technique, chosen for effort versus learning (A §2.2–2.6):

| Candidate | Needs | Verdict |
|---|---|---|
| Diphone + LPC resynthesis | a diphone inventory; no pitch marks for the basic mode | **chosen**: measured filter instead of ruled filter, concatenation, parametric joins; inventory can be generated from our own Klatt tracks, so it is buildable without a microphone |
| Diphone + TD-PSOLA | recordings with reliable pitch marks | added later as an excitation mode on the LPC residual and as a resynthesis vocoder |
| Unit selection | hours of labelled recordings | too much data for too little insight per byte (A §2.4) |
| HMM / statistical parametric | corpus, full-context labeller, training, vocoder | a large project of its own (A §2.5) |
| WORLD-style vocoder | nothing | analysis tool, not a TTS backend (A §2.6) |

Diphones are recorded once in monotone nonsense words (A §2.3); Festival's diphone voices use residual-excited LPC rather than PSOLA (A §2.3). We follow that line and add our own autolabelling by dynamic time warping against Klatt-rendered guide prompts.

### 1.3 Runtime

D §2.1 found `ort` to be the only runtime that executes Piper and Kokoro. That question is moot: nothing is executed that we did not write. For the stretch model, `burn` is preferred over candle because training is first-class there.

### 1.4 Text frontend

- Lexicon first, rules as fallback is what every serious system does (A §2.8). Ours: user lexicon → built-in exceptions written by us → CMUdict if the file is present → morphology → letter-to-sound.
- Letter-to-sound, two of our own implementations to compare: hand-written context rules in the NRL/Elovitz style (A §2.8: about 400 rules, public-domain report), and a CART trained by our code on CMUdict (Black, Lenzo & Pagel 1998: 58 % of held-out words fully correct).
- CMUdict (A §2.8): 135,166 entries, ARPAbet with stress, BSD-style licence. It is a word list, not a model. It also serves as the training data for our tree.
- Normalization by hand-written rules (A §2.7): production systems keep rule grammars because neural normalizers make unacceptable errors such as reading the wrong number.
- Prosody: Klatt's duration model and a declination-plus-tune F0 model (A §2.1.7, A §2.9).
- No word may be dropped. The lexicon-only frontend in sherpa-onnx 2.0 silently drops numbers and unknown words (D §2.3); that is the failure to avoid.

### 1.5 Latency

Formant and LPC synthesis cost a handful of second-order filters per sample; A §2.1.8 estimates thousands of times real time. With no model to load, time to first audio is argument parsing plus audio device open. For reference, `say -o x.aiff "hello world"` takes 0.55–0.65 s wall time here (D §2.1, re-measured). Chunking (C §2.16) is kept at sentence level for prosody, not for latency.

### 1.6 Voices and control

| Control | Klatt | Diphone LPC |
|---|---|---|
| Voice | preset of source and tract parameters | inventory file (`boot` from Klatt, or recorded by us) |
| Rate | duration rules | same durations; frames time-warped |
| Pitch | F0 rules drive the glottal source | same F0 targets drive the excitation |
| Voice quality | open quotient, tilt, flutter, diplophonia, breathiness; three source models | excitation mode: pulse, glottal, residual |
| Phoneme input | ARPAbet or `.pho` | same |

### 1.7 Quality versus size

The brief asked about Piper, Kokoro, StyleTTS 2, F5-TTS and XTTS. Report B has those numbers (B §3); none of them is used. What we get instead:

| | Klatt | Diphone LPC (`boot`) | Diphone LPC (own recordings) |
|---|---|---|---|
| Data on disk | 0 | ~6 MB, built locally in seconds | ~6–40 MB plus the recordings |
| Expected quality | intelligible, robotic | Klatt through an LPC coder | recognisably the speaker, buzzy |
| Startup | none | read one file | read one file |

### 1.8 Licensing

| Item | Licence | Policy |
|---|---|---|
| `speak` code | MIT OR Apache-2.0 | written here |
| Library crates (clap, cpal, hound, rubato, realfft, serde, …) | MIT / Apache-2.0 / BSD | allowed |
| CMUdict | BSD-2-Clause-style, © Carnegie Mellon University | optional data; notice reproduced in `NOTICE` |
| NRL letter-to-sound report (Elovitz et al. 1976) | US government work | reference for rule writing |
| Klatt 1980 / 1990 papers | published algorithms | implemented from the papers |
| GPL implementations (espeak-ng incl. `klatt.c`, rhdunn/klatt, klatt80, Praat, svara), Festival/MBROLA code and voices, DECtalk | GPL / AGPL / non-commercial / unclear | not read into the codebase, not used |
| LJSpeech (stretch) | public domain | training data only |

Source: A §4.7, A §2.1.8.

## 2. Backend comparison matrix

| | Klatt | Diphone LPC |
|---|---|---|
| Technique | cascade/parallel formant synthesis by rule | concatenated diphones, LPC lattice resynthesis |
| Where the vocal-tract filter comes from | rule tables (formant targets and transitions) | linear prediction of audio |
| Excitation | KLGLOTT88, LF or impulse; frication and aspiration noise | pulse, KLGLOTT88 or stored residual with PSOLA; noise |
| Input | timed phones + F0 targets | the same |
| Voice data | none | inventory of ~2,000 diphones we build |
| Sample rate | 16 kHz | 16 kHz |
| Coarticulation | computed (ranks, loci, context rules) | recorded inside each diphone |
| Joins | none (continuous tracks) | parameter interpolation at phone midpoints |
| What it teaches | source-filter theory, acoustic phonetics, rule writing | LPC, unit inventories, alignment (DTW), PSOLA |
| Reports | A §2.1 | A §2.2, A §2.3 |

Stretch only: a FastSpeech-2-shaped model we train from scratch in burn on LJSpeech, with our Griffin-Lim for the waveform (B §2.4, C §2.1; `spec.md` §17).

## 3. Facts verified while writing this synthesis (2026-11-20)

- Klatt 1980 Tables I–III were re-read from `papers/1980-klatt-cascade-parallel.pdf`; the values in `spec.md` §8.1.9 come from there. Klatt 1987 Table II lists the duration rules in words only; the percentages come from Klatt 1979 via secondary sources and are starting values.
- CMUdict at `cmusphinx/cmudict` commit `74790861f652b15e4ac49015a90074ad62a27690`: `cmudict.dict` is 3,618,488 bytes, 135,166 lines, SHA-256 `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`; the `LICENSE` file is the two-clause BSD text with a CMU copyright.
- `whisper` (Python CLI) is installed with `base.en` cached; `say`, `afinfo` and `ffmpeg` are available. They are used in tests only.

## 4. Index

### Reports

| File | Area | Use it for |
|---|---|---|
| `00-brief.md` | Objective, the eight questions, report template | scope |
| `A-classic-and-frontend.md` | Klatt, LPC, PSOLA, unit selection, HTS, WORLD; normalization, G2P, prosody, SSML | Klatt equations (§2.1), rule strategy (§2.1.6–7), G2P options (§2.8) |
| `B-neural-acoustic-models.md` | Tacotron 2 → VITS → Kokoro → codec LMs | background; FastSpeech 2 design for the stretch model (§2.4) |
| `C-vocoders-codecs-streaming.md` | Griffin-Lim → HiFi-GAN → Vocos; codecs; streaming | Rust iSTFT recipe (§2.14), Griffin-Lim (§2.1), chunking (§2.16) |
| `D-rust-ecosystem.md` | Runtimes, audio crates, G2P crates, existing Rust TTS, comparable CLIs, model hosting | `say` flags (§2.5), audio and DSP crates (§2.2); the runtime and model sections are background |

### Papers (`papers/`)

Classic synthesis and frontend (area A):

| File | Paper |
|---|---|
| `1980-klatt-cascade-parallel.pdf` | Klatt, Software for a cascade/parallel formant synthesizer |
| `1987-klatt-tts-review.pdf` | Klatt, Review of text-to-speech conversion for English |
| `1989-psola-diphones.pdf` | Charpentier & Moulines, Pitch-synchronous waveform processing using diphones |
| `1990-klatt-klatt-voice-quality.pdf` | Klatt & Klatt, Voice quality variations (KLSYN88, KLGLOTT88) |
| `1996-hunt-black-unit-selection.pdf` | Hunt & Black, Unit selection in a concatenative system |
| `1996-mbrola.pdf` | Dutoit et al., The MBROLA project |
| `1998-black-letter-to-sound.pdf` | Black, Lenzo & Pagel, Issues in building general letter to sound rules |
| `2013-hmm-speech-synthesis-overview.pdf` | Tokuda et al., Speech synthesis based on hidden Markov models |
| `2015-zen-spss-hmm-to-lstm.pdf` | Zen, Acoustic modeling in SPSS: from HMM to LSTM-RNN |
| `2015-seq2seq-g2p.pdf` | Yao & Zweig, Sequence-to-sequence models for G2P |
| `2016-neural-text-normalization.pdf` | Sproat & Jaitly, RNN approaches to text normalization |
| `2016-world-vocoder.pdf` | Morise et al., WORLD vocoder |
| `2022-byt5-g2p.pdf` | Zhu et al., ByT5 for multilingual G2P |

Neural acoustic models (area B):

| File | Paper |
|---|---|
| `2017-tacotron2.pdf` | Shen et al., Tacotron 2 |
| `2020-fastspeech2.pdf` | Ren et al., FastSpeech 2 |
| `2020-glow-tts.pdf` | Kim et al., Glow-TTS |
| `2021-vits.pdf` | Kim et al., VITS (Piper's architecture) |
| `2021-yourtts.pdf` | Casanova et al., YourTTS |
| `2022-styletts.pdf` | Li et al., StyleTTS |
| `2023-styletts2.pdf` | Li et al., StyleTTS 2 (Kokoro's ancestor) |
| `2023-matcha-tts.pdf` | Mehta et al., Matcha-TTS |
| `2023-vall-e.pdf` | Wang et al., VALL-E |
| `2023-soundstorm.pdf` | Borsos et al., SoundStorm |
| `2024-e2-tts.pdf` | Eskimez et al., E2 TTS |
| `2024-f5-tts.pdf` | Chen et al., F5-TTS |
| `2024-xtts.pdf` | Casanova et al., XTTS |
| `2024-parler-tts.pdf` | Lyth & King, Parler-TTS |
| `2024-cosyvoice2.pdf` | Du et al., CosyVoice 2 |
| `2025-calm-pocket-tts.pdf` | Kyutai, Continuous Audio Language Models (Pocket TTS) |

Vocoders and codecs (area C):

| File | Paper |
|---|---|
| `2016-wavenet.pdf` | van den Oord et al., WaveNet |
| `2018-wavernn.pdf` | Kalchbrenner et al., WaveRNN |
| `2018-lpcnet.pdf` | Valin & Skoglund, LPCNet |
| `2018-waveglow.pdf` | Prenger et al., WaveGlow |
| `2019-melgan.pdf` | Kumar et al., MelGAN |
| `2020-hifigan.pdf` | Kong et al., HiFi-GAN |
| `2022-istftnet.pdf` | Kaneko et al., iSTFTNet (Kokoro's decoder head) |
| `2022-bigvgan.pdf` | Lee et al., BigVGAN |
| `2023-vocos.pdf` | Siuzdak, Vocos |
| `2022-encodec.pdf` | Défossez et al., EnCodec |
| `2023-dac.pdf` | Kumar et al., DAC |
| `2024-moshi-mimi.pdf` | Défossez et al., Moshi / Mimi |

Area D added no papers.
