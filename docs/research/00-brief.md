# Research brief: building a Rust voice synthesizer

## Objective

Learn how text-to-speech systems are built, from classic formant synthesis to current neural models. The findings feed `spec.md` for `speak`: a Rust command-line tool like macOS `say` that we build ourselves to experiment with. It runs locally and offline, starts fast, plays audio straight away, and we can swap its parts out to compare synthesis techniques.

## Key design questions

1. **Pipeline shape:** which stages does every TTS system have (text normalization → G2P/phonemes → prosody → acoustic model → vocoder → audio out)? Which stages do end-to-end models (VITS, Kokoro) merge?
2. **First backend:** for a Rust tool, what should come first? Options: a hand-written formant synthesizer (Klatt), a small neural model run through ONNX (Piper/Kokoro), or both, side by side?
3. **Inference runtime:** `ort` (ONNX Runtime), `candle`, `burn` or `tract`? Weigh binary size, CPU speed, Apple Silicon support, model coverage and dependency weight.
4. **Text frontend:** espeak-ng (C library, GPL) for phonemes, a pure-Rust G2P, or a lexicon (CMUdict) plus a fallback? How do Piper and Kokoro do it?
5. **Latency and streaming:** how do systems get audio playing before the whole utterance is synthesized? Think sentence chunking, streaming vocoders, real-time factor.
6. **Voices and control:** how are voices packaged (model files, speaker embeddings, style vectors)? How are rate, pitch and volume exposed? Is SSML worth supporting?
7. **Quality versus size:** model size, real-time factor on a laptop CPU and perceived quality (MOS) for Piper, Kokoro, StyleTTS 2, F5-TTS and XTTS.
8. **Licensing:** which models, weights and phonemizers can a personal or MIT tool ship? Watch for espeak-ng GPL, and for weights under non-commercial licences.

## Areas (one agent each)

| # | Area | Covers |
|---|---|---|
| A | Classic synthesis + text frontend | Klatt/formant, source-filter, LPC, diphone/concatenative, PSOLA, unit selection (Festival, MBROLA), HMM/statistical parametric (HTS), WORLD vocoder. Text normalization, G2P (CMUdict, espeak-ng, neural G2P, misaki), prosody/ToBI, SSML |
| B | Neural acoustic models | Tacotron 2, FastSpeech 2, Glow-TTS, VITS, YourTTS, StyleTTS 2, Kokoro, Piper, Matcha-TTS, F5-TTS, E2-TTS, XTTS, Parler-TTS, codec LMs (VALL-E, SoundStorm, Bark, Orpheus/Llama-based TTS) |
| C | Vocoders, codecs, streaming | WaveNet, WaveRNN, LPCNet, WaveGlow, MelGAN, HiFi-GAN, UnivNet, BigVGAN, Vocos, iSTFTNet; neural codecs (EnCodec, DAC, Mimi, SNAC); streaming and real-time factor |
| D | Rust ecosystem + existing tools | `ort`, `candle`, `burn`, `tract`; `cpal`, `rodio`; espeak-ng bindings, Rust phonemizers, existing Rust TTS crates and projects (piper-rs, kokoro-rs, sherpa-rs, tts-rs); comparable CLIs (piper, sherpa-onnx, macOS `say`/AVSpeechSynthesizer, espeak-ng, Festival) |

## Report template (per area)

1. Overview: what the area covers and the timeline of techniques
2. Techniques: for each one, how it works, its inputs and outputs, model size, speed, quality, licence and key paper
3. Comparison table
4. What it means for `speak`: concrete recommendations
5. Sources: every URL, and the papers downloaded to `papers/`

Mark every section with a confidence level: 🟢 primary source (paper, repo, docs), 🟡 secondary (blog, benchmark), 🔴 inferred.

Papers go in `docs/research/papers/` as `<year>-<short-name>.pdf`, for example `2021-vits.pdf`. Each area also lists its papers in its report.
