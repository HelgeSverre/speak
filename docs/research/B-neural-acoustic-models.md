# B. Neural acoustic models

Research for `speak`, a Rust CLI like `say` that runs offline on an Apple Silicon laptop CPU. This area covers how neural TTS models turn text or phonemes into audio, and which of them a Rust program can actually run.

Confidence markers: 🟢 primary source (paper, repo, model card, or our own measurement), 🟡 secondary (blog, benchmark, leaderboard), 🔴 inferred or unverified.

Research date: 2026-11-20. Model cards and repos were read through the GitHub and Hugging Face APIs on that date. The measurements in §2.1 were run on this machine (Apple M2 Max, 8 performance + 4 efficiency cores, onnxruntime 1.32.0 CPU provider through Python).

---

## 1. Overview 🟢

Neural TTS has gone through four generations. Each one changes what goes in, what comes out, and how much of the pipeline a single network covers.

| Generation | Years | Shape | Examples |
|---|---|---|---|
| **Two-stage autoregressive** | 2017–2019 | characters/phonemes → attention seq2seq → mel spectrogram → separate neural vocoder | Tacotron 2 (+WaveNet / WaveGlow / HiFi-GAN) |
| **Two-stage non-autoregressive** | 2019–2021 | phonemes → parallel encoder + explicit durations (+pitch/energy) → mel → vocoder | FastSpeech 2, Glow-TTS, Matcha-TTS (2023) |
| **End-to-end small models** | 2021–2025 | phonemes → one network that holds its own vocoder → waveform | VITS, Piper (VITS), YourTTS, StyleTTS 2, Kokoro, Kitten TTS |
| **Large zero-shot / LM-based** | 2023–2026 | text tokens + reference audio → language model over codec tokens (or flow matching over mels/latents) → codec decoder or vocoder | VALL-E, SoundStorm, Bark, XTTS, F5/E2-TTS, CosyVoice, Orpheus, Sesame CSM, Chatterbox, Qwen3-TTS, VoxCPM2 |
| **Small streaming latent LMs (new in 2026)** | 2026 | text → small transformer predicts continuous codec latents one 80 ms frame at a time → codec decoder (Mimi) | Kyutai Pocket TTS, Gradium Phonon, VoxRT LinnetTTS, MOSS-TTS-Nano |

The trend that matters for `speak`: the end-to-end small models (Piper, Kokoro, Kitten) already ship as **single ONNX graphs** with the vocoder inside. Rust only has to produce phoneme IDs and play a float buffer. The large LM-based models make better and more expressive audio and can clone voices, but they are 0.3–3B parameters and usually Python/PyTorch-only. The new 2026 streaming latent models (30–100M parameters) are a middle ground: they stream on CPU and some can clone voices, but none of them ships a plain ONNX graph yet (they use GGUF, safetensors or their own runtimes).

### How the stages merge (design question 1) 🟢

```
Classic two-stage:  text → normalize → G2P → [acoustic model → mel] → [vocoder → wave]
VITS / Piper:       text → normalize → espeak-ng IPA → ids → [text enc + flow + duration + HiFi-GAN decoder] → wave
Kokoro / Kitten:    text → normalize → misaki or espeak IPA → ids + style vec → [PL-BERT + prosody predictor + iSTFTNet decoder] → wave
Codec LM:           text → BPE tokens (+ reference audio codes) → [LM over codec tokens] → [codec decoder] → wave
Flow matching (F5): text chars + reference mel → [DiT, N ODE steps] → mel → [Vocos/BigVGAN] → wave
```

Every model in this report still needs **text normalization outside the model** (numbers, dates, abbreviations). Every small model also needs **G2P outside the model**. Only the LM-based models (Bark, Orpheus, XTTS, F5, Chatterbox, Qwen3-TTS) take raw text or BPE tokens and learn pronunciation themselves. That costs them hundreds of millions of parameters.

---

## 2. Techniques

### 2.1 Piper (VITS, ONNX): extra depth 🟢

**What it is.** Piper is a VITS model trained per voice, exported to one ONNX file, and run with espeak-ng phonemes. It was built for the Raspberry Pi and Home Assistant.

**Repos and licences.**
- `rhasspy/piper`: MIT, **archived** (last push 2025-08, last release 2023.11.14-2). Its C++ engine used `piper-phonemize`, which links espeak-ng.
- `OHF-Voice/piper1-gpl`: the active project under the Open Home Foundation, **GPL-3.0**, v1.8.0 released 2026-09-04. It embeds espeak-ng directly and has a C API, `libpiper`.
- The **voice models carry no licence grant from the Piper project**. Each voice's rights come from its training data, recorded in a `MODEL_CARD` file next to the voice (`docs/LICENSING.md`). Many voices are fine-tuned from `en_US-lessac`, whose Blizzard 2013 corpus licence allows research use only. In late 2026 the project retrained many voices from two clean bases: `nl_BE-rdh-medium` (CC0) and `en_US-hifi_tts_92-medium` (CC BY 4.0). `docs/COMMERCIAL_USE.md` lists the voices whose whole chain is permissive. English examples: `en_US-joe-medium`, `en_US-mike-medium` (CC0 via rdh), `en_US-ljspeech-medium/high`, `en_US-kristin-medium`, `en_US-norman-medium` (public domain, trained from scratch), `en_GB-cori-high` (PD), `en_US-libritts-high` (CC BY). **The retrained voices kept their old names, so a copy downloaded before the retrain is still the lessac-derived one.** Check its `MODEL_CARD`.
- The ONNX graph is just weights. Running it from our own Rust code does not pull in GPL code. The GPL question comes only from **espeak-ng** (GPL-3.0+), which Piper's phoneme inventory depends on (see area A/D).

**Architecture (VITS).** A transformer text encoder over phoneme IDs produces a prior distribution. A stochastic duration predictor (a flow) picks how many frames each phoneme lasts. The prior is expanded to frame rate, a normalizing flow maps it into the latent `z`, and a HiFi-GAN-style decoder turns `z` straight into a waveform (hop length 256). There is no mel spectrogram and no separate vocoder at inference. Monotonic alignment search is used only in training.

**Quality levels** (from `voices.json`, 183 voices, 53 language families: 129 medium, 26 low, 14 x_low, 14 high):

| Quality | Sample rate | Measured `.onnx` size | Parameters |
|---|---|---|---|
| x_low | 16 kHz | not measured | 🔴 ~5–7M (smaller generator) |
| low | 16 kHz | 63.2 MB (`en_US-lessac-low`) | 🔴 ~15.7M; the file is byte-for-byte the same size as medium |
| medium | 22.05 kHz | 63.2 MB (`en_US-lessac-medium`) | 🟢 **15.65M** initializer elements (counted by us) |
| high | 22.05 kHz | 113.9 MB (`en_US-lessac-high`) | 🔴 ~28M (size / 4 bytes) |
| multi-speaker | 22.05 kHz | 78.6 MB (`en_US-libritts_r-medium`, 904 speakers) | — |

**ONNX interface** (from `src/piper/train/export_onnx.py`, checked against the downloaded `en_US-lessac-medium.onnx`, opset 15, standard ONNX ops only, no LSTM or STFT):

| Tensor | dtype | Shape | Meaning |
|---|---|---|---|
| `input` | int64 | `[1, N]` | phoneme IDs |
| `input_lengths` | int64 | `[1]` | `N` |
| `scales` | float32 | `[3]` | `[noise_scale, length_scale, noise_w]`, defaults `[0.667, 1.0, 0.8]` |
| `sid` | int64 | `[1]` | speaker ID, **only present when `num_speakers > 1`** |
| `output` | float32 | `[1, 1, 1, samples]` (measured; squeeze it) | waveform in about [-1, 1] at `audio.sample_rate` |

Patched voices (`piper.patch_voice_with_alignment`) add a second output: frames per phoneme ID. Multiply by `hop_length` (256) to get samples per phoneme, which is useful for word timing and visemes.

**Phoneme ID encoding** (`phoneme_ids.py`): `^` (BOS, id 1), then `_` (PAD, 0) **after every phoneme**, then `$` (EOS, 2). So `[1, 0, p1, 0, p2, 0, …, pn, 0, 2]`. Each phoneme is one Unicode codepoint of espeak-ng IPA, including stress marks `ˈ ˌ`, length `ː`, space (3) and punctuation (`!`, `,`, `.`, `?`, …), which the model hears as tokens. Newer configs may add multi-codepoint keys (diphthongs `aɪ aʊ ɔɪ eɪ oʊ`, ids 161–165) and a `vowel_clusters` list to merge longest-first. A Rust port must handle both (libpiper had a bug here until recently, per CHANGELOG). Piper calls `espeak_TextToPhonemesWithTerminator` so that clause punctuation survives phonemization. Text inside `[[ … ]]` is taken as raw phonemes.

**Voice config `<voice>.onnx.json`** (real `en_US-lessac-medium`, `phoneme_id_map` shortened; it has 154 entries, `num_symbols` 256):

```json
{
  "audio": { "sample_rate": 22050, "quality": "medium" },
  "espeak": { "voice": "en-us" },
  "inference": { "noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8 },
  "phoneme_type": "espeak",
  "phoneme_map": {},
  "num_symbols": 256,
  "num_speakers": 1,
  "speaker_id_map": {},
  "piper_version": "1.0.0",
  "language": { "code": "en_US", "family": "en", "region": "US", ... },
  "dataset": "lessac",
  "phoneme_id_map": { "_": [0], "^": [1], "$": [2], " ": [3], "!": [4], ",": [8], ".": [10], "?": [13],
                      "a": [14], "b": [15], "ð": [41], "ə": [59], "ɪ": [74], "ˈ": [120], "ˌ": [121], "ː": [122], ... }
}
```

Optional keys in the current `PiperConfig`: `hop_length` (default 256), `vowel_clusters`, `default_speaker_id`, `phoneme_counts`. `phoneme_type` can be `espeak` (most voices), `text` (raw characters), `pinyin`, `hebrew`, `japanese` (OpenJTalk), `thai` or `lithuanian`.

**Controls.** `length_scale` sets the rate (<1 is faster). `noise_scale` sets how varied the output sounds and `noise_w` how varied the durations are. Pitch has no control input. Volume is applied after synthesis. The Python API normalizes audio to full range by default.

**Streaming (new, 2026).** `python -m piper.split_voice_for_streaming model.onnx` cuts the graph at the decoder input (`dec.conv_pre`, tensor `z`) into `model.encoder.onnx` (IDs → `z`) and `model.decoder.onnx` (`z` [+ `sid`] → audio). libpiper then decodes `z` in overlapping chunks wider than the decoder's receptive field, so the output matches whole-sentence decoding. CHANGELOG: first sound in **about 0.2 s whatever the sentence length**, compared with "about 4 seconds for a 6 second sentence of a high voice used by VoiceOver on an Apple M3" before (that figure was also before a fix for libpiper running ORT single-threaded). This is the cleanest streaming design in this survey, and it is easy to copy in Rust.

**Measured on this M2 Max** (Python + onnxruntime 1.32 CPU, `en_US-lessac-medium`, 243 phoneme IDs, hand-written IPA for a 5.6 s sentence, median of 5 runs; whisper transcribed the output correctly):

| Threads | Session load | Inference | RTF |
|---|---|---|---|
| 1 | 0.16 s | 0.444 s | **0.079** |
| 4 | 0.16 s | 0.140 s | **0.025** |

A third party (VoxRT, which has a conflict of interest) measured 🟡 RTF 0.058 on one core and 0.021 on four on an i9-13900K, with 180 ms to first audio and 503 MB peak RSS for the whole Python process.

**Quality.** 🟡 Community reputation is "clear, robotic-ish but very intelligible". Medium voices are fine for screen readers and assistants but clearly below Kokoro for natural prosody. The VITS paper reports MOS 4.43 against 4.46 for ground truth on LJSpeech (🟢), but that was a full-size VITS. Piper voices are smaller, and many were fine-tuned from limited data.

### 2.2 Kokoro-82M (StyleTTS 2 derivative, ONNX): extra depth 🟢

**What it is.** An 82M-parameter model by hexgrad. v1.0 was released 2025-01-27, with 54 voices in 8 languages: American and British English, Japanese, Mandarin, Spanish, French, Hindi, Italian and Brazilian Portuguese. A Chinese-focused `Kokoro-82M-v1.1-zh` (103 voices) followed on 2025-03-04. **No newer hexgrad release exists as of 2026-11** (the HF org's last change was 2025-04-10). It was trained on a few hundred hours of permissive, public-domain and synthetic audio for about $1000 of A100 time.

**Licences.** Weights Apache-2.0, `hexgrad/kokoro` code Apache-2.0, `misaki` G2P Apache-2.0, `onnx-community` exports Apache-2.0, `thewh1teagle/kokoro-onnx` MIT. CC BY data was used for training (Koniwa, SIWIS), and the card lists those attributions. **The one GPL risk is the optional espeak-ng fallback for out-of-vocabulary words.** It is optional for English, but the non-English languages lean on espeak.

**Architecture.** Kokoro is "decoder only: no diffusion, no encoder release" (model card). Of StyleTTS 2 it keeps:
1. **PL-BERT** (ALBERT, hidden 768, 12 attention heads, 12 layers with shared weights, `max_position_embeddings` 512) over phoneme tokens. It feeds a linear projection to 512.
2. **Prosody predictor**: an AdaLN text encoder conditioned on the *prosody half* of the style vector (`ref_s[:, 128:]`), then an LSTM duration head (`sigmoid(...).sum() / speed`, rounded, at least 1 frame per token), then F0 and energy (N) curves.
3. **Text encoder** (CNN + LSTM, kernel size 5) aligned to the predicted durations.
4. **Decoder**: a StyleTTS 2 decoder plus an **iSTFTNet** vocoder head (upsample rates 10×6, n_fft 20, hop 5), conditioned on the *acoustic half* `ref_s[:, :128]`. Output is a 24 kHz waveform.

Training used StyleTTS 2's diffusion style sampler and the reference encoder. Those are not released, so **you cannot clone a voice or create a new style vector from audio** with the public model. You can only pick or blend the shipped voice packs.

**Model config** (`hexgrad/Kokoro-82M/config.json`): `n_token` 178, `hidden_dim` 512, `style_dim` 128, `n_mels` 80, `n_layer` 3, `max_dur` 50, `dim_in` 64, `max_conv_dim` 512, `multispeaker` true, `istftnet` `{upsample_rates [10,6], upsample_kernel_sizes [20,12], gen_istft_n_fft 20, gen_istft_hop_size 5, upsample_initial_channel 512, resblock_kernel_sizes [3,7,11], resblock_dilation_sizes [[1,3,5]×3]}`, plus a `vocab` of **114 symbols** mapped to IDs up to 177. The symbols are punctuation `;:,.!?—…"()“”`, space, IPA letters, misaki's capital-letter diphthongs `A I O Q W Y`, `ᵊ`, stress `ˈ ˌ`, length `ː`, and intonation arrows `↓ → ↗ ↘`. PAD is ID 0 (`$` in `tokenizer_config.json`).

**ONNX interface** (onnx-community `Kokoro-82M-v1.0-ONNX`, checked on the downloaded `model.onnx` and `model_quantized.onnx`, opset 20):

| Tensor | dtype | Shape | Meaning |
|---|---|---|---|
| `input_ids` | int64 | `[1, T]`, T ≤ 512 | `[0, *tokens, 0]`: at most 510 phoneme tokens plus a pad at each end |
| `style` | float32 | `[1, 256]` | one row of the voice pack |
| `speed` | float32 | `[1]` | duration divisor, 1.0 = normal (kokoro-onnx limits it to 0.5–2.0) |
| `waveform` | float32 | `[1, samples]` | 24 kHz mono |

The older thewh1teagle exports name the first input `tokens`, and some exports take an **integer** `speed`. kokoro-onnx checks the input names and dtypes at runtime, and a Rust loader should do the same.

**Voice packs.** One voice file holds `float32[510, 1, 256]` = 522,240 bytes (`voices/af_heart.bin`, raw little-endian floats; the hexgrad repo has `.pt` tensors of the same shape). kokoro-onnx bundles all voices in one `voices-v1.0.bin` (an npz archive, 28 MB). **The row is picked by token count: Kokoro trains a different style vector for each utterance length.** There is an off-by-one between sources: the official `KPipeline` uses `pack[len(ps) - 1]`, and the onnx-community README uses `voices[len(tokens)]`. 🔴 The audible difference is probably negligible, but use the official `len - 1`. Voices can be blended by a weighted average of packs (KPipeline accepts `"af_bella,af_sky"`). Voice quality varies a lot (VOICES.md grades `af_heart` A and `af_bella` A-, many others C or D). The card says most voices are best at **100–200 tokens**, weaker below 10–20 tokens and "rushing" above 400. So chunking does more than set latency: it changes quality.

**ONNX variants** (`onnx-community/Kokoro-82M-v1.0-ONNX`): `model.onnx` 325.5 MB fp32, `model_fp16` 163 MB, `model_uint8f16` 114 MB, `model_uint8` 177 MB, `model_quantized` (int8 dynamic) 92.4 MB, `model_q8f16` 86 MB, `model_q4` 305 MB, `model_q4f16` 155 MB. The fp32 graph uses only standard ONNX ops but contains **6 LSTM nodes and 1 STFT node**. The int8 graph uses ORT-only `com.microsoft` ops (`DynamicQuantizeLSTM`, `FastGelu`, `SkipLayerNormalization`, `FusedMatMul`), so **it runs only on ONNX Runtime**, not tract or candle (see area D).

**G2P: misaki.** For English it looks words up in a gold lexicon, then a silver lexicon (`us_gold.json` 3.0 MB, `us_silver.json` 3.1 MB, and GB equivalents), with spaCy for POS tags and tokenization and num2words for numbers. Words not in the lexicons go to an optional fallback: espeak-ng (via `phonemizer-fork`) or, planned, a seq2seq model. The phoneme set is **49 symbols** for English (EN_PHONES.md). Its own conventions include `A`=eɪ, `I`=aɪ, `O`=oʊ, `W`=aʊ, `Y`=ɔɪ, `Q`=əʊ, `ʤ ʧ` as single tokens, and `ᵊ` (small schwa). misaki ships a `FROM_ESPEAKS` table for mapping espeak IPA to misaki symbols, so **Kokoro can run on espeak-ng output alone** with some quality loss. kokoro-onnx and Kitten do exactly that. Other languages: pyopenjtalk for Japanese, jieba plus a PaddleSpeech frontend for Chinese, g2pK for Korean, espeak for the rest.

**Measured on this M2 Max** (Python + ORT 1.32 CPU, the onnx-community example tokens: 141 tokens, 143 with pads, `af_heart`, median of 3 runs; whisper transcribed the fp32 output correctly):

| Model file | Threads | Load | Inference | Audio | RTF |
|---|---|---|---|---|---|
| `model_quantized.onnx` (92 MB) | 1 | 0.21 s | 2.09 s | 8.18 s | **0.256** |
| `model_quantized.onnx` | 4 | 0.20 s | 0.88 s | 8.18 s | **0.107** |
| `model.onnx` fp32 (326 MB) | 1 | 0.33 s | 5.68 s | 8.25 s | **0.689** |
| `model.onnx` fp32 | 4 | 0.58 s | 1.67 s | 8.25 s | **0.202** |

So Kokoro is **about 4–8× slower than Piper medium** for the same audio, but still faster than real time. The ONNX graph does not stream: first audio arrives only after the whole chunk is synthesized, about 0.9–1.7 s for an 8 s chunk on 4 threads. Getting audio out sooner means short first chunks (§4). 🟡 VoxRT's i9 numbers: RTF 0.517 on one core, 0.172 on four, 1.5 s first audio, 1.25 GB RSS.

**Quality.** 🟡 Kokoro is the best-rated *small* open model. Artificial Analysis Speech Arena (Nov 2026, secondary): Kokoro 1064, Chatterbox 1023. HF TTS Arena (Oct 2026): Kokoro is about 32nd, Elo 1464, now behind mostly hosted APIs. 🟢 The StyleTTS 2 paper it derives from reports CMOS +0.28 over the *ground-truth recordings* on LJSpeech.

### 2.3 Tacotron 2 (2017), paper `2017-tacotron2.pdf` 🟢
- **Flow:** characters → conv + BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder → 80-band mel frames → modified WaveNet vocoder.
- **Input:** characters (phonemes in many reimplementations). **Vocoder:** separate (WaveNet in the paper; WaveGlow or HiFi-GAN in practice).
- **Size/speed:** about 28M parameters for the acoustic model 🔴. Autoregressive, so slow and attention can fail (skipped or repeated words) on long or unusual input.
- **Quality:** MOS 4.53 against 4.58 for ground truth on Google's internal data. In the VITS paper's re-evaluation on LJSpeech with HiFi-GAN: 3.77, or 4.25 after fine-tuning.
- **Licence/ONNX:** no official weights; NVIDIA's BSD-3 implementation has LJSpeech checkpoints and ONNX/TensorRT exports. Historically important; nothing to build on now.

### 2.4 FastSpeech 2 (2020), `2020-fastspeech2.pdf` 🟢
- **Flow:** phonemes → feed-forward transformer encoder → **variance adaptor** (duration, pitch and energy predictors, trained on MFA alignments and extracted F0/energy) → length regulator → transformer decoder → mel → vocoder. FastSpeech 2s goes straight to waveform.
- **Input:** phonemes (MFA/CMUdict in the paper). **Vocoder:** separate (Parallel WaveGAN in the paper).
- **Size/speed:** 41.2M parameters with RTF 0.010 on GPU (Matcha paper, Table 1). Non-autoregressive and robust.
- **Quality:** MOS 3.83 (+PWG) against 4.30 GT in its paper; 3.29 in the Matcha paper's re-run.
- **Why it matters:** the explicit **pitch/energy/duration controls** are the cleanest way to give a neural model `say`-style rate and pitch knobs. ONNX export is easy (community repos, sherpa-onnx). No canonical weights or licence.

### 2.5 Glow-TTS (2020), `2020-glow-tts.pdf` 🟢
- **Flow:** phonemes → transformer encoder → prior → **monotonic alignment search** (training) and a deterministic duration predictor (inference) → invertible flow decoder → mel → vocoder.
- **Size/speed:** 28.6M parameters 🔴. Glow-TTS + HiFi-GAN ran at 27× real time on GPU (VITS paper Table 4).
- **Quality:** MOS 4.14, or 4.32 fine-tuned, on LJSpeech + HiFi-GAN (VITS paper).
- **Licence:** code MIT. Its alignment search and flow became the core of VITS.

### 2.6 VITS (2021), `2021-vits.pdf` 🟢
- **Flow:** phonemes → transformer text encoder → conditional VAE prior + normalizing flow + **stochastic duration predictor** → HiFi-GAN decoder → **waveform**, trained end to end with adversarial and feature-matching losses.
- **Input:** espeak phonemes via the `phonemizer` package. **Vocoder:** built in.
- **Size/speed:** 36.3M parameters; RTF 0.074 on an RTX 3090 including everything (Matcha Table 1); 67× real time on GPU (VITS Table 4).
- **Quality:** MOS 4.43 against 4.46 GT on LJSpeech, and 4.38 against 4.38 on VCTK multi-speaker.
- **Speaker control:** speaker-ID embedding. **Licence:** code MIT (`jaywalnut310/vits`); official LJ/VCTK checkpoints have no licence file 🔴. **ONNX:** yes. Piper, sherpa-onnx and Coqui all export it.
- Piper is VITS with a smaller generator, trained per voice.

### 2.7 YourTTS (2021), `2021-yourtts.pdf` 🟢
- VITS plus an external **speaker encoder (H/ASP)**, a language embedding and speaker-consistency loss, giving **zero-shot cloning** from a short reference and cross-lingual voice conversion (English, Portuguese, French).
- **Input:** characters, not phonemes. **Licence:** code MPL-2.0 (Coqui TTS). 🔴 Weights are reportedly CC BY-NC-ND 4.0. Shipped in Coqui TTS. Superseded by XTTS.

### 2.8 StyleTTS 2 (2023), `2023-styletts2.pdf` (and `2022-styletts.pdf`) 🟢
- **Flow:** phonemes → PL-BERT + text encoder → a **style diffusion** model samples a 256-dim style vector (128 acoustic + 128 prosodic) from text alone, or a reference encoder takes it from audio → prosody predictor (duration, F0, energy) → HiFi-GAN or iSTFTNet decoder → waveform. Training adds WavLM adversarial losses and differentiable duration modeling.
- **Input:** espeak-ng IPA via `phonemizer` (GPL). The README warns that inference "depends on a GPL-licensed package". An MIT `styletts2` PyPI package swaps in gruut, at lower quality.
- **Quality:** CMOS **+0.28 over ground truth** on LJSpeech, on par with GT on VCTK, CMOS +1.07 over NaturalSpeech. Zero-shot on LibriTTS.
- **Licence:** code MIT. The **pretrained weights come with a use condition**: you must tell listeners the speech is synthesized unless you have the voice owner's permission. **ONNX:** not official. Kokoro and Kitten are its practical ONNX descendants.

### 2.9 Matcha-TTS (2023), `2023-matcha-tts.pdf` 🟢
- **Flow:** phonemes → transformer encoder + duration predictor (monotonic alignment search) → **U-Net decoder trained with optimal-transport conditional flow matching**, solved with 2–10 Euler ODE steps → mel → HiFi-GAN.
- **Input:** espeak phonemes via `phonemizer`. **Vocoder:** separate (13.9M HiFi-GAN). The official ONNX export (`matcha/onnx/export.py`) can **fold the vocoder into the graph**.
- **Size/speed:** 18.2M parameters (smaller than VITS even with the vocoder added); RTF 0.038 (10 steps) to 0.015 (2 steps) on RTX 3090.
- **Quality:** MOS 3.84 with 10 steps against VITS 3.71 in the same test; WER 2.09, the lowest of all systems tested.
- **Licence:** code MIT; LJSpeech/VCTK checkpoints (LJ is public domain). sherpa-onnx ships Matcha models.
- **For `speak`:** a fine second ONNX backend. The step count trades quality for speed at run time, which suits a tool built to experiment.

### 2.10 Kitten TTS (2025–2026): small StyleTTS 2 ONNX models 🟢
- KittenML: **nano 0.8, 15M parameters, 56 MB fp32 ONNX** (int8 variant available), plus mini 0.8 (the docs give the family as 15M–80M parameters, 25–80 MB). Apache-2.0 code and weights for the 0.8 ONNX models. Acknowledges "StyleTTS 2 architecture".
- **ONNX interface (checked on the download):** same as Kokoro, `input_ids [1,T] int64`, `style [1,256] float32`, `speed [1] float32` → `waveform [samples]`, plus a `duration` int64 output per token (free word timing). Standard ops only, with 5 LSTM nodes. Voices: `voices.npz` with 8 entries (`expr-voice-{2..5}-{m,f}`), each `float32[400, 256]`. 24 kHz output. Phonemes come from espeak-ng via `phonemizer`/`espeakng_loader`, so the same GPL concern applies.
- **Kitten TTS 2** (2026-09-30) is a different design: an LM plus decoder, multi-GB, PyTorch/GGUF, under the "Stellon Labs community licence", which 🟡 ends free commercial use above USD 1M revenue or funding. Not relevant here.
- Quality and RTF of nano 0.8 were not measured. 🟡 Community reputation: decent for its size, below Kokoro.

### 2.11 XTTS v2 (2023–24), `2024-xtts.pdf` 🟢
- **Flow:** BPE text plus conditioning latents from a 6 s reference (Perceiver) → **443M-parameter GPT-2-style decoder** predicts discrete VQ-VAE audio codes (13M VQ-VAE, 1024 codes) → **HiFi-GAN decoder (26M)** turns the GPT latents into 24 kHz audio. Built on Tortoise ideas.
- 17 languages, zero-shot cloning, streaming (under 200 ms first chunk on GPU 🟡). About 1.87 GB `model.pth`.
- **Licence:** code MPL-2.0 (`coqui-ai/TTS` unmaintained since 2024; the `idiap/coqui-ai-TTS` fork is active). **Weights: Coqui Public Model License, non-commercial.** Coqui shut down in early 2024, so no commercial licence can be bought. No official ONNX.
- On CPU, RTF is 🔴 about 1 or worse. Not a fit for `speak`.

### 2.12 F5-TTS and E2 TTS (2024), `2024-f5-tts.pdf`, `2024-e2-tts.pdf` 🟢
- **E2 TTS (Microsoft):** characters padded with filler tokens to the length of the mel → a flat transformer with **flow matching** fills in the masked mel from a reference prompt. No duration model, no phonemizer, no alignment. Microsoft released no weights; F5-TTS includes a reproduction.
- **F5-TTS:** the same idea with a **DiT backbone and ConvNeXt text refinement**, plus "Sway Sampling" at inference. 335.8M parameters, RTF **0.15 on GPU with 16 NFE**. The vocoder is separate (Vocos by default, BigVGAN optional). The total duration must be given (the reference speed sets it).
- **Input:** characters/pinyin. **Cloning:** zero-shot from a reference clip with its transcript.
- **Licence:** code MIT, **weights CC BY-NC 4.0** (Emilia training data). 1.35 GB checkpoints. Community ONNX exports exist 🟡, but CPU RTF is far above real time on a laptop 🔴. Good quality, wrong size class.

### 2.13 Codec language models: VALL-E, SoundStorm, Bark 🟢
- **VALL-E** (`2023-vall-e.pdf`): phonemes plus a 3 s acoustic prompt → an autoregressive transformer predicts the first EnCodec codebook → a non-autoregressive transformer predicts codebooks 2–8 → EnCodec decoder. It framed TTS as **conditional language modeling over codec tokens**, and that framing is the ancestor of the whole LLM-TTS line. No official weights (Microsoft cited misuse risk).
- **SoundStorm** (`2023-soundstorm.pdf`, Google): parallel, MaskGIT-style, non-autoregressive decoding of SoundStream tokens from AudioLM semantic tokens, about 100× faster than autoregressive decoding. No weights.
- **Bark** (Suno, 2023): GPT-style text → semantic → coarse → fine EnCodec tokens. Makes laughter, music and non-speech sounds; quality varies and it often hallucinates. **MIT code and weights.** About 5 GB per stage at full size (a small variant exists). Slow on CPU. No ONNX.

### 2.14 Parler-TTS (2024), `2024-parler-tts.pdf` 🟢
- **Flow:** a **natural-language description** of the voice ("a female speaker with a calm voice, close-sounding, slightly fast") → frozen Flan-T5 encoder; the transcript → token embeddings; a decoder-only transformer generates **DAC codec tokens** (delay pattern) → DAC decoder.
- Mini v1.1 is 880M (3.75 GB safetensors); Large is 2.3B. **Apache-2.0 code and weights.** No ONNX. Repo inactive since 2024-12.
- An interesting control idea (describe the voice in a prompt), but the size is too large for `speak`.

### 2.15 LLM-based TTS (2025–2026) 🟢/🟡
These put a pretrained text LLM in the middle, emitting audio tokens or latents. All are hundreds of MB to several GB and PyTorch/MLX/GGUF-first.

| Model | Released | Architecture | Size | Licence (code / weights) | ONNX? |
|---|---|---|---|---|---|
| **Orpheus** (Canopy) | 2025-03 | Llama-3.2-3B fine-tuned to emit **SNAC** 24 kHz codec tokens; emotion tags | 3B (smaller ones announced) | Apache-2.0 / card says Apache-2.0; 🟡 Llama 3.2 base licence likely applies too | No (GGUF via llama.cpp) |
| **Sesame CSM-1B** | 2025-03 | Llama-style 1B backbone + small audio decoder over **Mimi** RVQ codes; context-aware conversational speech | 1B (+100M decoder) | Apache-2.0 / Apache-2.0 | No |
| **CosyVoice 2 / Fun-CosyVoice3-0.5B** (Alibaba), `2024-cosyvoice2.pdf` | 2024-12 / 2025-12 | Qwen2.5-0.5B LM → supervised semantic speech tokens → **flow matching** to mel → HiFi-GAN; chunk-aware streaming | 0.5B LM + about 0.3B flow | Apache-2.0 / Apache-2.0 | Partial: flow estimator and tokenizer ship as `.onnx`, the LM does not |
| **Chatterbox / Turbo / Nano** (Resemble) | 2025-05 / 2025-12 / 2026 | "T3" Llama-style LM → S3 speech tokens → S3Gen flow-matching decoder (Turbo distilled to 1 step) → vocoder; paralinguistic tags; built-in PerTh watermark | 500M / 350M / 110M (Nano: "3× real time on 8 CPU cores") | MIT / MIT | **Yes**: `ResembleAI/chatterbox-turbo-ONNX` (multi-graph: embed, speech encoder, LM, decoder; 1.3 GB LM) |
| **Qwen3-TTS** (Alibaba) | 2026-01-22 | LM over a 12 Hz speech tokenizer; Base (3 s cloning), CustomVoice, VoiceDesign variants | 0.6B / 1.7B (0.6B Base = 1.8 GB + 0.7 GB tokenizer) | Apache-2.0 / Apache-2.0 | No (MLX ports exist) |
| **VoxCPM2** (OpenBMB) | 2026-04 | Tokenizer-free, continuous-latent LM with an AudioVAE; 30 languages, 48 kHz | 2B (4.6 GB) | Apache-2.0 | No (GGUF) |
| **VibeVoice** (Microsoft) | 2025-08 | LLM (Qwen2.5) + continuous acoustic/semantic tokenizers + diffusion head; long-form multi-speaker | 1.5B / 7B | MIT, 🟡 research-only; repo was pulled for misuse | No |
| **NeuTTS Air** (Neuphonic) | 2025-09 | 0.5B Qwen-based LM + NeuCodec; designed for on-device use via GGUF | 0.5B (Q4 GGUF) | Apache-2.0 | No |

**Status note:** the Chatterbox README now says "**As of August 2026, Chatterbox is retired research and is no longer maintained**", with the code kept as an MIT archive 🟢.

### 2.16 Small streaming latent models (2026): the newest class 🟢/🟡
These matter because they target exactly `speak`'s case: laptop CPU, streaming, sometimes cloning.

- **Kyutai Pocket TTS** (2026-01; training code released 2026-08). 100M parameters. A **continuous audio language model** (paper `2025-calm-pocket-tts.pdf`, arXiv 2509.06926): a transformer predicts continuous **Mimi** latents frame by frame with a small flow/consistency head, and the Mimi decoder produces 24 kHz audio. Claims (card): about 200 ms to first chunk, about 6× real time on a MacBook Air M4 using **only 2 cores**, voice cloning, 6 languages, unlimited text length. **Weights CC BY 4.0** (gated, with a prohibited-use clause for impersonation); code MIT. No ONNX. Community GGUF and CoreML ports exist.
- **Gradium Phonon** (2026-10-19). Builds on Pocket TTS and Mimi: a "flow-matching transformer predicts audio latents from the text one 80 ms frame at a time". **phonon-31m-en is Apache-2.0** (64.5 MB q8 GGUF); the 40M English, French, German, Spanish, Portuguese and multilingual variants are **CC BY-NC 4.0**. Runtimes: Python `ptts`, a browser npm package, and its own C++/"xn" runtime. 🟡 Reported 30–100 ms first audio on one M2 core.
- **VoxRT LinnetTTS** (2026-11-04). 41M parameters (the repo is named `linnet-en-35m`), English, autoregressive flow LM over a 25 Hz continuous codec latent, **Rust runtime**. Weights are under the custom "Linnet Weights License" (commercial use allowed, cloning and training licensed separately). 🟡 Their own benchmark: RTF 0.106 on one core, 87 ms first audio, 187 MB RSS.
- **MOSS-TTS-Nano** (OpenMOSS, 2026-04). 100M parameters, multilingual cloning, 48 kHz stereo, streams on 4 CPU cores, Apache-2.0. 🟡 A "standalone ONNX CPU version" is mentioned by a secondary source; the HF repo shows only `pytorch_model.bin`. Unverified.

None of these runs as a one-call ONNX graph. They are autoregressive *loops* (KV cache + a per-frame flow head + the Mimi decoder), so Rust would have to implement the loop itself, most likely in candle or burn (Kyutai's own Rust stack, `moshi`, is candle-based 🔴). That is more work, but it is also the only small-model path to **real streaming and voice cloning on CPU**.

---

## 3. Comparison table

RTF is CPU unless marked GPU. "Ours" means measured in §2.1/§2.2 on an M2 Max with 4 threads.

| Model | Year | Input | Own vocoder | Params | File | RTF | Quality | Cloning / control | ONNX | Code licence | Weights licence | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tacotron 2 | 2017 | chars | no (WaveNet/HiFi-GAN) | ~28M + voc | — | slow, autoregressive | MOS 4.53 (internal) | spk-emb variants | community | BSD (NVIDIA) | varies | 🟢/🔴 |
| FastSpeech 2 | 2020 | phonemes | no | 41M | — | 0.010 GPU | MOS 3.83 | **pitch/energy/duration knobs** | easy | various | varies | 🟢 |
| Glow-TTS | 2020 | phonemes | no | ~29M | — | 27× RT GPU (+HiFi-GAN) | MOS 4.14–4.32 | spk-ID | yes | MIT | — | 🟢 |
| VITS | 2021 | espeak IPA | **yes** | 36M | ~145 MB 🔴 | 0.074 GPU | MOS 4.43 (GT 4.46) | spk-ID | yes | MIT | unclear | 🟢 |
| **Piper medium** | 2023–26 | espeak IPA ids | **yes** | 15.7M | 63 MB | **0.025 (ours)**, 0.079 on 1 thread | 🟡 clear, a bit flat | spk-ID (multi), length/noise scales | **official** | GPL-3 (piper1) / MIT (archived) | per voice; CC0/PD voices exist | 🟢 |
| YourTTS | 2021 | chars | yes | ~40M 🔴 | — | — | good for its time | zero-shot | partial | MPL-2.0 | 🔴 CC BY-NC-ND | 🟢/🔴 |
| StyleTTS 2 | 2023 | espeak IPA | yes | ~150M 🔴 | — | fast on GPU | CMOS +0.28 over GT | ref-audio style, diffusion | no official | MIT | use condition (disclose synthesis) | 🟢 |
| **Kokoro-82M** | 2025 | misaki/espeak IPA (178 vocab) | **yes** (iSTFTNet) | 82M | 326 MB fp32 / 92 MB int8 | **0.107 int8, 0.202 fp32 (ours)** | 🟡 best small model; AA 1064 | 54 preset voices, blendable; no cloning | **yes** (community) | Apache-2.0 | **Apache-2.0** | 🟢 |
| Kitten nano 0.8 | 2026 | espeak IPA | yes | 15M | 56 MB | not measured | 🟡 decent | 8 voices | **yes** | Apache-2.0 | Apache-2.0 | 🟢 |
| Matcha-TTS | 2023 | espeak IPA | no (HiFi-GAN; can be folded into ONNX) | 18M + 14M | ~130 MB 🔴 | 0.015–0.038 GPU | MOS 3.84 > VITS 3.71 | spk-ID, step count, temperature | **official** | MIT | LJ PD-based | 🟢 |
| XTTS v2 | 2023 | BPE text | yes (HiFi-GAN) | ~480M | 1.9 GB | 🔴 ≥1 on CPU | 🟡 very good | 6 s zero-shot, 17 languages | no | MPL-2.0 | **CPML non-commercial** | 🟢 |
| F5-TTS | 2024 | chars | no (Vocos) | 336M | 1.35 GB | 0.15 GPU | 🟡 excellent cloning | zero-shot | community | MIT | **CC BY-NC** | 🟢 |
| E2 TTS | 2024 | chars | no | ~333M | — | — | similar to F5 | zero-shot | — | (no release) | (no release) | 🟢 |
| VALL-E / SoundStorm | 2023 | phonemes / semantic tokens | codec | — | — | — | landmark papers | 3 s prompt | — | no release | no release | 🟢 |
| Bark | 2023 | text | codec (EnCodec) | ~1B (3 stages) | ~5–12 GB | slow | 🟡 expressive but erratic | history prompts | no | MIT | MIT | 🟢 |
| Parler-TTS mini | 2024 | text + description | codec (DAC) | 880M | 3.75 GB | slow on CPU 🔴 | 🟡 good | **text-described voice** | no | Apache-2.0 | Apache-2.0 | 🟢 |
| Orpheus 3B | 2025 | text | codec (SNAC) | 3B | ~6.6 GB bf16 | GPU-class | 🟡 very natural, emotive | tags, finetune | no (GGUF) | Apache-2.0 | Apache-2.0 + Llama? | 🟢/🟡 |
| CosyVoice 2/3 | 2024–25 | text | flow + HiFi-GAN | 0.5B + flow | ~5–10 GB repo | GPU-class | 🟡 strong, streaming | zero-shot, instructions | partial | Apache-2.0 | Apache-2.0 | 🟢 |
| Sesame CSM-1B | 2025 | text + context | codec (Mimi) | 1B | 6.2 GB | GPU-class | 🟡 conversational | context/speaker | no | Apache-2.0 | Apache-2.0 | 🟢 |
| Chatterbox Turbo/Nano | 2025–26 | text | flow + vocoder | 350M / 110M | ~1.3 GB LM ONNX | Nano 3× RT on 8 cores (claim) | 🟡 AA 1023, top open on HF arena | zero-shot, tags | **yes** (Turbo, multi-graph) | MIT (**retired 2026-08**) | MIT | 🟢 |
| Qwen3-TTS | 2026 | text | codec | 0.6B / 1.7B | 2.5 GB (0.6B) | GPU-class | 🟡 strong | 3 s cloning, voice design | no | Apache-2.0 | Apache-2.0 | 🟢 |
| Pocket TTS | 2026 | text | Mimi decoder | 100M | ~236 MB 🟡 | ~0.17 on 2 cores, M4 (claim) | 🟡 good | **cloning**, streaming | no | MIT | CC BY 4.0 (gated) | 🟢 |
| Phonon-31m-en | 2026 | text | Mimi decoder | 31M | 64.5 MB q8 GGUF | 🟡 fast, 1 core | 🟡 new, unrated | streaming | no | ? | Apache-2.0 (en-31m only) | 🟢/🟡 |
| LinnetTTS | 2026 | text | codec decoder | 41M | 83 MB fp16 | 🟡 0.034 on 4 cores (vendor) | unrated | cloning (licensed separately) | no (Rust runtime) | Apache-2.0 | custom | 🟡 |

---

## 4. What it means for `speak` 🔴 (recommendations, built on the 🟢 facts above)

1. **Make Piper the first neural backend, Kokoro the second, behind one trait.** Both are single ONNX graphs with the vocoder inside, phoneme IDs in and float PCM out, so one `ort` session wrapper covers both. Piper is 4–8× faster (RTF 0.025 against 0.11 on 4 threads here), loads in 0.16 s, is 63 MB, and covers 53 languages. Kokoro sounds clearly better in English and is Apache-2.0 end to end. Kitten nano (15M, same tensor interface as Kokoro) comes almost free once Kokoro works, and Matcha-TTS is a good fourth backend for comparing flow-matching step counts.

2. **Suggested backend trait shape**, given the interfaces above:
   - `phonemize(text) -> Vec<Sentence<Phoneme>>` (frontend, from area A)
   - `encode(phonemes) -> Vec<i64>` (voice-specific: Piper's `phoneme_id_map` with PAD interleaving and BOS/EOS; Kokoro's `vocab` with `[0, …, 0]`)
   - `synthesize(ids, VoiceParams) -> Vec<f32>` plus `sample_rate()` (22050 Piper medium/high, 16000 low/x_low, 24000 Kokoro/Kitten)
   - `VoiceParams { rate, speaker, noise, style }`: rate maps to `length_scale = 1/rate` for Piper and `speed = rate` for Kokoro. **Neither model takes a pitch input.** Pitch for `say -p`-style control would have to be DSP post-processing (resampling or PSOLA, area C) or a FastSpeech 2-style model.

3. **Phonemizer: decide the espeak-ng question for each backend.** Every Piper voice expects espeak-ng IPA exactly as espeak emits it, including the clause terminator handling. A different G2P gives audibly worse output 🔴. Kokoro was trained on misaki symbols, whose English lexicons (gold and silver, about 6 MB of JSON, Apache-2.0) can be loaded from Rust directly, with espeak only for missing words. So a GPL-free English path exists for Kokoro (lexicon + a fallback such as a small neural G2P or CMUdict mapping) but **not really for Piper**. For a personal tool, linking espeak-ng (GPL-3) is fine, but the `speak` binary is then GPL. See areas A and D.

4. **Streaming:** copy Piper's split. Run the encoder once per sentence and decode `z` in overlapping chunks wider than the receptive field. This gets about 0.2 s to first sound with the same audio. Kokoro has no such split, so do it at the text level: send a **short first chunk** (a clause), then chunks of 100–200 tokens at clause or sentence boundaries, like misaki's waterfall `!.?… → :; → ,—`. Keep the next chunk synthesizing while the current one plays. Kokoro is weak below 10–20 tokens, so don't make the first chunk *too* short. Pick the style row by each chunk's token count (`pack[n-1]`).

5. **Voices and licences to ship or recommend:** Kokoro (Apache-2.0, 54 voices; download `voices-v1.0.bin` 28 MB or per-voice `.bin`s of 522,240 bytes each). Piper: default to a voice from `COMMERCIAL_USE.md` (`en_US-joe-medium`, `en_US-ljspeech-high`, `en_GB-cori-high`, `no_NO-talesyntese-medium` …), and **re-download after the late-2026 retrain**. Always keep each voice's `MODEL_CARD` next to it. Avoid lessac-derived voices as defaults if `speak` might ever be shared. Do not bundle XTTS, F5-TTS, YourTTS or 40M Phonon weights (non-commercial).

6. **Use fp32 or int8 with ORT for Kokoro; plan for portability.** The int8 graph uses ORT-only `com.microsoft` ops. The fp32 graph is standard ONNX but has LSTM and STFT nodes. Those are the ops to check first in tract, candle-onnx or burn-import (area D). Piper's graph is plain opset 15 with no LSTM or STFT, the easiest to port.

7. **For later experiments:** the 2026 small streaming latent models (Pocket TTS, Phonon-31m-en Apache-2.0, LinnetTTS) are the obvious "phase 3". They bring real frame-level streaming and, for Pocket TTS, voice cloning on 1–2 CPU cores. They need a hand-written autoregressive loop plus a Mimi decoder in Rust (candle), not a single ONNX call. The big LLM-TTS models (Orpheus, CosyVoice, Qwen3-TTS, Chatterbox Turbo) are worth trying only as a GPU/MLX side experiment. They don't meet "fast startup on a laptop CPU".

8. **Gaps to fill with our own tests:** listening-test quality of Piper medium against high against Kokoro; Kitten nano RTF; Rust-side load time (ORT session creation was 0.16–0.58 s in Python here, so the 326 MB fp32 Kokoro costs about 0.5 s startup unless memory-mapped or kept in a daemon); peak RSS in Rust (third-party Python numbers: Piper about 0.5 GB, Kokoro about 1.25 GB process RSS).

---

## 5. Sources

### Primary 🟢
- Piper (active): https://github.com/OHF-Voice/piper1-gpl, including `docs/LICENSING.md`, `docs/COMMERCIAL_USE.md`, `docs/STREAMING.md`, `docs/ALIGNMENTS.md`, `docs/TRAINING.md`, `docs/BUILDING.md`, `CHANGELOG.md`, `src/piper/train/export_onnx.py`, `src/piper/phoneme_ids.py`, `src/piper/config.py`, `src/piper/voice.py`, `src/piper/split_voice_for_streaming.py`; releases https://github.com/OHF-Voice/piper1-gpl/releases
- Piper (archived, MIT): https://github.com/rhasspy/piper
- Piper voices: https://huggingface.co/rhasspy/piper-voices (`voices.json`, `en/en_US/lessac/medium/en_US-lessac-medium.onnx(.json)`)
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M (README, `config.json`, VOICES.md), https://github.com/hexgrad/kokoro (`kokoro/model.py`, `kokoro/pipeline.py`), https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
- Kokoro ONNX: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX, https://github.com/thewh1teagle/kokoro-onnx (`src/kokoro_onnx/__init__.py`, `config.py`, releases `model-files-v1.0/1.1`)
- misaki: https://github.com/hexgrad/misaki (README, EN_PHONES.md, `misaki/data/`)
- Kitten TTS: https://github.com/KittenML/KittenTTS (`docs/onnx-models.md`), https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32, https://huggingface.co/KittenML/kitten-tts-2
- StyleTTS 2: https://github.com/yl4579/StyleTTS2 (README licence section)
- VITS: https://github.com/jaywalnut310/vits
- Matcha-TTS: https://github.com/shivammehta25/Matcha-TTS (`matcha/onnx/export.py`)
- F5-TTS: https://github.com/SWivid/F5-TTS, https://huggingface.co/SWivid/F5-TTS
- XTTS: https://huggingface.co/coqui/XTTS-v2, https://coqui.ai/cpml, https://github.com/coqui-ai/TTS, https://github.com/idiap/coqui-ai-TTS
- Parler-TTS: https://github.com/huggingface/parler-tts, https://huggingface.co/parler-tts/parler-tts-mini-v1.1
- Bark: https://github.com/suno-ai/bark, https://huggingface.co/suno/bark
- Orpheus: https://github.com/canopyai/Orpheus-TTS, https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice, https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B, https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- Sesame CSM: https://github.com/SesameAILabs/csm, https://huggingface.co/sesame/csm-1b
- Chatterbox: https://github.com/resemble-ai/chatterbox (README, retirement notice), https://huggingface.co/ResembleAI/chatterbox, https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX
- Qwen3-TTS: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- VoxCPM2: https://huggingface.co/openbmb/VoxCPM2
- VibeVoice: https://huggingface.co/microsoft/VibeVoice-1.5B
- NeuTTS Air: https://huggingface.co/neuphonic/neutts-air
- Pocket TTS: https://huggingface.co/kyutai/pocket-tts, https://github.com/kyutai-labs/pocket-tts, https://kyutai.org/blog/2026-01-13-pocket-tts
- Phonon: https://huggingface.co/gradium/phonon-31m-en, https://github.com/gradium-ai/xn-ptts
- LinnetTTS: https://huggingface.co/VoxRT/linnet-en-35m
- MOSS-TTS-Nano: https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M, https://github.com/OpenMOSS/MOSS-TTS-Nano
- Own measurements: scripts and outputs were kept in the session scratchpad (not in the repo). Method: onnxruntime 1.32.0 (Python, CPU EP), Apple M2 Max, `intra_op_num_threads` 1 and 4, median of 3–5 runs after one warm-up, outputs checked with whisper `base.en`.

### Secondary 🟡
- VoxRT CPU benchmark (vendor, conflict of interest): https://dev.to/voxrtio/four-on-device-tts-models-on-one-cpu-core-linnettts-piper-kokoro-pocket-tts-4lmc
- Leaderboard summaries: https://micdrop.dev/blog/top-open-source-text-to-speech-models, https://www.cartesia.ai/learn/open-source-tts, https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/, https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models, https://theopenweights.com/leaderboards/text-to-speech, https://www.yaps.ai/blog/best-local-tts-models, https://www.murmurtts.com/blog/best-local-tts-models-2026
- Kyutai TTS: https://kyutai.org/tts/

### Papers downloaded to `papers/` (this area)
| File | Paper | arXiv |
|---|---|---|
| `2017-tacotron2.pdf` | Shen et al., Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions | 1712.05884 |
| `2020-fastspeech2.pdf` | Ren et al., FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | 2006.04558 |
| `2020-glow-tts.pdf` | Kim et al., Glow-TTS: A Generative Flow for TTS via Monotonic Alignment Search | 2005.11129 |
| `2021-vits.pdf` | Kim et al., Conditional VAE with Adversarial Learning for End-to-End TTS | 2106.06103 |
| `2021-yourtts.pdf` | Casanova et al., YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot VC | 2112.02418 |
| `2022-styletts.pdf` | Li et al., StyleTTS: A Style-Based Generative Model for Natural and Diverse TTS | 2205.15439 |
| `2023-styletts2.pdf` | Li et al., StyleTTS 2: Towards Human-Level TTS through Style Diffusion and Adversarial Training with Large SLMs | 2306.07691 |
| `2023-matcha-tts.pdf` | Mehta et al., Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching | 2309.03199 |
| `2023-vall-e.pdf` | Wang et al., Neural Codec Language Models are Zero-Shot TTS Synthesizers | 2301.02111 |
| `2023-soundstorm.pdf` | Borsos et al., SoundStorm: Efficient Parallel Audio Generation | 2305.09636 |
| `2024-e2-tts.pdf` | Eskimez et al., E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS | 2406.18009 |
| `2024-f5-tts.pdf` | Chen et al., F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching | 2410.06885 |
| `2024-xtts.pdf` | Casanova et al., XTTS: a Massively Multilingual Zero-Shot TTS Model | 2406.04904 |
| `2024-parler-tts.pdf` | Lyth & King, Natural Language Guidance of High-Fidelity TTS with Synthetic Annotations | 2402.01912 |
| `2024-cosyvoice2.pdf` | Du et al., CosyVoice 2: Scalable Streaming Speech Synthesis with LLMs | 2412.10117 |
| `2025-calm-pocket-tts.pdf` | Kyutai, Continuous Audio Language Models (the basis of Pocket TTS) | 2509.06926 |

Each was checked to start with `%PDF` and to have the right first-page title. The iSTFTNet paper (Kokoro's vocoder head, `2022-istftnet.pdf`) and HiFi-GAN are in `papers/` from area C.
