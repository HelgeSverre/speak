# C. Vocoders, neural audio codecs and streaming

Area C of the `speak` research brief (`00-brief.md`). It covers the stage that turns an intermediate acoustic representation (mel spectrogram, latent, or codec tokens) into a waveform, and how TTS systems get audio playing before a whole utterance is synthesized.

Confidence markers: 🟢 primary source (paper, repo, docs, or a measurement made for this report), 🟡 secondary (blog, benchmark, third-party summary), 🔴 inferred.

---

## 1. Overview

A vocoder turns a compact, low-rate acoustic representation (typically an 80- or 100-band log-mel spectrogram at about 86–94 frames per second) into 16–44.1 kHz audio. That means producing 256–512 samples per frame, phase included. Each generation of vocoder answered the same question, "how do we produce phase cheaply?", in a different way.

| Era | Technique | Idea | Representative |
|---|---|---|---|
| 1984 → | Signal processing | Estimate phase iteratively from a magnitude spectrogram | Griffin-Lim (1984), Fast Griffin-Lim (2013) |
| 1990s–2010s | Parametric DSP vocoders | Source-filter: F0 + spectral envelope + aperiodicity | STRAIGHT, WORLD (area A) |
| 2016 | Autoregressive neural | Predict one sample at a time | WaveNet (2016), WaveRNN (2018), LPCNet (2018) |
| 2018 | Flow / distillation | Parallel generation by an invertible network | Parallel WaveNet, WaveGlow (2018) |
| 2019–2022 | GAN, time-domain | Transposed-conv upsampling trained with discriminators | MelGAN, Parallel WaveGAN (2019), HiFi-GAN (2020), UnivNet (2021), BigVGAN (2022) |
| 2022–2023 | GAN, Fourier-domain | Predict STFT magnitude + phase and let an inverse FFT do the upsampling | iSTFTNet (2022), Vocos (2023) |
| 2021 → | Neural audio codecs | Encoder + residual vector quantizer + decoder; the decoder is a vocoder for discrete tokens | SoundStream (2021), EnCodec (2022), DAC (2023), Mimi (2024), SNAC (2024) |

Two shifts matter for `speak`:

1. **End-to-end models swallowed the vocoder.** VITS/Piper put a HiFi-GAN generator inside the model, decoding a learned latent `z` rather than a mel. StyleTTS 2/Kokoro put an iSTFTNet-style decoder inside the model, conditioned on F0, energy and a style vector. For those models there is no separate mel → wave stage to swap. 🟢 (Piper's `split_voice_for_streaming.py` cuts the ONNX graph at `dec.conv_pre`; Kokoro's `istftnet.py`.)
2. **Fourier-domain vocoders made CPU vocoding nearly free.** Vocos runs at about 170× real time on a server CPU, against about 6× for HiFi-GAN V1 measured on the same machine (Vocos paper, Table 6). 🟢

---

## 2. Techniques

Shared mel vocabulary: **sr** = sample rate, **n_fft** = FFT size, **hop** = samples per frame (also the upsampling factor), **n_mels** = mel bands. "×RT" means seconds of audio generated per second of compute (higher is faster). Its inverse, RTF, is common too (lower is faster). Speeds from different papers come from different hardware and are not directly comparable.

### 2.1 Griffin-Lim (baseline, pure DSP) 🟢/🟡

- **How it works:** start from a magnitude spectrogram |X| with random or zero phase. Repeat: iSTFT → STFT → keep the new phase, put back the target magnitude. Each iteration lowers the spectral-convergence error. *Fast* Griffin-Lim (Perraudin et al. 2013) adds a momentum term (α≈0.99) and reaches the same quality in about half the iterations (30 FGLA ≈ 60 GLA, according to arXiv 2007.05764). 🟡
- **Input:** a **linear** magnitude spectrogram. From a mel spectrogram you first need an approximate inverse of the mel filterbank (pseudo-inverse or NNLS, as in `librosa.feature.inverse.mel_to_stft`). That inversion is lossy: high-frequency detail goes missing. 🟢 (librosa)
- **Output:** a waveform at the STFT's sr. Any config works.
- **Params:** none. **Licence:** an algorithm, nothing to license.
- **Speed (measured for this report, M2 Max, one thread, naive Rust + `realfft`):** 32 iterations at n_fft 1024 / hop 256 on 10 s of 24 kHz audio took **130 ms, ≈77× RT**. 🟢 (`istft-bench` in the scratchpad, see §4.4)
- **Quality:** characteristic "phasey", metallic, slightly reverberant sound, worst on high frequencies and transients. Tacotron 1 (2017) used Griffin-Lim and scored MOS 3.82. Deep Voice 3 found Griffin-Lim (3.62) close to WORLD (3.63) and below WaveNet (3.78). Every neural-vocoder study rates GL clearly below neural vocoders. 🟡
- **Streaming:** poor. GL is iterative over the whole spectrogram, and phase consistency across chunk borders needs overlap plus a few extra iterations. Online variants exist (RTISI-LA), with lower quality. 🟡
- **Role in `speak`:** zero-dependency reference vocoder and test oracle for the mel/STFT code, and a comparison point for the experimentation goal.

### 2.2 WaveNet (2016) 🟢 — `papers/2016-wavenet.pdf`

- **How:** autoregressive stack of dilated causal convolutions with gated activations. Outputs a 256-way softmax over μ-law-quantized samples (later versions use a mixture of logistics, MoL). Conditioned on linguistic features (original) or mel (Tacotron 2).
- **I/O:** 16 kHz in the original paper; 24 kHz in Tacotron 2 (mel, 12.5 ms hop).
- **Params:** not stated in the original paper. The HiFi-GAN paper's MoL WaveNet baseline has **24.73 M**.
- **Speed:** extremely slow. **0.003× RT on a V100** in the HiFi-GAN comparison, with no CPU number. One forward pass per sample.
- **Quality:** set the bar. MOS 4.21 against 3.86 for concatenative and 3.67 for parametric (US English, original paper). Tacotron 2 + WaveNet reached MOS 4.53 (🟡, Tacotron 2 paper, not downloaded).
- **Streaming:** naturally streaming, since it is sample by sample, but far too slow on CPU.
- **Licence:** paper only. Community implementations are MIT/Apache.
- **Relevance:** historical. Do not implement.

### 2.3 WaveRNN (2018) 🟢 — `papers/2018-wavernn.pdf`

- **How:** a single GRU layer (896 units) predicts each 16-bit sample as coarse + fine 8-bit halves (dual softmax). The paper adds weight pruning (Sparse WaveRNN) and *subscale* generation (16 samples per step) for parallelism.
- **I/O:** 24 kHz 16-bit. Conditioned on linguistic features or mel.
- **Speed:** **4× RT on a P100** with custom GPU kernels. The paper argues Sparse WaveRNN runs in real time on a mobile CPU. That was the first sequential neural model fast enough for on-device use.
- **Quality:** close to WaveNet (paper MOS around 4.4–4.5 for the large model, 🟡 from memory of Table 1; not re-checked).
- **Streaming:** sample-by-sample, so it streams.
- **Licence:** fatchord/WaveRNN implementation is MIT 🟢.
- **Relevance:** historical. The ancestor of LPCNet.

### 2.4 LPCNet (2018) 🟢 — `papers/2018-lpcnet.pdf`

- **How:** a WaveRNN variant that leaves the vocal-tract filter to classic **linear prediction**. A frame-rate network turns features into conditioning, and a sample-rate GRU predicts only the *excitation* (the LPC residual). Uses μ-law, pre-emphasis, and block-sparse GRU matrices.
- **I/O:** 16 kHz. Input is 20 features per 10 ms frame: 18 Bark-scale cepstral coefficients + pitch period + pitch correlation (not a mel).
- **Complexity:** **under 3 GFLOPS**, real time on one phone core (paper). 🟢
- **Quality:** good for 16 kHz speech. It matches WaveRNN at a much lower complexity in the paper's MOS tests.
- **Streaming:** fully causal, 10 ms frames, so excellent.
- **Licence:** **BSD-3-Clause** (xiph/LPCNet) 🟢. Xiph later shipped neural speech coding in Opus 1.5 (FARGAN, an LPCNet successor) 🟡.
- **Relevance:** the most "DSP-friendly" neural vocoder, and a natural fit for a hand-written LPC stack (area A). But its features are not a mel, so it does not pair with off-the-shelf acoustic models. It is also 16 kHz only.

### 2.5 WaveGlow (2018) 🟢 — `papers/2018-waveglow.pdf`

- **How:** normalizing flow (Glow + WaveNet-like coupling layers). Maps Gaussian noise to audio in one parallel pass, conditioned on mel.
- **I/O:** 22.05 kHz, 80 mels, n_fft 1024, hop 256 (Tacotron 2 mel config).
- **Params:** **87.73 M** (HiFi-GAN Table 1).
- **Speed:** "more than 500 kHz" on a V100. On CPU: **0.21× RT**, i.e. slower than real time (HiFi-GAN Table 1, Intel i7 2.6 GHz). 🟢
- **Quality:** MOS 3.81 in HiFi-GAN's comparison, 3.96 in its own paper.
- **Licence:** BSD-3-Clause (NVIDIA/waveglow).
- **Relevance:** none for a CPU tool. Too big and too slow.

### 2.6 MelGAN (2019) 🟢 — `papers/2019-melgan.pdf`

- **How:** first convincing GAN vocoder. Fully convolutional generator (transposed-conv upsampling 8×8×2×2 = 256, residual dilated blocks), multi-scale discriminator, feature-matching loss.
- **I/O:** 22.05 kHz, 80 mels, hop 256.
- **Params:** **4.26 M**.
- **Speed:** 2500 kHz on a GTX 1080 Ti and 50 kHz on CPU (paper). HiFi-GAN measured **6.59× RT on CPU**.
- **Quality:** MOS 3.79 (HiFi-GAN comparison). Audible buzz and metallic artifacts.
- **Licence:** MIT (descriptinc/melgan-neurips). Multi-band MelGAN (2020) is faster still (17.95× RT CPU in the iSTFTNet comparison, 2.54 M params) 🟢.
- **Relevance:** superseded by HiFi-GAN.

### 2.7 Parallel WaveGAN (2019) 🟢/🟡

- **How:** non-autoregressive WaveNet-like generator that maps noise to waveform under mel conditioning, trained with a GAN plus a **multi-resolution STFT loss**. That loss became a standard ingredient in later vocoders.
- **Params:** **1.44 M**. **Speed:** 28.68× RT on one GPU at 24 kHz (abstract). On CPU it is only **0.70× RT** in the iSTFTNet comparison 🟢. A small model is slow on CPU because the network runs at the full sample rate.
- **Quality:** MOS 4.16 in a Transformer-TTS setup (abstract).
- **Licence:** MIT (kan-bayashi/ParallelWaveGAN, which also hosts many pretrained vocoders).

### 2.8 HiFi-GAN (2020) 🟢 — `papers/2020-hifigan.pdf`

- **How:** transposed-conv upsampling generator with **multi-receptive-field fusion** (MRF: parallel residual blocks with kernels 3/7/11 and dilations 1/3/5), trained against a **multi-period discriminator** (MPD: views the waveform reshaped by periods 2, 3, 5, 7, 11) plus MelGAN's multi-scale discriminator, and an L1 mel loss (weight 45).
- **I/O:** 22.05 kHz, 80 mels, n_fft 1024, win 1024, hop 256 (upsample 8×8×2×2). fmax 8000 in the LJSpeech config. Retrains for other configs are common.
- **Variants (paper Table 1, CPU = Intel i7 2.6 GHz, fp32):**

| Variant | Params | MOS | CPU ×RT | V100 ×RT |
|---|---|---|---|---|
| V1 | 13.92 M | 4.36 | 1.43 | 167.9 |
| V2 | 0.92 M | 4.23 | 9.74 | 764.8 |
| V3 | 1.46 M | 4.05 | 13.44 | 1186.8 |
| (ground truth) | | 4.45 | | |

- **Streaming:** non-causal but with a finite receptive field. Chunk the mel with a few frames of context on each side, and the output is bit-identical to a full decode (libpiper relies on this, see §2.17). 🟢
- **ONNX:** widely exported. sherpa-onnx ships `hifigan_v1/v2/v3.onnx` for Matcha-TTS. Piper voices embed a HiFi-GAN-style decoder inside the VITS ONNX graph. 🟢
- **Licence:** MIT (jik876/hifi-gan), weights included 🟢.
- **Relevance:** the default "real" neural vocoder for mel-based models (Matcha-TTS, FastSpeech 2, Tacotron 2). V2 is the CPU-friendly sweet spot.

### 2.9 UnivNet (2021) 🟢/🟡

- **How:** generator built from **location-variable convolutions** (LVC: convolution kernels predicted from the mel per segment), plus a **multi-resolution spectrogram discriminator** (MRD) on linear spectrograms. MRD was later reused by Vocos and BigVGAN.
- **I/O:** 24 kHz, 100 mels, hop 256 (fixed), fmax 12 kHz (maum-ai/univnet README).
- **Params:** c16 **4.00 M**, c32 **14.86 M** 🟢 (README). MOS 3.59 / 3.70 (README's own evaluation).
- **Speed:** "real time". No CPU number found 🔴.
- **Licence:** BSD-3-Clause (maum-ai/univnet).
- **Relevance:** mainly a stepping stone. Its 100-band, 24 kHz mel config is the one Vocos and BigVGAN-24k use.

### 2.10 BigVGAN (2022; v2 2024) 🟢 — `papers/2022-bigvgan.pdf`

- **How:** HiFi-GAN generator with **Snake** periodic activations (x + sin²(αx)/α) and **anti-aliased** nonlinearities (upsample 2×, activate, low-pass, downsample) inside every residual block (the AMP module). Plus MPD + MRD. Aimed at a *universal* vocoder for unseen speakers, languages, singing and instruments.
- **Params:** base **14 M**, large **112 M**.
- **I/O (v2 checkpoints, all MIT on Hugging Face):** 24 kHz / 100 bands / hop 256, 22 kHz / 80 bands / hop 256, 44 kHz / 128 bands / hop 512, and others. 🟢
- **Speed:** on CPU, **0.40× RT, slower than real time** (Vocos Table 6, AMD EPYC 7542, PyTorch). The anti-aliasing filters run at 2× the sample rate in every layer. v2 ships a CUDA kernel, which does nothing on a Mac.
- **Quality:** best objective scores among GAN vocoders. Vocos is roughly on par in the Vocos paper (UTMOS 3.749 vs 3.734, MOS 3.64 vs 3.62).
- **Streaming:** chunkable with context, like HiFi-GAN.
- **ONNX:** possible in principle (convs + elementwise ops), but no official export found 🔴.
- **Licence:** code and v2 weights **MIT** 🟢.
- **Relevance:** quality reference only. Too slow on laptop CPU without optimization.

### 2.11 iSTFTNet (2022) 🟢 — `papers/2022-istftnet.pdf`

- **How:** take HiFi-GAN and **remove the last upsampling blocks**. Instead of upsampling all the way to the sample rate, the network outputs a small STFT: magnitude (via `exp`) and phase (via `sin`) at a reduced time resolution. A cheap **iSTFT** does the final upsampling. The naming tells you how many upsampling blocks remain: "C8C8I" = two 8× conv blocks then iSTFT, with **n_fft 16, hop 4**.
- **I/O:** 22.05 kHz, 80 mels, n_fft 1024, hop 256 (mel analysis). The internal iSTFT is tiny.
- **Results (paper Table 1; CPU = MacBook Pro, 2.7 GHz Intel i7):**

| Model | MOS | CPU ×RT | Params |
|---|---|---|---|
| HiFi-GAN V1 | 4.22 | 1.34 | 13.94 M |
| V1-C8C8I | 4.26 | **2.33** | 13.26 M |
| HiFi-GAN V2 | 3.91 | 10.39 | 0.93 M |
| V2-C8C8I | 3.95 | **20.37** | 0.89 M |
| V1-C8I (too far) | 3.32 | 7.57 | 10.89 M |

  Removing two upsampling stages (C8C8I) roughly **doubles CPU speed with no MOS loss**. Removing three (C8I) breaks quality.
- **Used in:** StyleTTS 2 and **Kokoro**. Kokoro's decoder config is `upsample_rates [10, 6]`, `gen_istft_n_fft 20`, `gen_istft_hop_size 5`. Total upsampling 10·6·5 = 300 samples per frame at 24 kHz, with a harmonic-plus-noise (hn-NSF) sine source driven by predicted F0 and AdaIN style conditioning 🟢 (Kokoro `config.json`, `istftnet.py`). Kokoro also ships `CustomSTFT`, which implements STFT/iSTFT with `conv1d`/`conv_transpose1d` because ONNX has no iSTFT op and `torch.istft` does not export 🟢.
- **Licence:** paper only. rishikksh20/iSTFTNet-pytorch is Apache-2.0. StyleTTS 2 is MIT. Kokoro is Apache-2.0 🟢.

### 2.12 Vocos (2023) 🟢 — `papers/2023-vocos.pdf`

- **How:** goes all the way: **no upsampling layers at all**. A ConvNeXt backbone (8 blocks, dim 512, depthwise conv k=7) runs at the **frame rate** (an "isotropic" network). A linear head outputs n_fft+2 channels per frame, split into log-magnitude m and phase p. Magnitude = exp(m), and the complex spectrum = exp(m)·(cos p + j sin p). That puts phase on the unit circle, so the network never has to wrap it. One full-size iSTFT (n_fft 1024, hop 256) makes the waveform. Trained with MPD + MRD, a hinge loss and an L1 mel loss.
- **I/O (`charactr/vocos-mel-24khz`):** **24 kHz, n_fft 1024, hop 256, 100 mels**, center padding. Log-mel uses torchaudio-style features 🟢 (config.yaml). A separate checkpoint decodes **EnCodec tokens** (drop-in replacement for EnCodec's decoder, used with Bark). Community 22.05 kHz / 80-band ports exist (BSC-LT `vocos-mel-22khz`, Apache-2.0; sherpa-onnx `vocos-22khz-univ.onnx`, 51 MB) 🟢.
- **Params:** **13.5 M**.
- **Speed (paper Table 6, batch 16 × 1 s, PyTorch, A100 / AMD EPYC 7542):**

| Model | GPU ×RT | CPU ×RT | Params |
|---|---|---|---|
| HiFi-GAN | 495.5 | 5.84 | 14.0 M |
| BigVGAN | 98.6 | 0.40 | 14.0 M |
| iSTFTNet | 1045.9 | 14.44 | 13.3 M |
| **Vocos** | **6696.5** | **169.63** | 13.5 M |

- **Quality:** on par with BigVGAN-base and above HiFi-GAN/iSTFTNet (UTMOS 3.734; PESQ 3.70, the best of the group; MOS 3.62 ± 0.15 against ground truth 3.81 in its test).
- **Streaming:** non-causal, but the receptive field is small: embed conv k=7 + 8 × depthwise k=7 ≈ **±27 frames (≈290 ms)** 🔴 (computed from `vocos/models.py`), plus 3 frames of iSTFT overlap. Chunk with that much context for an exact match, or less for a near-match.
- **ONNX:** the standard export (sherpa-onnx, `scripts/vocos`) puts **only the backbone + head** in ONNX, outputting `mag`, `x = cos p`, `y = sin p` with shape (N, n_fft/2+1, frames). The host code does the iSTFT (sherpa-onnx uses `kaldi-native-fbank`'s `IStft`, metadata `n_fft`, `hop_length`, `center`, `window_type`) 🟢. That is exactly the split a Rust tool wants.
- **Licence:** code **MIT**, weights **MIT** (`charactr/vocos-mel-24khz`, `vocos-encodec-24khz`) 🟢.
- **Relevance:** **best mel vocoder for `speak`.** It is fast enough that vocoder cost becomes noise, and it is small and permissively licensed.

### 2.13 Why iSTFT-based vocoders are fast 🟢/🔴

1. **The network never runs at audio rate.** A time-domain GAN upsamples 256× through transposed convs, so its last, widest-in-time layers process 22,050–24,000 positions per second, and that is where the FLOPs go. Vocos processes ~94 frames per second, about 256× fewer positions. iSTFTNet C8C8I stops at 1/4 of the sample rate. 🟢 (Vocos §3.1; iSTFTNet §3)
2. **The inverse FFT is an exact, parameter-free, O(N log N) upsampler.** Generating 256 samples from 513 complex bins costs one 1024-point real iFFT plus a windowed overlap-add. Measured here: **≈1.5 ms for 10 s of audio, ≈6,700× RT, single thread** (§4.4). 🟢
3. **No aliasing fixes needed.** Transposed convs alias, which is why BigVGAN adds 2×-oversampled filtered activations that make it 14–400× slower on CPU. The iSTFT has no such problem. 🟢 (Vocos §3.1)
4. **Phase is the hard part, and the network only has to output it at frame rate.** Vocos's cos/sin trick avoids phase wrapping, and iSTFTNet's small FFT keeps the phase problem tractable. 🟢
5. **Isotropic ConvNeXt with depthwise convs** is BLAS-friendly: big pointwise matmuls (512 → 1536 → 512) at low time resolution vectorize well on Apple's AMX/NEON through ONNX Runtime or candle. 🔴

### 2.14 What a pure-Rust iSTFT needs 🟢

The full recipe, verified by the benchmark in §4.4:

- **Crates:** `realfft` 3.5.x (real ↔ complex wrapper over `rustfft`; an even-length N real FFT uses an N/2 complex FFT; inverse takes N/2+1 bins → N reals) and `rustfft` 6.4.x (pure Rust, any length including primes, **NEON on aarch64 auto-detected**, AVX/SSE on x86). No C dependencies. Both are MIT/Apache-2.0. 🟢 (READMEs, crates.io)
- **Plan once, reuse:** `RealFftPlanner::<f32>::new().plan_fft_inverse(n_fft)` returns an `Arc<dyn ComplexToReal<f32>>`. Keep the plan and scratch buffers in the vocoder struct.
- **Per frame:** copy N/2+1 complex bins into the input buffer. **Zero the imaginary part of the DC and Nyquist bins**: `realfft` returns an error otherwise, and a network's output is not guaranteed to have zeros there. Run the inverse, scale by 1/N (realfft is unnormalized), multiply by the synthesis window, and overlap-add at `frame * hop`.
- **Window-sum normalization:** accumulate `window²` per sample and divide at the end (or precompute the constant for hann with hop = n_fft/4). This is what `torch.istft` does, and it makes STFT→iSTFT the identity. Measured max reconstruction error: 1.5e-7.
- **Window:** periodic Hann (`torch.hann_window(periodic=True)`, i.e. divide by N, not N−1). It must match the training code. A 5-line function, so no crate needed.
- **Padding mode must match training:** `center=True` (torch default: reflect-pad n_fft/2 at both ends, then trim n_fft/2 from the output), or Vocos's `"same"` padding (trim (win−hop)/2). Kokoro's `CustomSTFT` uses replicate padding for ONNX. A mismatch gives clicks at the start/end and a time offset.
- **Streaming iSTFT:** overlap-add is naturally streamable. Keep an `n_fft − hop` tail buffer. Each new frame completes `hop` samples, which can be emitted as soon as all overlapping frames have arrived. Latency is n_fft − hop samples (32 ms at 24 kHz / 1024 / 256).
- **Odd sizes:** Kokoro's internal n_fft is 20 (hop 5). `rustfft` handles non-power-of-two sizes (measured: same ≈1.5 ms per 10 s). For very small N, a direct DFT matrix multiply is also fine.
- **Mel analysis (for Griffin-Lim, tests, or a voice-cloning/feature path):** you also need the forward STFT and a mel filterbank. Note that **librosa (Slaney scale, Slaney norm; HiFi-GAN, Matcha) and torchaudio defaults (HTK scale, no norm; Vocos) produce different filterbanks.** The vocoder must see exactly the mel it was trained on: same sr, n_fft, hop, win, n_mels, fmin/fmax, log base and clamp (HiFi-GAN: `ln(clamp(x, 1e-5))`; Vocos: `ln(clamp(x, 1e-7))` 🟡). This is the most common source of "the vocoder sounds broken" bugs. 🔴

### 2.15 Neural audio codecs

A codec is an encoder → quantizer → decoder trained end to end. For TTS, only the **decoder** matters, and only when the acoustic model emits codec tokens (VALL-E, Bark, Orpheus, Moshi, CSM, many LLM-based TTS models). The codec decoder then *is* the vocoder. RVQ = residual vector quantization: each codebook quantizes the residual left by the previous one, and dropping codebooks lowers the bitrate.

| Codec | Year | sr | Frame rate | Tokens / bitrate | Causal / streaming | Params | Licence (code / weights) |
|---|---|---|---|---|---|---|---|
| SoundStream 🟡 | 2021 | 24 kHz | 75 Hz | RVQ, 3–18 kbps | Yes, real time on a phone CPU (paper) | ? | Paper only. Lives on in Lyra v2 (Apache-2.0) |
| EnCodec 🟢 | 2022 | 24 kHz (mono, causal) / 48 kHz (stereo, non-causal) | 75 Hz at 24 kHz | Up to 32 × 1024-entry codebooks: 1.5/3/6/12/24 kbps | 24 kHz model streams with **13.3 ms** initial latency; **≈10× RT encode and decode on one MacBook Pro 2019 thread** at 6 kbps (Table 5) | ? (not checked) | MIT code. HF weight tag not set 🔴 |
| DAC 🟢 | 2023 | 44.1 kHz (also 16/24 kHz) | 86 Hz | 9 × 10-bit codebooks = 8 kbps (≈90× compression) | Non-causal | ? (≈70–80 M 🔴) | MIT code. HF weight tag not set 🔴 |
| Mimi 🟢 | 2024 | 24 kHz | **12.5 Hz** | 8 × 2048 codebooks in Moshi (1st distilled from WavLM = "semantic"), ≈1.1 kbps | Fully causal. **80 ms frame**: 80 ms in gives 80 ms out | ? | Code Apache-2.0 (kyutai-labs/moshi). Weights **CC-BY-4.0** |
| SNAC 🟢 | 2024 | 24 kHz (speech) / 32 / 44 kHz | Multi-scale: coarse tokens at a lower rate than fine ones | 24 kHz: 0.98 kbps | Non-causal. Orpheus decodes it with a sliding window | **19.8 M** (24 kHz), 54.5 M (32/44 kHz) | MIT / MIT |

Codec details worth knowing:

- **EnCodec's streaming mode:** all conv padding goes before the first time step, and weight norm replaces layer norm. The 48 kHz model needs 1 s chunks because of its normalization. Entropy coding adds one frame of latency. 🟢 (EnCodec §3.1, §4.6)
- **Mimi** adds Transformers to the encoder and decoder (causal masked) and gets streaming ASR/TTS out of Moshi with a theoretical 160 ms (≈200 ms in practice) full-duplex latency. 🟢
- **Orpheus + SNAC streaming:** the LLM emits 7 tokens per SNAC frame. Once 28 tokens (4 frames) are buffered, the decoder runs on the last 28 at every new frame and keeps samples 2048–4096 (85 ms at 24 kHz) of the decoded window. Brute-force overlap that re-decodes a 4-frame window for every frame. 🟢 (`orpheus_tts/decoder.py`)
- **Vocos-EnCodec** shows a Fourier-domain decoder can replace a codec's own decoder with better quality, at 7.9 M params in the paper's matched comparison. 🟢 (Vocos §4.4)

**Relevance to `speak`:** low for the first versions. Codec decoders only matter if `speak` adopts a codec-LM TTS (Orpheus/SNAC, CSM/Mimi, Bark/EnCodec), and those need a multi-billion-parameter LLM in front, which conflicts with fast startup on a laptop CPU (area B). Mimi (12.5 Hz, causal, 80 ms) and SNAC (20 M params, MIT) are the ones to keep in mind. 🔴

### 2.16 Streaming and latency: concepts 🟢/🔴

- **RTF** = compute time ÷ audio duration. To stream without gaps you need RTF < 1 *sustained*, with margin for scheduling jitter. Aim for RTF ≤ 0.3 on the target CPU.
- **Time to first audio (TTFA)** = text in → first sample at the DAC. For an offline CLI: TTFA = process start + model load + text frontend for chunk 1 + acoustic model for chunk 1 + vocoder for the first decoded block + audio device open/buffer. **Model load (mmap, ONNX session creation) often dominates in a CLI.** kokoro-onnx reportedly takes ≈1.7 s to load on an M3 Pro 🟡. Measure it separately.
- **Three levels of chunking**, from coarse to fine:
  1. **Text chunking:** split into sentences (then clauses at `, ; : —` if long). Synthesize chunk *k+1* while chunk *k* plays. This is the main trick in Piper, Kokoro and sherpa-onnx. Cost: prosody does not carry across sentence borders, so pauses between chunks must be inserted deliberately.
  2. **Decoder chunking:** run the non-autoregressive acoustic model/encoder on the whole sentence (cheap), then run the vocoder/decoder over the latent in chunks with context frames, and play the first chunk immediately (libpiper). Exact if the context ≥ receptive field.
  3. **Causal/frame-streaming models:** causal convs plus a state cache, emitting audio per frame (EnCodec-streaming, Mimi at 80 ms, LPCNet at 10 ms). Requires a model trained causally.
- **First-chunk sizing:** make the *first* chunk small (fast first sound), then grow chunks while staying ahead of playback. Small chunks waste compute on context overhead, so do not keep them small.

### 2.17 How existing engines stream

**Piper / libpiper (OHF-Voice/piper1-gpl, GPL-3.0)** 🟢

- espeak-ng phonemizes and **splits text into sentences**. `PiperVoice.synthesize()` yields one `AudioChunk` per sentence (`voice.py`), and the CLI can stream raw PCM per sentence.
- Newer libpiper adds **decoder streaming**. `split_voice_for_streaming.py` cuts the VITS ONNX graph at the decoder input `z` (`dec.conv_pre`) into `<model>.encoder.onnx` (phoneme ids → z, durations/alignments) and `<model>.decoder.onnx` (z → audio). libpiper then decodes `z` in chunks (`piper.cpp`):
  - `STREAM_CONTEXT_FRAMES = 12` latent frames on each side. The comment says this makes the output **identical** to a full decode, because the decoder's receptive field is smaller.
  - `STREAM_FIRST_CHUNK_FRAMES = 32` for a fast first sound. At 22.05 kHz and hop 256 that is ≈0.37 s of audio 🔴 (computed).
  - Later chunks between 24 and 256 frames, sized so decoding takes ≤ 50 % (`STREAM_BUFFER_SHARE`) of the audio still buffered. This adaptive scheme keeps playback gapless as context overhead shrinks.
  - `MAX_INTRA_OP_THREADS = 4`.
- Piper voices: 22.05 kHz (medium/high) or 16 kHz (low/x_low). Mel/STFT config is VITS's 1024/256/1024. No published Apple Silicon RTF. Third parties report roughly 5–10× faster than real time on desktop CPUs and around real time on a Raspberry Pi 4 🟡.

**Kokoro (hexgrad/kokoro, Apache-2.0)** 🟢

- `KPipeline` G2P (misaki) produces phonemes, and the text is chunked so each chunk stays **≤ 510 phoneme tokens** (the model's context limit). Split points follow a "waterfall": prefer `!.?…`, then `:;`, then `,—`. The pipeline is a generator yielding `(graphemes, phonemes, audio)` per chunk, so a caller can play chunk 1 while chunk 2 renders. Default `split_pattern = r'\n+'` splits on newlines first.
- Kokoro is non-autoregressive per chunk with no built-in decoder chunking. TTFA = full synthesis of chunk 1, so **a short first chunk** is the main lever 🔴.
- Speed: kokoro-onnx on an **M3 Pro CPU ≈ 0.16 RTF** (22 s of audio in 3.5 s) and ≈1.7 s model load 🟡. Browser WASM on an M1 Max: RTF 0.83 (8 threads, q8) to 2.0 (1 thread) 🟡. A 4-core Xeon gives ≈0.57–0.64 RTF 🟡. ONNX exports: fp32 325 MB, q8 92 MB (onnx-community) 🟡.

**sherpa-onnx (k2-fsa, Apache-2.0)** 🟢

- `OfflineTtsConfig.max_num_sentences = 1` by default. The VITS/Matcha/Kokoro implementations split text into sentences (on `. ! ? ; :` and CJK equivalents), synthesize in batches of `max_num_sentences`, and call a **`GeneratedAudioCallback(samples, n, progress)` after each batch**. The callback can return 0 to cancel. `silence_scale` (default 0.2) shrinks pauses. This is the "play as you go" hook (`offline-tts.h`, `offline-tts-vits-impl.h`).
- Separate vocoders for Matcha: `vocos-22khz-univ.onnx` (51 MB) or `hifigan_v1/2/3.onnx`. Matcha + Vocos on a **Raspberry Pi 4: RTF 0.94 on 1 thread, 0.41 on 4 threads** 🟢 (sherpa docs).

**Commercial cloud engines** 🟡

- Vendor claims: ElevenLabs Flash v2.5 "~75 ms" (model only, excluding network). Cartesia Sonic 40–90 ms.
- Independent measurements (Coval and others, 2026): **≈150–450 ms TTFA** end to end, including network and leading silence. They use autoregressive or frame-streaming models with WebSocket token-by-token text input.
- Conversational targets: Moshi's theoretical 160 ms. Human turn-taking averages ≈230 ms (Moshi paper) 🟢.

**macOS `say` / AVSpeechSynthesizer:** no published latency numbers found 🔴. Measure it as `speak`'s baseline (e.g. `time say -o /dev/null "Hello"` and time-to-first-callback in a small Swift harness). Subjectively, it starts almost instantly.

---

## 3. Comparison table

CPU ×RT figures come from each paper's own CPU and are only comparable within a column source. "In-paper" means the number comes from that model's own paper or from the comparison table named.

| Vocoder | Type | sr / n_fft / hop / n_mels | Params | CPU ×RT (source) | Quality (MOS/UTMOS, source) | Streaming | ONNX | Licence (code / weights) | Conf. |
|---|---|---|---|---|---|---|---|---|---|
| Griffin-Lim | DSP iterative | any (needs linear mag) | 0 | ≈77× at 32 iters, Rust, M2 Max (this report) | ~3.6 (Tacotron/DV3 era) | Poor | n/a (pure Rust) | none | 🟢/🟡 |
| WaveNet (MoL) | AR conv | 16–24 kHz | 24.7 M | GPU 0.003× (HiFi-GAN T1) | 4.02 (HiFi-GAN T1) | Yes, too slow | Impractical | paper | 🟢 |
| WaveRNN | AR RNN | 24 kHz | ? | real-time on mobile (claim) | ≈WaveNet | Yes | Rare | MIT (fatchord) | 🟢/🟡 |
| LPCNet | AR RNN + LPC | 16 kHz, 20 cepstral feats / 10 ms | small (<3 GFLOPS) | real-time on one phone core | good at 16 kHz | Excellent (causal) | Own C code | BSD-3 | 🟢 |
| WaveGlow | Flow | 22.05k / 1024 / 256 / 80 | 87.7 M | 0.21× (HiFi-GAN T1) | 3.81 | Chunkable | Yes (NVIDIA) | BSD-3 | 🟢 |
| MelGAN | GAN time | 22.05k / 1024 / 256 / 80 | 4.26 M | 6.6× (HiFi-GAN T1) | 3.79 | Chunkable | Yes | MIT | 🟢 |
| Parallel WaveGAN | GAN time | 24k / – / 300 / 80 | 1.44 M | 0.70× (iSTFTNet T1) | 3.47 (iSTFTNet T1) | Chunkable | Yes (community) | MIT | 🟢 |
| HiFi-GAN V1 | GAN time | 22.05k / 1024 / 256 / 80 | 13.9 M | 1.43× (own) / 5.84× (Vocos T6) | 4.36 (own) | Chunkable, exact with context | **Yes** (sherpa-onnx, Piper-internal) | MIT / MIT | 🟢 |
| HiFi-GAN V2 | GAN time | same | 0.92 M | 9.7× (own) | 4.23 (own) | Chunkable | **Yes** | MIT / MIT | 🟢 |
| HiFi-GAN V3 | GAN time | same | 1.46 M | 13.4× (own) | 4.05 (own) | Chunkable | **Yes** | MIT / MIT | 🟢 |
| UnivNet c32 | GAN LVC | 24k / 1024 / 256 / 100 | 14.9 M | ? | 3.70 (README) | Chunkable | Community | BSD-3 | 🟢/🔴 |
| BigVGAN v2 | GAN time + Snake + AA | 24k/100/256, 22k/80/256, 44k/128/512 | 14 M base / 112 M | **0.40×** (Vocos T6) | UTMOS 3.749 (Vocos T1), best | Chunkable | Not official | MIT / MIT | 🟢 |
| iSTFTNet V1-C8C8I | GAN + small iSTFT | 22.05k / 1024 / 256 / 80 (internal iSTFT 16/4) | 13.3 M | 2.33× (own) / 14.4× (Vocos T6) | 4.26 (own) | Chunkable | Inside Kokoro ONNX | Apache-2.0 (community) | 🟢 |
| Kokoro decoder | iSTFTNet + hn-NSF + AdaIN | 24k, internal iSTFT 20/5, ×300 | part of 82 M (split unknown) | whole model ≈0.16 RTF on M3 Pro 🟡 | Kokoro ranks high in TTS Arena (area B) | Per text chunk | Whole model only | Apache-2.0 | 🟢/🟡 |
| **Vocos (mel-24k)** | ConvNeXt + full iSTFT | **24k / 1024 / 256 / 100** | 13.5 M | **169.6×** (own, EPYC) | UTMOS 3.734, MOS 3.62 (≈BigVGAN) | Chunkable (±27 frames) | **Yes**: backbone ONNX + host iSTFT (sherpa-onnx) | **MIT / MIT** | 🟢 |
| EnCodec 24k decoder | Codec | 24k, 75 Hz tokens | ? | ≈10× on one thread (own) | codec | **Causal**, 13 ms | Community | MIT / unclear | 🟢/🔴 |
| Mimi decoder | Codec + Transformer | 24k, 12.5 Hz | ? | ? | codec | **Causal, 80 ms** | Community (MLX/candle/Rust in moshi repo) | Apache-2.0 / CC-BY-4.0 | 🟢 |
| SNAC 24k decoder | Codec, multi-scale | 24k, 0.98 kbps | 19.8 M | ? | codec | Sliding window | Community | MIT / MIT | 🟢 |

---

## 4. What it means for `speak`

### 4.1 Vocoder stage design 🔴 (recommendations, based on the 🟢 facts above)

1. **Make the vocoder a trait over "acoustic features → PCM", and give the features a typed config.** Something like `trait Vocoder { fn config(&self) -> &FeatureConfig; fn decode(&mut self, frames: FrameBlock, sink: &mut dyn PcmSink) }`, where `FeatureConfig { kind: Mel|Latent|CodecTokens|Lpc, sample_rate, n_fft, hop, win, n_mels, fmin, fmax, mel_scale: Slaney|Htk, log: Ln{clamp}|Log10 }`. Refuse to pair an acoustic model and vocoder whose configs differ. That catches the most common silent-quality bug at load time.
2. **Implement three vocoders, in this order:**
   - **Griffin-Lim (pure Rust, `realfft`).** Zero model files. It validates the STFT/mel/iSTFT code and is the "classic" baseline for A/B listening. ≈77× RT single-threaded with 32 iterations, so speed is not a problem. Use FGLA momentum (α ≈ 0.99) to get by with fewer iterations.
   - **Vocos via ONNX + Rust iSTFT.** Load the sherpa-onnx-style export (backbone outputs mag/cos/sin) and do the iSTFT in Rust. You get the best speed/quality/licence trade-off available (13.5 M params, MIT weights, ~170× RT on a server CPU) and keep the DSP tail in your own code. Use the 24 kHz/100-band model for new acoustic models, and the 22.05 kHz/80-band community port for Matcha-TTS / HiFi-GAN-config mels.
   - **HiFi-GAN V2/V1 via ONNX.** A second neural datapoint, and the vocoder most mel-based acoustic models were trained against.
   - Skip WaveNet, WaveRNN, WaveGlow and BigVGAN (too slow on CPU or obsolete). LPCNet is an interesting "classic + neural" hybrid if area A's LPC path gets built, but it needs its own 16 kHz cepstral features.
3. **Accept that end-to-end backends own their vocoder.** For Piper (VITS) and Kokoro, the "vocoder" is part of the ONNX graph. Model the backend boundary as "phonemes → PCM stream", and the vocoder trait as an *optional* inner stage for mel-producing acoustic models (Matcha, FastSpeech 2, a hand-written parametric model). Piper's encoder/decoder split (cut at `z`) shows how to get decoder chunking anyway. The same graph surgery could apply to Kokoro's decoder 🔴.
4. **Codec decoders (SNAC/Mimi/EnCodec) are out of scope for v1.** Add a `CodecTokens` feature kind later if a codec-LM backend is ever tried.

### 4.2 Rust iSTFT / DSP module

- Depend on `realfft` 3.5 (which pulls in `rustfft` 6.4 with NEON). Write your own periodic Hann window, overlap-add with window-sum normalization, `center`/`same` padding modes, and a streaming OLA struct with an `n_fft − hop` tail. About 150 lines, verified in §4.4.
- Zero the imaginary part of the DC/Nyquist bins before `process()`.
- Add the forward STFT + mel filterbank (Slaney and HTK variants) for Griffin-Lim and for golden tests against `librosa`/`torchaudio` outputs stored as fixtures.
- Resample (`rubato`) only at the output edge, e.g. 22.05/24 kHz → device rate. Better: open the cpal stream at the model's rate when the device supports it.

### 4.3 Streaming design for `speak`

1. **Sentence/clause chunker in the frontend.** Split on `. ! ? …` first, then on `; : —`, then on `,` for long sentences, with a token cap per chunk (Kokoro's waterfall, 510 tokens). **Make the first chunk short**: cut at the first clause boundary after a few words, so TTFA depends on a small chunk.
2. **Pipeline with bounded channels:** `text → [frontend thread] → phoneme chunks → [synth thread] → PCM blocks → [ring buffer] → cpal callback`. Synthesize chunk *k+1* while *k* plays, and insert explicit silence between chunks (sherpa-onnx scales pauses with `silence_scale`).
3. **Decoder chunking where the model allows it** (Piper split model, Vocos, HiFi-GAN): copy libpiper's scheme. Use a first chunk of ~32 frames (≈0.35 s), context = receptive field each side (12 frames for Piper's decoder, ≈27 for Vocos), and later chunks sized so decode time ≤ 50 % of buffered audio, capped at 256 frames. For Vocos, overlap-add the iSTFT across chunk borders instead of cross-fading.
4. **Measure TTFA as its own metric, broken down:** process start → model loaded → first phonemes → first PCM → first callback. Startup (ONNX session creation, model mmap) will likely dominate for a CLI. Consider lazy session creation, smaller (q8) models, and caching the optimized ONNX graph 🔴.
5. **Targets, inferred from the numbers above:** on an M-series laptop, a Piper-medium or Matcha+Vocos backend should reach **TTFA < 150 ms after model load** and RTF < 0.1. Kokoro, with a short first chunk, should reach **≈200–400 ms TTFA** and RTF ≈ 0.15. Griffin-Lim with a parametric model should be near-instant. For comparison, cloud engines measure 150–450 ms in independent tests. These are estimates for `spec.md` to adopt as goals, not measurements. 🔴

### 4.4 Measurement made for this report 🟢

A throwaway Rust benchmark (`realfft` 3.5.1, release build, Apple M2 Max, one thread, 10 s of 24 kHz audio, naive code with no buffer reuse across calls):

| Operation | Time | ×RT | Notes |
|---|---|---|---|
| iSTFT n_fft 1024 / hop 256 (Vocos-size) | 1.49 ms | ≈6,700× | max reconstruction error 1.5e-7 vs input |
| iSTFT n_fft 20 / hop 5 (Kokoro-internal size) | 1.48 ms | ≈6,800× | non-power-of-two works fine |
| Griffin-Lim, 32 iterations (STFT + iSTFT each) | 130 ms | ≈77× | could be multithreaded per frame |

Conclusion: in a Vocos-style vocoder, the DSP tail costs essentially nothing, so all the cost is in the neural backbone.

---

## 5. Sources

### Papers downloaded to `papers/` (all verified `%PDF`)

| File | Paper | arXiv |
|---|---|---|
| `2016-wavenet.pdf` | van den Oord et al., *WaveNet: A Generative Model for Raw Audio* | 1609.03499 |
| `2018-wavernn.pdf` | Kalchbrenner et al., *Efficient Neural Audio Synthesis* (WaveRNN) | 1802.08435 |
| `2018-lpcnet.pdf` | Valin & Skoglund, *LPCNet: Improving Neural Speech Synthesis Through Linear Prediction* | 1810.11846 |
| `2018-waveglow.pdf` | Prenger et al., *WaveGlow: A Flow-based Generative Network for Speech Synthesis* | 1811.00002 |
| `2019-melgan.pdf` | Kumar et al., *MelGAN: GANs for Conditional Waveform Synthesis* | 1910.06711 |
| `2020-hifigan.pdf` | Kong et al., *HiFi-GAN: GANs for Efficient and High Fidelity Speech Synthesis* | 2010.05646 |
| `2022-istftnet.pdf` | Kaneko et al., *iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse STFT* | 2203.02395 |
| `2022-bigvgan.pdf` | Lee et al., *BigVGAN: A Universal Neural Vocoder with Large-Scale Training* | 2206.04658 |
| `2023-vocos.pdf` | Siuzdak, *Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders* | 2306.00814 |
| `2022-encodec.pdf` | Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec) | 2210.13438 |
| `2023-dac.pdf` | Kumar et al., *High-Fidelity Audio Compression with Improved RVQGAN* (DAC) | 2306.06546 |
| `2024-moshi-mimi.pdf` | Défossez et al., *Moshi: a speech-text foundation model for real-time dialogue* (Mimi codec) | 2410.00037 |

Not downloaded but cited: Parallel WaveGAN https://arxiv.org/abs/1910.11480, UnivNet https://arxiv.org/abs/2106.07889, SoundStream https://arxiv.org/abs/2107.03312, SNAC https://arxiv.org/abs/2410.14411, Fast Griffin-Lim for TTS https://arxiv.org/abs/2007.05764.

### Repositories and model cards

- HiFi-GAN: https://github.com/jik876/hifi-gan (MIT)
- BigVGAN: https://github.com/NVIDIA/BigVGAN (MIT); weights https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x, https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x, https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x (MIT)
- Vocos: https://github.com/gemelo-ai/vocos (MIT); https://huggingface.co/charactr/vocos-mel-24khz (config.yaml; MIT); https://huggingface.co/charactr/vocos-encodec-24khz; https://huggingface.co/BSC-LT/vocos-mel-22khz (Apache-2.0); source files `vocos/models.py`, `vocos/modules.py`, `vocos/spectral_ops.py`
- iSTFTNet community: https://github.com/rishikksh20/iSTFTNet-pytorch (Apache-2.0)
- UnivNet: https://github.com/maum-ai/univnet (BSD-3)
- Parallel WaveGAN: https://github.com/kan-bayashi/ParallelWaveGAN (MIT)
- MelGAN: https://github.com/descriptinc/melgan-neurips (MIT)
- WaveGlow: https://github.com/NVIDIA/waveglow (BSD-3)
- WaveRNN: https://github.com/fatchord/WaveRNN (MIT)
- LPCNet: https://github.com/xiph/LPCNet (BSD-3)
- EnCodec: https://github.com/facebookresearch/encodec (MIT); https://huggingface.co/facebook/encodec_24khz
- DAC: https://github.com/descriptinc/descript-audio-codec (MIT); https://huggingface.co/descript/dac_44khz
- Mimi/Moshi: https://github.com/kyutai-labs/moshi (Apache-2.0); https://huggingface.co/kyutai/mimi (CC-BY-4.0)
- SNAC: https://github.com/hubertsiuzdak/snac (MIT); https://huggingface.co/hubertsiuzdak/snac_24khz
- Orpheus SNAC streaming decoder: https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/decoder.py
- Kokoro: https://github.com/hexgrad/kokoro (`kokoro/istftnet.py`, `kokoro/custom_stft.py`, `kokoro/pipeline.py`; Apache-2.0); https://huggingface.co/hexgrad/Kokoro-82M (`config.json`); https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX; https://github.com/thewh1teagle/kokoro-onnx (MIT)
- StyleTTS 2: https://github.com/yl4579/StyleTTS2 (MIT)
- Piper: https://github.com/OHF-Voice/piper1-gpl (GPL-3.0): `src/piper/voice.py`, `src/piper/split_voice_for_streaming.py`, `libpiper/src/piper.cpp`
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx (Apache-2.0): `sherpa-onnx/csrc/offline-tts.h`, `offline-tts-vits-impl.h`, `vocos-vocoder.cc`, `scripts/vocos/`; Matcha/vocoder docs https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html
- Rust crates: https://crates.io/crates/rustfft (6.4.1; README on NEON/AVX/SSE https://github.com/ejmahler/RustFFT), https://crates.io/crates/realfft (3.5.1; https://github.com/HEnquist/realfft), https://crates.io/crates/rubato
- librosa mel inversion: https://librosa.org/doc/main/generated/librosa.feature.inverse.mel_to_stft.html

### Secondary sources (🟡)

- Kokoro CPU speed on M3 Pro: https://www.visionstory.ai/open-source/kokoro-tts
- Kokoro in the browser on an M1 Max: https://briantung.me/blog/tts-engines-in-the-browser/
- Kokoro Xeon CPU benchmark: https://heyneo.com/blog/kokoro-supertonic-inflect-nano-cpu-tts-benchmark
- Piper speed claims: https://aivideosensei.com/guides/piper-tts-offline-voice-guide, https://www.promptquorum.com/power-local-llm/best-local-tts-raspberry-pi
- Commercial TTFA: https://www.coval.ai/blog/time-to-first-audio-ttfa-benchmark/, https://futureagi.com/blog/elevenlabs-vs-cartesia-tts-2026/, https://gradium.ai/content/best-low-latency-tts-apis-2026, https://onepin.ai/blog/streaming-tts-latency-ttfb-2026
- Griffin-Lim quality: https://arxiv.org/abs/2007.05764, https://arxiv.org/abs/1912.02461, https://www.researchgate.net/publication/346466743_Deep_Griffin-Lim_Iteration_Trainable_Iterative_Phase_Reconstruction_Using_Neural_Network

### Open unknowns 🔴

- Exact parameter counts for the EnCodec, DAC and Mimi decoders, and the Kokoro decoder's share of 82 M.
- Weight licences for EnCodec and DAC on Hugging Face (no licence tag; check the repo README/LICENSE before shipping).
- Native ONNX Runtime numbers for Vocos and HiFi-GAN on Apple Silicon (no published figures found; benchmark them in `speak`).
- macOS `say` / AVSpeechSynthesizer time to first audio.
