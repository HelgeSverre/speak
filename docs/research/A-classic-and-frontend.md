# Area A: Classic synthesis and the text frontend

Research for `speak`, a Rust CLI like macOS `say`. This area covers pre-neural synthesis (formant, LPC, diphone/PSOLA, unit selection, HMM/statistical parametric, WORLD) and the text frontend that every TTS system needs (normalization, G2P, prosody, SSML). It goes deepest on Klatt formant synthesis, a candidate first backend, and on how a frontend turns text into phonemes with durations and pitch.

Confidence markers: 🟢 primary source (paper, repo, docs, source code read directly), 🟡 secondary (blog, benchmark, search summary, or my own recollection of a primary source that I did not re-read this session), 🔴 inferred.

Research date: 2026-11-20.

---

## 1. Overview and timeline 🟢/🟡

The source-filter model underlies everything in this area: speech is a sound source (periodic glottal pulses for voiced sounds, turbulent noise for fricatives and aspiration) passed through a time-varying filter (the vocal tract, whose resonances are the formants), then the radiation at the lips. Classic systems differ in where the filter comes from:

| Era | Technique | Where the "voice" comes from | Representative systems |
|---|---|---|---|
| 1950s–80s | Formant synthesis by rule | Hand-written rules set formant targets and transitions | Holmes-Mattingly-Shearme (1964), MITalk / Klattalk (1979–87), DECtalk (1984), eSpeak |
| 1970s–90s | LPC synthesis | All-pole filter fitted to recorded speech | Speak & Spell (TI, 1978), LPC-10 |
| late 80s–90s | Diphone concatenation + PSOLA | One recorded instance of every phone-to-phone transition, re-pitched and re-timed | MBROLA (1996), Festival diphone voices, Flite `kal` |
| mid 90s–2010s | Unit selection | Large recorded corpus, search picks the best sequence of units | CHATR, Festival multisyn, AT&T Natural Voices, Nuance, early Siri |
| 2000s–2015 | Statistical parametric (HMM, then DNN) | Model predicts vocoder parameters frame by frame | HTS, Flite clustergen, Merlin |
| 2010s | High-quality analysis/synthesis vocoders | Signal processing (F0, spectral envelope, aperiodicity) | STRAIGHT, WORLD (2016) |
| 2016+ | Neural | See areas B and C | WaveNet, Tacotron, VITS, Kokoro |

What every system shares, then and now, is the **frontend**: text normalization, then grapheme-to-phoneme (G2P), then prosody (phrasing, duration, F0). End-to-end neural models such as VITS and Kokoro still need G2P (Piper uses espeak-ng; Kokoro uses misaki). They only absorb prosody and acoustics into the network.

---

## 2. Techniques

### 2.1 Klatt cascade/parallel formant synthesis (deep dive) 🟢

Key papers: Klatt 1980 (JASA 67:971, Fortran listing in the appendix), Klatt & Klatt 1990 (JASA 87:820, KLSYN88 and the KLGLOTT88 source), Klatt 1987 (JASA 82:737, a review of text-to-speech with the rule systems). All three are in `papers/`. The facts below were checked against those PDFs and the KLSYN manual unless marked otherwise.

#### 2.1.1 Building block: the digital resonator

Every formant is a second-order IIR filter (Klatt 1980, eq. 1–2, impulse-invariant transform):

```
y[n] = A·x[n] + B·y[n-1] + C·y[n-2]
C = -exp(-2π·BW·T)
B =  2·exp(-π·BW·T)·cos(2π·F·T)
A =  1 - B - C            (unity gain at DC)
```

`T` is the sample period, `F` the formant frequency and `BW` the bandwidth. An **antiresonator** (zero, used for the nasal zero FNZ and the glottal zero RGZ) is the inverse: `A' = 1/A, B' = -B/A, C' = -C/A`, and `y[n] = A'·x[n] + B'·x[n-1] + C'·x[n-2]` (FIR). That is the whole DSP vocabulary of a Klatt synth: resonators, antiresonators, a first-difference filter for lip radiation, a one-pole low-pass, a noise generator and gains. In Rust it comes to roughly 30 lines per filter type and needs no dependencies. 🟢

Coefficients are recomputed whenever parameters change. Klatt updated parameters every **5 ms** (each parameter frame covers NWS samples, default 50 at 10 kHz). He notes that formants must change "slowly and continuously relative to the 5-ms update interval". 🟢

#### 2.1.2 Topology

From Klatt 1980 Fig. 6 and the text 🟢 (component order also confirmed against klatt-syn and espeak-ng `klatt.c`):

```
                      ┌──────────── cascade branch (voicing + aspiration) ─────────────┐
 impulse/KLGLOTT88 ──►│ RNP(nasal pole) → RNZ(nasal zero) → R1 → R2 → R3 → R4 → R5 (→R6) │──┐
   + AH·noise         └──────────────────────────────────────────────────────────────────┘  │
                                                                                              ├─► first difference (radiation) ─► out
 frication noise ──►┌──── parallel branch: A1·R1, A2·R2, … A6·R6, AN·RNP, AB·bypass ────┐──┘
   (AF)             └─────────────────────────────────────────────────────────────────────┘
```

- **Cascade** (formant filters in series) is used for voiced sonorants and aspiration. Relative formant amplitudes then come out right automatically, as acoustic theory predicts for an all-pole tract, so vowels need only F1–F5 and B1–B5. The number of cascade formants (NFC) is 4–6; 5 at 10 kHz.
- **Parallel** (each formant has its own gain A1–A6, outputs summed) is used for frication and bursts, where the source sits in front of the constriction and the spectrum has zeros. Changing A-values shapes the fricative spectrum directly: /s/ is mostly A5/A6, /ʃ/ A3/A4, /f/ and /θ/ use the bypass AB. Klatt calibrated the parallel gains so that A1–A5 = 60 dB approximates the cascade output.
- An **all-parallel** mode exists (switch SW). Holmes preferred all-parallel for imitating natural speech (Klatt 1987). For rule-driven synthesis, cascade/parallel is the standard. 🟢
- **Radiation**: a first difference `y[n] = x[n] - x[n-1]`. Klatt also low-pass filters the noise (`y[n] = x[n] + y[n-1]`, which approximates integration) so the radiation differentiator cancels out for noise. 🟢

#### 2.1.3 Control parameters (Klatt 1980 Table I) 🟢

39 parameters. The variable ones (V) change every frame:

| Group | Parameters | Range / typical |
|---|---|---|
| Source amplitudes (dB) | AV (voicing), AVS (quasi-sinusoidal voicing), AH (aspiration), AF (frication) | 0–80; AV ≈ 60 in a strong vowel |
| Pitch | F0 | 0–500 Hz (KLSYN88: tenths of a Hz) |
| Formant frequencies | F1 150–900 (450), F2 500–2500 (1450), F3 1300–3500 (2450), F4 2500–4500 (3300), F5 3500–4900 (3750) | Hz |
| Bandwidths | B1 40–500 (50), B2 (70), B3 (110) variable; B4 250, B5 200, B6 1000 constant | Hz |
| Nasal | FNZ (zero, variable, 200–700), FNP 250, BNP/BNZ 100 | Hz |
| Parallel gains | A1–A6, AN, AB | dB |
| Constants | SR 10000, NWS 50, G0 47, NFC 5, glottal resonators FGP 0/BGP 100, FGZ 1500/BGZ 6000, BGS 200 | |

KLSYN88 (Klatt & Klatt 1990; KLSYN manual 🟢) adds voice-quality controls: **OQ** open quotient (10–80 %, default 50), **TL** spectral tilt (dB down at 3 kHz, 0–34), **SK** skew, **FL** flutter, **DI** diplophonia, **dF1/dB1** (F1 and B1 rising during the open glottis phase, from source-tract interaction), and a tracheal pole-zero pair. Defaults widen: F1 180–1300, F6 up to 4990, B1 30–1000.

#### 2.1.4 Glottal source models

1. **Impulsive (KLSYN 1980)** 🟢: an impulse train every SR/F0 samples, low-pass filtered by resonator RGP (F = 0, BW = 100 Hz, giving about −12 dB/octave above 50 Hz) and shaped by the antiresonator RGZ. Quasi-sinusoidal voicing (AVS, for voice bars and voiced fricatives) passes the impulse through RGP and also RGS (BW 200). Klatt himself warns that it "does not have the same phase spectrum as a typical glottal pulse". This is the buzzy, robotic-sounding source.
2. **KLGLOTT88** 🟢 (default in KLSYN88): during the open phase the glottal flow follows Rosenberg's polynomial **Ug(t) = a·t² − b·t³**, with *a* and *b* set by AV and the open-phase duration (OQ × T0), and zero during the closed phase. TL applies an extra soft low-pass (one-pole) tilt. AH adds aspiration noise that is amplitude-modulated in the open phase. Flutter is deterministic, not random:
   `ΔF0 = (FL/50)·(F0/100)·[sin(2π·12.7t) + sin(2π·7.1t) + sin(2π·4.7t)] Hz`. FL = 25 % sounds realistic. Klatt notes that random jitter sounds harsh.
3. **Modified LF (Liljencrants-Fant 1985)** 🟢: offered as a third option in KLSYN88. It is parameterized by AV, F0, OQ, SQ (speed quotient) and TL. Klatt chose LF because Fant et al. and Fujisaki & Ljungqvist found it the most accurate model of its complexity. It costs more to compute: you solve for the exponential growth constant per period (usually a few Newton iterations, 🟡).
4. **Implementation details from real code** 🟢: espeak-ng `klatt.c` runs the glottal source at **4× the output sample rate** and low-passes before decimation, to avoid the staircase effect of quantizing T0 to whole samples (a problem Klatt 1980 mentions). It offers `IMPULSIVE`, `NATURAL` (KLGLOTT88-like) and two `SAMPLED` sources (a stored natural glottal waveform). chdh/klatt-syn (MIT, TypeScript) implements `impulsive`, `natural` and `noise` sources. It notes that the abrupt return to zero at the end of the open phase is not smoothed in classic Klatt, whereas Praat's KlattGrid smooths it.

Practical point 🔴: for `speak`, KLGLOTT88 with a 2–4× oversampled source, plus flutter and some aspiration, is the cheapest big step up from the impulse-train buzz. LF can come later as a swappable `GlottalSource` trait implementation.

#### 2.1.5 Sources of noise 🟢
Frication and aspiration use a uniform random generator (classic code sums several draws to approximate a Gaussian 🟡), passed through the LPF integrator. When voicing is also present (as in /z/ or /v/), the noise is halved during the second half of each glottal period. espeak-ng: `if (nper > nmod) noise *= 0.5`.

#### 2.1.6 From phonemes to parameter tracks: synthesis by rule

The synthesizer only renders frames. The intelligence lives in the rule program that turns a phoneme string with durations and F0 into 5–10 ms parameter frames. These are the published approaches:

- **Target + transition tables (Holmes, Mattingly & Shearme 1964)** 🟢 (via Klatt 1987): each phoneme has target values per parameter plus a rank (dominance). At each boundary the higher-ranked phoneme dictates a boundary value and transition duration. Formants move in straight lines from consonant loci toward vowel targets, and short vowels **undershoot** their targets (Klatt 1987 Fig. 19). The whole system was a listing of tables, so it is entirely data-driven. Mattingly (1966/68) adapted it to American English with S-shaped transitions and found "little if any perceptual difference" from linear ones. **rsynth** (Nick Ing-Simmons, 1994, `holmes.c`) is an open implementation of this rule scheme driving a Klatt synth. Its successor, the SoLoud speech module, describes itself as rsynth-derived and public domain/CC0 🟡.
- **Locus equations (Klatt 1987, eq. 1)** 🟢: `F2onset = F2locus + k·(F2vowel − F2locus)`, with locus and coarticulation coefficient k fitted per consonant (for example, /d/ has a locus of about 1800 Hz). Klatt found pure locus theory oversimplified for stops (velars need different loci before front and back vowels), so Klattalk used tables of onset values by vowel context.
- **Smoothing filters (Rabiner 1968, Liljencrants 1969)** 🟢 (via Klatt 1987): instead of interpolating, run the step-function targets through a critically damped second-order filter whose time constant depends on the segment type.
- **Klatt 1980 Tables II and III** 🟢: target F1–F3/B1–B3 for American vowels, with diphthongized start and end values (for example [iy] 310/2020/2960 → 290/2070/2960; [a] 700/1220/2600, B1 130). Table III covers consonants (frication A-values, nasals and stops) and makes a good starting phoneme table.
- **eSpeak / eSpeak NG** 🟢: phonemes are defined in `phsource` with spectral "frame sequences" (formant peak frequency, height and width per frame) that were edited by hand, plus rules for duration and context-dependent variants. The *default* eSpeak voice does **not** use Klatt: `wavegen.c` is an additive harmonic synth that sets the amplitude of each harmonic of F0 from the formant peaks. The `+klatt`, `+klatt2` … `+klatt6` voice variants (`klatt N` selects the source type) route the same frames through `klatt.c`, which linearly interpolates peak frequency, bandwidth and amplitude across each frame. eSpeak is GPL-3.0-or-later, and so is its Klatt code (based on Iles & Ing-Simmons klatt 3.04).
- **MITalk / Klattalk / DECtalk** 🟢/🟡: the rules are documented in Allen, Hunnicutt & Klatt, *From Text to Speech: The MITalk System* (Cambridge UP, 1987; not freely available), with summaries in Klatt 1987. DECtalk source code is on GitHub (`dectalk/dectalk`), but its provenance and licence are unclear ("NOASSERTION", reportedly from a leak 🟡). **Do not copy from it.**

A concrete rule design for `speak` 🔴, built from the sources above:
1. A phoneme table (about 45 English phonemes, ARPAbet or misaki symbols). For each: class flags (vowel, nasal, stop, fricative, voiced, …), targets for F1–F3/B1–B3 (onset and offset for diphthongs), AV/AF/AH/AN, parallel A2–A6/AB for obstruents, inherent and minimum duration, and a Holmes-style rank.
2. Stops as sub-segments: closure (silence, or a voice bar via AVS for voiced stops), burst (5–10 ms frication spike shaped by place), aspiration for voiceless stops before stressed vowels (AH through the cascade, with B1 widened and F1 raised), then formant transitions from a locus-equation table.
3. Interpolation: linear or cosine transitions over a duration set by the dominant phoneme, with undershoot when the vowel is shorter than the transitions.
4. Output: 5 ms frames → Klatt renderer → `f32` samples at 16 or 22.05 kHz. Raising SR above 10 kHz needs F6 and B4–B6 to be set sensibly. KLSYN supports SR up to 20 kHz.

#### 2.1.7 Duration and F0 rules
- **Klatt duration model** 🟢 (Klatt 1987 Table II; MITalk ch. 9; implemented in Festival as `duration_klatt_params`): `DUR = MINDUR + (INHDUR − MINDUR) · PRCNT/100`. PRCNT starts at 100 and is multiplied by rule factors: about 140 % in a pre-pausal syllable, 60 % for vowels not in a phrase-final syllable, 85 % for vowels not in a word-final syllable, 80 % in polysyllabic words, plus rules for unstressed syllables, consonant clusters and the voicing of a postvocalic consonant 🟡 (factors from search summaries of the 1987 table; check against `papers/1987-klatt-tts-review.pdf` when implementing). Worked example from the MITalk book: /a/ with INHDUR 160 ms and MINDUR 100 ms, factors 0.6·0.85·0.8 → 124 ms 🟡.
- **F0** 🟡: Klattalk and MITalk used a "hat pattern" with declination. Pitch rises on the first stressed syllable, stays high, falls on the last stressed syllable, and the baseline declines over the clause, with local accents on stressed content words. eSpeak's model 🟢 (`docs/intonation.md`) splits each clause into **pre-head, head, nucleus and tail**. A "tune" is chosen by the clause punctuation (`.` `,` `?` `!`) and gives envelopes for each part, with the nucleus on the last stressed syllable. That is the simplest workable model for `speak`. Alternatives are the Fujisaki command-response model (phrase and accent commands through critically damped filters) and ToBI-labelled targets (§2.9).

#### 2.1.8 Quality, size, speed and licence
- **Quality** 🟡: intelligible but robotic. Klatt 1987 reports DECtalk-class rule systems reaching high segmental intelligibility (Klatt's own CVC rules scored 95 % with trained listeners). Naturalness is far below concatenative and neural systems. Most of the robotic quality comes from the source model and the rule-generated prosody, not from the filters.
- **Size and speed** 🟢/🔴: the code is a few hundred lines, with no model weights. Per output sample it runs about 10 biquads and a source update. That is far beyond real time on any CPU: thousands of times real time on a laptop 🔴 (estimate, not benchmarked).
- **Licence**: the algorithms are published and free to reimplement. Existing code: espeak-ng `klatt.c`, rhdunn/klatt and jh4xsy/klatt80 are **GPL-3.0**. chdh/klatt-syn and tgies/klattsch-rs (a Rust Klatt-1980 parallel-formant engine on crates.io as `klattsch-core`, 14 stars, created 2026) are **MIT**. Praat's KlattGrid is GPL. KLSYN by rsprouse has no licence listed on GitHub 🟢.

### 2.2 LPC synthesis 🟡
Linear prediction fits an all-pole filter (order about 10–16 at 8–10 kHz) to each 10–25 ms analysis frame of real speech, by autocorrelation and Levinson-Durbin. Synthesis drives that filter with an impulse train or noise, switched by a voicing decision. It is a source-filter model whose filter is *measured* rather than ruled, so it carries the speaker's timbre but sounds buzzy because of the binary excitation. Variants: CELP/codebook excitation (speech codecs), and LPCNet (2018, area C), which keeps the LPC filter and makes the excitation neural. For `speak` it matters as a simple analysis tool (estimating formants from recordings to tune Klatt tables, as Klatt 1980 Fig. 16 does) and as conceptual groundwork for LPCNet. The key paper, Atal & Hanauer 1971 (JASA), is not freely available.

### 2.3 Diphone concatenation and PSOLA 🟢
- **Diphones** run from the middle of one phone to the middle of the next, so the hard-to-model transition is *recorded*. A language needs about 1,500–2,500 diphones, recorded once in monotone nonsense words 🟡.
- **TD-PSOLA** (Charpentier & Moulines, Eurospeech 1989; Moulines & Charpentier 1990): place pitch marks at glottal closure instants and cut two-period Hann-windowed grains centred on each mark. To change pitch, re-space the grains at the new period and overlap-add. To change duration, repeat or drop grains. It is cheap and high-quality for modifications within about ±50 % (🟡), with phase and spectral mismatches at joins. Variants include FD-PSOLA, LP-PSOLA and MBR-PSOLA.
- **MBROLA** (Dutoit et al., ICSLP 1996) 🟢: diphone databases are resynthesized offline with constant pitch and harmonic phases (MBE-style), so that joins are smooth and they can be interpolated in the time domain. The input is a `.pho` file with one phoneme per line, its duration in ms and (position %, F0 Hz) pairs, a clean interface definition worth copying for `speak`'s internal phoneme stream. The engine has been **AGPL-3.0** since 2018. The voice databases have a separate non-commercial licence 🟡, so shipping them in an MIT tool is not allowed.
- Festival and Flite include diphone voices (`cmu_us_kal`, `kal16`) using residual-excited LPC instead of PSOLA 🟡.

### 2.4 Unit selection 🟢
Hunt & Black (ICASSP 1996, used in CHATR, then Festival multisyn): the corpus holds hours of read speech, segmented into phone or half-phone units with features. Synthesis is a Viterbi search over candidate units that minimizes the sum of a **target cost** (how well the unit's context, prosody and position match the specification) and a **concatenation cost** (spectral, F0 and energy discontinuity at the join). When it works it sounds like the recorded speaker, because it *is* the recorded speaker. Joins can be audible, out-of-domain text degrades, and voices are large (hundreds of MB to GBs of audio). It was the commercial standard until about 2016–2018. Not viable for `speak`: it needs a large licensed corpus and gives little to experiment with per byte. Its cost-function and Viterbi ideas still reappear in neural systems.

### 2.5 HMM / statistical parametric synthesis (HTS) 🟢
Zen, Tokuda & Black 2009; Tokuda et al. 2013 (Proc. IEEE, in `papers/`); Zen 2015 (HMM→LSTM, in `papers/`):
- **Training**: extract vocoder features per frame (mel-cepstrum, log F0, band aperiodicity, each with Δ and ΔΔ) and train **context-dependent 5-state HSMMs**. The labels are "full-context" (quinphone plus stress, position in syllable, word and phrase, POS, ToBI-ish features). Decision-tree clustering shares parameters across contexts, with separate trees for spectrum, F0 (multi-space distributions, which handle unvoiced frames) and duration.
- **Synthesis**: text → full-context labels → tree lookup → state durations → **maximum-likelihood parameter generation** using the Δ constraints, which gives smooth trajectories → vocoder (MLSA filter, later STRAIGHT or WORLD).
- **Character**: small (a voice of about 1–5 MB 🟡), fast and controllable (speaker adaptation, interpolation), but "buzzy and muffled" from vocoding plus over-smoothing. DNN and LSTM acoustic models (Zen 2013/2015) replaced the trees and paved the way for Tacotron.
- **Software**: HTS (patch to HTK; HTK's own licence restricts redistribution 🟡), **hts_engine API** (runtime, modified BSD 🟡), Flite+hts_engine, Open JTalk (Japanese), and Flite **clustergen** (CMU's statistical parametric voices such as `cmu_us_slt`).
- **For `speak`** 🔴: an interesting "middle" backend because it trains on CPU from a small corpus such as CMU ARCTIC (about 1,100 utterances), but building a full-context labeller is a big project. Its main lessons for `speak` are the full-context feature set and the Klatt-style duration modelling.

### 2.6 WORLD vocoder 🟢
Morise, Yokomori & Ozawa 2016 (IEICE, open access, in `papers/`). WORLD analyses speech into three streams and resynthesizes it:
- **F0**: DIO in the paper (fast); **Harvest** in later versions (more robust) 🟡.
- **Spectral envelope**: **CheapTrick** (F0-adaptive windowing, smoothing, cepstral liftering).
- **Aperiodicity**: PLATINUM in the 2016 paper; **D4C** in current versions 🟡.
- **Synthesis**: pitch-synchronous minimum-phase responses excited by pulses plus noise, mixed according to aperiodicity.

It beat STRAIGHT and TANDEM-STRAIGHT on quality and was much faster, running real-time on a CPU. The licence is modified BSD 🟢 (`mmorise/World`). WORLD was the standard vocoder for SPSS and for singing synthesis (NEUTRINO, many UTAU tools). For `speak` it is a strong **resynthesis and manipulation tool**: analyse a recorded word, change F0 and duration, resynthesize. It could also feed a parametric backend. It is C++ and would need porting or FFI. Pure-Rust ports are not verified here (area D).

---

### 2.7 Text normalization 🟢/🟡
This step turns written text into speakable words: "Dr. Smith paid $3.50 on 12/03/2024 at 5pm" → "doctor smith paid three dollars fifty on december third twenty twenty four at five p m".
- **Taxonomy** (Sproat et al. 2001, "Normalization of non-standard words", 🟡 from recollection): numbers (cardinal, ordinal, year, phone, currency, measure, fraction), dates, times, abbreviations, acronyms read as letters or as words ("NASA" vs "FBI"), URLs and emails, and symbols. Detection of these is ambiguous: "St." can be saint or street, "1/2" a date or a fraction, "Dr." doctor or drive, "IV" four or intravenous.
- **Classic approach**: tokenize → classify each token (rules/regex, or a classifier) → verbalize with per-class grammars. Google's Kestrel and open-source **Sparrowhawk** (Apache-2.0, archived) and **NVIDIA NeMo text processing** use weighted finite-state transducer (WFST) grammars written in Pynini 🟡.
- **Neural**: Sproat & Jaitly 2016 (in `papers/`) released a large Google normalization dataset and showed that RNNs make occasional "silly errors", such as reading the wrong number. That is unacceptable in TTS, so production systems keep grammars, or use neural classification with grammar-based verbalization 🟢.
- **What the reference systems do** 🟢: misaki uses `num2words` plus hand-written logic for currency, decimals, ordinals and years (en.py). espeak-ng has its own per-language number rules (`docs/numbers.md`).
- **Rust options** 🟡 (crates.io search, not evaluated): `verbalize` (MIT/Apache, CLDR-driven numbers, dates and prices; v0.2, 2026), `zyphra-textnorm` (ports of NeMo grammars), `lling-llang` (a WFST framework). All are young. A hand-written English normalizer covering numbers, ordinals, years, currency, times, common abbreviations and letter-spelling of all-caps tokens is a few hundred lines and fully testable 🔴.

### 2.8 Grapheme-to-phoneme (G2P)
English spelling is irregular, so every serious system uses **lexicon first, model fallback**:

| Approach | Details | Accuracy (word-level) | Licence |
|---|---|---|---|
| **CMUdict** 🟢 | 135,166 entries, ARPAbet with stress digits 0/1/2, American English; variant pronunciations as `word(2)` | Exact for listed words; no homograph POS info | BSD-2-clause-style (CMU) 🟢 |
| **Letter-to-sound rules** (NRL rules, Elovitz et al. 1976 🟡) | About 400 context rules, used by SAM, rsynth and SoLoud | Low; misses many words 🟡 | Public domain 🟡 |
| **CART LTS** (Black, Lenzo & Pagel 1998) 🟢 | Automatic letter-phone alignment, then a decision tree per letter. Used in Festival and Flite | 58 % words correct on CMUdict (held out), 75 % on OALD | Festival/Flite BSD-like 🟢 |
| **Joint-sequence / graphone n-grams** (Sequitur, Bisani & Ney 2008; Phonetisaurus WFST) | Long the state of the art | about 24.5 % WER on CMUdict 🟢 (as cited by Yao & Zweig) | Sequitur GPL-2.0, Phonetisaurus BSD-3 🟢 |
| **Bi-LSTM / seq2seq** (Yao & Zweig 2015) 🟢 | 3-layer bi-LSTM with alignment | 5.45 % PER, **23.55 % WER** on CMUdict | Paper only |
| **ByT5 / CharsiuG2P** (Zhu et al. 2022) 🟢 | Byte-level T5 for about 100 languages | ByT5-tiny-8L: **7.3 M params**, 9.8 % PER / 28.8 % WER (multilingual average), about 2,500 words/s | MIT 🟢 |
| **espeak-ng** 🟢 | Hand-written rules plus exception lists for 100+ languages, with stress and some POS-free homograph handling | Good for English and broad language coverage | **GPL-3.0** |
| **misaki** (Kokoro's G2P) 🟢 | See below | Highest for Kokoro, which was trained on its symbols | Apache-2.0 🟢 |

Note on the error rates: CMUdict "WER" here means *out-of-lexicon generalization*, so a 23 % error rate applies only to words not found in the dictionary 🔴. Real-text error is far lower, because most tokens hit the lexicon.

**misaki in detail** 🟢 (read from `misaki/en.py` and `EN_PHONES.md`):
- 49 phoneme symbols in total, 41 shared, plus 4 US-only and 4 GB-only. They are IPA-like with single-character diphthongs (`A`=eɪ, `I`=aɪ, `W`=aʊ, `Y`=ɔɪ, `O`=oʊ US, `Q`=əʊ GB), `ʤ`/`ʧ` as single tokens, `ᵊ` (small schwa) and `ɾ` (US flap), with stress marks `ˈ ˌ`. The symbols are designed as **network tokens, not linguistics**.
- Lexicons: `us_gold.json` (90,201 entries, 3.0 MB) and `us_silver.json` (3.1 MB), with GB equivalents. Gold entries can be **POS-keyed for homographs**: `read → {DEFAULT: ɹˈid, VBD: ɹˈɛd, …}`, `record → {DEFAULT: ɹˈɛkəɹd, VERB: ɹəkˈɔɹd}`. Where the lexicons come from is not documented ❓ (likely CMUdict-derived and corrected 🔴).
- The pipeline uses **spaCy** for tokenization and POS tagging (`en_core_web_sm`, or `trf` for the transformer model), then gold lookup (rating 4) → silver (rating 3) → morphological handling of suffixes such as -s, -ed and -ing, plus special cases ("the" before vowels, "to", "used") → **fallback**. The fallback is either a BART seq2seq model (`PeterReid/graphemes_to_phonemes_en_us`, run with torch) or `EspeakFallback` (requires espeak-ng, GPL). Numbers go through `num2words`.
- **Rust ports** 🟡 (crates.io, not evaluated): `misaki-rs` (MIT, "self-contained, POS-aware", 105 k downloads, updated 2026-11-19), `loqui-g2p` (MIT, "GPL-free … misaki lexicons plus an embedded neural fallback"), `sayd-misaki-en`. There is also `open-english-pronouncing-dictionary` (CC-BY-SA-4.0, about 280 k words fusing misaki, CMUdict and WikiPron). Note that CC-BY-SA data has share-alike obligations.

**How Piper does it** 🟢: Piper (now `OHF-Voice/piper1-gpl`, GPL-3.0) embeds **espeak-ng** for phonemization and maps the IPA output characters to integer IDs, configured per voice in the `.onnx.json`. That is why Piper became GPL. Area D covers the Rust bindings. `piper-plus-g2p` (MIT) claims an espeak-free G2P for Piper 🟡.

### 2.9 Prosody, ToBI and intonation models 🟡
- **What prosody is**: phrasing (where to pause), duration (§2.1.7), F0 contour, and stress and prominence. Phrase breaks are predicted from punctuation plus POS (Festival uses CART or n-gram models over POS; a rule version is: break at punctuation, and at long function-word boundaries in long clauses).
- **ToBI** (Tones and Break Indices; Silverman et al. 1992): a labelling standard. Pitch accents are H\*, L\*, L+H\*, L\*+H and H+!H\*; phrase accents L- and H-; boundary tones L% and H%; break indices run from 0 to 4. TTS systems (Festival, HTS full-context labels) predict ToBI accents and tones, then map them to F0 targets (Festival: `Int_Targets_LR` linear regression, or Tilt). In practice a common English simplification is H\* on content-word stressed syllables, L-L% for statements and H-H% for yes/no questions.
- **Fujisaki model**: log F0 = base + phrase commands (impulses through a critically damped filter) + accent commands (steps through a filter). It is compact and well suited to rule generation.
- **eSpeak tunes** 🟢 (§2.1.7): pre-head, head, nucleus and tail envelopes per punctuation type. This is the most practical starting point.
- **Neural systems** predict duration and F0 implicitly (VITS) or explicitly (FastSpeech 2 variance adaptor). Kokoro takes **no explicit prosody input** beyond the phonemes, punctuation and style vector (area B).

### 2.10 SSML 🟡
W3C SSML 1.1 (2010) elements: `<speak>`, `<p>`/`<s>`, `<break time="500ms" strength="…">`, `<prosody rate pitch volume contour range>`, `<emphasis>`, `<say-as interpret-as="date|cardinal|characters|telephone">`, `<phoneme alphabet="ipa" ph="…">`, `<sub alias="…">`, `<voice>`, `<audio>`, `<mark>`, `<lexicon>`. espeak-ng supports a subset (`docs/markup.md` 🟢 exists; coverage not checked). macOS `say` and NSSpeechSynthesizer use **embedded commands** instead, such as `[[rate 200]]`, `[[slnc 500]]`, `[[pbas …]]`, `[[volm …]]`, `[[emph +]]` and `[[inpt PHON]]` 🟡 (from recollection of Apple's Speech Synthesis Programming Guide; verify before cloning).

Note for `speak` 🔴: with a Klatt backend, rate, pitch, pitch range, volume, breaks and phoneme input are all trivial, because they are explicit parameters. With Kokoro or Piper, only rate (the length scale or speed), breaks (by splitting sentences) and phoneme input are easy. Pitch needs post-processing such as PSOLA or WORLD. A small SSML subset (`break`, `prosody rate/pitch/volume`, `say-as`, `phoneme`, `sub`) plus `say`-style `[[slnc]]`/`[[rate]]` covers almost all real use.

---

## 3. Comparison table

| Technique | Inputs → outputs | Voice data size | Speed (CPU) | Quality (rough) | Licence of best open implementation | Key paper |
|---|---|---|---|---|---|---|
| Klatt formant by rule | phonemes + durations + F0 → params → samples | ~0 (rule tables, KB) | ≫1000× RT 🔴 | Intelligible, robotic | Algorithm free; espeak/klatt C GPL; klatt-syn, klattsch MIT | Klatt 1980, 1990 |
| LPC | LPC frames + excitation → samples | per-frame coefficients of recordings | ≫RT | Buzzy, speaker-like | Many; trivial | Atal & Hanauer 1971 |
| Diphone + PSOLA (MBROLA) | phonemes + dur + F0 targets → concatenated grains | 1–10 MB per voice 🟡 | ≫RT | Natural timbre, glitchy joins, flat prosody | MBROLA AGPL; voices non-commercial | Moulines & Charpentier 1990; Dutoit 1996 |
| Unit selection | full-context spec → unit search → concatenation | 100 MB–GBs | RT, memory-heavy | Best pre-neural when in-domain | Festival multisyn (BSD-like) | Hunt & Black 1996 |
| HMM SPSS (HTS) | full-context labels → vocoder params → vocoder | 1–5 MB 🟡 | ≫RT | Smooth, muffled/buzzy | hts_engine (BSD) 🟡; HTS training needs HTK | Tokuda 2013, Zen 2009 |
| WORLD vocoder | F0 + envelope + aperiodicity ↔ samples | n/a (vocoder) | Real-time+ 🟢 | Near-transparent resynthesis | Modified BSD 🟢 | Morise 2016 |
| G2P: CMUdict + CART | word → ARPAbet | 3–4 MB raw 🟡 | µs/word | 58 % OOV words correct | BSD-like | Black et al. 1998 |
| G2P: espeak-ng | text → IPA (100+ langs) | ~MBs data | µs/word | Good, broad | GPL-3.0 | n/a |
| G2P: misaki | text + POS → misaki symbols | ~6 MB lexicons + spaCy + BART | ms/sentence (Python) | Best match for Kokoro | Apache-2.0 | n/a |
| G2P: ByT5-tiny | word → IPA | 7.3 M params (~30 MB fp32 🔴) | ~2,500 words/s (GPU? not stated 🟡) | 9.8 % PER multilingual | MIT | Zhu et al. 2022 |

---

## 4. What this means for `speak` 🔴 (recommendations, built on the sources above)

1. **Make the internal interface the MBROLA `.pho` idea**: a stream of `(phoneme, duration_ms, [(pos%, f0_hz)])` with phrase and break markers. Every classic backend (Klatt, diphone/PSOLA, HTS-like) consumes exactly this. Neural backends (Kokoro, Piper) consume only the phoneme string, and can ignore or approximate durations and F0. This one type is the seam that makes backends swappable, so put it in a crate of its own.
2. **Frontend: lexicon + fallback, no GPL in the default build.**
   - Normalizer: hand-written English rules in Rust (numbers, ordinals, years, currency, time, abbreviations, letter-spelled acronyms), unit-tested against Sproat-style categories. Evaluate the `verbalize` crate before writing your own.
   - Lexicon: use the misaki US gold+silver lexicon (Apache-2.0) as the primary dictionary. It already carries stress and POS-keyed homographs, and it is the symbol set Kokoro expects. CMUdict (BSD) can serve as an ARPAbet cross-check. A fixed misaki→Klatt phoneme mapping lets one lexicon feed both backends.
   - Homographs: start with a tiny rule POS tagger (or "DEFAULT" only). spaCy is not available in Rust, so it is the biggest gap against Python misaki. Evaluate `misaki-rs`'s POS handling before building one.
   - OOV fallback: first a CART or simple letter-to-sound model trained offline on the lexicon (Black et al. 1998: small and pure Rust at inference), later a small seq2seq or ByT5-tiny model through ONNX. Offer espeak-ng only as an opt-in, feature-gated GPL backend.
3. **First classic backend: a hand-written Klatt in pure Rust, designed as KLSYN88-lite.**
   - Cascade/parallel topology, 5 ms frames, at 16 kHz (keep F6) or 22.05 kHz.
   - Swappable `GlottalSource`: impulse (1980), KLGLOTT88 (default), LF (later). Oversample the source 2–4× like espeak-ng. Add flutter by the three-sine formula and TL tilt.
   - Port nothing from GPL code. Write it from the Klatt 1980 and 1990 papers and the KLSYN manual, and use MIT klatt-syn as a readable cross-check. Look at `klattsch-core` (MIT) before starting; it may be usable or at least a test oracle.
   - Rules: Holmes-style target and rank tables seeded from Klatt 1980 Tables II/III, locus equations for stops, the Klatt duration formula, and eSpeak-style head/nucleus/tail tunes for F0.
   - Test it the way the brief's other projects do: render offline to WAV, plot the spectrogram and the F1/F2 tracks, and compare against Table II targets with an LPC formant estimate. Listening is the final check.
4. **Expectations**: a rule-based Klatt will sound like DECtalk or eSpeak at best, after significant tuning. It is excellent for *learning* and for instant startup and zero model files, but not a replacement for Kokoro or Piper quality. Treat it as backend #1 for experiments, with a neural backend alongside it (areas B and D) once the phoneme interface exists.
5. **Worth reusing later**: PSOLA, roughly 150 lines, for pitch and rate control on neural output (making `say -r` and pitch options work on Kokoro). WORLD for analysis and resynthesis experiments. HTS-style full-context features if `speak` ever trains a small parametric model.
6. **SSML**: support a small subset plus `say`-compatible `[[rate]]`, `[[slnc]]` and `[[inpt PHON]]`. With the Klatt backend all prosody controls map directly onto parameters.
7. **Licence summary for an MIT `speak`**: OK to ship are CMUdict (BSD-style), the misaki lexicons (Apache-2.0, provenance undocumented ❓), WORLD (BSD), the Klatt algorithms, klatt-syn and klattsch (MIT), and CharsiuG2P (MIT). Avoid in the default build: espeak-ng, its Klatt code, Piper's phonemizer (all GPL-3.0), MBROLA (AGPL) and its voices (non-commercial), and DECtalk source (unclear provenance). Sequitur is GPL-2.0, and OpenEPD is CC-BY-SA.

### Open questions / unknowns
- ❓ Where misaki's gold and silver lexicons come from and what licence the data itself carries (the repo is Apache-2.0, but the data provenance is undocumented).
- ❓ Quality and maintenance of the very new Rust crates (`misaki-rs`, `loqui-g2p`, `klattsch-core`, `verbalize`, `espeak-ng` pure-Rust port, which is GPL). They were found by search and not tested.
- ❓ Exact factors in Klatt's duration rule table. Read Table II in `papers/1987-klatt-tts-review.pdf` before implementing; the factors above come from secondary summaries.
- ❓ The MITalk book's full phoneme parameter tables are not freely available. Klatt 1980 Tables II and III plus the Holmes et al. 1964 tables (listed in the original paper; not checked for an online copy) are the accessible substitutes.

---

## 5. Sources

### Papers downloaded to `papers/` (all verified to start with `%PDF`)
| File | Paper | Source URL |
|---|---|---|
| `1980-klatt-cascade-parallel.pdf` | Klatt, "Software for a cascade/parallel formant synthesizer", JASA 67(3), 1980 | https://www.fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf (University of Amsterdam course copy) |
| `1987-klatt-tts-review.pdf` | Klatt, "Review of text-to-speech conversion for English", JASA 82(3), 1987 | https://sail.usc.edu/~lgoldste/Ling582/Week%2013/Klatt1987.pdf (USC course copy) |
| `1989-psola-diphones.pdf` | Charpentier & Moulines, "Pitch-synchronous waveform processing techniques for TTS using diphones", Eurospeech 1989 | https://www.isca-archive.org/eurospeech_1989/charpentier89_eurospeech.pdf |
| `1990-klatt-klatt-voice-quality.pdf` | Klatt & Klatt, "Analysis, synthesis and perception of voice quality variations among female and male talkers", JASA 87, 1990 | https://www.source-code.biz/klattSyn/Klatt-1990.pdf |
| `1996-hunt-black-unit-selection.pdf` | Hunt & Black, "Unit selection in a concatenative speech synthesis system using a large speech database", ICASSP 1996 | http://www.cstr.inf.ed.ac.uk/downloads/publications/1996/Hunt_1996_a.pdf (author institution; PDF text layer is garbled, scan/fonts) |
| `1996-mbrola.pdf` | Dutoit et al., "The MBROLA project", ICSLP 1996 | https://www.isca-archive.org/icslp_1996/dutoit96_icslp.pdf |
| `1998-black-letter-to-sound.pdf` | Black, Lenzo & Pagel, "Issues in building general letter to sound rules", ESCA SSW3 1998 | https://www.isca-archive.org/ssw_1998/black98_ssw.pdf |
| `2013-hmm-speech-synthesis-overview.pdf` | Tokuda et al., "Speech synthesis based on hidden Markov models", Proc. IEEE 101(5), 2013 (author manuscript) | https://www.pure.ed.ac.uk/ws/files/15269212/Speech_Synthesis_Based_on_Hidden_Markov_Models.pdf |
| `2015-zen-spss-hmm-to-lstm.pdf` | Zen, "Acoustic modeling in statistical parametric speech synthesis – from HMM to LSTM-RNN", 2015 | https://research.google.com/pubs/archive/43893.pdf |
| `2015-seq2seq-g2p.pdf` | Yao & Zweig, "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion", Interspeech 2015 | https://arxiv.org/pdf/1506.00196 |
| `2016-neural-text-normalization.pdf` | Sproat & Jaitly, "RNN approaches to text normalization: a challenge", 2016 | https://arxiv.org/pdf/1611.00068 |
| `2016-world-vocoder.pdf` | Morise, Yokomori & Ozawa, "WORLD: a vocoder-based high-quality speech synthesis system for real-time applications", IEICE 2016 | https://www.jstage.jst.go.jp/article/transinf/E99.D/7/E99.D_2015EDP7457/_pdf |
| `2022-byt5-g2p.pdf` | Zhu, Zhang & Jurgens, "ByT5 model for massively multilingual grapheme-to-phoneme conversion", Interspeech 2022 | https://arxiv.org/pdf/2204.03067 |

Not downloaded (no free legitimate copy found or not searched): Moulines & Charpentier 1990 (Speech Communication; only the 1989 conference version is open), Zen, Tokuda & Black 2009 (Speech Communication), Atal & Hanauer 1971, Holmes et al. 1964, Sproat et al. 2001, Silverman et al. 1992 (ToBI), the MITalk book (1987), and Bisani & Ney 2008.

KLSYN manual (Klatt's C version) was read but not saved to `papers/`, since it is a manual, not a paper: https://www.source-code.biz/klattSyn/klsynman.pdf

### Code and docs read
- espeak-ng Klatt module: https://github.com/espeak-ng/espeak-ng/blob/master/src/libespeak-ng/klatt.c
- espeak-ng harmonic synth: https://github.com/espeak-ng/espeak-ng/blob/master/src/libespeak-ng/wavegen.c
- espeak-ng Klatt voice variants: https://github.com/espeak-ng/espeak-ng/tree/master/espeak-ng-data/voices/!v
- espeak-ng intonation docs: https://github.com/espeak-ng/espeak-ng/blob/master/docs/intonation.md
- chdh/klatt-syn (MIT, TypeScript): https://github.com/chdh/klatt-syn (and `src/Klatt.ts`)
- Klatt 1980 Fortran: https://github.com/jh4xsy/klatt80 · KLSYN C: https://github.com/rsprouse/klsyn · klatt 3.04: https://github.com/rhdunn/klatt · Praat KlattGrid: https://github.com/praat/praat/blob/master/dwtools/KlattGrid.cpp
- klattsch-rs (MIT, Rust): https://github.com/tgies/klattsch-rs
- DECtalk (unclear licence, do not copy): https://github.com/dectalk/dectalk
- misaki: https://github.com/hexgrad/misaki · `EN_PHONES.md` · `misaki/en.py` · `misaki/data/us_gold.json`
- Piper (GPL): https://github.com/OHF-Voice/piper1-gpl
- CMUdict (licence + 135,166 lines in `cmudict.dict`): https://github.com/cmusphinx/cmudict
- Flite licence: https://github.com/festvox/flite/blob/master/COPYING · Festival: https://github.com/festvox/festival · Festival Klatt duration: https://www.cstr.ed.ac.uk/projects/festival/manual/festival_19.html
- MBROLA (AGPL-3.0): https://github.com/numediart/MBROLA · voices: https://github.com/numediart/MBROLA-voices
- WORLD (modified BSD): https://github.com/mmorise/World
- CharsiuG2P (MIT): https://github.com/lingjzhu/CharsiuG2P · Phonetisaurus (BSD-3): https://github.com/AdolfVonKleist/Phonetisaurus · Sequitur (GPL-2.0): https://github.com/sequitur-g2p/sequitur-g2p · phonemizer (GPL-3.0): https://github.com/bootphon/phonemizer · gruut (MIT, archived): https://github.com/rhasspy/gruut · Sparrowhawk (Apache-2.0, archived): https://github.com/google/sparrowhawk
- crates.io API searches (g2p, phonemizer, klatt, formant, cmudict, misaki, text-normalization): https://crates.io/api/v1/crates?q=… ; crates: `misaki-rs`, `loqui-g2p`, `piper-plus-g2p`, `klattsch-core`, `verbalize`, `zyphra-textnorm`, `espeak-ng` (GPL pure-Rust port, https://github.com/eugenehp/espeak-ng-rs), `open-english-pronouncing-dictionary`

### Secondary sources
- rsynth licence history: https://mi.eng.cam.ac.uk/comp.speech/Section5/Synth/rsynth.html, https://cdn.netbsd.org/pub/pkgsrc/current/pkgsrc/audio/rsynth/index.html, SoLoud legal notes (rsynth-derived speech): https://git.allpurposem.at/mat/SoLoud-CMake/src/commit/ea92fccf9da99d2dfb77c5bb7e2365b6cc2437ba/doc/legal.tex
- Klatt duration formula summaries: https://www.cle.org.pk/Publication/theses/2005/duration_model_for_urdu.pdf, https://github.com/StrangeQuery/cardiac-speech, US patent 5592585
- ISCA landing pages: https://www.isca-archive.org/icslp_1996/dutoit96_icslp.html, https://www.isca-archive.org/eurospeech_1989/charpentier89_eurospeech.html
- J-STAGE WORLD page: https://www.jstage.jst.go.jp/article/transinf/E99.D/7/E99.D_2015EDP7457/_article
- Edinburgh research record for Hunt & Black: https://era.ed.ac.uk/items/07a1382d-ac96-4894-8532-b6c8eac44aad
