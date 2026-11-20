# `speak` — implementation plan

Build `speak` from `spec.md`. Section references (`§8.1.3`) point into `spec.md`; research references (`A §2.1`) point into `docs/research/`.

## Ground rules

1. **We write it ourselves.** No pretrained models, no inference runtime, no third-party phonemizer, tagger or number library. Allowed crates are listed in spec §5; anything else needs a written reason in the milestone summary. CMUdict is allowed as an optional data file (spec §7.4).
2. Layout: one crate with modules, no workspace (spec §5). Paths below written as `crates/speak-x/src/y.rs` mean `src/x/y.rs`; `speak-core` files sit at `src/`; `cargo test --workspace` means `cargo test`.
3. `spec.md` is the contract. Where this plan and the spec disagree, the spec wins. Where the spec leaves something open, take the default in spec §18 and move on.
3. Work the milestones in order. A milestone is done when every command under **Acceptance** behaves as stated and `cargo test --workspace` passes.
4. Do not open, copy or port the sources listed in spec §14 item 5 (espeak-ng, GPL Klatt implementations, Praat, Festival/Flite/MBROLA, DECtalk). Work from the papers in `docs/research/papers/` and the spec.
5. `say` and `whisper` are test instruments (spec §14 item 8). They appear in tests and scripts only, never in product code, never as a source of voice or training data.
6. Every milestone: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
7. Nothing built locally (`*.sdv`, `en-us.lex`, recordings, CMUdict itself) goes into git. Committed data: what we wrote, test fixtures, and `data/lts-en.cart` (spec §12). Commit `Cargo.lock`.
8. You cannot listen. Verify audio with file properties, the formant/level/F0 tests and whisper transcripts. In each milestone summary, list what still needs a human ear.
9. If Helge has asked for commits: one commit per milestone, message `M<n>: <what works now>`.

## Environment (verified 2026-11-20)

| Item | State |
|---|---|
| Machine | Apple M2 Max, macOS, zsh |
| Rust | rustc / cargo 1.99.0 |
| `whisper` (Python CLI) | `/opt/homebrew/bin/whisper`, model `base.en` cached in `~/.cache/whisper/` |
| `say`, `afinfo`, `ffmpeg`, `curl` | available |
| Network | needed for crates, and once for CMUdict in M5 |

Shell conventions used below (zsh: quote `?`, `*`, `[`):

```sh
cargo build --release
S=./target/release/speak
L=./target/release/speak-lab
T=$(mktemp -d)
transcribe() { whisper "$1" --model base.en --language en --fp16 False --output_format txt --output_dir "$T/w" >/dev/null 2>&1 && cat "$T/w/$(basename "${1%.*}").txt"; }
```

## Milestones

| # | Result | Technique learned |
|---|---|---|
| M1 | Vowels come out of the speakers and into a WAV | resonators, glottal source, cascade/parallel graph |
| M2 | Klatt speaks any ARPAbet string intelligibly | synthesis by rule: targets, transitions, duration and F0 rules |
| M3 | `speak "text"` end to end with our normalizer and rule LTS | text normalization, NRL-style letter-to-sound |
| M4 | `say` flag parity, file formats, config, embedded commands | — |
| M5 | Lexicon lookup and a letter-to-sound tree we train | alignment, CART |
| M6 | Klatt in depth: LF source, context rules, voice quality, analysis tools | acoustic phonetics, LPC formant analysis |
| M7 | Second backend: diphone LPC with a bootstrap voice | linear prediction, lattice filters, unit concatenation |
| M8 | DSP lab: STFT, Griffin-Lim, PSOLA, `--resynth`, residual excitation | phase reconstruction, pitch-synchronous overlap-add |
| M9 | Record-your-own diphone voice with DTW autolabelling | MFCC, dynamic time warping |
| M10 | Measured against `say`; calibration; Linux; `-i` | — |
| M11 | STRETCH: neural acoustic model trained from scratch | only on request |

---

## M1 — First sound

Goal: formant renderer, audio sink, minimal CLI. Three steady vowels, played and written.

### Tasks

1. Workspace: root `Cargo.toml` (`[workspace]`, `resolver = "3"`, `[workspace.dependencies]`, `[workspace.lints]`), `rust-toolchain.toml` (1.99), `.gitignore` (`target/`).
2. `speak-core`: `Phone` (§6.1), `Stress`, `Segment`, `Unit`, `Chunk`, `Controls`, `Terminator` (§6.3); traits `Synthesizer`, `PcmWriter`, `AudioSink`, `Backend`, `GlottalSource` (§4.2); `Error` (§15); ARPAbet parser and printer (§6.2).
3. `speak-dsp`: `Resonator`, `AntiResonator` (§8.1.2); 4:1 decimating FIR; `rms_dbfs`, `peak`; `source.rs` with `Klglott88`, `Impulse`, flutter, tilt filter (§8.1.4); minimal LPC (autocorrelation + Levinson-Durbin, §8.2.2) with formant peak picking, needed by the tests.
4. `speak-klatt`: `Frame` (§8.1.5), `gains.rs`, `render.rs` (graph of §8.1.3, 1 ms coefficient updates, 20 ms output blocks), `tables.rs` with the monophthong rows of §8.1.9, voice `klatt-male`. Temporary compiler: each vowel 300 ms steady, AV 60, F0 falling 120 → 100 Hz, 30 ms linear crossfades. Replaced in M2.
5. `speak-audio`: `FileSink` for WAVE `LEI16` (`hound`); `Playback` (cpal + `rtrb`, §10.1) with `rubato` resampling when the device refuses 16 kHz; `NullSink`.
6. `speak-cli`, binary `speak`: `--phonemes`, `-o`, `--timing` (keys `sample_rate`, `ttfa_ms`, `synth_total_ms`, `audio_s`, `rtf`, `underruns`), `--version`, `--help`. Synthesis on its own thread (§4.3).

### Files

```
Cargo.toml  rust-toolchain.toml  .gitignore
crates/speak-core/src/{lib,phone,chunk,traits,error,timing}.rs
crates/speak-dsp/src/{lib,resonator,fir,level,source,lpc}.rs
crates/speak-klatt/src/{lib,frame,render,gains,tables,voice}.rs
crates/speak-audio/src/{lib,playback,file,null,resample}.rs
crates/speak-cli/src/bin/speak.rs  crates/speak-cli/src/{lib,args,run}.rs
```

### Pitfalls

- cpal callback: no allocation, locks or logging. Zeros on underrun, counted.
- Many macOS devices reject 16 kHz streams. Query supported configs; fall back to the device default rate and resample. `SPEAK_FORCE_RESAMPLE=1` forces the resampler so both paths are testable.
- `finish()` must wait for the device latency or the last 50–100 ms is cut off.
- Changing resonator coefficients on a live filter: keep the state, change only A/B/C.
- The parallel branch uses alternating signs (§8.1.3). Plain summation leaves spectral holes between formants.
- Bypass a resonator at or above 0.47 × SR, or it goes unstable when the formant scale is raised.
- The source runs at 4× the output rate; phase and period are fractional samples. Quantizing the period to whole samples makes F0 step audibly.

### Tests

- Resonator, antiresonator, source period and open-quotient tests (§16.2).
- Tilt filter: attenuation at 3 kHz equals TL within 0.5 dB for TL = 6, 12, 24.
- LPC recovers a known all-pole filter (§16.4, first bullet).
- Formant test for `AA`, `IY`, `UW` (§16.2).
- Level test for `AA` (−20 ± 1 dBFS); fit `K_AV`, `K_OUT`.
- ARPAbet round trip on `HH AX L OW1 / W ER1 L D .`

### Acceptance

```sh
cargo test --workspace
$S --phonemes 'AA / IY / UW' -o $T/v.wav && afinfo $T/v.wav
#   → "1 ch,  16000 Hz", Int16; estimated duration 0.90 s ± 0.05
$S --phonemes 'AA / IY / UW' --timing
#   → audible; exit 0; stderr has "underruns 0", ttfa_ms < 100, rtf < 0.02; wall time ≥ 0.9 s
SPEAK_FORCE_RESAMPLE=1 $S --phonemes 'AA / IY / UW' --timing     # same, through the resampler
cargo tree -p speak-cli --prefix none | grep -c -E '^(ort|ort-sys|tract|candle|burn|espeak|misaki|num2words)'       # 0
```

---

## M2 — Klatt speaks phonemes

Goal: full phone inventory, rule compiler, duration and F0 rules.

### Tasks

1. `tables.rs`: every row of §8.1.9 plus class, rank, `k`, transition times, burst lengths (§8.1.7) and INH/MIN durations (§7.7). One row per phone; no numbers elsewhere.
2. `compile.rs`: steps 1–4 of §8.1.7 (the context rules of §8.1.8 come in M6; structure the tables so they can be added as columns).
3. `RuleProsody` (durations and F0, §7.7) behind the `Prosody` trait. It lives in `speak-klatt/src/prosody.rs` for now and moves to `speak-text` in M3. Syllable = a vowel plus the consonants before it back to the previous vowel within the word.
4. Voices `klatt-male`, `klatt-female`, `klatt-robot`, `klatt-whisper` (§8.1.6); `-v`; `-P source= oq= tl= flutter= formant_scale= breath=`.
5. CLI: `--phonemes=arpabet|pho`, `--dump=phonemes|pho|frames`, `-r`, `--pitch`.
6. Fit the remaining gain constants to the level tests (§16.2).
7. Golden fixtures (§16.2): generate, check each by whisper, commit under `tests/fixtures/klatt/`.

### Files

```
crates/speak-klatt/src/{compile,prosody,pho}.rs
crates/speak-klatt/tests/{formants,levels,golden,compile}.rs
tests/fixtures/klatt/*.wav  tests/fixtures/klatt/*.pho
```

### Pitfalls

- Aspiration belongs to the start of the *following* sonorant (§8.1.7), not to the stop's own duration. Getting this wrong makes `P T K` sound like `B D G`.
- The final voiced/voiceless contrast is carried mostly by the preceding vowel's duration (rule 8). Test `B EH1 D` vs `B EH1 T`.
- Duration percentages are starting values (§7.7). If whisper mishears systematically (dropped final consonants, merged words), adjust durations before touching formant tables.
- F0 must be continuous across unvoiced segments (hold), or voiced onsets click.
- Golden WAVs are compared by SNR, not bytes: `libm` differs across platforms.
- Tuning changes table rows, never adds special cases to `compile.rs`.

### Tests

- All of §16.2 except the context-rule bullet.
- Duration model: `AA` with INH 240 / MIN 100 and rules 3, 4, 5 (0.6 × 0.85 × 0.8) → 157 ms. `.pho` round trip.
- F0: statement ends below base, yes/no question above, wh-question below.
- `klatt-robot` has a constant period (autocorrelation peak within 1 % across the utterance).

### Acceptance

```sh
$S --phonemes 'HH AX L OW1 / W ER1 L D .' -o $T/h.wav && transcribe $T/h.wav
#   → contains "hello world" (case and punctuation ignored)
$S --phonemes 'DH AX / K W IH1 K / B R AW1 N / F AA1 K S / JH AH1 M P S / OW1 V AXR / DH AX / L EY1 Z IY0 / D AO1 G .' -o $T/f.wav && transcribe $T/f.wav
#   → at least 7 of the 9 words of "the quick brown fox jumps over the lazy dog"
$S --phonemes 'HH AX L OW1 / W ER1 L D .' --dump pho         # one line per phone; durations sum = audio length ± 5 ms
$S --phonemes 'HH AX L OW1 / W ER1 L D .' --dump pho > $T/h.pho && $S --phonemes=pho -f $T/h.pho -o $T/h2.wav && afinfo $T/h2.wav
#   → duration within 10 ms of $T/h.wav; transcribe gives "hello world"
for v in klatt-male klatt-female klatt-robot klatt-whisper; do $S -v $v --phonemes 'HH AX L OW1 / W ER1 L D .' -o $T/$v.wav || echo FAIL; done
$S -r 300 --phonemes 'HH AX L OW1 / W ER1 L D .' -o $T/fast.wav      # duration ≤ 0.7 × that of $T/h.wav
$S --pitch 200 --phonemes 'AA1' --dump pho | grep -c ' 2[0-9][0-9]'   # ≥ 1 (F0 targets near 200 Hz)
cargo test --workspace
```

---

## M3 — Text in, streaming out

Goal: `speak "Hello world"` with our own frontend and the threaded pipeline of §4.3. Letter-to-sound is the hand-written rule set; no CMUdict yet.

### Tasks

1. `speak-text`:
   - `normalize.rs`, `numbers.rs` (our own number expansion), `abbrev.rs`, `units.rs`: §7.1, every class.
   - `chunk.rs`: §7.3.
   - `lexicon/builtin.rs` + `data/builtin-lexicon.txt`: ~500 entries we write (§7.4 row 2), including letter names for spelling.
   - `lts/rules.rs` + `data/lts-rules.txt`: `RuleLts` (§7.5): rule-file parser, matcher with the context symbols, stress assignment. Write the full rule set (~330 rules).
   - `g2p.rs`: `LexiconG2p` (user lexicon → built-in → morphology → `LetterToSound`), recording each word's source (§6.3).
   - `postlexical.rs`: §7.6. `funcwords.rs`.
   - `prosody.rs`: move `RuleProsody` here; add clause-internal pauses and tunes (§7.7).
2. `speak-klatt`: `Backend` + `Synthesizer` implementations.
3. `speak`: positional text, `-f`, stdin (non-TTY), `-b`, `-v '?'`, `--list-voices`, `--list-backends`, `--dump=text|chunks`, full `--timing` (§11.3) and `--timing=json`, `--no-stream`, `-V`, `-P lts=rules`. Threads and channels as §4.3.
4. `speak-lab` binary with `wer` (§3.6).
5. Intelligibility harness: `speak-cli/tests/intelligibility.rs` (`#[ignore]`), `tests/fixtures/sentences.txt`; also synthesizes the set with `say -o` and prints the baseline WER next to ours (§16.5). Skips with a message when `whisper` or `say` is missing.

### Files

```
crates/speak-text/src/{lib,normalize,numbers,abbrev,units,commands,chunk,g2p,postlexical,prosody,funcwords}.rs
crates/speak-text/src/lexicon/{mod,builtin,user}.rs
crates/speak-text/src/lts/{mod,rules}.rs
crates/speak-text/data/{builtin-lexicon.txt,lts-rules.txt}
crates/speak-text/tests/{normalize,numbers,chunk,lts_rules,postlexical}.rs
crates/speak-cli/src/bin/speak-lab.rs  crates/speak-cli/src/{pipeline,listing,dump,wer}.rs
crates/speak-cli/tests/{cli,intelligibility}.rs
tests/fixtures/{sentences.txt,bench.txt,normalize_cases.tsv,lts_words.txt}
```

### Pitfalls

- No word and no number may be dropped: unmatched digits are read digit by digit; unknown words always go through letter-to-sound; `--dump phonemes` never contains a digit or an unknown token. Assert it.
- Rule order is the algorithm: within a letter's list the first match wins, so specific rules go before general ones and every list ends with a context-free default.
- English function words are irregular and frequent (`of`, `the`, `was`, `one`, `to`, `do`, `are`, `have`). They decide most of the WER; they belong in the built-in lexicon, not in the rules.
- Sentence splitting: `Dr.`, `e.g.`, `3.14`, `U.S.`, initials, and `5 p.m.` at the end of a sentence.
- Words from the input keep their case (acronym detection needs it); expansions are lowercase (§7.1).
- `tests/fixtures/lts_words.txt` is a 500-word reference list we check by hand. Do not paste CMUdict into the repository.
- Parse rule and lexicon files on the frontend thread; report `frontend_init_ms`.

### Tests

- §16.3 except the CMUdict and CART bullets: normalizer table (≥ 120 rows), number expansion, chunker, built-in lexicon parses, `RuleLts` PER ≤ 35 % on `lts_words.txt`, one test per postlexical rule.
- CLI: `-f` + positional → exit 2; empty input → exit 0 and no audio; `--dump` stages stable for a fixture sentence.

### Acceptance

```sh
$S --dump text 'Dr. Smith paid $3.50 on 12/03/2024 at 5pm.'
#   → doctor Smith paid three dollars fifty on december third twenty twenty four at five p m.
$S --dump phonemes 'Hello world.'
#   → HH AX L OW1 / W ER1 L D .        (HH EH0 L OW1 is also acceptable)
$S --dump phonemes 'butter'                                  # B AH1 DX AXR  (flap applied)
echo 'qzxwv blorptastic 1969' | $S --dump phonemes           # phones for all tokens; no digits; no unknown tokens
$S --dump chunks 'First sentence. Second one, with a clause! Third?'       # 3 chunks with terminators . ! ?
$S -V --dump phonemes 'The cat sat.' 2>&1 | grep -c -E 'builtin|rules'      # ≥ 1 (word sources are logged)
$S 'Hello world. This is the formant synthesizer speaking.' --timing
#   → audible; exit 0; chunks 2; ttfa_ms < 100; underruns 0
$S -o $T/t.wav 'The quick brown fox jumps over the lazy dog.' && transcribe $T/t.wav      # ≥ 6 of 9 words
printf 'Line one.\nLine two.\n' | $S -o $T/s.wav && afinfo $T/s.wav                       # duration > 1 s
$S -v '?' ; $S --list-backends                               # formats of §3.4; klatt only so far
cargo test --workspace
cargo test -p speak-cli -- --ignored intelligibility --nocapture
#   → prints "say WER x %" and "klatt WER y %"; y ≤ 35
```

If klatt WER is above 35 %: read the per-word confusions the test prints, fix the lexicon and rules first, durations second, formant tables last.

---

## M4 — `say` parity

Goal: every flag in §3.1, file formats, config, embedded commands, signals.

### Tasks

1. Output: `--file-format`, `--data-format` (all combinations of §3.1, `@rate` resampling), `--channels`, extension inference, `-o -`, AIFF writer (§10.2), raw.
2. `-a` device selection and `-a '?'`; `--file-format='?'`, `--data-format='?'`.
3. `--volume`, `--progress`, unsupported `say` flags → exit 2.
4. TTY stdin line mode (§3.1), detected with `std::io::IsTerminal`.
5. Embedded commands (§7.2): parser, chunk splitting on control changes, inline phonemes.
6. Config file and environment (§13), precedence, `--config`, aliases `[voices.*]` with `params`, unknown-key error. User lexicon file (§7.4 row 1).
7. Exit codes (§3.5) for every error variant; SIGINT handling with a valid output file.

### Files

```
crates/speak-audio/src/{aiff,format}.rs
crates/speak-cli/src/{config,signals,progress}.rs
crates/speak-cli/tests/{formats,config,exit_codes}.rs
```

### Pitfalls

- AIFF stores the sample rate as an 80-bit IEEE 754 extended float, big-endian. Write the conversion by hand; test 16000, 22050, 24000, 44100, 48000 against `afinfo`.
- AIFF chunks are even-padded; patch `FORM` and `SSND` sizes and `numSampleFrames` at the end.
- 24-bit samples: 3 bytes, sign handled manually.
- WAVE to stdout cannot seek: sizes are `0xFFFFFFFF`; do not use `hound` for that path.
- SIGINT while writing a file: finalize the header before exiting. Install the handler explicitly: a process started in the background by a non-interactive shell begins with SIGINT ignored.
- Config path on macOS is `~/.config/speak/config.toml` (§12).

### Tests

- For every container/data-format pair: write 0.5 s, read back, compare within quantization error. Invalid pairs → exit 2.
- Embedded commands: `[[slnc 500]]` adds 500 ± 10 ms; `[[rate 300]]` shortens what follows; `[[inpt PHON]] HH AX L OW1 [[inpt TEXT]]` equals `--phonemes`.
- Config precedence: file < env < flag. Alias resolves. User lexicon entry overrides the built-in one.

### Acceptance

```sh
$S -o $T/a.aiff 'Hello world.' && afinfo $T/a.aiff      # AIFF, 1 ch, 16000 Hz, Int16 big-endian
$S -o $T/noext 'Hello world.' && afinfo $T/noext        # AIFF (say's default)
$S -o $T/b.wav --data-format=LEF32@48000 'Hello world.' && afinfo $T/b.wav    # WAVE, 48000 Hz, Float32
$S -o - 'Hello world.' | ffmpeg -v error -i - -f null - && echo ok             # ok
$S --file-format='?' ; $S --data-format='?' ; $S -a '?'                        # three listings, exit 0
$S 'One.[[slnc 1000]] Two.' -o $T/p.wav         # ≥ 1.0 s longer than without the command
$S -n host:1 'x'; echo $?                        # "speak: -n is not supported", 2
$S -v nosuchvoice 'x'; echo $?                   # 2
printf '[voices.bot]\nbackend = "klatt"\nvoice = "klatt-robot"\n' > $T/c.toml
$S --config $T/c.toml -v bot -o $T/bot.wav 'Hello.'                            # exit 0
printf 'nonsense = 1\n' > $T/bad.toml; $S --config $T/bad.toml 'x'; echo $?    # names the key, 2
LONG=$(printf 'This is sentence number one. %.0s' {1..5000})
( $S "$LONG" & sleep 1; kill -INT $! ; wait $! ; echo $? )                     # playback stops at once; 130
( $S -o $T/int.wav "$LONG" & sleep 0.3; kill -INT $! ; wait $! ; echo $? ) ; afinfo $T/int.wav
#   → 130 and a readable WAV (if the run finishes before the signal, raise the repeat count)
cargo test --workspace
```

---

## M5 — Lexicon and a letter-to-sound tree we train

Goal: CMUdict as optional lookup data; `CartLts` trained by our own code; homographs.

### Tasks

1. `scripts/fetch-cmudict.sh`: `curl` the pinned URL of §7.4 into `<data>/lexicon/cmudict.dict`, verify size and SHA-256 with `shasum -a 256`, fetch the `LICENSE` next to it. Fails loudly on mismatch.
2. `speak-lab lexicon build`: parse CMUdict (variants, comments), map phones (§6.1), write sorted `en-us.lex`. `lexicon/compiled.rs`: whole-file read + binary search on bytes.
3. Morphology (§7.4 row 4) and homograph table (§7.4).
4. `speak-lab lts train` (§7.5 `CartLts` steps 1–5): allowables table, iterative alignment, per-letter trees, binary writer; `lts/cart.rs`: loader and predictor. Deterministic output for the same input.
5. `speak-lab lts eval --method rules|cart`: held-out letters/words correct; also PER on `lts_words.txt`.
6. Commit `data/lts-en.cart` (≤ 2 MB), compile it in, make `lts=cart` the default, add the CMUdict notice to `NOTICE`.
7. `--timing` line `lexicon_hits` (§11.3).

### Files

```
scripts/fetch-cmudict.sh
crates/speak-text/src/lexicon/{compiled,morph,homograph}.rs
crates/speak-text/src/lts/{cart,align,train}.rs
crates/speak-text/data/lts-allowables.txt
data/lts-en.cart   NOTICE
crates/speak-text/tests/{lexicon,cart,homograph}.rs
```

### Pitfalls

- CMUdict is data. It is looked up; it is not compiled into the binary and `speak` must work when the file is absent (tests run both ways via `SPEAK_DATA_DIR`).
- Alignment: letters and phones differ in count. Without epsilon (letter → nothing) and double phones (`x → K-S`) a large share of words cannot be aligned; the allowables table decides the quality of everything after it (Black et al. 1998, `docs/research/papers/1998-black-letter-to-sound.pdf`).
- Train on the first pronunciation variant only; drop entries with digits, apostrophes-only tokens and abbreviations with periods.
- Hold out every 10th word *before* training; never evaluate on training words.
- Stress is part of the label for vowels; the tree therefore also learns stress. Unstressed mapping (`AH0 → AX` etc.) happens after prediction.
- Tree size: stop at 5 examples first; if the file exceeds 2 MB raise the stop size and re-check accuracy.
- Do not parse `en-us.lex` at startup. Budget 5 ms.

### Tests

- §16.3 complete.
- Lexicon builder: variants, comments, phone mapping; lookup hit and miss; absent file is not an error.
- Aligner: `box`, `though`, `cute`, `knight` align; alignment failure rate ≤ 3 % of CMUdict.
- `CartLts` beats `RuleLts` on `lts_words.txt`.

### Acceptance

```sh
scripts/fetch-cmudict.sh                               # "cmudict.dict verified 81917843…"
$L lexicon build "$HOME/Library/Application Support/speak/lexicon/cmudict.dict"       # writes en-us.lex; prints entry count ≥ 120000
$L lts train "$HOME/Library/Application Support/speak/lexicon/cmudict.dict" --out data/lts-en.cart
#   → prints alignment failures ≤ 3 %, node count, file size ≤ 2 MB; finishes in < 10 min
$L lts eval  "$HOME/Library/Application Support/speak/lexicon/cmudict.dict" --method cart     # words ≥ 50 %, letters ≥ 88 %
$L lts eval  "$HOME/Library/Application Support/speak/lexicon/cmudict.dict" --method rules    # reported, for comparison
cargo build --release
$S --dump phonemes 'I read it yesterday. I will read it tomorrow.'      # R EH1 D then R IY1 D
$S -V --dump phonemes 'The synthesizer mispronounced blorptastic.' 2>&1 | grep -E 'cmudict|cart'    # both sources appear
SPEAK_DATA_DIR=$T/empty $S --dump phonemes 'Hello world.'               # still works without CMUdict
$S 'Hello.' --timing 2>&1 | grep frontend_init_ms                       # ≤ 15
cargo test --workspace
cargo test -p speak-cli -- --ignored intelligibility --nocapture        # klatt WER ≤ 25 %
```

---

## M6 — Klatt in depth

Goal: the formant synthesizer as the main product: a second glottal model, context-dependent rules, voice quality, and tools to see what it does.

### Tasks

1. `Lf` glottal source (§8.1.4) with the per-period Newton solve; `-P source=lf`, `sq`.
2. Voice quality: `di` (diplophonia), breathy offset and hard onset rules (§8.1.6); voices `klatt-child`, `klatt-breathy`, `klatt-creaky`.
3. Context rules (§8.1.8) as table columns: loci by vowel class, velar bursts, fricative coarticulation, dark `L`, `R` colouring, duration-dependent reduction, sonorant devoicing.
4. `speak-lab analyze` (F0, formants, LPC as TSV per 5 ms) and `speak-lab spectrogram` (PGM writer, log-magnitude STFT; a simple DFT loop is fine here, the real STFT arrives in M8).
5. Tuning loop: for each systematic whisper confusion, compare `speak-lab analyze` tracks of our output with the table targets, adjust rows, re-run the gate. Record each change in the commit message.
6. Regenerate golden fixtures once, at the end.

### Files

```
crates/speak-dsp/src/source_lf.rs
crates/speak-klatt/src/{context,quality}.rs
crates/speak-cli/src/{analyze,spectrogram}.rs
crates/speak-klatt/tests/{context,sources}.rs
```

### Pitfalls

- LF: the Newton iterations can diverge for extreme OQ/SQ. Clamp parameters (OQ 0.3–0.8, SQ 1.2–4), cap iterations, fall back to the previous period's solution.
- Compare sources at equal loudness: normalize each to the same negative peak (−1 at closure) before judging spectra.
- Context rules interact with the undershoot formula. Change one rule at a time and keep the formant tests green.
- Regenerating goldens hides regressions. Do it once, after the WER gate passes, in its own commit.
- Whisper is a proxy. A lower WER with obviously implausible parameter values (formants outside Klatt's ranges) is not an improvement.

### Tests

- Sources bullet and context-rules bullet of §16.2.
- `klatt-creaky`: alternate periods differ in length by 10–15 %.
- `speak-lab analyze --formants` on a rendered `AA` reports F1/F2 within 8 % of the table.

### Acceptance

```sh
for s in klglott88 lf impulse; do $S -P source=$s -o $T/src-$s.wav 'Testing the glottal source.' && transcribe $T/src-$s.wav; done
#   → all three exit 0; klglott88 and lf transcripts ≤ 1 word wrong
for v in klatt-child klatt-breathy klatt-creaky; do $S -v $v -o $T/$v.wav 'A different voice.' || echo FAIL; done
$S -o $T/g1.wav --phonemes 'G IY1' ; $S -o $T/g2.wav --phonemes 'G UW1'
$L analyze $T/g1.wav --formants | head -40 ; $L analyze $T/g2.wav --formants | head -40
#   → F2 at voicing onset ≥ 600 Hz higher in g1 than in g2
$L analyze $T/src-lf.wav --f0 | awk 'NR>1 && $2>0 {s+=$2; n++} END {print s/n}'       # mean F0 80–140 Hz
$L spectrogram $T/src-lf.wav $T/s.pgm && head -c 2 $T/s.pgm                             # P5
cargo test --workspace
cargo test -p speak-cli -- --ignored intelligibility --nocapture                        # klatt WER ≤ 20 %
```

If 20 % is not reached after the tuning loop stops paying off, stop and report: the WER, the ten most frequent confusions, and the `analyze` output for two of them.

---

## M7 — Diphone LPC backend

Goal: the second technique, end to end, with the bootstrap voice built from our own Klatt output.

### Tasks

1. `speak-dsp`: complete `lpc.rs` (pre-emphasis, Hamming, reflection coefficients, gain), `pitch.rs` (F0 + voicing, §8.2.2), `lattice.rs` (all-pole lattice synthesis with per-millisecond interpolation).
2. `speak-diphone`: `inventory.rs` (pairs, impossible list), `format.rs` (`.sdv` reader/writer, §8.2.4), `build.rs` (bootstrap builder using `speak-klatt` as a library), `select.rs` (lookup, time mapping, joins, fallback, §8.2.5), `synth.rs` (§8.2.3, excitation `pulse|glottal`), `Backend` + `Synthesizer`.
3. `speak-lab diphones build --bootstrap`, `diphones info`.
4. `speak`: `-b diphone`, voice listing, `--dump=units|frames` for the diphone backend, `-P excitation=`, exit 3 with the build hint when the voice is missing.
5. Golden fixtures for the diphone backend (generated from the local `boot.sdv`; tests skip when the voice is absent).

### Files

```
crates/speak-dsp/src/{lpc,pitch,lattice}.rs
crates/speak-diphone/src/{lib,inventory,format,build,select,synth}.rs
crates/speak-diphone/tests/{lpc_roundtrip,bootstrap,ab_pho}.rs
tests/fixtures/diphone/*.pho
```

### Pitfalls

- Interpolate reflection coefficients, never direct-form predictor coefficients: interpolated direct-form filters can be unstable. Clamp |k| ≤ 0.999.
- Pre-emphasis at analysis must be undone by de-emphasis at synthesis, exactly once.
- Gain: the stored gain is the residual RMS. A pulse train must be scaled to unit RMS per period (amplitude ∝ `sqrt(T0)`), or loudness changes with pitch.
- Autocorrelation LPC on a strongly periodic low-F0 signal smears formants; the 25 ms window spans ≥ 2.5 periods at 110 Hz, which is the minimum.
- Voicing decision errors are the main source of buzz in fricatives and hoarseness in vowels. For the bootstrap voice the truth is known from the Klatt frames (AV > 0): test the detector against it.
- Stop bursts must not be time-stretched (§8.2.5 rule 2).
- The bootstrap voice sounds like coded Klatt. Do not "improve" it by tuning Klatt for LPC; its job is to validate the pipeline.
- `.sdv` files are little-endian with a version in the magic; reject anything else with `DataInvalid`.

### Tests

- LPC, F0 tracker, bootstrap-voice and same-`.pho` bullets of §16.4.
- Format: write → read round trip; truncated file and wrong magic rejected.
- Join smoothing: no sample-to-sample jump above 0.5 at any join in the golden utterances.

### Acceptance

```sh
$S -b diphone 'x'; echo $?                               # "diphone voice 'boot' is not built. Run: speak-lab diphones build --bootstrap", 3
$L diphones build --bootstrap                             # < 60 s; prints diphone count and file size
$L diphones info boot                                     # ≥ 1900 diphones; 16000 Hz; order 18
$S -b diphone -o $T/d.wav 'The quick brown fox jumps over the lazy dog.' --timing && transcribe $T/d.wav
#   → ≥ 6 of 9 words; ttfa_ms < 100; rtf < 0.02
$S --dump pho 'Hello world.' > $T/ab.pho
$S -b klatt   --phonemes=pho -f $T/ab.pho -o $T/ab-k.wav
$S -b diphone --phonemes=pho -f $T/ab.pho -o $T/ab-d.wav
afinfo $T/ab-k.wav ; afinfo $T/ab-d.wav                   # durations within 10 ms of each other
$S -b diphone --dump units 'Hello.'                        # _-HH HH-AX AX-L L-OW OW-_ with source/target durations
for e in pulse glottal; do $S -b diphone -P excitation=$e -o $T/e-$e.wav 'Excitation test.' || echo FAIL; done
$S -b diphone --pitch +5st -r 220 'Pitch and rate are free with our own excitation.' --timing     # audible; underruns 0
$S -v '?' | grep -c diphone                                # ≥ 1
cargo test --workspace
cargo test -p speak-cli -- --ignored intelligibility --nocapture        # diphone boot WER ≤ 35 %
```

---

## M8 — DSP lab

Goal: STFT, mel, Griffin-Lim and PSOLA in our own code, exercised by `--resynth`; residual-excited diphones.

### Tasks

1. `speak-dsp`: `window.rs`, `stft.rs` (forward, inverse, streaming overlap-add), `mel.rs` (filterbank, log-mel, MFCC), `griffinlim.rs`, `epochs.rs`, `psola.rs` (§9).
2. `Analyzer` / `Vocoder` implementations `LpcVocoder`, `Psola`, `GriffinLim` (§9.2); `speak --resynth`, `--vocoder`, with `-r` and `--pitch` as modifications.
3. Diphone `residual` excitation: store residuals and epochs in `.sdv` (bootstrap epochs from `GlottalSource::epoch`), PSOLA on the residual, then the lattice filter (§8.2.3, §9.3). `-P excitation=residual`.
4. Switch `speak-lab spectrogram` to the real STFT.

### Files

```
crates/speak-dsp/src/{window,stft,mel,griffinlim,epochs,psola}.rs
crates/speak-dsp/tests/{stft,mel,psola,griffinlim}.rs
crates/speak-cli/src/resynth.rs
```

### Pitfalls

- `realfft` inverse returns an error unless the imaginary parts of the DC and Nyquist bins are zero. Zero them before every inverse (C §2.14).
- `realfft` is unnormalized: scale the inverse by `1/N`.
- Periodic Hann (divide by N), not symmetric (N − 1), or overlap-add does not sum to a constant.
- Griffin-Lim needs a linear magnitude; the mel pseudo-inverse can go negative: clamp before iterating.
- PSOLA quality is decided by the pitch marks. Marks must sit at consistent points of each period (energy peaks); marks that drift by a fraction of a period produce roughness.
- PSOLA on unvoiced speech with repeated grains sounds tonal: when time-stretching unvoiced regions, reverse every other repeated grain.
- Residual excitation and `glottal` excitation have different levels; match RMS per frame to the stored gain.

### Tests

- STFT, mel/MFCC, DTW-free bullets of §16.4; PSOLA and Griffin-Lim bullets; epoch detector vs true marks.
- `--resynth --vocoder lpc` with factor 1 on a Klatt utterance: duration equal, F0 contour within 5 %.
- Residual-excited `AA-AA` diphone at its original pitch reconstructs the source rendering with SNR ≥ 20 dB.

### Acceptance

```sh
say -o $T/ref.aiff 'The vocoder turns analysis parameters back into sound.'        # test input only
ffmpeg -v error -y -i $T/ref.aiff $T/ref.wav
for v in lpc psola griffinlim; do $S --resynth $T/ref.wav --vocoder $v -o $T/rs-$v.wav && transcribe $T/rs-$v.wav; done
#   → lpc and psola: ≤ 2 words wrong; griffinlim: ≤ 3 words wrong
$S --resynth $T/ref.wav --vocoder psola --pitch +4st -o $T/up.wav
$L analyze $T/ref.wav --f0 > $T/f0a.tsv ; $L analyze $T/up.wav --f0 > $T/f0b.tsv
#   → median voiced F0 ratio b/a = 1.26 ± 0.05; durations within 2 %
$S --resynth $T/ref.wav --vocoder lpc -r 260 -o $T/fastr.wav            # duration = 175/260 of ref ± 3 %
$S --resynth $T/ref.wav --vocoder griffinlim --pitch +2st; echo $?       # "griffinlim cannot change pitch", 2
$L diphones build --bootstrap                                            # now with residuals
$S -b diphone -P excitation=residual -o $T/res.wav 'Residual excited linear prediction.' && transcribe $T/res.wav    # ≤ 2 words wrong
cargo test --workspace
```

---

## M9 — Record your own voice

Goal: the tooling to replace the bootstrap inventory with our own recordings, verified without a microphone by a synthetic speaker.

### Tasks

1. `speak-dsp/src/dtw.rs`: banded DTW returning the path.
2. `speak-lab diphones prompts` (§8.2.6 step 1): `prompts.tsv`, guide WAVs, a frequency-ordered list so the most useful diphones come first (spec §18 #4).
3. `speak-audio`: capture from the default input device. `speak-lab diphones record` (step 2): play guide, record, accept/redo/quit, resume with `--from`.
4. `speak-diphone/src/autolabel.rs` + `diphones build --dir` (step 3): MFCC, DTW against the guide, boundary mapping, analysis, epochs, rejection list.
5. Hidden test helper `speak-lab diphones simulate --dir DIR --voice klatt-female --rate 0.9 --jitter 0.1 --seed 1`: writes "recordings" rendered by another Klatt voice with perturbed timing, plus the true boundaries.

### Files

```
crates/speak-dsp/src/dtw.rs
crates/speak-audio/src/capture.rs
crates/speak-diphone/src/{prompts,autolabel,record}.rs
crates/speak-diphone/tests/autolabel.rs
```

### Pitfalls

- macOS asks for microphone permission on first capture; from a terminal it is the terminal app that needs it. Detect an all-zero recording and say so.
- DTW cost on raw MFCCs is dominated by overall level and channel: apply cepstral mean normalization per file and drop C0.
- Constrain the warp with a band (± 300 ms) or it finds degenerate paths through silences.
- The synthetic speaker shares our phone set and timing model, so it is an easy case. Real recordings will label worse; the rejection threshold and the re-record list exist for that.
- Monotone recording is the point (A §2.3): the guide is rendered at flat F0 so the speaker can match it.
- Recordings are personal data. Keep them where `--dir` points; never under the repository.

### Tests

- DTW bullet and autolabeller bullet of §16.4.
- Prompt generator covers every inventory diphone exactly once; guide length matches the prompt's durations.
- Capture: `record` with no input device exits 4.

### Acceptance

```sh
$L diphones prompts --out $T/rec && wc -l $T/rec/prompts.tsv         # ≥ 1900 lines; guide WAVs exist
$L diphones simulate --dir $T/rec --voice klatt-female --rate 0.9 --jitter 0.1 --seed 1
$L diphones build --dir $T/rec --name simtest
#   → prints boundary error vs truth: median ≤ 15 ms, p90 ≤ 40 ms; rejected prompts ≤ 5 %
$S -v simtest -o $T/sim.wav 'This voice was built from recordings.' && transcribe $T/sim.wav      # ≤ 3 words wrong
$S -v '?' | grep simtest
cargo test --workspace
cargo test -p speak-cli -- --ignored intelligibility --nocapture     # simtest WER ≤ 40 %
# Manual, for Helge (not part of the automated gate):
#   speak-lab diphones prompts --out ~/speak-voice && speak-lab diphones record --dir ~/speak-voice
#   speak-lab diphones build --dir ~/speak-voice --name helge && speak -v helge "Hello."
```

---

## M10 — Measure against `say`, calibrate, port

Goal: numbers; the dependency gate; Linux; `-i`.

### Tasks

1. `--bench[=N]` (§3.2, §11.3): warm-up, N runs to `NullSink`, median/min/max.
2. `scripts/bench-vs-say.sh` (§16.6): wall time to file for `say` and both backends on two texts, 5 runs each; binary size; prints a table. WER table from the intelligibility test alongside.
3. Rate calibration (§8.3): measure words per minute at `-r 175` for each backend; fix `native_wpm`; test 175 ± 15.
4. `deny.toml`, `scripts/check-deps.sh` (§14 items 2–3), final `NOTICE`.
5. `-i` / `--interactive`: print the chunk, highlight the current word from segment timing (reverse video), restore the terminal on exit and SIGINT.
6. Linux: build and test in a container (`rust:1.99` with `libasound2-dev pkg-config`); fix `XDG_*` paths and cpal defaults; no test may depend on `afinfo`, `say` or `whisper` being present.
7. Review `--help` of both binaries: every flag and subcommand of §3 present with a one-line description.

### Files

```
crates/speak-cli/src/{bench,interactive}.rs
scripts/{bench-vs-say.sh,check-deps.sh}
deny.toml
```

### Pitfalls

- Measure TTFA from the first statement of `main`; for cold numbers the script times whole processes.
- `say` has no TTFA hook: compare wall time to file only.
- cpal on Linux needs ALSA headers at build time; containers have no audio device: playback must exit 4 with a clear message, not hang, and playback tests must skip.

### Acceptance

```sh
$S -b klatt   --bench=5 -f tests/fixtures/bench.txt     # ttfa_ms median ≤ 100, rtf ≤ 0.02
$S -b diphone --bench=5 -f tests/fixtures/bench.txt     # ttfa_ms median ≤ 100, rtf ≤ 0.02
scripts/bench-vs-say.sh                                  # table: say, klatt, diphone × 2 texts; speak rows faster than say
scripts/check-deps.sh                                    # "ok: no model runtimes or third-party frontends"; exit 0
cargo deny check licenses                                # passes (install cargo-deny if missing)
$S -i 'Each word lights up as it is spoken.'             # words highlighted in sequence; terminal restored
docker run --rm -v "$PWD":/src -w /src rust:1.99 sh -c \
  'apt-get update -qq && apt-get install -y -qq libasound2-dev pkg-config >/dev/null && cargo test --workspace && cargo run -q --release -p speak-cli --bin speak -- -o /tmp/l.wav "Hello from Linux." && test -s /tmp/l.wav'
#   → tests pass, file written. If Docker is unavailable, say so and leave Linux unverified.
cargo test --workspace                                   # includes the 175 ± 15 wpm calibration tests
```

---

## M11 — STRETCH: a neural model we train from scratch

Do not start without an explicit request from Helge. Spec §17 is the whole design. No pretrained checkpoint of any kind.

### Tasks

1. `scripts/fetch-ljspeech.sh` (public-domain corpus; data only) into `<data>/neural/LJSpeech-1.1/`; resample to 16 kHz.
2. `speak-lab neural prepare`: run our frontend on every transcript; align each clip to the Klatt rendering of its phones with the M9 aligner; write phones, durations, F0 (our tracker) and log-mels (our `mel.rs`); drop high-cost clips; 95/5 train/validation split.
3. `speak-neural` crate (`burn`, excluded from default workspace members): model of §17, training loop with checkpointing, `speak-lab neural train`, loss curves as TSV.
4. Inference: `Synthesizer` implementation behind the `neural` cargo feature; waveform through our Griffin-Lim.

### Pitfalls

- `burn` and its backends must stay out of the default build: `scripts/check-deps.sh` keeps running on default features.
- Alignment quality bounds everything. Inspect the distribution of warp costs and listen-check by whisper on a few aligned segments before training.
- Griffin-Lim caps audio quality well below neural vocoders (C §2.1); judge the model by WER and mel loss, not by comparison with Kokoro-class systems.
- Training time on a laptop is days, not hours. Checkpoint every 1,000 steps; make the run resumable.

### Acceptance

```sh
cargo build --release --features neural
$S -b neural -o $T/n.wav 'This model was trained from scratch on public domain speech.' && transcribe $T/n.wav
cargo test -p speak-cli --features neural -- --ignored intelligibility --nocapture     # neural WER ≤ 30 %; RTF ≤ 0.5
cargo build --release && scripts/check-deps.sh                                          # default build still clean
```

Not converged within the budget of §17: stop, report loss curves and alignment statistics.

---

## Final summary to hand back

1. Milestones completed, and any acceptance command that does not pass, with its actual output.
2. Table: WER for `say`, klatt (rules / CART / after M6), diphone (`boot`, `simtest`); TTFA and RTF per backend; `say` wall time; binary size.
3. Letter-to-sound: held-out accuracy of the tree, PER of rules vs tree.
4. Spec §18 open questions: defaults kept or changed, and why.
5. What needs a human ear: Klatt voice quality and source choice, diphone joins, excitation modes, pause lengths; and what needs Helge's microphone: a recorded voice.
