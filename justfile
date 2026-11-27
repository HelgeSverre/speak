# speak: offline text-to-speech with no model inside.
set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

install_dir := env_var('HOME') / ".local/bin"
cmudict := env_var('HOME') / "Library/Application Support/speak/lexicon/cmudict.dict"

[private]
default:
    @just --list --unsorted

# ── build ─────────────────────────────────────────────────────────────

# Build both binaries, optimized.
[group('build')]
build:
    cargo build --release

# Remove build outputs.
[group('build')]
clean:
    cargo clean

# ── run ───────────────────────────────────────────────────────────────

# Speak with the current source; arguments go to speak: `just run -v klatt-drama "Hello."`
[group('run')]
@run *args:
    cargo run --quiet --release --bin speak -- "$@"

# Run the workshop tool; arguments go to speak-lab: `just lab diphones info boot`
[group('run')]
@lab *args:
    cargo run --quiet --release --bin speak-lab -- "$@"

# ── install ───────────────────────────────────────────────────────────

# Build and install speak and speak-lab to a directory on your PATH.
[group('install')]
install dest=install_dir: build
    mkdir -p {{ dest }}
    install -m 0755 target/release/speak {{ dest }}/speak
    install -m 0755 target/release/speak-lab {{ dest }}/speak-lab
    @echo "Installed speak and speak-lab to {{ dest }}"

# Remove the installed binaries; voices, lexicon and config are kept.
[group('install')]
uninstall dest=install_dir:
    rm -f {{ dest }}/speak {{ dest }}/speak-lab
    @echo "Removed {{ dest }}/speak and {{ dest }}/speak-lab"

# Rebuild and replace the installed binaries.
[group('install')]
reinstall dest=install_dir: (uninstall dest) (install dest)

# ── data ──────────────────────────────────────────────────────────────

# Download the pinned CMUdict and compile it into the lookup file speak reads.
[group('data')]
lexicon: build
    scripts/fetch-cmudict.sh
    target/release/speak-lab lexicon build "{{ cmudict }}"

# Build the bootstrap diphone voice (`speak -b diphone`) from the formant synthesizer.
[group('data')]
voice: build
    target/release/speak-lab diphones build --bootstrap

# Retrain the letter-to-sound tree in data/ from CMUdict; rebuild afterwards to embed it.
[group('data')]
tree: build
    target/release/speak-lab lts train "{{ cmudict }}" --out data/lts-en.cart

# ── check ─────────────────────────────────────────────────────────────

# Format the source.
[group('check')]
fmt:
    cargo fmt --all

# Check formatting and run clippy with warnings as errors.
[group('check')]
lint:
    cargo fmt --all --check
    cargo clippy --all-targets -- -D warnings

# No model runtimes or third-party frontends in the build, and permissive licences only.
[group('check')]
deps:
    scripts/check-deps.sh
    cargo deny check licenses

# Full local gate.
[group('check')]
check: lint test deps

# ── test ──────────────────────────────────────────────────────────────

# Run the test suite; arguments go to `cargo test`: `just test --test klatt`
[group('test')]
@test *args:
    cargo test "$@"

# Word error rate on the 20 test sentences via whisper; arguments go to speak: `just wer -b diphone`
[group('test')]
@wer *args: build
    scripts/wer.sh "$@"

# Compare every system against macOS say with whisper; slow.
[group('test')]
intelligibility:
    cargo test --release --test intelligibility -- --ignored --nocapture

# ── bench ─────────────────────────────────────────────────────────────

# Wall time to a finished file against macOS say, plus in-process TTFA and RTF.
[group('bench')]
bench: build
    scripts/bench-vs-say.sh

# ── web ───────────────────────────────────────────────────────────

# Build the browser engine into website/pkg; needs wasm-pack.
[group('web')]
wasm:
    scripts/build-wasm.sh

# Build the browser engine and serve the website on http://localhost:8000.
[group('web')]
site: wasm
    python3 -m http.server 8000 --bind 127.0.0.1 --directory website

# Build the browser engine and deploy the website to Vercel (project speak).
[group('web')]
deploy: wasm
    cd website && vercel deploy --prod --yes --scope helgesverre

# ── demo ──────────────────────────────────────────────────────────────

# Regenerate the README clips in demo/ (MP3, spectrogram, videos); needs ffmpeg.
[group('demo')]
demo: build
    scripts/make-demo.sh
