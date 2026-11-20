#!/bin/sh
# Wall time to a finished audio file: macOS `say` against both `speak` backends (spec §16.6).
# `say` is a test instrument here; it has no time-to-first-audio hook, so whole processes are timed.
# Usage: scripts/bench-vs-say.sh        (needs a release build; the diphone rows need its voice)
set -eu
cd "$(dirname "$0")/.."
S=./target/release/speak
export SPEAK_LIGHTS=0
work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT
RUNS=5

# Median wall time in milliseconds of RUNS executions of a command.
median_ms() {
    i=0
    : > "$work/times"
    while [ "$i" -lt "$RUNS" ]; do
        start=$(perl -MTime::HiRes=time -e 'printf "%.6f", time')
        "$@" >/dev/null 2>&1
        end=$(perl -MTime::HiRes=time -e 'printf "%.6f", time')
        echo "$start $end" | awk '{printf "%.1f\n", ($2 - $1) * 1000}' >> "$work/times"
        i=$((i + 1))
    done
    sort -n "$work/times" | sed -n "$((RUNS / 2 + 1))p"
}

bench_text=$(cat tests/fixtures/bench.txt)
printf '%-22s %14s %14s\n' "system" "hello world" "bench.txt"
if command -v say >/dev/null 2>&1; then
    a=$(median_ms say -o "$work/a.aiff" "hello world")
    b=$(median_ms say -o "$work/b.aiff" "$bench_text")
    printf '%-22s %11s ms %11s ms\n' "say" "$a" "$b"
else
    echo "say: not available on this system"
fi
a=$(median_ms "$S" -b klatt -o "$work/a.wav" "hello world")
b=$(median_ms "$S" -b klatt -o "$work/b.wav" "$bench_text")
printf '%-22s %11s ms %11s ms\n' "speak -b klatt" "$a" "$b"
if "$S" -b diphone --dump units x >/dev/null 2>&1; then
    a=$(median_ms "$S" -b diphone -o "$work/a.wav" "hello world")
    b=$(median_ms "$S" -b diphone -o "$work/b.wav" "$bench_text")
    printf '%-22s %11s ms %11s ms\n' "speak -b diphone" "$a" "$b"
else
    echo "speak -b diphone: voice not built (speak-lab diphones build --bootstrap)"
fi

echo
echo "binary size: $(wc -c < "$S" | tr -d ' ') bytes"
echo
echo "in-process figures (no audio device, median of 5):"
"$S" -b klatt --bench=5 -f tests/fixtures/bench.txt | sed 's/^/  /'
