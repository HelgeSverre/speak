#!/bin/sh
# Intelligibility of the current release build: synthesizes tests/fixtures/sentences.txt,
# transcribes with whisper (a test instrument, spec §16.5) and prints the word error rate.
# Usage: scripts/wer.sh [speak arguments...]     e.g. scripts/wer.sh -v klatt-female
set -eu
cd "$(dirname "$0")/.."
S=./target/release/speak
L=./target/release/speak-lab
export SPEAK_LIGHTS=0
work=$(mktemp -d)
trap 'rm -rf "${work:?}"' EXIT
n=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    n=$((n + 1))
    "$S" "$@" -o "$work/$(printf '%02d' "$n").wav" "$line"
done < tests/fixtures/sentences.txt
whisper "$work"/*.wav --model base.en --language en --fp16 False --output_format txt --output_dir "$work/out" >/dev/null 2>&1
: > "$work/hyp.txt"
for f in "$work"/*.wav; do
    tr '\n' ' ' < "$work/out/$(basename "${f%.wav}").txt" >> "$work/hyp.txt"
    echo >> "$work/hyp.txt"
done
"$L" wer tests/fixtures/sentences.txt "$work/hyp.txt" | head -1
if [ "${VERBOSE:-0}" != 0 ]; then
    paste -d'\n' tests/fixtures/sentences.txt "$work/hyp.txt" | sed 'n;s/^/    -> /'
fi
