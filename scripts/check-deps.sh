#!/bin/sh
# Fails if the default build depends on a model runtime, a third-party speech frontend or a
# network client (spec §14 item 3).
set -eu
cd "$(dirname "$0")/.."
FORBIDDEN='^(ort|ort-sys|onnx[a-z0-9_-]*|tract[a-z0-9_-]*|candle[a-z0-9_-]*|burn[a-z0-9_-]*|tch|espeak[a-z0-9_-]*|misaki[a-z0-9_-]*|piper[a-z0-9_-]*|kokoro[a-z0-9_-]*|num2words|hf-hub|reqwest|ureq) '
found=$(cargo tree --prefix none --edges normal,build | sort -u | grep -E "$FORBIDDEN" || true)
if [ -n "$found" ]; then
    echo "check-deps: forbidden crates in the default build:" >&2
    echo "$found" >&2
    exit 1
fi
echo "ok: no model runtimes or third-party frontends"
