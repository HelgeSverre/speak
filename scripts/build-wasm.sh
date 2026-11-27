#!/usr/bin/env bash
# Builds the browser engine into website/pkg (speak.js + speak_bg.wasm).
set -euo pipefail
cd "$(dirname "$0")/.."
wasm-pack build --target web --release --no-typescript --no-pack --no-opt \
  --out-dir website/pkg --out-name speak \
  -- --no-default-features --features wasm
rm -f website/pkg/.gitignore
