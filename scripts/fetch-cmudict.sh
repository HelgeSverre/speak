#!/bin/sh
# Downloads the pinned CMUdict (data, not a model) and verifies it (spec §7.4).
# The product itself has no network code; this script is the only download.
set -eu
REV=74790861f652b15e4ac49015a90074ad62a27690
BASE="https://raw.githubusercontent.com/cmusphinx/cmudict/$REV"
DICT_SHA=81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22
DICT_SIZE=3618488
LICENSE_SHA=bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a

if [ -n "${SPEAK_DATA_DIR:-}" ]; then
    DATA=$SPEAK_DATA_DIR
elif [ "$(uname)" = "Darwin" ]; then
    DATA="$HOME/Library/Application Support/speak"
else
    DATA="${XDG_DATA_HOME:-$HOME/.local/share}/speak"
fi
DIR="$DATA/lexicon"
mkdir -p "$DIR"
# Downloads land in a scratch directory inside the lexicon directory and are moved into place
# only after verification.
WORK=$(mktemp -d "$DIR/fetch.XXXXXX")
trap 'rm -rf "${WORK:?}"' EXIT

sha() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        sha256sum "$1" | cut -d' ' -f1
    fi
}

fetch() { # file name, expected SHA-256
    curl -fsSL -o "$WORK/$1" "$BASE/$1"
    got=$(sha "$WORK/$1")
    if [ "$got" != "$2" ]; then
        echo "fetch-cmudict: $1: SHA-256 mismatch: got $got, expected $2" >&2
        exit 1
    fi
}

fetch cmudict.dict "$DICT_SHA"
size=$(wc -c < "$WORK/cmudict.dict" | tr -d ' ')
if [ "$size" != "$DICT_SIZE" ]; then
    echo "fetch-cmudict: cmudict.dict: size $size, expected $DICT_SIZE" >&2
    exit 1
fi
fetch LICENSE "$LICENSE_SHA"
mv "$WORK/cmudict.dict" "$DIR/cmudict.dict"
mv "$WORK/LICENSE" "$DIR/CMUDICT-LICENSE"
echo "cmudict.dict verified $DICT_SHA"
echo "$DIR/cmudict.dict"
