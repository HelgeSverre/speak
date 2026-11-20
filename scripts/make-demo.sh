#!/bin/sh
# Regenerates demo/: one MP3 and one spectrogram JPEG per clip, plus a few waveform videos.
# Needs a release build (cargo build --release) and ffmpeg with libmp3lame and libx264.
set -eu
cd "$(dirname "$0")/.."
S=./target/release/speak
export SPEAK_LIGHTS=0
mkdir -p demo
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

gen() {
    name=$1
    shift
    "$S" -o "$tmp/$name.wav" "$@"
    ffmpeg -v error -y -i "$tmp/$name.wav" -ar 44100 -ac 1 -codec:a libmp3lame -b:a 64k "demo/$name.mp3"
    ffmpeg -v error -y -i "$tmp/$name.wav" \
        -lavfi "showspectrumpic=s=1000x200:legend=0:color=magma:scale=log:stop=8000" -q:v 5 "demo/$name.jpg"
}

video() {
    ffmpeg -v error -y -i "demo/$1.mp3" -filter_complex \
        "[0:a]aresample=16000,showspectrum=s=960x240:mode=combined:color=magma:scale=log:slide=scroll:stop=8000:fps=30[s];[0:a]showwaves=s=960x120:mode=cline:colors=0xffb347:rate=30[w];[s][w]vstack=inputs=2,format=yuv420p[v]" \
        -map "[v]" -map 0:a -c:v libx264 -preset slow -crf 30 -c:a aac -b:a 64k -movflags +faststart "demo/$1.mp4"
}

gen hello 'Hello world. I am speak. Every sound you hear was computed from scratch: eight resonators, one glottal source, and no neural network.'
gen normalizer 'Dr. Smith paid $3.50 on 12/03/2024 at 5pm. The file is 2GB, which is 45% of the disk. Call 555-1234 or visit example.com.'
gen fox 'The quick brown fox jumps over the lazy dog.'
gen voice-female -v klatt-female 'I am the female preset. Higher pitch, shorter vocal tract, a little breath.'
gen voice-pip -v klatt-pip 'I am Pip. I talk fast, I ask questions, and I think everything is interesting. Is that a new terminal theme?'
gen voice-calm -v klatt-calm 'I am Calm. I have read your stack trace. Everything is going to be fine.'
gen voice-gravel -v klatt-gravel 'Gravel. I have been compiling since 1979. I do not trust garbage collectors.'
gen voice-drama -v klatt-drama 'Drama! To deploy, or not to deploy? That is never the question on a Friday.'
gen voice-robot -v klatt-robot 'I am a robot. Beep beep beep. My pitch does not move.'
gen voice-whisper -v klatt-whisper 'This is the whisper voice. No voicing at all, only noise through the same filters.'
gen fast -r 320 'At three hundred and twenty words per minute I can read the terms and conditions before you find the checkbox.'
gen question 'Is this a question? Yes, it is. And this, with a comma, is a statement!'
gen sing-daisy --sing=daisy
gen sing-twinkle -v klatt-pip --sing=twinkle
gen sing-ode -v klatt-calm --sing=ode
gen sing-absolutely-right --melody @tunes/youre-absolutely-right.melody --tempo 52 -f tunes/youre-absolutely-right.txt
gen sing-custom --melody 'C4 E4 G4 C5:2 G4 E4 C4:2' --tempo 140 'la la la la, do re mi fa'
gen phonemes --phonemes 'HH AX L OW1 / F R AH1 M / R AO1 / F OW1 N IY2 M Z .'

# The second backend needs its voice: speak-lab diphones build --bootstrap
if "$S" -b diphone --dump units 'x' >/dev/null 2>&1; then
    gen diphone -b diphone 'This is the second backend. Two thousand tiny pieces of the first voice, measured, chopped up, and glued back together.'
    gen diphone-pulse -b diphone -P excitation=pulse 'With a plain pulse train, I sound like a toy from 1978.'
    gen diphone-sing -b diphone --sing=twinkle
else
    echo "skipping diphone clips: voice not built" >&2
fi

for clip in hello sing-daisy sing-absolutely-right voice-drama; do
    video "$clip"
done
echo "demo/ regenerated"
