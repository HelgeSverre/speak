// Shared by every page: the WASM engine, playback, the `speak` command line, highlighting.

import init, { Speaker, Request, voices, tunes, default_voice } from "./pkg/speak.js";

let speaker = null;
let ready = null;

/** Loads the engine once. */
export function load() {
  ready ??= init().then(() => {
    speaker = new Speaker();
    return {
      voices: JSON.parse(voices()),
      tunes: JSON.parse(tunes()),
      defaultVoice: default_voice(),
    };
  });
  return ready;
}

export const DEFAULTS = { rate: 175, volume: 1 };

/** Voice parameters: `-P key=value`. */
export const PARAMS = [
  { key: "source", values: ["klglott88", "lf", "impulse"], what: "Glottal source model" },
  { key: "oq", lo: 0.3, hi: 0.8, step: 0.01, what: "Open quotient: share of each cycle the glottis is open" },
  { key: "tl", lo: 0, hi: 34, step: 1, what: "Spectral tilt in dB; higher is darker" },
  { key: "breath", lo: 0, hi: 70, step: 1, what: "Aspiration noise in vowels, dB" },
  { key: "formant_scale", lo: 0.7, hi: 1.6, step: 0.01, what: "Multiplies all formants; above 1 is a shorter vocal tract" },
  { key: "flutter", lo: 0, hi: 100, step: 1, what: "Slow pitch wander, percent" },
  { key: "jitter", lo: 0, hi: 10, step: 0.1, what: "Cycle-to-cycle period noise, percent" },
  { key: "shimmer", lo: 0, hi: 40, step: 1, what: "Cycle-to-cycle amplitude noise, percent" },
  { key: "di", lo: 0, hi: 80, step: 1, what: "Diplophonia: alternate cycles paired, percent" },
  { key: "fry", lo: 0, hi: 1, step: 0.05, what: "Creak at the end of falling phrases" },
  { key: "sq", lo: 1.2, hi: 4, step: 0.1, what: "Speed quotient of the LF source" },
  { key: "range", lo: 0, hi: 3, step: 0.05, what: "Pitch range multiplier" },
  { key: "pace", lo: 0.5, hi: 2, step: 0.05, what: "Speaking rate multiplier" },
  { key: "pause", lo: 0.2, hi: 3, step: 0.05, what: "Pause length multiplier" },
];

/** A blank request state: everything at its default, as if no flag were given. */
export function blank() {
  return {
    text: "",
    mode: "",
    voice: "",
    rate: DEFAULTS.rate,
    pitch: "",
    volume: DEFAULTS.volume,
    sing: "",
    melody: "",
    tempo: "",
    key: "",
    params: {},
    outfile: "",
    fileFormat: "",
    dataFormat: "",
    channels: "",
    interactive: false,
    progress: false,
    timing: false,
    noLights: false,
  };
}

/** Synthesizes `s`. Returns PCM, timings and the pipeline dumps. */
export function synthesize(s) {
  const r = new Request(s.text);
  if (s.voice) r.voice = s.voice;
  r.rate = Number(s.rate) || 0;
  r.pitch = s.pitch || "";
  r.volume = Number(s.volume ?? 1);
  r.phonemes = s.mode || "";
  r.sing = s.sing || "";
  r.melody = s.melody || "";
  r.tempo = Number(s.tempo) || 0;
  r.key = Number(s.key) || 0;
  r.params = Object.entries(s.params || {})
    .filter(([, v]) => v !== "" && v != null)
    .map(([k, v]) => `${k}=${v}`)
    .join(" ");
  const t0 = performance.now();
  const u = speaker.speak(r);
  const ms = performance.now() - t0;
  const out = {
    samples: u.samples,
    rate: u.sample_rate,
    singShift: u.sing_shift,
    words: JSON.parse(u.words),
    sidepulse: JSON.parse(u.sidepulse),
    text: u.text,
    phonemes: u.phonemes,
    pho: u.pho,
    ms,
  };
  u.free();
  r.free();
  return out;
}

// ── Playback ────────────────────────────────────────────────────────

let ctx = null;
let current = null;

export function audio() {
  ctx ??= new AudioContext();
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

/** Plays an utterance. `clock()` gives seconds since the first sample left the speaker. */
export function play(u, { analyser } = {}) {
  stop();
  const ac = audio();
  const buf = ac.createBuffer(1, u.samples.length, u.rate);
  buf.copyToChannel(u.samples, 0);
  const src = ac.createBufferSource();
  src.buffer = buf;
  if (analyser) {
    src.connect(analyser);
    analyser.connect(ac.destination);
  } else {
    src.connect(ac.destination);
  }
  const start = ac.currentTime + 0.03;
  src.start(start);
  const latency = ac.outputLatency || ac.baseLatency || 0;
  const handle = {
    duration: buf.duration,
    clock: () => ac.currentTime - start - latency,
    ended: new Promise((res) => (src.onended = res)),
    stop: () => {
      try {
        src.stop();
      } catch {}
    },
  };
  current = handle;
  handle.ended.then(() => {
    if (current === handle) current = null;
  });
  return handle;
}

export function stop() {
  current?.stop();
  current = null;
}

/** 16-bit PCM WAV. */
export function wav(u) {
  const n = u.samples.length;
  const b = new DataView(new ArrayBuffer(44 + n * 2));
  const str = (o, s) => [...s].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  b.setUint32(4, 36 + n * 2, true);
  str(8, "WAVEfmt ");
  b.setUint32(16, 16, true);
  b.setUint16(20, 1, true);
  b.setUint16(22, 1, true);
  b.setUint32(24, u.rate, true);
  b.setUint32(28, u.rate * 2, true);
  b.setUint16(32, 2, true);
  b.setUint16(34, 16, true);
  str(36, "data");
  b.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, u.samples[i]));
    b.setInt16(44 + i * 2, v * 32767, true);
  }
  return new Blob([b], { type: "audio/wav" });
}

// ── SidePulse strip ─────────────────────────────────────────────────

const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

/** The 8 LED colours at `ms`, eased between keyframes as the device does (cosine). */
export function sidepulseAt(keys, ms) {
  if (!keys.length || ms < keys[0][0]) return Array(8).fill("#000000");
  let i = keys.findIndex((k) => k[0] > ms);
  if (i < 0) return keys[keys.length - 1].slice(1);
  const [a, b] = [keys[i - 1], keys[i]];
  const f = (ms - a[0]) / Math.max(1, b[0] - a[0]);
  const e = (1 - Math.cos(Math.PI * f)) / 2;
  return a.slice(1).map((ca, j) => {
    const [x, y] = [rgb(ca), rgb(b[j + 1])];
    return `rgb(${x.map((v, k) => Math.round(v + (y[k] - v) * e)).join(",")})`;
  });
}

// ── The command line ────────────────────────────────────────────────

const SAFE = /^[A-Za-z0-9_@%+=:,./-]+$/;
/** Shell-quotes with double quotes, escaping what sh expands inside them. */
export const quote = (s) => (SAFE.test(s) ? s : `"${s.replace(/["\\$`]/g, "\\$&")}"`);

/** `speak` arguments for a state, as [text, kind] tokens. */
export function command(s, defaultVoice) {
  const t = [["speak", "cmd"]];
  const flag = (f, v) => {
    t.push([f, "flag"]);
    if (v !== undefined) t.push([quote(String(v)), "str"]);
  };
  if (s.voice && s.voice !== defaultVoice) flag("-v", s.voice);
  if (Number(s.rate) !== DEFAULTS.rate) flag("-r", s.rate);
  if (s.pitch) flag("--pitch", s.pitch);
  if (Number(s.volume) !== DEFAULTS.volume) flag("--volume", s.volume);
  if (s.mode === "arpabet") flag("--phonemes");
  if (s.mode === "pho") flag("--phonemes=pho");
  if (s.sing) flag(`--sing=${s.sing}`);
  if (s.melody) flag("--melody", s.melody);
  if (s.tempo) flag("--tempo", s.tempo);
  if (s.key) flag("--key", s.key);
  for (const [k, v] of Object.entries(s.params || {})) if (v !== "" && v != null) flag("-P", `${k}=${v}`);
  if (s.outfile) flag("-o", s.outfile);
  if (s.fileFormat) flag(`--file-format=${s.fileFormat}`);
  if (s.dataFormat) flag(`--data-format=${s.dataFormat}`);
  if (s.channels) flag("--channels", s.channels);
  if (s.interactive) flag("-i");
  if (s.progress) flag("--progress");
  if (s.timing) flag("--timing");
  if (s.noLights) flag("--no-lights");
  if (s.text.trim()) t.push([quote(s.text.trim()), "str"]);
  return t;
}

export const commandText = (tokens) => tokens.map((t) => t[0]).join(" ");

export function commandHtml(tokens) {
  return tokens.map(([v, k]) => `<span class="tok-${k}">${esc(v)}</span>`).join(" ");
}

/** Shell words of a command line: quotes and backslashes as sh reads them. */
function words(line) {
  const out = [];
  let cur = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === "'") {
      const j = line.indexOf("'", i + 1);
      if (j < 0) throw new Error("Unclosed quote");
      cur = (cur ?? "") + line.slice(i + 1, j);
      i = j;
    } else if (c === '"') {
      let j = i + 1;
      let v = "";
      for (; j < line.length && line[j] !== '"'; j++) v += line[j] === "\\" ? line[++j] : line[j];
      if (j >= line.length) throw new Error("Unclosed quote");
      cur = (cur ?? "") + v;
      i = j;
    } else if (c === "\\") {
      cur = (cur ?? "") + (line[++i] ?? "");
    } else if (/\s/.test(c)) {
      if (cur !== null) out.push(cur);
      cur = null;
    } else {
      cur = (cur ?? "") + c;
    }
  }
  if (cur !== null) out.push(cur);
  return out;
}

const VALUE = {
  "-v": "voice", "--voice": "voice",
  "-r": "rate", "--rate": "rate",
  "--pitch": "pitch",
  "--volume": "volume",
  "--melody": "melody",
  "--tempo": "tempo",
  "--key": "key",
  "-o": "outfile", "--output-file": "outfile",
  "--file-format": "fileFormat",
  "--data-format": "dataFormat",
  "--channels": "channels",
};
const SWITCH = {
  "-i": "interactive", "--interactive": "interactive",
  "--progress": "progress",
  "--timing": "timing",
  "--no-lights": "noLights",
};

/** Parses a `speak …` command line into a state. Throws on what this page cannot do. */
export function parse(line) {
  const w = words(line.trim().replace(/^\$\s*/, ""));
  if (w[0] === "speak") w.shift();
  const s = blank();
  const text = [];
  for (let i = 0; i < w.length; i++) {
    const a = w[i];
    if (!a.startsWith("-") || a === "-" || /^-\d/.test(a)) {
      text.push(a);
      continue;
    }
    const [name, inline] = a.includes("=") ? [a.slice(0, a.indexOf("=")), a.slice(a.indexOf("=") + 1)] : [a, undefined];
    if (name === "-P") {
      const kv = inline ?? w[++i];
      const [k, v] = (kv ?? "").split("=");
      if (!k || v === undefined) throw new Error(`-P ${kv ?? ""}: expected key=value`);
      s.params[k] = v;
    } else if (name.startsWith("-P") && name.length > 2) {
      const [k, v] = a.slice(2).split("=");
      s.params[k] = v;
    } else if (VALUE[name]) {
      const v = inline ?? w[++i];
      if (v === undefined) throw new Error(`${name} needs a value`);
      s[VALUE[name]] = v;
    } else if (SWITCH[name]) {
      s[SWITCH[name]] = true;
    } else if (name === "--phonemes") {
      s.mode = inline === "pho" ? "pho" : "arpabet";
    } else if (name === "--sing") {
      s.sing = inline ?? "daisy";
    } else {
      throw new Error(`${name} works on the command line only`);
    }
  }
  s.text = text.join(" ");
  s.rate = Number(s.rate);
  s.volume = Number(s.volume);
  return s;
}

// ── Highlighting for static code blocks ─────────────────────────────

export function esc(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

const span = (k, v) => `<span class="tok-${k}">${esc(v)}</span>`;

function highlightSh(src) {
  return src
    .split("\n")
    .map((line) => {
      if (/^\s*#/.test(line)) return span("com", line);
      let first = true;
      return line.replace(/('[^']*'|"[^"]*"|\S+|\s+)/g, (m) => {
        if (/^\s+$/.test(m)) return m;
        if (m === "&&" || m === "|") {
          first = true;
          return span("plain", m);
        }
        if (first) {
          first = false;
          return span("cmd", m);
        }
        if (/^['"]/.test(m)) return span("str", m);
        if (/^-/.test(m)) return span("flag", m);
        return span("plain", m);
      });
    })
    .join("\n");
}

function highlightToml(src) {
  return src
    .split("\n")
    .map((line) => {
      if (/^\s*#/.test(line)) return span("com", line);
      if (/^\s*\[/.test(line)) return span("key", line);
      const m = line.match(/^(\s*)([\w.-]+)(\s*=\s*)(.*)$/);
      if (!m) return esc(line);
      const val = m[4].replace(/("[^"]*")|(\b\d+(\.\d+)?\b)|(true|false)|([^"\d]+)/g, (x, str, num, _d, bool) =>
        str ? span("str", x) : num ? span("num", x) : bool ? span("num", x) : esc(x),
      );
      return `${m[1]}${span("flag", m[2])}${esc(m[3])}${val}`;
    })
    .join("\n");
}

function highlightLex(src) {
  return src
    .split("\n")
    .map((line) => line.replace(/^(\S+)(\s+)(.*)$/, (_, w, sp, p) => `${span("flag", w)}${sp}${span("str", p)}`))
    .join("\n");
}

/** Colours every `pre[data-lang]` and wires every `[data-copy]` button. */
export function decorate(root = document) {
  for (const pre of root.querySelectorAll("pre[data-lang]")) {
    const code = pre.querySelector("code") ?? pre;
    const src = code.textContent;
    const f = { sh: highlightSh, toml: highlightToml, lex: highlightLex }[pre.dataset.lang];
    if (f) code.innerHTML = f(src);
  }
  for (const btn of root.querySelectorAll("[data-copy]")) {
    btn.addEventListener("click", async () => {
      const el = document.querySelector(btn.dataset.copy);
      try {
        await navigator.clipboard.writeText(el.textContent.trim());
        btn.classList.add("done");
        btn.textContent = "Copied";
        setTimeout(() => {
          btn.classList.remove("done");
          btn.textContent = "Copy";
        }, 1400);
      } catch {}
    });
  }
}

export const magma = (() => {
  // Six stops of the magma colour map, interpolated to 256 entries.
  const stops = [
    [0, [0, 0, 4]],
    [0.2, [59, 15, 112]],
    [0.4, [140, 41, 129]],
    [0.6, [222, 73, 104]],
    [0.8, [254, 159, 109]],
    [1, [252, 253, 191]],
  ];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    const j = stops.findIndex((s) => s[0] >= x);
    const [a, b] = j <= 0 ? [stops[0], stops[0]] : [stops[j - 1], stops[j]];
    const f = b[0] === a[0] ? 0 : (x - a[0]) / (b[0] - a[0]);
    for (let k = 0; k < 3; k++) lut[i * 3 + k] = a[1][k] + (b[1][k] - a[1][k]) * f;
  }
  return lut;
})();

// ── Spectrogram ─────────────────────────────────────────────────
// Wideband: a 10 ms window zero-padded to 512 points shows formants rather than harmonics.
const N = 512;
const W = 160;
const win = Float32Array.from({ length: W }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (W - 1)));
const rev = (() => {
  const r = new Uint16Array(N);
  for (let i = 0, bits = Math.log2(N); i < N; i++) {
    let x = i, y = 0;
    for (let b = 0; b < bits; b++) (y = (y << 1) | (x & 1)), (x >>= 1);
    r[i] = y;
  }
  return r;
})();

function fft(re, im) {
  for (let i = 0; i < N; i++) {
    const j = rev[i];
    if (j > i) [re[i], re[j], im[i], im[j]] = [re[j], re[i], im[j], im[i]];
  }
  for (let size = 2; size <= N; size *= 2) {
    const half = size / 2, step = (-2 * Math.PI) / size;
    for (let i = 0; i < N; i += size) {
      for (let k = 0; k < half; k++) {
        const wr = Math.cos(step * k), wi = Math.sin(step * k);
        const a = i + k, b = a + half;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr, im[b] = im[a] - ti;
        re[a] += tr, im[a] += ti;
      }
    }
  }
}

/** Draws a wideband spectrogram (0–8 kHz, magma) of `samples` filling canvas `c`. */
export function drawSpectrogram(c, samples) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(c.clientWidth * dpr));
  const h = Math.max(1, Math.round(c.clientHeight * dpr));
  c.width = w;
  c.height = h;
  const g = c.getContext("2d");
  const cols = Math.min(w, 1600);
  const bins = N / 2;
  const db = new Float32Array(cols * bins);
  const re = new Float64Array(N), im = new Float64Array(N);
  const hop = Math.max(1, (samples.length - W) / cols);
  let max = -Infinity;
  for (let x = 0; x < cols; x++) {
    const o = Math.floor(x * hop);
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < W; i++) re[i] = (samples[o + i] || 0) * win[i];
    fft(re, im);
    for (let k = 0; k < bins; k++) {
      // A gentle tilt so the upper formants show.
      const v = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12) + 12 * Math.log10(1 + k / 24);
      db[x * bins + k] = v;
      if (v > max) max = v;
    }
  }
  const img = g.createImageData(cols, bins);
  const range = 62;
  for (let x = 0; x < cols; x++) {
    for (let k = 0; k < bins; k++) {
      const v = Math.max(0, Math.min(1, (db[x * bins + k] - (max - range)) / range));
      const i = Math.round(Math.pow(v, 1.35) * 255) * 3;
      const p = ((bins - 1 - k) * cols + x) * 4;
      img.data[p] = magma[i];
      img.data[p + 1] = magma[i + 1];
      img.data[p + 2] = magma[i + 2];
      img.data[p + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(cols, bins);
  off.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingQuality = "high";
  g.drawImage(off, 0, 0, w, h);
}
