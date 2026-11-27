import { blank, command, commandText, esc, load, parse, play, quote, sidepulseAt, stop, synthesize, wav, drawSpectrogram } from "./engine.js";

const $ = (s) => document.querySelector(s);
const ui = {
  text: $("#text"), play: $("#play"), mode: $("#mode"), voices: $("#voices"),
  rate: $("#rate"), rateFill: $("#rate-fill"), rateVal: $("#rate-val"), rateUnit: $("#rate-unit"),
  singbar: $("#singbar"), presets: $("#presets"), melody: $("#melody"),
  scope: $("#scope"), spec: $("#spec"), pitch: $("#pitch"), words: $("#words"), veil: $("#veil"),
  playhead: $("#playhead"), status: $("#status"), reset: $("#reset"), randomize: $("#randomize"), hint: $("#hint"),
  strip: $("#strip"), cmd: $("#cmd"), copy: $("#copy"), savePho: $("#save-pho"), saveWav: $("#save-wav"),
  saveMp3: $("#save-mp3"),
};

const VOWELS = new Set("AA AE AH AO AW AX AXR AY EH ER EY IH IX IY OW OY UH UW UX".split(" "));
const HZ = { lo: 50, hi: 500 };
const SVG = "http://www.w3.org/2000/svg";
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const RULER = {
  speak: { min: 80, max: 500, step: 5, unit: "wpm", label: "Rate in words per minute" },
  sing: { min: 40, max: 240, step: 2, unit: "bpm", label: "Tempo in beats per minute" },
};

let engine = null;
let mode = "speak";
let voice = "";
let rate = 175;
let tempo = 132;
let base = null; // the text as synthesized: { key, u, pho }
let offsets = []; // speech mode: semitones per vowel handle
let out = null; // what plays: base.u, or the re-pitched speech
let playing = null;

// ── Timeline: the audio, plus one beat of headroom when singing so the last note can grow ──

let spanLock = 0; // seconds, held while a note is dragged so the scale stays put

function span(u = out) {
  const audio = u.samples.length / u.rate;
  const want = mode === "sing" ? audio + 60 / tempo : audio;
  return Math.max(want, spanLock);
}

/** The spectrogram over the whole timeline: the audio, then silence to the end. */
function drawSpec(u = out) {
  const n = Math.round(span(u) * u.rate);
  if (n <= u.samples.length) return drawSpectrogram(ui.spec, u.samples);
  const padded = new Float32Array(n);
  padded.set(u.samples);
  drawSpectrogram(ui.spec, padded);
}

// ── Melody: tokens of `--melody`, edited by the note roll and the text field ──

let notes = []; // { midi: number|null, beats: number }

function parseNotes(text) {
  const out = [];
  for (const tok of text.trim().split(/\s+/).filter(Boolean)) {
    const [name, b] = tok.split(":");
    const beats = b === undefined ? 1 : Number(b);
    if (!(beats > 0)) throw new Error(`Bad note '${tok}'`);
    if (/^(r|_)$/i.test(name)) {
      out.push({ midi: null, beats });
      continue;
    }
    const m = name.match(/^([A-Ga-g])([#b]?)(\d)$/);
    if (!m) throw new Error(`Bad note '${tok}'`);
    const semi = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1].toUpperCase()] + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0);
    out.push({ midi: 12 * (Number(m[3]) + 1) + semi, beats });
  }
  return out;
}

const noteName = (midi) => `${NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
const fmtBeats = (b) => (b === 1 ? "" : `:${Number(b.toFixed(2))}`);
const melodyText = () => notes.map((n) => `${n.midi === null ? "R" : noteName(n.midi)}${fmtBeats(n.beats)}`).join(" ");

function setMelody(text, { field = true } = {}) {
  notes = parseNotes(text);
  if (field) ui.melody.value = melodyText();
}

/** A melody that follows the speech intonation of the text, one note per syllable. */
function melodyFromWords() {
  const u = synthesize(Object.assign(blank(), { text: ui.text.value, voice }));
  const vowels = parsePho(u.pho).vowels;
  if (!vowels.length) return "C4";
  return vowels
    .map((v, i) => {
      // Up a fifth, where the engine fits a sung melody, so it is sung as written.
      const midi = Math.round(69 + 12 * Math.log2((v.hz * 1.5) / 440));
      const beats = i === vowels.length - 1 || /1$/.test(v.p.tok) ? 2 : 1;
      return `${noteName(midi)}${fmtBeats(beats)}`;
    })
    .join(" ");
}

// ── .pho model: phones with pitch targets on a millisecond timeline ──

function parsePho(text) {
  const lines = [];
  const vowels = [];
  let t = 0;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith(";")) {
      lines.push({ raw });
      continue;
    }
    const [tok, dur, ...rest] = line.split(/\s+/);
    const d = Number(dur);
    const f0 = [];
    for (let i = 0; i + 1 < rest.length; i += 2) f0.push([Number(rest[i]), Number(rest[i + 1])]);
    const p = { tok, dur: d, f0, t0: t };
    if (VOWELS.has(tok.replace(/\d/g, ""))) {
      const hz = f0.length ? hzAt(p, 50) : 0;
      // A vowel's handle pulls at its centre, so it gets a target there.
      if (f0.length && !f0.some(([pct]) => pct === 50)) {
        f0.push([50, Math.round(hz)]);
        f0.sort((a, b) => a[0] - b[0]);
      }
      vowels.push({ p, t: t + d / 2, hz });
    }
    lines.push(p);
    t += d;
  }
  return { lines, vowels, total: t };
}

function hzAt(p, pct) {
  const f = p.f0;
  if (pct <= f[0][0]) return f[0][1];
  for (let i = 1; i < f.length; i++) {
    if (pct <= f[i][0]) {
      const [a, b] = [f[i - 1], f[i]];
      return a[1] + ((b[1] - a[1]) * (pct - a[0])) / Math.max(1, b[0] - a[0]);
    }
  }
  return f[f.length - 1][1];
}

const pitched = () => base.pho.vowels.filter((v) => v.hz > 0);

/** Semitone offset at `t` ms, linear between neighbouring handles. */
function offsetAt(t) {
  const v = pitched();
  if (!v.length) return 0;
  if (t <= v[0].t) return offsets[0];
  for (let i = 1; i < v.length; i++) {
    if (t <= v[i].t) return offsets[i - 1] + ((offsets[i] - offsets[i - 1]) * (t - v[i - 1].t)) / (v[i].t - v[i - 1].t);
  }
  return offsets[v.length - 1];
}

const shifted = (hz, t) => Math.max(40, Math.min(600, hz * 2 ** (offsetAt(t) / 12)));

/** The edited phones as .pho lines (no comments). */
function phoLines() {
  return base.pho.lines
    .filter((p) => p.raw === undefined)
    .map((p) => {
      const targets = p.f0.map(([pct, hz]) => ` ${pct} ${Math.round(shifted(hz, p.t0 + (pct / 100) * p.dur))}`).join("");
      return `${p.tok} ${p.dur}${targets}`;
    });
}

const edited = () => mode === "speak" && offsets.some((o) => Math.abs(o) > 0.01);

// ── SidePulse strip ──

const leds = Array.from({ length: 8 }, () => ui.strip.appendChild(document.createElement("i")));
function showStrip(colors) {
  leds.forEach((el, i) => {
    const c = colors?.[i];
    const dark = !c || c === "#000000" || c === "rgb(0,0,0)";
    el.style.setProperty("--led", dark ? "" : c);
    el.style.setProperty("--glow", dark ? "" : c);
  });
}

// ── The command line ──

function state() {
  const s = Object.assign(blank(), { text: ui.text.value, voice });
  if (mode === "speak") s.rate = rate;
  else Object.assign(s, { melody: melodyText(), tempo });
  return s;
}

/** Tokens as [text, kind, joinsPrevious]: a flag's value stays on the flag's line when wrapping. */
function commandTokens() {
  if (!edited()) {
    const t = command(state(), engine.defaultVoice);
    return t.map(([v, k], i) => [v, k, k === "str" && t[i - 1]?.[1] === "flag"]);
  }
  const t = [["speak", "cmd", false]];
  if (voice !== engine.defaultVoice) t.push(["-v", "flag", false], [voice, "str", true]);
  t.push(["--phonemes=pho", "flag", false], [quote(phoLines().join(" | ")), "str", false]);
  return t;
}

let commandLine = "speak";
function showCommand() {
  const t = commandTokens();
  commandLine = t.map((x) => x[0]).join(" ");
  // Display only: each flag with its value on its own indented line. The line breaks and the
  // trailing backslashes are CSS, so neither is ever copied.
  const groups = [];
  for (const tok of t) {
    if (tok[2] && groups.length) groups[groups.length - 1].push(tok);
    else groups.push([tok]);
  }
  const html = (g) => g.map(([v, k]) => `<span class="tok-${k}">${esc(v)}</span>`).join(" ");
  ui.cmd.innerHTML = `<code>${groups.map((g, i) => `<span class="arg${i ? "" : " first"}">${html(g)}</span>`).join("")}</code>`;
  ui.savePho.hidden = !edited();
  ui.reset.hidden = !edited();
  const plain = commandText(command(state(), engine.defaultVoice));
  history.replaceState(null, "", `#${encodeURIComponent(plain)}`);
}

// Copying, by button or by selection, gives one line with ordinary spaces.
ui.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(commandLine);
    ui.copy.textContent = "Copied";
    ui.copy.classList.add("done");
    setTimeout(() => {
      ui.copy.textContent = "Copy";
      ui.copy.classList.remove("done");
    }, 1400);
  } catch {}
});
ui.cmd.addEventListener("copy", (e) => {
  const sel = document.getSelection().toString();
  e.clipboardData.setData("text/plain", sel.replace(/\u00a0/g, " ").replace(/\s*\n\s*/g, " "));
  e.preventDefault();
});

// ── Rendering ──

function status(ok, text) {
  ui.status.classList.toggle("err", !ok);
  ui.status.textContent = text;
}

/** Synthesizes when the text, voice, mode, rate, tempo or melody changed. */
function renderBase() {
  const s = state();
  const key = JSON.stringify([mode, s.text, voice, s.rate, s.melody, s.tempo]);
  if (base?.key === key) return;
  try {
    const u = synthesize(s);
    base = { key, u, pho: parsePho(u.pho) };
    offsets = pitched().map(() => 0);
    out = u;
    drawSpec(u);
    placeWords(u);
    drawOverlay();
    const octaves = u.singShift / 12;
    const fit = mode === "sing" && octaves ? `, sung ${Math.abs(octaves)} octave${Math.abs(octaves) > 1 ? "s" : ""} ${octaves < 0 ? "down" : "up"}` : "";
    status(true, `${(u.samples.length / u.rate).toFixed(2)} s in ${u.ms.toFixed(0)} ms${fit}`);
  } catch (e) {
    base = null;
    out = null;
    ui.pitch.replaceChildren();
    status(false, String(e.message || e));
  }
  showCommand();
}

/** Speech mode: re-synthesizes from the edited .pho, same timing, new pitch. */
function renderPitch() {
  if (!base) return;
  if (!edited()) {
    out = base.u;
  } else {
    try {
      out = { ...synthesize(Object.assign(blank(), { mode: "pho", text: phoLines().join("\n"), voice })), words: base.u.words };
    } catch (e) {
      status(false, String(e.message || e));
      return;
    }
  }
  drawSpec();
  showCommand();
}

const frames = {};
function onFrame(name, f) {
  if (!frames[name]) frames[name] = requestAnimationFrame(() => ((frames[name] = 0), f()));
}

function placeWords(u) {
  const dur = span(u);
  ui.words.replaceChildren(
    ...u.words.map((wd, i) => {
      const s = document.createElement("span");
      s.textContent = wd.text;
      const next = u.words[i + 1]?.start ?? dur;
      s.style.left = `${(wd.start / dur) * 100}%`;
      s.style.width = `${((next - wd.start) / dur) * 100}%`;
      return s;
    }),
  );
}

// ── Overlay: the pitch curve (speak) or the note roll (sing) ──

function el(name, attrs, text) {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

let geom = null;

function drawOverlay(drag = -1) {
  if (!base) return;
  const w = ui.pitch.clientWidth;
  const h = ui.pitch.clientHeight;
  const totalMs = span(base.u) * 1000;
  ui.pitch.setAttribute("viewBox", `0 0 ${w} ${h}`);
  ui.pitch.setAttribute("preserveAspectRatio", "none");
  geom = { w, h, x: (t) => (t / totalMs) * w, msPerPx: totalMs / w };
  ui.pitch.replaceChildren(...(mode === "sing" ? roll(drag) : curve(drag)));
}

function curve(dragging) {
  const g = geom;
  const pad = g.h * 0.12;
  g.y = (hz) => pad + (g.h - 2 * pad) * (1 - Math.log(hz / HZ.lo) / Math.log(HZ.hi / HZ.lo));
  g.hz = (y) => HZ.lo * (HZ.hi / HZ.lo) ** (1 - (y - pad) / (g.h - 2 * pad));
  const parts = [];
  for (const hz of [100, 200, 400]) {
    parts.push(el("line", { class: "grid", x1: 0, x2: g.w, y1: g.y(hz), y2: g.y(hz) }));
    parts.push(el("text", { class: "grid-label", x: 12, y: g.y(hz) - 4 }, `${hz} Hz`));
  }
  let run = [];
  const runs = [];
  for (const p of base.pho.lines) {
    if (p.raw !== undefined) continue;
    if (!p.f0.length) {
      if (run.length) runs.push(run);
      run = [];
      continue;
    }
    for (const [pct, hz] of p.f0) {
      const t = p.t0 + (pct / 100) * p.dur;
      run.push(`${g.x(t).toFixed(1)},${g.y(shifted(hz, t)).toFixed(1)}`);
    }
  }
  if (run.length) runs.push(run);
  for (const r of runs) {
    parts.push(el("polyline", { class: "shadow", points: r.join(" ") }));
    parts.push(el("polyline", { class: "curve", points: r.join(" ") }));
  }
  pitched().forEach((v, i) => {
    const cx = g.x(v.t);
    const cy = g.y(v.hz * 2 ** (offsets[i] / 12));
    parts.push(el("circle", { class: `dot${i === dragging ? " drag" : ""}`, cx, cy, r: i === dragging ? 7 : 5 }));
    parts.push(el("circle", { class: "hit", cx, cy, r: 16, "data-i": i }));
    if (i === dragging) {
      const st = offsets[i];
      parts.push(el("text", { class: "hz", x: cx + 12, y: cy - 10 }, `${Math.round(v.hz * 2 ** (st / 12))} Hz  ${st >= 0 ? "+" : ""}${st.toFixed(1)} st`));
    }
  });
  return parts;
}

/** Syllable k sings the k-th note of the looping melody, skipping rests. */
function noteIndices(count) {
  const idx = [];
  if (!notes.some((n) => n.midi !== null)) return idx;
  for (let i = 0; idx.length < count; i = (i + 1) % notes.length) if (notes[i].midi !== null) idx.push(i);
  return idx;
}

function roll(dragging) {
  const g = geom;
  const syl = base.pho.vowels;
  const idx = noteIndices(syl.length);
  const pitches = notes.filter((n) => n.midi !== null).map((n) => n.midi);
  const lo = Math.min(...pitches) - 3;
  const hi = Math.max(lo + 14, Math.max(...pitches) + 3);
  const pad = 10;
  const row = (g.h - 2 * pad) / (hi - lo + 1);
  g.y = (midi) => pad + (hi - midi) * row;
  g.midi = (y) => hi - (y - pad) / row + 0.5;
  g.row = row;
  const beatMs = 60000 / tempo;
  const parts = [
    el("defs", {}),
  ];
  parts[0].innerHTML = `<linearGradient id="note-fill" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffd59a"/><stop offset="0.5" stop-color="#fe9f6d"/><stop offset="1" stop-color="#de4968"/></linearGradient>`;
  for (let m = lo; m <= hi; m++) {
    const c = m % 12 === 0;
    parts.push(el("line", { class: `row-line${c ? " c" : ""}`, x1: 0, x2: g.w, y1: g.y(m) + row / 2, y2: g.y(m) + row / 2 }));
    if (c || row >= 14) parts.push(el("text", { class: "row-label", x: 8, y: g.y(m) + row / 2 - 3 }, noteName(m)));
  }
  const seen = new Set();
  syl.forEach((v, k) => {
    const i = idx[k];
    const n = notes[i];
    const x = g.x(v.p.t0);
    const width = Math.max(10, g.x(v.p.t0 + n.beats * beatMs) - x - 2);
    const y = g.y(n.midi) + 1;
    const hgt = Math.max(6, row - 2);
    const looped = seen.has(i);
    seen.add(i);
    const grp = el("g", { class: `note${i === dragging ? " drag" : ""}${looped ? " loop" : ""}`, "data-n": i });
    grp.append(el("rect", { x, y, width, height: hgt, rx: Math.min(5, hgt / 2), "data-n": i }));
    if (width > 28 && hgt >= 12) grp.append(el("text", { x: x + 5, y: y + hgt / 2 + 3.5 }, noteName(n.midi)));
    grp.append(el("rect", { class: "edge", x: x + width - 8, y, width: 8, height: hgt, "data-edge": i }));
    parts.push(grp);
  });
  return parts;
}

// ── Dragging: pitch dots (speak) and notes (sing) ──

let drag = null;
ui.pitch.addEventListener("pointerdown", (e) => {
  const d = e.target.dataset;
  if (mode === "speak" && d?.i !== undefined) drag = { kind: "dot", i: Number(d.i) };
  else if (mode === "sing" && d?.edge !== undefined) drag = { kind: "len", i: Number(d.edge), x: e.clientX, beats: notes[Number(d.edge)].beats };
  else if (mode === "sing" && d?.n !== undefined) drag = { kind: "note", i: Number(d.n), y: e.clientY, midi: notes[Number(d.n)].midi };
  else return;
  e.preventDefault();
  drag.id = e.pointerId;
  if (drag.kind !== "dot") spanLock = span();
  try {
    ui.pitch.setPointerCapture(e.pointerId);
  } catch {}
  ui.hint.classList.add("gone");
  drawOverlay(drag.i);
});
ui.pitch.addEventListener("pointermove", (e) => {
  if (!drag || e.pointerId !== drag.id) return;
  if (drag.kind === "dot") {
    const r = ui.pitch.getBoundingClientRect();
    const v = pitched()[drag.i];
    offsets[drag.i] = Math.max(-12, Math.min(12, 12 * Math.log2(geom.hz(e.clientY - r.top) / v.hz)));
    drawOverlay(drag.i);
    onFrame("pitch", renderPitch);
    return;
  }
  const n = notes[drag.i];
  if (drag.kind === "note") {
    const midi = Math.max(24, Math.min(96, drag.midi + Math.round((drag.y - e.clientY) / geom.row)));
    if (midi === n.midi) return;
    n.midi = midi;
  } else {
    const beatPx = 60000 / tempo / geom.msPerPx;
    const beats = Math.max(0.25, Math.round((drag.beats + (e.clientX - drag.x) / beatPx) * 4) / 4);
    if (beats === n.beats) return;
    n.beats = beats;
  }
  ui.melody.value = melodyText();
  onFrame("base", () => {
    renderBase();
    if (drag) drawOverlay(drag.i);
  });
});
const endDrag = (e) => {
  if (!drag || e.pointerId !== drag.id) return;
  spanLock = 0;
  const kind = drag.kind;
  drag = null;
  if (kind === "dot") {
    renderPitch();
  } else {
    renderBase();
    // The scale was held during the drag; redraw at the final one.
    if (out) {
      drawSpec();
      placeWords(out);
    }
  }
  drawOverlay();
};
ui.pitch.addEventListener("pointerup", endDrag);
ui.pitch.addEventListener("pointercancel", endDrag);
ui.pitch.addEventListener("dblclick", (e) => {
  const i = e.target.dataset?.i;
  if (mode !== "speak" || i === undefined) return;
  offsets[Number(i)] = 0;
  drawOverlay();
  renderPitch();
});

ui.reset.addEventListener("click", () => {
  offsets = offsets.map(() => 0);
  drawOverlay();
  renderPitch();
});
ui.randomize.addEventListener("click", () => {
  if (!base) return;
  ui.hint.classList.add("gone");
  if (mode === "speak") {
    offsets = offsets.map(() => (Math.random() * 2 - 1) * 8);
    drawOverlay();
    renderPitch();
  } else {
    // A random walk on a major scale around the current notes.
    const scale = [0, 2, 4, 5, 7, 9, 11];
    let deg = 7;
    for (const n of notes) {
      if (n.midi === null) continue;
      deg = Math.max(0, Math.min(14, deg + Math.round((Math.random() * 2 - 1) * 2)));
      n.midi = 60 + 12 * Math.floor(deg / 7) + scale[deg % 7];
      n.beats = [0.5, 1, 1, 1, 2][Math.floor(Math.random() * 5)];
    }
    ui.melody.value = melodyText();
    renderBase();
  }
});

// ── Mode, voices, ruler ──

function setMode(m) {
  mode = m;
  for (const b of ui.mode.children) b.setAttribute("aria-checked", String(b.dataset.mode === m));
  ui.singbar.hidden = m !== "sing";
  ui.randomize.textContent = m === "sing" ? "Random melody" : "Randomize pitch";
  ui.hint.textContent = m === "sing" ? "Drag a note up or down; drag its right edge to lengthen it" : "Drag the dots to change the pitch";
  ui.hint.classList.remove("gone");
  if (m === "sing" && !notes.length) setMelody(melodyFromWords());
  setRuler();
}

function setVoice(name) {
  voice = name;
  for (const b of ui.voices.children) b.setAttribute("aria-checked", String(b.dataset.voice === name));
}

function setRuler(v) {
  const r = RULER[mode];
  const cur = Math.round(Math.min(r.max, Math.max(r.min, v ?? (mode === "speak" ? rate : tempo))) / r.step) * r.step;
  if (mode === "speak") rate = cur;
  else tempo = cur;
  Object.entries({ "aria-valuemin": r.min, "aria-valuemax": r.max, "aria-valuenow": cur, "aria-label": r.label }).forEach(([k, val]) =>
    ui.rate.setAttribute(k, val),
  );
  ui.rateVal.textContent = cur;
  ui.rateUnit.textContent = r.unit;
  ui.rateFill.style.width = `${((cur - r.min) / (r.max - r.min)) * 100}%`;
}

let rulerDrag = null;
function rulerFrom(e) {
  const r = ui.rate.getBoundingClientRect();
  const cfg = RULER[mode];
  setRuler(cfg.min + ((e.clientX - r.left) / r.width) * (cfg.max - cfg.min));
  onFrame("base", renderBase);
}
ui.rate.addEventListener("pointerdown", (e) => {
  rulerDrag = e.pointerId;
  ui.rate.setPointerCapture(e.pointerId);
  rulerFrom(e);
});
ui.rate.addEventListener("pointermove", (e) => e.pointerId === rulerDrag && rulerFrom(e));
ui.rate.addEventListener("pointerup", () => (rulerDrag = null));
ui.rate.addEventListener("keydown", (e) => {
  const step = RULER[mode].step;
  const d = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step, PageUp: step * 10, PageDown: -step * 10 }[e.key];
  if (d === undefined) return;
  e.preventDefault();
  setRuler((mode === "speak" ? rate : tempo) + d);
  renderBase();
});

// ── Playback ──

function idle() {
  playing = null;
  ui.play.classList.remove("on");
  ui.play.setAttribute("aria-label", "Speak");
  ui.playhead.hidden = true;
  ui.veil.hidden = true;
  for (const s of ui.words.children) s.classList.remove("now", "said");
  showStrip(null);
}

function speak() {
  renderBase();
  const u = out;
  if (!u) return;
  stop();
  const h = play(u);
  playing = h;
  ui.play.classList.add("on");
  ui.play.setAttribute("aria-label", "Stop");
  ui.playhead.hidden = false;
  ui.veil.hidden = false;
  const spans = [...ui.words.children];
  const frame = () => {
    if (playing !== h) return;
    const t = Math.max(0, h.clock());
    const x = Math.min(1, t / span(u));
    ui.playhead.style.transform = `translateX(${x * ui.scope.clientWidth}px)`;
    ui.veil.style.left = `${x * 100}%`;
    u.words.forEach((wd, i) => {
      spans[i]?.classList.toggle("now", t >= wd.start && t < wd.end);
      spans[i]?.classList.toggle("said", t >= wd.end);
    });
    showStrip(sidepulseAt(u.sidepulse, t * 1000));
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  h.ended.then(() => playing === h && idle());
}

function toggle() {
  if (playing) {
    stop();
    idle();
  } else {
    speak();
  }
}

// ── Downloads ──

function slug() {
  return ui.text.value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "speak";
}

function save(blob, ext) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${slug()}.${ext}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

let lame = null;
function loadLame() {
  lame ??= new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/lamejs/1.2.1/lame.min.js";
    s.onload = () => res(window.lamejs);
    s.onerror = () => {
      lame = null;
      rej(new Error("The MP3 encoder did not load"));
    };
    document.head.append(s);
  });
  return lame;
}

async function mp3(u) {
  const lj = await loadLame();
  const enc = new lj.Mp3Encoder(1, u.rate, 64);
  const pcm = Int16Array.from(u.samples, (v) => Math.max(-1, Math.min(1, v)) * 32767);
  const chunks = [];
  for (let i = 0; i < pcm.length; i += 1152) {
    const b = enc.encodeBuffer(pcm.subarray(i, i + 1152));
    if (b.length) chunks.push(new Uint8Array(b));
  }
  const end = enc.flush();
  if (end.length) chunks.push(new Uint8Array(end));
  return new Blob(chunks, { type: "audio/mpeg" });
}

ui.saveWav.addEventListener("click", () => out && save(wav(out), "wav"));
ui.saveMp3.addEventListener("click", async () => {
  if (!out) return;
  ui.saveMp3.disabled = true;
  try {
    save(await mp3(out), "mp3");
  } catch (e) {
    status(false, e.message);
  } finally {
    ui.saveMp3.disabled = false;
  }
});
ui.savePho.addEventListener("click", () => base && save(new Blob([`${phoLines().join("\n")}\n`], { type: "text/plain" }), "pho"));

// ── Wiring ──

ui.play.addEventListener("click", toggle);
ui.text.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    toggle();
  }
});
// One line: pasted line breaks become spaces, in what is heard and in the command.
ui.text.addEventListener("input", () => {
  if (/\n/.test(ui.text.value)) {
    const at = ui.text.selectionStart;
    ui.text.value = ui.text.value.replace(/\s*\n\s*/g, " ");
    ui.text.setSelectionRange(at, at);
  }
  onFrame("base", renderBase);
});
ui.melody.addEventListener("input", () => {
  try {
    setMelody(ui.melody.value, { field: false });
    ui.melody.classList.remove("bad");
    onFrame("base", renderBase);
  } catch (e) {
    ui.melody.classList.add("bad");
    status(false, e.message);
  }
});
ui.mode.addEventListener("click", (e) => {
  const m = e.target.closest("button")?.dataset.mode;
  if (!m || m === mode) return;
  setMode(m);
  renderBase();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && playing) toggle();
});
// Redraw whenever the spectrogram area changes size: the window, or the command growing a line.
let drawn = "";
new ResizeObserver(([entry]) => {
  const { width, height } = entry.contentRect;
  const size = `${Math.round(width)}x${Math.round(height)}`;
  if (size === drawn || !out) return;
  drawn = size;
  onFrame("resize", () => {
    drawSpec();
    drawOverlay();
  });
}).observe(ui.scope);

function presets() {
  const make = (label, apply) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset";
    b.textContent = label;
    b.addEventListener("click", () => {
      apply();
      renderBase();
    });
    return b;
  };
  ui.presets.replaceChildren(
    make("From the words", () => setMelody(melodyFromWords())),
    ...engine.tunes.map((t) =>
      make(t.name, () => {
        ui.text.value = t.lyrics;
        setMelody(t.melody);
        setRuler(t.tempo);
      }),
    ),
  );
}

/** Loads a `speak …` command from the URL: speech, or singing with any melody or tune. */
function loadHash() {
  const hash = decodeURIComponent(location.hash.slice(1));
  if (!hash.startsWith("speak")) return;
  try {
    const s = parse(hash);
    const tune = engine.tunes.find((t) => t.name === s.sing);
    if (s.voice) setVoice(s.voice);
    ui.text.value = s.text || tune?.lyrics || ui.text.value;
    if (s.melody || tune) {
      setMelody(s.melody || tune.melody);
      setMode("sing");
      setRuler(Number(s.tempo) || tune?.tempo || 120);
    } else if (s.rate) {
      setRuler(s.rate);
    }
  } catch {}
}

showStrip(null);
setRuler();

load()
  .then((e) => {
    engine = e;
    ui.voices.replaceChildren(
      ...e.voices.map((v) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip";
        b.setAttribute("role", "radio");
        b.dataset.voice = v.name;
        b.title = v.description;
        b.textContent = v.name.replace(/^klatt-/, "");
        b.addEventListener("click", () => {
          setVoice(v.name);
          renderBase();
        });
        return b;
      }),
    );
    setVoice(e.defaultVoice);
    presets();
    loadHash();
    ui.play.disabled = false;
    renderBase();
    ui.text.focus();
    ui.text.setSelectionRange(ui.text.value.length, ui.text.value.length);
  })
  .catch((err) => status(false, `The engine did not load: ${err.message || err}`));
