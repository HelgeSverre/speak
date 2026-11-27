// Voices: every voice and tune as a row with a spectrogram rendered live from the engine.

import { blank, command, commandText, drawSpectrogram, load, play, stop, synthesize } from "./engine.js";

const $ = (s) => document.querySelector(s);
let engine;
let active = null;

/** A row: name, description, and a spectrogram of `state()`; click to play it. */
function row(list, { name, desc, state }) {
  const li = document.createElement("li");
  li.className = "row";
  li.tabIndex = 0;
  li.innerHTML = `<div class="info"><span class="name"></span><span class="desc"></span><a class="open">open in editor</a></div>
    <div class="wave"><canvas></canvas><div class="playhead"></div></div>`;
  li.querySelector(".name").textContent = name;
  li.querySelector(".desc").textContent = desc;
  list.append(li);
  const canvas = li.querySelector("canvas");
  const head = li.querySelector(".playhead");
  const link = li.querySelector(".open");
  const r = { li, u: null, key: "", error: null };

  r.render = () => {
    const s = state();
    const key = JSON.stringify(s);
    if (key === r.key) return;
    r.key = key;
    link.href = `./#${encodeURIComponent(commandText(command(s, engine.defaultVoice)))}`;
    li.querySelector(".err")?.remove();
    try {
      r.u = synthesize(s);
      drawSpectrogram(canvas, r.u.samples);
    } catch (e) {
      r.u = null;
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      const p = document.createElement("p");
      p.className = "err";
      p.textContent = e.message;
      li.querySelector(".wave").append(p);
    }
  };

  const toggle = () => {
    if (active === r) {
      stop();
      return;
    }
    r.render();
    if (!r.u) return;
    active?.li.classList.remove("playing");
    const h = play(r.u);
    active = r;
    li.classList.add("playing");
    const frame = () => {
      if (active !== r) return;
      const x = Math.min(1, Math.max(0, h.clock()) / h.duration);
      head.style.transform = `translateX(${x * canvas.clientWidth}px)`;
      requestAnimationFrame(frame);
    };
    frame();
    h.ended.then(() => {
      if (active !== r) return;
      active = null;
      li.classList.remove("playing");
    });
  };
  li.addEventListener("click", (e) => {
    if (!e.target.closest("a")) toggle();
  });
  li.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggle();
    }
  });
  return r;
}

/** Renders dirty rows one per frame so typing stays smooth. */
function renderer(rows) {
  let queue = [];
  let raf = 0;
  const step = () => {
    const r = queue.shift();
    r?.render();
    raf = queue.length ? requestAnimationFrame(step) : 0;
  };
  return (subset = rows) => {
    queue = [...new Set([...subset, ...queue])];
    if (!raf) raf = requestAnimationFrame(step);
  };
}

function voices() {
  const sentence = $("#sentence");
  const rows = engine.voices.map((v) =>
    row($("#voice-rows"), {
      name: v.name,
      desc: v.description,
      state: () => Object.assign(blank(), { voice: v.name, text: sentence.value }),
    }),
  );
  const render = renderer(rows);
  sentence.addEventListener("input", () => render());
  render();
  return rows;
}

function sing() {
  const f = { melody: $("#melody"), lyrics: $("#lyrics"), voice: $("#sing-voice"), tempo: $("#tempo"), key: $("#key") };
  for (const v of engine.voices) f.voice.append(new Option(v.name, v.name));
  f.voice.value = engine.defaultVoice;
  const voiceFor = { twinkle: "klatt-pip", ode: "klatt-calm" };
  const mine = row($("#sing-rows"), {
    name: "Your melody",
    desc: "Notes and words above",
    state: () =>
      Object.assign(blank(), {
        melody: f.melody.value.trim(),
        text: f.lyrics.value,
        voice: f.voice.value,
        tempo: f.tempo.value,
        key: f.key.value === "0" ? "" : f.key.value,
      }),
  });
  const tunes = engine.tunes.map((t) =>
    row($("#sing-rows"), {
      name: t.name,
      desc: t.lyrics.split(/[.!?]/)[0],
      state: () => Object.assign(blank(), { sing: t.name, voice: voiceFor[t.name] ?? "" }),
    }),
  );
  const render = renderer([mine, ...tunes]);
  $("#sing-form").addEventListener("input", () => render([mine]));
  $("#sing-form").addEventListener("change", () => render([mine]));
  render();
  return [mine, ...tunes];
}

load().then((e) => {
  engine = e;
  const all = [...voices(), ...sing()];
  let t = 0;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => all.forEach((r) => r.u && drawSpectrogram(r.li.querySelector("canvas"), r.u.samples)), 150);
  });
  document.addEventListener("keydown", (ev) => ev.key === "Escape" && stop());
});
