// Runs the live previews on the Verbatica page and points the download buttons at the latest release.
"use strict";

const REPO = "TariqMK/Verbatica";
const ASSET = { windows: /windows-x64\.exe$/i, macos: /macos-arm64\.zip$/i, linux: /linux-x64\.tar\.gz$/i };
const OS_NAME = { windows: "Windows", macos: "macOS", linux: "Linux" };

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const C = {
  ink: "#ededed", done: "#a1a1a6", todo: "rgba(237, 237, 237, 0.16)", caret: "#2a82da",
};
const ICON = {
  file: '<path d="M4 2h5l3 3v8.5c0 .3-.2.5-.5.5h-7a.5.5 0 0 1-.5-.5z"/><path d="M9 2v3h3"/>',
  check: '<path d="M3.5 8.5l3 3 6-7"/>',
  reveal: '<path d="M7 3H3.5C2.7 3 2 3.7 2 4.5v8c0 .8.7 1.5 1.5 1.5h8c.8 0 1.5-.7 1.5-1.5V9"/><path d="M9.5 2H14v4.5M14 2L7.5 8.5"/>',
};
const svg = (name) => `<svg viewBox="0 0 16 16" aria-hidden="true">${ICON[name]}</svg>`;

// ---- Waveforms ------------------------------------------------------------------------------

// Small seeded random generator, so every visit shows the same recordings.
function rng(seed) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

// Speech-like loudness: words and pauses, with stretches of silence that stay flat.
function makePeaks(seed, count, silences = []) {
  const r = rng(seed);
  const out = [];
  let level = 0.6;
  for (let i = 0; i < count; i++) {
    const x = i / count;
    if (silences.some(([a, b]) => x >= a && x < b)) { out.push(0); continue; }
    if (r() < 0.06) level = 0.25 + r() * 0.75;
    const v = level * (0.45 + r() * 0.55);
    out.push(r() < 0.05 ? 0.12 : v);
  }
  return out;
}

// Draws like the app's Waveform widget: 2 px bars on a 4 px step, the caret at the point reached.
function drawWave(canvas, peaks, fraction, state) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const mid = h / 2;

  if (!peaks) {
    g.fillStyle = C.todo;
    g.fillRect(0, mid - 0.5, w, 1);
    return;
  }
  const step = 4, n = Math.floor(w / step);
  const cut = fraction * n;
  for (let i = 0; i < n; i++) {
    const a = Math.floor((i * peaks.length) / n), b = Math.max(Math.floor(((i + 1) * peaks.length) / n), a + 1);
    let v = 0;
    for (let j = a; j < b; j++) v = Math.max(v, peaks[j]);
    const bh = Math.max(2, v * h);
    g.fillStyle = i < cut ? (state === "done" ? C.done : C.ink) : C.todo;
    roundRect(g, i * step, mid - bh / 2, 2, bh, 1);
  }
  if (state === "active") {
    g.fillStyle = C.caret;
    roundRect(g, Math.min(cut * step, w - 2), 0, 2, h, 1);
  }
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h);
  g.fill();
}

// ---- Hero: the transcription queue ----------------------------------------------------------

const QUEUE = [
  { name: "interview-maria-chen.m4a", length: "48:12", words: 2964, seed: 11, silences: [[0.24, 0.25]] },
  { name: "team-standup-monday.mp3", length: "12:04", words: 811, seed: 23, silences: [[0.32, 0.4]] },
  { name: "lecture-03-thermodynamics.mp4", length: "1:31:40", words: 9327, seed: 37, silences: [[0.42, 0.5], [0.81, 0.84]] },
  { name: "podcast-episode-12.wav", length: "54:37", words: 6418, seed: 53, silences: [[0.18, 0.22], [0.66, 0.7]] },
];

const queue = {
  rows: [],
  build() {
    const host = document.getElementById("queue");
    if (!host) return;
    host.innerHTML = "";
    this.rows = QUEUE.map((f) => {
      const el = document.createElement("div");
      el.className = "row";
      el.innerHTML =
        `<span class="ico"></span>` +
        `<span class="name"><b></b><span></span></span>` +
        `<canvas height="28"></canvas>` +
        `<span class="stat"></span>` +
        `<span class="rev">${svg("reveal")}</span>`;
      el.querySelector("b").textContent = f.name;
      host.appendChild(el);
      return { f, el, peaks: makePeaks(f.seed, 240, f.silences), state: "waiting", fraction: 0, hold: 0 };
    });
    this.reset();
  },
  reset() {
    const [a, b, c, d] = this.rows;
    if (!a) return;
    Object.assign(a, { state: "done", fraction: 1 });
    Object.assign(b, { state: "done", fraction: 1 });
    Object.assign(c, { state: "active", fraction: 0.48 });
    Object.assign(d, { state: "waiting", fraction: 0 });
    this.finishedAt = 0;
    this.render();
  },
  current() { return this.rows.find((r) => r.state === "active"); },
  tick(now) {
    const r = this.current();
    if (!r) {
      if (!this.finishedAt) this.finishedAt = now;
      if (now - this.finishedAt > 5000) this.reset();
      return;
    }
    if (r.hold > 0) { r.hold--; return; }
    let next = r.fraction + 0.012 + Math.random() * 0.01;
    // Skipped silence: the caret jumps over the flat stretch.
    const s = r.f.silences.find(([a, b]) => next >= a && next < b);
    if (s) next = s[1] + 0.004;
    if (next >= 1) {
      r.state = "done";
      r.fraction = 1;
      const following = this.rows[this.rows.indexOf(r) + 1];
      if (following) { following.state = "active"; following.fraction = 0; following.hold = 6; }
    } else {
      r.fraction = next;
    }
    this.render();
  },
  render() {
    const done = this.rows.filter((r) => r.state === "done").length;
    const active = this.current();
    this.rows.forEach((r) => {
      const ico = r.el.querySelector(".ico");
      const want = r.state === "done" ? "check" : "file";
      if (ico.dataset.icon !== want) {
        ico.innerHTML = svg(want);
        ico.dataset.icon = want;
        ico.classList.toggle("ok", want === "check");
      }
      r.el.classList.toggle("active", r.state === "active");
      r.el.querySelector(".name span").textContent = r.state === "waiting" ? "Waiting" : r.f.length;
      r.el.querySelector(".stat").textContent =
        r.state === "done" ? `${r.f.words.toLocaleString("en-US")} words`
        : r.state === "active" ? `${Math.floor(r.fraction * 100)}%` : "";
      r.el.querySelector(".rev").classList.toggle("on", r.state === "done");
      const showPeaks = r.state !== "waiting" && !(r.state === "active" && r.hold > 0);
      drawWave(r.el.querySelector("canvas"), showPeaks ? r.peaks : null, r.fraction, r.state);
    });
    setText("q-count", `${this.rows.length} files · ${done} done`);
    setText("file-of", active ? `File ${this.rows.indexOf(active) + 1} of ${this.rows.length}` : "All done");
    const bar = document.getElementById("total-bar");
    if (bar) bar.style.width = `${((done + (active ? active.fraction : 0)) / this.rows.length) * 100}%`;
    const stop = document.querySelector(".app-foot .abtn");
    if (stop) stop.classList.toggle("dim", !active);
  },
};

function setText(id, text) {
  const el = document.getElementById(id);
  if (el && el.textContent !== text) el.textContent = text;
}

// ---- Output formats: the same transcript as Text, SRT and VTT -------------------------------

const SEGMENTS = [
  [0.0, 4.2, "Thanks for making the time today."],
  [4.2, 9.8, "Of course. I've wanted to talk about the river survey for a while."],
  [9.8, 13.6, "Let's start there. How did the project begin?"],
  [13.6, 21.1, "With one water sample that didn't match the records from the year before."],
  [21.1, 24.9, "And that single sample changed the whole plan?"],
];

function stamp(sec, mark) {
  let ms = Math.round(sec * 1000);
  const h = Math.floor(ms / 3600000); ms -= h * 3600000;
  const m = Math.floor(ms / 60000); ms -= m * 60000;
  const s = Math.floor(ms / 1000); ms -= s * 1000;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(s)}${mark}${p(ms, 3)}`;
}

const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const CARET = '<span class="typing-caret"></span>';

const outputs = {
  seg: 0, chars: 0, hold: 0,
  render() {
    const txt = document.getElementById("out-txt");
    const srt = document.getElementById("out-srt");
    const vtt = document.getElementById("out-vtt");
    if (!txt) return;
    const parts = [];
    for (let i = 0; i <= Math.min(this.seg, SEGMENTS.length - 1); i++) {
      const [a, b, text] = SEGMENTS[i];
      const shown = i < this.seg ? text : text.slice(0, this.chars);
      parts.push({ i, a, b, shown, typing: i === this.seg });
    }
    const caret = (p) => (p.typing ? CARET : "");
    txt.innerHTML = parts.map((p) => esc(p.shown) + caret(p)).join(" ");
    srt.innerHTML = parts
      .map((p) => `<span class="t">${p.i + 1}\n${stamp(p.a, ",")} --> ${stamp(p.b, ",")}</span>\n${esc(p.shown)}${caret(p)}`)
      .join("\n\n");
    vtt.innerHTML = '<span class="t">WEBVTT</span>\n\n' + parts
      .map((p) => `<span class="t">${stamp(p.a, ".")} --> ${stamp(p.b, ".")}</span>\n${esc(p.shown)}${caret(p)}`)
      .join("\n\n");
  },
  tick() {
    if (this.hold > 0) { this.hold--; return; }
    if (this.seg >= SEGMENTS.length) { this.seg = 0; this.chars = 0; this.render(); return; }
    const text = SEGMENTS[this.seg][2];
    this.chars = Math.min(text.length, this.chars + 2 + Math.floor(Math.random() * 3));
    if (this.chars >= text.length) {
      this.seg++;
      this.chars = 0;
      this.hold = this.seg >= SEGMENTS.length ? 40 : 6;
    }
    this.render();
  },
  still() { this.seg = 3; this.chars = SEGMENTS[3][2].length; this.render(); },
};

// ---- Silence skipping -------------------------------------------------------------------------

const silence = {
  length: 80, // seconds
  gap: [0.43, 0.62],
  pos: 0, hold: 0,
  peaks: null,
  render() {
    const c = document.getElementById("silence-wave");
    if (!c) return;
    if (!this.peaks) this.peaks = makePeaks(71, 200, [this.gap]);
    drawWave(c, this.peaks, this.pos, "active");
    const t = (f) => { const s = Math.round(f * this.length); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
    setText("silence-time", `${t(this.pos)} / ${t(1)}`);
    const state = document.getElementById("silence-state");
    const inGap = this.pos >= this.gap[0] && this.pos < this.gap[1];
    state.classList.toggle("skip", inGap);
    setText("silence-state", inGap ? `Skipping silence ${t(this.gap[0])}–${t(this.gap[1])}` : this.pos >= 1 ? "Done" : "Transcribing");
  },
  tick() {
    if (this.hold > 0) { this.hold--; return; }
    if (this.pos >= 1) { this.pos = 0; this.render(); return; }
    const inGap = this.pos >= this.gap[0] && this.pos < this.gap[1];
    this.pos = Math.min(1, this.pos + (inGap ? 0.02 : 0.004 + Math.random() * 0.003));
    if (this.pos >= 1) this.hold = 25;
    this.render();
  },
  still() { this.pos = 0.5; this.render(); },
};

// ---- Verified model download ---------------------------------------------------------------

const MODEL_FILES = [
  { name: "config.json", size: "2.4 KB", hash: "b55496ac7940", weight: 1 },
  { name: "model.bin", size: "461 MB", hash: "3e305921506d", weight: 60 },
  { name: "tokenizer.json", size: "2.2 MB", hash: "fb7b63191e9b", weight: 3 },
  { name: "vocabulary.txt", size: "449 KB", hash: "34ce3fe1c504", weight: 2 },
];

const download = {
  rows: [], index: 0, hold: 0,
  build() {
    const host = document.getElementById("dl");
    if (!host) return;
    this.rows = MODEL_FILES.map((f) => {
      const el = document.createElement("div");
      el.className = "dl-row";
      el.innerHTML = `<span class="f"></span><span class="s"></span><span class="v">${svg("check")}</span>` +
        `<span class="bar"><i></i></span><span class="h"></span>`;
      el.querySelector(".f").textContent = f.name;
      el.querySelector(".s").textContent = f.size;
      host.appendChild(el);
      return { f, el, p: 0, ok: false };
    });
  },
  reset() { this.rows.forEach((r) => { r.p = 0; r.ok = false; }); this.index = 0; this.render(); },
  render() {
    this.rows.forEach((r) => {
      r.el.querySelector(".bar i").style.width = `${r.p * 100}%`;
      r.el.querySelector(".v").classList.toggle("on", r.ok);
      r.el.querySelector(".h").textContent = r.ok ? `SHA256 ${r.f.hash}… verified` : r.p >= 1 ? "Checking SHA256…" : "";
    });
  },
  tick() {
    if (this.hold > 0) { this.hold--; return; }
    const r = this.rows[this.index];
    if (!r) { this.reset(); return; }
    if (r.p < 1) {
      r.p = Math.min(1, r.p + (0.5 + Math.random() * 0.6) / (r.f.weight + 3));
    } else {
      r.ok = true;
      this.index++;
      if (this.index >= this.rows.length) this.hold = 40;
    }
    this.render();
  },
  still() { this.rows.forEach((r) => { r.p = 1; r.ok = true; }); this.render(); },
};

// ---- Run ------------------------------------------------------------------------------------

function start() {
  queue.build();
  download.build();
  outputs.render();
  silence.render();
  download.render();

  if (reduced) {
    // One still frame: the queue mid-file, the transcript part typed, the caret in the silence.
    Object.assign(queue.rows[2] || {}, { fraction: 0.62 });
    queue.render();
    outputs.still();
    silence.still();
    download.still();
    window.addEventListener("resize", () => { queue.render(); silence.render(); });
    return;
  }

  let ticks = 0;
  setInterval(() => {
    if (document.hidden) return;
    ticks++;
    const now = performance.now();
    if (ticks % 3 === 0) queue.tick(now);
    outputs.tick();
    silence.tick();
    download.tick();
  }, 100);
  window.addEventListener("resize", () => { queue.render(); silence.render(); });
}

// ---- Downloads ------------------------------------------------------------------------------

function detectOS() {
  const ua = navigator.userAgent;
  if (/iPhone|iPad|iPod|Android/i.test(ua)) return null;
  if (navigator.maxTouchPoints > 1 && /Macintosh/.test(ua)) return null; // iPadOS
  const platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || ua;
  if (/win/i.test(platform)) return "windows";
  if (/mac/i.test(platform)) return "macos";
  if (/linux|x11/i.test(platform)) return "linux";
  return null;
}

function wireDownloads() {
  const os = detectOS();
  const hero = document.getElementById("hero-download");
  if (os) {
    hero.textContent = `Download for ${OS_NAME[os]}`;
    const card = document.querySelector(`.card[data-os="${os}"]`);
    if (card) card.classList.add("match");
  } else {
    hero.setAttribute("href", "#download");
  }

  // Any failure (offline, rate limit, no release yet) leaves the Releases page links in place.
  fetch(`https://api.github.com/repos/${REPO}/releases/latest`, { headers: { Accept: "application/vnd.github+json" } })
    .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
    .then((rel) => {
      const tag = rel.tag_name;
      if (!tag) return;
      const date = new Date(rel.published_at).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
      setText("dl-title", `Download Verbatica ${tag}`);
      setText("dl-sub", `Released ${date}. Free and open source. One file per platform.`);
      setText("hero-version", `${tag} · Free`);
      for (const asset of rel.assets || []) {
        for (const [key, pattern] of Object.entries(ASSET)) {
          if (!pattern.test(asset.name)) continue;
          const card = document.querySelector(`.card[data-os="${key}"]`);
          if (card) {
            card.href = asset.browser_download_url;
            const size = card.querySelector("[data-size]");
            if (size) size.textContent += ` · ${Math.round(asset.size / 1048576)} MB`;
          }
          if (key === os) hero.href = asset.browser_download_url;
        }
      }
    })
    .catch(() => {});
}

document.addEventListener("DOMContentLoaded", () => {
  start();
  wireDownloads();
});
