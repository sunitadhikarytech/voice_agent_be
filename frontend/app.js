/* VANI frontend — talks to the Voice AI Agent backend.
   Text (pills) → grounded /voice/slow. Hold-to-talk → /voice/{slow|fast} with mic audio.
   Streams SSE (transcript.final → answer.delta → audio.chunk → done) and plays PCM16@24kHz. */
(() => {
  "use strict";
  const API = (window.API_BASE || "http://127.0.0.1:8080/api/v1").replace(/\/$/, "");
  const ORIGIN = API.replace(/\/api\/v1$/, "");

  const $ = (id) => document.getElementById(id);
  const hero = $("hero"), convo = $("convo"), thread = $("thread"), statusEl = $("status");
  const talk = $("talk"), talkLabel = $("talkLabel"), conn = $("conn"), scrollCue = $("scrollCue");
  let mode = "slow", busy = false;

  // ---------- connection indicator ----------
  (async function ping() {
    try {
      const r = await fetch(`${ORIGIN}/healthz`, { cache: "no-store" });
      const ok = r.ok && (await r.json()).ok;
      conn.textContent = ok ? "online" : "offline";
      conn.classList.toggle("online", !!ok);
    } catch {
      conn.textContent = "offline (start the backend on :8080)";
    }
  })();

  // ---------- mode toggle ----------
  document.querySelectorAll(".mode-btn").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((x) => x.classList.remove("is-active"));
      b.classList.add("is-active");
      mode = b.dataset.mode;
    })
  );

  // ---------- suggestion pills (text → grounded) ----------
  document.querySelectorAll(".pill").forEach((p) =>
    p.addEventListener("click", () => runText(p.dataset.q))
  );

  $("scrollCue").addEventListener("click", () =>
    convo.scrollIntoView({ behavior: "smooth" })
  );
  $("reset").addEventListener("click", () => {
    thread.innerHTML = "";
    convo.hidden = true;
    hero.scrollIntoView({ behavior: "smooth" });
  });

  // ---------- hold to talk ----------
  let recorder, chunks = [];
  const press = async (e) => {
    e.preventDefault();
    if (busy) { stopPlayback(); }            // barge-in: talking over the reply cuts it off
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream, pickMime());
      chunks = [];
      recorder.ondataavailable = (ev) => ev.data.size && chunks.push(ev.data);
      recorder.start();
      document.body.classList.add("is-listening");
      talkLabel.textContent = "LISTENING…";
    } catch {
      talkLabel.textContent = "MIC BLOCKED";
    }
  };
  const release = async (e) => {
    e.preventDefault();
    if (!recorder || recorder.state === "inactive") return;
    const done = new Promise((res) => (recorder.onstop = res));
    recorder.stop();
    await done;
    recorder.stream.getTracks().forEach((t) => t.stop());
    document.body.classList.remove("is-listening");
    talkLabel.innerHTML = "HOLD&nbsp;TO&nbsp;TALK";
    if (!chunks.length) return;
    const b64 = await blobToB64(new Blob(chunks, { type: "audio/webm" }));
    runTurn({ kind: "audio", audio_b64: b64 }, mode);
  };
  talk.addEventListener("pointerdown", press);
  talk.addEventListener("pointerup", release);
  talk.addEventListener("pointerleave", release);

  function pickMime() {
    for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return { mimeType: m };
    }
    return {};
  }

  // ---------- turns ----------
  function runText(text) { runTurn({ kind: "text", text }, "slow"); }

  async function runTurn(input, ep) {
    if (busy) return;
    busy = true;
    reveal();
    const youText = input.kind === "text" ? input.text : "…";
    const you = addTurn("you", "YOU", youText);
    const vani = addTurn("vani", "VANI", "");
    setStatus("thinking");
    try {
      const resp = await fetch(`${API}/voice/${ep}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ input }),
      });
      if (!resp.ok || !resp.body) {
        vani.q.textContent = `— error ${resp.status} (is the backend running + CORS allowing this origin?)`;
        return;
      }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "", answer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, i); buf = buf.slice(i + 2);
          const { event, data } = parseFrame(frame);
          if (!event) continue;
          if (event === "transcript.final" && input.kind === "audio") you.q.textContent = data.text;
          else if (event === "answer.delta") { answer += data.text; vani.q.textContent = answer; }
          else if (event === "audio.chunk") { setStatus("speaking"); playPcm(data.audio_b64); }
          else if (event === "done") { /* end */ }
        }
      }
      if (!answer && input.kind !== "audio") vani.q.textContent = "(no answer — check the LLM key)";
    } catch (err) {
      vani.q.textContent = "— network error reaching the backend";
    } finally {
      setStatus("idle");
      busy = false;
    }
  }

  function parseFrame(frame) {
    let event = null, data = {};
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) { try { data = JSON.parse(line.slice(5).trim()); } catch {} }
    }
    return { event, data };
  }

  // ---------- UI helpers ----------
  function reveal() {
    if (convo.hidden) { convo.hidden = false; scrollCue.classList.add("hidden"); }
    convo.scrollIntoView({ behavior: "smooth" });
  }
  function addTurn(kind, who, text) {
    const el = document.createElement("div");
    el.className = `turn ${kind}`;
    el.innerHTML = `<span class="who"></span><div class="bubble"></div>`;
    el.querySelector(".who").textContent = who;
    const q = el.querySelector(".bubble");
    q.textContent = text;
    thread.appendChild(el);
    el.scrollIntoView({ behavior: "smooth", block: "end" });
    return { el, q };
  }
  function setStatus(s) {
    statusEl.textContent = s;
    document.body.classList.toggle("is-speaking", s === "speaking");
  }

  // ---------- PCM16 @ 24kHz playback ----------
  let ctx = null, playHead = 0;
  function playPcm(b64) {
    if (!ctx) { ctx = new (window.AudioContext || window.webkitAudioContext)(); playHead = 0; }
    const bin = atob(b64), n = bin.length >> 1;
    if (!n) return;
    const buf = ctx.createBuffer(1, n, 24000), ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) {
      let s = (bin.charCodeAt(i * 2 + 1) << 8) | bin.charCodeAt(i * 2);
      if (s >= 32768) s -= 65536;
      ch[i] = s / 32768;
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const t = Math.max(ctx.currentTime, playHead);
    src.start(t);
    playHead = t + buf.duration;
  }
  function stopPlayback() {
    if (ctx) { ctx.close().catch(() => {}); ctx = null; playHead = 0; }
    document.body.classList.remove("is-speaking");
  }

  function blobToB64(blob) {
    return new Promise((res) => {
      const r = new FileReader();
      r.onloadend = () => res(String(r.result).split(",")[1] || "");
      r.readAsDataURL(blob);
    });
  }
})();
