/* VANI — a voice-only interface. Speak in, hear the answer; no text is ever shown.
   Tap the mic → it listens (green, reacting to your voice) → auto-stops on silence →
   streams the answer from /voice/{slow|fast} and plays it (teal) → idle.
   Grounded uses /voice/slow, Realtime uses /voice/fast; both are SSE. Only audio.chunk
   events are used — transcript/answer text events are deliberately ignored. */
(() => {
  "use strict";
  const API = (window.API_BASE || location.origin + "/api/v1").replace(/\/$/, "");
  const ORIGIN = API.replace(/\/api\/v1$/, "");
  const $ = (id) => document.getElementById(id);
  const mic = $("mic");

  const SPEECH_RMS = 0.025;     // above this = voice present
  const SILENCE_MS = 1400;      // auto-stop after this much quiet (once you've spoken)
  const NO_SPEECH_MS = 7000;    // give up if nothing is said
  const MAX_LISTEN_MS = 20000;  // hard cap

  let mode = "slow";
  let state = "idle";           // idle | listening | thinking | speaking
  let recorder = null, chunks = [], micStream = null;
  let vadCtx = null, rafId = 0;

  function setState(s) {
    state = s;
    document.body.className = "is-" + s;
  }
  function flashError() {
    document.body.className = "is-error";
    setTimeout(() => { if (state === "idle") document.body.className = "is-idle"; }, 700);
    setTimeout(() => setState("idle"), 700);
  }

  // ---------- connection dot (visual only) ----------
  (async function ping() {
    try {
      const r = await fetch(`${ORIGIN}/healthz`, { cache: "no-store" });
      $("conn").classList.toggle("online", r.ok && (await r.json()).ok);
    } catch { /* stays offline-red */ }
  })();

  // ---------- controls ----------
  document.querySelectorAll(".mode-btn").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((x) => x.classList.remove("is-active"));
      b.classList.add("is-active"); mode = b.dataset.mode;
    })
  );
  document.querySelectorAll(".pill").forEach((p) =>
    p.addEventListener("click", () => { if (state === "idle") sendTurn({ kind: "text", text: p.dataset.q }, "slow"); })
  );

  // ---------- the one control: the mic ----------
  mic.addEventListener("click", () => {
    if (state === "idle") startListening();
    else if (state === "listening") stopListening(true);
    else if (state === "speaking") { stopPlayback(); startListening(); }   // barge-in
    // thinking: ignore (busy)
  });

  async function startListening() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch { flashError(); return; }
    chunks = [];
    recorder = new MediaRecorder(micStream, pickMime());
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.start();
    setState("listening");
    runVad(micStream);
  }

  function stopListening(send) {
    cancelAnimationFrame(rafId);
    document.documentElement.style.setProperty("--level", 0);
    if (vadCtx) { vadCtx.close().catch(() => {}); vadCtx = null; }
    if (recorder && recorder.state !== "inactive") {
      recorder.onstop = async () => {
        micStream && micStream.getTracks().forEach((t) => t.stop());
        if (!send || !chunks.length) { setState("idle"); return; }
        setState("thinking");
        const b64 = await blobToB64(new Blob(chunks, { type: "audio/webm" }));
        sendTurn({ kind: "audio", audio_b64: b64 }, mode);
      };
      recorder.stop();
    } else if (send) { setState("thinking"); }
  }

  // ---------- voice-activity detection: reactive level + silence auto-stop ----------
  function runVad(stream) {
    vadCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = vadCtx.createMediaStreamSource(stream);
    const analyser = vadCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    const started = performance.now();
    let lastVoice = 0, spoke = false, smooth = 0;

    (function loop() {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const x = (buf[i] - 128) / 128; sum += x * x; }
      const rms = Math.sqrt(sum / buf.length);
      smooth = smooth * 0.8 + rms * 0.2;
      document.documentElement.style.setProperty("--level", Math.min(1, smooth * 9).toFixed(3));

      const now = performance.now();
      if (rms > SPEECH_RMS) { spoke = true; lastVoice = now; }
      const elapsed = now - started;
      const quietFor = now - lastVoice;
      if ((spoke && quietFor > SILENCE_MS) || (!spoke && elapsed > NO_SPEECH_MS) || elapsed > MAX_LISTEN_MS) {
        stopListening(spoke);   // only send if the caller actually spoke
        return;
      }
      rafId = requestAnimationFrame(loop);
    })();
  }

  // ---------- one turn: stream audio back, play it, no text ----------
  async function sendTurn(input, ep) {
    setState("thinking");
    let played = false;
    try {
      const resp = await fetch(`${API}/voice/${ep}`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ input }),
      });
      if (!resp.ok || !resp.body) { flashError(); return; }
      const reader = resp.body.getReader(), dec = new TextDecoder();
      let sseBuf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        sseBuf += dec.decode(value, { stream: true });
        let i;
        while ((i = sseBuf.indexOf("\n\n")) >= 0) {
          const frame = sseBuf.slice(0, i); sseBuf = sseBuf.slice(i + 2);
          const ev = parseEvent(frame);
          if (ev.event === "audio.chunk" && ev.data.audio_b64) {
            if (!played) { played = true; setState("speaking"); }
            playPcm(ev.data.audio_b64);
          }
          // transcript.final / answer.delta / done: intentionally ignored — voice only
        }
      }
    } catch { flashError(); return; }
    finishWhenPlaybackEnds(played);
  }

  function parseEvent(frame) {
    let event = null, data = {};
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) { try { data = JSON.parse(line.slice(5).trim()); } catch {} }
    }
    return { event, data };
  }

  // ---------- PCM16 @ 24kHz streaming playback ----------
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
    const node = ctx.createBufferSource();
    node.buffer = buf; node.connect(ctx.destination);
    const t = Math.max(ctx.currentTime, playHead);
    node.start(t); playHead = t + buf.duration;
  }
  function stopPlayback() {
    if (ctx) { ctx.close().catch(() => {}); ctx = null; playHead = 0; }
  }
  function finishWhenPlaybackEnds(played) {
    if (!played || !ctx) { if (!played) flashError(); else setState("idle"); return; }
    const remainingMs = Math.max(0, (playHead - ctx.currentTime) * 1000);
    setTimeout(() => { stopPlayback(); setState("idle"); }, remainingMs + 150);
  }

  // ---------- helpers ----------
  function pickMime() {
    for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return { mimeType: m };
    }
    return {};
  }
  function blobToB64(blob) {
    return new Promise((res) => {
      const r = new FileReader();
      r.onloadend = () => res(String(r.result).split(",")[1] || "");
      r.readAsDataURL(blob);
    });
  }
})();
