// Voice AI Agent — minimal reference dashboard (VA-51..VA-56).
//
// Demonstrates the client half of the design: capture mic audio (VA-51), select the endpoint
// per feature (VA-52), consume the SSE stream and render live captions (VA-53), play streamed
// audio (VA-54), barge-in (VA-55), and a debug panel (VA-56). Vanilla JS, no build step.
//
// The endpoint URL is the only selector — there is no routing field (VA-20/21). Streaming
// endpoints return Server-Sent Events over POST, so we read them with fetch + a stream reader
// (native EventSource is GET-only).

const API_PREFIX = "/api/v1";
const $ = (id) => document.getElementById(id);

const state = {
  playing: false,
  audioCtx: null,
  playHead: 0,
  chunks: 0,
  sessionId: null,
};

function setState(name) {
  $("dbg-state").textContent = name;
}

function resetTurnUi(endpoint) {
  $("transcript").textContent = "";
  $("answer").textContent = "";
  $("dbg-endpoint").textContent = endpoint;
  $("dbg-audio").textContent = "0";
  $("dbg-latency").textContent = "—";
  state.chunks = 0;
}

// --- audio playback (raw PCM16 @ 24 kHz, matching the backend output_format) --------------

function audioContext() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    state.playHead = state.audioCtx.currentTime;
  }
  return state.audioCtx;
}

function playPcmChunk(base64) {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const samples = new Int16Array(bytes.buffer);
  const ctx = audioContext();
  const buffer = ctx.createBuffer(1, samples.length, 24000);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  const startAt = Math.max(ctx.currentTime, state.playHead);
  src.start(startAt);
  state.playHead = startAt + buffer.duration;
  state.playing = true;
  $("stop").disabled = false;
}

function stopAudio() {
  // Barge-in / stop: drop the queue by recreating the context (VA-55).
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  state.playing = false;
  $("stop").disabled = true;
}

// --- SSE over fetch -----------------------------------------------------------------------

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = {};
      for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) event.name = line.slice(7).trim();
        else if (line.startsWith("data: ")) event.data = line.slice(6);
      }
      if (event.name) onEvent(event.name, event.data ? JSON.parse(event.data) : {});
    }
  }
}

function handleEvent(name, data) {
  if (name === "transcript.partial") $("transcript").textContent = data.text + " …";
  else if (name === "transcript.final") $("transcript").textContent = data.text;
  else if (name === "answer.delta") $("answer").textContent += data.text;
  else if (name === "audio.chunk") {
    state.chunks += 1;
    $("dbg-audio").textContent = String(state.chunks);
    setState("speaking");
    playPcmChunk(data.audio_b64);
  } else if (name === "done") {
    setState("idle");
    if (data.session_id) {
      state.sessionId = data.session_id;
      $("dbg-session").textContent = data.session_id;
    }
    $("dbg-latency").textContent = JSON.stringify(data.latency_ms || {});
  }
}

// --- turn dispatch ------------------------------------------------------------------------

async function sendTurn(input) {
  const endpoint = $("endpoint").value;
  resetTurnUi(endpoint);
  stopAudio(); // barge-in: a new turn cancels any in-flight playback
  setState("listening");
  const body = JSON.stringify({ session_id: state.sessionId, input });
  const resp = await fetch(`${API_PREFIX}/voice/${endpoint}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });
  if (!resp.ok) {
    $("answer").textContent = `Error ${resp.status}`;
    setState("idle");
    return;
  }
  setState("thinking");
  if (endpoint === "complete") {
    const result = await resp.json();
    $("transcript").textContent = result.transcript || "";
    $("answer").textContent = result.answer_text || "";
    state.sessionId = result.session_id;
    $("dbg-session").textContent = result.session_id || "—";
    $("dbg-latency").textContent = JSON.stringify(result.latency_ms || {});
    setState("idle");
  } else {
    await readSse(resp, handleEvent);
  }
}

// --- mic capture (VA-51) ------------------------------------------------------------------

let mediaRecorder = null;
let recordedChunks = [];

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stopAudio(); // barge-in
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    mediaRecorder.ondataavailable = (e) => e.data.size && recordedChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(recordedChunks, { type: "audio/webm;codecs=opus" });
      const b64 = await blobToBase64(blob);
      await sendTurn({ kind: "audio", audio_b64: b64 });
    };
    mediaRecorder.start();
    setState("listening");
  } catch (err) {
    $("answer").textContent = "Microphone permission denied.";
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
}

function blobToBase64(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.readAsDataURL(blob);
  });
}

// --- wiring -------------------------------------------------------------------------------

$("send").addEventListener("click", () => {
  const text = $("text").value.trim();
  if (text) sendTurn({ kind: "text", text });
});
$("mic").addEventListener("mousedown", startRecording);
$("mic").addEventListener("mouseup", stopRecording);
$("mic").addEventListener("mouseleave", stopRecording);
$("stop").addEventListener("click", stopAudio);
