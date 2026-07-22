# frontend/ — VANI browser client (served at `/ui`)

The **VANI** landing/console — ask the Constitution of India out loud (or tap a suggested
question) and get a spoken, grounded answer. Dependency-free static HTML/CSS/JS, no build step.

**Source of truth:** these files are maintained in their own repo,
[`sunitadhikarytech/FE`](https://github.com/sunitadhikarytech/FE), and **vendored here** so the
backend serves them **same-origin at `/ui`** — no CORS. `app.js` defaults `API_BASE` to
`location.origin + "/api/v1"`, so it targets whatever host serves it. To update: edit in the
`FE` repo, then copy `index.html` / `styles.css` / `app.js` here (a submodule/build step can
automate this later).

## Run

```bash
uvicorn app.main:app --port 8080
# open http://127.0.0.1:8080/ui/
```

Covers mic capture (VA-51), endpoint/mode selection (VA-52), SSE live captions (VA-53), streamed
audio playback (VA-54), and barge-in (VA-55). Self-contained — no external hosts/CDNs
(CSP-friendly, offline). Suggested pills run grounded text turns on `/voice/slow`; hold-to-talk
streams mic audio to `/voice/{slow,fast}` with live captions + PCM16@24kHz playback. Text/SSE
work against `mock` providers; real spoken answers need the real providers
(`DEEPGRAM_/GOOGLE_/CARTESIA_/OPENAI_*`).

## Files
- `index.html` — hero + console, inline-SVG mic, runtime config
- `app.js` — connection check, mic capture, SSE reader, PCM playback, barge-in
- `styles.css` — the aesthetic (cream canvas, serif wordmark, pills, mic)
