# Frontend — reference dashboard

A minimal, dependency-free browser dashboard (vanilla JS, no build step) that demonstrates the
client half of the design. It captures mic audio and calls the voice endpoints directly — the
endpoint URL selects the pipeline; there is no smart router.

Covers: mic capture (VA-51), endpoint selection (VA-52), SSE live captions (VA-53), streamed
audio playback (VA-54), barge-in (VA-55), and a debug panel (VA-56).

## Run

The backend serves these assets at **`/ui`** when they're present:

```bash
uvicorn app.main:app --port 8080
# open http://127.0.0.1:8080/ui/
```

Serving same-origin avoids CORS. Text turns and the SSE stream work against the `mock`
providers out of the box; audio playback assumes the backend's PCM16 @ 24 kHz output, so real
spoken replies need the real providers configured (`DEEPGRAM_/GOOGLE_/CARTESIA_/OPENAI_*`).

## Files
- `index.html` — the dashboard UI
- `app.js` — mic capture, endpoint dispatch, SSE reader, PCM playback, barge-in, debug panel
- `styles.css` — styling

This is a **reference client**; a production frontend would live in its own repo/app.
