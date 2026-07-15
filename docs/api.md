# API reference

The service exposes its two pipelines as HTTP endpoints. There is **no routing field**: the
client selects a pipeline purely by *which endpoint URL it calls*. All application endpoints
live under the configurable API prefix (default `/api/v1`); the liveness probe and root live at
the top level.

Interactive docs are always available: **Swagger UI at `/docs`**, the raw schema at
`/openapi.json`.

- [Authentication](#authentication)
- [Voice endpoints](#voice-endpoints)
- [Request body](#request-body)
- [Streaming responses (SSE)](#streaming-responses-sse)
- [Complete response (JSON)](#complete-response-json)
- [Contract endpoints](#contract-endpoints)
- [Operations endpoints](#operations-endpoints)
- [Errors](#errors)

---

## Authentication

Bearer-JWT auth (VA-15) is **enabled when `JWT_SECRET_KEY` is configured** — always the case
in `dev`/`prod`, optional locally. When on, every request under the API prefix requires:

```
Authorization: Bearer <jwt>
```

The token must be **HS256**-signed with the shared secret and carry three claims: `sub`
(caller identity), `tenant` (scopes sessions and usage metering), and `exp`. Anything else —
a missing/expired/badly-signed token, a missing claim, or a non-HS256 algorithm — returns a
problem-shaped **`401`** with a `WWW-Authenticate: Bearer` header.

`/healthz`, `/`, the docs (`/docs`, `/openapi.json`), and the `/ui` dashboard stay public.

```python
import jwt, time
token = jwt.encode(
    {"sub": "client-1", "tenant": "acme", "exp": int(time.time()) + 3600},
    SECRET, algorithm="HS256",
)
```

---

## Voice endpoints

| Method & path | Pipeline | Delivery | Input | Use |
| --- | --- | --- | --- | --- |
| `POST /api/v1/voice/fast` | realtime | SSE | audio | Low-latency voice-to-voice |
| `POST /api/v1/voice/slow` | traditional | SSE | text or audio | Document-grounded, tool-capable turn |
| `POST /api/v1/voice/complete` | traditional | JSON | text or audio | Same as slow, returned as one payload |
| `POST /api/v1/voice/stream` | traditional | SSE | text or audio | Traditional turn as an SSE stream |

`fast` is voice-to-voice and therefore requires **audio** input. The traditional endpoints
accept either text or audio (audio is transcribed first).

### Example — streamed traditional turn

```bash
curl -N -X POST http://localhost:8080/api/v1/voice/slow \
  -H 'content-type: application/json' \
  -d '{"input": {"kind": "text", "text": "What does Article 21 guarantee?"}}'
```

### Example — complete (JSON) turn

```bash
curl -X POST http://localhost:8080/api/v1/voice/complete \
  -H 'content-type: application/json' \
  -d '{"session_id": "sess-1", "input": {"kind": "text", "text": "What does Article 21 guarantee?"}}'
```

---

## Request body

Every voice endpoint accepts the **same** body (`VoiceTurnRequest`). Unknown fields are
rejected (`extra="forbid"`) — notably any `architecture`/`pipeline`/`mode` routing hint, since
the URL is the only selector.

```jsonc
{
  "session_id": "sess-1",          // optional; omit to start a new session
  "input": { /* one of the two variants below */ }
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `session_id` | string \| null | no | Conversation id for continuity; omit to start a new session |
| `input` | object | yes | Discriminated on `kind` |

**Text input**

```json
{ "kind": "text", "text": "your question" }
```

**Audio input** (Opus in a WebM container, base64-encoded)

```json
{ "kind": "audio", "audio_b64": "<base64>", "mime": "audio/webm;codecs=opus" }
```

`text` must be non-empty; `mime` defaults to `audio/webm;codecs=opus`.

---

## Streaming responses (SSE)

`fast`, `slow` and `stream` return `text/event-stream`. Each event is an `event:` line, a JSON
`data:` line, and a blank separator. Events arrive in this order:

```
transcript.partial*  →  transcript.final  →  answer.delta*  →  audio.chunk*  →  done
```

The realtime (`fast`) path is voice-to-voice, so it emits only `audio.chunk*` then `done`.

| Event | Payload | Meaning |
| --- | --- | --- |
| `transcript.partial` | `{ "text": string }` | Interim transcript (may repeat) |
| `transcript.final` | `{ "text": string }` | Final transcript of the utterance |
| `answer.delta` | `{ "text": string }` | A chunk of the answer text (repeats) |
| `audio.chunk` | `{ "audio_b64": string, "seq": int }` | A chunk of synthesized audio; `seq` is monotonic from 0 |
| `done` | `{ "session_id": string\|null, "latency_ms": { stage: ms } }` | Terminates the stream |

Example wire output:

```
event: transcript.final
data: {"event":"transcript.final","text":"What does Article 21 guarantee?"}

event: answer.delta
data: {"event":"answer.delta","text":"Article 21 protects "}

event: audio.chunk
data: {"event":"audio.chunk","audio_b64":"...","seq":0}

event: done
data: {"event":"done","session_id":"sess-1","latency_ms":{"stt_ms":180.0,"llm_ms":420.0,"first_audio_ms":540.0}}
```

---

## Complete response (JSON)

`POST /api/v1/voice/complete` returns one `VoiceTurnResult`:

```jsonc
{
  "session_id": "sess-1",
  "transcript": "What does Article 21 guarantee?",
  "answer_text": "Article 21 protects the right to life and personal liberty …",
  "audio_url": null,           // spoken answer is streamed as audio.chunk on the SSE paths
  "tools_called": [],
  "latency_ms": { "stt_ms": 180.0, "llm_ms": 420.0, "first_audio_ms": 540.0 }
}
```

---

## Contract endpoints

Documentation/validation helpers — they do **not** run a turn. They publish the shared schemas
into `/openapi.json` and let a client verify a request body.

| Method & path | Purpose |
| --- | --- |
| `POST /api/v1/contract/validate` | Validate a `VoiceTurnRequest` and echo it back; `422` if invalid |
| `GET /api/v1/contract/schema` | Example request + one of every SSE event, in emission order |

---

## Operations endpoints

| Method & path | Returns |
| --- | --- |
| `GET /healthz` | `{"ok": true}` — dependency-free liveness/readiness probe |
| `GET /` | `{"service", "environment"}` |
| `GET /api/v1/config` | Effective configuration, secrets redacted |
| `GET /api/v1/metrics` | Per-path latency aggregates `{path: {stage: {count, p50, p95, max}}}` |
| `GET /api/v1/usage` | Per-path, per-tenant `{tokens, audio_seconds, turns}` |
| `GET /api/v1/counters` | Per-path `{turns, errors, fallbacks, error_rate, fallback_rate}` |

`/healthz` never calls a downstream provider, so it cannot fail on an upstream outage.

---

## Errors

Every error returns a consistent `Problem` body (loosely RFC 7807) — never a stack trace — and
echoes an `X-Request-ID` header.

```json
{
  "status": 422,
  "title": "Validation error",
  "detail": null,
  "correlation_id": "b22ef4802514473ca0a62ad9845642af",
  "errors": [{ "loc": ["body", "input"], "msg": "Field required", "type": "missing" }]
}
```

| Field | Notes |
| --- | --- |
| `status` | HTTP status code |
| `title` | Short summary of the error type |
| `detail` | Human-readable explanation (nullable) |
| `correlation_id` | Also returned as `X-Request-ID`; appears in the structured logs |
| `errors` | Field-level validation details, when applicable |

Common statuses: **422** (invalid body — missing `input`, a routing field, empty text),
**405** (wrong method), **500** (internal error). Send an inbound `X-Request-ID` to correlate
your request with the server logs.
