# Voice AI Agent

A dual-pipeline voice assistant served as a single FastAPI service.

- **Traditional (slow) path** — streaming STT → LLM → streaming TTS. Handles complex,
  document-grounded, tool-capable turns. The full source document is supplied to the model
  as context (no vector store / RAG).
- **Realtime (fast) path** — voice-to-voice over WebSocket for fast, natural conversation.

There is **no backend router**: the client chooses a pipeline purely by which endpoint it
calls. Providers and pipelines sit behind small adapter interfaces so they are swappable by
configuration rather than code.

The backend is functionally complete: the four voice endpoints, both pipelines, the provider
adapters, full-document grounding, tools, session/memory/state, and observability — all
exercised by an offline test suite. The cloud/CI infrastructure (GCP, deploy) is tracked in
its own tickets.

📖 **Docs:** [API reference](docs/api.md) · [Configuration](docs/configuration.md) ·
[Architecture & design](docs/architecture.md)

## Project layout

```
app/
  main.py              # app factory: settings, document, sessions, observability, pipelines, ops endpoints
  config.py            # typed, fail-fast settings (VA-19)
  dispatch.py          # endpoint -> (architecture, delivery); no router (VA-21)
  api/voice.py         # the four voice endpoints (VA-24..27)
  streaming/           # shared request schema + SSE event contract (VA-20)
  pipelines/
    traditional/       # STT -> LLM -> TTS slow path (VA-45)
    realtime/          # voice-to-voice fast path (VA-48)
  providers/           # STT/LLM/TTS/realtime adapters + config-driven factory (VA-30..46)
  context/             # full-document grounding, no RAG (VA-35..37)
  tools/               # tool / function-calling registry (VA-38/39)
  session/             # session store, rolling memory, turn state machine (VA-40..42)
  observability/       # structured logging, latency, usage, counters (VA-57..60)
evaluation/            # offline accuracy + latency + grounding harness (VA-65/66)
scripts/               # operational scripts: endpoint smoke (VA-67)
frontend/              # browser dashboard, served at /ui (VA-51..56)
docs/                  # api, configuration, architecture references
tests/                 # offline test suite (mock providers), run in CI
```

## Local quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# run the service
uvicorn app.main:app --reload --port 8080
# -> http://127.0.0.1:8080/healthz  ->  {"ok": true}
# -> http://127.0.0.1:8080/docs     ->  Swagger UI
# -> http://127.0.0.1:8080/ui/      ->  reference dashboard

# run the tests (offline, mock providers)
pytest

# offline endpoint smoke + evaluation harness
python -m scripts.smoke
python -m evaluation
```

Run with no API keys by selecting the mock providers — `STT_PROVIDER=mock LLM_PROVIDER=mock
TTS_PROVIDER=mock REALTIME_PROVIDER=mock uvicorn app.main:app`. See
[docs/configuration.md](docs/configuration.md) for every setting.

Copy `.env.example` to `.env` for local configuration. **Never commit real secrets** —
`.env` is gitignored and secrets move to a secret manager in a later ticket.

## Run with Docker

A multi-stage, slim image (`python:3.12-slim`, non-root) runs the same locally, in CI, and
on Cloud Run — which injects `$PORT` at runtime.

```bash
docker build -t voice-ai-agent .
docker run --rm -p 8080:8080 -e PORT=8080 voice-ai-agent
# -> http://127.0.0.1:8080/healthz  ->  {"ok": true}
```

The image is built on every PR by `.github/workflows/docker-image.yml`, so a broken
Dockerfile blocks merge.

## Working agreement

- **One branch per ticket**, named `<owner>/<ticket>` — e.g. `Shiva/va-01`.
  No work is committed directly to `main`.
- **Every branch opens a pull request back to `main`**; nothing merges without review
  (see `.github/CODEOWNERS`).
- **A Dockerfile is required** — the service must build and run identically locally, in CI,
  and in the cloud (added in VA-02).
