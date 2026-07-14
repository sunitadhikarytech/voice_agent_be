# Voice AI Agent

A dual-pipeline voice assistant served as a single FastAPI service.

- **Traditional (slow) path** — streaming STT → LLM → streaming TTS. Handles complex,
  document-grounded, tool-capable turns. The full source document is supplied to the model
  as context (no vector store / RAG).
- **Realtime (fast) path** — voice-to-voice over WebSocket for fast, natural conversation.

There is **no backend router**: the client chooses a pipeline purely by which endpoint it
calls. Providers and pipelines sit behind small adapter interfaces so they are swappable by
configuration rather than code.

> Status: **VA-01 — service scaffold.** This commit establishes the module boundaries and a
> service that boots with a liveness probe. Endpoints, auth, streaming, pipelines and
> providers arrive in their own tickets.

## Project layout

```
app/
  main.py              # FastAPI app factory + /healthz liveness probe
  config.py            # typed settings (expanded into a fail-fast loader in VA-19)
  dispatch.py          # run_turn seam — no router; endpoint URL selects the pipeline (VA-21)
  pipelines/
    base.py            # Pipeline interface
    traditional/       # STT -> LLM -> TTS slow path (VA-45)
    realtime/          # voice-to-voice fast path (VA-48)
  providers/
    base.py            # STT / LLM / TTS adapter interfaces (VA-30)
  context/             # full-document grounding, no RAG (VA-35..VA-37)
  tools/               # tool / function-calling registry (VA-38)
  streaming/           # SSE event + request/response schemas (VA-20)
tests/                 # unit tests (run in CI on every PR)
frontend/              # browser dashboard (VA-51+)
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

# run the tests
pytest
```

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
