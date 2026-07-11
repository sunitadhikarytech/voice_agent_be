"""FastAPI application entrypoint.

VA-01 stands up the service skeleton so `uvicorn app.main:app` boots locally and the
module boundaries the rest of the backlog builds against are in place. Only a
dependency-free liveness endpoint is wired here; the four voice endpoints, auth
middleware, streaming and pipelines arrive in their own tickets.
"""
from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings


def create_app() -> FastAPI:
    """Application factory. Keeping construction in a function makes the app easy to
    configure per-environment and to instantiate fresh inside tests."""
    settings = get_settings()

    app = FastAPI(
        title="Voice AI Agent",
        version="0.1.0",
        summary="Dual-pipeline voice assistant (traditional STT→LLM→TTS + realtime voice-to-voice).",
    )

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, bool]:
        """Dependency-free liveness/readiness probe.

        Must never call a downstream provider so it cannot fail on an upstream outage.
        The full readiness contract is finalized in VA-06.
        """
        return {"ok": True}

    @app.get("/", tags=["ops"])
    def root() -> dict[str, str]:
        return {"service": settings.app_name, "environment": settings.environment}

    return app


app = create_app()
