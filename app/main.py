"""FastAPI application entrypoint.

VA-01 stood up the skeleton; VA-19 wires the typed, fail-fast configuration into the app
factory and injects it into request handlers. Settings are resolved once at startup — a
misconfigured service fails to boot with a clear error instead of failing mid-request.

The four voice endpoints, auth middleware, streaming and pipelines arrive in their own
tickets.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from app.api.voice import router as voice_router
from app.config import LogLevel, Settings, get_settings
from app.context import load_document
from app.errors import ERROR_RESPONSES, install_error_handling
from app.pipelines.factory import build_pipeline_registry
from app.session import SessionStore
from app.streaming.contract import router as contract_router


def get_app_settings(request: Request) -> Settings:
    """Dependency: the settings the running app was built with (stored on ``app.state``)."""
    return request.app.state.settings


# Handlers annotate with this to receive the app's settings by injection.
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def _apply_log_level(level: LogLevel) -> None:
    """Set the root log level from configuration. Structured JSON logging, correlation IDs
    and handlers are added in VA-57; this only honours the configured verbosity."""
    logging.getLogger().setLevel(level.value)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Resolving settings here (rather than at import time) keeps construction explicit and
    lets tests inject their own ``Settings`` without touching global state. The settings are
    stored on ``app.state`` and reach handlers via :func:`get_app_settings`.
    """
    app_settings = settings or get_settings()
    _apply_log_level(app_settings.log_level)

    app = FastAPI(
        title="Voice AI Agent",
        version="0.1.0",
        summary="Dual-pipeline voice assistant (traditional STT→LLM→TTS + realtime voice-to-voice).",
    )
    app.state.settings = app_settings

    # Load the full-document context once at startup (VA-35). A configured-but-missing or
    # oversized document fails fast here rather than mid-turn; None when grounding is off.
    app.state.document = load_document(app_settings)

    # Per-conversation state, tenant-scoped and keyed by session id (VA-40).
    app.state.session_store = SessionStore()

    # Traditional + realtime pipelines, registered by architecture for dispatch (VA-45/48).
    app.state.pipelines = build_pipeline_registry(
        app_settings, document=app.state.document, session_store=app.state.session_store
    )

    # Consistent problem-shaped errors + correlation ids on every response (VA-28).
    install_error_handling(app)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, bool]:
        """Dependency-free liveness/readiness probe.

        Must never call a downstream provider so it cannot fail on an upstream outage. The
        full readiness contract is finalized in VA-06.
        """
        return {"ok": True}

    @app.get("/", tags=["ops"])
    def root(settings: SettingsDep) -> dict[str, str]:
        return {"service": settings.app_name, "environment": settings.environment.value}

    @app.get(f"{app_settings.api_prefix}/config", tags=["ops"])
    def config(settings: SettingsDep) -> dict[str, object]:
        """Log-safe view of the effective configuration (secrets redacted)."""
        return settings.public_dict()

    # Shared voice-turn + SSE contract (VA-20).
    app.include_router(
        contract_router,
        prefix=f"{app_settings.api_prefix}/contract",
        tags=["contract"],
        responses=ERROR_RESPONSES,
    )

    # The four voice endpoints (VA-24..27) + delivery modes (VA-23).
    app.include_router(
        voice_router,
        prefix=f"{app_settings.api_prefix}/voice",
        tags=["voice"],
        responses=ERROR_RESPONSES,
    )

    return app


app = create_app()
