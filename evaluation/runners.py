"""Runners that answer eval questions (VA-65).

:class:`AppTurnRunner` drives an in-process app through ``/voice/complete`` and measures the
end-to-end latency of each turn. It defaults to mock providers so the harness runs offline in
CI; construct it with a :class:`~app.config.Settings` selecting real providers (keys via env)
to measure real document-grounded accuracy + latency.
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from evaluation.harness import RunResult

_COMPLETE = "/api/v1/voice/complete"


def mock_settings() -> Settings:
    """Settings that select the offline mock providers for every stage."""
    return Settings(
        _env_file=None,
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        realtime_provider="mock",
    )


class AppTurnRunner:
    """Answers each question via the traditional ``/voice/complete`` endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._client = TestClient(create_app(settings or mock_settings()))

    async def __call__(self, question: str) -> RunResult:
        start = time.monotonic()
        resp = self._client.post(_COMPLETE, json={"input": {"kind": "text", "text": question}})
        elapsed_ms = round((time.monotonic() - start) * 1000, 3)
        resp.raise_for_status()
        body = resp.json()
        return RunResult(
            answer_text=body["answer_text"],
            latency_ms=elapsed_ms,
            stages_ms=body.get("latency_ms", {}),
        )
