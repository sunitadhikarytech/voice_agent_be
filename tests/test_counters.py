"""VA-60 — error and fallback-rate counters."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.observability import EventCounters


def test_counts_and_rates_per_path():
    c = EventCounters()
    for _ in range(3):
        c.turn("traditional")
    c.error("traditional")
    c.fallback("traditional")
    summary = c.summary()["traditional"]
    assert summary["turns"] == 3
    assert summary["errors"] == 1
    assert summary["fallbacks"] == 1
    assert summary["error_rate"] == round(1 / 3, 4)
    assert summary["fallback_rate"] == round(1 / 3, 4)


def test_zero_turns_has_zero_rates():
    c = EventCounters()
    c.error("realtime")  # error with no counted turn
    assert c.summary()["realtime"]["error_rate"] == 0.0


def _client() -> TestClient:
    settings = Settings(
        _env_file=None, stt_provider="mock", llm_provider="mock", tts_provider="mock",
        realtime_provider="mock",
    )
    return TestClient(create_app(settings))


def test_successful_turns_counted_no_errors():
    client = _client()
    client.post("/api/v1/voice/complete", json={"input": {"kind": "text", "text": "hi"}})
    client.post("/api/v1/voice/slow", json={"input": {"kind": "text", "text": "hi"}})
    summary = client.get("/api/v1/counters").json()
    assert summary["traditional"]["turns"] == 2
    assert summary["traditional"]["errors"] == 0
    assert summary["traditional"]["error_rate"] == 0.0


def test_fast_path_with_text_input_counts_an_error():
    client = _client()
    # /voice/fast requires audio; text triggers a pipeline error -> counted for the realtime path
    with client:
        try:
            client.post("/api/v1/voice/fast", json={"input": {"kind": "text", "text": "no audio"}})
        except Exception:
            pass  # the streaming error may surface as a broken stream; the counter is what matters
    summary = client.get("/api/v1/counters").json()
    assert summary["realtime"]["turns"] == 1
    assert summary["realtime"]["errors"] == 1
