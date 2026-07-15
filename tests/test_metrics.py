"""VA-58 — per-stage and first-audio latency metrics."""
import base64

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.observability import LatencyMetrics, percentile


def test_percentile_linear_interpolation():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(vals, 0.5) == 30.0
    assert percentile(vals, 0.95) == 48.0  # 40 + (50-40)*0.8
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.9) == 7.0


def test_records_per_path_and_stage():
    m = LatencyMetrics()
    m.record("traditional", {"stt_ms": 100, "llm_ms": 300})
    m.record("traditional", {"stt_ms": 200, "llm_ms": 500})
    summary = m.summary()
    assert summary["traditional"]["stt_ms"]["count"] == 2
    assert summary["traditional"]["stt_ms"]["p50"] == 150.0
    assert summary["traditional"]["llm_ms"]["max"] == 500.0


def test_fast_and_slow_paths_are_comparable():
    m = LatencyMetrics()
    m.record("traditional", {"first_audio_ms": 900})
    m.record("realtime", {"first_audio_ms": 200})
    summary = m.summary()
    assert "traditional" in summary and "realtime" in summary
    assert summary["realtime"]["first_audio_ms"]["p50"] == 200.0


def test_reset_clears_samples():
    m = LatencyMetrics()
    m.record("traditional", {"stt_ms": 1})
    m.reset()
    assert m.summary() == {}


# --- integration: a real turn feeds the collector, exposed at /metrics ------------------

def _client() -> TestClient:
    settings = Settings(
        _env_file=None, stt_provider="mock", llm_provider="mock", tts_provider="mock",
        realtime_provider="mock",
    )
    return TestClient(create_app(settings))


def test_turns_populate_the_metrics_endpoint():
    client = _client()
    client.post("/api/v1/voice/complete", json={"input": {"kind": "text", "text": "hi"}})
    client.post(
        "/api/v1/voice/fast",
        json={"input": {"kind": "audio", "audio_b64": base64.b64encode(b"\x01").decode()}},
    )
    summary = client.get("/api/v1/metrics").json()
    assert "traditional" in summary and "stt_ms" in summary["traditional"]
    assert "realtime" in summary and "first_audio_ms" in summary["realtime"]
