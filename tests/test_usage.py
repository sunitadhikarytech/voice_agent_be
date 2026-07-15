"""VA-59 — cost metering (tokens + audio-seconds per request)."""
import base64

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.observability import UsageMetrics, audio_seconds


def test_audio_seconds_from_pcm16_bytes():
    # 24000 Hz * 2 bytes/sample => 48000 bytes per second
    assert audio_seconds(48_000) == 1.0
    assert audio_seconds(24_000) == 0.5
    assert audio_seconds(0) == 0.0


def test_records_and_aggregates_per_path_and_tenant():
    u = UsageMetrics()
    u.record("traditional", "tenant-a", tokens=100, audio_seconds=1.5)
    u.record("traditional", "tenant-a", tokens=50, audio_seconds=0.5)
    u.record("realtime", "tenant-b", audio_seconds=2.0)
    summary = u.summary()
    assert summary["traditional"]["tenant-a"] == {"tokens": 150, "audio_seconds": 2.0, "turns": 2}
    assert summary["realtime"]["tenant-b"]["audio_seconds"] == 2.0
    assert summary["realtime"]["tenant-b"]["tokens"] == 0


def _client() -> TestClient:
    settings = Settings(
        _env_file=None, stt_provider="mock", llm_provider="mock", tts_provider="mock",
        realtime_provider="mock",
    )
    return TestClient(create_app(settings))


def test_turns_populate_the_usage_endpoint():
    client = _client()
    client.post("/api/v1/voice/complete", json={"input": {"kind": "text", "text": "a question"}})
    client.post(
        "/api/v1/voice/fast",
        json={"input": {"kind": "audio", "audio_b64": base64.b64encode(b"\x00" * 100).decode()}},
    )
    summary = client.get("/api/v1/usage").json()
    assert summary["traditional"]["default"]["tokens"] > 0     # llm tokens metered
    assert summary["traditional"]["default"]["turns"] == 1
    assert summary["realtime"]["default"]["audio_seconds"] >= 0
