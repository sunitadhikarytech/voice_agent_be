"""VA-23..27 — the four voice endpoints + delivery modes (end to end, mock providers)."""
import base64

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _client() -> TestClient:
    settings = Settings(
        _env_file=None,
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        realtime_provider="mock",
    )
    return TestClient(create_app(settings))


client = _client()

_TEXT = {"input": {"kind": "text", "text": "what is article 21?"}}
_AUDIO = {"input": {"kind": "audio", "audio_b64": base64.b64encode(b"\x01\x02").decode()}}


# --- complete (JSON) --------------------------------------------------------------------

def test_complete_returns_json_result():
    resp = client.post("/api/v1/voice/complete", json=_TEXT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "what is article 21?"
    assert body["answer_text"] == "mock answer"
    assert "stt_ms" in body["latency_ms"]


# --- slow / stream (SSE, traditional) ---------------------------------------------------

def test_slow_streams_sse_events_in_order():
    resp = client.post("/api/v1/voice/slow", json=_TEXT)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    # events arrive in contract order, terminating with done
    assert (
        body.index("transcript.final")
        < body.index("answer.delta")
        < body.index("audio.chunk")
        < body.index("done")
    )
    assert body.rstrip().endswith(body[body.rindex("event: done"):].rstrip())


def test_stream_endpoint_emits_done():
    resp = client.post("/api/v1/voice/stream", json=_TEXT)
    assert resp.status_code == 200
    assert "event: done" in resp.text


# --- fast (SSE, realtime) ---------------------------------------------------------------

def test_fast_streams_audio_from_realtime():
    resp = client.post("/api/v1/voice/fast", json=_AUDIO)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: audio.chunk" in resp.text
    assert "event: done" in resp.text


# --- contract enforcement ---------------------------------------------------------------

def test_missing_input_returns_422_problem():
    resp = client.post("/api/v1/voice/complete", json={"session_id": "s1"})
    assert resp.status_code == 422
    assert resp.json()["status"] == 422  # VA-28 problem shape


def test_architecture_field_rejected():
    resp = client.post(
        "/api/v1/voice/complete",
        json={"architecture": "realtime", "input": {"kind": "text", "text": "hi"}},
    )
    assert resp.status_code == 422


# --- discoverability --------------------------------------------------------------------

def test_endpoints_documented_in_openapi():
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/v1/voice/fast", "/api/v1/voice/slow", "/api/v1/voice/complete", "/api/v1/voice/stream"):
        assert p in paths
