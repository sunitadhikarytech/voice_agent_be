"""VA-20 — the contract endpoints publish the schemas and enforce the request contract."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

client = TestClient(create_app(Settings(_env_file=None)))


def test_validate_accepts_text_request():
    resp = client.post(
        "/api/v1/contract/validate", json={"input": {"kind": "text", "text": "hi"}}
    )
    assert resp.status_code == 200
    assert resp.json()["input"] == {"kind": "text", "text": "hi"}


def test_validate_accepts_audio_request():
    resp = client.post(
        "/api/v1/contract/validate",
        json={"session_id": "s1", "input": {"kind": "audio", "audio_b64": "AAAA"}},
    )
    assert resp.status_code == 200
    assert resp.json()["input"]["mime"] == "audio/webm;codecs=opus"


def test_invalid_body_returns_422():
    resp = client.post("/api/v1/contract/validate", json={"session_id": "s1"})  # no input
    assert resp.status_code == 422


def test_architecture_field_rejected_422():
    resp = client.post(
        "/api/v1/contract/validate",
        json={"architecture": "realtime", "input": {"kind": "text", "text": "hi"}},
    )
    assert resp.status_code == 422


def test_schema_endpoint_publishes_contract():
    body = client.get("/api/v1/contract/schema").json()
    assert body["request_example"]["input"]["kind"] == "text"
    names = [e["event"] for e in body["sse_events"]]
    assert names == [
        "transcript.partial",
        "transcript.final",
        "answer.delta",
        "audio.chunk",
        "done",
    ]


def test_schemas_appear_in_openapi():
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("VoiceTurnRequest", "TextInput", "AudioInput", "AudioChunk", "Done"):
        assert name in schemas
