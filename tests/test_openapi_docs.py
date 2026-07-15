"""VA-29 — Swagger/OpenAPI descriptions + examples.

The published schema must let an integrator build a client without reading source: request
examples on the shared body, an event-stream example on every SSE endpoint, a result example
on /complete, and real descriptions everywhere.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

PREFIX = "/api/v1"
VOICE_PATHS = ["/voice/fast", "/voice/slow", "/voice/complete", "/voice/stream"]
SSE_PATHS = ["/voice/fast", "/voice/slow", "/voice/stream"]

client = TestClient(create_app(Settings(_env_file=None)))
openapi = client.get("/openapi.json").json()


def _post(path: str) -> dict:
    return openapi["paths"][PREFIX + path]["post"]


# --- request examples -------------------------------------------------------------------------

def test_request_schema_carries_examples():
    examples = openapi["components"]["schemas"]["VoiceTurnRequest"]["examples"]
    assert len(examples) >= 3
    kinds = {e["input"]["kind"] for e in examples}
    assert kinds == {"text", "audio"}  # both input modes shown
    assert any("session_id" in e for e in examples)  # continuity shown


def test_input_variants_carry_examples():
    schemas = openapi["components"]["schemas"]
    assert schemas["TextInput"]["examples"][0]["kind"] == "text"
    assert schemas["AudioInput"]["examples"][0]["mime"] == "audio/webm;codecs=opus"


def test_result_schema_carries_example():
    example = openapi["components"]["schemas"]["VoiceTurnResult"]["examples"][0]
    assert example["transcript"] and example["answer_text"]
    assert "latency_ms" in example


# --- endpoint documentation --------------------------------------------------------------------

def test_every_voice_endpoint_has_summary_and_description():
    for path in VOICE_PATHS:
        op = _post(path)
        assert op["summary"], path
        assert len(op.get("description", "")) > 40, f"{path} needs a real description"


def test_sse_endpoints_declare_event_stream_with_example():
    for path in SSE_PATHS:
        content = _post(path)["responses"]["200"]["content"]
        assert "text/event-stream" in content, path
        example = content["text/event-stream"]["example"]
        # the example is a real rendered stream in emission order, ending with done
        assert example.index("event: transcript.partial") < example.index("event: done")
        assert example.rstrip().endswith("}")


def test_complete_declares_json_result():
    responses = _post("/voice/complete")["responses"]["200"]
    schema_ref = responses["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref.endswith("/VoiceTurnResult")


def test_error_responses_documented_on_voice_endpoints():
    for path in VOICE_PATHS:
        responses = _post(path)["responses"]
        for code in ("401", "422", "429", "500"):
            assert code in responses, f"{path} missing {code}"


def test_swagger_ui_serves():
    assert client.get("/docs").status_code == 200
