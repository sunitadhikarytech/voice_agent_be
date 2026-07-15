"""VA-18 — PII-safe logging: rendered log lines never contain conversation content,
personal data, or credentials.

The guarantee lives at the formatter boundary (every line passes through the scrubber), so
it holds even if future code mistakenly passes a transcript or a token as a log extra.
"""
from __future__ import annotations

import base64
import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.observability import JsonFormatter, REDACTED, bind_log_context, reset_log_context
from app.observability.pii import scrub, scrub_text


def _record(message: str = "msg", **extras) -> logging.LogRecord:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, message, (), None)
    for key, value in extras.items():
        setattr(record, key, value)
    return record


# --- scrub_text: sensitive shapes inside strings -------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "reach me at jane.doe+spam@example.co.uk please",
        "call +1 (415) 555-0134 tomorrow",
        "header was Bearer sk-abc123.def456",
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1MSJ9.sig-part",
    ],
)
def test_scrub_text_redacts_sensitive_shapes(text):
    scrubbed = scrub_text(text)
    assert REDACTED in scrubbed
    for fragment in ("@example", "555-0134", "sk-abc123", "eyJhbGci"):
        assert fragment not in scrubbed


def test_scrub_text_leaves_plain_text_alone():
    assert scrub_text("traditional turn complete") == "traditional turn complete"


def test_scrub_text_preserves_operational_identifiers():
    # hex ids contain letters → never match the digit-run pattern
    line = "session 4238607f6ff34796a03bfb952cdb17ed corr d7aa2002694c4a62976c435c788c3737"
    assert scrub_text(line) == line


def test_scrub_text_preserves_iso_timestamps_inside_text():
    line = "started at 2026-07-15T16:39:49 and finished"
    assert scrub_text(line) == line


# --- scrub: sensitive keys, recursion, non-strings ------------------------------------------

@pytest.mark.parametrize(
    "key",
    ["transcript", "text", "prompt", "answer_text", "audio_b64", "api_key",
     "jwt_secret_key", "authorization", "Access_Token"],
)
def test_sensitive_keys_redacted_wholesale(key):
    assert scrub({key: "anything at all"})[key] == REDACTED


@pytest.mark.parametrize("key", ["tokens", "max_tokens", "token_count", "path", "seq"])
def test_operational_keys_survive(key):
    # counter keys are NOT credentials: matching is boundary-anchored, not substring
    assert scrub({key: 42}) == {key: 42}


def test_nested_structures_are_walked():
    value = {"turn": {"transcript": "hello", "latency_ms": {"stt_ms": 1.5}}, "tags": ["a@b.co"]}
    scrubbed = scrub(value)
    assert scrubbed["turn"]["transcript"] == REDACTED
    assert scrubbed["turn"]["latency_ms"] == {"stt_ms": 1.5}  # numbers untouched
    assert scrubbed["tags"] == [REDACTED]


def test_numbers_and_none_pass_through():
    assert scrub({"tokens": 13, "audio_seconds": 0.0, "flag": True, "none": None}) == {
        "tokens": 13,
        "audio_seconds": 0.0,
        "flag": True,
        "none": None,
    }


# --- JsonFormatter end to end ----------------------------------------------------------------

def test_formatter_scrubs_extras_and_message():
    line = JsonFormatter().format(
        _record("user email is spam@example.com", transcript="secret words", tokens=7)
    )
    assert "spam@example.com" not in line
    assert "secret words" not in line
    assert '"tokens": 7' in line


def test_formatter_scrubs_exception_text():
    try:
        raise RuntimeError("failed for user boss@example.com")
    except RuntimeError:
        record = logging.LogRecord(
            "app.test", logging.ERROR, __file__, 1, "boom", (), __import__("sys").exc_info()
        )
    line = JsonFormatter().format(record)
    assert "boss@example.com" not in line
    assert "RuntimeError" in line  # the operational part of the traceback survives


def test_formatter_keeps_bound_context_ids():
    tokens = bind_log_context(session_id="4238607f6ff34796a03bfb952cdb17ed", tenant_id="default")
    try:
        line = JsonFormatter().format(_record("ok"))
        assert "4238607f6ff34796a03bfb952cdb17ed" in line
    finally:
        reset_log_context(tokens)


# --- full-turn regression: the app never logs conversation content --------------------------

def test_full_turn_logs_contain_no_conversation_content(caplog):
    client = TestClient(
        create_app(
            Settings(
                _env_file=None,
                stt_provider="mock", llm_provider="mock",
                tts_provider="mock", realtime_provider="mock",
            )
        )
    )
    utterance = "my email is private.person@example.com and my number is +1 415 555 0134"
    with caplog.at_level(logging.INFO):
        resp = client.post(
            "/api/v1/voice/complete", json={"input": {"kind": "text", "text": utterance}}
        )
    assert resp.status_code == 200

    formatter = JsonFormatter()
    rendered = [formatter.format(r) for r in caplog.records if r.name.startswith("app.")]
    assert rendered, "expected app log lines for the turn"
    for line in rendered:
        assert "private.person@example.com" not in line
        assert "555 0134" not in line
        assert utterance not in line


def test_audio_payloads_never_rendered(caplog):
    client = TestClient(
        create_app(
            Settings(
                _env_file=None,
                stt_provider="mock", llm_provider="mock",
                tts_provider="mock", realtime_provider="mock",
            )
        )
    )
    audio_b64 = base64.b64encode(b"\x01\x02\x03\x04" * 8).decode()
    with caplog.at_level(logging.INFO):
        resp = client.post(
            "/api/v1/voice/fast", json={"input": {"kind": "audio", "audio_b64": audio_b64}}
        )
    assert resp.status_code == 200
    formatter = JsonFormatter()
    for record in caplog.records:
        assert audio_b64 not in formatter.format(record)
