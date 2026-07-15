"""VA-57 — structured JSON logging with correlation/session/tenant context."""
import json
import logging

import pytest

from app.observability import (
    JsonFormatter,
    bind_log_context,
    configure_logging,
    reset_log_context,
)
from app.observability.logging import _JSON_HANDLER_FLAG


@pytest.fixture(autouse=True)
def _clear_context():
    # ensure no context leaks between tests
    tokens = bind_log_context(correlation_id=None)  # no-op; explicit reset below
    yield
    reset_log_context(tokens)


def _record(msg="hello", **extra) -> logging.LogRecord:
    return logging.makeLogRecord(
        {"name": "app.test", "levelno": logging.INFO, "levelname": "INFO", "msg": msg, **extra}
    )


def test_format_emits_json_with_core_fields():
    payload = json.loads(JsonFormatter().format(_record("a turn happened")))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "a turn happened"
    assert "ts" in payload


def test_bound_context_appears_on_every_line():
    tokens = bind_log_context(correlation_id="cid-1", session_id="sess-1", tenant_id="tenant-1")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        reset_log_context(tokens)
    assert payload["correlation_id"] == "cid-1"
    assert payload["session_id"] == "sess-1"
    assert payload["tenant_id"] == "tenant-1"


def test_context_absent_when_unbound():
    payload = json.loads(JsonFormatter().format(_record()))
    assert "correlation_id" not in payload
    assert "session_id" not in payload


def test_reset_restores_previous_context():
    tokens = bind_log_context(correlation_id="cid-1")
    reset_log_context(tokens)
    payload = json.loads(JsonFormatter().format(_record()))
    assert "correlation_id" not in payload


def test_structured_extras_are_included():
    payload = json.loads(JsonFormatter().format(_record(latency_ms=123)))
    assert payload["latency_ms"] == 123


def test_configure_logging_installs_json_handler_and_is_idempotent():
    configure_logging("DEBUG")
    configure_logging("INFO")  # second call must not stack handlers
    root = logging.getLogger()
    json_handlers = [h for h in root.handlers if getattr(h, _JSON_HANDLER_FLAG, False)]
    assert len(json_handlers) == 1
    assert isinstance(json_handlers[0].formatter, JsonFormatter)
    assert root.level == logging.INFO
