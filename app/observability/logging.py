"""Structured JSON logging with correlation/session/tenant context (VA-57).

Every log line is a JSON object carrying the correlation id (per request, from VA-28's
middleware) and the session/tenant ids (per turn, from the pipelines), so a turn is traceable
end to end. Context is propagated with ``contextvars`` — set once at the boundary, and every
downstream log line across STT/LLM/TTS picks it up automatically.

PII safety (VA-18): every rendered line passes through :mod:`app.observability.pii` — the
message, extras (recursively), and exception text are scrubbed of conversation content,
personal data, and credential shapes before they reach stdout.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone

from app.observability.pii import scrub, scrub_text

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)

_CONTEXT = {
    "correlation_id": correlation_id_var,
    "session_id": session_id_var,
    "tenant_id": tenant_id_var,
}

# Standard LogRecord attributes we never copy into the JSON payload as "extras".
_RESERVED = set(vars(logging.makeLogRecord({})))

_JSON_HANDLER_FLAG = "_va_json_handler"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line, including any bound context + extras.

    PII-safe (VA-18): the message, every extra (recursively), and exception text are
    scrubbed before rendering. The envelope fields (``ts``/``level``/``logger``) and the
    bound context ids are operational values set by our own code and pass through as-is.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": scrub_text(record.getMessage()),
        }
        for key, var in _CONTEXT.items():
            value = var.get()
            if value is not None:
                payload[key] = value
        # structured extras passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = scrub(value, key=key)
        if record.exc_info:
            payload["exc"] = scrub_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger and set the level.

    Idempotent: replaces only our own handler (so pytest's ``caplog`` and other handlers are
    left intact) and can be called on every app construction.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [h for h in root.handlers if not getattr(h, _JSON_HANDLER_FLAG, False)]
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _JSON_HANDLER_FLAG, True)
    root.addHandler(handler)


def bind_log_context(
    *, correlation_id: str | None = None, session_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Token]:
    """Bind context values for subsequent log lines; returns reset tokens."""
    tokens: dict[str, Token] = {}
    if correlation_id is not None:
        tokens["correlation_id"] = correlation_id_var.set(correlation_id)
    if session_id is not None:
        tokens["session_id"] = session_id_var.set(session_id)
    if tenant_id is not None:
        tokens["tenant_id"] = tenant_id_var.set(tenant_id)
    return tokens


def reset_log_context(tokens: dict[str, Token]) -> None:
    for key, token in tokens.items():
        _CONTEXT[key].reset(token)
