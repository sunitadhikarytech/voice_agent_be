"""PII-safe log redaction (VA-18).

Redaction happens at the formatter boundary — the last point every log line passes — so no
downstream mistake (an ``extra`` carrying a transcript, an exception message embedding an
email address) can leak conversation content, personal data, or credentials into the
structured logs. The pipelines already avoid logging payload content; this makes that a
guarantee rather than a convention.

Two complementary rules:

* **Sensitive keys** — extras whose key names denote conversation content (``transcript``,
  ``prompt``, ``audio_b64``, …) or credentials (anything containing ``secret``, ``token``,
  ``api_key``, …) are redacted wholesale, recursively through nested dicts and lists.
* **Sensitive shapes** — string values are scanned for patterns that are personal data or
  credentials wherever they appear: email addresses, phone-like digit runs, bearer tokens,
  and JWTs.

Deliberately conservative: operational identifiers must survive untouched — hex
correlation/session ids contain letters and never match the digit-run pattern, and numeric
metrics (latencies, token counts) are not strings, so they pass through unchanged.
"""
from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"

# Key names (lowercased) whose values are conversation/payload content.
_CONTENT_KEYS = frozenset(
    {
        "text",
        "transcript",
        "prompt",
        "answer",
        "answer_text",
        "utterance",
        "audio",
        "audio_b64",
        "user_input",
        "input",
    }
)

# Keys that denote credentials: the name ends in a credential word at a boundary
# (``api_key``, ``access_token``, ``jwt_secret_key``, ``authorization``). Deliberately NOT a
# substring match — operational counters like ``tokens`` / ``max_tokens`` must survive.
_CREDENTIAL_KEY = re.compile(
    r"(?:^|[_-])(?:secret|password|passwd|token|api_?key|key|authorization|auth|bearer|credentials?|jwt)$"
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # email addresses
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # phone-like digit runs (9+ digits with optional separators); hex ids contain letters
    # and never match, and a digit run followed by a word character (e.g. inside an ISO
    # timestamp) is left alone
    re.compile(r"(?<![\w/])\+?\d[\d\s().-]{7,}\d(?!\w)"),
    # bearer tokens / JWTs
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9._-]+"),
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _CONTENT_KEYS:
        return True
    return _CREDENTIAL_KEY.search(lowered) is not None


def scrub_text(value: str) -> str:
    """Redact PII/credential shapes inside a string, leaving the rest intact."""
    for pattern in _PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value


def scrub(value: Any, *, key: str | None = None) -> Any:
    """Redact ``value`` for logging.

    A value under a sensitive key is replaced wholesale; strings are pattern-scrubbed;
    dicts/lists are walked recursively; everything else (numbers, bools, None) passes
    through untouched.
    """
    if key is not None and _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {k: scrub(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value
