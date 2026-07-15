"""Cost metering: tokens + audio-seconds per request (VA-59).

Records the usage each turn consumes — LLM tokens and audio-seconds — attributed by path and
tenant, so cost per conversation is derivable and the fast-vs-slow trade-off is measured, not
assumed. Usage is both aggregated (queryable at ``/usage``) and emitted as a structured log
line per request (VA-57), so cost is derivable from logs too.

The token/audio figures are estimates here; a real deployment reconciles them against the
providers' reported usage (Gemini ``usage_metadata``, provider audio durations) once the
adapters surface it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

# PCM16 mono default (matches the Cartesia/OpenAI audio format used by the adapters).
DEFAULT_SAMPLE_RATE = 24_000
_BYTES_PER_SAMPLE = 2


def audio_seconds(num_bytes: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Duration in seconds of ``num_bytes`` of PCM16 mono audio."""
    if sample_rate <= 0:
        return 0.0
    return round(num_bytes / (sample_rate * _BYTES_PER_SAMPLE), 3)


@dataclass
class _Usage:
    tokens: int = 0
    audio_seconds: float = 0.0
    turns: int = 0


class UsageMetrics:
    """Aggregates token + audio-second usage per ``(path, tenant)``."""

    def __init__(self) -> None:
        self._usage: dict[tuple[str, str], _Usage] = defaultdict(_Usage)

    def record(self, path: str, tenant: str, *, tokens: int = 0, audio_seconds: float = 0.0) -> None:
        entry = self._usage[(path, tenant)]
        entry.tokens += int(tokens)
        entry.audio_seconds = round(entry.audio_seconds + float(audio_seconds), 3)
        entry.turns += 1

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        """Nested ``{path: {tenant: {tokens, audio_seconds, turns}}}``."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        for (path, tenant), u in self._usage.items():
            out.setdefault(path, {})[tenant] = {
                "tokens": u.tokens,
                "audio_seconds": u.audio_seconds,
                "turns": u.turns,
            }
        return out

    def reset(self) -> None:
        self._usage.clear()
