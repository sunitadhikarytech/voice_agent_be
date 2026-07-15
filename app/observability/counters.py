"""Error and fallback-rate counters (VA-60).

Counts turns, errors, and fallbacks per path so endpoint error rates and the
realtime→traditional fallback rate (VA-49) are visible over time. Queryable at ``/counters``.
"""
from __future__ import annotations

from collections import defaultdict


class EventCounters:
    """Per-path turn / error / fallback counters."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"turns": 0, "errors": 0, "fallbacks": 0}
        )

    def turn(self, path: str) -> None:
        self._counts[path]["turns"] += 1

    def error(self, path: str) -> None:
        self._counts[path]["errors"] += 1

    def fallback(self, path: str) -> None:
        self._counts[path]["fallbacks"] += 1

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-path counts + derived error/fallback rates."""
        out: dict[str, dict[str, float]] = {}
        for path, c in self._counts.items():
            turns = c["turns"]
            out[path] = {
                "turns": turns,
                "errors": c["errors"],
                "fallbacks": c["fallbacks"],
                "error_rate": round(c["errors"] / turns, 4) if turns else 0.0,
                "fallback_rate": round(c["fallbacks"] / turns, 4) if turns else 0.0,
            }
        return out

    def reset(self) -> None:
        self._counts.clear()
