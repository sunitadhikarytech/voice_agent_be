"""Per-stage and first-audio latency metrics (VA-58).

Records the per-turn latency the pipelines already measure (STT/LLM/TTS + first-audio, or the
realtime round-trip), bucketed by path, so fast-vs-slow medians are comparable and p50/p95 are
queryable — the latency the user actually feels. In production these also flow to Cloud
Monitoring/Trace; this in-process collector backs the `/metrics` endpoint.
"""
from __future__ import annotations

import math
from collections import defaultdict


def percentile(sorted_values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (``pct`` in [0, 1]); ``None`` for no samples."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


class LatencyMetrics:
    """Collects per-(path, stage) latency samples and reports aggregates."""

    def __init__(self) -> None:
        self._samples: dict[tuple[str, str], list[float]] = defaultdict(list)

    def record(self, path: str, latency_ms: dict[str, float]) -> None:
        for stage, value in latency_ms.items():
            self._samples[(path, stage)].append(float(value))

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        """Nested ``{path: {stage: {count, p50, p95, max}}}`` for querying."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        for (path, stage), values in self._samples.items():
            ordered = sorted(values)
            out.setdefault(path, {})[stage] = {
                "count": len(ordered),
                "p50": percentile(ordered, 0.5),
                "p95": percentile(ordered, 0.95),
                "max": ordered[-1],
            }
        return out

    def reset(self) -> None:
        self._samples.clear()
