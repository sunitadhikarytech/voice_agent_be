"""Evaluation harness: score a seed set for accuracy + latency (VA-65).

Deliberately provider-agnostic. :func:`evaluate` takes a sequence of :class:`EvalCase`, an
async :class:`TurnRunner` that answers a question (returning the answer text + end-to-end
latency), and a :data:`Scorer`. It returns an :class:`EvalReport` with per-case results,
overall accuracy, and latency percentiles.

The runner is the seam: back it with the real app + real providers to measure real quality
(see :class:`evaluation.runners.AppTurnRunner`), or with a stub in tests. VA-66 layers a
document-grounding metric on top of this same harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, Sequence

from app.observability.metrics import percentile


@dataclass(frozen=True)
class EvalCase:
    """One question and the keywords a correct answer must contain."""

    id: str
    question: str
    expected_keywords: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class RunResult:
    """What a runner returns for one question."""

    answer_text: str
    latency_ms: float  # end-to-end latency for the turn
    stages_ms: dict[str, float] = field(default_factory=dict)  # optional per-stage breakdown


class TurnRunner(Protocol):
    """Answers a question. Implementations decide how (real app, stub, remote, …)."""

    def __call__(self, question: str) -> Awaitable[RunResult]: ...


Scorer = Callable[[str, Sequence[str]], bool]


def keyword_scorer(answer: str, expected_keywords: Sequence[str]) -> bool:
    """Correct when every expected keyword appears in the answer (case-insensitive).

    A deliberately simple, transparent metric: no expectations → trivially correct; otherwise
    every keyword must be present. VA-66 adds a document-grounding check on top.
    """
    if not expected_keywords:
        return True
    text = answer.lower()
    return all(keyword.lower() in text for keyword in expected_keywords)


@dataclass(frozen=True)
class CaseResult:
    """The outcome of scoring one case."""

    case: EvalCase
    answer_text: str
    passed: bool
    latency_ms: float


@dataclass
class EvalReport:
    """Aggregate results of an evaluation run."""

    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def accuracy(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0

    def latency(self) -> dict[str, float]:
        """``count`` / ``p50`` / ``p95`` / ``max`` over per-case end-to-end latency (ms)."""
        values = sorted(r.latency_ms for r in self.results)
        if not values:
            return {}
        return {
            "count": len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": values[-1],
        }

    def format(self) -> str:
        """Render a compact, human-readable report table."""
        lines = [f"{'case':<30}  pass  {'latency_ms':>10}  answer", "-" * 78]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            answer = (r.answer_text[:34] + "…") if len(r.answer_text) > 35 else r.answer_text
            lines.append(f"{r.case.id:<30}  {mark:<4}  {r.latency_ms:>10.1f}  {answer}")
        lines.append("-" * 78)
        lines.append(f"accuracy: {self.accuracy:.1%}  ({self.passed}/{self.total})")
        lat = self.latency()
        if lat:
            lines.append(
                f"latency_ms  p50={lat['p50']:.1f}  p95={lat['p95']:.1f}  max={lat['max']:.1f}"
            )
        return "\n".join(lines)


async def evaluate(
    cases: Sequence[EvalCase],
    runner: TurnRunner,
    scorer: Scorer = keyword_scorer,
) -> EvalReport:
    """Run every case through ``runner`` and score it. Sequential so latency is uncontended."""
    results: list[CaseResult] = []
    for case in cases:
        run = await runner(case.question)
        results.append(
            CaseResult(
                case=case,
                answer_text=run.answer_text,
                passed=scorer(run.answer_text, case.expected_keywords),
                latency_ms=run.latency_ms,
            )
        )
    return EvalReport(results)
