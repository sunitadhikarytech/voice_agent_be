"""VA-65 (QA-04) — the evaluation harness computes accuracy + latency correctly."""
from __future__ import annotations

import asyncio

from evaluation.dataset import load_seed_set
from evaluation.harness import (
    CaseResult,
    EvalCase,
    EvalReport,
    RunResult,
    evaluate,
    keyword_scorer,
)


# --- scorer -----------------------------------------------------------------------------

def test_keyword_scorer_requires_all_keywords_case_insensitive():
    assert keyword_scorer("Article 21 protects LIFE and personal liberty", ["life", "personal liberty"])
    assert not keyword_scorer("mentions life only", ["life", "personal liberty"])
    assert not keyword_scorer("nothing relevant", ["life"])


def test_keyword_scorer_no_expectations_passes():
    assert keyword_scorer("anything at all", [])


# --- evaluate ---------------------------------------------------------------------------

class _StubRunner:
    """Returns canned (answer, latency) per question so scoring is deterministic."""

    def __init__(self, answers: dict[str, tuple[str, float]]) -> None:
        self._answers = answers

    async def __call__(self, question: str) -> RunResult:
        answer, latency = self._answers[question]
        return RunResult(answer_text=answer, latency_ms=latency)


def test_evaluate_computes_accuracy_and_latency():
    cases = [
        EvalCase("c1", "q1", ("alpha",)),
        EvalCase("c2", "q2", ("beta",)),
        EvalCase("c3", "q3", ("gamma",)),
    ]
    runner = _StubRunner(
        {
            "q1": ("contains alpha", 30.0),  # pass
            "q2": ("no match here", 40.0),   # fail
            "q3": ("has gamma inside", 50.0),  # pass
        }
    )
    report = asyncio.run(evaluate(cases, runner))

    assert report.total == 3
    assert report.passed == 2
    assert report.accuracy == round(2 / 3, 4)
    assert [r.passed for r in report.results] == [True, False, True]

    lat = report.latency()
    assert lat["count"] == 3
    assert lat["p50"] == 40.0
    assert lat["max"] == 50.0
    assert lat["p50"] <= lat["p95"] <= lat["max"]


def test_empty_report_has_zero_accuracy_and_no_latency():
    report = EvalReport([])
    assert report.accuracy == 0.0
    assert report.latency() == {}


def test_report_format_is_printable():
    report = EvalReport(
        [CaseResult(EvalCase("art21", "q", ("life",)), "life and liberty", True, 12.3)]
    )
    text = report.format()
    assert "art21" in text
    assert "PASS" in text
    assert "accuracy" in text.lower()


# --- seed set ---------------------------------------------------------------------------

def test_seed_set_loads_and_is_wellformed():
    cases = load_seed_set()
    assert len(cases) >= 5
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))  # unique ids
    for c in cases:
        assert c.id and c.question and c.expected_keywords
        assert all(k == k.lower() for k in c.expected_keywords)  # lowercase for the scorer


# --- end-to-end against the app ---------------------------------------------------------

def test_harness_runs_against_app_on_mock_providers():
    # Smoke: the harness + AppTurnRunner drive the real app end to end. Mock answers won't
    # match the document-grounded expectations, so accuracy is low — but it RUNS and every
    # case gets a latency sample.
    from evaluation.runners import AppTurnRunner

    cases = load_seed_set()
    report = asyncio.run(evaluate(cases, AppTurnRunner()))
    assert report.total == len(cases)
    assert 0.0 <= report.accuracy <= 1.0
    assert report.latency()["count"] == len(cases)
