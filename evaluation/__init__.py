"""Offline evaluation harness for the voice agent (VA-65/66).

Public surface: build cases (:func:`evaluation.dataset.load_seed_set`), run them through a
:class:`~evaluation.harness.TurnRunner`, and score with :func:`~evaluation.harness.evaluate`.
"""
from evaluation.grounding import (
    GroundingEvalReport,
    GroundingResult,
    evaluate_grounding,
    grounding_report,
    is_refusal,
    lexical_support,
)
from evaluation.harness import (
    CaseResult,
    EvalCase,
    EvalReport,
    RunResult,
    Scorer,
    TurnRunner,
    evaluate,
    keyword_scorer,
)

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "RunResult",
    "Scorer",
    "TurnRunner",
    "evaluate",
    "keyword_scorer",
    "GroundingEvalReport",
    "GroundingResult",
    "evaluate_grounding",
    "grounding_report",
    "is_refusal",
    "lexical_support",
]
