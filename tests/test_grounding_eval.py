"""VA-66 (QA-05) — document-grounded answer evaluation."""
from __future__ import annotations

import asyncio

from evaluation.dataset import load_seed_set
from evaluation.grounding import (
    evaluate_grounding,
    grounding_report,
    is_refusal,
    lexical_support,
)
from evaluation.harness import EvalCase, RunResult

# A tiny synthetic "document" so grounding assertions are deterministic.
DOC = (
    "Article 21 protects the right to life and personal liberty. "
    "Article 14 guarantees equality before the law."
)

GROUNDED = "Article 21 protects life and personal liberty."
HALLUCINATED = "Article 999 establishes a Ministry of Magic and unicorns."
UNCITED = "Refunds are available within thirty days of purchase."
REFUSAL = "That information is not contained in the document."
PARTIAL = "Article 14 covers equality and taxation policy."


# --- signals ----------------------------------------------------------------------------

def test_is_refusal():
    assert is_refusal(REFUSAL)
    assert is_refusal("The document does not contain that.")
    assert not is_refusal(GROUNDED)


def test_lexical_support_fraction():
    assert lexical_support(GROUNDED, DOC) == 1.0  # every content word is in the document
    assert lexical_support(HALLUCINATED, DOC) == 0.2  # only "article" overlaps (1/5)
    assert lexical_support("", DOC) == 0.0


# --- grounding_report -------------------------------------------------------------------

def test_grounded_answer_is_grounded():
    r = grounding_report(GROUNDED, DOC)
    assert r.cited == ["Article 21"]
    assert r.citations_supported is True
    assert r.support == 1.0
    assert r.grounded is True


def test_hallucinated_answer_is_not_grounded():
    r = grounding_report(HALLUCINATED, DOC)
    assert r.cited == ["Article 999"]
    assert r.citations_supported is False  # Article 999 is not in the document
    assert r.grounded is False


def test_uncited_answer_is_not_grounded():
    r = grounding_report(UNCITED, DOC)
    assert r.cited == []
    assert r.grounded is False  # no citation, low lexical support


def test_refusal_counts_as_grounded():
    r = grounding_report(REFUSAL, DOC)
    assert r.refused is True
    assert r.cited == []
    assert r.grounded is True  # honest decline is faithful, not a hallucination


def test_support_threshold_is_configurable():
    # PARTIAL cites a real article but only ~0.4 of its words are in the document
    assert grounding_report(PARTIAL, DOC, min_support=0.5).grounded is False
    assert grounding_report(PARTIAL, DOC, min_support=0.3).grounded is True


# --- aggregate over a seed set ----------------------------------------------------------

class _StubRunner:
    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    async def __call__(self, question: str) -> RunResult:
        return RunResult(answer_text=self._answers[question], latency_ms=1.0)


def test_evaluate_grounding_aggregates():
    cases = [
        EvalCase("grounded", "q1", ()),
        EvalCase("hallucinated", "q2", ()),
        EvalCase("uncited", "q3", ()),
        EvalCase("refusal", "q4", ()),
    ]
    runner = _StubRunner(
        {"q1": GROUNDED, "q2": HALLUCINATED, "q3": UNCITED, "q4": REFUSAL}
    )
    report = asyncio.run(evaluate_grounding(cases, runner, DOC))

    assert report.total == 4
    assert report.grounded == 2  # GROUNDED + REFUSAL
    assert report.grounding_rate == 0.5
    assert report.citation_rate == 0.25  # only GROUNDED cites a supported article
    assert report.mean_support == 0.3  # (1.0 + 0.2 + 0.0 + 0.0) / 4


def test_grounding_report_format_is_printable():
    cases = [EvalCase("grounded", "q1", ())]
    report = asyncio.run(evaluate_grounding(cases, _StubRunner({"q1": GROUNDED}), DOC))
    text = report.format()
    assert "grounded" in text
    assert "grounding_rate" in text


# --- end-to-end against the app ---------------------------------------------------------

def test_grounding_eval_runs_against_app():
    from evaluation.runners import AppTurnRunner

    runner = AppTurnRunner()  # mock providers, no document loaded
    cases = load_seed_set()
    report = asyncio.run(evaluate_grounding(cases, runner, runner.document_text))
    assert report.total == len(cases)
    assert 0.0 <= report.grounding_rate <= 1.0
    # mock answers cite nothing and there is no document → nothing is grounded
    assert report.grounding_rate == 0.0


def test_grounding_eval_runs_against_app_with_document(tmp_path):
    # Grounding is wired onto the LLM when a document is loaded; this exercises that path on
    # the mock providers (previously an AttributeError — MockLlm now honors the LLM interface).
    from app.config import Settings
    from evaluation.runners import AppTurnRunner

    doc = tmp_path / "doc.txt"
    doc.write_text("Article 21 protects the right to life and personal liberty.", encoding="utf-8")
    runner = AppTurnRunner(
        Settings(
            _env_file=None,
            source_doc_path=str(doc),
            stt_provider="mock",
            llm_provider="mock",
            tts_provider="mock",
            realtime_provider="mock",
        )
    )
    assert runner.document_text  # the document loaded → grounding is on
    report = asyncio.run(evaluate_grounding(load_seed_set(), runner, runner.document_text))
    assert report.total == len(load_seed_set())
    assert 0.0 <= report.grounding_rate <= 1.0
