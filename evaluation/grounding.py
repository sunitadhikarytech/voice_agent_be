"""Document-grounded answer evaluation (VA-66).

Builds on the VA-65 harness with a grounding metric: is an answer actually supported by the
source document, or is it ungrounded / hallucinated? It combines three signals from the
answer and the document text:

- **citations** — the article/clause citations the answer makes (reuses the app's VA-37
  ``extract_citations``);
- **citation support** — every cited article actually appears in the document;
- **lexical support** — the fraction of the answer's significant words found in the document.

An answer is *grounded* when it cites at least one article, every citation is supported by
the document, and its lexical support clears a threshold. An honest refusal ("not contained
in the document") is recognized as faithful and counts as grounded rather than being
penalized — the alternative (inventing an answer) is exactly what grounding guards against.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.context import extract_citations
from evaluation.harness import EvalCase, TurnRunner

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "by",
        "with", "that", "this", "it", "as", "be", "from", "at", "which", "under", "shall",
        "any", "all", "not", "no", "its", "their", "there", "what", "does", "provide",
    }
)
_REFUSAL_MARKERS = (
    "not contained in the document",
    "not in the document",
    "does not contain",
    "cannot find",
    "could not find",
    "no information",
    "not mentioned",
    "not covered",
)


def is_refusal(answer: str) -> bool:
    """True when the answer honestly declines because the document lacks the answer."""
    text = (answer or "").lower()
    return any(marker in text for marker in _REFUSAL_MARKERS)


def _significant_words(text: str) -> list[str]:
    """Content words of ``text``: length ≥ 4, not a stopword, lowercased."""
    return [w for w in _WORD_RE.findall((text or "").lower()) if len(w) >= 4 and w not in _STOPWORDS]


def lexical_support(answer: str, document_text: str) -> float:
    """Fraction of the answer's significant words that also appear in the document, in [0, 1]."""
    words = _significant_words(answer)
    if not words:
        return 0.0
    doc = set(_significant_words(document_text))
    hits = sum(1 for w in words if w in doc)
    return round(hits / len(words), 4)


@dataclass(frozen=True)
class GroundingResult:
    """The grounding verdict for a single answer."""

    answer_text: str
    cited: list[str]
    citations_supported: bool
    support: float
    refused: bool
    grounded: bool


def grounding_report(answer: str, document_text: str, *, min_support: float = 0.5) -> GroundingResult:
    """Score one answer for grounding against ``document_text``."""
    cited = extract_citations(answer)
    doc_lower = (document_text or "").lower()
    citations_supported = all(c.lower() in doc_lower for c in cited)
    support = lexical_support(answer, document_text)
    refused = is_refusal(answer)
    grounded = refused or (bool(cited) and citations_supported and support >= min_support)
    return GroundingResult(
        answer_text=answer,
        cited=cited,
        citations_supported=citations_supported,
        support=support,
        refused=refused,
        grounded=grounded,
    )


@dataclass
class GroundingEvalReport:
    """Aggregate grounding results across a seed set."""

    results: list[tuple[EvalCase, GroundingResult]]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def grounded(self) -> int:
        return sum(1 for _, g in self.results if g.grounded)

    @property
    def grounding_rate(self) -> float:
        return round(self.grounded / self.total, 4) if self.total else 0.0

    @property
    def citation_rate(self) -> float:
        """Fraction of answers that make at least one document-supported citation."""
        cited = sum(1 for _, g in self.results if g.cited and g.citations_supported)
        return round(cited / self.total, 4) if self.total else 0.0

    @property
    def mean_support(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(g.support for _, g in self.results) / self.total, 4)

    def format(self) -> str:
        lines = [f"{'case':<30}  grounded  cited  supp   citations", "-" * 78]
        for case, g in self.results:
            mark = "yes" if g.grounded else "no"
            cites = ", ".join(g.cited) if g.cited else ("(refused)" if g.refused else "—")
            lines.append(f"{case.id:<30}  {mark:<8}  {('y' if g.citations_supported and g.cited else 'n'):<5}  {g.support:>4.2f}   {cites}")
        lines.append("-" * 78)
        lines.append(
            f"grounding_rate: {self.grounding_rate:.1%}   "
            f"citation_rate: {self.citation_rate:.1%}   mean_support: {self.mean_support:.2f}"
        )
        return "\n".join(lines)


async def evaluate_grounding(
    cases,
    runner: TurnRunner,
    document_text: str,
    *,
    min_support: float = 0.5,
) -> GroundingEvalReport:
    """Run every case through ``runner`` and grade each answer's grounding against the document."""
    results: list[tuple[EvalCase, GroundingResult]] = []
    for case in cases:
        run = await runner(case.question)
        results.append((case, grounding_report(run.answer_text, document_text, min_support=min_support)))
    return GroundingEvalReport(results)
