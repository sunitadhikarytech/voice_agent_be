"""Answer grounding for the traditional path (VA-37).

The retrieval-free replacement for RAG: the whole source document is in context (VA-35/36) and
the model is instructed to answer STRICTLY from it, cite the exact article/clause, and decline
honestly when the answer is not in the document.

``ground_llm`` wires the document and these instructions onto the LLM — the grounded system
prompt is what VA-36 caches alongside the document. ``extract_citations`` pulls the cited
articles out of an answer for the turn trace. The traditional pipeline (VA-45) calls these.
"""
from __future__ import annotations

import re

GROUNDING_INSTRUCTIONS = (
    "Answer using ONLY the source document provided to you as context.\n"
    "- Ground every statement in the document's text; do not rely on outside knowledge.\n"
    '- Cite the exact article, section, or clause you relied on (e.g. "Article 21").\n'
    "- If the answer is not contained in the document, say so honestly — do not guess or "
    "invent an answer.\n"
    "Keep answers concise and conversational for spoken delivery."
)

# Matches "Article 21", "Articles 14", "Article 19(1)(a)", "Article 21A", case-insensitive.
_CITATION_RE = re.compile(r"\barticles?\s+\d+[A-Z]?(?:\s*\([^)]*\))*", re.IGNORECASE)


def build_grounded_prompt(base_prompt: str) -> str:
    """Layer the grounding instructions on top of the agent's base system prompt."""
    base = (base_prompt or "").strip()
    return f"{base}\n\n{GROUNDING_INSTRUCTIONS}" if base else GROUNDING_INSTRUCTIONS


def extract_citations(answer: str) -> list[str]:
    """Return the article/clause citations in ``answer``, whitespace-normalized and
    de-duplicated in order of first appearance (recorded on the turn trace)."""
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(answer or ""):
        citation = re.sub(r"\s+", " ", match.group(0)).strip()
        citation = re.sub(r"^articles?", "Article", citation, flags=re.IGNORECASE)
        seen.setdefault(citation, None)
    return list(seen)


def ground_llm(llm, document) -> None:
    """Attach ``document`` and the grounding instructions to an LLM adapter (e.g. GeminiLlm).

    Sets the document as the LLM's context and layers the grounding instructions onto its
    system prompt — that grounded prompt is what VA-36 caches with the document. Call once,
    at pipeline construction (VA-45).
    """
    llm.document_context = document.text
    llm.system_prompt = build_grounded_prompt(llm.system_prompt)
