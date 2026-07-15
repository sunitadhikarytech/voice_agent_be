"""Rolling conversation memory (VA-41).

Renders the session's recent dialogue into the short-term memory fed to the model each turn.
This is the *evolving conversation* — distinct from the static full-document context (VA-36).
It is bounded by a token budget: the most recent turns that fit are kept, and older turns are
either summarized (when a summarizer is configured) or dropped (truncation). Memory is derived
per-session, so it is inherently isolated and cleared/rotated with the session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.context import estimate_tokens
from app.session.store import Turn


@dataclass
class ConversationMemory:
    """Builds the bounded rolling memory from a session's turns."""

    token_budget: int = 2000
    # Optional summarizer for evicted (oldest) turns; when None, they are dropped.
    summarize: Callable[[list[Turn]], str] | None = None
    # Token estimator (injectable for tests); defaults to the shared chars/4 heuristic.
    estimate: Callable[[str], int] = estimate_tokens

    def _format(self, turn: Turn) -> str:
        return f"{turn.role}: {turn.text}"

    def build(self, turns: list[Turn]) -> str:
        """Return the memory context string within the token budget.

        The most recent turns that fit are kept in chronological order; older turns are
        summarized (if a summarizer is set) or dropped. The single most recent turn is always
        kept even if it alone exceeds the budget.
        """
        kept: list[Turn] = []
        used = 0
        for turn in reversed(turns):
            cost = self.estimate(self._format(turn))
            if kept and used + cost > self.token_budget:
                break
            kept.append(turn)
            used += cost
        kept.reverse()

        evicted = turns[: len(turns) - len(kept)]
        lines: list[str] = []
        if evicted and self.summarize is not None:
            lines.append(f"[Earlier conversation summary] {self.summarize(evicted)}")
        lines.extend(self._format(t) for t in kept)
        return "\n".join(lines)
