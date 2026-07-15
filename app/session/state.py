"""Agent turn state machine (VA-42).

A deterministic per-turn state machine that both pipelines drive, so barge-in, tool calls, and
streaming transitions are explicit and testable rather than ad-hoc. Every transition is
validated against an allowed-transition table, recorded on ``history`` (emitted for the debug
panel, VA-56), and logged.

    idle → listening → thinking → speaking → (idle | listening)
    speaking → interrupted → listening         (barge-in)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("app.session.state")


class TurnState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


# Allowed transitions out of each state.
_ALLOWED: dict[TurnState, frozenset[TurnState]] = {
    TurnState.IDLE: frozenset({TurnState.LISTENING}),
    TurnState.LISTENING: frozenset({TurnState.THINKING, TurnState.IDLE}),
    TurnState.THINKING: frozenset({TurnState.SPEAKING, TurnState.IDLE}),
    TurnState.SPEAKING: frozenset({TurnState.INTERRUPTED, TurnState.LISTENING, TurnState.IDLE}),
    TurnState.INTERRUPTED: frozenset({TurnState.LISTENING, TurnState.IDLE}),
}


class InvalidTransition(RuntimeError):
    """Raised when a transition is not allowed from the current state."""


@dataclass(frozen=True, slots=True)
class Transition:
    """A recorded state change (for the turn trace / debug panel)."""

    frm: TurnState
    to: TurnState


class TurnStateMachine:
    """Deterministic per-turn state machine."""

    def __init__(self, state: TurnState = TurnState.IDLE) -> None:
        self._state = state
        self.history: list[Transition] = []

    @property
    def state(self) -> TurnState:
        return self._state

    def can(self, to: TurnState) -> bool:
        return to in _ALLOWED.get(self._state, frozenset())

    def transition(self, to: TurnState) -> TurnState:
        if not self.can(to):
            raise InvalidTransition(
                f"cannot transition {self._state.value} -> {to.value}"
            )
        logger.debug("turn state %s -> %s", self._state.value, to.value)
        self.history.append(Transition(self._state, to))
        self._state = to
        return to

    # --- convenience transitions ---
    def listen(self) -> TurnState:
        return self.transition(TurnState.LISTENING)

    def think(self) -> TurnState:
        return self.transition(TurnState.THINKING)

    def speak(self) -> TurnState:
        return self.transition(TurnState.SPEAKING)

    def finish(self) -> TurnState:
        return self.transition(TurnState.IDLE)

    def barge_in(self) -> TurnState:
        """Handle a barge-in during playback: speaking → interrupted → listening.

        Deterministically ends in ``listening`` so the new utterance can be captured.
        """
        if self._state is not TurnState.SPEAKING:
            raise InvalidTransition(
                f"barge-in is only valid while speaking, not {self._state.value}"
            )
        self.transition(TurnState.INTERRUPTED)
        return self.transition(TurnState.LISTENING)
