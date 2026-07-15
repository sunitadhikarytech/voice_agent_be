"""VA-42 — agent turn state machine."""
import logging

import pytest

from app.session import InvalidTransition, TurnState, TurnStateMachine


def test_starts_idle():
    assert TurnStateMachine().state is TurnState.IDLE


def test_full_turn_path_records_history():
    m = TurnStateMachine()
    m.listen()
    m.think()
    m.speak()
    m.finish()
    assert m.state is TurnState.IDLE
    assert [(t.frm, t.to) for t in m.history] == [
        (TurnState.IDLE, TurnState.LISTENING),
        (TurnState.LISTENING, TurnState.THINKING),
        (TurnState.THINKING, TurnState.SPEAKING),
        (TurnState.SPEAKING, TurnState.IDLE),
    ]


def test_invalid_transition_raises():
    m = TurnStateMachine()
    with pytest.raises(InvalidTransition):
        m.speak()  # idle -> speaking is not allowed
    assert m.state is TurnState.IDLE  # unchanged


def test_barge_in_moves_speaking_to_listening_via_interrupted():
    m = TurnStateMachine()
    m.listen()
    m.think()
    m.speak()
    result = m.barge_in()
    assert result is TurnState.LISTENING and m.state is TurnState.LISTENING
    # the transition is explicit: speaking -> interrupted -> listening
    assert [(t.frm, t.to) for t in m.history[-2:]] == [
        (TurnState.SPEAKING, TurnState.INTERRUPTED),
        (TurnState.INTERRUPTED, TurnState.LISTENING),
    ]


def test_barge_in_only_valid_while_speaking():
    m = TurnStateMachine()
    with pytest.raises(InvalidTransition):
        m.barge_in()  # idle


def test_can_reflects_allowed_transitions():
    m = TurnStateMachine()
    assert m.can(TurnState.LISTENING) is True
    assert m.can(TurnState.SPEAKING) is False


def test_transitions_are_logged(caplog):
    m = TurnStateMachine()
    with caplog.at_level(logging.DEBUG, logger="app.session.state"):
        m.listen()
    messages = [r.getMessage() for r in caplog.records if r.name == "app.session.state"]
    assert any("idle -> listening" in msg for msg in messages)
