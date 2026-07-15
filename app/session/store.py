"""Session management and continuity (VA-40).

Maintains conversation state keyed by ``session_id`` and scoped by ``tenant_id`` across turns
and streaming reconnects. A new turn on an existing session restores its prior context; a
reconnect with the same session id resumes the same session; sessions are isolated per tenant.

The store is in-memory and per-instance — Cloud Run session affinity (VA-05) pins a session to
one instance. VA-41 adds the rolling conversation memory on top of ``Session``; VA-42 adds the
per-turn state machine.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

Role = Literal["user", "agent"]


@dataclass(frozen=True, slots=True)
class Turn:
    """One utterance in the conversation."""

    role: Role
    text: str


@dataclass
class Session:
    """Per-conversation state. Tenant-scoped and keyed by ``session_id``."""

    session_id: str
    tenant_id: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, role: Role, text: str) -> Turn:
        turn = Turn(role=role, text=text)
        self.turns.append(turn)
        return turn


class SessionStore:
    """In-memory session store, namespaced by ``(tenant_id, session_id)``."""

    def __init__(self, *, id_factory: Callable[[], str] | None = None) -> None:
        self._sessions: dict[tuple[str, str], Session] = {}
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)

    def get(self, tenant_id: str, session_id: str) -> Session | None:
        return self._sessions.get((tenant_id, session_id))

    def create(self, tenant_id: str) -> Session:
        """Start a new session with a freshly generated id."""
        session = Session(session_id=self._id_factory(), tenant_id=tenant_id)
        self._sessions[(tenant_id, session.session_id)] = session
        return session

    def get_or_create(self, tenant_id: str, session_id: str) -> Session:
        session = self._sessions.get((tenant_id, session_id))
        if session is None:
            session = Session(session_id=session_id, tenant_id=tenant_id)
            self._sessions[(tenant_id, session_id)] = session
        return session

    def resolve(self, tenant_id: str, session_id: str | None) -> Session:
        """Per-turn entry point: resume an existing session, or start a new one when the
        request carries no ``session_id``."""
        if not session_id:
            return self.create(tenant_id)
        return self.get_or_create(tenant_id, session_id)

    def drop(self, tenant_id: str, session_id: str) -> None:
        self._sessions.pop((tenant_id, session_id), None)

    def __len__(self) -> int:
        return len(self._sessions)
