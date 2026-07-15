"""Session & memory management.

VA-40 provides session continuity (``Session`` + ``SessionStore``, tenant-scoped and keyed by
session id). VA-41 adds rolling conversation memory; VA-42 adds the per-turn state machine.
"""
from app.session.store import Role, Session, SessionStore, Turn

__all__ = ["Role", "Session", "SessionStore", "Turn"]
