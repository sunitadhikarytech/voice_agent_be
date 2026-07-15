"""VA-40 — session management and continuity."""
from app.config import Settings
from app.main import create_app
from app.session import Session, SessionStore


def _store(**kwargs) -> SessionStore:
    return SessionStore(**kwargs)


# --- new sessions -----------------------------------------------------------------------

def test_resolve_without_id_starts_a_new_session():
    ids = iter(["sess-1", "sess-2"])
    store = _store(id_factory=lambda: next(ids))
    s1 = store.resolve("tenant-a", None)
    s2 = store.resolve("tenant-a", None)
    assert isinstance(s1, Session)
    assert (s1.session_id, s2.session_id) == ("sess-1", "sess-2")  # distinct sessions
    assert len(store) == 2


# --- continuity -------------------------------------------------------------------------

def test_reconnect_resumes_the_same_session_with_prior_context():
    store = _store()
    first = store.resolve("tenant-a", "sess-1")
    first.add_turn("user", "what is article 21?")
    first.add_turn("agent", "It protects life and liberty.")

    # a reconnect / next turn with the same id resumes the SAME session
    resumed = store.resolve("tenant-a", "sess-1")
    assert resumed is first
    assert [t.text for t in resumed.turns] == [
        "what is article 21?",
        "It protects life and liberty.",
    ]


def test_multi_turn_conversation_accumulates_context():
    store = _store()
    s = store.resolve("tenant-a", "sess-1")
    s.add_turn("user", "hi")
    store.resolve("tenant-a", "sess-1").add_turn("agent", "hello")
    store.resolve("tenant-a", "sess-1").add_turn("user", "bye")
    assert [t.role for t in store.get("tenant-a", "sess-1").turns] == ["user", "agent", "user"]


# --- tenant scoping ---------------------------------------------------------------------

def test_sessions_are_tenant_scoped():
    store = _store()
    a = store.resolve("tenant-a", "shared-id")
    b = store.resolve("tenant-b", "shared-id")
    a.add_turn("user", "tenant A secret")
    assert a is not b
    assert b.turns == []                     # tenant B cannot see tenant A's turns
    assert store.get("tenant-b", "shared-id").turns == []


def test_get_unknown_returns_none_and_drop_removes():
    store = _store()
    assert store.get("tenant-a", "nope") is None
    store.resolve("tenant-a", "sess-1")
    store.drop("tenant-a", "sess-1")
    assert store.get("tenant-a", "sess-1") is None


# --- app wiring -------------------------------------------------------------------------

def test_app_has_a_session_store():
    app = create_app(Settings(_env_file=None))
    assert isinstance(app.state.session_store, SessionStore)
