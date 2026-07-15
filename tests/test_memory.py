"""VA-41 — rolling conversation memory."""
from app.session import ConversationMemory, Session, Turn

# 1 token per formatted turn, for predictable budget math.
_ONE = lambda _s: 1  # noqa: E731


def _turns(*texts) -> list[Turn]:
    # alternate user/agent roles
    return [Turn(role="user" if i % 2 == 0 else "agent", text=t) for i, t in enumerate(texts)]


def test_recent_turns_influence_later_answers():
    mem = ConversationMemory(token_budget=100)
    out = mem.build(_turns("what is article 21?", "It protects life and liberty."))
    assert "user: what is article 21?" in out
    assert "agent: It protects life and liberty." in out


def test_truncates_oldest_when_over_budget():
    mem = ConversationMemory(token_budget=2, estimate=_ONE)
    out = mem.build(_turns("t1", "t2", "t3", "t4"))
    assert "t3" in out and "t4" in out       # two most recent kept
    assert "t1" not in out and "t2" not in out  # older dropped (no summarizer)


def test_summarizes_evicted_turns_when_summarizer_configured():
    mem = ConversationMemory(
        token_budget=2, estimate=_ONE, summarize=lambda evicted: f"{len(evicted)} earlier turns"
    )
    out = mem.build(_turns("t1", "t2", "t3", "t4"))
    assert out.startswith("[Earlier conversation summary] 2 earlier turns")
    assert "t3" in out and "t4" in out


def test_most_recent_turn_always_kept_even_if_over_budget():
    mem = ConversationMemory(token_budget=1, estimate=lambda _s: 10)
    out = mem.build(_turns("older", "newest"))
    assert out == "agent: newest"


def test_empty_conversation_is_empty():
    assert ConversationMemory().build([]) == ""


def test_memory_is_per_session_and_clearable():
    mem = ConversationMemory(token_budget=100)
    s = Session(session_id="s1", tenant_id="t1")
    s.add_turn("user", "remember this")
    assert "remember this" in mem.build(s.turns)
    s.clear()
    assert mem.build(s.turns) == ""
