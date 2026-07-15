"""VA-67 — the endpoint smoke passes against the in-process app (mock providers)."""
from __future__ import annotations

from scripts.smoke import InProcessClient, run, run_checks


def test_smoke_exits_zero_in_process():
    assert run() == 0


def test_every_check_passes_and_covers_the_key_endpoints():
    results = run_checks(InProcessClient())
    assert results, "no checks ran"
    assert all(ok for _, ok, _ in results), [n for n, ok, _ in results if not ok]

    names = " ".join(n for n, _, _ in results)
    for fragment in ("/healthz", "/voice/complete", "/voice/slow", "/voice/fast", "/counters", "/ui/"):
        assert fragment in names
