"""VA-17 — per-key / per-IP rate limiting (token bucket).

Buckets by the validated JWT subject when auth is on, else by client IP. Off by default
(``RATE_LIMIT_PER_MINUTE=0``); exhausted buckets return a problem-shaped 429 with
``Retry-After``. Time is driven with a fake clock — no sleeping.
"""
from __future__ import annotations

import time

import jwt as pyjwt
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.ratelimit import RateLimiter

SECRET = "rate-limit-test-secret-0123456789abcdef"

_MOCKS = dict(stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock")
_TEXT = {"input": {"kind": "text", "text": "hi"}}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _app(**overrides):
    return create_app(Settings(_env_file=None, **_MOCKS, **overrides))


def _bearer(sub: str = "client-1") -> dict[str, str]:
    token = pyjwt.encode(
        {"sub": sub, "tenant": "acme", "exp": int(time.time()) + 300}, SECRET, algorithm="HS256"
    )
    return {"authorization": f"Bearer {token}"}


# --- bucket math (unit) -----------------------------------------------------------------------

def test_bucket_allows_burst_then_blocks():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=60, burst=3, clock=clock)
    assert [limiter.check("k")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after, remaining = limiter.check("k")
    assert allowed is False and remaining == 0
    assert 0 < retry_after <= 1.0  # 60/min = 1 token/s


def test_bucket_refills_with_time():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=60, burst=1, clock=clock)
    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is False
    clock.now += 1.0  # one token refilled
    assert limiter.check("k")[0] is True


def test_bucket_never_exceeds_capacity():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=60, burst=2, clock=clock)
    clock.now += 3600  # an hour idle refills at most to capacity
    assert [limiter.check("k")[0] for _ in range(3)] == [True, True, False]


def test_keys_are_independent():
    limiter = RateLimiter(per_minute=60, burst=1, clock=FakeClock())
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is False
    assert limiter.check("b")[0] is True  # a's exhaustion never affects b


def test_key_flood_is_pruned_not_unbounded():
    limiter = RateLimiter(per_minute=60, burst=1, clock=FakeClock(), max_keys=100)
    for i in range(500):
        limiter.check(f"ip:{i}")
    assert len(limiter) <= 100


def test_backwards_clock_does_not_drain():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=60, burst=2, clock=clock)
    limiter.check("k")
    clock.now -= 100  # clamped: negative elapsed must not remove tokens
    assert limiter.check("k")[0] is True


# --- middleware: off by default ---------------------------------------------------------------

def test_disabled_by_default():
    app = _app()
    assert app.state.rate_limiter is None
    client = TestClient(app)
    for _ in range(5):
        assert client.post("/api/v1/voice/complete", json=_TEXT).status_code == 200


# --- middleware: per-IP (auth off) ------------------------------------------------------------

def test_ip_bucket_exhausts_and_recovers():
    app = _app(rate_limit_per_minute=60, rate_limit_burst=2)
    clock = FakeClock()
    app.state.rate_limiter.clock = clock
    client = TestClient(app)

    assert client.post("/api/v1/voice/complete", json=_TEXT).status_code == 200
    assert client.post("/api/v1/voice/complete", json=_TEXT).status_code == 200
    blocked = client.post("/api/v1/voice/complete", json=_TEXT)
    assert blocked.status_code == 429
    body = blocked.json()
    assert body["status"] == 429 and body["title"] == "Too Many Requests"
    assert "correlation_id" in body
    assert int(blocked.headers["retry-after"]) >= 1
    assert blocked.headers["x-ratelimit-remaining"] == "0"

    clock.now += 1.0  # refill one token
    assert client.post("/api/v1/voice/complete", json=_TEXT).status_code == 200


def test_forwarded_for_header_identifies_the_client():
    app = _app(rate_limit_per_minute=60, rate_limit_burst=1)
    app.state.rate_limiter.clock = FakeClock()
    client = TestClient(app)
    a = {"x-forwarded-for": "203.0.113.7"}
    b = {"x-forwarded-for": "203.0.113.8, 10.0.0.1"}
    assert client.post("/api/v1/voice/complete", json=_TEXT, headers=a).status_code == 200
    assert client.post("/api/v1/voice/complete", json=_TEXT, headers=a).status_code == 429
    # a different client (first XFF entry) has its own bucket
    assert client.post("/api/v1/voice/complete", json=_TEXT, headers=b).status_code == 200


# --- middleware: per-key (auth on) ------------------------------------------------------------

def test_authenticated_subjects_have_independent_buckets():
    app = _app(jwt_secret_key=SECRET, rate_limit_per_minute=60, rate_limit_burst=1)
    app.state.rate_limiter.clock = FakeClock()
    client = TestClient(app)

    assert client.post("/api/v1/voice/complete", json=_TEXT, headers=_bearer("key-a")).status_code == 200
    assert client.post("/api/v1/voice/complete", json=_TEXT, headers=_bearer("key-a")).status_code == 429
    # same IP, different API key → its own bucket
    assert client.post("/api/v1/voice/complete", json=_TEXT, headers=_bearer("key-b")).status_code == 200


def test_streaming_endpoints_are_limited_too():
    app = _app(rate_limit_per_minute=60, rate_limit_burst=1)
    app.state.rate_limiter.clock = FakeClock()
    client = TestClient(app)
    assert client.post("/api/v1/voice/slow", json=_TEXT).status_code == 200
    assert client.post("/api/v1/voice/slow", json=_TEXT).status_code == 429


# --- scope + headers ---------------------------------------------------------------------------

def test_public_surface_is_never_limited():
    app = _app(rate_limit_per_minute=60, rate_limit_burst=1)
    app.state.rate_limiter.clock = FakeClock()
    client = TestClient(app)
    for _ in range(5):
        assert client.get("/healthz").status_code == 200


def test_success_responses_carry_ratelimit_headers():
    app = _app(rate_limit_per_minute=60, rate_limit_burst=5)
    app.state.rate_limiter.clock = FakeClock()
    resp = TestClient(app).post("/api/v1/voice/complete", json=_TEXT)
    assert resp.headers["x-ratelimit-limit"] == "60"
    assert resp.headers["x-ratelimit-remaining"] == "4"


def test_rate_limit_visible_in_public_config():
    client = TestClient(_app(rate_limit_per_minute=120))
    assert client.get("/api/v1/config").json()["rate_limit_per_minute"] == 120
