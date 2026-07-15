"""Per-key / per-IP rate limiting (VA-17).

A token-bucket limiter applied to everything under the API prefix. Requests are bucketed by
the strongest identity available:

* **per key** — the validated JWT ``sub`` when auth (VA-15) is on;
* **per IP** — the client address (first ``X-Forwarded-For`` entry behind a proxy such as
  Cloud Run's load balancer, else the socket peer) when the API runs open.

Disabled by default (``RATE_LIMIT_PER_MINUTE=0``) so local/offline development is
unthrottled; deployments opt in via configuration. The store is in-memory and per-instance —
Cloud Run session affinity (VA-05) pins a client to one instance, which is exactly the scope
a bucket needs. Exhausted buckets return the standard problem shape (VA-28) as ``429`` with
a ``Retry-After`` header; successful responses expose ``X-RateLimit-Limit`` /
``X-RateLimit-Remaining``.

The clock is injectable (``RateLimiter.clock``) so tests drive time deterministically.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings
from app.errors import problem_response

# One flood of unique keys (e.g. spoofed IPs) must not grow memory unbounded: past this many
# buckets, the stalest half are pruned.
MAX_TRACKED_KEYS = 10_000


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """A classic token bucket per key: ``per_minute`` refill, ``burst`` capacity."""

    def __init__(
        self,
        per_minute: int,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = MAX_TRACKED_KEYS,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive — use install_rate_limiting to disable")
        self.per_minute = per_minute
        self.burst = max(1, burst)
        self.clock = clock  # mutable on purpose: tests swap in a fake
        self._rate = per_minute / 60.0  # tokens per second
        self._buckets: dict[str, _Bucket] = {}
        self._max_keys = max_keys

    def check(self, key: str) -> tuple[bool, float, int]:
        """Take one token for ``key``. Returns ``(allowed, retry_after_s, remaining)``."""
        now = self.clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self._max_keys:
                self._prune()
            bucket = self._buckets[key] = _Bucket(tokens=float(self.burst), updated=now)
        else:
            elapsed = max(0.0, now - bucket.updated)  # clamp: a swapped clock must not drain
            bucket.tokens = min(float(self.burst), bucket.tokens + elapsed * self._rate)
            bucket.updated = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0.0, int(bucket.tokens)
        retry_after = (1.0 - bucket.tokens) / self._rate
        return False, retry_after, 0

    def _prune(self) -> None:
        stale = sorted(self._buckets.items(), key=lambda item: item[1].updated)
        for key, _ in stale[: max(1, len(stale) // 2)]:
            del self._buckets[key]

    def __len__(self) -> int:
        return len(self._buckets)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce the limiter for every request under the API prefix."""

    def __init__(self, app, *, limiter: RateLimiter, api_prefix: str) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._protected = api_prefix.rstrip("/") + "/"

    async def dispatch(self, request: Request, call_next) -> Response:
        if not request.url.path.startswith(self._protected):
            return await call_next(request)

        allowed, retry_after, remaining = self._limiter.check(_key_of(request))
        if not allowed:
            return _too_many_requests(request, retry_after, self._limiter.per_minute)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limiter.per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


def _key_of(request: Request) -> str:
    """Strongest identity available: the authenticated subject, else the client IP."""
    auth = getattr(request.state, "auth", None)
    if auth is not None:
        return f"sub:{auth.subject}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return "ip:" + forwarded.split(",")[0].strip()
    return "ip:" + (request.client.host if request.client else "unknown")


def _too_many_requests(request: Request, retry_after: float, limit: int) -> JSONResponse:
    return problem_response(
        429,
        "Too Many Requests",
        "Rate limit exceeded; retry after the indicated delay.",
        getattr(request.state, "correlation_id", "unknown"),
        headers={
            "Retry-After": str(math.ceil(retry_after)),
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": "0",
        },
    )


def install_rate_limiting(app: FastAPI, settings: Settings) -> None:
    """Install the limiter when configured. ``RATE_LIMIT_PER_MINUTE=0`` (default) = off.

    The limiter instance is exposed as ``app.state.rate_limiter`` (``None`` when off) so
    ops/tests can inspect it.
    """
    if settings.rate_limit_per_minute <= 0:
        app.state.rate_limiter = None
        return
    limiter = RateLimiter(
        settings.rate_limit_per_minute,
        settings.rate_limit_burst or settings.rate_limit_per_minute,
    )
    app.state.rate_limiter = limiter
    app.add_middleware(RateLimitMiddleware, limiter=limiter, api_prefix=settings.api_prefix)
