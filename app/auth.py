"""API authentication (VA-15): Bearer JWT (HS256) with a tenant claim.

Authentication is **enabled by configuration**: setting ``JWT_SECRET_KEY`` turns it on.
The key is required in ``dev``/``prod`` (``REQUIRED_IN_CLOUD``), so cloud deployments are
always authenticated; locally it may be left unset for open, offline development — the same
"missing secrets disable the features that need them" posture the rest of the config uses.

Every request under the API prefix must present ``Authorization: Bearer <jwt>`` where the
token is HS256-signed with the shared secret and carries ``sub`` (caller identity),
``tenant`` (the tenant claim), and ``exp``. The algorithm list is pinned to HS256 so
algorithm-confusion tokens (``alg=none`` et al.) are rejected outright. Anything outside the
API prefix — ``/healthz``, ``/``, the OpenAPI docs, the ``/ui`` dashboard assets — stays
public: probes must never depend on auth, and the docs describe the API without granting it.

The validated identity is exposed two ways:

* ``request.state.auth`` — for handlers and downstream middleware;
* a contextvar (:func:`current_tenant`) — for the pipelines, which only ever see the body
  model. The var is set *before* the endpoint task is created, so streaming generators
  (which run while the response body is being written) inherit it.

Failures return the standard problem shape (VA-28) as ``401`` with ``WWW-Authenticate:
Bearer``; the detail names the operational reason (expired / bad signature / missing claim),
never the token itself.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import jwt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings
from app.errors import problem_response

DEFAULT_TENANT = "default"

# Claims a token must carry. ``exp`` is mandatory so leaked tokens age out.
REQUIRED_CLAIMS: tuple[str, ...] = ("sub", "tenant", "exp")


@dataclass(frozen=True, slots=True)
class AuthContext:
    """The validated caller identity attached to an authenticated request."""

    subject: str
    tenant: str


current_auth_var: ContextVar[AuthContext | None] = ContextVar("current_auth", default=None)


def current_tenant(default: str = DEFAULT_TENANT) -> str:
    """The tenant claim of the current request, or ``default`` when auth is off."""
    auth = current_auth_var.get()
    return auth.tenant if auth is not None else default


def decode_token(token: str, secret: str) -> AuthContext:
    """Validate ``token`` and return its identity.

    Raises :class:`jwt.InvalidTokenError` (or a subclass) on any problem: bad signature,
    wrong/none algorithm, expired, or a missing required claim.
    """
    claims = jwt.decode(
        token,
        secret,
        algorithms=["HS256"],  # pinned — never trust the token's own alg header
        options={"require": list(REQUIRED_CLAIMS)},
    )
    return AuthContext(subject=str(claims["sub"]), tenant=str(claims["tenant"]))


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a valid bearer JWT for every request under the API prefix."""

    def __init__(self, app, *, secret: str, api_prefix: str) -> None:
        super().__init__(app)
        self._secret = secret
        self._protected = api_prefix.rstrip("/") + "/"

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith(self._protected):
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return _unauthorized(request, "Missing bearer token.")
        try:
            auth = decode_token(token.strip(), self._secret)
        except jwt.InvalidTokenError as exc:
            # operational reason only (expired / signature / missing claim) — never the token
            return _unauthorized(request, f"Invalid token: {exc}.")

        request.state.auth = auth
        ctx_token = current_auth_var.set(auth)
        try:
            return await call_next(request)
        finally:
            # The streaming body task copied the context when it was spawned inside
            # call_next, so resetting here cannot strip the tenant from an in-flight turn.
            current_auth_var.reset(ctx_token)


def _unauthorized(request: Request, detail: str) -> JSONResponse:
    return problem_response(
        401,
        "Unauthorized",
        detail,
        getattr(request.state, "correlation_id", "unknown"),
        headers={"WWW-Authenticate": "Bearer"},
    )


def install_auth(app: FastAPI, settings: Settings) -> None:
    """Install bearer-JWT auth when a secret is configured; without one (local, offline)
    the API stays open and pipelines fall back to the ``default`` tenant."""
    secret = settings.jwt_secret_key.get_secret_value()
    if not secret:
        return
    app.add_middleware(AuthMiddleware, secret=secret, api_prefix=settings.api_prefix)
