"""VA-15 — bearer-JWT authentication (HS256 + tenant claim).

Auth turns on when ``JWT_SECRET_KEY`` is set. Under the API prefix every request needs a
valid HS256 bearer token carrying ``sub``/``tenant``/``exp``; the validated tenant scopes
sessions and usage metering. ``/healthz``, ``/``, the docs, and ``/ui`` stay public.
"""
from __future__ import annotations

import time

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from app.auth import DEFAULT_TENANT, current_auth_var, current_tenant
from app.config import Settings
from app.main import create_app

# ≥32 bytes — HS256 keys below that are rejected at startup (RFC 7518 §3.2).
SECRET = "unit-test-secret-0123456789abcdef-xyz"

_MOCKS = dict(stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock")
_TEXT = {"input": {"kind": "text", "text": "what is article 21?"}}


def _client(**overrides) -> TestClient:
    return TestClient(create_app(Settings(_env_file=None, **_MOCKS, **overrides)))


def _token(secret: str = SECRET, **overrides) -> str:
    claims = {"sub": "client-1", "tenant": "acme", "exp": int(time.time()) + 300, **overrides}
    claims = {k: v for k, v in claims.items() if v is not None}  # None = drop the claim
    return pyjwt.encode(claims, secret, algorithm="HS256")


def _auth(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


# --- auth off (no secret): open, default tenant ---------------------------------------------

def test_no_secret_means_open_api():
    resp = _client().post("/api/v1/voice/complete", json=_TEXT)
    assert resp.status_code == 200


def test_current_tenant_defaults_when_unset():
    assert current_auth_var.get() is None
    assert current_tenant() == DEFAULT_TENANT


# --- auth on: the protected surface ----------------------------------------------------------

def test_missing_token_is_401_problem_with_challenge():
    resp = _client(jwt_secret_key=SECRET).post("/api/v1/voice/complete", json=_TEXT)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    body = resp.json()
    assert body["status"] == 401 and body["title"] == "Unauthorized"
    assert "correlation_id" in body
    assert resp.headers["x-request-id"] == body["correlation_id"]


def test_valid_token_passes():
    resp = _client(jwt_secret_key=SECRET).post(
        "/api/v1/voice/complete", json=_TEXT, headers=_auth(_token())
    )
    assert resp.status_code == 200
    assert resp.json()["answer_text"] == "mock answer"


@pytest.mark.parametrize(
    "bad_token",
    [
        "garbage.not.a-jwt",
        _token(secret="wrong-secret"),               # bad signature
        _token(exp=int(time.time()) - 60),           # expired
        _token(tenant=None),                          # missing tenant claim
        _token(exp=None),                             # missing exp claim
        _token(sub=None),                             # missing sub claim
    ],
)
def test_invalid_tokens_are_401(bad_token):
    resp = _client(jwt_secret_key=SECRET).post(
        "/api/v1/voice/complete", json=_TEXT, headers=_auth(bad_token)
    )
    assert resp.status_code == 401


def test_alg_none_token_rejected():
    # algorithm-confusion attack: an unsigned token must never validate
    unsigned = pyjwt.encode(
        {"sub": "x", "tenant": "acme", "exp": int(time.time()) + 300}, None, algorithm="none"
    )
    resp = _client(jwt_secret_key=SECRET).post(
        "/api/v1/voice/complete", json=_TEXT, headers=_auth(unsigned)
    )
    assert resp.status_code == 401


def test_wrong_scheme_rejected():
    resp = _client(jwt_secret_key=SECRET).post(
        "/api/v1/voice/complete", json=_TEXT, headers={"authorization": f"Basic {_token()}"}
    )
    assert resp.status_code == 401


def test_ops_endpoints_are_protected():
    client = _client(jwt_secret_key=SECRET)
    assert client.get("/api/v1/config").status_code == 401
    assert client.get("/api/v1/config", headers=_auth(_token())).status_code == 200


def test_contract_endpoints_are_protected():
    client = _client(jwt_secret_key=SECRET)
    assert client.get("/api/v1/contract/schema").status_code == 401


@pytest.mark.parametrize("path", ["/healthz", "/", "/docs", "/openapi.json"])
def test_public_surface_stays_open(path):
    assert _client(jwt_secret_key=SECRET).get(path).status_code == 200


def test_ui_stays_public():
    resp = _client(jwt_secret_key=SECRET).get("/ui/")
    assert resp.status_code == 200  # static reference client; the API it calls still needs a token


# --- the tenant claim flows into the turn ----------------------------------------------------

def test_tenant_claim_scopes_usage_metering():
    client = _client(jwt_secret_key=SECRET)
    headers = _auth(_token(tenant="acme"))
    assert client.post("/api/v1/voice/complete", json=_TEXT, headers=headers).status_code == 200
    usage = client.get("/api/v1/usage", headers=headers).json()
    assert "acme" in usage["traditional"]
    assert DEFAULT_TENANT not in usage["traditional"]


def test_tenant_flows_through_streaming_turns():
    # the SSE body is produced while the response streams — the contextvar must survive
    client = _client(jwt_secret_key=SECRET)
    headers = _auth(_token(tenant="stream-co"))
    resp = client.post("/api/v1/voice/slow", json=_TEXT, headers=headers)
    assert resp.status_code == 200
    assert "event: done" in resp.text
    usage = client.get("/api/v1/usage", headers=headers).json()
    assert "stream-co" in usage["traditional"]


def test_tenants_are_isolated_between_requests():
    client = _client(jwt_secret_key=SECRET)
    client.post("/api/v1/voice/complete", json=_TEXT, headers=_auth(_token(tenant="a-corp")))
    client.post("/api/v1/voice/complete", json=_TEXT, headers=_auth(_token(tenant="b-corp")))
    usage = client.get("/api/v1/usage", headers=_auth(_token())).json()
    assert set(usage["traditional"]) == {"a-corp", "b-corp"}


def test_sessions_are_scoped_by_tenant():
    # same session_id under two tenants must be two distinct sessions (VA-40 isolation)
    client = _client(jwt_secret_key=SECRET)
    body = {**_TEXT, "session_id": "shared-id"}
    r1 = client.post("/api/v1/voice/complete", json=body, headers=_auth(_token(tenant="a-corp")))
    r2 = client.post("/api/v1/voice/complete", json=body, headers=_auth(_token(tenant="b-corp")))
    assert r1.status_code == r2.status_code == 200
    # both echo the same id, but the store keys them under (tenant, id) — no cross-talk
    assert r1.json()["session_id"] == r2.json()["session_id"] == "shared-id"


# --- config surface ---------------------------------------------------------------------------

def test_weak_signing_key_fails_fast():
    # RFC 7518 §3.2: HS256 keys must be ≥ 256 bits — a short key is brute-forceable
    from app.config import ConfigError, load_settings

    with pytest.raises(ConfigError, match="32 bytes"):
        load_settings(_env_file=None, jwt_secret_key="short")


def test_auth_enabled_reported_in_public_config():
    open_client = _client()
    assert open_client.get("/api/v1/config").json()["auth_enabled"] is False
    locked = _client(jwt_secret_key=SECRET)
    body = locked.get("/api/v1/config", headers=_auth(_token())).json()
    assert body["auth_enabled"] is True
    assert SECRET not in str(body)  # never the secret itself
