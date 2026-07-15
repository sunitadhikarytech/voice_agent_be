"""VA-16 — CORS lockdown: only explicitly configured origins may call the API cross-origin.

The default posture is *deny*: with no ``ALLOWED_ORIGINS`` configured the app adds no CORS
middleware, so browsers refuse cross-origin reads. Configuring origins opens exactly those
origins — wildcards and scheme-less values are rejected at startup.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigError, Settings, load_settings
from app.main import create_app

ALLOWED = "https://dashboard.example.com"
OTHER = "https://evil.example.com"

_MOCKS = dict(stt_provider="mock", llm_provider="mock", tts_provider="mock", realtime_provider="mock")


def _client(**overrides) -> TestClient:
    return TestClient(create_app(Settings(_env_file=None, **_MOCKS, **overrides)))


# --- default posture: deny -----------------------------------------------------------------

def test_no_config_sends_no_cors_headers():
    resp = _client().get("/healthz", headers={"origin": OTHER})
    assert resp.status_code == 200  # server still serves…
    assert "access-control-allow-origin" not in resp.headers  # …but browsers may not read it


def test_no_config_preflight_is_405_not_cors():
    # without the middleware there is no OPTIONS handler at all
    resp = _client().options(
        "/api/v1/voice/complete",
        headers={"origin": OTHER, "access-control-request-method": "POST"},
    )
    assert resp.status_code == 405
    assert "access-control-allow-origin" not in resp.headers


# --- configured origins --------------------------------------------------------------------

def test_preflight_allows_configured_origin():
    resp = _client(allowed_origins=ALLOWED).options(
        "/api/v1/voice/complete",
        headers={
            "origin": ALLOWED,
            "access-control-request-method": "POST",
            "access-control-request-headers": "authorization,content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == ALLOWED
    assert "POST" in resp.headers["access-control-allow-methods"]


def test_preflight_rejects_unknown_origin():
    resp = _client(allowed_origins=ALLOWED).options(
        "/api/v1/voice/complete",
        headers={"origin": OTHER, "access-control-request-method": "POST"},
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_simple_request_header_only_for_allowed_origin():
    client = _client(allowed_origins=ALLOWED)
    allowed = client.get("/healthz", headers={"origin": ALLOWED})
    assert allowed.headers["access-control-allow-origin"] == ALLOWED
    denied = client.get("/healthz", headers={"origin": OTHER})
    assert "access-control-allow-origin" not in denied.headers


def test_error_responses_carry_cors_for_allowed_origin():
    # an allowed browser client must be able to read problem-shaped errors (VA-28)
    resp = _client(allowed_origins=ALLOWED).post(
        "/api/v1/voice/complete", json={}, headers={"origin": ALLOWED}
    )
    assert resp.status_code == 422
    assert resp.headers["access-control-allow-origin"] == ALLOWED


def test_credentials_are_not_allowed():
    resp = _client(allowed_origins=ALLOWED).options(
        "/api/v1/voice/complete",
        headers={"origin": ALLOWED, "access-control-request-method": "POST"},
    )
    assert "access-control-allow-credentials" not in resp.headers


# --- configuration parsing / validation ----------------------------------------------------

def test_comma_separated_env_value_parses(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", f" {ALLOWED} , https://b.example.com/ ")
    s = Settings(_env_file=None)
    assert s.allowed_origins == [ALLOWED, "https://b.example.com"]


def test_default_is_empty():
    assert Settings(_env_file=None).allowed_origins == []


@pytest.mark.parametrize("bad", ["*", "https://*.example.com", "dashboard.example.com"])
def test_invalid_origins_fail_fast(bad):
    with pytest.raises(ConfigError):
        load_settings(_env_file=None, allowed_origins=bad)


def test_origins_visible_in_public_config():
    body = _client(allowed_origins=ALLOWED).get("/api/v1/config").json()
    assert body["allowed_origins"] == [ALLOWED]
