"""VA-19 — app factory, settings injection, and bootstrap endpoints."""
import time

import jwt as pyjwt
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

# ≥32 bytes — HS256 keys below that are rejected at startup (VA-15 / RFC 7518 §3.2).
TEST_SECRET = "topsecret-0123456789abcdef-0123456789abcdef"


def _client(**settings_kwargs) -> TestClient:
    """A TestClient over an app built with explicit, isolated settings."""
    settings = Settings(_env_file=None, **settings_kwargs)
    return TestClient(create_app(settings))


def _bearer(secret: str = TEST_SECRET) -> dict[str, str]:
    token = pyjwt.encode(
        {"sub": "test", "tenant": "default", "exp": int(time.time()) + 300},
        secret,
        algorithm="HS256",
    )
    return {"authorization": f"Bearer {token}"}


def test_app_boots_and_healthz_is_ok():
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_root_reflects_injected_settings():
    resp = _client(app_name="custom-svc", environment="dev", jwt_secret_key=TEST_SECRET).get("/")
    assert resp.status_code == 200
    assert resp.json() == {"service": "custom-svc", "environment": "dev"}


def test_config_endpoint_is_redacted():
    # setting the secret enables auth (VA-15), so the config view itself needs a token
    resp = _client(jwt_secret_key=TEST_SECRET).get("/api/v1/config", headers=_bearer())
    assert resp.status_code == 200
    body = resp.json()
    assert body["app_name"] == "voice-ai-agent"
    assert body["jwt_secret_key_configured"] is True
    assert TEST_SECRET not in resp.text


def test_config_endpoint_follows_api_prefix():
    client = _client(api_prefix="/api/v2")
    assert client.get("/api/v2/config").status_code == 200
    assert client.get("/api/v1/config").status_code == 404


def test_settings_available_on_app_state():
    settings = Settings(_env_file=None, app_name="stateful")
    app = create_app(settings)
    assert app.state.settings is settings


def test_openapi_lists_bootstrap_routes():
    paths = _client().get("/openapi.json").json()["paths"]
    assert {"/healthz", "/", "/api/v1/config"} <= set(paths)
