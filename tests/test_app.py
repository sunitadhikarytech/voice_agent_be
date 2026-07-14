"""VA-19 — app factory, settings injection, and bootstrap endpoints."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _client(**settings_kwargs) -> TestClient:
    """A TestClient over an app built with explicit, isolated settings."""
    settings = Settings(_env_file=None, **settings_kwargs)
    return TestClient(create_app(settings))


def test_app_boots_and_healthz_is_ok():
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_root_reflects_injected_settings():
    resp = _client(app_name="custom-svc", environment="dev", jwt_secret_key="x").get("/")
    assert resp.status_code == 200
    assert resp.json() == {"service": "custom-svc", "environment": "dev"}


def test_config_endpoint_is_redacted():
    resp = _client(jwt_secret_key="topsecret").get("/api/v1/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["app_name"] == "voice-ai-agent"
    assert body["jwt_secret_key_configured"] is True
    assert "topsecret" not in resp.text


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
