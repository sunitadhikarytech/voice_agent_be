"""VA-51..56 — the reference dashboard is served and self-contained."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def _client() -> TestClient:
    return TestClient(create_app(Settings(_env_file=None)))


def test_assets_exist():
    for name in ("index.html", "app.js", "styles.css"):
        assert (FRONTEND / name).is_file()


def test_dashboard_is_served_at_ui():
    resp = _client().get("/ui/")
    assert resp.status_code == 200
    assert "Voice AI Agent" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_app_js_is_served():
    resp = _client().get("/ui/app.js")
    assert resp.status_code == 200
    # references the endpoints and the SSE events it consumes
    assert "/voice/" in resp.text
    assert "audio.chunk" in resp.text


def test_client_is_self_contained_no_external_hosts():
    html = (FRONTEND / "index.html").read_text()
    # no CDN / external script or style hosts (CSP-friendly, offline)
    assert "http://" not in html and "https://" not in html
