"""Smoke test of the import-time module app (`app = create_app()` in app.main)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_module_app_healthz_ok():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
