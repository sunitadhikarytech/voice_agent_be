"""VA-01 scaffold tests: the app boots and the liveness probe answers."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz_ok():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_root_reports_service():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "voice-ai-agent"
