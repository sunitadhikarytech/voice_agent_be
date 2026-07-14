"""VA-28 — standardized problem responses + correlation ids."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import REQUEST_ID_HEADER
from app.main import create_app

app = create_app(Settings(_env_file=None))
client = TestClient(app)


def test_validation_error_uses_problem_shape():
    resp = client.post("/api/v1/contract/validate", json={"session_id": "s1"})  # missing input
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == 422
    assert body["title"] == "Unprocessable Entity"
    assert body["correlation_id"]
    assert isinstance(body["errors"], list) and body["errors"]
    # field-level errors expose loc/msg/type only — never the raw input
    assert set(body["errors"][0]) == {"loc", "msg", "type"}


def test_not_found_uses_problem_shape():
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == 404
    assert body["title"] == "Not Found"
    assert body["correlation_id"]


def test_correlation_id_is_generated_and_returned():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.headers.get(REQUEST_ID_HEADER)  # generated when absent


def test_inbound_correlation_id_is_honoured():
    resp = client.post(
        "/api/v1/contract/validate",
        json={"session_id": "s1"},  # invalid -> 422 problem echoes the id
        headers={REQUEST_ID_HEADER: "trace-xyz"},
    )
    assert resp.headers[REQUEST_ID_HEADER] == "trace-xyz"
    assert resp.json()["correlation_id"] == "trace-xyz"


def test_unhandled_exception_is_500_without_stack_trace():
    boom = create_app(Settings(_env_file=None))

    @boom.get("/_boom")
    def _boom():
        raise RuntimeError("secret internal detail")

    # don't re-raise so the registered handler produces the response
    boom_client = TestClient(boom, raise_server_exceptions=False)
    resp = boom_client.get("/_boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["status"] == 500
    assert body["title"] == "Internal Server Error"
    assert body["detail"] == "An unexpected error occurred."
    assert body["correlation_id"]
    # the exception message / traceback must never leak to the client
    assert "secret internal detail" not in resp.text
    assert "Traceback" not in resp.text


def test_problem_schema_documented_in_openapi():
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "Problem" in schemas
