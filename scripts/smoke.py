"""Endpoint smoke test (VA-67).

Boots the service in-process (mock providers, no network) — or targets a running server with
``--base-url`` — and checks the key endpoints respond correctly. It is a fast, black-box gate
for CI on every PR, and is reusable as the post-deploy smoke once the deploy pipeline lands
(VA-12).

    python -m scripts.smoke                          # in-process app, mock providers
    python -m scripts.smoke --base-url http://host   # against a running server

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

API = "/api/v1"
_AUDIO_B64 = base64.b64encode(b"\x00\x01\x02").decode()
_TEXT_TURN = {"input": {"kind": "text", "text": "what does article 21 guarantee?"}}
_AUDIO_TURN = {"input": {"kind": "audio", "audio_b64": _AUDIO_B64}}


@dataclass
class Response:
    status: int
    content_type: str
    text: str


class InProcessClient:
    """Runs checks against an in-process app with mock providers (no network)."""

    def __init__(self) -> None:
        from fastapi.testclient import TestClient

        from app.config import Settings
        from app.main import create_app

        settings = Settings(
            _env_file=None,
            stt_provider="mock",
            llm_provider="mock",
            tts_provider="mock",
            realtime_provider="mock",
        )
        self._client = TestClient(create_app(settings))

    def get(self, path: str) -> Response:
        r = self._client.get(path)
        return Response(r.status_code, r.headers.get("content-type", ""), r.text)

    def post(self, path: str, body: dict) -> Response:
        r = self._client.post(path, json=body)
        return Response(r.status_code, r.headers.get("content-type", ""), r.text)


class UrlClient:
    """Runs checks against a running server over HTTP (stdlib only)."""

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")

    def _do(self, path: str, body: dict | None) -> Response:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        req = urllib.request.Request(
            self._base + path, data=data, method="POST" if body is not None else "GET", headers=headers
        )
        # base_url is operator-supplied (CLI flag), targeting the service's own endpoints.
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                ctype = resp.headers.get("content-type", "")
                return Response(resp.status, ctype, resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            return Response(exc.code, exc.headers.get("content-type", ""), exc.read().decode("utf-8", "replace"))

    def get(self, path: str) -> Response:
        return self._do(path, None)

    def post(self, path: str, body: dict) -> Response:
        return self._do(path, body)


def run_checks(client) -> list[tuple[str, bool, str]]:
    """Return ``(name, passed, detail)`` for each endpoint check."""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: object, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    r = client.get("/healthz")
    check("GET /healthz → 200 {ok:true}", r.status == 200 and '"ok":true' in r.text.replace(" ", ""), f"status={r.status}")

    r = client.post(f"{API}/voice/complete", _TEXT_TURN)
    ok = r.status == 200 and json.loads(r.text or "{}").get("answer_text")
    check("POST /voice/complete → JSON answer", ok, f"status={r.status}")

    r = client.post(f"{API}/voice/slow", _TEXT_TURN)
    check(
        "POST /voice/slow → SSE ending in done",
        r.status == 200 and "text/event-stream" in r.content_type and "event: done" in r.text,
        f"status={r.status}",
    )

    r = client.post(f"{API}/voice/fast", _AUDIO_TURN)
    check("POST /voice/fast → SSE audio.chunk", r.status == 200 and "event: audio.chunk" in r.text, f"status={r.status}")

    r = client.post(f"{API}/voice/complete", {"session_id": "x"})  # missing input
    check("POST /voice/complete (bad body) → 422", r.status == 422, f"status={r.status}")

    for path in (f"{API}/config", f"{API}/metrics", f"{API}/usage", f"{API}/counters"):
        r = client.get(path)
        check(f"GET {path} → 200", r.status == 200, f"status={r.status}")

    r = client.get("/openapi.json")
    check("GET /openapi.json → 200", r.status == 200, f"status={r.status}")

    r = client.get("/ui/")
    check("GET /ui/ → dashboard", r.status == 200 and "VANI" in r.text, f"status={r.status}")

    return results


def run(base_url: str | None = None) -> int:
    client = UrlClient(base_url) if base_url else InProcessClient()
    print(f"smoke: {base_url or 'in-process app (mock providers)'}")
    results = run_checks(client)
    for name, ok, detail in results:
        line = f"  {'PASS' if ok else 'FAIL'}  {name}"
        print(line + (f"  [{detail}]" if not ok else ""))
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke")
    parser.add_argument("--base-url", default=None, help="target a running server instead of the in-process app")
    args = parser.parse_args(argv)
    return run(args.base_url)


if __name__ == "__main__":
    sys.exit(main())
