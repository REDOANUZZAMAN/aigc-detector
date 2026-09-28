import importlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

REAL_CLIENT = httpx.AsyncClient  # captured before any monkeypatching

BACKEND = {
    "likelihood": 0.995, "verdict": {"key": "likely_ai", "label": "Likely AI-generated"},
    "bands": {"likely_ai": 0.99, "likely_real": 0.1}, "model": "m", "model_version": "v",
    "image": {"width": 256, "height": 256, "analyzed_width": 256, "analyzed_height": 256},
    "grid": 4, "tiles": [], "indicators": [],
}


@pytest.fixture
def client(monkeypatch):
    from app import server
    server = importlib.reload(server)
    calls = []

    def handler(req: httpx.Request):
        calls.append(req)
        if req.url.path == "/v1/analyze":
            assert req.headers["authorization"] == "Bearer backend-key"
            return httpx.Response(200, json=BACKEND)
        if req.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"status": "warming"})

    real = REAL_CLIENT
    monkeypatch.setattr(server.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    c = TestClient(server.app)
    c.calls, c.server = calls, server
    return c


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "AIGC Detector" in r.text


def test_analyze_merges_metadata(client, png):
    r = client.post("/api/analyze", files={"file": ("a.png", png({"parameters": "x"}))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["likelihood"] == 0.995
    assert body["metadata"]["declared_ai"] is True
    assert body["file"]["format"] == "PNG"


def test_rejects_non_image(client):
    r = client.post("/api/analyze", files={"file": ("a.txt", b"hello")})
    assert r.status_code == 415
    assert not client.calls


def test_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr(client.server, "MAX_BYTES", 100)
    r = client.post("/api/analyze", files={"file": ("a.bin", b"x" * 200)})
    assert r.status_code == 413


def test_non_native_format_converted_to_png(client):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "BMP")
    r = client.post("/api/analyze", files={"file": ("a.bmp", buf.getvalue())})
    assert r.status_code == 200
    sent = client.calls[-1].content
    assert b"converted.png" in sent and b"\x89PNG" in sent


def test_rate_limit(client, png, monkeypatch):
    monkeypatch.setattr(client.server, "RATE_LIMIT", 2)
    f = {"file": ("a.png", png())}
    h = {"cf-connecting-ip": "1.2.3.4"}
    assert client.post("/api/analyze", files=f, headers=h).status_code == 200
    assert client.post("/api/analyze", files=f, headers=h).status_code == 200
    r = client.post("/api/analyze", files=f, headers=h)
    assert r.status_code == 429 and "Retry-After" in r.headers
    # A different IP is unaffected; an API key bypasses the limit.
    assert client.post("/api/analyze", files=f, headers={"cf-connecting-ip": "5.6.7.8"}).status_code == 200
    assert client.post("/api/analyze", files=f, headers={**h, "authorization": "Bearer test-api-key"}).status_code == 200


def test_bad_api_key(client, png):
    r = client.post("/api/analyze", files={"file": ("a.png", png())},
                    headers={"authorization": "Bearer nope"})
    assert r.status_code == 401


def test_password_gate(client, monkeypatch):
    monkeypatch.setattr(client.server, "APP_PASSWORD", "s3cret")
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("admin", "s3cret")).status_code == 200
    assert client.get("/api/health").status_code == 200  # healthcheck stays open


def test_backend_error_propagates(client, png, monkeypatch):
    real = REAL_CLIENT
    monkeypatch.setattr(client.server.httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")), **kw))
    r = client.post("/api/analyze", files={"file": ("a.png", png())})
    assert r.status_code == 502
