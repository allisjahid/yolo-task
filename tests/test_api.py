"""
tests/test_api.py — Integration tests for the detection API.
Run: pytest tests/ -v
"""

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped client that triggers FastAPI lifespan (model loading)."""
    with TestClient(app) as c:
        yield c


# ── Helpers ────────────────────────────────────────────────────────────────

def _jpeg(w=320, h=240, color=(120, 180, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="JPEG")
    return buf.getvalue()

def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), color=(200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


# ── Health ─────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_root_returns_html(client):
    assert client.get("/").status_code == 200

# ── /classes ───────────────────────────────────────────────────────────────

def test_classes_endpoint(client):
    r = client.get("/classes")
    assert r.status_code == 200
    body = r.json()
    assert "classes" in body and isinstance(body["classes"], list) and len(body["classes"]) > 0

# ── /detect — happy paths ──────────────────────────────────────────────────

def test_detect_jpeg(client):
    r = client.post("/detect", files={"file": ("test.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
    assert r.status_code == 200
    _check(r.json())

def test_detect_png(client):
    r = client.post("/detect", files={"file": ("test.png", io.BytesIO(_png()), "image/png")})
    assert r.status_code == 200
    _check(r.json())

def test_detect_custom_confidence(client):
    r = client.post("/detect?confidence=0.5", files={"file": ("test.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
    assert r.status_code == 200

def test_detect_response_structure(client):
    r = client.post("/detect", files={"file": ("shelf.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
    body = r.json()
    assert "detections" in body
    assert "inference_ms" in body
    assert body["filename"] == "shelf.jpg"

def test_detection_fields(client):
    r = client.post("/detect", files={"file": ("test.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
    for det in r.json()["detections"]:
        assert "class" in det and "confidence" in det
        assert 0.0 <= det["confidence"] <= 1.0

# ── /detect — error paths ──────────────────────────────────────────────────

def test_detect_no_file(client):
    assert client.post("/detect").status_code == 422

def test_confidence_too_high(client):
    r = client.post("/detect?confidence=1.5", files={"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
    assert r.status_code == 422

def test_confidence_too_low(client):
    r = client.post("/detect?confidence=0.0", files={"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
    assert r.status_code == 422

def test_corrupt_image(client):
    r = client.post("/detect", files={"file": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")})
    assert r.status_code == 400

# ── Helper ─────────────────────────────────────────────────────────────────

def _check(body):
    assert "detections" in body
    assert body["model_mode"] in ("fine-tuned", "coco", "mock")
    assert len(body["image_size"]) == 2
