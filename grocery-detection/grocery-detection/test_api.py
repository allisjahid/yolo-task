"""
test_api.py
===========
Integration tests for POST /detect API.

Run:
    pytest test_api.py -v
    pytest test_api.py -v -s        # show print output too
"""

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Allow import from project root
sys.path.insert(0, str(Path(__file__).parent))

from app import app


# ---------------------------------------------------------------------------
# Session-scoped client — loads model ONCE for all tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:   # triggers lifespan → loads model
        yield c


# ---------------------------------------------------------------------------
# Helper: make in-memory images
# ---------------------------------------------------------------------------
def make_jpeg(w=320, h=240, color=(80, 160, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def make_png(w=200, h=200, color=(200, 100, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_has_status_field(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_model_mode_present(self, client):
        body = client.get("/health").json()
        assert "model_mode" in body
        assert body["model_mode"] in ("fine-tuned", "coco", "mock")

    def test_classes_present(self, client):
        body = client.get("/health").json()
        assert "classes" in body
        assert isinstance(body["classes"], list)
        assert len(body["classes"]) > 0


# ---------------------------------------------------------------------------
# POST /detect — happy paths
# ---------------------------------------------------------------------------
class TestDetectSuccess:
    def test_jpeg_upload_returns_200(self, client):
        r = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg()), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_png_upload_returns_200(self, client):
        r = client.post(
            "/detect",
            files={"file": ("test.png", io.BytesIO(make_png()), "image/png")},
        )
        assert r.status_code == 200

    def test_response_has_detections_key(self, client):
        r = client.post(
            "/detect",
            files={"file": ("shelf.jpg", io.BytesIO(make_jpeg()), "image/jpeg")},
        )
        body = r.json()
        assert "detections" in body, f"Missing 'detections' key. Got: {body}"

    def test_detections_is_a_list(self, client):
        r = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg()), "image/jpeg")},
        )
        assert isinstance(r.json()["detections"], list)

    def test_each_detection_has_class_and_confidence(self, client):
        r = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg()), "image/jpeg")},
        )
        for det in r.json()["detections"]:
            assert "class"      in det, f"Missing 'class' in {det}"
            assert "confidence" in det, f"Missing 'confidence' in {det}"

    def test_confidence_is_float_between_0_and_1(self, client):
        r = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg()), "image/jpeg")},
        )
        for det in r.json()["detections"]:
            assert isinstance(det["confidence"], float)
            assert 0.0 <= det["confidence"] <= 1.0, \
                f"Confidence out of range: {det['confidence']}"

    def test_class_is_string(self, client):
        r = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg()), "image/jpeg")},
        )
        for det in r.json()["detections"]:
            assert isinstance(det["class"], str)
            assert len(det["class"]) > 0

    def test_exact_output_format(self, client):
        """Verify the response matches the exact required format."""
        r = client.post(
            "/detect",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg()), "image/jpeg")},
        )
        body = r.json()
        # Must have ONLY detections at the top level (+ status code 200)
        assert "detections" in body
        # Each detection: exactly class + confidence (at minimum)
        for det in body["detections"]:
            assert set(det.keys()) >= {"class", "confidence"}


# ---------------------------------------------------------------------------
# POST /detect — error paths
# ---------------------------------------------------------------------------
class TestDetectErrors:
    def test_no_file_returns_422(self, client):
        r = client.post("/detect")
        assert r.status_code == 422

    def test_corrupt_bytes_returns_400(self, client):
        r = client.post(
            "/detect",
            files={"file": ("bad.jpg", io.BytesIO(b"this is not an image"), "image/jpeg")},
        )
        assert r.status_code == 400

    def test_wrong_file_type_returns_415(self, client):
        r = client.post(
            "/detect",
            files={"file": ("doc.pdf", io.BytesIO(b"%PDF-fake"), "application/pdf")},
        )
        assert r.status_code == 415

    def test_empty_file_returns_400(self, client):
        r = client.post(
            "/detect",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert r.status_code == 400
