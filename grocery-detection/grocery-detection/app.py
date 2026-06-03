"""
app.py
======
Grocery Product Detection API — FastAPI

Endpoints:
    POST /detect   → upload image, get JSON detections
    GET  /health   → server + model status

Run:
    uvicorn app:app --reload
    uvicorn app:app --host 0.0.0.0 --port 8000

Test with curl:
    curl -X POST http://localhost:8000/detect -F "file=@image.jpg"

Test with Postman:
    POST http://localhost:8000/detect
    Body → form-data → key: file (File), value: select image
"""

import io
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from detector import GroceryDetector

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global detector — loaded once at startup, shared across all requests
# ---------------------------------------------------------------------------
_detector: GroceryDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global _detector
    logger.info("Starting up — loading model...")
    _detector = GroceryDetector()
    logger.info(f"Model ready | mode = {_detector._mode}")
    yield
    _detector = None
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Grocery Product Detection API",
    description="Upload an image → get detected grocery products as JSON",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(..., description="Image file (jpg / png / webp)")):
    """
    Detect grocery products in the uploaded image.

    **Input:**  multipart/form-data — field name `file`

    **Output:**
    ```json
    {
        "detections": [
            {"class": "rice", "confidence": 0.92},
            {"class": "oil",  "confidence": 0.88}
        ]
    }
    ```
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again.")

    # ── Validate file type ──────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    content_type  = file.content_type or ""
    filename      = (file.filename or "").lower()

    if content_type not in allowed_types and not filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Use JPG, PNG, or WEBP.",
        )

    # ── Read & decode image ─────────────────────────────────────────────
    try:
        raw   = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # ── Run detection ───────────────────────────────────────────────────
    detections = _detector.detect(image, conf=0.25)

    logger.info(f"Detected {len(detections)} object(s) in '{file.filename}'")

    # ── Return exact required format ────────────────────────────────────
    return JSONResponse(content={"detections": detections})


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Check if the server and model are running."""
    return {
        "status":     "ok",
        "model_mode": _detector._mode if _detector else "not loaded",
        "classes":    _detector.classes if _detector else [],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
