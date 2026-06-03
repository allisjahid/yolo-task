"""
Grocery Product Detection API
Camera-Based Inventory System using YOLOv8
"""

import io
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import uvicorn

try:
    from detector import GroceryDetector
except ModuleNotFoundError:
    from api.detector import GroceryDetector

# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Lifespan (model load/unload) ──────────────────────────────────────────
detector: Optional[GroceryDetector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Loading YOLOv8 grocery detection model...")
    detector = GroceryDetector()
    logger.info(f"Model loaded | mode={detector._mode} | classes={detector.class_names}")
    yield
    detector = None

# ─── App Setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Grocery Product Detection API",
    description="AI-powered camera-based product detection using YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for demo UI
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ─── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    """Serve demo web UI."""
    ui_file = static_path / "index.html"
    if ui_file.exists():
        return ui_file.read_text()
    return HTMLResponse("<h1>Grocery Detection API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "model_classes": detector.class_names if detector else [],
    }


@app.post("/detect", tags=["Detection"])
async def detect(
    file: UploadFile = File(..., description="Image file (jpg/png/webp)"),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0, description="Min confidence threshold"),
):
    """
    Detect grocery products in an uploaded image.

    Returns JSON with detected objects, confidence scores, and bounding boxes.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    ct = file.content_type or ""
    fname = (file.filename or "").lower()
    if ct not in allowed and not fname.endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ct}")

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    start = time.perf_counter()
    result = detector.detect(image, confidence_threshold=confidence)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    logger.info(f"Detected {len(result['detections'])} objects in {elapsed_ms}ms | file={file.filename}")

    return {
        **result,
        "inference_ms": elapsed_ms,
        "filename": file.filename,
    }


@app.get("/classes", tags=["Detection"])
async def list_classes():
    """Return all detectable product classes."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"classes": detector.class_names, "count": len(detector.class_names)}


# ─── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
