# ── Dockerfile ──────────────────────────────────────────────────────────────
# Builds a Docker image for the Grocery Detection API
#
# Build:  docker build -t grocery-detector .
# Run:    docker run -p 8000:8000 grocery-detector
# Test:   curl -X POST http://localhost:8000/detect -F "file=@image.jpg"
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System libraries required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only torch keeps image smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      fastapi \
      "uvicorn[standard]" \
      python-multipart \
      Pillow \
 && pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ultralytics

# Pre-download YOLOv8n weights so the container works offline
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Copy application code
COPY detector.py .
COPY app.py      .
COPY model/      ./model/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
