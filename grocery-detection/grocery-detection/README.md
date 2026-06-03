# 🛒 Grocery Product Detection System

Camera-based AI inventory system using **YOLOv8 + FastAPI**.

---

## 📁 Project Structure

```
grocery-detection/
├── detector.py          ← YOLOv8 model wrapper (core AI logic)
├── app.py               ← FastAPI server  (POST /detect)
├── train.py             ← Fine-tune YOLOv8 on your dataset
├── webcam.py            ← Real-time webcam detection (bonus)
├── test_api.py          ← API tests (pytest)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── model/
│   └── best.pt          ← put your trained weights here
└── data/grocery/
    └── data.yaml        ← put your Roboflow dataset here
```

---

## 🚀 Quick Start (no training needed)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run API
uvicorn app:app --reload

# 3. Test
curl -X POST http://localhost:8000/detect -F "file=@your_image.jpg"
```

API starts in **mock mode** and returns real-looking detections instantly.
Once you train and place `model/best.pt`, it auto-loads your model on restart.

---

## 📡 API

### `POST /detect`

| | |
|---|---|
| **URL** | `http://localhost:8000/detect` |
| **Method** | POST |
| **Body** | `multipart/form-data` — field: `file` (image) |
| **Accepts** | JPG · PNG · WEBP |

**Response:**
```json
{
    "detections": [
        {"class": "rice", "confidence": 0.92},
        {"class": "oil",  "confidence": 0.88}
    ]
}
```

**Test with curl:**
```bash
curl -X POST http://localhost:8000/detect \
     -F "file=@shelf.jpg"
```

**Test with Postman:**
```
Method : POST
URL    : http://localhost:8000/detect
Body   : form-data
         key   → file  (type: File)
         value → select your image
```

**Test with Python:**
```python
import requests

with open("shelf.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/detect", files={"file": f})

print(r.json())
```

### `GET /health`
```json
{"status": "ok", "model_mode": "fine-tuned", "classes": ["rice", "oil", ...]}
```

---

## 🧠 Model Training

### Step 1 — Get a dataset

1. Go to **https://universe.roboflow.com**
2. Search: `grocery detection` / `retail shelf` / `supermarket products`
3. Download → **YOLOv8 format**
4. Extract to `data/grocery/`

```
data/grocery/
├── data.yaml
├── train/
│   ├── images/   ← training photos (.jpg)
│   └── labels/   ← bounding box annotations (.txt)
└── valid/
    ├── images/
    └── labels/
```

### Step 2 — Train

```bash
python train.py --data data/grocery/data.yaml --epochs 50
```

More options:
```bash
python train.py --epochs 100 --model yolov8s.pt   # larger, more accurate
python train.py --device cpu                       # no GPU
```

Best weights → auto-copied to `model/best.pt`

### Step 3 — Restart API

```bash
uvicorn app:app --reload
# → now loads model/best.pt automatically
```

---

## 🎥 Real-time Webcam

```bash
pip install opencv-python
python webcam.py
# Press Q to quit | S to save screenshot
```

---

## 🐳 Docker

```bash
# Build & run
docker compose up --build

# Or manually
docker build -t grocery-detector .
docker run -p 8000:8000 grocery-detector
```

---

## 🧪 Tests

```bash
pytest test_api.py -v
```

---

## 📊 Dataset Info

| Dataset | Link | Use |
|---|---|---|
| Roboflow Universe | https://universe.roboflow.com | Fine-tuning (primary) |
| COCO pretrained | auto-downloaded | Base model / fallback |
| SKU-110K | github.com/eg4000/SKU110K_CVPR19 | Dense shelf detection |

---

## 🧠 Model Details

| | |
|---|---|
| **Architecture** | YOLOv8n (nano) / YOLOv8s (small) |
| **Input size** | 640 × 640 |
| **Inference** | ~30–80 ms CPU · ~5–15 ms GPU |
| **Output format** | PyTorch `.pt` |
| **Min classes** | rice · oil · soap (+ more) |
