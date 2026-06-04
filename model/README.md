# model/

Place your trained YOLOv8 weights here:

- `best.pt`  — fine-tuned grocery model (from `scripts/train.py`)

If this file is missing, the API will fall back to:
1. COCO-pretrained YOLOv8n (auto-downloaded, grocery classes only)
2. Mock detector (if ultralytics is not installed)
