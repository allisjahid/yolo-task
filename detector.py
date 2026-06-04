"""
detector.py
===========
YOLOv8 Grocery Product Detector

3-tier loading strategy:
  Tier 1 → model/best.pt        (your fine-tuned grocery model)
  Tier 2 → yolov8n.pt (COCO)   (pretrained, filters grocery classes)
  Tier 3 → Mock mode            (for API testing without GPU/model)

Usage:
    from detector import GroceryDetector
    d = GroceryDetector()
    results = d.detect(pil_image, conf=0.25)
    # [{"class": "rice", "confidence": 0.92}, ...]
"""

from __future__ import annotations
from pathlib import Path
import logging

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class IDs that map to grocery items
# When using the pretrained COCO model we only keep these classes
# ---------------------------------------------------------------------------
COCO_GROCERY_MAP = {
    39: "oil",          # bottle      → oil / sauce bottle
    45: "rice",         # bowl        → rice bowl
    46: "banana",
    47: "apple",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    79: "soap",         # toothbrush  → soap / hygiene product
}

# Classes expected from a fine-tuned grocery model
GROCERY_CLASSES = [
    "rice", "oil", "soap", "detergent",
    "water_bottle", "juice", "snack_packet", "bread",
]

BEST_PT = Path(__file__).parent / "model" / "best.pt"


class GroceryDetector:
    """
    Wraps a YOLOv8 model and exposes a single .detect() method.
    Auto-selects the best available model tier on init.
    """

    def __init__(self):
        self._model  = None
        self._mode   = "mock"       # "fine-tuned" | "coco" | "mock"
        self.classes = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed → MOCK mode")
            self.classes = GROCERY_CLASSES
            self._mode   = "mock"
            return

        # Tier 1 — fine-tuned weights
        if BEST_PT.exists():
            logger.info(f"Loading fine-tuned model: {BEST_PT}")
            self._model  = YOLO(str(BEST_PT))
            self.classes = list(self._model.names.values())
            self._mode   = "fine-tuned"
            logger.info(f"✅ Fine-tuned model ready | classes: {self.classes}")
            return

        # Tier 2 — COCO pretrained (downloads ~6 MB once)
        logger.info("best.pt not found → loading COCO pretrained yolov8n.pt")
        try:
            self._model  = YOLO("yolov8n.pt")
            self.classes = list(COCO_GROCERY_MAP.values())
            self._mode   = "coco"
            logger.info("✅ COCO model ready (grocery class filter active)")
        except Exception as e:
            logger.warning(f"YOLO load failed ({e}) → MOCK mode")
            self.classes = GROCERY_CLASSES
            self._mode   = "mock"

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, image: Image.Image, conf: float = 0.25) -> list[dict]:
        """
        Run detection on a PIL RGB image.

        Args:
            image : PIL.Image.Image  — the input image
            conf  : float            — minimum confidence threshold (0–1)

        Returns:
            List of dicts sorted by confidence descending:
            [{"class": "rice", "confidence": 0.92}, ...]
        """
        if self._mode == "mock":
            return self._mock_detections(conf)

        results = self._model.predict(image, conf=conf, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_score = round(float(box.conf[0]), 4)

            if self._mode == "fine-tuned":
                label = self._model.names.get(cls_id, f"class_{cls_id}")
            else:
                # COCO mode — skip non-grocery classes
                if cls_id not in COCO_GROCERY_MAP:
                    continue
                label = COCO_GROCERY_MAP[cls_id]

            detections.append({
                "class":      label,
                "confidence": conf_score,
            })

        # Sort highest confidence first
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)

    # ------------------------------------------------------------------
    # Mock (no model needed — for testing)
    # ------------------------------------------------------------------
    def _mock_detections(self, conf: float) -> list[dict]:
        candidates = [
            {"class": "rice",         "confidence": 0.92},
            {"class": "oil",          "confidence": 0.88},
            {"class": "soap",         "confidence": 0.81},
            {"class": "detergent",    "confidence": 0.76},
            {"class": "snack_packet", "confidence": 0.69},
        ]
        return [d for d in candidates if d["confidence"] >= conf]
