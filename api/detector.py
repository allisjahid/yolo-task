"""
GroceryDetector — wraps Ultralytics YOLOv8 for grocery product detection.

Strategy:
  1. Try to load a fine-tuned grocery model (best.pt / grocery_yolov8.pt).
  2. Fall back to COCO-pretrained YOLOv8n and filter/remap detections
     to grocery-relevant COCO classes.
  3. If ultralytics is unavailable, expose a mock detector for API testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ── COCO class ids that overlap with common grocery items ──────────────────
#   (class_id -> friendly grocery label)
COCO_GROCERY_MAP: dict[int, str] = {
    39: "bottle",       # bottle → oil / sauce bottle
    40: "wine_glass",   # wine glass
    41: "cup",          # cup
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",         # bowl → rice / cereal bowl
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot_dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    67: "cell_phone",
    73: "book",
    76: "scissors",
    79: "toothbrush",   # toothbrush → soap/hygiene product
}

# Custom grocery class names used when a fine-tuned model is present
GROCERY_CLASSES = [
    "rice",
    "oil_bottle",
    "soap",
    "detergent",
    "water_bottle",
    "juice",
    "snack_packet",
    "cereal_box",
    "canned_food",
    "bread",
]


class GroceryDetector:
    """
    Unified detector that works with:
      - Fine-tuned YOLOv8 grocery model  (best.pt)
      - Generic YOLOv8 COCO model        (yolov8n.pt)  [fallback]
      - Mock detector                     [testing fallback]
    """

    MODEL_SEARCH_PATHS = [
        Path(__file__).parent.parent / "model" / "best.pt",
        Path(__file__).parent.parent / "model" / "grocery_yolov8.pt",
    ]

    def __init__(self):
        self._model = None
        self._mode = "mock"  # "fine-tuned" | "coco" | "mock"
        self.class_names: list[str] = []
        self._load()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            logger.warning("ultralytics not installed — running in MOCK mode.")
            self.class_names = GROCERY_CLASSES
            self._mode = "mock"
            return

        # 1. Try fine-tuned grocery model
        for path in self.MODEL_SEARCH_PATHS:
            if path.exists():
                logger.info(f"Loading fine-tuned model: {path}")
                self._model = YOLO(str(path))
                self.class_names = list(self._model.names.values())
                self._mode = "fine-tuned"
                logger.info(f"Fine-tuned model loaded | classes={self.class_names}")
                return

        # 2. Fall back to COCO pretrained
        logger.info("No fine-tuned model found. Falling back to yolov8n (COCO).")
        try:
            self._model = YOLO("yolov8n.pt")  # downloads ~6 MB if needed
            self.class_names = list(COCO_GROCERY_MAP.values())
            self._mode = "coco"
            logger.info("COCO fallback model loaded.")
        except Exception as e:
            logger.warning(f"Could not load COCO model ({e}). Using MOCK mode.")
            self.class_names = GROCERY_CLASSES
            self._mode = "mock"

    # ── Inference ──────────────────────────────────────────────────────────

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """
        Run detection on a PIL image.

        Returns:
            {
              "detections": [{"class": str, "confidence": float, "bbox": [...]}],
              "model_mode": str,
              "image_size": [w, h],
            }
        """
        w, h = image.size
        base = {
            "model_mode": self._mode,
            "image_size": [w, h],
        }

        if self._mode == "mock":
            return {**base, "detections": self._mock_detections(confidence_threshold)}

        if self._mode == "fine-tuned":
            return {**base, "detections": self._run_fine_tuned(image, confidence_threshold)}

        if self._mode == "coco":
            return {**base, "detections": self._run_coco(image, confidence_threshold)}

        return {**base, "detections": []}

    # ── Internal runners ───────────────────────────────────────────────────

    def _run_fine_tuned(self, image: Image.Image, conf: float) -> list[dict]:
        results = self._model.predict(image, conf=conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self._model.names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "class": label,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)

    def _run_coco(self, image: Image.Image, conf: float) -> list[dict]:
        results = self._model.predict(image, conf=conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_GROCERY_MAP:
                continue
            label = COCO_GROCERY_MAP[cls_id]
            detections.append({
                "class": label,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)

    def _mock_detections(self, conf: float) -> list[dict]:
        """Return deterministic mock detections for API testing."""
        import random
        random.seed(42)
        candidates = [
            ("rice", 0.92),
            ("oil_bottle", 0.88),
            ("soap", 0.81),
            ("detergent", 0.76),
            ("snack_packet", 0.69),
        ]
        return [
            {"class": cls, "confidence": round(c + random.uniform(-0.03, 0.03), 4), "bbox": [10, 20, 120, 200]}
            for cls, c in candidates
            if c >= conf
        ]
