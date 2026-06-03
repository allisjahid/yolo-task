"""
validate.py — Evaluate trained model on validation/test set.

Usage:
    python scripts/validate.py
    python scripts/validate.py --model model/best.pt --data data/grocery/data.yaml
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Validate trained grocery detection model")
    p.add_argument("--model", default="model/best.pt",          help="Path to .pt weights")
    p.add_argument("--data",  default="data/grocery/data.yaml", help="Dataset YAML")
    p.add_argument("--imgsz", type=int, default=640,            help="Image size")
    p.add_argument("--conf",  type=float, default=0.25,         help="Confidence threshold")
    p.add_argument("--iou",   type=float, default=0.6,          help="IoU threshold for NMS")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Run: pip install ultralytics")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nTrain first: python scripts/train.py")

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    print(f"\n{'='*55}")
    print(f"  Validating: {model_path.name}")
    print(f"{'='*55}\n")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
    )

    print(f"\n{'='*55}")
    print(f"  Validation Results")
    print(f"{'='*55}")
    print(f"  mAP@0.50      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
