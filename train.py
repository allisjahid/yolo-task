"""
train.py
========
Fine-tune YOLOv8 on your grocery dataset.

Steps before running:
    1. Download a grocery dataset from https://universe.roboflow.com
       (search "grocery detection" → export YOLOv8 format)
    2. Place it at:  data/grocery/
                         ├── data.yaml
                         ├── train/images/  train/labels/
                         └── valid/images/  valid/labels/

Usage:
    python train.py                                      # defaults
    python train.py --epochs 100 --model yolov8s.pt     # bigger model
    python train.py --data data/grocery/data.yaml --epochs 50 --device cpu

After training:
    Best weights → model/best.pt  (auto-copied)
    Results      → runs/detect/grocery_run/
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on grocery dataset")
    p.add_argument("--data",   default="data/grocery/data.yaml",
                   help="Path to dataset data.yaml")
    p.add_argument("--model",  default="yolov8n.pt",
                   help="Base model: yolov8n.pt | yolov8s.pt | yolov8m.pt")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs")
    p.add_argument("--imgsz",  type=int, default=640,
                   help="Input image size")
    p.add_argument("--batch",  type=int, default=16,
                   help="Batch size (-1 = auto)")
    p.add_argument("--device", default="0",
                   help="GPU id (0, 1, ...) or 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Check ultralytics ───────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("❌  Run first:  pip install ultralytics")

    # ── Check dataset ───────────────────────────────────────────────────
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"\n❌  Dataset not found: {data_yaml}\n\n"
            "Steps to get a dataset:\n"
            "  1. Go to https://universe.roboflow.com\n"
            "  2. Search 'grocery detection'\n"
            "  3. Download → YOLOv8 format\n"
            "  4. Extract to data/grocery/\n"
        )

    print("\n" + "="*55)
    print("  YOLOv8 Grocery Detector — Training")
    print("="*55)
    print(f"  Base model : {args.model}")
    print(f"  Dataset    : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print("="*55 + "\n")

    # ── Train ───────────────────────────────────────────────────────────
    model = YOLO(args.model)

    model.train(
        data    = str(data_yaml),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        name    = "grocery_run",
        project = "runs/detect",
        patience= 15,       # early stopping — stops if no improvement for 15 epochs
        augment = True,     # mosaic, HSV shift, flip — improves generalization
        save    = True,
        plots   = True,
    )

    # ── Copy best weights ────────────────────────────────────────────────
    src = Path("runs/detect/grocery_run/weights/best.pt")
    dst = Path("model/best.pt")
    dst.parent.mkdir(exist_ok=True)

    if src.exists():
        shutil.copy(src, dst)
        print(f"\n✅  Best weights copied → {dst}")
        print("    Restart the API and it will auto-load your trained model.\n")
    else:
        print(f"\n⚠️   Could not find {src}")
        print(f"    Check runs/detect/grocery_run/weights/ manually.\n")


if __name__ == "__main__":
    main()
