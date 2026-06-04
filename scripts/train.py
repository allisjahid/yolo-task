"""
train.py — Fine-tune YOLOv8 on a Roboflow grocery dataset.

Usage:
    python scripts/train.py --data data/grocery/data.yaml --epochs 50
    python scripts/train.py --data data/grocery/data.yaml --epochs 100 --model yolov8s.pt

After training, copy the best weights:
    cp runs/detect/train/weights/best.pt model/best.pt
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on grocery dataset")
    p.add_argument("--data",    default="data/grocery/data.yaml", help="Path to dataset YAML")
    p.add_argument("--model",   default="yolov8n.pt",             help="Base YOLO model to fine-tune")
    p.add_argument("--epochs",  type=int, default=50,             help="Number of training epochs")
    p.add_argument("--imgsz",   type=int, default=640,            help="Input image size")
    p.add_argument("--batch",   type=int, default=16,             help="Batch size (-1 = auto)")
    p.add_argument("--device",  default="0",                      help="GPU id or 'cpu'")
    p.add_argument("--name",    default="grocery_v1",             help="Run name under runs/detect/")
    p.add_argument("--project", default="runs/detect",            help="Project directory")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed.\n"
            "Run: pip install ultralytics"
        )

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"Dataset YAML not found: {data_yaml}\n"
            "Download a Roboflow dataset and point --data to its data.yaml.\n"
            "See README.md for instructions."
        )

    print(f"\n{'='*60}")
    print(f"  YOLOv8 Grocery Detector Training")
    print(f"{'='*60}")
    print(f"  Base model : {args.model}")
    print(f"  Dataset    : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=15,           # early stopping
        save=True,
        plots=True,
        verbose=True,
        augment=True,          # built-in mosaic, HSV, flip augmentations
    )

    # ── Copy best weights to model/ directory ────────────────────────────
    best_src = Path(args.project) / args.name / "weights" / "best.pt"
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(exist_ok=True)
    best_dst = model_dir / "best.pt"

    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\n✅ Best weights saved to: {best_dst}")
    else:
        print(f"\n⚠️  Could not find {best_src}. Check {args.project}/{args.name}/weights/")

    print(f"\nTraining complete. Results in: {args.project}/{args.name}/")
    print("Run validation with:  python scripts/validate.py")
    print("Run the API with:     uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
