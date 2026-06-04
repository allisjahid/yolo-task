"""
infer.py — Run grocery detection on images from the command line.

Usage:
    python scripts/infer.py --source test_images/shelf.jpg
    python scripts/infer.py --source test_images/          # folder
    python scripts/infer.py --source 0                     # webcam
    python scripts/infer.py --source test_images/shelf.jpg --conf 0.4 --save
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Grocery product detector — inference script")
    p.add_argument("--source", required=True, help="Image path, folder, or webcam index (0)")
    p.add_argument("--model",  default=None,  help="Path to .pt file (default: auto-detect)")
    p.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--save",   action="store_true", help="Save annotated images to runs/")
    p.add_argument("--json",   action="store_true", help="Print raw JSON output only")
    return p.parse_args()


def run_webcam(detector, conf: float):
    """Real-time webcam detection loop."""
    try:
        import cv2
    except ImportError:
        print("opencv-python required for webcam. Run: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam detection running. Press Q to quit.\n")
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgb)

        t0 = time.perf_counter()
        result = detector.detect(pil_img, confidence_threshold=conf)
        ms = (time.perf_counter() - t0) * 1000
        fps_history.append(1000 / ms)
        fps = sum(fps_history[-10:]) / min(len(fps_history), 10)

        # Draw detections
        for det in result["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 100, 100])]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Grocery Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(detector, source: str, conf: float, save: bool, json_only: bool):
    from PIL import Image as PILImage

    paths = []
    p = Path(source)
    if p.is_dir():
        paths = sorted(p.glob("*.*"))
        paths = [x for x in paths if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    elif p.is_file():
        paths = [p]
    else:
        print(f"Source not found: {source}")
        return

    for img_path in paths:
        image = PILImage.open(img_path).convert("RGB")
        t0 = time.perf_counter()
        result = detector.detect(image, confidence_threshold=conf)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        result["inference_ms"] = ms
        result["filename"] = img_path.name

        if json_only:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n📷  {img_path.name}  ({ms} ms)")
            if result["detections"]:
                for d in result["detections"]:
                    bar = "█" * int(d["confidence"] * 20)
                    print(f"  ✅ {d['class']:<20} {d['confidence']:.3f}  {bar}")
            else:
                print("  ⚠️  No products detected (try lowering --conf)")

        if save:
            _save_annotated(image, result, img_path)


def _save_annotated(image, result, src_path: Path):
    try:
        import cv2
        import numpy as np
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for det in result["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 100, 100])]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
        out_dir = Path("runs/infer")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src_path.name
        cv2.imwrite(str(out_path), frame)
        print(f"  💾 Saved → {out_path}")
    except ImportError:
        print("  ℹ️  Install opencv-python to save annotated images.")


def main():
    args = parse_args()

    from api.detector import GroceryDetector
    detector = GroceryDetector()

    if args.source.isdigit():
        run_webcam(detector, args.conf)
    else:
        run_image(detector, args.source, args.conf, args.save, args.json)


if __name__ == "__main__":
    main()
