"""
webcam.py
=========
Real-time grocery product detection using your webcam.

Requirements:
    pip install opencv-python

Usage:
    python webcam.py
    python webcam.py --conf 0.4       # stricter confidence
    python webcam.py --camera 1       # use second camera

Controls:
    Q  →  quit
    S  →  save current frame as screenshot
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from PIL import Image
from detector import GroceryDetector


def parse_args():
    p = argparse.ArgumentParser(description="Real-time grocery detector (webcam)")
    p.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--camera", type=int,   default=0,    help="Camera index (0=default)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Import OpenCV ───────────────────────────────────────────────────
    try:
        import cv2
    except ImportError:
        raise SystemExit("❌  Run:  pip install opencv-python")

    # ── Load detector ───────────────────────────────────────────────────
    print("Loading model...")
    detector = GroceryDetector()
    print(f"✅  Model ready | mode = {detector._mode}")
    print("Press Q to quit | S to save screenshot\n")

    # ── Open webcam ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"❌  Cannot open camera index {args.camera}")

    # ── Colors per class (BGR) ───────────────────────────────────────────
    COLORS = [
        (0, 255, 120),   # green
        (255, 180, 0),   # blue-yellow
        (0, 180, 255),   # orange
        (255, 0, 180),   # pink
        (120, 255, 0),   # lime
        (0, 120, 255),   # orange-red
        (255, 255, 0),   # cyan
        (180, 0, 255),   # purple
    ]
    color_map = {}

    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  Failed to read frame from camera.")
            break

        # ── Convert frame to PIL, run detection ─────────────────────────
        t0        = time.perf_counter()
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        dets      = detector.detect(pil_image, conf=args.conf)
        ms        = (time.perf_counter() - t0) * 1000

        # ── FPS calculation ──────────────────────────────────────────────
        fps_history.append(1000 / max(ms, 1))
        fps = sum(fps_history[-10:]) / min(len(fps_history), 10)

        # ── Draw detections ──────────────────────────────────────────────
        h_frame, w_frame = frame.shape[:2]

        for i, det in enumerate(dets):
            cls_name = det["class"]
            conf_val = det["confidence"]

            # Assign a stable color per class
            if cls_name not in color_map:
                color_map[cls_name] = COLORS[len(color_map) % len(COLORS)]
            color = color_map[cls_name]

            # Draw label box at top-left (bbox not available in mock/filtered mode)
            label   = f"{cls_name}  {conf_val:.2f}"
            y_pos   = 40 + i * 36
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, y_pos - th - 6), (10 + tw + 8, y_pos + 4), color, -1)
            cv2.putText(frame, label, (14, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # ── HUD ─────────────────────────────────────────────────────────
        hud = f"FPS: {fps:.1f}  |  Objects: {len(dets)}  |  Mode: {detector._mode}  |  [Q] Quit  [S] Save"
        cv2.rectangle(frame, (0, h_frame - 30), (w_frame, h_frame), (20, 20, 20), -1)
        cv2.putText(frame, hud, (10, h_frame - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Grocery Detector — Real-time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{ts}.jpg"
            cv2.imwrite(path, frame)
            print(f"📸  Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()
