"""
download_dataset.py — Download a Roboflow grocery dataset automatically.

Usage:
    # Option 1: Use Roboflow API key (recommended)
    python scripts/download_dataset.py --api-key YOUR_KEY --workspace my-workspace --project grocery-detection --version 1

    # Option 2: Use a Roboflow public download URL
    python scripts/download_dataset.py --url "https://app.roboflow.com/ds/XXXX?key=YYYY"

    # Option 3: Create a tiny synthetic dataset for testing (no internet needed)
    python scripts/download_dataset.py --synthetic
"""

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download or generate a grocery detection dataset")
    p.add_argument("--api-key",   default=None,  help="Roboflow API key")
    p.add_argument("--workspace", default=None,  help="Roboflow workspace slug")
    p.add_argument("--project",   default=None,  help="Roboflow project slug")
    p.add_argument("--version",   type=int, default=1, help="Dataset version number")
    p.add_argument("--url",       default=None,  help="Direct Roboflow download URL")
    p.add_argument("--out",       default="data/grocery", help="Output directory")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate a tiny synthetic dataset (testing only)")
    return p.parse_args()


# ── Synthetic dataset generator ──────────────────────────────────────────
CLASSES = ["rice", "oil_bottle", "soap", "detergent", "snack_packet",
           "water_bottle", "juice", "cereal_box", "canned_food", "bread"]


def create_synthetic_dataset(out_dir: Path, n_train=80, n_val=20):
    """Generate a minimal YOLO-format dataset with solid-color 'product' bboxes."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("Pillow required: pip install Pillow")

    print("Generating synthetic dataset (for testing only)…")

    for split, n in [("train", n_train), ("valid", n_val)]:
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            w, h = 640, 480
            img = Image.new("RGB", (w, h), color=_rand_color())
            draw = ImageDraw.Draw(img)

            labels = []
            n_boxes = random.randint(1, 4)
            for _ in range(n_boxes):
                cls_id = random.randint(0, len(CLASSES) - 1)
                bw = random.randint(60, 200)
                bh = random.randint(80, 250)
                bx = random.randint(0, w - bw)
                by = random.randint(0, h - bh)
                draw.rectangle([bx, by, bx + bw, by + bh], fill=_rand_color(), outline="white", width=3)
                draw.text((bx + 4, by + 4), CLASSES[cls_id], fill="white")
                # YOLO format: cls cx cy w h (normalized)
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            img.save(img_dir / f"img_{i:04d}.jpg")
            (lbl_dir / f"img_{i:04d}.txt").write_text("\n".join(labels))

    # data.yaml
    yaml_content = f"""path: {out_dir.resolve()}
train: train/images
val: valid/images

nc: {len(CLASSES)}
names: {json.dumps(CLASSES)}
"""
    (out_dir / "data.yaml").write_text(yaml_content)
    print(f"✅ Synthetic dataset created at: {out_dir}")
    print(f"   {n_train} train + {n_val} val images | {len(CLASSES)} classes")
    print(f"   data.yaml: {out_dir / 'data.yaml'}")
    print("\n⚠️  This is synthetic data for pipeline testing only.")
    print("   For real training, use a Roboflow grocery dataset.\n")


def _rand_color():
    return (random.randint(50, 220), random.randint(50, 220), random.randint(50, 220))


# ── Roboflow download ──────────────────────────────────────────────────────

def download_roboflow(args, out_dir: Path):
    if args.api_key and args.workspace and args.project:
        try:
            from roboflow import Roboflow
        except ImportError:
            raise SystemExit("Run: pip install roboflow")

        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace(args.workspace).project(args.project)
        dataset = project.version(args.version).download("yolov8", location=str(out_dir))
        print(f"✅ Dataset downloaded to: {out_dir}")

    elif args.url:
        import urllib.request
        zip_path = out_dir.parent / "dataset.zip"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading from {args.url} …")
        urllib.request.urlretrieve(args.url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        zip_path.unlink()
        print(f"✅ Dataset extracted to: {out_dir}")
    else:
        raise SystemExit(
            "Provide either:\n"
            "  --api-key, --workspace, --project\n"
            "  --url\n"
            "  --synthetic"
        )


def main():
    args = parse_args()
    out_dir = Path(args.out)

    if args.synthetic:
        create_synthetic_dataset(out_dir)
    else:
        download_roboflow(args, out_dir)


if __name__ == "__main__":
    main()
