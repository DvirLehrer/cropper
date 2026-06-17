#! /usr/bin/env python3
"""
Build the polygon training dataset from OCR JSON files.

For each *_char_boxes.json in ocr-dir:
  - Compute the minimal enclosing polygon (4-8 sides) of all Hebrew char boxes
  - Copy the original raw image to polygon_dataset/images/{split}/
  - Write a YOLO-seg label  (0 x1 y1 ... xn yn, normalised) to labels/{split}/

Split: 70 / 15 / 15  train / val / test  (seeded, reproducible)

Usage:
  python extract_polygons.py
  python extract_polygons.py --ocr-dir test_output --out polygon_dataset --seed 42
"""
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from visualize_char_boxes import enclosing_polygon

HEBREW_RANGE = ('א', 'ת')
SPLITS       = ('train', 'val', 'test')
SPLIT_RATIOS = (0.70,    0.15,  0.15)


def load_hebrew_boxes(json_path: str) -> list:
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    return [b for b in data.get('boxes', [])
            if b.get('char', '') and HEBREW_RANGE[0] <= b['char'][0] <= HEBREW_RANGE[1]], data.get('image', '')


def polygon_to_yolo(poly: np.ndarray, img_w: int, img_h: int) -> str:
    coords = []
    for x, y in poly:
        coords.append(f"{np.clip(x / img_w, 0, 1):.6f}")
        coords.append(f"{np.clip(y / img_h, 0, 1):.6f}")
    return "0 " + " ".join(coords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-dir", default="test_output")
    parser.add_argument("--out",     default="polygon_dataset")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--padding",   type=int, default=6,
                        help="Polygon padding in pixels (default: 6)")
    parser.add_argument("--max-sides", type=int, default=20,
                        help="Max polygon sides (default: 20)")
    args = parser.parse_args()

    rng       = np.random.default_rng(args.seed)
    out       = Path(args.out)
    json_files = sorted(glob.glob(os.path.join(args.ocr_dir, "*_char_boxes.json")))

    if not json_files:
        raise FileNotFoundError(f"No *_char_boxes.json found in {args.ocr_dir}")
    print(f"Found {len(json_files)} JSON files\n")

    # Create output directories
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Assign splits up front (reproducible shuffle)
    order  = rng.permutation(len(json_files))
    n      = len(json_files)
    n_tr   = int(n * SPLIT_RATIOS[0])
    n_val  = int(n * SPLIT_RATIOS[1])
    split_map = {}
    for idx, i in enumerate(order):
        if idx < n_tr:
            split_map[i] = 'train'
        elif idx < n_tr + n_val:
            split_map[i] = 'val'
        else:
            split_map[i] = 'test'

    ok = skipped = 0
    split_counts = {s: 0 for s in SPLITS}

    for i, jp in enumerate(json_files):
        boxes_heb, img_path = load_hebrew_boxes(jp)

        if not boxes_heb:
            print(f"  SKIP (no Hebrew boxes): {os.path.basename(jp)}")
            skipped += 1
            continue

        if not os.path.exists(img_path):
            print(f"  SKIP (image not found): {img_path}")
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is None:
            print(f"  SKIP (cannot read image): {img_path}")
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]

        poly = enclosing_polygon(boxes_heb, max_sides=args.max_sides, padding=args.padding)
        if poly is None or len(poly) < 4:
            print(f"  SKIP (polygon degenerate): {os.path.basename(jp)}")
            skipped += 1
            continue

        split    = split_map[i]
        base     = Path(jp).stem.replace('_char_boxes', '')
        img_ext  = Path(img_path).suffix
        img_dst  = out / "images" / split / f"{base}{img_ext}"
        lbl_dst  = out / "labels" / split / f"{base}.txt"

        shutil.copy2(img_path, img_dst)
        lbl_dst.write_text(polygon_to_yolo(poly, img_w, img_h) + "\n", encoding='utf-8')

        ok += 1
        split_counts[split] += 1

    # Write data.yaml
    yaml = (
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: text_region\n"
    )
    (out / "data.yaml").write_text(yaml, encoding='utf-8')

    print(f"\nDone: {ok} images exported, {skipped} skipped")
    for s in SPLITS:
        print(f"  {s}: {split_counts[s]}")
    print(f"\ndata.yaml → {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
