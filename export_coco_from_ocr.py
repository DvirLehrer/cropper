#!/usr/bin/env python3

"""
Export a COCO-format dataset from `google_ocr.py` character box results.

Input per image:
  test_output/<image_base>_char_boxes.json

Output:
  <images-dir>/instances_chars.json (default), or `--out-json` if provided.

COCO notes:
- COCO `bbox` is axis-aligned [x,y,width,height] (we compute from the 4 vertices).
- We also include `segmentation` as a polygon from the 4 vertices.
- Categories correspond to OCR character classes (the `char` field).
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from PIL import Image


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def poly_from_vertices(verts: List[Dict[str, Any]]) -> List[float]:
    # COCO polygon: [x1,y1,x2,y2,x3,y3,x4,y4]
    pts = []
    for v in verts[:4]:
        pts.extend([float(safe_int(v.get("x"))), float(safe_int(v.get("y")))])
    return pts


def bbox_from_vertices(verts: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    xs = [float(safe_int(v.get("x"))) for v in verts[:4]]
    ys = [float(safe_int(v.get("y"))) for v in verts[:4]]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)


def area_from_bbox(bbox: Tuple[float, float, float, float]) -> float:
    return float(bbox[2] * bbox[3])


def iter_images(images_dir: str) -> List[str]:
    out = []
    for fn in os.listdir(images_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(os.path.join(images_dir, fn))
    out.sort()
    return out


def image_base(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_boxes_json(boxes_dir: str, img_path: str) -> Dict[str, Any]:
    base = image_base(img_path)
    p = os.path.join(boxes_dir, f"{base}_char_boxes.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Export COCO dataset from google_ocr.py char box JSON outputs.")
    parser.add_argument("--images-dir", default="images/before_crop", help="Directory containing source images.")
    parser.add_argument("--boxes-dir", default="test_output", help="Directory containing *_char_boxes.json files.")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Path to write COCO JSON. Defaults to <images-dir>/instances_chars.json (same folder as images).",
    )
    parser.add_argument("--dataset-name", default="stam_ocr_chars", help="info.description in COCO JSON.")
    parser.add_argument("--require-boxes", action="store_true", help="Fail if any image is missing its *_char_boxes.json.")
    args = parser.parse_args()

    images = iter_images(args.images_dir)
    if not images:
        raise SystemExit(f"No images found under: {args.images_dir}")

    # First pass: collect categories (unique chars)
    chars = set()
    per_image_boxes: Dict[str, List[Dict[str, Any]]] = {}
    for img_path in images:
        base = image_base(img_path)
        json_path = os.path.join(args.boxes_dir, f"{base}_char_boxes.json")
        if not os.path.exists(json_path):
            if args.require_boxes:
                raise SystemExit(f"Missing boxes JSON for {img_path}: {json_path}")
            print(f"SKIP (missing boxes): {img_path}")
            continue
        data = read_boxes_json(args.boxes_dir, img_path)
        boxes = data.get("boxes") or []
        per_image_boxes[img_path] = boxes
        for b in boxes:
            ch = (b.get("char") or "").strip()
            if ch:
                chars.add(ch)

    chars_sorted = sorted(chars, key=lambda s: (len(s), s))
    categories = []
    char_to_cat: Dict[str, int] = {}
    for i, ch in enumerate(chars_sorted, 1):
        char_to_cat[ch] = i
        categories.append(
            {
                "id": i,
                "name": ch,
                "supercategory": "character",
            }
        )

    coco_images = []
    coco_annotations = []

    ann_id = 1
    img_id = 1

    for img_path in images:
        if img_path not in per_image_boxes:
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        coco_images.append(
            {
                "id": img_id,
                # COCO expects `file_name` to be relative to where images live.
                # Since we write the JSON into `images-dir`, use the basename.
                "file_name": os.path.basename(img_path),
                "width": w,
                "height": h,
            }
        )

        for b in per_image_boxes[img_path]:
            ch = (b.get("char") or "").strip()
            verts = b.get("vertices") or []
            if not ch or len(verts) < 4:
                continue

            bbox = bbox_from_vertices(verts)
            seg = poly_from_vertices(verts)
            if len(seg) != 8:
                continue

            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    # "category_id": char_to_cat[ch],
                    "category_id": 1,
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                    "area": area_from_bbox(bbox),
                    "iscrowd": 0,
                    "segmentation": [seg],
                    # Non-standard extras (safe for most tools to ignore):
                    "text": ch,
                    "ocr_indices": {
                        "i": b.get("i"),
                        "page": b.get("page"),
                        "block": b.get("block"),
                        "paragraph": b.get("paragraph"),
                        "word": b.get("word"),
                        "symbol": b.get("symbol"),
                    },
                }
            )
            ann_id += 1

        img_id += 1

    out_path = args.out_json
    if not out_path:
        out_path = os.path.join(args.images_dir, "instances_chars.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    coco = {
        "info": {"description": args.dataset_name},
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"Wrote: {out_path}")
    print(f"images={len(coco_images)} annotations={len(coco_annotations)} categories={len(categories)}")


if __name__ == "__main__":
    main()


