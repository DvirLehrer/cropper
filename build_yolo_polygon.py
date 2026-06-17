#! /usr/bin/env python3
"""
Convert Roboflow COCO exports → YOLO-seg format for YOLOv8-seg training.

For each image, the polygon is RE-DERIVED from the OCR JSON (test_output/<base>_char_boxes.json)
using enclosing_polygon() with max_sides=20, giving much higher precision than the 8-sided
polygons baked into the Roboflow COCO export.

Augmented images (no OCR JSON) fall back to the COCO segmentation polygon as-is.

Sources
  train : stam_polygon_augmented/train/  (_annotations.coco.json + images)
  val   : stam polygon.coco/valid/       (_annotations.coco.json + images)
  test  : stam polygon.coco/test/        (_annotations.coco.json + images)

Output
  yolo_polygon/
    data.yaml
    images/{train,val,test}/
    labels/{train,val,test}/

Usage
  python build_yolo_polygon.py
  python build_yolo_polygon.py --out yolo_polygon --colab-drive-path /content/drive/MyDrive/stam_seg/yolo_polygon
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from visualize_char_boxes import enclosing_polygon

SPLITS = {
    'train': 'stam_polygon_augmented/train',
    'val':   'stam polygon.coco/valid',
    'test':  'stam polygon.coco/test',
}
OCR_DIR      = Path('test_output')
HEBREW_RANGE = ('א', 'ת')
MAX_SIDES    = 20
PADDING      = 6


def _ocr_polygon(base: str, W: int, H: int):
    """Recompute 20-sided polygon from OCR JSON. Returns flat normalised list or None."""
    ocr_path = OCR_DIR / f'{base}_char_boxes.json'
    if not ocr_path.exists():
        return None
    with open(ocr_path, encoding='utf-8') as f:
        data = json.load(f)
    boxes = [b for b in data.get('boxes', [])
             if b.get('char', '') and HEBREW_RANGE[0] <= b['char'][0] <= HEBREW_RANGE[1]]
    if not boxes:
        return None
    poly = enclosing_polygon(boxes, max_sides=MAX_SIDES, padding=PADDING)
    if poly is None or len(poly) < 4:
        return None
    coords = []
    for x, y in poly:
        coords.append(f'{max(0.0, min(1.0, x / W)):.6f}')
        coords.append(f'{max(0.0, min(1.0, y / H)):.6f}')
    return coords


def _coco_polygon(seg, W: int, H: int):
    """Normalise a COCO segmentation polygon (fallback for augmented images)."""
    coords = []
    for x, y in zip(seg[0::2], seg[1::2]):
        coords.append(f'{max(0.0, min(1.0, x / W)):.6f}')
        coords.append(f'{max(0.0, min(1.0, y / H)):.6f}')
    return coords


def convert_split(split_name: str, src_dir: Path, out_dir: Path) -> int:
    ann_path = src_dir / '_annotations.coco.json'
    with open(ann_path, encoding='utf-8') as f:
        coco = json.load(f)

    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    img_out = out_dir / 'images' / split_name
    lbl_out = out_dir / 'labels' / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ok = skipped = ocr_hit = coco_fallback = 0
    for img_info in coco['images']:
        iid   = img_info['id']
        fname = img_info['file_name']
        W, H  = img_info['width'], img_info['height']

        anns = anns_by_img.get(iid, [])
        if not anns:
            skipped += 1; continue

        src_img = src_dir / fname
        if not src_img.exists():
            skipped += 1; continue

        stem = Path(fname).stem
        # Use extra.name (original filename before Roboflow upload) for OCR JSON lookup
        orig_name = (img_info.get('extra') or {}).get('name', '')
        base = Path(orig_name).stem if orig_name else stem

        lines = []
        for ann in anns:
            coords = _ocr_polygon(base, W, H)
            if coords is not None:
                ocr_hit += 1
            else:
                # Fallback: use COCO polygon (augmented images or missing OCR)
                coords = _coco_polygon(ann['segmentation'][0], W, H)
                coco_fallback += 1
            lines.append('0 ' + ' '.join(coords))

        shutil.copy2(src_img, img_out / fname)
        (lbl_out / f'{stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        ok += 1

    print(f'  {split_name}: {ok} images  (OCR polygon: {ocr_hit}, COCO fallback: {coco_fallback}, skipped: {skipped})')
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='yolo_polygon')
    parser.add_argument('--colab-drive-path', default='',
                        help='If set, write this as "path:" in data.yaml (for Colab)')
    args = parser.parse_args()

    out  = Path(args.out)
    base = Path('.')

    total = 0
    for split, src_rel in SPLITS.items():
        src = base / src_rel
        if not src.exists():
            print(f'WARNING: source not found, skipping {split}: {src}')
            continue
        total += convert_split(split, src, out)

    yaml_path  = out / 'data.yaml'
    drive_path = args.colab_drive_path or str(out.resolve())
    yaml_path.write_text(
        f'path: {drive_path}\n'
        f'train: images/train\n'
        f'val:   images/val\n'
        f'test:  images/test\n'
        f'nc: 1\n'
        f'names:\n'
        f'  0: text_region\n',
        encoding='utf-8',
    )

    print(f'\nDone: {total} images total  (max_sides={MAX_SIDES}, padding={PADDING}px)')
    print(f'data.yaml → {yaml_path}')
    print(f'\nNext: zip {out}/ and upload to Google Drive, then open train_yolo.ipynb in Colab.')


if __name__ == '__main__':
    main()
