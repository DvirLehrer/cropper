#! /usr/bin/env python3
"""
Crop STaM manuscript images to the text polygon using the YOLOv8-seg model.

For each input image:
  1. Run the model to detect the text-region polygon
  2. Mask everything outside the polygon to white
  3. Crop to the polygon bounding box
  4. Save to --out-dir

For images with extreme aspect ratios (> SPLIT_RATIO), the image is tiled along the long
axis into near-square overlapping tiles. Each tile's polygon is merged back into full-image
coordinates via convex hull.

Usage:
  python crop_with_model.py --image images/benchmark/mezuza3.jpeg
  python crop_with_model.py --images-dir images/benchmark --out-dir test_output/cropped
  python crop_with_model.py --images-dir images/before_crop --model best.pt --out-dir test_output/cropped
"""
import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS      = {'.jpg', '.jpeg', '.png'}
IMGSZ         = 640
SPLIT_RATIO   = 10.0  # trigger tiled inference when max(W,H)/min(W,H) exceeds this
SPLIT_OVERLAP = 0.15
MIN_SHORT_PX  = 128   # ensure short dimension is at least this many px in model input


def _imgsz_for(W: int, H: int) -> int:
    """Compute imgsz so the short dimension maps to at least MIN_SHORT_PX pixels."""
    short, long = min(W, H), max(W, H)
    needed = int(long * MIN_SHORT_PX / short)
    needed = ((needed + 31) // 32) * 32   # round up to multiple of 32
    return max(IMGSZ, needed)


def _predict(model, img, conf: float, imgsz: int = IMGSZ):
    """
    Run model on a numpy array or path.
    If one mask: return its polygon directly (full model precision).
    If multiple masks: OR the mask bitmaps, find outer contour → precise union polygon.
    """
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.masks is None or not len(r.masks):
        return None
    if len(r.masks) == 1:
        return r.masks.xy[0]
    # Combine mask bitmaps and trace outer contour
    combined = np.zeros(r.masks.data[0].shape, dtype=np.uint8)
    for m in r.masks.data:
        combined = cv2.bitwise_or(combined, (m.numpy() * 255).astype(np.uint8))
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return r.masks.xy[0]
    # Scale contour coords from mask space back to image space
    mh, mw = combined.shape
    if isinstance(img, np.ndarray):
        ih, iw = img.shape[:2]
    else:
        tmp = cv2.imread(img, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        ih, iw = tmp.shape[:2]
    sx, sy = iw / mw, ih / mh
    biggest = max(contours, key=cv2.contourArea)
    poly = biggest.reshape(-1, 2).astype(np.float32)
    poly[:, 0] *= sx
    poly[:, 1] *= sy
    return poly


def _tiled_predict(model, img: np.ndarray, conf: float):
    """
    For extreme-ratio images: split into 3 overlapping tiles along the long axis,
    run inference on each tile with dynamic imgsz, merge all masks via convex hull.
    """
    H, W = img.shape[:2]
    horizontal = W >= H
    long_dim  = max(W, H)
    short_dim = min(W, H)
    n_tiles   = 3
    tile_size = long_dim // n_tiles
    overlap   = int(tile_size * SPLIT_OVERLAP)

    starts = [i * tile_size for i in range(n_tiles)]
    starts[-1] = long_dim - tile_size  # align last tile to end

    all_pts = []
    for s in starts:
        e = min(s + tile_size + overlap, long_dim)
        if horizontal:
            tile = img[:, s:e]
            offset = (s, 0)
        else:
            tile = img[s:e, :]
            offset = (0, s)

        th, tw = tile.shape[:2]
        imgsz = _imgsz_for(tw, th)
        poly = _predict(model, tile, conf, imgsz)
        if poly is None:
            continue
        poly = poly.copy()
        poly[:, 0] += offset[0]
        poly[:, 1] += offset[1]
        all_pts.append(poly)

    if not all_pts:
        return _predict(model, img, conf, _imgsz_for(W, H))

    # Merge via convex hull across tiles (tile polygons are already precise per-tile)
    merged = np.vstack(all_pts).astype(np.float32)
    return cv2.convexHull(merged).reshape(-1, 2)


def expand_to_blobs(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Post-process: absorb ink pixels that are just outside the polygon boundary.

    1. Binarize → ink mask
    2. Dilate polygon outward by ~char_size px
    3. AND with ink mask → ink just outside the polygon edge
    4. Remove very large regions (background bleed) by size-filtering components
    5. Add to polygon, find outer contour
    """
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly], 255)

    # Dilate outward by ~2% of short dimension (roughly one char width)
    ring_px = max(8, int(min(H, W) * 0.02))
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px*2+1, ring_px*2+1))
    expanded_mask = cv2.dilate(poly_mask, kernel)

    # Ink pixels in the expansion zone only (not already inside)
    new_ink = ink & (expanded_mask & ~poly_mask)

    # Estimate typical char area from ink inside the polygon
    ink_inside = ink & poly_mask
    char_area  = max(4, int(cv2.countNonZero(ink_inside) / max(1, len(poly))))
    max_blob   = char_area * 30  # reject blobs much larger than a few chars

    # Remove oversized connected regions from new_ink (noise / background bleed)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(new_ink)
    filtered = np.zeros_like(new_ink)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] <= max_blob:
            filtered[labels == lbl] = 255

    combined = cv2.bitwise_or(poly_mask, filtered)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return poly
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def crop_image(model, img_path: str, out_dir: str, conf: float = 0.25) -> bool:
    img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img is None:
        print(f'  SKIP (unreadable): {img_path}')
        return False

    H, W = img.shape[:2]
    ratio = max(W, H) / min(W, H)
    imgsz = _imgsz_for(W, H)
    t0 = time.perf_counter()
    poly_f = _predict(model, img_path, conf, imgsz)
    # If single pass misses >30% of the long dimension, fall back to tiling
    if poly_f is not None and ratio > SPLIT_RATIO:
        long_dim = max(W, H)
        span = (poly_f[:, 0].max() - poly_f[:, 0].min()) if W >= H else (poly_f[:, 1].max() - poly_f[:, 1].min())
        if span < long_dim * 0.70:
            poly_f = _tiled_predict(model, img, conf)
    elif poly_f is None and ratio > SPLIT_RATIO:
        poly_f = _tiled_predict(model, img, conf)
    elapsed = (time.perf_counter() - t0) * 1000

    if poly_f is None:
        print(f'  NO DETECTION ({elapsed:.0f}ms): {os.path.basename(img_path)}')
        return False

    poly_f = expand_to_blobs(img, poly_f.astype(np.int32))
    poly = poly_f.astype(np.int32)

    # Build mask and blank outside polygon to white
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    result = img.copy()
    result[mask == 0] = 255

    # Crop to polygon bounding box
    x, y, w, h = cv2.boundingRect(poly)
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    cropped = result[y:y+h, x:x+w]

    # Save
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(img_path).stem
    ext  = Path(img_path).suffix
    out_path = os.path.join(out_dir, f'{stem}_cropped{ext}')
    cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])

    mode_tag = f' [tiled]' if ratio > SPLIT_RATIO else ''
    print(f'  {os.path.basename(img_path):45s}  {elapsed:5.0f}ms  '
          f'{W}x{H} → {w}x{h}{mode_tag}  → {out_path}')
    return True


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',      help='Single image')
    src.add_argument('--images-dir', help='Directory of images')
    parser.add_argument('--model',   default='best.pt')
    parser.add_argument('--out-dir', default='test_output/cropped')
    parser.add_argument('--conf',    type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.image:
        images = [args.image]
    else:
        images = [p for p in sorted(glob.glob(os.path.join(args.images_dir, '*')))
                  if Path(p).suffix.lower() in IMG_EXTS]

    print(f'Processing {len(images)} image(s) with {args.model}\n')
    ok = sum(crop_image(model, p, args.out_dir, args.conf) for p in images)
    print(f'\nDone: {ok}/{len(images)} cropped → {args.out_dir}/')


if __name__ == '__main__':
    main()
