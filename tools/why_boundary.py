#!/usr/bin/env python3
"""Answer one question: where does the final crop boundary actually come from?

    python3 tools/why_boundary.py --image ../benchmark/02_rotate/<name>.jpg

On several rotate and perspective images the boundary visibly encloses areas of
table or wall that hold no writing. It has to come from the model polygon, from
the OCR hull, or from a bug in the union of the two — and the three are easy to
tell apart by area. This prints the numbers and writes a mask showing exactly
which pixels the boundary added beyond both inputs.

The distinction decides the fix: a sloppy segmentation mask is a morphology
problem, hallucinated characters on texture are a filtering problem, and a
boundary larger than both its inputs is neither.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default=str(REPO / "best.pt"))
    ap.add_argument("--out-dir", default=str(REPO / "test_output" / "why"))
    args = ap.parse_args()

    from ultralytics import YOLO

    import crop_stam
    from crop_with_model import SPLIT_RATIO, _imgsz_for, _predict, _tiled_predict
    from stam_io import imread_any

    img = imread_any(args.image)
    if img is None:
        raise SystemExit(f"unreadable: {args.image}")
    H, W = img.shape[:2]
    model = YOLO(args.model)
    ratio = max(W, H) / min(W, H)
    imgsz = _imgsz_for(W, H)

    def _seg():
        poly = _predict(model, img, crop_stam.CONF, imgsz)
        if poly is None and ratio > SPLIT_RATIO:
            poly = _tiled_predict(model, img, crop_stam.CONF)
        return poly

    def _raw_detections():
        """Every mask the model returned, before _predict ORs them together.

        _predict merges all of them into one region. One sound detection on the
        parchment plus a spurious one on the table is enough to make the merged
        region swallow the frame, which would explain why the failure clusters
        on photographs of a sheet lying on a surface rather than on flat scans.
        """
        r = model.predict(img, imgsz=imgsz, conf=crop_stam.CONF, verbose=False)[0]
        if r.masks is None or not len(r.masks):
            return []
        confs = (r.boxes.conf.tolist() if r.boxes is not None else [None] * len(r.masks))
        out = []
        for i, poly in enumerate(r.masks.xy):
            m = np.zeros((H, W), np.uint8)
            if len(poly) >= 3:
                cv2.fillPoly(m, [poly.astype(np.int32)], 255)
            out.append((confs[i] if i < len(confs) else None,
                        100.0 * int(cv2.countNonZero(m)) / (H * W)))
        return out

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_seg, f_ocr = ex.submit(_seg), ex.submit(crop_stam.run_ocr, img)
        poly_model, ocr_boxes = f_seg.result(), f_ocr.result()

    def mask_of(poly):
        m = np.zeros((H, W), np.uint8)
        if poly is not None:
            cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        return m

    model_mask = mask_of(poly_model)
    adj_pts, accepted = crop_stam.adjacent_ocr_corners(ocr_boxes, model_mask, H, W)
    ocr_hull = cv2.convexHull(adj_pts).reshape(-1, 2) if adj_pts is not None else None
    hull_mask = mask_of(ocr_hull)

    if poly_model is not None and ocr_hull is not None:
        boundary = crop_stam.union_polygon(H, W, poly_model, ocr_hull)
    else:
        boundary = poly_model if poly_model is not None else ocr_hull
    bnd_mask = mask_of(boundary)

    frame = H * W
    inputs = cv2.bitwise_or(model_mask, hull_mask)
    added = cv2.bitwise_and(bnd_mask, cv2.bitwise_not(inputs))

    def pct(m):
        return 100.0 * int(cv2.countNonZero(m)) / frame

    print(f"\nimage            {Path(args.image).name}   {W}x{H}")
    print(f"OCR characters   {len(ocr_boxes)} found, {len(accepted)} accepted into the hull")

    dets = _raw_detections()
    print(f"\nmodel returned   {len(dets)} mask(s)"
          + ("   — _predict merges them all into one region" if len(dets) > 1 else ""))
    for i, (conf, area) in enumerate(sorted(dets, key=lambda d: -(d[1] or 0)), 1):
        print(f"   mask {i}: {area:>5.1f}% of frame"
              + (f"   confidence {conf:.2f}" if conf is not None else ""))
    print(f"\n{'model polygon':<24}{pct(model_mask):>7.1f}% of frame")
    print(f"{'OCR hull':<24}{pct(hull_mask):>7.1f}%")
    print(f"{'model OR hull':<24}{pct(inputs):>7.1f}%   <- the boundary should equal this")
    print(f"{'final boundary':<24}{pct(bnd_mask):>7.1f}%")
    print(f"{'added by the union':<24}{pct(added):>7.1f}%   <- anything here is a bug")

    hull_only = cv2.bitwise_and(hull_mask, cv2.bitwise_not(model_mask))
    model_only = cv2.bitwise_and(model_mask, cv2.bitwise_not(hull_mask))
    print(f"\n{'hull beyond the model':<24}{pct(hull_only):>7.1f}%")
    print(f"{'model beyond the hull':<24}{pct(model_only):>7.1f}%")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vis = img.copy()
    vis[model_only > 0] = (0.5 * vis[model_only > 0] + 0.5 * np.array([0, 220, 60])).astype(np.uint8)
    vis[hull_only > 0] = (0.5 * vis[hull_only > 0] + 0.5 * np.array([220, 60, 0])).astype(np.uint8)
    vis[added > 0] = (0.5 * vis[added > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    path = out / f"{Path(args.image).stem}_why.jpg"
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("\ngreen = model only, blue = hull only, red = added by the union")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
