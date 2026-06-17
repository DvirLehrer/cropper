#!/usr/bin/env python3
"""
debug_combined.py — Benchmark visualiser for the STaM crop pipeline.

For each image, runs the full crop_stam pipeline and draws a debug overlay showing:
  - OCR char boxes (green = included, dark-red = remote/excluded)
  - OCR convex hull (blue)
  - Model polygon (green outline)
  - Union polygon / final boundary (red, thick)

Also saves cropped output identical to crop_stam.py.

Usage:
  python3 debug_combined.py [--images-dir images/benchmark] [--model best.pt]
  python3 debug_combined.py --image images/benchmark/mezuza3.jpeg
"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from crop_with_model import _predict, _tiled_predict, _imgsz_for, SPLIT_RATIO
from crop_stam import (
    run_ocr, adjacent_ocr_corners, union_polygon,
    CONF, IMG_EXTS, MODEL_PATH, OCR_MAX_PIXELS,
)

BENCH_DIR = 'images/benchmark'
OUT_DIR   = 'test_output/debug_combined'


def process(model, img_path: str, out_dir: str) -> dict:
    stem = Path(img_path).stem
    img  = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img is None:
        print(f'  SKIP (unreadable): {img_path}')
        return None

    H, W  = img.shape[:2]
    ratio = max(W, H) / min(W, H)
    imgsz = _imgsz_for(W, H)

    # Segmentation + OCR in parallel
    def _seg():
        t0   = time.perf_counter()
        poly = _predict(model, img_path, CONF, imgsz)
        if poly is not None and ratio > SPLIT_RATIO:
            span = (poly[:, 0].max() - poly[:, 0].min()) if W >= H \
                   else (poly[:, 1].max() - poly[:, 1].min())
            if span < max(W, H) * 0.70:
                poly = _tiled_predict(model, img, CONF)
        elif poly is None and ratio > SPLIT_RATIO:
            poly = _tiled_predict(model, img, CONF)
        return poly, (time.perf_counter() - t0) * 1000

    def _ocr():
        t0 = time.perf_counter()
        return run_ocr(img_path), (time.perf_counter() - t0) * 1000

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_seg = ex.submit(_seg)
        f_ocr = ex.submit(_ocr)
        poly_model, t_seg = f_seg.result()
        ocr_boxes,  t_ocr = f_ocr.result()
    t_wall = (time.perf_counter() - t_wall) * 1000

    model_mask = None
    if poly_model is not None:
        model_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(model_mask, [poly_model.astype(np.int32)], 255)

    t0 = time.perf_counter()
    adj_pts, adj_set = adjacent_ocr_corners(ocr_boxes, model_mask, H, W)
    ocr_hull = cv2.convexHull(adj_pts).reshape(-1, 2) if adj_pts is not None else None
    t_extend = (time.perf_counter() - t0) * 1000

    if poly_model is not None and ocr_hull is not None:
        boundary = union_polygon(H, W, poly_model, ocr_hull)
    elif poly_model is not None:
        boundary = poly_model
    elif ocr_hull is not None:
        boundary = ocr_hull
    else:
        print(f'  NO DATA: {stem}')
        return None

    # ── debug overlay ──────────────────────────────────────────────────────────
    debug = img.copy()

    for i, b in enumerate(ocr_boxes):
        pts   = np.array([[v['x'], v['y']] for v in b['vertices']], np.int32)
        color = (0, 200, 80) if i in adj_set else (0, 0, 180)
        cv2.polylines(debug, [pts], True, color, 1)

    if ocr_hull is not None:
        cv2.polylines(debug, [ocr_hull.astype(np.int32)], True, (200, 60, 0), 2)
    if poly_model is not None:
        cv2.polylines(debug, [poly_model.astype(np.int32)], True, (0, 220, 60), 2)
    cv2.polylines(debug, [boundary.astype(np.int32)], True, (0, 0, 220), 4)

    legend = [
        ('model (green)',            (0, 220, 60)),
        ('adj OCR hull (blue)',      (200, 60, 0)),
        ('union / boundary (red)',   (0, 0, 220)),
        ('remote OCR (excl.)',       (0, 0, 180)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(debug, txt, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    n_adj    = len(adj_set)
    n_remote = len(ocr_boxes) - n_adj
    print(f'  {stem:45s}  wall={t_wall+t_extend:4.0f}ms'
          f'  (seg={t_seg:.0f}  ocr={t_ocr:.0f}  extend={t_extend:.0f})'
          f'  adj={n_adj}  remote={n_remote}')

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_debug.jpg'), debug,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── cropped output (same as crop_stam.py) ─────────────────────────────────
    ref_mask  = model_mask if model_mask is not None else np.zeros((H, W), dtype=np.uint8)
    if model_mask is None:
        cv2.fillPoly(ref_mask, [boundary.astype(np.int32)], 255)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inside_px = gray[ref_mask == 255]
    if inside_px.size >= 100:
        thresh, _ = cv2.threshold(inside_px.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_pixels = img[ref_mask == 255][inside_px > thresh]
        bg_color  = tuple(int(c) for c in bg_pixels.mean(axis=0)) if len(bg_pixels) else (255, 255, 255)
    else:
        bg_color = (255, 255, 255)

    union_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(union_mask, [boundary.astype(np.int32)], 255)
    result = np.full_like(img, bg_color)
    result[union_mask == 255] = img[union_mask == 255]
    x, y, w, h = cv2.boundingRect(boundary.astype(np.int32))

    crop_dir = out_dir.replace('debug_combined', 'cropped_combined')
    os.makedirs(crop_dir, exist_ok=True)
    cv2.imwrite(os.path.join(crop_dir, f'{stem}_cropped.jpg'), result[y:y+h, x:x+w],
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {'seg': t_seg, 'ocr': t_ocr, 'extend': t_extend, 'wall': t_wall + t_extend}


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument('--image')
    src.add_argument('--images-dir', default=BENCH_DIR)
    parser.add_argument('--model',   default=MODEL_PATH)
    parser.add_argument('--out-dir', default=OUT_DIR)
    args = parser.parse_args()

    model  = YOLO(args.model)
    images = ([args.image] if args.image else
              sorted(p for p in glob.glob(os.path.join(args.images_dir, '*'))
                     if Path(p).suffix.lower() in IMG_EXTS))

    print(f'Processing {len(images)} images\n')
    results = [r for r in (process(model, p, args.out_dir) for p in images) if r]
    n = len(results)
    if n:
        avg = lambda k: sum(r[k] for r in results) / n
        print(f'\n{"─"*70}')
        print(f'  {"":45s}  {"wall":>7}  {"seg":>6}  {"ocr":>6}  {"extend":>7}')
        print(f'  {"AVERAGE":45s}  {avg("wall"):6.0f}ms  {avg("seg"):5.0f}ms'
              f'  {avg("ocr"):5.0f}ms  {avg("extend"):6.0f}ms')
    print(f'\nDone: {n}/{len(images)} → {args.out_dir}/')


if __name__ == '__main__':
    main()
