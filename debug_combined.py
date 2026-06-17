#!/usr/bin/env python3
"""
debug_combined.py — For each benchmark image, combine:
  1. YOLO model polygon
  2. Convex hull of Hebrew OCR char-box corners that are ADJACENT to the model polygon
     (remote OCR clusters far from the model are ignored)
  Union of the two masks → outer contour = final polygon.

"Adjacent" = char box overlaps a dilated version of the model polygon mask
(dilation ≈ ADJACENCY_PCT * short_dim, roughly one character width).

Usage:
  python3 debug_combined.py [--images-dir images/benchmark] [--model best.pt]
"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision
from ultralytics import YOLO

from crop_with_model import _predict, _tiled_predict, _imgsz_for, SPLIT_RATIO

IMG_EXTS         = {'.jpg', '.jpeg', '.png'}
BENCH_DIR        = 'images/benchmark'
OUT_DIR          = 'test_output/debug_combined'
MODEL_PATH       = 'best.pt'
OCR_MAX_PIXELS   = 1_000_000
CLUSTER_GAP_PCT  = 0.04
MODEL_REACH_PCT  = 0.04
EXTEND_WORK_PX   = 1000   # downscale mask to this max dimension before connectedComponents

HEBREW_RANGE = ('א', 'ת')

_vision_client = None

def _get_vision_client():
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


def run_ocr(img_path: str):
    """Call Google Vision document_text_detection; downscale to ≤OCR_MAX_PIXELS first.
    Returns list of {char, vertices} dicts in original image pixel coords."""
    raw = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if raw is None:
        return []

    h, w = raw.shape[:2]
    ext   = Path(img_path).suffix or '.jpg'
    scale = 1.0
    if h * w > OCR_MAX_PIXELS:
        scale = (OCR_MAX_PIXELS / (h * w)) ** 0.5
        small = cv2.resize(raw, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(ext, small)
        content = buf.tobytes() if ok else open(img_path, 'rb').read()
    else:
        content = open(img_path, 'rb').read()

    inv = 1.0 / scale
    response = _get_vision_client().document_text_detection(
        image=vision.Image(content=content))
    if response.error.message:
        return []

    boxes = []
    fta = response.full_text_annotation
    if fta and fta.pages:
        for page in fta.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        for sym in word.symbols:
                            ch = sym.text or ''
                            if not ch.strip():
                                continue
                            if not (HEBREW_RANGE[0] <= ch[0] <= HEBREW_RANGE[1]):
                                continue
                            verts = sym.bounding_box.vertices if sym.bounding_box else []
                            if not verts:
                                continue
                            boxes.append({'char': ch, 'vertices': [
                                {'x': int(round((v.x or 0) * inv)),
                                 'y': int(round((v.y or 0) * inv))} for v in verts]})
    return boxes




def adjacent_ocr_corners(boxes: list, model_mask: np.ndarray, H: int, W: int):
    """
    Cluster-based adjacency: group all OCR boxes into connected text clusters
    (by dilating their footprints to bridge inter-character gaps), then include
    an entire cluster if the model polygon touches any box within it.
    All mask work is done at a downscaled resolution (EXTEND_WORK_PX) to keep
    connectedComponents fast on large images.
    Returns (Nx2 pts array in original coords or None, accepted index set).
    """
    if not boxes or model_mask is None:
        return None, set()

    # Downscale factor so max(H,W) ≤ EXTEND_WORK_PX
    sc = min(1.0, EXTEND_WORK_PX / max(H, W))
    wW, wH = max(1, round(W * sc)), max(1, round(H * sc))

    work_model = cv2.resize(model_mask, (wW, wH), interpolation=cv2.INTER_NEAREST)

    # 1. Paint OCR footprints at work resolution, dilate to bridge char gaps
    ocr_mask = np.zeros((wH, wW), dtype=np.uint8)
    for b in boxes:
        verts = b.get('vertices', [])
        if verts:
            pts = np.array([[int(round(v['x'] * sc)), int(round(v['y'] * sc))]
                            for v in verts], np.int32)
            cv2.fillPoly(ocr_mask, [pts], 255)

    gap_px = max(3, int(min(wH, wW) * CLUSTER_GAP_PCT))
    k_gap  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap_px+1, 2*gap_px+1))
    ocr_clustered = cv2.dilate(ocr_mask, k_gap)

    # 2. Connected components = text clusters
    num_labels, labels = cv2.connectedComponents(ocr_clustered)

    # 3. Dilate model mask slightly to catch boxes at its boundary
    reach_px = max(3, int(min(wH, wW) * MODEL_REACH_PCT))
    k_reach  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*reach_px+1, 2*reach_px+1))
    model_reach = cv2.dilate(work_model, k_reach)

    # 4. Which clusters touch the model polygon?
    connected_labels = set()
    for lbl in range(1, num_labels):
        if cv2.countNonZero(cv2.bitwise_and((labels == lbl).astype(np.uint8), model_reach)):
            connected_labels.add(lbl)

    # 5. Collect original-coord corners of boxes in connected clusters
    accepted = set()
    pts = []
    for i, b in enumerate(boxes):
        verts = b.get('vertices', [])
        if not verts:
            continue
        for v in verts:
            x = int(np.clip(round(v['x'] * sc), 0, wW-1))
            y = int(np.clip(round(v['y'] * sc), 0, wH-1))
            if labels[y, x] in connected_labels:
                accepted.add(i)
                for vv in verts:
                    pts.append([vv['x'], vv['y']])
                break

    if not pts:
        return None, accepted
    return np.array(pts, dtype=np.float32), accepted


def union_polygon(H: int, W: int, poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    """Binary-mask union of two polygons → outer contour in image coords."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly1.astype(np.int32)], 255)
    cv2.fillPoly(mask, [poly2.astype(np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return poly1
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def process(model, img_path: str, out_dir: str) -> dict:
    stem = Path(img_path).stem
    img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img is None:
        print(f'  SKIP (unreadable): {img_path}')
        return None

    H, W = img.shape[:2]
    ratio = max(W, H) / min(W, H)
    imgsz = _imgsz_for(W, H)

    # --- Segmentation + OCR in parallel ---
    def _seg():
        t0 = time.perf_counter()
        poly = _predict(model, img_path, 0.25, imgsz)
        if poly is not None and ratio > SPLIT_RATIO:
            long_dim = max(W, H)
            span = (poly[:, 0].max() - poly[:, 0].min()) if W >= H \
                   else (poly[:, 1].max() - poly[:, 1].min())
            if span < long_dim * 0.70:
                poly = _tiled_predict(model, img, 0.25)
        elif poly is None and ratio > SPLIT_RATIO:
            poly = _tiled_predict(model, img, 0.25)
        return poly, (time.perf_counter() - t0) * 1000

    def _ocr():
        t0 = time.perf_counter()
        boxes = run_ocr(img_path)
        return boxes, (time.perf_counter() - t0) * 1000

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_seg = ex.submit(_seg)
        f_ocr = ex.submit(_ocr)
        poly_model, t_seg = f_seg.result()
        ocr_boxes, t_ocr  = f_ocr.result()
    t_wall = (time.perf_counter() - t_wall) * 1000

    # --- Model mask (needed for adjacency test) ---
    model_mask = None
    if poly_model is not None:
        model_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(model_mask, [poly_model.astype(np.int32)], 255)

    # --- Extend OCR (cluster adjacency + hull) ---
    t0 = time.perf_counter()
    adj_pts, adj_set = adjacent_ocr_corners(ocr_boxes, model_mask, H, W)
    hull = cv2.convexHull(adj_pts).reshape(-1, 2) if adj_pts is not None else None
    t_extend = (time.perf_counter() - t0) * 1000

    # --- Union ---
    if poly_model is not None and hull is not None:
        combined = union_polygon(H, W, poly_model, hull)
    elif poly_model is not None:
        combined = poly_model
    elif hull is not None:
        combined = hull
    else:
        print(f'  NO DATA: {stem}')
        return None

    # --- Debug image ---
    debug = img.copy()

    # OCR char boxes — green if adjacent (in connected cluster), dark red if remote

    for i, b in enumerate(ocr_boxes):
        pts = np.array([[v['x'], v['y']] for v in b['vertices']], np.int32)
        color = (0, 200, 80) if i in adj_set else (0, 0, 180)
        cv2.polylines(debug, [pts], True, color, 1)

    # OCR adjacent hull (blue)
    if hull is not None:
        cv2.polylines(debug, [hull.astype(np.int32)], True, (200, 60, 0), 2)

    # Model polygon (green, thick)
    if poly_model is not None:
        cv2.polylines(debug, [poly_model.astype(np.int32)], True, (0, 220, 60), 2)

    # Combined union polygon (red, thickest)
    cv2.polylines(debug, [combined.astype(np.int32)], True, (0, 0, 220), 4)

    # Legend
    legend = [
        ('model (green)',       (0, 220, 60)),
        ('adj OCR hull (blue)', (200, 60, 0)),
        ('union (red)',         (0, 0, 220)),
        ('remote OCR (dark red, ignored)', (0, 0, 180)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(debug, txt, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    n_adj    = len(adj_set)
    n_remote = len(ocr_boxes) - n_adj
    print(f'  {stem:45s}  wall={t_wall+t_extend:4.0f}ms'
          f'  (seg={t_seg:.0f}  ocr={t_ocr:.0f}  extend={t_extend:.0f})  adj={n_adj}  remote={n_remote}')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{stem}_combined.jpg')
    cv2.imwrite(out_path, debug, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # --- Cropped output: union polygon kept, outside filled with parchment background color ---
    # Compute background color from non-ink pixels inside the model polygon (Otsu separates ink from bg)
    ref_poly  = poly_model if poly_model is not None else combined
    ref_mask  = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(ref_mask, [ref_poly.astype(np.int32)], 255)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inside_px = gray[ref_mask == 255]
    if inside_px.size >= 100:
        thresh, _ = cv2.threshold(inside_px.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask_flat = inside_px > thresh          # True = background (lighter) pixels
        bg_pixels    = img[ref_mask == 255][bg_mask_flat]
        bg_color     = tuple(int(c) for c in bg_pixels.mean(axis=0)) if len(bg_pixels) else (255, 255, 255)
    else:
        bg_color = (255, 255, 255)

    union_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(union_mask, [combined.astype(np.int32)], 255)
    result = np.full_like(img, bg_color, dtype=np.uint8)
    result[union_mask == 255] = img[union_mask == 255]

    x, y, w, h = cv2.boundingRect(combined.astype(np.int32))
    cropped = result[y:y+h, x:x+w]

    crop_dir  = out_dir.replace('debug_combined', 'cropped_combined')
    os.makedirs(crop_dir, exist_ok=True)
    crop_path = os.path.join(crop_dir, f'{stem}_cropped.jpg')
    cv2.imwrite(crop_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {'seg': t_seg, 'ocr': t_ocr, 'extend': t_extend, 'wall': t_wall + t_extend}


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument('--image')
    src.add_argument('--images-dir', default=BENCH_DIR)
    parser.add_argument('--model',   default=MODEL_PATH)
    parser.add_argument('--out-dir', default=OUT_DIR)
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.image:
        images = [args.image]
    else:
        images = sorted(p for p in glob.glob(os.path.join(args.images_dir, '*'))
                        if Path(p).suffix.lower() in IMG_EXTS)

    print(f'Processing {len(images)} images\n')
    results = [process(model, p, args.out_dir) for p in images]
    results = [r for r in results if r]
    n = len(results)
    if n:
        avg = lambda k: sum(r[k] for r in results) / n
        tot = lambda k: sum(r[k] for r in results)
        print(f'\n{"─"*70}')
        print(f'  {"":45s}  {"seg":>8}  {"ocr":>8}  {"extend":>8}')
        print(f'  {"":45s}  {"wall":>7}  {"seg":>7}  {"ocr":>7}  {"extend":>7}')
        print(f'  {"AVERAGE":45s}  {avg("wall"):6.0f}ms  {avg("seg"):6.0f}ms  {avg("ocr"):6.0f}ms  {avg("extend"):6.0f}ms')
        print(f'  {"TOTAL":45s}  {tot("wall"):6.0f}ms  {tot("seg"):6.0f}ms  {tot("ocr"):6.0f}ms  {tot("extend"):6.0f}ms')
    print(f'\nDone: {n}/{len(images)} → {args.out_dir}/')


if __name__ == '__main__':
    main()
