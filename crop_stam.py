#!/usr/bin/env python3
"""
crop_stam.py — Production CLI: crop STaM manuscript images to the text region.

For each input image:
  1. YOLO segmentation + Google Vision OCR run in parallel
  2. OCR boxes in text clusters adjacent to the model polygon are included
  3. Union of model polygon and OCR convex hull = final boundary
  4. Outside the boundary is filled with the sampled parchment background colour
  5. Image is cropped to the boundary bounding box and saved

Usage:
  python3 crop_stam.py --image images/benchmark/mezuza3.jpeg
  python3 crop_stam.py --images-dir images/benchmark --out-dir test_output/cropped
  python3 crop_stam.py --images-dir images/before_crop --model best.pt
"""
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision
from ultralytics import YOLO

from crop_with_model import _predict, _tiled_predict, _imgsz_for, SPLIT_RATIO

# ── constants ──────────────────────────────────────────────────────────────────
IMG_EXTS        = {'.jpg', '.jpeg', '.png'}
MODEL_PATH      = 'best.pt'
OUT_DIR         = 'test_output/cropped'
CONF            = 0.25
OCR_MAX_PIXELS  = 1_000_000   # downscale before sending to Vision API
CLUSTER_GAP_PCT = 0.04        # dilation to bridge inter-character gaps (% of short dim)
MODEL_REACH_PCT = 0.04        # dilation on model mask to catch border boxes
EXTEND_WORK_PX  = 1000        # max dimension for connectedComponents work canvas
HEBREW_RANGE    = ('א', 'ת')
ROTATE_RATIO    = 3.0         # rotate CCW when crop H/W exceeds this

# ── Vision API client (singleton) ──────────────────────────────────────────────
_vision_client = None

def _get_vision_client():
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


# ── OCR ────────────────────────────────────────────────────────────────────────
def run_ocr(img_source) -> list:
    """Call Google Vision document_text_detection.
    img_source: file path (str) or numpy BGR array (e.g. after rotation).
    Downscales to ≤OCR_MAX_PIXELS before sending.
    Returns list of {char, vertices} dicts (Hebrew chars only) in original pixel coords."""
    if isinstance(img_source, np.ndarray):
        raw = img_source
    else:
        raw = cv2.imread(img_source, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if raw is None:
        return []
    h, w  = raw.shape[:2]
    scale = 1.0
    if h * w > OCR_MAX_PIXELS:
        scale = (OCR_MAX_PIXELS / (h * w)) ** 0.5
        raw   = cv2.resize(raw, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return []
    content = buf.tobytes()

    inv      = 1.0 / scale
    response = _get_vision_client().document_text_detection(
        image=vision.Image(content=content))
    if response.error.message:
        return []

    boxes = []
    fta   = response.full_text_annotation
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


# ── OCR extension ──────────────────────────────────────────────────────────────
def adjacent_ocr_corners(boxes: list, model_mask: np.ndarray, H: int, W: int):
    """Group Hebrew OCR boxes into connected text clusters; return corners of all
    clusters that touch the model polygon.

    A 'cluster' is a set of boxes whose dilated footprints are connected (bridges
    inter-character gaps). If the model polygon touches any box in a cluster, the
    entire cluster is included — so the model capturing one end of a text line
    automatically pulls in the rest of the line.

    All mask operations run at ≤EXTEND_WORK_PX resolution for speed.
    Returns (Nx2 float32 array of corners in original coords, set of accepted indices).
    """
    if not boxes or model_mask is None:
        return None, set()

    sc       = min(1.0, EXTEND_WORK_PX / max(H, W))
    wW, wH   = max(1, round(W * sc)), max(1, round(H * sc))
    wm       = cv2.resize(model_mask, (wW, wH), interpolation=cv2.INTER_NEAREST)

    ocr_mask = np.zeros((wH, wW), dtype=np.uint8)
    for b in boxes:
        verts = b.get('vertices', [])
        if verts:
            pts = np.array([[int(round(v['x'] * sc)), int(round(v['y'] * sc))]
                            for v in verts], np.int32)
            cv2.fillPoly(ocr_mask, [pts], 255)

    gap_px = max(3, int(min(wH, wW) * CLUSTER_GAP_PCT))
    k_gap  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap_px+1, 2*gap_px+1))
    num_labels, labels = cv2.connectedComponents(cv2.dilate(ocr_mask, k_gap))

    reach_px    = max(3, int(min(wH, wW) * MODEL_REACH_PCT))
    k_reach     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*reach_px+1, 2*reach_px+1))
    model_reach = cv2.dilate(wm, k_reach)

    connected = set()
    for lbl in range(1, num_labels):
        if cv2.countNonZero(cv2.bitwise_and((labels == lbl).astype(np.uint8), model_reach)):
            connected.add(lbl)

    accepted, pts = set(), []
    for i, b in enumerate(boxes):
        for v in b.get('vertices', []):
            x = int(np.clip(round(v['x'] * sc), 0, wW-1))
            y = int(np.clip(round(v['y'] * sc), 0, wH-1))
            if labels[y, x] in connected:
                accepted.add(i)
                for vv in b['vertices']:
                    pts.append([vv['x'], vv['y']])
                break

    return (np.array(pts, dtype=np.float32) if pts else None), accepted


# ── polygon union ──────────────────────────────────────────────────────────────
def union_polygon(H: int, W: int, poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    """Binary-mask OR of two polygons → outer contour in image coords."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly1.astype(np.int32)], 255)
    cv2.fillPoly(mask, [poly2.astype(np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return poly1
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


# ── core per-image function ────────────────────────────────────────────────────
def crop_image(model, img_path: str, out_dir: str, conf: float = CONF) -> bool:
    """Full pipeline for one image. Returns True on success."""
    img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img is None:
        print(f'  SKIP (unreadable): {img_path}')
        return False

    H, W  = img.shape[:2]
    ratio = max(W, H) / min(W, H)
    imgsz = _imgsz_for(W, H)

    # Segmentation + OCR in parallel
    def _seg():
        poly = _predict(model, img_path, conf, imgsz)
        if poly is not None and ratio > SPLIT_RATIO:
            span = (poly[:, 0].max() - poly[:, 0].min()) if W >= H \
                   else (poly[:, 1].max() - poly[:, 1].min())
            if span < max(W, H) * 0.70:
                poly = _tiled_predict(model, img, conf)
        elif poly is None and ratio > SPLIT_RATIO:
            poly = _tiled_predict(model, img, conf)
        return poly

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_seg = ex.submit(_seg)
        f_ocr = ex.submit(run_ocr, img_path)
        poly_model = f_seg.result()
        ocr_boxes  = f_ocr.result()

    model_mask = None
    if poly_model is not None:
        model_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(model_mask, [poly_model.astype(np.int32)], 255)

    # Extend OCR to adjacent text clusters, build convex hull
    adj_pts, _ = adjacent_ocr_corners(ocr_boxes, model_mask, H, W)
    ocr_hull   = cv2.convexHull(adj_pts).reshape(-1, 2) if adj_pts is not None else None

    # Union polygon
    if poly_model is not None and ocr_hull is not None:
        boundary = union_polygon(H, W, poly_model, ocr_hull)
    elif poly_model is not None:
        boundary = poly_model
    elif ocr_hull is not None:
        boundary = ocr_hull
    else:
        print(f'  NO DETECTION: {os.path.basename(img_path)}')
        return False

    # Sample parchment background colour from non-ink pixels inside model polygon
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

    # Mask and crop
    union_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(union_mask, [boundary.astype(np.int32)], 255)
    result              = np.full_like(img, bg_color)
    result[union_mask == 255] = img[union_mask == 255]

    x, y, w, h = cv2.boundingRect(boundary.astype(np.int32))
    cropped    = result[y:y+h, x:x+w]

    # Rotate very vertical crops CCW (ratio based on actual text region, not full image)
    if h / max(w, 1) >= ROTATE_RATIO:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Background contrast reduction: only on large, roughly square images.
    # Skipped for low-res (strokes too thin) and high-ratio (elongated strips).
    _ch, _cw = cropped.shape[:2]
    if min(_ch, _cw) >= 500 and max(_ch, _cw) / min(_ch, _cw) <= 4:
        lab       = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
        l, a, b   = cv2.split(lab)
        thresh, _ = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask   = l >= min(255, int(thresh) + 20)
        if bg_mask.any():
            lf        = l.astype(np.float32)
            mean_bg   = float(lf[bg_mask].mean())
            lf[bg_mask] = mean_bg + (lf[bg_mask] - mean_bg) * 0.3
            l_new     = np.clip(lf, 0, 255).astype(np.uint8)
            cropped   = cv2.cvtColor(cv2.merge([l_new, a, b]), cv2.COLOR_LAB2BGR)

    os.makedirs(out_dir, exist_ok=True)
    stem     = Path(img_path).stem
    out_path = os.path.join(out_dir, f'{stem}_cropped{Path(img_path).suffix}')
    cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'  {os.path.basename(img_path):50s} → {w}x{h}  {out_path}')
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Crop STaM manuscript images to text region.')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',      help='Single input image')
    src.add_argument('--images-dir', help='Directory of input images')
    parser.add_argument('--model',   default=MODEL_PATH, help='YOLOv8 model weights')
    parser.add_argument('--out-dir', default=OUT_DIR,    help='Output directory')
    parser.add_argument('--conf',    type=float, default=CONF, help='Detection confidence threshold')
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.image:
        images = [args.image]
    else:
        images = sorted(p for p in glob.glob(os.path.join(args.images_dir, '*'))
                        if Path(p).suffix.lower() in IMG_EXTS)

    print(f'Processing {len(images)} image(s) with {args.model}\n')
    ok = sum(crop_image(model, p, args.out_dir, args.conf) for p in images)
    print(f'\nDone: {ok}/{len(images)} → {args.out_dir}/')


if __name__ == '__main__':
    main()
