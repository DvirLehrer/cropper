#!/usr/bin/env python3
"""Probe: does Google Vision still read text on the arbitrarily-skewed 02_rotate images?

The 02_rotate benchmark folder is not "portrait vs landscape, flip 90°" — it is
parchment strips photographed at arbitrary in-plane angles. The planned fix is a
quad -> rectangle warp whose angle comes from the OCR character boxes, which is
only viable if Vision returns usable boxes at those angles.

All 436 cached char_boxes JSONs are essentially axis-aligned (434 within 0.5°),
so that question cannot be answered offline. This script answers it with one
Vision call per rotate image, caching results so a re-run is free.

Per image it reports:
  n_boxes   how many Hebrew character boxes came back (0 = Vision failed at angle)
  angle     median character-box baseline angle, i.e. the deskew angle
  mad       median absolute deviation of that angle (low = consistent = trustworthy)
  rect      minAreaRect angle over all box corners, as an independent cross-check

Writes an overlay per image so the boxes can be eyeballed against the parchment.

Usage:
    python3 tools/probe_rotate_ocr.py
    python3 tools/probe_rotate_ocr.py --folder ../benchmark/03_perspective
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crop_stam import run_ocr          # noqa: E402  same downscale/filter as production
from stam_io import imread_any, list_images  # noqa: E402

DEFAULT_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark', '02_rotate')
OUT_DIR        = os.path.join(os.path.dirname(__file__), '..', 'test_output', 'rotate_probe')
WORKERS        = 4


def box_angle(b: dict):
    """Baseline direction of one character box, from its first edge, in degrees."""
    v = b.get('vertices') or []
    if len(v) < 2:
        return None
    dx, dy = v[1]['x'] - v[0]['x'], v[1]['y'] - v[0]['y']
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def circular_median_180(angles: list):
    """Median of angles that are only meaningful mod 180° (a line has no head/tail).

    Doubling the angles maps the mod-180 space onto a full circle, where a
    vector mean is well defined; halving the result brings it back.
    """
    if not angles:
        return None, None
    rad = np.radians(np.array(angles) * 2.0)
    mean = math.degrees(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) / 2.0
    resid = [(a - mean + 90) % 180 - 90 for a in angles]
    return mean, float(np.median(np.abs(resid)))


def probe(path: str) -> dict:
    stem = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(OUT_DIR, f'{stem}_char_boxes.json')

    img = imread_any(path)
    if img is None:
        return {'file': os.path.basename(path), 'error': 'undecodable'}

    if os.path.exists(cache):
        boxes = json.load(open(cache))['boxes']
        cached = True
    else:
        boxes = run_ocr(img)
        os.makedirs(OUT_DIR, exist_ok=True)
        json.dump({'image': path, 'boxes': boxes}, open(cache, 'w'), ensure_ascii=False)
        cached = False

    h, w = img.shape[:2]
    rec = {'file': os.path.basename(path), 'dims': f'{w}x{h}',
           'n_boxes': len(boxes), 'cached': cached}

    angles = [a for a in (box_angle(b) for b in boxes) if a is not None]
    med, mad = circular_median_180(angles)
    rec['angle'], rec['mad'] = med, mad

    if boxes:
        pts = np.array([[v['x'], v['y']] for b in boxes for v in b['vertices']], np.float32)
        rec['rect_angle'] = float(cv2.minAreaRect(pts)[2])

        overlay = img.copy()
        for b in boxes:
            q = np.array([[v['x'], v['y']] for v in b['vertices']], np.int32)
            cv2.polylines(overlay, [q], True, (0, 255, 0), max(1, min(h, w) // 400))
        box = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.int32)
        cv2.polylines(overlay, [box], True, (0, 0, 255), max(2, min(h, w) // 200))
        cv2.imwrite(os.path.join(OUT_DIR, f'{stem}_overlay.jpg'), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default=DEFAULT_FOLDER)
    args = ap.parse_args()

    images = list_images(args.folder)
    print(f'Probing {len(images)} images in {os.path.normpath(args.folder)}\n')

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        recs = list(ex.map(probe, images))

    print(f'{"file":52s} {"dims":12s} {"n":>5s} {"angle":>8s} {"mad":>6s} {"rect":>7s}')
    for r in sorted(recs, key=lambda r: r['file']):
        if r.get('error'):
            print(f'{r["file"][:52]:52s} {r["error"]}')
            continue
        ang = f'{r["angle"]:8.2f}' if r['angle'] is not None else '     n/a'
        mad = f'{r["mad"]:6.2f}' if r['mad'] is not None else '   n/a'
        rct = f'{r["rect_angle"]:7.2f}' if 'rect_angle' in r else '    n/a'
        print(f'{r["file"][:52]:52s} {r["dims"]:12s} {r["n_boxes"]:5d} {ang} {mad} {rct}')

    ok = [r for r in recs if not r.get('error')]
    read = [r for r in ok if r['n_boxes'] > 0]
    skew = [r for r in read if r['angle'] is not None
            and min(abs(r['angle'] - k) for k in (-180, -90, 0, 90, 180)) > 5]
    tight = [r for r in read if r['mad'] is not None and r['mad'] < 5]

    print(f'\n{len(read)}/{len(ok)} images returned any Hebrew boxes')
    print(f'{len(skew)}/{len(read)} of those are >5° off-axis (genuinely skewed)')
    print(f'{len(tight)}/{len(read)} have a consistent angle (MAD < 5°) — usable as a deskew signal')
    print(f'\nOverlays: {os.path.normpath(OUT_DIR)}')


if __name__ == '__main__':
    main()
