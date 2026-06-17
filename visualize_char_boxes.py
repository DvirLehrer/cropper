#! /usr/bin/env python3
"""
Visualise character bounding boxes and enclosing polygon for STaM manuscript images.

For each *_char_boxes.json, draws:
  - All Hebrew character quads (thin blue outline)
  - The tight enclosing polygon (row-based, green, thick)

Usage:
  python visualize_char_boxes.py                    # first 10 JSONs in test_output/
  python visualize_char_boxes.py --limit 10 --out test_output/debug_poly
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

HEBREW_RANGE = ('א', 'ת')   # alef–tav


# ---------------------------------------------------------------------------
# Core: enclosing polygon
# ---------------------------------------------------------------------------

def _cluster_rows(boxes):
    """Group character boxes into text lines by y-center proximity."""
    def cy(b): return sum(v['y'] for v in b['vertices']) / 4.0
    def bh(b):
        ys = [v['y'] for v in b['vertices']]
        return max(ys) - min(ys)

    lh = float(np.median([bh(b) for b in boxes])) if boxes else 20.0
    threshold = lh * 0.55

    sorted_boxes = sorted(boxes, key=cy)
    rows, row_means = [], []
    for b in sorted_boxes:
        y = cy(b)
        if not rows or y - row_means[-1] > threshold:
            rows.append([b])
            row_means.append(y)
        else:
            rows[-1].append(b)
            n = len(rows[-1])
            row_means[-1] = row_means[-1] * (n - 1) / n + y / n
    return rows


def _row_boundary(rows, padding, side='right'):
    """
    Build a staircase boundary for one side, guaranteeing all character corners
    remain inside the polygon.

    Inward steps (boundary moving toward text center) are deferred until AFTER
    the current row's last character y, so no char corner is ever left outside.
    Outward steps happen as early as possible.
    """
    is_right = (side == 'right')

    rd = []
    for row in rows:
        xs = [v['x'] for b in row for v in b['vertices']]
        ys = [v['y'] for b in row for v in b['vertices']]
        rd.append({
            'x':    (max(xs) + padding) if is_right else (min(xs) - padding),
            'ymin': min(ys) - padding,
            'ymax': max(ys) + padding,
        })

    pts = [(rd[0]['x'], rd[0]['ymin'])]
    for i in range(len(rd) - 1):
        cur, nxt = rd[i], rd[i + 1]
        stepping_inward = (nxt['x'] < cur['x']) if is_right else (nxt['x'] > cur['x'])
        # Inward: wait until current row's chars are fully behind us
        # Outward: step as early as possible to cover next row's chars
        y_step = max(cur['ymax'], nxt['ymin']) if stepping_inward else min(cur['ymax'], nxt['ymin'])
        pts.append((cur['x'], y_step))
        pts.append((nxt['x'], y_step))
    pts.append((rd[-1]['x'], rd[-1]['ymax']))
    return pts


def enclosing_polygon(boxes, max_sides=20, min_sides=4, padding=6):
    """
    Tight polygon enclosing the text region by tracing per-row left/right margins.

    Groups boxes into text lines, builds a safe staircase polygon from per-row
    x-extents, then simplifies with Douglas-Peucker to max_sides. All character
    box corners are guaranteed to be inside the polygon.
    """
    if not boxes:
        return None

    rows = _cluster_rows(boxes)
    if not rows:
        return None

    right_pts = _row_boundary(rows, padding, side='right')
    left_pts  = _row_boundary(rows, padding, side='left')

    # Right side top→bottom, left side bottom→top
    poly = np.array(right_pts + list(reversed(left_pts)), dtype=np.float32)

    # Simplify with Douglas-Peucker to max_sides
    eps = 1.0
    simplified = poly
    for _ in range(60):
        s = cv2.approxPolyDP(poly.reshape(-1, 1, 2), eps, closed=True)
        simplified = s.reshape(-1, 2)
        if len(simplified) <= max_sides:
            break
        eps *= 1.3

    if len(simplified) < min_sides:
        return None

    return np.round(simplified).astype(np.int32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_debug(json_path: str, out_dir: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_path = data.get("image", "")
    img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img is None:
        print(f"  SKIP: cannot load {img_path}")
        return

    boxes_heb = [b for b in data.get("boxes", [])
                 if b.get("char", "") and HEBREW_RANGE[0] <= b["char"][0] <= HEBREW_RANGE[1]]

    if not boxes_heb:
        print(f"  SKIP: no Hebrew boxes in {os.path.basename(json_path)}")
        return

    overlay = img.copy()

    # Draw individual character quads (thin blue)
    for b in boxes_heb:
        verts = b.get("vertices", [])
        if len(verts) < 4:
            continue
        pts = np.array([[v["x"], v["y"]] for v in verts], dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(220, 100, 0), thickness=1)

    # Compute and draw enclosing polygon (thick green)
    poly = enclosing_polygon(boxes_heb)
    if poly is not None:
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 220, 50), thickness=6)
        # Label each vertex with its index
        for i, (x, y) in enumerate(poly):
            cv2.circle(overlay, (int(x), int(y)), 12, (0, 220, 50), -1)
            cv2.putText(overlay, str(i), (int(x)+14, int(y)+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 50), 2)

        n_sides = len(poly)
        cv2.putText(overlay, f"polygon: {n_sides} sides  |  {len(boxes_heb)} Hebrew chars",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 50), 3)

    # Blend overlay with original for semi-transparency on boxes
    result = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    os.makedirs(out_dir, exist_ok=True)
    base  = os.path.splitext(os.path.basename(json_path))[0].replace("_char_boxes", "")
    out_p = os.path.join(out_dir, f"{base}_debug.jpg")
    cv2.imwrite(out_p, result, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {base}: {n_sides if poly is not None else '?'}-sided polygon  →  {out_p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-dir", default="test_output")
    parser.add_argument("--out",     default="test_output/debug_poly")
    parser.add_argument("--limit",   type=int, default=10)
    args = parser.parse_args()

    jsons = sorted(glob.glob(os.path.join(args.ocr_dir, "*_char_boxes.json")))
    if args.limit:
        jsons = jsons[:args.limit]

    print(f"Processing {len(jsons)} images → {args.out}/\n")
    for jp in jsons:
        draw_debug(jp, args.out)


if __name__ == "__main__":
    main()
