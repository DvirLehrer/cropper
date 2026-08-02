#!/usr/bin/env python3
"""Check the rotation heuristic against cached Vision results — no API calls.

`crop_stam.py`'s `_ocr_is_vertical()` rotates whenever the vertical spread of
OCR character centroids exceeds the horizontal spread. The suspicion (see
HANDOFF.md) is that this fires on any portrait parchment with short lines,
not just genuinely vertical text — and is the likely cause of the rotate
category's 105% error rate in the company's July 2026 report.

This script finds the overlap between `test_output/*_char_boxes.json` (436
cached Vision results from past runs) and `../benchmark/manifest.csv` (the
current 157-image frozen benchmark), then compares:

  - `old`: the current heuristic, `_ocr_is_vertical()` verbatim.
  - `new`: dominant nearest-neighbour offset direction — for each character,
    find its nearest neighbour by centroid distance and classify that pair as
    horizontal or vertical; majority vote across all characters. This tracks
    local text flow instead of the global bounding-box shape, so short lines
    on a portrait image no longer read as "vertical".

Ground truth: images in `02_rotate` should trigger rotation; every other
category should not. Caveat found while writing this: none of the 28 images
that overlap the cache fall in `02_rotate` (all are `01_cropper` /
`05_drawings`), so this only measures the FALSE-POSITIVE rate — whether the
heuristics correctly leave non-rotate images alone. Confirming the TRUE
positive rate (does `new` still catch genuinely vertical scrolls) needs a real
Vision call on `02_rotate` images, i.e. `bench.py crop --run baseline`.

Usage:
    python3 tools/check_rotation_heuristic.py
"""
import csv
import glob
import json
import os

import numpy as np

MANIFEST     = os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark', 'manifest.csv')
CHAR_BOXES   = os.path.join(os.path.dirname(__file__), '..', 'test_output', '*_char_boxes.json')
NEIGHBOR_CAP = 6  # skip a nearest-neighbour pair if farther than this * median char size


def load_manifest_by_name(path: str) -> dict:
    """Map every filename an image is known under -> its manifest row."""
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    by_name = {}
    for r in rows:
        names = [r['filename']]
        if r['duplicate_names']:
            names += [n.strip() for n in r['duplicate_names'].split(';') if n.strip()]
        for n in names:
            by_name.setdefault(n, r)
    return by_name


def ocr_is_vertical_old(boxes: list) -> bool:
    """Verbatim copy of crop_stam.py's current heuristic."""
    if not boxes:
        return False
    cx = [sum(v['x'] for v in b['vertices']) / len(b['vertices']) for b in boxes]
    cy = [sum(v['y'] for v in b['vertices']) / len(b['vertices']) for b in boxes]
    return (max(cy) - min(cy)) > (max(cx) - min(cx))


def line_direction_vertical(boxes: list) -> bool:
    """Dominant nearest-neighbour offset direction: horizontal text flow -> False."""
    if len(boxes) < 3:
        return ocr_is_vertical_old(boxes)

    cents = np.array([[sum(v['x'] for v in b['vertices']) / len(b['vertices']),
                        sum(v['y'] for v in b['vertices']) / len(b['vertices'])] for b in boxes])
    sizes = []
    for b in boxes:
        xs = [v['x'] for v in b['vertices']]
        ys = [v['y'] for v in b['vertices']]
        sizes.append(max(max(xs) - min(xs), max(ys) - min(ys)))
    cap = np.median(sizes) * NEIGHBOR_CAP

    horiz = vert = 0
    for i in range(len(cents)):
        d = cents - cents[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        dist[i] = np.inf
        j = np.argmin(dist)
        if dist[j] > cap:
            continue
        dx, dy = abs(d[j, 0]), abs(d[j, 1])
        if dx > dy:
            horiz += 1
        else:
            vert += 1

    if horiz + vert == 0:
        return ocr_is_vertical_old(boxes)
    return vert > horiz


def main():
    by_name = load_manifest_by_name(MANIFEST)

    results = []
    for f in sorted(glob.glob(CHAR_BOXES)):
        d = json.load(open(f))
        base = os.path.basename(d.get('image', ''))
        row = by_name.get(base)
        if row is None:
            continue
        boxes = d['boxes']
        results.append({
            'file': base,
            'category': row['category'],
            'aspect': row['aspect'],
            'old': ocr_is_vertical_old(boxes),
            'new': line_direction_vertical(boxes),
            'n_boxes': len(boxes),
        })

    print(f'{len(results)} of {len(glob.glob(CHAR_BOXES))} cached Vision results overlap the frozen benchmark.\n')
    print(f'{"file":42s} {"category":12s} {"aspect":8s} {"old":6s} {"new":6s} n')
    for r in results:
        diff = '  <-- DIFF' if r['old'] != r['new'] else ''
        print(f'{r["file"]:42s} {r["category"]:12s} {r["aspect"]:8s} '
              f'{str(r["old"]):6s} {str(r["new"]):6s} {r["n_boxes"]}{diff}')

    non_rotate = [r for r in results if r['category'] != '02_rotate']
    old_fp = sum(r['old'] for r in non_rotate)
    new_fp = sum(r['new'] for r in non_rotate)
    rotate = [r for r in results if r['category'] == '02_rotate']

    print(f'\nNon-rotate images in overlap: {len(non_rotate)}')
    print(f'  old heuristic false-positive rotations: {old_fp}')
    print(f'  new heuristic false-positive rotations: {new_fp}')
    if rotate:
        print(f'\n02_rotate images in overlap: {len(rotate)} (true-positive rate measurable)')
    else:
        print('\n02_rotate images in overlap: 0 — true-positive rate NOT measurable from cache. '
              'Run tools/bench.py crop --run baseline for that.')


if __name__ == '__main__':
    main()
