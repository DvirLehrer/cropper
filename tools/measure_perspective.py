#!/usr/bin/env python3
"""Measure keystone in a finished crop, so the question can be settled by
numbers rather than by looking.

    python3 tools/measure_perspective.py --run final3
    python3 tools/measure_perspective.py --run final3 --category 03_perspective

A sheet photographed square-on gives parallel text lines of equal height. A
sheet photographed at an angle gives lines that fan and letters that shrink as
they recede. Deskew corrects the average angle and can do nothing about either.

Two figures per image:

  fan    degrees between the first text line and the last, after the overall
         slope is removed. Parallel lines give ~0.
  taper  how much letter height changes from the top of the block to the
         bottom, as a percentage of the mean. A square-on photograph gives a
         few percent from natural variation in the hand.

Neither needs Vision: letters are found as ink blobs, grouped into lines by
their vertical position, and each line is fitted by least squares.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def blobs(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Ink blobs that are plausibly letters -> (cx, cy, height)."""
    work = gray
    scale = 1.0
    if max(work.shape) > 1600:
        scale = 1600 / max(work.shape)
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    work = cv2.GaussianBlur(work, (3, 3), 0)
    _, ink = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(ink, 8)
    hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n)]
    if len(hs) < 30:
        return []
    med = float(np.median(hs))
    out = []
    for i in range(1, n):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        # a letter, not a speck, not a whole merged line
        if not (0.45 * med <= h <= 2.2 * med):
            continue
        if w > 6 * med or area < 0.1 * h * w:
            continue
        out.append((cent[i][0] / scale, cent[i][1] / scale, h / scale))
    return out


def lines(bl: list[tuple[float, float, float]]) -> list[list]:
    """Group blobs into text lines by vertical position."""
    if not bl:
        return []
    med_h = float(np.median([b[2] for b in bl]))
    bl = sorted(bl, key=lambda b: b[1])
    groups, cur = [], [bl[0]]
    for b in bl[1:]:
        if b[1] - cur[-1][1] <= 0.6 * med_h:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    return [g for g in groups if len(g) >= 8]


def measure(path: Path) -> dict | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ls = lines(blobs(gray))
    if len(ls) < 4:
        return {"lines": len(ls), "fan": None, "taper": None}

    angles, ys, heights = [], [], []
    for g in ls:
        xs = np.array([b[0] for b in g])
        yy = np.array([b[1] for b in g])
        if xs.max() - xs.min() < 1e-6:
            continue
        slope = np.polyfit(xs, yy, 1)[0]
        angles.append(np.degrees(np.arctan(slope)))
        ys.append(float(np.mean(yy)))
        heights.append(float(np.median([b[2] for b in g])))
    if len(angles) < 4:
        return {"lines": len(ls), "fan": None, "taper": None}

    a = np.array(angles)
    y = np.array(ys)
    h = np.array(heights)
    # fan: the systematic part of the angle change down the block, not the noise
    fan = float(np.polyfit(y, a, 1)[0] * (y.max() - y.min()))
    taper = float(np.polyfit(y, h, 1)[0] * (y.max() - y.min()) / max(np.mean(h), 1e-6) * 100)
    return {"lines": len(ls), "fan": round(abs(fan), 2), "taper": round(abs(taper), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final3")
    ap.add_argument("--category")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    run = REPO / "test_output" / "bench" / args.run
    report = run / "report.csv"
    if not report.exists():
        raise SystemExit(f"no report at {report}")
    rows = list(csv.DictReader(report.open(encoding="utf-8-sig")))
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]

    crops = run / "crops"
    index = {p.stem: p for p in crops.rglob("*_cropped.jpg")}

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    out = []
    for r in rows:
        stem = Path(r["image"]).stem + "_cropped"
        p = index.get(stem)
        if p is None:
            continue
        m = measure(p)
        if m is None:
            continue
        out.append({"category": r["category"], "image": Path(r["image"]).name,
                    "text": f(r["text_match_pct"]), **m})

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    out.sort(key=lambda d: -(d["fan"] or 0))
    print(f"{'image':<34}{'cat':<16}{'lines':>6}{'fan':>7}{'taper':>7}{'text':>7}")
    print("-" * 77)
    for d in out:
        fan = "-" if d["fan"] is None else f"{d['fan']:.2f}"
        tap = "-" if d["taper"] is None else f"{d['taper']:.1f}"
        print(f"{d['image'][:33]:<34}{d['category']:<16}{d['lines']:>6}"
              f"{fan:>7}{tap:>7}{d['text']:>7.0f}")

    ok = [d for d in out if d["fan"] is not None]
    if ok:
        print(f"\n{'category':<16}{'n':>4}{'median fan':>12}{'median taper':>14}")
        cats = sorted({d["category"] for d in ok})
        for c in cats:
            g = [d for d in ok if d["category"] == c]
            print(f"{c:<16}{len(g):>4}{np.median([d['fan'] for d in g]):>12.2f}"
                  f"{np.median([d['taper'] for d in g]):>14.1f}")
        print(f"{'ALL':<16}{len(ok):>4}{np.median([d['fan'] for d in ok]):>12.2f}"
              f"{np.median([d['taper'] for d in ok]):>14.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
