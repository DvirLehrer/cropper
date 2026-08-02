#!/usr/bin/env python3
"""Measure how much of the ruling a de-ruling pass actually removed.

Judging this by eye does not scale and does not settle arguments. Two numbers
per image, both computed on non-ink pixels only so the writing cannot influence
them:

  groove_depth  how much darker the ruled rows are than the parchment between
                them. This is the thing we are trying to remove, so lower is
                better after the pass.
  ink_delta     mean change on ink pixels. Must stay at essentially zero — if
                the letters moved, the cure is worse than the disease.

    python3 tools/measure_deruling.py --images-dir ../benchmark/05_drawings
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deruling                                            # noqa: E402
from stam_io import imread_any, list_images                # noqa: E402


def groove_depth(gray: np.ndarray, rows: list[int], period: int,
                 ink: np.ndarray, tracks: dict[int, np.ndarray]) -> float:
    """How much darker the groove is than the parchment beside it.

    Measured **along the tracked path**, not at the predicted row. Ruling bows
    across a page, so a row-average mixes groove and clean parchment together
    and understates both the problem and the fix. What matters downstream is
    whether a continuous dark path still runs between the letters, and that is
    what following the track measures.
    """
    if not rows or period < 4:
        return 0.0
    h, w = gray.shape
    g = gray.astype(np.float32)
    offset = max(2, period // 4)      # clean parchment, half-way to the next line
    on, off = [], []
    xs = np.arange(w)
    for r in rows:
        track = tracks.get(r)
        if track is None:
            continue
        for dy in (-1, 0, 1):
            ys = np.clip(track + dy, 0, h - 1)
            keep = ~ink[ys, xs]
            on.append(g[ys, xs][keep])
        for sign in (-1, 1):
            ys = np.clip(track + sign * offset, 0, h - 1)
            keep = ~ink[ys, xs]
            off.append(g[ys, xs][keep])
    on = np.concatenate([a for a in on if a.size]) if any(a.size for a in on) else None
    off = np.concatenate([a for a in off if a.size]) if any(a.size for a in off) else None
    if on is None or off is None:
        return 0.0
    return float(off.mean() - on.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--images-dir")
    args = ap.parse_args()

    paths = [args.image] if args.image else list_images(args.images_dir)
    print(f"{'image':<44}{'conf':>6}{'lines':>7}{'depth in':>10}"
          f"{'depth out':>11}{'removed':>9}{'ink Δ':>8}{'ms':>7}")
    print("-" * 102)

    removed_all, ink_all = [], []
    for p in paths:
        img = imread_any(p)
        if img is None:
            continue
        t0 = time.perf_counter()
        fixed, info = deruling.remove(img)
        ms = (time.perf_counter() - t0) * 1000

        gray_a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
        # Measure along exactly the geometry the corrector used, at full
        # resolution. Re-deriving the tracks here would score a slightly
        # different path than the one that was actually lifted.
        tracks = info.get("_tracks") or {}
        if not tracks:
            print(f"{Path(p).name[:42]:<44}{info['confidence']:>6.2f}{0:>7}"
                  f"{'-':>10}{'-':>11}{'-':>9}{'-':>8}{ms:>7.0f}")
            continue
        rows = sorted(tracks)
        period = info["_period_full"]
        ink = info["_ink_full"]

        before = groove_depth(gray_a, rows, period, ink, tracks)
        after = groove_depth(gray_b, rows, period, ink, tracks)
        pct = 100 * (before - after) / before if before > 0.01 else 0.0
        ink_delta = float(np.abs(gray_b[ink].astype(np.float32)
                                 - gray_a[ink].astype(np.float32)).mean()) if ink.any() else 0.0

        removed_all.append(pct)
        ink_all.append(ink_delta)
        print(f"{Path(p).name[:42]:<44}{info['confidence']:>6.2f}{len(rows):>7}"
              f"{before:>10.2f}{after:>11.2f}{pct:>8.0f}%{ink_delta:>8.2f}{ms:>7.0f}")

    if removed_all:
        print("-" * 102)
        print(f"{'MEDIAN':<44}{'':>6}{'':>7}{'':>10}{'':>11}"
              f"{statistics.median(removed_all):>8.0f}%{statistics.median(ink_all):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
