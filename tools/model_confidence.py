#!/usr/bin/env python3
"""What the segmentation model returns, and how sure it is.

    python3 tools/model_confidence.py --images-dir ../benchmark/02_rotate
    python3 tools/model_confidence.py --benchmark ../benchmark        # all folders

Runs the model only — no Vision call, so it costs nothing and can be run over
the whole benchmark freely.

The question it answers: is the model's own confidence a usable signal for when
to distrust it? One rotate image comes back with a single mask covering 82.6% of
the frame at confidence 0.43, against a detection threshold of 0.25. If sound
detections sit far above that, raising the threshold fixes the failure with one
number and no new heuristics. If they overlap, confidence is not the answer and
something else has to separate them.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images-dir")
    src.add_argument("--benchmark", type=Path)
    ap.add_argument("--model", default=str(REPO / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    from crop_with_model import _imgsz_for
    from stam_io import list_images

    if args.benchmark:
        folders = sorted(d for d in args.benchmark.iterdir()
                         if d.is_dir() and d.name[0].isdigit())
    else:
        folders = [Path(args.images_dir)]

    model = YOLO(args.model)
    print(f"{'image':<44}{'masks':>6}{'area%':>8}{'conf':>7}")
    print("-" * 65)

    rows = []
    for folder in folders:
        if len(folders) > 1:
            print(f"\n[{folder.name}]")
        for path in list_images(folder):
            from stam_io import imread_any

            img = imread_any(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            r = model.predict(img, imgsz=_imgsz_for(W, H), conf=args.conf,
                              verbose=False)[0]
            if r.masks is None or not len(r.masks):
                print(f"{Path(path).name[:42]:<44}{0:>6}{'-':>8}{'-':>7}")
                rows.append((folder.name, Path(path).name, 0, None, None))
                continue
            confs = r.boxes.conf.tolist() if r.boxes is not None else []
            merged = np.zeros((H, W), np.uint8)
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(merged, [poly.astype(np.int32)], 255)
            area = 100.0 * int(cv2.countNonZero(merged)) / (H * W)
            best = max(confs) if confs else None
            print(f"{Path(path).name[:42]:<44}{len(r.masks):>6}{area:>8.1f}"
                  + (f"{best:>7.2f}" if best is not None else f"{'-':>7}"))
            rows.append((folder.name, Path(path).name, len(r.masks), area, best))

    scored = [r for r in rows if r[4] is not None]
    if scored:
        print("\n" + "-" * 65)
        # The interesting split: does confidence separate a plausible detection
        # from one that has claimed most of the picture?
        big = [r[4] for r in scored if r[3] and r[3] > 60]
        ok = [r[4] for r in scored if r[3] and r[3] <= 60]
        if big:
            print(f"masks covering >60% of frame : n={len(big):<4} "
                  f"median confidence {statistics.median(big):.2f}  "
                  f"max {max(big):.2f}")
        if ok:
            print(f"masks covering <=60%         : n={len(ok):<4} "
                  f"median confidence {statistics.median(ok):.2f}  "
                  f"min {min(ok):.2f}")
        if big and ok:
            if max(big) < min(ok):
                print(f"\nconfidence separates them cleanly: "
                      f"a threshold anywhere in {max(big):.2f}-{min(ok):.2f} works.")
            else:
                print("\nconfidence does NOT separate them — the ranges overlap, "
                      "so raising the threshold would discard good detections too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
