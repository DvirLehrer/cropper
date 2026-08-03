#!/usr/bin/env python3
"""Show the segmentation step before and after straightening the image.

    python3 tools/show_pipeline.py --image ../benchmark/02_rotate/<name>.jpg

Produces one picture with four panels:

    1  the photograph as it arrives
    2  the same photograph straightened, using the angle of the writing
    3  what the model marks on panel 1
    4  what the model marks on panel 2

The model runs twice, on the two images, and nothing else differs between the
panels. If panel 4 is tighter than panel 3, running segmentation after the
straightening is worth its cost. If they are equally loose, the model's trouble
is with the picture rather than with its orientation, and no reordering of the
pipeline will help.

The angle comes from the OCR alone, so straightening never depends on the model.
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

PANEL = 620


def _panel(img, title, note=""):
    h, w = img.shape[:2]
    s = min(PANEL / w, PANEL / h)
    tile = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.full((PANEL + 54, PANEL, 3), 26, np.uint8)
    y, x = 54 + (PANEL - tile.shape[0]) // 2, (PANEL - tile.shape[1]) // 2
    canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (235, 235, 235), 1, cv2.LINE_AA)
    if note:
        cv2.putText(canvas, note, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (140, 200, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default=str(REPO / "best.pt"))
    ap.add_argument("--out-dir", default=str(REPO / "test_output" / "pipeline"))
    args = ap.parse_args()

    from ultralytics import YOLO

    import crop_stam
    from crop_with_model import _imgsz_for, _predict
    from stam_io import imread_any

    img = imread_any(args.image)
    if img is None:
        raise SystemExit(f"unreadable: {args.image}")
    model = YOLO(args.model)

    def segment(on):
        h, w = on.shape[:2]
        poly = _predict(model, on, crop_stam.CONF, _imgsz_for(w, h))
        mask = np.zeros((h, w), np.uint8)
        if poly is not None:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        area = 100.0 * int(cv2.countNonZero(mask)) / (h * w)
        vis = on.copy()
        vis[mask > 0] = (0.45 * vis[mask > 0]
                         + 0.55 * np.array([0, 220, 60])).astype(np.uint8)
        cv2.drawContours(vis, cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[0],
                         -1, (0, 0, 255), max(2, int(min(h, w) * 0.004)))
        return vis, area

    # The angle of the writing, measured without the model.
    with ThreadPoolExecutor(max_workers=1) as ex:
        boxes = ex.submit(crop_stam.run_ocr, img).result()
    theta = crop_stam.text_angle(boxes)
    if theta is None:
        raise SystemExit("no OCR characters — cannot measure an angle")

    bg = tuple(int(c) for c in np.median(img.reshape(-1, 3)[::97], axis=0))
    straight, _ = crop_stam._rotate_about_centre(img, theta, bg)

    seg_orig, area_orig = segment(img)
    seg_straight, area_straight = segment(straight)

    print(f"\nwriting angle          {theta:+.1f}°")
    print(f"model on the original  {area_orig:.1f}% of frame")
    print(f"model after straighten {area_straight:.1f}% of frame")
    verdict = ("tighter after straightening"
               if area_straight < area_orig - 3 else
               "no better after straightening")
    print(f"verdict                {verdict}")

    strip = np.hstack([
        _panel(img, "1  as photographed"),
        _panel(straight, "2  straightened", f"rotated {-theta:+.1f} deg"),
        _panel(seg_orig, "3  model on 1", f"marks {area_orig:.1f}% of frame"),
        _panel(seg_straight, "4  model on 2", f"marks {area_straight:.1f}% of frame"),
    ])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.image).stem}_pipeline.jpg"
    cv2.imwrite(str(path), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
