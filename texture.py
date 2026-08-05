#!/usr/bin/env python3
"""Weaken the grain of the parchment without softening the writing.

The problem this addresses is not cropping. On rough skin the downstream engine
cannot tell a grain of the surface from a stroke of a letter, so it reads
texture as ink: the roughness folder is the worst category in the benchmark by a
wide margin, with the engine recovering barely half the text.

Why the existing step does not solve it
--------------------------------------
`crop_stam.FLATTEN_BG` pulls the luminance of background pixels toward their
mean. That evens out *lighting* — a bright corner, a shadow — which is a
low-frequency problem. Grain is the opposite: high frequency, low amplitude,
spread evenly. Averaging toward the mean barely touches it.

What is done instead
--------------------
An edge-preserving smooth. A bilateral filter mixes a pixel only with
neighbours of similar brightness, so grain — which differs from its
surroundings by a few levels — is averaged away, while the boundary of a
letter — which differs by a hundred — is left alone. The colour sigma is
derived from the image's own ink-to-parchment contrast rather than fixed, so
the same setting holds for a pale scan and a dark photograph.

Two safeguards, because softening the writing would be worse than the grain:

* the filter runs only where the roughness actually is, and the ink is written
  back exactly as it was;
* the spatial radius is tied to the measured grain size, not to the letters, so
  the filter is always far smaller than a stroke.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# Smoothing strength, as a multiple of the measured grain. Overridable so it can
# be swept without editing the file.
GRAIN_SIGMA = float(os.environ.get("GRAIN_SIGMA", "1.5"))

# Non-local means instead of a bilateral filter. Measured on the two grainiest
# benchmark crops it removes 65-78% of the speckle against 38-50%, and moves the
# edge of a letter by 0.7-1.5 grey levels against 3.4-4.1.
USE_NLM = os.environ.get("USE_NLM", "1").strip().lower() in ("1", "true", "yes", "on")
NLM_H = float(os.environ.get("NLM_H", "0.5"))
NLM_SEARCH = int(os.environ.get("NLM_SEARCH", "21"))
NLM_SEARCH_BIG = int(os.environ.get("NLM_SEARCH_BIG", "11"))
NLM_BIG_MP = float(os.environ.get("NLM_BIG_MP", "4.0"))
NLM_PATCH = int(os.environ.get("NLM_PATCH", "7"))


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray <= thresh


def ink_mask_from_boxes(gray: np.ndarray, boxes: list) -> np.ndarray | None:
    """Mark the writing using the character boxes the OCR already returned.

    A single global threshold is the weakest way to tell ink from parchment, and
    on rough skin it is wrong in both directions: some grain is dark enough to be
    protected, and part of a thin or faded stroke is not, so the filter softens
    the letter it was supposed to preserve. That is what broke the two images
    where this made things worse rather than better.

    The boxes say where the letters are, which is information from outside the
    image rather than a guess made from it. But a box is a rectangle, so most of
    it is parchment — protecting the whole box would leave grain untouched
    exactly where it does the most damage, right against the strokes.

    So the box supplies the location and a threshold computed *within that box*
    supplies the shape. Thirty-odd pixels of one letter, under even lighting,
    split into ink and parchment far more reliably than a whole page with its
    shadows and gradients. Everything outside the boxes is parchment by
    definition and can be smoothed freely.

    Returns None when there is nothing usable to work from.
    """
    if not boxes:
        return None
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    used = 0
    for b in boxes:
        verts = b.get('vertices') or []
        if len(verts) < 3:
            continue
        xs = [v['x'] for v in verts]
        ys = [v['y'] for v in verts]
        # int(): the vertices are floats once they have been through a warp,
        # and these index the array directly.
        x0, x1 = max(0, int(min(xs))), min(w, int(round(max(xs))) + 1)
        y0, y1 = max(0, int(min(ys))), min(h, int(round(max(ys))) + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        patch = gray[y0:y1, x0:x1]
        # Otsu on the letter alone. A box holds ink and parchment in comparable
        # amounts, which is exactly the case the method is built for.
        t, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask[y0:y1, x0:x1] |= (patch <= t).astype(np.uint8) * 255
        used += 1
    if not used:
        return None
    return mask > 0


def text_area(shape: tuple, boxes: list, pad_frac: float = 0.5):
    """Bounding box of the writing, padded by half a character.

    Measuring grain over the whole crop reads the surface the sheet was lying
    on as though it were the sheet. On the benchmark that put the rotate folder
    at a median of 14.1 — higher than the rough-parchment folder's 7.9 — purely
    because those crops keep corners of table and stone. Half the images then
    cleared the threshold and paid for a filter they did not need.
    """
    if not boxes:
        return None
    xs, ys, hs = [], [], []
    for b in boxes:
        v = b.get('vertices') or []
        if len(v) < 2:
            continue
        bx = [p['x'] for p in v]
        by = [p['y'] for p in v]
        xs += bx
        ys += by
        hs.append(max(by) - min(by))
    if not xs:
        return None
    pad = int(round(pad_frac * (np.median(hs) if hs else 0)))
    h, w = shape[:2]
    # Rounded, because these index an array. The boxes arrive as floats once
    # they have been carried through a warp, and a float slice is a TypeError
    # that kills the whole crop — mezuzah1 came out 0x0 this way.
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(w, int(round(max(xs))) + pad), min(h, int(round(max(ys))) + pad))


def measure_grain(gray: np.ndarray, ink: np.ndarray | None = None,
                  boxes: list | None = None) -> dict:
    """Grain amplitude and the ink-to-parchment contrast it sits on.

    When character boxes are supplied the measurement is confined to the block
    of writing, so what is reported is the roughness of the parchment rather
    than of whatever the parchment was photographed on.
    """
    if boxes:
        area = text_area(gray.shape, boxes)
        if area:
            x0, y0, x1, y1 = area
            if x1 - x0 > 20 and y1 - y0 > 20:
                gray = gray[y0:y1, x0:x1]
                if ink is not None:
                    ink = ink[y0:y1, x0:x1]
    if ink is None:
        ink = _ink_mask(gray)
    parchment = ~ink
    if parchment.sum() < 500:
        return {"grain": 0.0, "contrast": 0.0, "ratio": 0.0}

    g = gray.astype(np.float32)
    detail = np.abs(g - cv2.blur(g, (7, 7)))[parchment]
    # Median absolute deviation: a few specks or a hair must not read as grain.
    grain = float(np.median(detail)) * 1.4826
    contrast = float(np.median(g[parchment]) - np.median(g[ink])) if ink.any() else 0.0
    return {
        "grain": round(grain, 2),
        "contrast": round(contrast, 1),
        "ratio": round(grain / contrast, 4) if contrast > 10 else 0.0,
    }


def suppress(bgr: np.ndarray, strength: float = 1.0,
             boxes: list | None = None) -> tuple[np.ndarray, dict]:
    """Smooth the parchment, leave the writing untouched.

    `strength` scales how aggressively the grain is attenuated; 0 disables.
    `boxes` are the OCR character quads; when given they define the writing far
    more reliably than a threshold over the whole image can.

    Returns (image, measurements).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = ink_mask_from_boxes(gray, boxes) if boxes else None
    if ink is None:
        ink = _ink_mask(gray)
    before = measure_grain(gray, ink)
    if strength <= 0 or before["contrast"] <= 10:
        return bgr, {**before, "applied": False}

    # How far apart two pixels may be in brightness and still be averaged
    # together. This has to come from the grain itself, not from the
    # ink-to-parchment contrast: on a sheet measuring 181 levels of contrast and
    # 13 of grain, a quarter of the contrast is 45 — three times wider than
    # anything the grain spans — and a filter that mixes across 45 levels also
    # mixes across the soft edge of a faded stroke. Measured on the two images
    # this used to ruin, the edges moved by 3.9 grey levels at that setting and
    # by 0.3 at a sigma tied to the grain.
    #
    # Grain and a letter's edge overlap in amplitude, so nothing chosen here
    # separates them completely. Staying close to the grain keeps the damage
    # to a fraction of a grey level, and accepts removing less of the speckle.
    sigma_colour = max(3.0, GRAIN_SIGMA * before["grain"])

    if USE_NLM:
        # Non-local means, and the difference is not marginal:
        #
        #                       noise removed   letter edge moved
        #   bilateral, wide          38-50%          3.4-4.1
        #   non-local means          65-78%          0.7-1.5
        #
        # A bilateral filter judges a neighbour by how far apart the two pixels
        # are in brightness, and grain and the soft edge of a faded stroke
        # occupy the same few grey levels — so it cannot remove one without
        # eroding the other, which is the wall this kept hitting.
        #
        # Non-local means asks a different question: does this patch appear
        # elsewhere in the image? A stroke of a letter recurs hundreds of times
        # across a page of script and is reinforced; a speck of grain occurs
        # once and averages away. The criterion is repetition, not contrast,
        # and STaM — the same alphabet written over and over on one sheet — is
        # close to the ideal case for it.
        #
        # It costs a few hundred milliseconds against five, paid only on the
        # images measured as grainy, and next to the half-second Vision call.
        # Search window, the parameter that dominates the cost — it grows with
        # its square. Measured on the two grainiest crops:
        #
        #   window 21   78% / 91% removed   edge 1.5 / 0.7   1076 / 1340 ms
        #   window 11   75% / 82%           edge 1.3 / 0.7    372 /  552 ms
        #
        # Three times faster for a few points of noise, and no worse at the
        # edges. Downscaling was tried first and is not an option: halving the
        # image before filtering is faster still, but the resample itself moves
        # the letter edges by 5.6 grey levels against 1.5, which defeats the
        # purpose.
        h = max(2.0, NLM_H * before["grain"] * strength)
        # The wide window is worth its cost. Measured over the rough-parchment
        # folder, narrowing it from 21 to 11 lost 5.7 points of text recovered,
        # 5.6 on the under-20 rule, and turned two identifiable scans into
        # unidentifiable ones — for 2.4 seconds a picture.
        #
        # The cost grows with the area of the image as well as with the square
        # of the window, so on a very large crop it reaches ten seconds, which
        # is too long to put in front of a user. Those are also the crops with
        # the most pixels to average over, where the narrower window gives up
        # least. So the window narrows only where it would otherwise be slowest.
        megapixels = bgr.shape[0] * bgr.shape[1] / 1e6
        search = NLM_SEARCH if megapixels <= NLM_BIG_MP else NLM_SEARCH_BIG
        smoothed = cv2.fastNlMeansDenoisingColored(bgr, None, h, h,
                                                   NLM_PATCH, search)
    else:
        radius = 2 if max(bgr.shape[:2]) < 1500 else 3
        smoothed = cv2.bilateralFilter(bgr, d=2 * radius + 1,
                                       sigmaColor=sigma_colour * strength,
                                       sigmaSpace=radius * 2.0)
    # Restore the writing exactly, including its antialiased rim, so no stroke
    # can be thinned by the filter.
    rim = cv2.dilate(ink.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    out = smoothed.copy()
    out[rim] = bgr[rim]

    after = measure_grain(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), ink)
    return out, {
        **before,
        "applied": True,
        "sigma_colour": round(sigma_colour, 1),
        "grain_after": after["grain"],
        "removed_pct": round(100 * (1 - after["grain"] / before["grain"]), 1)
        if before["grain"] > 0.01 else 0.0,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    import statistics
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stam_io import imread_any, list_images

    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--images-dir")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    paths = [args.image] if args.image else list_images(args.images_dir)
    out = Path(args.out_dir) if args.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    print(f"{'image':<44}{'grain':>7}{'after':>7}{'removed':>9}{'ink Δ':>8}{'ms':>7}")
    print("-" * 82)
    removed, inkd = [], []
    for p in paths:
        img = imread_any(p)
        if img is None:
            continue
        t0 = time.perf_counter()
        fixed, info = suppress(img, args.strength)
        ms = (time.perf_counter() - t0) * 1000
        if not info["applied"]:
            print(f"{Path(p).name[:42]:<44}{'-':>7}{'-':>7}{'skipped':>9}{'-':>8}{ms:>7.0f}")
            continue
        g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
        ink = _ink_mask(g0)
        d = float(np.abs(g1[ink].astype(np.float32) - g0[ink].astype(np.float32)).mean()) \
            if ink.any() else 0.0
        removed.append(info["removed_pct"])
        inkd.append(d)
        print(f"{Path(p).name[:42]:<44}{info['grain']:>7.2f}{info['grain_after']:>7.2f}"
              f"{info['removed_pct']:>8.0f}%{d:>8.2f}{ms:>7.0f}")
        if out:
            cv2.imwrite(str(out / f"{Path(p).stem}_smooth.jpg"), fixed,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
    if removed:
        print("-" * 82)
        print(f"{'MEDIAN':<44}{'':>7}{'':>7}{statistics.median(removed):>8.0f}%"
              f"{statistics.median(inkd):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
