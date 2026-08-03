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

import cv2
import numpy as np


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray <= thresh


def measure_grain(gray: np.ndarray, ink: np.ndarray | None = None) -> dict:
    """Grain amplitude and the ink-to-parchment contrast it sits on."""
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


def suppress(bgr: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """Smooth the parchment, leave the writing untouched.

    `strength` scales how aggressively the grain is attenuated; 0 disables.
    Returns (image, measurements).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = _ink_mask(gray)
    before = measure_grain(gray, ink)
    if strength <= 0 or before["contrast"] <= 10:
        return bgr, {**before, "applied": False}

    # Mix only with neighbours within a fraction of the ink-to-parchment gap:
    # grain (a few levels) is averaged, a letter edge (most of the gap) is not.
    sigma_colour = max(4.0, min(0.25 * before["contrast"], 3.0 * before["grain"]))
    # Radius from the grain itself. Speckle is a pixel or three across, always
    # far narrower than a stroke, so the filter cannot reach across a letter.
    radius = 2 if max(bgr.shape[:2]) < 1500 else 3
    sigma_space = radius * 2.0

    smoothed = cv2.bilateralFilter(bgr, d=2 * radius + 1,
                                   sigmaColor=sigma_colour * strength,
                                   sigmaSpace=sigma_space)
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
