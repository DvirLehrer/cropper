#!/usr/bin/env python3
"""Suppress the ruled lines scored into STaM parchment.

Before writing, a scribe rules the parchment with a stylus — a shallow groove,
one per line of text, that the letters hang from. It is not ink: it is a ridge
in the skin, lighter than the writing but darker than its surroundings, and it
runs the full width of the text block straight through the tops of the letters.

Cropping cannot help. When the downstream engine binarises the page, the groove
survives as dark pixels that bridge one letter to the next, which is why the
dominant error the engine reports on these scans is TOUCHING_LETTER_H, followed
by LETTER_SHAPE_CHANGED. Appendix A criterion 5 asks for exactly this: lines
excluded from letter detection.

Method — the same idea as the original `stripes.py`, moved into NumPy
---------------------------------------------------------------------
The ruling is *periodic*: the scribe rules at a fixed spacing. That is the
strongest signal available, and far more robust than trying to find each faint
line on its own.

  1. Build a row profile of darkness from non-ink pixels only, so that the
     letters — which are far darker and unevenly distributed — do not drown out
     the groove.
  2. Autocorrelate the profile to recover the ruling period. A confident peak
     is also the detector: if no period stands out, the page is not ruled and
     nothing is touched.
  3. Place one line at each period, then refine each to the darkest non-ink row
     in its neighbourhood.
  4. Follow each line across the image column by column — ruling is rarely
     perfectly straight — and lift the groove pixels back to the local
     parchment tone. Ink is protected by an explicit mask throughout.

The original did all of this with Python loops over PIL pixel lists, which cost
hours per image; the arithmetic here is identical but vectorised, and runs in
tens of milliseconds.
"""

from __future__ import annotations

import cv2
import numpy as np

# Analysis runs on a downscaled copy: the ruling period is tens of pixels, so
# nothing is lost, and every step below gets cheaper by the square of the ratio.
WORK_LONG_EDGE = 1600

MIN_PERIOD_PX = 8         # closer than this is texture, not ruling
MAX_PERIOD_FRAC = 0.25    # a period longer than this can't be a page of lines
MIN_PERIODICITY = 0.55    # autocorrelation strength below which we do nothing
MIN_CYCLES = 8            # a period must repeat at least this many times before
                          # we believe it. Four ruled lines on a tefilin strip
                          # are not evidence of a rhythm — they are four lines,
                          # and the autocorrelation will happily lock onto the
                          # spacing of the writing instead. Every image this
                          # cost us in measurement was one of those.
MAX_DEPTH_FRAC = 0.5      # a dip more than half-way from parchment to ink is
                          # ink the mask missed, not a scored line
BAND_FRAC = 0.22          # search radius around a predicted line, in periods
GROOVE_PX = 3             # half-width of the correction, at work scale.
                          # This is the groove itself — two or three pixels —
                          # and must not be derived from the period: spreading
                          # the same lift over a band tens of pixels tall
                          # tapers it down to nothing where it is needed.
LIFT = 1.0                # fraction of the groove's darkness to remove

# Two guards against the detector locking onto the rhythm of the *text lines*
# instead of the ruling. On a properly ruled page the two coincide and it makes
# no difference, but on a page whose ink mask is imperfect — a low-contrast
# tefilin strip, say — the leftover darkness of the letters is periodic too, and
# lifting along it bleaches the writing itself.
#
# A real groove is shallow, and it is parchment: much lighter than ink. Both
# guards follow from that and need no tuning per image.
# A ruled line is nearly straight: the scribe drags a stylus once across the
# sheet, so it bows gently at worst. A path that is chasing the edges of letters
# instead jumps up and down from column to column. Measuring how far the tracked
# path strays from its own straight-line fit separates the two cleanly, and
# needs no assumption about contrast, exposure or ink darkness.
MAX_TRACK_WOBBLE = 0.15   # robust residual from the fitted curve, in periods


def _work_scale(shape: tuple[int, int]) -> float:
    return min(1.0, WORK_LONG_EDGE / max(shape[:2]))


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels dark enough to be writing. Deliberately generous — anything we
    call ink is protected, and protecting a little parchment is harmless."""
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray <= thresh


def _row_profile(gray: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Mean darkness per row, ink excluded.

    The groove is a shallow dip of a few grey levels. Averaging the letters in
    would swamp it, so they are masked out and each row is averaged over the
    parchment it actually shows.
    """
    vals = np.where(ink, np.nan, gray.astype(np.float32))
    with np.errstate(invalid="ignore"):
        prof = np.nanmean(vals, axis=1)
    prof = np.nan_to_num(prof, nan=float(np.nanmedian(prof)))
    # Remove slow illumination drift so autocorrelation sees only the ruling.
    smooth = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 0), 0, sigmaY=15).ravel()
    return smooth - prof   # positive where the row is darker than its neighbours


def _dominant_period(profile: np.ndarray) -> tuple[int, float]:
    """Return (period in px, confidence 0-1) of the strongest repeat."""
    sig = profile - profile.mean()
    n = len(sig)
    if n < 4 * MIN_PERIOD_PX:
        return 0, 0.0
    denom = float(np.dot(sig, sig))
    if denom <= 0:
        return 0, 0.0
    hi = int(min(n // 3, n * MAX_PERIOD_FRAC))
    if hi <= MIN_PERIOD_PX:
        return 0, 0.0
    lags = np.arange(MIN_PERIOD_PX, hi)
    scores = np.array([np.dot(sig[:n - k], sig[k:]) / denom for k in lags])
    best = int(np.argmax(scores))
    return int(lags[best]), float(max(0.0, scores[best]))


def _line_rows(profile: np.ndarray, period: int) -> list[int]:
    """One row index per ruled line: the darkest row in each period-wide slot,
    anchored on the strongest response so the grid lands on real lines."""
    n = len(profile)
    anchor = int(np.argmax(profile))
    rows = []
    for direction in (-1, 1):
        y = anchor
        while 0 <= y < n:
            lo, hi = max(0, y - period // 4), min(n, y + period // 4 + 1)
            if hi > lo:
                rows.append(lo + int(np.argmax(profile[lo:hi])))
            y += direction * period
    return sorted(set(rows))


def _follow(gray: np.ndarray, ink: np.ndarray, row: int, band: int) -> np.ndarray:
    """For each column, the row of the groove near `row`.

    Ruling drifts and bows across a page, so a straight line would miss it at
    the edges. Columns whose neighbourhood is all ink keep the predicted row.
    """
    h, w = gray.shape
    lo, hi = max(0, row - band), min(h, row + band + 1)
    if hi - lo < 3:
        return np.full(w, row, dtype=np.int32)
    window = np.where(ink[lo:hi], np.uint8(255), gray[lo:hi])
    track = lo + np.argmin(window, axis=0).astype(np.int32)
    # A median along x keeps one noisy column from dragging the line.
    return cv2.medianBlur(track.astype(np.float32).reshape(1, -1), 5).ravel().astype(np.int32)


def detect(gray: np.ndarray) -> dict:
    """Measure the ruling without modifying anything.

    Returns period, confidence, and the detected line rows, all in the
    coordinates of the image passed in.
    """
    scale = _work_scale(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else gray
    ink = _ink_mask(small)
    profile = _row_profile(small, ink)
    period, confidence = _dominant_period(profile)
    cycles = (small.shape[0] / period) if period else 0.0
    rows: list[int] = []
    if period and confidence >= MIN_PERIODICITY and cycles >= MIN_CYCLES:
        rows = _line_rows(profile, period)
    return {
        "scale": scale,
        "period_px": period / scale if period else 0,
        "confidence": round(confidence, 3),
        "cycles": round(cycles, 1),
        "n_lines": len(rows),
        "_small": small,
        "_ink": ink,
        "_rows": rows,
        "_period": period,
    }


def remove(bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Lift the ruled lines out of a colour image.

    Returns (image, info). When no periodic ruling is found the input is
    returned untouched — a page without ruling must not be altered.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    info = detect(gray)
    rows, period, scale = info["_rows"], info["_period"], info["scale"]
    if not rows:
        return bgr, {k: v for k, v in info.items() if not k.startswith("_")}

    small, ink = info["_small"], info["_ink"]
    band = max(2, int(period * BAND_FRAC))

    # Detection is cheap and robust at reduced scale, but the correction is not:
    # the groove is only a few pixels tall, and computing a 3-pixel lift on the
    # small image and interpolating it back up smears it out and halves its
    # amplitude. So the geometry found here is mapped up, and the actual pixel
    # work happens at full resolution. The background estimate is the exception
    # — it is smooth by construction, so upsampling it costs nothing and saves a
    # large median filter over the full-size image.
    background_small = cv2.medianBlur(small, 2 * band + 1)
    if scale < 1.0:
        H, W = gray.shape
        background = cv2.resize(background_small, (W, H), interpolation=cv2.INTER_LINEAR)
        work = gray
        ink_full = _ink_mask(gray)
        inv = 1.0 / scale
        rows = [int(round(r * inv)) for r in rows]
        band = max(2, int(round(band * inv)))
        groove = max(1, int(round(GROOVE_PX * inv)))
    else:
        background = background_small
        work = small
        ink_full = ink
        groove = GROOVE_PX

    h, w = work.shape

    # Restore the local parchment tone across the whole band around each ruled
    # line, rather than lifting a fixed number of rows at the tracked centre.
    # The groove is not a constant width — it broadens where the stylus pressed
    # harder — and anything already at background level is left alone, so a
    # wider band costs nothing where there is nothing to correct.
    lift = np.zeros((h, w), dtype=np.float32)
    work_f = work.astype(np.float32)
    bg_f = background.astype(np.float32)
    xs = np.arange(w)
    period_full = period / scale if scale < 1.0 else period

    # How dark may a dip be before we call it ink? Measured against this image's
    # own ink-to-parchment range, so it holds for a pale scan and a dark photo
    # alike rather than depending on an absolute grey level.
    ink_level = float(np.median(work_f[ink_full])) if ink_full.any() else 0.0
    max_dip = MAX_DEPTH_FRAC * max(1.0, float(np.median(bg_f)) - ink_level)

    tracks: dict[int, np.ndarray] = {}
    rejected = 0
    for row in rows:
        raw_track = _follow(work, ink_full, row, band)
        # A ruled line bows gently, so fit a quadratic and use the fit rather
        # than the raw track: wherever the groove passes under a letter the
        # tracker has nothing to lock onto and picks an arbitrary row, and the
        # fit carries the line straight through those gaps.
        fit = np.polyval(np.polyfit(xs, raw_track, 2), xs)
        residual = np.abs(raw_track - fit)
        # Median, not standard deviation: the columns hidden by letters are
        # outliers by construction and would otherwise dominate the measure.
        wobble = float(np.median(residual)) / max(1.0, period_full)
        if wobble > MAX_TRACK_WOBBLE:
            rejected += 1
            continue
        track = np.clip(np.round(fit), 0, h - 1).astype(np.int32)
        tracks[row] = track
        for dy in range(-groove, groove + 1):
            ys = np.clip(track + dy, 0, h - 1)
            delta = np.clip(bg_f[ys, xs] - work_f[ys, xs], 0, None)
            delta[delta > max_dip] = 0.0   # too dark to be a groove: leave it
            np.maximum.at(lift, (ys, xs), delta * LIFT)

    if not tracks:
        public = {k: v for k, v in info.items() if not k.startswith("_")}
        public.update(n_lines=0, rejected=rejected, lifted_mean=0.0)
        return bgr, public

    lift[ink_full] = 0.0   # never touch the writing

    out = bgr.astype(np.float32) + lift[:, :, None]
    public = {k: v for k, v in info.items() if not k.startswith("_")}
    public["lifted_mean"] = round(float(lift[lift > 0].mean()) if (lift > 0).any() else 0.0, 2)
    # Full-resolution geometry, so a caller can verify exactly what was changed
    # rather than re-deriving it and measuring something slightly different.
    public["n_lines"] = len(tracks)
    public["rejected"] = rejected
    public["_tracks"] = tracks
    public["_ink_full"] = ink_full
    public["_period_full"] = int(round(period / scale)) if scale < 1.0 else period
    return np.clip(out, 0, 255).astype(np.uint8), public


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    import time
    from pathlib import Path

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stam_io import imread_any, list_images

    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--images-dir")
    ap.add_argument("--out-dir", default="test_output/deruled")
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()

    paths = [args.image] if args.image else list_images(args.images_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for p in paths:
        img = imread_any(p)
        if img is None:
            print(f"  unreadable: {p}")
            continue
        t0 = time.perf_counter()
        if args.detect_only:
            info = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            info = {k: v for k, v in info.items() if not k.startswith("_")}
        else:
            fixed, info = remove(img)
            cv2.imwrite(str(out / f"{Path(p).stem}_deruled.jpg"), fixed,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {Path(p).name[:44]:<46} {ms:6.0f}ms  "
              f"conf={info['confidence']:.2f}  period={info['period_px']:.0f}px  "
              f"lines={info['n_lines']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
