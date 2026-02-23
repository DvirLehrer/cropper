#!/usr/bin/env python3
"""Visualize periodic stripe pattern rows on preprocessed images."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Optional

from PIL import Image, ImageChops, ImageDraw, ImageFilter

MASK_THR = 250
FLAT_RANGE_MAX = 8
MAX_WORK_WIDTH = 1200
ROW_BAND_MARGIN_FRAC = 0.03
ROW_BAND_MARGIN_MIN_PX = 8
DEBUG_WHITE_BAND_HALF_PX = 5
DEBUG_FILL_TILE_SIZE_PX = 56
DEBUG_FILL_SAMPLE_STRIDE_PX = 4
ABORT_MIN_ENERGY_STRENGTH = 0.85
ABORT_MIN_ROW_EDGE_STRENGTH = 0.40
ABORT_MIN_MEAN_COVERAGE_STRENGTH = 0.40
ENABLE_PERIODIC_PATTERN_DEBUG = True


def _mask_pixels(gray: Image.Image) -> list[bool]:
    # Fast C-level approximation of "bright + locally flat" masked regions.
    bright = gray.point(lambda p: 255 if p >= MASK_THR else 0)
    local_min = gray.filter(ImageFilter.MinFilter(size=3))
    local_max = gray.filter(ImageFilter.MaxFilter(size=3))
    local_range = ImageChops.subtract(local_max, local_min)
    flat = local_range.point(lambda p: 255 if p <= FLAT_RANGE_MAX else 0)
    combined = ImageChops.multiply(bright, flat)
    return [v > 0 for v in combined.getdata()]


def _masked_row_band(mask: list[bool], w: int, h: int) -> Optional[tuple[int, int]]:
    min_row_masked = max(8, int(round(0.01 * w)))
    rows: list[tuple[int, int]] = []
    for y in range(h):
        s = y * w
        cnt = sum(1 for x in range(w) if mask[s + x])
        if cnt >= min_row_masked:
            rows.append((y, cnt))
    if not rows:
        return None
    clusters: list[tuple[int, int, int]] = []
    y0 = rows[0][0]
    yp = rows[0][0]
    mass = rows[0][1]
    for y, cnt in rows[1:]:
        if y <= yp + 1:
            yp = y
            mass += cnt
            continue
        clusters.append((y0, yp, mass))
        y0 = y
        yp = y
        mass = cnt
    clusters.append((y0, yp, mass))
    best = max(clusters, key=lambda c: c[2])
    return best[0], best[1]


def _sobel_abs(gray: Image.Image, kernel: tuple[int, ...]) -> Image.Image:
    signed = gray.filter(ImageFilter.Kernel((3, 3), kernel, scale=1, offset=128))
    return signed.point(lambda p: abs(p - 128))


def _line_energy(gray: Image.Image) -> list[int]:
    dy = _sobel_abs(gray, (-1, -2, -1, 0, 0, 0, 1, 2, 1))
    dx = _sobel_abs(gray, (-1, 0, 1, -2, 0, 2, -1, 0, 1))
    pdy = list(dy.getdata())
    pdx = list(dx.getdata())
    out: list[int] = []
    for a, b in zip(pdy, pdx):
        v = int(round(a - 0.45 * b))
        out.append(0 if v < 0 else v)
    return out


def _percentile(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    i = int(round(max(0.0, min(1.0, q)) * (len(s) - 1)))
    return s[i]


def _row_signal(binary: list[int], valid: list[bool], w: int, h: int) -> list[float]:
    sig = [0.0] * h
    for y in range(h):
        s = y * w
        num = 0
        den = 0
        for x in range(w):
            i = s + x
            if not valid[i]:
                continue
            den += 1
            if binary[i]:
                num += 1
        sig[y] = (num / den) if den else 0.0
    return sig


def _dark_ridge_row_signal(gray: Image.Image, valid: list[bool], w: int, h: int) -> list[float]:
    px = list(gray.getdata())
    sig = [0.0] * h
    if w < 3 or h < 3:
        return sig

    for y in range(1, h - 1):
        row = y * w
        acc = 0.0
        den = 0
        for x in range(1, w - 1):
            i = row + x
            if not valid[i]:
                continue
            iu = i - w
            idn = i + w
            if not valid[iu] or not valid[idn]:
                continue
            top = float(px[iu])
            mid = float(px[i])
            bot = float(px[idn])
            ridge = 0.5 * (top + bot) - mid
            if ridge <= 0.0:
                continue
            lr = abs(float(px[i + 1]) - float(px[i - 1]))
            v = ridge - 0.35 * lr
            if v <= 0.0:
                continue
            acc += v
            den += 1
        sig[y] = (acc / float(den)) if den else 0.0

    vals = [v for v in sig if v > 0.0]
    if not vals:
        return sig
    p95 = sorted(vals)[int(round(0.95 * (len(vals) - 1)))]
    scale = p95 if p95 > 1e-6 else max(vals)
    if scale <= 1e-6:
        return sig
    out = [min(1.0, v / scale) for v in sig]

    sm = [0.0] * h
    for y in range(h):
        y0 = max(0, y - 1)
        y1 = min(h - 1, y + 1)
        sm[y] = sum(out[y0 : y1 + 1]) / float(y1 - y0 + 1)
    return sm


def _row_signal_xrange(
    binary: list[int],
    valid: list[bool],
    w: int,
    h: int,
    x0: int,
    x1: int,
) -> list[float]:
    xa = max(0, min(w - 1, x0))
    xb = max(0, min(w - 1, x1))
    if xb < xa:
        xa, xb = xb, xa
    sig = [0.0] * h
    for y in range(h):
        s = y * w
        num = 0
        den = 0
        for x in range(xa, xb + 1):
            i = s + x
            if not valid[i]:
                continue
            den += 1
            if binary[i]:
                num += 1
        sig[y] = (num / den) if den else 0.0
    return sig


def _binary_from_local_thresholds(
    energy: list[int],
    valid: list[bool],
    w: int,
    h: int,
    *,
    q: float = 0.88,
    block_w: int = 96,
) -> list[int]:
    block_w = max(24, block_w)
    blocks = list(range(0, w, block_w))
    thrs: list[int] = []
    for x0 in blocks:
        x1 = min(w, x0 + block_w)
        vals: list[int] = []
        for y in range(h):
            row = y * w
            for x in range(x0, x1):
                i = row + x
                if valid[i]:
                    vals.append(energy[i])
        thrs.append(_percentile(vals, q) if vals else 255)

    out = [0] * (w * h)
    for bi, x0 in enumerate(blocks):
        x1 = min(w, x0 + block_w)
        thr = thrs[bi]
        for y in range(h):
            row = y * w
            for x in range(x0, x1):
                i = row + x
                if valid[i] and energy[i] >= thr:
                    out[i] = 1
    return out


def _autocorr_best(
    sig: list[float],
    y_lo: int,
    y_hi: int,
    *,
    min_lag: Optional[int] = None,
    max_lag: Optional[int] = None,
) -> tuple[int, float]:
    seg = sig[y_lo : y_hi + 1]
    n = len(seg)
    if n < 24:
        return 0, 0.0
    lag_lo = max(6, n // 40)
    lag_hi = min(max(lag_lo + 3, n // 3), 64)
    if min_lag is not None and min_lag > 0:
        lag_lo = max(lag_lo, min_lag)
    if max_lag is not None and max_lag > 0:
        lag_hi = min(lag_hi, max_lag)
    if lag_hi < lag_lo:
        return 0, 0.0
    best_lag = 0
    best_corr = 0.0
    for lag in range(lag_lo, lag_hi + 1):
        sxy = sx2 = sy2 = 0.0
        for i in range(0, n - lag):
            a = seg[i]
            b = seg[i + lag]
            sxy += a * b
            sx2 += a * a
            sy2 += b * b
        den = (sx2 * sy2) ** 0.5
        if den <= 1e-9:
            continue
        c = sxy / den
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return best_lag, best_corr


def _autocorr_at_lag(sig: list[float], y_lo: int, y_hi: int, lag: int) -> float:
    if lag <= 0:
        return 0.0
    seg = sig[y_lo : y_hi + 1]
    n = len(seg)
    if lag >= n:
        return 0.0
    sxy = sx2 = sy2 = 0.0
    for i in range(0, n - lag):
        a = seg[i]
        b = seg[i + lag]
        sxy += a * b
        sx2 += a * a
        sy2 += b * b
    den = (sx2 * sy2) ** 0.5
    if den <= 1e-9:
        return 0.0
    return sxy / den


def _periodic_peaks(sig: list[float], y_lo: int, y_hi: int, lag: int) -> tuple[list[int], float]:
    if lag <= 0:
        return [], 0.0
    seg = sig[y_lo : y_hi + 1]
    if len(seg) < 3:
        return [], 0.0
    med = sorted(seg)[len(seg) // 2]
    gate = med + 0.015
    peaks: list[int] = []
    min_sep = max(3, int(round(0.6 * lag)))
    for i in range(1, len(seg) - 1):
        if seg[i] < gate:
            continue
        if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]:
            y = y_lo + i
            if not peaks or (y - peaks[-1]) >= min_sep:
                peaks.append(y)
            elif seg[y - y_lo] > sig[peaks[-1]]:
                peaks[-1] = y

    if peaks:
        # Fill missing rows along the detected periodic spacing.
        low_gate = med + 0.006
        tol = max(2, int(round(0.28 * lag)))

        def _near_peak(y_center: int) -> int | None:
            lo = max(y_lo + 1, y_center - tol)
            hi = min(y_hi - 1, y_center + tol)
            best_y = -1
            best_v = -1.0
            for y in range(lo, hi + 1):
                i = y - y_lo
                v = seg[i]
                if v < low_gate:
                    continue
                if v >= seg[i - 1] and v > seg[i + 1] and v > best_v:
                    best_v = v
                    best_y = y
            return best_y if best_y >= 0 else None

        filled = set(peaks)
        ordered = sorted(peaks)

        # Fill large gaps between neighboring peaks.
        for a, b in zip(ordered, ordered[1:]):
            y = a + lag
            while y < b - tol:
                cand = _near_peak(int(round(y)))
                if cand is not None:
                    filled.add(cand)
                y += lag

        # Extend up/down a bit when periodic support exists.
        y = ordered[0] - lag
        while y >= y_lo + tol:
            cand = _near_peak(int(round(y)))
            if cand is None:
                break
            filled.add(cand)
            y -= lag
        y = ordered[-1] + lag
        while y <= y_hi - tol:
            cand = _near_peak(int(round(y)))
            if cand is None:
                break
            filled.add(cand)
            y += lag

        peaks = sorted(filled)

    if len(peaks) < 2:
        return peaks, 0.0
    gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std = var**0.5
    return peaks, max(0.0, 1.0 - (std / max(1.0, mean_gap)))


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        if not points:
            return 0.0, 0.0
        return 0.0, points[0][1]
    n = float(len(points))
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if abs(den) <= 1e-9:
        return 0.0, sy / n
    m = (n * sxy - sx * sy) / den
    b = (sy - m * sx) / n
    return m, b


def _two_segment_polyline_for_peak(
    binary: list[int],
    valid: list[bool],
    w: int,
    h: int,
    y0: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    search = 8
    mid = w // 2
    left_pts: list[tuple[float, float]] = []
    right_pts: list[tuple[float, float]] = []

    for x in range(w):
        best_y = -1
        best_score = -1
        for yy in range(max(0, y0 - search), min(h - 1, y0 + search) + 1):
            i = yy * w + x
            if not valid[i] or not binary[i]:
                continue
            # Prefer points closer to peak row.
            score = 100 - abs(yy - y0)
            if score > best_score:
                best_score = score
                best_y = yy
        if best_y < 0:
            continue
        if x <= mid:
            left_pts.append((float(x), float(best_y)))
        else:
            right_pts.append((float(x), float(best_y)))

    m1, b1 = _fit_line(left_pts)
    m2, b2 = _fit_line(right_pts)

    x_left = 0
    y_left = int(round(m1 * x_left + b1)) if left_pts else y0
    x_mid = mid
    y_mid_l = int(round(m1 * x_mid + b1)) if left_pts else y0
    y_mid_r = int(round(m2 * x_mid + b2)) if right_pts else y0
    y_mid = int(round((y_mid_l + y_mid_r) / 2.0))
    x_right = w - 1
    y_right = int(round(m2 * x_right + b2)) if right_pts else y0

    y_left = max(0, min(h - 1, y_left))
    y_mid = max(0, min(h - 1, y_mid))
    y_right = max(0, min(h - 1, y_right))
    return (x_left, y_left), (x_mid, y_mid), (x_right, y_right)


def _stripe_intervals_for_peak(
    binary: list[int],
    valid: list[bool],
    w: int,
    h: int,
    y0: int,
    lag: int,
) -> list[tuple[int, int]]:
    support = [0.0] * w
    for x in range(w):
        on = 0.0
        on_n = 0.0
        for yy in (y0 - 1, y0, y0 + 1):
            if yy < 0 or yy >= h:
                continue
            i = yy * w + x
            if not valid[i]:
                continue
            on_n += 1.0
            if binary[i]:
                on += 1.0
        on_score = (on / on_n) if on_n > 0 else 0.0

        nbr = 0.0
        nbr_n = 0.0
        if lag > 0:
            for yb in (y0 - lag, y0 + lag):
                if yb < 0 or yb >= h:
                    continue
                for yy in (yb - 1, yb, yb + 1):
                    if yy < 0 or yy >= h:
                        continue
                    i = yy * w + x
                    if not valid[i]:
                        continue
                    nbr_n += 1.0
                    if binary[i]:
                        nbr += 1.0
        nbr_score = (nbr / nbr_n) if nbr_n > 0 else 0.0
        support[x] = 0.62 * on_score + 0.38 * nbr_score

    # Smooth support
    sm = [0.0] * w
    for x in range(w):
        lo = max(0, x - 4)
        hi = min(w - 1, x + 4)
        sm[x] = sum(support[lo : hi + 1]) / float(hi - lo + 1)

    local_radius = max(28, int(round(0.07 * w)))
    q1 = w // 4
    q2 = w // 2
    q3 = (3 * w) // 4
    quarter_bounds = (
        (0, max(0, q1 - 1)),
        (q1, max(q1, q2 - 1)),
        (q2, max(q2, q3 - 1)),
        (q3, w - 1),
    )
    gate_x = [0.0] * w
    for x in range(w):
        qi = min(3, max(0, (4 * x) // max(1, w)))
        x_min, x_max = quarter_bounds[qi]
        lo = max(x_min, x - local_radius)
        hi = min(x_max, x + local_radius)
        seg = sm[lo : hi + 1]
        mean_v = sum(seg) / float(max(1, len(seg)))
        max_v = max(seg) if seg else 0.0
        gate_x[x] = max(0.06, mean_v + 0.10 * (max_v - mean_v))
    min_len = max(14, int(round(0.02 * w)))
    max_gap = max(6, int(round(0.015 * w)))

    intervals: list[tuple[int, int]] = []
    run = -1
    gap = 0
    last_good = -1
    for x, v in enumerate(sm):
        if v >= gate_x[x]:
            if run < 0:
                run = x
            last_good = x
            gap = 0
            continue
        if run < 0:
            continue
        gap += 1
        if gap > max_gap:
            if last_good - run + 1 >= min_len:
                intervals.append((run, last_good))
            run = -1
            gap = 0
            last_good = -1

    if run >= 0 and last_good - run + 1 >= min_len:
        intervals.append((run, last_good))
    return intervals


def _two_segment_from_intervals(
    binary: list[int],
    valid: list[bool],
    w: int,
    h: int,
    y0: int,
    intervals: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    search = 4
    step_x = 2
    mid = w // 2
    left_pts: list[tuple[float, float]] = []
    right_pts: list[tuple[float, float]] = []

    for x0, x1 in intervals:
        for x in range(max(0, x0), min(w - 1, x1) + 1, step_x):
            best_y = -1
            best_score = -1
            for yy in range(max(0, y0 - search), min(h - 1, y0 + search) + 1):
                i = yy * w + x
                if not valid[i] or not binary[i]:
                    continue
                score = 100 - abs(yy - y0)
                if score > best_score:
                    best_score = score
                    best_y = yy
            if best_y < 0:
                continue
            if x <= mid:
                left_pts.append((float(x), float(best_y)))
            else:
                right_pts.append((float(x), float(best_y)))

    if len(left_pts) < 2 and len(right_pts) < 2:
        return _two_segment_polyline_for_peak(binary, valid, w, h, y0)

    m1, b1 = _fit_line(left_pts if left_pts else right_pts)
    m2, b2 = _fit_line(right_pts if right_pts else left_pts)

    x_left = 0
    x_mid = mid
    x_right = w - 1
    y_left = int(round(m1 * x_left + b1))
    y_mid = int(round((m1 * x_mid + b1 + m2 * x_mid + b2) / 2.0))
    y_right = int(round(m2 * x_right + b2))

    y_left = max(0, min(h - 1, y_left))
    y_mid = max(0, min(h - 1, y_mid))
    y_right = max(0, min(h - 1, y_right))
    return (x_left, y_left), (x_mid, y_mid), (x_right, y_right)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[m])
    return 0.5 * float(s[m - 1] + s[m])


def _work_image(gray_full: Image.Image) -> tuple[Image.Image, float]:
    w, h = gray_full.size
    if w <= MAX_WORK_WIDTH:
        return gray_full, 1.0
    scale = MAX_WORK_WIDTH / float(max(1, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return gray_full.resize((new_w, new_h), Image.BILINEAR), scale


def _row_slope(p0: tuple[int, int], p1: tuple[int, int]) -> float:
    dx = float(max(1, p1[0] - p0[0]))
    return float((p1[1] - p0[1]) / dx)


def _clamp_row_to_slope_ballpark(
    p0: tuple[int, int],
    pm: tuple[int, int],
    p1: tuple[int, int],
    *,
    target_slope: float,
    tol: float,
    h: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    lo = target_slope - tol
    hi = target_slope + tol

    x0, y0 = p0
    xm, ym = pm
    x1, y1 = p1

    # Clamp left and right halves independently so one side does not pull the other off-pattern.
    left_dx = float(max(1, xm - x0))
    right_dx = float(max(1, x1 - xm))
    left_slope = float((ym - y0) / left_dx)
    right_slope = float((y1 - ym) / right_dx)

    if left_slope < lo:
        left_slope = lo
    elif left_slope > hi:
        left_slope = hi
    if right_slope < lo:
        right_slope = lo
    elif right_slope > hi:
        right_slope = hi

    y0n = int(round(ym - left_slope * left_dx))
    y1n = int(round(ym + right_slope * right_dx))
    y0n = max(0, min(h - 1, y0n))
    ymn = max(0, min(h - 1, ym))
    y1n = max(0, min(h - 1, y1n))
    return (x0, y0n), (xm, ymn), (x1, y1n)


def _save_vertical_lighten_debug(
    image: Image.Image,
    mask: list[bool],
    lines_full: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
    out_path: Path,
) -> None:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if not lines_full:
        rgb.save(out_path)
        return

    px = list(rgb.getdata())
    white_zone = [False] * (w * h)
    for p0, pm, p1 in lines_full:
        x0, y0 = p0
        xm, ym = pm
        x1, y1 = p1
        x0 = max(0, min(w - 1, x0))
        xm = max(0, min(w - 1, xm))
        x1 = max(0, min(w - 1, x1))

        def _paint_segment(xa: int, ya: int, xb: int, yb: int) -> None:
            if xb < xa:
                xa, ya, xb, yb = xb, yb, xa, ya
            xa = max(0, min(w - 1, xa))
            xb = max(0, min(w - 1, xb))
            if xb < xa:
                return
            dx = max(1, xb - xa)
            for x in range(xa, xb + 1):
                t = (x - xa) / float(dx)
                yc = int(round(ya + (yb - ya) * t))
                y_lo = max(0, yc - DEBUG_WHITE_BAND_HALF_PX)
                y_hi = min(h - 1, yc + DEBUG_WHITE_BAND_HALF_PX)
                for y in range(y_lo, y_hi + 1):
                    i = y * w + x
                    if mask[i]:
                        continue
                    white_zone[i] = True

        _paint_segment(x0, y0, xm, ym)
        _paint_segment(xm, ym, x1, y1)

    if False:  # Debug: set to True to visualize selected white-zone pixels.
        dbg = Image.new("RGB", (w, h), (0, 0, 0))
        dbg.putdata([(255, 255, 255) if white_zone[i] else (0, 0, 0) for i in range(w * h)])
        dbg.save(out_path.with_name(f"{out_path.stem}_white_zone_debug.png"))

    # Build coarse cached surrounding colors (non-masked, non-target) using sparse pseudo-random samples.
    tile = max(8, DEBUG_FILL_TILE_SIZE_PX)
    stride = max(1, DEBUG_FILL_SAMPLE_STRIDE_PX)
    gw = (w + tile - 1) // tile
    gh = (h + tile - 1) // tile
    tile_mean: list[tuple[int, int, int] | None] = [None] * (gw * gh)
    gsr = gsg = gsb = gsc = 0
    for gy in range(gh):
        y0 = gy * tile
        y1 = min(h, y0 + tile)
        for gx in range(gw):
            x0 = gx * tile
            x1 = min(w, x0 + tile)
            hsh = ((gx * 73856093) ^ (gy * 19349663)) & 0xFFFFFFFF
            off_x = hsh % stride
            off_y = (hsh >> 5) % stride
            sr = sg = sb = cnt = 0
            for y in range(y0 + off_y, y1, stride):
                row = y * w
                for x in range(x0 + off_x, x1, stride):
                    i = row + x
                    if mask[i] or white_zone[i]:
                        continue
                    r, g, b = px[i]
                    sr += r
                    sg += g
                    sb += b
                    cnt += 1
            if cnt > 0:
                mr = int(round(sr / float(cnt)))
                mg = int(round(sg / float(cnt)))
                mb = int(round(sb / float(cnt)))
                tile_mean[gy * gw + gx] = (mr, mg, mb)
                gsr += sr
                gsg += sg
                gsb += sb
                gsc += cnt
    if gsc > 0:
        global_mean = (
            int(round(gsr / float(gsc))),
            int(round(gsg / float(gsc))),
            int(round(gsb / float(gsc))),
        )
    else:
        global_mean = (255, 255, 255)

    # Fill selected stripe pixels from cached surrounding means, while excluding masked pixels.
    out_px: list[tuple[int, int, int]] = []
    for i, (r, g, b) in enumerate(px):
        if mask[i]:
            out_px.append((r, g, b))
            continue
        if not white_zone[i]:
            out_px.append((r, g, b))
            continue
        y = i // w
        x = i - y * w
        gx = x // tile
        gy = y // tile
        m = tile_mean[gy * gw + gx]
        out_px.append(m if m is not None else global_mean)
    rgb.putdata(out_px)
    rgb.save(out_path)


def draw_periodic(
    path: Path,
    *,
    light_debug_image: Optional[Image.Image] = None,
    light_debug_mask_image: Optional[Image.Image] = None,
    light_debug_out_path: Optional[Path] = None,
    min_lag_full_px: Optional[int] = None,
    max_lag_full_px: Optional[int] = None,
) -> dict[str, float | int | str]:
    out_path = path.with_name(f"{path.stem}_periodic_pattern.png")
    return _draw_periodic_from_gray_full(
        gray_full=Image.open(path).convert("L"),
        input_name=path.name,
        out_path=out_path,
        light_debug_image=light_debug_image,
        light_debug_mask_image=light_debug_mask_image,
        light_debug_out_path=light_debug_out_path,
        min_lag_full_px=min_lag_full_px,
        max_lag_full_px=max_lag_full_px,
    )


def _draw_periodic_from_gray_full(
    *,
    gray_full: Image.Image,
    input_name: str,
    out_path: Optional[Path],
    light_debug_image: Optional[Image.Image],
    light_debug_mask_image: Optional[Image.Image],
    light_debug_out_path: Optional[Path],
    min_lag_full_px: Optional[int],
    max_lag_full_px: Optional[int],
) -> dict[str, float | int | str]:
    t0 = time.perf_counter()
    w_full, h_full = gray_full.size
    if light_debug_mask_image is not None:
        mask_src = light_debug_mask_image.convert("L")
        if mask_src.size != (w_full, h_full):
            mask_src = mask_src.resize((w_full, h_full), Image.NEAREST)
        mask_full = [v >= 128 for v in mask_src.getdata()]
    else:
        mask_full = _mask_pixels(gray_full)

    gray, scale = _work_image(gray_full)
    w, h = gray.size
    inv_scale = 1.0 / scale
    min_lag_work: Optional[int] = None
    if min_lag_full_px is not None and min_lag_full_px > 0:
        min_lag_work = max(6, int(round(min_lag_full_px * scale)))
    max_lag_work: Optional[int] = None
    if max_lag_full_px is not None and max_lag_full_px > 0:
        max_lag_work = max(6, int(round(max_lag_full_px * scale)))

    valid = [True] * (w * h)
    # Do not constrain periodic search by mask band; search full vertical range.
    y_lo, y_hi = 0, h - 1

    energy = _line_energy(gray)
    vals = [v for i, v in enumerate(energy) if valid[i]]
    if vals:
        e_max = max(vals)
        e_p90 = _percentile(vals, 0.90)
        energy_strength = (float(e_p90) / float(max(1, e_max))) if e_max > 0 else 0.0
    else:
        energy_strength = 0.0
    if energy_strength < ABORT_MIN_ENERGY_STRENGTH:
        out = gray_full.convert("RGB")
        if out_path is not None:
            out.save(out_path)
        light_out_name = ""
        t_sec = time.perf_counter() - t0
        meta = {
            "input": input_name,
            "output": out_path.name if out_path is not None else "",
            "lag": 0,
            "corr": 0.0,
            "peaks": 0,
            "spacing_cons": 0.0,
            "strength": 0.0,
            "periodic_time_sec": t_sec,
            "scale": scale,
            "light_debug_output": light_out_name,
            "energy_strength": energy_strength,
            "binary_strength": 0.0,
            "row_edge_strength": 0.0,
            "row_dark_strength": 0.0,
            "row_combined_strength": 0.0,
            "mean_coverage_strength": 0.0,
            "aborted": 1,
            "abort_stage": "energy_strength",
        }
        return meta
    binary = _binary_from_local_thresholds(energy, valid, w, h, q=0.88, block_w=max(72, w // 14))
    valid_count = sum(1 for v in valid if v)
    binary_on_count = sum(binary[i] for i in range(w * h) if valid[i])
    binary_strength = (float(binary_on_count) / float(max(1, valid_count))) if valid_count > 0 else 0.0
    sig_edge = _row_signal(binary, valid, w, h)
    sig_dark = _dark_ridge_row_signal(gray, valid, w, h)
    sig = [0.30 * a + 0.70 * b for a, b in zip(sig_edge, sig_dark)]
    row_edge_strength = max(sig_edge) if sig_edge else 0.0
    row_dark_strength = max(sig_dark) if sig_dark else 0.0
    row_combined_strength = max(sig) if sig else 0.0
    if row_edge_strength < ABORT_MIN_ROW_EDGE_STRENGTH:
        out = gray_full.convert("RGB")
        if out_path is not None:
            out.save(out_path)
        light_out_name = ""
        t_sec = time.perf_counter() - t0
        meta = {
            "input": input_name,
            "output": out_path.name if out_path is not None else "",
            "lag": 0,
            "corr": 0.0,
            "peaks": 0,
            "spacing_cons": 0.0,
            "strength": 0.0,
            "periodic_time_sec": t_sec,
            "scale": scale,
            "light_debug_output": light_out_name,
            "energy_strength": energy_strength,
            "binary_strength": binary_strength,
            "row_edge_strength": row_edge_strength,
            "row_dark_strength": row_dark_strength,
            "row_combined_strength": row_combined_strength,
            "mean_coverage_strength": 0.0,
            "aborted": 1,
            "abort_stage": "row_edge_strength",
        }
        return meta
    quarter_lags: list[int] = []
    quarter_corrs: list[float] = []
    y_mid = (y_lo + y_hi) // 2
    quad_ranges = [
        (0, max(0, (w // 2) - 1), y_lo, y_mid),  # top-left
        (w // 2, w - 1, y_lo, y_mid),  # top-right
        (0, max(0, (w // 2) - 1), min(h - 1, y_mid + 1), y_hi),  # bottom-left
        (w // 2, w - 1, min(h - 1, y_mid + 1), y_hi),  # bottom-right
    ]
    for x0, x1, y0q, y1q in quad_ranges:
        if y1q < y0q:
            quarter_lags.append(0)
            quarter_corrs.append(0.0)
            continue
        sig_q = _row_signal_xrange(binary, valid, w, h, x0, x1)
        lag_q, corr_q = _autocorr_best(
            sig_q, y0q, y1q, min_lag=min_lag_work, max_lag=max_lag_work
        )
        eff_q = lag_q
        if lag_q >= 20:
            half_q = max(6, lag_q // 2)
            corr_half_q = _autocorr_at_lag(sig_q, y0q, y1q, half_q)
            if corr_half_q >= (0.88 * corr_q):
                eff_q = half_q
        quarter_lags.append(eff_q)
        quarter_corrs.append(corr_q)

    valid_q = [(l, c) for l, c in zip(quarter_lags, quarter_corrs) if l > 0]
    if valid_q:
        wsum = sum(max(0.0, c) for _, c in valid_q)
        if wsum > 1e-6:
            eff_lag = int(round(sum(l * max(0.0, c) for l, c in valid_q) / wsum))
        else:
            eff_lag = int(round(sum(l for l, _ in valid_q) / float(len(valid_q))))
        lag = eff_lag
        corr = sum(c for _, c in valid_q) / float(len(valid_q))
    else:
        lag, corr = _autocorr_best(sig, y_lo, y_hi, min_lag=min_lag_work, max_lag=max_lag_work)
        eff_lag = lag
        if lag >= 20:
            half = max(6, lag // 2)
            corr_half = _autocorr_at_lag(sig, y_lo, y_hi, half)
            if corr_half >= (0.88 * corr):
                eff_lag = half
    peaks, spacing_cons = _periodic_peaks(sig, y_lo, y_hi, eff_lag)

    # Build/cache per-row models to avoid recomputing expensive interval scans.
    row_models: dict[int, tuple[tuple[int, int], tuple[int, int], tuple[int, int], float]] = {}

    def _build_row_model(y: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], float]:
        cached = row_models.get(y)
        if cached is not None:
            return cached
        iv = _stripe_intervals_for_peak(binary, valid, w, h, y, eff_lag)
        p0, pm, p1 = _two_segment_from_intervals(binary, valid, w, h, y, iv)
        cov = (sum((b - a + 1) for a, b in iv) / float(max(1, w))) if iv else 0.0
        model = (p0, pm, p1, cov)
        row_models[y] = model
        return model

    # Use only rows directly produced by autocorrelation peak detection.
    dominant_slope = 0.0
    if eff_lag > 0 and peaks:
        for y in peaks:
            _build_row_model(y)

        angles = [
            float((m[2][1] - m[0][1]) / float(max(1, (m[2][0] - m[0][0]))))
            for m in row_models.values()
        ]
        dominant_slope = _median(angles)

    lines_full: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    covs: list[float] = []
    for y in peaks:
        p0, pm, p1, cov = _build_row_model(y)
        covs.append(cov)
        p0f = (int(round(p0[0] * inv_scale)), int(round(p0[1] * inv_scale)))
        pmf = (int(round(pm[0] * inv_scale)), int(round(pm[1] * inv_scale)))
        p1f = (int(round(p1[0] * inv_scale)), int(round(p1[1] * inv_scale)))
        lines_full.append((p0f, pmf, p1f))
    mean_cov = (sum(covs) / float(len(covs))) if covs else 0.0
    if mean_cov < ABORT_MIN_MEAN_COVERAGE_STRENGTH:
        out = gray_full.convert("RGB")
        if out_path is not None:
            out.save(out_path)
        light_out_name = ""
        t_sec = time.perf_counter() - t0
        strength = 0.0
        meta = {
            "input": input_name,
            "output": out_path.name if out_path is not None else "",
            "lag": int(round(eff_lag * inv_scale)),
            "corr": corr,
            "peaks": len(peaks),
            "spacing_cons": spacing_cons,
            "strength": strength,
            "periodic_time_sec": t_sec,
            "scale": scale,
            "light_debug_output": light_out_name,
            "energy_strength": energy_strength,
            "binary_strength": binary_strength,
            "row_edge_strength": row_edge_strength,
            "row_dark_strength": row_dark_strength,
            "row_combined_strength": row_combined_strength,
            "mean_coverage_strength": mean_cov,
            "aborted": 1,
            "abort_stage": "mean_coverage_strength",
        }
        return meta

    out = gray_full.convert("RGB")
    d = ImageDraw.Draw(out)
    for p0f, pmf, p1f in lines_full:
        # Extra layer: full-span stripe from end to end.
        d.line([p0f, p1f], fill=(255, 0, 180), width=1)
        d.line([p0f, pmf], fill=(0, 160, 255), width=2)
        d.line([pmf, p1f], fill=(0, 200, 90), width=2)
    lag_full = int(round(eff_lag * inv_scale))
    d.text(
        (8, 8),
        (
            f"lag={lag}->{eff_lag} (~{lag_full}px full) "
            f"q=[{quarter_lags[0]},{quarter_lags[1]},{quarter_lags[2]},{quarter_lags[3]}] "
            f"corr={corr:.3f} peaks={len(peaks)} spacing_cons={spacing_cons:.3f}"
        ),
        fill=(20, 20, 20),
    )

    if out_path is not None:
        out.save(out_path)
    light_out_name = ""
    if light_debug_image is not None and light_debug_out_path is not None:
        debug_src = light_debug_image
        if debug_src.size != (w_full, h_full):
            debug_src = debug_src.resize((w_full, h_full), Image.BILINEAR)
        _save_vertical_lighten_debug(debug_src, mask_full, lines_full, light_debug_out_path)
        light_out_name = light_debug_out_path.name
    strength = max(0.0, min(1.0, float(corr) * float(spacing_cons) * float(mean_cov)))
    t_sec = time.perf_counter() - t0
    meta = {
        "input": input_name,
        "output": out_path.name if out_path is not None else "",
        "lag": lag_full,
        "corr": corr,
        "peaks": len(peaks),
        "spacing_cons": spacing_cons,
        "strength": strength,
        "periodic_time_sec": t_sec,
        "scale": scale,
        "light_debug_output": light_out_name,
        "energy_strength": energy_strength,
        "binary_strength": binary_strength,
        "row_edge_strength": row_edge_strength,
        "row_dark_strength": row_dark_strength,
        "row_combined_strength": row_combined_strength,
        "mean_coverage_strength": mean_cov,
    }
    return meta


def draw_periodic_pattern_for_image(
    path: Path,
    *,
    light_debug_image: Optional[Image.Image] = None,
    light_debug_mask_image: Optional[Image.Image] = None,
    light_debug_out_path: Optional[Path] = None,
    min_lag_full_px: Optional[int] = None,
    max_lag_full_px: Optional[int] = None,
) -> dict[str, float | int | str]:
    return draw_periodic(
        path,
        light_debug_image=light_debug_image,
        light_debug_mask_image=light_debug_mask_image,
        light_debug_out_path=light_debug_out_path,
        min_lag_full_px=min_lag_full_px,
        max_lag_full_px=max_lag_full_px,
    )


def draw_periodic_pattern_for_pil(
    image: Image.Image,
    *,
    input_name: str = "in-memory",
    periodic_out_path: Optional[Path] = None,
    light_debug_image: Optional[Image.Image] = None,
    light_debug_mask_image: Optional[Image.Image] = None,
    light_debug_out_path: Optional[Path] = None,
    min_lag_full_px: Optional[int] = None,
    max_lag_full_px: Optional[int] = None,
) -> dict[str, float | int | str]:
    return _draw_periodic_from_gray_full(
        gray_full=image.convert("L"),
        input_name=input_name,
        out_path=periodic_out_path,
        light_debug_image=light_debug_image,
        light_debug_mask_image=light_debug_mask_image,
        light_debug_out_path=light_debug_out_path,
        min_lag_full_px=min_lag_full_px,
        max_lag_full_px=max_lag_full_px,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    args = ap.parse_args()
    if not args.image:
        raise SystemExit("Provide one image path. Example: python src/draw_periodic_pattern.py preprocessed/foo.png")
    draw_periodic(Path(args.image))


if __name__ == "__main__":
    main()
