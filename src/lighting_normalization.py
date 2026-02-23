#!/usr/bin/env python3
"""Lighting normalization helpers for OCR preprocessing."""

from __future__ import annotations

import time
from collections import deque

from PIL import Image, ImageFilter

MIN_BLUR_RADIUS = 12
MAX_BLUR_RADIUS = 72
BLUR_SIZE_FRAC = 0.045
DIVIDE_GAIN = 168
NORM_BLEND_ALPHA = 0.28
DARK_MASK_PERCENTILE = 0.20
MASK_VERTICAL_EXPAND_PERCENTILE = 0.30
CONTRAST_CUTOFF = 0.02
LIGHTEN_GAIN = 1.22
LIGHTEN_BIAS = 14
MASK_FILL_MEAN_WINDOW_PX = 16
MASK_FILL_NOISE_AMPLITUDE = 2


def _blur_radius_for_size(width: int, height: int) -> int:
    base = int(round(min(width, height) * BLUR_SIZE_FRAC))
    if base < MIN_BLUR_RADIUS:
        return MIN_BLUR_RADIUS
    if base > MAX_BLUR_RADIUS:
        return MAX_BLUR_RADIUS
    return base


def _blur_background_fast(gray: Image.Image, radius: int, fast_bg_max_dim: int | None) -> Image.Image:
    if not fast_bg_max_dim or fast_bg_max_dim <= 0:
        return gray.filter(ImageFilter.GaussianBlur(radius=radius))
    w, h = gray.size
    max_dim = max(w, h)
    if max_dim <= fast_bg_max_dim:
        return gray.filter(ImageFilter.GaussianBlur(radius=radius))

    scale = float(fast_bg_max_dim) / float(max_dim)
    w_small = max(1, int(round(w * scale)))
    h_small = max(1, int(round(h * scale)))
    small = gray.resize((w_small, h_small), Image.BILINEAR)
    radius_small = max(1, int(round(float(radius) * scale)))
    small_bg = small.filter(ImageFilter.GaussianBlur(radius=radius_small))
    return small_bg.resize((w, h), Image.BILINEAR)


def _percentile_threshold(values: list[int], frac: float) -> int:
    if not values:
        return 0
    hist = [0] * 256
    for v in values:
        hist[v] += 1
    target = int(round(max(0.0, min(1.0, frac)) * len(values)))
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target:
            return i
    return 255


def _augment_with_trapped_small_regions(
    dark_mask: list[bool],
    width: int,
    height: int,
    *,
    avg_char_size: float | None,
) -> list[bool]:
    if not avg_char_size or avg_char_size <= 0:
        return dark_mask
    max_span = max(2, int(round(avg_char_size)))
    max_area = max(4, int(round(avg_char_size * avg_char_size)))
    n = width * height
    visited = bytearray(n)
    out = list(dark_mask)

    for i in range(n):
        if visited[i] or dark_mask[i]:
            continue
        q = deque([i])
        visited[i] = 1
        comp: list[int] = []
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0
        touches_border = False

        while q:
            p = q.popleft()
            comp.append(p)
            y, x = divmod(p, width)
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                touches_border = True
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                ni = ny * width + nx
                if visited[ni] or dark_mask[ni]:
                    continue
                visited[ni] = 1
                q.append(ni)

        if touches_border:
            continue
        comp_w = max_x - min_x + 1
        comp_h = max_y - min_y + 1
        if len(comp) <= max_area and comp_w <= max_span and comp_h <= max_span:
            for p in comp:
                out[p] = True

    return out


def _expand_mask_2px(mask: list[bool], width: int, height: int) -> list[bool]:
    out = list(mask)
    for y in range(height):
        row = y * width
        for x in range(width):
            i = row + x
            if not mask[i]:
                continue
            for ny in range(y - 2, y + 3):
                if ny < 0 or ny >= height:
                    continue
                nrow = ny * width
                for nx in range(x - 2, x + 3):
                    if nx < 0 or nx >= width:
                        continue
                    out[nrow + nx] = True
    return out


def _expand_mask_vertical_by_percentile(
    px: list[int], mask: list[bool], width: int, height: int, frac: float
) -> list[bool]:
    out = list(mask)
    thr = _percentile_threshold(px, frac)
    eligible = [v <= thr for v in px]
    q = deque(i for i, m in enumerate(out) if m)
    while q:
        i = q.popleft()
        y, x = divmod(i, width)
        if y > 0:
            j = (y - 1) * width + x
            if (not out[j]) and eligible[j]:
                out[j] = True
                q.append(j)
        if y + 1 < height:
            j = (y + 1) * width + x
            if (not out[j]) and eligible[j]:
                out[j] = True
                q.append(j)
    return out


def _masked_autocontrast(px: list[int], mask: list[bool], cutoff: float) -> list[int]:
    valid = [v for i, v in enumerate(px) if not mask[i]]
    if not valid:
        return list(px)

    hist = [0] * 256
    for v in valid:
        hist[v] += 1

    total = len(valid)
    cut = int(round(max(0.0, min(0.2, cutoff)) * total))

    lo = 0
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc > cut:
            lo = i
            break

    hi = 255
    acc = 0
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc > cut:
            hi = i
            break

    if hi <= lo:
        return list(px)

    scale = 255.0 / float(hi - lo)
    out = list(px)
    for i, v in enumerate(px):
        if mask[i]:
            continue
        nv = int(round((v - lo) * scale))
        if nv < 0:
            nv = 0
        elif nv > 255:
            nv = 255
        out[i] = nv
    return out


def _fill_mask_lr_continuum(px: list[int], mask: list[bool], width: int, height: int) -> list[int]:
    out = list(px)
    for y in range(height):
        row = y * width
        x = 0
        while x < width:
            i = row + x
            if not mask[i]:
                x += 1
                continue

            run_start = x
            x += 1
            while x < width and mask[row + x]:
                x += 1
            run_end = x - 1

            lx = run_start - 1
            if lx < 0 or mask[row + lx]:
                lx = -1
            rx = x
            if rx >= width or mask[row + rx]:
                rx = -1

            run_len = run_end - run_start + 1
            if lx >= 0 and rx >= 0 and lx != rx:
                lv = px[row + lx]
                rv = px[row + rx]
                span = rx - lx
                for xi in range(run_start, run_end + 1):
                    t = xi - lx
                    # Linear interpolation between nearest left/right unmasked samples.
                    out[row + xi] = ((lv * (span - t)) + (rv * t) + (span // 2)) // span
            elif lx >= 0:
                out[row + run_start : row + run_end + 1] = [px[row + lx]] * run_len
            elif rx >= 0:
                out[row + run_start : row + run_end + 1] = [px[row + rx]] * run_len
            else:
                out[row + run_start : row + run_end + 1] = [255] * run_len
    return out


def _normalize_core(
    gray: Image.Image,
    avg_char_size: float | None,
    fast_bg_max_dim: int | None = None,
) -> tuple[list[int], list[bool], int, int]:
    gray = gray.convert("L")
    original = gray.copy()

    radius = _blur_radius_for_size(gray.width, gray.height)
    bg = _blur_background_fast(gray, radius, fast_bg_max_dim)
    src = list(gray.getdata())
    back = list(bg.getdata())

    out = []
    for p, b in zip(src, back):
        val = (DIVIDE_GAIN * (p + 1)) // (b + 1)
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        out.append(val)

    norm = Image.new("L", gray.size)
    norm.putdata(out)
    blended = Image.blend(original, norm, NORM_BLEND_ALPHA)

    w, h = blended.size
    px2 = list(blended.getdata())
    dark_thr = _percentile_threshold(px2, DARK_MASK_PERCENTILE)
    dark_mask = [v <= dark_thr for v in px2]
    dark_mask = _augment_with_trapped_small_regions(
        dark_mask,
        w,
        h,
        avg_char_size=avg_char_size,
    )
    dark_mask = _expand_mask_vertical_by_percentile(
        px2, dark_mask, w, h, MASK_VERTICAL_EXPAND_PERCENTILE
    )
    dark_mask = _expand_mask_2px(dark_mask, w, h)

    contrasted = _masked_autocontrast(px2, dark_mask, CONTRAST_CUTOFF)
    out2 = list(contrasted)
    for i, v0 in enumerate(contrasted):
        if dark_mask[i]:
            out2[i] = 255
            continue
        v = int(v0 * LIGHTEN_GAIN + LIGHTEN_BIAS)
        if v > 255:
            v = 255
        out2[i] = v

    return out2, dark_mask, w, h


def normalize_uneven_lighting(image: Image.Image, avg_char_size: float | None = None) -> Image.Image:
    """Normalize image, ignore masked dark regions, and increase contrast elsewhere."""
    gray = image.convert("L")
    out2, _, w, h = _normalize_core(gray, avg_char_size)
    final = Image.new("L", (w, h))
    final.putdata(out2)
    return final


def normalize_uneven_lighting_dark_mask(image: Image.Image, avg_char_size: float | None = None) -> Image.Image:
    gray = image.convert("L")
    _, dark_mask, w, h = _normalize_core(gray, avg_char_size)
    out = Image.new("L", (w, h))
    out.putdata([255 if m else 0 for m in dark_mask])
    return out


def normalize_uneven_lighting_mask_continuum_debug(
    image: Image.Image, avg_char_size: float | None = None
) -> Image.Image:
    """Debug image: mask region is filled by row-wise left/right nearest continuum."""
    gray = image.convert("L")
    out2, dark_mask, w, h = _normalize_core(gray, avg_char_size)
    debug_px = _fill_mask_lr_continuum(out2, dark_mask, w, h)
    out = Image.new("L", (w, h))
    out.putdata(debug_px)
    return out


def build_post_crop_stripes(
    image: Image.Image,
    avg_char_size: float | None = None,
    fast_bg_max_dim: int | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, float]]:
    """Build all stripe-related outputs from one shared normalization pass."""
    gray = image.convert("L")
    t0 = time.perf_counter()
    out2, dark_mask, w, h = _normalize_core(gray, avg_char_size, fast_bg_max_dim=fast_bg_max_dim)
    t1 = time.perf_counter()

    stripe_ready = Image.new("L", (w, h))
    stripe_ready.putdata(out2)
    t2 = time.perf_counter()

    stripe_dark_mask = Image.new("L", (w, h))
    stripe_dark_mask.putdata([255 if m else 0 for m in dark_mask])
    t3 = time.perf_counter()

    debug_px = _fill_mask_lr_continuum(out2, dark_mask, w, h)
    stripe_mask_continuum_debug = Image.new("L", (w, h))
    stripe_mask_continuum_debug.putdata(debug_px)
    t4 = time.perf_counter()

    timing = {
        "post_crop_stripes_core": t1 - t0,
        "post_crop_stripes_ready_image": t2 - t1,
        "post_crop_stripes_dark_mask_image": t3 - t2,
        "post_crop_stripes_mask_continuum_debug": t4 - t3,
    }
    return stripe_ready, stripe_dark_mask, stripe_mask_continuum_debug, timing
