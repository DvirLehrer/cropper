#!/usr/bin/env python3
"""Lighting normalization helpers for OCR preprocessing."""

from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image, ImageFilter
from config import settings


def _blur_radius_for_size(width: int, height: int) -> int:
    cfg = settings.lighting
    base = int(round(min(width, height) * cfg.blur_size_frac))
    if base < cfg.min_blur_radius:
        return cfg.min_blur_radius
    if base > cfg.max_blur_radius:
        return cfg.max_blur_radius
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


def _percentile_threshold(values: list[int] | np.ndarray, frac: float) -> int:
    if len(values) == 0:
        return 0
    vals = np.asarray(values, dtype=np.uint8).ravel()
    hist = np.bincount(vals, minlength=256)
    target = int(round(max(0.0, min(1.0, frac)) * len(values)))
    acc = 0
    for i, c in enumerate(hist.tolist()):
        acc += int(c)
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
    dark = np.asarray(dark_mask, dtype=np.bool_).reshape((height, width))
    non_dark = (~dark).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(non_dark, connectivity=4)
    out = dark.copy()
    for lbl in range(1, labels_count):
        left = int(stats[lbl, cv2.CC_STAT_LEFT])
        top = int(stats[lbl, cv2.CC_STAT_TOP])
        comp_w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        touches_border = (
            left == 0
            or top == 0
            or (left + comp_w) >= width
            or (top + comp_h) >= height
        )
        if touches_border:
            continue
        if area <= max_area and comp_w <= max_span and comp_h <= max_span:
            out[labels == lbl] = True
    return out.ravel().tolist()


def _expand_mask_2px(mask: list[bool], width: int, height: int) -> list[bool]:
    src = np.asarray(mask, dtype=np.uint8).reshape((height, width))
    kernel = np.ones((5, 5), dtype=np.uint8)
    out = cv2.dilate(src, kernel, iterations=1) > 0
    return out.ravel().tolist()


def _expand_mask_vertical_by_percentile(
    px: list[int], mask: list[bool], width: int, height: int, frac: float
) -> list[bool]:
    out = np.asarray(mask, dtype=np.bool_).reshape((height, width))
    thr = _percentile_threshold(px, frac)
    eligible = (np.asarray(px, dtype=np.uint8).reshape((height, width)) <= thr)
    for x in range(width):
        col_elig = eligible[:, x]
        if not col_elig.any():
            continue
        col_out = out[:, x]
        changes = np.diff(np.concatenate(([0], col_elig.view(np.uint8), [0])))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        for s, e in zip(starts, ends):
            active = col_out[s : e + 1].any()
            if not active and s > 0:
                active = bool(col_out[s - 1])
            if not active and (e + 1) < height:
                active = bool(col_out[e + 1])
            if active:
                col_out[s : e + 1] = True
        out[:, x] = col_out
    return out.ravel().tolist()


def _masked_autocontrast(px: list[int], mask: list[bool], cutoff: float) -> list[int]:
    px_arr = np.asarray(px, dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.bool_)
    valid = px_arr[~mask_arr]
    if valid.size == 0:
        return list(px)

    hist = np.bincount(valid, minlength=256)

    total = int(valid.size)
    cut = int(round(max(0.0, min(0.2, cutoff)) * total))

    lo = 0
    acc = 0
    for i, c in enumerate(hist.tolist()):
        acc += int(c)
        if acc > cut:
            lo = i
            break

    hi = 255
    acc = 0
    for i in range(255, -1, -1):
        acc += int(hist[i])
        if acc > cut:
            hi = i
            break

    if hi <= lo:
        return list(px)

    scale = 255.0 / float(hi - lo)
    out = px_arr.astype(np.int16, copy=True)
    vals = out[~mask_arr].astype(np.float32)
    mapped = np.rint((vals - float(lo)) * scale)
    mapped = np.clip(mapped, 0.0, 255.0).astype(np.uint8)
    out_u8 = px_arr.copy()
    out_u8[~mask_arr] = mapped
    return out_u8.tolist()


def _fill_mask_lr_continuum_numpy(px: list[int], mask: list[bool], width: int, height: int) -> list[int]:
    arr = np.asarray(px, dtype=np.uint8).reshape((height, width))
    mask_arr = np.asarray(mask, dtype=np.bool_).reshape((height, width))
    out = arr.astype(np.float32, copy=True)
    x = np.arange(width, dtype=np.float32)
    for y in range(height):
        row_mask = mask_arr[y]
        if not row_mask.any():
            continue
        valid_idx = np.flatnonzero(~row_mask)
        if valid_idx.size == 0:
            out[y, :] = 255.0
            continue
        row = out[y]
        row[row_mask] = np.interp(x[row_mask], valid_idx, row[valid_idx])
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8).ravel().tolist()


def _normalize_core(
    gray: Image.Image,
    avg_char_size: float | None,
    fast_bg_max_dim: int | None = None,
) -> tuple[list[int], list[bool], int, int]:
    cfg = settings.lighting
    gray = gray.convert("L")
    original = gray.copy()

    radius = _blur_radius_for_size(gray.width, gray.height)
    bg = _blur_background_fast(gray, radius, fast_bg_max_dim)
    src = np.asarray(gray, dtype=np.uint16)
    back = np.asarray(bg, dtype=np.uint16)
    out_arr = (int(cfg.divide_gain) * (src + 1)) // (back + 1)
    out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
    out = out_arr.ravel().tolist()

    norm = Image.new("L", gray.size)
    norm.putdata(out)
    blended = Image.blend(original, norm, cfg.norm_blend_alpha)

    w, h = blended.size
    px2_arr = np.asarray(blended, dtype=np.uint8)
    px2 = px2_arr.ravel().tolist()
    dark_thr = _percentile_threshold(px2, cfg.dark_mask_percentile)
    dark_mask = (px2_arr <= dark_thr).ravel().tolist()
    dark_mask = _augment_with_trapped_small_regions(
        dark_mask,
        w,
        h,
        avg_char_size=avg_char_size,
    )
    dark_mask = _expand_mask_vertical_by_percentile(
        px2, dark_mask, w, h, cfg.mask_vertical_expand_percentile
    )
    dark_mask = _expand_mask_2px(dark_mask, w, h)

    contrasted = np.asarray(_masked_autocontrast(px2, dark_mask, cfg.contrast_cutoff), dtype=np.uint8)
    mask_arr = np.asarray(dark_mask, dtype=np.bool_)
    out2_arr = contrasted.astype(np.float32)
    out2_arr[~mask_arr] = np.clip(
        (out2_arr[~mask_arr] * float(cfg.lighten_gain)) + float(cfg.lighten_bias), 0.0, 255.0
    )
    out2_arr[mask_arr] = 255.0
    out2 = out2_arr.astype(np.uint8).tolist()

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
    debug_px = _fill_mask_lr_continuum_numpy(out2, dark_mask, w, h)
    out = Image.new("L", (w, h))
    out.putdata(debug_px)
    return out


def build_post_crop_stripes(
    image: Image.Image,
    avg_char_size: float | None = None,
    fast_bg_max_dim: int | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, float]]:
    """Build stripe-related outputs from one shared normalization pass."""
    gray = image.convert("L")
    out_w, out_h = gray.size
    work_gray = gray
    work_avg_char = avg_char_size
    scale = 1.0
    max_pixels = settings.crop_service.post_crop_stripes_work_max_pixels
    if max_pixels > 0 and (out_w * out_h) > max_pixels:
        scale = (float(max_pixels) / float(out_w * out_h)) ** 0.5
        work_w = max(1, int(round(out_w * scale)))
        work_h = max(1, int(round(out_h * scale)))
        work_gray = gray.resize((work_w, work_h), Image.BILINEAR)
        if isinstance(avg_char_size, (int, float)) and avg_char_size > 0:
            work_avg_char = float(avg_char_size) * scale

    t0 = time.perf_counter()
    out2, dark_mask, w, h = _normalize_core(work_gray, work_avg_char, fast_bg_max_dim=fast_bg_max_dim)
    t1 = time.perf_counter()

    stripe_dark_mask = Image.new("L", (w, h))
    stripe_dark_mask.putdata([255 if m else 0 for m in dark_mask])
    t2 = time.perf_counter()

    debug_px = _fill_mask_lr_continuum_numpy(out2, dark_mask, w, h)
    stripe_mask_continuum_debug = Image.new("L", (w, h))
    stripe_mask_continuum_debug.putdata(debug_px)
    t3 = time.perf_counter()

    if (w, h) != (out_w, out_h):
        stripe_dark_mask = stripe_dark_mask.resize((out_w, out_h), Image.NEAREST)
        stripe_mask_continuum_debug = stripe_mask_continuum_debug.resize((out_w, out_h), Image.BILINEAR)

    timing = {
        "post_crop_stripes_core": t1 - t0,
        "post_crop_stripes_dark_mask_image": t2 - t1,
        "post_crop_stripes_mask_continuum_debug": t3 - t2,
    }
    return stripe_dark_mask, stripe_mask_continuum_debug, timing
