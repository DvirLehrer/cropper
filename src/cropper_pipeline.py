#!/usr/bin/env python3
"""Shared single-image crop pipeline used by CLI and web."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image, ImageFilter

from config import settings
from core.lighting import (
    build_post_crop_stripes,
)
from core.line_mesh import build_block_mesh_from_lines
from core.line_fix import apply_tilt, decide_correction
from core.line_models import build_pil_mesh
from core.edge_candidates import _median_char_size
from core.crop_alignment import compute_opt_crop_bbox
from core.geometry import _clamp_bbox, _cluster_bbox
from ocr import _ocr_image_pil, _ocr_image_pil_sparse_merge
from core.settings_texts import strip_newlines


def _log(progress_cb: Optional[Callable[[str], None]], message: str) -> None:
    if progress_cb:
        progress_cb(message)


def _work_scale_for_image(width: int, height: int, max_pixels: int) -> float:
    area = width * height
    if max_pixels <= 0 or area <= max_pixels:
        return 1.0
    return (float(max_pixels) / float(area)) ** 0.5


def _downscale_for_work(image: Image.Image, scale: float) -> Image.Image:
    if scale >= 1.0:
        return image
    out_w = max(1, int(round(image.width * scale)))
    out_h = max(1, int(round(image.height * scale)))
    return image.resize((out_w, out_h), Image.Resampling.LANCZOS)


def _sync_full_orientation_after_ocr(
    full_image: Image.Image,
    work_before_ocr: Image.Image,
    work_after_ocr: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    if work_after_ocr.size == work_before_ocr.size:
        return full_image, work_after_ocr
    if work_after_ocr.size == (work_before_ocr.height, work_before_ocr.width):
        return full_image.rotate(90, expand=True), work_after_ocr
    return full_image, work_after_ocr


def _map_bbox_to_full(
    work_bbox: Tuple[int, int, int, int],
    work_size: Tuple[int, int],
    full_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    work_w, work_h = work_size
    full_w, full_h = full_size
    if work_w <= 0 or work_h <= 0:
        return None
    x1, y1, x2, y2 = work_bbox
    sx = float(full_w) / float(work_w)
    sy = float(full_h) / float(work_h)
    full_bbox = (
        int(math.floor(x1 * sx)),
        int(math.floor(y1 * sy)),
        int(math.ceil(x2 * sx)),
        int(math.ceil(y2 * sy)),
    )
    return _clamp_bbox(full_bbox, full_w, full_h)


def _scale_mesh_to_full(
    mesh_work: list[
        tuple[
            tuple[int, int, int, int],
            tuple[float, float, float, float, float, float, float, float],
        ]
    ],
    work_size: Tuple[int, int],
    full_size: Tuple[int, int],
) -> list[
    tuple[
        tuple[int, int, int, int],
        tuple[float, float, float, float, float, float, float, float],
    ]
]:
    work_w, work_h = work_size
    full_w, full_h = full_size
    if work_w <= 1 or work_h <= 1:
        return []
    sx = float(full_w - 1) / float(work_w - 1)
    sy = float(full_h - 1) / float(work_h - 1)

    def _x(v: float) -> float:
        return v * sx

    def _y(v: float) -> float:
        return v * sy

    mesh_full = []
    for quad, src in mesh_work:
        qx0 = int(round(_x(quad[0])))
        qy0 = int(round(_y(quad[1])))
        qx1 = int(round(_x(quad[2])))
        qy1 = int(round(_y(quad[3])))
        qx0 = max(0, min(full_w - 1, qx0))
        qx1 = max(0, min(full_w - 1, qx1))
        qy0 = max(0, min(full_h - 1, qy0))
        qy1 = max(0, min(full_h - 1, qy1))
        if qx1 <= qx0 or qy1 <= qy0:
            continue
        scaled_src = (
            _x(src[0]),
            _y(src[1]),
            _x(src[2]),
            _y(src[3]),
            _x(src[4]),
            _y(src[5]),
            _x(src[6]),
            _y(src[7]),
        )
        mesh_full.append(((qx0, qy0, qx1, qy1), scaled_src))
    return mesh_full


def _apply_transform_to_full(
    full_image: Image.Image,
    transform: Dict[str, Any],
    work_size: Tuple[int, int],
) -> Image.Image:
    mode = transform.get("mode")
    if mode == "tilt":
        return apply_tilt(full_image, [], float(transform["slope"]))
    if mode == "warp":
        mesh_work = transform.get("mesh") or []
        mesh_full = _scale_mesh_to_full(mesh_work, work_size, full_image.size)
        if mesh_full:
            return full_image.transform(full_image.size, Image.MESH, mesh_full, resample=Image.BICUBIC)
    return full_image


def _apply_layout_correction(
    image_full: Image.Image,
    words: list[dict[str, Any]],
    line_words: list[list[dict[str, Any]]],
    *,
    lang: str,
    timing_detail: Dict[str, float],
) -> tuple[
    Image.Image,
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
    bool,
    Optional[Dict[str, Any]],
]:
    correction = decide_correction(line_words)
    changed = False
    applied_transform: Optional[Dict[str, Any]] = None

    if settings.crop_service.enable_line_warp and settings.crop_service.apply_line_correction and correction.mode == "warp":
        xs, ys, grid = build_block_mesh_from_lines(line_words, image_full.width, image_full.height, grid_step=80)
        mesh = build_pil_mesh(xs, ys, grid)
        if mesh:
            original_image = image_full
            original_words = words
            original_line_words = line_words
            t_transform = time.perf_counter()
            image_full = image_full.transform(image_full.size, Image.MESH, mesh, resample=Image.BICUBIC)
            timing_detail["crop_warp_transform"] = time.perf_counter() - t_transform
            warp_result = _ocr_image_pil_sparse_merge(
                image_full,
                lang=lang,
                use_lighting_normalization=False,
                timing=timing_detail,
                timing_prefix="crop_warp_",
            )
            words = warp_result["words"]
            line_words = warp_result["line_words"]
            post = decide_correction(line_words)
            if post.curve_std > correction.curve_std * 1.05 or post.resid_mean > correction.resid_mean * 1.05:
                image_full = original_image
                words = original_words
                line_words = original_line_words
            else:
                changed = True
                applied_transform = {"mode": "warp", "mesh": mesh}
    elif settings.crop_service.enable_line_warp and settings.crop_service.apply_line_correction and correction.mode == "tilt":
        t_tilt = time.perf_counter()
        image_full = apply_tilt(image_full, words, correction.slope)
        timing_detail["crop_tilt_transform"] = time.perf_counter() - t_tilt
        tilt_result = _ocr_image_pil_sparse_merge(
            image_full,
            lang=lang,
            use_lighting_normalization=False,
            timing=timing_detail,
            timing_prefix="crop_tilt_",
        )
        words = tilt_result["words"]
        line_words = tilt_result["line_words"]
        changed = True
        applied_transform = {"mode": "tilt", "slope": correction.slope}

    return image_full, words, line_words, changed, applied_transform


def _apply_initial_cluster_crop(
    image_full: Image.Image,
    words: list[dict[str, Any]],
) -> tuple[Image.Image, bool, Optional[Tuple[int, int, int, int]]]:
    cluster_result = _cluster_bbox(words, return_cluster=True)
    if not cluster_result:
        return image_full, False, None
    crop_bbox = _clamp_bbox(cluster_result[0], image_full.width, image_full.height)
    if not crop_bbox:
        return image_full, False, None
    return image_full.crop(crop_bbox), True, crop_bbox


def _finalize_crop(
    image_full: Image.Image,
    opt_crop_bbox: Optional[Tuple[int, int, int, int]],
    *,
    target_chars: Optional[int],
    ocr_text: str,
) -> tuple[Image.Image, int, float]:
    cropped = image_full.crop(opt_crop_bbox) if opt_crop_bbox else image_full
    effective_target_chars = target_chars if target_chars is not None else max(len(ocr_text), 1)
    crop_area = cropped.width * cropped.height
    px_per_char = crop_area / max(effective_target_chars, 1)
    if (
        settings.crop_service.apply_crop_denoise
        and settings.crop_service.crop_denoise_size > 1
        and px_per_char > 1000
    ):
        cropped = cropped.filter(ImageFilter.MedianFilter(size=settings.crop_service.crop_denoise_size))
    return cropped, crop_area, px_per_char


def crop_image(
    image_full: Image.Image,
    *,
    lang: str = settings.ocr.lang,
    target_chars: Optional[int] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    _log(progress_cb, "Starting OCR pipeline")
    timing_detail: Dict[str, float] = {}
    full_current = image_full
    work_scale = _work_scale_for_image(
        full_current.width,
        full_current.height,
        settings.crop_service.work_max_pixels,
    )
    image_work = _downscale_for_work(full_current, work_scale)
    t0 = time.perf_counter()
    result = _ocr_image_pil(
        image_work,
        lang=lang,
        use_lighting_normalization=False,
        timing=timing_detail,
        timing_prefix="stage1_",
    )
    full_current, image_work = _sync_full_orientation_after_ocr(
        full_current,
        image_work,
        result["image_pil"],
    )
    t1 = time.perf_counter()
    _log(progress_cb, f"First OCR complete ({len(result['words'])} words)")

    words = result["words"]
    line_words = result["line_words"]
    correction = decide_correction(line_words)
    work_before_transform = image_work
    image_work, words, line_words, corrected_changed, applied_transform = _apply_layout_correction(
        image_work,
        words,
        line_words,
        lang=lang,
        timing_detail=timing_detail,
    )
    if corrected_changed and applied_transform:
        full_current = _apply_transform_to_full(full_current, applied_transform, work_before_transform.size)
    if corrected_changed:
        _log(progress_cb, "Applied tilt/warp correction")
    work_before_crop = image_work
    image_work, initial_crop_changed, initial_crop_bbox = _apply_initial_cluster_crop(image_work, words)
    if initial_crop_changed and initial_crop_bbox:
        full_crop_bbox = _map_bbox_to_full(initial_crop_bbox, work_before_crop.size, full_current.size)
        if full_crop_bbox:
            full_current = full_current.crop(full_crop_bbox)
    if initial_crop_changed:
        _log(progress_cb, "Applied initial cluster crop")
    t3 = time.perf_counter()

    if corrected_changed or initial_crop_changed:
        work_before_stage2 = image_work
        result = _ocr_image_pil(
            image_work,
            lang=lang,
            use_lighting_normalization=False,
            timing=timing_detail,
            timing_prefix="stage2_",
        )
        full_current, image_work = _sync_full_orientation_after_ocr(
            full_current,
            work_before_stage2,
            result["image_pil"],
        )
        words = result["words"]
        line_words = result["line_words"]
        _log(progress_cb, f"Second OCR complete ({len(words)} words)")
    t4 = time.perf_counter()

    ocr_text = strip_newlines(result["text"])
    cluster_result2 = _cluster_bbox(words, return_cluster=True)
    cluster_words = cluster_result2[1] if cluster_result2 else []
    opt_crop_bbox = None
    if cluster_words:
        t_edge = time.perf_counter()
        opt_crop_bbox = compute_opt_crop_bbox(cluster_words, line_words, image_work.width, image_work.height)
        timing_detail["crop_edge_align"] = time.perf_counter() - t_edge
        if opt_crop_bbox:
            _log(progress_cb, "Computed final edge-aligned crop")
    full_opt_crop_bbox = None
    if opt_crop_bbox:
        full_opt_crop_bbox = _map_bbox_to_full(opt_crop_bbox, image_work.size, full_current.size)

    cropped, crop_area, px_per_char = _finalize_crop(
        full_current,
        full_opt_crop_bbox,
        target_chars=target_chars,
        ocr_text=ocr_text,
    )
    t6 = time.perf_counter()
    avg_char_size = _median_char_size(words) if words else None
    (
        stripe_dark_mask,
        stripe_mask_continuum_debug,
        stripes_timing,
    ) = build_post_crop_stripes(
        cropped,
        avg_char_size=avg_char_size,
        fast_bg_max_dim=settings.crop_service.post_crop_stripes_fast_bg_max_dim,
    )
    timing_detail["post_crop_stripes_normalize"] = float(stripes_timing["post_crop_stripes_core"])
    timing_detail["post_crop_stripes_dark_mask"] = float(
        stripes_timing["post_crop_stripes_dark_mask_image"]
    )
    timing_detail["post_crop_stripes_mask_continuum_debug"] = float(
        stripes_timing["post_crop_stripes_mask_continuum_debug"]
    )
    t7 = time.perf_counter()
    _log(progress_cb, f"Done ({cropped.width}x{cropped.height})")

    return {
        "cropped": cropped,
        "stripe_ready": cropped,
        "stripe_dark_mask": stripe_dark_mask,
        "stripe_mask_continuum_debug": stripe_mask_continuum_debug,
        "avg_char_size": avg_char_size,
        "ocr_text": ocr_text,
        "text": result["text"],
        "correction": correction,
        "crop_area": crop_area,
        "px_per_char": px_per_char,
        "timing": {
            "ocr1": t1 - t0,
            "crop": t3 - t1,
            "ocr2": t4 - t3,
            "debug": 0.0,
            "crop_finalize": t6 - t4,
            "post_crop_stripes": t7 - t6,
            "left": 0.0,
        },
        "timing_detail": timing_detail,
    }
