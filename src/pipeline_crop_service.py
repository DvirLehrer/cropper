#!/usr/bin/env python3
"""Shared single-image crop pipeline used by CLI and web."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from cropper_config import LANG
from line_block_mesh import build_block_mesh_from_lines
from line_correction import apply_tilt, decide_correction
from line_structure import build_pil_mesh
from ocr_utils import preprocess_image
from pipeline_align import compute_opt_crop_bbox
from pipeline_geometry import _clamp_bbox, _cluster_bbox, _stripe_roi_bbox
from pipeline_ocr import _ocr_image_pil, _ocr_image_pil_sparse_merge
from magnet_mask import suppress_margin_magnets
from target_texts import strip_newlines

ENABLE_LINE_WARP = True
APPLY_LINE_CORRECTION = True
APPLY_CROP_DENOISE = True
CROP_DENOISE_SIZE = 3


def _log(progress_cb: Optional[Callable[[str], None]], message: str) -> None:
    if progress_cb:
        progress_cb(message)


def _apply_layout_correction(
    image_full: Image.Image,
    words: list[dict[str, Any]],
    line_words: list[list[dict[str, Any]]],
    *,
    lang: str,
    timing_detail: Dict[str, float],
) -> tuple[Image.Image, list[dict[str, Any]], list[list[dict[str, Any]]], bool]:
    correction = decide_correction(line_words)
    changed = False

    if ENABLE_LINE_WARP and APPLY_LINE_CORRECTION and correction.mode == "warp":
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
    elif ENABLE_LINE_WARP and APPLY_LINE_CORRECTION and correction.mode == "tilt":
        t_tilt = time.perf_counter()
        image_full = apply_tilt(image_full, words, correction.slope)
        timing_detail["crop_tilt_transform"] = time.perf_counter() - t_tilt
        tilt_result = _ocr_image_pil_sparse_merge(
            image_full,
            lang=lang,
            timing=timing_detail,
            timing_prefix="crop_tilt_",
        )
        words = tilt_result["words"]
        line_words = tilt_result["line_words"]
        changed = True

    return image_full, words, line_words, changed


def _apply_initial_cluster_crop(
    image_full: Image.Image,
    words: list[dict[str, Any]],
) -> tuple[Image.Image, bool]:
    cluster_result = _cluster_bbox(words, return_cluster=True)
    if not cluster_result:
        return image_full, False
    crop_bbox = _clamp_bbox(cluster_result[0], image_full.width, image_full.height)
    if not crop_bbox:
        return image_full, False
    return image_full.crop(crop_bbox), True


def _write_debug_overlay(debug_path: Path, image_full: Image.Image, words: list[dict[str, Any]]) -> None:
    debug_image = image_full.copy()
    draw = ImageDraw.Draw(debug_image)
    for w in words:
        draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)
    debug_image.save(debug_path)


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
    if APPLY_CROP_DENOISE and CROP_DENOISE_SIZE > 1 and px_per_char > 1000:
        cropped = cropped.filter(ImageFilter.MedianFilter(size=CROP_DENOISE_SIZE))
    return cropped, crop_area, px_per_char


def crop_image(
    image_full: Image.Image,
    *,
    lang: str = LANG,
    target_chars: Optional[int] = None,
    debug_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    _log(progress_cb, "Starting OCR pipeline")
    timing_detail: Dict[str, float] = {}
    t0 = time.perf_counter()
    result = _ocr_image_pil(
        image_full,
        lang=lang,
        timing=timing_detail,
        timing_prefix="stage1_",
    )
    image_full = result["image_pil"]
    if len(result["words"]) < 50:
        pre_for_stripe = preprocess_image(image_full, upscale_factor=1.0, sharpen=False)
        stripe_bbox = _stripe_roi_bbox(pre_for_stripe)
        if stripe_bbox:
            image_full = image_full.crop(stripe_bbox)
            result = _ocr_image_pil(
                image_full,
                lang=lang,
                allow_rotate=False,
                timing=timing_detail,
                timing_prefix="stage1_retry_",
            )
            image_full = result["image_pil"]
            _log(progress_cb, "Applied stripe ROI retry")
    t1 = time.perf_counter()
    _log(progress_cb, f"First OCR complete ({len(result['words'])} words)")

    masked_image, circles, _ = suppress_margin_magnets(image_full)
    if circles:
        image_full = masked_image
        _log(progress_cb, f"Masked margin magnets ({len(circles)} circles)")
        result = _ocr_image_pil(
            image_full,
            lang=lang,
            timing=timing_detail,
            timing_prefix="stage1_masked_",
        )
        image_full = result["image_pil"]
        _log(progress_cb, f"OCR after masking complete ({len(result['words'])} words)")

    words = result["words"]
    line_words = result["line_words"]
    correction = decide_correction(line_words)
    t2 = time.perf_counter()
    _log(progress_cb, f"Layout mode: {correction.mode}")

    image_full, words, line_words, corrected_changed = _apply_layout_correction(
        image_full,
        words,
        line_words,
        lang=lang,
        timing_detail=timing_detail,
    )
    if corrected_changed:
        _log(progress_cb, "Applied tilt/warp correction")
    image_full, initial_crop_changed = _apply_initial_cluster_crop(image_full, words)
    if initial_crop_changed:
        _log(progress_cb, "Applied initial cluster crop")
    t3 = time.perf_counter()

    if corrected_changed or initial_crop_changed:
        result = _ocr_image_pil(
            image_full,
            lang=lang,
            timing=timing_detail,
            timing_prefix="stage2_",
        )
        words = result["words"]
        line_words = result["line_words"]
        image_full = result["image_pil"]
        _log(progress_cb, f"Second OCR complete ({len(words)} words)")
    t4 = time.perf_counter()

    ocr_text = strip_newlines(result["text"])
    cluster_result2 = _cluster_bbox(words, return_cluster=True)
    cluster_words = cluster_result2[1] if cluster_result2 else []
    opt_crop_bbox = None
    if cluster_words:
        t_edge = time.perf_counter()
        opt_crop_bbox = compute_opt_crop_bbox(cluster_words, line_words, image_full.width, image_full.height)
        timing_detail["crop_edge_align"] = time.perf_counter() - t_edge
        if opt_crop_bbox:
            _log(progress_cb, "Computed final edge-aligned crop")

    if debug_path:
        _write_debug_overlay(debug_path, image_full, words)
    t5 = time.perf_counter()

    cropped, crop_area, px_per_char = _finalize_crop(
        image_full,
        opt_crop_bbox,
        target_chars=target_chars,
        ocr_text=ocr_text,
    )
    _log(progress_cb, f"Done ({cropped.width}x{cropped.height})")

    return {
        "cropped": cropped,
        "ocr_text": ocr_text,
        "text": result["text"],
        "correction": correction,
        "crop_area": crop_area,
        "px_per_char": px_per_char,
        "timing": {
            "ocr1": t1 - t0,
            "layout": t2 - t1,
            "crop": t3 - t2,
            "ocr2": t4 - t3,
            "debug": t5 - t4,
            "left": 0.0,
        },
        "timing_detail": timing_detail,
    }
