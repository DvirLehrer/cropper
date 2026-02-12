#!/usr/bin/env python3
"""Shared single-image crop pipeline used by CLI and web."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, ImageFilter

from cropper_config import LANG
from line_block_mesh import build_block_mesh_from_lines
from line_correction import apply_tilt, decide_correction
from line_structure import build_line_models, build_pil_mesh
from ocr_utils import preprocess_image
from pipeline_align import compute_opt_crop_bbox
from pipeline_geometry import _clamp_bbox, _cluster_bbox, _stripe_roi_bbox
from pipeline_ocr import _ocr_image_pil, _ocr_image_pil_sparse_merge
from target_texts import strip_newlines

ENABLE_LINE_WARP = True
APPLY_LINE_CORRECTION = True
APPLY_CROP_DENOISE = True
CROP_DENOISE_SIZE = 3


def crop_image(
    image_full: Image.Image,
    *,
    lang: str = LANG,
    target_chars: Optional[int] = None,
    debug_path: Optional[Path] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = _ocr_image_pil(image_full, lang=lang)
    image_full = result["image_pil"]
    if len(result["words"]) < 50:
        pre_for_stripe = preprocess_image(image_full, upscale_factor=1.0, sharpen=False)
        stripe_bbox = _stripe_roi_bbox(pre_for_stripe)
        if stripe_bbox:
            image_full = image_full.crop(stripe_bbox)
            result = _ocr_image_pil(image_full, lang=lang, allow_rotate=False)
            image_full = result["image_pil"]
    t1 = time.perf_counter()

    words = result["words"]
    line_words = result["line_words"]
    correction = decide_correction(line_words)
    t2 = time.perf_counter()

    if ENABLE_LINE_WARP and APPLY_LINE_CORRECTION and correction.mode == "warp":
        models = build_line_models(line_words)
        xs, ys, grid = build_block_mesh_from_lines(line_words, image_full.width, image_full.height, grid_step=80)
        mesh = build_pil_mesh(xs, ys, grid)
        if mesh:
            original_image = image_full
            original_words = [dict(w) for w in words]
            original_line_words = [list(line) for line in line_words]
            image_full = image_full.transform(image_full.size, Image.MESH, mesh, resample=Image.BICUBIC)
            warp_result = _ocr_image_pil_sparse_merge(image_full, lang=lang)
            words = warp_result["words"]
            line_words = warp_result["line_words"]
            post = decide_correction(line_words)
            if post.curve_std > correction.curve_std * 1.05 or post.resid_mean > correction.resid_mean * 1.05:
                image_full = original_image
                words = original_words
                line_words = original_line_words
    elif ENABLE_LINE_WARP and APPLY_LINE_CORRECTION and correction.mode == "tilt":
        image_full = apply_tilt(image_full, words, correction.slope)
        tilt_result = _ocr_image_pil_sparse_merge(image_full, lang=lang)
        words = tilt_result["words"]
        line_words = tilt_result["line_words"]

    cluster_result = _cluster_bbox(words, return_cluster=True)
    if cluster_result:
        crop_bbox = _clamp_bbox(cluster_result[0], image_full.width, image_full.height)
        if crop_bbox:
            image_full = image_full.crop(crop_bbox)
    t3 = time.perf_counter()

    result = _ocr_image_pil(image_full, lang=lang)
    t4 = time.perf_counter()
    words = result["words"]
    line_words = result["line_words"]
    image_full = result["image_pil"]
    ocr_text = strip_newlines(result["text"])

    opt_crop_bbox = None
    cluster_result2 = _cluster_bbox(words, return_cluster=True)
    cluster_words = cluster_result2[1] if cluster_result2 else []
    if cluster_words:
        opt_crop_bbox = compute_opt_crop_bbox(cluster_words, line_words, image_full.width, image_full.height)

    if debug_path:
        debug_image = image_full.copy()
        draw = ImageDraw.Draw(debug_image)
        for w in words:
            draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)
        debug_image.save(debug_path)
    t5 = time.perf_counter()

    cropped = image_full.crop(opt_crop_bbox) if opt_crop_bbox else image_full
    effective_target_chars = target_chars if target_chars is not None else max(len(ocr_text), 1)
    crop_area = cropped.width * cropped.height
    px_per_char = crop_area / max(effective_target_chars, 1)
    if APPLY_CROP_DENOISE and CROP_DENOISE_SIZE > 1 and px_per_char > 1000:
        cropped = cropped.filter(ImageFilter.MedianFilter(size=CROP_DENOISE_SIZE))

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
    }
