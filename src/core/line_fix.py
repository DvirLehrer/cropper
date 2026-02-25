#!/usr/bin/env python3
"""Decide and apply line correction (none/tilt/warp) from OCR lines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

from core.line_models import build_line_models
from core.tilt_from_image import estimate_global_tilt_deg


@dataclass(frozen=True)
class CorrectionDecision:
    mode: Literal["none", "tilt", "warp"]
    slope: float
    mean_abs: float
    std: float
    resid_mean: float
    resid_std: float
    curve_std: float
    image_angle_deg: Optional[float] = None
    image_confidence: float = 0.0


def decide_correction(
    line_words: List[List[Dict[str, Any]]],
    image: Optional[Image.Image] = None,
) -> CorrectionDecision:
    models = build_line_models(line_words)
    slopes = [m.m for m in models]
    heights = [
        (w["y2"] - w["y1"])
        for line in line_words
        for w in line
        if w.get("y2") is not None and w.get("y1") is not None
    ]
    heights_sorted = sorted(heights)
    scale = heights_sorted[len(heights_sorted) // 2] if heights_sorted else 1.0
    if scale <= 0:
        scale = 1.0

    resid_means: List[float] = []
    resid_ranges: List[float] = []
    for line, model in zip(line_words, models):
        if not line:
            continue
        xs = [(w["x1"] + w["x2"]) / 2.0 for w in line]
        ys = [w["y2"] for w in line]
        if not xs:
            continue
        residuals = [abs(y - model.y_at(x)) for x, y in zip(xs, ys)]
        resid_means.append(sum(residuals) / len(residuals))
        resid_ranges.append(max(residuals) - min(residuals))

    resid_mean = (sum(resid_means) / len(resid_means)) if resid_means else 0.0
    resid_var = (
        sum((r - resid_mean) ** 2 for r in resid_means) / len(resid_means)
        if resid_means
        else 0.0
    )
    resid_std = resid_var ** 0.5
    curve_mean = (sum(resid_ranges) / len(resid_ranges)) if resid_ranges else 0.0
    curve_var = (
        sum((r - curve_mean) ** 2 for r in resid_ranges) / len(resid_ranges)
        if resid_ranges
        else 0.0
    )
    curve_std = curve_var ** 0.5

    resid_mean /= scale
    resid_std /= scale
    curve_std /= scale

    if len(slopes) < 2:
        image_angle_deg, image_confidence = estimate_global_tilt_deg(image)
        return CorrectionDecision(
            mode="none",
            slope=0.0,
            mean_abs=0.0,
            std=0.0,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
            image_angle_deg=image_angle_deg,
            image_confidence=image_confidence,
        )
    mean = sum(slopes) / len(slopes)
    mean_abs = sum(abs(s) for s in slopes) / len(slopes)
    var = sum((s - mean) ** 2 for s in slopes) / len(slopes)
    std = var ** 0.5

    image_angle_deg, image_confidence = estimate_global_tilt_deg(image)
    image_tilt = (
        image_angle_deg is not None
        and image_confidence >= 0.55
        and abs(image_angle_deg) >= 0.6
        and resid_mean <= 0.40
    )
    image_slope = math.tan(math.radians(image_angle_deg)) if image_angle_deg is not None else 0.0

    if mean_abs < 0.0015 and resid_mean < 0.12 and not image_tilt:
        return CorrectionDecision(
            mode="none",
            slope=0.0,
            mean_abs=mean_abs,
            std=std,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
            image_angle_deg=image_angle_deg,
            image_confidence=image_confidence,
        )
    if image_tilt:
        return CorrectionDecision(
            mode="tilt",
            slope=image_slope,
            mean_abs=mean_abs,
            std=std,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
            image_angle_deg=image_angle_deg,
            image_confidence=image_confidence,
        )
    if mean_abs >= 0.004 and std < 0.004 and resid_mean <= 0.16 and curve_std < 0.2:
        return CorrectionDecision(
            mode="tilt",
            slope=mean,
            mean_abs=mean_abs,
            std=std,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
            image_angle_deg=image_angle_deg,
            image_confidence=image_confidence,
        )
    if curve_std >= 0.2 or resid_mean >= 0.2:
        return CorrectionDecision(
            mode="warp",
            slope=mean,
            mean_abs=mean_abs,
            std=std,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
            image_angle_deg=image_angle_deg,
            image_confidence=image_confidence,
        )
    return CorrectionDecision(
        mode="none",
        slope=0.0,
        mean_abs=mean_abs,
        std=std,
        resid_mean=resid_mean,
        resid_std=resid_std,
        curve_std=curve_std,
        image_angle_deg=image_angle_deg,
        image_confidence=image_confidence,
    )


def apply_tilt(
    image: Image.Image,
    words: List[Dict[str, Any]],
    slope: float,
) -> Image.Image:
    del words  # OCR is re-run after transform; word boxes are not reused.
    angle_deg = math.degrees(math.atan(slope))
    if abs(angle_deg) < 1e-3:
        return image
    return image.rotate(
        angle_deg,
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
