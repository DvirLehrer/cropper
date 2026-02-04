#!/usr/bin/env python3
"""Decide and apply line correction (none/tilt/warp) from OCR lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

from PIL import Image

from line_structure import build_line_models


@dataclass(frozen=True)
class CorrectionDecision:
    mode: Literal["none", "tilt", "warp"]
    slope: float
    mean_abs: float
    std: float
    resid_mean: float
    resid_std: float
    curve_std: float


def decide_correction(line_words: List[List[Dict[str, Any]]]) -> CorrectionDecision:
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
        return CorrectionDecision(
            mode="none",
            slope=0.0,
            mean_abs=0.0,
            std=0.0,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
        )
    mean = sum(slopes) / len(slopes)
    mean_abs = sum(abs(s) for s in slopes) / len(slopes)
    var = sum((s - mean) ** 2 for s in slopes) / len(slopes)
    std = var ** 0.5

    if mean_abs < 0.0015 and resid_mean < 0.12:
        return CorrectionDecision(
            mode="none",
            slope=0.0,
            mean_abs=mean_abs,
            std=std,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
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
        )
    if curve_std >= 0.25 and resid_mean >= 0.13:
        return CorrectionDecision(
            mode="warp",
            slope=mean,
            mean_abs=mean_abs,
            std=std,
            resid_mean=resid_mean,
            resid_std=resid_std,
            curve_std=curve_std,
        )
    return CorrectionDecision(
        mode="none",
        slope=0.0,
        mean_abs=mean_abs,
        std=std,
        resid_mean=resid_mean,
        resid_std=resid_std,
        curve_std=curve_std,
    )


def apply_tilt(
    image: Image.Image,
    words: List[Dict[str, Any]],
    slope: float,
) -> Image.Image:
    if abs(slope) < 1e-6:
        return image
    cx = image.width / 2.0
    # output -> input mapping
    a, b, c = 1.0, 0.0, 0.0
    d, e, f = slope, 1.0, -slope * cx
    tilted = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)

    for w in words:
        x1, x2 = w["x1"], w["x2"]
        y1, y2 = w["y1"], w["y2"]
        y1a = y1 - slope * (x1 - cx)
        y1b = y1 - slope * (x2 - cx)
        y2a = y2 - slope * (x1 - cx)
        y2b = y2 - slope * (x2 - cx)
        w["y1"] = int(round(min(y1a, y1b, y2a, y2b)))
        w["y2"] = int(round(max(y1a, y1b, y2a, y2b)))
    return tilted
