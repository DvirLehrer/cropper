#!/usr/bin/env python3
"""Image-based global tilt estimation from structural line segments."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _weighted_median(values: List[float], weights: List[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total = float(sum(weights))
    if total <= 0:
        return None
    acc = 0.0
    for value, weight in pairs:
        acc += float(weight)
        if acc >= total * 0.5:
            return float(value)
    return float(pairs[-1][0])


def estimate_global_tilt_deg(image: Optional[Image.Image]) -> Tuple[Optional[float], float]:
    """Return (median angle in degrees, confidence in [0, 1])."""
    if image is None:
        return None, 0.0

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=90,
        minLineLength=max(40, int(min(image.size) * 0.08)),
        maxLineGap=18,
    )
    if lines is None:
        return None, 0.0

    angles: List[float] = []
    weights: List[float] = []
    for seg in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in seg]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        if abs(angle) > 20.0:
            continue
        length = math.hypot(dx, dy)
        if length < 25.0:
            continue
        angles.append(angle)
        weights.append(length)

    if not angles:
        return None, 0.0

    median_angle = _weighted_median(angles, weights)
    if median_angle is None:
        return None, 0.0
    deviations = [abs(a - median_angle) for a in angles]
    mad = _weighted_median(deviations, weights)
    if mad is None:
        mad = 0.0
    count_score = min(1.0, float(len(angles)) / 80.0)
    spread_score = max(0.0, 1.0 - float(mad) / 2.5)
    confidence = count_score * spread_score
    return float(median_angle), float(confidence)
