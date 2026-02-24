#!/usr/bin/env python3
"""Final edge alignment stage for crop boundaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.edge_shift import align_edges_shift_side
from core.edge_candidates import (
    _edge_flags,
    _filter_boxes_by_char_size,
    _filter_outliers,
    _median_char_size,
    _median_word_height,
    _next_candidates,
)
from core.geometry import _clamp_bbox, _median


def _line_bottoms(line_words: List[List[Dict[str, Any]]]) -> List[float]:
    bottoms = []
    for line in line_words:
        if line:
            bottoms.append(max(w["y2"] for w in line))
    return bottoms


def _best_line(
    side: str,
    values: List[float],
    char_size: float,
    epsilon: float,
) -> Optional[float]:
    if not values:
        return None
    keep_lower = side in ("left", "top")
    values = _filter_outliers(values, keep_lower=keep_lower)
    if not values:
        return None
    increasing = side in ("left", "top")
    res = align_edges_shift_side(
        side,
        _next_candidates(values, increasing=increasing),
        values,
        penalty_lambda=100.0,
        normalize_penalty=True,
        strategy="next",
        verbose=False,
    )
    best = res.line_x
    original = min(values) if keep_lower else max(values)
    if abs(best - original) < char_size:
        best = original
    if best != original:
        best += -epsilon if keep_lower else epsilon
    return best


def compute_opt_crop_bbox(
    words: List[Dict[str, Any]],
    line_words: List[List[Dict[str, Any]]],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    cluster_words = words
    if not cluster_words:
        return None
    filtered_words = _filter_boxes_by_char_size(cluster_words)
    left_flags, right_flags, top_flags, bottom_flags = _edge_flags(filtered_words)
    widths = [w["x2"] - w["x1"] for w in filtered_words]
    heights = [w["y2"] - w["y1"] for w in filtered_words]
    median_w = _median(widths)
    median_h = _median(heights)
    min_w = 0.2 * median_w if median_w > 0 else 0.0
    min_h = 0.2 * median_h if median_h > 0 else 0.0
    bottom_right_weight = 3
    left_lines: List[float] = []
    right_lines: List[float] = []
    top_lines: List[float] = []
    bottom_lines: List[float] = []

    for idx, w in enumerate(filtered_words):
        if left_flags[idx]:
            left_lines.append(w["x1"])
        if right_flags[idx]:
            right_lines.append(w["x2"])
        if top_flags[idx]:
            top_lines.append(w["y1"])
        if bottom_flags[idx] and (w["x2"] - w["x1"]) >= min_w and (w["y2"] - w["y1"]) >= min_h:
            y = w["y2"]
            bottom_lines.append(y)
            if right_flags[idx]:
                bottom_lines.extend([y] * (bottom_right_weight - 1))

    char_w = _median_char_size(filtered_words)
    char_h = _median_word_height(filtered_words)
    epsilon = 4.0
    best_left = _best_line("left", left_lines, char_w, epsilon)
    best_right = _best_line("right", right_lines, char_w, epsilon)
    best_top = _best_line("top", top_lines, char_h, epsilon)
    best_bottom = _best_line("bottom", bottom_lines, char_h, epsilon)

    bottoms = _line_bottoms(line_words)
    if bottoms:
        max_line_bottom = max(bottoms)
        tol = max(char_h * 0.6, 4.0)
        if best_bottom is None or best_bottom < max_line_bottom - tol:
            best_bottom = max_line_bottom + epsilon

    if any(v is None for v in (best_left, best_right, best_top, best_bottom)):
        return None
    return _clamp_bbox(
        (int(best_left), int(best_top), int(best_right), int(best_bottom)),
        image_width,
        image_height,
    )
