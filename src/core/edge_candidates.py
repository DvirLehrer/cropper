#!/usr/bin/env python3
"""Edge candidate helpers for final crop alignment."""

from __future__ import annotations

import bisect
from typing import Any, Dict, List, Tuple

from core.geometry import _median


def _median_char_size(words: List[Dict[str, Any]]) -> float:
    sizes = []
    for w in words:
        text = w.get("text", "")
        char_count = max(len(text), 1)
        sizes.append((w["x2"] - w["x1"]) / char_count)
    return _median(sizes)


def _median_word_height(words: List[Dict[str, Any]]) -> float:
    heights = [(w["y2"] - w["y1"]) for w in words]
    return _median(heights)


def _edge_flags(words: List[Dict[str, Any]]) -> Tuple[List[bool], List[bool], List[bool], List[bool]]:
    if not words:
        return [], [], [], []

    heights = [w["y2"] - w["y1"] for w in words]
    widths = [w["x2"] - w["x1"] for w in words]
    bin_h = _median(heights) or 10.0
    bin_w = _median(widths) or 10.0

    y_bins_x2: Dict[int, List[float]] = {}
    y_bins_x1: Dict[int, List[float]] = {}
    x_bins_y2: Dict[int, List[float]] = {}
    x_bins_y1: Dict[int, List[float]] = {}

    for w in words:
        y_start = int(w["y1"] // bin_h)
        y_end = int(w["y2"] // bin_h)
        for b in range(y_start, y_end + 1):
            y_bins_x2.setdefault(b, []).append(w["x2"])
            y_bins_x1.setdefault(b, []).append(w["x1"])

        x_start = int(w["x1"] // bin_w)
        x_end = int(w["x2"] // bin_w)
        for b in range(x_start, x_end + 1):
            x_bins_y2.setdefault(b, []).append(w["y2"])
            x_bins_y1.setdefault(b, []).append(w["y1"])

    for bins in (y_bins_x2, y_bins_x1, x_bins_y2, x_bins_y1):
        for key in bins:
            bins[key].sort()

    leftmost = [True] * len(words)
    rightmost = [True] * len(words)
    topmost = [True] * len(words)
    bottommost = [True] * len(words)

    for idx, w in enumerate(words):
        y_start = int(w["y1"] // bin_h)
        y_end = int(w["y2"] // bin_h)
        for b in range(y_start, y_end + 1):
            xs_left = y_bins_x2.get(b, [])
            if xs_left and xs_left[0] <= w["x1"]:
                if bisect.bisect_right(xs_left, w["x1"]) > 0:
                    leftmost[idx] = False
            xs_right = y_bins_x1.get(b, [])
            if xs_right and xs_right[-1] >= w["x2"]:
                if bisect.bisect_left(xs_right, w["x2"]) < len(xs_right):
                    rightmost[idx] = False

        x_start = int(w["x1"] // bin_w)
        x_end = int(w["x2"] // bin_w)
        for b in range(x_start, x_end + 1):
            ys_top = x_bins_y2.get(b, [])
            if ys_top and ys_top[0] <= w["y1"]:
                if bisect.bisect_right(ys_top, w["y1"]) > 0:
                    topmost[idx] = False
            ys_bottom = x_bins_y1.get(b, [])
            if ys_bottom and ys_bottom[-1] >= w["y2"]:
                if bisect.bisect_left(ys_bottom, w["y2"]) < len(ys_bottom):
                    bottommost[idx] = False

    return leftmost, rightmost, topmost, bottommost


def _filter_boxes_by_char_size(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []
    widths = [w["x2"] - w["x1"] for w in words]
    heights = [w["y2"] - w["y1"] for w in words]
    mean_w = sum(widths) / len(widths)
    mean_h = sum(heights) / len(heights)
    if mean_w <= 0 or mean_h <= 0:
        return list(words)
    kept = []
    for w, width, height in zip(words, widths, heights):
        w_bad = width < (0.5 * mean_w) or width > (2.0 * mean_w)
        h_bad = height < (0.5 * mean_h) or height > (2.0 * mean_h)
        if w_bad and h_bad:
            continue
        kept.append(w)
    return kept if kept else list(words)


def _filter_outliers(xs: List[float], keep_lower: bool) -> List[float]:
    if not xs:
        return []
    med = _median(xs)
    mad = _median([abs(x - med) for x in xs])
    if mad <= 0:
        return list(xs)
    cutoff = 2.0 * mad
    if keep_lower:
        return [x for x in xs if x <= med + cutoff]
    return [x for x in xs if x >= med - cutoff]


def _next_candidates(xs: List[float], increasing: bool) -> List[List[float]]:
    if not xs:
        return []
    sorted_xs = sorted(xs, reverse=not increasing)
    next_map: Dict[float, float] = {}
    for i in range(len(sorted_xs) - 1):
        next_map[sorted_xs[i]] = sorted_xs[i + 1]
    return [[x, next_map[x]] if x in next_map else [x] for x in xs]
