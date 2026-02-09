#!/usr/bin/env python3
"""Block-level mesh construction from top/bottom quadratic curves."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _solve_3x3(m: List[List[float]], v: List[float]) -> Optional[Tuple[float, float, float]]:
    a = [row[:] for row in m]
    b = v[:]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-8:
            return None
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
        inv = 1.0 / a[i][i]
        for c in range(i, 3):
            a[i][c] *= inv
        b[i] *= inv
        for r in range(3):
            if r == i:
                continue
            f = a[r][i]
            for c in range(i, 3):
                a[r][c] -= f * a[i][c]
            b[r] -= f * b[i]
    return b[0], b[1], b[2]


def _fit_quadratic(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float, float]]:
    if len(xs) < 3:
        return None
    s0 = float(len(xs))
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x * x * x for x in xs)
    s4 = sum(x * x * x * x for x in xs)
    t0 = sum(ys)
    t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum((x * x) * y for x, y in zip(xs, ys))
    mat = [
        [s4, s3, s2],
        [s3, s2, s1],
        [s2, s1, s0],
    ]
    return _solve_3x3(mat, [t2, t1, t0])


def build_block_mesh_from_lines(
    line_words: List[List[Dict[str, Any]]],
    width: int,
    height: int,
    grid_step: int = 80,
    smooth_window: int = 3,
    gain: float = 1.5,
    max_shift_frac: float = 0.6,
    min_words: int = 3,
) -> Tuple[List[int], List[int], List[List[float]]]:
    lines = [line for line in line_words if line and len(line) >= min_words]
    if not lines:
        return [], [], []
    lines.sort(key=lambda line: sum((w["y1"] + w["y2"]) / 2.0 for w in line) / len(line))
    top = lines[0]
    bottom = lines[-1]
    top_xs = [(w["x1"] + w["x2"]) / 2.0 for w in top]
    top_ys = [w["y1"] for w in top]
    bot_xs = [(w["x1"] + w["x2"]) / 2.0 for w in bottom]
    bot_ys = [w["y2"] for w in bottom]
    top_fit = _fit_quadratic(top_xs, top_ys)
    bot_fit = _fit_quadratic(bot_xs, bot_ys)
    if top_fit is None or bot_fit is None:
        return [], [], []
    top_target = sorted(top_ys)[len(top_ys) // 2]
    bot_target = sorted(bot_ys)[len(bot_ys) // 2]
    denom = bot_target - top_target
    if abs(denom) < 1e-6:
        return [], [], []

    xs = list(range(0, width, max(20, grid_step)))
    if xs and xs[-1] != width - 1:
        xs.append(width - 1)
    ys = list(range(0, height, max(20, grid_step)))
    if ys and ys[-1] != height - 1:
        ys.append(height - 1)

    top_shift = []
    bot_shift = []
    a, b, c = top_fit
    for x in xs:
        y = a * x * x + b * x + c
        top_shift.append((y - top_target) * gain)
    a, b, c = bot_fit
    for x in xs:
        y = a * x * x + b * x + c
        bot_shift.append((y - bot_target) * gain)

    if smooth_window > 1:
        half = smooth_window // 2

        def _smooth(arr: List[float]) -> List[float]:
            out = []
            for i in range(len(arr)):
                lo = max(0, i - half)
                hi = min(len(arr), i + half + 1)
                out.append(sum(arr[lo:hi]) / (hi - lo))
            return out

        top_shift = _smooth(top_shift)
        bot_shift = _smooth(bot_shift)

    heights = [
        (w["y2"] - w["y1"])
        for line in lines
        for w in line
        if w.get("y2") is not None and w.get("y1") is not None
    ]
    heights.sort()
    median_h = heights[len(heights) // 2] if heights else 0.0
    max_shift = max_shift_frac * median_h if median_h > 0 else None

    grid = []
    for y in ys:
        t = (y - top_target) / denom
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        row = []
        for xi in range(len(xs)):
            dy = (1.0 - t) * top_shift[xi] + t * bot_shift[xi]
            if max_shift is not None:
                if dy > max_shift:
                    dy = max_shift
                elif dy < -max_shift:
                    dy = -max_shift
            row.append(dy)
        grid.append(row)
    return xs, ys, grid
