#!/usr/bin/env python3
"""Line-structure detection from OCR line boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class LineModel:
    m: float
    b: float
    y_target: float

    def y_at(self, x: float) -> float:
        return self.m * x + self.b


def _fit_line(xs: List[float], ys: List[float]) -> LineModel:
    if len(xs) < 2:
        y = ys[0] if ys else 0.0
        return LineModel(m=0.0, b=y, y_target=y)
    n = float(len(xs))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = (n * sxx) - (sx * sx)
    if abs(denom) < 1e-8:
        mean_y = sy / n
        return LineModel(m=0.0, b=mean_y, y_target=mean_y)
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    y_sorted = sorted(ys)
    y_target = float(y_sorted[len(y_sorted) // 2])
    return LineModel(m=m, b=b, y_target=y_target)


def build_line_models(line_words: List[List[Dict[str, Any]]]) -> List[LineModel]:
    models: List[LineModel] = []
    for line in line_words:
        if not line:
            continue
        xs = [(w["x1"] + w["x2"]) / 2.0 for w in line]
        ys = [w["y2"] for w in line]
        models.append(_fit_line(xs, ys))
    return models


def line_segments(
    models: List[LineModel],
    width: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    segs = []
    for model in models:
        x0 = 0
        x1 = max(0, width - 1)
        y0 = int(round(model.y_at(x0)))
        y1 = int(round(model.y_at(x1)))
        segs.append(((x0, y0), (x1, y1)))
    return segs




def build_pil_mesh(
    xs: List[int],
    ys: List[int],
    grid: List[List[float]],
) -> List[Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float, float, float, float]]]:
    mesh = []
    if len(xs) < 2 or len(ys) < 2:
        return mesh
    for iy in range(len(ys) - 1):
        for ix in range(len(xs) - 1):
            x0, x1 = xs[ix], xs[ix + 1]
            y0, y1 = ys[iy], ys[iy + 1]
            quad = (x0, y0, x1, y1)
            dy00 = grid[iy][ix]
            dy10 = grid[iy][ix + 1]
            dy01 = grid[iy + 1][ix]
            dy11 = grid[iy + 1][ix + 1]
            top0 = y0 + dy00
            top1 = y0 + dy10
            bot1 = y1 + dy11
            bot0 = y1 + dy01
            if min(top0, top1) >= max(bot0, bot1):
                top0, top1, bot0, bot1 = y0, y0, y1, y1
            # PIL expects quad order: UL, LL, LR, UR
            src = (x0, top0, x0, bot0, x1, bot1, x1, top1)
            mesh.append((quad, src))
    return mesh

