#!/usr/bin/env python3
"""Geometry and ROI helpers for the crop pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

STRIPE_HEIGHT_FRACS = (0.15, 0.25, 0.35, 0.5)
STRIPE_OVERLAP = 0.5
STRIPE_TOP_K = 2
STRIPE_PAD_FRAC = 0.08


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _box_edge_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    dx = 0.0
    if a["x2"] < b["x1"]:
        dx = b["x1"] - a["x2"]
    elif b["x2"] < a["x1"]:
        dx = a["x1"] - b["x2"]
    dy = 0.0
    if a["y2"] < b["y1"]:
        dy = b["y1"] - a["y2"]
    elif b["y2"] < a["y1"]:
        dy = a["y1"] - b["y2"]
    return (dx * dx + dy * dy) ** 0.5


def _cluster_bbox(
    words: List[Dict[str, Any]],
    link_scale: float = 2.0,
    return_cluster: bool = False,
) -> tuple[int, int, int, int] | tuple[tuple[int, int, int, int], List[Dict[str, Any]]] | None:
    if not words:
        return None
    sizes = []
    for w in words:
        text = w.get("text", "")
        char_count = max(len(text), 1)
        sizes.append((w["x2"] - w["x1"]) / char_count)
    median_size = _median(sizes)
    if median_size <= 0:
        return None
    link_dist = median_size * link_scale

    visited = [False] * len(words)
    clusters: List[List[int]] = []
    for i in range(len(words)):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        cluster = [i]
        while queue:
            idx = queue.pop()
            a = words[idx]
            for j in range(len(words)):
                if visited[j]:
                    continue
                b = words[j]
                if _box_edge_distance(a, b) <= link_dist:
                    visited[j] = True
                    queue.append(j)
                    cluster.append(j)
        clusters.append(cluster)

    if not clusters:
        return None
    best = max(clusters, key=len)
    kept = [words[i] for i in best]
    if not best:
        return None
    left = min(w["x1"] for w in kept)
    right = max(w["x2"] for w in kept)
    top = min(w["y1"] for w in kept)
    bottom = max(w["y2"] for w in kept)
    bbox = (left, top, right, bottom)
    if return_cluster:
        return bbox, kept
    return bbox


def _stripe_roi_bbox(pre: Image.Image) -> Optional[tuple[int, int, int, int]]:
    if pre.mode != "L":
        pre = pre.convert("L")
    w, h = pre.size
    if w < 2 or h < 2:
        return None
    pixels = list(pre.getdata())
    row_edge = [0.0] * (h - 1)
    for y in range(h - 1):
        row_sum = 0
        idx0 = y * w
        idx1 = (y + 1) * w
        for x in range(w):
            row_sum += abs(pixels[idx1 + x] - pixels[idx0 + x])
        row_edge[y] = row_sum / w
    prefix = [0.0]
    for val in row_edge:
        prefix.append(prefix[-1] + val)

    stripes: List[tuple[float, int, int]] = []
    for frac in STRIPE_HEIGHT_FRACS:
        stripe_h = max(2, int(round(h * frac)))
        step = max(1, int(round(stripe_h * (1.0 - STRIPE_OVERLAP))))
        for y in range(0, h - stripe_h + 1, step):
            y2 = y + stripe_h
            score = prefix[y2 - 1] - prefix[y]
            stripes.append((score, y, y2))

    if not stripes:
        return None
    stripes.sort(key=lambda s: s[0], reverse=True)
    top = [s for s in stripes[:STRIPE_TOP_K] if s[0] > 0]
    if not top:
        return None
    y1 = min(s[1] for s in top)
    y2 = max(s[2] for s in top)
    pad = int(round(h * STRIPE_PAD_FRAC))
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    if y2 <= y1:
        return None
    return (0, y1, w, y2)


def _clamp_bbox(
    bbox: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[int, int, int, int] | None:
    left, top, right, bottom = bbox
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
