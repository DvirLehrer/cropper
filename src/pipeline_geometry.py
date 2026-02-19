#!/usr/bin/env python3
"""Geometry and ROI helpers for the crop pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


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
