#!/usr/bin/env python3
"""Debug median-char-size filter over all benchmark images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from cropper_config import BENCHMARK_DIR, DEBUG_DIR, LANG
from ocr_utils import ocr_image


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


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


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(BENCHMARK_DIR.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
            continue
        result = ocr_image(path, lang=LANG)
        words: List[Dict[str, Any]] = result["words"]
        if not words:
            continue

        sizes = []
        for w in words:
            text = w.get("text", "")
            char_count = max(len(text), 1)
            sizes.append((w["x2"] - w["x1"]) / char_count)
        median_size = _median(sizes)
        if median_size <= 0:
            continue
        kept = []
        removed = 0
        for idx, (w, size) in enumerate(zip(words, sizes)):
            nearest = None
            for jdx, other in enumerate(words):
                if jdx == idx:
                    continue
                dist = _box_edge_distance(w, other)
                if nearest is None or dist < nearest:
                    nearest = dist
            if nearest is None:
                nearest = 0.0
            dist_units = nearest / max(median_size, 1e-6)
            tol = 3.0 / (1.0 + dist_units)
            if tol < 1.0:
                tol = 1.0
            min_size = median_size / tol
            max_size = median_size * tol
            if size < min_size or size > max_size:
                removed += 1
                continue
            kept.append(w)

        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Full OCR bbox in red.
        left = min(w["x1"] for w in words)
        right = max(w["x2"] for w in words)
        top = min(w["y1"] for w in words)
        bottom = max(w["y2"] for w in words)
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)

        # Filtered bbox in cyan.
        if kept:
            kleft = min(w["x1"] for w in kept)
            kright = max(w["x2"] for w in kept)
            ktop = min(w["y1"] for w in kept)
            kbottom = max(w["y2"] for w in kept)
            draw.rectangle([kleft, ktop, kright, kbottom], outline=(0, 255, 255), width=3)

        out_path = DEBUG_DIR / f"{path.stem}_debug.png"
        image.save(out_path)
        print(
            f"{path.name}: median={median_size:.2f} "
            f"kept={len(kept)} removed={removed} -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
