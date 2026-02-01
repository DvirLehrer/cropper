#!/usr/bin/env python3
"""Debug leftmost boxes: boxes with no other box to their left."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

from cropper_config import BENCHMARK_DIR, DEBUG_DIR, LANG
from ocr_utils import ocr_image


def _vertical_overlap(a: Dict[str, int], b: Dict[str, int]) -> int:
    return max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))


def _is_leftmost(w: Dict[str, int], words: List[Dict[str, int]]) -> bool:
    for other in words:
        if other is w:
            continue
        if other["x2"] <= w["x1"] and _vertical_overlap(other, w) > 0:
            return False
    return True


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(BENCHMARK_DIR.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
            continue
        result = ocr_image(path, lang=LANG)
        words = result["words"]
        if not words:
            continue
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for w in words:
            draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)

        leftmost = [w for w in words if _is_leftmost(w, words)]
        for w in leftmost:
            draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(255, 0, 0), width=3)

        out_path = DEBUG_DIR / f"{path.stem}_leftmost_debug.png"
        image.save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
