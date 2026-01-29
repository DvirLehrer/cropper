#!/usr/bin/env python3
"""Crop areas inside OCR bbox but outside expected text box."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image

import sys

sys.path.append("src")

from cropper_config import BENCHMARK_DIR, LANG, TEXT_DIR, TYPES_CSV
from ocr_utils import load_types, ocr_image
from target_texts import load_target_texts
from expected_layout import estimate_layout
from ocr_benchmark import _clamp_layout_to_image, _filtered_bbox


def _rect_diff(
    outer: Tuple[int, int, int, int],
    inner: Tuple[int, int, int, int],
) -> List[Tuple[int, int, int, int]]:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    ix1 = max(ix1, ox1)
    iy1 = max(iy1, oy1)
    ix2 = min(ix2, ox2)
    iy2 = min(iy2, oy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return [outer]
    pieces = []
    # top
    if oy1 < iy1:
        pieces.append((ox1, oy1, ox2, iy1))
    # bottom
    if iy2 < oy2:
        pieces.append((ox1, iy2, ox2, oy2))
    # left
    if ox1 < ix1:
        pieces.append((ox1, iy1, ix1, iy2))
    # right
    if ix2 < ox2:
        pieces.append((ix2, iy1, ox2, iy2))
    return [p for p in pieces if p[2] > p[0] and p[3] > p[1]]


def main() -> None:
    out_dir = Path("debug_images") / "in_ocr_not_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)

    for path in sorted(BENCHMARK_DIR.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
            continue
        image_type = type_map.get(path.name)
        if not image_type:
            continue
        target_text = (
            target_texts["m"]
            if image_type == "m"
            else target_texts[
                {
                    "s": "shema",
                    "v": "vehaya",
                    "k": "kadesh",
                    "p": "peter",
                }[image_type]
            ]
        )
        result = ocr_image(path, lang=LANG)
        words = result["words"]
        line_words = result["line_words"]
        if not words or not line_words:
            continue

        boundary_word_index = len(target_texts["shema"].split()) if image_type == "m" else None
        layout = estimate_layout(
            line_words,
            words,
            target_text,
            window_words=6,
            max_skip=6,
            score_threshold=0.4,
            boundary_word_index=boundary_word_index,
        )
        if not layout:
            continue

        image = Image.open(path).convert("RGB")
        layout = _clamp_layout_to_image(layout, image.width, image.height)
        red_bbox = _filtered_bbox(words)
        if not red_bbox:
            continue

        doc_left, doc_top, doc_right, doc_bottom = layout["doc_box"]
        red = (int(red_bbox[0]), int(red_bbox[1]), int(red_bbox[2]), int(red_bbox[3]))
        cyan = (int(doc_left), int(doc_top), int(doc_right), int(doc_bottom))

        pieces = _rect_diff(red, cyan)
        for idx, (x1, y1, x2, y2) in enumerate(pieces, start=1):
            crop = image.crop((x1, y1, x2, y2))
            out_path = out_dir / f"{path.stem}_p{idx}.png"
            crop.save(out_path)


if __name__ == "__main__":
    main()
