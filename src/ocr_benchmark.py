#!/usr/bin/env python3
"""
Run Hebrew OCR on all images in benchmark/ and print text distance stats.

Dependencies:
  - pytesseract
  - pillow
  - Tesseract OCR with Hebrew language data installed (lang=heb)
"""

from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image, ImageDraw

from cropper_config import (
    BENCHMARK_DIR,
    DEBUG_DIR,
    LANG,
    OCR_TEXT_DIR,
    PREPROCESSED_DIR,
    TEXT_DIR,
    TYPES_CSV,
)
from target_texts import load_target_texts, strip_newlines
from expected_layout import estimate_layout
from ocr_utils import iter_images, levenshtein, load_types, ocr_image, preprocess_image

WINDOW_WORDS = 6
MAX_SKIP = 6


def _draw_expected_layout(
    image: Image.Image,
    words: List[Dict[str, Any]],
    line_words: List[List[Dict[str, Any]]],
    target_text: str,
    boundary_word_index: int | None = None,
) -> None:
    layout = estimate_layout(
        line_words,
        words,
        target_text,
        window_words=WINDOW_WORDS,
        max_skip=MAX_SKIP,
        score_threshold=0.4,
        boundary_word_index=boundary_word_index,
    )
    if not layout:
        return
    draw = ImageDraw.Draw(image)
    doc_left, doc_top, doc_right, doc_bottom = layout["doc_box"]
    draw.rectangle([doc_left, doc_top, doc_right, doc_bottom], outline=(0, 255, 255), width=3)
    exp_x = int(round(layout["expected_start_x"]))
    exp_x = max(0, min(exp_x, image.width - 1))
    draw.line([(exp_x, doc_top), (exp_x, doc_top + 20)], fill=(255, 0, 0), width=3)


def main() -> None:
    benchmark_dir = BENCHMARK_DIR
    if not benchmark_dir.exists():
        raise SystemExit(f"Benchmark dir not found: {benchmark_dir}")

    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)

    debug_dir = DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    ocr_text_dir = OCR_TEXT_DIR
    ocr_text_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = PREPROCESSED_DIR
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_images(benchmark_dir):
        print(f"processing: {path.name}")
        image_type = type_map.get(path.name)
        result = ocr_image(path, lang=LANG)
        ocr_text = strip_newlines(result["text"])
        target_for_image = (
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
        words = result["words"]
        line_bboxes = result["line_bboxes"]
        line_words = result["line_words"]

        pre = preprocess_image(Image.open(path))
        pre_out = preprocessed_dir / f"{path.stem}_pre.png"
        pre.save(pre_out)

        if words:
            left = min(w["x1"] for w in words)
            right = max(w["x2"] for w in words)
            top = min(w["y1"] for w in words)
            bottom = max(w["y2"] for w in words)
            image = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
            for w in words:
                draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)
            boundary_word_index = None
            if image_type == "m":
                boundary_word_index = len(target_texts["shema"].split())
            _draw_expected_layout(image, words, line_words, target_for_image, boundary_word_index)
            out_path = debug_dir / f"{path.stem}_debug.png"
            image.save(out_path)

        text_out = ocr_text_dir / f"{path.stem}.txt"
        text_out.write_text(result["text"], encoding="utf-8")

        if image_type == "m":
            target = target_texts["m"]
            distance = levenshtein(ocr_text, target)
            print(f"type: {image_type}  distance: {distance}")
        else:
            names = ("shema", "vehaya", "kadesh", "peter")
            distances = {name: levenshtein(ocr_text, target_texts[name]) for name in names}
            guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
            print(f"type: {image_type}  guess: {guess_name}  distance: {guess_distance}")


if __name__ == "__main__":
    main()
