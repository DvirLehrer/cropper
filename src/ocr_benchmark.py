#!/usr/bin/env python3
"""
Run Hebrew OCR on all images in benchmark/ and print text distance stats.

Dependencies:
  - pytesseract
  - pillow
  - Tesseract OCR with Hebrew language data installed (lang=heb)
"""

from __future__ import annotations

from typing import Dict, List

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
from best_sequence import find_best_sequences
from ocr_utils import iter_images, levenshtein, load_types, ocr_image, preprocess_image

TOP_K = 5
WINDOW_WORDS = 6
MAX_SKIP = 6


def _draw_best_sequences(
    image: Image.Image,
    line_words: List[List[Dict[str, Any]]],
    target_text: str,
) -> None:
    scored = find_best_sequences(
        line_words,
        target_text,
        window_words=WINDOW_WORDS,
        max_skip=MAX_SKIP,
        top_k=TOP_K,
    )
    draw = ImageDraw.Draw(image)
    for _, idx, (i, j) in scored:
        span_words = sorted(line_words[idx], key=lambda w: -w["x2"])[i : j + 1]
        x1 = min(w["x1"] for w in span_words)
        y1 = min(w["y1"] for w in span_words)
        x2 = max(w["x2"] for w in span_words)
        y2 = max(w["y2"] for w in span_words)
        draw.rectangle([x1, y1, x2, y2], outline=(160, 0, 255), width=3)


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
            for idx, line in enumerate(line_bboxes, start=1):
                draw.rectangle([line["x1"], line["y1"], line["x2"], line["y2"]], outline=(0, 128, 255), width=2)
            _draw_best_sequences(image, line_words, target_for_image)
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
