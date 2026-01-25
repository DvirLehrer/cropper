#!/usr/bin/env python3
"""
Run Hebrew OCR on all images in benchmark/ and print text + first/last word boxes.

Dependencies:
  - pytesseract
  - pillow
  - Tesseract OCR with Hebrew language data installed (lang=heb)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import pytesseract

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

def _word_boxes(image: Image.Image, lang: str) -> List[Dict[str, Any]]:
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words: List[Dict[str, Any]] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(
            {
                "text": text,
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
            }
        )
    return words

def _reading_order_hebrew(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Approximate Hebrew reading order: top-to-bottom, right-to-left within a line.
    return sorted(words, key=lambda w: (w["y1"], -w["x2"]))


def _preprocess_image(image: Image.Image) -> Image.Image:
    # Simplify image gently: grayscale -> light contrast stretch -> lift shadows.
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    # Gamma < 1 lifts shadows without hard binarization.
    gamma = 0.85
    gray = gray.point(lambda p: int(255 * ((p / 255) ** gamma)))
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray


def ocr_image(path: Path, lang: str) -> Dict[str, Any]:
    image = Image.open(path)
    pre = _preprocess_image(image)
    words = _word_boxes(pre, lang=lang)
    text = pytesseract.image_to_string(pre, lang=lang)
    ordered = _reading_order_hebrew(words)
    first_word = ordered[0] if ordered else None
    last_word = ordered[-1] if ordered else None
    return {
        "image": str(path),
        "text": text.strip(),
        "first_word": first_word,
        "last_word": last_word,
        "words": words,
    }


def iter_images(benchmark_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    return sorted([p for p in benchmark_dir.iterdir() if p.suffix.lower() in exts])


def _load_types(csv_path: Path) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["type"] for row in reader if row.get("filename")}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]




def main() -> None:
    benchmark_dir = BENCHMARK_DIR
    if not benchmark_dir.exists():
        raise SystemExit(f"Benchmark dir not found: {benchmark_dir}")

    type_map = _load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)
    target_text = target_texts["m"]

    debug_dir = DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    ocr_text_dir = OCR_TEXT_DIR
    ocr_text_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = PREPROCESSED_DIR
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_images(benchmark_dir):
        image_type = type_map.get(path.name)
        result = ocr_image(path, lang=LANG)
        ocr_text = strip_newlines(result["text"])
        words = result["words"]
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
            out_path = debug_dir / f"{path.stem}_debug.png"
            image.save(out_path)
            pre = _preprocess_image(Image.open(path))
            pre_out = preprocessed_dir / f"{path.stem}_pre.png"
            pre.save(pre_out)
        text_out = ocr_text_dir / f"{path.stem}.txt"
        text_out.write_text(result["text"], encoding="utf-8")
        if image_type == "m":
            distance = _levenshtein(ocr_text, target_text)
            print(f"type: {image_type}  distance: {distance}")
        else:
            names = ("shema", "vehaya", "kadesh", "peter")
            distances = {name: _levenshtein(ocr_text, target_texts[name]) for name in names}
            guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
            print(f"type: {image_type}  guess: {guess_name}  distance: {guess_distance}")


if __name__ == "__main__":
    main()
