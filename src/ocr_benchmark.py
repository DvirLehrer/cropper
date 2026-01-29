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
from expected_layout import estimate_layout
from ocr_utils import iter_images, levenshtein, load_types, ocr_image, preprocess_image

WINDOW_WORDS = 6
MAX_SKIP = 6


def _draw_layout(image: Image.Image, layout: Dict[str, Any], offset: tuple[int, int] = (0, 0)) -> None:
    draw = ImageDraw.Draw(image)
    dx, dy = offset
    doc_left, doc_top, doc_right, doc_bottom = layout["doc_box"]
    doc_left -= dx
    doc_right -= dx
    doc_top -= dy
    doc_bottom -= dy
    box_w = doc_right - doc_left
    box_h = doc_bottom - doc_top
    if box_w > image.width:
        doc_left = 0
        doc_right = image.width
    else:
        if doc_left < 0:
            doc_left = 0
            doc_right = box_w
        if doc_right > image.width:
            doc_right = image.width
            doc_left = doc_right - box_w
    if box_h > image.height:
        doc_top = 0
        doc_bottom = image.height
    else:
        if doc_top < 0:
            doc_top = 0
            doc_bottom = box_h
        if doc_bottom > image.height:
            doc_bottom = image.height
            doc_top = doc_bottom - box_h
    draw.rectangle([doc_left, doc_top, doc_right, doc_bottom], outline=(0, 255, 255), width=3)
    exp_x = int(round(layout["expected_start_x"] - dx))
    exp_x = max(0, min(exp_x, image.width - 1))
    draw.line([(exp_x, doc_top), (exp_x, doc_top + 20)], fill=(255, 0, 0), width=3)


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


def _filtered_bbox(words: List[Dict[str, Any]]) -> tuple[int, int, int, int] | None:
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
    kept = []
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
        if min_size <= size <= max_size:
            kept.append(w)
    if not kept:
        return None
    left = min(w["x1"] for w in kept)
    right = max(w["x2"] for w in kept)
    top = min(w["y1"] for w in kept)
    bottom = max(w["y2"] for w in kept)
    return left, top, right, bottom


def _ocr_image_pil(image: Image.Image, lang: str) -> Dict[str, Any]:
    pre = preprocess_image(image)
    data = pytesseract.image_to_data(
        pre,
        lang=lang,
        config="--psm 4",
        output_type=pytesseract.Output.DICT,
    )
    from ocr_utils import word_boxes_from_data, text_from_words

    words = word_boxes_from_data(data)
    text, line_bboxes, line_words = text_from_words(words)
    return {
        "image": "in-memory",
        "text": text.strip(),
        "words": words,
        "line_bboxes": line_bboxes,
        "line_words": line_words,
    }


def _clamp_layout_to_image(layout: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    doc_left, doc_top, doc_right, doc_bottom = layout["doc_box"]
    box_w = doc_right - doc_left
    box_h = doc_bottom - doc_top
    if box_w <= 0 or box_h <= 0:
        return layout
    if box_w > width:
        box_w = float(width)
        doc_left = 0.0
    if box_h > height:
        box_h = float(height)
        doc_top = 0.0
    new_left = doc_left
    new_top = doc_top
    if new_left < 0:
        new_left = 0
    if new_top < 0:
        new_top = 0
    if new_left + box_w > width:
        new_left = max(0.0, width - box_w)
    if new_top + box_h > height:
        new_top = max(0.0, height - box_h)
    dx = new_left - doc_left
    dy = new_top - doc_top
    if dx == 0 and dy == 0:
        return layout
    updated = dict(layout)
    updated["doc_box"] = (new_left, new_top, new_left + box_w, new_top + box_h)
    updated["expected_start_x"] = layout["expected_start_x"] + dx
    return updated


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
        line_words = result["line_words"]

        boundary_word_index = len(target_texts["shema"].split()) if image_type == "m" else None
        expected_layout = estimate_layout(
            line_words,
            words,
            target_for_image,
            window_words=WINDOW_WORDS,
            max_skip=MAX_SKIP,
            score_threshold=0.4,
            boundary_word_index=boundary_word_index,
        )
        detected_bbox = _filtered_bbox(words)
        crop_bbox = None
        image_full = Image.open(path).convert("RGB")
        if expected_layout:
            expected_layout = _clamp_layout_to_image(expected_layout, image_full.width, image_full.height)
        if detected_bbox and expected_layout:
            doc_left, doc_top, doc_right, doc_bottom = expected_layout["doc_box"]
            crop_left = min(detected_bbox[0], int(doc_left))
            crop_top = min(detected_bbox[1], int(doc_top))
            crop_right = max(detected_bbox[2], int(doc_right))
            crop_bottom = max(detected_bbox[3], int(doc_bottom))
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(image_full.width, crop_right)
            crop_bottom = min(image_full.height, crop_bottom)
            if crop_right > crop_left and crop_bottom > crop_top:
                crop_bbox = (crop_left, crop_top, crop_right, crop_bottom)

        if crop_bbox:
            image_full = image_full.crop(crop_bbox)
        result = _ocr_image_pil(image_full, lang=LANG)
        ocr_text = strip_newlines(result["text"])
        words = result["words"]
        line_words = result["line_words"]

        pre = preprocess_image(Image.open(path))
        pre_out = preprocessed_dir / f"{path.stem}_pre.png"
        pre.save(pre_out)

        if words:
            image = image_full.copy()
            draw = ImageDraw.Draw(image)
            for w in words:
                draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)

            kept_bbox = _filtered_bbox(words)
            if kept_bbox:
                left, top, right, bottom = kept_bbox
                draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)

            if expected_layout:
                offset = (crop_bbox[0], crop_bbox[1]) if crop_bbox else (0, 0)
                _draw_layout(image, expected_layout, offset)
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
