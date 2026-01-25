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
from typing import Any, Dict, List, Tuple

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
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--psm 4",
        output_type=pytesseract.Output.DICT,
    )
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


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _estimate_char_width_height(words: List[Dict[str, Any]]) -> Tuple[float, float]:
    heights = [w["y2"] - w["y1"] for w in words]
    widths = []
    for w in words:
        length = max(len(w["text"]), 1)
        widths.append((w["x2"] - w["x1"]) / length)
    return _median(widths), _median(heights)


def _cluster_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    heights = [w["y2"] - w["y1"] for w in words]
    line_height = _median(heights)
    tol = line_height * 0.7 if line_height > 0 else 10
    sorted_words = sorted(words, key=lambda w: ((w["y1"] + w["y2"]) / 2.0, w["x1"]))
    lines: List[List[Dict[str, Any]]] = []
    for w in sorted_words:
        center = (w["y1"] + w["y2"]) / 2.0
        placed = False
        for line in lines:
            line_center = sum((lw["y1"] + lw["y2"]) / 2.0 for lw in line) / len(line)
            if abs(center - line_center) <= tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    return sorted(lines, key=lambda line: min(w["y1"] for w in line))


def _build_lines_from_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lines = _cluster_lines(words)
    result = []
    for line_words in lines:
        ordered = sorted(line_words, key=lambda w: -w["x2"])
        line_text = ""
        line_word_texts = []
        char_boxes = []
        for w in ordered:
            text = w["text"]
            if not text:
                continue
            line_word_texts.append(text)
            w_width = w["x2"] - w["x1"]
            if w_width <= 0:
                continue
            char_w = w_width / max(len(text), 1)
            for i, ch in enumerate(text):
                x2 = w["x2"] - (i * char_w)
                x1 = x2 - char_w
                char_boxes.append(
                    {
                        "char": ch,
                        "x1": x1,
                        "y1": w["y1"],
                        "x2": x2,
                        "y2": w["y2"],
                    }
                )
                line_text += ch
        if not line_text:
            continue
        line_text_words = " ".join(line_word_texts)
        bbox = {
            "x1": min(w["x1"] for w in line_words),
            "y1": min(w["y1"] for w in line_words),
            "x2": max(w["x2"] for w in line_words),
            "y2": max(w["y2"] for w in line_words),
        }
        result.append(
            {
                "text": line_text,
                "text_words": line_text_words,
                "char_boxes": char_boxes,
                "bbox": bbox,
            }
        )
    return result


def _best_window_match(
    line_text: str, target_text: str, window: int = 10
) -> Tuple[int, int, int, str, str]:
    if len(line_text) < window or len(target_text) < window:
        return 0, 0, _levenshtein(line_text, target_text), line_text, target_text[: len(line_text)]
    best = (0, 0, 10**9, "", "")
    target_windows = [target_text[i : i + window] for i in range(len(target_text) - window + 1)]
    for i in range(len(line_text) - window + 1):
        chunk = line_text[i : i + window]
        for j, tgt in enumerate(target_windows):
            dist = _levenshtein(chunk, tgt)
            if dist < best[2]:
                best = (i, j, dist, chunk, tgt)
    return best


def _best_sample_across_lines(
    lines: List[Dict[str, Any]],
    target_text: str,
    window: int = 10,
    threshold: int = 3,
) -> Tuple[int, int, int, int, str, str]:
    best = (-1, 0, 0, 10**9, "", "")
    for line_idx, line in enumerate(lines):
        line_text = line["text"].strip()
        if not line_text:
            continue
        ocr_idx, target_idx, dist, ocr_sample, target_sample = _best_window_match(
            line_text, target_text, window=window
        )
        if dist < best[3]:
            best = (line_idx, ocr_idx, target_idx, dist, ocr_sample, target_sample)
        if dist < threshold:
            return line_idx, ocr_idx, target_idx, dist, ocr_sample, target_sample
    return best


def _best_target_for_sample(sample: str, target_text: str) -> Tuple[int, int, str]:
    if not sample:
        return 0, _levenshtein(sample, target_text), target_text[: len(sample)]
    if len(target_text) < len(sample):
        return 0, _levenshtein(sample, target_text), target_text
    best_idx = 0
    best_dist = 10**9
    best_segment = target_text[: len(sample)]
    span = len(sample)
    for i in range(len(target_text) - span + 1):
        segment = target_text[i : i + span]
        dist = _levenshtein(sample, segment)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_segment = segment
            if best_dist == 0:
                break
    return best_idx, best_dist, best_segment


def _display_hebrew(text: str) -> str:
    return text[::-1]


def _draw_sample_border(
    image: Image.Image,
    lines: List[Dict[str, Any]],
    target_text: str,
    words: List[Dict[str, Any]],
    window: int = 10,
) -> None:
    if not lines or not words:
        return
    line_idx, ocr_idx, target_idx, dist, ocr_sample, target_sample = _best_sample_across_lines(
        lines, target_text, window=window
    )
    if line_idx < 0:
        return
    if line_idx >= len(lines):
        return
    line = lines[line_idx]
    char_boxes = line["char_boxes"]
    if not char_boxes:
        return
    start = ocr_idx
    end = min(ocr_idx + window, len(char_boxes))
    sample_boxes = char_boxes[start:end]
    if not sample_boxes:
        return
    sample_widths = [b["x2"] - b["x1"] for b in sample_boxes]
    sample_heights = [b["y2"] - b["y1"] for b in sample_boxes]
    sample_char_width = _median(sample_widths)
    sample_char_height = _median(sample_heights)
    sample_x1 = min(b["x1"] for b in sample_boxes)
    sample_x2 = max(b["x2"] for b in sample_boxes)
    sample_y1 = min(b["y1"] for b in sample_boxes)
    sample_y2 = max(b["y2"] for b in sample_boxes)
    print(f"sample: {_display_hebrew(ocr_sample)}")
    print(f"target: {_display_hebrew(target_sample)}")
    print(f"levenshtein: {dist}")
    sample2_line_idx = line_idx + 1
    if sample2_line_idx < len(lines):
        line2 = lines[sample2_line_idx]
        line2_text = line2["text"].strip()
        line2_boxes = line2["char_boxes"]
        if line2_text and line2_boxes:
            start = min(
                range(len(line2_boxes)),
                key=lambda i: abs(line2_boxes[i]["x2"] - sample_x2),
            )
            max_len = min(20, len(line2_text) - start)
            ocr_sample2 = line2_text[start : start + max_len]
            target_idx2, dist2, target_sample2 = _best_target_for_sample(
                ocr_sample2, target_text
            )
            print(f"sample2: {_display_hebrew(ocr_sample2)}")
            print(f"target2: {_display_hebrew(target_sample2)}")
            print(f"levenshtein2: {dist2}")
            line_len_chars = target_idx2 - (target_idx + window)
            print(f"line_len_chars: {line_len_chars}")
            end = min(start + max_len, len(line2_boxes))
            sample2_boxes = line2_boxes[start:end]
            if sample2_boxes:
                sample_width = sample_x2 - sample_x1
                sample_height = sample_y2 - sample_y1
                x2_2 = int(round(sample_x2))
                x1_2 = int(round(sample_x2 - (2 * sample_width)))
                y1_2 = int(round(min(b["y1"] for b in sample2_boxes)))
                y2_2 = int(round(y1_2 + sample_height))
                draw = ImageDraw.Draw(image)
                draw.rectangle([x1_2, y1_2, x2_2, y2_2], outline=(0, 128, 255), width=3)
    x1 = int(round(sample_x1))
    x2 = int(round(sample_x2))
    y1 = int(round(sample_y1))
    y2 = int(round(sample_y2))
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=3)

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
    text = pytesseract.image_to_string(pre, lang=lang, config="--psm 4")
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
        print(f"processing: {path.name}")
        image_type = type_map.get(path.name)
        result = ocr_image(path, lang=LANG)
        ocr_text = strip_newlines(result["text"])
        target_for_image = (
            target_text
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
        pre = _preprocess_image(Image.open(path))
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
            lines = _build_lines_from_words(words)
            if lines:
                _draw_sample_border(image, lines, target_for_image, words)
            out_path = debug_dir / f"{path.stem}_debug.png"
            image.save(out_path)
        text_out = ocr_text_dir / f"{path.stem}.txt"
        text_out.write_text(result["text"], encoding="utf-8")
        if image_type == "m":
            distance = _levenshtein(ocr_text, target_for_image)
            print(f"type: {image_type}  distance: {distance}")
        else:
            names = ("shema", "vehaya", "kadesh", "peter")
            distances = {name: _levenshtein(ocr_text, target_texts[name]) for name in names}
            guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
            print(f"type: {image_type}  guess: {guess_name}  distance: {guess_distance}")


if __name__ == "__main__":
    main()
