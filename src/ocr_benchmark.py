#!/usr/bin/env python3
"""
Run Hebrew OCR on all images in benchmark/ and print text distance stats.

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


def _preprocess_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gamma = 0.85
    gray = gray.point(lambda p: int(255 * ((p / 255) ** gamma)))
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray


def _word_boxes_from_data(data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
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


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


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


def _text_from_words(
    words: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    if not words:
        return "", [], []
    lines = _cluster_lines(words)
    line_bboxes = []
    for line in lines:
        ordered = sorted(line, key=lambda w: -w["x2"])
        line_bboxes.append(
            {
                "x1": min(w["x1"] for w in line),
                "y1": min(w["y1"] for w in line),
                "x2": max(w["x2"] for w in line),
                "y2": max(w["y2"] for w in line),
            }
        )
    text = "\n".join(" ".join(w["text"] for w in sorted(line, key=lambda w: -w["x2"])) for line in lines)
    return text, line_bboxes, lines


def ocr_image(path: Path, lang: str) -> Dict[str, Any]:
    image = Image.open(path)
    pre = _preprocess_image(image)
    data = pytesseract.image_to_data(
        pre,
        lang=lang,
        config="--psm 4",
        output_type=pytesseract.Output.DICT,
    )
    words = _word_boxes_from_data(data)
    text, line_bboxes, line_words = _text_from_words(words)
    return {
        "image": str(path),
        "text": text.strip(),
        "words": words,
        "line_bboxes": line_bboxes,
        "line_words": line_words,
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


def _best_word_segment_distance_with_prefix(
    line_text: str,
    target_words: List[str],
    word_lens: List[int],
    prefix: List[int],
    length_tolerance: int = 5,
) -> Tuple[int, str]:
    line_text = line_text.strip()
    if not line_text:
        return 0, ""
    if not target_words:
        return _levenshtein(line_text, ""), ""
    n = len(target_words)

    def seg_len(i: int, j: int) -> int:
        if j < i:
            return 0
        return (prefix[j + 1] - prefix[i]) + (j - i)

    target_len = len(line_text)
    best_dist = 10**9
    best_segment = ""
    for i in range(n):
        j_min = i
        while j_min < n and seg_len(i, j_min) < target_len - length_tolerance:
            j_min += 1
        j_max = j_min
        while j_max < n and seg_len(i, j_max) <= target_len:
            j_max += 1
        for j in range(j_min, j_max):
            segment = " ".join(target_words[i : j + 1])
            dist = _levenshtein(line_text, segment)
            if dist < best_dist:
                best_dist = dist
                best_segment = segment
                if best_dist == 0:
                    return best_dist, best_segment
    return best_dist, best_segment


def _draw_best_sequences(
    image: Image.Image,
    line_bboxes: List[Dict[str, Any]],
    line_words: List[List[Dict[str, Any]]],
    target_text: str,
    image_name: str,
    top_k: int = 5,
    window_words: int = 6,
    max_skip: int = 6,
) -> None:
    target_words = [w for w in target_text.split() if w]
    word_lens = [len(w) for w in target_words]
    prefix = [0] * (len(word_lens) + 1)
    for i, l in enumerate(word_lens):
        prefix[i + 1] = prefix[i] + l
    scored = []
    for idx, words in enumerate(line_words):
        if not words or len(words) < window_words:
            continue
        ordered = sorted(words, key=lambda w: -w["x2"])
        best = None
        max_start = min(max_skip, max(len(ordered) - 1, 0))
        for start in range(0, max_start + 1):
            if start >= len(ordered):
                break
            end = min(len(ordered), start + window_words) - 1
            if end - start + 1 < window_words:
                continue
            ocr_sub = " ".join(w["text"] for w in ordered[start : end + 1])
            dist, segment = _best_word_segment_distance_with_prefix(
                ocr_sub, target_words, word_lens, prefix
            )
            denom = max(len(segment), 1)
            score = 1.0 - (dist / denom)
            if best is None or score > best[0]:
                best = (score, start, end, ocr_sub, segment, dist)
        if best is None:
            continue
        score, start, end, ocr_sub, segment, dist = best
        scored.append((score, idx, (start, end)))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    draw = ImageDraw.Draw(image)
    for _, idx, (i, j) in scored[:top_k]:
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

    type_map = _load_types(TYPES_CSV)
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
            for idx, line in enumerate(line_bboxes, start=1):
                draw.rectangle([line["x1"], line["y1"], line["x2"], line["y2"]], outline=(0, 128, 255), width=2)
            _draw_best_sequences(
                image,
                line_bboxes,
                line_words,
                target_for_image,
                image_name=path.name,
                top_k=5,
            )
            out_path = debug_dir / f"{path.stem}_debug.png"
            image.save(out_path)

        text_out = ocr_text_dir / f"{path.stem}.txt"
        text_out.write_text(result["text"], encoding="utf-8")

        if image_type == "m":
            target = target_texts["m"]
            distance = _levenshtein(ocr_text, target)
            print(f"type: {image_type}  distance: {distance}")
        else:
            names = ("shema", "vehaya", "kadesh", "peter")
            distances = {name: _levenshtein(ocr_text, target_texts[name]) for name in names}
            guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
            print(f"type: {image_type}  guess: {guess_name}  distance: {guess_distance}")


if __name__ == "__main__":
    main()
