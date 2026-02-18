#!/usr/bin/env python3
"""Shared OCR helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageFilter, ImageOps
import pytesseract
from lighting_normalization import normalize_uneven_lighting

OCR_CONFIG = "--psm 4"


def preprocess_image(
    image: Image.Image,
    *,
    upscale_factor: float = 1.0,
    sharpen: bool = False,
) -> Image.Image:
    if upscale_factor != 1.0:
        new_w = max(1, int(round(image.width * upscale_factor)))
        new_h = max(1, int(round(image.height * upscale_factor)))
        image = image.resize((new_w, new_h), Image.LANCZOS)
    gray = ImageOps.grayscale(image)
    gray = normalize_uneven_lighting(gray)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gamma = 0.85
    gray = gray.point(lambda p: int(255 * ((p / 255) ** gamma)))
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    if sharpen:
        gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    return gray


def word_boxes_from_data(data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
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


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def cluster_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    heights = [w["y2"] - w["y1"] for w in words]
    line_height = median(heights)
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


def text_from_words(
    words: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    if not words:
        return "", [], []
    lines = cluster_lines(words)
    line_bboxes = []
    for line in lines:
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
    pre = preprocess_image(image)
    data = pytesseract.image_to_data(
        pre,
        lang=lang,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )
    words = word_boxes_from_data(data)
    text, line_bboxes, line_words = text_from_words(words)
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


def load_types(csv_path: Path) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        import csv

        reader = csv.DictReader(f)
        mapping: Dict[str, str] = {}
        for row in reader:
            filename = row.get("filename")
            if not filename:
                continue
            f_clean = filename.strip()
            f_norm = " ".join(f_clean.split())
            f_type = row.get("type")
            if not f_type:
                continue
            mapping[f_clean] = f_type
            mapping[f_norm] = f_type
        return mapping


def levenshtein(a: str, b: str) -> int:
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
