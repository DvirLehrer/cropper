#!/usr/bin/env python3
"""Debug best-sequence selection for specific mezuzah OCR outputs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageFilter, ImageOps
import pytesseract

from cropper_config import BENCHMARK_DIR, LANG, TEXT_DIR, TYPES_CSV
from target_texts import load_target_texts


NON_HEBREW_RE = re.compile(r"[^\u0590-\u05FF]")


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


def _best_word_segment_distance(
    line_text: str, target_words: List[str], length_tolerance: int = 5
) -> Tuple[int, str]:
    line_text = line_text.strip()
    if not line_text:
        return 0, ""
    if not target_words:
        return _levenshtein(line_text, ""), ""
    n = len(target_words)
    word_lens = [len(w) for w in target_words]
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + word_lens[i]

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


def _line_text_and_meta(words: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    ordered = sorted(words, key=lambda w: -w["x2"])
    texts = [w["text"] for w in ordered]
    meta = []
    for t in texts:
        if not t:
            continue
        if NON_HEBREW_RE.search(t):
            meta.append(f"{t}*")
        else:
            meta.append(t)
    return " ".join(texts), meta


def _display_hebrew(text: str) -> str:
    return text[::-1]


def _load_types(csv_path: Path) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["type"] for row in reader if row.get("filename")}


def _ocr_words(path: Path) -> List[Dict[str, Any]]:
    image = Image.open(path)
    pre = _preprocess_image(image)
    data = pytesseract.image_to_data(
        pre,
        lang=LANG,
        config="--psm 4",
        output_type=pytesseract.Output.DICT,
    )
    return _word_boxes_from_data(data)


def main() -> None:
    type_map = _load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)
    targets = {
        "m": target_texts["m"],
        "s": target_texts["shema"],
        "v": target_texts["vehaya"],
        "k": target_texts["kadesh"],
        "p": target_texts["peter"],
    }
    target_words_map = {k: [w for w in v.split() if w] for k, v in targets.items()}

    for name in ("mezuza2.jpeg", "mezuza3.jpg", "mezuza5.jpeg"):
        path = BENCHMARK_DIR / name
        if not path.exists():
            print(f"{name}: missing")
            continue
        image_type = type_map.get(name)
        target_words = target_words_map[image_type]
        words = _ocr_words(path)
        lines = _cluster_lines(words)
        scored = []
        print(f"\n{name}")
        for idx, line in enumerate(lines, start=1):
            line_text, meta = _line_text_and_meta(line)
            dist, segment = _best_word_segment_distance(line_text, target_words)
            score = 1.0 - (dist / max(len(segment), 1))
            scored.append((score, idx, line_text, segment, meta))
            print(f"  line {idx}: score={score:.4f} dist={dist}")
            print(f"    ocr: {_display_hebrew(line_text)}")
            print(f"    target: {_display_hebrew(segment)}")
            if meta:
                print(f"    words: {_display_hebrew(' '.join(meta))}")
        top = sorted(scored, key=lambda r: r[0], reverse=True)[:5]
        print("  top5:")
        for score, idx, line_text, segment, _ in top:
            print(
                f"    line {idx}: score={score:.4f} "
                f"ocr={_display_hebrew(line_text)} "
                f"target={_display_hebrew(segment)}"
            )


if __name__ == "__main__":
    main()
