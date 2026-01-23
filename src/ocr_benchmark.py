#!/usr/bin/env python3
"""
Run Hebrew OCR on all images in benchmark/ and print text + first/last word boxes.

Dependencies:
  - pytesseract
  - pillow
  - Tesseract OCR with Hebrew language data installed (lang=heb)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import pytesseract


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


def _reverse_lines(text: str) -> str:
    pattern = re.compile(r"([^\r\n]*)(\r?\n|\r|$)")
    return "".join(m.group(1)[::-1] + m.group(2) for m in pattern.finditer(text))


def _load_types(csv_path: Path) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["type"] for row in reader if row.get("filename")}


def _read_text_file(root: Path, target_name: str) -> str:
    direct = root / target_name
    if direct.exists():
        return direct.read_text(encoding="utf-8")
    for entry in root.iterdir():
        if entry.is_file() and entry.name.strip() == target_name:
            return entry.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Text file not found: {target_name}")


def _load_target_text(root: Path) -> str:
    shema = _read_text_file(root, "shema")
    vehaya = _read_text_file(root, "vehaya")
    return shema + vehaya


def _strip_newlines(text: str) -> str:
    return text.replace("\n", "").replace("\r", "")


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
    parser = argparse.ArgumentParser(description="OCR benchmark images (Hebrew).")
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark",
        help="Path to benchmark images folder (default: benchmark).",
    )
    parser.add_argument("--lang", default="heb", help="Tesseract language (default: heb).")
    parser.add_argument(
        "--types-csv",
        default="text_type.csv",
        help="CSV mapping filename -> type (default: text_type.csv).",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug_images",
        help="Output directory for debug images (default: debug_images).",
    )
    parser.add_argument(
        "--ocr-text-dir",
        default="ocr_text",
        help="Output directory for per-image OCR text (default: ocr_text).",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="preprocessed",
        help="Output directory for preprocessed images (default: preprocessed).",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        raise SystemExit(f"Benchmark dir not found: {benchmark_dir}")

    type_map = _load_types(Path(args.types_csv))
    root = Path.cwd()
    target_text = _strip_newlines(_load_target_text(root))
    shema_text = _strip_newlines(_read_text_file(root, "shema"))
    vehaya_text = _strip_newlines(_read_text_file(root, "vehaya"))
    kadesh_text = _strip_newlines(_read_text_file(root, "kadesh"))
    peter_text = _strip_newlines(_read_text_file(root, "peter"))

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    ocr_text_dir = Path(args.ocr_text_dir)
    ocr_text_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = Path(args.preprocessed_dir)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_images(benchmark_dir):
        image_type = type_map.get(path.name)
        result = ocr_image(path, lang=args.lang)
        ocr_text = _strip_newlines(result["text"])
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
        # print(f"image: {result['image']}")
        # print("text:")
        # print(_reverse_lines(result["text"]))
        # print(f"first_word: {result['first_word']}")
        # print(f"last_word: {result['last_word']}")
        if image_type == "m":
            distance = _levenshtein(ocr_text, target_text)
            print(f"type: {image_type}  distance: {distance}")
        else:
            distances = {
                "shema": _levenshtein(ocr_text, shema_text),
                "vehaya": _levenshtein(ocr_text, vehaya_text),
                "kadesh": _levenshtein(ocr_text, kadesh_text),
                "peter": _levenshtein(ocr_text, peter_text),
            }
            guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
            print(f"type: {image_type}  guess: {guess_name}  distance: {guess_distance}")
        # print("=" * 40)


if __name__ == "__main__":
    main()
