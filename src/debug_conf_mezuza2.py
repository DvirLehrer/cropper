#!/usr/bin/env python3
"""List OCR word confidences for mezuza2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import pytesseract

from cropper_config import BENCHMARK_DIR, LANG
from ocr_utils import preprocess_image


def _display_hebrew(text: str) -> str:
    return text[::-1]


def _resolve_image_path(stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = BENCHMARK_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found in benchmark/: {stem}")


def main() -> None:
    path = _resolve_image_path("mezuza2")
    image = Image.open(path)
    pre = preprocess_image(image)
    data = pytesseract.image_to_data(
        pre,
        lang=LANG,
        config="--psm 4",
        output_type=pytesseract.Output.DICT,
    )

    words: List[Dict[str, Any]] = []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        conf_raw = data.get("conf", [None])[i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = None
        words.append(
            {
                "text": text,
                "conf": conf,
                "x1": int(data["left"][i]),
                "y1": int(data["top"][i]),
                "x2": int(data["left"][i] + data["width"][i]),
                "y2": int(data["top"][i] + data["height"][i]),
                "line": int(data.get("line_num", [0])[i]),
            }
        )

    words.sort(key=lambda w: (w["line"], w["y1"], w["x1"]))
    lines: Dict[int, List[Dict[str, Any]]] = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)

    for line_idx in sorted(lines.keys()):
        print(f"line {line_idx:02d}:")
        line_words = sorted(lines[line_idx], key=lambda w: -w["x2"])
        for w in line_words:
            conf_str = "none" if w["conf"] is None else f"{w['conf']:.1f}"
            print(
                f"  conf={conf_str} "
                f"bbox=({w['x1']},{w['y1']},{w['x2']},{w['y2']}) "
                f"text={_display_hebrew(w['text'])}"
            )


if __name__ == "__main__":
    main()
