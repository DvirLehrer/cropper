#!/usr/bin/env python3
"""OCR execution helpers for PIL images."""

from __future__ import annotations

from typing import Any, Dict

import pytesseract
from PIL import Image

from ocr_utils import preprocess_image


def _ocr_image_pil(
    image: Image.Image,
    lang: str,
    min_boxes_retry: int = 50,
    allow_rotate: bool = True,
) -> Dict[str, Any]:
    def _run_ocr(
        pil_image: Image.Image,
        *,
        upscale_factor: float = 1.0,
        sharpen: bool = False,
    ) -> Dict[str, Any]:
        pre = preprocess_image(pil_image, upscale_factor=upscale_factor, sharpen=sharpen)
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
            "text": text.strip(),
            "words": words,
            "line_bboxes": line_bboxes,
            "line_words": line_words,
            "preprocessed": pre,
        }

    used_image = image
    result = _run_ocr(image)
    if allow_rotate and len(result["words"]) < min_boxes_retry:
        rotated = image.rotate(90, expand=True)
        rotated_result = _run_ocr(rotated)
        if len(rotated_result["words"]) > len(result["words"]):
            used_image = rotated
            result = rotated_result
        elif len(result["words"]) == 0 and len(rotated_result["words"]) == 0:
            used_image = rotated
            result = rotated_result

    if len(result["words"]) == 0:
        boosted = _run_ocr(used_image, upscale_factor=2.0, sharpen=True)
        if len(boosted["words"]) > len(result["words"]):
            result = boosted

    result["image"] = "in-memory"
    result["image_pil"] = used_image
    return result


def _ocr_image_pil_sparse_merge(
    image: Image.Image,
    lang: str,
) -> Dict[str, Any]:
    base = _ocr_image_pil(image, lang=lang, allow_rotate=False)
    pre = base["preprocessed"]
    data = pytesseract.image_to_data(
        pre,
        lang=lang,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    from ocr_utils import word_boxes_from_data, text_from_words

    sparse_words = word_boxes_from_data(data)

    def _overlaps(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        x1 = max(a["x1"], b["x1"])
        y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"])
        y2 = min(a["y2"], b["y2"])
        if x2 <= x1 or y2 <= y1:
            return False
        inter = (x2 - x1) * (y2 - y1)
        area = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        if area <= 0:
            return False
        return (inter / area) > 0.3

    merged = list(base["words"])
    for w in sparse_words:
        if any(_overlaps(w, k) for k in merged):
            continue
        merged.append(w)

    text, line_bboxes, line_words = text_from_words(merged)
    base.update(
        {
            "text": text.strip(),
            "words": merged,
            "line_bboxes": line_bboxes,
            "line_words": line_words,
        }
    )
    return base
