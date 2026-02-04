#!/usr/bin/env python3
"""Generate and save margin slices for OCR debug."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
import pytesseract

from ocr_utils import OCR_CONFIG, preprocess_image, word_boxes_from_data


@dataclass(frozen=True)
class SliceResult:
    edge: str
    bbox: Tuple[int, int, int, int]
    words: List[Dict[str, Any]]
    annotated: Image.Image


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _median_char_size(words: List[Dict[str, Any]]) -> float:
    sizes = []
    for w in words:
        text = w.get("text", "")
        char_count = max(len(text), 1)
        sizes.append((w["x2"] - w["x1"]) / char_count)
    return _median(sizes)


def _clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def explore_margin_slices(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    words: List[Dict[str, Any]],
    lang: str,
) -> List[SliceResult]:
    if not words:
        return []
    mcs = _median_char_size(words)
    if mcs <= 0:
        return []

    line_h = mcs
    inside = 2.0 * line_h
    step = 3.0 * line_h
    left, top, right, bottom = bbox

    def _edge_bbox(edge: str, outside: float) -> Tuple[int, int, int, int]:
        if edge == "left":
            return (int(round(left - outside)), top, int(round(left + inside)), bottom)
        if edge == "right":
            return (int(round(right - inside)), top, int(round(right + outside)), bottom)
        if edge == "top":
            return (left, int(round(top - outside)), right, int(round(top + inside)))
        if edge == "bottom":
            return (left, int(round(bottom - inside)), right, int(round(bottom + outside)))
        raise ValueError(f"Unknown edge: {edge}")

    def _hit_new_band(
        edge: str,
        slice_words: List[Dict[str, Any]],
        origin: Tuple[int, int],
        prev_out: float,
        scale: float,
    ) -> bool:
        ox, oy = origin
        if edge == "left":
            threshold = left - prev_out
            return any((w["x1"] / scale + ox) < threshold for w in slice_words)
        if edge == "right":
            threshold = right + prev_out
            return any((w["x2"] / scale + ox) > threshold for w in slice_words)
        if edge == "top":
            threshold = top - prev_out
            return any((w["y1"] / scale + oy) < threshold for w in slice_words)
        if edge == "bottom":
            threshold = bottom + prev_out
            return any((w["y2"] / scale + oy) > threshold for w in slice_words)
        return False

    results: List[SliceResult] = []

    for edge in ("left", "right", "top", "bottom"):
        prev_out = 0.0
        outside = step
        last = None
        last_words: List[Dict[str, Any]] = []
        last_crop = None
        last_bbox = None
        last_scale = 1.0
        triggered = False

        while True:
            raw_bbox = _edge_bbox(edge, outside)
            clamped = _clamp_bbox(raw_bbox, image.width, image.height)
            if not clamped:
                break
            x1, y1, x2, y2 = clamped
            crop = image.crop((x1, y1, x2, y2))
            upscale = 1.5
            pre = preprocess_image(crop, upscale_factor=upscale, sharpen=False)
            data = pytesseract.image_to_data(
                pre,
                lang=lang,
                config=OCR_CONFIG,
                output_type=pytesseract.Output.DICT,
            )
            slice_words = word_boxes_from_data(data)
            if not slice_words:
                upscale = 2.0
                pre = preprocess_image(crop, upscale_factor=upscale, sharpen=True)
                data = pytesseract.image_to_data(
                    pre,
                    lang=lang,
                    config=OCR_CONFIG,
                    output_type=pytesseract.Output.DICT,
                )
                slice_words = word_boxes_from_data(data)

            last = (x1, y1, x2, y2)
            last_words = slice_words
            last_crop = crop
            last_bbox = clamped
            last_scale = upscale

            if not _hit_new_band(edge, slice_words, (x1, y1), prev_out, upscale):
                break
            if not triggered and prev_out == 0.0:
                triggered = True

            if edge == "left" and x1 <= 0:
                break
            if edge == "right" and x2 >= image.width:
                break
            if edge == "top" and y1 <= 0:
                break
            if edge == "bottom" and y2 >= image.height:
                break

            prev_out = outside
            outside += step

        if not triggered or not last or last_crop is None or last_bbox is None:
            continue

        x1, y1, x2, y2 = last_bbox
        abs_words: List[Dict[str, Any]] = []
        for w in last_words:
            ax1 = int(round(w["x1"] / last_scale)) + last_bbox[0]
            ay1 = int(round(w["y1"] / last_scale)) + last_bbox[1]
            ax2 = int(round(w["x2"] / last_scale)) + last_bbox[0]
            ay2 = int(round(w["y2"] / last_scale)) + last_bbox[1]
            abs_words.append(
                {
                    "text": w["text"],
                    "x1": ax1,
                    "y1": ay1,
                    "x2": ax2,
                    "y2": ay2,
                }
            )

        new_words: List[Dict[str, Any]] = []
        for w in abs_words:
            if edge == "left" and w["x1"] < left:
                new_words.append(w)
            elif edge == "right" and w["x2"] > right:
                new_words.append(w)
            elif edge == "top" and w["y1"] < top:
                new_words.append(w)
            elif edge == "bottom" and w["y2"] > bottom:
                new_words.append(w)

        if not new_words:
            continue

        annotated = last_crop.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for w in new_words:
            x1 = w["x1"] - last_bbox[0]
            y1 = w["y1"] - last_bbox[1]
            x2 = w["x2"] - last_bbox[0]
            y2 = w["y2"] - last_bbox[1]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)

        results.append(SliceResult(edge=edge, bbox=last_bbox, words=new_words, annotated=annotated))

    return results
