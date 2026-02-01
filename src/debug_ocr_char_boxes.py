#!/usr/bin/env python3
"""Run first-round OCR and report character box counts."""

from __future__ import annotations

from PIL import Image
import pytesseract

from cropper_config import BENCHMARK_DIR, LANG
from ocr_utils import OCR_CONFIG, iter_images, preprocess_image

MIN_BOXES_RETRY = 50


def main() -> None:
    for path in iter_images(BENCHMARK_DIR):
        print(f"processing: {path.name}")
        image = Image.open(path)
        pre = preprocess_image(image)
        boxes = pytesseract.image_to_boxes(pre, lang=LANG, config=OCR_CONFIG)
        count = 0 if not boxes else len(boxes.strip().splitlines())
        chosen = "original"
        if count < MIN_BOXES_RETRY:
            rotated = pre.rotate(90, expand=True)
            rotated_boxes = pytesseract.image_to_boxes(rotated, lang=LANG, config=OCR_CONFIG)
            rotated_count = 0 if not rotated_boxes else len(rotated_boxes.strip().splitlines())
            if rotated_count > count:
                boxes = rotated_boxes
                count = rotated_count
                chosen = "rotated_ccw"
        if count < MIN_BOXES_RETRY:
            print(f"{path.name}: low box count ({count}), using {chosen}")
        print(f"{path.name}: {count}")


if __name__ == "__main__":
    main()
