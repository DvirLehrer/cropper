#!/usr/bin/env python3
"""Run Hebrew OCR crop pipeline on benchmark images."""

from __future__ import annotations

import time

from PIL import Image

from config import settings
from core.ocr_utils import iter_images, levenshtein, load_types
from core.run_logging import (
    print_image_report,
    print_skip_unknown,
    target_chars_for_type,
)
from output import build_output_image_from_crop_result
from cropper_pipeline import crop_image
from core.settings_texts import load_target_texts, strip_newlines


def main() -> None:
    if not settings.paths.benchmark_dir.exists():
        raise SystemExit(f"Benchmark dir not found: {settings.paths.benchmark_dir}")

    type_map = load_types(settings.paths.types_csv)
    target_texts = load_target_texts(settings.paths.text_dir)
    settings.paths.ocr_text_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.cropped_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_images(settings.paths.benchmark_dir):
        t_total_start = time.perf_counter()
        image_type = type_map.get(path.name)
        if image_type is None:
            print_skip_unknown(path.name)
            continue
        target_chars = target_chars_for_type(image_type, target_texts, strip_newlines)
        if target_chars <= 0:
            print_skip_unknown(path.name)
            continue

        image_full = Image.open(path).convert("RGB")

        t_crop_start = time.perf_counter()
        result = crop_image(
            image_full,
            lang=settings.ocr.lang,
            target_chars=target_chars,
            debug_path=None,
        )
        crop_image_sec = time.perf_counter() - t_crop_start

        t_periodic_start = time.perf_counter()
        cropped_output, periodic_meta = build_output_image_from_crop_result(
            input_name=path.name,
            crop_result=result,
        )
        output_crop_path = settings.paths.cropped_dir / f"{path.stem}_crop.png"
        periodic_sec = time.perf_counter() - t_periodic_start

        t_io_start = time.perf_counter()
        cropped = cropped_output
        cropped.save(output_crop_path)
        (settings.paths.ocr_text_dir / f"{path.stem}.txt").write_text(result["text"], encoding="utf-8")
        io_sec = time.perf_counter() - t_io_start
        total_sec = time.perf_counter() - t_total_start

        print_image_report(
            path_name=path.name,
            image_type=image_type,
            target_chars=target_chars,
            cropped_width=cropped.width,
            cropped_height=cropped.height,
            result=result,
            periodic_meta=periodic_meta,
            target_texts=target_texts,
            levenshtein_fn=levenshtein,
            crop_image_sec=crop_image_sec,
            periodic_sec=periodic_sec,
            io_sec=io_sec,
            total_sec=total_sec,
        )


if __name__ == "__main__":
    main()
