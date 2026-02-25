#!/usr/bin/env python3
"""Run Hebrew OCR crop pipeline on benchmark images."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR crop benchmark and timing logs.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Randomly sample N images (0 means process all images).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used with --sample-size.",
    )
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=Path("segment_performance.csv"),
        help="Path to write per-image segment timing CSV.",
    )
    return parser.parse_args()


def _sample_paths(paths: list[Path], sample_size: int, seed: int | None) -> list[Path]:
    if sample_size <= 0 or sample_size >= len(paths):
        return paths
    rng = random.Random(seed)
    selected = rng.sample(paths, sample_size)
    return sorted(selected)


def _write_segments_csv(rows: list[dict[str, str | float | int]], csv_path: Path) -> None:
    if not rows:
        return
    base_cols = [
        "image",
        "type",
        "image_dimensions",
        "image_magnitude_px",
        "target_chars",
        "total_sec",
        "crop_image_sec",
        "periodic_sec",
        "io_sec",
    ]
    dynamic_cols = sorted({key for row in rows for key in row.keys() if key not in base_cols})
    fieldnames = base_cols + dynamic_cols
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    if not settings.paths.benchmark_dir.exists():
        raise SystemExit(f"Benchmark dir not found: {settings.paths.benchmark_dir}")

    type_map = load_types(settings.paths.types_csv)
    target_texts = load_target_texts(settings.paths.text_dir)
    settings.paths.ocr_text_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.cropped_dir.mkdir(parents=True, exist_ok=True)
    segment_rows: list[dict[str, str | float | int]] = []

    paths = _sample_paths(
        iter_images(settings.paths.benchmark_dir),
        sample_size=args.sample_size,
        seed=args.seed,
    )

    for path in paths:
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
        row: dict[str, str | float | int] = {
            "image": path.name,
            "type": image_type,
            "image_dimensions": f"{image_full.width}x{image_full.height}",
            "image_magnitude_px": _format_area_magnitude(image_full.width, image_full.height),
            "target_chars": target_chars,
            "total_sec": total_sec,
            "crop_image_sec": crop_image_sec,
            "periodic_sec": periodic_sec,
            "io_sec": io_sec,
        }
        for key, value in result.get("timing", {}).items():
            row[f"timing_{key}"] = float(value)
        for key, value in result.get("timing_detail", {}).items():
            row[f"detail_{key}"] = float(value)
        segment_rows.append(row)

    _write_segments_csv(segment_rows, args.segments_csv)
    print(f"[segments_csv] {args.segments_csv} rows={len(segment_rows)}")


def _format_area_magnitude(width: int, height: int) -> str:
    area = int(width) * int(height)
    if area <= 0:
        return "0*10**0"
    exp = int(math.floor(math.log10(area)))
    mant = area / (10**exp)
    return f"{mant:.3g}*10**{exp}"


if __name__ == "__main__":
    main()
