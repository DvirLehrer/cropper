#!/usr/bin/env python3
"""Run Hebrew OCR crop pipeline on benchmark images."""

from __future__ import annotations

import time
from typing import Iterable

from PIL import Image

from cropper_config import (
    BENCHMARK_DIR,
    CROPPED_DIR,
    LANG,
    OCR_TEXT_DIR,
    TEXT_DIR,
    TYPES_CSV,
)
from ocr_utils import iter_images, levenshtein, load_types
from pipeline_crop_service import crop_image
from draw_periodic_pattern import draw_periodic_pattern_for_pil
from target_texts import load_target_texts, strip_newlines


def _target_text_for_image(image_type: str, target_texts: dict[str, str]) -> str | None:
    if image_type == "m":
        return target_texts["m"]
    if image_type in ("s", "v", "k", "p"):
        key = {"s": "shema", "v": "vehaya", "k": "kadesh", "p": "peter"}[image_type]
        return target_texts[key]
    return None


def _fmt_seconds(value: float) -> str:
    return f"{value:.3f}s"


def _format_top_items(items: Iterable[tuple[str, float]], limit: int = 4) -> str:
    top = sorted(items, key=lambda item: item[1], reverse=True)[:limit]
    if not top:
        return "none"
    return ", ".join(f"{name}={_fmt_seconds(sec)}" for name, sec in top)


def _timing_report(
    timing: dict[str, float],
    timing_detail: dict[str, float],
    *,
    crop_image_sec: float,
    periodic_sec: float,
    io_sec: float,
    total_sec: float,
) -> str:
    lines: list[str] = []
    ocr1 = float(timing.get("ocr1", 0.0))
    crop = float(timing.get("crop", 0.0))
    ocr2 = float(timing.get("ocr2", 0.0))
    debug = float(timing.get("debug", 0.0))
    crop_finalize = float(timing.get("crop_finalize", 0.0))
    post_crop_stripes = float(timing.get("post_crop_stripes", 0.0))
    crop_core_sec = ocr1 + crop + ocr2 + debug + crop_finalize
    crop_image_other = max(0.0, crop_image_sec - crop_core_sec - post_crop_stripes)
    other_sec = max(0.0, total_sec - crop_image_sec - periodic_sec - io_sec)
    lines.append(
        "timing "
        f"total={_fmt_seconds(total_sec)} "
        f"crop_core={_fmt_seconds(crop_core_sec)} "
        f"post_crop_stripes={_fmt_seconds(post_crop_stripes)} "
        f"crop_other={_fmt_seconds(crop_image_other)} "
        f"periodic={_fmt_seconds(periodic_sec)} "
        f"io={_fmt_seconds(io_sec)} "
        f"other={_fmt_seconds(other_sec)}"
    )
    lines.append(
        "stage "
        f"ocr1={_fmt_seconds(ocr1)} "
        f"crop={_fmt_seconds(crop)} "
        f"ocr2={_fmt_seconds(ocr2)} "
        f"debug={_fmt_seconds(debug)} "
        f"crop_finalize={_fmt_seconds(crop_finalize)} "
        f"post_crop_stripes={_fmt_seconds(post_crop_stripes)}"
    )

    group_specs = [
        ("stage1_", "detail_ocr_stage1"),
        ("stage2_", "detail_ocr_stage2"),
        ("crop_warp_", "detail_warp"),
        ("crop_tilt_", "detail_tilt"),
        ("post_crop_stripes_", "detail_post_crop_stripes"),
    ]
    used_keys: set[str] = set()
    for prefix, label in group_specs:
        grouped = [(k[len(prefix) :], float(v)) for k, v in timing_detail.items() if k.startswith(prefix)]
        if grouped:
            used_keys.update(k for k in timing_detail if k.startswith(prefix))
            grouped_map = {name: sec for name, sec in grouped}
            if "sparse_total" in grouped_map:
                subtotal = grouped_map["sparse_total"] + grouped_map.get("transform", 0.0)
            else:
                subtotal = sum(sec for _, sec in grouped)
            lines.append(f"{label}={_fmt_seconds(subtotal)} ({_format_top_items(grouped)})")

    remaining = [(k, float(v)) for k, v in timing_detail.items() if k not in used_keys]
    if remaining:
        lines.append(f"detail_other ({_format_top_items(remaining, limit=6)})")

    return "\n  ".join(lines)


def main() -> None:
    if not BENCHMARK_DIR.exists():
        raise SystemExit(f"Benchmark dir not found: {BENCHMARK_DIR}")

    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)
    OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)

    for path in iter_images(BENCHMARK_DIR):
        t_total_start = time.perf_counter()
        image_type = type_map.get(path.name)
        if image_type is None:
            print(f"[skip] {path.name} unknown type")
            continue
        target_for_image = _target_text_for_image(image_type, target_texts)
        if target_for_image is None:
            print(f"[skip] {path.name} unknown type")
            continue

        target_chars = max(len(strip_newlines(target_for_image)), 1)
        image_full = Image.open(path).convert("RGB")

        t_crop_start = time.perf_counter()
        result = crop_image(
            image_full,
            lang=LANG,
            target_chars=target_chars,
            debug_path=None,
        )
        crop_image_sec = time.perf_counter() - t_crop_start

        t_periodic_start = time.perf_counter()
        avg_char_size = result.get("avg_char_size")
        min_lag_full_px = 8
        max_lag_full_px = None
        if isinstance(avg_char_size, (int, float)) and avg_char_size > 0:
            max_lag_full_px = max(6, int(round(1.85 * float(avg_char_size))))
        output_crop_path = CROPPED_DIR / f"{path.stem}_crop.png"
        periodic_meta = draw_periodic_pattern_for_pil(
            result["stripe_mask_continuum_debug"],
            input_name=path.name,
            light_debug_image=result["cropped"],
            light_debug_mask_image=result["stripe_dark_mask"],
            light_debug_out_path=output_crop_path,
            min_lag_full_px=min_lag_full_px,
            max_lag_full_px=max_lag_full_px,
        )
        periodic_sec = time.perf_counter() - t_periodic_start

        t_io_start = time.perf_counter()
        cropped = result["cropped"]
        if not periodic_meta.get("light_debug_output"):
            cropped.save(output_crop_path)
        (OCR_TEXT_DIR / f"{path.stem}.txt").write_text(result["text"], encoding="utf-8")
        io_sec = time.perf_counter() - t_io_start
        total_sec = time.perf_counter() - t_total_start

        ocr_text = result["ocr_text"]
        if image_type == "m":
            quality_line = f"type={image_type} distance={levenshtein(ocr_text, target_texts['m'])}"
        else:
            names = ("shema", "vehaya", "kadesh", "peter")
            distances = {name: levenshtein(ocr_text, target_texts[name]) for name in names}
            guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
            quality_line = f"type={image_type} guess={guess_name} distance={guess_distance}"

        correction = result["correction"]
        periodic_line = (
            "periodic "
            f"lag={int(periodic_meta['lag'])} "
            f"corr={float(periodic_meta['corr']):.3f} "
            f"peaks={int(periodic_meta['peaks'])} "
            f"spacing_cons={float(periodic_meta['spacing_cons']):.3f} "
            f"strength={float(periodic_meta['strength']):.3f}"
        )
        timing_line = _timing_report(
            result["timing"],
            result.get("timing_detail", {}),
            crop_image_sec=crop_image_sec,
            periodic_sec=periodic_sec,
            io_sec=io_sec,
            total_sec=total_sec,
        )
        print(
            f"[{path.name}] {quality_line}\n"
            f"  crop {cropped.width}x{cropped.height} area={result['crop_area']} "
            f"target_chars={target_chars} px_per_char={result['px_per_char']:.1f}\n"
            f"  correction mode={correction.mode} mean_abs={correction.mean_abs:.5f} "
            f"std={correction.std:.5f} resid_mean={correction.resid_mean:.3f} "
            f"resid_std={correction.resid_std:.3f} curve_std={correction.curve_std:.3f}\n"
            f"  {periodic_line}\n"
            f"  {timing_line}"
        )


if __name__ == "__main__":
    main()
