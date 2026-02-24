#!/usr/bin/env python3
"""Formatting/printing helpers for cropper pipeline logs."""

from __future__ import annotations

from typing import Any, Callable, Iterable


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


def target_chars_for_type(image_type: str, target_texts: dict[str, str], strip_newlines_fn: Callable[[str], str]) -> int:
    target_for_image = _target_text_for_image(image_type, target_texts)
    if target_for_image is None:
        return 0
    return max(len(strip_newlines_fn(target_for_image)), 1)


def print_skip_unknown(path_name: str) -> None:
    print(f"[skip] {path_name} unknown type")


def print_image_report(
    *,
    path_name: str,
    image_type: str,
    target_chars: int,
    cropped_width: int,
    cropped_height: int,
    result: dict[str, Any],
    periodic_meta: dict[str, float | int | str],
    target_texts: dict[str, str],
    levenshtein_fn: Callable[[str, str], int],
    crop_image_sec: float,
    periodic_sec: float,
    io_sec: float,
    total_sec: float,
) -> None:
    ocr_text = str(result["ocr_text"])
    if image_type == "m":
        quality_line = f"type={image_type} distance={levenshtein_fn(ocr_text, target_texts['m'])}"
    else:
        names = ("shema", "vehaya", "kadesh", "peter")
        distances = {name: levenshtein_fn(ocr_text, target_texts[name]) for name in names}
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
        f"[{path_name}] {quality_line}\n"
        f"  crop {cropped_width}x{cropped_height} area={result['crop_area']} "
        f"target_chars={target_chars} px_per_char={result['px_per_char']:.1f}\n"
        f"  correction mode={correction.mode} mean_abs={correction.mean_abs:.5f} "
        f"std={correction.std:.5f} resid_mean={correction.resid_mean:.3f} "
        f"resid_std={correction.resid_std:.3f} curve_std={correction.curve_std:.3f}\n"
        f"  {periodic_line}\n"
        f"  {timing_line}"
    )
