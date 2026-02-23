#!/usr/bin/env python3
"""Shared output rendering stage for CLI and web entrypoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from draw_periodic_pattern import draw_periodic_pattern_for_pil


def _periodic_lag_bounds(avg_char_size: float | None) -> tuple[int, int | None]:
    min_lag_full_px = 8
    max_lag_full_px = None
    if isinstance(avg_char_size, (int, float)) and avg_char_size > 0:
        max_lag_full_px = max(6, int(round(1.85 * float(avg_char_size))))
    return min_lag_full_px, max_lag_full_px


def build_output_image(
    *,
    input_name: str,
    cropped: Image.Image,
    stripe_mask_continuum_debug: Image.Image,
    stripe_dark_mask: Image.Image,
    avg_char_size: float | None,
) -> tuple[Image.Image, dict[str, float | int | str]]:
    """Return final output image after periodic stripe post-processing."""
    min_lag_full_px, max_lag_full_px = _periodic_lag_bounds(avg_char_size)
    with tempfile.TemporaryDirectory(prefix="cropper-output-") as tmpdir:
        light_debug_out_path = Path(tmpdir) / "light_debug_output.png"
        periodic_meta = draw_periodic_pattern_for_pil(
            stripe_mask_continuum_debug,
            input_name=input_name,
            light_debug_image=cropped,
            light_debug_mask_image=stripe_dark_mask,
            light_debug_out_path=light_debug_out_path,
            min_lag_full_px=min_lag_full_px,
            max_lag_full_px=max_lag_full_px,
        )
        if periodic_meta.get("light_debug_output") and light_debug_out_path.exists():
            with Image.open(light_debug_out_path) as rendered:
                return rendered.convert("RGB").copy(), periodic_meta
    return cropped.copy(), periodic_meta


def build_output_image_from_crop_result(
    *,
    input_name: str,
    crop_result: dict[str, Any],
) -> tuple[Image.Image, dict[str, float | int | str]]:
    """Adapter over crop_image() result dict for shared use."""
    return build_output_image(
        input_name=input_name,
        cropped=crop_result["cropped"],
        stripe_mask_continuum_debug=crop_result["stripe_mask_continuum_debug"],
        stripe_dark_mask=crop_result["stripe_dark_mask"],
        avg_char_size=crop_result.get("avg_char_size"),
    )
