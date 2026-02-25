#!/usr/bin/env python3
"""Single segmented configuration for cropper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    benchmark_dir: Path = Path("benchmark")
    types_csv: Path = Path("text_type.csv")
    debug_dir: Path = Path("debug_images")
    ocr_text_dir: Path = Path("ocr_text")
    preprocessed_dir: Path = Path("preprocessed")
    cropped_dir: Path = Path("cropped")
    text_dir: Path = Path("text")


@dataclass(frozen=True)
class OcrConfig:
    lang: str = "heb"
    tesseract_config: str = "--psm 4"


@dataclass(frozen=True)
class CropServiceConfig:
    enable_line_warp: bool = True
    apply_line_correction: bool = True
    apply_crop_denoise: bool = True
    crop_denoise_size: int = 3
    work_max_pixels: int = int(os.environ.get("CROP_WORK_MAX_PIXELS", "2500000"))
    post_crop_stripes_work_max_pixels: int = int(
        os.environ.get("POST_CROP_STRIPES_WORK_MAX_PIXELS", "0")
    )
    post_crop_stripes_fast_bg_max_dim: int = int(
        os.environ.get("POST_CROP_STRIPES_FAST_BG_MAX_DIM", "1400")
    )


@dataclass(frozen=True)
class PeriodicPatternConfig:
    mask_thr: int = 250
    flat_range_max: int = 8
    debug_white_band_half_px: int = 5
    debug_fill_tile_size_px: int = 56
    debug_fill_sample_stride_px: int = 4
    abort_min_energy_strength: float = 0.85
    abort_min_row_edge_strength: float = 0.40
    abort_min_mean_coverage_strength: float = 0.40


@dataclass(frozen=True)
class LightingConfig:
    min_blur_radius: int = 12
    max_blur_radius: int = 72
    blur_size_frac: float = 0.045
    divide_gain: int = 168
    norm_blend_alpha: float = 0.28
    dark_mask_percentile: float = 0.20
    mask_vertical_expand_percentile: float = 0.30
    contrast_cutoff: float = 0.02
    lighten_gain: float = 1.22
    lighten_bias: int = 14
    mask_fill_mean_window_px: int = 16
    mask_fill_noise_amplitude: int = 2


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig = PathsConfig()
    ocr: OcrConfig = OcrConfig()
    crop_service: CropServiceConfig = CropServiceConfig()
    periodic: PeriodicPatternConfig = PeriodicPatternConfig()
    lighting: LightingConfig = LightingConfig()


settings = Settings()
