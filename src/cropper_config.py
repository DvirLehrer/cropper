#!/usr/bin/env python3
"""Configuration for cropper benchmark scripts."""

from __future__ import annotations

import os
from pathlib import Path


BENCHMARK_DIR = Path("benchmark")
LANG = "heb"
TYPES_CSV = Path("text_type.csv")
DEBUG_DIR = Path("debug_images")
OCR_TEXT_DIR = Path("ocr_text")
PREPROCESSED_DIR = Path("preprocessed")
CROPPED_DIR = Path("cropped")
TEXT_DIR = Path("text")

# Fast path for post-crop stripes normalization.
# 0 disables downscale blur optimization (quality mode).
POST_CROP_STRIPES_FAST_BG_MAX_DIM = int(os.environ.get("POST_CROP_STRIPES_FAST_BG_MAX_DIM", "1400"))
