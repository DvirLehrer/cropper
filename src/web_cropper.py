#!/usr/bin/env python3
"""OCR-based crop service helpers for the local web app."""

from __future__ import annotations

from io import BytesIO
from typing import Callable, Optional

from PIL import Image

from pipeline_crop_service import crop_image


def crop_uploaded_image_bytes(
    image_bytes: bytes,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[bytes, str, dict]:
    """Crop uploaded image bytes and return (JPEG bytes, mime type, metadata)."""
    with Image.open(BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")

    result = crop_image(image, progress_cb=progress_cb)
    out = BytesIO()
    result["cropped"].save(out, format="JPEG", quality=95, optimize=True)
    metadata = {
        "timing": result["timing"],
        "timing_detail": result.get("timing_detail", {}),
        "correction_mode": result["correction"].mode,
        "crop_area": result["crop_area"],
        "px_per_char": result["px_per_char"],
        "size": {
            "width": result["cropped"].width,
            "height": result["cropped"].height,
        },
    }
    return out.getvalue(), "image/jpeg", metadata
