#!/usr/bin/env python3
"""OCR-based crop service helpers for the local web app."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from pipeline_crop_service import crop_image


def crop_uploaded_image_bytes(image_bytes: bytes) -> tuple[bytes, str]:
    """Crop uploaded image bytes and return (JPEG bytes, mime type)."""
    with Image.open(BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")

    result = crop_image(image)
    out = BytesIO()
    result["cropped"].save(out, format="JPEG", quality=95, optimize=True)
    return out.getvalue(), "image/jpeg"
