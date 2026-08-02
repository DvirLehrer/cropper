#!/usr/bin/env python3
"""Image loading for the STaM pipeline.

Why this module exists
----------------------
Two problems bit us on real user uploads, and both are decode-time problems:

1. **HEIC files with a .jpg extension.** iPhones produce HEIC; several upload
   paths rename the file to .jpg without transcoding it. ``cv2.imread`` returns
   ``None`` for those, so the image was silently skipped — the crop never ran
   and the failure looked like a detection failure. 17 of the 157 benchmark
   images are in this state.

2. **EXIF orientation applied inconsistently.** ``cv2.imread`` auto-applies the
   EXIF orientation tag; Google Vision ignores it. The pipeline therefore reads
   with ``IMREAD_IGNORE_ORIENTATION`` to stay in Vision's coordinate frame — but
   Ultralytics does its own ``cv2.imread`` internally when handed a *path*, and
   that decode *does* apply EXIF. For an image tagged orientation=6 the model
   polygon came back in a frame rotated 90° from the OCR boxes.

The fix for (2) is to decode exactly once here and pass the resulting array to
both YOLO and Vision, so there is only one coordinate frame by construction.
That also removes a redundant decode of every image.

``pillow_heif`` is an optional dependency: without it, HEIC files are reported
as unreadable with a clear message instead of silently vanishing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Magic bytes at offset 4 for the ISO-BMFF brands Apple uses for HEIC/HEIF.
_HEIF_BRANDS = {
    b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypheim",
    b"ftypheis", b"ftyphevm", b"ftyphevs", b"ftypmif1", b"ftypmsf1",
}

# Extensions we are willing to open. HEIC variants are included because real
# uploads use them; note that extension alone is not trusted — see sniff_format.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
            ".heic", ".heif"}

_heif_ready: bool | None = None


def _ensure_heif() -> bool:
    """Register the pillow-heif opener once. Returns False if unavailable."""
    global _heif_ready
    if _heif_ready is None:
        try:
            import pillow_heif  # noqa: WPS433 - optional dependency

            pillow_heif.register_heif_opener()
            _heif_ready = True
        except ImportError:
            _heif_ready = False
    return _heif_ready


def sniff_format(path: str | Path) -> str:
    """Identify a file by content, not by extension.

    Returns "heif", "jpeg", "png" or "other".
    """
    with open(path, "rb") as fh:
        head = fh.read(16)
    if len(head) >= 12 and head[4:12] in _HEIF_BRANDS:
        return "heif"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "other"


def true_suffix(path: str | Path) -> str:
    """The extension the file *should* have, based on its magic bytes."""
    return {"heif": ".heic", "jpeg": ".jpg", "png": ".png"}.get(
        sniff_format(path), Path(path).suffix.lower()
    )


def imread_any(path: str | Path) -> np.ndarray | None:
    """Decode any supported image to a BGR uint8 array.

    EXIF orientation is deliberately **not** applied, matching Google Vision,
    which ignores it. Every consumer in the pipeline must receive this same
    array so that model polygons and OCR boxes share one coordinate frame.

    Returns ``None`` if the file cannot be decoded.
    """
    path = str(path)

    if sniff_format(path) == "heif":
        if not _ensure_heif():
            print(f"  SKIP (HEIC, pillow-heif not installed): {path}")
            return None
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(path) as im:
                # No ImageOps.exif_transpose() here — see module docstring.
                rgb = im.convert("RGB")
                arr = np.asarray(rgb)
        except Exception as exc:  # corrupt or unsupported HEIC variant
            print(f"  SKIP (HEIC decode failed: {exc}): {path}")
            return None
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)


def list_images(directory: str | Path) -> list[str]:
    """Every decodable image in a directory, sorted, extension-tolerant."""
    directory = Path(directory)
    out = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMG_EXTS or sniff_format(p) != "other":
            out.append(str(p))
    return out
