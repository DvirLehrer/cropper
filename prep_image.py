#! /usr/bin/env python3
"""
Single-image STaM pre-processing pipeline.

Takes one mezuzah/tefillin photo plus its Google-OCR character boxes, fixes the
orientation, and crops to the text region for a downstream analysis system, writing a
debug image after every step so each stage can be inspected and tuned independently:

    01_original     - input (+ overlay of all OCR char boxes)
    02_orientation  - global rotation fix (90/180/270 flips + fine skew, one rotation)
    03_cropped      - blank outside the text's convex-hull polygon, crop to its bbox

The character boxes come from `google_ocr.py` (which calls Google Vision). We read the
saved `*_char_boxes.json` instead of re-OCRing, and propagate every box's coordinates
through the rotation, so iterating on the geometry costs no API calls.

Usage:
    python prep_image.py --image images/sample/<name>.jpg \
        [--char-boxes test_output/<name>_char_boxes.json] \
        [--out-dir test_output/debug] [--run-ocr]
"""

import os
import re
import json
import argparse

import cv2
import numpy as np


def contains_hebrew(text: str) -> bool:
    """True if the string contains a Hebrew codepoint (U+0590..U+05FF)."""
    return bool(re.search(r"[֐-׿]", text or ""))


# ---- I/O ------------------------------------------------------------------
def load_boxes(json_path: str):
    """Load char boxes as (N,4,2) float32 vertices + parallel list of chars.

    Returns (boxes, chars, hebrew_mask) where hebrew_mask[i] is True for Hebrew glyphs.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes, chars, hebrew = [], [], []
    for b in data.get("boxes", []):
        verts = b.get("vertices") or []
        if len(verts) < 4:
            continue
        quad = np.array([[v["x"], v["y"]] for v in verts[:4]], dtype=np.float32)
        boxes.append(quad)
        chars.append(b.get("char", ""))
        hebrew.append(contains_hebrew(b.get("char", "")))

    if not boxes:
        return np.zeros((0, 4, 2), np.float32), [], np.zeros((0,), bool)
    return np.stack(boxes), chars, np.array(hebrew, dtype=bool)


def imread_boxframe(path: str):
    """Read an image as BGR in the *raw* pixel frame, ignoring EXIF orientation.

    Google Vision emits box coordinates against the raw (un-rotated) pixels, but
    cv2.imread auto-applies EXIF orientation -- so for photos with an orientation
    tag (e.g. phone shots, tag 6 = rotate 90) the default load is rotated 90 from
    the boxes. IMREAD_IGNORE_ORIENTATION keeps pixels in the boxes' frame; the
    orientation step then straightens everything from the box geometry alone.
    """
    return cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)


def save_step(out_dir: str, name: str, img: np.ndarray):
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img)
    print(f"  saved {path}")


def draw_boxes(img: np.ndarray, boxes: np.ndarray, color=(0, 0, 255), thickness=2):
    """Return a copy of img with each quad outlined."""
    vis = img.copy()
    for quad in boxes:
        cv2.polylines(vis, [quad.astype(np.int32)], True, color, thickness)
    return vis


# ---- geometric transforms -------------------------------------------------
def apply_affine_to_boxes(M: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to (N,4,2) vertices."""
    if len(boxes) == 0:
        return boxes
    pts = boxes.reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    out = (M @ np.hstack([pts, ones]).T).T
    return out.reshape(-1, 4, 2).astype(np.float32)


def estimate_orientation(boxes: np.ndarray, hebrew_mask: np.ndarray) -> float:
    """Global text angle in degrees from the boxes' reading-direction vectors.

    Google Vision orders each box's vertices relative to the *detected reading
    orientation*, so v[1]-v[0] is the per-glyph reading direction in pixel space.
    The circular mean of these vectors (sum unit vectors, then atan2) gives one
    angle that captures both a gross 90/180/270 flip and the residual skew.
    """
    use = boxes[hebrew_mask] if hebrew_mask.any() else boxes
    if len(use) == 0:
        return 0.0
    d = use[:, 1, :] - use[:, 0, :]              # (M,2) top-edge vectors
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    u = d / norms                                # unit vectors
    sx, sy = float(u[:, 0].sum()), float(u[:, 1].sum())
    return float(np.degrees(np.arctan2(sy, sx)))


def rotate_bound(img: np.ndarray, angle_deg: float):
    """Rotate around center, expanding the canvas so nothing is clipped.

    Returns (rotated_image, 2x3_matrix). Calling getRotationMatrix2D with
    angle=text_angle maps the reading direction back to horizontal.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def text_hull(boxes: np.ndarray, hebrew_mask: np.ndarray, img_shape):
    """Convex hull (K,2 float32) of the text region's box corners.

    This is the *true* polygon used to mask away everything outside the text.
    Falls back to all boxes when there are no Hebrew glyphs. Returns None when
    there aren't enough points to form a polygon.
    """
    use = boxes[hebrew_mask] if hebrew_mask.any() else boxes
    pts = use.reshape(-1, 2).astype(np.int32)
    if len(pts) < 4:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    h, w = img_shape[:2]
    hull[:, 0] = np.clip(hull[:, 0], 0, w - 1)
    hull[:, 1] = np.clip(hull[:, 1], 0, h - 1)
    return hull


def mask_and_crop(img: np.ndarray, polygon: np.ndarray, fill: int = 255):
    """Blank everything outside the polygon, then crop to the polygon's bounding box."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    out = np.full_like(img, fill)
    out[mask == 255] = img[mask == 255]
    x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
    return out[y:y + h, x:x + w]


# ---- pipeline -------------------------------------------------------------
def run_pipeline(image_path: str, json_path: str, out_dir: str, max_pixels: int = 1_000_000):
    img = imread_boxframe(image_path)
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")
    boxes, chars, hebrew = load_boxes(json_path)
    print(f"Loaded {len(boxes)} boxes ({int(hebrew.sum())} Hebrew) from {json_path}")

    # Optionally downscale to cap total pixels (big speedup on multi-MP photos).
    # Boxes live in the full-res frame, so scale them by the same factor.
    h, w = img.shape[:2]
    if max_pixels and h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        boxes = boxes * scale
        print(f"  downscaled {w}x{h} -> {img.shape[1]}x{img.shape[0]} (scale {scale:.3f})")

    os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: original ---
    print("Step 1: original")
    save_step(out_dir, "01_original.png", img)
    save_step(out_dir, "01_original_overlay.png", draw_boxes(img, boxes))

    # --- Step 2: orientation ---
    print("Step 2: orientation")
    angle = estimate_orientation(boxes, hebrew)
    print(f"  detected text angle: {angle:.2f} deg")
    img2, A = rotate_bound(img, angle)
    boxes2 = apply_affine_to_boxes(A, boxes)
    save_step(out_dir, "02_orientation.png", img2)
    ov2 = draw_boxes(img2, boxes2)
    cv2.putText(ov2, f"angle={angle:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    save_step(out_dir, "02_orientation_overlay.png", ov2)

    # --- Step 3: crop to the text polygon ---
    print("Step 3: crop to polygon")
    hull = text_hull(boxes2, hebrew, img2.shape)
    if hull is None:
        print("  no usable hull; skipping crop")
        return
    # Overlay: green = text polygon we crop/mask to.
    mask_ov = draw_boxes(img2, boxes2)
    cv2.polylines(mask_ov, [hull.astype(np.int32)], True, (0, 255, 0), 2)
    save_step(out_dir, "03_crop_polygon.png", mask_ov)

    save_step(out_dir, "03_cropped.png", mask_and_crop(img2, hull))

    print("Done.")


def main():
    ap = argparse.ArgumentParser(description="Single-image STaM prep pipeline with per-step debug output.")
    ap.add_argument("--image", required=True, help="Path to one input image.")
    ap.add_argument("--char-boxes", default=None,
                    help="Path to <base>_char_boxes.json (default: test_output/<base>_char_boxes.json).")
    ap.add_argument("--out-dir", default="test_output/debug",
                    help="Debug output root; images go to <out-dir>/<base>/.")
    ap.add_argument("--run-ocr", action="store_true",
                    help="If the boxes JSON is missing, run google_ocr.detect_text to create it (needs Vision creds).")
    ap.add_argument("--max-pixels", type=int, default=1_000_000,
                    help="Downscale the image so total pixels <= this before processing (0 = no downscale).")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.image))[0]
    json_path = args.char_boxes or os.path.join("test_output", f"{base}_char_boxes.json")

    if not os.path.exists(json_path):
        if args.run_ocr:
            print(f"Boxes JSON not found; running OCR for {args.image} ...")
            from google_ocr import detect_text
            detect_text(args.image, output_dir="test_output", dataset_root=None)
        else:
            raise SystemExit(
                f"Char boxes not found: {json_path}\n"
                f"Run OCR first:  python google_ocr.py --image {args.image}\n"
                f"or pass --run-ocr to generate it here."
            )

    out_dir = os.path.join(args.out_dir, base)
    run_pipeline(args.image, json_path, out_dir, max_pixels=args.max_pixels)


if __name__ == "__main__":
    main()
