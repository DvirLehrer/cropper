#!/usr/bin/env python3
"""
crop_stam.py — Production CLI: crop STaM manuscript images to the text region.

For each input image:
  1. YOLO segmentation + Google Vision OCR run in parallel
  2. OCR boxes in text clusters adjacent to the model polygon are included
  3. Union of model polygon and the outline of those characters = final boundary
  4. Outside the boundary is filled with the sampled parchment background colour
  5. Image is cropped to the boundary bounding box and saved

Usage:
  python3 crop_stam.py --image images/benchmark/mezuza3.jpeg
  python3 crop_stam.py --images-dir images/benchmark --out-dir test_output/cropped
  python3 crop_stam.py --images-dir images/before_crop --model best.pt
"""
# Type annotations are evaluated lazily so that `X | None` and friends parse on
# Python 3.9, which is what `python3` still is on this machine.
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision
from ultralytics import YOLO

import deruling
import rectify
import texture
from crop_with_model import _predict, _tiled_predict, _imgsz_for, SPLIT_RATIO
from stam_io import imread_any, list_images

# ── tunables ───────────────────────────────────────────────────────────────────
# Every constant below can be overridden from the environment, so a sweep never
# requires editing this file:
#
#     TEXT_REACH_CHARS=1.2 python3 tools/bench.py crop --run reach12
#
# This exists because the alternative failed twice in one day: a run that was
# supposed to test a new value silently measured the old one, and the near
# identical numbers were nearly taken as evidence. bench.py records the active
# values with each run so a result can always be traced back to what produced it.

def _tune(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return type(default)(raw)


def active_settings() -> dict:
    """The tunables in force, for a run to record alongside its results."""
    return {n: globals()[n] for n in (
        "CONF", "OCR_MAX_PIXELS", "CLUSTER_GAP_PCT", "MODEL_REACH_PCT",
        "ROTATE_RATIO", "DESKEW_MIN_DEG", "MARGIN_CHARS", "TEXT_REACH_CHARS",
        "MODEL_LEASH_CHARS", "FLATTEN_BG", "ROUGH_MIN", "CLIP_TO_SHEET", "USE_TEXT_REGION",
        "RESEG_AFTER_DESKEW", "RESEG_MIN_DEG", "MODEL_MAX_RATIO",
        "MIN_CHAR_PX", "MAX_UPSCALE", "GRAIN_MIN", "DERULE",
    )} | {"RECTIFY": rectify.ENABLED, "RECTIFY_MIN_FAN": rectify.MIN_FAN}


# ── constants ──────────────────────────────────────────────────────────────────
IMG_EXTS        = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
MODEL_PATH      = 'best.pt'
OUT_DIR         = 'test_output/cropped'
CONF            = _tune('CONF', 0.25)
OCR_MAX_PIXELS  = _tune('OCR_MAX_PIXELS', 1_000_000)   # downscale before sending to Vision API
CLUSTER_GAP_PCT = _tune('CLUSTER_GAP_PCT', 0.04)        # dilation to bridge inter-character gaps (% of short dim)
MODEL_REACH_PCT = _tune('MODEL_REACH_PCT', 0.04)        # dilation on model mask to catch border boxes
EXTEND_WORK_PX  = 1000        # max dimension for connectedComponents work canvas
HEBREW_RANGE    = ('א', 'ת')
ROTATE_RATIO    = _tune('ROTATE_RATIO', 3.0)         # no-OCR fallback: rotate CCW when crop H/W exceeds this
DESKEW_MIN_DEG  = _tune('DESKEW_MIN_DEG', 0.5)         # below this the image is already straight; don't resample
MARGIN_CHARS    = _tune('MARGIN_CHARS', 0.0)         # crop margin, in median character heights.
                              # Held at 0 until the text-region change below is
                              # measured on its own; it has never been isolated.
# Outline the accepted characters rather than taking their convex hull.
#
# Off, and the story is worth keeping. A convex hull lets one stray character
# box drag the outline out and swallow the triangle in between, which is what
# filled the crops of slanted parchment with table, so outlining the characters
# was worth +3.7 points of text when it was introduced.
#
# Straightening the image before segmentation then removed the cause: text that
# has been levelled has a hull that is a tight rectangle, with no triangles to
# swallow. Measured with that in place, the hull is the better of the two —
# -0.9 text but +4.7 on the under-20 rule and half the errors — because
# outlining the characters fills the gaps between the lines with background and
# hands the engine a page cut into stripes.
#
# A workaround that outlived the problem it worked around.
USE_TEXT_REGION = _tune('USE_TEXT_REGION', False)
TEXT_REACH_CHARS = _tune('TEXT_REACH_CHARS', 0.6)        # dilation around each character when outlining
                              # the text region, in character heights
# Limiting how far the model polygon may reach past the recognised text looked
# obvious — a quarter of the median rotate crop is area with no *detected* ink
# near it — and it was a clear regression: perspective 93.8% -> 70.5% text
# recovered, rotate 69.0% -> 63.2%, and three times slower. The empty-looking
# area was not junk. It held text too faint or too shadowed for an Otsu
# threshold to call ink, and the model was right to keep it. The measurement was
# wrong, not the pipeline. Left here at 0 as a record of a road already walked.
MODEL_LEASH_CHARS = _tune('MODEL_LEASH_CHARS', 0.0)
# Background flattening pulls parchment luminance toward its mean to suppress
# texture. It predates the benchmark and has never been measured — it was added
# because it sounded right. The errors that dominate the images sitting just
# above the 20-error line are TOUCHING_LETTER_H, LETTER_SHAPE_CHANGED and
# ADDED_LETTER, which is exactly the signature of over-processing, so it is now
# a flag and gets tested like everything else.
FLATTEN_BG      = _tune('FLATTEN_BG', True)
# Confining the crop to the bright sheet was tried and abandoned unmeasured, on
# the reasonable objection that if a brightness threshold could separate sheet
# from surface it would have been the first cropping algorithm anyone wrote —
# and this pipeline uses a trained model precisely because it cannot. It breaks
# on dark parchment, a pale table, a shadow across the sheet, an overexposed
# photo, or a scan on white paper. The code stays for reference; the default is
# off, and it should not be turned on without a measurement behind it.
CLIP_TO_SHEET   = _tune('CLIP_TO_SHEET', False)
# Re-run segmentation after straightening the image, so the model's low-resolution
# mask meets a sheet whose edges run with the pixel grid instead of across it.
# Costs one extra inference, and only on images slanted enough to matter.
RESEG_AFTER_DESKEW = _tune('RESEG_AFTER_DESKEW', True)
RESEG_MIN_DEG   = _tune('RESEG_MIN_DEG', 2.0)
# Discard the model polygon when it exceeds the text region by this factor.
# Measured: a sound detection sits near 1x, a failed one reached 6.4x.
MODEL_MAX_RATIO = _tune('MODEL_MAX_RATIO', 3.0)
# Smallest character the downstream classifier copes with, in pixels; crops
# below it are enlarged. 0 disables. See the note at the resize step.
# Parchment grain above which the surface is denoised, measured on the block of
# writing rather than the whole crop — over the whole crop the reading is of
# whatever the sheet was lying on, which put the rotate folder above the rough
# one and had half the benchmark paying for a filter it did not need.
#
# 8 rather than 5: at 5 the filter also fires on four perspective images and
# ruins them, costing that folder 13.1 points against 10.6 gained on roughness.
# At 8 no perspective image qualifies, the five roughest still do, and the
# overall figure is the best measured — 86.6% text recovered against 86.1%
# without any of this, with roughness up 6.5 points and nothing else worse.
GRAIN_MIN       = _tune('GRAIN_MIN', 8.0)
MIN_CHAR_PX     = _tune('MIN_CHAR_PX', 24.0)
MAX_UPSCALE     = _tune('MAX_UPSCALE', 3.0)
ROUGH_MIN       = _tune('ROUGH_MIN', 0.0)   # flatten only above this roughness;
                                            # 0 = flatten everything, as before
# Ruled-line suppression (deruling.py). The first ungated version was a large
# win where the parchment really is ruled — drawings 13.0% -> 26.1% of images
# under the 20-error line, roughness 44.4% -> 50.0% with nothing made worse —
# but a net loss overall, 32.5% -> 31.8%, because it also fired on unruled pages
# and lifted the letters instead. It is now gated conservatively: enough
# repetitions to establish a rhythm, a confident autocorrelation, a line that
# holds close to its own fitted curve, and a dip too shallow to be ink. That
# gives up roughly half the ruled images in exchange for leaving the rest
# untouched. Whether the trade pays is a benchmark question, not a taste one.
#
# Held off for now so the crop margin below can be measured on its own. Two
# changes in one run tell you the sum and nothing else — which is exactly how a
# whole day was lost earlier to a before/after that turned out to be the same
# code twice.
DERULE          = _tune('DERULE', False)

# Filled by crop_image with per-image diagnostics, for the benchmark harness to
# record. A module-level dict rather than a changed return type, so that every
# existing caller keeps working.
LAST_DIAG: dict = {}

# ── Vision API client (singleton) ──────────────────────────────────────────────
_vision_client = None

def _get_vision_client():
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


# ── OCR ────────────────────────────────────────────────────────────────────────
def run_ocr(img_source) -> list:
    """Call Google Vision document_text_detection.
    img_source: file path (str) or numpy BGR array (e.g. after rotation).
    Downscales to ≤OCR_MAX_PIXELS before sending.
    Returns list of {char, vertices} dicts (Hebrew chars only) in original pixel coords."""
    if isinstance(img_source, np.ndarray):
        raw = img_source
    else:
        raw = imread_any(img_source)
    if raw is None:
        return []
    h, w  = raw.shape[:2]
    scale = 1.0
    if h * w > OCR_MAX_PIXELS:
        scale = (OCR_MAX_PIXELS / (h * w)) ** 0.5
        raw   = cv2.resize(raw, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return []
    content = buf.tobytes()

    inv      = 1.0 / scale
    response = _get_vision_client().document_text_detection(
        image=vision.Image(content=content))
    if response.error.message:
        return []

    boxes = []
    fta   = response.full_text_annotation
    if fta and fta.pages:
        for page in fta.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        for sym in word.symbols:
                            ch = sym.text or ''
                            if not ch.strip():
                                continue
                            if not (HEBREW_RANGE[0] <= ch[0] <= HEBREW_RANGE[1]):
                                continue
                            verts = sym.bounding_box.vertices if sym.bounding_box else []
                            if not verts:
                                continue
                            boxes.append({'char': ch, 'vertices': [
                                {'x': int(round((v.x or 0) * inv)),
                                 'y': int(round((v.y or 0) * inv))} for v in verts]})
    return boxes


# ── OCR extension ──────────────────────────────────────────────────────────────
def adjacent_ocr_corners(boxes: list, model_mask: np.ndarray, H: int, W: int):
    """Group Hebrew OCR boxes into connected text clusters; return corners of all
    clusters that touch the model polygon.

    A 'cluster' is a set of boxes whose dilated footprints are connected (bridges
    inter-character gaps). If the model polygon touches any box in a cluster, the
    entire cluster is included — so the model capturing one end of a text line
    automatically pulls in the rest of the line.

    All mask operations run at ≤EXTEND_WORK_PX resolution for speed.
    Returns (Nx2 float32 array of corners in original coords, set of accepted indices).
    """
    if not boxes:
        return None, set()

    if model_mask is None:
        # No model polygon, so cluster adjacency has nothing to test against:
        # fall back to every Hebrew box found. Returning None here instead made
        # crop_image's `elif ocr_hull is not None` branch unreachable, so an
        # image the model missed was reported NO DETECTION even when Vision had
        # read it perfectly — 8 of the 29 benchmark rotate images did exactly
        # that, YOLO finding nothing while OCR returned 200-717 characters.
        pts = np.array([[v['x'], v['y']] for b in boxes for v in b.get('vertices', [])],
                       dtype=np.float32)
        return (pts if len(pts) else None), set(range(len(boxes)))

    sc       = min(1.0, EXTEND_WORK_PX / max(H, W))
    wW, wH   = max(1, round(W * sc)), max(1, round(H * sc))
    wm       = cv2.resize(model_mask, (wW, wH), interpolation=cv2.INTER_NEAREST)

    ocr_mask = np.zeros((wH, wW), dtype=np.uint8)
    for b in boxes:
        verts = b.get('vertices', [])
        if verts:
            pts = np.array([[int(round(v['x'] * sc)), int(round(v['y'] * sc))]
                            for v in verts], np.int32)
            cv2.fillPoly(ocr_mask, [pts], 255)

    gap_px = max(3, int(min(wH, wW) * CLUSTER_GAP_PCT))
    k_gap  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap_px+1, 2*gap_px+1))
    num_labels, labels = cv2.connectedComponents(cv2.dilate(ocr_mask, k_gap))

    reach_px    = max(3, int(min(wH, wW) * MODEL_REACH_PCT))
    k_reach     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*reach_px+1, 2*reach_px+1))
    model_reach = cv2.dilate(wm, k_reach)

    connected = set()
    for lbl in range(1, num_labels):
        if cv2.countNonZero(cv2.bitwise_and((labels == lbl).astype(np.uint8), model_reach)):
            connected.add(lbl)

    accepted, pts = set(), []
    for i, b in enumerate(boxes):
        for v in b.get('vertices', []):
            x = int(np.clip(round(v['x'] * sc), 0, wW-1))
            y = int(np.clip(round(v['y'] * sc), 0, wH-1))
            if labels[y, x] in connected:
                accepted.add(i)
                for vv in b['vertices']:
                    pts.append([vv['x'], vv['y']])
                break

    return (np.array(pts, dtype=np.float32) if pts else None), accepted


def text_angle(boxes: list) -> float | None:
    """Directed baseline angle of the writing, in degrees. None if no OCR.

    Depends on the OCR alone — the segmentation model plays no part — which is
    what allows the image to be straightened before the model ever sees it.
    """
    rad = []
    for b in boxes:
        v = b.get('vertices') or []
        if len(v) < 4:
            continue
        dx, dy = v[1]['x'] - v[0]['x'], v[1]['y'] - v[0]['y']
        if dx or dy:
            rad.append(math.atan2(dy, dx))
    if not rad:
        return None
    a = np.array(rad)
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def _rotate_about_centre(img: np.ndarray, deg: float, fill):
    """Rotate so nothing is lost. Returns (image, 2x3 matrix used)."""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += nW / 2.0 - W / 2.0
    M[1, 2] += nH / 2.0 - H / 2.0
    out = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return out, M


def _apply_to_boxes(boxes: list, M: np.ndarray) -> list:
    """Move OCR character quads into a rotated frame."""
    moved = []
    for b in boxes:
        verts = b.get('vertices') or []
        if not verts:
            moved.append(b)
            continue
        pts = np.array([[[v['x'], v['y']]] for v in verts], dtype=np.float32)
        out = cv2.transform(pts, M).reshape(-1, 2)
        moved.append({**b, 'vertices': [{'x': int(round(x)), 'y': int(round(y))}
                                        for x, y in out]})
    return moved


def parchment_mask(img: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    """The sheet the writing sits on, as a mask. None if it cannot be found.

    Why this exists: the model's mask comes back at a fraction of the image
    resolution and is upscaled, so its edges are a staircase aligned to the
    pixel grid. When the parchment lies at an angle — which is the whole of the
    rotate challenge — a staircase cannot follow a straight diagonal edge, and
    each step swallows a square of table. Deskewing afterwards straightens the
    writing and carries those squares along with it, which is why the corners of
    those crops are full of wood.

    Restricting the boundary by *brightness* rather than by proximity to
    recognised text is what makes this safe. Text the OCR missed is still text
    on a bright sheet, so it survives — the earlier attempt to leash the model
    to the recognised characters cost perspective 93.8% -> 70.5% for exactly
    that reason.

    `seed` marks pixels known to be on the sheet: the writing itself.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1400 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else gray
    seed_small = cv2.resize(seed, (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    if cv2.countNonZero(seed_small) < 20:
        return None

    # The sheet is whatever is brighter than the midpoint between the writing
    # and the surface it was photographed on. Otsu finds that split directly.
    thresh, sheet = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close over the writing so letters do not punch holes in their own sheet.
    span = max(3, int(min(small.shape) * 0.02))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*span+1, 2*span+1))
    sheet = cv2.morphologyEx(sheet, cv2.MORPH_CLOSE, k)

    n, labels = cv2.connectedComponents(sheet)
    if n <= 1:
        return None
    # Keep every component the writing actually touches: a torn or folded sheet
    # can read as more than one.
    keep = set(np.unique(labels[seed_small > 0])) - {0}
    if not keep:
        return None
    out = np.isin(labels, list(keep)).astype(np.uint8) * 255
    if scale < 1.0:
        out = cv2.resize(out, (gray.shape[1], gray.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return out


def parchment_roughness(bgr: np.ndarray) -> float:
    """How grainy the parchment is, on a scale where ~1.0 is visibly rough.

    Texture is measured only on the parchment — ink is excluded, since strokes
    are high-frequency by nature and would swamp the reading — as the spread of
    each pixel around its local neighbourhood. That is then divided by the
    image's own ink-to-parchment contrast, so a dim photo and a bright scan of
    the same skin give the same answer.

    Flattening exists to suppress this grain and does so well: on the rough
    parchment folder it is worth +16.5 points of text recovered. Everywhere else
    it costs — rotate -12.5, cropper -2.1, perspective -1.7 — because it is
    smoothing a surface that was not the problem. Hence measuring first.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1200 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    parchment = gray > thresh
    if parchment.sum() < 500:
        return 0.0

    g = gray.astype(np.float32)
    local = cv2.blur(g, (7, 7))
    detail = np.abs(g - local)[parchment]
    # Median absolute deviation, not the mean: a few specks or a stray hair
    # should not read as a rough surface.
    grain = float(np.median(detail)) * 1.4826

    ink = gray[gray <= thresh]
    contrast = float(np.median(gray[parchment]) - np.median(ink)) if ink.size else 0.0
    if contrast < 10:
        return 0.0
    return round(grain / contrast * 20.0, 3)


def _text_region(boxes: list, accepted: set, adj_pts, H: int, W: int):
    """Outline of the accepted characters — the text's own shape, not its hull.

    A convex hull was the obvious thing and it is wrong here. Convexity means a
    single stray box drags the outline all the way out to it *and* swallows the
    whole triangle in between. On a parchment lying at an angle that produces
    exactly the wedges of table and wall seen in the crops: measured on one
    rotate image, the hull reached 4.1% of the frame beyond the model polygon,
    and all 327 characters Vision returned had been accepted without a single
    filter.

    Dilating the character boxes instead lets the outline follow the writing.
    A misplaced box then costs its own footprint and nothing more.

    Falls back to the convex hull when there is nothing to dilate.
    """
    if adj_pts is None or not accepted:
        return None
    if not USE_TEXT_REGION:
        return cv2.convexHull(adj_pts).reshape(-1, 2)

    heights = []
    for i in accepted:
        ys = [v['y'] for v in boxes[i].get('vertices', [])]
        if len(ys) >= 2 and max(ys) > min(ys):
            heights.append(max(ys) - min(ys))
    if not heights:
        return cv2.convexHull(adj_pts).reshape(-1, 2)

    char_h = float(np.median(heights))
    # Half a character bridges the gaps between letters and between lines
    # without reaching off the page.
    reach = max(2, int(round(char_h * TEXT_REACH_CHARS)))

    mask = np.zeros((H, W), dtype=np.uint8)
    for i in accepted:
        verts = boxes[i].get('vertices', [])
        if verts:
            pts = np.array([[v['x'], v['y']] for v in verts], np.int32)
            cv2.fillPoly(mask, [pts], 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*reach+1, 2*reach+1))
    mask = cv2.dilate(mask, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.convexHull(adj_pts).reshape(-1, 2)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def _margin_px(boxes: list, H: int, W: int) -> int:
    """Crop margin in pixels, measured in letters rather than in pixels.

    A fixed pixel margin is wrong twice over: too tight on a 10 MP scan, and a
    quarter of the parchment on a small one. The natural unit is the height of
    the writing, which the OCR boxes give us for free. Falls back to a fraction
    of the image when there is no OCR to measure.
    """
    heights = []
    for b in boxes:
        ys = [v['y'] for v in b.get('vertices', [])]
        if len(ys) >= 2:
            h = max(ys) - min(ys)
            if h > 0:
                heights.append(h)
    if heights:
        char_h = float(np.median(heights))
        return int(round(min(char_h * MARGIN_CHARS, min(H, W) * 0.05)))
    return int(round(min(H, W) * 0.005))


# ── polygon union ──────────────────────────────────────────────────────────────
def union_polygon(H: int, W: int, poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    """Binary-mask OR of two polygons → outer contour in image coords."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly1.astype(np.int32)], 255)
    cv2.fillPoly(mask, [poly2.astype(np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return poly1
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


# ── core per-image function ────────────────────────────────────────────────────
def crop_image(model, img_path: str, out_dir: str, conf: float = CONF) -> bool:
    """Full pipeline for one image. Returns True on success."""
    LAST_DIAG.clear()
    img = imread_any(img_path)
    if img is None:
        print(f'  SKIP (unreadable): {img_path}')
        return False

    H, W  = img.shape[:2]

    # Segmentation + OCR in parallel.
    # Both branches receive the SAME decoded array, never the path: handing a
    # path to Ultralytics makes it run its own cv2.imread, which applies EXIF
    # orientation while we (and Google Vision) deliberately ignore it — for an
    # image tagged orientation=6 that put the model polygon in a frame rotated
    # 90° from the OCR boxes. Passing the array also avoids decoding the file
    # three times, and is the only way HEIC input reaches the model at all.
    def _seg(on, iw, ih):
        r = max(iw, ih) / max(1, min(iw, ih))
        poly = _predict(model, on, conf, _imgsz_for(iw, ih))
        if poly is not None and r > SPLIT_RATIO:
            span = (poly[:, 0].max() - poly[:, 0].min()) if iw >= ih \
                   else (poly[:, 1].max() - poly[:, 1].min())
            if span < max(iw, ih) * 0.70:
                poly = _tiled_predict(model, on, conf)
        elif poly is None and r > SPLIT_RATIO:
            poly = _tiled_predict(model, on, conf)
        return poly

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_seg = ex.submit(_seg, img, W, H)
        f_ocr = ex.submit(run_ocr, img)
        poly_model = f_seg.result()
        ocr_boxes  = f_ocr.result()

    # Straighten before segmenting again, when the writing is slanted.
    #
    # The model returns its mask at a fraction of the image resolution, so its
    # edges are a staircase on the pixel grid. A staircase sits neatly along an
    # axis-aligned sheet edge and badly along a diagonal one, where every step
    # swallows a square of whatever the parchment was lying on — which is why
    # the corners of the rotate crops are full of table.
    #
    # The angle comes from the OCR alone, so the image can be levelled first and
    # the model given a sheet whose edges run with the grid. The cost is that
    # segmentation can no longer overlap the Vision call, so it is paid only
    # when there is something to gain: 96% of scans measure within half a degree
    # of straight and take the parallel path unchanged.
    theta = text_angle(ocr_boxes)
    pre_rotated = False
    if (RESEG_AFTER_DESKEW and theta is not None
            and abs(theta) >= RESEG_MIN_DEG and ocr_boxes):
        bg_guess = tuple(int(c) for c in np.median(
            img.reshape(-1, 3)[::97], axis=0))
        rot, M = _rotate_about_centre(img, theta, bg_guess)
        rot_boxes = _apply_to_boxes(ocr_boxes, M)
        rH, rW = rot.shape[:2]
        rot_poly = _seg(rot, rW, rH)
        if rot_poly is not None:
            img, ocr_boxes, poly_model = rot, rot_boxes, rot_poly
            H, W = rH, rW
            theta, pre_rotated = 0.0, True

    model_mask = None
    if poly_model is not None:
        model_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(model_mask, [poly_model.astype(np.int32)], 255)

    # Extend OCR to adjacent text clusters, then wrap the accepted characters.
    adj_pts, accepted = adjacent_ocr_corners(ocr_boxes, model_mask, H, W)
    ocr_hull = _text_region(ocr_boxes, accepted, adj_pts, H, W)

    # Distrust a model polygon that dwarfs the writing it is supposed to bound.
    #
    # On a sound detection the two are comparable — one benchmark image measures
    # 13.3% of the frame for the model against 14.8% for the text, and the model
    # earns its keep by reaching slightly further and catching letters the OCR
    # missed. On a failed one the model claimed 82.6% of the frame while the
    # text occupied 13.0%, and the crop came back larger than the photo.
    #
    # Clipping the model back to the text on *every* image was tried and cost
    # perspective 93.8% -> 70.5%, because the area past the recognised
    # characters is usually real text. So the rule fires only at the extreme,
    # where no reading of the picture supports keeping it.
    if (poly_model is not None and ocr_hull is not None
            and MODEL_MAX_RATIO > 0 and model_mask is not None):
        text_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(text_mask, [ocr_hull.astype(np.int32)], 255)
        text_area = cv2.countNonZero(text_mask)
        model_area = cv2.countNonZero(model_mask)
        if text_area > 0 and model_area > MODEL_MAX_RATIO * text_area:
            print(f'  model discarded: {100*model_area/(H*W):.0f}% of frame vs '
                  f'{100*text_area/(H*W):.0f}% of text')
            poly_model, model_mask = None, None

    # Union polygon.
    #
    # The union takes the whole model polygon, including wherever it has spilled
    # onto a shadow, a table edge or a torn corner — and nothing downstream ever
    # takes it back. Measured on the benchmark, a quarter of the median rotate
    # crop is area with no writing anywhere near it, rising to 45%, against 1-2%
    # in the categories that score well. The model is there to catch text the
    # OCR missed, not to annex the desk, so its contribution is limited to a
    # neighbourhood of the text that was actually recognised.
    if poly_model is not None and ocr_hull is not None:
        if MODEL_LEASH_CHARS > 0:
            leash = _margin_px(ocr_boxes, H, W) * (MODEL_LEASH_CHARS / max(MARGIN_CHARS, 1e-6))
            leash = int(round(min(leash, min(H, W) * 0.25)))
            if leash > 0:
                near_text = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(near_text, [ocr_hull.astype(np.int32)], 255)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*leash+1, 2*leash+1))
                near_text = cv2.dilate(near_text, k)

                model_mask_clipped = cv2.bitwise_and(model_mask, near_text)
                cnts, _ = cv2.findContours(model_mask_clipped, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    poly_model = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        boundary = union_polygon(H, W, poly_model, ocr_hull)
    elif poly_model is not None:
        boundary = poly_model
    elif ocr_hull is not None:
        boundary = ocr_hull
    else:
        print(f'  NO DETECTION: {os.path.basename(img_path)}')
        return False

    # Sample parchment background colour from non-ink pixels inside model polygon
    ref_mask  = model_mask if model_mask is not None else np.zeros((H, W), dtype=np.uint8)
    if model_mask is None:
        cv2.fillPoly(ref_mask, [boundary.astype(np.int32)], 255)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inside_px = gray[ref_mask == 255]
    if inside_px.size >= 100:
        thresh, _ = cv2.threshold(inside_px.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_pixels = img[ref_mask == 255][inside_px > thresh]
        bg_color  = tuple(int(c) for c in bg_pixels.mean(axis=0)) if len(bg_pixels) else (255, 255, 255)
    else:
        bg_color = (255, 255, 255)

    # Mask and crop.
    #
    # The boundary is a hull through the outermost OCR character corners, so it
    # passes exactly along the edge of the last letter with nothing to spare. In
    # STaM that is not close enough: the tagin on שעטנז ג"ץ rise above the glyph
    # body and Vision's boxes are tight to the body, so the crowns fall outside
    # the hull and get sliced off — and a sliced crown reads downstream as a
    # changed or broken letter. Give the boundary a margin scaled to the writing
    # itself, so it holds for a thumbnail and a 10 MP scan alike.
    union_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(union_mask, [boundary.astype(np.int32)], 255)

    # Clip to the sheet, so nothing the model annexed from the table survives.
    if CLIP_TO_SHEET and ocr_hull is not None:
        seed = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(seed, [ocr_hull.astype(np.int32)], 255)
        sheet = parchment_mask(img, seed)
        if sheet is not None:
            clipped = cv2.bitwise_and(union_mask, sheet)
            # Only accept the clip if the sheet was actually found: losing most
            # of the region means the brightness split went wrong, and the
            # original boundary is the safer answer.
            if cv2.countNonZero(clipped) > 0.5 * cv2.countNonZero(union_mask):
                union_mask = clipped
                cnts, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    boundary = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                    union_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(union_mask, [boundary.astype(np.int32)], 255)

    margin = _margin_px(ocr_boxes, H, W)
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1))
        union_mask = cv2.dilate(union_mask, k)
        contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            boundary = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

    result              = np.full_like(img, bg_color)
    result[union_mask == 255] = img[union_mask == 255]

    # Deskew: rotate the text upright by the measured angle of the OCR characters.
    #
    # Vision returns each character as a quad whose first edge (v0->v1) runs along
    # the text baseline, so atan2 over that edge gives a *directed* angle — it
    # tells upright text from the same text upside down, which an undirected line
    # fit cannot. Over the 29 benchmark rotate images the per-character agreement
    # is near-perfect (circular variance < 0.004) and v0->v3 lands at +90° in
    # every one, confirming the ordering. Their angles span the full range
    # (7°, 35°, 72°, -53°, 112°), so no multiple of 90° could have straightened
    # them — which is why the old aspect-ratio rule scored worse than no crop.
    #
    # This subsumes that rule: a parchment photographed vertically simply has a
    # ~90° text angle and is rotated by exactly that, satisfying criterion 3 as
    # a special case rather than as a separate branch.
    # theta was computed above, before the optional re-segmentation; if the
    # image was straightened there it is now 0 and this step is a plain crop.
    if not pre_rotated:
        theta = text_angle(ocr_boxes)

    crop_M = None   # transform from the pre-crop frame into the cropped image
    spun   = False  # the 90° fallback fired, so the boxes no longer line up
    if theta is not None and abs(theta) >= DESKEW_MIN_DEG:
        # Rotate and crop in one warp: no full-size intermediate, and the corners
        # exposed by the rotation take the sampled parchment colour, not black.
        cx, cy     = float(boundary[:, 0].mean()), float(boundary[:, 1].mean())
        M          = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        pts        = cv2.transform(boundary.reshape(-1, 1, 2), M).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        M[0, 2]   -= x
        M[1, 2]   -= y
        cropped    = cv2.warpAffine(result, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
        crop_M     = M
    else:
        # Already straight (434 of 436 cached corpus scans are within 0.5°), or no
        # OCR to measure. Slice rather than warp so a flat scan is never resampled.
        x, y, w, h = cv2.boundingRect(boundary.astype(np.int32))
        cropped    = result[y:y+h, x:x+w]
        # Only with no text direction at all does shape remain the sole signal.
        if theta is None and h / max(w, 1) >= ROTATE_RATIO:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            spun = True

    # One transform from the original frame into the cropped image, so the
    # character boxes can be carried along whichever branch was taken above and
    # whatever happens below. Everything downstream that needs to know where the
    # letters are reads this rather than reconstructing it.
    box_H = np.eye(3)
    if crop_M is not None:
        box_H[:2, :] = crop_M
    else:
        box_H[0, 2], box_H[1, 2] = -x, -y

    # Undo the keystone of a page photographed at an angle.
    #
    # Deskew above rotates the writing upright, which is everything a rotation
    # can do; it cannot make converging text lines parallel. This reads the two
    # vanishing points off the character boxes and maps them back to infinity.
    # See rectify.py for why the writing is a better guide than the edge of the
    # parchment, and for the mirror-image failure this took some care to avoid.
    #
    # Gated hard. Across the benchmark 93% of pages measure under 2° of
    # convergence and are already flat, so for almost every image this is a
    # measurement and nothing more.
    if rectify.ENABLED and not spun:
        moved = rectify.apply_to_boxes(ocr_boxes, box_H)
        cropped, rect_H, rect_info = rectify.rectify(cropped, moved, fill=bg_color)
        LAST_DIAG['fan'] = rect_info.get('fan')
        LAST_DIAG['rectified'] = rect_info.get('rectified')
        # Which guard turned it down. Without this a refusal is indistinguishable
        # from a page that needed nothing, and the first run could not tell them
        # apart on the two images that mattered.
        LAST_DIAG['rect_reason'] = rect_info.get('reason')
        if rect_H is not None:
            box_H = rect_H @ box_H

    # Background contrast reduction: only on large, roughly square images.
    # Skipped for low-res (strokes too thin) and high-ratio (elongated strips).
    _ch, _cw = cropped.shape[:2]
    _rough = parchment_roughness(cropped) if FLATTEN_BG else 0.0
    if (FLATTEN_BG and _rough >= ROUGH_MIN
            and min(_ch, _cw) >= 500 and max(_ch, _cw) / min(_ch, _cw) <= 4):
        lab       = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
        l, a, b   = cv2.split(lab)
        thresh, _ = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask   = l >= min(255, int(thresh) + 20)
        if bg_mask.any():
            lf        = l.astype(np.float32)
            mean_bg   = float(lf[bg_mask].mean())
            lf[bg_mask] = mean_bg + (lf[bg_mask] - mean_bg) * 0.3
            l_new     = np.clip(lf, 0, 255).astype(np.uint8)
            cropped   = cv2.cvtColor(cv2.merge([l_new, a, b]), cv2.COLOR_LAB2BGR)

    # Suppress the grain of the parchment, where there is grain to suppress.
    #
    # Not the same failure as small writing, and it needs the opposite remedy.
    # On the two worst images in the benchmark the characters are a comfortable
    # 40px, but the surface noise measures 7.8 and 6.1 grey levels against 2.8 on
    # a clean sheet — the engine reads the speckle as ink. An edge-preserving
    # smooth averages the speckle away while leaving the boundary of a letter,
    # which differs from its surroundings by a hundred levels, untouched.
    #
    # Gated on the measurement, so a clean sheet is never softened.
    if GRAIN_MIN > 0:
        # The boxes are in the frame of the original photograph; carry them
        # through everything done since, so they land on the letters in the
        # cropped image rather than a few hundred pixels away. Needed both to
        # measure the parchment rather than the table, and to protect the
        # writing during the filter itself.
        shifted = ([] if spun else rectify.apply_to_boxes(ocr_boxes, box_H))
        grain = texture.measure_grain(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
                                      boxes=shifted)
        LAST_DIAG['grain'] = grain["grain"]
        LAST_DIAG['denoised'] = grain["grain"] >= GRAIN_MIN
        if grain["grain"] >= GRAIN_MIN:
            cropped, _tex = texture.suppress(cropped, boxes=shifted)

    # Ruled-line suppression, deliberately after the grain has been taken out.
    #
    # A scored line is a faint continuous groove. On a speckled sheet it has to
    # be found among thousands of similar-amplitude specks, which is what made
    # the detector unreliable: the autocorrelation that recovers the ruling
    # period competes with the noise, and the per-column tracking wanders. Once
    # the surface is clean the line stands alone, the period is unambiguous, and
    # the track holds.
    #
    # So the order matters, and it is the opposite of what it was.
    if DERULE:
        cropped, _rule_info = deruling.remove(cropped)

    # Upscale when the writing is too small for the downstream classifier.
    #
    # Google Vision reads characters twelve pixels tall without difficulty; the
    # engine that follows was trained on letter crops and needs more. Measured
    # across the benchmark, text recovery is flat at 98% for characters of 20px
    # and above, drops to 86% between 14 and 20, and to 56% below that — 24 of
    # the 157 images sit under the line.
    #
    # Enlarging adds no detail, so this is not about making the letters clearer.
    # It is about handing the next stage something inside the size range it can
    # work with at all.
    if MIN_CHAR_PX > 0 and ocr_boxes:
        heights = [max(v['y'] for v in b['vertices']) - min(v['y'] for v in b['vertices'])
                   for b in ocr_boxes if len(b.get('vertices', [])) >= 2]
        heights = [h for h in heights if h > 0]
        if heights:
            char_px = float(np.median(heights))
            if 0 < char_px < MIN_CHAR_PX:
                factor = min(MIN_CHAR_PX / char_px, MAX_UPSCALE)
                ch, cw = cropped.shape[:2]
                cropped = cv2.resize(cropped, (int(cw * factor), int(ch * factor)),
                                     interpolation=cv2.INTER_CUBIC)

    os.makedirs(out_dir, exist_ok=True)
    stem     = Path(img_path).stem
    # Always write JPEG: cv2.imwrite cannot encode .heic, so echoing the input
    # extension would silently produce nothing for HEIC uploads.
    out_path = os.path.join(out_dir, f'{stem}_cropped.jpg')
    cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # How much writing did the crop leave behind?
    #
    # This is the one failure the cropper is squarely responsible for, and it is
    # free to measure: Vision already reported every character it found, in the
    # same coordinate frame as the crop rectangle, so the characters that fall
    # outside it are exactly the text that was cut away. No second pass, no API
    # call, no scoring engine.
    lost = kept = 0
    for b in ocr_boxes:
        verts = b.get('vertices') or []
        if not verts:
            continue
        cx = sum(v['x'] for v in verts) / len(verts)
        cy = sum(v['y'] for v in verts) / len(verts)
        if x <= cx <= x + w and y <= cy <= y + h:
            kept += 1
        else:
            lost += 1
    LAST_DIAG.update(chars_found=len(ocr_boxes), chars_kept=kept, chars_lost=lost,
                     chars_lost_pct=round(100 * lost / max(1, len(ocr_boxes)), 1))

    note = f"   LOST {lost} chars ({LAST_DIAG['chars_lost_pct']}%)" if lost else ""
    print(f'  {os.path.basename(img_path):50s} → {w}x{h}  {out_path}{note}')
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Crop STaM manuscript images to text region.')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',      help='Single input image')
    src.add_argument('--images-dir', help='Directory of input images')
    parser.add_argument('--model',   default=MODEL_PATH, help='YOLOv8 model weights')
    parser.add_argument('--out-dir', default=OUT_DIR,    help='Output directory')
    parser.add_argument('--conf',    type=float, default=CONF, help='Detection confidence threshold')
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.image:
        images = [args.image]
    else:
        images = list_images(args.images_dir)

    print(f'Processing {len(images)} image(s) with {args.model}\n')
    ok = sum(crop_image(model, p, args.out_dir, args.conf) for p in images)
    print(f'\nDone: {ok}/{len(images)} → {args.out_dir}/')


if __name__ == '__main__':
    main()
