#!/usr/bin/env python3
"""
Run Hebrew OCR on all images in benchmark/ and print text distance stats.

Dependencies:
  - pytesseract
  - pillow
  - Tesseract OCR with Hebrew language data installed (lang=heb)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import bisect
import time

from PIL import Image, ImageDraw
import pytesseract

from cropper_config import (
    BENCHMARK_DIR,
    CROPPED_DIR,
    DEBUG_DIR,
    LANG,
    OCR_TEXT_DIR,
    PREPROCESSED_DIR,
    TEXT_DIR,
    TYPES_CSV,
)
from target_texts import load_target_texts, strip_newlines
from expected_layout import estimate_layout
from edge_align_shift import align_edges_shift_side
from ocr_utils import iter_images, levenshtein, load_types, preprocess_image

WINDOW_WORDS = 6
MAX_SKIP = 6
DRAW_EXPECTED_LAYOUT = False
STRIPE_HEIGHT_FRACS = (0.15, 0.25, 0.35, 0.5)
STRIPE_OVERLAP = 0.5
STRIPE_TOP_K = 2
STRIPE_PAD_FRAC = 0.08


def _stripe_roi_bbox(pre: Image.Image) -> Optional[tuple[int, int, int, int]]:
    if pre.mode != "L":
        pre = pre.convert("L")
    w, h = pre.size
    if w < 2 or h < 2:
        return None
    pixels = list(pre.getdata())
    row_edge = [0.0] * (h - 1)
    for y in range(h - 1):
        row_sum = 0
        idx0 = y * w
        idx1 = (y + 1) * w
        for x in range(w):
            row_sum += abs(pixels[idx1 + x] - pixels[idx0 + x])
        row_edge[y] = row_sum / w
    prefix = [0.0]
    for val in row_edge:
        prefix.append(prefix[-1] + val)

    stripes: List[tuple[float, int, int]] = []
    for frac in STRIPE_HEIGHT_FRACS:
        stripe_h = max(2, int(round(h * frac)))
        step = max(1, int(round(stripe_h * (1.0 - STRIPE_OVERLAP))))
        for y in range(0, h - stripe_h + 1, step):
            y2 = y + stripe_h
            score = prefix[y2 - 1] - prefix[y]
            stripes.append((score, y, y2))

    if not stripes:
        return None
    stripes.sort(key=lambda s: s[0], reverse=True)
    top = [s for s in stripes[:STRIPE_TOP_K] if s[0] > 0]
    if not top:
        return None
    y1 = min(s[1] for s in top)
    y2 = max(s[2] for s in top)
    pad = int(round(h * STRIPE_PAD_FRAC))
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    if y2 <= y1:
        return None
    return (0, y1, w, y2)


def _save_stripe_debug(image: Image.Image, bbox: Optional[tuple[int, int, int, int]], out_path: Path) -> None:
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if bbox:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 200), width=3)
        if y1 > 0:
            draw.rectangle([0, 0, base.width, y1], fill=(255, 0, 0, 60))
        if y2 < base.height:
            draw.rectangle([0, y2, base.width, base.height], fill=(255, 0, 0, 60))
    combined = Image.alpha_composite(base.convert("RGBA"), overlay)
    combined.save(out_path)




def _draw_layout(image: Image.Image, layout: Dict[str, Any], offset: tuple[int, int] = (0, 0)) -> None:
    draw = ImageDraw.Draw(image)
    dx, dy = offset
    doc_left, doc_top, doc_right, doc_bottom = layout["doc_box"]
    doc_left -= dx
    doc_right -= dx
    doc_top -= dy
    doc_bottom -= dy
    box_w = doc_right - doc_left
    box_h = doc_bottom - doc_top
    if box_w > image.width:
        doc_left = 0
        doc_right = image.width
    else:
        if doc_left < 0:
            doc_left = 0
            doc_right = box_w
        if doc_right > image.width:
            doc_right = image.width
            doc_left = doc_right - box_w
    if box_h > image.height:
        doc_top = 0
        doc_bottom = image.height
    else:
        if doc_top < 0:
            doc_top = 0
            doc_bottom = box_h
        if doc_bottom > image.height:
            doc_bottom = image.height
            doc_top = doc_bottom - box_h
    draw.rectangle([doc_left, doc_top, doc_right, doc_bottom], outline=(0, 255, 255), width=3)
    exp_x = int(round(layout["expected_start_x"] - dx))
    exp_x = max(0, min(exp_x, image.width - 1))
    draw.line([(exp_x, doc_top), (exp_x, doc_top + 20)], fill=(255, 0, 0), width=3)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _median_char_size(words: List[Dict[str, Any]]) -> float:
    sizes = []
    for w in words:
        text = w.get("text", "")
        char_count = max(len(text), 1)
        sizes.append((w["x2"] - w["x1"]) / char_count)
    return _median(sizes)


def _median_word_height(words: List[Dict[str, Any]]) -> float:
    heights = [(w["y2"] - w["y1"]) for w in words]
    return _median(heights)


def _box_edge_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    dx = 0.0
    if a["x2"] < b["x1"]:
        dx = b["x1"] - a["x2"]
    elif b["x2"] < a["x1"]:
        dx = a["x1"] - b["x2"]
    dy = 0.0
    if a["y2"] < b["y1"]:
        dy = b["y1"] - a["y2"]
    elif b["y2"] < a["y1"]:
        dy = a["y1"] - b["y2"]
    return (dx * dx + dy * dy) ** 0.5


def _vertical_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    return max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))


def _horizontal_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    return max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))


def _edge_flags(words: List[Dict[str, Any]]) -> Tuple[List[bool], List[bool], List[bool], List[bool]]:
    if not words:
        return [], [], [], []

    heights = [w["y2"] - w["y1"] for w in words]
    widths = [w["x2"] - w["x1"] for w in words]
    bin_h = _median(heights) or 10.0
    bin_w = _median(widths) or 10.0

    y_bins_x2: Dict[int, List[float]] = {}
    y_bins_x1: Dict[int, List[float]] = {}
    x_bins_y2: Dict[int, List[float]] = {}
    x_bins_y1: Dict[int, List[float]] = {}

    for w in words:
        y_start = int(w["y1"] // bin_h)
        y_end = int(w["y2"] // bin_h)
        for b in range(y_start, y_end + 1):
            y_bins_x2.setdefault(b, []).append(w["x2"])
            y_bins_x1.setdefault(b, []).append(w["x1"])

        x_start = int(w["x1"] // bin_w)
        x_end = int(w["x2"] // bin_w)
        for b in range(x_start, x_end + 1):
            x_bins_y2.setdefault(b, []).append(w["y2"])
            x_bins_y1.setdefault(b, []).append(w["y1"])

    for bins in (y_bins_x2, y_bins_x1, x_bins_y2, x_bins_y1):
        for key in bins:
            bins[key].sort()

    leftmost = [True] * len(words)
    rightmost = [True] * len(words)
    topmost = [True] * len(words)
    bottommost = [True] * len(words)

    for idx, w in enumerate(words):
        y_start = int(w["y1"] // bin_h)
        y_end = int(w["y2"] // bin_h)
        for b in range(y_start, y_end + 1):
            xs_left = y_bins_x2.get(b, [])
            if xs_left and xs_left[0] <= w["x1"]:
                # Any x2 <= w.x1 => not leftmost
                if bisect.bisect_right(xs_left, w["x1"]) > 0:
                    leftmost[idx] = False
            xs_right = y_bins_x1.get(b, [])
            if xs_right and xs_right[-1] >= w["x2"]:
                if bisect.bisect_left(xs_right, w["x2"]) < len(xs_right):
                    rightmost[idx] = False

        x_start = int(w["x1"] // bin_w)
        x_end = int(w["x2"] // bin_w)
        for b in range(x_start, x_end + 1):
            ys_top = x_bins_y2.get(b, [])
            if ys_top and ys_top[0] <= w["y1"]:
                if bisect.bisect_right(ys_top, w["y1"]) > 0:
                    topmost[idx] = False
            ys_bottom = x_bins_y1.get(b, [])
            if ys_bottom and ys_bottom[-1] >= w["y2"]:
                if bisect.bisect_left(ys_bottom, w["y2"]) < len(ys_bottom):
                    bottommost[idx] = False

    return leftmost, rightmost, topmost, bottommost


def _mad(xs: List[float], center: float) -> float:
    return _median([abs(x - center) for x in xs])


def _filter_boxes_by_char_size(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []
    sizes = []
    for w in words:
        text = w.get("text", "")
        char_count = max(len(text), 1)
        sizes.append((w["x2"] - w["x1"]) / char_count)
    median_size = _median(sizes)
    if median_size <= 0:
        return list(words)
    min_size = median_size / 3.0
    max_size = median_size * 3.0
    kept = [w for w, size in zip(words, sizes) if min_size <= size <= max_size]
    return kept if kept else list(words)


def _filter_outliers(xs: List[float], keep_lower: bool) -> List[float]:
    if not xs:
        return []
    med = _median(xs)
    mad = _median([abs(x - med) for x in xs])
    if mad <= 0:
        return list(xs)
    cutoff = 2.0 * mad
    if keep_lower:
        return [x for x in xs if x <= med + cutoff]
    return [x for x in xs if x >= med - cutoff]


def _next_candidates(xs: List[float], increasing: bool) -> List[List[float]]:
    if not xs:
        return []
    sorted_xs = sorted(xs, reverse=not increasing)
    next_map: Dict[float, float] = {}
    for i in range(len(sorted_xs) - 1):
        next_map[sorted_xs[i]] = sorted_xs[i + 1]
    return [[x, next_map[x]] if x in next_map else [x] for x in xs]


def _best_aligned_edge_deadzone(xs: List[float], tau: float, penalty_lambda: float) -> tuple[float, List[float]]:
    if not xs:
        return 0.0, []
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    if n == 1:
        return xs_sorted[0], xs_sorted

    med = _median(xs_sorted)
    s0 = 1.4826 * _mad(xs_sorted, med)
    if s0 <= 1e-6:
        return med, xs_sorted

    best_obj = None
    best_mu = xs_sorted[0]
    best_subset: List[float] = []

    for mu in xs_sorted:
        losses = []
        for x in xs_sorted:
            u = (x - mu) / s0
            excess = abs(u) - tau
            loss = (excess * excess) if excess > 0 else 0.0
            losses.append(loss)
        losses_sorted = sorted((loss, x) for loss, x in zip(losses, xs_sorted))
        prefix_loss = [0.0]
        for loss, _ in losses_sorted:
            prefix_loss.append(prefix_loss[-1] + loss)
        for k in range(1, n + 1):
            mean_loss = prefix_loss[k] / k
            obj = mean_loss + penalty_lambda * (1.0 - (k / n))
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_mu = mu
                best_subset = [x for _, x in losses_sorted[:k]]

    return best_mu, best_subset


def _cluster_bbox(
    words: List[Dict[str, Any]],
    link_scale: float = 2.0,
    return_cluster: bool = False,
) -> tuple[int, int, int, int] | tuple[tuple[int, int, int, int], List[Dict[str, Any]]] | None:
    if not words:
        return None
    sizes = []
    for w in words:
        text = w.get("text", "")
        char_count = max(len(text), 1)
        sizes.append((w["x2"] - w["x1"]) / char_count)
    median_size = _median(sizes)
    if median_size <= 0:
        return None
    link_dist = median_size * link_scale

    visited = [False] * len(words)
    clusters: List[List[int]] = []
    for i in range(len(words)):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        cluster = [i]
        while queue:
            idx = queue.pop()
            a = words[idx]
            for j in range(len(words)):
                if visited[j]:
                    continue
                b = words[j]
                if _box_edge_distance(a, b) <= link_dist:
                    visited[j] = True
                    queue.append(j)
                    cluster.append(j)
        clusters.append(cluster)

    if not clusters:
        return None
    best = max(clusters, key=len)
    kept = [words[i] for i in best]
    if not best:
        return None
    left = min(w["x1"] for w in kept)
    right = max(w["x2"] for w in kept)
    top = min(w["y1"] for w in kept)
    bottom = max(w["y2"] for w in kept)
    bbox = (left, top, right, bottom)
    if return_cluster:
        return bbox, kept
    return bbox


def _ocr_image_pil(
    image: Image.Image,
    lang: str,
    min_boxes_retry: int = 50,
    allow_rotate: bool = True,
) -> Dict[str, Any]:
    def _run_ocr(
        pil_image: Image.Image,
        *,
        upscale_factor: float = 1.0,
        sharpen: bool = False,
    ) -> Dict[str, Any]:
        pre = preprocess_image(pil_image, upscale_factor=upscale_factor, sharpen=sharpen)
        data = pytesseract.image_to_data(
            pre,
            lang=lang,
            config="--psm 4",
            output_type=pytesseract.Output.DICT,
        )
        from ocr_utils import word_boxes_from_data, text_from_words

        words = word_boxes_from_data(data)
        text, line_bboxes, line_words = text_from_words(words)
        return {
            "text": text.strip(),
            "words": words,
            "line_bboxes": line_bboxes,
            "line_words": line_words,
        }

    used_image = image
    result = _run_ocr(image)
    if allow_rotate and len(result["words"]) < min_boxes_retry:
        rotated = image.rotate(90, expand=True)
        rotated_result = _run_ocr(rotated)
        if len(rotated_result["words"]) > len(result["words"]):
            used_image = rotated
            result = rotated_result
        elif len(result["words"]) == 0 and len(rotated_result["words"]) == 0:
            used_image = rotated
            result = rotated_result

    if len(result["words"]) == 0:
        boosted = _run_ocr(used_image, upscale_factor=2.0, sharpen=True)
        if len(boosted["words"]) > len(result["words"]):
            result = boosted

    result["image"] = "in-memory"
    result["image_pil"] = used_image
    return result


def _clamp_layout_to_image(layout: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    doc_left, doc_top, doc_right, doc_bottom = layout["doc_box"]
    box_w = doc_right - doc_left
    box_h = doc_bottom - doc_top
    if box_w <= 0 or box_h <= 0:
        return layout
    if box_w > width:
        box_w = float(width)
        doc_left = 0.0
    if box_h > height:
        box_h = float(height)
        doc_top = 0.0
    new_left = doc_left
    new_top = doc_top
    if new_left < 0:
        new_left = 0
    if new_top < 0:
        new_top = 0
    if new_left + box_w > width:
        new_left = max(0.0, width - box_w)
    if new_top + box_h > height:
        new_top = max(0.0, height - box_h)
    dx = new_left - doc_left
    dy = new_top - doc_top
    if dx == 0 and dy == 0:
        return layout
    updated = dict(layout)
    updated["doc_box"] = (new_left, new_top, new_left + box_w, new_top + box_h)
    updated["expected_start_x"] = layout["expected_start_x"] + dx
    return updated


def main() -> None:
    benchmark_dir = BENCHMARK_DIR
    if not benchmark_dir.exists():
        raise SystemExit(f"Benchmark dir not found: {benchmark_dir}")

    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)

    debug_dir = DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    ocr_text_dir = OCR_TEXT_DIR
    ocr_text_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = PREPROCESSED_DIR
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir = CROPPED_DIR
    cropped_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_images(benchmark_dir):
        print(f"processing: {path.name}")
        image_type = type_map.get(path.name)
        t0 = time.perf_counter()
        image_full = Image.open(path).convert("RGB")
        result = _ocr_image_pil(image_full, lang=LANG)
        image_full = result["image_pil"]
        base_image = image_full
        stripe_bbox = None
        stripe_debug_image = image_full
        if len(result["words"]) < 50:
            pre_for_stripe = preprocess_image(image_full, upscale_factor=1.0, sharpen=False)
            stripe_bbox = _stripe_roi_bbox(pre_for_stripe)
            if stripe_bbox:
                stripe_debug_image = image_full
                image_full = image_full.crop(stripe_bbox)
                result = _ocr_image_pil(image_full, lang=LANG, allow_rotate=False)
                image_full = result["image_pil"]
                base_image = image_full
        t1 = time.perf_counter()
        if image_type == "m":
            target_for_image = target_texts["m"]
        elif image_type in ("s", "v", "k", "p"):
            target_for_image = target_texts[
                {
                    "s": "shema",
                    "v": "vehaya",
                    "k": "kadesh",
                    "p": "peter",
                }[image_type]
            ]
        else:
            print(f"warning: unknown type for {path.name}, skipping")
            continue
        words = result["words"]
        line_words = result["line_words"]

        boundary_word_index = len(target_texts["shema"].split()) if image_type == "m" else None
        expected_layout = estimate_layout(
            line_words,
            words,
            target_for_image,
            window_words=WINDOW_WORDS,
            max_skip=MAX_SKIP,
            score_threshold=0.4,
            boundary_word_index=boundary_word_index,
        )
        t2 = time.perf_counter()
        cluster_result = _cluster_bbox(words, return_cluster=True)
        detected_bbox = cluster_result[0] if cluster_result else None
        crop_bbox = None
        if expected_layout:
            expected_layout = _clamp_layout_to_image(expected_layout, image_full.width, image_full.height)
        if detected_bbox and expected_layout:
            doc_left, doc_top, doc_right, doc_bottom = expected_layout["doc_box"]
            crop_left = min(detected_bbox[0], int(doc_left))
            crop_top = min(detected_bbox[1], int(doc_top))
            crop_right = max(detected_bbox[2], int(doc_right))
            crop_bottom = max(detected_bbox[3], int(doc_bottom))
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(image_full.width, crop_right)
            crop_bottom = min(image_full.height, crop_bottom)
            if crop_right > crop_left and crop_bottom > crop_top:
                crop_bbox = (crop_left, crop_top, crop_right, crop_bottom)

        if crop_bbox:
            image_full = image_full.crop(crop_bbox)
        t3 = time.perf_counter()
        result = _ocr_image_pil(image_full, lang=LANG)
        t4 = time.perf_counter()
        ocr_text = strip_newlines(result["text"])
        words = result["words"]
        line_words = result["line_words"]
        image_full = result["image_pil"]
        opt_crop_bbox = None

        pre = preprocess_image(base_image)
        pre_out = preprocessed_dir / f"{path.stem}_pre.png"
        pre.save(pre_out)

        t_left0 = time.perf_counter()
        t_left1 = t_left0
        if words:
            image = image_full.copy()
            draw = ImageDraw.Draw(image)
            for w in words:
                draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)

            cluster_result2 = _cluster_bbox(words, return_cluster=True)
            kept_bbox = cluster_result2[0] if cluster_result2 else None
            cluster_words = cluster_result2[1] if cluster_result2 else []
            if kept_bbox:
                left, top, right, bottom = kept_bbox
                draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
            if cluster_words:
                t_left0 = time.perf_counter()
                filtered_words = _filter_boxes_by_char_size(cluster_words)
                left_flags, right_flags, top_flags, bottom_flags = _edge_flags(filtered_words)
                left_lines = []
                right_lines = []
                top_lines = []
                bottom_lines = []
                best_left = None
                best_right = None
                best_top = None
                best_bottom = None
                widths = [w["x2"] - w["x1"] for w in filtered_words]
                heights = [w["y2"] - w["y1"] for w in filtered_words]
                median_w = _median(widths)
                median_h = _median(heights)
                min_w = 0.2 * median_w if median_w > 0 else 0.0
                min_h = 0.2 * median_h if median_h > 0 else 0.0
                bottom_right_weight = 3
                for idx, w in enumerate(filtered_words):
                    if left_flags[idx]:
                        x = w["x1"]
                        left_lines.append(x)
                        draw.line([(x, w["y1"]), (x, w["y2"])], fill=(255, 165, 0), width=3)
                    if right_flags[idx]:
                        x = w["x2"]
                        right_lines.append(x)
                        draw.line([(x, w["y1"]), (x, w["y2"])], fill=(255, 165, 0), width=3)
                    if top_flags[idx]:
                        y = w["y1"]
                        top_lines.append(y)
                        draw.line([(w["x1"], y), (w["x2"], y)], fill=(255, 165, 0), width=3)
                    if bottom_flags[idx]:
                        if (w["x2"] - w["x1"]) >= min_w and (w["y2"] - w["y1"]) >= min_h:
                            y = w["y2"]
                            bottom_lines.append(y)
                            if right_flags[idx] and (w["x2"] - w["x1"]) >= min_w and (w["y2"] - w["y1"]) >= min_h:
                                bottom_lines.extend([y] * (bottom_right_weight - 1))
                            draw.line([(w["x1"], y), (w["x2"], y)], fill=(255, 165, 0), width=3)
                char_w = _median_char_size(filtered_words)
                char_h = _median_word_height(filtered_words)
                epsilon = 4.0
                if left_lines:
                    left_lines = _filter_outliers(left_lines, keep_lower=True)
                    if left_lines:
                        candidates = _next_candidates(left_lines, increasing=True)
                        res = align_edges_shift_side(
                            "left",
                            candidates,
                            left_lines,
                            penalty_lambda=100.0,
                            normalize_penalty=True,
                            strategy="next",
                            verbose=False,
                        )
                        best_x = res.line_x
                        original_x = min(left_lines)
                        if abs(best_x - original_x) < char_w:
                            best_x = original_x
                        if best_x != original_x:
                            best_x -= epsilon
                        best_left = best_x
                        draw.line([(best_x, 0), (best_x, image.height - 1)], fill=(255, 120, 0), width=2)
                if right_lines:
                    right_lines = _filter_outliers(right_lines, keep_lower=False)
                    if right_lines:
                        candidates = _next_candidates(right_lines, increasing=False)
                        res = align_edges_shift_side(
                            "right",
                            candidates,
                            right_lines,
                            penalty_lambda=100.0,
                            normalize_penalty=True,
                            strategy="next",
                            verbose=False,
                        )
                        best_x = res.line_x
                        original_x = max(right_lines)
                        if abs(best_x - original_x) < char_w:
                            best_x = original_x
                        if best_x != original_x:
                            best_x += epsilon
                        best_right = best_x
                        draw.line([(best_x, 0), (best_x, image.height - 1)], fill=(255, 120, 0), width=2)
                if top_lines:
                    top_lines = _filter_outliers(top_lines, keep_lower=True)
                    if top_lines:
                        candidates = _next_candidates(top_lines, increasing=True)
                        res = align_edges_shift_side(
                            "top",
                            candidates,
                            top_lines,
                            penalty_lambda=100.0,
                            normalize_penalty=True,
                            strategy="next",
                            verbose=False,
                        )
                        best_y = res.line_x
                        original_y = min(top_lines)
                        if abs(best_y - original_y) < char_h:
                            best_y = original_y
                        if best_y != original_y:
                            best_y -= epsilon
                        best_top = best_y
                        draw.line([(0, best_y), (image.width - 1, best_y)], fill=(255, 120, 0), width=2)
                if bottom_lines:
                    bottom_lines = _filter_outliers(bottom_lines, keep_lower=False)
                    if bottom_lines:
                        candidates = _next_candidates(bottom_lines, increasing=False)
                        res = align_edges_shift_side(
                            "bottom",
                            candidates,
                            bottom_lines,
                            penalty_lambda=100.0,
                            normalize_penalty=True,
                            strategy="next",
                            verbose=False,
                        )
                        best_y = res.line_x
                        original_y = max(bottom_lines)
                        if abs(best_y - original_y) < char_h:
                            best_y = original_y
                        if best_y != original_y:
                            best_y += epsilon
                        best_bottom = best_y
                        draw.line([(0, best_y), (image.width - 1, best_y)], fill=(255, 120, 0), width=2)

                t_left1 = time.perf_counter()
                if all(v is not None for v in (best_left, best_right, best_top, best_bottom)):
                    left = max(0, int(best_left))
                    right = min(image_full.width, int(best_right))
                    top = max(0, int(best_top))
                    bottom = min(image_full.height, int(best_bottom))
                    if right > left and bottom > top:
                        opt_crop_bbox = (left, top, right, bottom)

            if expected_layout and DRAW_EXPECTED_LAYOUT:
                offset = (crop_bbox[0], crop_bbox[1]) if crop_bbox else (0, 0)
                _draw_layout(image, expected_layout, offset)
            out_path = debug_dir / f"{path.stem}_debug.png"
            image.save(out_path)
        t5 = time.perf_counter()

        crop_out = cropped_dir / f"{path.stem}_crop.png"
        if opt_crop_bbox:
            image_full.crop(opt_crop_bbox).save(crop_out)
        else:
            image_full.save(crop_out)

        print(
            "timing: "
            f"ocr1={(t1 - t0):.3f}s "
            f"layout={(t2 - t1):.3f}s "
            f"crop={(t3 - t2):.3f}s "
            f"ocr2={(t4 - t3):.3f}s "
            f"debug={(t5 - t4):.3f}s "
            f"left={(t_left1 - t_left0):.3f}s"
        )

        text_out = ocr_text_dir / f"{path.stem}.txt"
        text_out.write_text(result["text"], encoding="utf-8")

        if image_type == "m":
            target = target_texts["m"]
            distance = levenshtein(ocr_text, target)
            print(f"type: {image_type}  distance: {distance}")
        else:
            names = ("shema", "vehaya", "kadesh", "peter")
            distances = {name: levenshtein(ocr_text, target_texts[name]) for name in names}
            guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
            print(f"type: {image_type}  guess: {guess_name}  distance: {guess_distance}")


if __name__ == "__main__":
    main()
