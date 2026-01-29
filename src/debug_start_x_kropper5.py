#!/usr/bin/env python3
"""Debug start-x estimation details for אתגר הקרופר 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from best_sequence import find_best_sequences
from cropper_config import BENCHMARK_DIR, LANG, TEXT_DIR, TYPES_CSV
from ocr_utils import load_types, ocr_image
from target_texts import load_target_texts
from expected_layout import estimate_layout


def _display_hebrew(text: str) -> str:
    return text[::-1]


def _best_char_substring(target_text: str, ocr_text: str) -> tuple[int, int, str, int]:
    if not target_text or not ocr_text:
        return 0, 0, "", 0
    text = target_text
    pattern = ocr_text
    n = len(text)
    m = len(pattern)

    dp_dist = [0] * (n + 1)
    dp_start = list(range(n + 1))

    for i in range(1, m + 1):
        new_dist = [0] * (n + 1)
        new_start = [0] * (n + 1)
        new_dist[0] = i
        new_start[0] = 0
        pc = pattern[i - 1]
        for j in range(1, n + 1):
            tc = text[j - 1]
            del_cost = dp_dist[j] + 1
            ins_cost = new_dist[j - 1] + 1
            sub_cost = dp_dist[j - 1] + (pc != tc)

            best_cost = del_cost
            best_start = dp_start[j]

            if ins_cost < best_cost or (ins_cost == best_cost and new_start[j - 1] < best_start):
                best_cost = ins_cost
                best_start = new_start[j - 1]

            if sub_cost < best_cost or (sub_cost == best_cost and dp_start[j - 1] < best_start):
                best_cost = sub_cost
                best_start = dp_start[j - 1]

            new_dist[j] = best_cost
            new_start[j] = best_start

        dp_dist = new_dist
        dp_start = new_start

    best_end = 0
    best_dist = dp_dist[0]
    best_start = dp_start[0]
    for j in range(1, n + 1):
        if dp_dist[j] < best_dist or (
            dp_dist[j] == best_dist and dp_start[j] < best_start
        ):
            best_dist = dp_dist[j]
            best_end = j
            best_start = dp_start[j]

    best_sub = text[best_start:best_end]
    return best_start, best_end, best_sub, best_dist


def _resolve_image_path(stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = BENCHMARK_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found in benchmark/: {stem}")


def main() -> None:
    path = _resolve_image_path("אתגר הקרופר 5")
    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)
    image_type = type_map.get(path.name)
    target = target_texts[
        {
            "s": "shema",
            "v": "vehaya",
            "k": "kadesh",
            "p": "peter",
            "m": "m",
        }[image_type]
    ]

    result = ocr_image(path, lang=LANG)
    line_words = result["line_words"]
    words = result["words"]

    scored = find_best_sequences(
        line_words,
        target,
        window_words=6,
        max_skip=6,
        top_k=10**6,
    )
    if not scored:
        raise SystemExit("no sequences found")

    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    keep_n = max(3, int(len(scored) * 0.30))
    score_keep = [item for item in scored if item[0] >= 0.7]
    keep_n = max(keep_n, len(score_keep))
    kept = scored[:keep_n]

    chosen: List[Dict[str, Any]] = []
    for score, idx, (i, j) in kept:
        if idx >= len(line_words) or not line_words[idx]:
            continue
        span_words = sorted(line_words[idx], key=lambda w: -w["x2"])[i : j + 1]
        if not span_words:
            continue
        x1 = min(w["x1"] for w in span_words)
        y1 = min(w["y1"] for w in span_words)
        x2 = max(w["x2"] for w in span_words)
        y2 = max(w["y2"] for w in span_words)
        ocr_text = " ".join(w["text"] for w in span_words)
        t_start, t_end, t_sub, _ = _best_char_substring(target, ocr_text)
        per_word_sizes = []
        for w in span_words:
            w_text = w["text"]
            w_chars = len(w_text)
            if w_chars > 0:
                per_word_sizes.append((w["x2"] - w["x1"]) / w_chars)
        if per_word_sizes:
            per_word_sizes.sort()
            mid = len(per_word_sizes) // 2
            if len(per_word_sizes) % 2 == 1:
                letter_size = per_word_sizes[mid]
            else:
                letter_size = (per_word_sizes[mid - 1] + per_word_sizes[mid]) / 2.0
        else:
            char_count = len(ocr_text.replace(" ", ""))
            letter_size = (x2 - x1) / max(char_count, 1)
        chosen.append(
            {
                "score": score,
                "line": idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "ocr": ocr_text,
                "t_start": t_start,
                "t_end": t_end,
                "t_sub": t_sub,
                "letter_size": letter_size,
            }
        )

    chosen_sorted = sorted(chosen, key=lambda item: item["y1"])
    print(f"processing: {path.name} (type={image_type})")
    print("chosen spans:")
    for idx, item in enumerate(chosen_sorted, start=1):
        print(
            f"{idx:02d}. line={item['line']} score={item['score']:.4f} "
            f"t_start={item['t_start']} t_end={item['t_end']} "
            f"letter_size={item['letter_size']:.2f}"
        )
        print(f"  ocr: {_display_hebrew(item['ocr'])}")
        print(f"  target: {_display_hebrew(item['t_sub'])}")

    words_all = target.split()
    word_spans = []
    pos = 0
    for w in words_all:
        start = pos
        end = start + len(w)
        word_spans.append((start, end))
        pos = end + 1

    layout = estimate_layout(
        line_words,
        words,
        target,
        window_words=6,
        max_skip=6,
        score_threshold=0.4,
        boundary_word_index=None,
    )
    if not layout:
        raise SystemExit("layout estimation failed")
    lines = layout["lines"]

    print("layout lines:")
    for li, (_, _, text, _) in enumerate(lines):
        print(f"  line {li:02d}: {_display_hebrew(text)}")

    line_start_estimates = []
    print("start-x estimates:")
    for idx, item in enumerate(chosen_sorted, start=1):
        t_start = item["t_start"]
        word_idx = None
        for wi, (ws, we) in enumerate(word_spans):
            if we > t_start:
                word_idx = wi
                break
        if word_idx is None:
            continue
        line_start_word = None
        for (ws, we, _, _) in lines:
            if ws <= word_idx <= we:
                line_start_word = ws
                break
        if line_start_word is None:
            continue
        line_start_char = word_spans[line_start_word][0]
        offset_chars = max(0, t_start - line_start_char)
        est_x = item["x2"] + offset_chars * item["letter_size"]
        line_start_estimates.append(est_x)
        print(
            f"  span {idx:02d}: line={item['line']} t_start={t_start} "
            f"line_start_word={line_start_word} offset_chars={offset_chars} "
            f"letter_size={item['letter_size']:.2f} x2={item['x2']} "
            f"est_x={est_x:.1f}"
        )

    if line_start_estimates:
        line_start_estimates.sort()
        mid = len(line_start_estimates) // 2
        if len(line_start_estimates) % 2 == 1:
            expected_start_x = line_start_estimates[mid]
        else:
            expected_start_x = (line_start_estimates[mid - 1] + line_start_estimates[mid]) / 2.0
        expected_start_x = max(expected_start_x, max(item["x2"] for item in chosen_sorted))
        print(f"expected_start_x (median): {expected_start_x:.1f}")

    # Draw debug image with chosen spans and estimated box.
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if words:
        left = min(w["x1"] for w in words)
        right = max(w["x2"] for w in words)
        top = min(w["y1"] for w in words)
        bottom = max(w["y2"] for w in words)
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
    for item in chosen_sorted:
        draw.rectangle([item["x1"], item["y1"], item["x2"], item["y2"]], outline=(160, 0, 255), width=3)

    if words and line_start_estimates:
        doc_left, doc_top, doc_right, doc_bottom = layout["doc_box"]
        draw.rectangle([doc_left, doc_top, doc_right, doc_bottom], outline=(0, 255, 255), width=3)

        # Error metrics vs red box (OCR words bbox).
        est = (doc_left, doc_top, doc_right, doc_bottom)
        gt = (left, top, right, bottom)
        inter_left = max(est[0], gt[0])
        inter_top = max(est[1], gt[1])
        inter_right = min(est[2], gt[2])
        inter_bottom = min(est[3], gt[3])
        inter_w = max(0.0, inter_right - inter_left)
        inter_h = max(0.0, inter_bottom - inter_top)
        inter_area = inter_w * inter_h
        est_area = max(0.0, (est[2] - est[0]) * (est[3] - est[1]))
        gt_area = max(0.0, (gt[2] - gt[0]) * (gt[3] - gt[1]))
        union_area = est_area + gt_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        print(f"bbox error: iou={iou:.3f}")
        print(
            "bbox delta: "
            f"left={est[0]-gt[0]:.1f} top={est[1]-gt[1]:.1f} "
            f"right={est[2]-gt[2]:.1f} bottom={est[3]-gt[3]:.1f}"
        )

    out_path = Path("debug_images") / f"{path.stem}_startx_debug.png"
    image.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
