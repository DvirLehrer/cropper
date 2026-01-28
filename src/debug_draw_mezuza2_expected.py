#!/usr/bin/env python3
"""Draw line1/line2 seq boxes and expected line start on mezuza2_debug.png."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from cropper_config import BENCHMARK_DIR, DEBUG_DIR, LANG, TEXT_DIR, TYPES_CSV
from ocr_utils import load_types, ocr_image
from target_texts import load_target_texts


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def _best_word_segment_distance_anywhere(
    line_text: str,
    target_words: List[str],
    word_lens: List[int],
    prefix: List[int],
    length_tolerance: int = 5,
) -> Tuple[int, str]:
    line_text = line_text.strip()
    if not line_text:
        return 0, ""
    if not target_words:
        return _levenshtein(line_text, ""), ""
    n = len(target_words)

    def seg_len(i: int, j: int) -> int:
        if j < i:
            return 0
        return (prefix[j + 1] - prefix[i]) + (j - i)

    target_len = len(line_text)
    best_dist = 10**9
    best_segment = ""
    for i in range(n):
        j_min = i
        while j_min < n and seg_len(i, j_min) < target_len - length_tolerance:
            j_min += 1
        j_max = j_min
        while j_max < n and seg_len(i, j_max) <= target_len:
            j_max += 1
        for j in range(j_min, j_max):
            segment = " ".join(target_words[i : j + 1])
            dist = _levenshtein(line_text, segment)
            if dist < best_dist:
                best_dist = dist
                best_segment = segment
                if best_dist == 0:
                    return best_dist, best_segment
    return best_dist, best_segment


def _best_char_substring(target_text: str, ocr_text: str) -> Tuple[int, int, str, int]:
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


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _best_window_for_line(
    words: List[Dict[str, Any]],
    target_words: List[str],
    word_lens: List[int],
    prefix: List[int],
    window_words: int,
    max_skip: int,
) -> Tuple[float, int, int] | None:
    ordered = sorted(words, key=lambda w: -w["x2"])
    if len(ordered) < window_words:
        return None
    max_start = min(max_skip, max(len(ordered) - 1, 0))
    best = None
    for start in range(0, max_start + 1):
        end = min(len(ordered), start + window_words) - 1
        if end - start + 1 < window_words:
            continue
        ocr_sub = " ".join(w["text"] for w in ordered[start : end + 1])
        dist, segment = _best_word_segment_distance_anywhere(
            ocr_sub, target_words, word_lens, prefix
        )
        denom = max(len(segment), 1)
        score = 1.0 - (dist / denom)
        if best is None or score > best[0]:
            best = (score, start, end)
    return best


def main() -> None:
    name = "mezuza2.jpg"
    path = BENCHMARK_DIR / name
    if not path.exists():
        alt = BENCHMARK_DIR / "mezuza2.jpeg"
        if alt.exists():
            path = alt
        else:
            raise SystemExit("mezuza2 image not found in benchmark/")

    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)
    image_type = type_map.get(path.name)
    target = target_texts["m"] if image_type == "m" else target_texts["m"]
    target_words = [w for w in target.split() if w]
    word_lens = [len(w) for w in target_words]
    prefix = [0] * (len(word_lens) + 1)
    for i, l in enumerate(word_lens):
        prefix[i + 1] = prefix[i] + l

    result = ocr_image(path, lang=LANG)
    line_words = result["line_words"]

    window_words = 6
    max_skip = 6
    best_windows: Dict[int, Tuple[float, int, int]] = {}
    for idx, words in enumerate(line_words):
        if not words:
            continue
        best = _best_window_for_line(
            words, target_words, word_lens, prefix, window_words, max_skip
        )
        if best:
            best_windows[idx] = best

    line1_idx = None
    for idx in sorted(best_windows.keys()):
        if best_windows[idx][0] >= 0.4:
            line1_idx = idx
            break
    if line1_idx is None:
        raise SystemExit("no line passed threshold=0.4")

    score, start, end = best_windows[line1_idx]
    line1_words = sorted(line_words[line1_idx], key=lambda w: -w["x2"])
    span_words = line1_words[start : end + 1]
    x1 = min(w["x1"] for w in span_words)
    y1 = min(w["y1"] for w in span_words)
    x2 = max(w["x2"] for w in span_words)
    y2 = max(w["y2"] for w in span_words)
    ocr_sub = " ".join(w["text"] for w in span_words)
    char_count = len(ocr_sub.replace(" ", ""))
    letter_size = (x2 - x1) / max(char_count, 1)
    t_start1, _, _, _ = _best_char_substring(target, ocr_sub)
    expected_start_x = x2 + (t_start1 * letter_size)

    t_start1_dbg, t_end1_dbg, t_sub1_dbg, _ = _best_char_substring(target, ocr_sub)
    words = target.split()
    if not words:
        raise SystemExit("empty target text")
    line_text = ""
    line_chars = 0
    running = 0
    for i, w in enumerate(words):
        add = len(w) if i == 0 else len(w) + 1
        running += add
        line_text = " ".join(words[: i + 1])
        line_chars = len(line_text)
        if running >= t_end1_dbg:
            break
    if line_chars <= 0:
        raise SystemExit("invalid line length (chars)")
    line_px = line_chars * letter_size

    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for w in words:
        extra = len(w) if not current else len(w) + 1
        if current and current_len + extra > line_chars:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    line_count = len(lines)

    word_heights = [
        w["y2"] - w["y1"] for w in result.get("words", []) if w.get("y2") is not None
    ]
    line_height = _median(word_heights) or max(1.0, y2 - y1)
    doc_height = line_height * line_count
    doc_right = expected_start_x
    doc_left = doc_right - line_px
    doc_top = y1
    doc_bottom = doc_top + doc_height

    print("line length derivation:")
    print(f"  line1 ocr span: {ocr_sub[::-1]}")
    print(f"  line1 target segment: {t_sub1_dbg[::-1]}")
    print(f"  line1 target start: {t_start1_dbg}")
    print(f"  line text (for line length): {line_text[::-1]}")
    print(f"  line length chars: {len(line_text)}")

    print("expected text lines (word-wrapped):")
    for i, segment in enumerate(lines):
        print(f"  line {i:02d}: {segment[::-1]}")

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    if result["words"]:
        words = result["words"]
        left = min(w["x1"] for w in words)
        right = max(w["x2"] for w in words)
        top = min(w["y1"] for w in words)
        bottom = max(w["y2"] for w in words)
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
        for w in words:
            draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)

    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=3)
    exp_x = int(round(expected_start_x))
    exp_x = max(0, min(exp_x, image.width - 1))
    draw.line([(exp_x, y1), (exp_x, y2)], fill=(255, 0, 0), width=3)

    draw.rectangle(
        [doc_left, doc_top, doc_right, doc_bottom],
        outline=(0, 255, 255),
        width=3,
    )

    out_path = DEBUG_DIR / "mezuza2_debug_expected.png"
    image.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
