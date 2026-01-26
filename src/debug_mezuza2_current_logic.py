#!/usr/bin/env python3
"""Debug current find_best_sequences logic for mezuza2."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from cropper_config import BENCHMARK_DIR, LANG, TEXT_DIR, TYPES_CSV
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


def _display_hebrew(text: str) -> str:
    return text[::-1]


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


def _best_char_substring(
    target_text: str,
    ocr_text: str,
    length_tol: int = 6,
) -> Tuple[int, int, str, int]:
    if not target_text or not ocr_text:
        return 0, 0, "", 0
    text = target_text
    pattern = ocr_text
    n = len(text)
    m = len(pattern)

    # dp_dist[j] = edit distance between pattern[:i] and text[:j]
    # dp_start[j] = start index in text for the best alignment ending at j
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

    window_words = 6
    max_skip = 6
    top_k = 5
    first_line_words = 5

    result = ocr_image(path, lang=LANG)
    line_words = result["line_words"]

    target_words = [w for w in target.split() if w]
    first_target = " ".join(target_words[:first_line_words])
    word_lens = [len(w) for w in target_words]
    prefix = [0] * (len(word_lens) + 1)
    for i, l in enumerate(word_lens):
        prefix[i + 1] = prefix[i] + l

    print(f"image: {path.name}  type: {image_type}")
    print(f"first target ({first_line_words} words): {_display_hebrew(first_target)}")
    print("")

    scored: List[Tuple[float, int, Tuple[int, int]]] = []
    best_prefix: Tuple[float, int, Tuple[int, int]] | None = None
    best_windows: Dict[int, Tuple[float, int, int]] = {}

    for idx, words in enumerate(line_words):
        if not words:
            continue
        ordered = sorted(words, key=lambda w: -w["x2"])
        line_text = " ".join(w["text"] for w in ordered)
        max_start = min(max_skip, max(len(ordered) - 1, 0))

        print(f"line {idx}:")
        print(f"  ocr: {_display_hebrew(line_text)}")
        print("  tokens:")
        for i, w in enumerate(ordered):
            print(f"    {i}: {_display_hebrew(w['text'])}")

        if first_target:
            print("  prefix search (any length):")
            prefix_best = None
            target_len = len(first_target)
            length_tol = 6
            token_lens = [len(w["text"]) for w in ordered]
            for start in range(0, max_start + 1):
                span_len = 0
                for end in range(start, len(ordered)):
                    if end == start:
                        span_len = token_lens[end]
                    else:
                        span_len += 1 + token_lens[end]
                    if span_len < target_len - length_tol:
                        continue
                    if span_len > target_len + length_tol:
                        break
                    ocr_sub = " ".join(w["text"] for w in ordered[start : end + 1])
                    dist = _levenshtein(ocr_sub, first_target)
                    denom = max(len(first_target), 1)
                    score = 1.0 - (dist / denom)
                    print(
                        f"    start={start} end={end} score={score:.4f} dist={dist}"
                    )
                    print(f"      ocr: {_display_hebrew(ocr_sub)}")
                    print(f"      target: {_display_hebrew(first_target)}")
                    if prefix_best is None or score > prefix_best[0]:
                        prefix_best = (score, idx, (start, end))
            if prefix_best:
                score, _, (start, end) = prefix_best
                print(
                    f"  best prefix: score={score:.4f} start={start} end={end}"
                )
                if best_prefix is None or score > best_prefix[0]:
                    best_prefix = prefix_best

        if len(ordered) < window_words:
            print("  skip window search (line too short)")
            print("")
            continue

        best = None
        print(f"  window search (window_words={window_words}):")
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
            print(
                f"    start={start} end={end} score={score:.4f} dist={dist}"
            )
            print(f"      ocr: {_display_hebrew(ocr_sub)}")
            print(f"      target: {_display_hebrew(segment)}")
            if best is None or score > best[0]:
                best = (score, idx, (start, end))
        if best:
            score, _, (start, end) = best
            print(f"  best window: score={score:.4f} start={start} end={end}")
            scored.append(best)
            best_windows[idx] = (score, start, end)
        print("")

    if best_prefix:
        scored = [best_prefix] + [s for s in scored if s[1] != best_prefix[1]]

    print("final chosen (top_k order):")
    for score, line_idx, (start, end) in scored[:top_k]:
        ordered = sorted(line_words[line_idx], key=lambda w: -w["x2"])
        ocr_sub = " ".join(w["text"] for w in ordered[start : end + 1])
        print(
            f"  line {line_idx} score={score:.4f} start={start} end={end} "
            f"ocr={_display_hebrew(ocr_sub)}"
        )
        t_start, t_end, t_sub, dist = _best_char_substring(target, ocr_sub)
        print(
            "  best target substring (char-level): "
            f"start={t_start} end={t_end} dist={dist}"
        )
        print(f"    target: {_display_hebrew(t_sub)}")

    print("")
    print("derived metrics:")
    line1_idx = None
    line1_score = None
    for idx in sorted(best_windows.keys()):
        score, start, end = best_windows[idx]
        if score >= 0.4:
            line1_idx = idx
            line1_score = score
            break

    if line1_idx is None:
        print("  no line passed threshold=0.4")
        return

    line1_words = sorted(line_words[line1_idx], key=lambda w: -w["x2"])
    score, start, end = best_windows[line1_idx]
    span_words = line1_words[start : end + 1]
    x1 = min(w["x1"] for w in span_words)
    y1 = min(w["y1"] for w in span_words)
    x2 = max(w["x2"] for w in span_words)
    y2 = max(w["y2"] for w in span_words)
    ocr_sub = " ".join(w["text"] for w in span_words)
    char_count = len(ocr_sub.replace(" ", ""))
    letter_size = (x2 - x1) / max(char_count, 1)
    t_start1, t_end1, t_sub1, dist1 = _best_char_substring(target, ocr_sub)

    print(f"  line1 idx: {line1_idx} score={line1_score:.4f}")
    print(f"  line1 bbox: x1={x1} y1={y1} x2={x2} y2={y2}")
    print(f"  line1 ocr: {_display_hebrew(ocr_sub)}")
    print(f"  line1 target: {_display_hebrew(t_sub1)}")
    print(f"  line1 target start: {t_start1}")
    print(f"  letter size (px/char): {letter_size:.3f}")

    line1_bottom = y2
    line2_idx = None
    line2_delta = None
    for idx, words in enumerate(line_words):
        if not words:
            continue
        line_y1 = min(w["y1"] for w in words)
        line_y2 = max(w["y2"] for w in words)
        line_center = (line_y1 + line_y2) / 2.0
        if line_center <= line1_bottom:
            continue
        delta = line_center - line1_bottom
        if line2_delta is None or delta < line2_delta:
            line2_delta = delta
            line2_idx = idx

    if line2_idx is None:
        print("  no line found beneath line1")
        return

    line2_words = sorted(line_words[line2_idx], key=lambda w: -w["x2"])
    overlap_words = [
        w
        for w in line2_words
        if not (w["x2"] < x1 or w["x1"] > x2)
    ]
    if overlap_words:
        line2_ocr = " ".join(w["text"] for w in overlap_words)
    else:
        line2_ocr = " ".join(w["text"] for w in line2_words)
    t_start2, t_end2, t_sub2, dist2 = _best_char_substring(target, line2_ocr)
    line_chars = t_start2 - t_start1
    line_px = line_chars * letter_size

    print(f"  line2 idx: {line2_idx}")
    print(f"  line2 ocr (area): {_display_hebrew(line2_ocr)}")
    print(f"  line2 target: {_display_hebrew(t_sub2)}")
    print(f"  line2 target start: {t_start2}")
    print(f"  line length chars: {line_chars}")
    print(f"  line length px: {line_px:.1f}")


if __name__ == "__main__":
    main()
