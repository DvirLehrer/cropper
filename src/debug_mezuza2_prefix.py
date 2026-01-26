#!/usr/bin/env python3
"""Debug prefix-vs-ocr matching for mezuza2."""

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


def _ordered_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(words, key=lambda w: -w["x2"])


def _best_prefix_span(
    line_words: List[Dict[str, Any]],
    target_prefix: str,
    max_skip: int,
) -> Tuple[float, int, int, str]:
    ordered = _ordered_words(line_words)
    max_start = min(max_skip, max(len(ordered) - 1, 0))
    best = (-10.0, 0, 0, "")
    for start in range(0, max_start + 1):
        for end in range(start, len(ordered)):
            ocr_sub = " ".join(w["text"] for w in ordered[start : end + 1])
            dist = _levenshtein(ocr_sub, target_prefix)
            denom = max(len(target_prefix), 1)
            score = 1.0 - (dist / denom)
            if score > best[0]:
                best = (score, start, end, ocr_sub)
    return best


def _best_target_substring_for_ocr(
    ocr_sub: str,
    target_words: List[str],
    word_lens: List[int],
    prefix: List[int],
    length_tolerance: int = 5,
) -> Tuple[float, int, int, str, int, int]:
    if not ocr_sub or not target_words:
        return -10.0, 0, 0, "", 0, 0
    n = len(target_words)

    def seg_len(i: int, j: int) -> int:
        if j < i:
            return 0
        return (prefix[j + 1] - prefix[i]) + (j - i)

    target_len = len(ocr_sub)
    best_dist = 10**9
    best_segment = ""
    best_i = 0
    best_j = 0
    for i in range(n):
        j_min = i
        while j_min < n and seg_len(i, j_min) < target_len - length_tolerance:
            j_min += 1
        j_max = j_min
        while j_max < n and seg_len(i, j_max) <= target_len:
            j_max += 1
        for j in range(j_min, j_max):
            segment = " ".join(target_words[i : j + 1])
            dist = _levenshtein(ocr_sub, segment)
            if dist < best_dist:
                best_dist = dist
                best_segment = segment
                best_i = i
                best_j = j
                if best_dist == 0:
                    break
        if best_dist == 0:
            break

    denom = max(len(best_segment), 1)
    score = 1.0 - (best_dist / denom)
    start_char = prefix[best_i] + best_i
    return score, best_i, best_j, best_segment, best_dist, start_char


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
    first_line_words = 5
    target_prefix = " ".join(target_words[:first_line_words])

    result = ocr_image(path, lang=LANG)
    line_words = result["line_words"]

    print(f"image: {path.name}  type: {image_type}")
    print(f"target prefix (5 words): {_display_hebrew(target_prefix)}")
    print("")

    max_skip = 6
    if not line_words:
        print("no OCR lines found")
        return

    line = line_words[0]
    ordered = _ordered_words(line)
    line_text = " ".join(w["text"] for w in ordered)
    print("line 1 only:")
    print(f"  ocr line: {_display_hebrew(line_text)}")
    print("")

    max_start = min(max_skip, max(len(ordered) - 1, 0))

    def scan_spans(allowed_starts: List[int]) -> Tuple[
        Tuple[float, int, int, str, int, int, int, str, int] | None,
        List[Tuple[float, int, int, str, int, int]],
    ]:
        best_local = None
        all_rows: List[Tuple[float, int, int, str, int, int]] = []
        for start in allowed_starts:
            for end in range(start, len(ordered)):
                ocr_sub = " ".join(w["text"] for w in ordered[start : end + 1])
                score, t_start, t_end, segment, dist, start_char = _best_target_substring_for_ocr(
                    ocr_sub,
                    target_words,
                    word_lens,
                    prefix,
                )
                all_rows.append((score, start, end, ocr_sub, t_start, start_char))
                if best_local is None or score > best_local[0]:
                    best_local = (
                        score,
                        start,
                        end,
                        ocr_sub,
                        dist,
                        t_start,
                        t_end,
                        segment,
                        start_char,
                    )
        return best_local, all_rows

    best_all, rows_all = scan_spans(list(range(0, max_start + 1)))
    if not best_all:
        print("no spans evaluated")
        return

    score, start, end, ocr_sub, dist, t_start, t_end, segment, start_char = best_all
    print("initial best (all starts):")
    print(
        f"  score={score:.4f} dist={dist} start={start} end={end} "
        f"target_start_word={t_start} missing_chars={start_char}"
    )
    print(f"  ocr span: {_display_hebrew(ocr_sub)}")
    print(f"  target: {_display_hebrew(segment)}")
    print("")

    if t_start > 0:
        print("rule: goal start > 0, force OCR start=0")
        best_forced, rows_forced = scan_spans([0])
        if best_forced:
            score, start, end, ocr_sub, dist, t_start, t_end, segment, start_char = best_forced
            print("best with OCR start=0:")
            print(
                f"  score={score:.4f} dist={dist} start={start} end={end} "
                f"target_start_word={t_start} missing_chars={start_char}"
            )
            print(f"  ocr span: {_display_hebrew(ocr_sub)}")
            print(f"  target: {_display_hebrew(segment)}")
            print("")
    else:
        print("rule: goal start == 0, allow OCR start>0 (noise can be skipped)")


if __name__ == "__main__":
    main()
