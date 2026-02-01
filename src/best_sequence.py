#!/usr/bin/env python3
"""Best-sequence selection for OCR line substrings."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


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
    seg_cache: Dict[Tuple[int, int], str] = {}
    lev_cache: Dict[str, int] = {}
    for i in range(n):
        j_min = i
        while j_min < n and seg_len(i, j_min) < target_len - length_tolerance:
            j_min += 1
        j_max = j_min
        while j_max < n and seg_len(i, j_max) <= target_len:
            j_max += 1
        for j in range(j_min, j_max):
            key = (i, j)
            if key in seg_cache:
                segment = seg_cache[key]
            else:
                segment = " ".join(target_words[i : j + 1])
                seg_cache[key] = segment
            if segment in lev_cache:
                dist = lev_cache[segment]
            else:
                dist = _levenshtein(line_text, segment)
                lev_cache[segment] = dist
            if dist < best_dist:
                best_dist = dist
                best_segment = segment
                if best_dist == 0:
                    return best_dist, best_segment
    return best_dist, best_segment


def find_best_sequences(
    line_words: List[List[Dict[str, Any]]],
    target_text: str,
    window_words: int = 6,
    max_skip: int = 6,
    top_k: int = 5,
    first_line_words: int = 5,
) -> List[Tuple[float, int, Tuple[int, int]]]:
    """Return best substrings (score, line_idx, (start,end))."""
    target_words = [w for w in target_text.split() if w]
    first_target = " ".join(target_words[:first_line_words])
    word_lens = [len(w) for w in target_words]
    prefix = [0] * (len(word_lens) + 1)
    for i, l in enumerate(word_lens):
        prefix[i + 1] = prefix[i] + l

    scored: List[Tuple[float, int, Tuple[int, int]]] = []
    best_prefix: Tuple[float, int, Tuple[int, int]] | None = None
    for idx, words in enumerate(line_words):
        if not words:
            continue
        ordered = sorted(words, key=lambda w: -w["x2"])
        ordered_texts = [w["text"] for w in ordered]
        sub_cache: Dict[Tuple[int, int], str] = {}

        def _sub(start: int, end: int) -> str:
            key = (start, end)
            if key in sub_cache:
                return sub_cache[key]
            text = " ".join(ordered_texts[start : end + 1])
            sub_cache[key] = text
            return text
        max_start = min(max_skip, max(len(ordered) - 1, 0))

        if first_target and len(ordered) >= window_words:
            prefix_best = None
            for start in range(0, max_start + 1):
                end = min(len(ordered), start + window_words) - 1
                if end < start:
                    continue
                ocr_sub = _sub(start, end)
                dist = _levenshtein(ocr_sub, first_target)
                denom = max(len(first_target), 1)
                score = 1.0 - (dist / denom)
                if prefix_best is None or score > prefix_best[0]:
                    prefix_best = (score, idx, (start, end))
            if prefix_best and (best_prefix is None or prefix_best[0] > best_prefix[0]):
                best_prefix = prefix_best

        if len(ordered) < window_words:
            continue
        best = None
        for start in range(0, max_start + 1):
            end = min(len(ordered), start + window_words) - 1
            if end - start + 1 < window_words:
                continue
            ocr_sub = _sub(start, end)
            dist, segment = _best_word_segment_distance_anywhere(
                ocr_sub, target_words, word_lens, prefix
            )
            denom = max(len(segment), 1)
            score = 1.0 - (dist / denom)
            if best is None or score > best[0]:
                best = (score, idx, (start, end))
        if best is None:
            continue
        scored.append(best)

    if best_prefix:
        scored = [best_prefix] + [s for s in scored if s[1] != best_prefix[1]]
    return scored[:top_k]
