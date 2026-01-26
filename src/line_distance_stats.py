#!/usr/bin/env python3
"""Compute per-line OCR distance to target (shema + vehaya)."""

from __future__ import annotations

import statistics as stats
from pathlib import Path
from typing import List, Tuple

from cropper_config import OCR_TEXT_DIR, TEXT_DIR
from target_texts import load_target_texts, strip_newlines


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


def _best_word_segment_distance(
    line: str, target_words: List[str], length_tolerance: int = 5
) -> Tuple[int, str]:
    line = line.strip()
    if not line:
        return 0, ""
    if not target_words:
        return _levenshtein(line, ""), ""
    n = len(target_words)
    word_lens = [len(w) for w in target_words]
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + word_lens[i]

    def seg_len(i: int, j: int) -> int:
        if j < i:
            return 0
        return (prefix[j + 1] - prefix[i]) + (j - i)

    target_len = len(line)
    best_dist = 10**9
    best_segment = ""
    for i in range(n):
        j_min = i
        while j_min < n and seg_len(i, j_min) < target_len - length_tolerance:
            j_min += 1
        j_max = j_min
        while j_max < n and seg_len(i, j_max) <= target_len + length_tolerance:
            j_max += 1
        for j in range(j_min, j_max):
            segment = " ".join(target_words[i : j + 1])
            dist = _levenshtein(line, segment)
            if dist < best_dist:
                best_dist = dist
                best_segment = segment
                if best_dist == 0:
                    return best_dist, best_segment
    return best_dist, best_segment


def _display_hebrew(text: str) -> str:
    return text[::-1]


def _iter_mezuza_texts(ocr_dir: Path) -> List[Path]:
    return sorted(ocr_dir.glob("mezuza*.txt"))


def main() -> None:
    target_text = load_target_texts(TEXT_DIR)["m"]
    target_text = strip_newlines(target_text)
    target_words = [w for w in target_text.split() if w]
    ocr_dir = OCR_TEXT_DIR
    files = _iter_mezuza_texts(ocr_dir)
    if not files:
        raise SystemExit(f"No mezuza*.txt found in {ocr_dir}")

    for path in files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            print(f"{path.name}: no lines")
            continue
        matches = []
        for idx, line in enumerate(lines, start=1):
            clean = strip_newlines(line)
            if not clean:
                continue
            dist, segment = _best_word_segment_distance(clean, target_words)
            score = 1.0 - (dist / max(len(clean), 1))
            matches.append(
                {
                    "file": path.name,
                    "line": idx,
                    "ocr": clean,
                    "target": segment,
                    "score": score,
                    "length": len(clean),
                }
            )
        if not matches:
            print(f"{path.name}: no matches found")
            continue
        top = sorted(matches, key=lambda r: (r["score"], r["length"]), reverse=True)[:5]
        print(path.name)
        for rank, item in enumerate(top, start=1):
            print(f"  {rank}. line {item['line']} score={item['score']:.4f} len={item['length']}")
            print(f"     ocr: {_display_hebrew(item['ocr'])}")
            print(f"     target: {_display_hebrew(item['target'])}")


if __name__ == "__main__":
    main()
