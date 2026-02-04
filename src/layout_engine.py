#!/usr/bin/env python3
"""Pluggable layout engine interface and default implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from expected_layout import _best_char_substring, estimate_layout as _estimate_layout
from ocr_utils import levenshtein


class LayoutEngine(Protocol):
    def estimate(
        self,
        line_words: List[List[Dict[str, Any]]],
        words: List[Dict[str, Any]],
        target_text: str,
        *,
        window_words: int,
        max_skip: int,
        score_threshold: float,
        boundary_word_index: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Estimate layout from OCR output."""


@dataclass(frozen=True)
class ExpectedLayoutEngine:
    """Wrap the existing expected_layout.estimate_layout logic."""

    def estimate(
        self,
        line_words: List[List[Dict[str, Any]]],
        words: List[Dict[str, Any]],
        target_text: str,
        *,
        window_words: int,
        max_skip: int,
        score_threshold: float,
        boundary_word_index: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        return _estimate_layout(
            line_words,
            words,
            target_text,
            window_words=window_words,
            max_skip=max_skip,
            score_threshold=score_threshold,
            boundary_word_index=boundary_word_index,
        )


@dataclass(frozen=True)
class WholeTextWindowEngine:
    """Find the best contiguous line window by minimizing full-text distance."""

    def estimate(
        self,
        line_words: List[List[Dict[str, Any]]],
        words: List[Dict[str, Any]],
        target_text: str,
        *,
        window_words: int,
        max_skip: int,
        score_threshold: float,
        boundary_word_index: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not line_words or not target_text:
            return None
        line_texts: List[str] = []
        line_bboxes: List[Tuple[int, int, int, int]] = []
        for line in line_words:
            if not line:
                line_texts.append("")
                line_bboxes.append((0, 0, 0, 0))
                continue
            ordered = sorted(line, key=lambda w: -w["x2"])
            line_texts.append(" ".join(w["text"] for w in ordered))
            x1 = min(w["x1"] for w in line)
            y1 = min(w["y1"] for w in line)
            x2 = max(w["x2"] for w in line)
            y2 = max(w["y2"] for w in line)
            line_bboxes.append((x1, y1, x2, y2))

        best = None
        best_dist = None
        for i in range(len(line_texts)):
            for j in range(i, len(line_texts)):
                text = "\n".join(line_texts[i : j + 1]).strip()
                if not text:
                    continue
                dist = levenshtein(text, target_text.strip())
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = (i, j)

        if best is None:
            return None
        i, j = best
        xs = [line_bboxes[k][0] for k in range(i, j + 1)]
        ys = [line_bboxes[k][1] for k in range(i, j + 1)]
        xe = [line_bboxes[k][2] for k in range(i, j + 1)]
        ye = [line_bboxes[k][3] for k in range(i, j + 1)]
        doc_left = min(xs)
        doc_top = min(ys)
        doc_right = max(xe)
        doc_bottom = max(ye)
        expected_start_x = doc_left
        return {
            "doc_box": (float(doc_left), float(doc_top), float(doc_right), float(doc_bottom)),
            "expected_start_x": float(expected_start_x),
            "score": 0.0 if best_dist is None else -float(best_dist),
        }


@dataclass(frozen=True)
class BoxPruneEngine:
    """Search line windows, trimming margins (outside-in) with balanced trims."""

    max_trim: int = 5

    def estimate(
        self,
        line_words: List[List[Dict[str, Any]]],
        words: List[Dict[str, Any]],
        target_text: str,
        *,
        window_words: int,
        max_skip: int,
        score_threshold: float,
        boundary_word_index: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not line_words or not target_text:
            return None

        lines: List[List[Dict[str, Any]]] = []
        for line in line_words:
            if not line:
                continue
            ordered = sorted(line, key=lambda w: -w["x2"])
            lines.append(ordered)
        if not lines:
            return None

        def _trim_pairs(n: int) -> List[Tuple[int, int]]:
            if n <= 0:
                return []
            limit = min(self.max_trim, n - 1)
            pairs = []
            for k in range(0, limit + 1):
                pairs.append((k, k))
                if k + 1 <= limit:
                    pairs.append((k, k + 1))
                    pairs.append((k + 1, k))
            seen = set()
            out = []
            for a, b in pairs:
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                out.append((a, b))
            return out

        def _slice_line(line: List[Dict[str, Any]], left_trim: int, right_trim: int) -> List[Dict[str, Any]]:
            if left_trim + right_trim >= len(line):
                return []
            end = len(line) - right_trim
            return line[left_trim:end]

        best_score = None
        best_boxes: List[Dict[str, Any]] = []
        best_sub = ""

        for i in range(len(lines)):
            for j in range(i, len(lines)):
                first = lines[i]
                last = lines[j]
                first_pairs = _trim_pairs(len(first))
                last_pairs = _trim_pairs(len(last)) if j != i else first_pairs
                middle = lines[i + 1 : j] if j > i + 1 else []

                for f_left, f_right in first_pairs:
                    f_line = _slice_line(first, f_left, f_right)
                    if not f_line:
                        continue
                    for l_left, l_right in last_pairs:
                        l_line = f_line if j == i else _slice_line(last, l_left, l_right)
                        if not l_line:
                            continue

                        kept_lines = [f_line] + middle + ([l_line] if j != i else [])
                        kept_boxes = [w for line in kept_lines for w in line]
                        if not kept_boxes:
                            continue
                        text_lines = [" ".join(w["text"] for w in line) for line in kept_lines]
                        ocr_text = "\n".join(text_lines).strip()
                        if not ocr_text:
                            continue
                        _, _, sub, dist = _best_char_substring(target_text, ocr_text)
                        score = 1.0 - (dist / max(len(sub), 1))
                        if best_score is None or score > best_score:
                            best_score = score
                            best_boxes = kept_boxes
                            best_sub = sub

        if not best_boxes or best_score is None:
            return None
        doc_left = min(w["x1"] for w in best_boxes)
        doc_top = min(w["y1"] for w in best_boxes)
        doc_right = max(w["x2"] for w in best_boxes)
        doc_bottom = max(w["y2"] for w in best_boxes)
        expected_start_x = doc_left
        return {
            "doc_box": (float(doc_left), float(doc_top), float(doc_right), float(doc_bottom)),
            "expected_start_x": float(expected_start_x),
            "score": float(best_score),
            "target_sub": best_sub,
        }


@dataclass(frozen=True)
class SequentialTargetEngine:
    """Match each OCR line to the closest remaining target substring in order."""

    def estimate(
        self,
        line_words: List[List[Dict[str, Any]]],
        words: List[Dict[str, Any]],
        target_text: str,
        *,
        window_words: int,
        max_skip: int,
        score_threshold: float,
        boundary_word_index: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if not line_words or not target_text:
            return None

        ordered_lines: List[List[Dict[str, Any]]] = []
        for line in line_words:
            if not line:
                continue
            ordered_lines.append(sorted(line, key=lambda w: -w["x2"]))
        if not ordered_lines:
            return None

        remaining = target_text
        gaps: List[str] = []

        for line in ordered_lines:
            line_text = " ".join(w["text"] for w in line).strip()
            if not line_text:
                continue
            start, end, _, _ = _best_char_substring(remaining, line_text)
            if start > 0:
                gaps.append(remaining[:start])
            remaining = remaining[end:]

        if remaining:
            gaps.append(remaining)

        doc_left = min(w["x1"] for w in words)
        doc_top = min(w["y1"] for w in words)
        doc_right = max(w["x2"] for w in words)
        doc_bottom = max(w["y2"] for w in words)
        expected_start_x = doc_left
        return {
            "doc_box": (float(doc_left), float(doc_top), float(doc_right), float(doc_bottom)),
            "expected_start_x": float(expected_start_x),
            "score": 0.0,
            "target_gaps": gaps,
        }


def get_layout_engine(name: str = "expected") -> LayoutEngine:
    if name == "expected":
        return ExpectedLayoutEngine()
    if name == "window_lev":
        return WholeTextWindowEngine()
    if name == "box_prune":
        return BoxPruneEngine()
    if name == "seq_target":
        return SequentialTargetEngine()
    raise ValueError(f"Unknown layout engine: {name}")
