#!/usr/bin/env python3
"""Expected document layout estimation from OCR line 1."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from best_sequence import find_best_sequences


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
            best = (score, start, end)
    return best


def estimate_layout(
    line_words: List[List[Dict[str, Any]]],
    words: List[Dict[str, Any]],
    target_text: str,
    window_words: int,
    max_skip: int,
    score_threshold: float = 0.4,
    boundary_word_index: int | None = None,
) -> Dict[str, Any] | None:
    if not target_text or not line_words:
        return None

    line_index_map = list(range(len(line_words)))

    scored = find_best_sequences(
        line_words,
        target_text,
        window_words=window_words,
        max_skip=max_skip,
        top_k=120,
    )
    if not scored:
        return None

    if line_index_map:
        scored = [(s, line_index_map[idx], span) for (s, idx, span) in scored]

    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    keep_n = max(3, int(len(scored) * 0.20))
    score_keep = [item for item in scored if item[0] >= 0.7]
    keep_n = max(keep_n, len(score_keep))
    kept = scored[:keep_n]

    chosen = []
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
        t_start, t_end, _, _ = _best_char_substring(target_text, ocr_text)
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
                "start": i,
                "end": j,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "ocr": ocr_text,
                "t_start": t_start,
                "t_end": t_end,
                "letter_size": letter_size,
            }
        )

    if not chosen:
        return None

    chosen_sorted = sorted(chosen, key=lambda item: item["y1"])

    weighted_sum = 0.0
    weight_total = 0.0
    size_weighted_sum = 0.0
    size_weight_total = 0.0
    gap_line_lengths: List[float] = []
    for i in range(len(chosen_sorted) - 1):
        top = chosen_sorted[i]
        bottom = chosen_sorted[i + 1]
        overlap_left = max(top["x1"], bottom["x1"])
        overlap_right = min(top["x2"], bottom["x2"])
        if overlap_left >= overlap_right:
            continue
        overlap_x = overlap_left
        top_offset = int(round((top["x2"] - overlap_x) / max(top["letter_size"], 1e-6)))
        bottom_offset = int(round((bottom["x2"] - overlap_x) / max(bottom["letter_size"], 1e-6)))
        top_pos = top["t_start"] + top_offset
        bottom_pos = bottom["t_start"] + bottom_offset
        if bottom_pos <= top_pos:
            continue
        delta_lines = bottom["line"] - top["line"]
        if delta_lines <= 0:
            continue
        gap_text = target_text[top_pos:bottom_pos]
        line_len = len(gap_text) / delta_lines
        weight = (top["score"] + bottom["score"]) / 2.0
        weighted_sum += line_len * weight
        weight_total += weight
        size_weighted_sum += ((top["letter_size"] + bottom["letter_size"]) / 2.0) * weight
        size_weight_total += weight
        gap_line_lengths.append(line_len)

    line_chars = 0
    avg_letter_size = None
    if weight_total > 0:
        weighted_avg = weighted_sum / weight_total
        line_chars = max(1, int(round(weighted_avg)))
        if size_weight_total > 0:
            avg_letter_size = size_weighted_sum / size_weight_total

    gap_quality_weak = False
    if gap_line_lengths:
        mean_len = sum(gap_line_lengths) / len(gap_line_lengths)
        if len(gap_line_lengths) < 2:
            gap_quality_weak = True
        else:
            variance = sum((v - mean_len) ** 2 for v in gap_line_lengths) / len(gap_line_lengths)
            stdev = variance ** 0.5
            if mean_len > 0 and (stdev / mean_len) > 0.25:
                gap_quality_weak = True
    else:
        gap_quality_weak = True

    ocr_line_count = len([line for line in line_words if line])
    target_line_count = ocr_line_count if gap_quality_weak and ocr_line_count > 0 else None

    words_all = target_text.split()
    if not words_all:
        return None

    word_spans = []
    pos = 0
    for w in words_all:
        start = pos
        end = start + len(w)
        word_spans.append((start, end))
        pos = end + 1

    anchors = []
    for item in chosen_sorted:
        t_start = item["t_start"]
        t_end = item["t_end"]
        w_start = 0
        w_end = len(words_all) - 1
        for wi, (ws, we) in enumerate(word_spans):
            if we > t_start:
                w_start = wi
                break
        for wi, (ws, we) in enumerate(word_spans):
            if ws >= t_end:
                w_end = max(w_start, wi - 1)
                break
        anchors.append({"line": item["line"], "w_start": w_start, "w_end": w_end})

    anchor_end_by_start = {a["w_start"]: a["w_end"] for a in anchors}

    tokens: List[tuple[int, int, str, int, bool]] = []
    i = 0
    while i < len(words_all):
        if i in anchor_end_by_start:
            end = anchor_end_by_start[i]
            if boundary_word_index is not None and i < boundary_word_index <= end:
                left_text = " ".join(words_all[i:boundary_word_index])
                right_text = " ".join(words_all[boundary_word_index : end + 1])
                if left_text:
                    tokens.append((i, boundary_word_index - 1, left_text, len(left_text), True))
                if right_text:
                    tokens.append((boundary_word_index, end, right_text, len(right_text), True))
                i = end + 1
            else:
                text = " ".join(words_all[i : end + 1])
                tokens.append((i, end, text, len(text), True))
                i = end + 1
        else:
            text = words_all[i]
            tokens.append((i, i, text, len(text), False))
            i += 1

    def build_lines(token_slice: List[tuple[int, int, str, int, bool]]) -> List[tuple[int, int, str, list[int]]]:
        n = len(token_slice)
        prefix = [0] * (n + 1)
        for idx, tok in enumerate(token_slice):
            prefix[idx + 1] = prefix[idx] + tok[3]

        def line_length(i: int, j: int) -> int:
            return (prefix[j] - prefix[i]) + max(0, j - i - 1)

        dp = [0] * (n + 1)
        nxt = [n] * (n + 1)
        dp[n] = 0
        for i in range(n - 1, -1, -1):
            best_cost = None
            best_j = None
            for j in range(i + 1, n + 1):
                length = line_length(i, j)
                if i == 0 or j == n:
                    if line_chars and length > line_chars:
                        continue
                    cost = 0
                else:
                    cost = 0 if not line_chars else (length - line_chars) ** 2
                    if line_chars and length > line_chars:
                        break
                total = cost + dp[j]
                if best_cost is None or total < best_cost:
                    best_cost = total
                    best_j = j
            dp[i] = best_cost if best_cost is not None else 0
            nxt[i] = best_j if best_j is not None else n

        out = []
        i = 0
        while i < n:
            j = nxt[i]
            line_start = token_slice[i][0]
            line_end = token_slice[j - 1][1]
            line_text = " ".join(tok[2] for tok in token_slice[i:j])
            out.append((line_start, line_end, line_text, list(range(i, j))))
            i = j
        return out

    def build_lines_fixed_count(
        token_slice: List[tuple[int, int, str, int, bool]],
        desired_lines: int,
    ) -> List[tuple[int, int, str, list[int]]]:
        n = len(token_slice)
        prefix = [0] * (n + 1)
        for idx, tok in enumerate(token_slice):
            prefix[idx + 1] = prefix[idx] + tok[3]

        def line_length(i: int, j: int) -> int:
            return (prefix[j] - prefix[i]) + max(0, j - i - 1)

        if desired_lines <= 0:
            return []

        dp = [[None] * (desired_lines + 1) for _ in range(n + 1)]
        nxt = [[None] * (desired_lines + 1) for _ in range(n + 1)]
        dp[n][0] = 0
        for i in range(n - 1, -1, -1):
            for k in range(1, desired_lines + 1):
                best_cost = None
                best_j = None
                for j in range(i + 1, n + 1):
                    length = line_length(i, j)
                    if i == 0 or j == n:
                        if line_chars and length > line_chars:
                            continue
                        cost = 0
                    else:
                        cost = 0 if not line_chars else (length - line_chars) ** 2
                        if line_chars and length > line_chars:
                            break
                    if dp[j][k - 1] is None:
                        continue
                    total = cost + dp[j][k - 1]
                    if best_cost is None or total < best_cost:
                        best_cost = total
                        best_j = j
                dp[i][k] = best_cost
                nxt[i][k] = best_j

        out = []
        i = 0
        k = desired_lines
        while i < n and k > 0:
            j = nxt[i][k]
            if j is None:
                break
            line_start = token_slice[i][0]
            line_end = token_slice[j - 1][1]
            line_text = " ".join(tok[2] for tok in token_slice[i:j])
            out.append((line_start, line_end, line_text, list(range(i, j))))
            i = j
            k -= 1
        return out

    boundary_token = None
    if boundary_word_index is not None:
        for idx, tok in enumerate(tokens):
            if tok[0] >= boundary_word_index:
                boundary_token = idx
                break

    if boundary_token is None:
        if target_line_count and len(tokens) <= 400:
            lines = build_lines_fixed_count(tokens, target_line_count)
        else:
            lines = build_lines(tokens)
    else:
        if target_line_count is None:
            lines = build_lines(tokens[:boundary_token]) + build_lines(tokens[boundary_token:])
        else:
            shema_lines = build_lines(tokens[:boundary_token])
            remaining = max(1, target_line_count - len(shema_lines))
            if len(tokens) <= 400:
                lines = shema_lines + build_lines_fixed_count(tokens[boundary_token:], remaining)
            else:
                lines = shema_lines + build_lines(tokens[boundary_token:])

    if not lines:
        return None

    lengths = [len(text) for _, _, text, _ in lines]
    if len(lengths) > 2:
        mid_lengths = lengths[1:-1]
    else:
        mid_lengths = lengths
    line_chars = max(1, int(round(sum(mid_lengths) / len(mid_lengths))))

    line_width_px = None
    if avg_letter_size is not None:
        line_width_px = line_chars * avg_letter_size

    line_start_estimates = []
    for item in chosen_sorted:
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

    if not line_start_estimates:
        return None
    line_start_estimates.sort()
    mid = len(line_start_estimates) // 2
    if len(line_start_estimates) % 2 == 1:
        expected_start_x = line_start_estimates[mid]
    else:
        expected_start_x = (line_start_estimates[mid - 1] + line_start_estimates[mid]) / 2.0
    expected_start_x = max(expected_start_x, max(item["x2"] for item in chosen_sorted))

    if line_width_px is None:
        return None

    line_tops = []
    for line in line_words:
        if not line:
            continue
        line_tops.append(min(w["y1"] for w in line))
    line_tops.sort()
    line_steps = [b - a for a, b in zip(line_tops, line_tops[1:]) if b > a]
    line_height = _median(line_steps)
    if not line_height:
        word_heights = [w["y2"] - w["y1"] for w in words if w.get("y2") is not None]
        line_height = _median(word_heights) or max(1.0, chosen_sorted[0]["y2"] - chosen_sorted[0]["y1"])
    line_count = len(lines)
    doc_right = expected_start_x
    doc_left = doc_right - line_width_px

    best_item = max(chosen, key=lambda item: item["score"])
    # Map best_item t_start to a line index in the final layout.
    best_word_idx = None
    for wi, (ws, we) in enumerate(word_spans):
        if we > best_item["t_start"]:
            best_word_idx = wi
            break
    lines_before = 0
    if best_word_idx is not None:
        for li, (ws, we, _, _) in enumerate(lines):
            if ws <= best_word_idx <= we:
                lines_before = li
                break
    doc_top = best_item["y1"] - (line_height * lines_before)
    if doc_top < 0:
        doc_top = 0.0
    doc_bottom = doc_top + (line_height * line_count)

    top_item = chosen_sorted[0]
    return {
        "line1_idx": top_item["line"],
        "line1_score": top_item["score"],
        "line1_span": (top_item["start"], top_item["end"]),
        "line1_bbox": (top_item["x1"], top_item["y1"], top_item["x2"], top_item["y2"]),
        "expected_start_x": expected_start_x,
        "line_chars": line_chars,
        "line_count": line_count,
        "line_height": line_height,
        "lines": lines,
        "doc_box": (doc_left, doc_top, doc_right, doc_bottom),
    }
