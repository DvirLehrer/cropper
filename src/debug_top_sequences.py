#!/usr/bin/env python3
"""Debug top sequences (top 25% with min 3) on selected benchmark images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from best_sequence import find_best_sequences
from cropper_config import BENCHMARK_DIR, DEBUG_DIR, LANG, OCR_TEXT_DIR, TEXT_DIR, TYPES_CSV
from ocr_utils import load_types, ocr_image
from target_texts import load_target_texts


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


def _resolve_image_path(name: str) -> Path:
    path = BENCHMARK_DIR / name
    if path.exists():
        return path
    stem = Path(name).stem
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = BENCHMARK_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found in benchmark/: {name}")


def _process_image(path: Path) -> None:
    print(f"processing: {path.name}")

    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)
    image_type = type_map.get(path.name)
    target = (
        target_texts["m"]
        if image_type == "m"
        else target_texts[
            {
                "s": "shema",
                "v": "vehaya",
                "k": "kadesh",
                "p": "peter",
            }[image_type]
        ]
    )

    window_words = 6
    max_skip = 6

    result = ocr_image(path, lang=LANG)
    words = result["words"]
    line_words = result["line_words"]
    ocr_text_dir = OCR_TEXT_DIR
    ocr_text_dir.mkdir(parents=True, exist_ok=True)
    text_out = ocr_text_dir / f"{path.stem}.txt"
    text_out.write_text(result["text"], encoding="utf-8")

    scored = find_best_sequences(
        line_words,
        target,
        window_words=window_words,
        max_skip=max_skip,
        top_k=10**6,
    )
    if not scored:
        raise SystemExit("no sequences found")

    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    keep_n = max(3, int(len(scored) * 0.30))
    score_keep = [item for item in scored if item[0] >= 0.7]
    keep_n = max(keep_n, len(score_keep))
    kept = scored[:keep_n]
    line_width_px = None

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if words:
        left = min(w["x1"] for w in words)
        right = max(w["x2"] for w in words)
        top = min(w["y1"] for w in words)
        bottom = max(w["y2"] for w in words)
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
        for w in words:
            draw.rectangle([w["x1"], w["y1"], w["x2"], w["y2"]], outline=(0, 200, 0), width=2)

    target_words = [w for w in target.split() if w]
    word_lens = [len(w) for w in target_words]
    prefix = [0] * (len(word_lens) + 1)
    for k, l in enumerate(word_lens):
        prefix[k + 1] = prefix[k] + l
    from best_sequence import _best_word_segment_distance_anywhere as _best_seg

    chosen = []
    for rank, (score, idx, (i, j)) in enumerate(kept, start=1):
        span_words = sorted(line_words[idx], key=lambda w: -w["x2"])[i : j + 1]
        x1 = min(w["x1"] for w in span_words)
        y1 = min(w["y1"] for w in span_words)
        x2 = max(w["x2"] for w in span_words)
        y2 = max(w["y2"] for w in span_words)
        draw.rectangle([x1, y1, x2, y2], outline=(160, 0, 255), width=3)
        ocr_text = " ".join(w["text"] for w in span_words)
        dist, segment = _best_seg(ocr_text, target_words, word_lens, prefix)
        t_start, t_end, t_sub, t_dist = _best_char_substring(target, ocr_text)
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
                "rank": rank,
                "line": idx,
                "score": score,
                "ocr": ocr_text,
                "target": segment,
                "dist": dist,
                "t_start": t_start,
                "t_end": t_end,
                "t_sub": t_sub,
                "t_dist": t_dist,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "letter_size": letter_size,
            }
        )

    chosen_sorted = sorted(chosen, key=lambda item: item["y1"])
    print("chosen sequences by score:")
    print("chosen scores:", ", ".join(f"{item['score']:.4f}" for item in chosen))
    for item in chosen:
        print(f"{item['rank']:02d}. line={item['line']} score={item['score']:.4f}")
        print(f"  ocr: {_display_hebrew(item['ocr'])}")
        print(f"  target: {_display_hebrew(item['target'])}")
        print(f"  dist: {item['dist']}")

    print("gap-based line length estimates:")
    weighted_sum = 0.0
    weight_total = 0.0
    size_weighted_sum = 0.0
    size_weight_total = 0.0
    gap_line_lengths = []
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
        gap_text = target[top_pos:bottom_pos]
        delta_lines = bottom["line"] - top["line"]
        if delta_lines <= 0:
            continue
        line_len = len(gap_text) / delta_lines
        weight = (top["score"] + bottom["score"]) / 2.0
        weighted_sum += line_len * weight
        weight_total += weight
        size_weighted_sum += ((top["letter_size"] + bottom["letter_size"]) / 2.0) * weight
        size_weight_total += weight
        gap_line_lengths.append(line_len)
        print(
            f"  top_line={top['line']} bottom_line={bottom['line']} "
            f"gap_chars={len(gap_text)} delta_lines={delta_lines} "
            f"line_len={line_len:.2f} weight={weight:.4f}"
        )
        print(f"    gap_text: {_display_hebrew(gap_text)}")
        print(f"    assumed_lines: {delta_lines}")
    line_chars = None
    avg_letter_size = None
    if weight_total > 0:
        weighted_avg = weighted_sum / weight_total
        print(f"weighted avg line length (initial): {weighted_avg:.2f}")
        if size_weight_total > 0:
            avg_letter_size = size_weighted_sum / size_weight_total
            print(f"avg letter size (px/char): {avg_letter_size:.3f}")
        line_chars = max(1, int(round(weighted_avg)))
        print(f"line length chars (initial rounded): {line_chars}")

    # Fallback to OCR line count when gap quality is weak.
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
        print(
            "gap stats: "
            f"count={len(gap_line_lengths)} "
            f"mean={mean_len:.2f} "
            f"min={min(gap_line_lengths):.2f} "
            f"max={max(gap_line_lengths):.2f} "
            f"stdev={(stdev if len(gap_line_lengths) > 1 else 0.0):.2f}"
        )
    else:
        gap_quality_weak = True
        print("gap stats: count=0")

    ocr_line_count = len([line for line in line_words if line])
    target_line_count = None
    if gap_quality_weak and ocr_line_count > 0:
        target_line_count = ocr_line_count
        print(f"line count constraint (ocr fallback): {target_line_count}")

    if line_chars:
        words_all = target.split()
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
        anchor_ranges = [(a["w_start"], a["w_end"]) for a in anchors]

        def is_inside_anchor(idx: int) -> bool:
            for a_start, a_end in anchor_ranges:
                if a_start <= idx <= a_end:
                    return True
            return False

        print("target lines (anchors intact, balanced mid lines):")
        # Hard rule: Shema and Vehaya cannot share the same line (mezuzah only).
        words_all = target.split()
        boundary_word = None
        if image_type == "m":
            vehaya_text = target_texts.get("vehaya", "")
            vehaya_words = vehaya_text.split()
            shema_text = target_texts.get("shema", "")
            shema_words = shema_text.split()
            boundary_start = None
            if vehaya_words:
                first = vehaya_words[0]
                search_from = max(0, len(shema_words) - 2)
                for idx in range(search_from, len(words_all)):
                    if words_all[idx] == first:
                        boundary_start = idx
                        break
            if boundary_start is None:
                boundary_start = len(shema_words)
            boundary_word = boundary_start - 1 if boundary_start > 0 else None

        tokens = []
        i = 0
        boundary_start = boundary_word + 1 if boundary_word is not None else None
        while i < len(words_all):
            if i in anchor_end_by_start:
                end = anchor_end_by_start[i]
                if boundary_start is not None and i < boundary_start <= end:
                    # Split anchor token at boundary to enforce Vehaya on new line.
                    left_text = " ".join(words_all[i:boundary_start])
                    right_text = " ".join(words_all[boundary_start : end + 1])
                    if left_text:
                        tokens.append((i, boundary_start - 1, left_text, len(left_text), True))
                    if right_text:
                        tokens.append((boundary_start, end, right_text, len(right_text), True))
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
                        if length > line_chars:
                            continue
                        cost = 0
                    else:
                        cost = (length - line_chars) ** 2
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
                            if line_chars is not None and length > line_chars:
                                continue
                            cost = 0
                        else:
                            cost = 0 if line_chars is None else (length - line_chars) ** 2
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

        # Force split tokens at the Shema/Vehaya boundary.
        if boundary_start is not None:
            split_tokens = []
            for tok in tokens:
                s, e, text, tlen, is_anchor = tok
                if s < boundary_start <= e:
                    left_text = " ".join(words_all[s:boundary_start])
                    right_text = " ".join(words_all[boundary_start : e + 1])
                    if left_text:
                        split_tokens.append((s, boundary_start - 1, left_text, len(left_text), is_anchor))
                    if right_text:
                        split_tokens.append((boundary_start, e, right_text, len(right_text), is_anchor))
                else:
                    split_tokens.append(tok)
            tokens = split_tokens

        # Force a new line at Vehaya: build Shema lines then Vehaya lines.
        boundary_token = None
        if boundary_start is not None:
            for idx, tok in enumerate(tokens):
                if tok[0] >= boundary_start:
                    boundary_token = idx
                    break
        if boundary_token is None:
            if target_line_count is None:
                lines = build_lines(tokens)
            else:
                lines = build_lines_fixed_count(tokens, target_line_count)
        else:
            # Ensure Vehaya starts a new line: exclude the boundary token from Shema lines.
            if target_line_count is None:
                lines = build_lines(tokens[:boundary_token]) + build_lines(tokens[boundary_token:])
            else:
                shema_lines = build_lines(tokens[:boundary_token])
                remaining = target_line_count - len(shema_lines)
                lines = shema_lines + build_lines_fixed_count(tokens[boundary_token:], max(1, remaining))

        for line_idx, (_, _, text, _) in enumerate(lines):
            print(f"  line {line_idx:02d}: {_display_hebrew(text)}")

        print("anchor line index vs OCR (soft):")
        for a in anchors:
            for li, (ws, we, _, _) in enumerate(lines):
                if a["w_start"] >= ws and a["w_end"] <= we:
                    print(f"  anchor words {a['w_start']}-{a['w_end']} -> line {li:02d} (ocr line {a['line']})")
                    break

        # Recompute line length after final line arrangement.
        lengths = [len(text) for _, _, text, _ in lines]
        if lengths:
            if len(lengths) > 2:
                mid_lengths = lengths[1:-1]
            else:
                mid_lengths = lengths
            line_chars = max(1, int(round(sum(mid_lengths) / len(mid_lengths))))
            print(f"line length chars (recomputed): {line_chars}")
            if avg_letter_size is not None:
                line_width_px = line_chars * avg_letter_size
                print(f"line width (px, recomputed): {line_width_px:.1f}")

        # Estimate line start (right edge) from chosen spans.
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
        if line_start_estimates:
            line_start_estimates.sort()
            mid = len(line_start_estimates) // 2
            if len(line_start_estimates) % 2 == 1:
                line_start_x = line_start_estimates[mid]
            else:
                line_start_x = (line_start_estimates[mid - 1] + line_start_estimates[mid]) / 2.0
            max_right = max(item["x2"] for item in chosen_sorted)
            line_start_x = max(line_start_x, max_right)
            print(f"line start x (estimated): {line_start_x:.1f}")
        else:
            line_start_x = None

    out_path = DEBUG_DIR / f"{path.stem}_debug.png"
    if line_start_x is not None:
        x = int(round(line_start_x))
        x = max(0, min(x, image.width - 1))
        draw.line([(x, 0), (x, image.height - 1)], fill=(255, 128, 0), width=2)
    if line_width_px and line_start_x is not None:
        y_mid = image.height // 2
        lx1 = int(round(line_start_x - line_width_px))
        lx1 = max(0, min(lx1, image.width - 1))
        lx2 = int(round(line_start_x))
        draw.line([(lx1, y_mid), (lx2, y_mid)], fill=(255, 200, 0), width=3)
    image.save(out_path)
    print(f"saved: {out_path}")


def main() -> None:
    names = [
        "mezuza2",
        "mezuza3",
        "אתגר הקרופר 9",
    ]
    for name in names:
        path = _resolve_image_path(name)
        _process_image(path)


if __name__ == "__main__":
    main()
