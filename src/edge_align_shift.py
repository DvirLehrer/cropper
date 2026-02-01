#!/usr/bin/env python3
"""Line-sweep alignment with shift-on-crossing rule."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class AlignResult:
    y: List[float]
    line_x: float
    quad: float
    alterations: int
    objective: float


def _quad_sum(s1: float, s2: float, n: int, line_x: float) -> float:
    if n == 0:
        return 0.0
    mean_s1 = s1 / n
    return (s2 / n) - (2.0 * line_x * mean_s1) + (line_x * line_x)


def _choose_best_fit(cands: Sequence[float], target: float) -> float:
    if not cands:
        return 0.0
    idx = bisect_left(cands, target)
    if idx <= 0:
        return cands[0]
    if idx >= len(cands):
        return cands[-1]
    left = cands[idx - 1]
    right = cands[idx]
    return left if (target - left) <= (right - target) else right


def _group_by_left_edge(order: List[int], left_edges: Sequence[float]) -> Iterable[List[int]]:
    if not order:
        return
    group = [order[0]]
    current = left_edges[order[0]]
    for idx in order[1:]:
        if left_edges[idx] == current:
            group.append(idx)
        else:
            yield group
            group = [idx]
            current = left_edges[idx]
    yield group


def align_edges_shift(
    candidates: List[List[float]],
    left_edges: List[float],
    penalty_lambda: float = 200.0,
    normalize_penalty: bool = False,
    strategy: str = "next",
    verbose: bool = False,
) -> AlignResult:
    """
    Line-sweep alignment from line_sweep_alignment.md.
    - candidates[i] is strictly increasing; candidates[i][0] is the default edge.
    - crossing left edge forces a shift to candidates[i][k], k>=1.
    - objective: Q(line_x) + lambda * A(line_x) (optionally normalized by n).
    """
    n = len(candidates)
    if n == 0:
        return AlignResult([], 0.0, 0.0, 0, 0.0)
    if len(left_edges) != n:
        raise ValueError("left_edges must have the same length as candidates")

    order = sorted(range(n), key=lambda i: left_edges[i])
    y = [c[0] if c else 0.0 for c in candidates]
    s1 = sum(y)
    s2 = sum(v * v for v in y)
    line_x = left_edges[order[0]] if n else 0.0
    quad = _quad_sum(s1, s2, n, line_x)
    alterations = 0
    penalty_scale = (1.0 / n) if normalize_penalty else 1.0
    objective = quad + penalty_lambda * penalty_scale * alterations

    best = AlignResult(y=list(y), line_x=line_x, quad=quad, alterations=alterations, objective=objective)

    prev_quad = quad
    prev_obj = objective
    prev_alter = alterations

    for group in _group_by_left_edge(order, left_edges):
        line_x = left_edges[group[0]]
        # If any box in this event has no shifted candidate, crossing is invalid.
        if any(len(candidates[i]) < 2 for i in group):
            if verbose:
                blocked = sum(1 for i in group if len(candidates[i]) < 2)
                print(
                    f"sweep stop at line_x={line_x:.3f} "
                    f"(no shifted candidate; blocked={blocked})"
                )
            break

        for i in group:
            old = y[i]
            if strategy == "best_fit":
                target = line_x
                shifted_cands = candidates[i][1:]
                new = _choose_best_fit(shifted_cands, target)
            else:
                new = candidates[i][1]

            if verbose:
                edge_val = left_edges[i]
                print(f"shift i={i} edge={edge_val:.3f} {old:.3f}->{new:.3f}")
            y[i] = new
            s1 += new - old
            s2 += (new * new) - (old * old)

        alterations += len(group)
        quad = _quad_sum(s1, s2, n, line_x)
        objective = quad + penalty_lambda * penalty_scale * alterations

        if verbose:
            gain = prev_quad - quad
            loss = alterations - prev_alter
            delta_obj = prev_obj - objective
            print(
                f"event line_x={line_x:.3f} "
                f"gain={gain:.3f} loss={loss} "
                f"lost_by_shift={alterations} quad={quad:.3f} obj={objective:.3f} "
                f"delta_obj={delta_obj:.3f}"
            )
            prev_quad = quad
            prev_obj = objective
            prev_alter = alterations
        if objective < best.objective:
            best = AlignResult(y=list(y), line_x=line_x, quad=quad, alterations=alterations, objective=objective)

    return best


def align_edges_shift_side(
    side: str,
    candidates: List[List[float]],
    edges: List[float],
    penalty_lambda: float = 200.0,
    normalize_penalty: bool = False,
    strategy: str = "next",
    verbose: bool = False,
) -> AlignResult:
    """
    Generalized sweep for left/right/top/bottom.
    - left/top: same as align_edges_shift (edges increasing).
    - right/bottom: mirror coordinates (x'=-x or y'=-y) and reuse the sweep.
      For these sides, candidates should be strictly decreasing in original space.
    """
    if side not in {"left", "right", "top", "bottom"}:
        raise ValueError("side must be one of: left, right, top, bottom")

    if side in {"left", "top"}:
        return align_edges_shift(
            candidates,
            edges,
            penalty_lambda=penalty_lambda,
            normalize_penalty=normalize_penalty,
            strategy=strategy,
            verbose=verbose,
        )

    cand_neg = [[-x for x in c] for c in candidates]
    edges_neg = [-x for x in edges]
    res = align_edges_shift(
        cand_neg,
        edges_neg,
        penalty_lambda=penalty_lambda,
        normalize_penalty=normalize_penalty,
        strategy=strategy,
        verbose=verbose,
    )
    return AlignResult(
        y=[-v for v in res.y],
        line_x=-res.line_x,
        quad=res.quad,
        alterations=res.alterations,
        objective=res.objective,
    )


def _format_result(res: AlignResult) -> str:
    return (
        f"y={res.y}\n"
        f"line_x={res.line_x:.3f}\n"
        f"quad={res.quad:.3f}\n"
        f"alterations={res.alterations}\n"
        f"objective={res.objective:.3f}"
    )


def main() -> None:
    # Example usage with a tiny synthetic input.
    candidates = [
        [100, 110, 120],
        [102, 115],
        [98, 105, 130],
        [101, 112],
    ]
    left_edges = [c[0] for c in candidates]
    res = align_edges_shift(candidates, left_edges, strategy="next")
    print(_format_result(res))


if __name__ == "__main__":
    main()
