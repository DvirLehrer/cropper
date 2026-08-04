#!/usr/bin/env python3
"""Sweep the remaining parameters and print one table.

    python3 tools/search.py --stage 1        # GRAIN_MIN: 5 / 8 / 12
    python3 tools/search.py --stage 2 --grain 8   # then DERULE and MIN_CHAR_PX
    python3 tools/search.py --table           # just re-print what has been run

Each combination is a crop pass followed by a scoring pass on the development
set, then a single score:

    50%  share of images the engine reported under 20 errors
    30%  share of the known reference text it recovered
    20%  share of images it could identify at all

The weights follow what the report is for. The 20-error line is what a customer
sees, so it carries most of the weight. Text recovered is what makes that
possible and is the steadiest of the three. Failing to identify the document at
all is the worst outcome but the rarest, and it is already punished twice over —
an unidentified scan scores badly on the other two by construction — so it gets
the smallest share rather than double-counting.

The three components are printed beside the score. A winner that wins on one
component alone is worth looking at rather than accepting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

RUNS = REPO / "test_output" / "bench"
DEVSET = REPO.parent / "benchmark" / "dev_set.txt"
SCORING_PY = REPO / ".venv-scoring" / "bin" / "python"

W_UNDER, W_TEXT, W_IDENT = 0.5, 0.3, 0.2


def score_of(name: str) -> dict | None:
    from bench import _join, _summarise

    run_dir = RUNS / name
    if not (run_dir / "crops.csv").exists():
        return None
    rows = _join(run_dir, REPO.parent / "benchmark")
    s = _summarise(rows).get("ALL")
    if not s or s.get("under_limit_pct") is None:
        return None
    n = s["n"]
    ident = 100.0 * (n - s["unreadable_n"]) / n
    total = (W_UNDER * s["under_limit_pct"] + W_TEXT * s["text_match_pct"]
             + W_IDENT * ident)
    return {"n": n, "under": s["under_limit_pct"], "text": s["text_match_pct"],
            "ident": round(ident, 1), "score": round(total, 1)}


def run_one(name: str, env: dict, only: list[str] | None) -> None:
    if (RUNS / name / "crops.csv").exists():
        print(f"  {name}: already run, skipping")
        return
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    crop = [sys.executable, str(REPO / "tools" / "bench.py"), "crop",
            "--run", name, "--set", str(DEVSET)]
    if only:
        crop += ["--only", *only]
    print(f"  {name}: cropping  ({' '.join(f'{k}={v}' for k, v in env.items())})")
    subprocess.run(crop, env=e, cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL)
    if not SCORING_PY.exists():
        raise SystemExit(f"scoring interpreter not found: {SCORING_PY}")
    print(f"  {name}: scoring")
    subprocess.run([str(SCORING_PY), str(REPO / "tools" / "bench.py"), "score",
                    "--run", name, "--no-raw"],
                   env=e, cwd=REPO, check=True, stdout=subprocess.DEVNULL)


def table(names: list[str]) -> None:
    print(f"\n{'run':<16}{'settings':<34}{'<20':>7}{'text':>7}{'ident':>7}{'SCORE':>8}")
    print("-" * 79)
    rows = []
    for name in names:
        s = score_of(name)
        if not s:
            continue
        f = RUNS / name / "settings.json"
        cfg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        keep = {k: cfg[k] for k in ("GRAIN_MIN", "MIN_CHAR_PX", "DERULE") if k in cfg}
        rows.append((s["score"], name, keep, s))
    for sc, name, keep, s in sorted(rows, reverse=True):
        desc = " ".join(f"{k}={v}" for k, v in keep.items())
        print(f"{name:<16}{desc:<34}{s['under']:>7}{s['text']:>7}{s['ident']:>7}{sc:>8}")
    if rows:
        best = max(rows)[1]
        print(f"\nbest: {best}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2))
    ap.add_argument("--grain", type=float, help="winning GRAIN_MIN, for stage 2")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.stage == 1:
        combos = [(f"s_g{int(g)}", {"GRAIN_MIN": g}) for g in (5, 8, 12)]
    elif args.stage == 2:
        if args.grain is None:
            raise SystemExit("--grain is required for stage 2")
        g = args.grain
        combos = [
            (f"s_g{int(g)}_c0", {"GRAIN_MIN": g, "MIN_CHAR_PX": 0}),
            (f"s_g{int(g)}_d1", {"GRAIN_MIN": g, "DERULE": 1}),
            (f"s_g{int(g)}_c0d1", {"GRAIN_MIN": g, "MIN_CHAR_PX": 0, "DERULE": 1}),
        ]
    else:
        combos = []

    for name, env in combos:
        run_one(name, env, args.only)

    names = sorted({d.name for d in RUNS.iterdir()
                    if d.is_dir() and d.name.startswith("s_")})
    table(names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
