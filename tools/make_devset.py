#!/usr/bin/env python3
"""Pick a small, fixed development set for parameter search.

    python3 tools/make_devset.py --n 40 --from-run after

Tuning a threshold against all 157 images costs a Vision call per image and ten
minutes of scoring per candidate value, which is far too slow a loop to search
in — and most of those images tell you nothing, because they are either already
comfortable or hopeless whatever you do.

The sample is **stratified twice**: proportionally across the six challenge
folders, and within each folder across the range of outcomes measured on a
reference run — some images the engine already reads cleanly, some sitting just
above the error line, some it cannot read at all. A sample drawn only from the
hard cases would make every change look like an improvement; one drawn only
from the easy cases would hide every regression.

The result is written to a file and reused. It must not be redrawn between
experiments: a moving sample turns a comparison into a coin toss.

Search on the dev set, then confirm the winner on all 157 before believing it.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "test_output" / "bench"
DEFAULT_OUT = REPO.parent / "benchmark" / "dev_set.txt"

# Outcome bands, in errors. Sampling across these is what keeps the dev set
# representative rather than merely small.
BANDS = [
    ("clean", lambda e: e is not None and e < 20),
    ("near", lambda e: e is not None and 20 <= e < 40),
    ("poor", lambda e: e is not None and e >= 40),
    ("unread", lambda e: e is None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--from-run", default="after",
                    help="run whose results define the outcome bands")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    run_dir = args.runs_dir / getattr(args, "from_run")
    cache_path = args.runs_dir / "score_cache.jsonl"
    scores = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                scores[row["sha"]] = row

    with (run_dir / "crops.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    # (category, band) -> images
    buckets: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        s = scores.get(r["crop_sha"])
        errors = (s["letter_errors"] + s["word_errors"]) if s and s.get("ok") else None
        band = next(name for name, test in BANDS if test(errors))
        buckets.setdefault((r["category"], band), []).append(r["image"])

    rng = random.Random(args.seed)
    total = len(rows)
    picked: list[tuple[str, str]] = []
    for (category, band), images in sorted(buckets.items()):
        # Proportional to how much of the benchmark this bucket represents,
        # always at least one so no failure mode disappears entirely.
        want = max(1, round(args.n * len(images) / total))
        images = sorted(images)
        rng.shuffle(images)
        picked.extend((category, im) for im in images[:want])

    picked.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# Fixed development set for parameter search — do not redraw.\n"
        f"# {len(picked)} of {total} images, stratified by challenge and by outcome.\n"
        + "".join(f"{c}/{i}\n" for c, i in picked),
        encoding="utf-8",
    )

    from collections import Counter
    by_cat = Counter(c for c, _ in picked)
    print(f"wrote {args.out}  ({len(picked)} of {total} images)")
    for cat, n in sorted(by_cat.items()):
        whole = sum(1 for r in rows if r["category"] == cat)
        print(f"   {cat:<16} {n:>3} of {whole:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
