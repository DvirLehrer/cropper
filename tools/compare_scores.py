#!/usr/bin/env python3
"""Compare two scoring passes image by image, to prove nothing else moved.

    python3 tools/compare_scores.py \
        before=test_output/bench/score_cache.jsonl \
        after=test_output/bench/patched.jsonl

Written for one question: when the engine is changed so that it stops crashing,
does it still find the same scribal errors on everything that never crashed?

"It no longer fails" is not the standard. The product exists to report faults in
a scribe's work, so a change that rescues fourteen images while quietly altering
the verdict on the other hundred and forty-three would be a bad trade, and an
aggregate that went up would hide it.

So every image is compared on its own: the text recovered, the error count, and
the count of each individual error type. Anything that differs is listed. The
only differences that should appear are the images that used to crash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        raise SystemExit(f"no such cache: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["sha"]] = row          # append-only: the later line wins
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs=2, metavar="label=path")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    labels, paths = [], []
    for item in args.caches:
        if "=" not in item:
            raise SystemExit(f"expected label=path, got: {item}")
        label, path = item.split("=", 1)
        labels.append(label)
        paths.append(Path(path))
    (la, lb), (a, b) = labels, [load(p) for p in paths]

    shared = sorted(set(a) & set(b))
    print(f"{la}: {len(a)} records   {lb}: {len(b)} records   shared: {len(shared)}\n")

    rescued, broken, changed, identical = [], [], [], 0
    for sha in shared:
        x, y = a[sha], b[sha]
        if not x.get("ok") and y.get("ok"):
            rescued.append(sha)
            continue
        if x.get("ok") and not y.get("ok"):
            broken.append((sha, y.get("error", "")))
            continue
        if not x.get("ok") and not y.get("ok"):
            continue
        diff = {}
        for field in ("text_match_pct", "letter_errors", "word_errors",
                      "read_letters", "ref_letters", "text_type"):
            if x.get(field) != y.get(field):
                diff[field] = (x.get(field), y.get(field))
        types = set(x.get("errors_by_type") or {}) | set(y.get("errors_by_type") or {})
        for t in sorted(types):
            u = (x.get("errors_by_type") or {}).get(t, 0)
            v = (y.get("errors_by_type") or {}).get(t, 0)
            if u != v:
                diff[t] = (u, v)

        # Which faults, not how many of each: a touching-letter that moved from
        # one word to another leaves every count identical while telling the
        # scribe to look somewhere else entirely.
        fa, fb = set(x.get("error_fingerprint") or []), set(y.get("error_fingerprint") or [])
        if (x.get("error_fingerprint") is not None
                and y.get("error_fingerprint") is not None and fa != fb):
            gone, new = sorted(fa - fb), sorted(fb - fa)
            if gone or new:
                diff["_faults"] = (f"{len(gone)} dropped", f"{len(new)} added")
                for item in gone[:4]:
                    diff[f"    dropped {item}"] = ("", "")
                for item in new[:4]:
                    diff[f"    added   {item}"] = ("", "")
        if diff:
            changed.append((sha, diff))
        else:
            identical += 1

    print(f"  identical verdict      {identical}")
    print(f"  rescued by {lb:<12} {len(rescued)}")
    print(f"  broken by {lb:<13} {len(broken)}")
    print(f"  VERDICT CHANGED        {len(changed)}"
          + ("   <- these need explaining" if changed else "   <- nothing else moved"))

    for sha, err in broken[: args.show]:
        print(f"\n  broken  {sha[:12]}  {err[:90]}")
    for sha, diff in changed[: args.show]:
        print(f"\n  changed {sha[:12]}")
        for k, (u, v) in sorted(diff.items()):
            print(f"      {k:<28} {u!r:>12}  ->  {v!r}")
    if len(changed) > args.show:
        print(f"\n  ... and {len(changed) - args.show} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
