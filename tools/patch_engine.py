#!/usr/bin/env python3
"""Temporarily stop three crashes in the scoring engine, to see what is behind them.

    python3 tools/patch_engine.py --apply
    python3 tools/patch_engine.py --revert
    python3 tools/patch_engine.py --status

Fourteen of the 157 benchmark images — a third of the rough-parchment folder —
produce no report at all, and every one of them dies inside Stam-OCR rather than
in our code. Three distinct faults, all of them an index that is not there:

  MyDiffLib.py    fix_touching_letter_v   KeyError: -1                 8 images
  PolygonsList.py union                   AttributeError: 'NoneType'   5 images
  ShowResults.py  fix_l_swallowing        IndexError                   1 image

The question this answers is narrow: if those crashes did not happen, would the
images be readable? That decides whether the remaining gap is ours to close with
better cropping or theirs to close with a guard.

These are **not** fixes. Each edit only prevents the crash, leaving every
decision the code makes exactly as it was — because the intent behind
`lines[i - 1]` is genuinely unclear and guessing at it would answer a different
question. Reverting restores the files from git, so nothing is left behind.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_ENGINE = Path.home() / "dev" / "Stam-OCR"

# (file, exact text to find, replacement, why)
EDITS = [
    (
        "Utils/MyDiffLib.py",
        "for idx, poly in enumerate(scan_data.polygons_data.lines[i - 1]):",
        "for idx, poly in enumerate(scan_data.polygons_data.lines.get(i - 1, [])):",
        "i indexes the pair of lines a letter spans, 0 or 1 — not a line number, "
        "which lines_idxs holds. lines[-1] on a dict raises. Reading it as .get "
        "keeps the behaviour and skips the lookup when the key is absent.",
    ),
    (
        "Utils/MyDiffLib.py",
        "prev_poly = scan_data.polygons_data.lines[i - 1][idx - 1]\n"
        "                                                        next_poly = "
        "scan_data.polygons_data.lines[i - 1][idx + 1]",
        "_row = scan_data.polygons_data.lines.get(i - 1, [])\n"
        "                                                        if not (0 < idx < len(_row) - 1):\n"
        "                                                            continue\n"
        "                                                        prev_poly = _row[idx - 1]\n"
        "                                                        next_poly = _row[idx + 1]",
        "idx-1 and idx+1 run off the ends for the first and last polygon in a row.",
    ),
    (
        "PolygonsList.py",
        "        new_contours = poly1.contours + poly2.contours",
        "        if poly1 is None or poly2 is None:\n"
        "            return None   # index already removed by an earlier union\n"
        "        new_contours = poly1.contours + poly2.contours",
        "union is called with an index whose polygon a previous union already "
        "deleted, so get_by_index returns None.",
    ),
    (
        "ShowResults.py",
        "                                right_letter = line.sub_elements[word_num - 1].sub_elements[-1]",
        "                                _prev = line.sub_elements[word_num - 1].sub_elements\n"
        "                                right_letter = _prev[-1] if _prev else None",
        "the preceding word can hold no letters, and [-1] on it raises.",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--revert", action="store_true")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    args = ap.parse_args()

    root: Path = args.engine
    if not (root / "StamOcr.py").exists():
        raise SystemExit(f"Stam-OCR not found at {root}")

    files = sorted({f for f, *_ in EDITS})

    if args.status or args.revert:
        dirty = subprocess.run(["git", "-C", str(root), "status", "--short"] + files,
                               capture_output=True, text=True).stdout.strip()
        if args.status:
            print(f"engine: {root}")
            print("modified:\n" + (dirty or "  (clean — no patch applied)"))
            return 0
        if not dirty:
            print("nothing to revert; the engine is already clean")
            return 0
        subprocess.run(["git", "-C", str(root), "checkout", "--"] + files, check=True)
        print("reverted:\n  " + "\n  ".join(files))
        return 0

    # --apply
    dirty = subprocess.run(["git", "-C", str(root), "status", "--short"] + files,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("these files are already modified — revert first so the experiment "
              "starts from a known state:\n" + dirty, file=sys.stderr)
        return 1

    for filename, find, replace, why in EDITS:
        path = root / filename
        text = path.read_text(encoding="utf-8")
        if find not in text:
            print(f"  !! pattern not found in {filename} — the file has changed. "
                  f"Nothing applied.", file=sys.stderr)
            subprocess.run(["git", "-C", str(root), "checkout", "--"] + files)
            return 1
        path.write_text(text.replace(find, replace, 1), encoding="utf-8")
        print(f"  {filename}\n     {why}")

    print("\napplied. Re-score, then revert with --revert.")
    print("Nothing here changes what the engine decides — only whether it survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
