#!/usr/bin/env python3
"""Package the three crash fixes as a zip the company can drop in or ignore.

    python3 tools/build_engine_fix.py

Produces `dist/stam-ocr-crash-fixes.zip` containing the three patched files, a
readable diff, and a note explaining what each change does and what was measured.

Deliberately narrow. Each edit only stops an exception; not one decision the
engine makes is altered, which is what makes the result verifiable: re-scoring
all 157 benchmark images twice, once patched and once not, gives 143 identical
verdicts — same faults, same letters, same places — 14 scans rescued, and none
changed. A fix that also "improved" something would have made that comparison
meaningless.

The engine's own repository is left untouched: the patch is applied to a copy,
zipped, and the copy is discarded.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from patch_engine import EDITS                              # noqa: E402

DEFAULT_ENGINE = Path.home() / "dev" / "Stam-OCR"

NOTE = """\
Three crashes in Stam-OCR
=========================

Found while benchmarking a cropper against your engine on 157 real customer
images. Fourteen of them — 9% — produced no report at all, and every one died
inside the engine rather than in the cropper. No crop can prevent them: they
depend on where a letter sits in the text, not on the quality of the picture.

  Utils/MyDiffLib.py:831  fix_touching_letter_v   KeyError: -1                8 images
  PolygonsList.py:100     union                   AttributeError: 'NoneType'  5 images
  ShowResults.py:474      fix_l_swallowing        IndexError                  1 image

All three are an index that is not there. `create_scan` catches the exception
and returns None, so the whole scan is lost — not one letter misjudged, the
entire report.

What is in this zip
-------------------
Three files, each with the smallest change that prevents the exception. Nothing
else is altered: no threshold moved, no decision changed. Anywhere the code
previously crashed it now skips that check; anywhere it did not crash it behaves
exactly as before.

What was measured
-----------------
All 157 images were scored twice against the engine, once with these changes and
once without, and compared image by image — not just the error counts but which
fault was reported on which letter in which word:

  identical verdict      143      same faults, same letters, same places
  rescued                 14
  verdict changed          0
  broken                   0

  engine returned a result   91.1%  ->  100%
  text recovered from the
  known reference text       79.5%  ->  86.1%
  rough-parchment folder     54.6%  ->  75.2%

One thing left alone
--------------------
At MyDiffLib.py:831, `i` runs over [0, 1] and says which of the two lines a
letter spans; the line numbers themselves are in `lines_idxs`. The code reads
`polygons_data.lines[i - 1]`, which asks a dict for key -1 when i is 0 — the
crash. When i is 1 it does not crash, but it reads line 0, which is not the line
the letter touches either.

That looks like it was meant to be `lines_idxs[1 - i]`. It has NOT been changed
here, because that would alter what the engine decides and there was no way to
verify the intent from outside. Worth a look by whoever wrote it.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "stam-ocr-crash-fixes.zip")
    args = ap.parse_args()

    if not (args.engine / "StamOcr.py").exists():
        raise SystemExit(f"Stam-OCR not found at {args.engine}")

    files = sorted({f for f, *_ in EDITS})

    # Only the files being patched, and only genuine modifications — build
    # artefacts like __pycache__ are untracked and irrelevant. The point of the
    # check is that the diff shipped in the zip describes their code as it
    # stands, not as some earlier experiment left it.
    status = subprocess.run(
        ["git", "-C", str(args.engine), "status", "--porcelain", "--"] + files,
        capture_output=True, text=True).stdout.splitlines()
    dirty = [ln for ln in status if not ln.startswith("??")]
    if dirty:
        raise SystemExit(
            "these engine files are modified — revert them first "
            "(tools/patch_engine.py --revert) so the diff is honest:\n"
            + "\n".join(dirty))
    tmp = Path(tempfile.mkdtemp())
    diffs = []

    try:
        for filename in files:
            src = args.engine / filename
            text = original = src.read_text(encoding="utf-8")
            for f, find, replace, _why in EDITS:
                if f != filename:
                    continue
                if find not in text:
                    raise SystemExit(f"pattern not found in {filename}; "
                                     "the engine source has changed")
                text = text.replace(find, replace, 1)
            dest = tmp / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            diffs.extend(difflib.unified_diff(
                original.splitlines(keepends=True), text.splitlines(keepends=True),
                fromfile=f"a/{filename}", tofile=f"b/{filename}"))

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
            for filename in files:
                z.write(tmp / filename, f"patched/{filename}")
            z.writestr("crash-fixes.diff", "".join(diffs))
            z.writestr("README.txt", NOTE)

        print(f"wrote {args.out}")
        for filename in files:
            print(f"   patched/{filename}")
        print("   crash-fixes.diff")
        print("   README.txt")
        print(f"\nchanged lines in the diff: "
              f"{sum(1 for d in diffs if d.startswith(('+', '-')) and not d.startswith(('+++', '---')))}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
