#!/usr/bin/env python3
"""Put two or more crop outputs side by side, with the original.

    python3 tools/compare.py new=test_output/try old=test_output/try_old
    python3 tools/compare.py region=test_output/bench/region/crops \
                             sheet=test_output/bench/sheet/crops

Each argument is `label=directory`. Directories are searched recursively, and
images are matched across them by filename stem, so it works equally on a
single-image trial and on a whole benchmark run.

Writes `current/compare.html`: one row per image, the original first and then
one column per configuration. Click any panel for the full-resolution file.
A number is a summary; two crops next to each other are the thing itself.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from sheet import BROWSER_SAFE, thumb                      # noqa: E402

DEFAULT_BENCHMARK = REPO.parent / "benchmark"
OUT = REPO / "current"


def index(directory: Path) -> dict[str, Path]:
    """stem -> file, with the _cropped / _deruled suffixes normalised away."""
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in BROWSER_SAFE | {".heic", ".heif"}:
            continue
        stem = p.stem
        for suffix in ("_cropped", "_deruled"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        found.setdefault(stem, p)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", metavar="label=path")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out", type=Path, default=OUT / "compare.html")
    ap.add_argument("--width", type=int, default=760)
    args = ap.parse_args()

    columns = []
    for item in args.dirs:
        if "=" not in item:
            raise SystemExit(f"expected label=path, got: {item}")
        label, path = item.split("=", 1)
        columns.append((label, Path(path)))

    indexes = [(label, index(path)) for label, path in columns]
    for (label, idx), (_, path) in zip(indexes, columns):
        print(f"  {label:<12} {len(idx):>4} images   {path}")

    originals = index(args.benchmark)
    stems = sorted({s for _, idx in indexes for s in idx})
    if not stems:
        raise SystemExit("nothing to compare — check the paths")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for stem in stems:
        cells = []
        src = originals.get(stem)
        if src is not None:
            rel = f"_cmp/{stem}__orig.jpg"
            if thumb(src, args.out.parent / rel, args.width):
                full = (f"../../benchmark/{src.relative_to(args.benchmark)}"
                        if src.suffix.lower() in BROWSER_SAFE else rel)
                cells.append(("מקור", rel, full))

        for label, idx in indexes:
            p = idx.get(stem)
            if p is None:
                cells.append((label, "", ""))
                continue
            rel = f"_cmp/{stem}__{label}.jpg"
            if thumb(p, args.out.parent / rel, args.width):
                cells.append((label, rel, str(p.resolve())))
            else:
                cells.append((label, "", ""))

        panels = "".join(
            f'<div class="panel"><span class="tag">{html.escape(tag)}</span>'
            + (f'<img loading="lazy" src="{s}" data-full="{f}">' if s
               else '<div class="none">—</div>')
            + "</div>"
            for tag, s, f in cells
        )
        rows.append(f'<section><h2>{html.escape(stem)}</h2>'
                    f'<div class="row" style="grid-template-columns:repeat({len(cells)},1fr)">'
                    f'{panels}</div></section>')

    labels = " · ".join(label for label, _ in columns)
    args.out.write_text(f"""<!doctype html>
<html lang="he" dir="rtl"><meta charset="utf-8"><title>השוואה — {html.escape(labels)}</title>
<style>
 :root {{ color-scheme: dark }}
 body {{ margin:0; background:#111417; color:#e8eaed;
        font:14px/1.5 -apple-system,"SF Hebrew",system-ui,sans-serif }}
 header {{ position:sticky; top:0; background:#181c20; border-bottom:1px solid #2a3138;
           padding:10px 16px; font-weight:600; z-index:5 }}
 header small {{ font-weight:400; color:#93a1ad; margin-inline-start:10px }}
 section {{ padding:14px 16px; border-bottom:1px solid #20262c }}
 h2 {{ margin:0 0 8px; font-size:13px; font-weight:600; color:#c3ced8; word-break:break-all }}
 .row {{ display:grid; gap:10px; align-items:start }}
 .panel {{ position:relative; background:#0d1013; border:1px solid #262d34; border-radius:8px;
           overflow:hidden; min-height:80px }}
 .panel img {{ width:100%; height:auto; display:block; cursor:zoom-in }}
 .tag {{ position:absolute; top:4px; inset-inline-start:4px; z-index:2; font-size:11px;
         background:#000b; padding:1px 7px; border-radius:3px; color:#cfe3f5 }}
 .none {{ color:#6b7a86; text-align:center; padding:34px 0 }}
 dialog {{ border:0; padding:0; background:#000; max-width:98vw; max-height:98vh }}
 dialog img {{ max-width:98vw; max-height:98vh; display:block }}
 dialog::backdrop {{ background:#000d }}
</style>
<header>השוואה<small>{html.escape(labels)} · {len(stems)} תמונות</small></header>
{''.join(rows)}
<dialog id="zoom"><img></dialog>
<script>
 const dlg=document.getElementById('zoom');
 document.querySelectorAll('.panel img').forEach(i=>i.onclick=()=>{{
   dlg.querySelector('img').src=i.dataset.full||i.src; dlg.showModal();
 }});
 dlg.onclick=()=>dlg.close();
</script>
</html>""", encoding="utf-8")

    print(f"\nopen {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
