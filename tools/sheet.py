#!/usr/bin/env python3
"""Build `current/` — a single always-overwritten view of the whole benchmark.

    python3 tools/sheet.py                 # rebuild from the most recent run
    python3 tools/sheet.py --run baseline  # or from a named one

Produces:

    current/
      index.html      open this
      _thumbs/        side-by-side thumbnails, regenerated every time
      crops/          the current crop of every benchmark image, full size

There is exactly one `current/`. Every rebuild overwrites it, so whatever is in
there is the latest state of the pipeline — no run names to remember, no stale
copies to compare by accident. `bench.py crop` refreshes it while it works, so
opening index.html mid-run shows how far it has got.

The page is plain files on disk with relative image paths: it opens over
file:// with no server, and the whole folder can be zipped and sent to someone
else as-is.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

Image.MAX_IMAGE_PIXELS = None

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = REPO.parent / "benchmark"
CURRENT = REPO / "current"
RUNS_DIR = REPO / "test_output" / "bench"

# Cap the LONG edge, not the width. Capping width punishes landscape images:
# a crop that was rotated 90° puts its text lines along the width, so a
# width-capped thumbnail gives it far fewer pixels per line than the portrait
# original it came from — which is exactly the case we most need to eyeball.
THUMB_LONG = 720

BROWSER_SAFE = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

CATEGORY_LABELS = {
    "01_cropper": "אתגר הקרופר",
    "02_rotate": "אתגר הרוטייט",
    "03_perspective": "אתגר הפרספקטיבה",
    "04_objects": "אתגר העצמים",
    "05_drawings": "אתגר השרטוט",
    "06_roughness": "אתגר החספוס",
    "00_general": "כללי",
}


def thumb(src: Path, dest: Path, long_edge: int = THUMB_LONG) -> bool:
    """Write a JPEG preview whose longest side is at most `long_edge`.

    Returns False if src won't decode.
    """
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            longest = max(im.size)
            if longest > long_edge:
                scale = long_edge / longest
                im = im.resize((max(1, round(im.width * scale)),
                                max(1, round(im.height * scale))), Image.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, "JPEG", quality=82)
        return True
    except Exception:
        return False


def _read_rows(run_dir: Path) -> list[dict]:
    """Prefer the incremental jsonl so a running sweep is visible mid-flight."""
    jsonl = run_dir / "crops.jsonl"
    if jsonl.exists():
        rows = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if rows:
            return rows
    csv_path = run_dir / "crops.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig") as fh:
            return [
                {**r,
                 "ok": r["ok"] == "True",
                 "rotated": r["rotated"] == "True",
                 "seconds": float(r["seconds"] or 0),
                 "area_kept_pct": float(r["area_kept_pct"] or 0)}
                for r in csv.DictReader(fh)
            ]
    return []


def _read_scores(run_dir: Path) -> dict[str, dict]:
    cache = run_dir.parent / "score_cache.jsonl"
    scores: dict[str, dict] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                scores[row["sha"]] = row
    return scores


def build(run_dir: Path, benchmark: Path, out: Path = CURRENT) -> Path:
    rows = _read_rows(run_dir)
    if not rows:
        raise SystemExit(f"nothing to show — no crops recorded in {run_dir}")
    scores = _read_scores(run_dir)

    thumbs = out / "_thumbs"
    crops_out = out / "crops"
    for stale in (thumbs, crops_out):
        shutil.rmtree(stale, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    cards: list[dict] = []
    for r in rows:
        category = r["category"]
        name = r["image"]
        src = benchmark / category / name
        stem = Path(name).stem

        before = f"_thumbs/{category}/{stem}__before.jpg"
        has_before = thumb(src, out / before)

        # Zoom targets are the real files, not the previews. A JPEG/PNG original
        # is referenced in place out of the benchmark folder — no copy, no size
        # limit. HEIC has to be transcoded, because no browser will render it.
        if src.suffix.lower() in BROWSER_SAFE:
            before_full = f"../../benchmark/{category}/{name}"
        else:
            before_full = f"_full/{category}/{stem}__before.jpg"
            if not thumb(src, out / before_full, long_edge=2000):
                before_full = before

        after = after_full = ""
        if r.get("ok") and r.get("crop_path"):
            crop_src = run_dir / r["crop_path"]
            if crop_src.exists():
                kept = crops_out / category / crop_src.name
                kept.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(crop_src, kept)
                after = f"_thumbs/{category}/{stem}__after.jpg"
                if thumb(crop_src, out / after):
                    after_full = f"crops/{category}/{crop_src.name}"
                else:
                    after = ""

        score = scores.get(r.get("crop_sha", ""), {})
        recovery = None
        ref = score.get("ref_letters") or 0
        if score.get("ok") and ref:
            read = score.get("read_letters") or 0
            recovery = round(100 * max(0, read - (score.get("letter_errors") or 0)) / ref, 1)

        cards.append({
            "category": category,
            "name": name,
            "before": before if has_before else "",
            "before_full": before_full if has_before else "",
            "after": after,
            "after_full": after_full,
            "ok": bool(r.get("ok")),
            "rotated": bool(r.get("rotated")),
            "seconds": round(float(r.get("seconds") or 0), 2),
            "area": r.get("area_kept_pct") or 0,
            "size": f"{r.get('in_w')}×{r.get('in_h')} → {r.get('out_w')}×{r.get('out_h')}"
                    if r.get("ok") else f"{r.get('in_w')}×{r.get('in_h')}",
            "recovery": recovery,
            "text_type": score.get("text_type", ""),
            "error": r.get("error") or score.get("error", ""),
        })

    (out / "index.html").write_text(_render(cards, run_dir), encoding="utf-8")
    return out / "index.html"


def _render(cards: list[dict], run_dir: Path) -> str:
    counts: dict[str, list[int]] = {}
    for c in cards:
        agg = counts.setdefault(c["category"], [0, 0])
        agg[0] += 1
        agg[1] += c["ok"]

    chips = ['<button class="chip on" data-cat="all">הכל '
             f'<b>{len(cards)}</b></button>']
    for cat, (n, ok) in sorted(counts.items()):
        label = CATEGORY_LABELS.get(cat, cat)
        chips.append(f'<button class="chip" data-cat="{cat}">{label} '
                     f'<b>{ok}/{n}</b></button>')
    chips.append('<button class="chip warn" data-cat="fail">נכשלו '
                 f'<b>{sum(1 for c in cards if not c["ok"])}</b></button>')
    chips.append('<button class="chip warn" data-cat="rot">סובבו '
                 f'<b>{sum(1 for c in cards if c["rotated"])}</b></button>')

    items = []
    for c in cards:
        flags = []
        if c["rotated"]:
            flags.append('<span class="flag rot">סובב</span>')
        if not c["ok"]:
            flags.append('<span class="flag bad">נכשל</span>')
        if c["recovery"] is not None:
            cls = "good" if c["recovery"] >= 80 else "mid" if c["recovery"] >= 50 else "bad"
            flags.append(f'<span class="flag {cls}">שיחוור {c["recovery"]}%</span>')
        if c["text_type"]:
            flags.append(f'<span class="flag">{html.escape(c["text_type"])}</span>')

        after_cell = (f'<img loading="lazy" src="{c["after"]}" data-full="{c["after_full"]}">'
                      if c["after"]
                      else f'<div class="none">{html.escape(c["error"] or "אין פלט")}</div>')

        items.append(f"""
<figure class="card" data-cat="{c['category']}" data-ok="{int(c['ok'])}"
        data-rot="{int(c['rotated'])}" data-rec="{c['recovery'] if c['recovery'] is not None else -1}">
  <figcaption>
    <span class="name">{html.escape(c['name'])}</span>
    <span class="meta">{CATEGORY_LABELS.get(c['category'], c['category'])} · {c['size']} · {c['seconds']}s · {c['area']}% שטח</span>
    <span class="flags">{''.join(flags)}</span>
  </figcaption>
  <div class="pair">
    <div class="side"><span class="tag">מקור</span><img loading="lazy" src="{c['before']}" data-full="{c['before_full']}"></div>
    <div class="side"><span class="tag">אחרי</span>{after_cell}</div>
  </div>
</figure>""")

    built = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="he" dir="rtl"><meta charset="utf-8">
<title>מצב הקרופר — {html.escape(run_dir.name)}</title>
<style>
 :root {{ color-scheme: dark }}
 body {{ margin:0; background:#111417; color:#e8eaed;
        font:14px/1.5 -apple-system,"SF Hebrew",system-ui,sans-serif }}
 header {{ position:sticky; top:0; z-index:5; background:#181c20;
           border-bottom:1px solid #2a3138; padding:12px 16px }}
 h1 {{ margin:0 0 8px; font-size:15px; font-weight:600 }}
 h1 small {{ font-weight:400; color:#93a1ad; margin-inline-start:8px }}
 .chips {{ display:flex; flex-wrap:wrap; gap:6px }}
 .chip {{ background:#222a31; color:#c8d2da; border:1px solid #2f3941;
          border-radius:999px; padding:4px 12px; font:inherit; font-size:13px; cursor:pointer }}
 .chip.on {{ background:#2b6cb0; border-color:#2b6cb0; color:#fff }}
 .chip.warn {{ border-color:#7a4a2a }}
 .chip b {{ color:#8fa6b8; margin-inline-start:4px }}
 .chip.on b {{ color:#cfe3f5 }}
 main {{ padding:16px; display:grid; gap:16px;
         grid-template-columns:repeat(auto-fill,minmax(560px,1fr)) }}
 .card {{ margin:0; background:#171b1f; border:1px solid #262d34; border-radius:10px;
          overflow:hidden }}
 .card.hide {{ display:none }}
 figcaption {{ padding:8px 10px; border-bottom:1px solid #262d34;
               display:flex; flex-direction:column; gap:3px }}
 .name {{ font-weight:600; word-break:break-all }}
 .meta {{ color:#8b98a5; font-size:12px }}
 .flags {{ display:flex; gap:5px; flex-wrap:wrap; margin-top:2px }}
 .flag {{ font-size:11px; padding:1px 7px; border-radius:4px;
          background:#242c33; color:#a8b6c2 }}
 .flag.rot {{ background:#3a2d16; color:#e0b654 }}
 .flag.bad {{ background:#3d1f22; color:#e88b8b }}
 .flag.mid {{ background:#3a3418; color:#ddd06a }}
 .flag.good {{ background:#1d3524; color:#7fd39a }}
 .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#262d34 }}
 .side {{ position:relative; background:#0d1013; min-height:120px;
          display:flex; align-items:center; justify-content:center }}
 .side img {{ width:100%; height:auto; display:block; cursor:zoom-in }}
 .tag {{ position:absolute; top:4px; inset-inline-start:4px; z-index:2; font-size:11px;
         background:#000a; padding:1px 6px; border-radius:3px; color:#c3ced8 }}
 .none {{ color:#7d5252; font-size:12px; padding:28px 10px; text-align:center }}
 dialog {{ border:0; padding:0; background:#000; max-width:96vw; max-height:96vh }}
 dialog img {{ max-width:96vw; max-height:96vh; display:block }}
 dialog::backdrop {{ background:#000c }}
</style>
<header>
  <h1>מצב הקרופר<small>{html.escape(run_dir.name)} · נבנה {built} · {len(cards)} תמונות</small></h1>
  <div class="chips">{''.join(chips)}</div>
</header>
<main>{''.join(items)}</main>
<dialog id="zoom"><img></dialog>
<script>
 const cards = [...document.querySelectorAll('.card')];
 document.querySelectorAll('.chip').forEach(chip => chip.onclick = () => {{
   document.querySelectorAll('.chip').forEach(c => c.classList.remove('on'));
   chip.classList.add('on');
   const key = chip.dataset.cat;
   cards.forEach(card => {{
     const show = key === 'all'
       || (key === 'fail' && card.dataset.ok === '0')
       || (key === 'rot'  && card.dataset.rot === '1')
       || card.dataset.cat === key;
     card.classList.toggle('hide', !show);
   }});
 }});
 const dlg = document.getElementById('zoom');
 document.querySelectorAll('.side img').forEach(img => img.onclick = () => {{
   dlg.querySelector('img').src = img.dataset.full || img.src; dlg.showModal();
 }});
 dlg.onclick = () => dlg.close();
</script>
</html>"""


def latest_run(runs_dir: Path) -> Path:
    candidates = [d for d in runs_dir.iterdir()
                  if d.is_dir() and ((d / "crops.jsonl").exists() or (d / "crops.csv").exists())] \
        if runs_dir.is_dir() else []
    if not candidates:
        raise SystemExit(f"no runs found under {runs_dir} — run `bench.py crop` first")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run name; default is the most recent")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out", type=Path, default=CURRENT)
    args = ap.parse_args()

    run_dir = args.runs_dir / args.run if args.run else latest_run(args.runs_dir)
    page = build(run_dir, args.benchmark, args.out)
    print(f"open {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
