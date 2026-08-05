#!/usr/bin/env python3
"""Benchmark harness for the STaM cropper.

Three stages, run independently:

    python3 tools/bench.py crop   --run baseline
    python3 tools/bench.py score  --run baseline
    python3 tools/bench.py report --run baseline [--vs previous_run]

**crop** is the fast loop: run the cropper over the benchmark, record timing and
geometry. Seconds per image, no scoring engine, safe to run on every change.

**score** is the slow loop: feed each crop to the real production engine
(Stam-OCR) and count what it got wrong. Every result is cached by the SHA-256 of
the crop's bytes, so a change that only moves rotate images re-scores only those.
The first run is expensive; subsequent runs cost only what actually changed.

**report** joins the two and prints a per-challenge table, optionally as a diff
against an earlier run.

Why not reuse the company's metric
----------------------------------
Their July 2026 report scored `errors / words_detected`, averaged over images
that succeeded. Both halves of that are unstable:

* The denominator is what the OCR *found*. A cropper that cuts away half the
  parchment reports fewer words, therefore fewer errors, and scores better.
* Averaging over successes only means each cropper is graded on a different
  subset — a version that recovers a hard image is penalised for its errors,
  while a version that fails that image outright is simply not graded on it.

Stam-OCR diffs against a *known* reference text (`find_reference_text` picks the
matching mezuza/tefilin/torah page automatically). That gives a denominator that
is fixed per document type and independent of how good the crop was. So the
primary metric here is **recovery**: how much of the text that should be there
was actually read, with a failed crop scoring 0 rather than dropping out of the
average. Their metric is still computed and reported alongside, so the two can
be cross-checked against their CSV if we ever get it.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_BENCHMARK = REPO.parent / "benchmark"
DEFAULT_STAM_OCR = Path.home() / "dev" / "Stam-OCR"
RUNS_DIR = REPO / "test_output" / "bench"

CATEGORIES = [
    "01_cropper", "02_rotate", "03_perspective",
    "04_objects", "05_drawings", "06_roughness", "00_general",
]

HEBREW = re.compile(r"[א-ת]")

# A reference of this length means the engine matched the scan to a torah page.
#
# Read it as a symptom of our own failure, never as an outside quirk to be
# excluded from the numbers. The engine picks a reference by comparing what it
# read against each source, and accepts one only if the length lands within ±20%
# of it. When the image was poor enough that only part of the text came back,
# nothing fits, and it falls through to an unconstrained search over hundreds of
# torah pages — where something always looks vaguely similar.
#
# So a torah match on a mezuza means: the picture we handed over could not be
# read. That is the thing to fix. Filtering these images out of the metric would
# be hiding exactly the failures the cropper exists to prevent.
TORAH_REF_LETTERS = 1000

# The company's own read of its report: above roughly this many errors the
# engine has stopped finding real scribal faults and is simply failing to read
# the image. A genuine scan usually shows a handful. So the target is not zero
# errors — zero would mean it read almost nothing — but the share of images that
# land under the line. Recovery is the guard on the other side: a crop that
# throws text away also reports few errors, and would otherwise look like a win.
ERROR_LIMIT = 20

# Fields a cached score must carry to be reusable. Add to this whenever a new
# measurement is introduced, and stale records re-score themselves.
REQUIRED_SCORE_FIELDS = ("text_match_pct", "ocr_chars")

# How often the crop stage rebuilds `current/`. Low enough that the folder is
# never far behind, high enough that thumbnailing doesn't dominate the sweep.
SHEET_EVERY = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


# ── stage 1: crop ──────────────────────────────────────────────────────────────

@dataclass
class CropRecord:
    category: str
    image: str
    ok: bool
    error: str = ""
    seconds: float = 0.0
    in_w: int = 0
    in_h: int = 0
    out_w: int = 0
    out_h: int = 0
    rotated: bool = False       # inferred: portrait in, landscape out
    area_kept_pct: float = 0.0  # crop area as % of input area
    crop_path: str = ""
    crop_sha: str = ""
    # Characters Vision found that the crop rectangle then excluded — text the
    # cropper is directly responsible for losing.
    chars_found: int = 0
    chars_lost: int = 0
    chars_lost_pct: float = 0.0
    grain: float = 0.0          # parchment roughness, measured on the text block
    denoised: bool = False      # whether the denoiser actually fired
    fan: float = 0.0            # degrees of keystone measured across the text
    rectified: bool = False     # whether the keystone correction actually fired
    rect_reason: str = ""       # which guard declined, when it did


def _load_set(path: Path | None) -> set[str] | None:
    """Read a dev-set file into {"category/image", ...}."""
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"no such set file: {path} — run tools/make_devset.py")
    names = {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")}
    if not names:
        raise SystemExit(f"{path} is empty")
    return names


def stage_crop(bench: Path, run_dir: Path, model_path: str, only: list[str] | None,
               subset: set[str] | None = None) -> None:
    """Run the production cropper over the benchmark."""
    import cv2
    from ultralytics import YOLO

    import crop_stam
    from stam_io import imread_any, list_images

    import sheet

    model = YOLO(model_path)
    crops_dir = run_dir / "crops"
    records: list[CropRecord] = []

    # Written one line at a time so `current/` can be rebuilt mid-sweep and the
    # results of a run that dies at image 140 are not lost.
    run_dir.mkdir(parents=True, exist_ok=True)
    live = run_dir / "crops.jsonl"
    live.unlink(missing_ok=True)

    def emit(rec: CropRecord) -> None:
        records.append(rec)
        with live.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        if len(records) % SHEET_EVERY == 0:
            try:
                sheet.build(run_dir, bench)
            except Exception:                             # noqa: BLE001
                pass   # the preview is a convenience; never fail a sweep for it

    for category in CATEGORIES:
        if only and category not in only:
            continue
        src = bench / category
        if not src.is_dir():
            continue
        out = crops_dir / category
        out.mkdir(parents=True, exist_ok=True)

        for image_path in list_images(src):
            image = Path(image_path)
            if subset is not None and f"{category}/{image.name}" not in subset:
                continue
            rec = CropRecord(category=category, image=image.name, ok=False)

            raw = imread_any(image_path)
            if raw is None:
                rec.error = "undecodable"
                emit(rec)
                print(f"  UNREADABLE {image.name}")
                continue
            rec.in_h, rec.in_w = raw.shape[:2]

            t0 = time.perf_counter()
            try:
                ok = crop_stam.crop_image(model, image_path, str(out))
            except Exception as exc:                      # noqa: BLE001
                # A crash here is itself a finding: contract clause 2.5 forbids
                # new exceptions, so record it rather than aborting the sweep.
                rec.error = f"{type(exc).__name__}: {exc}"[:200]
                rec.seconds = time.perf_counter() - t0
                emit(rec)
                print(f"  CRASH {image.name}: {rec.error}")
                continue
            rec.seconds = time.perf_counter() - t0

            produced = out / f"{image.stem}_cropped.jpg"
            if not ok or not produced.exists():
                rec.error = rec.error or "no detection"
                emit(rec)
                continue

            crop = cv2.imread(str(produced))
            if crop is None:
                rec.error = "crop unreadable"
                emit(rec)
                continue

            rec.ok = True
            rec.out_h, rec.out_w = crop.shape[:2]
            rec.rotated = (rec.in_h > rec.in_w) and (rec.out_w > rec.out_h)
            rec.area_kept_pct = round(
                100 * (rec.out_w * rec.out_h) / max(1, rec.in_w * rec.in_h), 1
            )
            rec.crop_path = str(produced.relative_to(run_dir))
            rec.crop_sha = sha256_file(produced)
            diag = getattr(crop_stam, "LAST_DIAG", {})
            rec.chars_found = diag.get("chars_found", 0)
            rec.chars_lost = diag.get("chars_lost", 0)
            rec.chars_lost_pct = diag.get("chars_lost_pct", 0.0)
            rec.grain = round(diag.get("grain", 0.0), 2)
            rec.denoised = bool(diag.get("denoised", False))
            rec.fan = round(diag.get("fan") or 0.0, 2)
            rec.rectified = bool(diag.get("rectified", False))
            rec.rect_reason = diag.get("rect_reason", "") or ""
            emit(rec)

    if not records:
        raise SystemExit(f"no images found under {bench} — check --benchmark")

    # Record what this run was actually configured with. A result you cannot
    # trace back to its settings is not a result.
    settings = crop_stam.active_settings()
    (run_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    print("\nsettings: " + "  ".join(f"{k}={v}" for k, v in sorted(settings.items())))

    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "crops.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    ok = sum(r.ok for r in records)
    secs = [r.seconds for r in records if r.seconds]
    median = f"   median {statistics.median(secs):.2f}s" if secs else ""
    print(f"\ncropped {ok}/{len(records)}{median}")
    print(f"wrote {csv_path}")

    page = sheet.build(run_dir, bench)
    print(f"look at  {page}")


# ── stage 2: score ─────────────────────────────────────────────────────────────

@dataclass
class ScoreRecord:
    sha: str
    ok: bool
    error: str = ""
    seconds: float = 0.0
    text_type: str = ""
    page_num: int = 0
    ref_letters: int = 0       # fixed denominator: letters in the reference text
    ref_words: int = 0
    read_letters: int = 0      # letters the engine actually read
    read_words: int = 0
    letter_errors: int = 0
    word_errors: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    # How much of the expected text the engine actually came back with, 0-100.
    # This is the metric that matters. Counting error types is unreliable
    # precisely where it counts most: when the engine cannot read the image it
    # invents letters and then dutifully reports errors against its own
    # invention, so a hopeless scan and a nearly perfect one can post similar
    # error counts. Comparing the recognised text against the known reference
    # cannot be fooled that way — invented text simply fails to match.
    text_match_pct: float = 0.0
    ocr_chars: int = 0
    # A fingerprint of *which* faults were reported, not merely how many.
    # Counting by type would call it unchanged if a touching-letter moved from
    # the third word to the seventh, and the product's whole purpose is telling
    # a scribe which letter is wrong. Each entry is line/word/letter position,
    # the letter itself and the fault, so two passes can be compared exactly.
    error_fingerprint: list = field(default_factory=list)


def _load_stam_ocr(stam_ocr_dir: Path):
    """Import the production engine, with its own directory as cwd.

    Stam-OCR resolves `bot_folder='./bot'` relative to the process working
    directory, so it has to be run from its own root.
    """
    if not (stam_ocr_dir / "StamOcr.py").exists():
        raise SystemExit(f"Stam-OCR not found at {stam_ocr_dir} (pass --stam-ocr)")
    os.chdir(stam_ocr_dir)
    sys.path.insert(0, str(stam_ocr_dir))
    from StamOcr import StamOcr  # noqa: WPS433

    return StamOcr()


def _hebrew_only(text: str) -> str:
    """Strip everything that is not a Hebrew letter.

    Spacing, punctuation and line breaks differ between the reference files and
    the engine's output for reasons that have nothing to do with how well the
    parchment was read.
    """
    return "".join(HEBREW.findall(text or ""))


def text_match(reference: str, recognised: str) -> float:
    """Percentage of the reference text present, in order, in what was read.

    A ratio over the *reference* rather than a symmetric similarity: reading
    nothing must score 0, and padding the output with invented letters must not
    raise the score.
    """
    ref = _hebrew_only(reference)
    got = _hebrew_only(recognised)
    if not ref:
        return 0.0
    if not got:
        return 0.0
    matcher = difflib.SequenceMatcher(None, ref, got, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return round(100.0 * matched / len(ref), 1)


def _walk_letters(result: dict):
    for line in result.get("lines") or []:
        for word in line.get("words") or []:
            yield word, None
            for letter in word.get("letters") or []:
                yield word, letter


_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt)\b.*)$")


def _blame(stderr_text: str) -> str:
    """The exception line out of whatever the engine printed while failing.

    Not simply the last line: `StamOcr.create_scan` does
    `print(traceback.print_exc(), file=sys.stderr)`, and since print_exc writes
    the traceback and then returns None, the final line on stderr is the word
    "None". Taking the last line reports that, which is how the first attempt at
    this produced 'create_scan returned None: None' for every failure.
    """
    lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        m = _EXC_LINE.match(lines[i])
        if not m:
            continue
        # The frame just above the exception is where it was actually raised —
        # without it, "AttributeError: 'NoneType' has no attribute 'contours'"
        # could be any of thirteen places in their code, and none of them worth
        # patching on a guess.
        where = ""
        for j in range(i - 1, -1, -1):
            f = re.match(r'File "([^"]+)", line (\d+), in (\S+)', lines[j])
            if f:
                where = f" at {Path(f.group(1)).name}:{f.group(2)} in {f.group(3)}"
                break
        return m.group(1) + where
    # No recognisable exception: fall back to the last line that carries content.
    for ln in reversed(lines):
        if ln not in {"None", "e"} and not ln.startswith(("File ", "Traceback")):
            return ln
    return "no traceback captured"


class _Timeout(Exception):
    pass


def _deadline(seconds: int):
    """Abort a scan that runs too long.

    Their engine has no internal limit, and on a large image it can grind for
    minutes or exhaust memory — their own report logged 'bad allocation'
    failures on the same set. One stuck image must not cost the whole sweep, and
    an image the engine cannot process in time is a result in itself.
    """
    import contextlib
    import signal

    @contextlib.contextmanager
    def guard():
        if not seconds or not hasattr(signal, "SIGALRM"):
            yield
            return

        def fire(signum, frame):
            raise _Timeout(f"exceeded {seconds}s")

        previous = signal.signal(signal.SIGALRM, fire)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    return guard()


def score_one(engine, image_path: Path, model_name: str, type_scan: str,
              timeout: int = 0) -> ScoreRecord:
    """Score a single image with the production engine.

    The filename handed to `create_scan` is deliberately a hash, not the real
    name: `ScanData` derives `text_type` from the filename stem, so an image
    called `mezuza3.jpg` is forced to the mezuza reference while an image called
    `IMG_1219.jpg` is auto-detected. Uniform auto-detection keeps every image on
    the same footing.
    """
    sha = sha256_file(image_path)
    rec = ScoreRecord(sha=sha, ok=False)
    t0 = time.perf_counter()
    # create_scan catches almost everything and returns None, printing the real
    # traceback to stderr on its way. Without capturing that, every distinct
    # failure looks identical and there is nothing to act on.
    import contextlib
    import io

    err = io.StringIO()
    try:
        with _deadline(timeout), contextlib.redirect_stderr(err):
            scan = engine.create_scan(
                f"{sha[:16]}.jpg", str(image_path), "./bot", model_name, type_scan, 0
            )
    except _Timeout as exc:
        rec.error = f"timeout: {exc}"
        rec.seconds = time.perf_counter() - t0
        return rec
    except Exception as exc:                              # noqa: BLE001
        rec.error = f"{type(exc).__name__}: {exc}"[:200]
        rec.seconds = time.perf_counter() - t0
        return rec
    rec.seconds = time.perf_counter() - t0

    if scan is None:
        rec.error = f"create_scan returned None: {_blame(err.getvalue())}"[:300]
        return rec

    reference = getattr(scan.compare_data_module, "reference_text", "") or ""
    rec.text_type = getattr(scan, "text_type", "") or ""
    rec.page_num = int(getattr(scan, "page_num", 0) or 0)
    rec.ref_letters = len(HEBREW.findall(reference))
    rec.ref_words = len(reference.split())

    # The engine keeps the assembled text on the extended data; fall back to the
    # raw OCR pass if that stage never ran.
    recognised = ""
    for holder in ("extended_ocr_text_data", "ocr_text_data"):
        obj = getattr(scan, holder, None)
        text = getattr(obj, "ocr_text", None) if obj is not None else None
        if text:
            recognised = text
            break
    rec.ocr_chars = len(_hebrew_only(recognised))
    rec.text_match_pct = text_match(reference, recognised)

    try:
        result = json.loads(scan.result_json)
    except Exception:                                     # noqa: BLE001
        rec.error = "result_json unparseable"
        return rec

    fingerprint = []
    for li, line in enumerate(result.get("lines") or []):
        for wi, word in enumerate(line.get("words") or []):
            wt = word.get("error_type") or ""
            if wt:
                fingerprint.append(f"{li}.{wi}|{word.get('orig_text') or ''}|{wt}")
            for xi, letter in enumerate(word.get("letters") or []):
                lt = letter.get("error_type") or ""
                if lt:
                    fingerprint.append(
                        f"{li}.{wi}.{xi}|{letter.get('orig_text') or letter.get('ocr_text') or ''}|{lt}")
    rec.error_fingerprint = fingerprint

    by_type: dict[str, int] = {}
    for word, letter in _walk_letters(result):
        if letter is None:
            et = word.get("error_type") or ""
            if et:
                rec.word_errors += 1
                by_type[et] = by_type.get(et, 0) + 1
            rec.read_words += 1
            continue
        et = letter.get("error_type") or ""
        if et:
            rec.letter_errors += 1
            by_type[et] = by_type.get(et, 0) + 1
        if (letter.get("ocr_text") or "").strip():
            rec.read_letters += 1

    rec.errors_by_type = by_type
    rec.ok = True
    return rec


def stage_score(run_dir: Path, stam_ocr_dir: Path, model_name: str,
                type_scan: str, limit: int | None, bench: Path | None = None,
                timeout: int = 120, cache_file: Path | None = None,
                force: bool = False) -> None:
    """Score the crops, and optionally the untouched originals.

    Scoring the originals is what answers the only question the company
    actually cares about: how many errors does their app report without our
    crop, and how many with it. Both go into the same cache, keyed by file
    content, so nothing is scored twice.
    """
    # A separate cache keeps an experiment from overwriting the baseline it is
    # meant to be compared against.
    cache_path = cache_file or (run_dir.parent / "score_cache.jsonl")
    cache: dict[str, dict] = {}
    if cache_path.exists() and not force:
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    # A record written before a metric existed cannot answer for
                    # it. Silently treating it as current is how half a table
                    # ends up blank — or worse, half measured one way and half
                    # another. Anything missing a required field is re-scored.
                    if any(f not in row for f in REQUIRED_SCORE_FIELDS):
                        continue
                    # A failure is not worth caching. It is cheap to repeat,
                    # there are few of them, and the reason we record for one
                    # may improve as the instrumentation does — which is exactly
                    # what happened here.
                    if not row.get("ok"):
                        continue
                    cache[row["sha"]] = row

    crops_csv = run_dir / "crops.csv"
    if not crops_csv.exists():
        raise SystemExit(f"no {crops_csv} — run the crop stage first")
    with crops_csv.open(encoding="utf-8-sig") as fh:
        crops = [r for r in csv.DictReader(fh)]

    # (label, absolute path) for everything that still needs a score.
    todo: list[tuple[str, Path]] = []
    for r in crops:
        if r["ok"] == "True" and r["crop_sha"] not in cache:
            todo.append((f"crop  {r['category']}/{r['image']}", run_dir / r["crop_path"]))
    if bench is not None:
        for r in crops:
            src = bench / r["category"] / r["image"]
            if src.exists() and sha256_file(src) not in cache:
                todo.append((f"raw   {r['category']}/{r['image']}", src))

    if limit:
        todo = todo[:limit]
    print(f"{len(crops)} crops, {len(cache)} already scored, {len(todo)} to score")

    # _load_stam_ocr chdir's into the engine's directory, so anything we touch
    # afterwards has to be an absolute path.
    cache_path = cache_path.resolve()
    run_dir = run_dir.resolve()
    bench = bench.resolve() if bench is not None else None

    import sheet

    engine = _load_stam_ocr(stam_ocr_dir) if todo else None
    for i, (label, path) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {label}", flush=True)
        rec = score_one(engine, path, model_name, type_scan, timeout)
        status = "ok" if rec.ok else rec.error[:60]
        print(f"          {rec.seconds:5.1f}s  {status}", flush=True)
        cache[rec.sha] = asdict(rec)
        # Append as we go: a 157-image sweep takes long enough that losing it
        # to a crash at image 150 is not acceptable, and it lets `report` run
        # against partial results from another terminal while this is going.
        with cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        if i % SHEET_EVERY == 0 and bench is not None:
            try:
                sheet.build(run_dir, bench)
            except Exception:                             # noqa: BLE001
                pass

    print(f"cache now holds {len(cache)} scored crops → {cache_path}")

    # A short, self-contained block at the very end. The engine prints hundreds
    # of lines of its own while it works, and the numbers that matter should not
    # have to be hunted for in the middle of it.
    try:
        rows = _join(run_dir, bench)
        summary = _summarise(rows)
        overall = summary.get("ALL")
        if overall:
            name = run_dir.name
            settings = {}
            sfile = run_dir / "settings.json"
            if sfile.exists():
                settings = json.loads(sfile.read_text(encoding="utf-8"))
            interesting = {k: v for k, v in settings.items() if k in (
                "GRAIN_MIN", "MIN_CHAR_PX", "RESEG_AFTER_DESKEW", "USE_TEXT_REGION",
                "FLATTEN_BG", "DERULE", "ROUGH_MIN", "MODEL_MAX_RATIO")}
            print("\n" + "=" * 58)
            print(f"  RUN {name}   {overall['n']} images")
            print(f"  text recovered        {overall['text_match_pct']}%")
            print(f"  under {ERROR_LIMIT} errors        {overall['under_limit_pct']}%")
            print(f"  unidentifiable        {overall['unreadable_n']}")
            print(f"  median errors         {overall['median_errors']}")
            print("  settings              "
                  + "  ".join(f"{k}={v}" for k, v in sorted(interesting.items())))
            print("=" * 58)
    except Exception:                                     # noqa: BLE001
        pass

    # Refresh `current/` so the recovery numbers land next to the pictures.
    import sheet

    try:
        print(f"look at  {sheet.build(run_dir, DEFAULT_BENCHMARK)}")
    except Exception as exc:                              # noqa: BLE001
        print(f"(sheet not rebuilt: {exc})")


# ── stage 3: report ────────────────────────────────────────────────────────────

def _recovery(s: dict) -> float | None:
    """Letters read and not flagged, over letters the reference text says exist."""
    ref = s.get("ref_letters") or 0
    if not s.get("ok") or not ref:
        return None
    read = s.get("read_letters") or 0
    return round(100 * max(0, read - (s.get("letter_errors") or 0)) / ref, 1)


def _join(run_dir: Path, bench: Path | None = None) -> list[dict]:
    cache_path = run_dir.parent / "score_cache.jsonl"
    scores: dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    # Later lines win: the file is append-only, so a re-scored
                    # record supersedes the stale one it replaces.
                    scores[row["sha"]] = row

    with (run_dir / "crops.csv").open(encoding="utf-8-sig") as fh:
        crops = list(csv.DictReader(fh))

    joined = []
    for c in crops:
        s = scores.get(c["crop_sha"], {})
        cropped_ok = c["ok"] == "True"
        scored_ok = bool(s.get("ok"))
        ref = s.get("ref_letters") or 0
        read = s.get("read_letters") or 0
        letter_errors = s.get("letter_errors") or 0
        word_errors = s.get("word_errors") or 0

        # Recovery: of the letters that should be on this parchment, how many
        # were read *and* not flagged. Letters only — a word-level error such as
        # ALTERED_WORD also flags its constituent letters, so adding the two
        # together would double-count. A crop that failed scores 0; it is not
        # excluded from the average, which is the whole point of the metric.
        recovery = 0.0
        if cropped_ok and scored_ok and ref:
            recovery = 100 * max(0, read - letter_errors) / ref

        # The same engine on the untouched original — the with/without-us number.
        raw_recovery = raw_rate = raw_errors = None
        ref_mismatch = False
        raw = {}
        if bench is not None:
            src = bench / c["category"] / c["image"]
            if src.exists():
                raw = scores.get(sha256_file(src), {})
                # The engine picks the reference text by matching what it read.
                # On a mezuza, whose text is shma + vehaya, it can land on a
                # tefilin parsha instead — and land somewhere *different* for
                # the photo than for the crop. When that happens the two sides
                # were diffed against different documents and the comparison is
                # meaningless, so it is excluded rather than averaged in.
                if raw and s and raw.get("ok") and s.get("ok"):
                    ref_mismatch = raw.get("text_type") != s.get("text_type")
                if raw and not ref_mismatch:
                    raw_recovery = _recovery(raw) or 0.0
                    if raw.get("ok"):
                        raw_errors = (raw.get("letter_errors") or 0) + (raw.get("word_errors") or 0)
                    if raw.get("read_words"):
                        raw_rate = round(
                            100 * ((raw.get("letter_errors") or 0)
                                   + (raw.get("word_errors") or 0)) / raw["read_words"], 1)

        # Their metric, kept for cross-checking against the company's CSV.
        # This one does add word and letter errors, because that is what their
        # report counted.
        their_rate = None
        if scored_ok and s.get("read_words"):
            their_rate = 100 * (letter_errors + word_errors) / s["read_words"]

        joined.append({
            "category": c["category"],
            "image": c["image"],
            "cropped": cropped_ok,
            "scored": scored_ok,
            "seconds": float(c["seconds"] or 0),
            "rotated": c["rotated"] == "True",
            "area_kept_pct": float(c["area_kept_pct"] or 0),
            "text_type": s.get("text_type", ""),
            "raw_text_type": (scores.get(sha256_file(bench / c["category"] / c["image"]), {})
                              .get("text_type", "") if bench is not None
                              and (bench / c["category"] / c["image"]).exists() else ""),
            "ref_mismatch": ref_mismatch,
            "ref_letters": ref,
            "read_letters": read,
            "letter_errors": letter_errors,
            "word_errors": word_errors,
            # The headline: did the app come back with a believable number of
            # errors, or did it drown? An image that failed to score never made
            # it under the line.
            # Flagged, not excluded: see TORAH_REF_LETTERS. These are images the
            # engine could not read well enough to even identify, which is the
            # cropper's problem to solve, not a measurement artefact to discount.
            "unreadable_ref": bool(scored_ok and (s.get("ref_letters") or 0)
                                   >= TORAH_REF_LETTERS),
            "text_match_pct": s.get("text_match_pct") if scored_ok else 0.0,
            "raw_text_match_pct": (raw.get("text_match_pct")
                                   if bench is not None and raw and raw.get("ok") else None),
            "ocr_chars": s.get("ocr_chars") if scored_ok else None,
            "errors_total": (letter_errors + word_errors) if scored_ok else None,
            "under_limit": bool(scored_ok and (letter_errors + word_errors) < ERROR_LIMIT),
            "raw_errors_total": raw_errors,
            "raw_under_limit": bool(raw_errors is not None and raw_errors < ERROR_LIMIT),
            "recovery_pct": round(recovery, 1),
            "raw_recovery_pct": raw_recovery,
            "gain_pct": (round(recovery - raw_recovery, 1)
                         if raw_recovery is not None else None),
            "their_error_rate_pct": round(their_rate, 1) if their_rate is not None else None,
            "raw_error_rate_pct": raw_rate,
            "error": c["error"] or s.get("error", ""),
        })
    return joined


def _summarise(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for category in CATEGORIES + ["ALL"]:
        sel = rows if category == "ALL" else [r for r in rows if r["category"] == category]
        if not sel:
            continue
        secs = [r["seconds"] for r in sel if r["seconds"]]
        theirs = [r["their_error_rate_pct"] for r in sel
                  if r["their_error_rate_pct"] is not None]
        raw_rates = [r["raw_error_rate_pct"] for r in sel
                     if r.get("raw_error_rate_pct") is not None]
        raw_recs = [r["raw_recovery_pct"] for r in sel
                    if r.get("raw_recovery_pct") is not None]
        matches = [r["text_match_pct"] for r in sel if r["text_match_pct"] is not None]
        raw_matches = [r["raw_text_match_pct"] for r in sel
                       if r.get("raw_text_match_pct") is not None]
        recovery = round(sum(r["recovery_pct"] for r in sel) / len(sel), 1)
        raw_recovery = round(sum(raw_recs) / len(raw_recs), 1) if raw_recs else None
        errs = [r["errors_total"] for r in sel if r["errors_total"] is not None]
        raw_errs = [r["raw_errors_total"] for r in sel if r["raw_errors_total"] is not None]
        scored_any = any(r["scored"] for r in sel)
        out[category] = {
            "n": len(sel),
            "crop_ok_pct": round(100 * sum(r["cropped"] for r in sel) / len(sel), 1),
            "scored_ok_pct": round(100 * sum(r["scored"] for r in sel) / len(sel), 1),
            # Share of images the app reported fewer than ERROR_LIMIT errors on.
            # Mean over ALL images, a failed scan counting as 0 rather than
            # dropping out — the whole point of a fixed reference.
            "text_match_pct": round(sum(r["text_match_pct"] or 0 for r in sel) / len(sel), 1)
                              if matches else None,
            "raw_text_match_pct": round(sum(raw_matches) / len(sel), 1) if raw_matches else None,
            "unreadable_n": sum(r["unreadable_ref"] for r in sel),
            "under_limit_pct": (round(100 * sum(r["under_limit"] for r in sel) / len(sel), 1)
                                if scored_any else None),
            "raw_under_limit_pct": (round(100 * sum(r["raw_under_limit"] for r in sel) / len(sel), 1)
                                    if raw_errs else None),
            "median_errors": round(statistics.median(errs)) if errs else None,
            "raw_median_errors": round(statistics.median(raw_errs)) if raw_errs else None,
            # Mean over ALL images, failures included as 0.
            "recovery_pct": recovery,
            "raw_recovery_pct": raw_recovery,
            "gain_pct": round(recovery - raw_recovery, 1) if raw_recovery is not None else None,
            "rotated_n": sum(r["rotated"] for r in sel),
            "median_sec": round(statistics.median(secs), 2) if secs else None,
            "their_error_rate_pct": round(statistics.mean(theirs), 1) if theirs else None,
            "raw_error_rate_pct": round(statistics.mean(raw_rates), 1) if raw_rates else None,
        }
    return out


def _print_settings_diff(run_dir: Path, vs: Path | None) -> None:
    """Show what actually differed between two runs' configurations.

    Twice in one session a run believed to be testing a new value turned out to
    have measured the old one, and the near-identical numbers were nearly read
    as a finding. If two runs were configured identically, say so loudly.
    """
    def load(p):
        f = p / "settings.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    mine, theirs = load(run_dir), load(vs) if vs else None
    if mine is None or theirs is None:
        return
    changed = {k: (theirs.get(k), v) for k, v in mine.items() if theirs.get(k) != v}
    if changed:
        print("\nsettings changed vs " + vs.name + ":")
        for k, (was, now) in sorted(changed.items()):
            print(f"   {k}: {was} -> {now}")
    else:
        print(f"\n!! settings are IDENTICAL to {vs.name} — this run changed nothing.")


def stage_report(run_dir: Path, vs: Path | None, bench: Path | None = None) -> None:
    rows = _join(run_dir, bench)
    summary = _summarise(rows)
    _print_settings_diff(run_dir, vs)

    detail = run_dir / "report.csv"
    with detail.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    base = _summarise(_join(vs, bench)) if vs else {}

    def fmt(v, width):
        return f"{'-' if v is None else v:>{width}}"

    head = (f"{'category':<16}{'n':>4}"
            f"{'raw text':>10}{'our text':>10}"
            f"{'raw <20':>9}{'ours <20':>10}"
            f"{'unread':>8}{'our errs':>10}{'sec':>7}")
    print("\n" + head)
    print("-" * len(head))
    for category, s in summary.items():
        line = (f"{category:<16}{s['n']:>4}"
                + fmt(s["raw_text_match_pct"], 10)
                + fmt(s["text_match_pct"], 10)
                + fmt(s["raw_under_limit_pct"], 9)
                + fmt(s["under_limit_pct"], 10)
                + fmt(s["unreadable_n"], 8)
                + fmt(s["median_errors"], 10)
                + fmt(s["median_sec"], 7))
        if category in base and base[category].get("text_match_pct") is not None \
                and s.get("text_match_pct") is not None:
            d = s["text_match_pct"] - base[category]["text_match_pct"]
            line += f"   {d:+.1f} text vs {vs.name}"
        print(line)
    mismatched = sum(r.get("ref_mismatch") for r in rows)
    if mismatched:
        print(f"\n  {mismatched} image(s) excluded from the raw comparison: the engine matched"
              "\n  the photo and the crop to different reference texts, so those two numbers"
              "\n  were never measuring the same thing. See ref_mismatch in the detail file.")

    print("\n  text  how much of the KNOWN reference text the engine actually came back"
          "\n        with. The primary metric. Error counts mislead exactly where it"
          "\n        matters most — an engine that cannot read the image invents letters"
          "\n        and then reports errors against its own invention — but invented"
          "\n        text cannot match the reference. A failed scan counts as 0."
          f"\n  <20   share of images reported with fewer than {ERROR_LIMIT} errors; the"
          "\n        company's own rule of thumb for 'this is a real report, not noise'."
          "\n  errs  median error count."
          "\n  unread  images the engine could not identify at all: it matched them to"
          "\n        a torah page because too little text came back for any reference to"
          "\n        fit. A crop failure, counted as one — not excluded."
          "\n  raw   the untouched photo through the same engine; ours, after our crop.")
    print(f"\nper-image detail: {detail}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["crop", "score", "report"])
    ap.add_argument("--run", default="latest", help="name of this run")
    ap.add_argument("--vs", help="earlier run name to diff against (report only)")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--model", default=str(REPO / "best.pt"))
    ap.add_argument("--only", nargs="*", help="restrict to these categories")
    ap.add_argument("--stam-ocr", type=Path, default=DEFAULT_STAM_OCR)
    ap.add_argument("--ocr-model", default="model-square-epoch27-20231226-122820__13M.h5")
    ap.add_argument("--type-scan", default="", help="leave empty for auto-detect")
    ap.add_argument("--limit", type=int, help="score at most N images (smoke test)")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip scoring the untouched originals")
    ap.add_argument("--timeout", type=int, default=120,
                    help="give up on a single scan after N seconds (0 disables)")
    ap.add_argument("--cache", type=Path,
                    help="write scores to this file instead of the shared cache")
    ap.add_argument("--force", action="store_true",
                    help="re-score everything, ignoring what is cached")
    ap.add_argument("--set", type=Path,
                    help="restrict to a dev-set file (see tools/make_devset.py)")
    args = ap.parse_args()

    run_dir = args.runs_dir / args.run
    bench = None if args.no_raw else args.benchmark
    if args.stage == "crop":
        stage_crop(args.benchmark, run_dir, args.model, args.only, _load_set(args.set))
    elif args.stage == "score":
        stage_score(run_dir, args.stam_ocr, args.ocr_model, args.type_scan,
                    args.limit, bench, args.timeout, args.cache, args.force)
    else:
        stage_report(run_dir, args.runs_dir / args.vs if args.vs else None, bench)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
