#!/usr/bin/env python3
"""Reorganise the raw benchmark drop into one folder per contract task.

Input  : a directory containing loose images and/or Google-Drive zip exports
         whose top-level folder name identifies the challenge
         (e.g. "אתגר הרוטייט-20260730T112002Z-1-001.zip").
Output : <root>/<NN_slug>/ folders, deduplicated by file content hash,
         plus manifest.csv and a _zips/ archive of the original downloads.

Idempotent: re-running on an already-organised folder is a no-op beyond
refreshing the manifest.

    python3 tools/organize_benchmark.py --root /path/to/benchmark
    python3 tools/organize_benchmark.py --root /path/to/benchmark --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stam_io import sniff_format, true_suffix  # noqa: E402

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # manifest will report 0x0 for HEIC files
    pass

Image.MAX_IMAGE_PIXELS = None

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp",
           ".heic", ".heif"}

# ---------------------------------------------------------------------------
# Challenge taxonomy — mirrors נספח א' of the contract.
# ---------------------------------------------------------------------------

CATEGORIES: list[tuple[str, str, str]] = [
    # (folder slug, Hebrew label, contract task)
    ("01_cropper",     "אתגר הקרופר",      "1 - crop correctly on any background / lighting"),
    ("02_rotate",      "אתגר הרוטייט",     "3 - detect vertical parchment, rotate 90 CCW"),
    ("03_perspective", "אתגר הפרספקטיבה",  "2 - perspective correction, unaligned corners"),
    ("04_objects",     "אתגר העצמים",      "4 - exclude magnets, tape, table background"),
    ("05_drawings",    "אתגר השרטוט",      "5 - ignore ruled lines, frames, engravings"),
    ("06_roughness",   "אתגר החספוס",      "6 - too-high-quality images, parchment texture"),
    ("00_general",     "כללי",             "unlabelled - legacy dev benchmark"),
]

SLUG_BY_LABEL = {label: slug for slug, label, _ in CATEGORIES}
LABEL_BY_SLUG = {slug: label for slug, label, _ in CATEGORIES}
TASK_BY_SLUG = {slug: task for slug, _, task in CATEGORIES}

# Loose filenames are classified by their Hebrew prefix. Longest match wins,
# so "אתגר השרטוטים" must be tried before "אתגר השרטוט".
LOOSE_PATTERNS: list[tuple[str, str]] = [
    (r"^אתגר\s+הקרופר",       "01_cropper"),
    (r"^עוד\s+אתגר\s+הקרופר", "01_cropper"),
    (r"^אתגר\s+הרוטייט",      "02_rotate"),
    (r"^אתגר\s+הפרספקטיבה",   "03_perspective"),
    (r"^אתגר\s+העצמים",       "04_objects"),
    (r"^אתגר\s+השרטוט",       "05_drawings"),   # also matches השרטוטים
    (r"^אתגר\s+החספוס",       "06_roughness",),
]

# Filenames that carry no information about the failure mode.
GENERIC_NAME = re.compile(
    r"""^(
          scan[0-9a-f]{16,}          # stamscanner export id
        | IMG[-_]?\d+                # camera roll
        | IMG-\d{8}-WA\d+            # whatsapp via android
        | PHOTO-[\d-]+               # whatsapp via ios
        | תמונה\ של\ WhatsApp.*      # whatsapp hebrew export
        | hgjh                       # keyboard mash
        )$""",
    re.VERBOSE,
)

# "אתגר הקרופר 7" — categorised but says nothing about what fails.
ENUMERATED_NAME = re.compile(r"^(עוד\s+)?אתגר\s+\S+(\s+\d+)?$")


def name_score(stem: str) -> int:
    """Higher = more worth keeping as the canonical filename.

    A name like "המגנט למעלה משמאל משבש" tells us why the image is in the
    benchmark; "scan691c17e4..." does not. When two paths hold identical
    bytes we keep the informative one.
    """
    if GENERIC_NAME.match(stem):
        return 0
    if ENUMERATED_NAME.match(stem):
        return 1
    return 2  # a real Hebrew description of the failure


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def classify_loose(stem: str) -> str:
    for pattern, slug in LOOSE_PATTERNS:
        if re.match(pattern, stem):
            return slug
    return "00_general"


def zip_category(folder_name: str) -> str | None:
    """Map a zip's internal top-level folder to a category slug."""
    folder_name = folder_name.strip()
    if folder_name in SLUG_BY_LABEL:
        return SLUG_BY_LABEL[folder_name]
    for label, slug in SLUG_BY_LABEL.items():
        if folder_name.startswith(label):
            return slug
    return None


@dataclass
class Candidate:
    """One physical file found in the input, before dedup."""

    path: Path
    category: str
    origin: str          # "loose" or the zip filename
    digest: str = ""
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------


def collect(root: Path, stage: Path) -> list[Candidate]:
    """Extract every zip and enumerate all candidate images."""
    candidates: list[Candidate] = []

    # Already-organised category folders (makes the script idempotent).
    for slug in LABEL_BY_SLUG:
        folder = root / slug
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                candidates.append(Candidate(p, slug, "loose"))

    # Loose images sitting at the root of the drop.
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            candidates.append(Candidate(p, classify_loose(p.stem), "loose"))

    # Zips, both at the root and already moved into _zips/.
    zips = sorted(root.glob("*.zip")) + sorted((root / "_zips").glob("*.zip"))
    for z in zips:
        target = stage / z.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Google Drive writes UTF-8 names without the UTF-8 flag set,
                # so zipfile mis-decodes them as cp437. Recover the bytes.
                raw = info.filename
                if not info.flag_bits & 0x800:
                    try:
                        raw = raw.encode("cp437").decode("utf-8")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                inner = Path(raw)
                if inner.suffix.lower() not in IMG_EXT:
                    continue
                slug = zip_category(inner.parts[0]) if len(inner.parts) > 1 else None
                if slug is None:
                    slug = classify_loose(inner.stem)
                out = target / inner.name
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                candidates.append(Candidate(out, slug, z.name))

    return candidates


def dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse identical bytes to a single winner.

    Preference order: a specific category beats 00_general (an image filed
    under "objects" is more useful than the same image filed as generic),
    then a descriptive filename beats a generated one.
    """
    for c in candidates:
        c.digest = sha256(c.path)

    by_digest: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_digest[c.digest].append(c)

    winners: list[Candidate] = []
    for digest, group in by_digest.items():
        group.sort(
            key=lambda c: (
                c.category == "00_general",     # False sorts first
                -name_score(c.path.stem),
                len(c.path.stem) * -1,
                c.path.name,
            )
        )
        winner, *rest = group
        seen = {winner.path.name}
        for other in rest:
            if other.path.name not in seen:
                winner.aliases.append(other.path.name)
                seen.add(other.path.name)
            if other.category != winner.category:
                winner.aliases.append(f"[{other.category}]{other.path.name}")
        winners.append(winner)

    winners.sort(key=lambda c: (c.category, c.path.name))
    return winners


def image_meta(path: Path) -> tuple[int, int, float]:
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w, h, round(w * h / 1e6, 2)
    except Exception:
        return 0, 0, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--stage", type=Path, default=Path("/tmp/bm_stage_run"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    stage: Path = args.stage
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    candidates = collect(root, stage)
    winners = dedupe(candidates)

    dropped = len(candidates) - len(winners)
    print(f"found {len(candidates)} files, {len(winners)} unique, {dropped} duplicates")

    counts: dict[str, int] = defaultdict(int)
    for c in winners:
        counts[c.category] += 1
    for slug, label, _ in CATEGORIES:
        print(f"  {slug:<16} {label:<18} {counts[slug]:>3}")

    if args.dry_run:
        return 0

    # --- write the new tree ------------------------------------------------
    for slug, _, _ in CATEGORIES:
        (root / slug).mkdir(exist_ok=True)

    rows = []
    final_paths: set[Path] = set()
    renamed = 0
    for c in winners:
        # Trust magic bytes over the extension: 17 of these files are HEIC
        # saved as .jpg, which every cv2-based tool silently refuses to read.
        real = true_suffix(c.path)
        name = c.path.name
        if real and real != c.path.suffix.lower():
            name = c.path.stem + real
            renamed += 1
        dest = root / c.category / name
        if c.path.resolve() != dest.resolve():
            if dest.exists():
                dest.unlink()
            shutil.copy2(c.path, dest)
        final_paths.add(dest)
        w, h, mp = image_meta(dest)
        rows.append(
            {
                "category": c.category,
                "challenge": LABEL_BY_SLUG[c.category],
                "contract_task": TASK_BY_SLUG[c.category],
                "filename": dest.name,
                "width": w,
                "height": h,
                "megapixels": mp,
                "format": sniff_format(dest),
                "aspect": round(max(w, h) / min(w, h), 2) if w and h else 0,
                "bytes": dest.stat().st_size,
                "sha256": c.digest[:16],
                "source": c.origin,
                "duplicate_names": " | ".join(c.aliases),
            }
        )

    # --- clear the root ----------------------------------------------------
    # Originals are moved aside, never deleted: the category folders hold
    # copies, and some mounted filesystems refuse unlink() outright.
    zip_dir = root / "_zips"
    zip_dir.mkdir(exist_ok=True)
    for z in sorted(root.glob("*.zip")):
        shutil.move(str(z), str(zip_dir / z.name))

    orig_dir = root / "_originals"
    orig_dir.mkdir(exist_ok=True)
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            shutil.move(str(p), str(orig_dir / p.name))

    # Files left in category folders by an earlier run that are no longer
    # winners — park them rather than delete.
    stale_dir = root / "_stale"
    for slug in LABEL_BY_SLUG:
        for p in sorted((root / slug).iterdir()):
            if p.is_file() and p not in final_paths:
                stale_dir.mkdir(exist_ok=True)
                shutil.move(str(p), str(stale_dir / p.name))

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest} ({len(rows)} rows)")
    if renamed:
        print(f"corrected {renamed} extension(s) to match actual file format")

    shutil.rmtree(stage, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
