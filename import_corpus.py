#! /usr/bin/env python3
"""
Sample images from the hard-drive corpus into images/corpus/.

Samples across subdirectories (one session/scribe per subdir) to maximise diversity.
Each category gets its own subfolder: images/corpus/<category_slug>/

Usage:
  python import_corpus.py --src "/Volumes/Samsung USB/ספרי תורה מגילות ומזוזות"
  python import_corpus.py --src "..." --dry-run
"""
import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# How many images to sample per category
TARGETS = {
    "מזוזות":    125,
    "ספרי תורה": 125,
    "מגילות":    75,
    "תפילין":    75,
}

SLUG = {
    "מזוזות":    "mezuzot",
    "ספרי תורה": "torah",
    "מגילות":    "megilot",
    "תפילין":    "tefillin",
}


def list_images(directory: Path):
    return [p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS]


def sample_from_category(cat_dir: Path, n: int, rng) -> list:
    """
    Sample n images spread across subdirectories.
    Strategy: pick ceil(n / n_subdirs) images per subdir, randomly,
    then trim to exactly n.
    """
    subdirs = sorted(d for d in cat_dir.iterdir() if d.is_dir())
    if not subdirs:
        # Flat directory
        imgs = list_images(cat_dir)
        idx = rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)
        return [imgs[i] for i in sorted(idx)]

    # Shuffle subdirs, then pick round-robin until we have n
    order = list(rng.permutation(len(subdirs)))
    selected = []
    while len(selected) < n:
        made_progress = False
        for i in order:
            if len(selected) >= n:
                break
            imgs = list_images(subdirs[i])
            if not imgs:
                continue
            # Pick one random image from this subdir that hasn't been picked
            candidates = [p for p in imgs if p not in selected]
            if not candidates:
                continue
            pick = candidates[int(rng.integers(len(candidates)))]
            selected.append(pick)
            made_progress = True
        if not made_progress:
            break  # exhausted all subdirs

    return selected[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",     required=True,
                        help="Root of the hard-drive corpus")
    parser.add_argument("--dst",     default="images/corpus",
                        help="Destination directory (default: images/corpus)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be copied without copying")
    args = parser.parse_args()

    src  = Path(args.src)
    dst  = Path(args.dst)
    rng  = np.random.default_rng(args.seed)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    print(f"Source:  {src}")
    print(f"Dest:    {dst}")
    print(f"Dry run: {args.dry_run}\n")

    total = 0
    for cat_heb, n_target in TARGETS.items():
        cat_dir = src / cat_heb
        if not cat_dir.exists():
            print(f"  WARNING: category not found: {cat_dir}")
            continue

        slug    = SLUG[cat_heb]
        out_dir = dst / slug

        selected = sample_from_category(cat_dir, n_target, rng)
        print(f"{cat_heb} ({slug}): {len(selected)} images selected")

        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src_path in selected:
            # Build a collision-safe destination filename:
            # slug_subdir_filename  e.g. mezuzot_01.05.22_001.jpg
            rel   = src_path.relative_to(cat_dir)
            parts = list(rel.parts)
            safe  = re.sub(r"[^\w.\-]", "_", "_".join(parts))
            dst_path = out_dir / safe

            if args.dry_run:
                print(f"  {src_path}  →  {dst_path}")
            else:
                shutil.copy2(src_path, dst_path)
            copied += 1

        if not args.dry_run:
            print(f"  → copied {copied} to {out_dir}/")
        total += copied

    if args.dry_run:
        print(f"\nDry run total: {total} images would be copied")
    else:
        print(f"\nDone: {total} images copied to {dst}/")


if __name__ == "__main__":
    main()
