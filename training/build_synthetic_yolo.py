#! /usr/bin/env python3
"""
Generate a YOLO-format synthetic training dataset of Hebrew STaM scroll images.

Each image is built by generate_synthetic_scroll.generate() using random text,
then placed into the standard YOLO directory layout with a data.yaml.

Usage:
  python build_synthetic_yolo.py
  python build_synthetic_yolo.py --count 40 --target-chars 500 --output-dir yolo_synthetic
"""
import argparse
import shutil
import sys
from pathlib import Path

# Suppress argparse inside generate_synthetic_scroll when imported as module
_saved_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from generate_synthetic_scroll import generate, HEBREW_CHARS
sys.argv = _saved_argv

ROWS_PER_500 = 27   # calibrated: rows=27 ≈ 500 chars on a 1400px canvas


def build(
    count: int = 40,
    rows: int = ROWS_PER_500,
    output_dir: str = "yolo_synthetic",
    dataset_dir: str = "test_output/char_dataset_clean",
    seed: int = 0,
):
    out = Path(output_dir)
    img_dir = out / "images" / "train"
    lbl_dir = out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    tmp_img = Path("/tmp/_synth_scroll.png")
    tmp_lbl = Path("/tmp/_synth_scroll.txt")

    total_chars = 0

    for i in range(count):
        img_seed = seed + i
        generate(
            output=str(tmp_img),
            rows=rows,
            seed=img_seed,
            dataset_dir=dataset_dir,
        )

        fname = f"scroll_{i+1:04d}"
        shutil.move(str(tmp_img), img_dir / f"{fname}.png")
        shutil.move(str(tmp_lbl), lbl_dir / f"{fname}.txt")

        n = len((lbl_dir / f"{fname}.txt").read_text().splitlines())
        total_chars += n
        print(f"  [{i+1:2d}/{count}] {fname}.png  {n} chars")

    # data.yaml
    yaml_lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/train",
        f"nc: {len(HEBREW_CHARS)}",
        "names:",
    ] + [f"  {i}: {ch}" for i, ch in enumerate(HEBREW_CHARS)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"\nDone: {count} images, {total_chars} total chars "
          f"(avg {total_chars//count}/image) → {out}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",      type=int, default=40)
    parser.add_argument("--rows",       type=int, default=ROWS_PER_500,
                        help=f"Rows per image (default {ROWS_PER_500} ≈ 500 chars)")
    parser.add_argument("--output-dir",   default="yolo_synthetic")
    parser.add_argument("--dataset-dir",  default="test_output/char_dataset_clean",
                        help="Character crop dataset directory (default: char_dataset_clean)")
    parser.add_argument("--seed",         type=int, default=0,
                        help="Base random seed; each image uses seed+i")
    args = parser.parse_args()
    build(count=args.count, rows=args.rows, output_dir=args.output_dir,
          dataset_dir=args.dataset_dir, seed=args.seed)


if __name__ == "__main__":
    main()
