# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Hebrew character recognition pipeline for STaM (Sefer Torah, Tefillin, Mezuzah) imagery. Four standalone Python scripts wired together by a shared on-disk layout under `test_output/` — no package, no requirements file, no tests.

## End-to-end pipeline

The scripts are independent CLIs that communicate via files on disk, in this order:

1. **`google_ocr.py`** — runs Google Cloud Vision `document_text_detection` on input images. For each image it writes:
   - `test_output/<base>_char_boxes.json` — per-symbol records with `char`, 4-vertex `vertices`, and OCR hierarchy indices (`page`/`block`/`paragraph`/`word`/`symbol`). All symbols are saved, not only Hebrew.
   - `test_output/<base>_bbox.<ext>` and `_warped.<ext>` — only produced when Hebrew characters exist. The page-level convex-hull → 4-point quadrilateral is approximated from Hebrew-only points and used for a perspective warp.
   - `test_output/<base>_all_boxes.<ext>` — overlay of every character quad (file name is "all_boxes" for historical reasons; contents are character-level, not word-level).
   - `test_output/char_dataset/<class>/...png` (when `--dataset-out` is set) — per-character crops rectified via perspective transform, grouped by class directory `U<HEX>_<char>` (codepoint-prefixed for stable, collision-free names).

2. **`export_coco_from_ocr.py`** — converts the `*_char_boxes.json` files into a single COCO JSON. Note: `category_id` is currently hard-coded to `1` in `coco_annotations` (the `char_to_cat` map is built but unused). Output defaults to `<images-dir>/instances_chars.json`.

3. **`train_cnn.py`** — trains a small CNN classifier (~5 conv layers + AdaptiveAvgPool + Linear, defined inline in `build_model`) on the per-class char crops produced by step 1. Writes `model.pt`, `meta.json`, `confusion_matrix.csv`, `per_class_accuracy.json` to `test_output/cnn_model/`.

4. **`infer_cnn_boxes.py`** — uses the trained CNN to **classify** the OCR-proposed character boxes (the CNN does not localize). Reads `<base>_char_boxes.json` from step 1, re-crops with matching padding, runs `model(crop)`, and writes overlaid PNGs to `test_output/cnn_boxes/`.

The implication: `google_ocr.py` is the source of truth for box geometry — both training crops and inference proposals come from it. If you change the crop logic (padding, perspective, polarity), keep `google_ocr.py::export_char_dataset` and `infer_cnn_boxes.py::crop_quad_with_padding` in sync, and re-run OCR + retrain.

## Critical pre-processing detail

`train_cnn.py::pil_to_tensor_gray` performs more than resize+normalize: it autocontrasts, **inverts polarity** if the median pixel is dark (so glyphs end up dark-on-light), and **Otsu-binarizes** by default. `infer_cnn_boxes.py` imports and reuses this exact function, so the same transforms apply at inference. Toggling `--no-binarize` during training without rebuilding inference will silently produce a distribution mismatch.

## Running things

Each script is a standalone CLI. There is no build system, no lint config, and no tests in the repo.

```bash
# 1. OCR + per-character crop export (requires GOOGLE_APPLICATION_CREDENTIALS)
python google_ocr.py --images-dir images/before_crop --output-dir test_output \
    --dataset-out test_output/char_dataset --skip-existing

# 2. COCO export (optional, only if you need detection-style annotations)
python export_coco_from_ocr.py --images-dir images/before_crop --boxes-dir test_output

# 3. Train classifier
python train_cnn.py --data-dir test_output/char_dataset --out-dir test_output/cnn_model

# 4. Run classifier on OCR boxes and write annotated images
python infer_cnn_boxes.py --test-dir images/_test_set --char-boxes-dir test_output \
    --model-dir test_output/cnn_model --out-dir test_output/cnn_boxes
```

Single-image OCR run: `python google_ocr.py --image path/to/file.jpg`.

When no input flag is passed, `google_ocr.py` defaults to globbing `images/_test_set/*.{jpg,jpeg,png}`.

## Image folder conventions

- `images/_test_set/` — small hand-picked set used as the default OCR input.
- `images/before_crop/`, `images/after_crop/`, `images/aligned_cropped/` — staged corpora at different points of manual pre-processing. They are not produced by these scripts; they are inputs.
- `images/before_crop/instances_chars.json` — output of `export_coco_from_ocr.py` lives next to its images (file names in the COCO JSON are basenames, relative to that directory).

## Dependencies

There is no `requirements.txt`. Imports tell you what must be installed in the active environment:

- `google.cloud.vision` (and `GOOGLE_APPLICATION_CREDENTIALS` env var pointing at a service-account JSON) — only `google_ocr.py`.
- `torch` — only `train_cnn.py` and `infer_cnn_boxes.py`. Imported lazily inside functions so that the other scripts work without it.
- `opencv-python` (`cv2`), `Pillow` (`PIL`), `numpy` — used across scripts.

If you add new dependencies, prefer keeping torch imports lazy so the OCR and COCO scripts stay runnable in environments without it.
