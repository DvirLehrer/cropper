# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Image **pre-processing** for STaM (Sefer Torah, Tefillin, Mezuzah) imagery: take a photo/scan of
parchment text and produce an orientation-corrected, tightly-cropped image to feed to a
downstream analysis system. Two standalone Python scripts wired together by files on disk under
`test_output/` — no package, no requirements file, no tests.

(An earlier CNN-classification branch — `export_coco_from_ocr.py`, `train_cnn.py`,
`infer_cnn_boxes.py` — was removed. There is no model training/inference here anymore.)

## End-to-end pipeline

Two independent CLIs that communicate via files on disk, in order:

1. **`google_ocr.py`** — runs Google Cloud Vision `document_text_detection` on input images and is
   the **source of truth for box geometry**. For each image it writes:
   - `test_output/<base>_char_boxes.json` — per-symbol records with `char`, 4-vertex `vertices`,
     and OCR hierarchy indices (`page`/`block`/`paragraph`/`word`/`symbol`). All symbols are
     saved, not only Hebrew. **This is the only output `prep_image.py` consumes.**
   - `test_output/<base>_bbox.<ext>`, `_warped.<ext>`, `_all_boxes.<ext>`, and (with
     `--dataset-out`) per-character crops under `char_dataset/` — legacy artifacts from the old
     CNN flow. Still produced, but nothing downstream reads them now.

2. **`prep_image.py`** — reads `<base>_char_boxes.json` (no re-OCR, so iterating on geometry costs
   no API calls) and runs a short pipeline, writing a debug PNG after each step to
   `test_output/debug/<base>/`:
   - `01_original` (+ `_overlay` of all OCR char quads)
   - `02_orientation` (+ `_overlay`) — global rotation fix. `estimate_orientation` takes the
     circular mean of each box's reading-direction vector (`v[1]-v[0]`); `rotate_bound` rotates the
     canvas (expanding so nothing clips) to bring text to horizontal. Box coords are carried
     through the same affine.
   - `03_crop_polygon` — overlay of the text region's convex hull (`text_hull`, Hebrew points
     only when present).
   - `03_cropped` — **the final output**: `mask_and_crop` blanks everything outside the hull
     polygon (to white) and crops to the polygon's bounding box.

## Critical detail: EXIF orientation / coordinate frame

Google Vision reports box coordinates against the **raw, un-rotated** pixels and ignores EXIF
orientation. `cv2.imread`, by contrast, **auto-applies** EXIF orientation. So for any photo with a
rotation tag (e.g. phone shots with EXIF Orientation = 6), a plain `cv2.imread` loads pixels 90°
out of sync with the saved boxes, and orientation correction comes out 90° wrong.

`prep_image.py::imread_boxframe` loads with `cv2.IMREAD_IGNORE_ORIENTATION` so pixels stay in the
boxes' frame; the orientation step then straightens everything from box geometry alone. If you add
any new image read in this pipeline, use `imread_boxframe`, not `cv2.imread`, or boxes and pixels
will desync. (`google_ocr.py` uses plain `cv2.imread` for its own legacy overlays/warps, so those
artifacts can be misaligned for EXIF-rotated inputs — but `prep_image.py` does not rely on them.)

## Running things

Each script is a standalone CLI. No build system, no lint config, no tests.

```bash
# 1. OCR -> char boxes (needs Vision credentials; see Dependencies)
python google_ocr.py --images-dir images/before_crop --output-dir test_output \
    --dataset-out "" --skip-existing
# single image:
python google_ocr.py --image path/to/file.jpg --output-dir test_output --dataset-out ""

# 2. Orientation fix + crop to text polygon (reads the boxes from step 1)
python prep_image.py --image images/before_crop/<base>.jpg \
    --char-boxes test_output/<base>_char_boxes.json
# if boxes are missing, --run-ocr will invoke google_ocr.detect_text first
```

When no input flag is passed, `google_ocr.py` defaults to globbing
`images/_test_set/*.{jpg,jpeg,png}`. `prep_image.py` defaults `--char-boxes` to
`test_output/<base>_char_boxes.json` and writes debug output under `test_output/debug/<base>/`.

## Image folder conventions

- `images/_test_set/` — small hand-picked set used as the default OCR input.
- `images/before_crop/`, `images/after_crop/`, `images/aligned_cropped/` — staged corpora at
  different points of manual pre-processing. They are inputs, not produced by these scripts.

## Dependencies

There is no `requirements.txt`. Imports tell you what must be installed in the active environment:

- `google.cloud.vision` — only `google_ocr.py`. Authenticates via Application Default Credentials
  (`gcloud auth application-default login`) or `GOOGLE_APPLICATION_CREDENTIALS` pointing at a
  service-account JSON.
- `opencv-python` (`cv2`), `Pillow` (`PIL`), `numpy` — used by both scripts.

`torch` is no longer a dependency (the CNN scripts were removed).
