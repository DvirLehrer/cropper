# Cropper OCR Benchmark + Web Pipeline

This repository contains a Hebrew OCR-based crop pipeline for mezuzah/tefillin scans, with:
- a batch benchmark runner
- a web upload flow
- shared crop/output logic used by both paths

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run benchmark on all images in `benchmark/`:
   - `python3 src/benchmark_run.py`
3. Run web app:
   - `python3 src/web_app.py`
   - open `http://localhost:8000`

## Current Structure

- `benchmark/`: input images
- `text/`: canonical texts (`shema`, `vehaya`, `kadesh`, `peter`)
- `text_type.csv`: type mapping per image
- `src/benchmark_run.py`: batch benchmark orchestration
- `src/cropper_pipeline.py`: single-image crop pipeline (shared core service API)
- `src/output.py`: output rendering stage (periodic stripes post-process)
- `src/stripes.py`: periodic stripe detection and debug light-fill output
- `src/ocr.py`: OCR execution wrappers over Tesseract
- `src/web_api.py`: web-facing adapter around crop/output pipeline
- `src/web_app.py`: Flask app (HTTP/deployment envelope only)
- `src/config.py`: single segmented configuration object (`settings`)
- `src/core/`: internal reusable geometry/OCR/line/alignment/logging modules

Generated outputs:
- `cropped/`: final cropped images
- `ocr_text/`: OCR text per image
- `debug_images/`: optional debug overlays
- `preprocessed/`: optional intermediate images

## Algorithm Flow

### 1) Batch entry (`benchmark_run.py`)
For each image:
1. Load image type from `text_type.csv`.
2. Resolve expected target text from `text/`.
3. Compute `target_chars` from newline-stripped target text.
4. Run `crop_image(...)` from `cropper_pipeline.py`.
5. Run output post-process (`output.py`) for periodic stripe cleanup.
6. Save cropped image + OCR text.
7. Compute OCR quality (Levenshtein distance) and print timing/quality report.

### 2) Single-image crop pipeline (`cropper_pipeline.py`)
`crop_image(...)` executes:
1. OCR pass 1 (`ocr._ocr_image_pil`).
2. Line-based correction decision (`core.line_fix.decide_correction`):
   - `none`, `tilt`, or `warp`.
3. Apply correction when enabled:
   - tilt via affine transform, or
   - warp via mesh built from OCR line structure.
4. Optional OCR pass 2 after geometry changes.
5. Cluster OCR words and compute final crop bbox with edge alignment:
   - `core.edge_candidates`
   - `core.crop_alignment`
   - `core.edge_shift`
6. Final crop + optional denoise.
7. Build stripe-related intermediates (`core.lighting.build_post_crop_stripes`).
8. Return cropped image, OCR text, correction metadata, and timing details.

### 3) Output post-process (`output.py` + `stripes.py`)
1. Estimate lag bounds from average character size.
2. Detect periodic stripe rows on the normalized stripe image.
3. Build light debug output by filling stripe zones from surrounding statistics.
4. Return final rendered output image + periodic metadata.

### 4) Web path (`web_app.py` -> `web_api.py`)
1. `web_app.py` handles upload/job lifecycle/streamed logs.
2. `web_api.py` calls shared pipeline (`crop_image`) and output stage (`build_output_image_from_crop_result`).
3. Returns JPEG bytes + metadata JSON.

## Design Rules (Implemented)

- Web app is transport/deployment only (`web_app.py`).
- Crop/output behavior lives in reusable functions (`cropper_pipeline.py`, `output.py`, `src/core/*`).
- CLI benchmark and web flow share the same crop/output logic.

## Configuration

All tunables are centralized in `src/config.py` under `settings`, segmented by domain:
- `settings.paths`
- `settings.ocr`
- `settings.crop_service`
- `settings.periodic`
- `settings.lighting`

## Deployment

- `render.yaml` + `Dockerfile` are provided for Render deployment.
- OCR is CPU-bound; higher CPU instances materially improve throughput.
