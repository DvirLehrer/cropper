Cropper OCR Benchmarks

This repo contains OCR benchmarking scripts and image assets for mezuzah scans.

Quick Start
- Run OCR benchmark on all images:
  - `python3 src/ocr_benchmark.py`
- Output folders (generated):
  - `ocr_text/` : per-image OCR text
  - `debug_images/` : debug overlays (overall bbox, word boxes, line boxes, edge lines)
  - `preprocessed/` : preprocessed images used for OCR

Repository Structure
- `benchmark/` : input images
- `text/` : canonical Hebrew texts (shema, vehaya, kadesh, peter)
- `src/ocr_benchmark.py` : main benchmark runner
- `src/ocr_utils.py` : shared OCR helpers (preprocess, OCR, line clustering, Levenshtein)
- `src/edge_align_shift.py` : edge alignment sweep for crop boundaries
- `src/line_*` : line structure, correction, and block mesh utilities

Key Behavior
- OCR is done with Tesseract `--psm 4` on preprocessed images.
- Text output is assembled from the same word boxes used for debug overlays.
- Line boxes are derived via y-clustered word grouping.

Goal & Strategy
We aim to produce a precise crop of the mezuzah and tefillin text regions. The current workflow:
- Detect the initial text area from OCR word boxes.
- Fit line structure and optionally correct tilt/warp.
- Estimate tight crop boundaries from edge-aligned word boxes.

This is important because OCR boxes can be incomplete in shadows or noisy at the periphery. The goal is to keep the tightest crop that preserves all text.

Glossary
- OCR word box: A per-word bounding box returned by Tesseract.
- Line words: OCR word boxes grouped into a single text line by y-clustering.
- Line box: The bounding box that encloses all words in a line.
- Document bbox: The bounding box around detected OCR content before edge refinement.
- Optimal lines: The tightest crop boundaries (left, top, right, bottom) that define the final crop.
- Bottom line anomaly: When short final lines get cropped off by the optimal lines.
- Edge alignment line: A sweep line used to align left or right edges from candidate boxes.
- Tilt correction: A small rotation applied when OCR line baselines are slanted.
- Warp correction: A non-linear adjustment to reduce line curvature based on OCR line structure.

Target Texts
- Mezuzah uses the combined Shema + Vehaya text.
- Tefillin uses four texts: Shema, Vehaya, Kadesh, Peter (per `text/` files).

Configurable Parameters
Defined at the top of `src/ocr_benchmark.py`:
- `APPLY_CROP_DENOISE` : toggle median-filter denoise on saved crops
- `CROP_DENOISE_SIZE` : median filter kernel size
- `ENABLE_LINE_WARP` : toggle non-linear line warp correction
- `APPLY_LINE_CORRECTION` : toggle tilt/warp correction

Output Conventions
- Hebrew output in debug or scripts is displayed reversed for readability.
- Source files should not exceed 200 lines; split into modules as needed.

Notes
- `ocr_text/` and `debug_images/` must be regenerated after logic changes.
- If text/boxes look misaligned, verify that `ocr_text` is built from the same OCR word boxes.
