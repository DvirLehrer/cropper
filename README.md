Cropper OCR Benchmarks

This repo contains OCR benchmarking scripts and image assets for mezuzah scans.

Quick Start
- Run OCR benchmark on all images:
  - `python3 src/ocr_benchmark.py`
- Output folders (generated):
  - `ocr_text/` : per-image OCR text
  - `debug_images/` : debug overlays (overall bbox, word boxes, line boxes, best-sequence boxes)
  - `preprocessed/` : preprocessed images used for OCR

Repository Structure
- `benchmark/` : input images
- `text/` : canonical Hebrew texts (shema, vehaya, kadesh, peter)
- `src/ocr_benchmark.py` : main benchmark runner (kept <= 200 LOC)
- `src/ocr_utils.py` : shared OCR helpers (preprocess, OCR, line clustering, Levenshtein)
- `src/best_sequence.py` : best-substring scoring module
- `src/ocr_benchmark_old.py` : snapshot of older script state

Key Behavior
- OCR is done with Tesseract `--psm 4` on preprocessed images.
- Text output is assembled from the same word boxes used for debug overlays.
- Line boxes are derived via y-clustered word grouping.
- Best sequences are chosen per line as OCR word-boundary substrings and matched to full target-word segments.

Goal & Strategy
We aim to produce a precise crop of the mezuzah and tefillin text regions. The current workflow:
- Detect the initial text area from OCR word boxes.
- Extract reliable OCR substrings (best sequences) to anchor alignment with the canonical target text.
- Use those anchors to estimate letter size and line length, then infer the expected text area even in shadowed regions.

This is important because OCR boxes can be incomplete in shadows or noisy at the periphery. The goal is to infer the full expected text block (based on the target text + estimated character size/line length) and then:
- Recover missing boxes in dark/shadowed areas.
- Remove peripheral noisy boxes that do not fit the expected block.

Target Texts
- Mezuzah uses the combined Shema + Vehaya text.
- Tefillin uses four texts: Shema, Vehaya, Kadesh, Peter (per `text/` files).

TODO (Open Work)
- Estimate letter size and line length from reliable OCR substrings.
- Compute the expected text block area from target text structure.
- Expand or adjust boxes to include shadowed text while filtering peripheral noise.

Configurable Parameters
Defined at the top of `src/ocr_benchmark.py`:
- `TOP_K` : number of best sequences to highlight
- `WINDOW_WORDS` : size of OCR word window per line
- `MAX_SKIP` : max starting offset (in words) for window search

Output Conventions
- Hebrew output in debug or scripts is displayed reversed for readability.
- Source files should not exceed 200 lines; split into modules as needed.

Notes
- `ocr_text/` and `debug_images/` must be regenerated after logic changes.
- If text/boxes look misaligned, verify that `ocr_text` is built from the same OCR word boxes.
