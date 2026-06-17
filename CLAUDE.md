# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Image **pre-processing** for STaM (Sefer Torah, Tefillin, Mezuzah) imagery: take a raw photo or
scan of parchment text and produce a tightly-cropped image of just the text region, ready for a
downstream analysis system.

Two distinct areas:

| Area | Scripts | Purpose |
|------|---------|---------|
| **Production** | `crop_stam.py`, `crop_with_model.py`, `google_ocr.py` | Crop an image in production |
| **Training** | `training/` directory | Build datasets and retrain the model |

---

## Production pipeline — `crop_stam.py`

```bash
python3 crop_stam.py --image path/to/photo.jpg
python3 crop_stam.py --images-dir images/benchmark --out-dir test_output/cropped
```

### Steps (per image)

1. **Load** with `cv2.IMREAD_IGNORE_ORIENTATION` — keeps pixels in the same frame as Google Vision
   coordinates (Vision ignores EXIF; cv2 normally auto-applies it).

2. **Segmentation + OCR in parallel** (`ThreadPoolExecutor(max_workers=2)`)

   - **Segmentation** (`crop_with_model.py`): YOLOv8-seg (`best.pt`) → text-region polygon.
     Dynamic `imgsz` ensures the short dimension is ≥ 128 px in model input.
     For images with aspect ratio > 10:1, 3-tile tiled inference with 15% overlap; tile polygons
     merged via convex hull.

   - **OCR** (`crop_stam.py::run_ocr`): Google Vision `document_text_detection`, downscaled to
     ≤ 1 MP before sending. Returns Hebrew char boxes (`א`–`ת`) in original pixel coords.

3. **Extend OCR** (`crop_stam.py::adjacent_ocr_corners`): group OCR boxes into connected text
   clusters (dilate footprints to bridge inter-character gaps → `connectedComponents`). Include
   every cluster that touches the model polygon — so if the model captures one end of a text line,
   the whole line is included. Runs on a ≤ 1000 px downscaled canvas for speed.

4. **OCR convex hull**: `cv2.convexHull` over accepted OCR box corners.

5. **Union polygon** (`crop_stam.py::union_polygon`): binary-mask OR of model polygon and OCR
   hull → outer contour = final text boundary.

6. **Background colour**: Otsu-threshold pixels inside the model polygon to separate ink from
   parchment; average the parchment pixels → fill colour for outside area.

7. **Output**: fill outside union polygon with background colour, crop to polygon bounding box,
   save as JPEG.

### Timing (benchmark average over 39 images)
- Wall time ≈ 630 ms (seg + OCR run in parallel ≈ 580 ms, extend ≈ 47 ms)
- Bottleneck: Google Vision API (~580 ms, irreducible)

---

## Debug / benchmark — `debug_combined.py`

Runs the same pipeline as `crop_stam.py` and additionally draws polygon overlays + timing:

```bash
python3 debug_combined.py --images-dir images/benchmark
python3 debug_combined.py --image images/benchmark/mezuza3.jpeg
```

Writes `test_output/debug_combined/<stem>_debug.jpg` (overlay) and
`test_output/cropped_combined/<stem>_cropped.jpg` (cropped output).

Colour key: green boxes = included OCR, dark-red = remote/excluded, blue = OCR hull,
bright-green = model polygon, red (thick) = final union boundary.

---

## Interactive tester — `roi_inference.py`

OpenCV window to run the model on a user-drawn ROI and inspect the polygon interactively.

```bash
python3 roi_inference.py   # loads images from images/benchmark/
```

Controls: N/P = next/prev image, drag = select ROI, Enter = run model, R = clear, Q = quit.

---

## Training pipeline — `training/`

Use when model quality degrades or new failure patterns appear.

### Steps

1. **OCR** (if not already done):
   ```bash
   python3 google_ocr.py --images-dir images/before_crop --output-dir test_output --skip-existing
   ```

2. **Edit labels** — `training/polygon_editor.py`: interactive tool to approve/adjust YOLO
   polygon labels image by image.

3. **Build dataset** — `training/build_yolo_polygon.py`: writes `yolo_polygon/` in YOLO seg
   format. Optional synthetic augmentation via `training/generate_synthetic_scroll.py` +
   `training/build_synthetic_yolo.py`.

4. **Train** — upload `yolo_polygon/` to Google Drive, run `training/train_yolo.ipynb` on Colab.

5. **Deploy** — replace `best.pt` in the repo root; production pipeline picks it up automatically.

---

## Dependencies

No `requirements.txt`. Active imports tell you what's needed:

| Package | Used by |
|---------|---------|
| `ultralytics` | `crop_stam.py`, `crop_with_model.py` |
| `google-cloud-vision` | `crop_stam.py`, `google_ocr.py` |
| `opencv-python` (`cv2`) | everything |
| `numpy` | everything |
| `Pillow` (`PIL`) | `training/` scripts |

Google Vision authenticates via Application Default Credentials
(`gcloud auth application-default login`) or `GOOGLE_APPLICATION_CREDENTIALS` env var.

## Image folder conventions

- `images/benchmark/` — 39 hand-picked test images used for development and benchmarking.
- `images/before_crop/`, `images/after_crop/` — staged corpora at different preprocessing steps.
- `images/_test_set/` — small default set for `google_ocr.py`.
