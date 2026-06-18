# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project purpose

Image pre-processing for STaM (Sefer Torah, Tefillin, Mezuzah) manuscripts: take a raw photo or
scan of parchment text and produce a tightly-cropped image of the text region, ready for downstream
analysis.

Two distinct areas:

| Area | Scripts | Purpose |
|------|---------|---------|
| **Production** | `crop_stam.py`, `crop_with_model.py` | Crop images |
| **Training** | `training/` directory | Build datasets and retrain the model |

`google_ocr.py` is a training utility (caches Vision API results to JSON for labelling), not part
of the production pipeline.

---

## Production pipeline — `crop_stam.py`

```bash
python3 crop_stam.py --image path/to/photo.jpg
python3 crop_stam.py --images-dir images/benchmark --out-dir test_output/cropped
```

### Steps (per image)

1. **Load** with `cv2.IMREAD_IGNORE_ORIENTATION` — keeps pixels in the same coordinate frame as
   Google Vision (Vision ignores EXIF; cv2 normally auto-applies it).

2. **Segmentation + OCR in parallel** (`ThreadPoolExecutor(max_workers=2)`)

   - **Segmentation** (`crop_with_model.py`): YOLOv8-seg (`best.pt`) → text-region polygon.
     Dynamic `imgsz` ensures the short dimension maps to ≥ 128 px in model input.
     For images with aspect ratio > 10:1, 3-tile tiled inference with 15% overlap; tile polygons
     merged via convex hull.

   - **OCR** (`run_ocr`): Google Vision `document_text_detection`, downscaled to ≤ 1 MP before
     sending. Accepts a file path or a numpy array. Returns Hebrew char boxes (`א`–`ת`) in
     original pixel coordinates.

3. **Extend OCR** (`adjacent_ocr_corners`): group OCR boxes into connected text clusters (dilate
   footprints to bridge inter-character gaps → `connectedComponents`). Include every cluster that
   touches the model polygon — so capturing one end of a text line pulls in the whole line.
   Runs on a ≤ 1000 px canvas for speed.

4. **OCR convex hull**: `cv2.convexHull` over accepted OCR box corners.

5. **Union polygon** (`union_polygon`): binary-mask OR of model polygon and OCR hull → outer
   contour = final text boundary.

6. **Background colour**: Otsu-threshold pixels inside the model polygon to separate ink from
   parchment; average the parchment pixels → fill colour for the exterior.

7. **Fill and crop**: fill outside the union polygon with the background colour; crop to bounding
   box.

8. **Rotate**: if the cropped region has H/W ≥ `ROTATE_RATIO` (3.0), rotate 90° CCW. Ratio is
   computed on the actual cropped text region, not the full image.

9. **Background flattening**: for large, roughly square crops only (`min(h,w) ≥ 500` and aspect
   ratio ≤ 4:1) — pull background pixel luminance (L channel, above Otsu threshold + 20) toward
   the mean background tone with a 0.3 retention factor. Suppresses parchment texture without
   affecting ink or changing overall brightness.

10. **Save** as JPEG (quality 92).

### Key constants

```python
ROTATE_RATIO    = 3.0    # rotate CCW when crop H/W exceeds this
OCR_MAX_PIXELS  = 1_000_000
CLUSTER_GAP_PCT = 0.04   # dilation gap for text cluster bridging
MODEL_REACH_PCT = 0.04   # dilation on model mask to catch border boxes
EXTEND_WORK_PX  = 1000   # max canvas size for connectedComponents step
CONF            = 0.25   # YOLO detection confidence threshold
```

### Timing (benchmark average over 39 images)
- Wall time ≈ 620 ms (seg + OCR in parallel ≈ 575 ms, extend ≈ 45 ms)
- Bottleneck: Google Vision API (~575 ms, irreducible)

---

## Debug / benchmark — `debug_combined.py`

Runs the same pipeline as `crop_stam.py` and draws polygon overlays with per-image timing.

```bash
python3 debug_combined.py --images-dir images/benchmark
python3 debug_combined.py --image images/benchmark/mezuza3.jpeg
```

Writes:
- `test_output/debug_combined/<stem>_debug.jpg` — overlay image
- `test_output/cropped_combined/<stem>_cropped.jpg` — cropped output (identical to `crop_stam.py`)

Colour key: green = included OCR boxes, dark-red = excluded, blue = OCR hull,
bright-green = model polygon, red (thick) = final union boundary.

---

## Interactive tester — `roi_inference.py`

OpenCV window to run the model on a user-drawn ROI.

```bash
python3 roi_inference.py   # loads images/benchmark/
```

Controls: N/P = next/prev image, drag = select ROI, Enter = run model, R = clear, Q = quit.

---

## Training pipeline — `training/`

Use when model quality degrades or new failure patterns appear.

### Steps

1. **OCR new images** (if needed):
   ```bash
   python3 google_ocr.py --images-dir images/corpus --output-dir test_output --skip-existing
   ```

2. **Edit labels** — `training/polygon_editor.py`: interactive tool to approve/adjust YOLO
   polygon labels image by image.

3. **Build dataset** — `training/build_yolo_polygon.py`: writes `yolo_polygon/` in YOLO seg
   format. Optional synthetic augmentation via `training/generate_synthetic_scroll.py` +
   `training/build_synthetic_yolo.py`.

4. **Train** — upload `yolo_polygon/` to Google Drive, run `training/train_yolo.ipynb` on Colab.

5. **Deploy** — replace `best.pt` in the repo root; production pipeline picks it up automatically.

### Training data layout

```
stam polygon.coco/          # COCO-format labelled dataset (test + val splits)
stam_polygon_augmented/     # Augmented training split
yolo_polygon/               # Built by build_yolo_polygon.py, uploaded to Colab
```

---

## Dependencies

| Package | Used by |
|---------|---------|
| `ultralytics` | `crop_stam.py`, `crop_with_model.py` |
| `google-cloud-vision` | `crop_stam.py`, `google_ocr.py` |
| `opencv-python` (`cv2`) | everything |
| `numpy` | everything |
| `Pillow` (`PIL`) | `training/` scripts |

Google Vision authenticates via Application Default Credentials
(`gcloud auth application-default login`) or `GOOGLE_APPLICATION_CREDENTIALS` env var.

## Image folders

- `images/benchmark/` — 39 hand-picked test images for development and benchmarking.
- `images/corpus/` — full labelled corpus organised by document type (megilot, mezuzot, tefillin, torah).
