# STaM Crop

Image pre-processing for STaM (Sefer Torah, Tefillin, Mezuzah) manuscripts: takes a raw photo or
scan of parchment text and produces a tightly-cropped image of the text region, ready for downstream
analysis.

## Installation

### 1. Python dependencies

Requires Python 3.8+.

```bash
pip install ultralytics google-cloud-vision opencv-python numpy Pillow
```

### 2. Google Cloud Vision API

The OCR step calls the Google Cloud Vision API, which requires a Google Cloud project with billing enabled.

**One-time setup:**

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and create or select a project.
2. Enable the **Cloud Vision API** for that project:
   [console.cloud.google.com/apis/library/vision.googleapis.com](https://console.cloud.google.com/apis/library/vision.googleapis.com)
3. Set up billing for the project (Vision API has a free tier of 1,000 units/month).

**Authentication — choose one option:**

**Option A: `gcloud` CLI (easiest for local development)**

```bash
# Install the gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth application-default login
```

**Option B: Service account key file**

1. In the Cloud Console, go to **IAM & Admin → Service Accounts**.
2. Create a service account and grant it the **Cloud Vision API User** role.
3. Create a JSON key for it and download it.
4. Point the environment variable at the file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"
```

Add that line to your `~/.zshrc` or `~/.bashrc` to make it permanent.

### 3. Model weights

The YOLOv8 segmentation model (`best.pt`) must be present in the repo root. It is not included in
the repository due to file size. Download it from the project's release assets and place it at:

```
cropper/best.pt
```

## Quick start

```bash
# Single image
python3 crop_stam.py --image path/to/photo.jpg

# Batch
python3 crop_stam.py --images-dir images/benchmark --out-dir test_output/cropped
```

## Dependencies

| Package | Used by |
|---------|---------|
| `ultralytics` | `crop_stam.py`, `crop_with_model.py` |
| `google-cloud-vision` | `crop_stam.py`, `google_ocr.py` |
| `opencv-python` | everything |
| `numpy` | everything |
| `Pillow` | `training/` scripts |

Google Vision authenticates via Application Default Credentials (`gcloud auth application-default login`)
or the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

## Project layout

```
crop_stam.py          # Production CLI
crop_with_model.py    # YOLO inference helpers (imported by crop_stam.py)
google_ocr.py         # Standalone OCR tool for labelling new training data
debug_combined.py     # Benchmark visualiser — runs the pipeline and draws overlays
roi_inference.py      # Interactive model tester (OpenCV window)
best.pt               # YOLOv8-seg model weights
images/
  benchmark/          # 39 hand-picked test images
  corpus/             # Full image corpus (megilot, mezuzot, tefillin, torah)
training/             # Dataset building and retraining tools
```

## Pipeline

Each image goes through the following steps in `crop_stam.py`:

1. **Load** with `cv2.IMREAD_IGNORE_ORIENTATION` — preserves raw pixel coordinates to match
   Google Vision (which also ignores EXIF rotation).

2. **Segmentation + OCR in parallel** (`ThreadPoolExecutor`)
   - **Segmentation**: YOLOv8-seg (`best.pt`) detects the text-region polygon. For extreme-ratio
     images (> 10:1), 3-tile tiled inference with 15% overlap; polygons merged via convex hull.
   - **OCR**: Google Vision `document_text_detection`, downscaled to ≤ 1 MP before sending.
     Returns Hebrew character bounding boxes (`א`–`ת`) in original pixel coordinates.

3. **Extend OCR**: Hebrew boxes are grouped into connected text clusters (dilated footprints →
   `connectedComponents`). Any cluster that touches the model polygon is included in full — so
   if the model captures one end of a line, the rest of the line is pulled in automatically.

4. **Union polygon**: convex hull over included OCR corners, then binary-mask OR with the model
   polygon → outer contour = final text boundary.

5. **Fill and crop**: sample parchment colour from non-ink pixels inside the model polygon (Otsu
   threshold on L channel). Fill everything outside the boundary with that colour. Crop to the
   boundary bounding box.

6. **Rotate**: if the cropped region is taller than 3× its width, rotate 90° counter-clockwise.

7. **Background flattening**: for large, roughly square crops (short side ≥ 500 px, aspect ≤ 4:1),
   pull background pixel luminance toward the mean background tone, reducing parchment texture
   contrast without touching the ink.

**Average wall time: ~620 ms** (segmentation ~65 ms + OCR ~575 ms in parallel; extend ~45 ms).
Bottleneck is the Google Vision API call.

## Debug / benchmark

```bash
python3 debug_combined.py --images-dir images/benchmark
python3 debug_combined.py --image images/benchmark/mezuza3.jpeg
```

Writes to `test_output/debug_combined/` (polygon overlays) and `test_output/cropped_combined/`
(cropped output identical to `crop_stam.py`).

Overlay colour key: green = included OCR boxes, dark-red = excluded, blue = OCR hull,
bright-green = model polygon, red (thick) = final union boundary.

## Retraining

Use the tools in `training/` when the model needs updating:

1. **Label new images** — `training/polygon_editor.py`: interactive tool to approve/edit YOLO
   polygon labels.
2. **OCR new images** (if needed) — `google_ocr.py`: caches Vision API results to JSON.
3. **Build dataset** — `training/build_yolo_polygon.py`: converts labels to YOLO seg format in
   `yolo_polygon/`. Optional synthetic data via `generate_synthetic_scroll.py` +
   `build_synthetic_yolo.py`.
4. **Train** — upload `yolo_polygon/` to Google Drive, run `training/train_yolo.ipynb` on Colab.
5. **Deploy** — replace `best.pt` in the repo root.
