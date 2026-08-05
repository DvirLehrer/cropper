# STaM Crop

Takes a raw photo or scan of a STaM parchment — Sefer Torah, Tefillin, Mezuzah,
Megillah — and produces a tightly cropped, straightened image of the writing,
ready for a letter recogniser to read.

It is judged on whether the recogniser reads the parchment correctly afterwards,
not on whether the crop looks right. On a frozen set of 157 real customer
photographs, scored against the production engine:

| | untouched photo | after this cropper |
|---|---|---|
| the engine returns a result at all | 52.9% | 99.4% |
| of the known reference text, how much it read | 40.5% | 86.6% |
| 90% or more of the text recovered | — | 69.4% |

Six times as many images come back with fewer than twenty reported errors.

Median 0.99 s an image.

The first row assumes the three crashes patched in `dist/stam-ocr-crash-fixes.zip`
are applied. Without them the engine throws on 14 of the 157 no matter what the
cropper hands it, and the figure is 91.1%.

## Installation

Python 3.9 or newer.

```bash
pip install ultralytics google-cloud-vision opencv-python numpy Pillow pillow-heif
```

`pillow-heif` is not optional. iPhone uploads frequently arrive as HEIC with a
`.jpg` extension; `cv2.imread` returns `None` for those, and without this package
they are skipped silently rather than failing loudly.

### Google Cloud Vision

The OCR step calls the Cloud Vision API, which needs a Google Cloud project with
billing enabled. Free tier is 1,000 units a month.

1. Create or select a project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the [Cloud Vision API](https://console.cloud.google.com/apis/library/vision.googleapis.com)
3. Authenticate, either with the CLI:

   ```bash
   gcloud auth application-default login
   ```

   or with a service account key — **IAM & Admin → Service Accounts**, role
   *Cloud Vision API User*, download the JSON, then:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
   ```

### Model weights

`best.pt` is in the repository root. Nothing to download.

## Running it

```bash
# one image
python3 crop_stam.py --image path/to/photo.jpg

# a directory
python3 crop_stam.py --images-dir photos/ --out-dir cropped/
```

Output is always JPEG at quality 92, named `<stem>_cropped.jpg`, regardless of
the input extension — `cv2.imwrite` cannot encode HEIC.

## Calling it from an application

```python
from ultralytics import YOLO
from crop_stam import crop_image

model = YOLO('best.pt')          # load once, reuse — loading costs ~1 s
ok = crop_image(model, 'upload.jpg', out_dir='cropped/')
```

`crop_image` returns `False` and leaves nothing behind when the image cannot be
decoded or no text is found. It writes to disk rather than returning an array;
if in-memory output would suit the calling code better, that is a small change
to the tail of the function.

After a call, `crop_stam.LAST_DIAG` holds what the pipeline decided for that
image — measured grain, whether denoising fired, the deskew angle. Useful when a
particular upload comes out wrong.

## What it does, in order

1. **Decode** via `stam_io.imread_any` — JPEG, PNG and HEIC. EXIF orientation is
   deliberately *not* applied, so the pixels stay in the same coordinate frame as
   Google Vision, which also ignores it.

2. **Segmentation and OCR, in parallel.** Both receive the same decoded array,
   never the file path: handing a path to Ultralytics makes it run its own
   `cv2.imread`, which *does* apply EXIF, and for an image tagged
   `orientation=6` that puts the model polygon 90° away from the OCR boxes.

   - YOLOv8-seg (`best.pt`) returns the text-region polygon. Images beyond 10:1
     get three overlapping tiles, merged by convex hull.
   - Google Vision `document_text_detection`, downscaled to ≤ 1 MP, returns
     Hebrew character boxes in original pixel coordinates.

3. **Deskew.** The mean baseline angle over the Vision character quads. Applied
   as one `warpAffine` that rotates and crops together. Below 0.5° nothing
   happens.

4. **Re-segment.** After straightening, the model is run again: a slanted
   parchment is out of distribution for it, and the second polygon on the
   corrected image is materially better. Only above 2°.

5. **Boundary.** Character boxes are grouped into connected text clusters, every
   cluster touching the model polygon is taken whole — so catching one end of a
   line pulls in the rest of it — and the convex hull of those is OR'd with the
   model polygon.

6. **Fill and crop.** Parchment colour is sampled from non-ink pixels inside the
   polygon and painted outside the boundary; the result is cropped to it.

7. **Denoise, if the parchment is grainy.** Grain is measured on the block of
   writing rather than the whole crop — measured over the whole crop it is the
   grain of the table the sheet was lying on. Above the threshold, non-local
   means; ink is restored exactly afterwards. Roughly one image in twenty
   qualifies.

8. **Upscale, if the letters are small.** The engine reads 97% of the text when
   characters are 20 px and above, 86% between 14 and 20, and 56% below that.
   Enlarging adds no detail; it hands the next stage something inside the size
   range it can work with at all.

9. **Flatten the background** on large squarish crops, and save.

## Tuning

Every constant is overridable from the environment, and each run records what it
used:

```bash
GRAIN_MIN=12 MIN_CHAR_PX=32 python3 crop_stam.py --images-dir photos/
```

The defaults were each chosen by measurement against the scoring engine, and
`HANDOFF.md` records what was tried and what the numbers said. Changing one
without re-measuring is likely to cost more than it gains.

## Looking at the output

```bash
python3 debug_combined.py --images-dir photos/
```

Writes overlays to `test_output/debug_combined/`. Green = included character
boxes, dark red = excluded, blue = the OCR hull, bright green = the model
polygon, thick red = the final boundary.

## Measuring a change

```bash
./tools/setup_scoring_macos.sh                          # once
python3 tools/bench.py crop  --run mychange
.venv-scoring/bin/python tools/bench.py score --run mychange
python3 tools/bench.py report --run mychange --vs final3
```

`report --vs` prints which settings differed between the two runs, and says so
loudly when none did. `current/index.html` shows every crop beside its original,
filterable, rebuilt as the run proceeds.

## Layout

```
crop_stam.py          production pipeline and CLI
crop_with_model.py    YOLO inference
stam_io.py            decoding, including HEIC
texture.py            grain measurement and suppression
deruling.py           ruled-line suppression (written, switched off)
best.pt               model weights
tools/                benchmark harness, parameter search, diagnostics
training/             dataset building and retraining
dist/                 the crash patch for the company's engine
HANDOFF.md            why the code looks like this, and what to do next
```

## Retraining

When the model starts missing a new kind of image:

1. Label — `training/polygon_editor.py`
2. OCR new images if needed — `google_ocr.py` caches Vision results to JSON
3. Build — `training/build_yolo_polygon.py` writes `yolo_polygon/`
4. Train — upload `yolo_polygon/` to Drive, run `training/train_yolo.ipynb` on Colab
5. Deploy — replace `best.pt`
