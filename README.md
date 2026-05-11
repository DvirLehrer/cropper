# stam

Hebrew character recognition pipeline for STaM (Sefer Torah, Tefillin, Mezuzah) imagery.

Four standalone Python scripts that communicate via files on disk under `test_output/`:

1. **`google_ocr.py`** — runs Google Cloud Vision `document_text_detection` on input images and writes:
   - `<base>_char_boxes.json` — per-symbol records (character, 4-vertex polygon, OCR hierarchy indices).
   - `<base>_bbox.<ext>` + `<base>_warped.<ext>` — page-level quad (fit to Hebrew points only) and its perspective-warped rectangle.
   - `<base>_all_boxes.<ext>` — overlay of every character quad.
   - `char_dataset/<class>/...png` — per-character crops grouped by class folder `U<HEX>_<char>`, ready for training.

2. **`export_coco_from_ocr.py`** — turns the `*_char_boxes.json` files into a single COCO JSON for detection-style use.

3. **`train_cnn.py`** — trains a small CNN classifier on the per-class crops. Writes `model.pt`, `meta.json`, `confusion_matrix.csv`, `per_class_accuracy.json`.

4. **`infer_cnn_boxes.py`** — uses the trained CNN to classify the OCR-proposed boxes from step 1 and writes annotated overlays. The CNN does not localize — Google Vision is still the source of box geometry.

## Setup

No `requirements.txt`. Install what each script imports into your active Python env:

- All scripts: `numpy`, `Pillow`, `opencv-python`
- `google_ocr.py`: `google-cloud-vision`, plus `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service-account JSON
- `train_cnn.py`, `infer_cnn_boxes.py`: `torch`

Torch is imported lazily, so the OCR and COCO scripts run fine without it.

## Usage

```bash
# 1. OCR + per-character crop export
python google_ocr.py --images-dir images/before_crop --output-dir test_output \
    --dataset-out test_output/char_dataset --skip-existing

# 2. COCO export (optional)
python export_coco_from_ocr.py --images-dir images/before_crop --boxes-dir test_output

# 3. Train classifier
python train_cnn.py --data-dir test_output/char_dataset --out-dir test_output/cnn_model

# 4. Classify OCR boxes and write annotated images
python infer_cnn_boxes.py --test-dir images/_test_set --char-boxes-dir test_output \
    --model-dir test_output/cnn_model --out-dir test_output/cnn_boxes
```

Single-image OCR: `python google_ocr.py --image path/to/file.jpg`.
With no input flag, `google_ocr.py` defaults to `images/_test_set/*.{jpg,jpeg,png}`.

## Notes

- `train_cnn.py::pil_to_tensor_gray` autocontrasts, flips polarity if the median pixel is dark, and Otsu-binarizes by default. `infer_cnn_boxes.py` reuses the same function — keep `--binarize` consistent between training and inference.
- `export_coco_from_ocr.py` currently writes `category_id: 1` for every annotation (single-class detection setup); the per-character category map is built but unused.
- Images, model artifacts, and derived `test_output/` data are gitignored. The repo tracks code only.
