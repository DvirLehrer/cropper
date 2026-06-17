#!/usr/bin/env python3
"""
roi_inference.py — Interactive ROI model tester for benchmark images.

Select a region on the image, press Enter to run the model on that region
and see the segmented polygon overlaid in full-image coordinates.

Controls:
  N / P          next / previous image
  click + drag   draw ROI rectangle
  Enter / Space  run model on current ROI
  R              clear ROI
  Q / Esc        quit
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from crop_with_model import _predict, _imgsz_for

BENCH_DIR = Path('images/benchmark')
OCR_DIR   = Path('test_output')
MODEL_PATH = 'best.pt'
HEBREW    = ('א', 'ת')

DISP_W, DISP_H = 1400, 800
WIN = 'ROI Inference  [N/P=image  drag=select  Enter=run  R=clear  Q=quit]'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_ocr(stem):
    p = OCR_DIR / f'{stem}_char_boxes.json'
    if p.exists():
        return p
    base = re.sub(r'_jpg$', '', stem.split('.rf.')[0]).replace('-', '.')
    p2 = OCR_DIR / f'{base}_char_boxes.json'
    return p2 if p2.exists() else None


def load_hebrew_boxes(ocr_path):
    with open(ocr_path, encoding='utf-8') as f:
        data = json.load(f)
    return [b for b in data.get('boxes', [])
            if b.get('char', '') and HEBREW[0] <= b['char'][0] <= HEBREW[1]]


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class App:
    def __init__(self, model):
        self.model    = model
        self.images   = sorted(p for p in BENCH_DIR.glob('*')
                               if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        self.idx      = 0
        self.img      = None        # original full-res image
        self.H = self.W = 0
        self.scale    = 1.0         # display scale
        self.canvas   = None        # what we draw on
        self.ocr_boxes = []

        # ROI drag state (display coords)
        self.drag_start = None
        self.drag_end   = None
        self.dragging   = False

        # Last result polygon (image coords)
        self.result_poly = None

        self.load_image(0)

    # ------------------------------------------------------------------
    def load_image(self, idx):
        self.idx = idx % len(self.images)
        path = self.images[self.idx]
        self.img = cv2.imread(str(path), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if self.img is None:
            print(f'Cannot load {path}')
            return
        self.H, self.W = self.img.shape[:2]
        self.scale = min(DISP_W / self.W, DISP_H / self.H)

        ocr_path = find_ocr(path.stem)
        self.ocr_boxes = load_hebrew_boxes(ocr_path) if ocr_path else []

        self.drag_start = self.drag_end = None
        self.result_poly = None
        self.draw()
        print(f'\n[{self.idx+1}/{len(self.images)}] {path.name}  '
              f'{self.W}x{self.H}  {len(self.ocr_boxes)} Hebrew boxes')

    # ------------------------------------------------------------------
    def img_to_disp(self, x, y):
        return int(x * self.scale), int(y * self.scale)

    def disp_to_img(self, dx, dy):
        return dx / self.scale, dy / self.scale

    def roi_img_coords(self):
        """Return (x0,y0,x1,y1) in image coords, or None."""
        if self.drag_start is None or self.drag_end is None:
            return None
        dx0, dy0 = self.drag_start
        dx1, dy1 = self.drag_end
        x0, y0 = self.disp_to_img(min(dx0, dx1), min(dy0, dy1))
        x1, y1 = self.disp_to_img(max(dx0, dx1), max(dy0, dy1))
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.W, int(x1)), min(self.H, int(y1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return x0, y0, x1, y1

    # ------------------------------------------------------------------
    def draw(self):
        disp = cv2.resize(self.img, (int(self.W * self.scale), int(self.H * self.scale)))

        # OCR boxes (blue)
        for b in self.ocr_boxes:
            pts = np.array([[int(v['x'] * self.scale), int(v['y'] * self.scale)]
                            for v in b['vertices']], np.int32)
            cv2.polylines(disp, [pts], True, (220, 120, 0), 1)

        # Result polygon (green)
        if self.result_poly is not None:
            pts = np.array([[int(x * self.scale), int(y * self.scale)]
                            for x, y in self.result_poly], np.int32)
            cv2.polylines(disp, [pts], True, (0, 220, 50), 3)
            for pt in pts:
                cv2.circle(disp, tuple(pt), 5, (0, 220, 50), -1)

        # ROI rectangle (white dashed-look via yellow)
        if self.drag_start and self.drag_end:
            dx0, dy0 = self.drag_start
            dx1, dy1 = self.drag_end
            x0, x1 = min(dx0, dx1), max(dx0, dx1)
            y0, y1 = min(dy0, dy1), max(dy0, dy1)
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 200, 255), 2)

        # Status bar
        dh, dw = disp.shape[:2]
        bar = np.zeros((36, dw, 3), dtype=np.uint8)
        name = self.images[self.idx].name
        roi = self.roi_img_coords()
        roi_str = f'ROI: {roi[0]},{roi[1]} → {roi[2]},{roi[3]}  ({roi[2]-roi[0]}x{roi[3]-roi[1]})' if roi else 'drag to select ROI'
        label = f'[{self.idx+1}/{len(self.images)}] {name}   {roi_str}'
        cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        self.canvas = np.vstack([disp, bar])
        cv2.imshow(WIN, self.canvas)

    # ------------------------------------------------------------------
    def run_model(self):
        roi = self.roi_img_coords()
        if roi is None:
            print('No ROI selected — drag a rectangle first')
            return
        x0, y0, x1, y1 = roi
        crop = self.img[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        imgsz = _imgsz_for(cw, ch)
        print(f'Running model on ROI {x0},{y0}→{x1},{y1} ({cw}x{ch}) imgsz={imgsz}…')

        poly_f = _predict(self.model, crop, 0.25, imgsz)
        if poly_f is None:
            print('  NO DETECTION')
            self.result_poly = None
        else:
            # Shift back to full-image coords
            poly = poly_f.copy()
            poly[:, 0] += x0
            poly[:, 1] += y0
            self.result_poly = poly
            print(f'  {len(poly)} points detected')
        self.draw()

    # ------------------------------------------------------------------
    def on_mouse(self, event, dx, dy, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (dx, dy)
            self.drag_end   = (dx, dy)
            self.result_poly = None
            self.draw()
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (dx, dy)
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_end = (dx, dy)
            self.draw()

    # ------------------------------------------------------------------
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DISP_W, DISP_H + 36)
        cv2.setMouseCallback(WIN, self.on_mouse)

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('n'):
                self.load_image(self.idx + 1)
            elif k == ord('p'):
                self.load_image(self.idx - 1)
            elif k in (13, 32):   # Enter / Space
                self.run_model()
            elif k == ord('r'):
                self.drag_start = self.drag_end = None
                self.result_poly = None
                self.draw()

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------

def main():
    print('Loading model…')
    model = YOLO(MODEL_PATH)
    app = App(model)
    app.run()


if __name__ == '__main__':
    main()
