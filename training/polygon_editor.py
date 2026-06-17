#!/usr/bin/env python3
"""
Interactive polygon editor for YOLO-seg annotations.

Layout  : image area on top, info panel below (no overlay on the image).
Zoom    : scroll wheel (centered on cursor) or +/- keys; 0 to fit.
Pan     : left-drag on empty space (no vertex or edge under cursor).
Vertices: left-drag to move, left-click edge to insert, right-click to delete.

s / Enter  save annotation to label file
r          reset polygon to last saved state
n / Space  next image (auto-saves if modified)
p          previous image (auto-saves if modified)
q / Esc    quit

Usage:
  python polygon_editor.py
  python polygon_editor.py --split train
  python polygon_editor.py --start 42
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

WIN     = 'Polygon Editor'
DISP_W  = 1100   # image-area width  (px)
DISP_H  = 800    # image-area height (px)
PANEL_H = 60     # info panel height below image (px)
VRAD    = 8      # vertex dot radius (display px)
PICK_R  = 14     # vertex click-detection radius (display px)
EDGE_R  = 10     # edge   click-detection radius (display px)


# ---------------------------------------------------------------------------
# OCR JSON lookup (Hebrew char-box overlay, optional reference)
# ---------------------------------------------------------------------------

def _find_ocr_json(stem):
    direct = Path('test_output') / f'{stem}_char_boxes.json'
    if direct.exists():
        return direct
    # Roboflow mangled names: "01-05-22_001_jpg.rf.HASH" → "01.05.22_001"
    if '.rf.' in stem:
        base = re.sub(r'_jpg$', '', stem.split('.rf.')[0]).replace('-', '.')
        p = Path('test_output') / f'{base}_char_boxes.json'
        if p.exists():
            return p
    return None


def load_ocr_boxes(stem):
    p = _find_ocr_json(stem)
    if p is None:
        return []
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    return [b for b in data.get('boxes', [])
            if b.get('char', '') and 'א' <= b['char'][0] <= 'ת']


# ---------------------------------------------------------------------------
# YOLO-seg label I/O
# ---------------------------------------------------------------------------

def load_label(lbl_path, W, H):
    """Return (N,2) float32 in image px, or None."""
    if not lbl_path.exists():
        return None
    lines = lbl_path.read_text().strip().splitlines()
    if not lines:
        return None
    parts = lines[0].split()
    coords = list(map(float, parts[1:]))
    pts = [[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords) - 1, 2)]
    return np.array(pts, dtype=np.float32)


def save_label(lbl_path, poly, W, H):
    coords = ' '.join(
        f'{max(0.0, min(1.0, x / W)):.6f} {max(0.0, min(1.0, y / H)):.6f}'
        for x, y in poly
    )
    lbl_path.write_text(f'0 {coords}\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------

class Editor:
    def __init__(self, pairs):
        self.pairs      = pairs
        self.idx        = 0
        self.img        = None
        self.H = self.W = 0
        self.base_scale = 1.0
        self.zoom       = 1.0   # multiplier on top of base_scale
        self.pan_x      = 0.0   # image-coord at display (0, 0)
        self.pan_y      = 0.0
        self.poly       = np.empty((0, 2), dtype=np.float32)
        self.poly_saved = np.empty((0, 2), dtype=np.float32)
        self.ocr_boxes  = []
        self.modified   = False
        self.sel        = -1
        self.hover_v    = -1
        self.hover_e    = -1
        self.dragging   = False
        self.drag_mode  = None   # 'vertex' | 'pan'
        self.drag_last  = (0, 0)

    # --- coordinate helpers -------------------------------------------------

    def _ez(self):
        return self.zoom * self.base_scale

    def to_img(self, dx, dy):
        ez = self._ez()
        return np.array([dx / ez + self.pan_x, dy / ez + self.pan_y], dtype=np.float32)

    def to_disp(self, pt):
        ez = self._ez()
        return (int((pt[0] - self.pan_x) * ez), int((pt[1] - self.pan_y) * ez))

    def _init_view(self):
        self.zoom  = 1.0
        ez = self.base_scale
        self.pan_x = -(DISP_W - self.W * ez) / (2 * ez)
        self.pan_y = -(DISP_H - self.H * ez) / (2 * ez)

    def _apply_zoom(self, factor, dx, dy):
        ez_old = self._ez()
        ix = dx / ez_old + self.pan_x
        iy = dy / ez_old + self.pan_y
        self.zoom  = max(0.25, min(20.0, self.zoom * factor))
        ez_new = self._ez()
        self.pan_x = ix - dx / ez_new
        self.pan_y = iy - dy / ez_new

    # --- load / save --------------------------------------------------------

    def load(self):
        img_path, lbl_path = self.pairs[self.idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is None:
            print(f'Cannot read {img_path}')
            return False
        self.img        = img
        self.H, self.W  = img.shape[:2]
        self.base_scale = min(DISP_W / self.W, DISP_H / self.H)
        self.ocr_boxes  = load_ocr_boxes(Path(img_path).stem)
        self._init_view()

        poly = load_label(lbl_path, self.W, self.H)
        self.poly       = poly if poly is not None else np.empty((0, 2), dtype=np.float32)
        self.poly_saved = self.poly.copy()
        self.modified   = False
        self.sel = self.hover_v = self.hover_e = -1
        return True

    def save(self):
        _, lbl_path = self.pairs[self.idx]
        if len(self.poly) < 3:
            print('Need ≥3 vertices to save.')
            return
        save_label(lbl_path, self.poly, self.W, self.H)
        self.poly_saved = self.poly.copy()
        self.modified   = False
        print(f'Saved  ({len(self.poly)} pts)  →  {lbl_path}')

    def reset(self):
        self.poly     = self.poly_saved.copy()
        self.modified = False
        self.sel      = -1

    # --- geometry helpers ---------------------------------------------------

    def _nearest_vertex(self, dpt):
        n = len(self.poly)
        if n == 0:
            return -1
        dpts  = np.array([self.to_disp(p) for p in self.poly])
        dists = np.linalg.norm(dpts - dpt, axis=1)
        i = int(np.argmin(dists))
        return i if dists[i] <= PICK_R else -1

    def _nearest_edge(self, dpt):
        n = len(self.poly)
        if n < 2:
            return -1, 0.0
        best_d, best_i, best_t = float(EDGE_R), -1, 0.0
        dpt_f = np.array(dpt, dtype=float)
        for i in range(n):
            p1 = np.array(self.to_disp(self.poly[i]),           dtype=float)
            p2 = np.array(self.to_disp(self.poly[(i + 1) % n]), dtype=float)
            d  = p2 - p1
            ls = float(np.dot(d, d))
            if ls < 1e-6:
                continue
            t    = float(np.clip(np.dot(dpt_f - p1, d) / ls, 0.0, 1.0))
            dist = float(np.linalg.norm(p1 + t * d - dpt_f))
            if dist < best_d:
                best_d, best_i, best_t = dist, i, t
        return best_i, best_t

    def _insert_on_edge(self, ei, t):
        p1, p2 = self.poly[ei], self.poly[(ei + 1) % len(self.poly)]
        new_pt = (p1 + t * (p2 - p1)).reshape(1, 2)
        self.poly     = np.insert(self.poly, ei + 1, new_pt, axis=0)
        self.modified = True

    # --- draw ---------------------------------------------------------------

    def draw(self):
        canvas = np.full((DISP_H + PANEL_H, DISP_W, 3), 55, dtype=np.uint8)

        # --- render image region ---
        ez  = self._ez()
        x0d = int(-self.pan_x * ez)           # display x where image left edge lands
        y0d = int(-self.pan_y * ez)           # display y where image top  edge lands
        x1d = int((self.W - self.pan_x) * ez)
        y1d = int((self.H - self.pan_y) * ez)

        dst_x0, dst_x1 = max(0, x0d), min(DISP_W, x1d)
        dst_y0, dst_y1 = max(0, y0d), min(DISP_H, y1d)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            s_x0 = int(max(0.0,        dst_x0 / ez + self.pan_x))
            s_y0 = int(max(0.0,        dst_y0 / ez + self.pan_y))
            s_x1 = min(self.W, int(dst_x1 / ez + self.pan_x) + 1)
            s_y1 = min(self.H, int(dst_y1 / ez + self.pan_y) + 1)
            crop = self.img[s_y0:s_y1, s_x0:s_x1]
            if crop.size:
                scaled = cv2.resize(crop, (dst_x1 - dst_x0, dst_y1 - dst_y0))
                canvas[dst_y0:dst_y1, dst_x0:dst_x1] = scaled

        # --- char-box overlay (thin blue, reference only) ---
        for b in self.ocr_boxes:
            verts = b.get('vertices', [])
            if len(verts) < 4:
                continue
            pts = np.array([self.to_disp((v['x'], v['y'])) for v in verts], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (200, 100, 0), 1)

        # --- polygon edges ---
        n = len(self.poly)
        if n >= 2:
            for i in range(n):
                p1 = self.to_disp(self.poly[i])
                p2 = self.to_disp(self.poly[(i + 1) % n])
                col = (0, 180, 255) if i == self.hover_e else (0, 220, 50)
                cv2.line(canvas, p1, p2, col, 2)

        # --- vertices ---
        for i in range(n):
            dp = self.to_disp(self.poly[i])
            if i == self.sel:
                col, r = (0, 0, 255),   VRAD + 4   # red: active
            elif i == self.hover_v:
                col, r = (0, 255, 255), VRAD + 2   # yellow: hover
            else:
                col, r = (0, 220, 50),  VRAD        # green: normal
            cv2.circle(canvas, dp, r, col, -1)
            cv2.circle(canvas, dp, r, (0, 0, 0), 1)
            cv2.putText(canvas, str(i), (dp[0] + r + 2, dp[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # --- info panel (below image, no overlap) ---
        img_name = Path(self.pairs[self.idx][0]).name
        zoom_pct = int(self.zoom * 100)
        status = (f'{self.idx + 1}/{len(self.pairs)}   {img_name}   '
                  f'{n} pts   zoom {zoom_pct}%'
                  + ('   * MODIFIED' if self.modified else ''))
        hint = ('L-drag: move pt    L-click edge: add pt    L-click + Del: delete pt    '
                'drag empty: pan    scroll / +- : zoom    0: fit    '
                's: save    r: reset    n/Space: next    p: prev    q: quit')
        y_base = DISP_H
        cv2.line(canvas, (0, y_base), (DISP_W, y_base), (80, 80, 80), 1)
        cv2.putText(canvas, status, (10, y_base + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)
        cv2.putText(canvas, hint,   (10, y_base + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

        cv2.imshow(WIN, canvas)

    # --- mouse callback -----------------------------------------------------

    def mouse_cb(self, event, dx, dy, flags, _):
        # Ignore clicks inside the panel area
        if dy >= DISP_H and event != cv2.EVENT_MOUSEWHEEL:
            return

        dpt    = (dx, dy)
        img_pt = self.to_img(dx, dy)

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = (1.0 / 1.12) if flags > 0 else 1.12
            self._apply_zoom(factor, dx, dy)
            self.draw()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                if self.drag_mode == 'vertex' and self.sel >= 0:
                    self.poly[self.sel] = img_pt
                    self.modified = True
                elif self.drag_mode == 'pan':
                    ez = self._ez()
                    self.pan_x -= (dx - self.drag_last[0]) / ez
                    self.pan_y -= (dy - self.drag_last[1]) / ez
                    self.drag_last = (dx, dy)
            else:
                vi = self._nearest_vertex(dpt)
                ei, _ = self._nearest_edge(dpt)
                self.hover_v = vi
                self.hover_e = -1 if vi >= 0 else ei
            self.draw()

        elif event == cv2.EVENT_LBUTTONDOWN:
            vi = self._nearest_vertex(dpt)
            if vi >= 0:
                self.sel, self.dragging, self.drag_mode = vi, True, 'vertex'
            else:
                ei, t = self._nearest_edge(dpt)
                if ei >= 0:
                    self._insert_on_edge(ei, t)
                    self.sel, self.dragging, self.drag_mode = ei + 1, True, 'vertex'
                else:
                    self.dragging, self.drag_mode = True, 'pan'
                    self.drag_last = (dx, dy)
            self.draw()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging  = False
            self.drag_mode = None


    # --- main loop ----------------------------------------------------------

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.mouse_cb)
        self.load()
        self.draw()

        while True:
            key = cv2.waitKey(20)
            if key < 0:
                continue
            k = key & 0xFF

            if k in (ord('q'), 27):
                if self.modified:
                    print('Unsaved changes — press q again to discard, or s to save.')
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 not in (ord('q'), 27):
                        continue
                break
            elif k in (ord('s'), 13):
                self.save();  self.draw()
            elif k == ord('r'):
                self.reset(); self.draw()
            elif k in (ord('n'), ord(' ')):
                if self.modified: self.save()
                self.idx = (self.idx + 1) % len(self.pairs)
                self.load(); self.draw()
            elif k == ord('p'):
                if self.modified: self.save()
                self.idx = (self.idx - 1) % len(self.pairs)
                self.load(); self.draw()
            elif k in (ord('+'), ord('=')):
                self._apply_zoom(1.2, DISP_W // 2, DISP_H // 2); self.draw()
            elif k == ord('-'):
                self._apply_zoom(1 / 1.2, DISP_W // 2, DISP_H // 2); self.draw()
            elif k == ord('0'):
                self._init_view(); self.draw()
            elif k in (8, 127, 255):   # Backspace / Delete / Fn+Delete
                if self.sel >= 0 and len(self.poly) > 3:
                    self.poly     = np.delete(self.poly, self.sel, axis=0)
                    self.modified = True
                    self.sel      = -1
                    self.draw()

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_pairs(yolo_dir, split):
    splits = [split] if split else ['train', 'val', 'test']
    pairs  = []
    for sp in splits:
        img_dir = yolo_dir / 'images' / sp
        lbl_dir = yolo_dir / 'labels' / sp
        if not img_dir.exists():
            continue
        for ip in sorted(img_dir.glob('*')):
            if ip.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                pairs.append((ip, lbl_dir / (ip.stem + '.txt')))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo-dir', default='yolo_polygon')
    ap.add_argument('--split',    choices=['train', 'val', 'test'])
    ap.add_argument('--start',    type=int, default=0)
    args = ap.parse_args()

    pairs = collect_pairs(Path(args.yolo_dir), args.split)
    if not pairs:
        print(f'No images found in {args.yolo_dir}')
        sys.exit(1)

    editor     = Editor(pairs)
    editor.idx = max(0, min(args.start, len(pairs) - 1))
    print(f'Loaded {len(pairs)} images.  s=save  n/p=navigate  q=quit')
    editor.run()


if __name__ == '__main__':
    main()
