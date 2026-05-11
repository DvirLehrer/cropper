#!/usr/bin/env python3

"""
Use the trained CNN classifier to label character bounding boxes (proposals) and
write annotated images.

Important: the CNN does NOT "find" boxes by itself (it's a classifier). We use
the character boxes produced by `google_ocr.py` (Google Vision) as proposals,
then classify each crop and draw it back on the original image.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch

from train_cnn import build_model, pil_to_tensor_gray


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points as: top-left, top-right, bottom-right, bottom-left.
    pts: (4,2)
    """
    pts = np.asarray(pts, dtype=np.float32)
    pts_sorted = pts[np.argsort(pts[:, 1])]
    top_points = pts_sorted[:2]
    bottom_points = pts_sorted[2:]
    top_points = top_points[np.argsort(top_points[:, 0])]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    rect = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.float32)
    return rect


def crop_quad_with_padding(img_cv: np.ndarray, vertices: List[Dict[str, int]], padding_frac: float) -> Image.Image:
    pts = np.array([[v["x"], v["y"]] for v in vertices[:4]], dtype=np.float32)
    rect = order_points(pts)

    w_top = np.linalg.norm(rect[1] - rect[0])
    w_bottom = np.linalg.norm(rect[2] - rect[3])
    h_left = np.linalg.norm(rect[3] - rect[0])
    h_right = np.linalg.norm(rect[2] - rect[1])
    out_w = int(max(w_top, w_bottom))
    out_h = int(max(h_left, h_right))
    if out_w < 2 or out_h < 2:
        raise ValueError("quad too small")

    max_dim = max(out_w, out_h)
    pad_px = int(round(max_dim * float(padding_frac)))
    if pad_px < 1:
        pad_px = 1
    scale = (max_dim + 2.0 * pad_px) / float(max_dim)
    center = rect.mean(axis=0, keepdims=True)
    rect_expanded = center + (rect - center) * scale

    w_top2 = np.linalg.norm(rect_expanded[1] - rect_expanded[0])
    w_bottom2 = np.linalg.norm(rect_expanded[2] - rect_expanded[3])
    h_left2 = np.linalg.norm(rect_expanded[3] - rect_expanded[0])
    h_right2 = np.linalg.norm(rect_expanded[2] - rect_expanded[1])
    out_w2 = int(max(w_top2, w_bottom2))
    out_h2 = int(max(h_left2, h_right2))
    out_w2 = max(2, out_w2)
    out_h2 = max(2, out_h2)

    dst = np.array([[0, 0], [out_w2 - 1, 0], [out_w2 - 1, out_h2 - 1], [0, out_h2 - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect_expanded.astype(np.float32), dst)
    patch = cv2.warpPerspective(img_cv, M, (out_w2, out_h2))

    # cv2 -> PIL (RGB)
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return Image.fromarray(patch_rgb)


def load_meta(model_dir: str) -> Tuple[int, Dict[int, str], int]:
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()} if isinstance(meta.get("idx_to_class"), dict) else meta["idx_to_class"]
    num_classes = len(meta["class_to_idx"])
    image_size = int(meta.get("image_size", 32))
    return num_classes, idx_to_class, image_size


def load_model(model_dir: str, device: torch.device):
    num_classes, idx_to_class, image_size = load_meta(model_dir)
    model = build_model(num_classes=num_classes).to(device)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, idx_to_class, image_size


def iter_test_images(test_dir: str) -> List[str]:
    out = []
    for fn in os.listdir(test_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(os.path.join(test_dir, fn))
    out.sort()
    return out


def main():
    parser = argparse.ArgumentParser(description="Infer CNN labels for OCR character boxes and draw overlays.")
    parser.add_argument("--test-dir", default="images/_test_set", help="Directory of input images.")
    parser.add_argument("--char-boxes-dir", default="test_output", help="Where *_char_boxes.json files are located.")
    parser.add_argument("--model-dir", default="test_output/cnn_model", help="Directory with model.pt + meta.json.")
    parser.add_argument("--out-dir", default="test_output/cnn_boxes", help="Where to write annotated images.")
    parser.add_argument("--padding-frac", type=float, default=0.25, help="Padding frac used when cropping boxes (match export).")
    parser.add_argument("--conf-threshold", type=float, default=0.20, help="Only draw boxes with prob >= threshold.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class, image_size = load_model(args.model_dir, device)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()

    images = iter_test_images(args.test_dir)
    if not images:
        raise SystemExit(f"No images found under: {args.test_dir}")

    for image_path in images:
        base = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(args.char_boxes_dir, f"{base}_char_boxes.json")
        if not os.path.exists(json_path):
            print(f"SKIP (missing boxes): {image_path} -> expected {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        boxes: List[Dict[str, Any]] = data.get("boxes", [])
        if not boxes:
            print(f"SKIP (no boxes): {image_path}")
            continue

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"SKIP (cannot read): {image_path}")
            continue

        img_pil = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)

        drawn = 0
        for b in boxes:
            verts = b.get("vertices") or []
            if len(verts) < 4:
                continue
            try:
                patch_pil = crop_quad_with_padding(img_cv, verts, padding_frac=float(args.padding_frac))
            except Exception:
                continue

            x = pil_to_tensor_gray(patch_pil, size=image_size).unsqueeze(0).to(device)  # (1,1,H,W)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
                conf_f = float(conf.item())
                pred_i = int(pred_idx.item())

            if conf_f < float(args.conf_threshold):
                continue

            label = idx_to_class.get(pred_i, str(pred_i))

            pts = [(int(v["x"]), int(v["y"])) for v in verts[:4]]
            draw.polygon(pts, outline="red", width=2)
            # label at first vertex
            txt = f"{label} {conf_f:.2f}"
            draw.text((pts[0][0] + 2, pts[0][1] + 2), txt, fill="yellow", font=font)
            drawn += 1

        out_path = os.path.join(args.out_dir, f"{base}_cnn_boxes.png")
        img_pil.save(out_path)
        print(f"OK: {base} boxes={len(boxes)} drawn={drawn} -> {out_path}")


if __name__ == "__main__":
    main()


