#! /usr/bin/env python3
import os
import json
import unicodedata
import numpy as np
import cv2


def _class_dirname(ch: str) -> str:
    if ch is None:
        ch = ""
    ch = str(ch)
    ch_clean = ch.replace(os.sep, "_").replace("/", "_").strip()
    if not ch_clean or unicodedata.category(ch_clean[0]).startswith("C"):
        return "UNKNOWN"
    cp = ord(ch_clean[0])
    return f"U{cp:04X}_{ch_clean[0]}"


def _order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    pts_sorted = pts[np.argsort(pts[:, 1])]
    top_points = pts_sorted[:2]
    bottom_points = pts_sorted[2:]
    top_points = top_points[np.argsort(top_points[:, 0])]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    return np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.float32)


def _crop_char_patch(img_cv, b: dict, min_size: int = 6, padding_frac: float = 0.20):
    """Return a perspective-corrected crop for one character box, or None if too small."""
    verts = b.get("vertices") or []
    if len(verts) < 4:
        return None
    pts = np.array([[v["x"], v["y"]] for v in verts[:4]], dtype=np.float32)
    rect = _order_points(pts)
    out_w = int(max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3])))
    out_h = int(max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1])))
    if out_w < min_size or out_h < min_size:
        return None
    pad_px = max(1, int(round(max(out_w, out_h) * float(padding_frac))))
    scale = (max(out_w, out_h) + 2.0 * pad_px) / float(max(out_w, out_h))
    center = rect.mean(axis=0, keepdims=True)
    rect_exp = center + (rect - center) * scale
    out_w2 = int(max(np.linalg.norm(rect_exp[1] - rect_exp[0]), np.linalg.norm(rect_exp[2] - rect_exp[3])))
    out_h2 = int(max(np.linalg.norm(rect_exp[3] - rect_exp[0]), np.linalg.norm(rect_exp[2] - rect_exp[1])))
    if out_w2 < min_size or out_h2 < min_size:
        return None
    dst = np.array([[0, 0], [out_w2 - 1, 0], [out_w2 - 1, out_h2 - 1], [0, out_h2 - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect_exp, dst)
    return cv2.warpPerspective(img_cv, M, (out_w2, out_h2))


def _save_crop(patch, b: dict, image_base: str, output_root: str):
    ch = b.get("char", "")
    cls_dir = os.path.join(output_root, _class_dirname(ch))
    os.makedirs(cls_dir, exist_ok=True)
    fname = (
        f"{image_base}_{b.get('i', 0):06d}_p{b.get('page', 0)}_b{b.get('block', 0)}"
        f"_r{b.get('paragraph', 0)}_w{b.get('word', 0)}_s{b.get('symbol', 0)}.png"
    )
    cv2.imwrite(os.path.join(cls_dir, fname), patch)


def export_dataset_from_json(
    json_path: str,
    output_root: str,
    min_size: int = 6,
    padding_frac: float = 0.20,
    min_confidence: float = 0.80,
) -> int:
    """Export character crops from an existing _char_boxes.json without re-running OCR."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    image_path = data.get("image", "")
    img_cv = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img_cv is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image_base = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_root, exist_ok=True)
    written = skipped_conf = 0
    for b in data.get("boxes", []):
        if b.get("confidence", 0.0) < min_confidence:
            skipped_conf += 1
            continue
        patch = _crop_char_patch(img_cv, b, min_size, padding_frac)
        if patch is None:
            continue
        _save_crop(patch, b, image_base, output_root)
        written += 1
    print(f"  {os.path.basename(json_path)}: exported {written}, skipped {skipped_conf} (conf < {min_confidence:.4f})")
    return written


def export_dataset_topup(
    json_paths: list,
    output_root: str,
    min_size: int = 6,
    padding_frac: float = 0.20,
    min_confidence: float = 0.9937,
    min_per_class: int = 20,
) -> dict:
    """
    Export the top-confidence dataset with per-class top-up.

    First exports all chars with confidence >= min_confidence (the primary set).
    Then, for any character class that still has fewer than min_per_class examples,
    fills up to min_per_class by adding the next-highest-confidence chars for that class.
    """
    from collections import defaultdict

    # Load all boxes from all JSONs, tagged with their source image
    image_cache = {}
    all_records = []
    for jp in json_paths:
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        image_path = data.get("image", "")
        if image_path not in image_cache:
            img = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            image_cache[image_path] = img
        img_cv = image_cache[image_path]
        if img_cv is None:
            print(f"  WARNING: cannot load {image_path}, skipping {os.path.basename(jp)}")
            continue
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        for b in data.get("boxes", []):
            if "confidence" not in b:
                continue
            all_records.append((b, img_cv, image_base))

    # Group by character, sort each group by confidence descending
    by_char = defaultdict(list)
    for rec in all_records:
        by_char[rec[0].get("char", "")].append(rec)
    for ch in by_char:
        by_char[ch].sort(key=lambda r: r[0]["confidence"], reverse=True)

    os.makedirs(output_root, exist_ok=True)
    stats = {}
    total_high = total_topup = 0

    for ch, recs in sorted(by_char.items()):
        high = [(b, img, base) for b, img, base in recs if b["confidence"] >= min_confidence]
        low  = [(b, img, base) for b, img, base in recs if b["confidence"] <  min_confidence]
        needed = max(0, min_per_class - len(high))
        to_export = high + low[:needed]

        written = topup = 0
        for b, img_cv, image_base in to_export:
            patch = _crop_char_patch(img_cv, b, min_size, padding_frac)
            if patch is None:
                continue
            _save_crop(patch, b, image_base, output_root)
            written += 1
            if b["confidence"] < min_confidence:
                topup += 1

        stats[ch] = {"high_conf": len(high), "topup": topup, "exported": written, "available": len(recs)}
        total_high += len(high)
        total_topup += topup

    print(f"\nClasses: {len(stats)}  |  high-conf crops: {total_high}  |  top-up crops: {total_topup}  "
          f"|  total: {total_high + total_topup}")
    return stats


def detect_text(
    path,
    output_dir: str = "test_output",
    max_pixels: int = 1_000_000,
):
    """Run Google Vision document_text_detection and save character-level boxes to JSON.

    Output: test_output/<base>_char_boxes.json
    """
    from google.cloud import vision

    output_dir = output_dir or "test_output"
    os.makedirs(output_dir, exist_ok=True)

    name, ext = os.path.splitext(os.path.basename(path))

    # Downscale large images before sending to Vision; scale vertices back afterward.
    scale = 1.0
    raw = cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR) if max_pixels else None
    if raw is not None and raw.shape[0] * raw.shape[1] > max_pixels:
        h, w = raw.shape[:2]
        scale = (max_pixels / (h * w)) ** 0.5
        small = cv2.resize(raw, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(ext or ".jpg", small)
        if not ok:
            raise RuntimeError(f"failed to encode downscaled image for {path}")
        content = buf.tobytes()
        print(f"Downscaled {w}x{h} -> {small.shape[1]}x{small.shape[0]} for OCR")
    else:
        with open(path, "rb") as f:
            content = f.read()
    inv_scale = 1.0 / scale

    def _norm_vertex(v):
        x = 0 if v.x is None else int(round(v.x * inv_scale))
        y = 0 if v.y is None else int(round(v.y * inv_scale))
        return {"x": x, "y": y}

    client   = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=vision.Image(content=content))

    if response.error.message:
        raise Exception(response.error.message)

    fta = response.full_text_annotation

    char_boxes = []
    hebrew_count = 0
    if fta and fta.pages:
        symbol_index = 0
        for page_idx, page in enumerate(fta.pages):
            for block_idx, block in enumerate(page.blocks):
                for para_idx, para in enumerate(block.paragraphs):
                    for word_idx, word in enumerate(para.words):
                        for sym_idx, sym in enumerate(word.symbols):
                            ch = sym.text or ""
                            if not ch.strip():
                                symbol_index += 1
                                continue
                            verts = sym.bounding_box.vertices if sym.bounding_box else []
                            if not verts:
                                continue
                            if "\u0590" <= ch[0] <= "\u05FF":
                                hebrew_count += 1
                            char_boxes.append({
                                "i":          symbol_index,
                                "char":       ch,
                                "vertices":   [_norm_vertex(v) for v in verts],
                                "confidence": round(float(sym.confidence or 0.0), 4),
                                "page":       page_idx,
                                "block":      block_idx,
                                "paragraph":  para_idx,
                                "word":       word_idx,
                                "symbol":     sym_idx,
                            })
                            symbol_index += 1

    print(f"  {hebrew_count} Hebrew chars, {len(char_boxes)} total symbols")

    out_path = os.path.join(output_dir, f"{name}_char_boxes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"image": os.path.abspath(path), "text": (fta.text if fta else ""), "boxes": char_boxes},
            f, ensure_ascii=False, indent=2,
        )
    return out_path

if __name__ == "__main__":
    import glob
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Google Vision OCR helper for STaM (character-level boxes).")
    parser.add_argument("--image", help="Path to a single image to process.")
    parser.add_argument("--test-set", action="store_true", help="Process all images in images/_test_set/")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Process all images in this directory (non-recursive). Overrides --test-set when provided.",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output",
        help="Where to write outputs (e.g. *_char_boxes.json, overlays).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images if <output-dir>/<base>_char_boxes.json already exists.",
    )
    parser.add_argument("--max-pixels", type=int, default=1_000_000,
                        help="Downscale images so total pixels <= this before OCR (0 = no downscale).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many images (for testing).")
    parser.add_argument(
        "--reexport-only",
        action="store_true",
        help="Skip OCR; re-export the dataset from existing *_char_boxes.json files in --output-dir.",
    )
    parser.add_argument(
        "--top-percentile",
        type=float,
        default=None,
        metavar="N",
        help="Keep only the top N%% of characters by confidence rank (e.g. 20 = top 20%%). "
             "Overrides --min-confidence when set.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=20,
        help="When using --reexport-only, top-up classes below this count with the next-best "
             "confidence examples for that character (default: 20).",
    )
    args = parser.parse_args()

    out_dir = (args.output_dir or "test_output").strip()

    # --- Re-export mode: read existing JSONs, no API calls ---
    if args.reexport_only:
        if not dataset_root:
            print("--reexport-only requires --dataset-out to be set.")
            raise SystemExit(1)
        json_files = sorted(glob.glob(os.path.join(out_dir, "*_char_boxes.json")))
        if not json_files:
            print(f"No *_char_boxes.json files found in {out_dir}")
            raise SystemExit(1)

        min_conf = args.min_confidence

        if args.top_percentile is not None:
            # Compute threshold dynamically: keep the top N% by confidence rank
            all_conf = []
            for jp in json_files:
                with open(jp, encoding="utf-8") as f:
                    d = json.load(f)
                for b in d.get("boxes", []):
                    if "confidence" in b:
                        all_conf.append(b["confidence"])
            if not all_conf:
                print("No confidence data found in any JSON — re-run OCR first.")
                raise SystemExit(1)
            threshold_pct = 100.0 - args.top_percentile
            min_conf = float(np.percentile(all_conf, threshold_pct))
            print(f"Top {args.top_percentile:.0f}% threshold (p{threshold_pct:.0f} of {len(all_conf)} chars): {min_conf:.4f}")

        print(f"Re-exporting dataset from {len(json_files)} JSON file(s) "
              f"(min confidence {min_conf:.4f}, top-up to {args.min_per_class}/class) -> {dataset_root}\n")
        stats = export_dataset_topup(
            json_files, dataset_root,
            min_size=args.min_size,
            padding_frac=args.padding_frac,
            min_confidence=min_conf,
            min_per_class=args.min_per_class,
        )
        print("\nPer-class breakdown:")
        for ch, s in sorted(stats.items()):
            note = f"  [+{s['topup']} top-up]" if s["topup"] else ""
            print(f"  {ch}: {s['exported']} exported (high-conf={s['high_conf']}, avail={s['available']}){note}")
        raise SystemExit(0)

    # --- Normal OCR mode ---
    if not args.image and not args.test_set and not args.images_dir:
        args.test_set = True

    if args.image:
        image_files = [args.image]
    elif args.images_dir:
        image_files = []
        for fn in os.listdir(args.images_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(args.images_dir, fn))
        image_files.sort()
    else:
        image_patterns = [
            "images/_test_set/*.jpg",
            "images/_test_set/*.jpeg",
            "images/_test_set/*.png",
        ]
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(pattern))

    import time
    # When --limit is combined with --skip-existing, filter pending first then cap
    if args.limit and args.skip_existing:
        image_files = [p for p in image_files
                       if not os.path.exists(os.path.join(out_dir,
                           os.path.splitext(os.path.basename(p))[0] + "_char_boxes.json"))]
        image_files = image_files[:args.limit]
    elif args.limit:
        image_files = image_files[:args.limit]
    print(f"Found {len(image_files)} images to process\n")

    done = skipped = errors = 0
    times = []
    batch_start = time.time()

    for i, image_path in enumerate(image_files, 1):
        pct = 100.0 * i / len(image_files)
        print(f"[{i}/{len(image_files)}  {pct:.1f}%]  {os.path.basename(image_path)}")

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        if args.skip_existing:
            expected_json = os.path.join(out_dir, f"{image_id}_char_boxes.json")
            if os.path.exists(expected_json):
                print(f"  ↷ skip (exists)")
                skipped += 1
                continue

        t0 = time.time()
        try:
            output_path = detect_text(image_path, output_dir=out_dir, max_pixels=int(args.max_pixels))
            elapsed = time.time() - t0
            times.append(elapsed)
            avg = sum(times) / len(times)
            remaining = avg * (len(image_files) - i)
            print(f"  ✓  {elapsed:.1f}s  |  avg {avg:.1f}s  |  ~{remaining/60:.1f} min left")
            done += 1
        except Exception as e:
            print(f"  ✗  {e}")
            errors += 1

    total = time.time() - batch_start
    print(f"\nDone: {done} processed, {skipped} skipped, {errors} errors  |  total {total/60:.1f} min")
