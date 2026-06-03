#! /usr/bin/env python3

def detect_text(
    path,
    dataset_root=None,
    dataset_image_id=None,
    output_dir: str = "test_output",
    padding_frac: float = 0.20,
    min_size: int = 6,
    max_pixels: int = 1_000_000,
):
    """Detects text in the file."""
    from google.cloud import vision
    from PIL import Image
    import os
    import re
    import cv2
    import numpy as np
    import json
    import unicodedata

    def contains_hebrew(text):
        """Check if text contains Hebrew characters."""
        # Hebrew Unicode range: U+0590 to U+05FF
        return bool(re.search(r'[\u0590-\u05FF]', text))

    def order_points(pts):
        """
        Order points as: top-left, top-right, bottom-right, bottom-left.
        Works for (4,2) arrays.
        """
        pts = np.asarray(pts, dtype=np.float32)
        pts_sorted = pts[np.argsort(pts[:, 1])]
        top_points = pts_sorted[:2]
        bottom_points = pts_sorted[2:]
        top_points = top_points[np.argsort(top_points[:, 0])]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
        rect = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.float32)
        return rect

    def class_dirname(ch: str) -> str:
        """
        Create a filesystem-safe class directory name for a character.
        Keeps the character (useful for humans) but prefixes codepoint for stability.
        """
        if ch is None:
            ch = ""
        ch = str(ch)
        # Disallow path separators / weird whitespace
        ch_clean = ch.replace(os.sep, "_").replace("/", "_").strip()
        if not ch_clean or unicodedata.category(ch_clean[0]).startswith("C"):
            return "UNKNOWN"
        cp = ord(ch_clean[0])
        # Prefix with codepoint to avoid collisions (e.g. lookalikes) and keep stable ordering.
        return f"U{cp:04X}_{ch_clean[0]}"

    def export_char_dataset(
        img_cv,
        char_boxes,
        output_root: str,
        min_size: int = 6,
        padding_frac: float = 0.20,
        image_id: str = "img",
    ):
        """
        Save one image per character under output_root/<class>/....
        Each crop is rectified from the 4-point quad using a local perspective transform.
        """
        os.makedirs(output_root, exist_ok=True)
        written = 0

        for b in char_boxes:
            ch = b.get("char", "")
            verts = b.get("vertices") or []
            if len(verts) < 4:
                continue

            pts = np.array([[v["x"], v["y"]] for v in verts[:4]], dtype=np.float32)
            rect = order_points(pts)

            # Compute output patch size from rect edges
            w_top = np.linalg.norm(rect[1] - rect[0])
            w_bottom = np.linalg.norm(rect[2] - rect[3])
            h_left = np.linalg.norm(rect[3] - rect[0])
            h_right = np.linalg.norm(rect[2] - rect[1])
            out_w = int(max(w_top, w_bottom))
            out_h = int(max(h_left, h_right))

            if out_w < min_size or out_h < min_size:
                continue

            # Expand the source quad outward so the character isn't cut off.
            # We do this by scaling the quad around its center.
            max_dim = max(out_w, out_h)
            pad_px = int(round(max_dim * float(padding_frac)))
            if pad_px < 1:
                pad_px = 1
            scale = (max_dim + 2.0 * pad_px) / float(max_dim)
            center = rect.mean(axis=0, keepdims=True)
            rect_expanded = center + (rect - center) * scale

            # Recompute output size from expanded quad so we keep aspect reasonable.
            w_top2 = np.linalg.norm(rect_expanded[1] - rect_expanded[0])
            w_bottom2 = np.linalg.norm(rect_expanded[2] - rect_expanded[3])
            h_left2 = np.linalg.norm(rect_expanded[3] - rect_expanded[0])
            h_right2 = np.linalg.norm(rect_expanded[2] - rect_expanded[1])
            out_w2 = int(max(w_top2, w_bottom2))
            out_h2 = int(max(h_left2, h_right2))
            if out_w2 < min_size or out_h2 < min_size:
                continue

            dst = np.array([[0, 0], [out_w2 - 1, 0], [out_w2 - 1, out_h2 - 1], [0, out_h2 - 1]], dtype=np.float32)
            Mch = cv2.getPerspectiveTransform(rect_expanded.astype(np.float32), dst)
            patch = cv2.warpPerspective(img_cv, Mch, (out_w2, out_h2))

            # Save as PNG for dataset use (avoid JPEG artifacts)
            cls = class_dirname(ch)
            cls_dir = os.path.join(output_root, cls)
            os.makedirs(cls_dir, exist_ok=True)

            # Use both global symbol index and position indices for uniqueness
            fname = (
                f"{image_id}_"
                f"{b.get('i', 0):06d}_p{b.get('page', 0)}_b{b.get('block', 0)}_r{b.get('paragraph', 0)}_"
                f"w{b.get('word', 0)}_s{b.get('symbol', 0)}.png"
            )
            out_path = os.path.join(cls_dir, fname)
            cv2.imwrite(out_path, patch)
            written += 1

        return written

    output_dir = output_dir or "test_output"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)

    # Downscale large images before sending to Vision (faster upload + processing,
    # no measurable box-count loss at ~1MP). We OCR the smaller image but scale the
    # returned vertices back to the ORIGINAL frame (via inv_scale below), so
    # char_boxes.json -- and every internal overlay/warp, which read the original
    # `path` -- all stay in one coordinate frame regardless of downscaling.

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
        print(f"Downscaled {w}x{h} -> {small.shape[1]}x{small.shape[0]} (<= {max_pixels} px) for OCR")
        ocr_input_path = os.path.join(output_dir, f"{name}_ocr_input{ext}")
        cv2.imwrite(ocr_input_path, small)
        print(f"  saved OCR input (downscaled) to: {ocr_input_path}")
    else:
        with open(path, "rb") as image_file:
            content = image_file.read()
    inv_scale = 1.0 / scale

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)

    # NOTE: `text_detection` mostly yields word-ish boxes via `text_annotations`.
    # For true character-level boxes we need `document_text_detection` and traverse
    # `full_text_annotation` down to words.symbols.
    response = client.document_text_detection(image=image)
    fta = response.full_text_annotation
    print("Text (full_text_annotation):")
    if fta and fta.text:
        print(f'\n"{fta.text}"')
    else:
        print("\n<no text>")

    # Track the extreme characters/words
    leftmost_text = None
    rightmost_text = None
    topmost_text = None
    bottommost_text = None

    leftmost_x = float('inf')
    rightmost_x = float('-inf')
    topmost_y = float('inf')
    bottommost_y = float('-inf')

    hebrew_text_count = 0

    def _norm_vertex(v):
        # Vision sometimes returns None for x/y; normalize to ints and scale back
        # to the original frame (inv_scale == 1.0 when no downscaling happened).
        x = 0 if v.x is None else int(round(v.x * inv_scale))
        y = 0 if v.y is None else int(round(v.y * inv_scale))
        return {"x": x, "y": y}

    def _bbox_minmax(vertices):
        xs = [0 if v.x is None else int(round(v.x * inv_scale)) for v in vertices]
        ys = [0 if v.y is None else int(round(v.y * inv_scale)) for v in vertices]
        return min(xs), max(xs), min(ys), max(ys)

    # Extract character-level (symbol) boxes.
    # For the "match Google OCR style" task, we store ALL symbols (not only Hebrew).
    char_boxes = []
    if fta and fta.pages:
        symbol_index = 0
        for page_idx, page in enumerate(fta.pages):
            for block_idx, block in enumerate(page.blocks):
                for para_idx, para in enumerate(block.paragraphs):
                    for word_idx, word in enumerate(para.words):
                        for sym_idx, sym in enumerate(word.symbols):
                            ch = sym.text or ""
                            verts = sym.bounding_box.vertices if sym.bounding_box else []
                            if not verts:
                                continue

                            # Skip empty/whitespace-like symbols (rare)
                            if not ch.strip():
                                symbol_index += 1
                                continue

                            if contains_hebrew(ch):
                                hebrew_text_count += 1

                                min_x, max_x, min_y, max_y = _bbox_minmax(verts)
                                if min_x < leftmost_x:
                                    leftmost_x = min_x
                                    leftmost_text = {"text": ch}
                                if max_x > rightmost_x:
                                    rightmost_x = max_x
                                    rightmost_text = {"text": ch}
                                if min_y < topmost_y:
                                    topmost_y = min_y
                                    topmost_text = {"text": ch}
                                if max_y > bottommost_y:
                                    bottommost_y = max_y
                                    bottommost_text = {"text": ch}

                            char_boxes.append(
                                {
                                    "i": symbol_index,
                                    "char": ch,
                                    "vertices": [_norm_vertex(v) for v in verts],
                                    "page": page_idx,
                                    "block": block_idx,
                                    "paragraph": para_idx,
                                    "word": word_idx,
                                    "symbol": sym_idx,
                                }
                            )
                            symbol_index += 1

    print(f"\nFound {hebrew_text_count} Hebrew text segments")

    # Collect all vertices from Hebrew character boxes (for page quad)
    all_hebrew_points = []
    if leftmost_text:
        print(f"Leftmost char: '{leftmost_text['text']}' at x={leftmost_x}")
        print(f"Rightmost char: '{rightmost_text['text']}' at x={rightmost_x}")
        print(f"Topmost char: '{topmost_text['text']}' at y={topmost_y}")
        print(f"Bottommost char: '{bottommost_text['text']}' at y={bottommost_y}")

        for b in char_boxes:
            for v in b["vertices"]:
                all_hebrew_points.append([v["x"], v["y"]])

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    # If we didn't find any Hebrew characters, skip the page-level quad/warp,
    # but still export per-symbol (character) boxes JSON (and an overlay) so COCO export
    # can include this image (possibly with 0 annotations).
    if not all_hebrew_points:
        print("\nNo Hebrew text found - skipping bounding box/warp (still exporting per-symbol boxes).")

        output_path_chars = os.path.join(output_dir, f"{name}_char_boxes.json")
        with open(output_path_chars, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": os.path.abspath(path),
                    "text": (fta.text if fta else ""),
                    "boxes": char_boxes,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved character-level bounding boxes to: {output_path_chars}")

        from PIL import ImageDraw

        img2 = Image.open(path)
        draw2 = ImageDraw.Draw(img2)
        boxes_drawn = 0
        for b in char_boxes:
            pts = [(v["x"], v["y"]) for v in b["vertices"]]
            if len(pts) >= 4:
                draw2.polygon(pts, outline="black", width=3)
                try:
                    draw2.text((pts[0][0] + 2, pts[0][1] + 2), b["char"], fill="red")
                except Exception:
                    pass
                boxes_drawn += 1

        output_path2 = os.path.join(output_dir, f"{name}_all_boxes{ext}")
        img2.save(output_path2)
        print(f"Saved {boxes_drawn} character bounding boxes overlay to: {output_path2}")

        return output_path_chars

    # Convert to numpy array for OpenCV
    points_array = np.array(all_hebrew_points, dtype=np.int32)

    # Find convex hull
    hull = cv2.convexHull(points_array)

    # Approximate the hull to a quadrilateral (4 vertices)
    # Use a small epsilon to get a tight fit
    epsilon = 0.02 * cv2.arcLength(hull, True)
    quad = cv2.approxPolyDP(hull, epsilon, True)

    # If approximation doesn't give us exactly 4 points, adjust epsilon
    attempts = 0
    while len(quad) != 4 and attempts < 20:
        if len(quad) > 4:
            epsilon *= 1.5  # Increase to get fewer points
        else:
            epsilon *= 0.5  # Decrease to get more points
        quad = cv2.approxPolyDP(hull, epsilon, True)
        attempts += 1

    # If we still don't have 4 points, fall back to convex hull
    if len(quad) != 4:
        print(f"Warning: Could not approximate to exactly 4 points (got {len(quad)}), using convex hull")
        quad = hull

    quad = quad.reshape(-1, 2)
    print(f"\nMinimal bounding quadrilateral corners ({len(quad)} points): {quad.tolist()}")

    # Get image dimensions to clip the quadrilateral
    from PIL import Image as PIL_Image
    temp_img = PIL_Image.open(path)
    img_width, img_height = temp_img.size
    print(f"Image dimensions: {img_width}x{img_height}")

    # Clip quadrilateral points to image bounds
    quad[:, 0] = np.clip(quad[:, 0], 0, img_width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, img_height - 1)

    box = quad
    print(f"Clipped box points: {[f'[{p[0]:.0f},{p[1]:.0f}]' for p in box]}")

    # Import ImageDraw here
    from PIL import ImageDraw, ImageFont

    # Perform perspective transformation to convert quadrilateral to rectangle
    ordered_quad = order_points(box.astype(np.float32))
    print(f"Ordered quadrilateral: TL={ordered_quad[0]}, TR={ordered_quad[1]}, BR={ordered_quad[2]}, BL={ordered_quad[3]}")

    # Output 1: Draw minimal bounding quadrilateral (using ORDERED points)
    img1 = Image.open(path)
    draw1 = ImageDraw.Draw(img1)
    # Convert ordered points to list of tuples for PIL
    ordered_box_points = [(int(p[0]), int(p[1])) for p in ordered_quad]
    draw1.polygon(ordered_box_points, outline="red", width=5)

    # Draw corner labels for debugging
    try:
        font = ImageFont.truetype("Arial", 40)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except OSError:
            font = ImageFont.load_default()

    for i, (x, y) in enumerate(ordered_box_points):
        label = ["TL", "TR", "BR", "BL"][i]
        draw1.ellipse([x-20, y-20, x+20, y+20], fill="blue", outline="blue")
        draw1.text((x+30, y-30), label, font=font, fill="green")

    output_path1 = os.path.join(output_dir, f"{name}_bbox{ext}")
    img1.save(output_path1)
    print(f"Saved minimal bounding quadrilateral to: {output_path1}")

    # Calculate width and height of the destination rectangle
    # Width is the maximum of top width and bottom width
    top_width = np.linalg.norm(ordered_quad[1] - ordered_quad[0])
    bottom_width = np.linalg.norm(ordered_quad[2] - ordered_quad[3])
    max_width = int(max(top_width, bottom_width))

    # Height is the maximum of left height and right height
    left_height = np.linalg.norm(ordered_quad[3] - ordered_quad[0])
    right_height = np.linalg.norm(ordered_quad[2] - ordered_quad[1])
    max_height = int(max(left_height, right_height))

    print(f"Destination dimensions: width={max_width}, height={max_height}")
    print(f"Top width: {top_width:.1f}, Bottom width: {bottom_width:.1f}")
    print(f"Left height: {left_height:.1f}, Right height: {right_height:.1f}")

    # Define destination rectangle corners (top-left, top-right, bottom-right, bottom-left)
    dst_rect = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]
    ], dtype=np.float32)

    print(f"Destination rect: {dst_rect}")

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(ordered_quad, dst_rect)

    print(f"\nPerspective transformation matrix:")
    print(M)

    # Load original image with cv2 for transformation
    img_cv = cv2.imread(path)
    print(f"Original image shape: {img_cv.shape}")

    # Apply perspective transformation
    warped = cv2.warpPerspective(img_cv, M, (max_width, max_height))

    print(f"Warped image shape: {warped.shape}")

    # Test: Transform the source corners to see where they end up
    src_corners_test = np.array([ordered_quad], dtype=np.float32)
    dst_corners_test = cv2.perspectiveTransform(src_corners_test, M)
    print(f"Source corners transformed: {dst_corners_test[0]}")

    # Save the warped (rectangular) image
    output_path_warped = os.path.join(output_dir, f"{name}_warped{ext}")
    cv2.imwrite(output_path_warped, warped)
    print(f"Saved perspective-transformed rectangle to: {output_path_warped}")

    # Output 2: Save character-level boxes as JSON (Hebrew chars only)
    output_path_chars = os.path.join(output_dir, f"{name}_char_boxes.json")
    with open(output_path_chars, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": os.path.abspath(path),
                "text": (fta.text if fta else ""),
                "boxes": char_boxes,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved character-level bounding boxes to: {output_path_chars}")

    # Output 2b: export per-character crops grouped by class (for CNN training)
    if dataset_root:
        img_cv_for_crops = cv2.imread(path)
        if img_cv_for_crops is None:
            print(f"Warning: could not read image with cv2 for dataset export: {path}")
        else:
            img_id = dataset_image_id or name
            written = export_char_dataset(
                img_cv_for_crops,
                char_boxes,
                dataset_root,
                min_size=min_size,
                padding_frac=padding_frac,
                image_id=img_id,
            )
            print(f"Saved {written} character crops grouped by class under: {dataset_root}")

    # Output 3: Draw all character boxes (debug)
    img2 = Image.open(path)
    draw2 = ImageDraw.Draw(img2)

    boxes_drawn = 0
    for b in char_boxes:
        pts = [(v["x"], v["y"]) for v in b["vertices"]]
        if len(pts) >= 4:
            draw2.polygon(pts, outline="black", width=3)
            # Label with char (optional; can get noisy)
            try:
                draw2.text((pts[0][0] + 2, pts[0][1] + 2), b["char"], fill="red")
            except Exception:
                pass
            boxes_drawn += 1

    # Backwards compatible name: historically this file was word-level `*_all_boxes`.
    # It is now character-level boxes.
    output_path2 = os.path.join(output_dir, f"{name}_all_boxes{ext}")
    img2.save(output_path2)
    print(f"Saved {boxes_drawn} character bounding boxes overlay to: {output_path2}")

    return output_path_chars

if __name__ == "__main__":
    import glob
    import argparse
    import os

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
    parser.add_argument(
        "--dataset-out",
        default="test_output/char_dataset",
        help="Output root for CNN dataset export (per-class folders). Set empty to disable.",
    )
    parser.add_argument("--padding-frac", type=float, default=0.20, help="Padding as fraction of char box size (e.g. 0.2 = 20%).")
    parser.add_argument("--min-size", type=int, default=6, help="Skip crops with width/height smaller than this (px).")
    parser.add_argument("--max-pixels", type=int, default=1_000_000,
                        help="Downscale images so total pixels <= this before OCR (0 = no downscale).")
    args = parser.parse_args()

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

    print(f"Found {len(image_files)} images to process\n")

    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(image_files)}: {image_path}")
        print(f"{'='*80}")

        try:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = (args.output_dir or "test_output").strip()
            if args.skip_existing:
                expected_json = os.path.join(out_dir, f"{image_id}_char_boxes.json")
                if os.path.exists(expected_json):
                    print(f"↷ Skip (already exists): {expected_json}")
                    continue

            dataset_out = (args.dataset_out or "").strip()
            dataset_root = dataset_out if dataset_out else None
            output_path = detect_text(
                image_path,
                dataset_root=dataset_root,
                dataset_image_id=image_id,
                output_dir=out_dir,
                padding_frac=float(args.padding_frac),
                min_size=int(args.min_size),
                max_pixels=int(args.max_pixels),
            )
            if output_path:
                print(f"✓ Successfully processed: {output_path}")
            else:
                print(f"⚠ Skipped: {image_path}")
        except Exception as e:
            print(f"✗ Error processing {image_path}: {e}")

    print(f"\n{'='*80}")
    print(f"Completed processing {len(image_files)} images")
    print(f"{'='*80}")
