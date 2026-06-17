#! /usr/bin/env python3
"""
Generate a synthetic parchment-scroll image by randomly compositing
character crops from char_dataset_clean/ AND glyphs from the fonts/ directory
(TTF renders + realsefarad / realA reference images) into rows of Hebrew words.

Produces:
  <output>.png   — the synthetic scroll image
  <output>.txt   — YOLO-format annotations (class cx cy bw bh, normalised)

Usage:
  python generate_synthetic_scroll.py
  python generate_synthetic_scroll.py --seed 42 --rows 4 --words-per-row 6
  python generate_synthetic_scroll.py --no-ocr-crops   # fonts only
  python generate_synthetic_scroll.py --no-font-sources  # OCR crops only
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEBREW_CHARS = [chr(cp) for cp in range(0x05D0, 0x05EB)]
CHAR_TO_ID   = {ch: i for i, ch in enumerate(HEBREW_CHARS)}

# Letters whose bounding box overflows the row line
DESCENDERS = {'ק', 'ן', 'ץ', 'ך'}   # extend below the row
ASCENDERS  = {'ל'}                    # extend above the row
DESCENDER_FRAC = 0.38   # descender adds this fraction of char_height below
ASCENDER_FRAC  = 0.32   # ascender adds this fraction of char_height above

# Mezuzah text with word spacing (Shema + V'ahavta + V'haya paragraphs)
MEZUZA_WORDS = (
    "שמע ישראל יהוה אלהינו יהוה אחד "
    "ואהבת את יהוה אלהיך בכל לבבך ובכל נפשך ובכל מאדך "
    "והיו הדברים האלה אשר אנכי מצוך היום על לבבך "
    "ושננתם לבניך ודברת בם בשבתך בביתך ובלכתך בדרך ובשכבך ובקומך "
    "וקשרתם לאות על ידך והיו לטטפת בין עיניך "
    "וכתבתם על מזזות ביתך ובשעריך "
    "והיה אם שמוע תשמעו אל מצותי אשר אנכי מצוה אתכם היום "
    "לאהבה את יהוה אלהיכם ולעבדו בכל לבבכם ובכל נפשכם "
    "ונתתי מטר ארצכם בעתו יורה ומלקוש ואספת דגנך ותירשך ויצהרך "
    "ונתתי עשב בשדך לבהמתך ואכלת ושבעת "
    "השמרו לכם פן יפתה לבבכם וסרתם ועבדתם אלהים אחרים והשתחויתם להם "
    "וחרה אף יהוה בכם ועצר את השמים ולא יהיה מטר "
    "והאדמה לא תתן את יבולה ואבדתם מהרה מעל הארץ הטבה אשר יהוה נתן לכם "
    "ושמתם את דברי אלה על לבבכם ועל נפשכם "
    "וקשרתם אתם לאות על ידכם והיו לטוטפת בין עיניכם "
    "ולמדתם אתם את בניכם לדבר בם בשבתך בביתך ובלכתך בדרך ובשכבך ובקומך "
    "וכתבתם על מזוזות ביתך ובשעריך "
    "למען ירבו ימיכם וימי בניכם על האדמה "
    "אשר נשבע יהוה לאבתיכם לתת להם כימי השמים על הארץ"
).split()

PARCHMENT_BASE = np.array([200, 178, 145], dtype=np.uint8)   # warm tan (RGB)
INK_COLOR      = np.array([18,  14,  10],  dtype=np.uint8)   # near-black, warm


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def load_pool(dataset_dir: Path) -> dict:
    """Return dict: char -> list of Path for every Hebrew character class found."""
    pool = {}
    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        m = re.match(r"U([0-9A-Fa-f]{4})_(.)$", cls_dir.name)
        if not m or not (0x05D0 <= int(m.group(1), 16) <= 0x05EA):
            continue
        ch    = m.group(2)
        paths = sorted(cls_dir.glob("*.png"))
        if paths:
            pool[ch] = paths
    return pool


# ---------------------------------------------------------------------------
# Font sources loader
# ---------------------------------------------------------------------------

# TTF sizes to render at (multiple sizes add natural scale variety in the pool)
_TTF_SIZES = [48, 64, 80, 96, 112]


def load_font_sources(fonts_dir: Path) -> dict:
    """
    Build extra pool entries from the fonts/ directory:
      - fonts/realsefarad/{ch}.jpg  and  fonts/realA/{ch}.png  → grayscale arrays
      - fonts/**/*.ttf (skipping stam 2/) rendered at _TTF_SIZES → grayscale arrays

    Returns dict: char -> list of np.ndarray (grayscale, ink is dark / bg is light).
    """
    sources = {ch: [] for ch in HEBREW_CHARS}

    # Real image directories (one image per character)
    for subdir in sorted(fonts_dir.iterdir()):
        if not subdir.is_dir() or subdir.name == "stam 2":
            continue
        for ch in HEBREW_CHARS:
            for ext in ("jpg", "jpeg", "png"):
                p = subdir / f"{ch}.{ext}"
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        sources[ch].append(img)
                    break

    # TTF fonts
    ttf_files = [p for p in sorted(fonts_dir.rglob("*.ttf")) if "stam 2" not in p.parts]
    dummy = ImageDraw.Draw(Image.new("L", (1, 1)))
    for font_path in ttf_files:
        for size in _TTF_SIZES:
            try:
                fnt = ImageFont.truetype(str(font_path), size)
            except OSError:
                continue
            for ch in HEBREW_CHARS:
                try:
                    l, t, r, b = dummy.textbbox((0, 0), ch, font=fnt)
                except Exception:
                    continue
                gw, gh = r - l, b - t
                if gw <= 0 or gh <= 0:
                    continue
                # Add padding matching OCR crop proportions (~15% of max dim per side)
                pad = max(2, int(max(gw, gh) * 0.15))
                img_pil = Image.new("L", (gw + 2 * pad, gh + 2 * pad), 255)
                ImageDraw.Draw(img_pil).text((pad - l, pad - t), ch, font=fnt, fill=0)
                sources[ch].append(np.array(img_pil, dtype=np.uint8))

    n_chars = sum(1 for v in sources.values() if v)
    n_imgs  = sum(len(v) for v in sources.values())
    print(f"Font sources: {n_imgs} glyphs across {n_chars} characters")
    return {ch: v for ch, v in sources.items() if v}


# ---------------------------------------------------------------------------
# Character rendering
# ---------------------------------------------------------------------------

def render_char(source, target_h: int):
    """
    Accept either a Path (load from disk) or a np.ndarray (already grayscale).
    Resize to target_h, Otsu-threshold to ink mask.
    Returns (ink_mask, width) or (None, 0) if the source is unusable.
    """
    if isinstance(source, np.ndarray):
        img = source
    else:
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None, 0

    h, w = img.shape
    new_w = max(1, int(w * target_h / h))
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_frac = mask.sum() / (255 * mask.size)
    if ink_frac < 0.01:
        return None, 0

    return mask, new_w


def sample_char(rng, pool: dict, ch: str, target_h: int, max_tries: int = 5):
    """Randomly pick a usable source (Path or array) for character ch."""
    sources = pool.get(ch, [])
    if not sources:
        return None, 0
    indices = rng.permutation(len(sources))[:max_tries]
    for idx in indices:
        mask, w = render_char(sources[idx], target_h)
        if mask is not None:
            return mask, w
    return None, 0


def _effective_h(ch: str, char_height: int):
    """Return (render_height, y_offset) for a character relative to the row top.
    y_offset < 0 means the character starts above the row (ascender).
    render_height > char_height means it extends beyond the row (descender or ascender).
    """
    if ch in DESCENDERS:
        extra = int(char_height * DESCENDER_FRAC)
        return char_height + extra, 0          # top-aligned, overflows below
    if ch in ASCENDERS:
        extra = int(char_height * ASCENDER_FRAC)
        return char_height + extra, -extra     # bottom-aligned, overflows above
    return char_height, 0


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def make_parchment(rng, height: int, width: int) -> np.ndarray:
    """
    Create a highly variable parchment-like background.

    Each call can produce very different looks: light cream, aged tan, dark brown,
    heavily stained, bleached, mottled — controlled entirely by rng.
    Pass a fresh per-image rng to get visually distinct results.
    """
    # --- Base colour: wide range across parchment/paper palette ---
    palettes = [
        [230, 220, 195],   # light cream / vellum
        [200, 178, 145],   # warm tan (classic)
        [175, 145, 105],   # aged brown
        [215, 205, 180],   # pale grey-yellow
        [190, 160, 120],   # darker ochre
        [240, 230, 210],   # near-white bleached
        [160, 130,  95],   # very dark aged
    ]
    base = np.array(palettes[rng.integers(len(palettes))], dtype=np.float32)
    base += rng.uniform(-15, 15, size=3)
    canvas = np.full((height, width, 3), base, dtype=np.float32)

    # --- Large-scale colour variation (uneven aging, moisture zones) ---
    n_blobs = int(rng.integers(5, 14))
    blob_sigma = float(rng.uniform(15, 50))   # how strong each blob is
    for _ in range(n_blobs):
        bh = max(1, height // int(rng.integers(2, 6)))
        bw = max(1, width  // int(rng.integers(2, 6)))
        blob = rng.normal(0, blob_sigma, (bh, bw, 3))
        blob_up = cv2.resize(blob, (width, height), interpolation=cv2.INTER_CUBIC)
        canvas += blob_up

    # --- Medium texture ---
    mh = max(1, height // 5)
    mw = max(1, width  // 5)
    med_sigma = float(rng.uniform(6, 18))
    medium = rng.normal(0, med_sigma, (mh, mw, 3))
    canvas += cv2.resize(medium, (width, height), interpolation=cv2.INTER_LINEAR)

    # --- Fine grain ---
    canvas += rng.normal(0, float(rng.uniform(2, 8)), (height, width, 3))

    # --- Stains: variable count and strength ---
    stain_layer = np.zeros((height, width), dtype=np.float32)
    n_stains = int(rng.integers(0, 8))
    for _ in range(n_stains):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        rx = int(rng.integers(width  // 12, width  // 2))
        ry = int(rng.integers(height // 12, height // 2))
        angle  = float(rng.uniform(0, 180))
        strength = float(rng.uniform(10, 45))
        cv2.ellipse(stain_layer, (cx, cy), (rx, ry), angle, 0, 360, strength, -1)
    blur_k = int(max(width, height) // 20) | 1
    stain_layer = cv2.GaussianBlur(stain_layer, (blur_k, blur_k), 0)
    # stains can darken OR lighten
    if rng.random() < 0.5:
        canvas -= stain_layer[:, :, np.newaxis]
    else:
        canvas += stain_layer[:, :, np.newaxis] * 0.5

    # --- Directional gradient (lighting, uneven illumination) ---
    Y, X = np.mgrid[:height, :width]
    angle = float(rng.uniform(0, 2 * np.pi))
    grad = X / width * np.cos(angle) + Y / height * np.sin(angle)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-9)
    grad_strength = float(rng.uniform(10, 50))
    canvas += (grad * grad_strength - grad_strength / 2)[:, :, np.newaxis]

    # --- Vignette (variable shape and strength) ---
    cx2, cy2 = width / 2, height / 2
    dist = np.sqrt(((X - cx2) / (cx2 + 1)) ** 2 + ((Y - cy2) / (cy2 + 1)) ** 2)
    v_strength = float(rng.uniform(0.1, 0.55))
    vignette = np.clip(1.0 - v_strength * dist, 0.5, 1.0)
    canvas *= vignette[:, :, np.newaxis]

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
    dataset_dir: str = "test_output/char_dataset_clean",
    fonts_dir: str = "fonts",
    output: str = "test_output/synthetic_scroll.png",
    rows: int = 4,
    words_per_row: int = 6,
    char_height: int = 72,
    char_spacing: int = 4,
    word_spacing: int = 44,
    line_spacing: float = 1.75,
    margin: int = 90,
    canvas_width: int = 1400,
    seed: int = None,
    use_ocr_crops: bool = True,
    use_font_sources: bool = True,
    mezuzah: bool = False,
):
    rng = np.random.default_rng(seed)
    dataset_dir = Path(dataset_dir)
    fonts_dir   = Path(fonts_dir)
    output      = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build merged pool: char -> list of (Path | np.ndarray)
    pool: dict = {ch: [] for ch in HEBREW_CHARS}

    if use_ocr_crops and dataset_dir.exists():
        ocr_pool = load_pool(dataset_dir)
        for ch, paths in ocr_pool.items():
            pool[ch].extend(paths)
        total_ocr = sum(len(v) for v in ocr_pool.values())
        print(f"OCR crops: {total_ocr} images across {len(ocr_pool)} characters")

    if use_font_sources and fonts_dir.exists():
        font_pool = load_font_sources(fonts_dir)
        for ch, arrays in font_pool.items():
            pool[ch].extend(arrays)

    chars_available = [ch for ch in HEBREW_CHARS if pool[ch]]
    if not chars_available:
        raise RuntimeError("No character sources found — check --dataset-dir and --fonts-dir")
    total = sum(len(pool[ch]) for ch in chars_available)
    print(f"Combined pool: {total} sources across {len(chars_available)} characters")

    line_height = int(char_height * line_spacing)

    # Build list of words to render
    if mezuzah:
        # Filter to characters available in pool; keep word structure
        words_to_render = []
        for word in MEZUZA_WORDS:
            filtered = [ch for ch in word if ch in pool and pool[ch]]
            if filtered:
                words_to_render.append(filtered)
        # Two-pass layout: first sample actual widths to get true row count,
        # then render. We sample one crop per character to measure real widths.
        def _sample_w(ch):
            sources = pool.get(ch, [])
            if not sources:
                return int(char_height * 0.55)
            eff_h, _ = _effective_h(ch, char_height)
            src = sources[int(rng.integers(len(sources)))]
            mask, w = render_char(src, eff_h)
            return w if mask is not None else int(char_height * 0.55)

        sim_x = canvas_width - margin
        sim_rows = 0
        for word in words_to_render:
            word_w = sum(_sample_w(ch) + char_spacing for ch in word) + word_spacing
            if sim_x - word_w < margin:
                sim_rows += 1
                sim_x = canvas_width - margin
            sim_x -= word_w
        sim_rows += 1  # final row
        canvas_height = 2 * margin + (sim_rows + 2) * line_height  # +2 row safety buffer
    else:
        canvas_height = 2 * margin + rows * line_height

    canvas = make_parchment(rng, canvas_height, canvas_width)
    placements = []  # list of (ch, x1, y1, x2, y2)

    def _place_word(word_chars, x_cursor, y_top):
        """Composite a list of (ch, mask, w) RTL at x_cursor; return new x_cursor."""
        for ch, mask, w in word_chars:
            eff_h, y_off = _effective_h(ch, char_height)
            x1 = x_cursor - w;  x2 = x_cursor
            y1 = y_top + y_off; y2 = y1 + eff_h
            # Clamp to canvas
            cx1 = max(0, min(canvas_width,  x1))
            cx2 = max(0, min(canvas_width,  x2))
            cy1 = max(0, min(canvas_height, y1))
            cy2 = max(0, min(canvas_height, y2))
            if cx2 > cx1 and cy2 > cy1:
                mx1 = cx1 - x1;  mx2 = cx2 - x1
                my1 = cy1 - y1;  my2 = cy2 - y1
                roi_mask   = mask[my1:my2, mx1:mx2]
                roi_canvas = canvas[cy1:cy2, cx1:cx2]
                roi_canvas[roi_mask == 255] = INK_COLOR
                canvas[cy1:cy2, cx1:cx2] = roi_canvas
            placements.append((ch, x1, y1, x2, y2))
            x_cursor = x1 - char_spacing
        return x_cursor

    if mezuzah:
        row_idx  = 0
        x_cursor = canvas_width - margin
        for word in words_to_render:
            # Pre-sample
            word_chars = []
            for ch in word:
                eff_h, _ = _effective_h(ch, char_height)
                mask, w = sample_char(rng, pool, ch, eff_h)
                if mask is not None:
                    word_chars.append((ch, mask, w))
            if not word_chars:
                continue
            total_w = sum(w for _, _, w in word_chars) + char_spacing * (len(word_chars) - 1)
            if x_cursor - total_w < margin:
                row_idx += 1
                x_cursor = canvas_width - margin
            y_top    = margin + row_idx * line_height
            x_cursor = _place_word(word_chars, x_cursor, y_top)
            x_cursor -= word_spacing
    else:
        word_len_choices = [2, 3, 4, 5, 6]
        word_len_weights = [1, 3, 4, 3, 1]
        for row_idx in range(rows):
            y_top    = margin + row_idx * line_height
            x_cursor = canvas_width - margin
            words_this_row = int(rng.integers(max(1, words_per_row - 1), words_per_row + 2))
            word_lengths   = rng.choice(
                word_len_choices, size=words_this_row,
                p=np.array(word_len_weights) / sum(word_len_weights),
            )
            for word_len in word_lengths:
                if x_cursor - margin < word_len * (char_height // 2):
                    break
                word_chars = []
                for _ in range(int(word_len)):
                    ch = rng.choice(chars_available)
                    eff_h, _ = _effective_h(ch, char_height)
                    mask, w = sample_char(rng, pool, ch, eff_h)
                    if mask is not None:
                        word_chars.append((ch, mask, w))
                if not word_chars:
                    continue
                total_word_w = sum(w for _, _, w in word_chars) + char_spacing * (len(word_chars) - 1)
                if x_cursor - total_word_w < margin:
                    break
                x_cursor = _place_word(word_chars, x_cursor, y_top)
                x_cursor -= word_spacing

    # Save image (OpenCV expects BGR)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Saved image: {output}  ({canvas_width}×{canvas_height}px, {len(placements)} characters)")

    # Save YOLO annotation
    yolo_lines = []
    for ch, x1, y1, x2, y2 in placements:
        if ch not in CHAR_TO_ID:
            continue
        cx = ((x1 + x2) / 2) / canvas_width
        cy = ((y1 + y2) / 2) / canvas_height
        bw = (x2 - x1) / canvas_width
        bh = (y2 - y1) / canvas_height
        yolo_lines.append(f"{CHAR_TO_ID[ch]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    yolo_path = output.with_suffix(".txt")
    yolo_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
    print(f"Saved YOLO labels: {yolo_path}  ({len(yolo_lines)} boxes)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic STaM scroll image from char_dataset_clean crops."
    )
    parser.add_argument("--dataset-dir",    default="test_output/char_dataset_clean")
    parser.add_argument("--fonts-dir",      default="fonts")
    parser.add_argument("--output",         default="test_output/synthetic_scroll.png")
    parser.add_argument("--rows",           type=int,   default=4)
    parser.add_argument("--words-per-row",  type=int,   default=6)
    parser.add_argument("--char-height",    type=int,   default=72)
    parser.add_argument("--char-spacing",   type=int,   default=4)
    parser.add_argument("--word-spacing",   type=int,   default=44)
    parser.add_argument("--line-spacing",   type=float, default=1.75)
    parser.add_argument("--margin",         type=int,   default=90)
    parser.add_argument("--canvas-width",   type=int,   default=1400)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--mezuzah",         action="store_true",
                        help="Use actual mezuzah text instead of random words")
    parser.add_argument("--no-ocr-crops",    dest="use_ocr_crops",    action="store_false",
                        help="Exclude OCR crops from char_dataset_clean/")
    parser.add_argument("--no-font-sources", dest="use_font_sources", action="store_false",
                        help="Exclude font directory sources")
    args = parser.parse_args()

    generate(
        dataset_dir=args.dataset_dir,
        fonts_dir=args.fonts_dir,
        output=args.output,
        rows=args.rows,
        words_per_row=args.words_per_row,
        char_height=args.char_height,
        char_spacing=args.char_spacing,
        word_spacing=args.word_spacing,
        line_spacing=args.line_spacing,
        margin=args.margin,
        canvas_width=args.canvas_width,
        seed=args.seed,
        use_ocr_crops=args.use_ocr_crops,
        use_font_sources=args.use_font_sources,
        mezuzah=args.mezuzah,
    )


if __name__ == "__main__":
    main()
