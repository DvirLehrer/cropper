#!/usr/bin/env python3
"""Run Hebrew OCR crop pipeline on benchmark images."""

from __future__ import annotations

from PIL import Image

from cropper_config import (
    BENCHMARK_DIR,
    CROPPED_DIR,
    DEBUG_DIR,
    LANG,
    OCR_TEXT_DIR,
    PREPROCESSED_DIR,
    TEXT_DIR,
    TYPES_CSV,
)
from ocr_utils import iter_images, levenshtein, load_types
from pipeline_crop_service import crop_image
from draw_periodic_pattern import draw_periodic_pattern_for_image
from target_texts import load_target_texts, strip_newlines


def _target_text_for_image(image_type: str, target_texts: dict[str, str]) -> str | None:
    if image_type == "m":
        return target_texts["m"]
    if image_type in ("s", "v", "k", "p"):
        key = {"s": "shema", "v": "vehaya", "k": "kadesh", "p": "peter"}[image_type]
        return target_texts[key]
    return None


def main() -> None:
    if not BENCHMARK_DIR.exists():
        raise SystemExit(f"Benchmark dir not found: {BENCHMARK_DIR}")

    type_map = load_types(TYPES_CSV)
    target_texts = load_target_texts(TEXT_DIR)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for path in iter_images(BENCHMARK_DIR):
        print(f"processing: {path.name}")
        image_type = type_map.get(path.name)
        if image_type is None:
            print(f"warning: unknown type for {path.name}, skipping")
            continue
        target_for_image = _target_text_for_image(image_type, target_texts)
        if target_for_image is None:
            print(f"warning: unknown type for {path.name}, skipping")
            continue

            target_chars = max(len(strip_newlines(target_for_image)), 1)
            image_full = Image.open(path).convert("RGB")
            result = crop_image(
                image_full,
                lang=LANG,
                target_chars=target_chars,
                debug_path=DEBUG_DIR / f"{path.stem}_debug.png",
            )
            preprocessed_path = PREPROCESSED_DIR / f"{path.stem}_dark_masked.png"
            result["stripe_ready"].save(preprocessed_path)
            mask_continuum_path = PREPROCESSED_DIR / f"{path.stem}_dark_masked_mask_continuum_debug.png"
            result["stripe_mask_continuum_debug"].save(mask_continuum_path)
            print(f"mask-continuum-debug: {mask_continuum_path.name}")
            avg_char_size = result.get("avg_char_size")
            min_lag_full_px = 8
            max_lag_full_px = None
            if isinstance(avg_char_size, (int, float)) and avg_char_size > 0:
                # Option 1: never sparser than OCR line-scale (can be denser).
                max_lag_full_px = max(6, int(round(1.85 * float(avg_char_size))))
            periodic_meta = draw_periodic_pattern_for_image(
                mask_continuum_path,
                light_debug_image=result["cropped"],
                light_debug_mask_image=result["stripe_dark_mask"],
                light_debug_out_path=DEBUG_DIR / f"{path.stem}_stripe_light_debug.png",
                min_lag_full_px=min_lag_full_px,
                max_lag_full_px=max_lag_full_px,
            )
            print(
                "periodic: "
                f"lag={int(periodic_meta['lag'])} "
                f"corr={float(periodic_meta['corr']):.3f} "
                f"peaks={int(periodic_meta['peaks'])} "
                f"spacing_cons={float(periodic_meta['spacing_cons']):.3f} "
                f"strength={float(periodic_meta['strength']):.3f} "
                f"time={float(periodic_meta['periodic_time_sec']):.3f}s"
            )
            if periodic_meta.get("light_debug_output"):
                print(f"stripe-light-debug: {periodic_meta['light_debug_output']}")
            correction = result["correction"]
            print(
                f"correction: {correction.mode} "
                f"(mean_abs={correction.mean_abs:.5f} std={correction.std:.5f} "
                f"resid_mean={correction.resid_mean:.3f} resid_std={correction.resid_std:.3f} "
                f"curve_std={correction.curve_std:.3f})"
            )

            cropped = result["cropped"]
            cropped.save(CROPPED_DIR / f"{path.stem}_crop.png")
            print(
                "crop stats: "
                f"{cropped.width}x{cropped.height} "
                f"area={result['crop_area']} "
                f"target_chars={target_chars} "
                f"px_per_char={result['px_per_char']:.1f}"
            )
            print(
                "timing: "
                f"ocr1={result['timing']['ocr1']:.3f}s "
                f"layout={result['timing']['layout']:.3f}s "
                f"crop={result['timing']['crop']:.3f}s "
                f"ocr2={result['timing']['ocr2']:.3f}s "
                f"debug={result['timing']['debug']:.3f}s "
                f"left={result['timing']['left']:.3f}s"
            )

            (OCR_TEXT_DIR / f"{path.stem}.txt").write_text(result["text"], encoding="utf-8")
            ocr_text = result["ocr_text"]
            if image_type == "m":
                distance = levenshtein(ocr_text, target_texts["m"])
                print(f"type: {image_type}  distance: {distance}")
            else:
                names = ("shema", "vehaya", "kadesh", "peter")
                distances = {name: levenshtein(ocr_text, target_texts[name]) for name in names}
                guess_name, guess_distance = min(distances.items(), key=lambda item: item[1])
                print(f"type: {image_type}  guess: {guess_name}  distance: {guess_distance}")


if __name__ == "__main__":
    main()
