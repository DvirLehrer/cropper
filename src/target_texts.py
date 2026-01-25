#!/usr/bin/env python3
"""Load canonical target texts used by OCR benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def strip_newlines(text: str) -> str:
    return text.replace("\n", "").replace("\r", "")


def _read_text_file(root: Path, target_name: str) -> str:
    direct = root / target_name
    if direct.exists():
        return direct.read_text(encoding="utf-8")
    for entry in root.iterdir():
        if entry.is_file() and entry.name.strip() == target_name:
            return entry.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Text file not found: {target_name}")


def load_target_texts(text_dir: Path) -> Dict[str, str]:
    """Return newline-stripped target texts keyed by name."""
    shema = strip_newlines(_read_text_file(text_dir, "shema"))
    vehaya = strip_newlines(_read_text_file(text_dir, "vehaya"))
    kadesh = strip_newlines(_read_text_file(text_dir, "kadesh"))
    peter = strip_newlines(_read_text_file(text_dir, "peter"))
    return {
        "shema": shema,
        "vehaya": vehaya,
        "kadesh": kadesh,
        "peter": peter,
        "m": shema + vehaya,
    }
