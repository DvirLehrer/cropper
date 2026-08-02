#!/usr/bin/env bash
# Build a native Apple Silicon environment for the Stam-OCR scoring engine.
#
#   ./tools/setup_scoring_macos.sh [path-to-Stam-OCR]
#
# Creates .venv-scoring/ next to this repo, installs the pinned mac-native
# dependency set, then proves the environment actually works by loading both
# models the engine needs — the Keras .h5 letter classifier and the PyTorch
# .pth context model. An install that imports cleanly but cannot load the
# models is worse than useless, so the check is part of setup, not an
# afterthought.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAM_OCR="${1:-$HOME/dev/Stam-OCR}"
VENV="$REPO/.venv-scoring"
REQ="$REPO/tools/requirements-scoring-macos.txt"

# ── preconditions ─────────────────────────────────────────────────────────────

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS. On Linux use Stam-OCR/requirements.txt as-is." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Not Apple Silicon (uname -m = $(uname -m))." >&2
  echo "On Intel macs install tensorflow-cpu==2.15.1 instead of tensorflow-macos." >&2
  exit 1
fi

if [[ ! -f "$STAM_OCR/StamOcr.py" ]]; then
  echo "Stam-OCR not found at: $STAM_OCR" >&2
  echo "Pass its path as the first argument." >&2
  exit 1
fi

# TensorFlow 2.15 ships no cp312+ wheels, so 3.11 is not a preference, it is the
# ceiling. Look for it in the usual places before giving up.
PY311=""
for candidate in \
    "$(command -v python3.11 || true)" \
    /opt/homebrew/bin/python3.11 \
    /opt/homebrew/opt/python@3.11/bin/python3.11 \
    "$HOME/.pyenv/versions/3.11"*/bin/python3.11 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then PY311="$candidate"; break; fi
done

if [[ -z "$PY311" ]]; then
  echo "Python 3.11 not found. Install it with:" >&2
  echo "    brew install python@3.11" >&2
  echo "(TensorFlow 2.15 has no wheels for 3.12 or newer.)" >&2
  exit 1
fi

echo "python:    $PY311 ($("$PY311" --version 2>&1))"
echo "Stam-OCR:  $STAM_OCR"
echo "venv:      $VENV"
echo

# ── build ─────────────────────────────────────────────────────────────────────

"$PY311" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel
"$VENV/bin/pip" install -r "$REQ"

# ── verify ────────────────────────────────────────────────────────────────────

echo
echo "verifying…"
STAM_OCR="$STAM_OCR" "$VENV/bin/python" - <<'PY'
import os, sys, pathlib

stam = pathlib.Path(os.environ["STAM_OCR"])
os.chdir(stam)
sys.path.insert(0, str(stam))

import numpy, cv2, torch
try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise SystemExit(
        "  tensorflow did not install.\n"
        "  tensorflow-macos 2.15.1 and 2.16.x are empty redirector wheels: pip\n"
        "  reports success and installs no module. Pin 2.15.0, the last real build."
    )
import keras   # under TF 2.15 this is Keras 2; tf.keras is a lazy alias with no __version__
print(f"  numpy {numpy.__version__}   opencv {cv2.__version__}")
print(f"  torch {torch.__version__}   tensorflow {tf.__version__}   keras {keras.__version__}")
if not keras.__version__.startswith("2."):
    raise SystemExit("  Keras 3 detected. The legacy .h5 classifier needs Keras 2 — "
                     "stay on TensorFlow 2.15.")

from tensorflow.keras.models import load_model
h5 = stam / "jupyter" / "model-square-epoch27-20231226-122820__13M.h5"
model = load_model(str(h5), compile=False)
print(f"  keras classifier loaded: output {model.output_shape}")

pth = next(stam.glob("models/*insane*.pth"))
state = torch.load(str(pth), map_location="cpu")
print(f"  torch context model loaded: {len(state)} tensors")

from StamOcr import StamOcr           # exercises the full import graph
StamOcr()
print("  StamOcr imports and constructs")
PY

cat <<EOF

Done. Score with:

    $VENV/bin/python tools/bench.py score --run baseline --stam-ocr "$STAM_OCR"

The crop stage keeps using your normal python (it needs ultralytics and
google-cloud-vision, which this environment deliberately does not carry) —
the two stages are separate processes, so two environments is fine.
EOF
