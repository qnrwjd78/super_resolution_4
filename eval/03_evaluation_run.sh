#!/usr/bin/env bash
set -euo pipefail

# Example: run 03_evaluation.py three times (edit JSON paths as needed).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_PY="$SCRIPT_DIR/03_evaluation.py"

# Input JSONs (our format: [{"res": "...", "hr": "..."}, ...])
INPUT_DIR="/sr/data/result/meta"
J1="$INPUT_DIR/result.dat.json"
J2="$INPUT_DIR/result_mambairv2.json"
J3="$INPUT_DIR/result_swinir.json"
J4="$INPUT_DIR/result_swin2sr.json"
J5="$INPUT_DIR/result_hat.json"

# Where to write evaluation outputs
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/output/eval}"

# Common options
DEVICE="${DEVICE:-auto}"       # auto|cuda|cpu
FR_RESIZE="${FR_RESIZE:-to_ref}" # to_ref|none

eval python "$EVAL_PY" -i "$J1" -o "$OUT_DIR/eval_dat.json" --device "$DEVICE" --fr_resize "$FR_RESIZE"
eval python "$EVAL_PY" -i "$J2" -o "$OUT_DIR/eval_mambairv2.json" --device "$DEVICE" --fr_resize "$FR_RESIZE"
eval python "$EVAL_PY" -i "$J3" -o "$OUT_DIR/eval_swinir.json" --device "$DEVICE" --fr_resize "$FR_RESIZE"
eval python "$EVAL_PY" -i "$J4" -o "$OUT_DIR/eval_swin2sr.json" --device "$DEVICE" --fr_resize "$FR_RESIZE"
eval python "$EVAL_PY" -i "$J5" -o "$OUT_DIR/eval_hat.json" --device "$DEVICE" --fr_resize "$FR_RESIZE"

echo "[OK] wrote outputs under: $OUT_DIR"
