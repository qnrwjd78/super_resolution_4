#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIS_PY="$SCRIPT_DIR/04_visualization.py"
LATEX_PY="$SCRIPT_DIR/05_export_latex_table.py"
DEFAULT_CONFIG_JSON="$SCRIPT_DIR/latex_config.json"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <input_dir> <output_dir> [latex_config_json]" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CONFIG_JSON="${3:-$DEFAULT_CONFIG_JSON}"
TEX_OUT="${TEX_OUT:-$OUTPUT_DIR/mean_metrics.tex}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: input directory not found: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$VIS_PY" "$INPUT_DIR" "$OUTPUT_DIR"
"$PYTHON_BIN" "$LATEX_PY" "$INPUT_DIR" --config_json "$CONFIG_JSON" -o "$TEX_OUT"

echo "[OK] wrote visualization and LaTeX summary under: $OUTPUT_DIR"
