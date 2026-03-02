#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_OPTIONS_JSON="$STAGE2_DIR/options/inference.json"
OPTIONS_JSON="${OPTIONS_JSON:-$DEFAULT_OPTIONS_JSON}"

if [[ $# -gt 0 && "$1" == *.json ]]; then
  OPTIONS_JSON="$1"
  shift
fi

has() { command -v "$1" >/dev/null 2>&1; }

run_flux2_python() {
  if has flux2; then
    flux2 python "$@"
  elif has conda; then
    conda run -n flux2 --no-capture-output python "$@"
  else
    echo "ERROR: flux2 wrapper (or conda env 'flux2') not found." >&2
    exit 1
  fi
}

load_options_json() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    if [[ "$path" == "$DEFAULT_OPTIONS_JSON" ]]; then
      return 0
    fi
    echo "ERROR: options JSON not found: $path" >&2
    exit 1
  fi

  eval "$(
    run_flux2_python - "$path" <<'PY'
import json
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"ERROR: failed to read options JSON '{path}': {exc}")

if not isinstance(data, dict):
    raise SystemExit("ERROR: options JSON must be a JSON object.")

mapping = {
    "model_path": "CFG_MODEL_PATH",
    "input_json": "CFG_INPUT_JSON",
    "output_dir": "CFG_OUTPUT_DIR",
    "output_json": "CFG_OUTPUT_JSON",
    "default_prompt": "CFG_DEFAULT_PROMPT",
    "lora_weights_path": "CFG_LORA_WEIGHTS_PATH",
    "mode": "CFG_MODE",
    "resolution": "CFG_RESOLUTION",
    "guidance_scale": "CFG_GUIDANCE_SCALE",
    "num_inference_steps": "CFG_NUM_INFERENCE_STEPS",
    "dtype": "CFG_DTYPE",
    "device": "CFG_DEVICE",
    "seed": "CFG_SEED",
    "cpu_offload": "CFG_CPU_OFFLOAD",
    "gpu_devices": "CFG_GPU_DEVICES",
}

for json_key, shell_name in mapping.items():
    value = data.get(json_key)
    if value is None:
        continue
    if isinstance(value, bool):
        value = "1" if value else "0"
    else:
        value = str(value)
    print(f"{shell_name}={shlex.quote(value)}")
PY
  )"
}

load_options_json "$OPTIONS_JSON"

MODEL_PATH="${MODEL_PATH:-${CFG_MODEL_PATH:-$STAGE2_DIR/weights/flux2-klein-base-9b}}"
INPUT_JSON="${INPUT_JSON:-${CFG_INPUT_JSON:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUTPUT_DIR:-$STAGE2_DIR/outputs/inference}}"
OUTPUT_JSON="${OUTPUT_JSON:-${CFG_OUTPUT_JSON:-}}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-${CFG_DEFAULT_PROMPT:-}}"
LORA_WEIGHTS_PATH="${LORA_WEIGHTS_PATH:-${CFG_LORA_WEIGHTS_PATH:-}}"

MODE="${MODE:-${CFG_MODE:-full}}"
RESOLUTION="${RESOLUTION:-${CFG_RESOLUTION:-512}}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-${CFG_GUIDANCE_SCALE:-4.0}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-${CFG_NUM_INFERENCE_STEPS:-50}}"
DTYPE="${DTYPE:-${CFG_DTYPE:-bf16}}"
DEVICE="${DEVICE:-${CFG_DEVICE:-cuda}}"
SEED="${SEED:-${CFG_SEED:-0}}"
CPU_OFFLOAD="${CPU_OFFLOAD:-${CFG_CPU_OFFLOAD:-1}}"
GPU_DEVICES="${GPU_DEVICES:-${CFG_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"

if [[ -z "$INPUT_JSON" ]]; then
  echo "ERROR: set INPUT_JSON or provide it in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

ARGS=(
  "$STAGE2_DIR/models/lora_inference.py"
  --pretrained_model_name_or_path "$MODEL_PATH"
  --input_json "$INPUT_JSON"
  --output_dir "$OUTPUT_DIR"
  --mode "$MODE"
  --resolution "$RESOLUTION"
  --guidance_scale "$GUIDANCE_SCALE"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ -n "$OUTPUT_JSON" ]]; then
  ARGS+=(--output_json "$OUTPUT_JSON")
fi

if [[ -n "$DEFAULT_PROMPT" ]]; then
  ARGS+=(--default_prompt "$DEFAULT_PROMPT")
fi

if [[ -n "$LORA_WEIGHTS_PATH" ]]; then
  ARGS+=(--lora_weights_path "$LORA_WEIGHTS_PATH")
fi

if [[ "$CPU_OFFLOAD" == "1" ]]; then
  ARGS+=(--cpu_offload)
fi

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

echo "OPTIONS_JSON      : $OPTIONS_JSON"
echo "MODEL_PATH        : $MODEL_PATH"
echo "INPUT_JSON        : $INPUT_JSON"
echo "OUTPUT_DIR        : $OUTPUT_DIR"
echo "LORA_WEIGHTS_PATH : ${LORA_WEIGHTS_PATH:-<none>}"
echo "MODE              : $MODE"
echo "RESOLUTION        : $RESOLUTION"
echo "GPU_DEVICES       : ${GPU_DEVICES:-<default>}"
echo "DEVICE            : $DEVICE"
echo

run_flux2_python "${ARGS[@]}"
