#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$STAGE2_DIR/.." && pwd)"
DEFAULT_OPTIONS_JSON="$STAGE2_DIR/options/base_inference.json"
DEFAULT_MODEL_PATH="$STAGE2_DIR/weights/flux2-klein-base-9b"
DEFAULT_OUTPUT_DIR="$STAGE2_DIR/outputs/base_inference"
DEFAULT_PROMPTS_JSON="$STAGE2_DIR/prompts.json"
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
    "prompts_json": "CFG_PROMPTS_JSON",
    "prompt_name": "CFG_PROMPT_NAME",
    "patch_size": "CFG_PATCH_SIZE",
    "guidance_scale": "CFG_GUIDANCE_SCALE",
    "num_inference_steps": "CFG_NUM_INFERENCE_STEPS",
    "dtype": "CFG_DTYPE",
    "device": "CFG_DEVICE",
    "seed": "CFG_SEED",
    "cpu_offload": "CFG_CPU_OFFLOAD",
    "gpu_devices": "CFG_GPU_DEVICES",
    "test_name": "CFG_TEST_NAME",
    "evaluation_output_dir": "CFG_EVALUATION_OUTPUT_DIR",
    "evaluation_device": "CFG_EVALUATION_DEVICE",
    "evaluation_gpu_devices": "CFG_EVALUATION_GPU_DEVICES",
    "evaluation_fr_resize": "CFG_EVALUATION_FR_RESIZE",
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

MODEL_PATH="${MODEL_PATH:-${CFG_MODEL_PATH:-$DEFAULT_MODEL_PATH}}"
INPUT_JSON="${INPUT_JSON:-${CFG_INPUT_JSON:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}}"
OUTPUT_JSON="${OUTPUT_JSON:-${CFG_OUTPUT_JSON:-}}"
PROMPTS_JSON="${PROMPTS_JSON:-${CFG_PROMPTS_JSON:-$DEFAULT_PROMPTS_JSON}}"
PROMPT_NAME="${PROMPT_NAME:-${CFG_PROMPT_NAME:-}}"
PATCH_SIZE="${PATCH_SIZE:-${CFG_PATCH_SIZE:-512}}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-${CFG_GUIDANCE_SCALE:-4.0}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-${CFG_NUM_INFERENCE_STEPS:-50}}"
DTYPE="${DTYPE:-${CFG_DTYPE:-bf16}}"
DEVICE="${DEVICE:-${CFG_DEVICE:-cuda}}"
SEED="${SEED:-${CFG_SEED:-0}}"
CPU_OFFLOAD="${CPU_OFFLOAD:-${CFG_CPU_OFFLOAD:-1}}"
GPU_DEVICES="${GPU_DEVICES:-${CFG_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
TEST_NAME="${TEST_NAME:-${CFG_TEST_NAME:-}}"
TEST_NAME="${TEST_NAME%.json}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-${CFG_EVALUATION_OUTPUT_DIR:-}}"
EVALUATION_DEVICE="${EVALUATION_DEVICE:-${CFG_EVALUATION_DEVICE:-auto}}"
EVALUATION_GPU_DEVICES="${EVALUATION_GPU_DEVICES:-${CFG_EVALUATION_GPU_DEVICES:-$GPU_DEVICES}}"
EVALUATION_FR_RESIZE="${EVALUATION_FR_RESIZE:-${CFG_EVALUATION_FR_RESIZE:-to_ref}}"

if [[ -n "$TEST_NAME" ]]; then
  OUTPUT_JSON="$OUTPUT_DIR/$TEST_NAME.json"
fi

RUN_EVAL=0
if [[ -n "$EVALUATION_OUTPUT_DIR" ]]; then
  RUN_EVAL=1
fi

if [[ -z "$INPUT_JSON" ]]; then
  echo "ERROR: set INPUT_JSON or provide it in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ -z "$PROMPT_NAME" ]]; then
  echo "ERROR: set PROMPT_NAME or provide it in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ -n "$TEST_NAME" && "$TEST_NAME" == */* ]]; then
  echo "ERROR: TEST_NAME must not contain '/'." >&2
  exit 1
fi

if [[ "$RUN_EVAL" == "1" && -z "$TEST_NAME" ]]; then
  echo "ERROR: set TEST_NAME when EVALUATION_OUTPUT_DIR is provided." >&2
  exit 1
fi

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

ARGS=(
  "$STAGE2_DIR/models/base.py"
  --model_path "$MODEL_PATH"
  --input_json "$INPUT_JSON"
  --output_dir "$OUTPUT_DIR"
  --prompts_json "$PROMPTS_JSON"
  --prompt_name "$PROMPT_NAME"
  --patch_size "$PATCH_SIZE"
  --guidance_scale "$GUIDANCE_SCALE"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ -n "$OUTPUT_JSON" ]]; then
  ARGS+=(--output_json "$OUTPUT_JSON")
fi

if [[ "$CPU_OFFLOAD" == "1" ]]; then
  ARGS+=(--cpu_offload)
fi

if [[ -n "$GPU_DEVICES" ]]; then
  ARGS+=(--gpu_devices "$GPU_DEVICES")
fi

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

echo "OPTIONS_JSON         : $OPTIONS_JSON"
echo "MODEL_PATH           : $MODEL_PATH"
echo "INPUT_JSON           : $INPUT_JSON"
echo "OUTPUT_DIR           : $OUTPUT_DIR"
echo "PROMPTS_JSON         : $PROMPTS_JSON"
echo "PROMPT_NAME          : $PROMPT_NAME"
echo "PATCH_SIZE           : $PATCH_SIZE"
echo "GPU_DEVICES          : ${GPU_DEVICES:-<default>}"
echo "DEVICE               : $DEVICE"
if [[ -n "$TEST_NAME" ]]; then
  echo "TEST_NAME            : $TEST_NAME"
fi
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "EVAL_OUTPUT_DIR      : $EVALUATION_OUTPUT_DIR"
fi
echo

run_flux2_python "${ARGS[@]}"

if [[ "$RUN_EVAL" == "1" ]]; then
  EVAL_SCRIPT="$PROJECT_DIR/eval/03_evaluation.py"
  if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "ERROR: evaluation script not found: $EVAL_SCRIPT" >&2
    exit 1
  fi

  mkdir -p "$EVALUATION_OUTPUT_DIR"
  EVALUATION_OUTPUT_JSON="$EVALUATION_OUTPUT_DIR/$TEST_NAME.json"

  EVAL_ARGS=(
    "$EVAL_SCRIPT"
    --input "$OUTPUT_JSON"
    --out "$EVALUATION_OUTPUT_JSON"
    --device "$EVALUATION_DEVICE"
    --fr_resize "$EVALUATION_FR_RESIZE"
  )

  if [[ -n "$EVALUATION_GPU_DEVICES" ]]; then
    EVAL_ARGS+=(--gpu_devices "$EVALUATION_GPU_DEVICES")
  fi

  echo "Running evaluation..."
  echo "INFER_OUTPUT_JSON    : $OUTPUT_JSON"
  echo "EVAL_OUTPUT_JSON     : $EVALUATION_OUTPUT_JSON"
  echo
  run_flux2_python "${EVAL_ARGS[@]}"
fi
