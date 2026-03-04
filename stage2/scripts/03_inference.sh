#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$STAGE2_DIR/.." && pwd)"
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

run_eval_python() {
  if has conda; then
    conda run -n eval --no-capture-output python "$@"
  elif [[ -x /usr/local/bin/eval ]]; then
    /usr/local/bin/eval python "$@"
  else
    echo "ERROR: conda env 'eval' (or /usr/local/bin/eval wrapper) not found." >&2
    exit 1
  fi
}

to_bool_01() {
  local raw="${1:-}"
  local normalized="${raw,,}"
  case "$normalized" in
    1|true|yes|y|on)
      echo "1"
      ;;
    0|false|no|n|off|"")
      echo "0"
      ;;
    *)
      echo "ERROR: invalid boolean value: '$raw' (expected true/false)." >&2
      exit 1
      ;;
  esac
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
    "eval": "CFG_EVAL",
    "evaluation_output_json": "CFG_EVALUATION_OUTPUT_JSON",
    "evaluation_device": "CFG_EVALUATION_DEVICE",
    "evaluation_gpu_devices": "CFG_EVALUATION_GPU_DEVICES",
    "evaluation_fr_resize": "CFG_EVALUATION_FR_RESIZE",
    "prompts_json": "CFG_PROMPTS_JSON",
    "prompt_name": "CFG_PROMPT_NAME",
    "default_prompt": "CFG_DEFAULT_PROMPT",
    "lora_weights_path": "CFG_LORA_WEIGHTS_PATH",
    "revision": "CFG_REVISION",
    "variant": "CFG_VARIANT",
    "mode": "CFG_MODE",
    "crop_mode": "CFG_CROP_MODE",
    "resolution": "CFG_RESOLUTION",
    "tile_size_px": "CFG_TILE_SIZE_PX",
    "tile_overlap_px": "CFG_TILE_OVERLAP_PX",
    "tile_batch_size": "CFG_TILE_BATCH_SIZE",
    "tile_sigma_ratio": "CFG_TILE_SIGMA_RATIO",
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
EVAL="${EVAL:-${CFG_EVAL:-0}}"
EVALUATION_OUTPUT_JSON="${EVALUATION_OUTPUT_JSON:-${CFG_EVALUATION_OUTPUT_JSON:-}}"
EVALUATION_DEVICE="${EVALUATION_DEVICE:-${CFG_EVALUATION_DEVICE:-auto}}"
EVALUATION_FR_RESIZE="${EVALUATION_FR_RESIZE:-${CFG_EVALUATION_FR_RESIZE:-to_ref}}"
PROMPTS_JSON="${PROMPTS_JSON:-${CFG_PROMPTS_JSON:-$STAGE2_DIR/prompts.json}}"
PROMPT_NAME="${PROMPT_NAME:-${CFG_PROMPT_NAME:-}}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-${CFG_DEFAULT_PROMPT:-}}"
LORA_WEIGHTS_PATH="${LORA_WEIGHTS_PATH:-${CFG_LORA_WEIGHTS_PATH:-}}"
REVISION="${REVISION:-${CFG_REVISION:-}}"
VARIANT="${VARIANT:-${CFG_VARIANT:-}}"

MODE="${MODE:-${CFG_MODE:-plain}}"
CROP_MODE="${CROP_MODE:-${CFG_CROP_MODE:-full}}"
RESOLUTION="${RESOLUTION:-${CFG_RESOLUTION:-512}}"
TILE_SIZE_PX="${TILE_SIZE_PX:-${CFG_TILE_SIZE_PX:-1024}}"
TILE_OVERLAP_PX="${TILE_OVERLAP_PX:-${CFG_TILE_OVERLAP_PX:-256}}"
TILE_BATCH_SIZE="${TILE_BATCH_SIZE:-${CFG_TILE_BATCH_SIZE:-4}}"
TILE_SIGMA_RATIO="${TILE_SIGMA_RATIO:-${CFG_TILE_SIGMA_RATIO:-0.15}}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-${CFG_GUIDANCE_SCALE:-4.0}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-${CFG_NUM_INFERENCE_STEPS:-50}}"
DTYPE="${DTYPE:-${CFG_DTYPE:-bf16}}"
DEVICE="${DEVICE:-${CFG_DEVICE:-cuda}}"
SEED="${SEED:-${CFG_SEED:-0}}"
CPU_OFFLOAD="${CPU_OFFLOAD:-${CFG_CPU_OFFLOAD:-1}}"
GPU_DEVICES="${GPU_DEVICES:-${CFG_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
EVALUATION_GPU_DEVICES="${EVALUATION_GPU_DEVICES:-${CFG_EVALUATION_GPU_DEVICES:-${GPU_DEVICES:-}}}"

if [[ -z "$INPUT_JSON" ]]; then
  echo "ERROR: set INPUT_JSON or provide it in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ "$MODE" == "full" || "$MODE" == "center_crop" || "$MODE" == "random_crop" ]]; then
  CROP_MODE="$MODE"
  MODE="plain"
fi

RUN_EVAL="$(to_bool_01 "$EVAL")"

INFER_OUTPUT_JSON="$OUTPUT_JSON"
if [[ -z "$INFER_OUTPUT_JSON" ]]; then
  INFER_OUTPUT_JSON="$OUTPUT_DIR/results.json"
fi

if [[ "$RUN_EVAL" == "1" && -z "$EVALUATION_OUTPUT_JSON" ]]; then
  if [[ "$INFER_OUTPUT_JSON" == *.* ]]; then
    EVALUATION_OUTPUT_JSON="${INFER_OUTPUT_JSON%.*}_evaluation.json"
  else
    EVALUATION_OUTPUT_JSON="${INFER_OUTPUT_JSON}.evaluation.json"
  fi
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
  --crop_mode "$CROP_MODE"
  --resolution "$RESOLUTION"
  --tile_size_px "$TILE_SIZE_PX"
  --tile_overlap_px "$TILE_OVERLAP_PX"
  --tile_batch_size "$TILE_BATCH_SIZE"
  --tile_sigma_ratio "$TILE_SIGMA_RATIO"
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

if [[ -n "$PROMPT_NAME" ]]; then
  ARGS+=(--prompts_json "$PROMPTS_JSON" --prompt_name "$PROMPT_NAME")
fi

if [[ -n "$REVISION" ]]; then
  ARGS+=(--revision "$REVISION")
fi

if [[ -n "$VARIANT" ]]; then
  ARGS+=(--variant "$VARIANT")
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
echo "OUTPUT_JSON       : $INFER_OUTPUT_JSON"
echo "EVAL              : $RUN_EVAL"
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "EVAL_OUTPUT_JSON  : $EVALUATION_OUTPUT_JSON"
fi
echo "PROMPTS_JSON      : $PROMPTS_JSON"
echo "PROMPT_NAME       : ${PROMPT_NAME:-<input_json_or_default_prompt>}"
echo "LORA_WEIGHTS_PATH : ${LORA_WEIGHTS_PATH:-<none>}"
echo "REVISION          : ${REVISION:-<none>}"
echo "VARIANT           : ${VARIANT:-<none>}"
echo "MODE              : $MODE"
echo "CROP_MODE         : $CROP_MODE"
echo "RESOLUTION        : $RESOLUTION"
echo "TILE_SIZE_PX      : $TILE_SIZE_PX"
echo "TILE_OVERLAP_PX   : $TILE_OVERLAP_PX"
echo "TILE_BATCH_SIZE   : $TILE_BATCH_SIZE"
echo "TILE_SIGMA_RATIO  : $TILE_SIGMA_RATIO"
echo "GPU_DEVICES       : ${GPU_DEVICES:-<default>}"
echo "DEVICE            : $DEVICE"
echo

run_flux2_python "${ARGS[@]}"

if [[ "$RUN_EVAL" == "1" ]]; then
  EVAL_SCRIPT="$PROJECT_DIR/eval/03_evaluation.py"
  if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "ERROR: evaluation script not found: $EVAL_SCRIPT" >&2
    exit 1
  fi

  EVAL_ARGS=(
    "$EVAL_SCRIPT"
    --input "$INFER_OUTPUT_JSON"
    --out "$EVALUATION_OUTPUT_JSON"
    --device "$EVALUATION_DEVICE"
    --fr_resize "$EVALUATION_FR_RESIZE"
  )

  if [[ -n "$EVALUATION_GPU_DEVICES" ]]; then
    EVAL_ARGS+=(--gpu_devices "$EVALUATION_GPU_DEVICES")
  fi

  echo "Running evaluation..."
  echo "INFER_OUTPUT_JSON : $INFER_OUTPUT_JSON"
  echo "EVAL_OUTPUT_JSON  : $EVALUATION_OUTPUT_JSON"
  echo
  run_eval_python "${EVAL_ARGS[@]}"
fi
