#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$STAGE2_DIR/.." && pwd)"
DEFAULT_OPTIONS_JSON="$STAGE2_DIR/options/inference/inference.json"
DEFAULT_PRETRAINED_MODEL_PATH="$STAGE2_DIR/weights/flux2-klein-base-9b"
DEFAULT_OUTPUT_DIR="$STAGE2_DIR/outputs/inference"
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

parse_model_names() {
  local raw="${1:-}"
  raw="${raw//,/ }"

  local token
  for token in $raw; do
    [[ -n "$token" ]] && printf "%s\n" "$token"
  done
}

calc_overlap_px_from_ratio() {
  local tile_size_px="$1"
  local overlap_ratio="$2"
  run_flux2_python - "$tile_size_px" "$overlap_ratio" <<'PY'
import sys

tile_size = int(sys.argv[1])
ratio = float(sys.argv[2])

if tile_size <= 0:
    raise SystemExit("ERROR: TILE_SIZE_PX must be > 0.")
if ratio < 0 or ratio >= 1:
    raise SystemExit("ERROR: TILE_OVERLAP_RATIO must satisfy 0 <= overlap_ratio < 1.")

overlap_px = int(round(tile_size * ratio))
if overlap_px >= tile_size:
    overlap_px = tile_size - 1
print(overlap_px)
PY
}

calc_overlap_ratio_from_px() {
  local tile_size_px="$1"
  local overlap_px="$2"
  run_flux2_python - "$tile_size_px" "$overlap_px" <<'PY'
import sys

tile_size = int(sys.argv[1])
overlap_px = int(sys.argv[2])

if tile_size <= 0:
    raise SystemExit("ERROR: TILE_SIZE_PX must be > 0.")
if overlap_px < 0 or overlap_px >= tile_size:
    raise SystemExit("ERROR: TILE_OVERLAP_PX must satisfy 0 <= overlap_px < tile_size_px.")

print(f"{overlap_px / tile_size:.6f}")
PY
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
    "pretrained_model_name_or_path": "CFG_PRETRAINED_MODEL_PATH",
    "base_model_path": "CFG_PRETRAINED_MODEL_PATH",
    "model_path": "CFG_MODEL_PATH",
    "model_name": "CFG_MODEL_NAME",
    "model_names": "CFG_MODEL_NAMES",
    "input_json": "CFG_INPUT_JSON",
    "output_dir": "CFG_OUTPUT_DIR",
    "output_path": "CFG_OUTPUT_PATH_DIR",
    "output_path_dir": "CFG_OUTPUT_PATH_DIR",
    "eval": "CFG_EVAL",
    "evaluation_output_dir": "CFG_EVALUATION_OUTPUT_DIR",
    "evaluation_device": "CFG_EVALUATION_DEVICE",
    "evaluation_gpu_devices": "CFG_EVALUATION_GPU_DEVICES",
    "evaluation_fr_resize": "CFG_EVALUATION_FR_RESIZE",
    "prompts_json": "CFG_PROMPTS_JSON",
    "prompt_name": "CFG_PROMPT_NAME",
    "default_prompt": "CFG_DEFAULT_PROMPT",
    "lora_weights": "CFG_LORA_WEIGHTS",
    "lora_weights_path": "CFG_LORA_WEIGHTS_PATH",
    "revision": "CFG_REVISION",
    "variant": "CFG_VARIANT",
    "mode": "CFG_MODE",
    "crop_mode": "CFG_CROP_MODE",
    "resolution": "CFG_RESOLUTION",
    "tile_size_px": "CFG_TILE_SIZE_PX",
    "tile_overlap_ratio": "CFG_TILE_OVERLAP_RATIO",
    "tile_overlap_px": "CFG_TILE_OVERLAP_PX",
    "tile_batch_size": "CFG_TILE_BATCH_SIZE",
    "tile_sigma_ratio": "CFG_TILE_SIGMA_RATIO",
    "canvas_padding_mode": "CFG_CANVAS_PADDING_MODE",
    "canvas_padding_position": "CFG_CANVAS_PADDING_POSITION",
    "canvas_padding_value": "CFG_CANVAS_PADDING_VALUE",
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
    elif isinstance(value, list):
        value = ",".join(str(v).strip() for v in value if str(v).strip())
    else:
        value = str(value)
    print(f"{shell_name}={shlex.quote(value)}")
PY
  )"
}

load_options_json "$OPTIONS_JSON"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${CFG_PRETRAINED_MODEL_PATH:-$DEFAULT_PRETRAINED_MODEL_PATH}}"
MODEL_PATH="${MODEL_PATH:-${CFG_MODEL_PATH:-}}"
MODEL_NAMES_RAW="${MODEL_NAMES:-${MODEL_NAME:-${CFG_MODEL_NAMES:-${CFG_MODEL_NAME:-}}}}"
INPUT_JSON="${INPUT_JSON:-${CFG_INPUT_JSON:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}}"
OUTPUT_PATH_DIR="${OUTPUT_PATH_DIR:-${CFG_OUTPUT_PATH_DIR:-}}"
EVAL="${EVAL:-${CFG_EVAL:-0}}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-${CFG_EVALUATION_OUTPUT_DIR:-}}"
EVALUATION_DEVICE="${EVALUATION_DEVICE:-${CFG_EVALUATION_DEVICE:-auto}}"
EVALUATION_FR_RESIZE="${EVALUATION_FR_RESIZE:-${CFG_EVALUATION_FR_RESIZE:-to_ref}}"
PROMPTS_JSON="${PROMPTS_JSON:-${CFG_PROMPTS_JSON:-$DEFAULT_PROMPTS_JSON}}"
PROMPT_NAME="${PROMPT_NAME:-${CFG_PROMPT_NAME:-}}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-${CFG_DEFAULT_PROMPT:-}}"
LORA_WEIGHTS="${LORA_WEIGHTS:-${CFG_LORA_WEIGHTS:-}}"
LORA_WEIGHTS_PATH_LEGACY="${LORA_WEIGHTS_PATH:-${CFG_LORA_WEIGHTS_PATH:-}}"
REVISION="${REVISION:-${CFG_REVISION:-}}"
VARIANT="${VARIANT:-${CFG_VARIANT:-}}"

MODE="${MODE:-${CFG_MODE:-plain}}"
CROP_MODE="${CROP_MODE:-${CFG_CROP_MODE:-full}}"
RESOLUTION="${RESOLUTION:-${CFG_RESOLUTION:-512}}"
TILE_SIZE_PX="${TILE_SIZE_PX:-${CFG_TILE_SIZE_PX:-1024}}"
TILE_OVERLAP_RATIO="${TILE_OVERLAP_RATIO:-${CFG_TILE_OVERLAP_RATIO:-}}"
TILE_OVERLAP_PX_LEGACY="${TILE_OVERLAP_PX:-${CFG_TILE_OVERLAP_PX:-}}"
TILE_BATCH_SIZE="${TILE_BATCH_SIZE:-${CFG_TILE_BATCH_SIZE:-4}}"
TILE_SIGMA_RATIO="${TILE_SIGMA_RATIO:-${CFG_TILE_SIGMA_RATIO:-0.15}}"
CANVAS_PADDING_MODE="${CANVAS_PADDING_MODE:-${CFG_CANVAS_PADDING_MODE:-none}}"
CANVAS_PADDING_POSITION="${CANVAS_PADDING_POSITION:-${CFG_CANVAS_PADDING_POSITION:-one_side}}"
CANVAS_PADDING_VALUE="${CANVAS_PADDING_VALUE:-${CFG_CANVAS_PADDING_VALUE:-0.0}}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-${CFG_GUIDANCE_SCALE:-4.0}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-${CFG_NUM_INFERENCE_STEPS:-50}}"
DTYPE="${DTYPE:-${CFG_DTYPE:-bf16}}"
DEVICE="${DEVICE:-${CFG_DEVICE:-cuda}}"
SEED="${SEED:-${CFG_SEED:-0}}"
CPU_OFFLOAD="${CPU_OFFLOAD:-${CFG_CPU_OFFLOAD:-1}}"
GPU_DEVICES="${GPU_DEVICES:-${CFG_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"

# backward compatibility: if only legacy evaluation_gpu_devices is set, reuse it as common gpu_devices
if [[ -z "$GPU_DEVICES" && -n "${CFG_EVALUATION_GPU_DEVICES:-}" ]]; then
  GPU_DEVICES="${CFG_EVALUATION_GPU_DEVICES}"
fi

if [[ -z "$INPUT_JSON" ]]; then
  echo "ERROR: set INPUT_JSON or provide it in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ -z "$MODEL_NAMES_RAW" ]]; then
  echo "ERROR: set MODEL_NAME/MODEL_NAMES or provide model_name/model_names in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: set MODEL_PATH (common root directory for model checkpoints)." >&2
  exit 1
fi

if [[ -z "$LORA_WEIGHTS" && -n "$LORA_WEIGHTS_PATH_LEGACY" ]]; then
  LORA_WEIGHTS="$LORA_WEIGHTS_PATH_LEGACY"
fi

if [[ -z "$LORA_WEIGHTS" ]]; then
  echo "ERROR: set LORA_WEIGHTS (common checkpoint path segment)." >&2
  exit 1
fi

if [[ "$MODE" == "full" || "$MODE" == "center_crop" || "$MODE" == "random_crop" ]]; then
  CROP_MODE="$MODE"
  MODE="plain"
fi

RUN_EVAL="$(to_bool_01 "$EVAL")"

if [[ -z "$OUTPUT_PATH_DIR" ]]; then
  OUTPUT_PATH_DIR="$OUTPUT_DIR/output_path"
elif [[ "$OUTPUT_PATH_DIR" != /* ]]; then
  OUTPUT_PATH_DIR="$OUTPUT_DIR/$OUTPUT_PATH_DIR"
fi

if [[ -z "$EVALUATION_OUTPUT_DIR" ]]; then
  EVALUATION_OUTPUT_DIR="$OUTPUT_DIR/eval"
elif [[ "$EVALUATION_OUTPUT_DIR" != /* ]]; then
  EVALUATION_OUTPUT_DIR="$OUTPUT_DIR/$EVALUATION_OUTPUT_DIR"
fi

if [[ -n "$TILE_OVERLAP_RATIO" ]]; then
  TILE_OVERLAP_PX="$(calc_overlap_px_from_ratio "$TILE_SIZE_PX" "$TILE_OVERLAP_RATIO")"
elif [[ -n "$TILE_OVERLAP_PX_LEGACY" ]]; then
  if [[ ! "$TILE_OVERLAP_PX_LEGACY" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TILE_OVERLAP_PX must be an integer: '$TILE_OVERLAP_PX_LEGACY'" >&2
    exit 1
  fi
  TILE_OVERLAP_PX="$TILE_OVERLAP_PX_LEGACY"
  TILE_OVERLAP_RATIO="$(calc_overlap_ratio_from_px "$TILE_SIZE_PX" "$TILE_OVERLAP_PX")"
else
  TILE_OVERLAP_RATIO="0.25"
  TILE_OVERLAP_PX="$(calc_overlap_px_from_ratio "$TILE_SIZE_PX" "$TILE_OVERLAP_RATIO")"
fi

mapfile -t MODEL_NAMES < <(parse_model_names "$MODEL_NAMES_RAW")
if [[ ${#MODEL_NAMES[@]} -eq 0 ]]; then
  echo "ERROR: no valid model names found in '$MODEL_NAMES_RAW'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/output" "$OUTPUT_PATH_DIR"
if [[ "$RUN_EVAL" == "1" ]]; then
  mkdir -p "$EVALUATION_OUTPUT_DIR"
fi

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

EVAL_SCRIPT="$PROJECT_DIR/eval/03_evaluation.py"
if [[ "$RUN_EVAL" == "1" && ! -f "$EVAL_SCRIPT" ]]; then
  echo "ERROR: evaluation script not found: $EVAL_SCRIPT" >&2
  exit 1
fi

echo "OPTIONS_JSON           : $OPTIONS_JSON"
echo "PRETRAINED_MODEL_PATH  : $PRETRAINED_MODEL_PATH"
echo "MODEL_PATH             : $MODEL_PATH"
echo "MODEL_NAMES            : ${MODEL_NAMES[*]}"
echo "LORA_WEIGHTS           : $LORA_WEIGHTS"
echo "INPUT_JSON             : $INPUT_JSON"
echo "OUTPUT_DIR             : $OUTPUT_DIR"
echo "OUTPUT_PATH_DIR        : $OUTPUT_PATH_DIR"
echo "EVAL                   : $RUN_EVAL"
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "EVAL_OUTPUT_DIR        : $EVALUATION_OUTPUT_DIR"
fi
echo "PROMPTS_JSON           : $PROMPTS_JSON"
echo "PROMPT_NAME            : ${PROMPT_NAME:-<input_json_or_default_prompt>}"
echo "REVISION               : ${REVISION:-<none>}"
echo "VARIANT                : ${VARIANT:-<none>}"
echo "MODE                   : $MODE"
echo "CROP_MODE              : $CROP_MODE"
echo "RESOLUTION             : $RESOLUTION"
echo "TILE_SIZE_PX           : $TILE_SIZE_PX"
echo "TILE_OVERLAP_RATIO     : $TILE_OVERLAP_RATIO"
echo "TILE_OVERLAP_PX        : $TILE_OVERLAP_PX"
echo "TILE_BATCH_SIZE        : $TILE_BATCH_SIZE"
echo "TILE_SIGMA_RATIO       : $TILE_SIGMA_RATIO"
echo "CANVAS_PADDING_MODE    : $CANVAS_PADDING_MODE"
echo "CANVAS_PADDING_POS     : $CANVAS_PADDING_POSITION"
echo "CANVAS_PADDING_VALUE   : $CANVAS_PADDING_VALUE"
echo "GPU_DEVICES            : ${GPU_DEVICES:-<default>}"
echo "DEVICE                 : $DEVICE"
echo

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  MODEL_LORA_WEIGHTS_PATH="$LORA_WEIGHTS"
  if [[ "$MODEL_LORA_WEIGHTS_PATH" != /* ]]; then
    MODEL_LORA_WEIGHTS_PATH="${MODEL_PATH%/}/${MODEL_NAME}/${MODEL_LORA_WEIGHTS_PATH#/}"
  fi

  MODEL_IMAGE_OUTPUT_DIR="$OUTPUT_DIR/output/$MODEL_NAME"
  MODEL_OUTPUT_JSON="$OUTPUT_PATH_DIR/$MODEL_NAME.json"
  MODEL_EVAL_JSON="$EVALUATION_OUTPUT_DIR/$MODEL_NAME.json"

  ARGS=(
    "$STAGE2_DIR/models/lora_inference.py"
    --pretrained_model_name_or_path "$PRETRAINED_MODEL_PATH"
    --input_json "$INPUT_JSON"
    --output_dir "$MODEL_IMAGE_OUTPUT_DIR"
    --output_json "$MODEL_OUTPUT_JSON"
    --mode "$MODE"
    --crop_mode "$CROP_MODE"
    --resolution "$RESOLUTION"
    --tile_size_px "$TILE_SIZE_PX"
    --tile_overlap_px "$TILE_OVERLAP_PX"
    --tile_batch_size "$TILE_BATCH_SIZE"
    --tile_sigma_ratio "$TILE_SIGMA_RATIO"
    --canvas_padding_mode "$CANVAS_PADDING_MODE"
    --canvas_padding_position "$CANVAS_PADDING_POSITION"
    --canvas_padding_value "$CANVAS_PADDING_VALUE"
    --guidance_scale "$GUIDANCE_SCALE"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --dtype "$DTYPE"
    --device "$DEVICE"
    --seed "$SEED"
  )

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

  ARGS+=(--lora_weights_path "$MODEL_LORA_WEIGHTS_PATH")

  if [[ "$CPU_OFFLOAD" == "1" ]]; then
    ARGS+=(--cpu_offload)
  fi

  if [[ $# -gt 0 ]]; then
    ARGS+=("$@")
  fi

  echo "Running inference for model: $MODEL_NAME"
  echo "MODEL_CHECKPOINT       : $MODEL_LORA_WEIGHTS_PATH"
  echo "MODEL_IMAGE_OUTPUT_DIR : $MODEL_IMAGE_OUTPUT_DIR"
  echo "MODEL_OUTPUT_JSON      : $MODEL_OUTPUT_JSON"
  echo
  run_flux2_python "${ARGS[@]}"

  if [[ "$RUN_EVAL" == "1" ]]; then
    EVAL_ARGS=(
      "$EVAL_SCRIPT"
      --input "$MODEL_OUTPUT_JSON"
      --out "$MODEL_EVAL_JSON"
      --device "$EVALUATION_DEVICE"
      --fr_resize "$EVALUATION_FR_RESIZE"
    )

    if [[ -n "$GPU_DEVICES" ]]; then
      EVAL_ARGS+=(--gpu_devices "$GPU_DEVICES")
    fi

    echo "Running evaluation for model: $MODEL_NAME"
    echo "INFER_OUTPUT_JSON      : $MODEL_OUTPUT_JSON"
    echo "EVAL_OUTPUT_JSON       : $MODEL_EVAL_JSON"
    echo
    run_eval_python "${EVAL_ARGS[@]}"
  fi
done
