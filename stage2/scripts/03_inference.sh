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

CLI_EXTRA_ARGS=("$@")

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

parse_lora_weights() {
  local raw="${1:-}"
  raw="${raw//,/ }"

  local token
  for token in $raw; do
    [[ -n "$token" ]] && printf "%s\n" "$token"
  done
}

run_inference_job() {
  local gpu_device="$1"
  local run_name="$2"
  local model_checkpoint_path="$3"
  local sem_checkpoint_path="$4"
  local model_image_output_dir="$5"
  local model_output_json="$6"
  local model_eval_json="$7"

  local -a args=(
    "$STAGE2_DIR/models/lora_inference.py"
    --pretrained_model_name_or_path "$PRETRAINED_MODEL_PATH"
    --input_json "$INPUT_JSON"
    --output_dir "$model_image_output_dir"
    --output_json "$model_output_json"
    --mode "$MODE"
    --crop_mode "$CROP_MODE"
    --resolution "$RESOLUTION"
    --tile_size_px "$TILE_SIZE_PX"
    --tile_overlap_px "$TILE_OVERLAP_PX"
    --tile_batch_size "$TILE_BATCH_SIZE"
    --sample_batch_size "$SAMPLE_BATCH_SIZE"
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

  if [[ -n "$HR_DIR" ]]; then
    args+=(--hr_dir "$HR_DIR/$run_name")
  fi

  if [[ -n "$DEFAULT_PROMPT" ]]; then
    args+=(--default_prompt "$DEFAULT_PROMPT")
  fi

  if [[ -n "$PROMPT_NAME" ]]; then
    args+=(--prompts_json "$PROMPTS_JSON" --prompt_name "$PROMPT_NAME")
  fi

  if [[ -n "$REVISION" ]]; then
    args+=(--revision "$REVISION")
  fi

  if [[ -n "$VARIANT" ]]; then
    args+=(--variant "$VARIANT")
  fi

  if [[ -n "$PIX_LORA_WEIGHTS_PATH" ]]; then
    args+=(
      --pix_lora_weights_path "$PIX_LORA_WEIGHTS_PATH"
      --sem_lora_weights_path "$sem_checkpoint_path"
      --pix_adapter_name "$PIX_ADAPTER_NAME"
      --sem_adapter_name "$SEM_ADAPTER_NAME"
      --pix_adapter_scale "$PIX_ADAPTER_SCALE"
      --sem_adapter_scale "$SEM_ADAPTER_SCALE"
    )
  else
    args+=(--lora_weights_path "$model_checkpoint_path")
  fi

  if [[ "$CPU_OFFLOAD" == "1" ]]; then
    args+=(--cpu_offload)
  fi

  if [[ $# -gt 7 ]]; then
    args+=("${@:8}")
  fi

  echo "Running inference for: $run_name on GPU $gpu_device"
  if [[ -n "$PIX_LORA_WEIGHTS_PATH" ]]; then
    echo "PIX_CHECKPOINT         : $PIX_LORA_WEIGHTS_PATH"
    echo "SEM_CHECKPOINT         : $sem_checkpoint_path"
  else
    echo "MODEL_CHECKPOINT       : $model_checkpoint_path"
  fi
  echo "MODEL_IMAGE_OUTPUT_DIR : $model_image_output_dir"
  echo "MODEL_OUTPUT_JSON      : $model_output_json"
  echo
  CUDA_VISIBLE_DEVICES="$gpu_device" run_flux2_python "${args[@]}"

  if [[ "$RUN_EVAL" == "1" ]]; then
    local -a eval_args=(
      "$EVAL_SCRIPT"
      --input "$model_output_json"
      --out "$model_eval_json"
      --device "$EVALUATION_DEVICE"
      --fr_resize "$EVALUATION_FR_RESIZE"
      --gpu_devices "$gpu_device"
    )

    echo "Running evaluation for: $run_name on GPU $gpu_device"
    echo "INFER_OUTPUT_JSON      : $model_output_json"
    echo "EVAL_OUTPUT_JSON       : $model_eval_json"
    echo
    CUDA_VISIBLE_DEVICES="$gpu_device" run_eval_python "${eval_args[@]}"
  fi
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
    "pix_lora_weights_path": "CFG_PIX_LORA_WEIGHTS_PATH",
    "sem_lora_weights_path": "CFG_SEM_LORA_WEIGHTS_PATH",
    "pix_adapter_name": "CFG_PIX_ADAPTER_NAME",
    "sem_adapter_name": "CFG_SEM_ADAPTER_NAME",
    "pix_adapter_scale": "CFG_PIX_ADAPTER_SCALE",
    "sem_adapter_scale": "CFG_SEM_ADAPTER_SCALE",
    "input_json": "CFG_INPUT_JSON",
    "output_dir": "CFG_OUTPUT_DIR",
    "hr_dir": "CFG_HR_DIR",
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
    "sample_batch_size": "CFG_SAMPLE_BATCH_SIZE",
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
PIX_LORA_WEIGHTS_PATH="${PIX_LORA_WEIGHTS_PATH:-${CFG_PIX_LORA_WEIGHTS_PATH:-}}"
SEM_LORA_WEIGHTS_PATH="${SEM_LORA_WEIGHTS_PATH:-${CFG_SEM_LORA_WEIGHTS_PATH:-}}"
PIX_ADAPTER_NAME="${PIX_ADAPTER_NAME:-${CFG_PIX_ADAPTER_NAME:-pix}}"
SEM_ADAPTER_NAME="${SEM_ADAPTER_NAME:-${CFG_SEM_ADAPTER_NAME:-sem}}"
PIX_ADAPTER_SCALE="${PIX_ADAPTER_SCALE:-${CFG_PIX_ADAPTER_SCALE:-1.0}}"
SEM_ADAPTER_SCALE="${SEM_ADAPTER_SCALE:-${CFG_SEM_ADAPTER_SCALE:-1.0}}"
INPUT_JSON="${INPUT_JSON:-${CFG_INPUT_JSON:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}}"
HR_DIR="${HR_DIR:-${CFG_HR_DIR:-}}"
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
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-${CFG_SAMPLE_BATCH_SIZE:-1}}"
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

if [[ -z "$LORA_WEIGHTS" && -n "$LORA_WEIGHTS_PATH_LEGACY" ]]; then
  LORA_WEIGHTS="$LORA_WEIGHTS_PATH_LEGACY"
fi

if [[ -z "$PIX_LORA_WEIGHTS_PATH" ]]; then
  if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: set MODEL_PATH (common root directory for model checkpoints)." >&2
    exit 1
  fi
  if [[ -z "$LORA_WEIGHTS" ]]; then
    echo "ERROR: set LORA_WEIGHTS (common checkpoint path segment)." >&2
    exit 1
  fi
else
  if [[ -z "$SEM_LORA_WEIGHTS_PATH" ]]; then
    if [[ -z "$MODEL_PATH" ]]; then
      echo "ERROR: dual-LoRA mode requires MODEL_PATH as the sem checkpoint root when SEM_LORA_WEIGHTS_PATH is unset." >&2
      exit 1
    fi
    if [[ -z "$LORA_WEIGHTS" ]]; then
      echo "ERROR: dual-LoRA mode requires LORA_WEIGHTS as the sem checkpoint path segment when SEM_LORA_WEIGHTS_PATH is unset." >&2
      exit 1
    fi
  fi
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

mapfile -t LORA_WEIGHT_ITEMS < <(parse_lora_weights "$LORA_WEIGHTS")
if [[ ${#LORA_WEIGHT_ITEMS[@]} -eq 0 ]]; then
  echo "ERROR: no valid lora_weights found in '$LORA_WEIGHTS'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/output" "$OUTPUT_PATH_DIR"
if [[ "$RUN_EVAL" == "1" ]]; then
  mkdir -p "$EVALUATION_OUTPUT_DIR"
fi

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

mapfile -t GPU_DEVICE_LIST < <(parse_model_names "$GPU_DEVICES")
if [[ ${#GPU_DEVICE_LIST[@]} -eq 0 ]]; then
  GPU_DEVICE_LIST=("0")
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
echo "LORA_WEIGHTS           : ${LORA_WEIGHT_ITEMS[*]}"
echo "PIX_LORA_WEIGHTS_PATH  : ${PIX_LORA_WEIGHTS_PATH:-<none>}"
echo "SEM_LORA_WEIGHTS_PATH  : ${SEM_LORA_WEIGHTS_PATH:-<per-model>}"
echo "PIX_ADAPTER_NAME       : $PIX_ADAPTER_NAME"
echo "SEM_ADAPTER_NAME       : $SEM_ADAPTER_NAME"
echo "PIX_ADAPTER_SCALE      : $PIX_ADAPTER_SCALE"
echo "SEM_ADAPTER_SCALE      : $SEM_ADAPTER_SCALE"
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
echo "SAMPLE_BATCH_SIZE      : $SAMPLE_BATCH_SIZE"
echo "TILE_SIGMA_RATIO       : $TILE_SIGMA_RATIO"
echo "CANVAS_PADDING_MODE    : $CANVAS_PADDING_MODE"
echo "CANVAS_PADDING_POS     : $CANVAS_PADDING_POSITION"
echo "CANVAS_PADDING_VALUE   : $CANVAS_PADDING_VALUE"
echo "GPU_DEVICES            : ${GPU_DEVICES:-<default>}"
echo "GPU_WORKERS            : ${#GPU_DEVICE_LIST[@]}"
echo "DEVICE                 : $DEVICE"
echo

declare -a SLOT_PIDS
declare -a SLOT_GPUS
declare -a SLOT_NAMES

for ((i=0; i<${#GPU_DEVICE_LIST[@]}; i++)); do
  SLOT_PIDS[i]=""
  SLOT_GPUS[i]="${GPU_DEVICE_LIST[i]}"
  SLOT_NAMES[i]=""
done

wait_for_any_slot() {
  while true; do
    for ((slot_idx=0; slot_idx<${#SLOT_PIDS[@]}; slot_idx++)); do
      local pid="${SLOT_PIDS[slot_idx]}"
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        local finished_name="${SLOT_NAMES[slot_idx]}"
        if wait "$pid"; then
          echo "Finished job: $finished_name on GPU ${SLOT_GPUS[slot_idx]}"
        else
          echo "ERROR: job failed: $finished_name on GPU ${SLOT_GPUS[slot_idx]}" >&2
          exit 1
        fi
        SLOT_PIDS[slot_idx]=""
        SLOT_NAMES[slot_idx]=""
        return 0
      fi
    done
    sleep 1
  done
}

launch_on_slot() {
  local slot_idx="$1"
  local gpu_device="$2"
  local run_name="$3"
  local model_checkpoint_path="$4"
  local sem_checkpoint_path="$5"
  local model_image_output_dir="$6"
  local model_output_json="$7"
  local model_eval_json="$8"

  (
    run_inference_job \
      "$gpu_device" \
      "$run_name" \
      "$model_checkpoint_path" \
      "$sem_checkpoint_path" \
      "$model_image_output_dir" \
      "$model_output_json" \
      "$model_eval_json" \
      "${CLI_EXTRA_ARGS[@]}"
  ) &
  SLOT_PIDS[slot_idx]="$!"
  SLOT_NAMES[slot_idx]="$run_name"
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for LORA_WEIGHT_ITEM in "${LORA_WEIGHT_ITEMS[@]}"; do
    MODEL_LORA_WEIGHTS_PATH="$LORA_WEIGHT_ITEM"
    MODEL_SEM_LORA_WEIGHTS_PATH="$SEM_LORA_WEIGHTS_PATH"
    RUN_NAME="${MODEL_NAME}_${LORA_WEIGHT_ITEM}"
    if [[ -z "$PIX_LORA_WEIGHTS_PATH" ]]; then
      if [[ "$MODEL_LORA_WEIGHTS_PATH" != /* ]]; then
        MODEL_LORA_WEIGHTS_PATH="${MODEL_PATH%/}/${MODEL_NAME}/${MODEL_LORA_WEIGHTS_PATH#/}"
      fi
    else
      if [[ -z "$MODEL_SEM_LORA_WEIGHTS_PATH" ]]; then
        MODEL_SEM_LORA_WEIGHTS_PATH="$LORA_WEIGHT_ITEM"
        if [[ "$MODEL_SEM_LORA_WEIGHTS_PATH" != /* ]]; then
          MODEL_SEM_LORA_WEIGHTS_PATH="${MODEL_PATH%/}/${MODEL_NAME}/${MODEL_SEM_LORA_WEIGHTS_PATH#/}"
        fi
      elif [[ "$MODEL_SEM_LORA_WEIGHTS_PATH" != /* ]]; then
        MODEL_SEM_LORA_WEIGHTS_PATH="${MODEL_PATH%/}/${MODEL_NAME}/${MODEL_SEM_LORA_WEIGHTS_PATH#/}"
      fi
    fi

    MODEL_IMAGE_OUTPUT_DIR="$OUTPUT_DIR/output/$RUN_NAME"
    MODEL_OUTPUT_JSON="$OUTPUT_PATH_DIR/$RUN_NAME.json"
    MODEL_EVAL_JSON="$EVALUATION_OUTPUT_DIR/$RUN_NAME.json"

    while true; do
      free_slot_idx=""
      for ((slot_idx=0; slot_idx<${#SLOT_PIDS[@]}; slot_idx++)); do
        if [[ -z "${SLOT_PIDS[slot_idx]}" ]]; then
          free_slot_idx="$slot_idx"
          break
        fi
      done

      if [[ -n "$free_slot_idx" ]]; then
        launch_on_slot \
          "$free_slot_idx" \
          "${SLOT_GPUS[free_slot_idx]}" \
          "$RUN_NAME" \
          "$MODEL_LORA_WEIGHTS_PATH" \
          "$MODEL_SEM_LORA_WEIGHTS_PATH" \
          "$MODEL_IMAGE_OUTPUT_DIR" \
          "$MODEL_OUTPUT_JSON" \
          "$MODEL_EVAL_JSON"
        break
      fi

      wait_for_any_slot
    done
  done
done

for ((slot_idx=0; slot_idx<${#SLOT_PIDS[@]}; slot_idx++)); do
  if [[ -n "${SLOT_PIDS[slot_idx]}" ]]; then
    if wait "${SLOT_PIDS[slot_idx]}"; then
      echo "Finished job: ${SLOT_NAMES[slot_idx]} on GPU ${SLOT_GPUS[slot_idx]}"
    else
      echo "ERROR: job failed: ${SLOT_NAMES[slot_idx]} on GPU ${SLOT_GPUS[slot_idx]}" >&2
      exit 1
    fi
  fi
done
