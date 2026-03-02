#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$STAGE2_DIR/.." && pwd)"
DEFAULT_OPTIONS_JSON="$STAGE2_DIR/options/train.json"
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

def emit_shell_var(name, value):
    print(f"{name}={shlex.quote(value)}")

def emit_shell_array(name, values):
    print(f"{name}=(")
    for value in values:
        print(f"  {shlex.quote(value)}")
    print(")")

mapping = {
    "model_path": "CFG_MODEL_PATH",
    "pretrained_model_name_or_path": "CFG_MODEL_PATH",
    "revision": "CFG_REVISION",
    "variant": "CFG_VARIANT",
    "train_data_json": "CFG_TRAIN_DATA_JSON",
    "output_dir": "CFG_OUTPUT_DIR",
    "instance_prompt": "CFG_INSTANCE_PROMPT",
    "lora_weights_path": "CFG_LORA_WEIGHTS_PATH",
    "validation_prompt": "CFG_VALIDATION_PROMPT",
    "validation_image": "CFG_VALIDATION_IMAGE",
    "resolution": "CFG_RESOLUTION",
    "train_batch_size": "CFG_TRAIN_BATCH_SIZE",
    "num_train_epochs": "CFG_NUM_TRAIN_EPOCHS",
    "learning_rate": "CFG_LEARNING_RATE",
    "mixed_precision": "CFG_MIXED_PRECISION",
    "report_to": "CFG_REPORT_TO",
    "seed": "CFG_SEED",
    "repeats": "CFG_REPEATS",
    "checkpointing_steps": "CFG_CHECKPOINTING_STEPS",
    "skip_final_inference": "CFG_SKIP_FINAL_INFERENCE",
    "random_flip": "CFG_RANDOM_FLIP",
    "center_crop": "CFG_CENTER_CROP",
    "offload": "CFG_OFFLOAD",
    "resume_from_checkpoint": "CFG_RESUME_FROM_CHECKPOINT",
    "gpu_devices": "CFG_GPU_DEVICES",
    "test_data_json": "CFG_TEST_DATA_JSON",
    "test_name": "CFG_TEST_NAME",
    "inference_output_dir": "CFG_INFERENCE_OUTPUT_DIR",
    "evaluation_output_dir": "CFG_EVALUATION_OUTPUT_DIR",
    "inference_mode": "CFG_INFERENCE_MODE",
    "inference_resolution": "CFG_INFERENCE_RESOLUTION",
    "inference_guidance_scale": "CFG_INFERENCE_GUIDANCE_SCALE",
    "inference_num_inference_steps": "CFG_INFERENCE_NUM_INFERENCE_STEPS",
    "inference_dtype": "CFG_INFERENCE_DTYPE",
    "inference_device": "CFG_INFERENCE_DEVICE",
    "inference_seed": "CFG_INFERENCE_SEED",
    "inference_cpu_offload": "CFG_INFERENCE_CPU_OFFLOAD",
    "evaluation_device": "CFG_EVALUATION_DEVICE",
    "evaluation_gpu_devices": "CFG_EVALUATION_GPU_DEVICES",
    "evaluation_fr_resize": "CFG_EVALUATION_FR_RESIZE",
}

store_true_keys = {
    "do_fp8_training",
    "skip_final_inference",
    "center_crop",
    "random_flip",
    "gradient_checkpointing",
    "scale_lr",
    "use_8bit_adam",
    "push_to_hub",
    "allow_tf32",
    "cache_latents",
    "upcast_before_saving",
    "offload",
    "enable_npu_flash_attention",
    "fsdp_text_encoder",
}

bool_value_keys = {
    "prodigy_decouple",
    "prodigy_use_bias_correction",
    "prodigy_safeguard_warmup",
}

shell_values = {}
source_keys = {}
extra_args = []

for json_key, value in data.items():
    if value is None:
        continue

    if json_key in mapping:
        shell_name = mapping[json_key]
        if shell_name in source_keys:
            other_key = source_keys[shell_name]
            raise SystemExit(
                f"ERROR: options JSON defines both '{other_key}' and '{json_key}', which map to the same setting."
            )

        if isinstance(value, bool):
            shell_values[shell_name] = "1" if value else "0"
        elif isinstance(value, (list, dict)):
            raise SystemExit(f"ERROR: option '{json_key}' must be a scalar value.")
        else:
            shell_values[shell_name] = str(value)

        source_keys[shell_name] = json_key
        continue

    flag = f"--{json_key}"

    if isinstance(value, bool):
        if json_key in store_true_keys:
            if value:
                extra_args.append(flag)
        elif json_key in bool_value_keys:
            extra_args.extend([flag, "1" if value else ""])
        else:
            raise SystemExit(
                f"ERROR: boolean option '{json_key}' is ambiguous in options JSON. "
                "Pass it on the command line or add it to 02_train.sh."
            )
        continue

    if isinstance(value, dict):
        raise SystemExit(f"ERROR: option '{json_key}' must be a scalar value, not an object.")

    if isinstance(value, list):
        if not value:
            continue
        for item in value:
            if item is None or isinstance(item, (list, dict)):
                raise SystemExit(f"ERROR: option '{json_key}' list items must be scalar values.")
            extra_args.extend([flag, str(item)])
        continue

    if isinstance(value, str) and value == "":
        continue

    extra_args.extend([flag, str(value)])

for shell_name, value in shell_values.items():
    emit_shell_var(shell_name, value)

emit_shell_array("CFG_JSON_EXTRA_ARGS", extra_args)
PY
  )"
}

CFG_JSON_EXTRA_ARGS=()
load_options_json "$OPTIONS_JSON"

MODEL_PATH="${MODEL_PATH:-${CFG_MODEL_PATH:-$STAGE2_DIR/weights/flux2-klein-base-9b}}"
REVISION="${REVISION:-${CFG_REVISION:-}}"
VARIANT="${VARIANT:-${CFG_VARIANT:-}}"
TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-${CFG_TRAIN_DATA_JSON:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUTPUT_DIR:-$STAGE2_DIR/outputs/lora_train}}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-${CFG_INSTANCE_PROMPT:-Enhance the perceptual quality of the condition image while preserving the original content, structure, and colors.}}"
LORA_WEIGHTS_PATH="${LORA_WEIGHTS_PATH:-${CFG_LORA_WEIGHTS_PATH:-}}"
VALIDATION_PROMPT="${VALIDATION_PROMPT:-${CFG_VALIDATION_PROMPT:-}}"
VALIDATION_IMAGE="${VALIDATION_IMAGE:-${CFG_VALIDATION_IMAGE:-}}"

RESOLUTION="${RESOLUTION:-${CFG_RESOLUTION:-512}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${CFG_TRAIN_BATCH_SIZE:-1}}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-${CFG_NUM_TRAIN_EPOCHS:-1}}"
LEARNING_RATE="${LEARNING_RATE:-${CFG_LEARNING_RATE:-1e-4}}"
MIXED_PRECISION="${MIXED_PRECISION:-${CFG_MIXED_PRECISION:-bf16}}"
REPORT_TO="${REPORT_TO:-${CFG_REPORT_TO:-tensorboard}}"
SEED="${SEED:-${CFG_SEED:-0}}"
REPEATS="${REPEATS:-${CFG_REPEATS:-1}}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-${CFG_CHECKPOINTING_STEPS:-500}}"
SKIP_FINAL_INFERENCE="${SKIP_FINAL_INFERENCE:-${CFG_SKIP_FINAL_INFERENCE:-1}}"
RANDOM_FLIP="${RANDOM_FLIP:-${CFG_RANDOM_FLIP:-0}}"
CENTER_CROP="${CENTER_CROP:-${CFG_CENTER_CROP:-0}}"
OFFLOAD="${OFFLOAD:-${CFG_OFFLOAD:-1}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${CFG_RESUME_FROM_CHECKPOINT:-}}"
GPU_DEVICES="${GPU_DEVICES:-${CFG_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"

TEST_DATA_JSON="${TEST_DATA_JSON:-${CFG_TEST_DATA_JSON:-}}"
TEST_NAME="${TEST_NAME:-${CFG_TEST_NAME:-}}"
TEST_NAME="${TEST_NAME%.json}"
INFERENCE_OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:-${CFG_INFERENCE_OUTPUT_DIR:-}}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-${CFG_EVALUATION_OUTPUT_DIR:-}}"

INFERENCE_MODE="${INFERENCE_MODE:-${CFG_INFERENCE_MODE:-full}}"
INFERENCE_RESOLUTION="${INFERENCE_RESOLUTION:-${CFG_INFERENCE_RESOLUTION:-$RESOLUTION}}"
INFERENCE_GUIDANCE_SCALE="${INFERENCE_GUIDANCE_SCALE:-${CFG_INFERENCE_GUIDANCE_SCALE:-4.0}}"
INFERENCE_NUM_INFERENCE_STEPS="${INFERENCE_NUM_INFERENCE_STEPS:-${CFG_INFERENCE_NUM_INFERENCE_STEPS:-50}}"
if [[ "$MIXED_PRECISION" == "no" ]]; then
  DEFAULT_INFERENCE_DTYPE="fp32"
else
  DEFAULT_INFERENCE_DTYPE="$MIXED_PRECISION"
fi
INFERENCE_DTYPE="${INFERENCE_DTYPE:-${CFG_INFERENCE_DTYPE:-$DEFAULT_INFERENCE_DTYPE}}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-${CFG_INFERENCE_DEVICE:-cuda}}"
INFERENCE_SEED="${INFERENCE_SEED:-${CFG_INFERENCE_SEED:-$SEED}}"
INFERENCE_CPU_OFFLOAD="${INFERENCE_CPU_OFFLOAD:-${CFG_INFERENCE_CPU_OFFLOAD:-1}}"
EVALUATION_DEVICE="${EVALUATION_DEVICE:-${CFG_EVALUATION_DEVICE:-auto}}"
EVALUATION_GPU_DEVICES="${EVALUATION_GPU_DEVICES:-${CFG_EVALUATION_GPU_DEVICES:-$GPU_DEVICES}}"
EVALUATION_FR_RESIZE="${EVALUATION_FR_RESIZE:-${CFG_EVALUATION_FR_RESIZE:-to_ref}}"

RUN_POST_EVAL=0
if [[ -n "$TEST_DATA_JSON" || -n "$TEST_NAME" || -n "$INFERENCE_OUTPUT_DIR" || -n "$EVALUATION_OUTPUT_DIR" ]]; then
  RUN_POST_EVAL=1
fi

if [[ -z "$TRAIN_DATA_JSON" ]]; then
  echo "ERROR: set TRAIN_DATA_JSON or provide it in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ -n "$VALIDATION_PROMPT" && -z "$VALIDATION_IMAGE" ]]; then
  echo "ERROR: VALIDATION_IMAGE is required when VALIDATION_PROMPT is set." >&2
  exit 1
fi

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

if [[ "$RUN_POST_EVAL" == "1" ]]; then
  if [[ -z "$TEST_DATA_JSON" || -z "$TEST_NAME" || -z "$INFERENCE_OUTPUT_DIR" || -z "$EVALUATION_OUTPUT_DIR" ]]; then
    echo "ERROR: to run post-train inference/evaluation, set TEST_DATA_JSON, TEST_NAME, INFERENCE_OUTPUT_DIR, and EVALUATION_OUTPUT_DIR." >&2
    exit 1
  fi

  if [[ ! -f "$TEST_DATA_JSON" ]]; then
    echo "ERROR: TEST_DATA_JSON not found: $TEST_DATA_JSON" >&2
    exit 1
  fi

  if [[ "$TEST_NAME" == */* ]]; then
    echo "ERROR: TEST_NAME must not contain '/'." >&2
    exit 1
  fi
fi

ARGS=(
  "$STAGE2_DIR/models/lora.py"
  --pretrained_model_name_or_path "$MODEL_PATH"
  --train_data_json "$TRAIN_DATA_JSON"
  --output_dir "$OUTPUT_DIR"
  --instance_prompt "$INSTANCE_PROMPT"
  --resolution "$RESOLUTION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --mixed_precision "$MIXED_PRECISION"
  --report_to "$REPORT_TO"
  --seed "$SEED"
  --repeats "$REPEATS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
)

if [[ -n "$REVISION" ]]; then
  ARGS+=(--revision "$REVISION")
fi

if [[ -n "$VARIANT" ]]; then
  ARGS+=(--variant "$VARIANT")
fi

if [[ -n "$LORA_WEIGHTS_PATH" ]]; then
  ARGS+=(--lora_weights_path "$LORA_WEIGHTS_PATH")
fi

if [[ -n "$VALIDATION_PROMPT" ]]; then
  ARGS+=(--validation_prompt "$VALIDATION_PROMPT" --validation_image "$VALIDATION_IMAGE")
fi

if [[ "$SKIP_FINAL_INFERENCE" == "1" ]]; then
  ARGS+=(--skip_final_inference)
fi

if [[ "$RANDOM_FLIP" == "1" ]]; then
  ARGS+=(--random_flip)
fi

if [[ "$CENTER_CROP" == "1" ]]; then
  ARGS+=(--center_crop)
fi

if [[ "$OFFLOAD" == "1" ]]; then
  ARGS+=(--offload)
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ ${#CFG_JSON_EXTRA_ARGS[@]} -gt 0 ]]; then
  ARGS+=("${CFG_JSON_EXTRA_ARGS[@]}")
fi

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

echo "OPTIONS_JSON      : $OPTIONS_JSON"
echo "MODEL_PATH        : $MODEL_PATH"
echo "TRAIN_DATA_JSON   : $TRAIN_DATA_JSON"
echo "OUTPUT_DIR        : $OUTPUT_DIR"
echo "LORA_WEIGHTS_PATH : ${LORA_WEIGHTS_PATH:-<none>}"
echo "RESUME_CHECKPOINT : ${RESUME_FROM_CHECKPOINT:-<none>}"
echo "GPU_DEVICES       : ${GPU_DEVICES:-<default>}"
echo "REPORT_TO         : $REPORT_TO"
if [[ "$RUN_POST_EVAL" == "1" ]]; then
  echo "TEST_DATA_JSON    : $TEST_DATA_JSON"
  echo "TEST_NAME         : $TEST_NAME"
  echo "INFER_OUT_DIR     : $INFERENCE_OUTPUT_DIR"
  echo "EVAL_OUT_DIR      : $EVALUATION_OUTPUT_DIR"
fi
echo

run_flux2_python "${ARGS[@]}"

if [[ "$RUN_POST_EVAL" == "1" ]]; then
  EVAL_SCRIPT="$PROJECT_DIR/eval/03_evaluation.py"
  if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "ERROR: evaluation script not found: $EVAL_SCRIPT" >&2
    exit 1
  fi

  mkdir -p "$INFERENCE_OUTPUT_DIR" "$EVALUATION_OUTPUT_DIR"

  INFERENCE_RUN_DIR="$INFERENCE_OUTPUT_DIR/$TEST_NAME"
  INFERENCE_OUTPUT_JSON="$INFERENCE_OUTPUT_DIR/$TEST_NAME.json"
  EVALUATION_OUTPUT_JSON="$EVALUATION_OUTPUT_DIR/$TEST_NAME.json"

  INFER_ARGS=(
    "$STAGE2_DIR/models/lora_inference.py"
    --pretrained_model_name_or_path "$MODEL_PATH"
    --input_json "$TEST_DATA_JSON"
    --output_dir "$INFERENCE_RUN_DIR"
    --output_json "$INFERENCE_OUTPUT_JSON"
    --lora_weights_path "$OUTPUT_DIR"
    --mode "$INFERENCE_MODE"
    --resolution "$INFERENCE_RESOLUTION"
    --guidance_scale "$INFERENCE_GUIDANCE_SCALE"
    --num_inference_steps "$INFERENCE_NUM_INFERENCE_STEPS"
    --dtype "$INFERENCE_DTYPE"
    --device "$INFERENCE_DEVICE"
    --seed "$INFERENCE_SEED"
  )

  if [[ -n "$INSTANCE_PROMPT" ]]; then
    INFER_ARGS+=(--default_prompt "$INSTANCE_PROMPT")
  fi

  if [[ -n "$REVISION" ]]; then
    INFER_ARGS+=(--revision "$REVISION")
  fi

  if [[ -n "$VARIANT" ]]; then
    INFER_ARGS+=(--variant "$VARIANT")
  fi

  if [[ "$INFERENCE_CPU_OFFLOAD" == "1" ]]; then
    INFER_ARGS+=(--cpu_offload)
  fi

  echo "Running post-train inference..."
  echo "INFER_IMAGES_DIR  : $INFERENCE_RUN_DIR"
  echo "INFER_OUTPUT_JSON : $INFERENCE_OUTPUT_JSON"
  echo
  run_flux2_python "${INFER_ARGS[@]}"

  EVAL_ARGS=(
    "$EVAL_SCRIPT"
    --input "$INFERENCE_OUTPUT_JSON"
    --out "$EVALUATION_OUTPUT_JSON"
    --device "$EVALUATION_DEVICE"
    --fr_resize "$EVALUATION_FR_RESIZE"
  )

  if [[ -n "$EVALUATION_GPU_DEVICES" ]]; then
    EVAL_ARGS+=(--gpu_devices "$EVALUATION_GPU_DEVICES")
  fi

  echo "Running evaluation..."
  echo "EVAL_OUTPUT_JSON  : $EVALUATION_OUTPUT_JSON"
  echo
  run_flux2_python "${EVAL_ARGS[@]}"
fi
