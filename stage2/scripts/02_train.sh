#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$STAGE2_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
DEFAULT_OPTIONS_JSON="$STAGE2_DIR/options/train.json"
DEFAULT_PROMPTS_JSON="$STAGE2_DIR/prompts.json"
OPTIONS_JSON="${OPTIONS_JSON:-$DEFAULT_OPTIONS_JSON}"

if [[ $# -gt 0 && "$1" == *.json ]]; then
  OPTIONS_JSON="$1"
  shift
fi

has() { command -v "$1" >/dev/null 2>&1; }

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

count_csv_items() {
  local raw="$1"
  local item=""
  local count=0
  local old_ifs="$IFS"
  IFS=","
  read -r -a _items <<< "$raw"
  IFS="$old_ifs"
  for item in "${_items[@]}"; do
    item="${item// /}"
    if [[ -n "$item" ]]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

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

run_flux2_accelerate() {
  if has flux2; then
    flux2 python -m accelerate.commands.launch "$@"
  elif has conda; then
    conda run -n flux2 --no-capture-output python -m accelerate.commands.launch "$@"
  else
    echo "ERROR: flux2 wrapper (or conda env 'flux2') not found." >&2
    exit 1
  fi
}

run_cuda_preflight() {
  run_flux2_python - <<'PY'
import os
import torch

def _short(value, limit=240):
    if value is None:
        return "<unset>"
    value = str(value)
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"

cvis = os.environ.get("CUDA_VISIBLE_DEVICES")
nvis = os.environ.get("NVIDIA_VISIBLE_DEVICES")
ld = os.environ.get("LD_LIBRARY_PATH")
nvml_check = os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK")

print(f"CUDA_VISIBLE_DEVICES : {_short(cvis)}")
print(f"NVIDIA_VISIBLE_DEVICES : {_short(nvis)}")
print(f"PYTORCH_NVML_BASED_CUDA_CHECK : {_short(nvml_check)}")
print(f"LD_LIBRARY_PATH : {_short(ld)}")
print(f"torch.__version__ : {torch.__version__}")
print(f"torch.version.cuda : {torch.version.cuda}")

available = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"torch.cuda.is_available : {available}")
print(f"torch.cuda.device_count : {count}")

init_error = None
try:
    torch.cuda.init()
    print("torch.cuda.init : OK")
except Exception as exc:
    init_error = repr(exc)
    print(f"torch.cuda.init : FAIL ({init_error})")

if available and count > 0:
    for i in range(count):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = "<unknown>"
        print(f"cuda:{i} -> {name}")
    raise SystemExit(0)

raise SystemExit(7)
PY
}

resolve_prompt_by_name() {
  local prompts_json="$1"
  local prompt_name="$2"
  run_flux2_python - "$prompts_json" "$prompt_name" <<'PY'
import json
import sys
from pathlib import Path

prompts_path = Path(sys.argv[1]).expanduser().resolve()
prompt_name = sys.argv[2].strip()

if not prompt_name:
    raise SystemExit("ERROR: prompt name must be a non-empty string.")
if not prompts_path.exists():
    raise SystemExit(f"ERROR: prompts JSON not found: {prompts_path}")

with prompts_path.open("r", encoding="utf-8") as handle:
    entries = json.load(handle)

if not isinstance(entries, list):
    raise SystemExit("ERROR: prompts JSON must be a JSON array.")

available_names = []
for entry in entries:
    if not isinstance(entry, dict):
        continue
    name = str(entry.get("name", "")).strip()
    prompt = str(entry.get("prompt", "")).strip()
    if name:
        available_names.append(name)
    if name == prompt_name:
        if not prompt:
            raise SystemExit(f"ERROR: prompt entry '{prompt_name}' has empty 'prompt' text.")
        sys.stdout.write(prompt)
        raise SystemExit(0)

choices = ", ".join(sorted(available_names)) if available_names else "<none>"
raise SystemExit(f"ERROR: prompt name '{prompt_name}' was not found in {prompts_path}. Available names: {choices}")
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
    "pix_lora_weights_path": "CFG_PIX_LORA_WEIGHTS_PATH",
    "pix_adapter_name": "CFG_PIX_ADAPTER_NAME",
    "sem_adapter_name": "CFG_SEM_ADAPTER_NAME",
    "pix_adapter_scale": "CFG_PIX_ADAPTER_SCALE",
    "sem_adapter_scale": "CFG_SEM_ADAPTER_SCALE",
    "train_data_json": "CFG_TRAIN_DATA_JSON",
    "output_dir": "CFG_OUTPUT_DIR",
    "prompts_json": "CFG_PROMPTS_JSON",
    "instance_prompt": "CFG_INSTANCE_PROMPT",
    "instance_prompt_name": "CFG_INSTANCE_PROMPT_NAME",
    "lora_weights_path": "CFG_LORA_WEIGHTS_PATH",
    "resolution": "CFG_RESOLUTION",
    "train_batch_size": "CFG_TRAIN_BATCH_SIZE",
    "gradient_accumulation_steps": "CFG_GRADIENT_ACCUMULATION_STEPS",
    "num_processes": "CFG_NUM_PROCESSES",
    "num_machines": "CFG_NUM_MACHINES",
    "machine_rank": "CFG_MACHINE_RANK",
    "main_process_ip": "CFG_MAIN_PROCESS_IP",
    "main_process_port": "CFG_MAIN_PROCESS_PORT",
    "num_train_epochs": "CFG_NUM_TRAIN_EPOCHS",
    "learning_rate": "CFG_LEARNING_RATE",
    "mixed_precision": "CFG_MIXED_PRECISION",
    "report_to": "CFG_REPORT_TO",
    "seed": "CFG_SEED",
    "repeats": "CFG_REPEATS",
    "nr_iqa_metric": "CFG_NR_IQA_METRIC",
    "q_metric_weights": "CFG_Q_METRIC_WEIGHTS",
    "checkpointing_steps": "CFG_CHECKPOINTING_STEPS",
    "random_flip": "CFG_RANDOM_FLIP",
    "center_crop": "CFG_CENTER_CROP",
    "offload": "CFG_OFFLOAD",
    "resume_from_checkpoint": "CFG_RESUME_FROM_CHECKPOINT",
    "gpu_devices": "CFG_GPU_DEVICES",
}

store_true_keys = {
    "do_fp8_training",
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
    "use_nr_iqa_loss",
    "train_sem_only",
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
PIX_LORA_WEIGHTS_PATH="${PIX_LORA_WEIGHTS_PATH:-${CFG_PIX_LORA_WEIGHTS_PATH:-}}"
PIX_ADAPTER_NAME="${PIX_ADAPTER_NAME:-${CFG_PIX_ADAPTER_NAME:-pix}}"
SEM_ADAPTER_NAME="${SEM_ADAPTER_NAME:-${CFG_SEM_ADAPTER_NAME:-sem}}"
PIX_ADAPTER_SCALE="${PIX_ADAPTER_SCALE:-${CFG_PIX_ADAPTER_SCALE:-1.0}}"
SEM_ADAPTER_SCALE="${SEM_ADAPTER_SCALE:-${CFG_SEM_ADAPTER_SCALE:-1.0}}"
TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-${CFG_TRAIN_DATA_JSON:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CFG_OUTPUT_DIR:-$STAGE2_DIR/outputs/lora_train}}"
PROMPTS_JSON="${PROMPTS_JSON:-${CFG_PROMPTS_JSON:-$DEFAULT_PROMPTS_JSON}}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-${CFG_INSTANCE_PROMPT:-Enhance the perceptual quality of the condition image while preserving the original content, structure, and colors.}}"
INSTANCE_PROMPT_NAME="${INSTANCE_PROMPT_NAME:-${CFG_INSTANCE_PROMPT_NAME:-}}"
LORA_WEIGHTS_PATH="${LORA_WEIGHTS_PATH:-${CFG_LORA_WEIGHTS_PATH:-}}"

RESOLUTION="${RESOLUTION:-${CFG_RESOLUTION:-512}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${CFG_TRAIN_BATCH_SIZE:-1}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${CFG_GRADIENT_ACCUMULATION_STEPS:-1}}"
NUM_PROCESSES="${NUM_PROCESSES:-${CFG_NUM_PROCESSES:-}}"
NUM_MACHINES="${NUM_MACHINES:-${CFG_NUM_MACHINES:-1}}"
MACHINE_RANK="${MACHINE_RANK:-${CFG_MACHINE_RANK:-0}}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${CFG_MAIN_PROCESS_IP:-}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${CFG_MAIN_PROCESS_PORT:-}}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-${CFG_NUM_TRAIN_EPOCHS:-1}}"
LEARNING_RATE="${LEARNING_RATE:-${CFG_LEARNING_RATE:-1e-4}}"
MIXED_PRECISION="${MIXED_PRECISION:-${CFG_MIXED_PRECISION:-bf16}}"
REPORT_TO="${REPORT_TO:-${CFG_REPORT_TO:-tensorboard}}"
SEED="${SEED:-${CFG_SEED:-0}}"
REPEATS="${REPEATS:-${CFG_REPEATS:-1}}"
NR_IQA_METRIC="${NR_IQA_METRIC:-${CFG_NR_IQA_METRIC:-musiq}}"
Q_METRIC_WEIGHTS="${Q_METRIC_WEIGHTS:-${CFG_Q_METRIC_WEIGHTS:-}}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-${CFG_CHECKPOINTING_STEPS:-500}}"
RANDOM_FLIP="${RANDOM_FLIP:-${CFG_RANDOM_FLIP:-0}}"
CENTER_CROP="${CENTER_CROP:-${CFG_CENTER_CROP:-0}}"
OFFLOAD="${OFFLOAD:-${CFG_OFFLOAD:-1}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${CFG_RESUME_FROM_CHECKPOINT:-}}"
# Prefer per-run JSON config first so concurrent jobs can target different GPUs reliably.
GPU_DEVICES="${CFG_GPU_DEVICES:-${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
GPU_DEVICES="${GPU_DEVICES// /}"

if [[ -n "$INSTANCE_PROMPT_NAME" ]]; then
  INSTANCE_PROMPT="$(resolve_prompt_by_name "$PROMPTS_JSON" "$INSTANCE_PROMPT_NAME")"
fi

if [[ -z "$TRAIN_DATA_JSON" ]]; then
  echo "ERROR: set TRAIN_DATA_JSON or provide it in $OPTIONS_JSON." >&2
  exit 1
fi

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  # Avoid accidental CPU-forced runs from inherited shell environment.
  unset ACCELERATE_USE_CPU

  CUDA_PREFLIGHT_OUTPUT=""
  if ! CUDA_PREFLIGHT_OUTPUT="$(run_cuda_preflight)"; then
    if [[ -n "$CUDA_PREFLIGHT_OUTPUT" ]]; then
      echo "$CUDA_PREFLIGHT_OUTPUT"
    fi
    echo "WARN: CUDA is not visible in the current runtime. Retrying with cleaned CUDA env..." >&2

    CUDA_PREFLIGHT_CLEAN_OUTPUT=""
    if CUDA_PREFLIGHT_CLEAN_OUTPUT="$(
      LD_LIBRARY_PATH="" CUDA_HOME="" CUDA_INC_DIR="" CUDA_PATH="" PYTORCH_NVML_BASED_CUDA_CHECK=0 run_cuda_preflight
    )"; then
      unset LD_LIBRARY_PATH CUDA_HOME CUDA_INC_DIR CUDA_PATH
      export PYTORCH_NVML_BASED_CUDA_CHECK=0
      echo "$CUDA_PREFLIGHT_CLEAN_OUTPUT"
      echo "INFO: Applied cleaned CUDA environment for this run." >&2
    else
      if [[ -n "$CUDA_PREFLIGHT_CLEAN_OUTPUT" ]]; then
        echo "$CUDA_PREFLIGHT_CLEAN_OUTPUT"
      fi
      echo "ERROR: CUDA is not visible in the training runtime. Aborting before launch." >&2
      echo "       Check GPU_DEVICES/CUDA_VISIBLE_DEVICES and ensure ACCELERATE_USE_CPU is not forced." >&2
      echo "       Also verify flux2 env torch build and driver linkage." >&2
      exit 1
    fi
  else
    echo "$CUDA_PREFLIGHT_OUTPUT"
  fi
fi

if [[ -z "$NUM_PROCESSES" ]]; then
  if [[ -n "$GPU_DEVICES" ]]; then
    NUM_PROCESSES="$(count_csv_items "$GPU_DEVICES")"
    if [[ "$NUM_PROCESSES" -lt 1 ]]; then
      NUM_PROCESSES=1
    fi
  else
    NUM_PROCESSES=1
  fi
fi

for pair in \
  "TRAIN_BATCH_SIZE:$TRAIN_BATCH_SIZE" \
  "GRADIENT_ACCUMULATION_STEPS:$GRADIENT_ACCUMULATION_STEPS" \
  "NUM_PROCESSES:$NUM_PROCESSES" \
  "NUM_MACHINES:$NUM_MACHINES" \
  "MACHINE_RANK:$MACHINE_RANK"; do
  key="${pair%%:*}"
  value="${pair#*:}"
  if ! is_uint "$value"; then
    echo "ERROR: $key must be a non-negative integer. Got: $value" >&2
    exit 1
  fi
done

if [[ "$TRAIN_BATCH_SIZE" -lt 1 || "$GRADIENT_ACCUMULATION_STEPS" -lt 1 || "$NUM_PROCESSES" -lt 1 || "$NUM_MACHINES" -lt 1 ]]; then
  echo "ERROR: TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, NUM_PROCESSES, and NUM_MACHINES must be >= 1." >&2
  exit 1
fi

if [[ "$NUM_MACHINES" -gt 1 ]]; then
  if [[ -z "$MAIN_PROCESS_IP" || -z "$MAIN_PROCESS_PORT" ]]; then
    echo "ERROR: MAIN_PROCESS_IP and MAIN_PROCESS_PORT are required when NUM_MACHINES > 1." >&2
    exit 1
  fi
fi

GLOBAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_PROCESSES))
USE_DISTRIBUTED=0
if [[ "$NUM_PROCESSES" -gt 1 || "$NUM_MACHINES" -gt 1 ]]; then
  USE_DISTRIBUTED=1
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  RESUME_PATH="$RESUME_FROM_CHECKPOINT"
  if [[ "$RESUME_FROM_CHECKPOINT" != "latest" && ! "$RESUME_FROM_CHECKPOINT" = /* ]]; then
    RESUME_PATH="$OUTPUT_DIR/$RESUME_FROM_CHECKPOINT"
  fi

  if [[ "$RESUME_FROM_CHECKPOINT" != "latest" && ! -d "$RESUME_PATH" ]]; then
    echo "ERROR: RESUME_FROM_CHECKPOINT directory not found: $RESUME_PATH" >&2
    exit 1
  fi

  if [[ "$RESUME_FROM_CHECKPOINT" != "latest" && "$NUM_PROCESSES" -gt 1 ]]; then
    MISSING_RANDOM_STATE=0
    for ((i=0; i<NUM_PROCESSES; i++)); do
      if [[ ! -f "$RESUME_PATH/random_states_${i}.pkl" ]]; then
        MISSING_RANDOM_STATE=1
        break
      fi
    done

    if [[ "$MISSING_RANDOM_STATE" == "1" ]]; then
      FOUND_RANDOM_STATES=0
      while IFS= read -r _line; do
        FOUND_RANDOM_STATES=$((FOUND_RANDOM_STATES + 1))
      done < <(find "$RESUME_PATH" -maxdepth 1 -type f -name 'random_states_*.pkl' | sort)

      echo "ERROR: Resume checkpoint and NUM_PROCESSES mismatch." >&2
      echo "       RESUME_PATH         : $RESUME_PATH" >&2
      echo "       NUM_PROCESSES       : $NUM_PROCESSES" >&2
      echo "       random_states found : $FOUND_RANDOM_STATES" >&2
      echo "       Fix: use matching NUM_PROCESSES, or clear RESUME_FROM_CHECKPOINT to start a new run." >&2
      exit 1
    fi
  fi
fi

TRAIN_SCRIPT="$STAGE2_DIR/models/lora.py"
ARGS=(
  --pretrained_model_name_or_path "$MODEL_PATH"
  --train_data_json "$TRAIN_DATA_JSON"
  --output_dir "$OUTPUT_DIR"
  --instance_prompt "$INSTANCE_PROMPT"
  --resolution "$RESOLUTION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --mixed_precision "$MIXED_PRECISION"
  --report_to "$REPORT_TO"
  --seed "$SEED"
  --repeats "$REPEATS"
  --nr_iqa_metric "$NR_IQA_METRIC"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
)

if [[ -n "$Q_METRIC_WEIGHTS" ]]; then
  ARGS+=(--q_metric_weights "$Q_METRIC_WEIGHTS")
fi

if [[ -n "$REVISION" ]]; then
  ARGS+=(--revision "$REVISION")
fi

if [[ -n "$VARIANT" ]]; then
  ARGS+=(--variant "$VARIANT")
fi

if [[ -n "$PIX_LORA_WEIGHTS_PATH" ]]; then
  ARGS+=(--pix_lora_weights_path "$PIX_LORA_WEIGHTS_PATH")
fi

if [[ -n "$PIX_ADAPTER_NAME" ]]; then
  ARGS+=(--pix_adapter_name "$PIX_ADAPTER_NAME")
fi

if [[ -n "$SEM_ADAPTER_NAME" ]]; then
  ARGS+=(--sem_adapter_name "$SEM_ADAPTER_NAME")
fi

if [[ -n "$PIX_ADAPTER_SCALE" ]]; then
  ARGS+=(--pix_adapter_scale "$PIX_ADAPTER_SCALE")
fi

if [[ -n "$SEM_ADAPTER_SCALE" ]]; then
  ARGS+=(--sem_adapter_scale "$SEM_ADAPTER_SCALE")
fi

if [[ -n "$LORA_WEIGHTS_PATH" ]]; then
  ARGS+=(--lora_weights_path "$LORA_WEIGHTS_PATH")
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
echo "PROMPTS_JSON      : $PROMPTS_JSON"
if [[ -n "$INSTANCE_PROMPT_NAME" ]]; then
  echo "INSTANCE_PROMPT_NAME : $INSTANCE_PROMPT_NAME"
fi
echo "LORA_WEIGHTS_PATH : ${LORA_WEIGHTS_PATH:-<none>}"
echo "RESUME_CHECKPOINT : ${RESUME_FROM_CHECKPOINT:-<none>}"
echo "GPU_DEVICES       : ${GPU_DEVICES:-<default>}"
echo "TRAIN_BATCH_SIZE  : $TRAIN_BATCH_SIZE (per device)"
echo "GRAD_ACC_STEPS    : $GRADIENT_ACCUMULATION_STEPS"
echo "NUM_PROCESSES     : $NUM_PROCESSES"
echo "NUM_MACHINES      : $NUM_MACHINES"
echo "MACHINE_RANK      : $MACHINE_RANK"
echo "GLOBAL_BATCH_SIZE : $GLOBAL_BATCH_SIZE"
if [[ "$USE_DISTRIBUTED" == "1" ]]; then
  echo "TRAIN_LAUNCHER    : accelerate"
else
  echo "TRAIN_LAUNCHER    : python"
fi
echo "NR_IQA_METRIC     : $NR_IQA_METRIC"
if [[ -n "$Q_METRIC_WEIGHTS" ]]; then
  echo "Q_METRIC_WEIGHTS  : $Q_METRIC_WEIGHTS"
fi
echo "REPORT_TO         : $REPORT_TO"
echo

if [[ "$USE_DISTRIBUTED" == "1" ]]; then
  LAUNCH_ARGS=(
    --num_processes "$NUM_PROCESSES"
    --num_machines "$NUM_MACHINES"
    --machine_rank "$MACHINE_RANK"
  )
  if [[ -n "$MAIN_PROCESS_IP" ]]; then
    LAUNCH_ARGS+=(--main_process_ip "$MAIN_PROCESS_IP")
  fi
  if [[ -n "$MAIN_PROCESS_PORT" ]]; then
    LAUNCH_ARGS+=(--main_process_port "$MAIN_PROCESS_PORT")
  fi
  run_flux2_accelerate "${LAUNCH_ARGS[@]}" "$TRAIN_SCRIPT" "${ARGS[@]}"
else
  run_flux2_python "$TRAIN_SCRIPT" "${ARGS[@]}"
fi
