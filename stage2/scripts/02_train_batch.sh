#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/02_train.sh"

if [[ ! -x "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: worker script not found or not executable: $TRAIN_SCRIPT" >&2
  exit 1
fi

parse_csv_items() {
  local raw="${1:-}"
  local old_ifs="$IFS"
  local item=""
  IFS=","
  read -r -a _items <<< "$raw"
  IFS="$old_ifs"
  for item in "${_items[@]}"; do
    item="${item// /}"
    if [[ -n "$item" ]]; then
      printf '%s\n' "$item"
    fi
  done
}

usage() {
  cat <<EOF
Usage:
  GPU_DEVICES=0,1,2 $0 <json1> [json2 ...] [-- extra args for 02_train.sh]

Examples:
  GPU_DEVICES=0,1 $0 $STAGE2_DIR/options/train/sem_lora/latent/train_sem_*_w1_0p1_512.json
  GPU_DEVICES=2,3 $0 $STAGE2_DIR/options/train/sem_lora/pixel/*.json -- --mixed_precision bf16

Notes:
  - One training job is assigned to one GPU.
  - This batch launcher forces \`NUM_PROCESSES=1\` for each job.
  - Per-json \`gpu_devices\` values are ignored during batch execution.
EOF
}

JSON_FILES=()
CLI_EXTRA_ARGS=()
PARSING_JSONS=1

for arg in "$@"; do
  if [[ "$arg" == "--" && "$PARSING_JSONS" == "1" ]]; then
    PARSING_JSONS=0
    continue
  fi
  if [[ "$PARSING_JSONS" == "1" ]]; then
    JSON_FILES+=("$arg")
  else
    CLI_EXTRA_ARGS+=("$arg")
  fi
done

if [[ ${#JSON_FILES[@]} -eq 0 ]]; then
  usage >&2
  exit 1
fi

mapfile -t GPU_LIST < <(parse_csv_items "${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}")
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  GPU_LIST=("0")
fi

for json_path in "${JSON_FILES[@]}"; do
  if [[ ! -f "$json_path" ]]; then
    echo "ERROR: options JSON not found: $json_path" >&2
    exit 1
  fi
done

mkdir -p "$STAGE2_DIR/outputs/train_batch_logs"

echo "TRAIN_SCRIPT : $TRAIN_SCRIPT"
echo "JSON_COUNT   : ${#JSON_FILES[@]}"
echo "GPU_DEVICES  : ${GPU_LIST[*]}"
echo "LOG_DIR      : $STAGE2_DIR/outputs/train_batch_logs"
if [[ ${#CLI_EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "EXTRA_ARGS   : ${CLI_EXTRA_ARGS[*]}"
fi
echo

declare -a SLOT_PIDS
declare -a SLOT_GPUS
declare -a SLOT_NAMES
declare -a SLOT_LOGS

for ((i=0; i<${#GPU_LIST[@]}; i++)); do
  SLOT_PIDS[i]=""
  SLOT_GPUS[i]="${GPU_LIST[i]}"
  SLOT_NAMES[i]=""
  SLOT_LOGS[i]=""
done

wait_for_any_slot() {
  while true; do
    for ((slot_idx=0; slot_idx<${#SLOT_PIDS[@]}; slot_idx++)); do
      local pid="${SLOT_PIDS[slot_idx]}"
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        local finished_name="${SLOT_NAMES[slot_idx]}"
        local finished_log="${SLOT_LOGS[slot_idx]}"
        if wait "$pid"; then
          echo "Finished job: $finished_name on GPU ${SLOT_GPUS[slot_idx]}"
          echo "  log: $finished_log"
        else
          echo "ERROR: job failed: $finished_name on GPU ${SLOT_GPUS[slot_idx]}" >&2
          echo "       log: $finished_log" >&2
          exit 1
        fi
        SLOT_PIDS[slot_idx]=""
        SLOT_NAMES[slot_idx]=""
        SLOT_LOGS[slot_idx]=""
        return 0
      fi
    done
    sleep 2
  done
}

launch_on_slot() {
  local slot_idx="$1"
  local gpu_device="$2"
  local json_path="$3"
  local job_name="$4"
  local log_path="$5"

  echo "Launching job: $job_name on GPU $gpu_device"
  echo "  json: $json_path"
  echo "  log : $log_path"

  (
    export GPU_DEVICES="$gpu_device"
    export NUM_PROCESSES=1
    "$TRAIN_SCRIPT" "$json_path" "${CLI_EXTRA_ARGS[@]}"
  ) >"$log_path" 2>&1 &

  SLOT_PIDS[slot_idx]="$!"
  SLOT_NAMES[slot_idx]="$job_name"
  SLOT_LOGS[slot_idx]="$log_path"
}

for json_path in "${JSON_FILES[@]}"; do
  job_name="$(basename "$json_path" .json)"
  log_path="$STAGE2_DIR/outputs/train_batch_logs/${job_name}.log"

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
        "$json_path" \
        "$job_name" \
        "$log_path"
      break
    fi

    wait_for_any_slot
  done
done

for ((slot_idx=0; slot_idx<${#SLOT_PIDS[@]}; slot_idx++)); do
  if [[ -n "${SLOT_PIDS[slot_idx]}" ]]; then
    if wait "${SLOT_PIDS[slot_idx]}"; then
      echo "Finished job: ${SLOT_NAMES[slot_idx]} on GPU ${SLOT_GPUS[slot_idx]}"
      echo "  log: ${SLOT_LOGS[slot_idx]}"
    else
      echo "ERROR: job failed: ${SLOT_NAMES[slot_idx]} on GPU ${SLOT_GPUS[slot_idx]}" >&2
      echo "       log: ${SLOT_LOGS[slot_idx]}" >&2
      exit 1
    fi
  fi
done

echo
echo "All jobs finished."
