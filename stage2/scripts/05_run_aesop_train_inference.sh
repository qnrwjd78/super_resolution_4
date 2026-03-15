#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/02_train.sh"
INFERENCE_SCRIPT="$SCRIPT_DIR/03_inference.sh"

TRAIN_CONFIGS=(
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_dists_maniqa_w1_0p15_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_dists_maniqa_w1_0p2_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_dists_maniqa_w1_0p25_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_dists_maniqa_w1_0p3_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_lpips_musiq_w1_0p15_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_lpips_musiq_w1_0p2_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_lpips_musiq_w1_0p25_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_lpips_musiq_w1_0p3_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_lpips_musiq_w1_0p35_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_lpips_musiq_w1_0p4_512.json"
  "$STAGE2_DIR/options/train/sem_lora/aesop/train_sem_lpips_musiq_w1_0p45_512.json"
)

INFERENCE_CONFIGS=(
  "$STAGE2_DIR/options/inference/aesop/inference512_dual_sem_aesop.json"
  "$STAGE2_DIR/options/inference/aesop/inference512_dual_sem_aesop_bs2.json"
  "$STAGE2_DIR/options/inference/aesop/inference512_dual_sem_aesop_bs4.json"
  "$STAGE2_DIR/options/inference/aesop/inference512_dual_sem_aesop_bs8.json"
  "$STAGE2_DIR/options/inference/aesop/inference512_dual_sem_aesop_bs16.json"
  "$STAGE2_DIR/options/inference/aesop/inference512_dual_sem_aesop_bs32.json"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path" >&2
    exit 1
  fi
}

run_section() {
  local label="$1"
  local worker="$2"
  shift 2
  local configs=("$@")
  local total="${#configs[@]}"
  local idx=0
  local cfg=""

  echo "========================="
  echo "$label"
  echo "========================="

  for cfg in "${configs[@]}"; do
    idx=$((idx + 1))
    require_file "$cfg"
    echo
    echo "[$label $idx/$total] $(basename "$cfg")"
    echo "bash $worker $cfg"
    bash "$worker" "$cfg"
  done
}

require_file "$TRAIN_SCRIPT"
require_file "$INFERENCE_SCRIPT"

run_section "TRAIN" "$TRAIN_SCRIPT" "${TRAIN_CONFIGS[@]}"
run_section "INFERENCE" "$INFERENCE_SCRIPT" "${INFERENCE_CONFIGS[@]}"

echo
echo "All aesop train + inference jobs finished."
