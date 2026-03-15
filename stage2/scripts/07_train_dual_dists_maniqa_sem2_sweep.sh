#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_BATCH_SCRIPT="$SCRIPT_DIR/02_train_batch.sh"

TRAIN_CONFIGS=(
  "$STAGE2_DIR/options/train/sem_lora/latent/dual_lora/train_dual_dists_maniqa_w1_0p1_sem2_r4_512.json"
  "$STAGE2_DIR/options/train/sem_lora/latent/dual_lora/train_dual_dists_maniqa_w1_0p2_sem2_r8_512.json"
  "$STAGE2_DIR/options/train/sem_lora/latent/dual_lora/train_dual_dists_maniqa_w1_0p25_sem2_r16_512.json"
  "$STAGE2_DIR/options/train/sem_lora/latent/dual_lora/train_dual_dists_maniqa_w1_0p3_sem2_r32_512.json"
  "$STAGE2_DIR/options/train/sem_lora/latent/dual_lora/train_dual_dists_maniqa_w1_0p4_sem2_r64_512.json"
  "$STAGE2_DIR/options/train/sem_lora/latent/dual_lora/train_dual_dists_maniqa_w1_0p5_sem2_r128_512.json"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path" >&2
    exit 1
  fi
}

require_file "$TRAIN_BATCH_SCRIPT"
for cfg in "${TRAIN_CONFIGS[@]}"; do
  require_file "$cfg"
done

echo "STAGE2_DIR    : $STAGE2_DIR"
echo "TRAIN_SCRIPT  : $TRAIN_BATCH_SCRIPT"
echo "GPU_DEVICES   : ${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-6}}"
echo "TRAIN_CONFIGS : ${#TRAIN_CONFIGS[@]}"
printf '  %s\n' "${TRAIN_CONFIGS[@]}"
echo

bash "$TRAIN_BATCH_SCRIPT" "${TRAIN_CONFIGS[@]}" "$@"
