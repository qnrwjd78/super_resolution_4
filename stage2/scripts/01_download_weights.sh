#!/usr/bin/env bash
set -euo pipefail

# Download full Diffusers-style FLUX.2 repositories into:
#   stage2/weights/<model_key>/...
#
# Default (one model):
#   ./01_download_weights.sh
#
# Multiple models:
#   MODELS="flux.2-klein-4b flux.2-klein-base-9b flux.2-dev" ./01_download_weights.sh
#
# Aliases:
#   klein-4b, klein-9b, klein-base-4b, klein-base-9b, dev, all
#
# Force re-download of an existing local snapshot:
#   FORCE=1 ./01_download_weights.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHT_DIR="${WEIGHT_DIR:-$STAGE2_DIR/weights}"
MODELS="${MODELS:-flux.2-klein-base-9b}"
HF_LOGIN="${HF_LOGIN:-auto}"   # auto|force|skip
FORCE="${FORCE:-0}"            # 0|1

has() { command -v "$1" >/dev/null 2>&1; }

run_flux2_python() {
  if has flux2; then
    flux2 python "$@"
  elif has conda; then
    conda run -n flux2 --no-capture-output python "$@"
  elif has python3; then
    python3 "$@"
  elif has python; then
    python "$@"
  else
    echo "ERROR: python runtime not found." >&2
    exit 1
  fi
}

hf_cli() {
  if has hf; then
    hf "$@"
  elif has flux2; then
    flux2 hf "$@"
  elif has conda; then
    conda run -n flux2 --no-capture-output hf "$@"
  else
    return 127
  fi
}

has_hf_token() {
  run_flux2_python - <<'PY'
from huggingface_hub import HfFolder
import sys
sys.exit(0 if HfFolder.get_token() else 1)
PY
}

ensure_hf_login() {
  if [[ "$HF_LOGIN" == "skip" ]]; then
    echo "[INFO] HF_LOGIN=skip -> skipping login check."
    return 0
  fi

  if [[ "$HF_LOGIN" == "auto" ]] && has_hf_token; then
    echo "[OK] Hugging Face token already available."
    return 0
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[INFO] Logging into Hugging Face with HF_TOKEN..."
    hf_cli auth login --token "$HF_TOKEN" >/dev/null || true
  else
    echo "[INFO] Hugging Face login required. Running 'hf auth login'..."
    hf_cli auth login
  fi

  if ! has_hf_token; then
    echo "ERROR: Hugging Face login failed or token not available." >&2
    exit 1
  fi
  echo "[OK] Hugging Face login ready."
}

normalize_model_name() {
  case "$1" in
    flux.2-klein-4b|klein-4b|4b)
      echo "flux.2-klein-4b"
      ;;
    flux.2-klein-9b|klein-9b|9b)
      echo "flux.2-klein-9b"
      ;;
    flux.2-klein-base-4b|klein-base-4b|base-4b)
      echo "flux.2-klein-base-4b"
      ;;
    flux.2-klein-base-9b|klein-base-9b|base-9b)
      echo "flux.2-klein-base-9b"
      ;;
    flux.2-dev|dev)
      echo "flux.2-dev"
      ;;
    *)
      echo "ERROR: unknown model key: $1" >&2
      echo "Supported: flux.2-klein-4b flux.2-klein-9b flux.2-klein-base-4b flux.2-klein-base-9b flux.2-dev all" >&2
      exit 1
      ;;
  esac
}

set_repo_id() {
  case "$1" in
    flux.2-klein-4b) echo "black-forest-labs/FLUX.2-klein-4B" ;;
    flux.2-klein-9b) echo "black-forest-labs/FLUX.2-klein-9B" ;;
    flux.2-klein-base-4b) echo "black-forest-labs/FLUX.2-klein-base-4B" ;;
    flux.2-klein-base-9b) echo "black-forest-labs/FLUX.2-klein-base-9B" ;;
    flux.2-dev) echo "black-forest-labs/FLUX.2-dev" ;;
    *)
      echo "ERROR: unsupported model: $1" >&2
      exit 1
      ;;
  esac
}

is_diffusers_layout_complete() {
  local dir="$1"
  [[ -f "$dir/model_index.json" ]] &&
    [[ -d "$dir/scheduler" ]] &&
    [[ -d "$dir/tokenizer" ]] &&
    [[ -d "$dir/text_encoder" ]] &&
    [[ -d "$dir/transformer" ]] &&
    [[ -d "$dir/vae" ]]
}

download_hf_snapshot() {
  local repo_id="$1"
  local out_dir="$2"
  local force="$3"

  if [[ "$force" != "1" ]] && is_diffusers_layout_complete "$out_dir"; then
    echo "[SKIP] Diffusers layout already exists: $out_dir"
    return 0
  fi

  mkdir -p "$out_dir"
  echo "[DOWN]  $repo_id"
  echo "   ->  $out_dir"

  run_flux2_python - "$repo_id" "$out_dir" "$force" <<'PY'
import os
import shutil
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    raise SystemExit(
        f"ERROR: failed to import huggingface_hub ({e}). "
        "Run this inside the container (with flux2 env) or install huggingface_hub."
    )

repo_id, out_dir, force = sys.argv[1:4]

if force == "1" and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=out_dir,
        ignore_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.mp4"],
    )
except Exception as e:
    msg = str(e)
    if "GatedRepoError" in type(e).__name__ or "Cannot access gated repo" in msg:
        raise SystemExit(
            f"ERROR: access denied for gated repo '{repo_id}'.\n"
            f"Request access at: https://huggingface.co/{repo_id}"
        )
    raise

required = [
    "model_index.json",
    "scheduler",
    "tokenizer",
    "text_encoder",
    "transformer",
    "vae",
]
missing = []
for name in required:
    p = os.path.join(out_dir, name)
    if not os.path.exists(p):
        missing.append(name)

if missing:
    raise SystemExit(
        "ERROR: incomplete Diffusers snapshot. Missing: " + ", ".join(missing)
    )

print(f"[OK]   {out_dir}")
PY
}

expand_models() {
  local raw="$1"
  local out=()
  for m in $raw; do
    if [[ "$m" == "all" ]]; then
      out+=(
        "flux.2-klein-4b"
        "flux.2-klein-9b"
        "flux.2-klein-base-4b"
        "flux.2-klein-base-9b"
        "flux.2-dev"
      )
    else
      out+=("$(normalize_model_name "$m")")
    fi
  done
  printf '%s\n' "${out[@]}" | awk '!seen[$0]++'
}

echo "STAGE2_DIR : $STAGE2_DIR"
echo "WEIGHT_DIR : $WEIGHT_DIR"
echo "MODELS     : $MODELS"
echo "HF_LOGIN   : $HF_LOGIN"
echo "FORCE      : $FORCE"
echo
echo "Note: gated models require access approval + Hugging Face auth."
echo

ensure_hf_login
mapfile -t MODEL_LIST < <(expand_models "$MODELS")

for model_name in "${MODEL_LIST[@]}"; do
  repo_id="$(set_repo_id "$model_name")"
  model_dir="$WEIGHT_DIR/${model_name//./}"
  download_hf_snapshot "$repo_id" "$model_dir" "$FORCE"
done

echo
echo "Done."
echo "Downloaded Diffusers model directories:"
for model_name in "${MODEL_LIST[@]}"; do
  model_dir="$WEIGHT_DIR/${model_name//./}"
  echo " - $model_dir"
done
echo
echo "Example:"
echo "  flux2 python stage2/models/flux2_base.py"
echo "  # or for training:"
echo "  # --pretrained_model_name_or_path $WEIGHT_DIR/flux2-klein-base-9b"
