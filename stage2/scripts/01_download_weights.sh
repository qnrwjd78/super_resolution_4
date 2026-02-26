#!/usr/bin/env bash
set -euo pipefail

# Download FLUX.2 weights into super_resolution_4/stage2/weights/<model>/...
#
# Default downloads only klein-9b:
#   ./01_download_weights.sh
#
# Download multiple models:
#   MODELS="flux.2-klein-4b flux.2-klein-9b flux.2-dev" ./01_download_weights.sh
#
# Aliases supported:
#   klein-4b, klein-9b, klein-base-4b, klein-base-9b, dev, all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHT_DIR="${WEIGHT_DIR:-$STAGE2_DIR/weights}"
MODELS="${MODELS:-flux.2-klein-9b}"
INCLUDE_AE="${INCLUDE_AE:-1}"
# FLUX.2 autoencoder is published at FLUX.2-dev repo.
AE_REPO_ID="${AE_REPO_ID:-black-forest-labs/FLUX.2-dev}"
HF_LOGIN="${HF_LOGIN:-auto}"  # auto|force|skip

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

set_repo_and_filename() {
  case "$1" in
    flux.2-klein-4b)
      REPO_ID="black-forest-labs/FLUX.2-klein-4B"
      MODEL_FILE="flux-2-klein-4b.safetensors"
      ;;
    flux.2-klein-9b)
      REPO_ID="black-forest-labs/FLUX.2-klein-9B"
      MODEL_FILE="flux-2-klein-9b.safetensors"
      ;;
    flux.2-klein-base-4b)
      REPO_ID="black-forest-labs/FLUX.2-klein-base-4B"
      MODEL_FILE="flux-2-klein-base-4b.safetensors"
      ;;
    flux.2-klein-base-9b)
      REPO_ID="black-forest-labs/FLUX.2-klein-base-9B"
      MODEL_FILE="flux-2-klein-base-9b.safetensors"
      ;;
    flux.2-dev)
      REPO_ID="black-forest-labs/FLUX.2-dev"
      MODEL_FILE="flux2-dev.safetensors"
      ;;
    *)
      echo "ERROR: unsupported model: $1" >&2
      exit 1
      ;;
  esac
}

download_hf_file() {
  local repo_id="$1"
  local filename="$2"
  local out_path="$3"

  if [[ -f "$out_path" ]]; then
    echo "[SKIP] exists: $out_path"
    return 0
  fi

  mkdir -p "$(dirname "$out_path")"
  echo "[DOWN]  $repo_id :: $filename"
  echo "   ->  $out_path"

  run_flux2_python - "$repo_id" "$filename" "$out_path" <<'PY'
import os
import shutil
import sys

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    raise SystemExit(
        f"ERROR: failed to import huggingface_hub ({e}). "
        "Run this inside the container (with flux2 env) or install huggingface_hub."
    )

repo_id, filename, out_path = sys.argv[1:4]
try:
    tmp_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
except Exception as e:
    msg = str(e)
    if "GatedRepoError" in type(e).__name__ or "Cannot access gated repo" in msg:
        raise SystemExit(
            f"ERROR: access denied for gated repo '{repo_id}'.\n"
            f"Request access at: https://huggingface.co/{repo_id}\n"
            "If you need immediate download without gating, use MODELS=flux.2-klein-4b."
        )
    raise

os.makedirs(os.path.dirname(out_path), exist_ok=True)
shutil.copy2(tmp_path, out_path)
print(f"[OK]   {out_path}")
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
echo "INCLUDE_AE : $INCLUDE_AE"
echo "AE_REPO_ID : $AE_REPO_ID"
echo "HF_LOGIN   : $HF_LOGIN"
echo
echo 'Note: gated models require access approval + Hugging Face auth.'
echo

ensure_hf_login

mapfile -t MODEL_LIST < <(expand_models "$MODELS")

for model_name in "${MODEL_LIST[@]}"; do
  set_repo_and_filename "$model_name"
  model_dir="$WEIGHT_DIR/${model_name//./}"

  download_hf_file "$REPO_ID" "$MODEL_FILE" "$model_dir/$MODEL_FILE"
  if [[ "$INCLUDE_AE" == "1" ]]; then
    download_hf_file "$AE_REPO_ID" "ae.safetensors" "$model_dir/ae.safetensors"
  fi
done

echo
echo "Done."
echo "Downloaded files:"
for model_name in "${MODEL_LIST[@]}"; do
  set_repo_and_filename "$model_name"
  model_dir="$WEIGHT_DIR/${model_name//./}"
  echo " - $model_dir/$MODEL_FILE"
  if [[ "$INCLUDE_AE" == "1" ]]; then
    echo " - $model_dir/ae.safetensors"
  fi
done
