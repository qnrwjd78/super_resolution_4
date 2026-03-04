#!/usr/bin/env bash
set -euo pipefail

# Rebuilds flux2 conda env to match the original Dockerfile flux2 section.
#
# Usage:
#   bash stage2/scripts/00_repair_flux2_env.sh
# Optional:
#   ENV_NAME=flux2 PYTHON_VERSION=3.10 RECREATE=1 bash stage2/scripts/00_repair_flux2_env.sh

ENV_NAME="${ENV_NAME:-flux2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RECREATE="${RECREATE:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found." >&2
  exit 1
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

if [[ "$RECREATE" == "1" ]] && env_exists; then
  echo "[1/7] Removing existing conda env: $ENV_NAME"
  conda remove -n "$ENV_NAME" --all -y
else
  echo "[1/7] Skip removal (RECREATE=$RECREATE, exists=$(env_exists && echo 1 || echo 0))"
fi

echo "[2/7] Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
conda create -n "$ENV_NAME" -y --override-channels -c conda-forge "python=${PYTHON_VERSION}"

echo "[3/7] Installing base pip/setuptools/wheel"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install -U \
  "pip<25" \
  "setuptools<82" \
  wheel

echo "[4/7] Installing torch/cu117 stack"
conda run -n "$ENV_NAME" --no-capture-output pip install --index-url "$TORCH_INDEX_URL" \
  torch==2.0.1 \
  torchvision==0.15.2 \
  torchaudio==2.0.2

echo "[5/7] Installing Flux2 LoRA trainer dependencies (Dockerfile-original)"
conda run -n "$ENV_NAME" --no-capture-output pip install \
  "numpy<2" \
  pillow \
  tqdm \
  packaging \
  pyyaml \
  requests \
  "accelerate>=0.31.0" \
  "transformers>=4.41.2" \
  "peft>=0.11.1" \
  datasets \
  sentencepiece \
  ftfy \
  tensorboard \
  Jinja2 \
  bitsandbytes \
  prodigyopt \
  huggingface-hub \
  wandb \
  safetensors

echo "[6/7] Installing diffusers from main (Dockerfile-original)"
conda run -n "$ENV_NAME" --no-capture-output pip install \
  "diffusers @ git+https://github.com/huggingface/diffusers.git"

echo "[7/7] Verifying environment was created"
conda run -n "$ENV_NAME" --no-capture-output python -V

echo "Done."
