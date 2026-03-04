#!/usr/bin/env bash
set -euo pipefail

# Repairs an existing flux2 conda env by adding pyiqa safely.
# Keeps existing transformers stack; installs pyiqa with --no-deps.
#
# Usage:
#   bash stage2/scripts/00_repair_flux2_env.sh
# Optional:
#   ENV_NAME=flux2 INSTALL_OPENAI_CLIP=1 bash stage2/scripts/00_repair_flux2_env.sh

ENV_NAME="${ENV_NAME:-flux2}"
INSTALL_OPENAI_CLIP="${INSTALL_OPENAI_CLIP:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found." >&2
  exit 1
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

if ! env_exists; then
  echo "ERROR: conda env '$ENV_NAME' does not exist. Create it first." >&2
  exit 1
fi

echo "[1/7] Uninstalling existing pyiqa (if any)"
conda run -n "$ENV_NAME" --no-capture-output python -m pip uninstall -y pyiqa || true

echo "[2/7] Checking transformers version before install"
conda run -n "$ENV_NAME" --no-capture-output python - <<'PY'
import transformers
print("transformers(before):", transformers.__version__)
PY

echo "[3/7] Restoring transformers baseline for FLUX2"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install "transformers>=4.41.2"

echo "[4/7] Installing pyiqa only (no deps)"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install --no-deps "pyiqa==0.1.14.1"

echo "[5/7] Installing required runtime deps for pyiqa"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install \
  addict \
  einops \
  future \
  icecream \
  lmdb \
  opencv-python-headless \
  pandas \
  scikit-image \
  scipy \
  "timm>=0.8"

if [[ "$INSTALL_OPENAI_CLIP" == "1" ]]; then
  echo "[6/7] Installing openai-clip (for CLIP-IQA)"
  conda run -n "$ENV_NAME" --no-capture-output python -m pip install openai-clip
else
  echo "[6/7] Skipping openai-clip install (INSTALL_OPENAI_CLIP=$INSTALL_OPENAI_CLIP)"
fi

echo "[7/7] Verifying pyiqa metrics and transformers"
conda run -n "$ENV_NAME" --no-capture-output python - <<'PY'
import transformers
import pyiqa

print("transformers:", transformers.__version__)
print("pyiqa import ok")

for name in ["musiq", "maniqa"]:
    metric = pyiqa.create_metric(name, device="cpu", as_loss=True, loss_reduction="none")
    print(name, "ok | lower_better =", metric.lower_better)

try:
    metric = pyiqa.create_metric("clipiqa", device="cpu", as_loss=True, loss_reduction="none")
    print("clipiqa ok | lower_better =", metric.lower_better)
except Exception as exc:
    print("clipiqa failed:", exc)

try:
    metric = pyiqa.create_metric("niqe", device="cpu")
    print("niqe ok | lower_better =", metric.lower_better)
except Exception as exc:
    print("niqe failed:", exc)
PY

echo "Done."
