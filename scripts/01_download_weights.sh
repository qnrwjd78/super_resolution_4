#!/usr/bin/env bash
set -euo pipefail

# Download pretrained weights to the exact paths expected by your run_*.sh scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # .../super_resolution_4
REPOS_DIR="$SR_DIR/repos"

# Choose models to download (space-separated).
# default: the three you are missing now
MODELS="${MODELS:-mambairv2 dat hat}"
# You may also include: swin2sr swinir
# e.g. MODELS="mambairv2 dat hat swin2sr swinir"

has() { command -v "$1" >/dev/null 2>&1; }

download() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" ]]; then
    echo "[SKIP] exists: $out"
    return 0
  fi

  mkdir -p "$(dirname "$out")"
  echo "[DOWN]  $url"
  echo "   ->  $out"

  if has curl; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
  elif has wget; then
    wget -O "$out" "$url"
  else
    echo "ERROR: need curl or wget." >&2
    return 1
  fi

  echo "[OK]   $out"
}

echo "SR_DIR   : $SR_DIR"
echo "REPOS_DIR: $REPOS_DIR"
echo "MODELS   : $MODELS"
echo

for m in $MODELS; do
  case "$m" in
    mambairv2)
      # Official GitHub release asset (tag v1.0)
      # file expected by your script:
      # repos/MambaIR/experiments/pretrained_models/mambairv2_classicSR_Base_x4.pth
      download \
        "https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_classicSR_Base_x4.pth" \
        "$REPOS_DIR/MambaIR/experiments/pretrained_models/mambairv2_classicSR_Base_x4.pth"
      ;;

    dat)
      # DAT official provides weights via Google Drive, but Drive download is annoying in headless Docker.
      # So we use a HuggingFace mirror pinned to a commit (stable path).
      # expected:
      # repos/DAT/experiments/pretrained_models/DAT/DAT_x4.pth
      download \
        "https://huggingface.co/w-e-w/DAT/resolve/0282d9e2afcf7d1b4069a9e379e437c9f1ecc392/experiments/pretrained_models/DAT/DAT_x4.pth" \
        "$REPOS_DIR/DAT/experiments/pretrained_models/DAT/DAT_x4.pth"
      ;;

    hat)
      # HAT official provides weights via Google Drive/Baidu.
      # We use a HuggingFace mirror pinned to a commit (stable path).
      # expected:
      # repos/HAT/experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth
      download \
        "https://huggingface.co/Acly/hat/resolve/8403819bcbf5959d54c72383f0725f2525472d30/HAT_SRx4_ImageNet-pretrain.pth" \
        "$REPOS_DIR/HAT/experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth"
      ;;

    *)
      echo "ERROR: unknown model key: $m"
      exit 1
      ;;
  esac
done

echo
echo "Done."
echo "Downloaded files:"
for m in $MODELS; do
  case "$m" in
    mambairv2) echo " - $REPOS_DIR/MambaIR/experiments/pretrained_models/mambairv2_classicSR_Base_x4.pth" ;;
    dat)      echo " - $REPOS_DIR/DAT/experiments/pretrained_models/DAT/DAT_x4.pth" ;;
    hat)      echo " - $REPOS_DIR/HAT/experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth" ;;
  esac
done
