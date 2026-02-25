#!/usr/bin/env bash
set -euo pipefail

# Download pretrained weights into super_resolution_4/weight/<model>/...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # .../super_resolution_4
WEIGHT_DIR="$SR_DIR/weight"

# Choose models to download (space-separated).
# Default: download everything we support here.
# Available keys: hat mambair mambairv2 dat swinir swin2sr
# e.g. MODELS="hat swinir"
MODELS="${MODELS:-hat mambair dat swinir swin2sr}"

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
echo "WEIGHT_DIR: $WEIGHT_DIR"
echo "MODELS   : $MODELS"
echo

for m in $MODELS; do
  case "$m" in
    hat)
      HAT_BASE="https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT"
      download "$HAT_BASE/HAT_SRx4.pth" "$WEIGHT_DIR/hat/HAT_SRx4.pth"
      download "$HAT_BASE/HAT_SRx4_ImageNet-pretrain.pth" "$WEIGHT_DIR/hat/HAT_SRx4_ImageNet-pretrain.pth"
      download "$HAT_BASE/HAT-L_SRx4_ImageNet-pretrain.pth" "$WEIGHT_DIR/hat/HAT-L_SRx4_ImageNet-pretrain.pth"
      download "$HAT_BASE/HAT-S_SRx4.pth" "$WEIGHT_DIR/hat/HAT-S_SRx4.pth"
      download "$HAT_BASE/Real_HAT_GAN_SRx4.pth" "$WEIGHT_DIR/hat/Real_HAT_GAN_SRx4.pth"
      # upstream filename is Real_HAT_GAN_sharper.pth (rename on save)
      download "$HAT_BASE/Real_HAT_GAN_sharper.pth" "$WEIGHT_DIR/hat/Real_HAT_GAN_SRx4_sharper.pth"
      ;;

    mambair)
      MAMBAIR_BASE="https://huggingface.co/cguoh/MambaIR/resolve/main"
      # rename on save to match your requested filenames
      download "$MAMBAIR_BASE/MambaIRv1_ckpt/MambaIR_classicSRx4.pth" "$WEIGHT_DIR/mambair/MambaIR_SRx4.pth"
      download "$MAMBAIR_BASE/MambaIRv2_ckpt/mambairv2_classicSR_Base_x4.pth" "$WEIGHT_DIR/mambair/MambaIRv2_SRx4.pth"
      ;;

    mambairv2)
      MAMBAIR_BASE="https://huggingface.co/cguoh/MambaIR/resolve/main"
      download "$MAMBAIR_BASE/MambaIRv2_ckpt/mambairv2_classicSR_Base_x4.pth" "$WEIGHT_DIR/mambair/MambaIRv2_SRx4.pth"
      ;;

    swinir)
      SWINIR_BASE="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"
      download "$SWINIR_BASE/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth" "$WEIGHT_DIR/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
      download "$SWINIR_BASE/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth" "$WEIGHT_DIR/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
      download "$SWINIR_BASE/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth" "$WEIGHT_DIR/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
      download "$SWINIR_BASE/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" "$WEIGHT_DIR/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
      download "$SWINIR_BASE/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" "$WEIGHT_DIR/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
      ;;

    swin2sr)
      SWIN2SR_BASE="https://github.com/mv-lab/swin2sr/releases/download/v0.0.1"
      download "$SWIN2SR_BASE/Swin2SR_ClassicalSR_X4_64.pth" "$WEIGHT_DIR/swin2sr/Swin2SR_ClassicalSR_X4_64.pth"
      download "$SWIN2SR_BASE/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth" "$WEIGHT_DIR/swin2sr/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"
      download "$SWIN2SR_BASE/Swin2SR_CompressedSR_X4_48.pth" "$WEIGHT_DIR/swin2sr/Swin2SR_CompressedSR_X4_48.pth"
      ;;

    dat)
      DAT_BASE="https://huggingface.co/w-e-w/DAT/resolve/0282d9e2afcf7d1b4069a9e379e437c9f1ecc392/experiments/pretrained_models"
      download "$DAT_BASE/DAT/DAT_x4.pth" "$WEIGHT_DIR/dat/DAT_x4.pth"
      download "$DAT_BASE/DAT-S/DAT_S_x4.pth" "$WEIGHT_DIR/dat/DAT_S_x4.pth"
      download "$DAT_BASE/DAT-2/DAT_2_x4.pth" "$WEIGHT_DIR/dat/DAT_2_x4.pth"
      download "$DAT_BASE/DAT-light/DAT_light_x4.pth" "$WEIGHT_DIR/dat/DAT_light_x4.pth"
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
    hat)
      echo " - $WEIGHT_DIR/hat/HAT_SRx4.pth"
      echo " - $WEIGHT_DIR/hat/HAT_SRx4_ImageNet-pretrain.pth"
      echo " - $WEIGHT_DIR/hat/HAT-L_SRx4_ImageNet-pretrain.pth"
      echo " - $WEIGHT_DIR/hat/HAT-S_SRx4.pth"
      echo " - $WEIGHT_DIR/hat/Real_HAT_GAN_SRx4.pth"
      echo " - $WEIGHT_DIR/hat/Real_HAT_GAN_SRx4_sharper.pth"
      ;;
    mambair)
      echo " - $WEIGHT_DIR/mambair/MambaIR_SRx4.pth"
      echo " - $WEIGHT_DIR/mambair/MambaIRv2_SRx4.pth"
      ;;
    mambairv2)
      echo " - $WEIGHT_DIR/mambair/MambaIRv2_SRx4.pth"
      ;;
    swinir)
      echo " - $WEIGHT_DIR/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
      echo " - $WEIGHT_DIR/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
      echo " - $WEIGHT_DIR/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
      echo " - $WEIGHT_DIR/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
      echo " - $WEIGHT_DIR/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
      ;;
    swin2sr)
      echo " - $WEIGHT_DIR/swin2sr/Swin2SR_ClassicalSR_X4_64.pth"
      echo " - $WEIGHT_DIR/swin2sr/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"
      echo " - $WEIGHT_DIR/swin2sr/Swin2SR_CompressedSR_X4_48.pth"
      ;;
    dat)
      echo " - $WEIGHT_DIR/dat/DAT_x4.pth"
      echo " - $WEIGHT_DIR/dat/DAT_S_x4.pth"
      echo " - $WEIGHT_DIR/dat/DAT_2_x4.pth"
      echo " - $WEIGHT_DIR/dat/DAT_light_x4.pth"
      ;;
  esac
done
