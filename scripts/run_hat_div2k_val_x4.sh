#!/usr/bin/env bash
set -euo pipefail

# HAT x4 inference on DIV2K val LR (paths from $VAL_JSON).
# - Requires you to download a HAT pretrained model and point WEIGHT at it.
# - Tile mode is enabled by default to reduce GPU memory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$SR_DIR/../.." && pwd)"

VAL_JSON="${VAL_JSON:-$WORKSPACE_DIR/data/val.json}"

LR_DIR="$(
  VAL_JSON="$VAL_JSON" python3 - <<'PY'
import json, os
val_json = os.environ["VAL_JSON"]
d = json.load(open(val_json))
dirs = sorted({os.path.dirname(e["lr"]) for e in d})
if len(dirs) != 1:
    raise SystemExit(f"Expected 1 lr dir in {val_json}, got: {dirs}")
print(dirs[0])
PY
)"

REPO_DIR="$SR_DIR/repos/HAT"

NAME="${NAME:-HAT_SRx4_ImageNet-LR_DIV2K_val}"
WEIGHT="${WEIGHT:-$REPO_DIR/experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth}"

# Tile mode for limited GPU memory.
# HAT uses window_size=16; keeping tile_size a multiple of 16 is recommended.
TILE_SIZE="${TILE_SIZE:-512}"
TILE_PAD="${TILE_PAD:-32}"

if [[ ! -f "$WEIGHT" ]]; then
  echo "Missing weight file: $WEIGHT" >&2
  echo "Download HAT_SRx4_ImageNet-pretrain.pth and set WEIGHT=/path/to/HAT_SRx4_ImageNet-pretrain.pth" >&2
  exit 1
fi

OPT_FILE="$(mktemp -t hat_div2k_val_XXXX.yml)"
trap 'rm -f "$OPT_FILE"' EXIT

cat >"$OPT_FILE" <<YML
name: ${NAME}
model_type: HATModel
scale: 4
num_gpu: 1  # set num_gpu: 0 for cpu mode
manual_seed: 0

tile:
  tile_size: ${TILE_SIZE}
  tile_pad: ${TILE_PAD}

datasets:
  test_1:
    name: DIV2K_val
    type: SingleImageDataset
    dataroot_lq: "${LR_DIR}"
    io_backend:
      type: disk

network_g:
  type: HAT
  upscale: 4
  in_chans: 3
  img_size: 64
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'

path:
  pretrain_network_g: "${WEIGHT}"
  strict_load_g: true
  param_key_g: 'params_ema'

val:
  save_img: true
  suffix: ~
YML

cd "$REPO_DIR"
python3 hat/test.py -opt "$OPT_FILE"

echo "Results saved under: $REPO_DIR/results/${NAME}/visualization/DIV2K_val"
