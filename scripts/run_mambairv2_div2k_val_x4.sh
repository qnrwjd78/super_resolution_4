#!/usr/bin/env bash
set -euo pipefail

# MambaIRv2 x4 inference on DIV2K val LR (paths from $VAL_JSON).
# - Requires you to download a MambaIRv2 pretrained model and point WEIGHT at it.
# - MambaIRv2Model uses partitioned testing internally (memory friendly).

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

REPO_DIR="$SR_DIR/repos/MambaIR"

NAME="${NAME:-MambaIRv2_x4_DIV2K_val}"
WEIGHT="${WEIGHT:-$REPO_DIR/experiments/pretrained_models/mambairv2_classicSR_Base_x4.pth}"

if [[ ! -f "$WEIGHT" ]]; then
  echo "Missing weight file: $WEIGHT" >&2
  echo "Download a MambaIRv2 x4 SR weight and set WEIGHT=/path/to/weight.pth" >&2
  exit 1
fi

OPT_FILE="$(mktemp -t mambairv2_div2k_val_XXXX.yml)"
trap 'rm -f "$OPT_FILE"' EXIT

cat >"$OPT_FILE" <<YML
name: ${NAME}
model_type: MambaIRv2Model
scale: 4
num_gpu: 1
manual_seed: 10

datasets:
  test_1:
    name: DIV2K_val
    type: SingleImageDataset
    dataroot_lq: "${LR_DIR}"
    io_backend:
      type: disk

network_g:
  type: MambaIRv2
  upscale: 4
  in_chans: 3
  img_size: 64
  img_range: 1.
  embed_dim: 174
  d_state: 16
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  window_size: 16
  inner_rank: 64
  num_tokens: 128
  convffn_kernel_size: 5
  mlp_ratio: 2.
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'

path:
  pretrain_network_g: "${WEIGHT}"
  strict_load_g: true

val:
  save_img: true
  suffix: ~
YML

cd "$REPO_DIR"
python3 basicsr/test.py -opt "$OPT_FILE"

echo "Results saved under: $REPO_DIR/results/${NAME}/visualization/DIV2K_val"
