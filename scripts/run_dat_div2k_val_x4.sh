#!/usr/bin/env bash
set -euo pipefail

# DAT x4 inference on DIV2K val LR (paths from $VAL_JSON).
# - Requires you to download a DAT pretrained model and point WEIGHT at it.
# - Optional "use_chop" can reduce memory on large images.

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

REPO_DIR="$SR_DIR/repos/DAT"

NAME="${NAME:-DAT_x4_DIV2K_val}"
WEIGHT="${WEIGHT:-$REPO_DIR/experiments/pretrained_models/DAT/DAT_x4.pth}"
USE_CHOP="${USE_CHOP:-false}" # true/false

if [[ ! -f "$WEIGHT" ]]; then
  echo "Missing weight file: $WEIGHT" >&2
  echo "Download DAT_x4.pth and set WEIGHT=/path/to/DAT_x4.pth" >&2
  exit 1
fi

OPT_FILE="$(mktemp -t dat_div2k_val_XXXX.yml)"
trap 'rm -f "$OPT_FILE"' EXIT

cat >"$OPT_FILE" <<YML
name: ${NAME}
model_type: DATModel
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
  type: DAT
  upscale: 4
  in_chans: 3
  img_size: 64
  img_range: 1.
  split_size: [8, 32]
  depth: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  expansion_factor: 4
  resi_connection: '1conv'

path:
  pretrain_network_g: "${WEIGHT}"
  strict_load_g: true

val:
  save_img: true
  suffix: 'x4'
  use_chop: ${USE_CHOP}
YML

cd "$REPO_DIR"
python3 basicsr/test.py -opt "$OPT_FILE"

echo "Results saved under: $REPO_DIR/results/${NAME}/visualization/DIV2K_val"
