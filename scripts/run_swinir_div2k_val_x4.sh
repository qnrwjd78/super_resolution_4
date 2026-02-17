#!/usr/bin/env bash
set -euo pipefail

# SwinIR classical SR x4 inference on DIV2K val (paths from $VAL_JSON).
# - Weight is auto-downloaded by SwinIR if missing (by basename).
# - "training_patch_size" must match the chosen weight (see SwinIR README).

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

HR_DIR="$(
  VAL_JSON="$VAL_JSON" python3 - <<'PY'
import json, os
val_json = os.environ["VAL_JSON"]
d = json.load(open(val_json))
dirs = sorted({os.path.dirname(e["hr"]) for e in d})
if len(dirs) != 1:
    raise SystemExit(f"Expected 1 hr dir in {val_json}, got: {dirs}")
print(dirs[0])
PY
)"

REPO_DIR="$SR_DIR/repos/SwinIR"

TASK="${TASK:-classical_sr}" # classical_sr / lightweight_sr / real_sr / ...
SCALE="${SCALE:-4}"
TRAINING_PATCH_SIZE="${TRAINING_PATCH_SIZE:-48}"
MODEL_PATH="${MODEL_PATH:-model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth}"

# Optional tiling to reduce GPU memory (must be a multiple of window_size=8).
TILE="${TILE:-}"               # e.g. 512; leave empty to disable
TILE_OVERLAP="${TILE_OVERLAP:-32}"

cd "$REPO_DIR"

args=(
  --task "$TASK"
  --scale "$SCALE"
  --training_patch_size "$TRAINING_PATCH_SIZE"
  --model_path "$MODEL_PATH"
  --folder_lq "$LR_DIR"
  --folder_gt "$HR_DIR"
  --tile_overlap "$TILE_OVERLAP"
)
if [[ -n "$TILE" ]]; then
  args+=(--tile "$TILE")
fi

python3 main_test_swinir.py "${args[@]}"

echo "Results saved under: $REPO_DIR/results/swinir_${TASK}_x${SCALE}"
