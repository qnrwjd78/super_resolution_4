#!/usr/bin/env bash
set -euo pipefail

# FLUX.2 [klein-base-9b] image-to-image inference launcher.
#
# This script is intentionally fixed to:
#   - model: flux.2-klein-base-9b
#   - mode : i2i (input image(s) required)
#
# Examples:
#   ./02_base_inference.sh \
#     --prompt "turn this into watercolor illustration" \
#     --input-image /abs/path/input.png
#
#   ./02_base_inference.sh \
#     --prompt "cinematic relight while preserving identity" \
#     --input-image /abs/path/ref_a.png \
#     --input-image /abs/path/ref_b.png \
#     --match-image-size 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_NAME="flux.2-klein-base-9b"
MODEL_ENV_VAR="KLEIN_9B_BASE_MODEL_PATH"
MODEL_FILE="flux-2-klein-base-9b.safetensors"
MODEL_SUBDIR="${MODEL_NAME//./}"

WEIGHT_DIR="${WEIGHT_DIR:-$STAGE2_DIR/weights}"
FLUX_REPO_DIR="${FLUX_REPO_DIR:-$STAGE2_DIR/repos/flux2}"
MODEL_PATH="${MODEL_PATH:-$WEIGHT_DIR/$MODEL_SUBDIR/$MODEL_FILE}"
AE_PATH="${AE_PATH:-$WEIGHT_DIR/$MODEL_SUBDIR/ae.safetensors}"

PROMPT="${PROMPT:-}"
SINGLE_EVAL="${SINGLE_EVAL:-1}"
CPU_OFFLOADING="${CPU_OFFLOADING:-0}"
WIDTH="${WIDTH:-}"
HEIGHT="${HEIGHT:-}"
NUM_STEPS="${NUM_STEPS:-}"
GUIDANCE="${GUIDANCE:-}"
MATCH_IMAGE_SIZE="${MATCH_IMAGE_SIZE:-0}"
UPSAMPLE_PROMPT_MODE="${UPSAMPLE_PROMPT_MODE:-}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-}"

declare -a INPUT_IMAGES_ARR=()
EXTRA_ARGS=()

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./02_base_inference.sh [options] [-- <extra fire args>]

Options:
  --prompt "<text>"             Prompt text (required)
  --input-image <path>          Input image path (repeatable, required)
  --input-images <csv_paths>    Comma-separated input image paths (optional alternative)
  --match-image-size <index>    0-based index to copy output size from input image (default: 0)
  --interactive                 Keep CLI interactive (default is one-shot single eval)
  --cpu-offloading              Enable cpu offloading
  --width <int>
  --height <int>
  --num-steps <int>
  --guidance <float>
  --upsample-mode <mode>        none | local | openrouter
  --openrouter-model <name>     OpenRouter model name
  --repo-dir <path>             Override flux2 repo path
  --weights-dir <path>          Override stage2 weights root
  --model-path <path>           Override model weight path
  --ae-path <path>              Override AE weight path
  -h, --help                    Show this help

Env overrides:
  WEIGHT_DIR, FLUX_REPO_DIR, MODEL_PATH, AE_PATH, PROMPT
EOF
}

has() { command -v "$1" >/dev/null 2>&1; }

run_flux2_python() {
  if has flux2; then
    flux2 env PYTHONPATH=src python "$@"
  elif has conda; then
    conda run -n flux2 --no-capture-output env PYTHONPATH=src python "$@"
  elif has python3; then
    PYTHONPATH=src python3 "$@"
  elif has python; then
    PYTHONPATH=src python "$@"
  else
    echo "ERROR: python runtime not found." >&2
    exit 1
  fi
}

add_csv_images() {
  local csv="$1"
  IFS=',' read -r -a _tmp <<< "$csv"
  for p in "${_tmp[@]}"; do
    p="${p#"${p%%[![:space:]]*}"}"
    p="${p%"${p##*[![:space:]]}"}"
    [[ -n "$p" ]] && INPUT_IMAGES_ARR+=("$p")
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --input-image)
      INPUT_IMAGES_ARR+=("$2")
      shift 2
      ;;
    --input-images)
      add_csv_images "$2"
      shift 2
      ;;
    --match-image-size)
      MATCH_IMAGE_SIZE="$2"
      shift 2
      ;;
    --interactive)
      SINGLE_EVAL=0
      shift
      ;;
    --cpu-offloading)
      CPU_OFFLOADING=1
      shift
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --guidance)
      GUIDANCE="$2"
      shift 2
      ;;
    --upsample-mode)
      UPSAMPLE_PROMPT_MODE="$2"
      shift 2
      ;;
    --openrouter-model)
      OPENROUTER_MODEL="$2"
      shift 2
      ;;
    --repo-dir)
      FLUX_REPO_DIR="$2"
      shift 2
      ;;
    --weights-dir)
      WEIGHT_DIR="$2"
      MODEL_PATH="$WEIGHT_DIR/$MODEL_SUBDIR/$MODEL_FILE"
      AE_PATH="$WEIGHT_DIR/$MODEL_SUBDIR/ae.safetensors"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --ae-path)
      AE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "$PROMPT" ]] || die "--prompt is required."
[[ "${#INPUT_IMAGES_ARR[@]}" -gt 0 ]] || die "I2I mode requires at least one --input-image (or --input-images)."
[[ "$MATCH_IMAGE_SIZE" =~ ^[0-9]+$ ]] || die "--match-image-size must be a non-negative integer."
(( MATCH_IMAGE_SIZE < ${#INPUT_IMAGES_ARR[@]} )) || die "--match-image-size is out of range for provided input images."

for p in "${INPUT_IMAGES_ARR[@]}"; do
  [[ -f "$p" ]] || die "input image not found: $p"
done

INPUT_IMAGES_CSV="$(IFS=,; echo "${INPUT_IMAGES_ARR[*]}")"

[[ -d "$FLUX_REPO_DIR" ]] || die "flux2 repo not found: $FLUX_REPO_DIR"
[[ -f "$MODEL_PATH" ]] || die "model weight not found: $MODEL_PATH
Hint: MODELS=\"$MODEL_NAME\" \"$SCRIPT_DIR/01_download_weights.sh\""
[[ -f "$AE_PATH" ]] || die "AE weight not found: $AE_PATH
Hint: MODELS=\"$MODEL_NAME\" \"$SCRIPT_DIR/01_download_weights.sh\""

# The upstream CLI writes to ./output under repo root.
if [[ -d "$FLUX_REPO_DIR/output" ]]; then
  [[ -w "$FLUX_REPO_DIR/output" ]] || die "cannot write to $FLUX_REPO_DIR/output (check ownership/permissions)"
else
  [[ -w "$FLUX_REPO_DIR" ]] || die "cannot create $FLUX_REPO_DIR/output (check repo ownership/permissions)"
fi

export "${MODEL_ENV_VAR}=${MODEL_PATH}"
export "AE_MODEL_PATH=${AE_PATH}"

CLI_ARGS=(scripts/cli.py "--model_name=$MODEL_NAME")
CLI_ARGS+=("--prompt=$PROMPT")
CLI_ARGS+=("--input_images=$INPUT_IMAGES_CSV")
CLI_ARGS+=("--match_image_size=$MATCH_IMAGE_SIZE")

if [[ "$SINGLE_EVAL" == "1" ]]; then
  CLI_ARGS+=("--single_eval=True")
fi
if [[ "$CPU_OFFLOADING" == "1" ]]; then
  CLI_ARGS+=("--cpu_offloading=True")
fi
if [[ -n "$WIDTH" ]]; then
  CLI_ARGS+=("--width=$WIDTH")
fi
if [[ -n "$HEIGHT" ]]; then
  CLI_ARGS+=("--height=$HEIGHT")
fi
if [[ -n "$NUM_STEPS" ]]; then
  CLI_ARGS+=("--num_steps=$NUM_STEPS")
fi
if [[ -n "$GUIDANCE" ]]; then
  CLI_ARGS+=("--guidance=$GUIDANCE")
fi
if [[ -n "$UPSAMPLE_PROMPT_MODE" ]]; then
  CLI_ARGS+=("--upsample_prompt_mode=$UPSAMPLE_PROMPT_MODE")
fi
if [[ -n "$OPENROUTER_MODEL" ]]; then
  CLI_ARGS+=("--openrouter_model=$OPENROUTER_MODEL")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CLI_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "MODEL_NAME : $MODEL_NAME"
echo "MODEL_PATH : $MODEL_PATH"
echo "AE_PATH    : $AE_PATH"
echo "REPO_DIR   : $FLUX_REPO_DIR"
echo "I2I_INPUTS : $INPUT_IMAGES_CSV"
echo "PROMPT     : $PROMPT"
echo

(
  cd "$FLUX_REPO_DIR"
  run_flux2_python "${CLI_ARGS[@]}"
)
