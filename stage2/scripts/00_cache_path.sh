#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source ./00_cache_path.sh
#
# Notes:
# - This script is intended to be sourced so exported variables persist
#   in the current shell.
# - Default cache root is /sr/.hf_cache (override via HF_CACHE_ROOT).

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/sr/.hf_cache}"
HF_CACHE_HUB="${HF_CACHE_HUB:-$HF_CACHE_ROOT/hub}"
HF_CACHE_TRANSFORMERS="${HF_CACHE_TRANSFORMERS:-$HF_CACHE_ROOT/transformers}"
HF_TMPDIR="${HF_TMPDIR:-/sr/.tmp}"
HF_XDG_CACHE_HOME="${HF_XDG_CACHE_HOME:-/sr/.cache}"

unset HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_HUB_CACHE

mkdir -p "$HF_CACHE_HUB" "$HF_CACHE_TRANSFORMERS" "$HF_TMPDIR" "$HF_XDG_CACHE_HOME"

export HF_HOME="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_HUB"
export HF_HUB_CACHE="$HF_CACHE_HUB"
export TRANSFORMERS_CACHE="$HF_CACHE_TRANSFORMERS"
export XDG_CACHE_HOME="$HF_XDG_CACHE_HOME"
export TMPDIR="$HF_TMPDIR"

echo "Configured Hugging Face cache paths:"
echo "  HF_HOME=$HF_HOME"
echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "  TMPDIR=$TMPDIR"


# source ./00_cache_path.sh 이걸로 실행