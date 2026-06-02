#!/usr/bin/env bash
# Run the MLX-LM generation benchmark for one prompt file.
# Env: MLX_MODEL (hf id or path), MLX_VENV (venv dir), HF_HOME (optional cache),
#      MAX_TOKENS, ITERS, OUT (bundle dir)
# Arg 1: prompt file
set -euo pipefail
PROMPT_FILE="${1:?usage: run-mlx-lm.sh <prompt_file>}"
MLX_MODEL="${MLX_MODEL:?set MLX_MODEL (e.g. mlx-community/Llama-3.2-3B-Instruct-8bit)}"
MLX_VENV="${MLX_VENV:?set MLX_VENV to the mlx venv dir}"
MAX_TOKENS="${MAX_TOKENS:-128}"
ITERS="${ITERS:-10}"
OUT="${OUT:?set OUT to the bundle dir}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
label="$(basename "$PROMPT_FILE" .txt)"
mkdir -p "$OUT/raw/mlx-lm"

# shellcheck disable=SC1091
source "$MLX_VENV/bin/activate"
[ -n "${HF_HOME:-}" ] && export HF_HOME

echo "[mlx-lm] $label: max_tokens=$MAX_TOKENS iters=$ITERS (1 warmup)"
/usr/bin/time -l python3 "$HERE/lib/mlx_generate.py" --model "$MLX_MODEL" \
  --prompt-file "$PROMPT_FILE" --max-tokens "$MAX_TOKENS" --temperature 0 \
  --warmup --iterations "$ITERS" \
  >"$OUT/raw/mlx-lm/$label.jsonl" 2>"$OUT/raw/mlx-lm/$label.time"
echo "[mlx-lm] $label done -> $OUT/raw/mlx-lm/$label.jsonl"
