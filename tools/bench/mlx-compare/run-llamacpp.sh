#!/usr/bin/env bash
# Reference baseline: llama.cpp via llama-bench (prefill pp + decode tg tok/s).
# llama-bench does not expose per-request TTFT or peak RSS, so this is a
# throughput reference only, not a full-schema row.
# Env: MODEL (gguf path), MAX_TOKENS, OUT (bundle dir)
# Arg 1: prompt file (used only for its token count target via -p)
set -euo pipefail
PROMPT_FILE="${1:?usage: run-llamacpp.sh <prompt_file>}"
MODEL="${MODEL:?set MODEL to a .gguf path}"
MAX_TOKENS="${MAX_TOKENS:-128}"
OUT="${OUT:?set OUT to the bundle dir}"

command -v llama-bench >/dev/null 2>&1 || { echo "llama-bench not found; skipping" >&2; exit 0; }

label="$(basename "$PROMPT_FILE" .txt)"
mkdir -p "$OUT/raw/llamacpp"

# Approximate prompt token count from words (reference only).
words="$(wc -w < "$PROMPT_FILE" | tr -d ' ')"
pp=$(( words * 4 / 3 )); [ "$pp" -lt 1 ] && pp=1

echo "[llamacpp] $label: pp=$pp tg=$MAX_TOKENS (reference)"
llama-bench -m "$MODEL" -p "$pp" -n "$MAX_TOKENS" -o json \
  >"$OUT/raw/llamacpp/$label.json" 2>"$OUT/raw/llamacpp/$label.log" || {
    echo "[llamacpp] $label failed (see log)" >&2; exit 0; }
echo "[llamacpp] $label done -> $OUT/raw/llamacpp/$label.json"
