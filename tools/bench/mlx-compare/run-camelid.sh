#!/usr/bin/env bash
# Run the Camelid generation benchmark for one prompt file.
# Env: MODEL (gguf path), CAMELID_BIN (optional), MAX_TOKENS, ITERS, OUT (bundle dir)
# Arg 1: prompt file
set -euo pipefail
PROMPT_FILE="${1:?usage: run-camelid.sh <prompt_file>}"
MODEL="${MODEL:?set MODEL to a .gguf path}"
MAX_TOKENS="${MAX_TOKENS:-128}"
ITERS="${ITERS:-10}"
OUT="${OUT:?set OUT to the bundle dir}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${CAMELID_BIN:-$REPO_ROOT/target/release/camelid}"
[ -x "$BIN" ] || { echo "camelid binary not found at $BIN (build it or set CAMELID_BIN)" >&2; exit 1; }

label="$(basename "$PROMPT_FILE" .txt)"
mkdir -p "$OUT/raw/camelid"
commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "[camelid] $label: max_tokens=$MAX_TOKENS iters=$ITERS (1 warmup)"
/usr/bin/time -l env CAMELID_COMMIT="$commit" "$BIN" bench-generate "$MODEL" \
  --prompt-file "$PROMPT_FILE" --max-tokens "$MAX_TOKENS" --temperature 0 \
  --warmup --iterations "$ITERS" --json \
  >"$OUT/raw/camelid/$label.jsonl" 2>"$OUT/raw/camelid/$label.time"
echo "[camelid] $label done -> $OUT/raw/camelid/$label.jsonl"
