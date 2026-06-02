#!/usr/bin/env bash
# Verify Camelid pipeline-parallel correctness on one machine: split a model across
# two local processes (master 0..k, worker k..N) and confirm it generates and that
# each node holds only part of the weights. Compare the printed text to a single-node
# run of the same prompt (greedy ⇒ should match exactly).
#
# Usage: loopback-verify.sh <model.gguf> [layers_total] [split]
set -euo pipefail
MODEL="${1:?usage: loopback-verify.sh <model.gguf> [layers_total] [split]}"
TOTAL="${2:-28}"      # transformer layers in the model (Llama-3.2-3B = 28)
SPLIT="${3:-14}"      # master owns 0..SPLIT, worker owns SPLIT..TOTAL
PROMPT="${PROMPT:-Explain what a Rust borrow checker does in two sentences.}"
MAX_TOKENS="${MAX_TOKENS:-48}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${CAMELID_BIN:-$REPO_ROOT/target/release/camelid}"
[ -x "$BIN" ] || { echo "camelid binary not found at $BIN (build it or set CAMELID_BIN)" >&2; exit 1; }

WORK="$(mktemp -d)"
echo "[loopback] master 0..$SPLIT  worker $SPLIT..$TOTAL  model=$MODEL"

/usr/bin/time -l "$BIN" distribute-worker "$MODEL" \
  --addr 127.0.0.1:5005 --layers "$SPLIT..$TOTAL" --master-addr 127.0.0.1:5006 \
  >"$WORK/worker.out" 2>"$WORK/worker.time" &
WPID=$!

"$BIN" distribute-master "$MODEL" \
  --worker-addr 127.0.0.1:5005 --layers "0..$SPLIT" --addr 127.0.0.1:5006 \
  --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" >"$WORK/master.out" 2>&1 || true
sleep 1; kill "$WPID" 2>/dev/null || true

echo "--- distributed output ---"
sed -n '/Encoded prompt/,$p' "$WORK/master.out"
echo "--- per-node peak RSS ---"
grep -i "maximum resident" "$WORK/worker.time" | awk '{printf "  worker: %.2f GB\n", $1/1073741824}'
echo "--- compare to single node (greedy should match): ---"
echo "  $BIN bench-generate $MODEL --prompt \"$PROMPT\" --max-tokens $MAX_TOKENS --temperature 0"
echo "artifacts in $WORK"
