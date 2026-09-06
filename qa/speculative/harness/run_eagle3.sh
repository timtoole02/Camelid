#!/bin/bash
# Pinned Llama 3.2 3B Q4_K_M + thoughtworks EAGLE-3 resident-Metal sweep.
# The benchmark itself fails nonzero on target-stream divergence; pipefail keeps
# an optional caller-side tee from hiding that failure.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$ROOT/target/release/camelid}
MODEL=${MODEL:-$HOME/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf}
EAGLE3=${EAGLE3:-$HOME/models/Llama-3.2-3B-Instruct-Eagle3}
PROMPT_FILE=${PROMPT_FILE:-$ROOT/qa/speculative/harness/prompts/eagle3_prose.txt}
OUT_DIR=${OUT_DIR:-$HOME/camelid-overnight/receipts}
MAX_TOKENS=${MAX_TOKENS:-96}
CHAT=${CHAT:-0}

# Command substitution deliberately removes the file's terminal newline. The
# committed tuning prompt was measured without one; that one byte changes the
# continuation and therefore the learned-head acceptance distribution.
PROMPT=$(<"$PROMPT_FILE")
if [[ "$CHAT" == 1 ]]; then
  PROMPT_MODE=chat
  CHAT_ARGS=(--chat)
else
  PROMPT_MODE=raw
  CHAT_ARGS=()
fi

if (($# == 0)); then
  GAMMAS=(1 2 3 4 5 6)
else
  GAMMAS=("$@")
fi

mkdir -p "$OUT_DIR"
for gamma in "${GAMMAS[@]}"; do
  workload="llama32-3b-q4km-eagle3-resident-${PROMPT_MODE}-gamma${gamma}-prose${MAX_TOKENS}"
  json="$OUT_DIR/$workload.jsonl"
  stderr="$OUT_DIR/$workload.stderr"
  env \
    CAMELID_METAL_LINEAR=1 \
    CAMELID_METAL_Q8=1 \
    CAMELID_METAL_RESIDENT_DECODE=1 \
    CAMELID_METAL_RESIDENT_PREFILL=1 \
    CAMELID_METAL_WIRE=1 \
    CAMELID_METAL_WIRE_NSG8=1 \
    CAMELID_METAL_F32Y=1 \
    CAMELID_METAL_NOCOPY=1 \
    CAMELID_METAL_KQUANT=1 \
    CAMELID_KQUANT_V2=1 \
    CAMELID_KQUANT_MMA=1 \
    CAMELID_SPEC_TREE=0 \
    "$BIN" bench-eagle3 "$MODEL" \
      --eagle3 "$EAGLE3" \
      --draft-tokens "$gamma" \
      --prompt "$PROMPT" \
      "${CHAT_ARGS[@]}" \
      --max-tokens "$MAX_TOKENS" \
      --workload "$workload" \
      >"$json" 2>"$stderr"
  jq -e '
    .lossless == true and
    .first_divergent_generated_token_index == -1 and
    .resident_verify_rounds > 0 and
    .cpu_verify_rounds == 0
  ' "$json" >/dev/null
  jq -c '{draft_tokens,plain_tokens_per_second,eagle3_tokens_per_second,speedup,accept_rate,mean_emitted_tokens_per_round,lossless,binary_sha256}' "$json"
done
