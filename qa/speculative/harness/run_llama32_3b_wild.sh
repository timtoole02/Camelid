#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 BIN TARGET.gguf EAGLE3_DIR [RECEIPT.jsonl]" >&2
  exit 2
fi

bin=$1
target=$2
eagle3=$3
receipt=${4:-/dev/stdout}
cycles=${CYCLES:-24}
max_tokens=${MAX_TOKENS:-96}
trace=${TRACE:-0}
expected_binary_sha256=${EXPECTED_BINARY_SHA256:-}

# 24 cycles tokenizes to the 115-token ceiling prompt. 450 cycles tokenizes to
# the deepest valid receipt (1,819 prompt tokens). Keep this reproducer inside
# the exact measured prompt/decode envelope; the unreceipted 4k stress shape is
# deliberately unavailable here.
if [[ ! $cycles =~ ^[0-9]+$ ]]; then
  echo "CYCLES must be a decimal integer" >&2
  exit 2
fi
if [[ ! $max_tokens =~ ^[0-9]+$ ]]; then
  echo "MAX_TOKENS must be a decimal integer" >&2
  exit 2
fi
cycles=$((10#$cycles))
max_tokens=$((10#$max_tokens))
if (( cycles < 2 || cycles > 450 )); then
  echo "CYCLES must be in 2..450; 450 is the deepest measured width-16 prompt" >&2
  exit 2
fi
if (( max_tokens < 16 || max_tokens > 96 )); then
  echo "MAX_TOKENS must be in 16..96; 96 is the measured decode envelope" >&2
  exit 2
fi
if [[ $trace != 0 && $trace != 1 ]]; then
  echo "TRACE must be 0 or 1" >&2
  exit 2
fi

prompt='Continue the exact repeating pattern below. Output only the pattern continuation and never explain:'
for ((i = 0; i < cycles; i++)); do
  prompt+=' alpha beta gamma delta'
done
prompt+=' alpha beta'

trace_env=()
if [[ $trace == 1 ]]; then
  trace_env=(CAMELID_SPEC_VERIFY_TRACE=1 CAMELID_KQUANT_V4_TRACE=1)
fi

if [[ $receipt == /dev/stdout ]]; then
  receipt_dir=${TMPDIR:-/tmp}
else
  receipt_dir=$(dirname "$receipt")
  mkdir -p "$receipt_dir"
fi
tmp_receipt=$(mktemp "$receipt_dir/.llama32-3b-wild.XXXXXX")
trap 'rm -f "$tmp_receipt"' EXIT

# Start with a clean environment so ambient experimental variables cannot
# silently change the head coverage, dispatch route, or timing. TRACE=1 is a
# diagnostic run: stderr carries the q4/q6-wide and attention ownership proof.
env -i \
  PATH="$PATH" \
  TMPDIR="${TMPDIR:-/tmp}" \
  "${trace_env[@]}" \
  CAMELID_EAGLE3_LM_HEAD_Q8=1 \
  CAMELID_METAL_LINEAR=1 \
  CAMELID_METAL_Q8=1 \
  CAMELID_METAL_RESIDENT_DECODE=1 \
  CAMELID_METAL_RESIDENT_PREFILL=1 \
  CAMELID_METAL_ATTN2=1 \
  CAMELID_METAL_ATTN_BATCH_K=1 \
  CAMELID_METAL_KV_DTYPE=f16 \
  CAMELID_METAL_WIRE=1 \
  CAMELID_METAL_WIRE_NSG8=1 \
  CAMELID_METAL_F32Y=1 \
  CAMELID_METAL_NOCOPY=1 \
  CAMELID_METAL_KQUANT=1 \
  CAMELID_KQUANT_V2=1 \
  CAMELID_KQUANT_V3=1 \
  CAMELID_KQUANT_V4=1 \
  CAMELID_KQUANT_MMA=1 \
  CAMELID_SPEC_TREE=1 \
  CAMELID_NO_OPEN=1 \
  "$bin" bench-eagle3 "$target" \
    --eagle3 "$eagle3" \
    --draft-tokens 15 \
    --tree-nodes 16 \
    --tree-topk 4 \
    --tree-expansions 4 \
    --suffix-first \
    --prompt "$prompt" \
    --max-tokens "$max_tokens" \
    --workload "llama32-3b-wild-cycle-${cycles}" >"$tmp_receipt"

expected_prompt_tokens=$((4 * cycles + 19))
jq -e \
  --argjson prompt_tokens "$expected_prompt_tokens" \
  --argjson max_tokens "$max_tokens" '
    .lossless == true and
    .first_divergent_generated_token_index == -1 and
    .prompt_tokens == $prompt_tokens and
    .max_tokens == $max_tokens and
    .draft_mode == "suffix_then_dynamic_tree" and
    .resident_verify_rounds > 0 and
    .cpu_verify_rounds == 0 and
    .resident_normal_steps == 0 and
    .suffix_rounds > 0 and
    .dynamic_tree_rounds == 0 and
    .drafted == .accepted_drafts and
    .accept_rate == 1 and
    .eagle3_tokens_per_second > 0
  ' "$tmp_receipt" >/dev/null

if [[ -n $expected_binary_sha256 ]]; then
  jq -e --arg sha "$expected_binary_sha256" '.binary_sha256 == $sha' \
    "$tmp_receipt" >/dev/null
fi

if [[ $receipt == /dev/stdout ]]; then
  command cat "$tmp_receipt"
  rm -f "$tmp_receipt"
else
  mv "$tmp_receipt" "$receipt"
fi
trap - EXIT
