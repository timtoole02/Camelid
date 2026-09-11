#!/bin/bash
# The drafter is now the binding constraint: the n-gram drafter proposes only
# 3.09 tokens/round of the 7 asked, so the round is paid in full for a third of
# a window. Two things to try before funding a trained head, both already in the
# tree and reachable by env:
#   * CAMELID_SPEC_TREE=1 -> the TREE lane, which uses SuffixDecodingDrafter --
#     built for exactly this (frequency-weighted suffix tree over prompt+output,
#     literature-best on agentic replay) instead of a fixed-length n-gram match.
#   * wider n-gram match bounds, in case the default min/max is what caps it.
# Depth sweep too: at 4k the k x prefix-KV term is 47% of round bytes, so whether
# the round grows with depth decides if shared-prefix verify is the next lever.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/drafter.jsonl; : > $OUT
FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1
      CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1)
run() { # label extra-env... -- drafts prompt
  local label=$1; shift
  local -a extra=()
  while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  local k=$1 p=$2
  echo "### $label k=$k $p $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${FAST[@]}" "${extra[@]}" \
    "$BIN" bench-speculative "$Q4" --drafter ngram --draft-tokens "$k" --workload "$label" \
      --prompt-file "$HOME/prompts/$p.txt" --max-tokens 96 --warmup \
    2>>$HOME/drafter.stderr | sed "s/^/{\"label\":\"$label\",\"drafts\":$k,\"prompt\":\"$p\",\"record\":/; s/\$/}/" >> $OUT
  echo "    done $(date +%H:%M:%S)" >&2
}
# A: the suffix/tree drafter at two widths
run tree_k7  CAMELID_SPEC_TREE=1 CAMELID_SPEC_TREE_GATE=0 -- 7  agentic_4k
run tree_k15 CAMELID_SPEC_TREE=1 CAMELID_SPEC_TREE_GATE=0 -- 15 agentic_4k
# B: n-gram with a shorter minimum match, in case the default floor is the cap
run ngram_min2 CAMELID_SPEC_TREE=0 CAMELID_SPEC_NGRAM_MIN=2 -- 7 agentic_4k
# C: depth sweep on the winning kernel lane, fixed width
for p in ag_512 ag_1024 ag_2048; do run depth_$p CAMELID_SPEC_TREE=0 -- 7 $p; done
echo "DRAFTER done" >&2
