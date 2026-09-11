#!/bin/bash
# THE combination: suffix-quality drafting (measured A=7.33 at k=7, 12.57 at k=15)
# fed into the batched-column MMA chain verify (measured 388 ms/round) instead of
# the tree verify that made it a 0.33x loss. Both halves measured separately;
# this is the first run with them connected.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/suffix.jsonl; : > $OUT
FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1 CAMELID_SPEC_TREE=0
      CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1)
run() { # drafter drafts
  echo "### $1 k=$2 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${FAST[@]}" \
    "$BIN" bench-speculative "$Q4" --drafter "$1" --draft-tokens "$2" --workload "sfx_$1_k$2" \
      --prompt-file "$HOME/prompts/agentic_4k.txt" --max-tokens 96 --warmup \
    2>>$HOME/suffix.stderr | sed "s/^/{\"drafter\":\"$1\",\"drafts\":$2,\"record\":/; s/\$/}/" >> $OUT
  echo "    done $(date +%H:%M:%S)" >&2
}
run suffix 7
run suffix 15
run suffix 11
run ngram  7
echo "SUFFIX done" >&2
