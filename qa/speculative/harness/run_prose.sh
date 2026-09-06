#!/bin/bash
# What an MTP head actually has to beat.
#
# The suffix drafter already reaches A=7.33 at 91% acceptance on AGENTIC text, so
# a trained head buys little there. Its real value is PROSE: a training-free
# drafter can only propose tokens the context already contains, so on
# non-repetitive text it should collapse toward ~1.0x. That floor is the number
# the head must beat, and nobody has measured it on this lane.
# depth_4k.txt is repo prose (deduped, code fences stripped), matched in length
# to the agentic prompt so only the CONTENT differs.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/prose.jsonl; : > $OUT
FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1 CAMELID_SPEC_TREE=0
      CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1)
run() { # drafter prompt
  echo "### $1 $2 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${FAST[@]}" \
    "$BIN" bench-speculative "$Q4" --drafter "$1" --draft-tokens 7 --workload "pr_$1_$2" \
      --prompt-file "$HOME/prompts/$2.txt" --max-tokens 96 --warmup \
    2>>$HOME/prose.stderr | sed "s/^/{\"drafter\":\"$1\",\"prompt\":\"$2\",\"record\":/; s/\$/}/" >> $OUT
}
run suffix depth_4k
run ngram  depth_4k
run suffix agentic_4k
echo "PROSE done" >&2
