#!/bin/bash
# Nail down the v1 K-quant losslessness failure.
#
# The exact config that reported lossless=false (divergence at token 58) was:
#   CAMELID_KQUANT_V2=0, ngram drafter, 7 drafts, ag_256.
# It now reports true. Two candidates: (a) the split-K change fixed it as a side
# effect -- that edit is NOT behind the V2 gate, so the v1 lane gets it too; or
# (b) it is intermittent.
#
# CAMELID_METAL_ATTN_SPLITK=0 sends kv16 down the SAME v2-attention path it took
# before the change, on the same binary -- so this is a clean A/B, no rebuild.
# 5 repeats each: a single pass proves nothing about an intermittent fault.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/repro.jsonl; : > $OUT
FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1 CAMELID_SPEC_TREE=0
      CAMELID_KQUANT_V2=0 CAMELID_KQUANT_MMA=1)
run() { # label splitk
  echo "### $1 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${FAST[@]}" CAMELID_METAL_ATTN_SPLITK="$2" \
    "$BIN" bench-speculative "$Q4" --drafter ngram --draft-tokens 7 --workload "rp_$1" \
      --prompt-file "$HOME/prompts/ag_256.txt" --max-tokens 96 --warmup \
    2>>$HOME/repro.stderr | sed "s/^/{\"arm\":\"$1\",\"record\":/; s/\$/}/" >> $OUT
}
for i in 1 2 3 4 5; do run oldattn_splitk0 0; done
for i in 1 2 3 4 5; do run newattn_splitk1 1; done
echo "REPRO done" >&2
