#!/bin/bash
# THE unmeasured number: the end-to-end speculative multiplier at a depth where
# acceptance is actually high.
#
# Everything measured so far was at 304-556 tokens, where the n-gram drafter only
# accepts 28-36% -- so speculation measured ~1.0x no matter how cheap the verify
# got. Phase 0 established acceptance is 90.8% (A=7.35, window SATURATED) on a
# 4137-token agentic transcript. The verify round is now 6.2x cheaper and, per the
# lane's own microbench, roughly FLAT in window width. So this asks two things at
# once:
#   1. v1 vs v2+MMA at the same draft width -> the real multiplier
#   2. width 5/7/11/15 -> does A keep climbing once the window stops binding?
set -u
S=/private/tmp/claude-501/-Users-timtoole/abb2295f-8181-40db-9186-de1e33166d1a/scratchpad
OUT=$S/receipts; mkdir -p "$OUT"
BIN=/Volumes/Untitled/cargo-targets/Camelid-3x/release/camelid
Q4=/Volumes/Untitled/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
P=$S/prompts/agentic_4k.txt

FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1 CAMELID_SPEC_TREE=0)

run() { # run <arm> <v2> <mma> <drafts>
  echo "### $1 drafts=$4 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID "$HOME"/bin/cam-lock.sh env "${FAST[@]}" \
    CAMELID_KQUANT_V2="$2" CAMELID_KQUANT_MMA="$3" \
    "$BIN" bench-speculative "$Q4" \
      --drafter ngram --draft-tokens "$4" --workload "win_$1_k$4" \
      --prompt-file "$P" --max-tokens 96 --warmup \
    2>>"$OUT/window.stderr" \
    | sed "s/^/{\"arm\":\"$1\",\"drafts\":$4,\"record\":/; s/\$/}/" >> "$OUT/window.jsonl"
  echo "    done $(date +%H:%M:%S)" >&2
}

# v1 control at the width Phase 0 used, so the multiplier has today's default as
# its baseline rather than v2's own plain decode.
run v1 0 0 7
# v2+MMA across widths. MAX_VERIFY_K is 16, so 15 drafts is the widest legal window.
for k in 7 11 15 5; do run v2mma 1 1 "$k"; done
echo "WINDOW done -> $OUT/window.jsonl" >&2
