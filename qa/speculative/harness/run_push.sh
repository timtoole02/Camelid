#!/bin/bash
# Push for tok/s. Four things, cheapest and most-likely-first, all measurement.
#
# A. RE-ATTRIBUTE the round. The ablation that found attention (67% of per-column
#    cost) ran BEFORE the split-K fix made attention ~3x cheaper. Whatever
#    dominates now is a different stage, and guessing has been expensive.
# B. WIDER WINDOW on chain+splitK. The old width sweep was run with the slow
#    attention, where wider always lost. Attention is now 3x cheaper, so the
#    tradeoff may have inverted. A was 12.57 at k=15 on the tree.
# C. TREE verify, re-measured. It cost 2770 ms/round -- but that was the OLD
#    attention path, and the tree issues k*nodes attention dispatches, so it
#    should have benefited most from the fix. Trees carry the highest A.
# D. Q8_0 vs Q4_K_M head to head on the fixed lane, since Q8_0 attention was
#    never the bottleneck and its GEMV runs at 85% of wall.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
Q8=$HOME/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf
OUT=$HOME/push.jsonl; : > $OUT
BASE=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=true CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1)
run() { # label model drafts extra...
  local label=$1 model=$2 k=$3; shift 3
  echo "### $label $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${BASE[@]}" "$@" \
    "$BIN" bench-speculative "$model" --drafter suffix --draft-tokens "$k" --workload "ps_$label" \
      --prompt-file "$HOME/prompts/agentic_4k.txt" --max-tokens 96 --warmup \
    2>>$HOME/push.stderr | sed "s/^/{\"label\":\"$label\",\"drafts\":$k,\"record\":/; s/\$/}/" >> $OUT
  echo "    done $(date +%H:%M:%S)" >&2
}
# B: does a wider window pay now that attention is cheap?
run width_k7  "$Q4" 7  CAMELID_SPEC_TREE=0
run width_k11 "$Q4" 11 CAMELID_SPEC_TREE=0
run width_k15 "$Q4" 15 CAMELID_SPEC_TREE=0
# C: tree verify re-measured on the fixed attention path
run tree_k7   "$Q4" 7  CAMELID_SPEC_TREE=1 CAMELID_SPEC_TREE_GATE=0
run tree_k15  "$Q4" 15 CAMELID_SPEC_TREE=1 CAMELID_SPEC_TREE_GATE=0
# D: Q8_0 on the fixed lane
run q8_k7     "$Q8" 7  CAMELID_SPEC_TREE=0
# A: re-attribution of whatever now dominates the round
for st in attn rope scatter argmax; do
  echo "### ablate_$st $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${BASE[@]}" CAMELID_SPEC_TREE=0 CAMELID_VERIFY_ABLATE=$st \
    "$BIN" bench-speculative "$Q4" --drafter suffix --draft-tokens 7 --workload "abl2_$st" \
      --prompt-file "$HOME/prompts/agentic_4k.txt" --max-tokens 96 --warmup \
    2>>$HOME/push.stderr | sed "s/^/{\"label\":\"ablate_$st\",\"drafts\":7,\"record\":/; s/\$/}/" >> $OUT
done
echo "PUSH done" >&2
