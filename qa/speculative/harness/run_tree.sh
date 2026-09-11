#!/bin/bash
# Tree verify with kv16 admitted to split-K. The tree lane carries the highest
# acceptance measured anywhere here (A=12.57 at k=15) but cost 348.9 ms per
# verify column -- 13x the linear path -- purely because encode_attention_tree
# still excluded kv16 primaries. If it now lands near the linear path's 27 ms/col,
# A=12.57 at a ~215 ms round is ~58 tok/s.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/tree.jsonl; : > $OUT
BASE=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=true CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1)
run() { # label drafts tree prompt
  echo "### $1 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${BASE[@]}" CAMELID_SPEC_TREE=$3 CAMELID_SPEC_TREE_GATE=0 \
    "$BIN" bench-speculative "$Q4" --drafter suffix --draft-tokens "$2" --workload "tr_$1" \
      --prompt-file "$HOME/prompts/$4.txt" --max-tokens 96 --warmup \
    2>>$HOME/tree.stderr | sed "s/^/{\"label\":\"$1\",\"drafts\":$2,\"record\":/; s/\$/}/" >> $OUT
  echo "    done $(date +%H:%M:%S)" >&2
}
run tree_k7_fixed  7  1 agentic_4k
run tree_k15_fixed 15 1 agentic_4k
run chain_k7_ref   7  0 agentic_4k
# and the prose case, where the chain drafter collapses to a LOSS
run tree_k15_prose 15 1 hard_creative_writing
run chain_k7_prose 7  0 hard_creative_writing
echo "TREE done" >&2
