#!/bin/bash
# Attribute the 70 ms per verify column. Each arm omits ONE per-row stage, so the
# drop in round time is that stage's share. Outputs are garbage in the ablated
# arms by construction -- lossless=false is EXPECTED and is not a defect here.
# Baseline first and last, to catch thermal drift across the set.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/ablate.jsonl; : > $OUT
FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1 CAMELID_SPEC_TREE=0
      CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1)
run() { # label ablate-value
  echo "### $1 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${FAST[@]}" CAMELID_VERIFY_ABLATE="$2" \
    "$BIN" bench-speculative "$Q4" --drafter suffix --draft-tokens 7 --workload "abl_$1" \
      --prompt-file "$HOME/prompts/agentic_4k.txt" --max-tokens 96 --warmup \
    2>>$HOME/ablate.stderr | sed "s/^/{\"ablate\":\"$1\",\"record\":/; s/\$/}/" >> $OUT
  echo "    done $(date +%H:%M:%S)" >&2
}
run baseline ""
run no_attn   attn
run no_rope   rope
run no_scatter scatter
run no_argmax argmax
run no_qknorm qknorm
run baseline2 ""
echo "ABLATE done" >&2
