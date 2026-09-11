#!/bin/bash
# Hypothesis: the v1 K-quant lane is non-lossless because multi-token dispatches
# go to q4k_linear_tiled while single-token decode uses q4k_linear_simd, and
# nothing binds those two to the same fast-math contraction. The mc kernel is
# ALREADY proven bit-identical to the single-token one
# (metal_kquant_mc_gemv_bit_identical), and there is already an env gate that
# routes multi-token through it -- default OFF. If that gate flips lossless
# false->true, the fix is a default, not a new kernel or a fast-math change.
# Repeat each arm 3x: the failure was intermittent (false at 304 tok, true at 4137).
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/lossless.jsonl; : > $OUT
FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1 CAMELID_SPEC_TREE=0 CAMELID_KQUANT_V2=0)
run() { # label mc prompt
  echo "### $1 $3 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${FAST[@]}" CAMELID_KQUANT_MC_GEMV="$2" \
    "$BIN" bench-speculative "$Q4" --drafter ngram --draft-tokens 7 --workload "ll_$1_$3" \
      --prompt-file "$HOME/prompts/$3.txt" --max-tokens 96 --warmup \
    2>>$HOME/lossless.stderr | sed "s/^/{\"arm\":\"$1\",\"prompt\":\"$3\",\"record\":/; s/\$/}/" >> $OUT
}
for i in 1 2 3; do run v1_tiled 0 ag_256; done
for i in 1 2 3; do run v1_mc    1 ag_256; done
run v1_tiled 0 ag_512
run v1_mc    1 ag_512
echo "LOSSLESS done" >&2
