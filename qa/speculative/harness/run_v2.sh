#!/bin/bash
# The v2 lane on the 8B, which is the model Phase 0 actually cares about.
# The session that built v2 measured it on Qwen3-4B; this asks whether it
# transfers to Llama-3-8B Q4_K_M -- the config Phase 0 disqualified at 47% of
# the memory wall using the DEFAULT (v1) K-quant kernels.
#
# Three arms off one build, gated by env, so nothing but the kernel changes:
#   v1      -- today's default, the Phase 0 baseline
#   v2      -- strict-math K-quant GEMV v2, scalar multi-column verify
#   v2+mma  -- v2 with the simdgroup-matrix verify (the flat-in-k claim)
set -u
S=/private/tmp/claude-501/-Users-timtoole/abb2295f-8181-40db-9186-de1e33166d1a/scratchpad
OUT=$S/receipts; mkdir -p "$OUT"
BIN=/Volumes/Untitled/cargo-targets/Camelid-3x/release/camelid
Q4=/Volumes/Untitled/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
cd /Volumes/Untitled/Camelid-3x || exit 1

echo "=== build + bit-identity gates ==="
CAM_SESSION_PID=$PPID CARGO_TARGET_DIR=/Volumes/Untitled/cargo-targets/Camelid-3x \
  "$HOME"/bin/cam-lock.sh sh -c \
  'cargo test --lib -- metal_kquant_v2_pair_bit_identical metal_kquant_q6k_v2_bit_identical metal_kquant_mc_gemv_bit_identical metal_verify_gemv_batched_bit_identical --nocapture && cargo build --release' \
  > "$S/v2_build.log" 2>&1
rc=$?; echo "build/test exit $rc"
grep -E "^test result:|^error" "$S/v2_build.log" | head -4
[ $rc -ne 0 ] && { echo "ABORT: gate or build failed"; exit 1; }

FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1 CAMELID_SPEC_TREE=0)

run() { # run <arm> <v2> <mma> <prompt> <label>
  echo "### $1 / $5 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID "$HOME"/bin/cam-lock.sh env "${FAST[@]}" \
    CAMELID_KQUANT_V2="$2" CAMELID_KQUANT_MMA="$3" \
    "$BIN" bench-speculative "$Q4" \
      --drafter ngram --draft-tokens 7 --workload "$1_$5" \
      --prompt-file "$4" --max-tokens 96 --warmup \
    2>>"$OUT/v2.stderr" | sed "s/^/{\"arm\":\"$1\",\"prompt\":\"$5\",\"record\":/; s/$/}/" >> "$OUT/v2.jsonl"
}

# short prompt first: Q4_K prefill on the v1 lane is 110 ms/token, so a deep
# prompt would spend all its time there rather than in what is being compared.
for arm in "v1 0 1" "v2 1 0" "v2mma 1 1"; do
  set -- $arm
  run "$1" "$2" "$3" "$S/prompts/ag_256.txt" ag256
done
echo "V2 done -> $OUT/v2.jsonl" >&2
