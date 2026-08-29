#!/bin/bash
# Second question for the v2 lane, and possibly the bigger one: PREFILL.
#
# Phase 0 measured Q4_K prefill on the default kernels at 110 ms per prompt
# token, growing superlinearly -- a 465-token prefill took 63.7 s, which is what
# disqualified Q4_K_M as a target regardless of decode. But the multi-column
# K-quant GEMV explicitly serves batched prefill ("wide batches run as
# consecutive column groups through the same kernel"), and the MMA lane engages
# at >= 4 columns. So the same change may fix prefill as a side effect.
#
# bench-generate reports prefill_ms directly, so this measures it cleanly
# without speculation in the way.
set -u
S=/private/tmp/claude-501/-Users-timtoole/abb2295f-8181-40db-9186-de1e33166d1a/scratchpad
OUT=$S/receipts; mkdir -p "$OUT"
BIN=/Volumes/Untitled/cargo-targets/Camelid-3x/release/camelid
Q4=/Volumes/Untitled/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
Q8=/Volumes/Untitled/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf

FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=1)

gen() { # gen <arm> <v2> <mma> <model> <prompt> <label>
  echo "### gen $1 / $6 $(date +%H:%M:%S)" >&2
  CAM_SESSION_PID=$PPID /Users/timtoole/bin/cam-lock.sh env "${FAST[@]}" \
    CAMELID_KQUANT_V2="$2" CAMELID_KQUANT_MMA="$3" \
    "$BIN" bench-generate "$4" --prompt-file "$5" --max-tokens 64 --iterations 2 --warmup \
    2>>"$OUT/v2depth.stderr" | sed "s/^/{\"arm\":\"$1\",\"prompt\":\"$6\",\"record\":/; s/$/}/" \
    >> "$OUT/v2depth.jsonl"
  echo "    done $(date +%H:%M:%S)" >&2
}

# prefill at 512-token depth: v1 took 63.7 s here, so run it LAST and only if
# the v2 arms show the lane is worth the comparison.
gen v2mma 1 1 "$Q4" "$S/prompts/ag_512.txt" d512
gen v2    1 0 "$Q4" "$S/prompts/ag_512.txt" d512
# Q8_0 control at the same depth -- the lane Phase 0 says is healthy (8.7 ms/tok prefill)
gen q8    0 1 "$Q8" "$S/prompts/ag_512.txt" d512
# and the v1 K-quant baseline that produced the 110 ms/token number
gen v1    0 1 "$Q4" "$S/prompts/ag_512.txt" d512
echo "V2DEPTH done -> $OUT/v2depth.jsonl" >&2
