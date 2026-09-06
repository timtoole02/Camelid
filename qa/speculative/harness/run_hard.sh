#!/bin/bash
# The decisive MTP-value test.
#
# On repo-documentation prose the suffix drafter did NOT collapse (A=7.42,
# 2.24x) -- but doc prose is stylistically repetitive, so that is a soft test.
# These three prompts ship in qa/speed/prompts.json flagged spec_friendly=FALSE;
# adversarial_lowaccept literally asks for 60 unrelated unpredictable words.
# If the suffix drafter holds up here, a trained MTP head has little left to buy
# on this workload. If it collapses, that collapse is exactly the gap the head
# fills, and the weeks are justified.
set -u
BIN=$HOME/camelid-integrity-check/target/release/camelid
Q4=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
OUT=$HOME/hard.jsonl; : > $OUT
FAST=(CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 CAMELID_METAL_WIRE=1
      CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 CAMELID_METAL_NOCOPY=1
      CAMELID_METAL_KQUANT=1 CAMELID_NO_OPEN=true CAMELID_SPEC_TREE=0
      CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1)
for p in hard_adversarial_lowaccept hard_creative_writing hard_normal_chat; do
  for dr in suffix ngram; do
    echo "### $dr $p $(date +%H:%M:%S)" >&2
    CAM_SESSION_PID=$PPID ~/bin/cam-lock.sh env "${FAST[@]}" \
      "$BIN" bench-speculative "$Q4" --drafter "$dr" --draft-tokens 7 --workload "hd_${dr}_$p" \
        --prompt-file "$HOME/prompts/$p.txt" --max-tokens 128 --warmup \
      2>>$HOME/hard.stderr | sed "s/^/{\"drafter\":\"$dr\",\"prompt\":\"$p\",\"record\":/; s/\$/}/" >> $OUT
  done
done
echo "HARD done" >&2
