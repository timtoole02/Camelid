#!/usr/bin/env bash
# Greedy-parity capture for the gemma4 E2B Q4_0 row: CUDA-resident lane vs the CPU
# gemma4 runtime, one engine resident at a time (this box has 6 GB VRAM / 16 GB RAM).
# Prompts carry the gemma chat template, because that is the shape serve feeds and
# the shape the original mis-decode was reported on.
set -u
BIN=./target/release/camelid.exe
MODEL="${1:?usage: run-q4_0-parity.sh <gguf>}"
OUT="${2:-qa/gemma3-cuda/phase5/q4_0-parity.json}"
MAXTOK=32

PROMPTS=(
  "Name the capital of France in one word."
  "What color is the sky on a clear day?"
  "Name the largest ocean on Earth."
  "What is 2 + 2?"
  "List three primary colors."
)

echo "{" >"$OUT"
echo "  \"model\": \"$(basename "$MODEL")\"," >>"$OUT"
echo "  \"max_tokens\": $MAXTOK," >>"$OUT"
echo "  \"legs\": [" >>"$OUT"

all_pass=true
first=true
for p in "${PROMPTS[@]}"; do
  tmpl=$'<start_of_turn>user\n'"$p"$'<end_of_turn>\n<start_of_turn>model\n'
  cuda=$("$BIN" gemma4-cuda-generate "$MODEL" --prompt "$tmpl" --max-tokens $MAXTOK 2>&1 |
    grep -oP '(?<=token_ids: ).*' | tail -1)
  cpu=$("$BIN" gemma4-generate "$MODEL" --prompt "$tmpl" --max-tokens $MAXTOK 2>&1 |
    grep -oP '(?<=token_ids: ).*' | tail -1)
  if [ "$cuda" = "$cpu" ]; then match=true; else match=false; all_pass=false; fi
  echo "    prompt: $p"
  echo "      cuda: $cuda"
  echo "      cpu : $cpu"
  echo "      match: $match"
  $first || echo "," >>"$OUT"
  first=false
  printf '    {"prompt": %s, "cuda_ids": "%s", "cpu_ids": "%s", "token_identical": %s}' \
    "$(printf '%s' "$p" | sed 's/"/\\"/g; s/^/"/; s/$/"/')" "$cuda" "$cpu" "$match" >>"$OUT"
done

echo "" >>"$OUT"
echo "  ]," >>"$OUT"
echo "  \"all_pass\": $all_pass" >>"$OUT"
echo "}" >>"$OUT"
echo "all_pass: $all_pass  -> $OUT"
