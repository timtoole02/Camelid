#!/usr/bin/env bash
# Token-identity + throughput receipt for the FP8 KV cache lanes.
#
# Runs one greedy generation per --kv-quant format against the SAME model and
# prompt, then compares each quantized lane's token stream against the f16
# reference. This is the comparison that matters for a lossy KV format: not how
# it scores on synthetic data, but how far its output drifts from the
# unquantized baseline on a real model.
#
# Every lane is pinned to the CPU (--gpu off) because the resident CUDA decoder
# only honours f16 and q8_0; without the pin, f16/q8_0 could run on the GPU
# while fp8 fell back to the CPU and the comparison would be meaningless.
#
# Usage: scripts/kv-fp8-receipt.sh <model.gguf> [max_tokens]
set -euo pipefail

MODEL="${1:?usage: kv-fp8-receipt.sh <model.gguf> [max_tokens]}"
MAX_TOKENS="${2:-96}"
BIN="${CAMELID_BIN:-./target/release/camelid}"
PORT="${PORT:-8731}"
OUT="${OUT:-./target/kv-fp8-receipt}"
# Kept as a single line with no characters needing JSON escaping, so the request
# body below can be assembled without a quoting pipeline.
PROMPT='Explain in one paragraph why a quantized key-value cache changes the memory cost of long-context inference. Be precise and concrete.'

mkdir -p "$OUT"
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

run_lane() {
  local quant="$1"
  local log="$OUT/$quant.server.log"
  local body="$OUT/$quant.json"

  "$BIN" serve --addr "127.0.0.1:$PORT" --gpu off --deterministic \
    --kv-quant "$quant" --model "$MODEL" >"$log" 2>&1 &
  local pid=$!
  trap 'kill '"$pid"' 2>/dev/null || true' RETURN

  # `/v1/health` answers 200 while the engine is still warming up, and a request
  # sent then comes back "model is not loaded" — fast, uniform across every lane,
  # and therefore trivially "identical". Wait for the model to be genuinely
  # loadable by probing generation itself.
  local ready=""
  for _ in $(seq 1 300); do
    # The server prints this only once the engine is actually built and the model
    # is resident; `/v1/health` answers 200 well before that.
    if grep -q "Camelid is ready" "$log" 2>/dev/null; then ready=1; break; fi
    kill -0 "$pid" 2>/dev/null || { echo "server for $quant exited early; see $log" >&2; return 1; }
    sleep 2
  done
  [ -n "$ready" ] || { echo "server for $quant never became ready; see $log" >&2; return 1; }

  # Extracted with parameter expansion rather than a pipeline: any stage that exits
  # early (`grep -m1`, `head -1`) SIGPIPEs the stage feeding it, which trips
  # `pipefail` and aborts the run under `set -e`.
  local models rest model_id
  models=$(curl -s "http://127.0.0.1:$PORT/v1/models")
  rest="${models#*\"id\":\"}"
  model_id="${rest%%\"*}"
  if [ -z "$model_id" ] || [ "$model_id" = "$models" ]; then
    echo "could not read a model id for $quant from: $models" >&2
    return 1
  fi

  local start end
  start=$(date +%s%N)
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_id\",\"temperature\":0,\"top_p\":1,\"seed\":0,\"max_tokens\":$MAX_TOKENS,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
    >"$body"
  end=$(date +%s%N)

  # Strip to the assistant text so lanes can be diffed as plain token streams.
  sed 's/.*"content" *: *"//; s/".*//' "$body" >"$OUT/$quant.txt"
  echo "$(( (end - start) / 1000000 ))" >"$OUT/$quant.ms"

  # A lane that errored produces a short, identical body across every format,
  # which reads as a perfect match. Refuse to report on one.
  local produced
  produced=$(wc -c <"$OUT/$quant.txt")
  if [ "$produced" -lt 200 ]; then
    echo "lane $quant produced only $produced chars; body was:" >&2
    head -c 400 "$body" >&2
    echo >&2
    return 1
  fi

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

echo "model:      $MODEL"
echo "max_tokens: $MAX_TOKENS"
echo

for quant in f16 q4_0 q8_0 fp8_e4m3 fp8_e5m2; do
  printf 'running %-10s ... ' "$quant"
  run_lane "$quant"
  printf 'done (%s ms)\n' "$(cat "$OUT/$quant.ms")"
done

echo
echo "=== token-stream agreement with the f16 reference ==="
ref="$OUT/f16.txt"
for quant in q4_0 q8_0 fp8_e4m3 fp8_e5m2; do
  if cmp -s "$ref" "$OUT/$quant.txt"; then
    printf '  %-10s IDENTICAL to f16\n' "$quant"
  else
    common=$(cmp "$ref" "$OUT/$quant.txt" 2>/dev/null | sed 's/.*char \([0-9]*\).*/\1/' || echo 0)
    total=$(wc -c <"$ref")
    printf '  %-10s diverges at char %s of %s\n' "$quant" "$common" "$total"
  fi
done

echo
echo "artifacts in $OUT/"
