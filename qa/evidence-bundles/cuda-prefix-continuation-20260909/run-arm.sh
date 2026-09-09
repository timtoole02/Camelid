#!/usr/bin/env bash
# Run ONE arm of the prefix-continuation A/B: start a server with the given
# CAMELID_CUDA_PREFIX_CONTINUATION setting, drive a growing conversation, stop it.
#
# One server per arm is not a convenience: `continuation_enabled` is read from the
# process environment, and the resident CUDA engine is process-global, so the arm
# has to BE the process. Turn 1 has nothing resident and is expected to match
# across arms — it is the built-in control.
set -uo pipefail

ARM="$1"          # label, e.g. on-1
CONT="$2"         # "1" or "0"
PORT="$3"
TURNS="${4:-4}"

BIN="./target/release/camelid.exe"
MODEL="~/Documents/GitHub/Camelid/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
OUT="$SCRATCH/arm-$ARM"
mkdir -p "$OUT"

# Refuse to start on top of another engine. Two engines on one 6 GiB card starve
# each other and the numbers become thermal//VRAM artifacts rather than a signal.
if netstat -ano 2>/dev/null | grep -q "127.0.0.1:$PORT .*LISTENING"; then
  echo "port $PORT already listening; refusing to start a second engine" >&2
  exit 2
fi

CAMELID_CUDA_PREFIX_CONTINUATION="$CONT" \
CAMELID_RESIDENT_TRACE=1 \
  "$BIN" serve --addr "127.0.0.1:$PORT" --model "$MODEL" \
  >"$OUT/serve.out" 2>"$OUT/serve.err" &

# Wait for the MODEL, not just the socket. /v1/health answers 200 while the model
# is still loading (loaded_now:false), so gating on the socket measured a cold load
# as if it were a turn.
ready=0
for _ in $(seq 1 300); do
  if curl -s "http://127.0.0.1:$PORT/v1/health" 2>/dev/null | grep -q '"generation_ready":true'; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "model did not become generation_ready" >&2
  tail -20 "$OUT/serve.err" >&2
  exit 3
fi

# One untimed warm-up request is deliberately NOT done: turn 1 IS the cold case and
# we want it in the record, since "turn 1 unaffected" is part of the claim.
node "$SCRATCH/bench-turns.mjs" "http://127.0.0.1:$PORT" "$TURNS" "$ARM" | tee "$OUT/turns.jsonl"

PID=$(netstat -ano 2>/dev/null | grep "127.0.0.1:$PORT .*LISTENING" | awk '{print $5}' | head -1)
if [ -n "${PID:-}" ]; then
  taskkill //PID "$PID" //F >/dev/null 2>&1 || true
fi
# Confirm the port is actually free before the next arm starts.
for _ in $(seq 1 30); do
  netstat -ano 2>/dev/null | grep -q "127.0.0.1:$PORT .*LISTENING" || break
  sleep 1
done
grep -c "prefix continuation" "$OUT/serve.err" 2>/dev/null | sed "s/^/$ARM continuation-trace-lines: /"
