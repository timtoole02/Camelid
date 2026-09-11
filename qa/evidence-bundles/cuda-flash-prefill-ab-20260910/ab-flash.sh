#!/usr/bin/env bash
# Flash prefill A/B across context lengths, ABBA with a cool gate.
# Arm = CAMELID_FLASH_PREFILL. Same binary, same prompts (fixed nonce) across arms.
set -uo pipefail
S="~/AppData/Local/Temp/claude/C--Users-timto/da34b761-d343-4f3f-9153-e0a13209f877/scratchpad"
M="~/Documents/GitHub/Camelid/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
LENS="${LENS:-1400,3000,6000}"
NONCE=fixed7788
cd ~/wt-flash

cool() {
  for _ in $(seq 1 40); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -n "$t" ] && [ "$t" -le 55 ] && break
    sleep 3
  done
  echo "[cool] gpu=${t}C"
}

arm() { # label, flash(0/1)
  local L="$1" F="$2"
  cool
  echo "=== arm $L (CAMELID_FLASH_PREFILL=$F) ==="
  CAMELID_FLASH_PREFILL="$F" CAMELID_RESIDENT_TRACE=1 \
    ./target/release/camelid.exe serve --addr 127.0.0.1:8090 --model "$M" \
    >/dev/null 2>"$S/fl-$L.err" &
  for _ in $(seq 1 300); do
    curl -s http://127.0.0.1:8090/v1/health 2>/dev/null | grep -q '"generation_ready":true' && break
    sleep 1
  done
  node "$S/bench-flash.mjs" http://127.0.0.1:8090 "$LENS" "$L" "$NONCE" | tee "$S/fl-$L.jsonl"
  P=$(netstat -ano 2>/dev/null | grep "127.0.0.1:8090 .*LISTENING" | awk '{print $5}' | head -1)
  [ -n "${P:-}" ] && taskkill //PID "$P" //F >/dev/null 2>&1
  for _ in $(seq 1 30); do
    netstat -ano 2>/dev/null | grep -q "127.0.0.1:8090 .*LISTENING" || break
    sleep 1
  done
  echo "  prefill lines:"; grep "GPU prefill" "$S/fl-$L.err" | sed 's/^/    /'
}

arm off-1 0
arm on-1 1
arm on-2 1
arm off-2 0
echo "=== DONE ==="
