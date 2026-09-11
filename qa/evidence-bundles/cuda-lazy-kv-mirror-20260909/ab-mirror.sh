#!/usr/bin/env bash
# Lazy vs eager KV mirror, one binary, ABBA with a cool gate. Arm = CAMELID_CUDA_EAGER_KV_MIRROR.
set -uo pipefail
S="~/AppData/Local/Temp/claude/C--Users-timto/da34b761-d343-4f3f-9153-e0a13209f877/scratchpad"
M="~/Documents/GitHub/Camelid/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
cd ~/wt-kvmirror

cool() {
  for _ in $(seq 1 40); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -n "$t" ] && [ "$t" -le 55 ] && break
    sleep 3
  done
  echo "[cool] gpu=${t}C"
}

arm() { # label, eager(0/1)
  local L="$1" E="$2"
  cool
  echo "=== arm $L (CAMELID_CUDA_EAGER_KV_MIRROR=$E) ==="
  CAMELID_CUDA_EAGER_KV_MIRROR="$E" CAMELID_RESIDENT_TRACE=1 \
    ./target/release/camelid.exe serve --addr 127.0.0.1:8095 --model "$M" \
    >/dev/null 2>"$S/mir-$L.err" &
  for _ in $(seq 1 300); do
    curl -s http://127.0.0.1:8095/v1/health 2>/dev/null | grep -q '"generation_ready":true' && break
    sleep 1
  done
  node "$S/bench-turns.mjs" http://127.0.0.1:8095 5 "$L" | tee "$S/mir-$L.jsonl"
  node "$S/bench-replies.mjs" http://127.0.0.1:8095 5 > "$S/mir-replies-$L.txt" 2>&1
  P=$(netstat -ano 2>/dev/null | grep "127.0.0.1:8095 .*LISTENING" | awk '{print $5}' | head -1)
  [ -n "${P:-}" ] && taskkill //PID "$P" //F >/dev/null 2>&1
  for _ in $(seq 1 30); do
    netstat -ano 2>/dev/null | grep -q "127.0.0.1:8095 .*LISTENING" || break
    sleep 1
  done
}

arm lazy-1 0
arm eager-1 1
arm eager-2 1
arm lazy-2 0
echo "=== DONE ==="
