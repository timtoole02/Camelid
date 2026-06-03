#!/usr/bin/env bash
# Run Camelid pipeline-parallel across two Macs over the Thunderbolt bridge, and
# print the generated text + per-node peak RSS. The master runs on THIS machine; the
# worker (last node) runs on $WORKER_HOST via ssh. Each node needs the camelid binary
# and the GGUF model present; this stages them to the worker if missing.
#
# Required env:
#   MODEL          GGUF path (must be identical on both nodes; staged to worker here)
#   CAMELID_BIN    path to the camelid release binary on this machine
#   WORKER_HOST    ssh host for the worker (e.g. tims-mac-mini-2.local)
#   MASTER_TB_IP   this machine's Thunderbolt-bridge IP (e.g. 169.254.147.29)
#   WORKER_TB_IP   worker's Thunderbolt-bridge IP (e.g. 169.254.156.89)
#   TOTAL_LAYERS   transformer layers in the model (Llama-2-13B = 40)
# Optional:
#   SPLIT (default TOTAL_LAYERS/2)  PROMPT  MAX_TOKENS (default 64)
#   REMOTE_BIN (default /tmp/camelid)  REMOTE_MODEL (default same path as MODEL)
set -euo pipefail
: "${MODEL:?}" "${CAMELID_BIN:?}" "${WORKER_HOST:?}" "${MASTER_TB_IP:?}" "${WORKER_TB_IP:?}" "${TOTAL_LAYERS:?}"
SPLIT="${SPLIT:-$((TOTAL_LAYERS/2))}"
PROMPT="${PROMPT:-Explain what a Rust borrow checker does in two sentences.}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REMOTE_BIN="${REMOTE_BIN:-/tmp/camelid}"
EXTRA_ENV="${EXTRA_ENV:-}"   # extra VAR=VAL pairs exported on BOTH nodes (e.g. CAMELID_DISTRIBUTED_TRACE=1)
# Repack OFF by default: the GPU-resident decode path needs plain (un-packed) Q8_0 blocks,
# and the nodes' residency gate hard-fails on repacked storage. Set REPACK=1 for CPU-decode runs.
REPACK="${REPACK:-0}"
# Ghost mesh: when set, each node streams its layer shard per token from its local .cghost
# (made with `repack-ghost --layers a..b`) instead of holding it resident.
MASTER_CGHOST="${MASTER_CGHOST:-}"   # local .cghost path for the master's 0..SPLIT shard
WORKER_CGHOST="${WORKER_CGHOST:-}"   # REMOTE .cghost path for the worker's SPLIT..TOTAL shard
MASTER_GHOST_ARGS=""; [ -n "$MASTER_CGHOST" ] && MASTER_GHOST_ARGS="--cghost $MASTER_CGHOST"
WORKER_GHOST_ARGS=""; [ -n "$WORKER_CGHOST" ] && WORKER_GHOST_ARGS="--cghost $WORKER_CGHOST"
REMOTE_MODEL="${REMOTE_MODEL:-$MODEL}"
PORT_W=5005
PORT_M=5006

echo "[two-mac] master 0..$SPLIT on $(hostname -s) ($MASTER_TB_IP)"
echo "[two-mac] worker $SPLIT..$TOTAL_LAYERS on $WORKER_HOST ($WORKER_TB_IP)"

# 1) Stage binary + model to the worker if missing (over the fast TB link).
ssh "$WORKER_HOST" "mkdir -p \"\$(dirname '$REMOTE_MODEL')\""
# Re-stage the binary whenever it differs (size check) — a stale worker binary silently
# running old code is exactly the class of failure the residency gate exists to kill.
remote_bin_size="$(ssh "$WORKER_HOST" "stat -f%z '$REMOTE_BIN' 2>/dev/null || echo 0")"
local_bin_size="$(stat -f%z "$CAMELID_BIN")"
if [ "$remote_bin_size" != "$local_bin_size" ]; then
  echo "[two-mac] copying binary -> $WORKER_HOST:$REMOTE_BIN"
  scp -q "$CAMELID_BIN" "$WORKER_HOST:$REMOTE_BIN"
  ssh "$WORKER_HOST" "chmod +x '$REMOTE_BIN'"
fi
remote_size="$(ssh "$WORKER_HOST" "stat -f%z '$REMOTE_MODEL' 2>/dev/null || echo 0")"
local_size="$(stat -f%z "$MODEL")"
if [ "$remote_size" != "$local_size" ]; then
  echo "[two-mac] copying model ($(echo "$local_size" | awk '{printf "%.1f GB", $1/1073741824}')) -> $WORKER_HOST:$REMOTE_MODEL (over Thunderbolt)"
  scp -q "$MODEL" "$WORKER_HOST:$REMOTE_MODEL"
fi

# 2) Launch the worker (last node) on the remote, bound to its TB IP.
echo "[two-mac] starting remote worker..."
ssh "$WORKER_HOST" "CAMELID_MAC_Q8_REPACK=$REPACK $EXTRA_ENV /usr/bin/time -l '$REMOTE_BIN' distribute-worker '$REMOTE_MODEL' \
  --addr $WORKER_TB_IP:$PORT_W --layers $SPLIT..$TOTAL_LAYERS --master-addr $MASTER_TB_IP:$PORT_M $WORKER_GHOST_ARGS" \
  >/tmp/two_mac_worker.out 2>&1 &
SSH_PID=$!

# 3) Run the master (first node) here; it connects to the worker over TB.
echo "[two-mac] starting master..."
START=$(python3 -c 'import time;print(time.time())')
env CAMELID_MAC_Q8_REPACK=$REPACK $EXTRA_ENV /usr/bin/time -l "$CAMELID_BIN" distribute-master "$MODEL" \
  --worker-addr "$WORKER_TB_IP:$PORT_W" --layers "0..$SPLIT" --addr "$MASTER_TB_IP:$PORT_M" \
  --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" $MASTER_GHOST_ARGS \
  >/tmp/two_mac_master.out 2>/tmp/two_mac_master.time || true
END=$(python3 -c 'import time;print(time.time())')

# 4) Clean up the remote worker.
ssh "$WORKER_HOST" "pkill -f distribute-worker" 2>/dev/null || true
kill "$SSH_PID" 2>/dev/null || true

echo "=== generated text ==="
sed -n '/Encoded prompt/,$p' /tmp/two_mac_master.out
echo "=== timing: $(python3 -c "print(f'{$END-$START:.1f}s total (incl. load)')") ==="
echo "=== master peak RSS ==="; grep -i "maximum resident" /tmp/two_mac_master.time | awk '{printf "  %.2f GB\n",$1/1073741824}'
echo "=== worker peak RSS (remote) ==="; grep -i "maximum resident" /tmp/two_mac_worker.out | tail -1 | awk '{printf "  %.2f GB\n",$1/1073741824}'
echo "=== decode rate ==="; grep -i "\[distributed\] decode" /tmp/two_mac_master.out || true
echo "(worker stdout: /tmp/two_mac_worker.out)"
