#!/usr/bin/env bash
# serve.sh <binary> <data dir> <port> [serve args...]: start a detached camelid serve, wait until healthy.
set -u
BIN="$1"; DATA="$2"; PORT="$3"; shift 3
MODELS_DIR="${MODELS_DIR:-$HOME/Camelid/models}"
mkdir -p "$DATA"
cd "$HOME/Camelid" || exit 1
# Refuse to start if something already answers on the port: the health check below would
# otherwise report that other server as this one.
if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/v1/health"; then
  echo "port $PORT is already serving; refusing to start" >&2
  exit 1
fi
setsid nohup env CAMELID_DATA_DIR="$DATA" RUST_MIN_STACK=8388608 "$BIN" serve --addr "127.0.0.1:$PORT" \
  --models-dir "$MODELS_DIR" "$@" > "$DATA/server.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$DATA/server.pid"
echo "pid $PID"
for _ in $(seq 1 120); do curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/v1/health" && break; sleep 1; done
kill -0 "$PID" 2>/dev/null || { echo "server $PID exited" >&2; tail -5 "$DATA/server.log" >&2; exit 1; }
curl -s "http://127.0.0.1:$PORT/v1/health" | python3 -c 'import json,sys; h=json.load(sys.stdin); print("health: build", h.get("build"), "| model", h.get("active_model_id"), "| ready", h.get("generation_ready"))'
