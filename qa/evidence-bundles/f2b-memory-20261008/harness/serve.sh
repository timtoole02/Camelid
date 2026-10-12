#!/usr/bin/env bash
# serve.sh <data dir> <port> <model file>: start camelid serve detached and wait until it answers.
set -u
DATA="$1"; PORT="$2"; MODEL="$3"
mkdir -p "$DATA"
if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/v1/health"; then echo "port $PORT busy" >&2; exit 1; fi
setsid nohup env CAMELID_DATA_DIR="$DATA" RUST_MIN_STACK=8388608 "$HOME/camelid-f2a-target/release/camelid" serve --addr "127.0.0.1:$PORT" \
  --models-dir "$HOME/Camelid/models" --model "$HOME/Camelid/models/$MODEL" > "$DATA/server.log" 2>&1 < /dev/null &
echo $! > "$DATA/server.pid"
for _ in $(seq 1 300); do curl -s -m 2 "http://127.0.0.1:$PORT/v1/health" | grep -q "\"generation_ready\":true" && break; sleep 1; done
curl -s "http://127.0.0.1:$PORT/v1/health" | python3 -c "import json,sys; h=json.load(sys.stdin); print(\"build\", h.get(\"build\"), \"model\", h.get(\"active_model_id\"), \"ready\", h.get(\"generation_ready\"))"
