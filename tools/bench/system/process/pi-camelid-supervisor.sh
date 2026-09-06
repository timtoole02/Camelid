#!/bin/sh
set -eu

if [ "$#" -lt 9 ]; then
  printf '%s\n' 'usage: pi-camelid-supervisor CANDIDATE MODEL ADDR MODEL_ID CONTEXT_WINDOW SOURCE_SHA MODEL_SHA PI PI_ARGS...' >&2
  exit 64
fi

candidate=$1
model=$2
addr=$3
model_id=$4
context_window=$5
source_sha=$6
model_sha=$7
pi=$8
shift 8

server_pid=''

cleanup() {
  if [ -z "$server_pid" ] || ! kill -0 "$server_pid" 2>/dev/null; then
    return
  fi
  kill "$server_pid" 2>/dev/null || true
  remaining=50
  while kill -0 "$server_pid" 2>/dev/null && [ "$remaining" -gt 0 ]; do
    sleep 0.1
    remaining=$((remaining - 1))
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"$candidate" serve --model "$model" --addr "$addr" --no-open >&2 &
server_pid=$!

ready_timeout=${CAMELID_PI_READY_TIMEOUT_SECONDS:-300}
case "$ready_timeout" in
  ''|*[!0-9]*)
    printf '%s\n' 'CAMELID_PI_INVALID_READY_TIMEOUT' >&2
    exit 64
    ;;
esac
started=$(date +%s)
models_response=/tmp/camelid-pi-models.json
while true; do
  if curl -fsS --max-time 2 "http://$addr/v1/models" >"$models_response" 2>/dev/null; then
    if python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); matches=[item for item in data if isinstance(item, dict) and item.get("id") == sys.argv[1]]; raise SystemExit(0 if len(matches) == 1 and matches[0].get("meta", {}).get("n_ctx_train") == int(sys.argv[2]) else 2 if len(data) == 0 else 1)' "$model_id" "$context_window" <"$models_response"; then
      break
    else
      model_status=$?
    fi
    if [ "$model_status" -ne 2 ]; then
      printf '%s\n' 'CAMELID_PI_MODEL_ID_MISMATCH' >&2
      exit 72
    fi
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid" || true
    printf '%s\n' 'CAMELID_PI_SERVER_EXITED_BEFORE_READY' >&2
    exit 70
  fi
  now=$(date +%s)
  if [ $((now - started)) -ge "$ready_timeout" ]; then
    printf '%s\n' 'CAMELID_PI_SERVER_READY_TIMEOUT' >&2
    exit 71
  fi
  sleep 0.1
done

current_model=/tmp/camelid-pi-current-model.json
if ! curl -fsS --max-time 15 "http://$addr/api/models/current" >"$current_model" 2>/dev/null; then
  printf '%s\n' 'CAMELID_PI_PROVENANCE_UNAVAILABLE' >&2
  exit 73
fi
if ! python3 -c 'import json,sys; model=json.load(sys.stdin); lane=model.get("lane", {}); ok=model.get("id") == sys.argv[1] and lane.get("camelid_commit") == sys.argv[2] and lane.get("gguf_sha256") == sys.argv[3]; raise SystemExit(0 if ok else 1)' "$model_id" "$source_sha" "$model_sha" <"$current_model"; then
  printf '%s\n' 'CAMELID_PI_PROVENANCE_MISMATCH' >&2
  exit 73
fi

printf '%s\n' 'CAMELID_PI_SERVER_READY' >&2
set +e
"$pi" "$@"
pi_exit=$?
set -e
exit "$pi_exit"