#!/bin/zsh
set -euo pipefail
umask 077

readonly PORT=${CAMELID_50TPS_PORT:-8189}
readonly MIN_DISK_AVAILABLE_KIB=20971520
readonly MIN_FREE_MEMORY_PERCENT=20
readonly PROFILE=39,40,33,30,30,31,31,30,34,30,26,28,30,31,28,37,31,30,31,32,31,32,30,31,32,35,32,34,34,37
readonly script_dir=${0:A:h}
readonly repo_root=$(/usr/bin/git -C "$script_dir" rev-parse --show-toplevel)
readonly source_commit=$(/usr/bin/git -C "$repo_root" rev-parse HEAD)
readonly target_dir=$(cd "$repo_root" && /usr/bin/env cargo metadata --no-deps --format-version 1 | /usr/bin/jq -r .target_directory)
readonly binary=${CAMELID_50TPS_BINARY:-"$target_dir/release/camelid"}
readonly model=${CAMELID_50TPS_MODEL:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf}
readonly cghost=${CAMELID_50TPS_CGHOST:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost}
readonly assistant=${CAMELID_50TPS_ASSISTANT:-/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors}
readonly request="$script_dir/request-48.json"
readonly expected="$script_dir/expected-48-token-ids.json"
readonly analyzer="$script_dir/gate_50tps.py"
readonly run_id=$(/bin/date -u '+%Y%m%dT%H%M%SZ')-$$
readonly receipt_root=${CAMELID_50TPS_RECEIPT_ROOT:-"$repo_root/qa/perf/gemma4-mtp-bandwidth-2026-08-21/50tps-gate-$run_id"}
readonly server_log="$receipt_root/server.log"
readonly response="$receipt_root/response.json"
readonly verdict="$receipt_root/verdict.json"

typeset child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]] && /bin/kill -0 "$child_pid" 2>/dev/null; then
    /bin/kill -INT "$child_pid" 2>/dev/null || true
    for _ in {1..30}; do
      /bin/kill -0 "$child_pid" 2>/dev/null || break
      /bin/sleep 1
    done
    if /bin/kill -0 "$child_pid" 2>/dev/null; then
      /bin/kill -TERM "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

[[ -x "$binary" && -f "$binary" && ! -L "$binary" ]] || {
  print -u2 "REFUSED: build the release binary first: $binary"
  exit 75
}
for input in "$model" "$cghost" "$assistant" "$request" "$expected" "$analyzer"; do
  [[ -f "$input" && ! -L "$input" ]] || {
    print -u2 "REFUSED: missing or symlinked input: $input"
    exit 75
  }
done
if /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "REFUSED: TCP port $PORT is already in use"
  exit 76
fi
if /usr/sbin/lsof -nP -iTCP:8181 -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "REFUSED: the user WebUI engine is active on 8181; it will not be touched"
  exit 76
fi
typeset available_kib
available_kib=$(/bin/df -k /System/Volumes/Data | /usr/bin/awk 'NR==2 {print $4}')
(( available_kib >= MIN_DISK_AVAILABLE_KIB )) || {
  print -u2 "REFUSED: disk headroom ${available_kib}KiB is below ${MIN_DISK_AVAILABLE_KIB}KiB"
  exit 75
}
typeset free_percent
free_percent=$(/usr/bin/memory_pressure -Q | /usr/bin/awk '/free percentage/ {gsub(/%/, "", $5); print $5}')
(( free_percent >= MIN_FREE_MEMORY_PERCENT )) || {
  print -u2 "REFUSED: free memory ${free_percent}% is below ${MIN_FREE_MEMORY_PERCENT}%"
  exit 75
}

/bin/mkdir -p "$receipt_root"
{
  print -r -- "schema_version=1"
  print -r -- "source_commit=$source_commit"
  print -r -- "binary=$binary"
  /usr/bin/shasum -a 256 "$binary" "$model" "$cghost" "$assistant" "$request" "$expected" "$analyzer" "$0"
  /usr/sbin/sysctl kern.boottime vm.swapusage
  /usr/bin/memory_pressure -Q
  /bin/df -h /System/Volumes/Data
} > "$receipt_root/baseline.txt"

/usr/bin/env -i \
  HOME=/Users/timtoole \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  TMPDIR=/tmp \
  CAMELID_GHOST_ALLOW_LEGACY_SPARSE=0 \
  CAMELID_GEMMA4_GHOST_METAL=1 \
  CAMELID_GEMMA4_GHOST_METAL_SLOTS=1 \
  CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST=1 \
  CAMELID_GEMMA4_GHOST_METAL_TURBO=1 \
  CAMELID_GEMMA4_GHOST_METAL_COMMON=1 \
  CAMELID_GEMMA4_GHOST_METAL_CONTEXT=1024 \
  CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT=0 \
  CAMELID_GEMMA4_GHOST_READ_THREADS=8 \
  CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY=1 \
  CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1 \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32 \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER="$PROFILE" \
  CAMELID_GEMMA4_GHOST_METAL_DECODE_PROMOTION=0 \
  CAMELID_GEMMA4_SLOT_PIN=0 \
  CAMELID_GEMMA4_GHOST_METAL_HOT_PIN=0 \
  CAMELID_GEMMA4_VICTIM_CACHE=0 \
  CAMELID_GEMMA4_VICTIM_MB=0 \
  CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS=0 \
  CAMELID_GEMMA4_CHAINED_PREDICT=0 \
  CAMELID_SPEC_DECODE=off \
  CAMELID_GEMMA4_SPEC_K1_LANE=chained \
  CAMELID_GEMMA4_SPEC_CHUNK_MAX=8 \
  CAMELID_GEMMA4_SPEC_DRAFT_TOKENS=8 \
  CAMELID_GEMMA4_MTP_ASSISTANT_PATH="$assistant" \
  CAMELID_GEMMA4_MTP_DEVICE_CHAIN=1 \
  CAMELID_GEMMA4_MTP_PREFILL_SEED_BOOTSTRAP=1 \
  CAMELID_GEMMA4_DENSE_K8_GENERIC=1 \
  CAMELID_GEMMA4_SPEC_TIMING=1 \
  CAMELID_GEMMA4_GHOST_METAL_TIMING=1 \
  CAMELID_GEMMA4_ROUTE_TRACE=1 \
  "$binary" serve \
    --addr 127.0.0.1:$PORT \
    --model "$model" \
    --cghost "$cghost" \
    --expert-cache-mib 0 --gpu on --no-open \
    > "$server_log" 2>&1 &
child_pid=$!

typeset health="" ready=0
for _ in {1..900}; do
  if health=$(/usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/health" 2>/dev/null); then
    if print -r -- "$health" | /usr/bin/jq -e '
      .generation_ready == true and
      .gemma4_serve_lane == "ghost_moe" and
      .gemma4_ghost_execution_mode == "full_common_metal" and
      .gemma4_ghost_common_metal_active == true and
      .gemma4_ghost_experts_metal_active == true and
      .gemma4_ghost_head_metal_active == true
    ' >/dev/null; then
      ready=1
      break
    fi
  fi
  /bin/kill -0 "$child_pid" 2>/dev/null || {
    print -u2 "REFUSED: server exited before readiness"
    exit 75
  }
  /bin/sleep 1
done
print -r -- "$health" > "$receipt_root/health.json"
(( ready == 1 )) || {
  print -u2 "REFUSED: server did not reach full-Metal readiness"
  exit 75
}

/usr/bin/memory_pressure -Q > "$receipt_root/pre-request-memory.txt"
/usr/sbin/sysctl vm.swapusage >> "$receipt_root/pre-request-memory.txt"
/usr/bin/curl -fsS --max-time 1800 \
  -w '%{time_total}\n' -o "$response" \
  -H 'Content-Type: application/json' \
  --data-binary "@$request" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  > "$receipt_root/http-wall-seconds.txt"
/usr/bin/memory_pressure -Q > "$receipt_root/post-request-memory.txt"
/usr/sbin/sysctl vm.swapusage >> "$receipt_root/post-request-memory.txt"

/bin/kill -INT "$child_pid"
wait "$child_pid" || true
child_pid=""

/usr/bin/python3 "$analyzer" \
  --response "$response" \
  --server-log "$server_log" \
  --expected-token-ids "$expected" \
  --output "$verdict"
print -r -- "50TPS_RECEIPT=$receipt_root"
