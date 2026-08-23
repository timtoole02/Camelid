#!/bin/zsh
set -euo pipefail
umask 077

readonly EX_USAGE=64
readonly EX_REFUSED=75
readonly EX_PORT_BUSY=76
readonly EX_RECEIPT_EXISTS=77
readonly PORT=8189
readonly MIN_DISK_AVAILABLE_KIB=20971520
readonly MAX_DISK_USED_PERCENT=90
readonly MIN_BASELINE_HEADROOM_BYTES=8589934592
readonly MIN_RUNTIME_HEADROOM_BYTES=2147483648
readonly MAX_CHILD_FOOTPRINT_BYTES=8053063680
readonly MAX_HOST_WIRED_BYTES=8589934592

usage() {
  print -u2 "usage: $0 load-only|smoke-k8|smoke-k1|promotion-k8|promotion-k1"
  exit "$EX_USAGE"
}

refuse() {
  print -u2 "REFUSED: $1"
  exit "${2:-$EX_REFUSED}"
}

(( $# == 1 )) || usage
readonly stage="$1"
case "$stage" in
  load-only|smoke-k8|smoke-k1|promotion-k8|promotion-k1) ;;
  *) usage ;;
esac

readonly script_dir=${0:A:h}
readonly repo_root=$(/usr/bin/git -C "$script_dir" rev-parse --show-toplevel)
readonly watchdog="$repo_root/qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/run_load_only_watchdog.py"
readonly analyzer="$script_dir/hybrid_receipt.py"
readonly receipt_root=${CAMELID_HYBRID_RECEIPT_ROOT:-"$repo_root/qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot32-mapped-cold-2026-08-22-v1"}
readonly server_binary=${CAMELID_HYBRID_SERVER_BINARY:-"$receipt_root/camelid"}
readonly load_binary=${CAMELID_HYBRID_LOAD_BINARY:-"$receipt_root/gemma4-mtp-assistant-experiment"}
readonly model=${CAMELID_HYBRID_MODEL:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf}
readonly cghost=${CAMELID_HYBRID_CGHOST:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost}
readonly assistant=${CAMELID_HYBRID_ASSISTANT:-/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors}
readonly cam_lock=${CAMELID_CAM_LOCK:-/Users/timtoole/bin/cam-lock.sh}
readonly load_contract=${CAMELID_HYBRID_LOAD_CONTRACT:-"$receipt_root/hybrid-load-only-schema-v1.json"}
readonly telemetry_contract=${CAMELID_HYBRID_TELEMETRY_CONTRACT:-"$receipt_root/hybrid-telemetry-schema-v1.json"}

for key in \
  CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER \
  CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER \
  CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS; do
  if (( ${+parameters[$key]} )); then
    refuse "legacy physical-slot environment key is inherited: $key"
  fi
done

[[ -d "$receipt_root" && ! -L "$receipt_root" ]] || \
  refuse "receipt root must already be a real directory: $receipt_root"
[[ -x "$cam_lock" && ! -L "$cam_lock" ]] || refuse "missing host lock helper: $cam_lock"
[[ -f "$watchdog" && ! -L "$watchdog" ]] || refuse "missing watchdog: $watchdog"
[[ -f "$analyzer" && ! -L "$analyzer" ]] || refuse "missing analyzer: $analyzer"
for input in "$model" "$cghost" "$assistant"; do
  [[ -f "$input" ]] || refuse "missing required internal model input: $input"
done

typeset lane_kind expected_tokens lane_dir request expected_request_sha predecessor executable contract
case "$stage" in
  load-only)
    lane_kind=load-only
    expected_tokens=0
    lane_dir="$receipt_root/01-load-only"
    request=""
    expected_request_sha=""
    predecessor=""
    executable="$load_binary"
    contract="$load_contract"
    ;;
  smoke-k8)
    lane_kind=k8
    expected_tokens=9
    lane_dir="$receipt_root/02-smoke-9t/k8"
    request="$script_dir/request-9.json"
    expected_request_sha="a612ca079082b32a1cf80cd51f76d41ffe6f26cf22266089e148b9aed966a0d4"
    predecessor="$receipt_root/01-load-only/verdict.json"
    executable="$server_binary"
    contract="$telemetry_contract"
    ;;
  smoke-k1)
    lane_kind=k1
    expected_tokens=9
    lane_dir="$receipt_root/02-smoke-9t/k1"
    request="$script_dir/request-9.json"
    expected_request_sha="a612ca079082b32a1cf80cd51f76d41ffe6f26cf22266089e148b9aed966a0d4"
    predecessor="$receipt_root/02-smoke-9t/k8/verdict.json"
    executable="$server_binary"
    contract="$telemetry_contract"
    ;;
  promotion-k8)
    lane_kind=k8
    expected_tokens=48
    lane_dir="$receipt_root/03-promotion-48t/k8"
    request="$script_dir/request-48.json"
    expected_request_sha="b2f1110079fc726699cc936a628a268a7ec5bf2076fa970899de39d4ea903939"
    predecessor="$receipt_root/02-smoke-9t/parity.json"
    executable="$server_binary"
    contract="$telemetry_contract"
    [[ "${CAMELID_HYBRID_PROMOTION_ACK:-}" == "smoke-parity-reviewed" ]] || \
      refuse "promotion-k8 requires CAMELID_HYBRID_PROMOTION_ACK=smoke-parity-reviewed"
    ;;
  promotion-k1)
    lane_kind=k1
    expected_tokens=48
    lane_dir="$receipt_root/03-promotion-48t/k1"
    request="$script_dir/request-48.json"
    expected_request_sha="b2f1110079fc726699cc936a628a268a7ec5bf2076fa970899de39d4ea903939"
    predecessor="$receipt_root/03-promotion-48t/k8/verdict.json"
    executable="$server_binary"
    contract="$telemetry_contract"
    ;;
esac

[[ -x "$executable" && -f "$executable" && ! -L "$executable" ]] || \
  refuse "frozen executable is missing, non-regular, symlinked, or not executable: $executable"
[[ -z "$request" || ( -f "$request" && ! -L "$request" ) ]] || \
  refuse "request fixture is missing or symlinked: $request"
if [[ -n "$request" ]]; then
  typeset source_request_sha
  source_request_sha=$(/usr/bin/shasum -a 256 "$request" | /usr/bin/awk '{print $1}')
  [[ "$source_request_sha" == "$expected_request_sha" ]] || \
    refuse "request fixture bytes do not match the canonical stage request"
  /usr/bin/jq -e --argjson tokens "$expected_tokens" '
    .max_tokens == $tokens and
    .temperature == 0 and
    .top_k == 1 and
    .seed == 0 and
    .stream == false and
    .camelid_receipt == true and
    .camelid_enable_thinking == false
  ' "$request" >/dev/null || refuse "request fixture does not match the stage token/receipt contract"
fi

typeset predecessor_sha=""
if [[ -n "$predecessor" ]]; then
  [[ -f "$predecessor" && ! -L "$predecessor" ]] || \
    refuse "required predecessor PASS receipt is absent: $predecessor"
  /usr/bin/jq -e '.pass == true' "$predecessor" >/dev/null || \
    refuse "required predecessor did not pass: $predecessor"
  predecessor_sha=$(/usr/bin/shasum -a 256 "$predecessor" | /usr/bin/awk '{print $1}')
fi

readonly executable_sha=$(/usr/bin/shasum -a 256 "$executable" | /usr/bin/awk '{print $1}')
[[ -f "$contract" && ! -L "$contract" ]] || \
  refuse "required integration contract is absent; current binary is not admitted: $contract"
if [[ "$stage" == "load-only" ]]; then
  /usr/bin/jq -e --arg sha "$executable_sha" '
    .schema_version == 1 and
    .load_binary_sha256 == $sha and
    .test_name == "gemma4_mtp_assistant_load_only_probe" and
    .hybrid_hot_slots == 32 and
    .assistant_residency_receipted == true and
    .evict_page_cache == false
  ' "$contract" >/dev/null || refuse "load-only integration contract does not match the frozen binary"
else
  /usr/bin/jq -e --arg sha "$executable_sha" '
    .schema_version == 1 and
    .server_binary_sha256 == $sha and
    .response_field == "camelid.hybrid_telemetry" and
    .coverage == "every_completed_measured_round_and_layer" and
    .q4_assistant_head_fail_closed == true
  ' "$contract" >/dev/null || refuse "structured hybrid telemetry contract does not match the frozen server"
fi

readonly disk_available_kib=$(/bin/df -Pk /System/Volumes/Data | /usr/bin/awk 'NR == 2 {print $4}')
readonly disk_used_percent=$(/bin/df -Pk /System/Volumes/Data | /usr/bin/awk 'NR == 2 {gsub("%", "", $5); print $5}')
(( disk_available_kib >= MIN_DISK_AVAILABLE_KIB && disk_used_percent <= MAX_DISK_USED_PERCENT )) || \
  refuse "Data volume needs >=20 GiB available and <=90% used"

if /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  refuse "TCP port $PORT is already in use" "$EX_PORT_BUSY"
fi

typeset lock_dir="$receipt_root/.hybrid-runner.lock"
typeset lock_owned=0 cleanup_started=0 supervisor_pid="" child_pid="" child_pgid=""
/bin/mkdir -m 700 "$lock_dir" 2>/dev/null || \
  refuse "runner lock exists (review before removing): $lock_dir"
lock_owned=1

refresh_child_identity() {
  if [[ -s "$lane_dir/watchdog.jsonl" ]]; then
    [[ -n "$child_pid" ]] || child_pid=$(
      /usr/bin/jq -r 'select(.event == "child_started") | .pid' \
        "$lane_dir/watchdog.jsonl" 2>/dev/null | /usr/bin/tail -1
    )
    [[ -n "$child_pgid" ]] || child_pgid=$(
      /usr/bin/jq -r 'select(.event == "child_started") | .process_group' \
        "$lane_dir/watchdog.jsonl" 2>/dev/null | /usr/bin/tail -1
    )
    [[ "$child_pid" == "null" ]] && child_pid=""
    [[ "$child_pgid" == "null" ]] && child_pgid=""
  fi
}

group_alive() {
  [[ -n "$child_pgid" ]] && /bin/kill -0 -- "-$child_pgid" 2>/dev/null
}

cleanup() {
  set +e
  (( cleanup_started == 0 )) || return 0
  cleanup_started=1
  refresh_child_identity
  if group_alive; then
    /bin/kill -TERM -- "-$child_pgid" 2>/dev/null
  fi
  if [[ -n "$supervisor_pid" ]] && /bin/kill -0 "$supervisor_pid" 2>/dev/null; then
    /bin/kill -TERM "$supervisor_pid" 2>/dev/null
  fi
  /bin/sleep 1
  if group_alive; then
    /bin/kill -KILL -- "-$child_pgid" 2>/dev/null
  fi
  if [[ -n "$supervisor_pid" ]]; then
    wait "$supervisor_pid" 2>/dev/null
  fi
  if (( lock_owned == 1 )); then
    /bin/rmdir "$lock_dir" 2>/dev/null
    lock_owned=0
  fi
}

abort_on_signal() {
  typeset status="$1"
  trap - EXIT INT TERM HUP
  cleanup
  exit "$status"
}

trap cleanup EXIT
trap 'abort_on_signal 130' INT
trap 'abort_on_signal 143' TERM
trap 'abort_on_signal 129' HUP

typeset lane_parent=${lane_dir:h}
if [[ ! -e "$lane_parent" ]]; then
  /bin/mkdir -m 700 "$lane_parent"
fi
[[ -d "$lane_parent" && ! -L "$lane_parent" ]] || \
  refuse "lane parent is not a real directory: $lane_parent"
[[ ! -e "$lane_dir" && ! -L "$lane_dir" ]] || \
  refuse "receipt lane already exists: $lane_dir" "$EX_RECEIPT_EXISTS"
/bin/mkdir -m 700 "$lane_dir"

typeset frozen_request="" request_sha=""
if [[ -n "$request" ]]; then
  frozen_request="$lane_dir/request.json"
  /bin/cp "$request" "$frozen_request"
  /bin/chmod 600 "$frozen_request"
  [[ -f "$frozen_request" && ! -L "$frozen_request" ]] || \
    refuse "failed to freeze request fixture as a regular file"
  request_sha=$(/usr/bin/shasum -a 256 "$frozen_request" | /usr/bin/awk '{print $1}')
fi

readonly source_commit=$(/usr/bin/git -C "$repo_root" rev-parse HEAD)
readonly runner_sha=$(/usr/bin/shasum -a 256 "$0" | /usr/bin/awk '{print $1}')
readonly watchdog_sha=$(/usr/bin/shasum -a 256 "$watchdog" | /usr/bin/awk '{print $1}')
readonly analyzer_sha=$(/usr/bin/shasum -a 256 "$analyzer" | /usr/bin/awk '{print $1}')
readonly contract_sha=$(/usr/bin/shasum -a 256 "$contract" | /usr/bin/awk '{print $1}')
readonly run_nonce=$(/usr/bin/uuidgen)
readonly boot_identity=$(/usr/sbin/sysctl -n kern.boottime)
readonly model_size=$(/usr/bin/stat -f '%z' "$model")
readonly model_mtime=$(/usr/bin/stat -f '%m' "$model")
readonly cghost_size=$(/usr/bin/stat -f '%z' "$cghost")
readonly cghost_mtime=$(/usr/bin/stat -f '%m' "$cghost")
readonly assistant_size=$(/usr/bin/stat -f '%z' "$assistant")
readonly assistant_mtime=$(/usr/bin/stat -f '%m' "$assistant")

/usr/bin/jq -n \
  --arg nonce "$run_nonce" \
  --arg stage "$stage" \
  --arg lane "$lane_kind" \
  --arg commit "$source_commit" \
  --arg executable "$executable" \
  --arg executable_sha "$executable_sha" \
  --arg runner_sha "$runner_sha" \
  --arg watchdog_sha "$watchdog_sha" \
  --arg analyzer_sha "$analyzer_sha" \
  --arg contract "$contract" \
  --arg contract_sha "$contract_sha" \
  --arg request_source "$request" \
  --arg request_frozen "$frozen_request" \
  --arg request_sha "$request_sha" \
  --arg predecessor "$predecessor" \
  --arg predecessor_sha "$predecessor_sha" \
  --arg boot "$boot_identity" \
  --arg model "$model" --argjson model_size "$model_size" --argjson model_mtime "$model_mtime" \
  --arg cghost "$cghost" --argjson cghost_size "$cghost_size" --argjson cghost_mtime "$cghost_mtime" \
  --arg assistant "$assistant" --argjson assistant_size "$assistant_size" --argjson assistant_mtime "$assistant_mtime" \
  --argjson expected_tokens "$expected_tokens" \
  --argjson disk_available_kib "$disk_available_kib" \
  --argjson disk_used_percent "$disk_used_percent" '
  {
    schema_version: 1,
    nonce: $nonce,
    stage: $stage,
    lane: $lane,
    expected_tokens: $expected_tokens,
    source_commit: $commit,
    executable: {path: $executable, sha256: $executable_sha},
    tooling: {runner_sha256: $runner_sha, watchdog_sha256: $watchdog_sha, analyzer_sha256: $analyzer_sha},
    integration_contract: {path: $contract, sha256: $contract_sha},
    request: {source_path: $request_source, frozen_path: $request_frozen, sha256: $request_sha},
    predecessor: {path: $predecessor, sha256: $predecessor_sha},
    boot_identity: $boot,
    disk: {available_kib: $disk_available_kib, used_percent: $disk_used_percent},
    input_identity: {
      hash_policy: "stat-only-during-fresh-baseline",
      model: {path: $model, size: $model_size, mtime_epoch: $model_mtime},
      cghost: {path: $cghost, size: $cghost_size, mtime_epoch: $cghost_mtime},
      assistant: {path: $assistant, size: $assistant_size, mtime_epoch: $assistant_mtime}
    },
    profile: {
      demand_load_only: 1,
      file_mapped_experts: 1,
      hybrid_hot_slots: 32,
      slot_pin: 0,
      assistant_residency_policy: "observed_from_assistant_ledger",
      physical_slots_per_layer: "unset"
    }
  }' > "$lane_dir/.intent.json.tmp"
/bin/mv "$lane_dir/.intent.json.tmp" "$lane_dir/intent.json"

{
  /bin/date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  print -r -- "stage=$stage"
  print -r -- "source_commit=$source_commit"
  print -r -- "run_nonce=$run_nonce"
  /usr/sbin/sysctl kern.boottime vm.swapusage
  /usr/bin/memory_pressure -Q
  /usr/bin/vm_stat
  /bin/df -h /System/Volumes/Data
  /usr/bin/shasum -a 256 "$executable" "$0" "$watchdog" "$analyzer" "$contract"
  if [[ -n "$frozen_request" ]]; then
    /usr/bin/shasum -a 256 "$frozen_request"
  fi
} > "$lane_dir/.baseline.txt.tmp"
/bin/mv "$lane_dir/.baseline.txt.tmp" "$lane_dir/baseline.txt"

typeset -a common_env lane_env watchdog_args
common_env=(
  HOME=/Users/timtoole
  PATH=/usr/bin:/bin:/usr/sbin:/sbin
  TMPDIR=/tmp
  CAMELID_GHOST_ALLOW_LEGACY_SPARSE=0
  CAMELID_GEMMA4_GHOST_METAL=1
  CAMELID_GEMMA4_GHOST_METAL_SLOTS=1
  CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST=1
  CAMELID_GEMMA4_GHOST_METAL_TURBO=1
  CAMELID_GEMMA4_GHOST_METAL_COMMON=1
  CAMELID_GEMMA4_GHOST_METAL_CONTEXT=1024
  CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT=0
  CAMELID_GEMMA4_GHOST_READ_THREADS=1
  CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY=1
  CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32
  CAMELID_GEMMA4_SLOT_PIN=0
  CAMELID_GEMMA4_GHOST_METAL_HOT_PIN=0
  CAMELID_GEMMA4_VICTIM_CACHE=0
  CAMELID_GEMMA4_VICTIM_MB=0
  CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS=0
  CAMELID_GEMMA4_CHAINED_PREDICT=0
  CAMELID_SPEC_DECODE=off
  CAMELID_GEMMA4_SPEC_K1_LANE=chained
  CAMELID_GEMMA4_SPEC_TIMING=1
  CAMELID_GEMMA4_GHOST_METAL_TIMING=1
  CAMELID_GEMMA4_ROUTE_TRACE=1
)
if [[ "$lane_kind" == "k1" ]]; then
  lane_env=(
    CAMELID_GEMMA4_SPEC_CHUNK_MAX=1
    CAMELID_GEMMA4_SPEC_DRAFT_TOKENS=1
    CAMELID_GEMMA4_CHAINED_K1=1
  )
else
  lane_env=(
    CAMELID_GEMMA4_SPEC_CHUNK_MAX=8
    CAMELID_GEMMA4_SPEC_DRAFT_TOKENS=8
    CAMELID_GEMMA4_MTP_ASSISTANT_PATH="$assistant"
  )
fi
watchdog_args=(
  --baseline-soak-seconds 60
  --minimum-baseline-reclaimable-headroom-bytes "$MIN_BASELINE_HEADROOM_BYTES"
  --minimum-runtime-reclaimable-headroom-bytes "$MIN_RUNTIME_HEADROOM_BYTES"
  --maximum-child-physical-footprint-bytes "$MAX_CHILD_FOOTPRINT_BYTES"
  --maximum-host-wired-bytes "$MAX_HOST_WIRED_BYTES"
  --require-zero-current-swap
  --reject-swapin-growth
)

port_clear_receipt() {
  for _ in {1..10}; do
    if ! /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
      /usr/bin/jq -n --argjson port "$PORT" '{schema_version:1, port:$port, clear:true}' \
        > "$lane_dir/.port-clear.json.tmp"
      /bin/mv "$lane_dir/.port-clear.json.tmp" "$lane_dir/port-clear.json"
      return 0
    fi
    /bin/sleep 1
  done
  refuse "known child process group did not clear TCP port $PORT"
}

if [[ "$stage" == "load-only" ]]; then
  CAM_SESSION_PID=$$ "$cam_lock" /usr/bin/env -i \
    "${common_env[@]}" \
    "${lane_env[@]}" \
    CAMELID_GEMMA4_26B_GGUF="$model" \
    CAMELID_GEMMA4_26B_CGHOST="$cghost" \
    SPEC50_CACHE_MIB=0 \
    /usr/bin/python3 "$watchdog" \
      "${watchdog_args[@]}" \
      --report "$lane_dir/load-only-report.json" \
      --watchdog-log "$lane_dir/watchdog.jsonl" \
      --child-log "$lane_dir/child.log" \
      -- "$load_binary" \
        gemma4_mtp_assistant_load_only_probe \
        --ignored --exact --nocapture --test-threads=1
  port_clear_receipt
  /usr/bin/python3 "$analyzer" load-only \
    --lane-dir "$lane_dir" \
    --output "$lane_dir/verdict.json"
  print -r -- "STAGE_COMPLETE $stage"
  exit 0
fi

CAM_SESSION_PID=$$ "$cam_lock" /usr/bin/env -i \
  "${common_env[@]}" \
  "${lane_env[@]}" \
  /usr/bin/python3 "$watchdog" \
    "${watchdog_args[@]}" \
    --external-report-producer \
    --report "$lane_dir/response.json" \
    --watchdog-log "$lane_dir/watchdog.jsonl" \
    --child-log "$lane_dir/server.log" \
    -- "$server_binary" serve \
      --addr 127.0.0.1:$PORT \
      --model "$model" \
      --cghost "$cghost" \
      --expert-cache-mib 0 --gpu on --no-open &
supervisor_pid=$!

for _ in {1..1200}; do
  refresh_child_identity
  if [[ -n "$child_pid" && -n "$child_pgid" ]]; then
    [[ "$child_pid" == "$child_pgid" ]] || refuse "watchdog child PID/PGID identity drift"
    break
  fi
  if ! /bin/kill -0 "$supervisor_pid" 2>/dev/null; then
    set +e
    wait "$supervisor_pid"
    typeset status=$?
    set -e
    refuse "watchdog exited before child_started (status $status)"
  fi
  /bin/sleep 1
done
[[ -n "$child_pid" && -n "$child_pgid" ]] || refuse "watchdog never recorded child_started"

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
  if ! /bin/kill -0 "$supervisor_pid" 2>/dev/null; then
    set +e
    wait "$supervisor_pid"
    typeset status=$?
    set -e
    refuse "watchdog exited before readiness (status $status)"
  fi
  /bin/sleep 1
done
print -r -- "$health" > "$lane_dir/.health.json.tmp"
/bin/mv "$lane_dir/.health.json.tmp" "$lane_dir/health.json"
(( ready == 1 )) || refuse "server failed full-Metal readiness gate"

/usr/bin/curl -fsS --max-time 1800 \
  -H 'Content-Type: application/json' \
  --data-binary "@$frozen_request" \
  "http://127.0.0.1:$PORT/v1/chat/completions" > "$lane_dir/response.tmp"
/usr/bin/jq -e --argjson tokens "$expected_tokens" '
  .usage.completion_tokens == $tokens and
  (.camelid.generated_token_ids | length) == $tokens and
  .choices[0].finish_reason == "length" and
  (.choices[0].message.content | type) == "string"
' "$lane_dir/response.tmp" >/dev/null
/bin/mv "$lane_dir/response.tmp" "$lane_dir/response.json"

/bin/kill -INT "$child_pid"
set +e
wait "$supervisor_pid"
typeset supervisor_status=$?
set -e
supervisor_pid=""
(( supervisor_status == 0 )) || refuse "watchdog supervisor returned $supervisor_status"
port_clear_receipt
child_pid=""
child_pgid=""

/usr/bin/python3 "$analyzer" lane \
  --lane-dir "$lane_dir" \
  --lane "$lane_kind" \
  --expected-tokens "$expected_tokens" \
  --output "$lane_dir/verdict.json"

if [[ "$stage" == "smoke-k1" ]]; then
  /usr/bin/python3 "$analyzer" parity \
    --receipt-root "$receipt_root" \
    --stage smoke \
    --output "$receipt_root/02-smoke-9t/parity.json"
elif [[ "$stage" == "promotion-k1" ]]; then
  /usr/bin/python3 "$analyzer" parity \
    --receipt-root "$receipt_root" \
    --stage promotion \
    --output "$receipt_root/03-promotion-48t/parity.json"
fi

print -r -- "STAGE_COMPLETE $stage"
