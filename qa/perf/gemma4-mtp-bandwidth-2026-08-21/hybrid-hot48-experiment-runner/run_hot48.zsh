#!/bin/zsh
set -euo pipefail
umask 077

readonly EX_REFUSED=75
readonly EX_PORT_BUSY=76
readonly EX_RECEIPT_EXISTS=77
readonly PORT=8189
readonly PROTECTED_PORT=8181
readonly MIN_DISK_AVAILABLE_KIB=20971520
readonly MIN_BASELINE_HEADROOM_BYTES=8589934592
readonly MIN_RUNTIME_HEADROOM_BYTES=2147483648
readonly MAX_CHILD_FOOTPRINT_BYTES=8053063680
readonly MAX_HOST_WIRED_BYTES=8589934592
readonly REQUEST_SHA256=b2f1110079fc726699cc936a628a268a7ec5bf2076fa970899de39d4ea903939
readonly EXPECTED_IDS_SHA256=45e65ac09155d7627373c262f1edd1faf6188fb6dad26c5d5994fe5226a97975

refuse() {
  print -u2 "REFUSED: $1"
  exit "${2:-$EX_REFUSED}"
}

(( $# == 0 )) || refuse "this exact benchmark accepts no positional arguments"

readonly script_dir=${0:A:h}
readonly runner=${0:A}
readonly repo_root=$(/usr/bin/git -C "$script_dir" rev-parse --show-toplevel)
readonly analyzer="$script_dir/analyze_hot48.py"
readonly watchdog="$repo_root/qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/run_load_only_watchdog.py"
readonly request_source="$script_dir/request-48.json"
readonly expected_ids_source="$script_dir/expected-48-token-ids.json"
typeset source_worktree_status
source_worktree_status=$(/usr/bin/git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
[[ -z "$source_worktree_status" ]] || \
  refuse "repository must be completely clean before admitting a provenance-bound run"
(( ${+parameters[CAMELID_HOT48_BINARY]} )) || \
  refuse "set CAMELID_HOT48_BINARY to an absolute prebuilt binary on the internal Data volume"
readonly binary=$CAMELID_HOT48_BINARY
readonly model=${CAMELID_HOT48_MODEL:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf}
readonly cghost=${CAMELID_HOT48_CGHOST:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost}
readonly assistant=${CAMELID_HOT48_ASSISTANT:-/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors}
readonly run_id=$(/bin/date -u '+%Y%m%dT%H%M%SZ')-$$
readonly default_receipt_root="$repo_root/qa/perf/gemma4-mtp-bandwidth-2026-08-21/hot48-experiment-$run_id"
readonly receipt_root=${CAMELID_HOT48_RECEIPT_ROOT:-"$default_receipt_root"}
readonly lock_dir="$script_dir/.hot48-experiment.lock"

for key in \
  CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER \
  CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER \
  CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER; do
  if (( ${+parameters[$key]} )); then
    refuse "inherited per-layer slot override is forbidden: $key"
  fi
done

[[ "$receipt_root" == /* ]] || refuse "receipt root must be absolute"
[[ ! -e "$receipt_root" && ! -L "$receipt_root" ]] || \
  refuse "receipt root already exists: $receipt_root" "$EX_RECEIPT_EXISTS"
[[ -d "${receipt_root:h}" && ! -L "${receipt_root:h}" ]] || \
  refuse "receipt parent must already be a real directory: ${receipt_root:h}"

readonly data_volume_device=$(/usr/bin/stat -f '%d' /System/Volumes/Data)
internal_data_path() {
  [[ -e "$1" && ! -L "$1" ]] || return 1
  [[ "$(/usr/bin/stat -f '%d' "$1")" == "$data_volume_device" ]]
}
internal_data_path "$repo_root" || refuse "repository working directory is not on the internal Data volume"
internal_data_path "${receipt_root:h}" || refuse "receipt parent is not on the internal Data volume"

for input in "$model" "$cghost" "$assistant" "$request_source" \
  "$expected_ids_source" "$analyzer" "$watchdog" "$runner"; do
  [[ -f "$input" && ! -L "$input" ]] || \
    refuse "required input is missing, non-regular, or symlinked: $input"
done
[[ -x "$binary" && -f "$binary" && ! -L "$binary" ]] || \
  refuse "prebuilt binary is missing, non-regular, symlinked, or not executable: $binary"
for input in "$binary" "$model" "$cghost" "$assistant"; do
  internal_data_path "$input" || refuse "runtime input is not on the internal Data volume: $input"
done

readonly disk_available_kib=$(/bin/df -Pk /System/Volumes/Data | /usr/bin/awk 'NR == 2 {print $4}')
[[ "$disk_available_kib" == <-> ]] || refuse "could not measure Data-volume disk headroom"
(( disk_available_kib >= MIN_DISK_AVAILABLE_KIB )) || \
  refuse "Data volume needs at least 20 GiB available before the benchmark"

typeset request_source_sha expected_ids_source_sha
request_source_sha=$(/usr/bin/shasum -a 256 "$request_source" | /usr/bin/awk '{print $1}')
expected_ids_source_sha=$(/usr/bin/shasum -a 256 "$expected_ids_source" | /usr/bin/awk '{print $1}')
[[ "$request_source_sha" == "$REQUEST_SHA256" ]] || \
  refuse "frozen request fixture SHA-256 drifted"
[[ "$expected_ids_source_sha" == "$EXPECTED_IDS_SHA256" ]] || \
  refuse "frozen expected-token fixture SHA-256 drifted"
/usr/bin/jq -e '
  .max_tokens == 48 and .temperature == 0 and .top_k == 1 and .seed == 0 and
  .stream == false and .camelid_receipt == true and
  .camelid_enable_thinking == false
' "$request_source" >/dev/null || refuse "request fixture is not the exact deterministic contract"
/usr/bin/jq -e '
  type == "array" and length == 48 and
  all(.[]; type == "number" and floor == . and . >= 0 and . <= 4294967295)
' "$expected_ids_source" >/dev/null || refuse "expected-token fixture is not 48 uint32 IDs"

port_listening() {
  /usr/sbin/lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

port_listening "$PROTECTED_PORT" && \
  refuse "the user WebUI engine is active on protected port $PROTECTED_PORT" "$EX_PORT_BUSY"
port_listening "$PORT" && \
  refuse "benchmark port $PORT is already in use" "$EX_PORT_BUSY"

typeset lock_owned=0 cleanup_started=0 supervisor_pid="" child_pid="" child_pgid=""
/bin/mkdir -m 700 "$lock_dir" 2>/dev/null || \
  refuse "another Hot48 benchmark owns the runner lock: $lock_dir"
lock_owned=1

refresh_child_identity() {
  if [[ -s "$receipt_root/watchdog.jsonl" ]]; then
    if [[ -z "$child_pid" ]]; then
      child_pid=$(/usr/bin/jq -r 'select(.event == "child_started") | .pid' \
        "$receipt_root/watchdog.jsonl" 2>/dev/null | /usr/bin/tail -1)
      [[ "$child_pid" == <-> ]] || child_pid=""
    fi
    if [[ -z "$child_pgid" ]]; then
      child_pgid=$(/usr/bin/jq -r 'select(.event == "child_started") | .process_group' \
        "$receipt_root/watchdog.jsonl" 2>/dev/null | /usr/bin/tail -1)
      [[ "$child_pgid" == <-> ]] || child_pgid=""
    fi
  fi
}

group_alive() {
  [[ "$child_pgid" == <-> ]] && /bin/kill -0 -- "-$child_pgid" 2>/dev/null
}

cleanup() {
  set +e
  (( cleanup_started == 0 )) || return 0
  cleanup_started=1
  refresh_child_identity
  if group_alive; then
    /bin/kill -TERM -- "-$child_pgid" 2>/dev/null
  fi
  if [[ "$supervisor_pid" == <-> ]] && /bin/kill -0 "$supervisor_pid" 2>/dev/null; then
    /bin/kill -TERM "$supervisor_pid" 2>/dev/null
  fi
  /bin/sleep 1
  if group_alive; then
    /bin/kill -KILL -- "-$child_pgid" 2>/dev/null
  fi
  if [[ "$supervisor_pid" == <-> ]]; then
    wait "$supervisor_pid" 2>/dev/null
  fi
  if (( lock_owned == 1 )); then
    /bin/rmdir "$lock_dir" 2>/dev/null
    lock_owned=0
  fi
}

abort_signal() {
  typeset status="$1"
  trap - EXIT INT TERM HUP
  cleanup
  exit "$status"
}

trap cleanup EXIT
trap 'abort_signal 130' INT
trap 'abort_signal 143' TERM
trap 'abort_signal 129' HUP

/bin/mkdir -m 700 "$receipt_root"
readonly request_frozen="$receipt_root/request.json"
readonly expected_ids_frozen="$receipt_root/expected-token-ids.json"
/bin/cp "$request_source" "$request_frozen"
/bin/cp "$expected_ids_source" "$expected_ids_frozen"
/bin/chmod 600 "$request_frozen" "$expected_ids_frozen"
[[ -f "$request_frozen" && ! -L "$request_frozen" ]] || refuse "request freeze failed"
[[ -f "$expected_ids_frozen" && ! -L "$expected_ids_frozen" ]] || refuse "token-ID freeze failed"

print -r -- "Hashing the prebuilt binary, model inputs, frozen fixtures, and tooling..."
typeset binary_sha model_sha cghost_sha assistant_sha request_frozen_sha expected_ids_frozen_sha
typeset runner_sha analyzer_sha watchdog_sha
binary_sha=$(/usr/bin/shasum -a 256 "$binary" | /usr/bin/awk '{print $1}')
model_sha=$(/usr/bin/shasum -a 256 "$model" | /usr/bin/awk '{print $1}')
cghost_sha=$(/usr/bin/shasum -a 256 "$cghost" | /usr/bin/awk '{print $1}')
assistant_sha=$(/usr/bin/shasum -a 256 "$assistant" | /usr/bin/awk '{print $1}')
request_frozen_sha=$(/usr/bin/shasum -a 256 "$request_frozen" | /usr/bin/awk '{print $1}')
expected_ids_frozen_sha=$(/usr/bin/shasum -a 256 "$expected_ids_frozen" | /usr/bin/awk '{print $1}')
runner_sha=$(/usr/bin/shasum -a 256 "$runner" | /usr/bin/awk '{print $1}')
analyzer_sha=$(/usr/bin/shasum -a 256 "$analyzer" | /usr/bin/awk '{print $1}')
watchdog_sha=$(/usr/bin/shasum -a 256 "$watchdog" | /usr/bin/awk '{print $1}')
[[ "$request_frozen_sha" == "$REQUEST_SHA256" ]] || refuse "frozen request copy drifted"
[[ "$expected_ids_frozen_sha" == "$EXPECTED_IDS_SHA256" ]] || refuse "frozen token-ID copy drifted"

readonly source_commit=$(/usr/bin/git -C "$repo_root" rev-parse HEAD)
readonly run_nonce=$(/usr/bin/uuidgen)
readonly boot_identity=$(/usr/sbin/sysctl -n kern.boottime)
typeset binary_size model_size cghost_size assistant_size request_source_size request_frozen_size
typeset expected_source_size expected_frozen_size runner_size analyzer_size watchdog_size
binary_size=$(/usr/bin/stat -f '%z' "$binary")
model_size=$(/usr/bin/stat -f '%z' "$model")
cghost_size=$(/usr/bin/stat -f '%z' "$cghost")
assistant_size=$(/usr/bin/stat -f '%z' "$assistant")
request_source_size=$(/usr/bin/stat -f '%z' "$request_source")
request_frozen_size=$(/usr/bin/stat -f '%z' "$request_frozen")
expected_source_size=$(/usr/bin/stat -f '%z' "$expected_ids_source")
expected_frozen_size=$(/usr/bin/stat -f '%z' "$expected_ids_frozen")
runner_size=$(/usr/bin/stat -f '%z' "$runner")
analyzer_size=$(/usr/bin/stat -f '%z' "$analyzer")
watchdog_size=$(/usr/bin/stat -f '%z' "$watchdog")

/usr/bin/jq -n \
  --arg source_commit "$source_commit" --arg nonce "$run_nonce" --arg boot "$boot_identity" \
  --argjson disk_available_kib "$disk_available_kib" \
  --arg binary "$binary" --arg binary_sha "$binary_sha" --argjson binary_size "$binary_size" \
  --arg model "$model" --arg model_sha "$model_sha" --argjson model_size "$model_size" \
  --arg cghost "$cghost" --arg cghost_sha "$cghost_sha" --argjson cghost_size "$cghost_size" \
  --arg assistant "$assistant" --arg assistant_sha "$assistant_sha" --argjson assistant_size "$assistant_size" \
  --arg request_source "$request_source" --arg request_source_sha "$request_source_sha" --argjson request_source_size "$request_source_size" \
  --arg request_frozen "$request_frozen" --arg request_frozen_sha "$request_frozen_sha" --argjson request_frozen_size "$request_frozen_size" \
  --arg expected_source "$expected_ids_source" --arg expected_source_sha "$expected_ids_source_sha" --argjson expected_source_size "$expected_source_size" \
  --arg expected_frozen "$expected_ids_frozen" --arg expected_frozen_sha "$expected_ids_frozen_sha" --argjson expected_frozen_size "$expected_frozen_size" \
  --arg runner "$runner" --arg runner_sha "$runner_sha" --argjson runner_size "$runner_size" \
  --arg analyzer "$analyzer" --arg analyzer_sha "$analyzer_sha" --argjson analyzer_size "$analyzer_size" \
  --arg watchdog "$watchdog" --arg watchdog_sha "$watchdog_sha" --argjson watchdog_size "$watchdog_size" '
  def artifact($path; $sha; $size): {path: $path, sha256: $sha, size_bytes: $size};
  {
    schema_version: 1,
    benchmark: "gemma4-uniform-hot48-experiment",
    source_commit: $source_commit,
    source_worktree_clean: true,
    nonce: $nonce,
    boot_identity: $boot,
    expected_tokens: 48,
    port: 8189,
    protected_port: 8181,
    disk: {
      available_kib: $disk_available_kib,
      minimum_available_kib: 20971520,
      volume: "/System/Volumes/Data"
    },
    geometry: {
      layers: 30,
      logical_slots_per_layer: 128,
      hot_slots_per_layer: 48,
      anonymous_hot_capacity_slots: 1440,
      anonymous_hot_capacity_bytes: 4836556800,
      file_mapped_addressable_slots: 3840,
      file_mapped_address_span_bytes: 12897484800,
      overflow_slots: 0,
      victim_slots: 0,
      host_cache_budget_bytes: 0
    },
    watchdog_contract: {
      schema_version: 3,
      sample_period_ns: 250000000,
      baseline_soak_seconds: 60,
      minimum_baseline_reclaimable_headroom_bytes: 8589934592,
      minimum_runtime_reclaimable_headroom_bytes: 2147483648,
      maximum_child_physical_footprint_bytes: 8053063680,
      maximum_host_wired_bytes: 8589934592,
      reject_swapin_growth: true,
      require_zero_current_swap: false
    },
    artifacts: {
      binary: artifact($binary; $binary_sha; $binary_size),
      model: artifact($model; $model_sha; $model_size),
      cghost: artifact($cghost; $cghost_sha; $cghost_size),
      assistant: artifact($assistant; $assistant_sha; $assistant_size),
      request_source: artifact($request_source; $request_source_sha; $request_source_size),
      request_frozen: artifact($request_frozen; $request_frozen_sha; $request_frozen_size),
      expected_ids_source: artifact($expected_source; $expected_source_sha; $expected_source_size),
      expected_ids_frozen: artifact($expected_frozen; $expected_frozen_sha; $expected_frozen_size),
      runner: artifact($runner; $runner_sha; $runner_size),
      analyzer: artifact($analyzer; $analyzer_sha; $analyzer_size),
      watchdog: artifact($watchdog; $watchdog_sha; $watchdog_size)
    }
  }' > "$receipt_root/.intent.json.tmp"
/bin/mv "$receipt_root/.intent.json.tmp" "$receipt_root/intent.json"

{
  /bin/date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  print -r -- "source_commit=$source_commit"
  print -r -- "source_worktree_clean=true"
  print -r -- "nonce=$run_nonce"
  print -r -- "binary_sha256=$binary_sha $binary"
  print -r -- "model_sha256=$model_sha $model"
  print -r -- "cghost_sha256=$cghost_sha $cghost"
  print -r -- "assistant_sha256=$assistant_sha $assistant"
  print -r -- "request_sha256=$request_frozen_sha $request_frozen"
  print -r -- "expected_ids_sha256=$expected_ids_frozen_sha $expected_ids_frozen"
  print -r -- "runner_sha256=$runner_sha $runner"
  print -r -- "analyzer_sha256=$analyzer_sha $analyzer"
  print -r -- "watchdog_sha256=$watchdog_sha $watchdog"
  /usr/sbin/sysctl kern.boottime vm.swapusage
  /usr/bin/memory_pressure -Q
  /usr/bin/vm_stat
  /bin/df -h /System/Volumes/Data
} > "$receipt_root/.baseline.txt.tmp"
/bin/mv "$receipt_root/.baseline.txt.tmp" "$receipt_root/baseline.txt"

typeset -a experiment_env watchdog_args
experiment_env=(
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
  CAMELID_GEMMA4_GHOST_READ_THREADS=8
  CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY=1
  CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT48_EXPERIMENT=1
  CAMELID_GEMMA4_GHOST_METAL_DECODE_PROMOTION=0
  CAMELID_GEMMA4_GHOST_METAL_MAPPED_READAHEAD=1
  CAMELID_GEMMA4_SLOT_PIN=0
  CAMELID_GEMMA4_GHOST_METAL_HOT_PIN=0
  CAMELID_GEMMA4_VICTIM_CACHE=0
  CAMELID_GEMMA4_VICTIM_MB=0
  CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS=0
  CAMELID_GEMMA4_CHAINED_PREDICT=0
  CAMELID_SPEC_DECODE=off
  CAMELID_GEMMA4_SPEC_K1_LANE=chained
  CAMELID_GEMMA4_SPEC_CHUNK_MAX=8
  CAMELID_GEMMA4_SPEC_DRAFT_TOKENS=8
  CAMELID_GEMMA4_MTP_ASSISTANT_PATH="$assistant"
  CAMELID_GEMMA4_MTP_DEVICE_CHAIN=1
  CAMELID_GEMMA4_MTP_FULL_Q4=1
  CAMELID_GEMMA4_MTP_PREFILL_SEED_BOOTSTRAP=1
  CAMELID_GEMMA4_DENSE_K8_GENERIC=1
  CAMELID_GEMMA4_HEAD_SPEC50_K8_COMPACT=1
  CAMELID_GEMMA4_SPEC_TIMING=1
  CAMELID_GEMMA4_GHOST_METAL_TIMING=1
  CAMELID_GEMMA4_ROUTE_TRACE=1
)
watchdog_args=(
  --baseline-soak-seconds 60
  --minimum-baseline-reclaimable-headroom-bytes "$MIN_BASELINE_HEADROOM_BYTES"
  --minimum-runtime-reclaimable-headroom-bytes "$MIN_RUNTIME_HEADROOM_BYTES"
  --maximum-child-physical-footprint-bytes "$MAX_CHILD_FOOTPRINT_BYTES"
  --maximum-host-wired-bytes "$MAX_HOST_WIRED_BYTES"
  --reject-swapin-growth
)

cd "$repo_root"
/usr/bin/env -i \
  "${experiment_env[@]}" \
  /usr/bin/python3 "$watchdog" \
    "${watchdog_args[@]}" \
    --external-report-producer \
    --report "$receipt_root/response.json" \
    --watchdog-log "$receipt_root/watchdog.jsonl" \
    --child-log "$receipt_root/server.log" \
    -- "$binary" serve \
      --addr 127.0.0.1:$PORT \
      --model "$model" \
      --cghost "$cghost" \
      --expert-cache-mib 0 --gpu on --no-open &
supervisor_pid=$!

for _ in {1..360}; do
  refresh_child_identity
  if [[ -n "$child_pid" && -n "$child_pgid" ]]; then
    [[ "$child_pid" == "$child_pgid" ]] || refuse "watchdog child PID/PGID identity drifted"
    break
  fi
  if port_listening "$PROTECTED_PORT"; then
    refuse "the user WebUI engine started on protected port $PROTECTED_PORT" "$EX_PORT_BUSY"
  fi
  if ! /bin/kill -0 "$supervisor_pid" 2>/dev/null; then
    set +e
    wait "$supervisor_pid"
    typeset early_status=$?
    set -e
    supervisor_pid=""
    refuse "watchdog exited before child_started (status $early_status)"
  fi
  /bin/sleep 1
done
[[ -n "$child_pid" && -n "$child_pgid" ]] || refuse "watchdog never recorded child_started"

typeset health="" ready=0
for _ in {1..900}; do
  if port_listening "$PROTECTED_PORT"; then
    refuse "the user WebUI engine started on protected port $PROTECTED_PORT" "$EX_PORT_BUSY"
  fi
  if health=$(/usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/health" 2>/dev/null); then
    if print -r -- "$health" | /usr/bin/jq -e --arg source_commit "$source_commit" '
      (.build // "") as $build |
      .source_commit == $source_commit and
      ($build | type == "string") and
      ($build | length > 0) and
      ($build | endswith("-dirty") | not) and
      .generation_ready == true and
      .gemma4_serve_lane == "ghost_moe" and
      .gemma4_ghost_execution_mode == "full_common_metal" and
      .gemma4_ghost_common_metal_active == true and
      .gemma4_ghost_experts_metal_active == true and
      .gemma4_ghost_head_metal_active == true and
      .gemma4_mtp_assistant_loaded == true and
      .gemma4_mtp_full_q4_active == true
    ' >/dev/null; then
      ready=1
      break
    fi
  fi
  if ! /bin/kill -0 "$supervisor_pid" 2>/dev/null; then
    set +e
    wait "$supervisor_pid"
    typeset readiness_status=$?
    set -e
    supervisor_pid=""
    refuse "watchdog exited before full readiness (status $readiness_status)"
  fi
  /bin/sleep 1
done
print -r -- "$health" > "$receipt_root/.health.json.tmp"
/bin/mv "$receipt_root/.health.json.tmp" "$receipt_root/health.json"
(( ready == 1 )) || refuse "server did not reach full-Metal + full-Q4 MTP readiness"

/usr/bin/curl -fsS --max-time 1800 \
  -H 'Content-Type: application/json' \
  --data-binary "@$request_frozen" \
  "http://127.0.0.1:$PORT/v1/chat/completions" > "$receipt_root/response.tmp"
/usr/bin/jq -e '
  .usage.completion_tokens == 48 and
  (.camelid.generated_token_ids | length) == 48 and
  .choices[0].finish_reason == "length"
' "$receipt_root/response.tmp" >/dev/null || refuse "response did not reach the exact 48-token boundary"
/bin/mv "$receipt_root/response.tmp" "$receipt_root/response.json"

/bin/kill -INT "$child_pid"
set +e
wait "$supervisor_pid"
typeset supervisor_status=$?
set -e
supervisor_pid=""
(( supervisor_status == 0 )) || refuse "watchdog supervisor returned $supervisor_status"
child_pid=""
child_pgid=""

for _ in {1..20}; do
  if ! port_listening "$PROTECTED_PORT" && ! port_listening "$PORT"; then
    /usr/bin/jq -n '{schema_version:1, ports:{"8181":{clear:true}, "8189":{clear:true}}}' \
      > "$receipt_root/.port-clear.json.tmp"
    /bin/mv "$receipt_root/.port-clear.json.tmp" "$receipt_root/port-clear.json"
    break
  fi
  /bin/sleep 1
done
[[ -f "$receipt_root/port-clear.json" ]] || \
  refuse "ports 8181/8189 did not both clear after the supervised run" "$EX_PORT_BUSY"

/usr/bin/python3 "$analyzer" \
  --run-dir "$receipt_root" \
  --output "$receipt_root/verdict.json"

print -r -- "HOT48_EXPERIMENT_COMPLETE $receipt_root"
