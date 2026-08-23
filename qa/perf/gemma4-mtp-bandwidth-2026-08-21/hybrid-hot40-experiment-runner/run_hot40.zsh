#!/bin/zsh
set -euo pipefail
umask 077

readonly EX_REFUSED=75
readonly EX_PORT_BUSY=76
readonly EX_RECEIPT_EXISTS=77
readonly PORT=8189
readonly PROTECTED_PORT=8181
readonly MIN_DISK_AVAILABLE_KIB=20971520
readonly MIN_BASELINE_HEADROOM_BYTES=8053063680
readonly MIN_RUNTIME_HEADROOM_BYTES=2147483648
readonly MAX_CHILD_FOOTPRINT_BYTES=8053063680
readonly MAX_HOST_WIRED_BYTES=8589934592
readonly REQUEST_SHA256=b2f1110079fc726699cc936a628a268a7ec5bf2076fa970899de39d4ea903939
readonly EXPECTED_IDS_SHA256=45e65ac09155d7627373c262f1edd1faf6188fb6dad26c5d5994fe5226a97975
readonly DEFAULT_MODEL=/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf
readonly DEFAULT_CGHOST=/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost
readonly DEFAULT_ASSISTANT=/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors
readonly DEFAULT_MODEL_SHA256=66bfa72e759bfa8509634ec0589057df4283183ab4927635c110819690fe972d
readonly DEFAULT_CGHOST_SHA256=b3352d21b6c84abf2950f4551a9b47606f2cb003acde6e839118313c51aa3757
readonly DEFAULT_ASSISTANT_SHA256=c082cc581c3ec90d70285c1a41c81544ff56cbc96650f16c900a280940655801
readonly DEFAULT_PROVENANCE_REFERENCE=/Users/timtoole/Documents/Camelid-hot48-receipts/run-hot48-301da730-v1/intent.json
readonly DEFAULT_PROVENANCE_REFERENCE_SHA256=1009914429ef0417b44f2886299fab6f6be64165afa78e72fe8d6e7ebd715cc3
readonly PROVENANCE_LIMITATION="Large model, cghost, and assistant contents are not hashed by this run. Historical SHA-256 claims are carried forward and bound only to live device/inode/size/mtime stat identity, avoiding pre-spawn page-cache pollution but providing weaker provenance than a fresh content hash."

refuse() {
  print -u2 "REFUSED: $1"
  exit "${2:-$EX_REFUSED}"
}

(( $# == 0 )) || refuse "this exact benchmark accepts no positional arguments"

readonly script_dir=${0:A:h}
readonly runner=${0:A}
readonly repo_root=$(/usr/bin/git -C "$script_dir" rev-parse --show-toplevel)
readonly analyzer="$script_dir/analyze_hot40.py"
readonly host_sampler="$script_dir/capture_host_memory.py"
readonly hasher="$script_dir/sha256_nocache.py"
readonly watchdog="$repo_root/qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/run_load_only_watchdog.py"
readonly request_source="$script_dir/request-48.json"
readonly expected_ids_source="$script_dir/expected-48-token-ids.json"
readonly harness_commit=$(/usr/bin/git -C "$repo_root" rev-parse --verify "HEAD^{commit}")
typeset source_worktree_status
source_worktree_status=$(/usr/bin/git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
[[ -z "$source_worktree_status" ]] || \
  refuse "repository must be completely clean before admitting a provenance-bound run"
typeset binary_source_revision="$harness_commit"
typeset binary_source_defaulted=true
if (( ${+parameters[CAMELID_HOT40_BINARY_SOURCE_COMMIT]} )); then
  binary_source_revision=$CAMELID_HOT40_BINARY_SOURCE_COMMIT
  binary_source_defaulted=false
fi
[[ -n "$binary_source_revision" ]] || \
  refuse "CAMELID_HOT40_BINARY_SOURCE_COMMIT must not be empty"
typeset canonical_binary_source_commit
canonical_binary_source_commit=$(
  /usr/bin/git -C "$repo_root" rev-parse --verify --end-of-options \
    "${binary_source_revision}^{commit}" 2>/dev/null
) || refuse "CAMELID_HOT40_BINARY_SOURCE_COMMIT does not resolve to a commit"
print -rn -- "$canonical_binary_source_commit" | \
  /usr/bin/grep -Eq '^[0-9a-f]{40}$' || \
  refuse "binary source commit did not canonicalize to a full 40-character commit"
readonly binary_source_commit="$canonical_binary_source_commit"
/usr/bin/git -C "$repo_root" merge-base --is-ancestor \
  "$binary_source_commit" "$harness_commit" || \
  refuse "binary source commit must be an ancestor of the harness commit"
/usr/bin/git -C "$repo_root" diff --quiet \
  "$binary_source_commit" "$harness_commit" -- \
  src Cargo.toml Cargo.lock build.rs || \
  refuse "runtime source differs between the binary and harness commits"
readonly source_commit="$binary_source_commit"
(( ${+parameters[CAMELID_HOT40_BINARY]} )) || \
  refuse "set CAMELID_HOT40_BINARY to an absolute prebuilt binary on the internal Data volume"
readonly binary=$CAMELID_HOT40_BINARY
readonly model=${CAMELID_HOT40_MODEL:-$DEFAULT_MODEL}
readonly cghost=${CAMELID_HOT40_CGHOST:-$DEFAULT_CGHOST}
readonly assistant=${CAMELID_HOT40_ASSISTANT:-$DEFAULT_ASSISTANT}
readonly run_id=$(/bin/date -u '+%Y%m%dT%H%M%SZ')-$$
readonly default_receipt_root="$repo_root/qa/perf/gemma4-mtp-bandwidth-2026-08-21/hot40-experiment-$run_id"
readonly receipt_root=${CAMELID_HOT40_RECEIPT_ROOT:-"$default_receipt_root"}
readonly lock_dir="$script_dir/.hot40-experiment.lock"

typeset model_preverified_sha="$DEFAULT_MODEL_SHA256"
typeset cghost_preverified_sha="$DEFAULT_CGHOST_SHA256"
typeset assistant_preverified_sha="$DEFAULT_ASSISTANT_SHA256"
typeset provenance_reference="$DEFAULT_PROVENANCE_REFERENCE"
typeset provenance_reference_sha="$DEFAULT_PROVENANCE_REFERENCE_SHA256"
typeset custom_large_input=0
if [[ "$model" != "$DEFAULT_MODEL" ]]; then
  (( ${+parameters[CAMELID_HOT40_MODEL_PREVERIFIED_SHA256]} )) || \
    refuse "a model override requires CAMELID_HOT40_MODEL_PREVERIFIED_SHA256"
  model_preverified_sha=$CAMELID_HOT40_MODEL_PREVERIFIED_SHA256
  custom_large_input=1
fi
if [[ "$cghost" != "$DEFAULT_CGHOST" ]]; then
  (( ${+parameters[CAMELID_HOT40_CGHOST_PREVERIFIED_SHA256]} )) || \
    refuse "a cghost override requires CAMELID_HOT40_CGHOST_PREVERIFIED_SHA256"
  cghost_preverified_sha=$CAMELID_HOT40_CGHOST_PREVERIFIED_SHA256
  custom_large_input=1
fi
if [[ "$assistant" != "$DEFAULT_ASSISTANT" ]]; then
  (( ${+parameters[CAMELID_HOT40_ASSISTANT_PREVERIFIED_SHA256]} )) || \
    refuse "an assistant override requires CAMELID_HOT40_ASSISTANT_PREVERIFIED_SHA256"
  assistant_preverified_sha=$CAMELID_HOT40_ASSISTANT_PREVERIFIED_SHA256
  custom_large_input=1
fi
for sha in "$model_preverified_sha" "$cghost_preverified_sha" "$assistant_preverified_sha"; do
  print -rn -- "$sha" | /usr/bin/grep -Eq '^[0-9a-f]{64}$' || \
    refuse "preverified SHA-256 metadata must be canonical lowercase hex"
done
if (( custom_large_input == 1 )); then
  (( ${+parameters[CAMELID_HOT40_PREVERIFIED_PROVENANCE]} )) || \
    refuse "large-input overrides require CAMELID_HOT40_PREVERIFIED_PROVENANCE"
  provenance_reference=$CAMELID_HOT40_PREVERIFIED_PROVENANCE
  [[ -n "$provenance_reference" ]] || refuse "preverified provenance reference must be nonempty"
  provenance_reference_sha=""
fi

for key in \
  CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER \
  CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER \
  CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT48_EXPERIMENT; do
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
  "$expected_ids_source" "$analyzer" "$host_sampler" "$hasher" "$watchdog" "$runner"; do
  [[ -f "$input" && ! -L "$input" ]] || \
    refuse "required input is missing, non-regular, or symlinked: $input"
done
[[ -x "$binary" && -f "$binary" && ! -L "$binary" ]] || \
  refuse "prebuilt binary is missing, non-regular, symlinked, or not executable: $binary"
for input in "$binary" "$model" "$cghost" "$assistant"; do
  [[ "$input" == /* ]] || refuse "runtime input path must be absolute: $input"
  internal_data_path "$input" || refuse "runtime input is not on the internal Data volume: $input"
done

# Large model inputs are intentionally stat-only before child spawn. Do not
# open, hash, mmap, copy, or otherwise read their contents in this runner.
typeset model_device model_inode model_size model_mtime
typeset cghost_device cghost_inode cghost_size cghost_mtime
typeset assistant_device assistant_inode assistant_size assistant_mtime
model_device=$(/usr/bin/stat -f '%d' "$model")
model_inode=$(/usr/bin/stat -f '%i' "$model")
model_size=$(/usr/bin/stat -f '%z' "$model")
model_mtime=$(/usr/bin/stat -f '%m' "$model")
cghost_device=$(/usr/bin/stat -f '%d' "$cghost")
cghost_inode=$(/usr/bin/stat -f '%i' "$cghost")
cghost_size=$(/usr/bin/stat -f '%z' "$cghost")
cghost_mtime=$(/usr/bin/stat -f '%m' "$cghost")
assistant_device=$(/usr/bin/stat -f '%d' "$assistant")
assistant_inode=$(/usr/bin/stat -f '%i' "$assistant")
assistant_size=$(/usr/bin/stat -f '%z' "$assistant")
assistant_mtime=$(/usr/bin/stat -f '%m' "$assistant")

readonly disk_available_kib=$(/bin/df -Pk /System/Volumes/Data | /usr/bin/awk 'NR == 2 {print $4}')
[[ "$disk_available_kib" == <-> ]] || refuse "could not measure Data-volume disk headroom"
(( disk_available_kib >= MIN_DISK_AVAILABLE_KIB )) || \
  refuse "Data volume needs at least 20 GiB available before the benchmark"

typeset request_source_sha expected_ids_source_sha
sha256_file() {
  /usr/bin/python3 "$hasher" "$1"
}
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
  refuse "protected WebUI port $PROTECTED_PORT is already in use; it will not be touched" "$EX_PORT_BUSY"
port_listening "$PORT" && \
  refuse "benchmark port $PORT is already in use" "$EX_PORT_BUSY"

typeset lock_owned=0 cleanup_started=0 supervisor_pid="" child_pid="" child_pgid="" request_pid=""
/bin/mkdir -m 700 "$lock_dir" 2>/dev/null || \
  refuse "another Hot40 benchmark owns the runner lock: $lock_dir"
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
  if [[ "$request_pid" == <-> ]] && /bin/kill -0 "$request_pid" 2>/dev/null; then
    /bin/kill -TERM "$request_pid" 2>/dev/null
    wait "$request_pid" 2>/dev/null
  fi
  request_pid=""
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

readonly run_nonce=$(/usr/bin/uuidgen)
readonly boot_identity=$(/usr/sbin/sysctl -n kern.boottime)
readonly hashing_memory_before="$receipt_root/hashing-memory-before.json"
readonly hashing_memory_after="$receipt_root/hashing-memory-after.json"
/usr/bin/python3 "$host_sampler" --watchdog "$watchdog" --output "$hashing_memory_before"
/usr/bin/jq -e \
  --arg boot "$boot_identity" \
  --argjson minimum_headroom "$MIN_BASELINE_HEADROOM_BYTES" \
  --argjson maximum_wired "$MAX_HOST_WIRED_BYTES" '
  .schema_version == 1 and
  .telemetry_source == "run_load_only_watchdog.NativeTelemetry.sample_host" and
  .boot_identity == $boot and
  .host.pressure_level_raw == 1 and
  .host.reclaimable_headroom_bytes >= $minimum_headroom and
  .host.wired_bytes <= $maximum_wired
' "$hashing_memory_before" >/dev/null || \
  refuse "host memory was not clean enough to begin integrity hashing"

print -r -- "Hashing only the prebuilt binary, frozen fixtures, and tooling; large inputs remain stat-only..."
typeset binary_sha request_frozen_sha expected_ids_frozen_sha
typeset runner_sha analyzer_sha host_sampler_sha hasher_sha watchdog_sha
typeset hashing_memory_before_sha hashing_memory_after_sha
request_source_sha=$(sha256_file "$request_source")
expected_ids_source_sha=$(sha256_file "$expected_ids_source")
binary_sha=$(sha256_file "$binary")
request_frozen_sha=$(sha256_file "$request_frozen")
expected_ids_frozen_sha=$(sha256_file "$expected_ids_frozen")
runner_sha=$(sha256_file "$runner")
analyzer_sha=$(sha256_file "$analyzer")
host_sampler_sha=$(sha256_file "$host_sampler")
hasher_sha=$(sha256_file "$hasher")
watchdog_sha=$(sha256_file "$watchdog")
[[ "$request_source_sha" == "$REQUEST_SHA256" ]] || \
  refuse "frozen request fixture SHA-256 drifted"
[[ "$expected_ids_source_sha" == "$EXPECTED_IDS_SHA256" ]] || \
  refuse "frozen expected-token fixture SHA-256 drifted"
[[ "$request_frozen_sha" == "$REQUEST_SHA256" ]] || refuse "frozen request copy drifted"
[[ "$expected_ids_frozen_sha" == "$EXPECTED_IDS_SHA256" ]] || refuse "frozen token-ID copy drifted"

/usr/bin/python3 "$host_sampler" --watchdog "$watchdog" --output "$hashing_memory_after"
/usr/bin/jq -e -s \
  --arg boot "$boot_identity" \
  --argjson minimum_headroom "$MIN_BASELINE_HEADROOM_BYTES" \
  --argjson maximum_wired "$MAX_HOST_WIRED_BYTES" '
  .[0] as $before | .[1] as $after |
  all($before, $after;
    .schema_version == 1 and
    .telemetry_source == "run_load_only_watchdog.NativeTelemetry.sample_host" and
    .boot_identity == $boot and
    .host.pressure_level_raw == 1 and
    .host.reclaimable_headroom_bytes >= $minimum_headroom and
    .host.wired_bytes <= $maximum_wired
  ) and
  $before.host.swapins_pages == $after.host.swapins_pages and
  $before.host.swapouts_pages == $after.host.swapouts_pages
' "$hashing_memory_before" "$hashing_memory_after" >/dev/null || \
  refuse "small-artifact hashing changed swap or left an unsafe pre-spawn baseline"
hashing_memory_before_sha=$(sha256_file "$hashing_memory_before")
hashing_memory_after_sha=$(sha256_file "$hashing_memory_after")

typeset binary_size request_source_size request_frozen_size
typeset expected_source_size expected_frozen_size runner_size analyzer_size host_sampler_size
typeset hasher_size watchdog_size hashing_memory_before_size hashing_memory_after_size
binary_size=$(/usr/bin/stat -f '%z' "$binary")
request_source_size=$(/usr/bin/stat -f '%z' "$request_source")
request_frozen_size=$(/usr/bin/stat -f '%z' "$request_frozen")
expected_source_size=$(/usr/bin/stat -f '%z' "$expected_ids_source")
expected_frozen_size=$(/usr/bin/stat -f '%z' "$expected_ids_frozen")
runner_size=$(/usr/bin/stat -f '%z' "$runner")
analyzer_size=$(/usr/bin/stat -f '%z' "$analyzer")
host_sampler_size=$(/usr/bin/stat -f '%z' "$host_sampler")
hasher_size=$(/usr/bin/stat -f '%z' "$hasher")
watchdog_size=$(/usr/bin/stat -f '%z' "$watchdog")
hashing_memory_before_size=$(/usr/bin/stat -f '%z' "$hashing_memory_before")
hashing_memory_after_size=$(/usr/bin/stat -f '%z' "$hashing_memory_after")

/usr/bin/jq -n \
  --arg source_commit "$source_commit" \
  --arg binary_source_commit "$binary_source_commit" \
  --arg harness_commit "$harness_commit" \
  --argjson binary_source_defaulted "$binary_source_defaulted" \
  --arg nonce "$run_nonce" --arg boot "$boot_identity" \
  --argjson disk_available_kib "$disk_available_kib" \
  --arg binary "$binary" --arg binary_sha "$binary_sha" --argjson binary_size "$binary_size" \
  --arg model "$model" --arg model_sha "$model_preverified_sha" --argjson model_size "$model_size" \
  --argjson model_device "$model_device" --argjson model_inode "$model_inode" --argjson model_mtime "$model_mtime" \
  --arg cghost "$cghost" --arg cghost_sha "$cghost_preverified_sha" --argjson cghost_size "$cghost_size" \
  --argjson cghost_device "$cghost_device" --argjson cghost_inode "$cghost_inode" --argjson cghost_mtime "$cghost_mtime" \
  --arg assistant "$assistant" --arg assistant_sha "$assistant_preverified_sha" --argjson assistant_size "$assistant_size" \
  --argjson assistant_device "$assistant_device" --argjson assistant_inode "$assistant_inode" --argjson assistant_mtime "$assistant_mtime" \
  --arg provenance_reference "$provenance_reference" \
  --arg provenance_reference_sha "$provenance_reference_sha" \
  --arg provenance_limitation "$PROVENANCE_LIMITATION" \
  --arg request_source "$request_source" --arg request_source_sha "$request_source_sha" --argjson request_source_size "$request_source_size" \
  --arg request_frozen "$request_frozen" --arg request_frozen_sha "$request_frozen_sha" --argjson request_frozen_size "$request_frozen_size" \
  --arg expected_source "$expected_ids_source" --arg expected_source_sha "$expected_ids_source_sha" --argjson expected_source_size "$expected_source_size" \
  --arg expected_frozen "$expected_ids_frozen" --arg expected_frozen_sha "$expected_ids_frozen_sha" --argjson expected_frozen_size "$expected_frozen_size" \
  --arg runner "$runner" --arg runner_sha "$runner_sha" --argjson runner_size "$runner_size" \
  --arg analyzer "$analyzer" --arg analyzer_sha "$analyzer_sha" --argjson analyzer_size "$analyzer_size" \
  --arg host_sampler "$host_sampler" --arg host_sampler_sha "$host_sampler_sha" --argjson host_sampler_size "$host_sampler_size" \
  --arg hasher "$hasher" --arg hasher_sha "$hasher_sha" --argjson hasher_size "$hasher_size" \
  --arg hashing_before "$hashing_memory_before" --arg hashing_before_sha "$hashing_memory_before_sha" --argjson hashing_before_size "$hashing_memory_before_size" \
  --arg hashing_after "$hashing_memory_after" --arg hashing_after_sha "$hashing_memory_after_sha" --argjson hashing_after_size "$hashing_memory_after_size" \
  --slurpfile hashing_before_json "$hashing_memory_before" \
  --slurpfile hashing_after_json "$hashing_memory_after" \
  --arg watchdog "$watchdog" --arg watchdog_sha "$watchdog_sha" --argjson watchdog_size "$watchdog_size" '
  def artifact($path; $sha; $size): {
    path: $path,
    sha256: $sha,
    size_bytes: $size,
    verification: "sha256-this-run"
  };
  def large_input($path; $sha; $size; $device; $inode; $mtime): {
    path: $path,
    preverified_sha256: $sha,
    size_bytes: $size,
    stat: {
      device: $device,
      inode: $inode,
      mtime_seconds: $mtime
    },
    verification: "historical-sha256-plus-live-stat-no-run-content-read",
    content_read_before_spawn: false
  };
  {
    schema_version: 1,
    benchmark: "gemma4-uniform-hot40-experiment",
    source_commit: $source_commit,
    binary_source_commit: $binary_source_commit,
    harness_commit: $harness_commit,
    source_worktree_clean: true,
    harness_worktree_clean: true,
    binary_source_contract: {
      environment: "CAMELID_HOT40_BINARY_SOURCE_COMMIT",
      defaulted_to_harness_commit: $binary_source_defaulted,
      canonical_full_commit: true,
      ancestor_of_harness_commit: true,
      runtime_source_diff_empty: true,
      runtime_source_paths: ["src", "Cargo.toml", "Cargo.lock", "build.rs"]
    },
    nonce: $nonce,
    boot_identity: $boot,
    expected_tokens: 48,
    port: 8189,
    protected_port: 8181,
    protected_port_policy: "never-bound-connected-or-signaled",
    disk: {
      available_kib: $disk_available_kib,
      minimum_available_kib: 20971520,
      volume: "/System/Volumes/Data"
    },
    geometry: {
      layers: 30,
      logical_slots_per_layer: 128,
      hot_slots_per_layer: 40,
      anonymous_hot_capacity_slots: 1200,
      anonymous_hot_capacity_bytes: 4030464000,
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
      minimum_baseline_reclaimable_headroom_bytes: 8053063680,
      minimum_runtime_reclaimable_headroom_bytes: 2147483648,
      maximum_child_physical_footprint_bytes: 8053063680,
      maximum_host_wired_bytes: 8589934592,
      reject_swapin_growth: true,
      reject_swapout_growth: true,
      require_zero_current_swap: false
    },
    hashing_contract: {
      schema_version: 1,
      algorithm: "sha256",
      platform: "darwin",
      f_rdahead_command: 45,
      f_rdahead_value: 0,
      f_nocache_command: 48,
      f_nocache_value: 1,
      read_chunk_bytes: 4194304,
      post_hash_cooldown_seconds: 0,
      helper_artifact_label: "nocache_hasher",
      host_sampler_artifact_label: "host_sampler",
      telemetry_watchdog_artifact_label: "watchdog",
      telemetry_source: "run_load_only_watchdog.NativeTelemetry.sample_host",
      minimum_pre_hash_reclaimable_headroom_bytes: 8053063680,
      minimum_post_hash_reclaimable_headroom_bytes: 8053063680,
      maximum_host_wired_bytes: 8589934592,
      require_normal_pressure: true,
      reject_swapin_growth: true,
      reject_swapout_growth: true,
      large_inputs_content_hashed_this_run: false,
      large_inputs_content_read_before_spawn: false,
      large_input_binding: "historical-sha256-plus-live-stat",
      provenance_reference: $provenance_reference,
      provenance_reference_sha256: $provenance_reference_sha,
      provenance_limitation: $provenance_limitation,
      memory_before: $hashing_before_json[0],
      memory_after: $hashing_after_json[0]
    },
    large_inputs: {
      model: large_input($model; $model_sha; $model_size; $model_device; $model_inode; $model_mtime),
      cghost: large_input($cghost; $cghost_sha; $cghost_size; $cghost_device; $cghost_inode; $cghost_mtime),
      assistant: large_input($assistant; $assistant_sha; $assistant_size; $assistant_device; $assistant_inode; $assistant_mtime)
    },
    artifacts: {
      binary: artifact($binary; $binary_sha; $binary_size),
      request_source: artifact($request_source; $request_source_sha; $request_source_size),
      request_frozen: artifact($request_frozen; $request_frozen_sha; $request_frozen_size),
      expected_ids_source: artifact($expected_source; $expected_source_sha; $expected_source_size),
      expected_ids_frozen: artifact($expected_frozen; $expected_frozen_sha; $expected_frozen_size),
      runner: artifact($runner; $runner_sha; $runner_size),
      analyzer: artifact($analyzer; $analyzer_sha; $analyzer_size),
      host_sampler: artifact($host_sampler; $host_sampler_sha; $host_sampler_size),
      nocache_hasher: artifact($hasher; $hasher_sha; $hasher_size),
      hashing_memory_before: artifact($hashing_before; $hashing_before_sha; $hashing_before_size),
      hashing_memory_after: artifact($hashing_after; $hashing_after_sha; $hashing_after_size),
      watchdog: artifact($watchdog; $watchdog_sha; $watchdog_size)
    }
  }' > "$receipt_root/.intent.json.tmp"
/bin/mv "$receipt_root/.intent.json.tmp" "$receipt_root/intent.json"

{
  /bin/date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  print -r -- "source_commit=$source_commit"
  print -r -- "binary_source_commit=$binary_source_commit"
  print -r -- "harness_commit=$harness_commit"
  print -r -- "binary_source_defaulted_to_harness=$binary_source_defaulted"
  print -r -- "runtime_source_diff_empty=true paths=src,Cargo.toml,Cargo.lock,build.rs"
  print -r -- "source_worktree_clean=true"
  print -r -- "harness_worktree_clean=true"
  print -r -- "nonce=$run_nonce"
  print -r -- "binary_sha256=$binary_sha $binary"
  print -r -- "model_historical_sha256=$model_preverified_sha $model"
  print -r -- "model_live_stat=$model_device:$model_inode:$model_size:$model_mtime"
  print -r -- "cghost_historical_sha256=$cghost_preverified_sha $cghost"
  print -r -- "cghost_live_stat=$cghost_device:$cghost_inode:$cghost_size:$cghost_mtime"
  print -r -- "assistant_historical_sha256=$assistant_preverified_sha $assistant"
  print -r -- "assistant_live_stat=$assistant_device:$assistant_inode:$assistant_size:$assistant_mtime"
  print -r -- "large_inputs_content_read_before_spawn=false"
  print -r -- "large_input_provenance_reference=$provenance_reference"
  print -r -- "large_input_provenance_reference_sha256=$provenance_reference_sha"
  print -r -- "provenance_limitation=$PROVENANCE_LIMITATION"
  print -r -- "request_sha256=$request_frozen_sha $request_frozen"
  print -r -- "expected_ids_sha256=$expected_ids_frozen_sha $expected_ids_frozen"
  print -r -- "runner_sha256=$runner_sha $runner"
  print -r -- "analyzer_sha256=$analyzer_sha $analyzer"
  print -r -- "host_sampler_sha256=$host_sampler_sha $host_sampler"
  print -r -- "nocache_hasher_sha256=$hasher_sha $hasher"
  print -r -- "hashing_memory_before_sha256=$hashing_memory_before_sha $hashing_memory_before"
  print -r -- "hashing_memory_after_sha256=$hashing_memory_after_sha $hashing_memory_after"
  print -r -- "watchdog_sha256=$watchdog_sha $watchdog"
  /usr/bin/jq -c '{hashing_memory_before:.}' "$hashing_memory_before"
  /usr/bin/jq -c '{hashing_memory_after:.}' "$hashing_memory_after"
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
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT40_EXPERIMENT=1
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
    refuse "protected WebUI port $PROTECTED_PORT became active; benchmark child will be stopped" "$EX_PORT_BUSY"
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
    refuse "protected WebUI port $PROTECTED_PORT became active; benchmark child will be stopped" "$EX_PORT_BUSY"
  fi
  if health=$(/usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/health" 2>/dev/null); then
    if print -r -- "$health" | /usr/bin/jq -e \
      --arg binary_source_commit "$binary_source_commit" '
      (.build // "") as $build |
      .source_commit == $binary_source_commit and
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
  "http://127.0.0.1:$PORT/v1/chat/completions" > "$receipt_root/response.tmp" &
request_pid=$!
while /bin/kill -0 "$request_pid" 2>/dev/null; do
  if port_listening "$PROTECTED_PORT"; then
    refuse "protected WebUI port $PROTECTED_PORT became active during the request; benchmark child will be stopped" "$EX_PORT_BUSY"
  fi
  if ! /bin/kill -0 "$supervisor_pid" 2>/dev/null; then
    refuse "watchdog exited while the measured request was active"
  fi
  /bin/sleep 1
done
set +e
wait "$request_pid"
typeset request_status=$?
set -e
request_pid=""
(( request_status == 0 )) || refuse "measured request failed (curl status $request_status)"
port_listening "$PROTECTED_PORT" && \
  refuse "protected WebUI port $PROTECTED_PORT became active at request completion" "$EX_PORT_BUSY"
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
    /usr/bin/jq -n '{
      schema_version: 1,
      ports: {
        "8181": {clear: true, policy: "never-bound-connected-or-signaled"},
        "8189": {clear: true, policy: "benchmark-only"}
      }
    }' > "$receipt_root/.port-clear.json.tmp"
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

print -r -- "HOT40_EXPERIMENT_COMPLETE $receipt_root"
