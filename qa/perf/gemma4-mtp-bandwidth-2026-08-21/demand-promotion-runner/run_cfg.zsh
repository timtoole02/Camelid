#!/bin/zsh
# Parameterized Gemma4 26B MoE serve benchmark.
#   run_cfg.zsh <label> <envfile>
# envfile holds one KEY=VALUE per line (no export, no quotes). A literal
# multi-line value can be loaded from a file relative to envfile with
# KEY@FILE=path; this is needed for CAMELID_GEMMA4_SPEC_SEED_TEXT fixtures.
set -euo pipefail
zmodload zsh/datetime

(( $# == 2 )) || {
  print -u2 "usage: ${0:t} <label> <envfile>"
  exit 64
}
readonly label=$1
readonly envfile=$2
[[ -n $label && $label == [A-Za-z0-9]* && $label != *[^A-Za-z0-9._-]* ]] || {
  print -u2 "REFUSED: label must be one filename component beginning with an alphanumeric: $label"
  exit 75
}
readonly PORT=${CAMELID_BENCH_PORT:-8189}
typeset repo_candidate=${CAMELID_BENCH_SOURCE_ROOT:-${0:A:h}/../../../..}
readonly repo=${repo_candidate:A}
readonly runner=$repo/qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner
readonly watchdog=$repo/qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/run_load_only_watchdog.py
readonly manual_safety_sampler=${0:A:h}/capture_manual_safety.py
readonly binary=${CAMELID_BENCH_BINARY:-/Volumes/Untitled/cargo-targets/global/release/camelid}
readonly model=/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf
readonly cghost=/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost
readonly assistant=/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors
readonly request=${CAMELID_BENCH_REQUEST:-${0:A:h}/request-48-plain.json}
readonly expected_token_ids=$runner/expected-48-token-ids.json
readonly out=${CAMELID_BENCH_OUT:-${0:A:h}/runs}/$label
readonly cache_mib=${CAMELID_BENCH_CACHE_MIB:-0}
readonly no_watchdog=${CAMELID_BENCH_NO_WATCHDOG:-0}
readonly MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES=8053063679
readonly MAX_HOST_WIRED_BYTES=8589934591
readonly MAX_PRESSURE_LEVEL_RAW=1

[[ -x $binary ]] || { print -u2 "REFUSED: no binary $binary"; exit 75 }
[[ -f $request && ! -L $request ]] || {
  print -u2 "REFUSED: request fixture must be a regular non-symlink file: $request"; exit 75
}
[[ $no_watchdog == 0 || $no_watchdog == 1 ]] || {
  print -u2 "REFUSED: CAMELID_BENCH_NO_WATCHDOG must be exactly 0 or 1"; exit 75
}
if (( no_watchdog == 0 )); then
  [[ -f $watchdog && ! -L $watchdog ]] || {
    print -u2 "REFUSED: no watchdog $watchdog"; exit 75
  }
else
  [[ -f $manual_safety_sampler && ! -L $manual_safety_sampler ]] || {
    print -u2 "REFUSED: no manual safety sampler $manual_safety_sampler"; exit 75
  }
fi
[[ -f $envfile ]] || { print -u2 "REFUSED: no envfile $envfile"; exit 75 }
[[ -f $expected_token_ids && ! -L $expected_token_ids ]] || {
  print -u2 "REFUSED: no immutable expected-token fixture $expected_token_ids"; exit 75
}
/usr/bin/jq -e 'type == "array" and length == 48 and all(.[]; type == "number" and floor == .)' \
  "$expected_token_ids" >/dev/null || {
  print -u2 "REFUSED: expected-token fixture is not exactly 48 integer IDs"; exit 75
}
[[ $out == /* ]] || { print -u2 "REFUSED: output path must be absolute: $out"; exit 75 }
if /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "REFUSED: port $PORT busy"; exit 76
fi
typeset free_percent
free_percent=$(/usr/bin/memory_pressure -Q | /usr/bin/awk '/free percentage/ {gsub(/%/,"",$5); print $5}')
(( free_percent >= 55 )) || { print -u2 "REFUSED: free memory ${free_percent}% < 55%"; exit 75 }

typeset source_commit source_tree_status source_tree_clean binary_sha binary_size binary_version
typeset runner_sha request_sha manual_safety_sampler_sha supervision_mode
source_commit=$(/usr/bin/git -C "$repo" rev-parse --verify HEAD) || {
  print -u2 "REFUSED: could not resolve source commit in $repo"; exit 75
}
[[ ${#source_commit} == 40 && $source_commit != *[^0-9a-f]* ]] || {
  print -u2 "REFUSED: source commit is not a full lowercase SHA-1: $source_commit"; exit 75
}
source_tree_status=$(/usr/bin/git -C "$repo" status --porcelain --untracked-files=all) || {
  print -u2 "REFUSED: could not inspect source worktree $repo"; exit 75
}
if [[ -z $source_tree_status ]]; then
  source_tree_clean=1
else
  source_tree_clean=0
fi
(( no_watchdog == 0 || source_tree_clean == 1 )) || {
  print -u2 "REFUSED: source worktree is not clean: $repo"; exit 75
}
binary_sha=$(/usr/bin/shasum -a 256 "$binary" | /usr/bin/awk '{print $1}')
binary_size=$(/usr/bin/stat -f '%z' "$binary")
binary_version=$("$binary" --version) || {
  print -u2 "REFUSED: could not read binary version from $binary"; exit 75
}
[[ -n $binary_version && $binary_version != *$'\n'* ]] || {
  print -u2 "REFUSED: binary version must be one nonempty line"; exit 75
}
[[ ${#binary_sha} == 64 && $binary_sha != *[^0-9a-f]* && $binary_size == <1-> ]] || {
  print -u2 "REFUSED: binary provenance is malformed"; exit 75
}
runner_sha=$(/usr/bin/shasum -a 256 "${0:A}" | /usr/bin/awk '{print $1}')
request_sha=$(/usr/bin/shasum -a 256 "$request" | /usr/bin/awk '{print $1}')
manual_safety_sampler_sha=$(/usr/bin/shasum -a 256 "$manual_safety_sampler" | /usr/bin/awk '{print $1}')
[[ ${#request_sha} == 64 && $request_sha != *[^0-9a-f]* ]] || {
  print -u2 "REFUSED: request fixture SHA-256 is malformed"; exit 75
}
if (( no_watchdog == 1 )); then
  supervision_mode=manual-no-watchdog
else
  supervision_mode=strict-watchdog
fi

/bin/rm -rf -- "$out"
/bin/mkdir -p -- "$out"
readonly server_log=$out/server.log
readonly response=$out/response.json
readonly response_tmp=$out/response.tmp
readonly watchdog_log=$out/memory-watchdog.jsonl
readonly manual_safety=$out/manual-safety.json
readonly manual_pre=$out/.manual-safety-pre.tmp.json
readonly manual_ready=$out/.manual-safety-ready.tmp.json
readonly manual_post=$out/.manual-safety-post.tmp.json

typeset -a envargs manifest_lines
typeset -A seen_env_keys
envargs=()
manifest_lines=(manifest_format=base64-v1)
readonly file_value_sentinel='__CAMELID_ENV_FILE_VALUE_EOF_7C89D1A6__'

manifest_encode() {
  print -rn -- "$1" | /usr/bin/base64 | /usr/bin/tr -d '\n'
}

record_manifest_value() {
  typeset key=$1 value=$2 encoded
  encoded=$(manifest_encode "$value")
  manifest_lines+=("$key@BASE64=$encoded")
}

add_env_value() {
  typeset key=$1 value=$2
  [[ -n $key && $key == [A-Za-z_]* && $key != *[^A-Za-z0-9_]* ]] || {
    print -u2 "REFUSED: invalid environment identifier: $key"
    exit 75
  }
  [[ -z ${seen_env_keys[$key]:-} ]] || {
    print -u2 "REFUSED: duplicate environment identifier: $key"
    exit 75
  }
  seen_env_keys[$key]=1
  envargs+=("$key=$value")
  record_manifest_value "$key" "$value"
}

add_env_value HOME /Users/timtoole
add_env_value PATH /usr/bin:/bin:/usr/sbin:/sbin
add_env_value TMPDIR /tmp
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  if [[ $line == *'@FILE='* ]]; then
    typeset key=${line%%@FILE=*}
    typeset value_path=${line#*@FILE=}
    [[ -n $key && $key == [A-Za-z_]* && $key != *[^A-Za-z0-9_]* && -n $value_path ]] || {
      print -u2 "REFUSED: malformed KEY@FILE entry in $envfile: $line"; exit 75
    }
    [[ $value_path = /* ]] || value_path="${envfile:A:h}/$value_path"
    value_path=${value_path:A}
    [[ -f $value_path ]] || {
      print -u2 "REFUSED: no value file $value_path for $key"; exit 75
    }
    # Command substitution normally strips every trailing newline. Append a
    # sentinel first, then remove exactly that suffix so the environment value
    # remains byte-for-byte identical to the text file (NUL is not representable
    # in a Unix environment value).
    typeset file_value_with_sentinel file_value
    file_value_with_sentinel="$(/bin/cat -- "$value_path"; print -rn -- "$file_value_sentinel")"
    [[ $file_value_with_sentinel == *$file_value_sentinel ]] || {
      print -u2 "REFUSED: could not preserve value file $value_path for $key"; exit 75
    }
    file_value=${file_value_with_sentinel%$file_value_sentinel}
    add_env_value "$key" "$file_value"
    continue
  fi
  line=${line//__ASSISTANT__/$assistant}
  [[ $line == *=* ]] || {
    print -u2 "REFUSED: malformed KEY=VALUE entry in $envfile: $line"; exit 75
  }
  typeset key=${line%%=*}
  typeset value=${line#*=}
  add_env_value "$key" "$value"
done < "$envfile"
record_manifest_value binary "$binary"
record_manifest_value cache_mib "$cache_mib"
record_manifest_value supervision_mode "$supervision_mode"
record_manifest_value source_commit "$source_commit"
record_manifest_value source_tree_clean "$source_tree_clean"
record_manifest_value binary_sha256 "$binary_sha"
record_manifest_value binary_size "$binary_size"
record_manifest_value binary_version "$binary_version"
record_manifest_value expected_token_ids_sha256 \
  "$(/usr/bin/shasum -a 256 "$expected_token_ids" | /usr/bin/awk '{print $1}')"
record_manifest_value runner_sha256 "$runner_sha"
record_manifest_value request "$request"
record_manifest_value request_sha256 "$request_sha"
record_manifest_value manual_safety_sampler_sha256 "$manual_safety_sampler_sha"
print -rl -- "${manifest_lines[@]}" > "$out/env.txt"
/usr/bin/memory_pressure -Q > $out/pre-memory.txt
/usr/sbin/sysctl vm.swapusage >> $out/pre-memory.txt

typeset child_pid="" child_pgid="" supervisor_pid="" staging_dir="" staged_binary=""
typeset -i cleanup_started=0

group_alive() {
  [[ $child_pgid == <-> ]] && /bin/kill -0 -- "-$child_pgid" 2>/dev/null
}

cleanup() {
  set +e
  (( cleanup_started == 0 )) || return 0
  cleanup_started=1
  if [[ $supervisor_pid == <-> ]] && /bin/kill -0 $supervisor_pid 2>/dev/null; then
    /bin/kill -TERM $supervisor_pid 2>/dev/null
  fi
  if group_alive; then
    /bin/kill -TERM -- "-$child_pgid" 2>/dev/null
  fi
  for _ in {1..40}; do
    typeset supervisor_live=0 group_live=0
    [[ $supervisor_pid == <-> ]] && /bin/kill -0 $supervisor_pid 2>/dev/null && supervisor_live=1
    group_alive && group_live=1
    (( supervisor_live == 1 || group_live == 1 )) || break
    /bin/sleep 0.25
  done
  if group_alive; then
    /bin/kill -KILL -- "-$child_pgid" 2>/dev/null
  fi
  if [[ $supervisor_pid == <-> ]] && /bin/kill -0 $supervisor_pid 2>/dev/null; then
    /bin/kill -KILL $supervisor_pid 2>/dev/null
  fi
  if [[ $supervisor_pid == <-> ]]; then
    wait $supervisor_pid 2>/dev/null
  fi
  if [[ -n $staged_binary ]]; then
    /bin/rm -f -- "$staged_binary"
  fi
  if [[ -n $staging_dir && $staging_dir == /private/tmp/camelid-demand-bench.* ]]; then
    /bin/rmdir "$staging_dir" 2>/dev/null
  fi
}

abort_signal() {
  typeset abort_status=$1
  trap - EXIT INT TERM HUP
  cleanup
  exit $abort_status
}

trap cleanup EXIT
trap 'abort_signal 130' INT
trap 'abort_signal 143' TERM
trap 'abort_signal 129' HUP

if (( no_watchdog == 1 )); then
  /usr/bin/python3 "$manual_safety_sampler" --phase pre > "$manual_pre"
  /usr/bin/jq -e \
    --argjson maximum_wired "$MAX_HOST_WIRED_BYTES" \
    --argjson maximum_pressure "$MAX_PRESSURE_LEVEL_RAW" '
      .schema_version == 1 and
      .phase == "pre" and
      .process == null and
      .host.swapped_pages_current == 0 and
      .host.swapped_bytes_current == 0 and
      .host.pressure_level_raw <= $maximum_pressure and
      .host.wired_bytes <= $maximum_wired
    ' "$manual_pre" >/dev/null || {
    print -u2 "REFUSED: manual preflight point sample is outside the configured caps"
    exit 75
  }
fi

staging_dir=$(/usr/bin/mktemp -d /private/tmp/camelid-demand-bench.XXXXXX)
staged_binary=$staging_dir/camelid
/bin/cp -p -- "$binary" "$staged_binary"
[[ -x $staged_binary && -f $staged_binary && ! -L $staged_binary ]] || {
  print -u2 "REFUSED: could not stage an internal benchmark child"; exit 75
}
/usr/bin/cmp -s -- "$binary" "$staged_binary" || {
  print -u2 "REFUSED: staged benchmark child differs from $binary"; exit 75
}

typeset -F t_launch=$EPOCHREALTIME
if (( no_watchdog == 1 )); then
  # This tiny launcher creates an isolated process group and immediately execs
  # the server.  It performs no sampling, monitoring, or termination policy.
  /usr/bin/env -i "${envargs[@]}" \
    /usr/bin/python3 -c \
      'import os,sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
      "$staged_binary" serve --addr 127.0.0.1:$PORT --model $model --cghost $cghost \
      --expert-cache-mib $cache_mib --gpu on --no-open \
      > "$server_log" 2>&1 &
  supervisor_pid=$!
  child_pid=$supervisor_pid
  for _ in {1..200}; do
    typeset discovered_pgid=""
    discovered_pgid=$(/bin/ps -o pgid= -p "$child_pid" 2>/dev/null | /usr/bin/tr -d ' ') \
      || discovered_pgid=""
    if [[ $discovered_pgid == $child_pid ]]; then
      child_pgid=$discovered_pgid
      break
    fi
    if ! /bin/kill -0 $child_pid 2>/dev/null; then
      set +e
      wait $child_pid
      typeset child_status=$?
      set -e
      supervisor_pid=""
      print -u2 "REFUSED: no-watchdog child exited before process-group isolation (status $child_status)"
      exit 75
    fi
    /bin/sleep 0.01
  done
  [[ $child_pid == <-> && $child_pgid == $child_pid ]] || {
    print -u2 "REFUSED: no-watchdog child did not become an isolated process-group leader"
    exit 75
  }
else
  /usr/bin/env -i "${envargs[@]}" \
    /usr/bin/python3 "$watchdog" \
      --maximum-child-physical-footprint-bytes "$MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES" \
      --maximum-host-wired-bytes "$MAX_HOST_WIRED_BYTES" \
      --require-zero-current-swap \
      --reject-swapin-growth \
      --external-report-producer \
      --report "$response" \
      --watchdog-log "$watchdog_log" \
      --child-log "$server_log" \
      -- "$staged_binary" serve --addr 127.0.0.1:$PORT --model $model --cghost $cghost \
        --expert-cache-mib $cache_mib --gpu on --no-open &
  supervisor_pid=$!

  for _ in {1..200}; do
    if [[ -s $watchdog_log ]]; then
      typeset discovered_pid discovered_pgid
      discovered_pid=$(/usr/bin/jq -r 'select(.event == "child_started") | .pid' \
        "$watchdog_log" 2>/dev/null | /usr/bin/tail -1) || discovered_pid=""
      discovered_pgid=$(/usr/bin/jq -r 'select(.event == "child_started") | .process_group' \
        "$watchdog_log" 2>/dev/null | /usr/bin/tail -1) || discovered_pgid=""
      [[ $discovered_pid == <-> ]] && child_pid=$discovered_pid
      [[ $discovered_pgid == <-> ]] && child_pgid=$discovered_pgid
    fi
    if [[ $child_pid == <-> && $child_pgid == <-> ]]; then
      [[ $child_pid == $child_pgid ]] || {
        print -u2 "REFUSED: watchdog child PID/PGID identity drifted"; exit 75
      }
      break
    fi
    if ! /bin/kill -0 $supervisor_pid 2>/dev/null; then
      set +e
      wait $supervisor_pid
      typeset supervisor_status=$?
      set -e
      supervisor_pid=""
      print -u2 "REFUSED: watchdog exited before child_started (status $supervisor_status)"
      exit 75
    fi
    /bin/sleep 0.05
  done
  [[ $child_pid == <-> && $child_pgid == <-> ]] || {
    print -u2 "REFUSED: watchdog never recorded child_started"; exit 75
  }

  # Do not mistake a child that exits between spawn and the watchdog's first
  # process sample for a valid server start. This also makes the later nonzero
  # peak-footprint receipt check an observed invariant rather than a race.
  typeset first_process_sample=""
  for _ in {1..200}; do
    first_process_sample=$(/usr/bin/jq -sc --argjson expected_pid "$child_pid" '
      [ .[] | select(
          .event == "sample" and
          .pid == $expected_pid and
          .violations == [] and
          (.process.physical_footprint_bytes // 0) > 0
        ) ][0] // empty
    ' "$watchdog_log" 2>/dev/null) || first_process_sample=""
    [[ -n $first_process_sample ]] && break
    if ! /bin/kill -0 $supervisor_pid 2>/dev/null; then
      set +e
      wait $supervisor_pid
      typeset supervisor_status=$?
      set -e
      supervisor_pid=""
      print -u2 "REFUSED: watchdog exited before the first nonzero child sample (status $supervisor_status)"
      exit 75
    fi
    /bin/sleep 0.05
  done
  [[ -n $first_process_sample ]] || {
    print -u2 "REFUSED: watchdog never captured a nonzero child process sample"; exit 75
  }
fi

typeset health="" ready=0
for _ in {1..1200}; do
  if health=$(/usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/health" 2>/dev/null); then
    if print -r -- "$health" | /usr/bin/jq -e '.generation_ready == true' >/dev/null 2>&1; then ready=1; break; fi
  fi
  if ! /bin/kill -0 $supervisor_pid 2>/dev/null; then
    set +e
    wait $supervisor_pid
    typeset supervisor_status=$?
    set -e
    supervisor_pid=""
    print -u2 "REFUSED: benchmark supervisor exited before readiness (status $supervisor_status)"
    /usr/bin/tail -30 $server_log >&2
    exit 75
  fi
  /bin/kill -0 $child_pid 2>/dev/null || {
    print -u2 "REFUSED: server exited early"; /usr/bin/tail -30 $server_log >&2; exit 75
  }
  /bin/sleep 1
done
typeset -F t_ready=$EPOCHREALTIME
print -r -- "$health" > $out/health.json
(( ready == 1 )) || { print -u2 "REFUSED: never ready"; exit 75 }
if (( no_watchdog == 1 )); then
  /usr/bin/jq -e --arg source_commit "$source_commit" '
    .source_commit == $source_commit and
    (.build | type == "string" and (endswith("-dirty") | not))
  ' "$out/health.json" >/dev/null || {
    print -u2 "REFUSED: running binary does not bind the clean source commit $source_commit"
    exit 75
  }
fi
print -r -- "load_seconds=$(( t_ready - t_launch ))" > $out/timings.txt

/usr/bin/memory_pressure -Q > $out/ready-memory.txt
/usr/sbin/sysctl vm.swapusage >> $out/ready-memory.txt
if (( no_watchdog == 1 )); then
  /usr/bin/python3 "$manual_safety_sampler" \
    --phase ready --pid "$child_pid" --process-group "$child_pgid" \
    > "$manual_ready"
  /usr/bin/jq -e \
    --slurpfile pre "$manual_pre" \
    --argjson expected_pid "$child_pid" \
    --argjson expected_pgid "$child_pgid" \
    --argjson maximum_child "$MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES" \
    --argjson maximum_wired "$MAX_HOST_WIRED_BYTES" \
    --argjson maximum_pressure "$MAX_PRESSURE_LEVEL_RAW" '
      .schema_version == 1 and
      .phase == "ready" and
      .host.swapped_pages_current == 0 and
      .host.swapped_bytes_current == 0 and
      .host.swapins_pages == $pre[0].host.swapins_pages and
      .host.swapouts_pages == $pre[0].host.swapouts_pages and
      .host.pressure_level_raw <= $maximum_pressure and
      .host.wired_bytes <= $maximum_wired and
      .process.pid == $expected_pid and
      .process.process_group == $expected_pgid and
      (.process.member_pids | index($expected_pid)) != null and
      .process.physical_footprint_bytes > 0 and
      .process.physical_footprint_bytes <= $maximum_child
    ' "$manual_ready" >/dev/null || {
    print -u2 "REFUSED: manual readiness point sample is outside the configured caps"
    exit 75
  }
fi

/usr/bin/curl -sS --max-time 3600 -w '%{time_total}\n' -o $response_tmp \
  -H 'Content-Type: application/json' --data-binary "@$request" \
  "http://127.0.0.1:$PORT/v1/chat/completions" > $out/http-wall-seconds.txt
/bin/mv "$response_tmp" "$response"

# Exact token parity is part of the benchmark contract, not a post-hoc note.
# Annotate the response itself and reject any 47/48 (or malformed) candidate
# before it can be counted as a warmup or measurement.
typeset -i exact_match_expected=0
if /usr/bin/jq -e --slurpfile expected "$expected_token_ids" '
  (.camelid.generated_token_ids // null) as $actual |
  (($actual | type) == "array") and
  (($actual | length) == 48) and
  ($actual == $expected[0]) and
  (.usage.completion_tokens == 48)
' "$response" >/dev/null; then
  /usr/bin/jq \
    '.camelid.exact_match_expected = true | .camelid.exact_match_count = 48' \
    "$response" > "$response_tmp"
  /bin/mv "$response_tmp" "$response"
  exact_match_expected=1
  print -r -- "exact_match_expected=true" >> $out/timings.txt
else
  /usr/bin/jq \
    '.camelid.exact_match_expected = false | .camelid.exact_match_count = 0' \
    "$response" > "$response_tmp" 2>/dev/null || true
  [[ -s $response_tmp ]] && /bin/mv "$response_tmp" "$response"
  print -u2 "REFUSED: response failed exact 48/48 expected-token parity"
fi

/usr/bin/memory_pressure -Q > $out/post-memory.txt
/usr/sbin/sysctl vm.swapusage >> $out/post-memory.txt
if (( no_watchdog == 1 )); then
  /usr/bin/python3 "$manual_safety_sampler" \
    --phase post --pid "$child_pid" --process-group "$child_pgid" \
    > "$manual_post"
fi
if (( no_watchdog == 1 )); then
  /bin/kill -INT -- "-$child_pgid"
else
  /bin/kill -INT $child_pid
fi
set +e
wait $supervisor_pid
typeset supervisor_status=$?
set -e
supervisor_pid=""
if (( no_watchdog == 1 )); then
  typeset -i process_group_empty=0 report_exists=0 report_regular=0 report_symlink=0
  typeset -i report_size=0
  for _ in {1..40}; do
    if ! group_alive; then
      process_group_empty=1
      break
    fi
    /bin/sleep 0.05
  done
  [[ -e $response ]] && report_exists=1
  [[ -f $response ]] && report_regular=1
  [[ -L $response ]] && report_symlink=1
  if (( report_regular == 1 )); then
    report_size=$(/usr/bin/stat -f '%z' "$response")
  fi

  /usr/bin/jq -n \
    --slurpfile pre "$manual_pre" \
    --slurpfile ready "$manual_ready" \
    --slurpfile post "$manual_post" \
    --argjson maximum_child "$MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES" \
    --argjson maximum_wired "$MAX_HOST_WIRED_BYTES" \
    --argjson maximum_pressure "$MAX_PRESSURE_LEVEL_RAW" \
    --argjson child_pid "$child_pid" \
    --argjson child_pgid "$child_pgid" \
    --argjson child_returncode "$supervisor_status" \
    --argjson process_group_empty "$process_group_empty" \
    --argjson report_exists "$report_exists" \
    --argjson report_regular "$report_regular" \
    --argjson report_symlink "$report_symlink" \
    --argjson report_size "$report_size" '
      ($pre[0]) as $pre_sample |
      ($ready[0]) as $ready_sample |
      ($post[0]) as $post_sample |
      ([$pre_sample, $ready_sample, $post_sample]) as $samples |
      (all($samples[]; .host.swapped_pages_current == 0)) as $current_swap_zero |
      (all($samples[]; .host.swapins_pages == $pre_sample.host.swapins_pages))
        as $swapins_unchanged |
      (all($samples[]; .host.swapouts_pages == $pre_sample.host.swapouts_pages))
        as $swapouts_unchanged |
      (all($samples[]; .host.pressure_level_raw <= $maximum_pressure))
        as $pressure_within_cap |
      (all($samples[]; .host.wired_bytes <= $maximum_wired))
        as $host_wired_within_cap |
      (all([$ready_sample, $post_sample][];
        .process.physical_footprint_bytes > 0 and
        .process.physical_footprint_bytes <= $maximum_child
      )) as $child_physical_footprint_within_cap |
      ($child_returncode == 0) as $child_returned_zero |
      ($process_group_empty == 1) as $process_group_is_empty |
      (
        $report_exists == 1 and
        $report_regular == 1 and
        $report_symlink == 0 and
        $report_size > 0
      ) as $report_valid |
      {
        schema_version: 1,
        mode: "manual-no-watchdog",
        qualifying: false,
        point_samples_only: true,
        configured_caps: {
          maximum_child_physical_footprint_bytes: $maximum_child,
          maximum_host_wired_bytes: $maximum_wired,
          maximum_pressure_level_raw: $maximum_pressure,
          require_zero_current_swap: true,
          reject_swapin_growth: true,
          reject_swapout_growth: true
        },
        samples: {
          pre: $pre_sample,
          ready: $ready_sample,
          post: $post_sample
        },
        sampled_peaks: {
          child_rss_bytes: ([$ready_sample, $post_sample] | map(.process.rss_bytes) | max),
          child_physical_footprint_bytes:
            ([$ready_sample, $post_sample] | map(.process.physical_footprint_bytes) | max),
          host_wired_bytes: ($samples | map(.host.wired_bytes) | max)
        },
        child: {
          pid: $child_pid,
          process_group: $child_pgid,
          returncode: $child_returncode,
          process_group_empty: $process_group_is_empty
        },
        report: {
          exists: ($report_exists == 1),
          is_regular_file: ($report_regular == 1),
          is_symlink: ($report_symlink == 1),
          size_bytes: $report_size
        },
        checks: {
          current_swap_zero: $current_swap_zero,
          swapins_unchanged: $swapins_unchanged,
          swapouts_unchanged: $swapouts_unchanged,
          pressure_within_cap: $pressure_within_cap,
          host_wired_within_cap: $host_wired_within_cap,
          child_physical_footprint_within_cap:
            $child_physical_footprint_within_cap,
          child_returned_zero: $child_returned_zero,
          process_group_empty: $process_group_is_empty,
          report_valid: $report_valid,
          all_passed: (
            $current_swap_zero and
            $swapins_unchanged and
            $swapouts_unchanged and
            $pressure_within_cap and
            $host_wired_within_cap and
            $child_physical_footprint_within_cap and
            $child_returned_zero and
            $process_group_is_empty and
            $report_valid
          )
        }
      }
    ' > "$manual_safety"
  /bin/rm -f -- "$manual_pre" "$manual_ready" "$manual_post"

  /usr/bin/jq -e '
    .schema_version == 1 and
    .mode == "manual-no-watchdog" and
    .qualifying == false and
    .point_samples_only == true and
    .checks.all_passed == true
  ' "$manual_safety" >/dev/null || {
    print -u2 "REFUSED: manual point samples or final cleanup violated the no-watchdog contract"
    exit 75
  }
else
  (( supervisor_status == 0 )) || {
    print -u2 "REFUSED: watchdog supervisor returned $supervisor_status"; exit 75
  }

  if ! /usr/bin/jq -s -e \
    --argjson expected_pid "$child_pid" \
    --argjson expected_pgid "$child_pgid" \
    --argjson maximum_child "$MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES" \
    --argjson maximum_wired "$MAX_HOST_WIRED_BYTES" '
      def events($name): [.[] | select(.event == $name)];
      (events("clean_parent_baseline")) as $baselines |
      (events("child_started")) as $starts |
      (events("post_exit_sample")) as $posts |
      (events("final")) as $finals |
      all(.[]; .schema_version == 3) and
      (.[-1].event == "final") and
      (($baselines | length) == 1) and
      (($starts | length) == 1) and
      (($posts | length) == 1) and
      (($finals | length) == 1) and
      ($baselines[0].violations == []) and
      ($baselines[0].host.swapped_pages_current == 0) and
      ($starts[0].pid == $expected_pid) and
      ($starts[0].process_group == $expected_pgid) and
      ($starts[0].pid == $starts[0].process_group) and
      ($starts[0].maximum_child_physical_footprint_bytes == $maximum_child) and
      ($starts[0].maximum_host_wired_bytes == $maximum_wired) and
      ($starts[0].require_zero_current_swap == true) and
      ($starts[0].reject_swapin_growth == true) and
      ($starts[0].report_producer == "external") and
      ($posts[0].violations == []) and
      ($posts[0].host.swapped_pages_current == 0) and
      ($posts[0].host.swapins_pages == $baselines[0].host.swapins_pages) and
      ($posts[0].host.swapouts_pages == $baselines[0].host.swapouts_pages) and
      ($finals[0].pid == $expected_pid) and
      ($finals[0].watchdog_aborted == false) and
      ($finals[0].abort_reasons == []) and
      ($finals[0].child_returncode == 0) and
      ($finals[0].process_group == $expected_pgid) and
      ($finals[0].process_group_empty == true) and
      ($finals[0].process_accounting_scope == "isolated_process_group_aggregate") and
      ($finals[0].peak_child_physical_footprint_bytes > 0) and
      ($finals[0].peak_child_physical_footprint_bytes <= $maximum_child) and
      ($finals[0].peak_host_wired_bytes > 0) and
      ($finals[0].peak_host_wired_bytes <= $maximum_wired) and
      ($finals[0].baseline_swapins_pages == $baselines[0].host.swapins_pages) and
      ($finals[0].baseline_swapouts_pages == $baselines[0].host.swapouts_pages) and
      ($finals[0].report_exists == true) and
      ($finals[0].report_is_regular_file == true) and
      ($finals[0].report_is_symlink == false) and
      ($finals[0].report_size_bytes > 0)
    ' "$watchdog_log" >/dev/null; then
    print -u2 "REFUSED: watchdog receipt is missing, malformed, or outside the memory contract"
    exit 75
  fi
fi

(( exact_match_expected == 1 )) || {
  print -u2 "REFUSED: completed run did not have exact_match_expected=true"
  exit 75
}

child_pid=""
child_pgid=""
print -r -- "RUN=$out"
