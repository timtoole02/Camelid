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
readonly repo=/Users/timtoole/Documents/Camelid
readonly runner=$repo/qa/perf/gemma4-mtp-bandwidth-2026-08-21/hybrid-hot48-runner
readonly binary=${CAMELID_BENCH_BINARY:-/Volumes/Untitled/cargo-targets/global/release/camelid}
readonly model=/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf
readonly cghost=/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost
readonly assistant=/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors
readonly request=${CAMELID_BENCH_REQUEST:-${0:A:h}/request-48-plain.json}
readonly out=${CAMELID_BENCH_OUT:-${0:A:h}/runs}/$label
readonly cache_mib=${CAMELID_BENCH_CACHE_MIB:-0}

[[ -x $binary ]] || { print -u2 "REFUSED: no binary $binary"; exit 75 }
[[ -f $envfile ]] || { print -u2 "REFUSED: no envfile $envfile"; exit 75 }
if /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "REFUSED: port $PORT busy"; exit 76
fi
typeset free_percent
free_percent=$(/usr/bin/memory_pressure -Q | /usr/bin/awk '/free percentage/ {gsub(/%/,"",$5); print $5}')
(( free_percent >= 55 )) || { print -u2 "REFUSED: free memory ${free_percent}% < 55%"; exit 75 }

/bin/rm -rf -- "$out"
/bin/mkdir -p -- "$out"
readonly server_log=$out/server.log
readonly response=$out/response.json

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
print -rl -- "${manifest_lines[@]}" > "$out/env.txt"
/usr/bin/memory_pressure -Q > $out/pre-memory.txt
/usr/sbin/sysctl vm.swapusage >> $out/pre-memory.txt

typeset child_pid=""
cleanup() {
  if [[ -n $child_pid ]] && /bin/kill -0 $child_pid 2>/dev/null; then
    /bin/kill -INT $child_pid 2>/dev/null || true
    for _ in {1..40}; do /bin/kill -0 $child_pid 2>/dev/null || break; /bin/sleep 1; done
    /bin/kill -KILL $child_pid 2>/dev/null || true
    wait $child_pid 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

typeset -F t_launch=$EPOCHREALTIME
/usr/bin/env -i "${envargs[@]}" \
  $binary serve --addr 127.0.0.1:$PORT --model $model --cghost $cghost \
    --expert-cache-mib $cache_mib --gpu on --no-open > $server_log 2>&1 &
child_pid=$!

typeset health="" ready=0
for _ in {1..1200}; do
  if health=$(/usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/health" 2>/dev/null); then
    if print -r -- "$health" | /usr/bin/jq -e '.generation_ready == true' >/dev/null 2>&1; then ready=1; break; fi
  fi
  /bin/kill -0 $child_pid 2>/dev/null || { print -u2 "REFUSED: server exited early"; tail -30 $server_log >&2; exit 75 }
  /bin/sleep 1
done
typeset -F t_ready=$EPOCHREALTIME
print -r -- "$health" > $out/health.json
(( ready == 1 )) || { print -u2 "REFUSED: never ready"; exit 75 }
print -r -- "load_seconds=$(( t_ready - t_launch ))" > $out/timings.txt

/usr/bin/memory_pressure -Q > $out/ready-memory.txt
/usr/sbin/sysctl vm.swapusage >> $out/ready-memory.txt

/usr/bin/curl -sS --max-time 3600 -w '%{time_total}\n' -o $response \
  -H 'Content-Type: application/json' --data-binary "@$request" \
  "http://127.0.0.1:$PORT/v1/chat/completions" > $out/http-wall-seconds.txt

/usr/bin/memory_pressure -Q > $out/post-memory.txt
/usr/sbin/sysctl vm.swapusage >> $out/post-memory.txt
/bin/kill -INT $child_pid; wait $child_pid 2>/dev/null || true; child_pid=""
print -r -- "RUN=$out"
