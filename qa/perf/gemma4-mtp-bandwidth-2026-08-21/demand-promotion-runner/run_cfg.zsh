#!/bin/zsh
# Parameterized Gemma4 26B MoE serve benchmark.
#   run_cfg.zsh <label> <envfile>
# envfile holds one KEY=VALUE per line (no export, no quotes).
set -euo pipefail
zmodload zsh/datetime

readonly label=$1
readonly envfile=$2
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

/bin/rm -rf $out; /bin/mkdir -p $out
readonly server_log=$out/server.log
readonly response=$out/response.json

typeset -a envargs
envargs=(HOME=/Users/timtoole PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR=/tmp)
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  line=${line//__ASSISTANT__/$assistant}
  envargs+=("$line")
done < $envfile
print -rl -- "${envargs[@]}" > $out/env.txt
print -r -- "binary=$binary" >> $out/env.txt
print -r -- "cache_mib=$cache_mib" >> $out/env.txt
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
