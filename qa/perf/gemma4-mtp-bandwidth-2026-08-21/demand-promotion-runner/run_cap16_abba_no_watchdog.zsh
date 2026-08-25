#!/bin/zsh
# Run one clean Mini2 cap8/cap16/cap16/cap8 campaign under cam-lock.
set -euo pipefail

readonly lock=${CAMELID_CAM_LOCK:-$HOME/bin/cam-lock.sh}
readonly script=${0:A}
readonly here=${script:h}

if [[ ${1:-} != --inside-cam-lock ]]; then
  (( $# == 1 )) || {
    print -u2 "usage: ${0:t} <label-prefix>"
    exit 64
  }
  readonly requested_prefix=$1
  [[ -n $requested_prefix && $requested_prefix == [A-Za-z0-9]* \
    && $requested_prefix != *[^A-Za-z0-9._-]* ]] || {
    print -u2 "REFUSED: label prefix must begin alphanumeric and contain only [A-Za-z0-9._-]"
    exit 75
  }
  [[ -x $lock ]] || { print -u2 "REFUSED: no serialization lock $lock"; exit 75 }
  export CAM_SESSION_PID=$$
  exec "$lock" "$script" --inside-cam-lock "$requested_prefix"
fi

(( $# == 2 )) || { print -u2 "REFUSED: malformed internal invocation"; exit 75 }
shift
readonly prefix=$1
typeset lock_holder
lock_holder=$(/bin/cat /tmp/camelid-build.lock/pid 2>/dev/null) || lock_holder=""
[[ $lock_holder == $PPID ]] || {
  print -u2 "REFUSED: ABBA campaign is not the direct child of the active cam-lock holder"
  exit 75
}

readonly runner=$here/run_cfg.zsh
readonly analyzer=$here/analyze_live_sequential_cap16.py
readonly summarizer=$here/summarize_cap16_abba.py
readonly canonical_request=$here/request-48-plain.json
readonly control=$here/env/H49-live-hidden-sequential-fast-predict-dual-reader-kv192-control
readonly candidate=$here/env/H69-live-hidden-sequential-fast-predict-dual-reader-kv192-cap16
typeset output_candidate=${CAMELID_BENCH_OUT:-$here/runs}
readonly output_root=${output_candidate:A}
readonly summary=$output_root/$prefix-cap16-abba-summary.json

[[ -x $runner && -f $analyzer && -f $summarizer ]] || {
  print -u2 "REFUSED: ABBA runner or analyzer is absent"
  exit 75
}
[[ -f $canonical_request && ! -L $canonical_request ]] || {
  print -u2 "REFUSED: canonical 48-token request fixture is absent or a symlink"
  exit 75
}
[[ -f $control && -f $candidate ]] || {
  print -u2 "REFUSED: cap8 control or cap16 candidate profile is absent"
  exit 75
}

typeset -a labels profiles run_dirs
labels=(
  "$prefix-cap8-a1"
  "$prefix-cap16-b1"
  "$prefix-cap16-b2"
  "$prefix-cap8-a2"
)
profiles=("$control" "$candidate" "$candidate" "$control")
run_dirs=()
typeset run_dir
for label in $labels; do
  run_dir=$output_root/$label
  [[ ! -e $run_dir && ! -L $run_dir ]] || {
    print -u2 "REFUSED: run destination already exists: $run_dir"
    exit 75
  }
  run_dirs+=("$run_dir")
done
[[ ! -e $summary && ! -L $summary ]] || {
  print -u2 "REFUSED: summary destination already exists: $summary"
  exit 75
}

export CAMELID_BENCH_NO_WATCHDOG=1
export CAMELID_BENCH_REQUEST=$canonical_request
export CAMELID_BENCH_OUT=$output_root
export CARGO_BUILD_JOBS=2
for index in {1..4}; do
  print -u2 "ABBA $index/4: ${labels[$index]}"
  "$runner" "${labels[$index]}" "${profiles[$index]}"
  /usr/bin/python3 "$analyzer" "${run_dirs[$index]}" \
    --output "${run_dirs[$index]}/h69-analysis.json" >/dev/null
done

/usr/bin/python3 "$summarizer" "${run_dirs[@]}" > "$summary"
/bin/cat "$summary"
print -u2 "ABBA summary: $summary"
