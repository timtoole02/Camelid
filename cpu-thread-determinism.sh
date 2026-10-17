#!/usr/bin/env bash
# =============================================================================
# cpu-thread-determinism.sh: is greedy decode bit-identical across thread counts?
# =============================================================================
# Camelid documents that GPU f32 reduction order is device-specific, and scopes
# its parity bundles to an exact GPU/driver/CUDA combination because of it.
# The CPU path is the parity reference and the default lane, and it is
# parallelised with Rayon. `verify` and `serve` both expose `--threads` to
# override the worker count, which implies thread count is a variable someone
# already considered.
#
# The analogous question for the CPU lane does not appear to be recorded
# anywhere: does the same model, same prompt, same build produce the same tokens
# when the Rayon worker count changes? A parallel reduction that sums partial
# results in completion order rather than a fixed order can produce different
# last-bit results at different widths, and a near-tie between two candidate
# tokens can then flip.
#
# This script answers that on one host by sweeping thread counts and comparing
# greedy output byte-for-byte against the single-threaded run.
#
# WHY IT MATTERS
#   If output is stable across thread counts, that is a useful negative result
#   for a project that documents non-claims, and it strengthens the CPU lane's
#   standing as the reference.
#   If it is not stable, then any CPU parity claim needs a thread count recorded
#   alongside it, exactly as GPU claims record a driver version.
#
# HOSTS WORTH RUNNING THIS ON
#   Thread count is the variable, so a machine with more cores tests more of the
#   space. 16, 32, and 44 logical threads give three quite different sweeps.
#
# SCOPE
#   One model, one prompt, one build, one host per run. Greedy decode only
#   (temperature 0), because anything stochastic makes the comparison
#   meaningless. This tests determinism across thread counts. It is not a parity
#   test against llama.cpp and makes no parity claim.
#
# Usage:
#   ./cpu-thread-determinism.sh [--model PATH] [--threads "1 2 4 8 16"]
#                               [--tokens N] [--out DIR]
# =============================================================================

set -uo pipefail

MODEL=""
THREAD_LIST=""
TOKENS=64
OUT="./evidence-out"
BINARY="./target/release/camelid"
PORT=18435

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --threads) THREAD_LIST="$2"; shift 2 ;;
        --tokens)  TOKENS="$2"; shift 2 ;;
        --out)     OUT="$2"; shift 2 ;;
        --binary)  BINARY="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -x "$BINARY" ]] || { echo "camelid not executable: $BINARY" >&2; exit 2; }
[[ -n "$MODEL" ]] || MODEL="$(ls models/*.gguf 2>/dev/null | head -1)"
[[ -f "$MODEL" ]] || { echo "no model; pass --model" >&2; exit 2; }

ncores() {
    case "$(uname -s)" in
        Linux)  nproc ;;
        Darwin) sysctl -n hw.ncpu ;;
        *)      echo 4 ;;
    esac
}
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# Camelid exposes a real readiness contract at /v1/health: `generation_ready`
# says the engine can generate, `active_model_id` names what is loaded, and
# `execution_plan`/`backend` say which lane serves it. An earlier version of this
# harness polled /v1/models and treated any HTTP 200 as ready, which races
# because serve answers 200 with an empty data array while weights load. Use the
# endpoint the engine actually provides; it is public (no auth) by design so a
# load balancer can probe it.
wait_for_model() {   # $1 = port, $2 = max seconds; echoes the model id on success
    local port="$1" deadline=$(( $2 / 2 )) i id
    for ((i=0; i<deadline; i++)); do
        id="$(curl -s --max-time 3 "http://127.0.0.1:${port}/v1/health" 2>/dev/null \
            | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d.get('active_model_id') or '' if d.get('generation_ready') else '')
except Exception: print('')
" 2>/dev/null)"
        [[ -n "$id" ]] && { printf '%s' "$id"; return 0; }
        # Fall back to /v1/models for older builds that predate the health fields.
        id="$(curl -s --max-time 3 "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
            | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')
except Exception: print('')
" 2>/dev/null)"
        [[ -n "$id" ]] && { printf '%s' "$id"; return 0; }
        sleep 2
    done
    return 1
}

# Default sweep: powers of two up to the host's core count, plus the exact core
# count itself, since that is what an unconfigured run would use.
if [[ -z "$THREAD_LIST" ]]; then
    N="$(ncores)"; THREAD_LIST="1 2 4"
    for t in 8 16 32 44; do (( t <= N )) && THREAD_LIST="$THREAD_LIST $t"; done
    case " $THREAD_LIST " in *" $N "*) ;; *) THREAD_LIST="$THREAD_LIST $N" ;; esac
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIR="${OUT}/cpu-thread-determinism-${STAMP}-head-${SHORT}"
mkdir -p "$DIR/raw"

echo "model   : $(basename "$MODEL")"
echo "threads : ${THREAD_LIST}"
echo "tokens  : ${TOKENS} (greedy, temperature 0)"
echo "out     : $DIR"
echo

PROMPT="List the first eight prime numbers, separated by commas, then explain in one sentence why one is not among them."

SERVE_PID=""
cleanup() { [[ -n "$SERVE_PID" ]] && { kill -TERM "$SERVE_PID" 2>/dev/null; sleep 2; kill -KILL "$SERVE_PID" 2>/dev/null; }; }
trap cleanup EXIT INT TERM

RESULTS=""
BASELINE_HASH=""
BASELINE_T=""

for T in $THREAD_LIST; do
    printf '  threads=%-4s ' "$T"

    # CAMELID_LAZY_Q8_0_LINEAR keeps the CPU weight materialization inside the
    # documented safety limit for larger rows on smaller hosts. It selects the
    # slower file-backed Q8 path, which is a documented option, not a bypass.
    CAMELID_LAZY_Q8_0_LINEAR=1 "$BINARY" serve \
        --addr "127.0.0.1:${PORT}" --model "$MODEL" --threads "$T" \
        > "$DIR/raw/serve-t${T}.out" 2> "$DIR/raw/serve-t${T}.err" &
    SERVE_PID=$!

    MODEL_ID="$(wait_for_model "$PORT" 180)"
    if [[ -z "$MODEL_ID" ]]; then
        echo "model never became ready"
        RESULTS="${RESULTS}$(printf '{"threads":%s,"status":"model_not_ready"},' "$T")"
        cleanup; SERVE_PID=""; continue
    fi

    # temperature 0 and a fixed prompt: any difference is the engine, not sampling.
    python3 - "$DIR/raw/req-t${T}.json" "$MODEL_ID" "$PROMPT" "$TOKENS" <<'PY'
import json, sys
path, mid, prompt, toks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
json.dump({"model": mid,
           "messages": [{"role": "user", "content": prompt}],
           "temperature": 0, "top_p": 1, "seed": 0,
           "max_tokens": toks}, open(path, "w"))
PY

    curl -s --max-time 900 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$DIR/raw/req-t${T}.json" \
        > "$DIR/raw/resp-t${T}.json" 2>&1

    CONTENT="$(python3 -c "
import json,sys
try:
    d=json.load(open('$DIR/raw/resp-t${T}.json'))
    print(d['choices'][0]['message'].get('content') or '')
except Exception:
    print('__ERROR__')
" 2>/dev/null)"

    cleanup; SERVE_PID=""

    if [[ "$CONTENT" == "__ERROR__" || -z "$CONTENT" ]]; then
        echo "no usable completion"
        RESULTS="${RESULTS}$(printf '{"threads":%s,"status":"no_completion"},' "$T")"
        continue
    fi

    printf '%s' "$CONTENT" > "$DIR/raw/out-t${T}.txt"
    H="$(sha256_of "$DIR/raw/out-t${T}.txt")"

    if [[ -z "$BASELINE_HASH" ]]; then
        BASELINE_HASH="$H"; BASELINE_T="$T"
        echo "baseline  ${H:0:16}"
        VERDICT="baseline"
    elif [[ "$H" == "$BASELINE_HASH" ]]; then
        echo "IDENTICAL ${H:0:16}"
        VERDICT="identical_to_baseline"
    else
        echo "DIVERGED  ${H:0:16}"
        VERDICT="diverged_from_baseline"
    fi
    RESULTS="${RESULTS}$(printf '{"threads":%s,"status":"ok","sha256":"%s","verdict":"%s"},' \
        "$T" "$H" "$VERDICT")"
done

DIVERGED=$(printf '%s' "$RESULTS" | grep -o 'diverged_from_baseline' | wc -l | tr -d ' ')
# Count runs that actually produced output. Without this, a sweep in which every
# run failed reports diverged_count 0, which reads as "determinism held" when in
# fact nothing was compared. An all-failed sweep is INCONCLUSIVE, not a pass.
COMPLETED=$(printf '%s' "$RESULTS" | grep -o '"status":"ok"' | wc -l | tr -d ' ')
if (( COMPLETED < 2 )); then VERDICT_OVERALL="inconclusive_insufficient_runs"
elif (( DIVERGED == 0 )); then VERDICT_OVERALL="no_divergence"
else VERDICT_OVERALL="divergence_found"; fi

cat > "$DIR/receipt.json" <<JSON
{
  "schema": "camelid.cpu-thread-determinism.v1",
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "question": "Does greedy CPU decode produce byte-identical output as the Rayon worker count changes?",
  "camelid_version": "$($BINARY --version 2>/dev/null | head -1)",
  "camelid_git_head_short": "${SHORT}",
  "host": {
    "os": "$(uname -s) $(uname -r)",
    "arch": "$(uname -m)",
    "logical_cores": $(ncores)
  },
  "model": { "file": "$(basename "$MODEL")", "sha256": "$(sha256_of "$MODEL")" },
  "request": { "temperature": 0, "top_p": 1, "seed": 0, "max_tokens": ${TOKENS} },
  "env": { "CAMELID_LAZY_Q8_0_LINEAR": "1" },
  "baseline_threads": "${BASELINE_T}",
  "diverged_count": ${DIVERGED},
  "completed_runs": ${COMPLETED},
  "verdict": "${VERDICT_OVERALL}",
  "verdict_note": "A diverged_count of 0 is only meaningful when completed_runs is at least 2. An all-failed sweep is inconclusive, never a determinism pass.",
  "runs": [${RESULTS%,}],
  "scope": "One model, one prompt, one build, one host. Greedy decode only. Tests determinism across Rayon worker counts. This is NOT a parity test against llama.cpp and makes no parity claim. A divergence here would mean CPU parity claims need a thread count recorded alongside them; no divergence is a useful negative result."
}
JSON

( cd "$DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
  | while IFS= read -r -d '' f; do printf '%s  %s\n' "$(sha256_of "$f")" "${f#./}"; done \
  > SHA256SUMS )

echo
if (( COMPLETED < 2 )); then
    echo "RESULT: INCONCLUSIVE - only ${COMPLETED} run(s) produced output"
    echo "        nothing was compared; this is not a determinism result"
elif (( DIVERGED == 0 )); then
    echo "RESULT: no divergence across ${COMPLETED} completed run(s) of ${THREAD_LIST}"
else
    echo "RESULT: ${DIVERGED} thread count(s) diverged from the ${BASELINE_T}-thread baseline"
    echo "        diff the out-t*.txt files in $DIR/raw to see where"
fi
echo "receipt: $DIR/receipt.json"
