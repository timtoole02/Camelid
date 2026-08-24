#!/usr/bin/env bash
# =============================================================================
# r007-retest.sh: re-test pxx receipt R-007's findings against current Camelid
# =============================================================================
# pxx receipt R-007 (docs/RECEIPTS.md in cdnwetzel/pxx) recorded three findings
# against Camelid v0.4.4 on Apple Silicon, and states:
#
#   "Findings 1 and 3 are reported upstream; this entry updates when the lanes
#    change."
#
# Camelid has since moved to v0.6.1-191. This script re-runs the probes so the
# receipt can be updated with evidence rather than assumption, and extends it to
# a platform R-007 never covered (Linux x86_64 + CUDA).
#
# THE THREE FINDINGS
#
#   F1  Default serve panicked on macOS ARM64 for qwen3_4b_q4_k_m
#       (metal_resident.rs unwrap on None), deterministic, on both 8 and 16 GB.
#       On Linux the analogous question is whether the CUDA-resident path
#       refuses cleanly with a typed error or panics the worker.
#
#   F2  The deterministic CPU lane completed a full pxx round trip
#       (~1.2 tok/s on 8 GB, ~8.7 tok/s on M4). Re-measure for a third point.
#
#   F3  The OpenAI /v1/chat/completions `tools` surface accepted the parameter
#       but never executed: `tool_calls` was always null and the model's
#       Qwen-native <tool_call> block was returned verbatim as `content`.
#       This is the finding that gates agent loops, and the one worth
#       re-testing first.
#
# F3 is deliberately isolated with a direct curl carrying an explicit tools
# array, exactly as R-007 did, so the comparison is like-for-like. It does NOT
# use pxx, because introducing pxx would confound a serve-layer question with a
# client-layer one.
#
# SCOPE. This tests the OpenAI-compatible surface of one build on one host with
# one model. It says nothing about Camelid's own agent lane, which has its own
# receipt and passes independently (`camelid agent-eval`).
#
# Usage:
#   ./r007-retest.sh [--model PATH] [--port N] [--out DIR]
# =============================================================================

set -uo pipefail

MODEL=""
PORT=18434
OUT="./evidence-out"
BINARY="./target/release/camelid"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        --out)    OUT="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -x "$BINARY" ]] || { echo "camelid not executable: $BINARY" >&2; exit 2; }
if [[ -z "$MODEL" ]]; then
    MODEL="$(ls models/*.gguf 2>/dev/null | head -1)"
fi
[[ -f "$MODEL" ]] || { echo "no model found; pass --model" >&2; exit 2; }

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

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIR="${OUT}/r007-retest-${STAMP}-head-${SHORT}"
mkdir -p "$DIR/raw"

VER="$($BINARY --version 2>/dev/null | head -1)"
echo "Camelid: $VER"
echo "model:   $(basename "$MODEL")"
echo "out:     $DIR"
echo

SERVE_PID=""
cleanup() {
    if [[ -n "$SERVE_PID" ]]; then
        kill -TERM "$SERVE_PID" 2>/dev/null
        sleep 2
        kill -KILL "$SERVE_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# F1: does serve come up and survive a first completion, or panic like v0.4.4?
# ---------------------------------------------------------------------------
echo "[F1] starting serve on 127.0.0.1:${PORT}"
# --addr, not --port (verified against `camelid serve --help` on v0.6.1).
# --log-acceleration records which execution lane the engine actually selected,
# which is the context R-007's finding 1 turned on: "auto-select the safest
# validated execution plan" routing into an unvalidated lane.
"$BINARY" serve --addr "127.0.0.1:${PORT}" --model "$MODEL" --log-acceleration \
    > "$DIR/raw/serve.out" 2> "$DIR/raw/serve.err" &
SERVE_PID=$!

# Readiness means the model is loaded, not that the socket answers. The served
# id is whatever /v1/models reports; assuming an alias such as "local" gets a
# model_not_found error that looks like a lane failure but is a client mistake,
# so read it rather than guess it. Both facts come from the same response.
MODEL_ID="$(wait_for_model "$PORT" 300)"
curl -s --max-time 10 "http://127.0.0.1:${PORT}/v1/models" > "$DIR/raw/v1-models.json" 2>&1

if [[ -z "$MODEL_ID" ]]; then
    # Distinguish a dead server from a live one that never finished loading.
    if kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[F1] serve is up but no model became ready within the window"
        F1_RESULT="serve_ready_model_never_loaded"
    else
        echo "[F1] serve did not stay up"
        F1_RESULT="serve_not_ready"
    fi
    MODEL_ID="unknown"
else
    echo "[F1] serve is up, served model id: $MODEL_ID"
    F1_RESULT="serve_ready"
fi

# A plain completion first. In v0.4.4 this is where macOS panicked.
curl -s --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":32}" \
    > "$DIR/raw/f1-plain-completion.json" 2>&1
if kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[F1] worker survived a plain completion"
    F1_RESULT="${F1_RESULT}_survived_plain_completion"
else
    echo "[F1] worker DIED on a plain completion (v0.4.4 behaviour)"
    F1_RESULT="${F1_RESULT}_worker_died"
fi

# ---------------------------------------------------------------------------
# F3: the finding that gates agent loops. Explicit tools array, as R-007 did.
# ---------------------------------------------------------------------------
echo "[F3] probing the OpenAI tools surface"
cat > "$DIR/raw/f3-request.json" <<REQ
{
  "model": "${MODEL_ID}",
  "messages": [
    {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "City name"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 160
}
REQ

curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"$DIR/raw/f3-request.json" \
    > "$DIR/raw/f3-response.json" 2>&1

# Classify strictly. The distinction that matters is whether the serve layer
# parsed the model's tool call into the structured field, or passed the raw
# marker through as prose.
F3_VERDICT="$(python3 - "$DIR/raw/f3-response.json" <<'PY'
import json, sys, re
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("unparseable_response"); raise SystemExit
try:
    msg = d["choices"][0]["message"]
except Exception:
    print("no_choices_in_response"); raise SystemExit
tc = msg.get("tool_calls")
content = msg.get("content") or ""
if tc:
    print("FIXED_tool_calls_populated")
elif re.search(r"<tool_call>|</tool_call>", content):
    print("UNCHANGED_raw_marker_in_content")
elif "get_weather" in content:
    print("PARTIAL_tool_named_in_content_no_markers")
else:
    print("NO_TOOL_ATTEMPT")
PY
)"
echo "[F3] verdict: $F3_VERDICT"

cleanup; SERVE_PID=""

# ---------------------------------------------------------------------------
# Receipt, in pxx's R-NNN shape so it can drop straight into docs/RECEIPTS.md
# ---------------------------------------------------------------------------
cat > "$DIR/receipt.json" <<JSON
{
  "schema": "pxx.r007-retest.v1",
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "retests": "pxx docs/RECEIPTS.md R-007, findings 1 and 3, originally recorded against Camelid v0.4.4 on Apple Silicon",
  "camelid_version": "${VER}",
  "camelid_git_head_short": "${SHORT}",
  "host": {
    "os": "$(uname -s) $(uname -r)",
    "arch": "$(uname -m)",
    "accelerator": "$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo 'metal-or-cpu')"
  },
  "model": {
    "file": "$(basename "$MODEL")",
    "sha256": "$(sha256_of "$MODEL")"
  },
  "findings": {
    "f1_serve_stability": "${F1_RESULT}",
    "f3_openai_tools_surface": "${F3_VERDICT}"
  },
  "scope": "One build, one host, one model, one request per finding. Tests the OpenAI-compatible serve surface only. Says nothing about Camelid's own agent lane, which has a separate receipt and passes independently via 'camelid agent-eval'. No parity, quality, or throughput claim.",
  "verdict_legend": {
    "FIXED_tool_calls_populated": "serve parsed the tool call into the structured tool_calls field; R-007 finding 3 is resolved on this build",
    "UNCHANGED_raw_marker_in_content": "raw <tool_call> markers returned as content with tool_calls null; R-007 finding 3 still present",
    "PARTIAL_tool_named_in_content_no_markers": "tool named in prose but not structured; needs manual reading of raw/f3-response.json",
    "NO_TOOL_ATTEMPT": "model did not attempt the tool; inconclusive for the serve layer, re-run with a tool-capable row"
  }
}
JSON

( cd "$DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
  | while IFS= read -r -d '' f; do printf '%s  %s\n' "$(sha256_of "$f")" "${f#./}"; done \
  > SHA256SUMS )

echo
echo "receipt: $DIR/receipt.json"
echo "  F1 serve stability : ${F1_RESULT}"
echo "  F3 tools surface   : ${F3_VERDICT}"
