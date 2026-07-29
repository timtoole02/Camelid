#!/usr/bin/env bash
# E2E capability smoke for the sampling lane + n_choices + stream_usage.
#
# Usage:  qa/capability/smoke.sh <model.gguf> [port] [out-suffix]
#   CAMELID_EXE        camelid binary (default: target/release/camelid.exe).
#                      Set this to run the smoke test against another build.
#   CAMELID_SMOKE_OUT  output dir (default: qa/capability/smoke_out<out-suffix>).
#
# Evidence captured: sampling lane (min_p/repeat_penalty accepted + seed
# reproducibility), n_choices (n=3 shape + n>1/stream fail-closed), stream_usage.
# Requests OMIT "model" so the single loaded model is used (get_or_load_model(None)).
set -u
EXE="${CAMELID_EXE:-target/release/camelid.exe}"
MODEL="${1:?usage: smoke.sh <model.gguf> [port] [out-suffix]}"
PORT="${2:-8231}"
BASE="http://127.0.0.1:${PORT}"
OUT="${CAMELID_SMOKE_OUT:-qa/capability/smoke_out${3:-}}"
mkdir -p "$OUT"
CT='Content-Type: application/json'

export CUDA_VISIBLE_DEVICES=-1
"$EXE" serve --addr 127.0.0.1:"${PORT}" --model "$MODEL" > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
echo "serve pid=$SERVE_PID"
cleanup() { kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null; }
trap cleanup EXIT

# Readiness: /api/models/current returns 404 until the model is active. --fail makes
# 404 an error so --retry-all-errors keeps retrying (curl paces it; no foreground sleep).
if ! curl -s --fail --retry-connrefused --retry-all-errors --retry 180 --retry-delay 1 \
     --max-time 10 "${BASE}/api/models/current" -o "$OUT/current.json"; then
  echo "MODEL NOT READY"; tail -30 "$OUT/serve.log"; exit 1
fi
echo "--- current model ---"; cat "$OUT/current.json"; echo

# (1) Sampling lane: min_p + repeat_penalty ACCEPTED (200, not the old 400 stub).
SAMP_CODE=$(curl -s -o "$OUT/samplers.json" -w "%{http_code}" --max-time 180 -H "$CT" \
  -d '{"messages":[{"role":"user","content":"Name a color."}],"temperature":0,"min_p":0.05,"repeat_penalty":1.3,"max_tokens":6}' \
  "${BASE}/v1/chat/completions")
echo "samplers_http=$SAMP_CODE"

# (1b) out-of-range min_p must be a typed 400.
BADMINP_CODE=$(curl -s -o "$OUT/bad_minp.json" -w "%{http_code}" --max-time 30 -H "$CT" \
  -d '{"messages":[{"role":"user","content":"hi"}],"min_p":1.5,"max_tokens":2}' \
  "${BASE}/v1/chat/completions")
echo "bad_minp_http=$BADMINP_CODE"

# (2) Seed reproducibility end-to-end: two identical seeded temp>0 requests -> identical text.
REQ='{"messages":[{"role":"user","content":"Tell me a word."}],"temperature":0.8,"seed":42,"max_tokens":6}'
curl -s --max-time 180 -H "$CT" -d "$REQ" "${BASE}/v1/chat/completions" > "$OUT/seed_a.json"
curl -s --max-time 180 -H "$CT" -d "$REQ" "${BASE}/v1/chat/completions" > "$OUT/seed_b.json"

# (3) n_choices: n=3 -> 3 choices with indices 0,1,2 and aggregated usage.
curl -s --max-time 300 -H "$CT" \
  -d '{"messages":[{"role":"user","content":"Say something."}],"temperature":0.9,"seed":7,"n":3,"max_tokens":6}' \
  "${BASE}/v1/chat/completions" > "$OUT/n3.json"

# (3b) n>1 + stream must be a typed 400.
NSTREAM_CODE=$(curl -s -o "$OUT/n_stream.json" -w "%{http_code}" --max-time 30 -H "$CT" \
  -d '{"messages":[{"role":"user","content":"hi"}],"n":2,"stream":true,"max_tokens":2}' \
  "${BASE}/v1/chat/completions")
echo "n_stream_http=$NSTREAM_CODE"

# (4) stream_options.include_usage terminal usage chunk.
curl -s --max-time 180 -N -H "$CT" \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":4}' \
  "${BASE}/v1/chat/completions" > "$OUT/stream_usage.txt"

echo "SMOKE_DONE"
