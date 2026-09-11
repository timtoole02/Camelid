#!/bin/bash
# Reproduce the parity probe that left llama3_1_8b_instruct_q8_0 amber on
# 2026-08-11. The fixture pins the artifact sha256, the oracle build
# (llama.cpp b9632 / acd79d603), the prompt IDs and the expected greedy IDs, and
# recorded a divergence at generated index 3 on the CPU reference lane.
# Question one: does it STILL fail on the current build?
set -u
S=/private/tmp/claude-501/-Users-timtoole/abb2295f-8181-40db-9186-de1e33166d1a/scratchpad
M=/Volumes/Untitled/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
BIN=/Volumes/Untitled/cargo-targets/Camelid-3x/release/camelid
OUT=$S/receipts; mkdir -p "$OUT"; : > "$OUT/probe.jsonl"
PORT=8231

echo "=== artifact check: the fixture pins this exact sha256 ==="
want=9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283
# cached: re-hashing 8.5 GB per run costs 2 min and the file has not changed
got=$(cut -d' ' -f1 < "$S/l31q8.sha256")
echo "  expected $want"; echo "  actual   $got"
if [ "$want" != "$got" ]; then echo "  !! MISMATCH — a parity result against a different artifact proves nothing"; exit 1; fi
echo "  MATCHES the pinned fixture"

# CPU reference lane, f32 KV, no repack — the lane the failure was recorded on.
CAM_SESSION_PID=$$ "$HOME"/bin/cam-lock.sh env \
  CAMELID_METAL_RESIDENT_DECODE=0 CAMELID_METAL_RESIDENT_PREFILL=0 \
  "$BIN" serve --addr 127.0.0.1:$PORT --model "$M" > "$OUT/probe_serve.log" 2>&1 &
LOCKPID=$!
# camelid reachable != loaded: gate on generation_ready, not on the port opening.
for i in $(seq 1 240); do
  r=$(curl -s -m 3 "http://127.0.0.1:$PORT/v1/health" 2>/dev/null)
  echo "$r" | grep -q '"generation_ready":true' && { echo "  model loaded after ${i}0s"; break; }
  sleep 10
done

probe() { # name ids expect_ids expect_text
  echo "### $1" >&2
  local body="{\"model\":\"$(basename $M)\",\"camelid_prompt_token_ids\":[$2],\"max_tokens\":5,\"temperature\":0,\"stream\":false}"
  local resp=$(curl -s -m 120 -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/v1/completions")
  python3 - "$1" "$3" "$4" <<PY >> "$OUT/probe.jsonl"
import json,sys
name,exp_ids,exp_txt=sys.argv[1],sys.argv[2],sys.argv[3]
r=json.loads('''$resp''') if '''$resp'''.strip() else {}
txt=(r.get('choices') or [{}])[0].get('text','')
print(json.dumps({"probe":name,"oracle_ids":exp_ids,"oracle_text":exp_txt,"camelid_text":txt,
                  "match":txt==exp_txt,"raw_keys":list(r)[:6]}))
PY
}
probe hello  "128000,9906"                    "11,358,1097,264,220"   ", I am a "
probe france "128000,791,6864,315,9822,374"   "264,3363,315,30363,11" " a city of romance,"
probe once   "128000,12805,5304,264,892"      "11,304,264,2678,14458" ", in a small village"

curl -s -m 5 -X POST "http://127.0.0.1:$PORT/api/shutdown" >/dev/null 2>&1
kill -TERM $LOCKPID 2>/dev/null
echo "PROBE done" >&2
