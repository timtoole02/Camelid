#!/usr/bin/env bash
# Prove CAMELID_METAL_KQUANT_ATTN_MM actually changes the prefill code path before
# spending a timed run on it. CAMELID_PREFILL_TRACE=1 splits each prefill stage into
# its own command buffer and reports hardware GPU-busy per stage: if the flag engages,
# the stage SET differs between arms. Tracing inflates wall time (per-stage waits), so
# this run is for attribution only -- the timed A/B runs untraced.
set -uo pipefail
PORT=8132
BIN="$HOME/camelid-memctx/camelid"
MODEL="$HOME/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

for flag in 0 1; do
  OUT="$HOME/camelid-memctx/pre-$flag"; mkdir -p "$OUT"
  CAMELID_METAL_KQUANT_ATTN_MM="$flag" CAMELID_PREFILL_TRACE=1 \
    "$BIN" serve --addr "127.0.0.1:$PORT" --model "$MODEL" \
    >"$OUT/serve.out" 2>"$OUT/serve.err" &
  SRV=$!
  ready=0
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$PORT/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && { ready=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then echo "flag=$flag: server not ready"; tail -5 "$OUT/serve.err"; kill $SRV 2>/dev/null; continue; fi

  python3 - "$PORT" <<'PY'
import json,sys,urllib.request
base='http://127.0.0.1:'+sys.argv[1]
V=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','sigma','tau','omega','rho','phi']
s=7; w=[]
for _ in range(2200):
    s=(s*1103515245+12345)&0xFFFFFFFF; w.append(V[s%len(V)])
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))['data'][0]['id']
b=json.dumps({'model':m,'messages':[{'role':'user','content':'Notes:\n'+' '.join(w)+'\nAck.'}],
              'temperature':0,'max_tokens':8}).encode()
r=urllib.request.Request(base+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})
print(json.load(urllib.request.urlopen(r,timeout=900))['choices'][0]['message']['content'][:80])
PY

  kill $SRV 2>/dev/null; for _ in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 1; done; kill -9 $SRV 2>/dev/null
  echo "----- flag=$flag prefill stage trace -----"
  grep -iE "prefill|stage|attn|mm" "$OUT/serve.err" | tail -25
done
