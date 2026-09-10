#!/usr/bin/env bash
# Reference: what does each prompt produce as the ONLY request to a fresh server?
# That is an unambiguous cold GPU prefill with an empty cache -- the bit-exact path.
# The fix claims turn-2 output should now equal this. The old behaviour's CPU-tail
# resume is the arm that has to justify itself against it.
set -uo pipefail
BIN="$HOME/camelid-memctx/camelid-fixed"
MODEL="$HOME/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
port=8150
for sd in 500 501 502; do
  out="$HOME/camelid-memctx/parity-$sd"; mkdir -p "$out"
  "$BIN" serve --addr 127.0.0.1:$port --model "$MODEL" >"$out/serve.out" 2>"$out/serve.err" &
  srv=$!
  for _ in $(seq 1 300); do curl -s "http://127.0.0.1:$port/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && break; sleep 1; done
  python3 - "$port" "$sd" <<'PY'
import json,sys,urllib.request
base='http://127.0.0.1:'+sys.argv[1]; sd=int(sys.argv[2])
V=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','sigma','tau','omega','rho','phi']
s=sd&0xFFFFFFFF; o=[]
for _ in range(350):
    s=(s*1103515245+12345)&0xFFFFFFFF; o.append(V[s%len(V)])
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))['data'][0]['id']
PRE='You are a terse assistant. Answer in one short sentence. Use the notes below and do not speculate beyond them.'
b=json.dumps({'model':m,'messages':[{'role':'system','content':PRE},
  {'role':'user','content':'Here are my notes, please read them carefully:\n\n'+' '.join(o)+'\nAck.'}],
  'temperature':0,'max_tokens':16,'stream':False}).encode()
r=urllib.request.Request(base+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})
print(f'seed {sd} COLD-REFERENCE: {json.load(urllib.request.urlopen(r,timeout=900))["choices"][0]["message"]["content"][:120]}',flush=True)
PY
  kill $srv 2>/dev/null; for _ in $(seq 1 30); do kill -0 $srv 2>/dev/null || break; sleep 1; done; kill -9 $srv 2>/dev/null
  port=$((port+1)); sleep 20
done
