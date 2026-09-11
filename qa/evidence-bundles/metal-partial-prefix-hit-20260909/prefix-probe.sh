#!/usr/bin/env bash
# Does a PARTIAL prompt-prefix-cache hit make Metal SLOWER than a cold miss?
# Predicted: a hit rolls the session back to position>0, try_metal_resident_prefill
# declines (metal_resident.rs:302 `|| self.kv_cache.position != 0`), and the divergent
# tail falls to the CPU causal walk. If so, sharing a prefix is a large REGRESSION.
set -uo pipefail
PORT=8136
BIN="$HOME/camelid-memctx/camelid"; MODEL="$HOME/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
OUT="$HOME/camelid-memctx/prefixprobe"; mkdir -p "$OUT"
CAMELID_METAL_KQUANT_ATTN_MM=1 "$BIN" serve --addr 127.0.0.1:$PORT --model "$MODEL" >"$OUT/serve.out" 2>"$OUT/serve.err" &
SRV=$!
for _ in $(seq 1 300); do curl -s "http://127.0.0.1:$PORT/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && break; sleep 1; done
python3 - "$PORT" <<'PY'
import json,sys,time,urllib.request
base='http://127.0.0.1:'+sys.argv[1]
V=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','sigma','tau','omega','rho','phi']
def filler(sd,n):
    s=sd&0xFFFFFFFF; o=[]
    for _ in range(n):
        s=(s*1103515245+12345)&0xFFFFFFFF; o.append(V[s%len(V)])
    return ' '.join(o)
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))['data'][0]['id']
LONG_PREAMBLE='You are a terse assistant. Answer in one short sentence. Use the notes below and do not speculate beyond them.'
def go(sd,shared_prefix):
    msgs=([{'role':'system','content':LONG_PREAMBLE}] if shared_prefix else [])
    lead='Here are my notes, please read them carefully:\n\n' if shared_prefix else 'N:\n'
    msgs=msgs+[{'role':'user','content':lead+filler(sd,350)+'\nAck.'}]
    b=json.dumps({'model':m,'messages':msgs,'temperature':0,'max_tokens':8,'stream':False}).encode()
    r=urllib.request.Request(base+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})
    t=time.time()
    with urllib.request.urlopen(r,timeout=900) as resp: resp.read()
    return (time.time()-t)*1000
print('--- NO shared prefix (each request a cache MISS) ---',flush=True)
n=300
for i in range(3):
    print(f'  req{i}: {go(n,False):9.1f} ms',flush=True); n+=1
print('--- SHARED long prefix (req0 miss, then PARTIAL HITS) ---',flush=True)
n=400
for i in range(3):
    print(f'  req{i}: {go(n,True):9.1f} ms',flush=True); n+=1
PY
kill $SRV 2>/dev/null; for _ in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 1; done; kill -9 $SRV 2>/dev/null
