#!/usr/bin/env bash
# Would a Q8_0 model be better or worse on an F16 KV primary?
#
# No code change needed: CAMELID_METAL_KV_DTYPE=f16 IS that override. metal.rs records
# this having been tried before and costing "~26% decode and ~2.2x prefill on a 2k-token
# Q8_0 run" -- but split-K has since gained kv16-primary support, so the decode half may
# be stale. Measure rather than assume, in both directions.
#
# ABBA. One server per arm (the format latches in a OnceLock). Decode tok/s from a long
# generation on a short prompt; prefill from TTFT on a long prompt.
set -uo pipefail
BIN="$HOME/camelid-memctx/camelid"      # stock origin/main
MODEL="$HOME/models/Llama-3.2-3B-Instruct-Q8_0.gguf"

run_arm() {
  local label="$1" dtype="$2" port="$3"
  local out="$HOME/camelid-memctx/q8kv-$label"; mkdir -p "$out"
  if [ "$dtype" = "default" ]; then
    "$BIN" serve --addr 127.0.0.1:$port --model "$MODEL" >"$out/serve.out" 2>"$out/serve.err" &
  else
    CAMELID_METAL_KV_DTYPE="$dtype" "$BIN" serve --addr 127.0.0.1:$port --model "$MODEL" >"$out/serve.out" 2>"$out/serve.err" &
  fi
  local srv=$!
  for _ in $(seq 1 300); do curl -s "http://127.0.0.1:$port/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && break; sleep 1; done
  python3 - "$port" "$label" <<'PY'
import json,sys,time,urllib.request
base='http://127.0.0.1:'+sys.argv[1]; lbl=sys.argv[2]
V=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','sigma','tau','omega','rho','phi']
def filler(sd,n):
    s=sd&0xFFFFFFFF; o=[]
    for _ in range(n):
        s=(s*1103515245+12345)&0xFFFFFFFF; o.append(V[s%len(V)])
    return ' '.join(o)
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))['data'][0]['id']
def go(prompt,maxtok):
    b=json.dumps({'model':m,'messages':[{'role':'user','content':prompt}],
                  'temperature':0,'max_tokens':maxtok,'stream':False}).encode()
    r=urllib.request.Request(base+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})
    t=time.time()
    with urllib.request.urlopen(r,timeout=900) as resp: j=json.load(resp)
    return (time.time()-t)*1000, j['usage']['completion_tokens'], j['choices'][0]['message']['content']
go('Hi.',8)  # warm
# DECODE: short prompt, long generation -> wall is dominated by decode
ms,ntok,_=go('Write a long detailed essay about the history of the printing press.',256)
print(json.dumps({'arm':lbl,'metric':'decode','ms':round(ms,1),'tokens':ntok,'tok_per_s':round(ntok/(ms/1000),2)}),flush=True)
# PREFILL: long prompt, tiny generation -> wall is dominated by prefill
for w in (700,2200):
    ms,_,txt=go('Notes:\n'+filler(21,w)+'\nAck.',4)
    print(json.dumps({'arm':lbl,'metric':f'prefill-{w}w','ms':round(ms,1),'text':txt[:60]}),flush=True)
# SHORT-CONTEXT PARITY: below 128 positions the f32 primary is read directly
for w in (20,40):
    ms,_,txt=go('Notes: '+filler(31,w)+'\nAck.',24)
    print(json.dumps({'arm':lbl,'metric':f'short-{w}w','ms':round(ms,1),'text':txt[:120]}),flush=True)
PY
  kill $srv 2>/dev/null; for _ in $(seq 1 30); do kill -0 $srv 2>/dev/null || break; sleep 1; done; kill -9 $srv 2>/dev/null
  sleep 20
}
run_arm f32-1 default 8180
run_arm f16-1 f16     8181
run_arm f16-2 f16     8182
run_arm f32-2 default 8183
