#!/usr/bin/env bash
# Same binary, both arms. CAMELID_METAL_PREFIX_PARTIAL_RESUME=1 restores the old
# unconditional resume, so the only difference between arms is the decision this PR
# changes. ABBA over the two arms so drift cannot masquerade as the fix.
set -uo pipefail
BIN="$HOME/camelid-memctx/camelid-fixed"
MODEL="$HOME/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

run_arm() {
  local label="$1" forced="$2" port="$3"
  local out="$HOME/camelid-memctx/fix-$label"; mkdir -p "$out"
  CAMELID_METAL_PREFIX_PARTIAL_RESUME="$forced" CAMELID_RESIDENT_TRACE=1 \
    "$BIN" serve --addr 127.0.0.1:$port --model "$MODEL" >"$out/serve.out" 2>"$out/serve.err" &
  local srv=$!
  for _ in $(seq 1 300); do curl -s "http://127.0.0.1:$port/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && break; sleep 1; done
  echo "--- arm $label (CAMELID_METAL_PREFIX_PARTIAL_RESUME=$forced) ---"
  python3 - "$port" "$label" <<'PY'
import json,sys,time,urllib.request
base='http://127.0.0.1:'+sys.argv[1]
V=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','sigma','tau','omega','rho','phi']
def filler(sd,n):
    s=sd&0xFFFFFFFF; o=[]
    for _ in range(n):
        s=(s*1103515245+12345)&0xFFFFFFFF; o.append(V[s%len(V)])
    return ' '.join(o)
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))['data'][0]['id']
PRE='You are a terse assistant. Answer in one short sentence. Use the notes below and do not speculate beyond them.'
def go(sd):
    msgs=[{'role':'system','content':PRE},
          {'role':'user','content':'Here are my notes, please read them carefully:\n\n'+filler(sd,350)+'\nAck.'}]
    b=json.dumps({'model':m,'messages':msgs,'temperature':0,'max_tokens':16,'stream':False}).encode()
    r=urllib.request.Request(base+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})
    t=time.time()
    with urllib.request.urlopen(r,timeout=900) as resp: j=json.load(resp)
    return (time.time()-t)*1000, j['choices'][0]['message']['content']
for i,sd in enumerate([500,501,502]):
    ms,txt=go(sd)
    print(json.dumps({'arm':sys.argv[2],'turn':i,'ms':round(ms,1),'text':txt[:120]}),flush=True)
PY
  kill $srv 2>/dev/null; for _ in $(seq 1 30); do kill -0 $srv 2>/dev/null || break; sleep 1; done; kill -9 $srv 2>/dev/null
  echo "declines: $(grep -c 'declined at site' "$out/serve.err" 2>/dev/null)"
  sleep 25
}
run_arm fixed-1 0 8140
run_arm old-1   1 8141
run_arm old-2   1 8142
run_arm fixed-2 0 8143
