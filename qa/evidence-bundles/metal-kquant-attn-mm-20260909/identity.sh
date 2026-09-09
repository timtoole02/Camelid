#!/usr/bin/env bash
# Does CAMELID_METAL_KQUANT_ATTN_MM change greedy output on 3B-Q4_K_M?
#
# The flag is opt-in for a PRECISION reason, not a qualification gap: metal.rs records
# "divergence at generated token 4 on Llama-3.2-1B-Q4_K_M". No equivalent measurement
# exists for any other model. If 3B is token-identical that is new information; if it
# diverges, the opt-in is correct and should stay.
#
# One server per arm (the gate latches in a OnceLock). Greedy, 128 tokens, several prompt
# shapes including natural language -- divergence shows up where logits are flattest.
# Uses the PR #739 binary so a partial prefix-cache hit cannot confound the comparison.
set -uo pipefail
BIN="$HOME/camelid-memctx/camelid-fixed"
MODEL="$HOME/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
port=8160
for flag in 0 1; do
  out="$HOME/camelid-memctx/ident-$flag"; mkdir -p "$out"
  CAMELID_METAL_KQUANT_ATTN_MM="$flag" "$BIN" serve --addr 127.0.0.1:$port --model "$MODEL" \
    >"$out/serve.out" 2>"$out/serve.err" &
  srv=$!
  for _ in $(seq 1 300); do curl -s "http://127.0.0.1:$port/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && break; sleep 1; done
  python3 - "$port" "$flag" > "$out/out.jsonl" <<'PY'
import json,sys,urllib.request
base='http://127.0.0.1:'+sys.argv[1]; flag=sys.argv[2]
V=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','sigma','tau','omega','rho','phi']
def filler(sd,n):
    s=sd&0xFFFFFFFF; o=[]
    for _ in range(n):
        s=(s*1103515245+12345)&0xFFFFFFFF; o.append(V[s%len(V)])
    return ' '.join(o)
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))['data'][0]['id']
ESSAY=("The history of computing is often told as a story of hardware, but the decisive "
       "shifts were usually about memory. Early machines were limited less by how fast they "
       "could compute than by how much they could hold and how quickly they could reach it. ")
CASES=[('filler-350', 'Notes:\n'+filler(11,350)+'\nSummarise.'),
       ('filler-1100','Notes:\n'+filler(12,1100)+'\nSummarise.'),
       ('filler-2200','Notes:\n'+filler(13,2200)+'\nSummarise.'),
       ('prose-900',  (ESSAY*12)+'\nContinue this essay in the same register.'),
       ('prose-1800', (ESSAY*24)+'\nContinue this essay in the same register.')]
for name,content in CASES:
    b=json.dumps({'model':m,'messages':[{'role':'user','content':content}],
                  'temperature':0,'max_tokens':128,'stream':False}).encode()
    r=urllib.request.Request(base+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})
    j=json.load(urllib.request.urlopen(r,timeout=900))
    print(json.dumps({'case':name,'flag':flag,
                      'prompt_tokens':j.get('usage',{}).get('prompt_tokens'),
                      'text':j['choices'][0]['message']['content']}),flush=True)
PY
  kill $srv 2>/dev/null; for _ in $(seq 1 30); do kill -0 $srv 2>/dev/null || break; sleep 1; done; kill -9 $srv 2>/dev/null
  port=$((port+1)); sleep 25
done
python3 - <<'PY'
import json,os
h=os.path.expanduser('~')
a={json.loads(l)['case']:json.loads(l) for l in open(f'{h}/camelid-memctx/ident-0/out.jsonl')}
b={json.loads(l)['case']:json.loads(l) for l in open(f'{h}/camelid-memctx/ident-1/out.jsonl')}
print(f"{'case':14s} {'ptok':>6s}  {'identical?':11s}  first divergence")
for k in a:
    ta,tb=a[k]['text'],b[k]['text']
    if ta==tb:
        print(f"{k:14s} {str(a[k]['prompt_tokens']):>6s}  {'IDENTICAL':11s}  -")
    else:
        i=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y), min(len(ta),len(tb)))
        print(f"{k:14s} {str(a[k]['prompt_tokens']):>6s}  {'DIVERGES':11s}  char {i}: off={ta[i:i+40]!r} on={tb[i:i+40]!r}")
PY
