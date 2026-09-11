#!/usr/bin/env bash
# What does the F16 KV primary actually save? Grow the KV cache with a deep prompt, then
# read physical footprint. The saving is the whole point of #2, so measure it rather than
# quoting arithmetic.
set -uo pipefail
BIN="$HOME/camelid-memctx/camelid-both"
MODEL="$HOME/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
for arm in f32:default f16:f16; do
  lbl="${arm%%:*}"; dt="${arm##*:}"; port=$((8230 + RANDOM % 40))
  out="$HOME/camelid-memctx/kvmem-$lbl"; mkdir -p "$out"
  if [ "$dt" = "default" ]; then "$BIN" serve --addr 127.0.0.1:$port --model "$MODEL" >"$out/o" 2>"$out/e" &
  else CAMELID_METAL_KV_DTYPE=f16 "$BIN" serve --addr 127.0.0.1:$port --model "$MODEL" >"$out/o" 2>"$out/e" & fi
  srv=$!
  for _ in $(seq 1 300); do curl -s "http://127.0.0.1:$port/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && break; sleep 1; done
  fp() { footprint -p "$srv" 2>/dev/null | grep -iE "phys_footprint|Physical footprint" | tail -1 | grep -oE "[0-9.]+ *[KMG]B?" | tail -1; }
  python3 - "$port" >/dev/null <<'PY'
import json,sys,urllib.request
base='http://127.0.0.1:'+sys.argv[1]
V=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta']
s=21; o=[]
for _ in range(3200):
    s=(s*1103515245+12345)&0xFFFFFFFF; o.append(V[s%len(V)])
m=json.load(urllib.request.urlopen(base+'/v1/models',timeout=30))['data'][0]['id']
b=json.dumps({'model':m,'messages':[{'role':'user','content':'Notes:\n'+' '.join(o)+'\nWrite at length.'}],
              'temperature':0,'max_tokens':200}).encode()
r=urllib.request.Request(base+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'})
urllib.request.urlopen(r,timeout=1800).read()
PY
  sleep 4
  printf '%-6s deep-context footprint: %s\n' "$lbl" "$(fp)"
  kill $srv 2>/dev/null; for _ in $(seq 1 30); do kill -0 $srv 2>/dev/null || break; sleep 1; done; kill -9 $srv 2>/dev/null
  sleep 20
done
