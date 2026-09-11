#!/bin/bash
# Is the 3.1 divergence a NEAR-TIE or a real error? That distinction decides
# whether promotion needs a bug fix or the "documented parity/tolerance decision"
# the ledger's own next_step asks for.
#
# Oracle: "Hello" -> [11, 358, 1097, 264, 220] = ", I am a "
# Camelid diverges at generated index 3: picks something else instead of 264.
#
# Feed the ORACLE's own prefix [.., 11, 358, 1097] and ask for logprobs at the
# next position. If token 264 is camelid's #2 by a hair, this is a tie, not an
# error -- and the qwen35 precedent shows the discriminating test: a real bug put
# the oracle token 70th and 8-19 nats out, which no tolerance can excuse.
set -u
M=$HOME/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
BIN=$HOME/camelid-integrity-check/target/release/camelid
PORT=8232
CAM_SESSION_PID=$$ ~/bin/cam-lock.sh env \
  CAMELID_METAL_RESIDENT_DECODE=0 CAMELID_METAL_RESIDENT_PREFILL=0 CAMELID_LAZY_Q8_0_LINEAR=1 \
  "$BIN" serve --addr 127.0.0.1:$PORT --model "$M" --no-open > ~/logit_serve.log 2>&1 &
for i in $(seq 1 120); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/health" 2>/dev/null | grep -q '"generation_ready":true' && break
  sleep 10
done
# teacher-forced: give it the oracle's own first 3 generated tokens, ask what comes next
curl -s -m 300 -H 'Content-Type: application/json' -d \
 '{"model":"Meta Llama 3.1 8B Instruct","camelid_prompt_token_ids":[128000,9906,11,358,1097],"max_tokens":1,"temperature":0,"logprobs":10,"stream":false}' \
 "http://127.0.0.1:$PORT/v1/completions" > ~/logit_probe.json 2>&1
echo "--- top candidates at the divergence position ---"
python3 - <<'PY'
import json
try: r=json.load(open('"$HOME"/logit_probe.json'))
except Exception as e: print('parse fail:',e); raise SystemExit
ch=(r.get('choices') or [{}])[0]
print('text:', repr(ch.get('text')))
lp=ch.get('logprobs') or {}
print('logprobs keys:', list(lp))
top=lp.get('top_logprobs') or []
if top:
    for i,d in enumerate(top[:1]):
        items=sorted(d.items(), key=lambda kv:-kv[1])
        print(f'position {i}:')
        for tok,v in items[:10]: print(f'   {tok!r:22} {v:.6f}')
else:
    print(json.dumps(r)[:800])
PY
pkill -f "camelid serve" 2>/dev/null
echo "LOGIT done"
