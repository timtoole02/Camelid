#!/bin/bash
# Gold-standard parity evidence: the EXACT pinned oracle build (acd79d603 = b9632,
# the build the fixture names) serving the SAME artifact, asked for its own top-10
# at the exact position camelid diverges.
#
# camelid at that position:  interested -2.142938 | a -2.173546   (gap 0.0306 nats)
# If llama.cpp's distribution matches to a few hundredths of a nat and differs
# only in which side of the tie it lands, that is a documented parity/tolerance
# decision -- which is precisely what the ledger's next_step asks for.
set -u
S=/private/tmp/claude-501/-Users-timtoole/abb2295f-8181-40db-9186-de1e33166d1a/scratchpad
SRV=/Volumes/Untitled/llama.cpp-metal-parity/build/bin/llama-server
M=/Volumes/Untitled/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
PORT=8399
"$SRV" -m "$M" --port $PORT --host 127.0.0.1 -c 512 -ngl 99 --no-warmup > $S/oracle_srv.log 2>&1 &
SRVPID=$!
for i in $(seq 1 90); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && break
  sleep 2
done
# Teacher-forced on the oracle's own prefix, same as the camelid probe:
# [128000, 9906, 11, 358, 1097] = "Hello" + ", I am"
curl -s -m 120 -H 'Content-Type: application/json' -d \
 '{"prompt":[128000,9906,11,358,1097],"n_predict":1,"temperature":0,"top_k":1,"n_probs":10,"cache_prompt":false,"seed":0}' \
 "http://127.0.0.1:$PORT/completion" > $S/oracle_dist.json 2>&1
kill -TERM $SRVPID 2>/dev/null
echo "ORACLE_DIST done"
