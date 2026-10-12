#!/usr/bin/env bash
# run-matrix.sh <label> <binary> <model file> <port> <filler> <abort ms> <stress rounds> <stress max abort ms>
# SERVE_EXTRA="--gpu off" adds serve arguments.
# SKIP_QUEUE=1 skips the engine-queue check (the runnable lane has no queue).
# Extra env (e.g. CAMELID_QWEN35_CUDA=1) is passed through to the server.
# Runs every check against one fresh server and appends JSON lines to ~/r803/<label>.jsonl.
set -u
LABEL=$1 BIN=$2 MODEL=$3 PORT=$4 FILLER=$5 ABORT=$6 ROUNDS=$7 MAXABORT=$8
OUT=~/r803/$LABEL.jsonl
DATA=~/r803/data-$LABEL
rm -rf "$DATA"; : > "$OUT"
bash ~/r803/serve.sh "$BIN" "$DATA" "$PORT" --model ~/Camelid/models/"$MODEL" ${SERVE_EXTRA:-} || exit 1
for _ in $(seq 1 300); do
  curl -s "127.0.0.1:$PORT/v1/health" | grep -q '"generation_ready":true' && break
  sleep 1
done
cd ~/r803
[ "${SKIP_QUEUE:-0}" = 1 ] || node queue.mjs "$PORT" "$FILLER" | sed 's/^/{"check":"queue","r":/; s/$/}/' >> "$OUT"
for mode in nonstream stream; do
  node cancel-bench.mjs "$PORT" "$mode" "$ABORT" "$FILLER" 32 | sed "s/^/{\"check\":\"cancel-$mode\",\"r\":/; s/\$/}/" >> "$OUT"
done
node stress.mjs "$PORT" 0 "$FILLER" "$MAXABORT" 7 | sed 's/^/{"check":"stress-reference","r":/; s/$/}/' >> "$OUT"
kill "$(cat "$DATA/server.pid")"; sleep 3
# The stress run gets its own fresh server so the reference above cannot be served
# from the prompt cache.
bash ~/r803/serve.sh "$BIN" "$DATA-stress" "$PORT" --model ~/Camelid/models/"$MODEL" ${SERVE_EXTRA:-} || exit 1
for _ in $(seq 1 300); do
  curl -s "127.0.0.1:$PORT/v1/health" | grep -q '"generation_ready":true' && break
  sleep 1
done
if [ "$ROUNDS" -gt 0 ]; then
  node stress.mjs "$PORT" "$ROUNDS" "$FILLER" "$MAXABORT" 7 | sed 's/^/{"check":"stress","r":/; s/$/}/' >> "$OUT"
fi
grep -c . "$OUT"
ps -o rss= -p "$(cat "$DATA-stress/server.pid")" | sed 's/^/server rss kB after stress: /' >> "$OUT.txt"
kill "$(cat "$DATA-stress/server.pid")"
echo DONE >> "$OUT.txt"
