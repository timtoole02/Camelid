#!/usr/bin/env bash
# After the first SciFact run: serve the same library with the head build on an otherwise idle host,
# confirm the index is complete, and repeat the search and verify phases.
set -u
EVAL=/mnt/disks/data/camelid-eval
DATA=$EVAL/lib-scifact-1000
OUT=$EVAL/out-scifact-1000-head
grep -q RUN_DONE /tmp/f2h/scifact_run.log || { echo "first run not finished"; exit 1; }
bash /tmp/f2h/stop.sh "$DATA"
rm -rf "$OUT"
mkdir -p "$OUT"
bash /tmp/f2h/serve.sh /tmp/f2h/bin/camelid-head "$DATA" 8190 || exit 1
cd "$EVAL" || exit 1
export CAMELID_BASE=http://127.0.0.1:8190 SCIFACT_DIR=scifact SCIFACT_CORPUS=corpus-subset-1000.jsonl OUT_DIR=$OUT
/tmp/f2h/bin/camelid-head --version > "$OUT/build.txt"
uptime > "$OUT/load-before.txt"
python3 eval_scifact.py index
python3 eval_scifact.py search
python3 eval_scifact.py verify
uptime > "$OUT/load-after.txt"
ps -o rss= -p "$(cat "$DATA/server.pid")" > "$OUT/server-rss-kib.txt"
du -b "$DATA/documents_rag.sqlite3" > "$OUT/db-bytes.txt"
bash /tmp/f2h/stop.sh "$DATA"
bash /tmp/f2h/run_base_compare.sh
echo HEAD_RUN_DONE
