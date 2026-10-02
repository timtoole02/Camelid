#!/usr/bin/env bash
# Full SciFact-subset run against a fresh library on port 8190: ingest, wait for the index, search, verify.
set -u
EVAL=/mnt/disks/data/camelid-eval
DATA=$EVAL/lib-scifact-1000
OUT=$EVAL/out-scifact-1000
rm -rf "$DATA" "$OUT"
bash /tmp/f2h/serve.sh "$HOME/Camelid/target/eval/release/camelid" "$DATA" 8190
cd "$EVAL" || exit 1
export CAMELID_BASE=http://127.0.0.1:8190 SCIFACT_DIR=scifact SCIFACT_CORPUS=corpus-subset-1000.jsonl OUT_DIR=$OUT
python3 eval_scifact.py ingest
python3 eval_scifact.py index
python3 eval_scifact.py search
python3 eval_scifact.py verify
ps -o rss= -p "$(cat "$DATA/server.pid")" > "$OUT/server-rss-kib.txt"
du -b "$DATA/documents_rag.sqlite3" > "$OUT/db-bytes.txt"
echo RUN_DONE
