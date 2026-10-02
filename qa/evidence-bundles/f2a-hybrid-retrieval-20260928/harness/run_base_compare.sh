#!/usr/bin/env bash
# Serve the SciFact library with the previous build and compare its rankings with keyword mode.
set -u
EVAL=/mnt/disks/data/camelid-eval
DATA=$EVAL/lib-scifact-1000
bash /tmp/f2h/serve.sh /tmp/f2h/bin/camelid-base "$DATA" 8190 || exit 1
cd "$EVAL" || exit 1
CAMELID_BASE=http://127.0.0.1:8190 SCIFACT_DIR=scifact PER_QUERY=$EVAL/out-scifact-1000-head/per-query.json \
  OUT=$EVAL/out-scifact-1000-head python3 base_keyword_compare.py
echo "COMPARE_EXIT=$?"
bash /tmp/f2h/stop.sh "$DATA"
