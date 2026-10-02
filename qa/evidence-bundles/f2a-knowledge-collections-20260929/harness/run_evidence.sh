#!/usr/bin/env bash
# All live evidence for knowledge collections, one after another on an otherwise idle host.
set -u
BIN=/tmp/f2c/bin/camelid-collections
cd ~/Camelid || exit 1
echo "commit=$(git rev-parse --short HEAD) tree_clean=$([ -z "$(git status --porcelain --untracked-files=no)" ] && echo yes || echo no)"
"$BIN" --version
for port in 8181 8192 8193; do
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$port/v1/health"; then echo "port $port busy"; exit 1; fi
done
NEW_BIN=$BIN OLD_BIN=/tmp/f2h/bin/camelid-head OUT=/tmp/f2c/edge-out python3 /tmp/f2c/edge_collections.py
echo "EDGE_EXIT=$?"
BIN=$BIN SRC_LIB=/mnt/disks/data/camelid-eval/lib-scifact-1000 LIB=/mnt/disks/data/camelid-eval/lib-scifact-collections \
  OUT=/tmp/f2c/scale-out SCIFACT_DIR=/mnt/disks/data/camelid-eval/scifact python3 /tmp/f2c/scale_collections.py
echo "SCALE_EXIT=$?"
bash /tmp/f2c/capture_run.sh
ps -eo pid,args | grep -E "/tmp/f2[ch]/bin/camelid-" | grep -v grep || echo "no evidence servers left"
echo EVIDENCE_DONE
