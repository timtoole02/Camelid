#!/usr/bin/env bash
# All live evidence for whole-library search, one run after another, once the build and tests are done.
until grep -q FULL_DONE /tmp/f2l/full2.log 2>/dev/null; do sleep 15; done
BIN=/tmp/f2l/bin/camelid-library
cd ~/Camelid || exit 1
echo "commit=$(git rev-parse --short HEAD) tree_clean=$([ -z "$(git status --porcelain --untracked-files=no)" ] && echo yes || echo no)"
"$BIN" --version
for port in 8181 8196 8197; do
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$port/v1/health"; then echo "port $port busy"; exit 1; fi
done
NEW_BIN=$BIN OUT=/tmp/f2l/edge-out python3 /tmp/f2l/edge_library.py
echo "EDGE_EXIT=$?"
BIN=$BIN OUT=/tmp/f2l/endpoint-out python3 /tmp/f2l/endpoint_calibration.py
echo "ENDPOINT_EXIT=$?"
bash /tmp/f2l/capture_run.sh
ps -eo pid,args | grep -E "/tmp/f2[chl]/bin/camelid-" | grep -v grep || echo "no evidence servers left"
echo EVIDENCE_DONE
