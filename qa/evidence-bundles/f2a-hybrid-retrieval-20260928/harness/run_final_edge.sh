#!/usr/bin/env bash
# Final evidence runs on the head build: the edge suite, then the before/after skip-recovery check.
set -u
if curl -s -m 2 -o /dev/null http://127.0.0.1:8192/v1/health; then echo "port 8192 busy"; exit 1; fi
cd /tmp/f2h || exit 1
NEW_BIN=/tmp/f2h/bin/camelid-head BASE_BIN=/tmp/f2h/bin/camelid-base OUT=/tmp/f2h/edge-final \
  python3 /tmp/f2h/edge_hybrid.py
echo "EDGE_EXIT=$?"
OLD_BIN=/tmp/f2h/bin/camelid-63746aa8 NEW_BIN=/tmp/f2h/bin/camelid-head OUT=/tmp/f2h/edge-final \
  python3 /tmp/f2h/causal_skip.py
echo "CAUSAL_EXIT=$?"
OLD_BIN=/tmp/f2h/bin/camelid-5fb64e0f NEW_BIN=/tmp/f2h/bin/camelid-head OUT=/tmp/f2h/edge-final \
  python3 /tmp/f2h/causal_queue.py
echo "QUEUE_CAUSAL_EXIT=$?"
ps -eo pid,args | grep "/tmp/f2h/bin/camelid-" | grep -v grep || echo "no edge servers left"
