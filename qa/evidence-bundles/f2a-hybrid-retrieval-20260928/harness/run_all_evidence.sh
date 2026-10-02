#!/usr/bin/env bash
# Every evidence run on the head build, one after another on an otherwise idle host.
set -u
cd /tmp/f2h || exit 1
/tmp/f2h/bin/camelid-head --version
echo "== SciFact search and verify on the head build, then the previous build's keyword comparison"
bash /tmp/f2h/run_scifact_head.sh
echo "SCIFACT_EXIT=$?"
echo "== edge cases and skip recovery"
bash /tmp/f2h/run_final_edge.sh
echo "== screenshots"
bash /tmp/f2h/capture_hybrid_run.sh
echo "CAPTURE_EXIT=$?"
ps -eo pid,args | grep -E "/tmp/f2h/bin/camelid-|release/camelid serve" | grep -v grep || echo "no servers left"
echo ALL_EVIDENCE_DONE
