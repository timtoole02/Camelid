#!/usr/bin/env bash
# Run every validation gate exactly like CI's validation-scripts job.
cd ~/Camelid
pass=0; fail=0
for t in scripts/test-*.mjs; do
  if node "$t" > /tmp/f2v_gate.log 2>&1; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $t"; tail -8 /tmp/f2v_gate.log; fi
done
echo "validation gates: $pass passed, $fail failed"
