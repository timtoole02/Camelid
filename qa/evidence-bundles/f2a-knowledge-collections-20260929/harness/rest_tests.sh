#!/usr/bin/env bash
# The test targets cargo did not reach after the debug tinyllama smoke was stopped.
cd ~/Camelid || exit 1
source ~/.cargo/env
until grep -q FULL_DONE /tmp/f2c/full.log; do sleep 10; done
cargo test --all-features --test runnable_smoke -- --skip smoke_admits_tinyllama > /tmp/f2c/rest1.log 2>&1
echo "RUNNABLE_SMOKE_EXIT=$?"
grep -E "^test result:" /tmp/f2c/rest1.log
cargo test --all-features --no-fail-fast --test runnable_tokenizer --test spec_draft_rollback --test tensor_primitives \
  --test tensor_store --test tokenizer --test workspace_api --examples > /tmp/f2c/rest2.log 2>&1
echo "REST_EXIT=$?"
grep -E "^test result:" /tmp/f2c/rest2.log | awk '{p+=$4; f+=$6; i+=$8} END {print "passed", p, "failed", f, "ignored", i, "binaries", NR}'
grep -E "FAILED|panicked" /tmp/f2c/rest2.log | head
git status --short --untracked-files=no
echo REST_DONE
