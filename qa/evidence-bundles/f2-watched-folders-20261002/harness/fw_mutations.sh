#!/usr/bin/env bash
# Each mutation must make at least one folder unit test fail; the tree is restored after each.
cd /tmp/f2r/wt-folders || exit 1
source ~/.cargo/env
export CARGO_TARGET_DIR="$HOME/Camelid/target"
: > /tmp/f2r/fw-mutations.txt
for name in open_routes follow_links read_hidden drop_unreadable rehash_ignored allow_overlap; do
  python3 /tmp/f2r/fw_mutate.py "$name" >> /tmp/f2r/fw-mutations.txt || { echo "MUTATION $name did not apply"; continue; }
  cargo test --lib -- api::document_folders > /tmp/f2r/fw-mutant.log 2>&1
  status=$?
  failed=$(grep -E "^test api::document_folders::tests::.* FAILED$" /tmp/f2r/fw-mutant.log | sed -E 's/^test api::document_folders::tests::(.*) \.\.\. FAILED$/\1/' | tr '\n' ' ')
  echo "mutation $name: exit $status, failing: ${failed:-none}" | tee -a /tmp/f2r/fw-mutations.txt
  git checkout -- src/api/document_folders.rs
done
git diff --quiet && echo "tree restored" | tee -a /tmp/f2r/fw-mutations.txt
