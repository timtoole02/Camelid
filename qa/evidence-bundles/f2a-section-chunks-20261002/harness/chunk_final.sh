#!/usr/bin/env bash
# Section chunking, final: tests, commit, release build, after-eval, edge suite,
# then the exact branch base (cfa1e191) as the before-eval.
cd /tmp/f2r/wt-library || exit 1
source ~/.cargo/env
export CARGO_TARGET_DIR="$HOME/Camelid/target"
cargo fmt --all && cargo fmt --all -- --check > /tmp/f2r/chunk-fmt.log 2>&1; echo "FMT_EXIT=$?"
git status --short | grep -v '^??'
cargo test --lib -- api:: > /tmp/f2r/chunk-unit.log 2>&1; echo "UNIT_EXIT=$?"
grep -E "^test result:|FAILED|panicked" /tmp/f2r/chunk-unit.log | head
[ "$(grep -c '^test result: ok' /tmp/f2r/chunk-unit.log)" -ge 1 ] || { echo "unit tests failed"; echo CHUNK_FINAL_DONE; exit 1; }
git add src/api/citations.rs
git -c user.email=karan68@users.noreply.github.com -c user.name=karan68 commit -q -F /tmp/f2r/msg_chunks.txt
git log --oneline -2
build() {  # build <checkout> <dest>
  (cd "$1" && CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    cargo build --release --bin camelid --target-dir "$HOME/Camelid/target/eval" > /tmp/f2r/chunk-release.log 2>&1) \
    && grep -q Finished /tmp/f2r/chunk-release.log && cp "$HOME/Camelid/target/eval/release/camelid" "$2" && "$2" --version
}
build /tmp/f2r/wt-library /tmp/f2r/bin/camelid-chunks || { echo "after build failed"; echo CHUNK_FINAL_DONE; exit 1; }
rm -rf /tmp/f2r/lib-chunk-*-after
BIN=/tmp/f2r/bin/camelid-chunks TAG=after OUT=/tmp/f2r/chunk-out python3 /tmp/f2r/chunk_eval.py > /tmp/f2r/chunk-after.log 2>&1
echo "EVAL_AFTER_EXIT=$?"
bash /tmp/f2r/edge_chunks.sh 2>&1 | tail -3
[ -d /tmp/f2r/wt-base ] || git worktree add -q --detach /tmp/f2r/wt-base cfa1e191
ln -sfn /tmp/f2r/wt-library/frontend/node_modules /tmp/f2r/wt-base/frontend/node_modules
build /tmp/f2r/wt-base /tmp/f2r/bin/camelid-786 || { echo "base build failed"; echo CHUNK_FINAL_DONE; exit 1; }
rm -rf /tmp/f2r/lib-chunk-*-before
BIN=/tmp/f2r/bin/camelid-786 TAG=before OUT=/tmp/f2r/chunk-out python3 /tmp/f2r/chunk_eval.py > /tmp/f2r/chunk-before.log 2>&1
echo "EVAL_BEFORE_EXIT=$?"
python3 /tmp/f2r/chunk_compare.py
echo CHUNK_FINAL_DONE
