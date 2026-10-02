#!/usr/bin/env bash
# Lint the section-chunking branch, then assemble and check its evidence bundle.
set -u
WT=/tmp/f2r/wt-library
B=$WT/qa/evidence-bundles/f2a-section-chunks-20261002
cd "$WT" || exit 1
source ~/.cargo/env
export CARGO_TARGET_DIR="$HOME/Camelid/target"
{
  echo "commit $(git rev-parse --short HEAD), tracked changes $(git status --porcelain --untracked-files=no | wc -l)"
  echo "$ cargo fmt --all -- --check"; cargo fmt --all -- --check; echo "FMT_EXIT=$?"
  echo "$ cargo clippy --all-targets -- -D warnings"; cargo clippy --all-targets -- -D warnings 2>&1 | tail -2; echo "CLIPPY_DEFAULT_EXIT=${PIPESTATUS[0]}"
  echo "$ cargo clippy --all-targets --all-features -- -D warnings"; cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -2; echo "CLIPPY_ALL_EXIT=${PIPESTATUS[0]}"
} > /tmp/f2r/chunk-lint.txt 2>&1
cat /tmp/f2r/chunk-lint.txt

rm -rf "$B" && mkdir -p "$B"/{data,harness,tests,source}
cp /tmp/f2r/chunk-bundle/README.md "$B/"
cp /tmp/f2r/chunk-bundle/harness/check_claims.py "$B/harness/"
cp /tmp/f2r/chunk-out/chunk-eval-before.json /tmp/f2r/chunk-out/chunk-eval-after.json /tmp/f2r/chunk-out/probe-chunks.json "$B/data/"
cp /tmp/f2r/edge-chunks/edge-library.json "$B/data/"
cp "$HOME/Camelid/qa/evidence-bundles/f2a-verifiable-citations-20260927/source/support-policy.txt" "$B/source/"
cp qa/evidence-bundles/f2a-knowledge-collections-20260929/source/escalation-guide.txt "$B/source/"
git -C "$HOME/Camelid" show d9318fa7:docs/PROJECT_CONTEXT.md > "$B/source/PROJECT_CONTEXT.md"
cp /tmp/f2r/chunk_eval.py /tmp/f2r/chunk_compare.py /tmp/f2r/dump_probe_chunks.py /tmp/f2r/edge_chunks.sh \
   /tmp/f2r/chunk_final.sh /tmp/f2r/assemble_chunk_bundle.sh /tmp/f2h/serve.sh /tmp/f2h/stop.sh "$B/harness/"
cp qa/evidence-bundles/f2a-library-search-20260930/harness/edge_library.py "$B/harness/"
cp /tmp/f2r/chunk-lint.txt "$B/tests/lint.txt"
{
  grep -E "^test api::citations::tests::(each_section|sections_shorter|prose_without|headings_are)" /tmp/f2r/chunk-unit.log
  grep -E "^test result:" /tmp/f2r/chunk-unit.log
} > "$B/tests/unit.txt"
cd "$B"
find . -type f ! -name SHA256SUMS | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
python3 -B harness/check_claims.py
echo "privacy hits:"; grep -rnEf /tmp/f2r/privacy.pat . || echo none
du -sh .
echo CHUNK_BUNDLE_DONE
