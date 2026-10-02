#!/usr/bin/env bash
# Folder watch: lint and unit tests on the committed head, the frontend job again, then assemble and check the bundle.
set -u
WT=/tmp/f2r/wt-folders
B=$WT/qa/evidence-bundles/f2-watched-folders-20261002
cd "$WT" || exit 1
source ~/.cargo/env
export CARGO_TARGET_DIR="$HOME/Camelid/target"
CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
export CHROME_PATH="$CHROME" PUPPETEER_EXECUTABLE_PATH="$CHROME"
if [ -z "${SKIP_HEAVY:-}" ]; then
{
  echo "commit $(git rev-parse --short HEAD), tracked changes $(git status --porcelain --untracked-files=no | wc -l)"
  echo "$ cargo fmt --all -- --check"; cargo fmt --all -- --check; echo "FMT_EXIT=$?"
  echo "$ cargo clippy --all-targets -- -D warnings"; cargo clippy --all-targets -- -D warnings 2>&1 | tail -1; echo "CLIPPY_DEFAULT_EXIT=${PIPESTATUS[0]}"
  echo "$ cargo clippy --all-targets --all-features -- -D warnings"; cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -1; echo "CLIPPY_ALL_EXIT=${PIPESTATUS[0]}"
} > /tmp/f2r/fw-lint.txt 2>&1
cat /tmp/f2r/fw-lint.txt
cargo test --lib -- api:: > /tmp/f2r/fw-api-head.log 2>&1; echo "API_TEST_EXIT=$?"
REPO="$WT" python3 /tmp/f2r/ci_job_wt.py frontend > /tmp/f2r/fw-ci-frontend-2.log 2>&1
grep -E "^FAIL|JOB_DONE" /tmp/f2r/fw-ci-frontend-2.log

fi
rm -rf "$B" && mkdir -p "$B"/{data,harness,tests,source,screenshots}
cp /tmp/f2r/fw-bundle/README.md "$B/"
cp /tmp/f2r/fw-bundle/harness/check_claims.py "$B/harness/"
cp /tmp/f2r/fw-edge-out/folder-edge.json /tmp/f2r/fw-shots/capture-folders.json "$B/data/"
cp /tmp/f2r/fw-shots/screenshots/*.png "$B/screenshots/"
cp /tmp/f2r/wt-library/qa/evidence-bundles/f2a-section-chunks-20261002/source/support-policy.txt \
   /tmp/f2r/wt-library/qa/evidence-bundles/f2a-section-chunks-20261002/source/escalation-guide.txt "$B/source/"
cp /tmp/f2r/folder_edge.py /tmp/f2r/capture_folders.mjs /tmp/f2r/capture_run.sh /tmp/f2r/fw_mutate.py /tmp/f2r/fw_mutations.sh \
   /tmp/f2r/fw_round2.sh /tmp/f2r/fe_compare.sh /tmp/f2r/repin_seal.py /tmp/f2r/assemble_fw_bundle.sh /tmp/f2h/serve.sh /tmp/f2h/stop.sh "$B/harness/"
cp /tmp/f2r/fw-lint.txt "$B/tests/lint.txt"
grep -v "^applied" /tmp/f2r/fw-mutations.txt > "$B/tests/mutations.txt"
{ grep -E "^test api::document_folders::tests::" /tmp/f2r/fw-api-head.log | sort; grep -E "^test result:" /tmp/f2r/fw-api-head.log; } > "$B/tests/unit.txt"
tail -3 /tmp/f2r/fw-smoke.log > "$B/tests/smoke.txt"
{
  echo "== frontend, folder branch, PUPPETEER_EXECUTABLE_PATH set"
  grep -E "^(OK|FAIL)|JOB_DONE" /tmp/f2r/fw-ci-frontend-2.log
  echo "== frontend, folder branch, PUPPETEER_EXECUTABLE_PATH unset"
  grep -E "^(OK|FAIL)|JOB_DONE" /tmp/f2r/fe-fw-nopp.log
  echo "== frontend, #788 (no folder changes), PUPPETEER_EXECUTABLE_PATH set"
  grep -E "^(OK|FAIL)|JOB_DONE" /tmp/f2r/fe-788.log
  echo "== public-scrub, validation-scripts, gates"
  grep -E "JOB_DONE public-scrub|JOB_DONE validation-scripts|validation gates:" /tmp/f2r/fw_round2.log
} > "$B/tests/ci.txt"
grep -E "^==|FAIL|JOB_DONE|gates" "$B/tests/ci.txt"
cd "$B"
find . -type f ! -name SHA256SUMS | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
python3 -B harness/check_claims.py
echo "privacy hits:"; grep -rnEf /tmp/f2r/privacy.pat . || echo none
du -sh .
echo FW_BUNDLE_DONE
