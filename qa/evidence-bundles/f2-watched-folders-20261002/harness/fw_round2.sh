#!/usr/bin/env bash
# Folder watch, round 2: the UI refresh fix and smoke, the seal re-pin, amend, rebuild, and every check again.
WT=/tmp/f2r/wt-folders
cd "$WT" || exit 1
source ~/.cargo/env
export CARGO_TARGET_DIR="$HOME/Camelid/target"
CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
export CHROME_PATH="$CHROME" PUPPETEER_EXECUTABLE_PATH="$CHROME"
python3 /tmp/f2r/repin_seal.py
git add frontend/src/components/knowledge/WatchedFolders.jsx frontend/scripts/watched-folders-browser-smoke.mjs \
  qa/model-qualification/fixtures/smollm3-default-thinking-runtime-envelope-v1.json \
  scripts/hf-qualification-smollm3-chat-parity.mjs scripts/test-hf-qualification-smollm3-chat-parity.mjs
git -c user.email=karan68@users.noreply.github.com -c user.name=karan68 commit -q --amend -F /tmp/f2r/msg_folders.txt
git log --oneline -2
git show --stat HEAD | tail -22
echo "tracked changes after commit: $(git status --porcelain --untracked-files=no | wc -l)"

(cd frontend && npm run build > /tmp/f2r/fw-fe-build.log 2>&1); echo "FE_BUILD_EXIT=$?"
(cd frontend && npm run smoke:watched-folders-browser > /tmp/f2r/fw-smoke.log 2>&1); echo "SMOKE_EXIT=$?"
tail -2 /tmp/f2r/fw-smoke.log
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release --bin camelid \
  --target-dir "$HOME/Camelid/target/eval" > /tmp/f2r/fw-release.log 2>&1; echo "BUILD_EXIT=$?"
cp "$HOME/Camelid/target/eval/release/camelid" /tmp/f2r/bin/camelid-folders && /tmp/f2r/bin/camelid-folders --version
NEW_BIN=/tmp/f2r/bin/camelid-folders OUT=/tmp/f2r/fw-edge-out SOURCE=/tmp/f2r/wt-library/qa/evidence-bundles/f2a-section-chunks-20261002/source \
  python3 /tmp/f2r/folder_edge.py > /tmp/f2r/fw-edge.log 2>&1; echo "EDGE_EXIT=$?"
grep -E "FAIL|FOLDER EDGE" -A1 /tmp/f2r/fw-edge.log | cut -c1-300
bash /tmp/f2r/capture_run.sh > /tmp/f2r/fw-capture.log 2>&1
grep -E "CAPTURE_EXIT|captured|Error" /tmp/f2r/fw-capture.log | cut -c1-300
bash /tmp/f2r/fw_mutations.sh > /dev/null 2>&1; cat /tmp/f2r/fw-mutations.txt | grep -v "^applied"

for job in frontend public-scrub validation-scripts; do
  echo "== $job"
  REPO="$WT" python3 /tmp/f2r/ci_job_wt.py "$job" > /tmp/f2r/fw-ci-$job.log 2>&1
  grep -E "^FAIL|JOB_DONE" /tmp/f2r/fw-ci-$job.log
done
pass=0; fail=0
for t in scripts/test-*.mjs; do
  if node "$t" > /tmp/f2r/gate-fw.log 2>&1; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $t"; tail -5 /tmp/f2r/gate-fw.log; fi
done
echo "validation gates: $pass passed, $fail failed"
echo "tracked changes at end: $(git status --porcelain --untracked-files=no | wc -l)"
echo FW_ROUND2_DONE
