#!/usr/bin/env bash
# Copy frontend changes in, build, and run the new smoke plus every smoke near the code it touches.
set -u
cd "$HOME/Camelid" || exit 1
SRC=/tmp/f2c/front
for f in $(cd "$SRC" && find . -type f | sed 's|^\./||'); do
  mkdir -p "$(dirname "$f")"
  cp "$SRC/$f" "$f"
done
cd frontend
npm run build > /tmp/f2c/fe_build.log 2>&1 && echo BUILD_OK || { echo BUILD_FAILED; tail -30 /tmp/f2c/fe_build.log; }
export CHROME_PATH=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
export PUPPETEER_EXECUTABLE_PATH="$CHROME_PATH" CI=true
for s in knowledge-collections-browser project-context project-context-browser semantic-index-browser citations-browser document-viewer-browser; do
  if timeout 600 npm run --silent "smoke:$s" > "/tmp/f2c/smoke-$s.log" 2>&1; then
    echo "PASS $s"
  else
    echo "FAIL $s"; grep -E "AssertionError|Error:|expected|actual|at .*smoke" "/tmp/f2c/smoke-$s.log" | head -25
  fi
done
echo SMOKES_DONE
