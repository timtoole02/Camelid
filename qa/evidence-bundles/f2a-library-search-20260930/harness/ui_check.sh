#!/usr/bin/env bash
# Build the UI and run the library smoke with the smokes nearest to the code it touches; optionally mutate.
cd ~/Camelid/frontend || exit 1
npm run build > /tmp/f2l/fe_build.log 2>&1 && echo BUILD_OK || { echo BUILD_FAILED; tail -30 /tmp/f2l/fe_build.log; echo UI_DONE; exit 1; }
export CHROME_PATH=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
export PUPPETEER_EXECUTABLE_PATH="$CHROME_PATH" CI=true
for s in library-search-browser project-context knowledge-collections-browser project-context-browser semantic-index-browser citations-browser document-viewer-browser; do
  if timeout 600 npm run --silent "smoke:$s" > "/tmp/f2l/smoke-$s.log" 2>&1; then
    echo "PASS $s"
  else
    echo "FAIL $s"; grep -E "AssertionError|Error:|expected|actual|at .*smoke" "/tmp/f2l/smoke-$s.log" | head -25
  fi
done
if [ "${1:-}" = mutate ]; then python3 /tmp/f2l/mutate_library.py; fi
echo UI_DONE
