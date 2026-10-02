#!/usr/bin/env bash
# Screenshots of watched folders from the committed build, against a live server with TinyLlama loaded.
set -u
BIN=/tmp/f2r/bin/camelid-folders
OUT=/tmp/f2r/fw-shots
ROOT=/tmp/f2r/fw-ui
DATA=/tmp/f2r/lib-fw-ui
SRC=/tmp/f2r/wt-library/qa/evidence-bundles/f2a-section-chunks-20261002/source
CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
LAUNCHER=/tmp/f2r/wt-folders/frontend/scripts/lib/launch-browser.mjs
rm -rf "$OUT" "$ROOT" "$DATA"
mkdir -p "$OUT" "$ROOT/watched/team" "$ROOT/archive"
cp "$SRC/support-policy.txt" "$SRC/escalation-guide.txt" "$ROOT/watched/"
printf '# Team notes\n\nThe quarterly review meets on the first Monday of each quarter.\n' > "$ROOT/watched/team/notes.md"
printf '%%PDF-1.7 not really a PDF' > "$ROOT/watched/broken.pdf"
printf '   \n' > "$ROOT/watched/empty.txt"
"$BIN" --version
bash /tmp/f2h/serve.sh "$BIN" "$DATA" 8181 --model "$HOME/Camelid/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf" --no-open || exit 1
for _ in $(seq 1 60); do
  r=$(curl -s -m 5 http://127.0.0.1:8181/v1/health | python3 -c 'import json,sys; print(json.load(sys.stdin).get("generation_ready"))' 2>/dev/null)
  [ "$r" = "True" ] && break
  sleep 2
done
(cd /tmp/f2r/wt-folders/frontend && CHROME_PATH="$CHROME" PUPPETEER_EXECUTABLE_PATH="$CHROME" \
  node /tmp/f2r/capture_folders.mjs http://127.0.0.1:8181 "$OUT" "$LAUNCHER" "$ROOT" "$ROOT/watched")
echo "CAPTURE_EXIT=$?"
bash /tmp/f2h/stop.sh "$DATA"
ls -la "$OUT/screenshots"
echo CAPTURE_DONE
