#!/usr/bin/env bash
# Capture the collection screenshots from the committed build with Llama-3.2-3B and the pinned encoder.
set -u
REPO="$HOME/Camelid"
BIN=/tmp/f2c/bin/camelid-collections
OUT=/tmp/f2c/shots
LLAMA="$REPO/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
POLICY="$REPO/qa/evidence-bundles/f2a-verifiable-citations-20260927/source/support-policy.txt"
GUIDE=/tmp/f2c/source/escalation-guide.txt
README=/tmp/f2c/source/README.md
CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
LAUNCHER="$REPO/frontend/scripts/lib/launch-browser.mjs"

rm -rf "$OUT" /tmp/f2c/lib-shots
mkdir -p "$OUT" /tmp/f2c/source
cp "$REPO/README.md" "$README"
cd "$REPO" || exit 1
echo "commit=$(git rev-parse --short HEAD) tree_clean=$([ -z "$(git status --porcelain --untracked-files=no)" ] && echo yes || echo no)"

wait_ready() {
  for _ in $(seq 1 120); do
    r=$(curl -s -m 5 "http://127.0.0.1:$1/v1/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("generation_ready"))' 2>/dev/null)
    [ "$r" = "True" ] && return 0
    sleep 5
  done
  return 1
}

bash /tmp/f2h/serve.sh "$BIN" /tmp/f2c/lib-shots 8181 --model "$LLAMA" || exit 1
wait_ready 8181 && echo "ready 8181"
CHROME_PATH="$CHROME" timeout 1800 node /tmp/f2c/capture_collections.mjs http://127.0.0.1:8181 "$OUT" "$LAUNCHER" "$POLICY" "$GUIDE" "$README"
echo "CAPTURE_EXIT=$?"
bash /tmp/f2h/stop.sh /tmp/f2c/lib-shots
echo CAPTURE_RUN_DONE
