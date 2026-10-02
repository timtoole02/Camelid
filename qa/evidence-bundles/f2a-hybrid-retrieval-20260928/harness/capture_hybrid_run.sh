#!/usr/bin/env bash
# Capture both phases from the committed build: with the encoder, then without it.
set -u
REPO="$HOME/Camelid"
BIN=/tmp/f2h/bin/camelid-head
OUT=/tmp/f2h/shots
LLAMA="$REPO/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
POLICY="$REPO/qa/evidence-bundles/f2a-verifiable-citations-20260927/source/support-policy.txt"
PROGRESS=/tmp/f2h/capture-docs/README.md
NO_ENCODER=/tmp/f2h/models-no-encoder
CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
LAUNCHER="$REPO/frontend/scripts/lib/launch-browser.mjs"

rm -rf "$OUT" /tmp/f2h/lib-shots /tmp/f2h/lib-shots-keyword
mkdir -p "$OUT" "$NO_ENCODER" "$(dirname "$PROGRESS")"
cp "$REPO/README.md" "$PROGRESS"
ln -sf "$LLAMA" "$NO_ENCODER/$(basename "$LLAMA")"
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

bash /tmp/f2h/serve.sh "$BIN" /tmp/f2h/lib-shots 8181 --model "$LLAMA"
wait_ready 8181 && echo "ready 8181"
curl -s http://127.0.0.1:8181/v1/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print("build", h.get("build"))'
CHROME_PATH="$CHROME" timeout 1800 node /tmp/f2h/capture_hybrid.mjs semantic http://127.0.0.1:8181 "$OUT" "$LAUNCHER" "$POLICY" "$PROGRESS"
echo "SEMANTIC_EXIT=$?"
ps -o rss= -p "$(cat /tmp/f2h/lib-shots/server.pid)" > "$OUT/server-rss-kib-with-encoder.txt"
bash /tmp/f2h/stop.sh /tmp/f2h/lib-shots

MODELS_DIR="$NO_ENCODER" bash /tmp/f2h/serve.sh "$BIN" /tmp/f2h/lib-shots-keyword 8181 --model "$NO_ENCODER/$(basename "$LLAMA")"
wait_ready 8181 && echo "ready 8181 (no encoder)"
CHROME_PATH="$CHROME" timeout 1800 node /tmp/f2h/capture_hybrid.mjs keyword-only http://127.0.0.1:8181 "$OUT" "$LAUNCHER" "$POLICY"
echo "KEYWORD_EXIT=$?"
ps -o rss= -p "$(cat /tmp/f2h/lib-shots-keyword/server.pid)" > "$OUT/server-rss-kib-without-encoder.txt"
bash /tmp/f2h/stop.sh /tmp/f2h/lib-shots-keyword
echo CAPTURE_RUN_DONE
