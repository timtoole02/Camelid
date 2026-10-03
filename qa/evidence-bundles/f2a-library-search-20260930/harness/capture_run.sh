#!/usr/bin/env bash
# Screenshots from the committed build: with the encoder (Llama-3.2-3B answering), then without it.
set -u
REPO="$HOME/Camelid"
BIN=/tmp/f2l/bin/camelid-library
OUT=/tmp/f2l/shots
LLAMA="$REPO/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
POLICY="$REPO/qa/evidence-bundles/f2a-verifiable-citations-20260927/source/support-policy.txt"
GUIDE="$REPO/qa/evidence-bundles/f2a-knowledge-collections-20260929/source/escalation-guide.txt"
DOCS="$REPO/docs/PROJECT_CONTEXT.md"
NO_ENCODER=/tmp/f2l/models-shots-no-encoder
CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
LAUNCHER="$REPO/frontend/scripts/lib/launch-browser.mjs"

rm -rf "$OUT" /tmp/f2l/lib-shots /tmp/f2l/lib-shots-no-encoder "$NO_ENCODER"
mkdir -p "$OUT" "$NO_ENCODER"
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

bash /tmp/f2h/serve.sh "$BIN" /tmp/f2l/lib-shots 8181 --model "$LLAMA" || exit 1
wait_ready 8181 && echo "ready 8181"
CHROME_PATH="$CHROME" timeout 1800 node /tmp/f2l/capture_library.mjs encoder http://127.0.0.1:8181 "$OUT" "$LAUNCHER" "$POLICY" "$GUIDE" "$DOCS"
echo "ENCODER_EXIT=$?"
bash /tmp/f2h/stop.sh /tmp/f2l/lib-shots

MODELS_DIR="$NO_ENCODER" bash /tmp/f2h/serve.sh "$BIN" /tmp/f2l/lib-shots-no-encoder 8181 --model "$NO_ENCODER/$(basename "$LLAMA")" || exit 1
wait_ready 8181 && echo "ready 8181 (no encoder)"
CHROME_PATH="$CHROME" timeout 1800 node /tmp/f2l/capture_library.mjs no-encoder http://127.0.0.1:8181 "$OUT" "$LAUNCHER" "$POLICY"
echo "NO_ENCODER_EXIT=$?"
bash /tmp/f2h/stop.sh /tmp/f2l/lib-shots-no-encoder
echo CAPTURE_RUN_DONE
