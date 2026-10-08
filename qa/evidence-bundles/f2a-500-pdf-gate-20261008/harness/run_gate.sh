#!/usr/bin/env bash
# Full F2a 500-PDF gate on the release binary. Logs to run/gate.log.
set -u
ROOT=$HOME/f2a-500
RUN=$ROOT/run
BIN=$HOME/camelid-f2a-target/release/camelid
SERVE=$HOME/camelid-f2a/qa/evidence-bundles/f2-watched-folders-20261002/harness/serve.sh
MODEL=$HOME/Camelid/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
PORT=18793
export MODELS_DIR=$HOME/Camelid/models
# Keyword ranking only, so the run is deterministic; search by meaning is measured separately.
export CAMELID_DOCUMENT_SEMANTIC=0

rm -rf "$RUN" && mkdir -p "$RUN/data" "$RUN/out" "$RUN/corpus"
cp -r "$ROOT/corpus/planted" "$RUN/corpus/"
cp "$ROOT/corpus/planted-facts.json" "$ROOT/corpus/arxiv-manifest.json" "$RUN/corpus/"
ln -s "$ROOT/corpus/pdf" "$RUN/corpus/pdf"

gate() { python3 "$ROOT/gate.py" $PORT "$RUN/corpus" "$RUN/data" "$RUN/out" "$@"; }
stop() { kill "$(cat "$RUN/data/server.pid")"; for _ in $(seq 1 30); do kill -0 "$(cat "$RUN/data/server.pid")" 2>/dev/null || break; sleep 1; done; }

echo "== start $(date -u +%FT%TZ)"
bash "$SERVE" "$BIN" "$RUN/data" $PORT --model "$MODEL" || exit 1
gate ingest
gate verify before
gate retrieve
gate chat
gate corrupt
gate verify after-corrupt
echo "== restart $(date -u +%FT%TZ)"
stop
bash "$SERVE" "$BIN" "$RUN/data" $PORT --model "$MODEL" || exit 1
gate verify after-restart
gate retrieve after-restart
stop
echo "== done $(date -u +%FT%TZ)"
