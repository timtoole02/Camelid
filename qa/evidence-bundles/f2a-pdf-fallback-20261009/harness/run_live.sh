#!/usr/bin/env bash
set -u
ROOT=$HOME/f2a-500; RUN=$ROOT/run-799; PORT=18810
SERVE=$HOME/camelid-799/qa/evidence-bundles/f2-watched-folders-20261002/harness/serve.sh
BIN=$HOME/camelid-f2a-target/release/camelid
export MODELS_DIR=$HOME/Camelid/models CAMELID_DOCUMENT_SEMANTIC=0
rm -rf "$RUN" && mkdir -p "$RUN/data" "$RUN/out" "$RUN/corpus"
cp -r "$ROOT/corpus/planted" "$RUN/corpus/"
cp "$ROOT/corpus/planted-facts.json" "$ROOT/corpus/arxiv-manifest.json" "$RUN/corpus/"
ln -s "$ROOT/corpus/pdf" "$RUN/corpus/pdf"
bash "$SERVE" "$BIN" "$RUN/data" $PORT || exit 1
GATE_REAL_TARGET=500 python3 "$ROOT/gate.py" $PORT "$RUN/corpus" "$RUN/data" "$RUN/out" ingest > "$RUN/ingest.log" 2>&1
python3 "$ROOT/gate.py" $PORT "$RUN/corpus" "$RUN/data" "$RUN/out" verify all > "$RUN/verify.log" 2>&1
kill "$(cat "$RUN/data/server.pid")"
echo DONE > "$RUN/done"
