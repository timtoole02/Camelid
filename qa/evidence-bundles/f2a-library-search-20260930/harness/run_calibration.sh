#!/usr/bin/env bash
# Collect whole-library cosine similarities on copies of both calibration libraries, then choose the floor.
set -u
BIN=/tmp/f2c/bin/camelid-collections
EVAL=/mnt/disks/data/camelid-eval
export OUT_DIR=$EVAL/calibration CAMELID_BASE=http://127.0.0.1:8195
for lib in scifact fiqa; do
  if [ "$lib" = fiqa ]; then
    until grep -q FIQA_DONE /tmp/f2l/fiqa_build.log; do sleep 20; done
  fi
  SRC=$EVAL/lib-$lib-1000
  COPY=$EVAL/lib-$lib-calib
  rm -rf "$COPY" && mkdir -p "$COPY" && cp "$SRC/documents_rag.sqlite3" "$COPY/"
  bash /tmp/f2h/serve.sh "$BIN" "$COPY" 8195 || exit 1
  python3 /tmp/f2l/collect_similarity.py "$lib"
  echo "COLLECT_${lib}_EXIT=$?"
  bash /tmp/f2h/stop.sh "$COPY"
done
python3 /tmp/f2l/analyze_floor.py "$OUT_DIR"
echo CALIBRATION_DONE
