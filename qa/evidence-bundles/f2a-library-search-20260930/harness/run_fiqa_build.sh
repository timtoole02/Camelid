#!/usr/bin/env bash
# Build the FiQA calibration library with the #785 build (same Rust as the new branch's base).
set -u
BIN=/tmp/f2c/bin/camelid-collections
LIB=/mnt/disks/data/camelid-eval/lib-fiqa-1000
cd /mnt/disks/data/camelid-eval || exit 1
python3 /tmp/f2l/build_fiqa_library.py subset || exit 1
rm -rf "$LIB"
bash /tmp/f2h/serve.sh "$BIN" "$LIB" 8194 || exit 1
python3 /tmp/f2l/build_fiqa_library.py ingest
python3 /tmp/f2l/build_fiqa_library.py index
bash /tmp/f2h/stop.sh "$LIB"
echo FIQA_DONE
