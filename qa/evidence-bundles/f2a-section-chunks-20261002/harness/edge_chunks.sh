#!/usr/bin/env bash
# Run #786's library-search edge suite against the section-chunking binary.
SRC=/tmp/f2r/wt-library/qa/evidence-bundles/f2a-library-search-20260930/harness/edge_library.py
sed -e 's|^REPO = .*|REPO = "/tmp/f2r/wt-library"|' \
    -e 's|^MODELS = .*|MODELS = os.path.expanduser("~/Camelid/models")|' \
    "$SRC" > /tmp/f2r/edge_library_chunks.py
grep -n "^REPO\|^MODELS" /tmp/f2r/edge_library_chunks.py
NEW_BIN=/tmp/f2r/bin/camelid-chunks OUT=/tmp/f2r/edge-chunks python3 /tmp/f2r/edge_library_chunks.py
echo "EDGE_EXIT=$?"
echo EDGE_DONE
