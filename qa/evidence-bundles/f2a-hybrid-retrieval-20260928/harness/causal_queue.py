#!/usr/bin/env python3
"""Is the late-upload fix causal? The same scenario on the build before it and on the build with it.

A three-copy README (about 110 chunks) is uploaded; once the indexer has stored its first batch, a
one-chunk note is uploaded. Recorded: how long the note took to be embedded, and how much of the large
file was embedded by then.

usage: causal_queue.py   (env: OLD_BIN, NEW_BIN, OUT)
"""
import json
import os
import sys
import time

os.environ.setdefault("BASE_BIN", os.environ["OLD_BIN"])
import edge_hybrid as e  # noqa: E402

e.DATA = "/tmp/f2h/lib-causal-queue"
e.DB = os.path.join(e.DATA, "documents_rag.sqlite3")


def scenario(binary):
    os.system(f"rm -rf {e.DATA}")
    e.start(binary)
    large = e.ingest("large.md", "\n\n".join([e.README] * 3))
    started = time.time()
    while time.time() - started < 600:
        entry = e.coverage(large)[1]
        if entry and entry["indexed_chunks"] >= 1:
            break
        time.sleep(0.5)
    at_upload = entry
    uploaded = time.time()
    late = e.ingest("late.txt", "A late note: the lighthouse ferry leaves at seven.")
    late_entry = e.wait_indexed(late, timeout=900)
    waited = round(time.time() - uploaded, 1)
    large_entry = e.coverage(large)[1]
    server = e.SERVERS[-1]
    e.stop()
    return {"build": server["build"], "binary": server["binary"], "large_when_note_uploaded": at_upload,
            "note_embedded_after_s": waited, "note": late_entry, "large_when_note_embedded": large_entry,
            "note_went_first": late_entry["indexed_chunks"] == 1
            and large_entry["indexed_chunks"] < large_entry["indexable_chunks"]}


def main():
    old = scenario(os.environ["OLD_BIN"])
    new = scenario(os.environ["NEW_BIN"])
    result = {"before_fix": old, "with_fix": new,
              "causal": (not old["note_went_first"]) and new["note_went_first"]}
    os.makedirs(os.environ["OUT"], exist_ok=True)
    with open(os.path.join(os.environ["OUT"], "late-upload-causal.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["causal"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        os.system(f"bash /tmp/f2h/stop.sh {e.DATA} >/dev/null 2>&1")
