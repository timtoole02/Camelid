#!/usr/bin/env python3
"""Is the skip-recovery fix causal? Run the same scenario on the build before it and on the build with it.

usage: causal_skip.py   (env: OLD_BIN, NEW_BIN, OUT)
Reuses the edge suite's helpers on its own data directory and port.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("BASE_BIN", os.environ["OLD_BIN"])
import edge_hybrid as e  # noqa: E402

e.DATA = "/tmp/f2h/lib-causal"
e.DB = os.path.join(e.DATA, "documents_rag.sqlite3")
TEXT = "Skip test: the courier leaves at dawn from the east gate."
DRIFTED = "Skip test: the courier leaves at noon from the east gate."
WAIT_S = 180


def scenario(binary):
    os.system(f"rm -rf {e.DATA}")
    e.start(binary)
    e.http("GET", "/api/documents/index-status")  # creates the schema
    e.stop()
    digest = hashlib.sha256(TEXT.encode()).hexdigest()
    with e.db() as conn:
        conn.execute("INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at, source_sha256, text_sha256, source_text) "
                     "VALUES ('skipdoc', 'skip.txt', 'txt', ?, 1, 2, ?, ?, ?)", (len(TEXT), digest, digest, TEXT))
        cur = conn.execute("INSERT INTO document_chunks (doc_id, chunk_index, content, byte_start, byte_end, chunk_sha256) VALUES ('skipdoc', 0, ?, 0, ?, ?)",
                           (DRIFTED, len(TEXT), digest))
        conn.execute("INSERT INTO document_chunks_fts (rowid, content) VALUES (?, ?)", (cur.lastrowid, TEXT))
        chunk = cur.lastrowid
    e.start(binary)
    skipped = e.wait_indexed("skipdoc", timeout=WAIT_S)
    with e.db() as conn:
        conn.execute("UPDATE document_chunks SET content = ? WHERE id = ?", (TEXT, chunk))
    started, entry = time.time(), None
    while time.time() - started < WAIT_S:
        entry = e.coverage("skipdoc")[1]
        if entry["indexed_chunks"] == 1:
            break
        time.sleep(2)
    server = e.SERVERS[-1]
    e.stop()
    return {"build": server["build"], "binary": server["binary"], "after_drift": skipped,
            "after_restore": entry, "seconds_waited": round(time.time() - started, 1),
            "recovered": entry["indexed_chunks"] == 1 and entry["skipped_chunks"] == 0}


def main():
    old = scenario(os.environ["OLD_BIN"])
    new = scenario(os.environ["NEW_BIN"])
    result = {"wait_limit_s": WAIT_S, "before_fix": old, "with_fix": new,
              "causal": (not old["recovered"]) and new["recovered"]
              and old["after_drift"]["skipped_chunks"] == 1 and new["after_drift"]["skipped_chunks"] == 1}
    os.makedirs(os.environ["OUT"], exist_ok=True)
    with open(os.path.join(os.environ["OUT"], "skip-recovery-causal.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["causal"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        os.system(f"bash /tmp/f2h/stop.sh {e.DATA} >/dev/null 2>&1")
