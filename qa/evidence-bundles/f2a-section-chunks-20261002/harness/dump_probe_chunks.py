#!/usr/bin/env python3
"""Dump every chunk of the probe libraries, before and after, as JSON.

usage: dump_probe_chunks.py <out.json>
"""
import json
import sqlite3
import sys

out = {}
for tag in ("before", "after"):
    conn = sqlite3.connect(f"/tmp/f2r/lib-chunk-probe-{tag}/documents_rag.sqlite3")
    rows = conn.execute(
        "SELECT d.filename, c.chunk_index, c.byte_start, c.byte_end, c.content "
        "FROM document_chunks c JOIN documents d ON d.id = c.doc_id ORDER BY d.filename, c.chunk_index").fetchall()
    docs = {}
    for filename, index, start, end, content in rows:
        docs.setdefault(filename, []).append({"index": index, "start": start, "end": end, "text": content})
    out[tag] = docs
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print({tag: {name: len(chunks) for name, chunks in docs.items()} for tag, docs in out.items()})
