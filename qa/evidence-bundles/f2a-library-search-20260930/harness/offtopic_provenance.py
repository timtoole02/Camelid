#!/usr/bin/env python3
"""Write the off-topic set's provenance: source, row and the SHA-256 of each text.

The third-party texts (MT-Bench, Alpaca, GSM8K) are not copied into the bundle;
make_offtopic.py regenerates them, and each hash here identifies the exact text.
The author-written messages are kept in full.

usage: offtopic_provenance.py <offtopic.jsonl> <out.jsonl>
"""
import hashlib
import json
import sys

with open(sys.argv[2], "w", encoding="utf-8") as out:
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        row = {"source": r["source"], "row": r["row"], "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()}
        if r["source"] == "author-written":
            row["text"] = r["text"]
        out.write(json.dumps(row) + "\n")
