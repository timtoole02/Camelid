#!/usr/bin/env python3
"""Fetch BEIR FiQA-2018 for the library-search relevance calibration and record its provenance.

usage: fetch_fiqa.py <dest dir>
"""
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
dest = sys.argv[1]
os.makedirs(dest, exist_ok=True)
archive = os.path.join(dest, "fiqa.zip")
if not os.path.exists(archive):
    urllib.request.urlretrieve(URL, archive)
digest = hashlib.sha256(open(archive, "rb").read()).hexdigest()
with zipfile.ZipFile(archive) as zf:
    zf.extractall(dest)
root = os.path.join(dest, "fiqa")
counts = {}
for name in ("corpus.jsonl", "queries.jsonl"):
    with open(os.path.join(root, name), encoding="utf-8") as fh:
        counts[name] = sum(1 for line in fh if line.strip())
with open(os.path.join(root, "qrels", "test.tsv"), encoding="utf-8") as fh:
    counts["qrels/test.tsv"] = sum(1 for _ in fh) - 1
json.dump({"url": URL, "sha256": digest, "bytes": os.path.getsize(archive), "counts": counts},
          open(os.path.join(dest, "fiqa-source.json"), "w"), indent=2)
print(json.dumps({"sha256": digest, "counts": counts}))
