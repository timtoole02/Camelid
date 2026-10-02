#!/usr/bin/env python3
"""Write the dataset provenance for the SciFact evidence: archive and file hashes, the subset recipe, its ids."""
import hashlib
import json
import os
import sys

EVAL = "/mnt/disks/data/camelid-eval"
OUT = sys.argv[1]


def sha(path):
    with open(os.path.join(EVAL, path), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


subset = [json.loads(line) for line in open(os.path.join(EVAL, "scifact/corpus-subset-1000.jsonl"), encoding="utf-8")]
corpus_count = sum(1 for line in open(os.path.join(EVAL, "scifact/corpus.jsonl"), encoding="utf-8") if line.strip())
relevant = set()
queries = set()
with open(os.path.join(EVAL, "scifact/qrels/test.tsv"), encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        qid, did, score = line.rstrip("\n").split("\t")
        if int(score) > 0:
            relevant.add(did)
            queries.add(qid)
ids = [doc["_id"] for doc in subset]
meta = {
    "dataset": "BEIR SciFact",
    "archive": "scifact.zip",
    "archive_sha256": sha("scifact.zip"),
    "archive_matches_fresh_download_of": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "files_sha256": {name: sha(f"scifact/{name}") for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")},
    "corpus_documents": corpus_count,
    "test_queries": len(queries),
    "test_relevant_documents": len(relevant),
    "subset": {
        "recipe": "every test-relevant document, plus random others (Python random.Random(20260929).shuffle of the sorted remaining ids) up to 1,000",
        "documents": len(ids),
        "relevant_included": len(relevant & set(ids)),
        "file": "corpus-subset-1000.jsonl",
        "file_sha256": sha("scifact/corpus-subset-1000.jsonl"),
        "ids_file": "subset-ids.txt",
    },
    "document_text": "title + blank line + abstract, ingested as one document per abstract",
}
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "dataset.json"), "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
with open(os.path.join(OUT, "subset-ids.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(ids) + "\n")
print(json.dumps(meta, indent=2))
