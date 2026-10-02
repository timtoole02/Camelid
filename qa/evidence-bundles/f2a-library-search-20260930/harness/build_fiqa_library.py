#!/usr/bin/env python3
"""Build and ingest the FiQA calibration library: the relevant documents of a seeded
sample of test questions plus seeded random others, 1,000 documents in all.

usage: build_fiqa_library.py subset|ingest|index   (env: CAMELID_BASE, FIQA_DIR, OUT_DIR)
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

SEED = 20260930
QUERIES = 200
SIZE = 1000
BASE = os.environ.get("CAMELID_BASE", "http://127.0.0.1:8194")
FIQA = os.environ.get("FIQA_DIR", "/mnt/disks/data/camelid-eval/fiqa-src/fiqa")
OUT = os.environ.get("OUT_DIR", "/mnt/disks/data/camelid-eval/fiqa-1000")


def http(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            return res.status, json.loads(res.read() or b"null")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null")


def test_qrels():
    rel = {}
    with open(os.path.join(FIQA, "qrels", "test.tsv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                rel.setdefault(qid, set()).add(did)
    return rel


def subset():
    os.makedirs(OUT, exist_ok=True)
    rel = test_qrels()
    qids = sorted(rel)
    random.Random(SEED).shuffle(qids)
    chosen_q = sorted(qids[:QUERIES])
    relevant = set().union(*(rel[q] for q in chosen_q))
    corpus = [json.loads(line) for line in open(os.path.join(FIQA, "corpus.jsonl"), encoding="utf-8") if line.strip()]
    others = sorted(d["_id"] for d in corpus if d["_id"] not in relevant and (d.get("text") or "").strip())
    random.Random(SEED).shuffle(others)
    chosen = relevant | set(others[: SIZE - len(relevant)])
    docs = [d for d in corpus if d["_id"] in chosen]
    with open(os.path.join(OUT, "corpus-1000.jsonl"), "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    json.dump({"seed": SEED, "queries": chosen_q, "relevant_docs": len(relevant), "documents": len(docs)},
              open(os.path.join(OUT, "subset.json"), "w"), indent=1)
    print(f"{len(chosen_q)} queries, {len(relevant)} relevant docs, {len(docs)} documents")


def ingest():
    failures, chunks = [], 0
    docs = [json.loads(line) for line in open(os.path.join(OUT, "corpus-1000.jsonl"), encoding="utf-8")]
    for n, d in enumerate(docs):
        text = f"{d.get('title', '').strip()}\n\n{d.get('text', '').strip()}".strip()
        status, body = http("POST", "/api/documents/ingest", {"doc_id": f"fiqa-{d['_id']}", "filename": f"{d['_id']}.txt", "content": text})
        if status != 200:
            failures.append({"id": d["_id"], "status": status, "body": body})
        else:
            chunks += body["chunk_count"]
        if n % 250 == 0:
            print(f"ingested {n + 1}/{len(docs)}", flush=True)
    json.dump({"documents": len(docs), "chunks": chunks, "failures": failures}, open(os.path.join(OUT, "ingest.json"), "w"), indent=1)
    print(f"chunks {chunks}, failures {len(failures)}")


def index():
    started = time.time()
    while True:
        _, body = http("GET", "/api/documents/index-status")
        docs = body["documents"]
        indexable = sum(d["indexable_chunks"] for d in docs)
        done = sum(d["indexed_chunks"] + d["skipped_chunks"] for d in docs)
        print(f"{time.time() - started:6.0f}s {done}/{indexable}", flush=True)
        if done >= indexable:
            break
        time.sleep(20)
    json.dump({"indexable_chunks": indexable, "indexed_or_skipped": done, "seconds": round(time.time() - started)},
              open(os.path.join(OUT, "index.json"), "w"), indent=1)


{"subset": subset, "ingest": ingest, "index": index}[sys.argv[1]]()
