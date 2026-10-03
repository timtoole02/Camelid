#!/usr/bin/env python3
"""Collect the evidence for the library-search relevance floor.

Every query is sent to POST /api/documents/search over the whole library in
`semantic` mode with top_k 20, where each result's score is its cosine
similarity to the query. Nothing here chooses the floor; analyze_floor.py does,
by the rule written down there before any of these files existed.

usage: collect_similarity.py <library name> <positives source> (env: CAMELID_BASE, OUT_DIR)
  library name: scifact | fiqa
Writes OUT_DIR/<library>.jsonl, one line per query:
  {"set": "positive"|"negative", "source", "id", "gold": [...doc ids...], "results": [[doc_id, chunk_index, cosine], ...]}
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("CAMELID_BASE", "http://127.0.0.1:8195")
OUT = os.environ.get("OUT_DIR", "/mnt/disks/data/camelid-eval/calibration")
EVAL = "/mnt/disks/data/camelid-eval"
TOP_K = 20


def http(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            return res.status, json.loads(res.read() or b"null")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null")


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def qrels(path, prefix):
    rel = {}
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                rel.setdefault(qid, []).append(f"{prefix}{did}")
    return rel


def scifact_queries():
    rel = qrels(f"{EVAL}/scifact/qrels/test.tsv", "scifact-")
    text = {q["_id"]: q["text"] for q in jsonl(f"{EVAL}/scifact/queries.jsonl")}
    return [{"source": "scifact", "id": qid, "text": text[qid], "gold": rel[qid]} for qid in sorted(rel, key=int)]


def fiqa_queries():
    chosen = json.load(open(f"{EVAL}/fiqa-1000/subset.json"))["queries"]
    rel = qrels(f"{EVAL}/fiqa-src/fiqa/qrels/test.tsv", "fiqa-")
    text = {q["_id"]: q["text"] for q in jsonl(f"{EVAL}/fiqa-src/fiqa/queries.jsonl")}
    return [{"source": "fiqa", "id": qid, "text": text[qid], "gold": rel[qid]} for qid in sorted(chosen, key=int)]


def offtopic():
    return [{"source": r["source"], "id": f"{r['source']}-{r['row']}", "text": r["text"], "gold": []}
            for r in jsonl(f"{OUT}/offtopic.jsonl")]


def main():
    library = sys.argv[1]
    _, health = http("GET", "/v1/health")
    _, status = http("GET", "/api/documents/index-status")
    docs = status["documents"]
    indexable = sum(d["indexable_chunks"] for d in docs)
    indexed = sum(d["indexed_chunks"] for d in docs)
    assert status["semantic"]["available"] and indexed == indexable, (status["semantic"], indexed, indexable)
    own, other = (scifact_queries(), fiqa_queries()) if library == "scifact" else (fiqa_queries(), scifact_queries())
    work = [("positive", q) for q in own] + [("negative", {**q, "gold": []}) for q in other] + [("negative", q) for q in offtopic()]
    out_path = os.path.join(OUT, f"{library}.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for n, (kind, q) in enumerate(work):
            code, body = http("POST", "/api/documents/search", {"query": q["text"], "top_k": TOP_K, "mode": "semantic"})
            assert code == 200 and body["retrieval"]["mode"] in ("semantic", "none"), (q["id"], code, body)
            fh.write(json.dumps({"set": kind, "source": q["source"], "id": q["id"], "gold": q["gold"],
                                 "results": [[r["doc_id"], r["chunk_index"], r["score"]] for r in body["results"]]}) + "\n")
            if n % 200 == 0:
                print(f"{library}: {n + 1}/{len(work)}", flush=True)
    json.dump({"library": library, "build": health.get("build"), "documents": len(docs), "indexed_chunks": indexed, "queries": len(work)},
              open(os.path.join(OUT, f"{library}-meta.json"), "w"), indent=1)
    print(f"{library}: wrote {len(work)} queries")


if __name__ == "__main__":
    main()
