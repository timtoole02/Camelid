#!/usr/bin/env python3
"""Knowledge Library retrieval on BEIR SciFact through the live HTTP API.

Ingests every SciFact abstract as one document, waits for the semantic index,
then runs the 300 test claims in keyword, semantic and hybrid mode and scores
each against the official test qrels (nDCG@10, Recall@10, MRR@10).

usage: eval_scifact.py ingest|index|search|verify
       (env: CAMELID_BASE, SCIFACT_DIR, SCIFACT_CORPUS, OUT_DIR)
"""
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("CAMELID_BASE", "http://127.0.0.1:8190")
SCIFACT = os.environ.get("SCIFACT_DIR", "scifact")
CORPUS = os.environ.get("SCIFACT_CORPUS", "corpus.jsonl")
OUT = os.environ.get("OUT_DIR", "out")
MODES = ("keyword", "semantic", "hybrid")
K = 10
TOP_K = 20  # the API maximum; chunks are deduplicated to documents before scoring


def http(method, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            status, raw = res.status, res.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    elapsed = (time.perf_counter() - started) * 1000
    return status, (json.loads(raw) if raw else None), elapsed


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def qrels():
    rel = {}
    with open(os.path.join(SCIFACT, "qrels", "test.tsv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                rel.setdefault(qid, {})[did] = int(score)
    return rel


def doc_id(corpus_id):
    return f"scifact-{corpus_id}"


def ingest():
    corpus_path = os.path.join(SCIFACT, CORPUS)
    corpus = jsonl(corpus_path)
    with open(corpus_path, "rb") as fh:
        corpus_sha256 = hashlib.sha256(fh.read()).hexdigest()
    started, chunks, failures = time.perf_counter(), 0, []
    for index, doc in enumerate(corpus):
        text = f"{doc.get('title', '').strip()}\n\n{doc.get('text', '').strip()}".strip()
        status, body, _ = http("POST", "/api/documents/ingest", {
            "doc_id": doc_id(doc["_id"]), "filename": f"{doc['_id']}.txt", "content": text,
        })
        if status != 200:
            failures.append({"id": doc["_id"], "status": status, "body": body})
        else:
            chunks += body["chunk_count"]
        if index % 500 == 0:
            print(f"ingested {index + 1}/{len(corpus)}", flush=True)
    summary = {"corpus": CORPUS, "corpus_sha256": corpus_sha256, "documents": len(corpus), "chunks": chunks,
               "failures": failures, "seconds": round(time.perf_counter() - started, 1)}
    write("ingest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), "failures:", len(failures))


def index():
    started, samples = time.perf_counter(), []
    while True:
        status, body, _ = http("GET", "/api/documents/index-status")
        assert status == 200, (status, body)
        docs = body["documents"]
        indexable = sum(d["indexable_chunks"] for d in docs)
        indexed = sum(d["indexed_chunks"] for d in docs)
        skipped = sum(d["skipped_chunks"] for d in docs)
        elapsed = time.perf_counter() - started
        samples.append((round(elapsed, 1), indexed))
        print(f"{elapsed:7.0f}s indexed {indexed}/{indexable} skipped {skipped} indexing={body['semantic']['indexing']}", flush=True)
        if indexed + skipped >= indexable or not body["semantic"]["indexing"]:
            break
        time.sleep(15)
    write("index.json", {"semantic": body["semantic"], "indexable_chunks": indexable, "indexed_chunks": indexed,
                         "skipped_chunks": skipped, "wall_seconds_observed": samples[-1][0], "samples": samples})


def ndcg(ranked, relevant):
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:K]))
    ideal = sorted(relevant.values(), reverse=True)[:K]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def search():
    queries = {q["_id"]: q["text"] for q in jsonl(os.path.join(SCIFACT, "queries.jsonl"))}
    rel = qrels()
    per_query, latency, modes_seen = {}, {m: [] for m in MODES}, {m: {} for m in MODES}
    for n, qid in enumerate(sorted(rel, key=int)):
        per_query[qid] = {}
        for mode in MODES:
            status, body, ms = http("POST", "/api/documents/search", {"query": queries[qid], "top_k": TOP_K, "mode": mode})
            assert status == 200, (qid, mode, status, body)
            latency[mode].append(ms)
            ran = body["retrieval"]["mode"]
            modes_seen[mode][ran] = modes_seen[mode].get(ran, 0) + 1
            ranked = []
            for result in body["results"]:
                corpus_id = result["doc_id"].removeprefix("scifact-")
                if corpus_id not in ranked:
                    ranked.append(corpus_id)
            relevant = rel[qid]
            first = next((i + 1 for i, d in enumerate(ranked[:K]) if d in relevant), None)
            per_query[qid][mode] = {
                "ndcg10": ndcg(ranked, relevant),
                "recall10": len(set(ranked[:K]) & set(relevant)) / len(relevant),
                "mrr10": 1 / first if first else 0.0,
                "ranked": ranked[:K],
            }
        if n % 50 == 0:
            print(f"query {n + 1}/{len(rel)}", flush=True)

    def mean(mode, metric):
        return round(statistics.fmean(q[mode][metric] for q in per_query.values()), 4)

    summary = {
        "queries": len(per_query),
        "metrics": {m: {x: mean(m, x) for x in ("ndcg10", "recall10", "mrr10")} for m in MODES},
        "latency_ms": {m: {"p50": round(statistics.median(v), 1), "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 1)} for m, v in latency.items()},
        "ranked_by": modes_seen,
        "hybrid_vs_keyword_ndcg10": {
            "better": sum(q["hybrid"]["ndcg10"] > q["keyword"]["ndcg10"] for q in per_query.values()),
            "worse": sum(q["hybrid"]["ndcg10"] < q["keyword"]["ndcg10"] for q in per_query.values()),
            "same": sum(q["hybrid"]["ndcg10"] == q["keyword"]["ndcg10"] for q in per_query.values()),
        },
    }
    write("search.json", summary)
    write("per-query.json", per_query)
    print(json.dumps(summary, indent=2))


def verify():
    """Re-resolve every hybrid result of the first 50 test queries through the citation endpoint."""
    queries = {q["_id"]: q["text"] for q in jsonl(os.path.join(SCIFACT, "queries.jsonl"))}
    checked, failures = 0, []
    for qid in sorted(qrels(), key=int)[:50]:
        status, body, _ = http("POST", "/api/documents/search", {"query": queries[qid], "top_k": TOP_K, "mode": "hybrid"})
        for result in body["results"]:
            s, resolved, _ = http("POST", "/api/documents/citation/resolve", {
                "doc_id": result["doc_id"], "chunk_index": result["chunk_index"],
                "chunk_sha256": result["chunk_sha256"], "doc_sha256": result["doc_sha256"]})
            checked += 1
            if s != 200 or resolved["span"] != result["excerpt"]:
                failures.append({"query": qid, "doc": result["doc_id"], "status": s})
    write("verify.json", {"checked": checked, "failures": failures})
    print(f"resolved {checked} hybrid results, {len(failures)} failures")


def write(name, value):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2)


if __name__ == "__main__":
    {"ingest": ingest, "index": index, "search": search, "verify": verify}[sys.argv[1]]()
