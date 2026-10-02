#!/usr/bin/env python3
"""A 500-document collection on a copy of the indexed 1,000-document SciFact library.

usage: scale_collections.py   (env: BIN, SRC_LIB, LIB, OUT, SCIFACT_DIR)
Copies SRC_LIB to LIB (the source is never written), serves LIB on port 8193 with
BIN, collects every other document (by id) into one collection, and runs the 300
SciFact test claims three ways: scoped by the collection, scoped by the same ids
as doc_ids, and over the whole library.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 8193
BASE = f"http://127.0.0.1:{PORT}"
BIN = os.environ["BIN"]
SRC_LIB = os.environ["SRC_LIB"]
LIB = os.environ["LIB"]
OUT = os.environ["OUT"]
SCIFACT = os.environ["SCIFACT_DIR"]
K = 10
TOP_K = 20


def http(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            status, raw = res.status, res.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    return status, (json.loads(raw) if raw else None), (time.perf_counter() - started) * 1000


def pct(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))], 1)


def summary(values):
    return {"n": len(values), "p50_ms": pct(values, 50), "p95_ms": pct(values, 95), "max_ms": round(max(values), 1),
            "mean_ms": round(statistics.fmean(values), 1)}


def main():
    assert not os.path.exists(os.path.join(LIB, "server.pid")), "a server may be using the copy"
    shutil.rmtree(LIB, ignore_errors=True)
    shutil.copytree(SRC_LIB, LIB, ignore=shutil.ignore_patterns("server.pid", "server.log"))
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["bash", "/tmp/f2h/serve.sh", BIN, LIB, str(PORT)], check=True)
    try:
        return run()
    finally:
        subprocess.run(["bash", "/tmp/f2h/stop.sh", LIB], check=True)


def run():
    _, health, _ = http("GET", "/v1/health")
    _, status, _ = http("GET", "/api/documents/index-status")
    docs = status["documents"]
    indexable = sum(d["indexable_chunks"] for d in docs)
    indexed = sum(d["indexed_chunks"] for d in docs)
    assert status["semantic"]["available"] and indexed == indexable, (status["semantic"], indexed, indexable)

    ids = sorted(d["id"] for d in docs)
    members = ids[::2]
    code, collection, _ = http("POST", "/api/collections", {"name": "SciFact half"})
    assert code == 201, (code, collection)
    code, added, add_ms = http("POST", f"/api/collections/{collection['id']}/documents", {"doc_ids": members})
    assert code == 200 and added["doc_ids"] == members, code
    list_ms = [http("GET", "/api/collections")[2] for _ in range(20)]
    member_set = set(members)

    queries = {}
    with open(os.path.join(SCIFACT, "queries.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                q = json.loads(line)
                queries[q["_id"]] = q["text"]
    relevant = {}
    with open(os.path.join(SCIFACT, "qrels", "test.tsv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                relevant.setdefault(qid, set()).add(f"scifact-{did}")

    latency = {"collection": [], "doc_ids": [], "whole_library": [], "collection_keyword": []}
    violations, identical, modes = [], 0, {}
    in_scope = {"queries": 0, "collection_recall10": [], "whole_recall10": []}
    for qid in sorted(relevant, key=int):
        query = queries[qid]
        code, scoped, ms = http("POST", "/api/documents/search", {"query": query, "top_k": TOP_K, "mode": "hybrid", "collection_ids": [collection["id"]]})
        assert code == 200, (qid, code, scoped)
        latency["collection"].append(ms)
        modes[scoped["retrieval"]["mode"]] = modes.get(scoped["retrieval"]["mode"], 0) + 1
        code, by_ids, ms = http("POST", "/api/documents/search", {"query": query, "top_k": TOP_K, "mode": "hybrid", "doc_ids": members})
        assert code == 200, (qid, code)
        latency["doc_ids"].append(ms)
        code, whole, ms = http("POST", "/api/documents/search", {"query": query, "top_k": TOP_K, "mode": "hybrid"})
        assert code == 200, (qid, code)
        latency["whole_library"].append(ms)
        code, keyword, ms = http("POST", "/api/documents/search", {"query": query, "top_k": TOP_K, "mode": "keyword", "collection_ids": [collection["id"]]})
        assert code == 200, (qid, code)
        latency["collection_keyword"].append(ms)

        outside = [r["doc_id"] for r in scoped["results"] + keyword["results"] if r["doc_id"] not in member_set]
        if outside:
            violations.append({"query": qid, "outside": outside})
        if [(r["doc_id"], r["chunk_index"]) for r in scoped["results"]] == [(r["doc_id"], r["chunk_index"]) for r in by_ids["results"]]:
            identical += 1

        gold = relevant[qid]
        if gold <= member_set:
            def ranked(body):
                seen = []
                for r in body["results"]:
                    if r["doc_id"] not in seen:
                        seen.append(r["doc_id"])
                return seen[:K]
            in_scope["queries"] += 1
            in_scope["collection_recall10"].append(len(set(ranked(scoped)) & gold) / len(gold))
            in_scope["whole_recall10"].append(len(set(ranked(whole)) & gold) / len(gold))

    report = {
        "build": health.get("build"),
        "library": {"documents": len(ids), "indexable_chunks": indexable, "indexed_chunks": indexed},
        "collection": {"members": len(members), "selection": "every other document id, sorted", "add_request_ms": round(add_ms, 1),
                       "list_collections": summary(list_ms)},
        "queries": len(relevant),
        "retrieval_modes_seen": modes,
        "results_outside_the_collection": len(violations),
        "violations": violations[:10],
        "collection_equals_doc_ids_ranking": identical,
        "gold_inside_collection": {
            "queries": in_scope["queries"],
            "recall10_collection": round(statistics.fmean(in_scope["collection_recall10"]), 4) if in_scope["queries"] else None,
            "recall10_whole_library": round(statistics.fmean(in_scope["whole_recall10"]), 4) if in_scope["queries"] else None,
        },
        "latency": {name: summary(values) for name, values in latency.items()},
        "load_at_end": open("/proc/loadavg").read().split()[:3],
    }
    with open(os.path.join(OUT, "scale-collections.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    ok = not violations and identical == len(relevant)
    print("SCALE", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
