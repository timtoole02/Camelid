#!/usr/bin/env python3
"""Does keyword mode rank exactly as search did before this change?

Serves the SciFact library with the previous build (which has no `mode` and no vectors), runs the same
300 test claims, collapses chunks to documents the same way as eval_scifact.py, and compares with the
keyword-mode rankings in per-query.json. Also records the previous build's latency.

usage: base_keyword_compare.py   (env: CAMELID_BASE, SCIFACT_DIR, PER_QUERY, OUT)
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_scifact as ev  # noqa: E402

PER_QUERY = os.environ["PER_QUERY"]
OUT = os.environ["OUT"]


def main():
    expected = json.load(open(PER_QUERY, encoding="utf-8"))
    queries = {q["_id"]: q["text"] for q in ev.jsonl(os.path.join(ev.SCIFACT, "queries.jsonl"))}
    status, health, _ = ev.http("GET", "/v1/health")
    same, differing, latency = 0, [], []
    for qid in sorted(expected, key=int):
        status, body, ms = ev.http("POST", "/api/documents/search", {"query": queries[qid], "top_k": ev.TOP_K})
        assert status == 200, (qid, status, body)
        assert "retrieval" not in body, "this is not the previous build"
        latency.append(ms)
        ranked = []
        for result in body["results"]:
            corpus_id = result["doc_id"].removeprefix("scifact-")
            if corpus_id not in ranked:
                ranked.append(corpus_id)
        if ranked[:ev.K] == expected[qid]["keyword"]["ranked"]:
            same += 1
        else:
            differing.append({"query": qid, "previous": ranked[:ev.K], "keyword_mode": expected[qid]["keyword"]["ranked"]})
    result = {
        "previous_build": health.get("build"),
        "queries": len(expected),
        "identical_top10_documents": same,
        "differing": differing,
        "previous_build_latency_ms": {"p50": round(statistics.median(latency), 1),
                                      "p95": round(sorted(latency)[int(0.95 * (len(latency) - 1))], 1)},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "keyword-vs-previous-build.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "differing"}, indent=2), "differing:", len(differing))


if __name__ == "__main__":
    main()
