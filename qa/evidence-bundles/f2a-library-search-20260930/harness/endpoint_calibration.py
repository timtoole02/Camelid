#!/usr/bin/env python3
"""Replay every calibration query through the shipped endpoint with `library: true`.

For each library (a fresh copy of the one calibration used), every query is sent
twice: with `library: true, top_k: 20`, and as a plain whole-library search with
`top_k: 20` for comparison. It checks that no returned passage is below the floor,
that the endpoint returns something exactly when the calibration's best cosine
reached the floor, and reports what the chat itself uses: the first 4 passages. Each library's floor is the one the
server reports for its indexed chunks.

usage: endpoint_calibration.py   (env: BIN, OUT)
"""
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/tmp/f2l")
import collect_similarity as queries  # noqa: E402  (the calibration's own query builders)

PORT = 8197
BASE = f"http://127.0.0.1:{PORT}"
BIN = os.environ["BIN"]
OUT = os.environ["OUT"]
EVAL = "/mnt/disks/data/camelid-eval"
CALIB = f"{EVAL}/calibration"
SEED = 20260930  # analyze_floor.py's split
CHAT_TOP_K = 4


def http(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/api/documents/search", data=data, method="POST", headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            status, body = res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        status, body = err.code, json.loads(err.read() or b"null")
    return status, body, (time.perf_counter() - started) * 1000


def pct(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))], 1)


def halves(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["library"], r["set"], r["source"]), []).append(r)
    for members in groups.values():
        members.sort(key=lambda r: r["id"])
        random.Random(SEED).shuffle(members)
        cut = math.ceil(len(members) / 2)
        for i, r in enumerate(members):
            r["half"] = "calibration" if i < cut else "held_out"


def run_library(library):
    own, other = (queries.scifact_queries(), queries.fiqa_queries()) if library == "scifact" else (queries.fiqa_queries(), queries.scifact_queries())
    work = [("positive", q) for q in own] + [("negative", {**q, "gold": []}) for q in other] + [("negative", q) for q in queries.offtopic()]
    calibration = {(r["set"], r["source"], r["id"]): r for r in map(json.loads, open(f"{CALIB}/{library}.jsonl", encoding="utf-8"))}
    src, copy = f"{EVAL}/lib-{library}-1000", f"{EVAL}/lib-{library}-endpoint"
    shutil.rmtree(copy, ignore_errors=True)
    os.makedirs(copy)
    shutil.copy2(f"{src}/documents_rag.sqlite3", copy)
    subprocess.run(["bash", "/tmp/f2h/serve.sh", BIN, copy, str(PORT)], check=True)
    rows, lat_library, lat_plain, floors = [], [], [], set()
    try:
        for n, (kind, q) in enumerate(work):
            status, body, ms = http({"query": q["text"], "top_k": 20, "library": True})
            assert status == 200, (q["id"], status, body)
            lat_library.append(ms)
            floor = body["retrieval"]["relevance_floor"]
            floors.add(floor)
            status_plain, _, ms_plain = http({"query": q["text"], "top_k": 20})
            assert status_plain == 200, (q["id"], status_plain)
            lat_plain.append(ms_plain)
            results = body["results"]
            cal = calibration[(kind, q["source"], q["id"])]
            cal_top = max((c for _, _, c in cal["results"]), default=0.0)
            gold = set(q["gold"])
            rows.append({
                "library": library, "set": kind, "source": q["source"], "id": q["id"],
                "returned": len(results),
                "below_floor": sum(1 for r in results if r.get("similarity") is None or r["similarity"] < floor),
                "calibration_top": round(cal_top, 6),
                "agrees": bool(results) == (cal_top >= floor),
                "gold_in_chat_passages": any(r["doc_id"] in gold for r in results[:CHAT_TOP_K]),
                "chat_passages": min(len(results), CHAT_TOP_K),
            })
            if n % 200 == 0:
                print(f"{library}: {n + 1}/{len(work)}", flush=True)
    finally:
        subprocess.run(["bash", "/tmp/f2h/stop.sh", copy], check=True)
    assert len(floors) == 1, floors
    return rows, lat_library, lat_plain, floors.pop()


def rates(rows):
    pos = [r for r in rows if r["set"] == "positive"]
    neg = [r for r in rows if r["set"] == "negative"]
    return {
        "positives": len(pos),
        "positives_with_a_relevant_passage_in_the_4": sum(r["gold_in_chat_passages"] for r in pos),
        "negatives": len(neg),
        "negatives_with_no_passage": sum(r["returned"] == 0 for r in neg),
        "negatives_mean_passages": round(statistics.fmean(r["chat_passages"] for r in neg), 3) if neg else None,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    build = subprocess.check_output([BIN, "--version"], text=True).strip()
    all_rows, latency, floors = [], {}, {}
    for library in ("scifact", "fiqa"):
        rows, lat_library, lat_plain, floors[library] = run_library(library)
        all_rows += rows
        latency[library] = {
            "library": {"p50_ms": pct(lat_library, 50), "p95_ms": pct(lat_library, 95)},
            "plain_whole_library": {"p50_ms": pct(lat_plain, 50), "p95_ms": pct(lat_plain, 95)},
            "n": len(lat_library),
        }
    halves(all_rows)
    held_out = [r for r in all_rows if r["half"] == "held_out"]
    report = {
        "build": build,
        "floors": floors,
        "queries": len(all_rows),
        "results_below_the_floor": sum(r["below_floor"] for r in all_rows),
        "endpoint_agrees_with_calibration": sum(r["agrees"] for r in all_rows),
        "disagreements": [r for r in all_rows if not r["agrees"]][:20],
        "chat_top_k": CHAT_TOP_K,
        "held_out": rates(held_out),
        "held_out_by_library": {lib: rates([r for r in held_out if r["library"] == lib]) for lib in ("scifact", "fiqa")},
        "latency": latency,
        "load_at_end": open("/proc/loadavg").read().split()[:3],
    }
    with open(os.path.join(OUT, "endpoint-calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(OUT, "endpoint-per-query.jsonl"), "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=1))
    ok = report["results_below_the_floor"] == 0 and report["endpoint_agrees_with_calibration"] == report["queries"]
    print("ENDPOINT", "OK" if ok else "DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
