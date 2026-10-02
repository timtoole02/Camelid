#!/usr/bin/env python3
"""Choose the library-search relevance floor from collect_similarity.py's output,
by a rule fixed before the data was collected:

1. Per query: `top` is the best cosine among its 20 whole-library results (0.0
   when there are none); `gold` is the best cosine among results from one of its
   relevant documents, or none when no relevant document is in the 20.
2. Split: within each (library, set, source) group, queries are sorted by id,
   shuffled with SEED, and the first half (rounded up) calibrates; the rest is
   held out and only ever reported.
3. At a floor t, a positive is served when gold >= t (a positive with no gold in
   its 20 is never served), and a negative stays quiet when top < t.
4. t runs from 0.300 to 0.900 in steps of 0.005. The floor is the t with the
   highest TPR + TNR - 1 on the pooled calibration halves of both libraries;
   ties go to the smallest t.

usage: analyze_floor.py <calibration dir>   (reads scifact.jsonl, fiqa.jsonl, offtopic.jsonl; writes floor.json)
"""
import json
import math
import os
import random
import sys

SEED = 20260930
GRID = [round(0.300 + 0.005 * i, 3) for i in range(121)]
DIR = sys.argv[1]


def jsonl(name):
    with open(os.path.join(DIR, name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


rows = []
for library in ("scifact", "fiqa"):
    for r in jsonl(f"{library}.jsonl"):
        cosines = [c for _, _, c in r["results"]]
        gold = [c for d, _, c in r["results"] if d in set(r["gold"])]
        rows.append({"library": library, "set": r["set"], "source": r["source"], "id": r["id"],
                     "top": max(cosines) if cosines else 0.0, "gold": max(gold) if gold else None})

groups = {}
for r in rows:
    groups.setdefault((r["library"], r["set"], r["source"]), []).append(r)
for key, members in groups.items():
    members.sort(key=lambda r: r["id"])
    random.Random(SEED).shuffle(members)
    cut = math.ceil(len(members) / 2)
    for i, r in enumerate(members):
        r["half"] = "calibration" if i < cut else "held_out"


def served(r, t):
    return r["gold"] is not None and r["gold"] >= t


def quiet(r, t):
    return r["top"] < t


def rates(subset, t):
    pos = [r for r in subset if r["set"] == "positive"]
    neg = [r for r in subset if r["set"] == "negative"]
    tpr = sum(served(r, t) for r in pos) / len(pos) if pos else None
    tnr = sum(quiet(r, t) for r in neg) / len(neg) if neg else None
    return {"positives": len(pos), "negatives": len(neg),
            "tpr": None if tpr is None else round(tpr, 4), "tnr": None if tnr is None else round(tnr, 4)}


calibration = [r for r in rows if r["half"] == "calibration"]
held_out = [r for r in rows if r["half"] == "held_out"]
best_t, best_j = None, -2.0
curve = []
for t in GRID:
    m = rates(calibration, t)
    j = m["tpr"] + m["tnr"] - 1
    curve.append({"floor": t, **m, "j": round(j, 4)})
    if j > best_j + 1e-12:
        best_t, best_j = t, j

report = {
    "rule": __doc__.split("usage:")[0].strip(),
    "seed": SEED,
    "floor": best_t,
    "calibration": {**rates(calibration, best_t), "j": round(best_j, 4)},
    "held_out": rates(held_out, best_t),
    "held_out_by_library": {lib: rates([r for r in held_out if r["library"] == lib], best_t) for lib in ("scifact", "fiqa")},
    "held_out_negatives_by_source": {},
    "ceiling": {lib: round(sum(r["gold"] is not None for r in rows if r["library"] == lib and r["set"] == "positive")
                           / sum(1 for r in rows if r["library"] == lib and r["set"] == "positive"), 4)
                for lib in ("scifact", "fiqa")},
    "curve": curve,
}
for lib in ("scifact", "fiqa"):
    for src in sorted({r["source"] for r in held_out if r["library"] == lib and r["set"] == "negative"}):
        sub = [r for r in held_out if r["library"] == lib and r["set"] == "negative" and r["source"] == src]
        report["held_out_negatives_by_source"][f"{lib}/{src}"] = {"n": len(sub), "quiet": sum(quiet(r, best_t) for r in sub)}
loudest = sorted((r for r in held_out if r["set"] == "negative" and not quiet(r, best_t)), key=lambda r: -r["top"])
report["held_out_negatives_above_floor"] = [{"library": r["library"], "source": r["source"], "id": r["id"], "top": round(r["top"], 4)} for r in loudest]
json.dump(report, open(os.path.join(DIR, "floor.json"), "w"), indent=1)
print(json.dumps({k: report[k] for k in ("floor", "calibration", "held_out", "held_out_by_library", "held_out_negatives_by_source", "ceiling")}, indent=1))
print("held-out negatives above the floor:", len(loudest))
