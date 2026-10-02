#!/usr/bin/env python3
"""Choose a library-size-aware relevance floor, by a rule fixed before this script was run.

Why: the fixed floor (0.69) passed its held-out test on two 1,000-document libraries, then left out
the right passage for natural questions on a live 3-document library. The best chance match an
unrelated message finds grows with how many passages a library holds, so a floor fitted at one size
is too strict for smaller libraries. This script measures that and fits a floor that moves with size.

Rule (fixed before this script was first run):
1. Data: every calibration query's vector (query-vectors.npz, checked against the server's own
   scores by embed_queries.py) and every indexed chunk vector of the SciFact and FiQA libraries.
   Query sets and the calibration/held-out split are analyze_floor.py's: per library, its own test
   questions are positives, the other library's questions and the 405 off-topic messages negatives;
   within each (library, set, source) group, queries sorted by id, shuffled with SEED 20260930, the
   first half (rounded up) calibrates and the rest is held out.
2. Simulated libraries: for sizes D in 3, 10, 30, 100, 300 and 1000 documents, each query gets R = 5
   seeded samples (one at 1,000, the whole library). A positive's sample is its relevant documents
   plus seeded others up to max(D, number of relevant documents); a negative's is D seeded documents.
   N is the sample's number of indexed chunks, top its best cosine to the query, gold (positives) the
   best cosine among relevant documents' chunks.
3. At floor t, a positive is served when gold >= t and a negative stays quiet when top < t.
4. For each D, t*_D is the floor on the grid 0.300 to 0.900 (step 0.005) with the highest TPR + TNR
   - 1 over the calibration half, both libraries and all samples pooled; ties go to the smallest t.
5. floor(N) = a + b * ln(N), with a and b fitted by ordinary least squares of t*_D on the mean ln(N)
   of size D's calibration samples, rounded to 4 decimals, and clamped to [0.300, 0.900].
6. Only the held-out half is reported as evidence: per size and pooled, TPR and TNR of floor(N) at
   each sample's own N, beside the fixed 0.69 on the same samples.

usage: analyze_size_floor.py <calibration dir>   (writes size-floor.json)
"""
import hashlib
import json
import math
import os
import random
import sqlite3
import sys

import numpy as np

SEED = 20260930
SIZES = [3, 10, 30, 100, 300, 1000]
REPLICATES = 5
GRID = [round(0.300 + 0.005 * i, 3) for i in range(121)]
CLAMP = (0.300, 0.900)
FIXED = 0.69
DIR = sys.argv[1]
EVAL = os.path.dirname(DIR.rstrip("/"))


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def library_chunks(library):
    conn = sqlite3.connect(f"{EVAL}/lib-{library}-1000/documents_rag.sqlite3")
    rows = conn.execute(
        "SELECT c.doc_id, v.vector FROM document_chunk_vectors v JOIN document_chunks c ON c.id = v.chunk_id "
        "WHERE v.vector IS NOT NULL AND v.chunk_sha256 = c.chunk_sha256 ORDER BY c.id").fetchall()
    docs = sorted({doc for doc, _ in rows})
    doc_index = {doc: i for i, doc in enumerate(docs)}
    owner = np.array([doc_index[doc] for doc, _ in rows])
    matrix = np.stack([np.frombuffer(blob, dtype="<f4") for _, blob in rows]).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return docs, owner, matrix


def seeded(*parts):
    return random.Random(int(hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).hexdigest()[:16], 16))


store = np.load(os.path.join(DIR, "query-vectors.npz"))
qvec = {(s, i): v / np.linalg.norm(v) for s, i, v in zip(store["sources"], store["ids"], store["vectors"])}

rows = []
for library in ("scifact", "fiqa"):
    docs, owner, matrix = library_chunks(library)
    doc_set = set(docs)
    by_doc = {}
    for chunk, d in enumerate(owner):
        by_doc.setdefault(docs[d], []).append(chunk)
    for r in jsonl(os.path.join(DIR, f"{library}.jsonl")):
        cos = matrix @ qvec[(r["source"], r["id"])]
        gold = [d for d in r["gold"] if d in doc_set]
        others = [d for d in docs if d not in set(gold)]
        for size in SIZES:
            for rep in range(1 if size == 1000 else REPLICATES):
                if size == 1000:
                    sample = docs
                else:
                    extra = max(size, len(gold)) - len(gold) if r["set"] == "positive" else size
                    sample = (gold if r["set"] == "positive" else []) + seeded(library, r["source"], r["id"], size, rep).sample(others if r["set"] == "positive" else docs, extra)
                chunks = [c for d in sample for c in by_doc.get(d, [])]
                gold_chunks = [c for d in gold for c in by_doc.get(d, [])] if r["set"] == "positive" else []
                rows.append({"library": library, "set": r["set"], "source": r["source"], "id": r["id"], "size": size, "rep": rep,
                             "n": len(chunks), "top": float(cos[chunks].max()) if chunks else 0.0,
                             "gold": float(cos[gold_chunks].max()) if gold_chunks else None})

groups = {}
for r in {(r["library"], r["set"], r["source"], r["id"]): r for r in rows}.values():
    groups.setdefault((r["library"], r["set"], r["source"]), []).append(r)
half = {}
for key, members in groups.items():
    members.sort(key=lambda r: r["id"])
    random.Random(SEED).shuffle(members)
    cut = math.ceil(len(members) / 2)
    for i, r in enumerate(members):
        half[(r["library"], r["set"], r["source"], r["id"])] = "calibration" if i < cut else "held_out"
for r in rows:
    r["half"] = half[(r["library"], r["set"], r["source"], r["id"])]


def rates(subset, floor_of):
    pos = [r for r in subset if r["set"] == "positive"]
    neg = [r for r in subset if r["set"] == "negative"]
    tpr = sum(r["gold"] is not None and r["gold"] >= floor_of(r) for r in pos) / len(pos)
    tnr = sum(r["top"] < floor_of(r) for r in neg) / len(neg)
    return {"positives": len(pos), "negatives": len(neg), "tpr": round(tpr, 4), "tnr": round(tnr, 4)}


per_size = []
for size in SIZES:
    cal = [r for r in rows if r["half"] == "calibration" and r["size"] == size]
    best_t, best_j = None, -2.0
    for t in GRID:
        m = rates(cal, lambda r, t=t: t)
        j = m["tpr"] + m["tnr"] - 1
        if j > best_j + 1e-12:
            best_t, best_j = t, j
    per_size.append({"size": size, "mean_ln_n": sum(math.log(max(r["n"], 1)) for r in cal) / len(cal),
                     "mean_chunks": round(sum(r["n"] for r in cal) / len(cal), 1), "t_star": best_t, "j": round(best_j, 4)})

x = np.array([p["mean_ln_n"] for p in per_size])
y = np.array([p["t_star"] for p in per_size])
b, a = np.polyfit(x, y, 1)
a, b = round(float(a), 4), round(float(b), 4)


def fitted(n):
    return min(CLAMP[1], max(CLAMP[0], a + b * math.log(max(n, 1))))


held = [r for r in rows if r["half"] == "held_out"]
report = {
    "rule": __doc__.split("usage:")[0].strip(),
    "seed": SEED,
    "a": a, "b": b, "clamp": CLAMP,
    "floor_at": {n: round(fitted(n), 4) for n in (5, 20, 50, 200, 1000, 2649, 4406, 20000, 100000)},
    "calibration_per_size": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()} for p in per_size],
    "held_out": {
        "size_aware": rates(held, lambda r: fitted(r["n"])),
        "fixed_0_69": rates(held, lambda r: FIXED),
        "by_size": [{"size": s, "size_aware": rates([r for r in held if r["size"] == s], lambda r: fitted(r["n"])),
                     "fixed_0_69": rates([r for r in held if r["size"] == s], lambda r: FIXED)} for s in SIZES],
        "by_library_at_1000": {lib: {"size_aware": rates([r for r in held if r["size"] == 1000 and r["library"] == lib], lambda r: fitted(r["n"])),
                                     "fixed_0_69": rates([r for r in held if r["size"] == 1000 and r["library"] == lib], lambda r: FIXED)}
                               for lib in ("scifact", "fiqa")},
    },
    "samples": len(rows),
}
json.dump(report, open(os.path.join(DIR, "size-floor.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "rule"}, indent=1))
