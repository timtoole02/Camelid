#!/usr/bin/env python3
"""Size a SciFact subset: every test-relevant abstract plus seeded random distractors."""
import json
import random
import sys

SEED = 20260929
size = int(sys.argv[1])
corpus = [json.loads(line) for line in open("scifact/corpus.jsonl", encoding="utf-8")]
relevant = set()
with open("scifact/qrels/test.tsv", encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        _, did, score = line.split("\t")
        if int(score) > 0:
            relevant.add(did)
lengths = [len(f"{d.get('title','')}\n\n{d.get('text','')}") for d in corpus]
print("corpus docs", len(corpus), "mean chars", round(sum(lengths) / len(lengths)), "relevant docs", len(relevant))
others = sorted(d["_id"] for d in corpus if d["_id"] not in relevant)
random.Random(SEED).shuffle(others)
chosen = set(relevant) | set(others[: max(0, size - len(relevant))])
subset = [d for d in corpus if d["_id"] in chosen]
chars = sum(len(f"{d.get('title','')}\n\n{d.get('text','')}") for d in subset)
print("subset docs", len(subset), "chars", chars, "approx chunks", round(chars / 448))
with open(f"scifact/corpus-subset-{size}.jsonl", "w", encoding="utf-8") as out:
    for d in subset:
        out.write(json.dumps(d) + "\n")
