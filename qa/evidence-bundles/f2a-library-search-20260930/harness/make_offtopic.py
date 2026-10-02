#!/usr/bin/env python3
"""Off-topic chat messages for the library-search relevance calibration, with provenance.

Sources: the cached Hugging Face snapshots of MT-Bench prompts (all first turns),
Alpaca (seeded sample of 200) and GSM8K test (seeded sample of 100), plus 25 short
chit-chat messages written for this calibration (source "author-written").

usage: make_offtopic.py <out.jsonl>
"""
import glob
import json
import random
import sys

import pandas as pd

SEED = 20260930
HUB = "/mnt/disks/data/hf/hub"
CHITCHAT = [
    "hello", "hi there", "thanks!", "thank you, that helps", "good morning", "how are you?",
    "what can you do?", "who are you?", "ok", "cool", "tell me a joke", "can you help me?",
    "what's your name?", "bye", "nice, thanks", "please continue", "can you make that shorter?",
    "summarize our conversation so far", "translate that into French", "what time is it?",
    "I'm bored", "good night", "sounds good", "let's start over", "explain that again more simply",
]


def parquet(pattern):
    (path,) = glob.glob(f"{HUB}/{pattern}")
    return path, pd.read_parquet(path)


rows = []
path, mt = parquet("datasets--HuggingFaceH4--mt_bench_prompts/snapshots/*/data/*.parquet")
for i, r in mt.iterrows():
    rows.append({"source": "mt_bench", "row": int(i), "category": r["category"], "text": list(r["prompt"])[0]})
path, alpaca = parquet("datasets--tatsu-lab--alpaca/snapshots/*/data/*.parquet")
for i in sorted(random.Random(SEED).sample(range(len(alpaca)), 200)):
    r = alpaca.iloc[i]
    text = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
    rows.append({"source": "alpaca", "row": int(i), "text": text})
path, gsm = parquet("datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")
for i in sorted(random.Random(SEED).sample(range(len(gsm)), 100)):
    rows.append({"source": "gsm8k", "row": int(i), "text": gsm.iloc[i]["question"]})
for i, text in enumerate(CHITCHAT):
    rows.append({"source": "author-written", "row": i, "text": text})

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
counts = {}
for r in rows:
    counts[r["source"]] = counts.get(r["source"], 0) + 1
print(counts)
