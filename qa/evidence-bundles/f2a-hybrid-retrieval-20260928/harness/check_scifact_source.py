#!/usr/bin/env python3
"""Confirm the local scifact.zip is BEIR's published archive by hashing a fresh download."""
import hashlib
import urllib.request

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
LOCAL = "/mnt/disks/data/camelid-eval/scifact.zip"
with urllib.request.urlopen(URL, timeout=120) as res:
    fresh = hashlib.sha256(res.read()).hexdigest()
local = hashlib.sha256(open(LOCAL, "rb").read()).hexdigest()
print("fresh", fresh)
print("local", local)
print("MATCH" if fresh == local else "DIFFERENT")
