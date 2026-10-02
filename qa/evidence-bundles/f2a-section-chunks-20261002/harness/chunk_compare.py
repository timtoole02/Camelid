#!/usr/bin/env python3
"""Before/after summary of the section-chunking probe."""
import json

runs = {tag: json.load(open(f"/tmp/f2r/chunk-out/chunk-eval-{tag}.json")) for tag in ["before", "after"]}
print({tag: (r["build"], {k: len(v) if isinstance(v, list) else v for k, v in r["chunks"].items()}) for tag, r in runs.items()})
p0 = runs["before"]["probes"][0]
print("probe keys:", list(p0.keys()))
for i, probe in enumerate(runs["before"]["probes"]):
    after = runs["after"]["probes"][i]
    print("\nQ:", probe["question"])
    for tag, p in (("before", probe), ("after", after)):
        rest = {k: v for k, v in p.items() if k not in ("question", "top_by_meaning")}
        top = [(t[0][:10], t[1], t[2]) for t in p["top_by_meaning"][:3]]
        print(f"  {tag:6} top3={top}")
        print(f"         {json.dumps(rest)[:600]}")
