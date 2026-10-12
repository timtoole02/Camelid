"""Sort the answers that state the decoy's amount (issue #800).

Each planted fact has a decoy: the same sentence about a neighbouring part
(south clarifier, secondary mirror, west turntable) with a different amount.
An answer that states the decoy amount either says which part it belongs to
(labelled: the reply names the decoy's part word) or presents it as an answer
to the question (misattributed: the part word is missing).

The part word is the word in the decoy sentence that is not in the fact
sentence. Usage: python3 classify.py <planted-facts.json> <result json>...
"""
import json
import re
import sys

words = lambda text: re.findall(r"[a-z]+", text.lower())
facts = {f["id"]: f for f in json.load(open(sys.argv[1]))}
part = {}
for fid, f in facts.items():
    extra = [w for w in words(f["decoy"]) if w not in words(f["fact"])]
    assert len(extra) == 1, (fid, extra)
    part[fid] = extra[0]

report = []
for path in sys.argv[2:]:
    run = json.load(open(path))
    for name, result in run["prompts"].items():
        labelled, misattributed = [], []
        for row in result["rows"]:
            if row["states_decoy"]:
                (labelled if part[row["id"]] in words(row["answer"]) else misattributed).append(row["id"])
        entry = {"file": path.rsplit("/", 1)[-1], "split": run["split"], "prompt": name, **result["summary"],
                 "decoy_labelled": len(labelled), "decoy_misattributed": len(misattributed),
                 "labelled_ids": labelled, "misattributed_ids": misattributed}
        report.append(entry)
        print(f"{entry['split']:4} {name:7} right={entry['right_amount']}/{entry['questions']} "
              f"cited_ok={entry['right_and_cited_passage_has_it']} marker={entry['has_citation_marker']} "
              f"decoy_stated={entry['decoy_amount_also']} misattributed={len(misattributed)} labelled={len(labelled)}")
json.dump(report, open("classification.json", "w"), indent=1)
