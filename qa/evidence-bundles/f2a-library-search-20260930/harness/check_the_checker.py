#!/usr/bin/env python3
"""Check the checker: each mutation of the bundle must make check_claims.py fail.

Works on copies under /tmp/cc; the bundle itself is never touched.
"""
import json
import os
import shutil
import subprocess

REPO = os.path.expanduser("~/Camelid/qa/evidence-bundles")
NAME = "f2a-library-search-20260930"
ROOT = "/tmp/cc"


def fresh():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    for sibling in ("f2a-knowledge-collections-20260929", "f2a-verifiable-citations-20260927"):
        os.symlink(os.path.join(REPO, sibling), os.path.join(ROOT, sibling))
    shutil.copytree(os.path.join(REPO, NAME), os.path.join(ROOT, NAME))
    return os.path.join(ROOT, NAME)


def resum(bundle):
    subprocess.run("find . -type f ! -name SHA256SUMS | sed 's|^\\./||' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS",
                   shell=True, cwd=bundle, check=True)


def edit_text(rel, old, new):
    def apply(bundle):
        p = os.path.join(bundle, rel)
        body = open(p, encoding="utf-8").read()
        assert old in body, (rel, old)
        open(p, "w", encoding="utf-8").write(body.replace(old, new, 1))
    return apply


def edit_json(rel, change):
    def apply(bundle):
        p = os.path.join(bundle, rel)
        data = json.load(open(p, encoding="utf-8"))
        change(data)
        json.dump(data, open(p, "w", encoding="utf-8"), indent=2)
    return apply


def edit_jsonl(rel, index, key, value):
    def apply(bundle):
        p = os.path.join(bundle, rel)
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        rows[index][key] = value
        with open(p, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    return apply


def remove(rel):
    return lambda bundle: os.remove(os.path.join(bundle, rel))


def add_file(rel):
    return lambda bundle: open(os.path.join(bundle, rel), "wb").write(b"x")


MUTATIONS = [
    ("a held-out rate in the README", edit_text("README.md", "served\n85.2% of questions", "served\n85.3% of questions"), True),
    ("the incident score in the README", edit_text("README.md", "scored\n0.6344", "scored\n0.6345"), True),
    ("an endpoint table cell", edit_text("README.md", "| FiQA | 76 of 100 (76.0%)", "| FiQA | 77 of 100 (77.0%)"), True),
    ("a latency cell", edit_text("README.md", "422.6 / 1,273.2", "412.6 / 1,273.2"), True),
    ("a mutation bullet dropped", edit_text("README.md", "- the switch is not saved with the chat\n", ""), True),
    ("an edge check dropped from the README", edit_text("README.md", "- top_k still bounds a library search\n", ""), True),
    ("the fitted rule", edit_text("README.md", "0.6408 + 0.0058", "0.6408 + 0.0060"), True),
    ("the Rust count", edit_text("README.md", "3,181 passed", "3,182 passed"), True),
    ("a pre-registered script edited", edit_text("harness/analyze_floor.py", "\n", "\n\n"), True),
    ("an endpoint row disagrees", edit_jsonl("endpoint/endpoint-per-query.jsonl", 7, "agrees", False), True),
    ("an endpoint row below the floor", edit_jsonl("endpoint/endpoint-per-query.jsonl", 9, "below_floor", 1), True),
    ("an edge check fails", edit_json("edge/edge-library.json", lambda d: d["checks"][5].update({"pass": False})), True),
    ("the answer gets a citation pill", edit_json("capture-encoder.json", lambda d: d.update({"pills": ["View source citation [1]"]})), True),
    ("the UI sends doc_ids too", edit_json("capture-encoder.json", lambda d: d["answered"]["searches"][0].update({"doc_ids": ["x"]})), True),
    ("the hard question gets the incident passage", edit_json("capture-encoder.json", lambda d: d["hard"]["citations"][0].update({"holds_hard_passage": True})), True),
    ("a screenshot removed", remove("screenshots/06-unrelated-message.png"), True),
    ("an unlisted file", add_file("extra.txt"), True),
]


def run(bundle):
    return subprocess.run(["python3", "-B", "harness/check_claims.py"], cwd=bundle, capture_output=True, text=True)


baseline = run(fresh())
print("baseline:", baseline.stdout.strip().splitlines()[-1], "exit", baseline.returncode)
assert baseline.returncode == 0
caught = 0
for label, mutate, with_resum in MUTATIONS:
    bundle = fresh()
    mutate(bundle)
    if with_resum and os.path.exists(os.path.join(bundle, "SHA256SUMS")) and label != "an unlisted file":
        resum(bundle)  # so only the claim itself can catch it, not the checksum
    result = run(bundle)
    ok = result.returncode != 0
    caught += ok
    first = [l for l in result.stdout.splitlines() if "claims failed" not in l][:1]
    print("CAUGHT" if ok else "MISSED", label, "->", first[0][:140] if first else result.stdout.strip()[-120:])
print(f"{caught}/{len(MUTATIONS)} checker mutations caught")
shutil.rmtree(ROOT, ignore_errors=True)
