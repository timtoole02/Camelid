#!/usr/bin/env python3
"""Re-derive every number in README.md from this bundle's files.

usage: python3 harness/check_claims.py   (from the bundle root)
"""
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []
CHECKS = 0


def check(name, condition, detail=""):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as handle:
        return json.load(handle)


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as handle:
        return handle.read()


# Checksums.
for line in read("SHA256SUMS").splitlines():
    digest, rel = line.split("  ", 1)
    with open(os.path.join(ROOT, rel), "rb") as handle:
        check(f"sha256 {rel}", hashlib.sha256(handle.read()).hexdigest() == digest)
listed = {line.split("  ", 1)[1] for line in read("SHA256SUMS").splitlines()}
on_disk = {os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/")
           for d, _, files in os.walk(ROOT) for f in files if f != "SHA256SUMS"}
check("SHA256SUMS lists every file", listed == on_disk, sorted(listed ^ on_disk))

before, after = load("data/chunk-eval-before.json"), load("data/chunk-eval-after.json")
check("builds", (before["build"], after["build"]) == ("v0.7.8-25-gcfa1e191", "v0.7.8-26-g50570094"))
check("chunk counts before", before["chunks"] == {"support-policy.txt": 5, "escalation-guide.txt": 2, "PROJECT_CONTEXT.md": 22}, before["chunks"])
check("chunk counts after", after["chunks"] == {"support-policy.txt": 7, "escalation-guide.txt": 4, "PROJECT_CONTEXT.md": 21}, after["chunks"])

# The floor follows the indexed chunk count: 0.6408 + 0.0058 ln(n).
for run, chunks, shown in ((before, 29, 0.6603), (after, 32, 0.6609)):
    check(f"{run['tag']} indexes {chunks} chunks", sum(run["chunks"].values()) == chunks)
    for probe in run["probes"]:
        check(f"{run['tag']} floor", round(probe["floor"], 4) == shown and abs(probe["floor"] - (0.6408 + 0.0058 * math.log(chunks))) < 1e-4, probe["floor"])

probes = {tag: {p["question"]: p for p in run["probes"]} for tag, run in (("before", before), ("after", after))}
HACKERS = "If hackers steal my information, when will you tell me?"
BREACH = "When will you tell me about a data breach?"
BILLING = "Who handles a billing dispute?"
REFUNDS = "How long do refunds take?"
OFFTOPIC = ["hello", "thanks!", "Tell me a joke about penguins.", "What is 17 times 23?", "Write a haiku about autumn leaves."]
check("nine questions", all(len(p) == 9 for p in probes.values()) and set(probes["before"]) == {HACKERS, BREACH, BILLING, REFUNDS, *OFFTOPIC})

served = {tag: {q: [(f, i, round(s, 4)) for f, i, s in p["library_served"]] for q, p in qs.items()} for tag, qs in probes.items()}
G, P = "escalation-guide.txt", "support-policy.txt"
expected = {
    "before": {HACKERS: [(G, 1, 0.69)], BREACH: [(G, 1, 0.7031)], BILLING: [(G, 1, 0.6642)],
               REFUNDS: [(P, 2, 0.8344), (P, 3, 0.7122), (G, 1, 0.7021)]},
    "after": {HACKERS: [(G, 2, 0.7459), (P, 5, 0.6773)], BREACH: [(G, 2, 0.7789), (P, 5, 0.6826)],
              BILLING: [(G, 3, 0.7743)], REFUNDS: [(P, 3, 0.8386)]},
}
for tag, questions in expected.items():
    for question, want in questions.items():
        check(f"{tag} serves {question!r}", served[tag][question] == want, served[tag][question])
    for question in OFFTOPIC:
        check(f"{tag} serves nothing for {question!r}", served[tag][question] == [], served[tag][question])

chunks = load("data/probe-chunks.json")
heading = {tag: {f: [c["text"].split("\n", 1)[0] for c in cs] for f, cs in docs.items()} for tag, docs in chunks.items()}
for filename, index, title in ((G, 2, "2. Security reports"), (G, 3, "3. Billing disputes"), (P, 3, "3. Refunds"), (P, 5, "5. Incident handling")):
    check(f"after {filename} chunk {index} is {title!r}", heading["after"][filename][index] == title, heading["after"][filename][index])


def top(tag, question, filename, index):
    return [round(s, 4) for f, i, s, _ in probes[tag][question]["top_by_meaning"] if (f, i) == (filename, index)]


old_mixed = chunks["before"][P][3]["text"]
check("before policy chunk 3 spans refunds to incidents",
      old_mixed.startswith("not eligible for refunds") and "4. Data export" in old_mixed
      and "5. Incident handling" in old_mixed and old_mixed.endswith("6. Changes to this policy"), old_mixed[:80])
check("breach scored 0.6578 on the mixed chunk", top("before", BREACH, P, 3) == [0.6578], top("before", BREACH, P, 3))
check("hackers scored 0.6344 on the mixed chunk", top("before", HACKERS, P, 3) == [0.6344], top("before", HACKERS, P, 3))
check("the mixed chunk was under the floor", 0.6578 < probes["before"][BREACH]["floor"])
guide_old = chunks["before"][G][1]["text"]
check("before guide chunk 1 holds security and billing", "2. Security reports" in guide_old and "3. Billing disputes" in guide_old and "refund is promised" in guide_old)
check("billing margin 0.0039 -> 0.1134",
      (round(0.6642 - probes["before"][BILLING]["floor"], 4), round(0.7743 - probes["after"][BILLING]["floor"], 4)) == (0.0039, 0.1134))
best = {tag: round(max(t[2] for q in OFFTOPIC for t in probes[tag][q]["top_by_meaning"]), 4) for tag in probes}
check("off-topic best similarity", best == {"before": 0.5942, "after": 0.5875}, best)

# Exact slices, and where neighbouring chunks overlap.
sources = {name: open(os.path.join(ROOT, "source", name), "rb").read() for name in (P, G, "PROJECT_CONTEXT.md")}
for tag, docs in chunks.items():
    for filename, cs in docs.items():
        check(f"{tag} {filename} chunks are exact slices",
              all(sources[filename][c["start"]:c["end"]].decode("utf-8") == c["text"] for c in cs))
pairs = {tag: [(f, a["end"] > b["start"]) for f, cs in docs.items() for a, b in zip(cs, cs[1:])] for tag, docs in chunks.items()}
check("before: 26 of 26 pairs overlap", (len(pairs["before"]), sum(o for _, o in pairs["before"])) == (26, 26))
check("after: 16 of 29 pairs overlap", (len(pairs["after"]), sum(o for _, o in pairs["after"])) == (29, 16))
check("after: overlaps only in PROJECT_CONTEXT.md", {f for f, o in pairs["after"] if o} == {"PROJECT_CONTEXT.md"})

for run in (before, after):
    calibration = run["calibration_libraries"]
    check(f"{run['tag']} scifact unchanged", (calibration["scifact"]["documents"], calibration["scifact"]["documents_with_changed_chunks"], calibration["scifact"]["chunks_after"]) == (1000, 0, 4406))
    check(f"{run['tag']} fiqa unchanged", (calibration["fiqa"]["documents"], calibration["fiqa"]["documents_with_changed_chunks"], calibration["fiqa"]["chunks_after"]) == (1000, 0, 2649))

edge = load("data/edge-library.json")
check("edge 33 of 33", (edge["passed"], edge["total"]) == (33, 33) and all(c["pass"] for c in edge["checks"]) and len(edge["checks"]) == 33)
check("edge ran this change", {s["build"] for s in edge["servers"]} == {"v0.7.8-26-g50570094"})

unit = read("tests/unit.txt")
for name in ("each_section_gets_its_own_chunk_without_overlap", "sections_shorter_than_a_quarter_window_share_a_chunk",
             "prose_without_headings_still_overlaps", "headings_are_short_lines_or_markdown_after_a_blank_line"):
    check(f"unit {name}", f"test api::citations::tests::{name} ... ok" in unit)
check("unit result ok", "test result: ok." in unit and "FAILED" not in unit)
lint = read("tests/lint.txt")
for key in ("FMT_EXIT=0", "CLIPPY_DEFAULT_EXIT=0", "CLIPPY_ALL_EXIT=0"):
    check(f"lint {key}", key in lint)

if FAILURES:
    print("\n".join(FAILURES))
    print(f"{len(FAILURES)} of {CHECKS} claims failed")
    sys.exit(1)
print(f"all {CHECKS} claims hold")
