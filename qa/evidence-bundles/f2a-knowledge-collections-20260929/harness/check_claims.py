#!/usr/bin/env python3
"""Check that every number and quoted result in this bundle's README comes from
the bundle's own data, and that SHA256SUMS matches the files.

usage: python3 harness/check_claims.py
Prints each failed claim and exits 1 if there is any.
"""
import hashlib
import json
import os
import re
import sys

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []
checked = 0


def path(name):
    return os.path.join(BUNDLE, name)


def load(name):
    with open(path(name), encoding="utf-8") as fh:
        return json.load(fh)


def text(name):
    with open(path(name), encoding="utf-8") as fh:
        return fh.read()


def sha256_file(name):
    with open(path(name), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def normalize(value):
    value = value.replace("\\\n", " ")
    value = re.sub(r"\s+", " ", value)
    return re.sub(r" ([.,;:])", r"\1", value).strip()


README = normalize(re.sub(r"(?m)^> ?", "", text("README.md")))


def claim(label, needle):
    """The README must state `needle` (whitespace-insensitive)."""
    global checked
    checked += 1
    if normalize(needle) not in README:
        failures.append(f"README does not state {label}: {needle!r}")


def fact(label, condition, detail=""):
    """A property of the data that the README relies on."""
    global checked
    checked += 1
    if not condition:
        failures.append(f"{label}: {detail}" if detail else label)


def ms(value):
    return f"{value:,.1f} ms"


edge = load("edge/edge-collections.json")
scale = load("scale/scale-collections.json")
capture = load("capture-collections.json")
mutations = text("ui/mutations.txt")
smokes = text("ui/smokes.txt")

# ---- builds --------------------------------------------------------------------------------------------
head_build = capture["build"]
claim("the build under test", f"`{head_build}`")
fact("the scale run used the same build", scale["build"] == head_build, scale["build"])
new_servers = [s for s in edge["servers"] if s["binary"] == "camelid-collections"]
old_servers = [s for s in edge["servers"] if s["binary"] != "camelid-collections"]
fact("every edge server of this change ran the build under test", new_servers and all(s["build"] == head_build for s in new_servers))
fact("one previous build served the downgrade", len({s["build"] for s in old_servers}) == 1, {s["build"] for s in old_servers})
previous_build = old_servers[0]["build"]
claim("the previous build", f"`{previous_build}`")

# ---- edge ----------------------------------------------------------------------------------------------
checks = [c for c in edge["checks"] if "check" in c]
fact("the edge counts are consistent", edge["passed"] == sum(c["pass"] for c in checks) and edge["total"] == len(checks))
fact("every edge check passed", edge["passed"] == edge["total"], [c["check"] for c in checks if not c["pass"]])
claim("the edge result", f"{edge['passed']} of {edge['total']} checks pass")
for c in checks:
    claim(f"edge check {c['check']!r}", c["check"])
note = next(c for c in edge["checks"] if c.get("observation"))
fact("the previous build's re-ingest was observed to drop the membership",
     note["status"] == 200 and note["memberships_after"] == [], note)
claim("the downgrade re-ingest finding", "re-ingesting a document with the previous build drops it from its collections")
lan = [s for s in edge["servers"] if s["api_surface"] == "lan_chat_only"]
fact("the LAN checks ran against a LAN chat surface", len(lan) == 1 and "--lan-chat-only" in lan[0]["args"], lan)

# ---- scale ---------------------------------------------------------------------------------------------
lib = scale["library"]
claim("the library size", f"{lib['documents']:,} documents")
claim("the chunk count", f"{lib['indexed_chunks']:,} chunks")
fact("the library was fully indexed", lib["indexed_chunks"] == lib["indexable_chunks"])
claim("the collection size", f"{scale['collection']['members']} documents")
claim("the query count", f"{scale['queries']} SciFact test claims")
fact("every scoped search ranked in hybrid mode", scale["retrieval_modes_seen"] == {"hybrid": scale["queries"]}, scale["retrieval_modes_seen"])
fact("no result fell outside the collection", scale["results_outside_the_collection"] == 0 and scale["violations"] == [])
claim("the outside count", "0 results from outside the collection")
fact("the collection ranking equals the doc_ids ranking for every query", scale["collection_equals_doc_ids_ranking"] == scale["queries"])
claim("the ranking equality", f"identical for all {scale['queries']}")
gold = scale["gold_inside_collection"]
claim("the in-collection query count", f"{gold['queries']} claims")
claim("recall inside the collection", f"{gold['recall10_collection']:.4f}")
claim("recall over the whole library", f"{gold['recall10_whole_library']:.4f}")
for name, label in (("collection", "collection"), ("doc_ids", "doc_ids"), ("whole_library", "whole library"), ("collection_keyword", "collection, keyword")):
    lat = scale["latency"][name]
    claim(f"{name} p50", f"| {label} | {ms(lat['p50_ms'])} | {ms(lat['p95_ms'])} |")
claim("the add request time", ms(scale["collection"]["add_request_ms"]))
claim("the list time", ms(scale["collection"]["list_collections"]["p50_ms"]))

# ---- screenshots ---------------------------------------------------------------------------------------
fact("the capture raised no page errors", capture["page_errors"] == [])
fact("the UI's one search sent the collection and no doc_ids",
     len(capture["ui_search_requests"]) == 1 and set(capture["ui_search_requests"][0]) == {"query", "collection_ids", "top_k"}
     and capture["ui_search_requests"][0]["collection_ids"] == [capture["collection_api"]["id"]])
fact("the collection holds the two documents shown", sorted(m.split(".txt")[0] for m in capture["library_members"]) == ["escalation-guide", "support-policy"])
claim("the composer chip", "**Customer support · 2 docs**")
fact("the composer chip text", capture["composer_chips"] == ["Customer support2 docs"], capture["composer_chips"])
passages = len(capture["chat_citations"])
fact("the message chip counts the passages", capture["message_chips"] == [f"Customer support{passages} passages used"], capture["message_chips"])
claim("the message chip", f"**Customer support · {passages} passages used**")
fact("every cited passage came from the collection", capture["citations_all_members"] is True)
by_file = {}
for c in capture["chat_citations"]:
    by_file[c["filename"]] = by_file.get(c["filename"], 0) + 1
for name, count in by_file.items():
    claim(f"passages from {name}", f"{count} from `{name}`")
fact("the replayed search returns the same passages", [(r["filename"], r["chunk_index"]) for r in capture["search_replay"]["results"]]
     == [(r["filename"], r["chunk_index"]) for r in capture["chat_citations"]])
n = capture["expected_citation_number"]
fact("the incident passage is one citation", n is not None and capture["chat_citations"][n - 1]["holds_expected_passage"])
claim("the incident citation number", f"citation [{n}]")
modal = capture["citation_modal"]
claim("the citation badge", f"**{modal['badge']}**")
claim("the citation provenance", f"**{modal['found']}**")
fact("the verified span holds the incident sentence", "reported to the account owner within seventy-two hours" in modal["span"])
first_sentence = capture["answer"].split("\n")[0].strip()
claim("the answer's opening words", first_sentence)
fact("the project editor offered the collection", capture["project_editor_collections"] == ["Customer support2 docs"])
fact("the project chat chips", capture["project_chat_chips"] == ["Customer support2 docs · project", "Collection unavailable"], capture["project_chat_chips"])
claim("the project chip", "**Customer support · 2 docs · project**")
claim("the unavailable chip", "**Collection unavailable**")
guide = sha256_file("source/escalation-guide.txt")
claim("the guide's size", f"{os.path.getsize(path('source/escalation-guide.txt')):,}-byte")

# ---- UI tests ------------------------------------------------------------------------------------------
caught = re.search(r"(\d+)/(\d+) mutations caught", mutations)
fact("every mutation was caught", caught and caught.group(1) == caught.group(2), caught and caught.group(0))
claim("the mutation count", f"{caught.group(1)} of {caught.group(2)}")
for line in mutations.splitlines():
    if line.startswith("CAUGHT"):
        label = line[len("CAUGHT"):].split(":", 1)[0].strip()
        claim(f"mutation {label!r}", label)
fact("the smokes passed", "PASS knowledge-collections-browser" in smokes and "FAIL" not in smokes)

# ---- Rust tests ----------------------------------------------------------------------------------------
cargo = text("tests/cargo-test.txt")
main_run = re.search(r"passed (\d+) failed (\d+) ignored (\d+) binaries (\d+)", cargo)
rest_run = re.findall(r"passed (\d+) failed (\d+) ignored (\d+) binaries (\d+)", cargo)[1]
fact("the main run had no failures", main_run and main_run.group(2) == "0")
claim("the main run", f"{int(main_run.group(1)):,} passed, 0 failed, {main_run.group(3)} ignored, across {main_run.group(4)} test binaries")
claim("the remaining targets", f"{rest_run[0]} passed and {rest_run[1]} failed")
fact("the one remaining failure is the tokenizer comparison", rest_run[1] == "1" and "test spm_tokenizer_matches_hf ... FAILED" in cargo)
fact("runnable_smoke's other case passed", "RUNNABLE_SMOKE_EXIT=0" in cargo and "1 passed; 0 failed" in cargo)

# ---- checksums -----------------------------------------------------------------------------------------
listed = {}
for line in text("SHA256SUMS").splitlines():
    digest, name = line.split("  ", 1)
    listed[name] = digest
on_disk = sorted(os.path.relpath(os.path.join(root, f), BUNDLE) for root, _, files in os.walk(BUNDLE) for f in files if f != "SHA256SUMS")
fact("SHA256SUMS lists every file", sorted(listed) == on_disk, set(on_disk) ^ set(listed))
for name, digest in listed.items():
    fact(f"checksum of {name}", os.path.exists(path(name)) and sha256_file(name) == digest)

if failures:
    print("\n".join(failures))
    print(f"{len(failures)} of {checked} claims failed")
    sys.exit(1)
print(f"all {checked} claims hold")
