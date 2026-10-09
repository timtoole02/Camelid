"""A/B of the chat's document prompt on the planted-fact questions (issue #800).

Each question goes through the chat's steps: search the collection (top 4),
build the prompt, ask the loaded model at temperature 0. Prompts are named in
PROMPTS; `current` is what ChatWorkspace.jsx sends today.

Usage: python3 eval_rag_prompt.py <port> <corpus dir> <state.json> <out json> <split: dev|test|all> <prompt name>...
"""
import json
import re
import sys
import time
import urllib.request

PORT, CORPUS, STATE, OUT, SPLIT = sys.argv[1:6]
NAMES = sys.argv[6:]
BASE = f"http://127.0.0.1:{PORT}"


def current(question, citations):
    context = "\n\n".join(f"[Citation {i + 1} from {c['filename']}]:\n{c['excerpt']}" for i, c in enumerate(citations))
    return ("Refer to the following retrieved document excerpts to answer the prompt. Cite your sources inline using [1], [2], etc.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context}\n--- END CONTEXT ---\n\nUser Question: {question}")


def exact(question, citations):
    context = "\n\n".join(f"[{i + 1}] (from {c['filename']}):\n{c['excerpt']}" for i, c in enumerate(citations))
    return ("Answer the question from the document excerpts below. Use an excerpt only if it is about exactly what the question asks: "
            "one about something similar - another item, part, person, date or place - does not answer it, so leave it out. "
            "Cite each fact inline as [1], [2], matching the excerpt it came from.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context}\n--- END CONTEXT ---\n\nQuestion: {question}")


def exact2(question, citations):
    context = "\n\n".join(f"[{i + 1}] (from {c['filename']}):\n{c['excerpt']}" for i, c in enumerate(citations))
    return ("Answer the question from the document excerpts below. Use an excerpt only if it is about exactly what the question asks: "
            "one about something similar - another item, part, person, date or place - does not answer it, so leave it out. "
            "Cite each fact inline as [1], [2], matching the excerpt it came from; every answer cites at least one excerpt.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context}\n--- END CONTEXT ---\n\nQuestion: {question}")


PROMPTS = {"current": current, "exact": exact, "exact2": exact2}

facts = json.load(open(f"{CORPUS}/planted-facts.json"))
state = json.load(open(STATE))
by_file = {p["file"]: p["doc_id"] for p in state["planted"]}
if SPLIT == "dev":
    facts = [f for f in facts if f["id"] % 2 == 0]
elif SPLIT == "test":
    facts = [f for f in facts if f["id"] % 2 == 1]


def call(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


model = json.loads(urllib.request.urlopen(f"{BASE}/v1/health").read())["active_model_id"]
digits = lambda text: re.sub(r"[\s\u202f\u00a0,]", "", text)
report = {"model": model, "split": SPLIT, "prompts": {}}
for name in NAMES:
    rows = []
    for fact in facts:
        found = call("/api/documents/search", {"query": fact["question"], "collection_ids": [state["collection_id"]], "top_k": 4})
        citations = found.get("results", [])
        started = time.time()
        out = call("/v1/chat/completions", {"model": model, "temperature": 0, "max_tokens": 160, "stream": False,
                                            "messages": [{"role": "user", "content": PROMPTS[name](fact["question"], citations)}]})
        answer = out["choices"][0]["message"]["content"]
        flat = digits(answer)
        states_answer = digits(fact["answer"]) in flat
        states_decoy = digits(fact["decoy_answer"]) in flat
        markers = [int(a or b) for a, b in re.findall(r"\[(?:Citation\s+|Source\s+)?(\d+)\]|\(Citation\s+(\d+)(?:\s+from\s+[^()]+)?\)", answer, re.I)]
        cited_ok = any(0 < n <= len(citations) and fact["answer"] in citations[n - 1]["excerpt"] for n in markers)
        decoy_in_context = any(fact["decoy_answer"] in c["excerpt"] for c in citations)
        rows.append({"id": fact["id"], "question": fact["question"], "answer": answer, "states_answer": states_answer,
                     "states_decoy": states_decoy, "decoy_in_context": decoy_in_context, "pill": bool(markers),
                     "cited_passage_has_answer": cited_ok, "seconds": round(time.time() - started, 1)})
        print(f"{name} {fact['id']}: answer={states_answer} decoy={states_decoy} pill={bool(markers)} cited_ok={cited_ok}", flush=True)
    summary = {
        "questions": len(rows),
        "right_amount": sum(r["states_answer"] for r in rows),
        "decoy_amount_also": sum(r["states_decoy"] for r in rows),
        "decoy_in_context": sum(r["decoy_in_context"] for r in rows),
        "right_and_cited_passage_has_it": sum(r["states_answer"] and r["cited_passage_has_answer"] for r in rows),
        "has_citation_marker": sum(r["pill"] for r in rows),
    }
    report["prompts"][name] = {"summary": summary, "rows": rows}
    print(name, json.dumps(summary), flush=True)
json.dump(report, open(OUT, "w"), indent=1)
