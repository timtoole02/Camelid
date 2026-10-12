"""F2a 500-PDF success gate, driven against a live `camelid serve`.

Spec (CAMELID_PRODUCT_SPEC F2): "Ingest 500 PDFs, ask a question, get an answer
with citations; click a citation and land on the exact highlighted span;
corrupt one source file and that citation refuses rather than silently
drifting."

Usage: python3 gate.py <port> <corpus dir> <data dir> <out dir> <phase> [label]
Phases: ingest | verify <label> | retrieve | chat | corrupt
State is kept in <out dir>/state.json; each phase writes <out dir>/<phase>[-label].json.
"""
import base64
import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request

PORT, CORPUS, DATA, OUT, PHASE = sys.argv[1:6]
CORPUS, DATA, OUT = (os.path.abspath(p) for p in (CORPUS, DATA, OUT))
LABEL = sys.argv[6] if len(sys.argv) > 6 else ""
BASE = f"http://127.0.0.1:{PORT}"
DB = os.path.join(DATA, "documents_rag.sqlite3")
STATE = os.path.join(OUT, "state.json")
REAL_TARGET = int(os.environ.get("GATE_REAL_TARGET", "450"))
os.makedirs(OUT, exist_ok=True)


def call(method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Origin": BASE})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as err:
        raw, status = err.read(), err.code
    except Exception as err:  # noqa: BLE001 - a dropped connection is a result too
        return 0, {"transport_error": str(err)[:300]}, (time.perf_counter() - started) * 1000
    elapsed = (time.perf_counter() - started) * 1000
    try:
        return status, (json.loads(raw) if raw else None), elapsed
    except ValueError:
        return status, {"raw": raw[:300].decode(errors="replace")}, elapsed


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))], 1)


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(name, payload):
    json.dump(payload, open(os.path.join(OUT, name), "w"), indent=1)


def error_code(body):
    return ((body or {}).get("error") or {}).get("code") if isinstance(body, dict) else None


def ingest():
    state = {"started": time.time()}
    status, coll, _ = call("POST", "/api/collections", {"name": "F2a 500-PDF gate"})
    assert status == 201, (status, coll)
    state["collection_id"] = coll["id"]

    planted_dir = os.path.join(CORPUS, "planted")
    t0 = time.perf_counter()
    status, folder, _ = call("POST", "/api/folders", {"path": planted_dir, "collection_id": coll["id"]})
    assert status == 201, (status, folder)
    state["folder_id"] = folder["id"]
    for _ in range(600):
        _, folders, _ = call("GET", "/api/folders")
        entry = next(f for f in folders if f["id"] == folder["id"])
        if entry.get("last_scan_at") and not entry["scanning"] and not entry["queued"]:
            break
        time.sleep(1)
    state["planted_scan"] = {"documents": entry["documents"], "skipped": entry["skipped"],
                             "seconds": round(time.perf_counter() - t0, 1)}

    manifest = json.load(open(os.path.join(CORPUS, "arxiv-manifest.json")))
    real, rejected = [], []
    t0 = time.perf_counter()
    for arxiv_id, meta in manifest.items():
        if len(real) >= REAL_TARGET:
            break
        if meta.get("status") != "ok":
            continue
        path = os.path.join(CORPUS, "pdf", meta["file"])
        data = open(path, "rb").read()
        status, body, ms = call("POST", "/api/documents/ingest", {
            "filename": meta["file"], "content": base64.b64encode(data).decode(),
            "is_base64": True, "collection_ids": [coll["id"]]})
        row = {"arxiv_id": arxiv_id, "file": meta["file"], "bytes": len(data), "sha256": sha(data),
               "category": meta["category"], "ms": round(ms, 1), "status": status}
        if status == 200:
            row.update(doc_id=body["doc_id"], chunk_count=body["chunk_count"], text_bytes=body["byte_size"])
            real.append(row)
        else:
            row["code"] = error_code(body) or body
            rejected.append(row)
        print(f"ingest {len(real)}/{REAL_TARGET} {arxiv_id} {status} {round(ms)}ms", flush=True)
    real_seconds = time.perf_counter() - t0

    _, docs, _ = call("GET", "/api/documents")
    by_name = {d["filename"]: d for d in docs}
    facts = json.load(open(os.path.join(CORPUS, "planted-facts.json")))
    planted = []
    for fact in facts:
        doc = by_name[fact["file"]]
        data = open(os.path.join(planted_dir, fact["file"]), "rb").read()
        planted.append({"file": fact["file"], "doc_id": doc["id"], "chunk_count": doc["chunk_count"],
                        "bytes": len(data), "sha256": sha(data), "text_bytes": doc["byte_size"]})

    state.update(real=real, rejected=rejected, planted=planted, corrupted=[],
                 real_ingest_seconds=round(real_seconds, 1))
    json.dump(state, open(STATE, "w"), indent=1)
    summary = {
        "documents_ingested": len(real) + len(planted),
        "real_pdfs": len(real), "planted_pdfs": len(planted),
        "rejected": [{"arxiv_id": r["arxiv_id"], "status": r["status"], "code": r["code"]} for r in rejected],
        "chunks": sum(d["chunk_count"] for d in real + planted),
        "real_pdf_bytes": sum(d["bytes"] for d in real),
        "real_ingest_seconds": round(real_seconds, 1),
        "real_ingest_ms_p50": pct([d["ms"] for d in real], 50),
        "real_ingest_ms_p95": pct([d["ms"] for d in real], 95),
        "real_ingest_ms_max": max(d["ms"] for d in real) if real else None,
        "planted_scan": state["planted_scan"],
        "categories": {c: sum(1 for d in real if d["category"] == c) for c in sorted({d["category"] for d in real})},
    }
    save("ingest.json", summary)
    print(json.dumps(summary, indent=1))


def verify():
    state = load_state()
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    corrupted = {c["doc_id"]: c for c in state["corrupted"] if c["kind"] == "source_text"}
    report = {"label": LABEL, "documents": 0, "citations_resolved": 0, "citations_refused_as_expected": 0,
              "failures": [], "source_hash_mismatches": [], "uncovered_documents": [], "leaks": []}
    resolve_ms, source_ms = [], []
    for doc in state["real"] + state["planted"]:
        report["documents"] += 1
        doc_id = doc["doc_id"]
        status, src, ms = call("GET", f"/api/documents/{doc_id}/source")
        source_ms.append(ms)
        (stored_source_sha, chunk_count) = conn.execute(
            "SELECT source_sha256, chunk_count FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if stored_source_sha != doc["sha256"]:
            report["source_hash_mismatches"].append({"doc_id": doc_id, "file": doc["file"]})
        if doc_id in corrupted:
            expected = corrupted[doc_id]
            if status == 200:
                report["failures"].append({"doc_id": doc_id, "what": "corrupted source served"})
            for index in range(chunk_count):
                s, body, ms = call("POST", "/api/documents/citation/resolve", {"doc_id": doc_id, "chunk_index": index})
                resolve_ms.append(ms)
                text = json.dumps(body)
                if s == 409 and error_code(body) == expected["expected_code"] and "span" not in (body or {}):
                    report["citations_refused_as_expected"] += 1
                else:
                    report["failures"].append({"doc_id": doc_id, "chunk": index, "status": s, "code": error_code(body)})
                if expected.get("marker") and expected["marker"] in text:
                    report["leaks"].append({"doc_id": doc_id, "chunk": index})
            continue
        if status != 200:
            report["failures"].append({"doc_id": doc_id, "what": "source refused", "status": status, "code": error_code(src)})
            continue
        text_bytes = src["text"].encode()
        if sha(text_bytes) != src["doc_sha256"]:
            report["failures"].append({"doc_id": doc_id, "what": "served text does not hash to doc_sha256"})
        covered = bytearray(len(text_bytes))
        for index in range(chunk_count):
            s, body, ms = call("POST", "/api/documents/citation/resolve", {"doc_id": doc_id, "chunk_index": index})
            resolve_ms.append(ms)
            if s != 200:
                report["failures"].append({"doc_id": doc_id, "chunk": index, "status": s, "code": error_code(body)})
                continue
            span = text_bytes[body["byte_start"]:body["byte_end"]]
            ok = (span.decode() == body["span"] and sha(span) == body["chunk_sha256"]
                  and body["doc_sha256"] == src["doc_sha256"])
            if not ok:
                report["failures"].append({"doc_id": doc_id, "chunk": index, "what": "span mismatch"})
                continue
            report["citations_resolved"] += 1
            covered[body["byte_start"]:body["byte_end"]] = b"\x01" * (body["byte_end"] - body["byte_start"])
        uncovered = sum(1 for i, b in enumerate(text_bytes) if not covered[i] and not chr(b).isspace() and b < 128)
        uncovered += sum(1 for i, b in enumerate(text_bytes) if not covered[i] and b >= 128)
        if uncovered:
            report["uncovered_documents"].append({"doc_id": doc_id, "uncovered_bytes": uncovered})
        print(f"verify[{LABEL}] {report['documents']} docs, {report['citations_resolved']} citations", flush=True)
    report["resolve_ms_p50"], report["resolve_ms_p95"] = pct(resolve_ms, 50), pct(resolve_ms, 95)
    report["source_ms_p50"], report["source_ms_p95"] = pct(source_ms, 50), pct(source_ms, 95)
    report["passed"] = not (report["failures"] or report["source_hash_mismatches"]
                            or report["uncovered_documents"] or report["leaks"])
    save(f"verify-{LABEL}.json", report)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)} |
                     {k: len(v) for k, v in report.items() if isinstance(v, list)}, indent=1))


def planted_lookup(state):
    facts = json.load(open(os.path.join(CORPUS, "planted-facts.json")))
    by_file = {p["file"]: p["doc_id"] for p in state["planted"]}
    return facts, by_file


def retrieve():
    state = load_state()
    facts, by_file = planted_lookup(state)
    rows, search_ms = [], []
    for fact in facts:
        status, res, ms = call("POST", "/api/documents/search", {
            "query": fact["question"], "collection_ids": [state["collection_id"]], "top_k": 5})
        search_ms.append(ms)
        results = (res or {}).get("results", [])
        rank = next((i + 1 for i, r in enumerate(results)
                     if r["doc_id"] == by_file[fact["file"]] and fact["answer"] in r["excerpt"]), None)
        decoy_first = bool(results) and results[0]["doc_id"] == by_file[fact["decoy_file"]] \
            and fact["decoy_answer"] in results[0]["excerpt"]
        rows.append({"id": fact["id"], "status": status, "rank": rank, "decoy_first": decoy_first,
                     "mode": (res or {}).get("retrieval", {}).get("mode") if isinstance((res or {}).get("retrieval"), dict) else (res or {}).get("mode")})
    report = {
        "questions": len(rows),
        "first": sum(1 for r in rows if r["rank"] == 1),
        "top5": sum(1 for r in rows if r["rank"]),
        "decoy_first": sum(1 for r in rows if r["decoy_first"]),
        "search_ms_p50": pct(search_ms, 50), "search_ms_p95": pct(search_ms, 95),
        "rows": rows,
    }
    save(f"retrieve{('-' + LABEL) if LABEL else ''}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


def chat():
    state = load_state()
    facts, by_file = planted_lookup(state)
    _, health, _ = call("GET", "/v1/health")
    model = health.get("active_model_id")
    rows = []
    for fact in facts:
        status, res, search_ms = call("POST", "/api/documents/search", {
            "query": fact["question"], "collection_ids": [state["collection_id"]], "top_k": 4})
        citations = (res or {}).get("results", [])
        # Exactly the prompt ChatWorkspace.jsx builds for a message with document context.
        context = "\n\n".join(f"[Citation {i + 1} from {c['filename']}]:\n{c['excerpt']}" for i, c in enumerate(citations))
        content = ("Refer to the following retrieved document excerpts to answer the prompt. Cite your sources inline using [1], [2], etc.\n\n"
                   f"--- DOCUMENT CONTEXT ---\n{context}\n--- END CONTEXT ---\n\nUser Question: {fact['question']}")
        status, out, chat_ms = call("POST", "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 120, "stream": False}, timeout=1800)
        answer = ((out or {}).get("choices") or [{}])[0].get("message", {}).get("content", "") if status == 200 else ""
        flat = answer.replace(" ", "").replace("\u202f", "").replace("\u00a0", "")
        states_answer = fact["answer"] in answer or fact["answer"].replace(",", "") in flat
        states_decoy = fact["decoy_answer"] in answer or fact["decoy_answer"].replace(",", "") in flat
        # The markers the chat UI turns into citation pills, before and after it
        # learned the "(Citation N from file)" form (frontend/src/lib/markdown.jsx).
        old_pills = [int(m) for m in re.findall(r"\[(?:Citation\s+|Source\s+)?(\d+)\]", answer, re.I)]
        new_pills = [int(a or b) for a, b in re.findall(
            r"\[(?:Citation\s+|Source\s+)?(\d+)\]|\(Citation\s+(\d+)(?:\s+from\s+[^()]+)?\)", answer, re.I)]
        cited = sorted({n for n in new_pills if 0 < n <= len(citations)})
        cited_holds_answer = []
        for n in cited:
            c = citations[n - 1]
            s, body, _ = call("POST", "/api/documents/citation/resolve", {
                "doc_id": c["doc_id"], "chunk_index": c["chunk_index"],
                "chunk_sha256": c["chunk_sha256"], "doc_sha256": c["doc_sha256"]})
            cited_holds_answer.append(s == 200 and fact["answer"] in body.get("span", ""))
        rows.append({"id": fact["id"], "question": fact["question"], "answer": answer.strip(),
                     "expected": fact["answer"], "decoy": fact["decoy_answer"], "status": status,
                     "states_answer": states_answer, "states_decoy": states_decoy, "cited": cited,
                     "pill_before_fix": bool(old_pills), "pill_after_fix": bool(new_pills),
                     "a_cited_passage_resolves_to_the_answer": any(cited_holds_answer),
                     "citations": [{"n": i + 1, "file": c["filename"], "chunk_index": c["chunk_index"],
                                    "has_answer": fact["answer"] in c["excerpt"],
                                    "has_decoy": fact["decoy_answer"] in c["excerpt"]} for i, c in enumerate(citations)],
                     "passages": len(citations), "search_ms": round(search_ms, 1), "chat_ms": round(chat_ms, 1),
                     "usage": (out or {}).get("usage")})
        print(f"chat {fact['id']}: answer={states_answer} decoy={states_decoy} cited={cited} {round(chat_ms)}ms", flush=True)
    report = {
        "model": model, "questions": len(rows),
        "states_right_number": sum(1 for r in rows if r["states_answer"]),
        "states_decoy_number": sum(1 for r in rows if r["states_decoy"]),
        "has_a_citation_pill_before_fix": sum(1 for r in rows if r["pill_before_fix"]),
        "has_a_citation_pill_after_fix": sum(1 for r in rows if r["pill_after_fix"]),
        "right_and_cited_passage_resolves_to_it": sum(1 for r in rows if r["states_answer"] and r["a_cited_passage_resolves_to_the_answer"]),
        "chat_ms_p50": pct([r["chat_ms"] for r in rows], 50), "chat_ms_p95": pct([r["chat_ms"] for r in rows], 95),
        "rows": rows,
    }
    save(f"chat{('-' + LABEL) if LABEL else ''}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


def corrupt():
    """Three ways a source can go wrong, each made while the server runs."""
    state = load_state()
    facts, by_file = planted_lookup(state)
    report = {}
    conn = sqlite3.connect(DB, timeout=30)

    # 1. A real PDF's stored text is altered in the library database.
    victim = state["real"][0]
    (text,) = conn.execute("SELECT source_text FROM documents WHERE id = ?", (victim["doc_id"],)).fetchone()
    marker = "CORRUPTED-BY-GATE"
    position = len(text) // 2
    conn.execute("UPDATE documents SET source_text = ? WHERE id = ?",
                 (text[:position] + marker + text[position:], victim["doc_id"]))
    conn.commit()
    state["corrupted"].append({"doc_id": victim["doc_id"], "file": victim["file"], "kind": "source_text",
                               "expected_code": "citation_source_corrupted", "marker": marker})
    report["real_source_text_tampered"] = victim["file"]

    # 2. A planted fact's document is altered in the database: its question must
    #    not be answered from it, and its passage must not reach the model.
    fact = facts[7]
    doc_id = by_file[fact["file"]]
    (text,) = conn.execute("SELECT source_text FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.execute("UPDATE documents SET source_text = ? WHERE id = ?",
                 (text.replace(fact["answer"], fact["answer"].replace(",", ".")), doc_id))
    conn.commit()
    state["corrupted"].append({"doc_id": doc_id, "file": fact["file"], "kind": "source_text",
                               "expected_code": "citation_source_corrupted"})
    s, res, _ = call("POST", "/api/documents/search", {
        "query": fact["question"], "collection_ids": [state["collection_id"]], "top_k": 5})
    results = (res or {}).get("results", [])
    report["planted_fact_7"] = {
        "file": fact["file"],
        "results_from_tampered_document": sum(1 for r in results if r["doc_id"] == doc_id),
        "top_result": {"file": results[0]["filename"], "excerpt_has_decoy": fact["decoy_answer"] in results[0]["excerpt"]} if results else None,
    }

    # 3. A stored excerpt (chunk row) is altered while its source is intact:
    #    search must never hand the altered excerpt to the model.
    fact = facts[12]
    doc_id = by_file[fact["file"]]
    chunk_id, index, content = conn.execute(
        "SELECT id, chunk_index, content FROM document_chunks WHERE doc_id = ? AND content LIKE ?",
        (doc_id, f"%{fact['answer']}%")).fetchone()
    forged = content.replace(fact["answer"], "999,999")
    conn.execute("UPDATE document_chunks SET content = ? WHERE id = ?", (forged, chunk_id))
    conn.execute("UPDATE document_chunks_fts SET content = ? WHERE rowid = ?", (forged, chunk_id))
    conn.commit()
    s, res, _ = call("POST", "/api/documents/search", {
        "query": fact["question"], "collection_ids": [state["collection_id"]], "top_k": 5})
    results = (res or {}).get("results", [])
    s2, body, _ = call("POST", "/api/documents/citation/resolve", {"doc_id": doc_id, "chunk_index": index})
    report["planted_fact_12_excerpt_forged"] = {
        "forged_excerpt_served": any("999,999" in r["excerpt"] for r in results),
        "resolve_status": s2, "resolve_returns_original_text": s2 == 200 and fact["answer"] in body.get("span", ""),
    }

    # 4. A watched source file changes on disk. A citation shown before the
    #    change must refuse afterwards instead of resolving to the new text.
    fact = facts[20]
    doc_id = by_file[fact["file"]]
    s, res, _ = call("POST", "/api/documents/search", {
        "query": fact["question"], "doc_ids": [doc_id], "top_k": 5})
    before = next(r for r in res["results"] if fact["answer"] in r["excerpt"])
    path = os.path.join(CORPUS, "planted", fact["file"])
    data = open(path, "rb").read()
    new_answer = fact["answer"][:-3] + ("700" if not fact["answer"].endswith("700") else "800")
    assert data.count(fact["answer"].encode()) == 1
    open(path, "wb").write(data.replace(fact["answer"].encode(), new_answer.encode()))
    call("POST", f"/api/folders/{state['folder_id']}/scan")
    for _ in range(120):
        _, folders, _ = call("GET", "/api/folders")
        entry = next(f for f in folders if f["id"] == state["folder_id"])
        if not entry["scanning"] and not entry["queued"] and (entry.get("last_changes") or {}).get("updated"):
            break
        time.sleep(1)
    s, body, _ = call("POST", "/api/documents/citation/resolve", {
        "doc_id": before["doc_id"], "chunk_index": before["chunk_index"],
        "chunk_sha256": before["chunk_sha256"], "doc_sha256": before["doc_sha256"]})
    s_new, res_new, _ = call("POST", "/api/documents/search", {
        "query": fact["question"], "doc_ids": [doc_id], "top_k": 5})
    report["watched_file_changed_on_disk"] = {
        "file": fact["file"], "old_amount": fact["answer"], "new_amount": new_answer,
        "scan_changes": entry.get("last_changes"),
        "old_citation_status": s, "old_citation_code": error_code(body),
        "old_citation_leaks_text": "span" in (body or {}),
        "search_now_returns_new_amount": any(new_answer in r["excerpt"] for r in (res_new or {}).get("results", [])),
        "search_still_returns_old_amount": any(fact["answer"] in r["excerpt"] for r in (res_new or {}).get("results", [])),
    }
    # The re-indexed document is consistent again; record its new hash for the sweep.
    for doc in state["planted"]:
        if doc["doc_id"] == doc_id:
            doc["sha256"] = sha(open(path, "rb").read())
    json.dump(state, open(STATE, "w"), indent=1)
    save("corrupt.json", report)
    print(json.dumps(report, indent=1))


{"ingest": ingest, "verify": verify, "retrieve": retrieve, "chat": chat, "corrupt": corrupt}[PHASE]()
