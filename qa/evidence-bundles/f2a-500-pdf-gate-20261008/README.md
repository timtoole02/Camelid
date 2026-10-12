# F2a 500-PDF success gate — 2026-10-08

The product spec's F2 success test, run against a live release build:

> Ingest 500 PDFs, ask a question, get an answer with citations; click a
> citation and land on the exact highlighted span; corrupt one source file and
> that citation refuses rather than silently drifting.

Build `v0.7.8-28-g9da2be85` (this branch before the citation-pill commit; that
commit changes only the web UI, and `data/chat-v2.json` scores answers with the
UI's own marker rules before and after it). Linux x86_64, 4 vCPU Intel Xeon
@ 2.20 GHz, 15 GiB RAM, CPU only. Chat model `Llama-3.2-3B-Instruct-Q4_K_M.gguf`,
temperature 0. `CAMELID_DOCUMENT_SEMANTIC=0`: ranking is keyword (BM25) so the run
is deterministic; see *Not covered*.

## Corpus: 500 PDFs

- **450 real PDFs** from arXiv (recent submissions in 10 categories, fetched
  through the arXiv API at one request per 3 s by `harness/fetch_arxiv.py`),
  uploaded one by one through `POST /api/documents/ingest`, as the UI does.
  821.8 MB of PDF. The PDFs are not redistributed here; `source/arxiv-manifest.json`
  lists every id, title, size and sha256.
- **50 generated PDFs** (`harness/make_planted_pdfs.py`, deterministic) picked up by
  a **watched folder**. Each carries one planted fact ("The primary mirror of the
  Varnholt observatory was recoated in 2011 at a cost of 419,100 euros.") and a
  near-duplicate decoy of another document's fact (the *secondary* mirror, a
  different year and amount). `source/planted-facts.json` has every fact, decoy
  and question.

483 real PDFs were tried to get 450 in. The 33 refused are listed in
`data/ingest.json`:

- **22 × `422 extract_failed`.** The PDF reader (pdf-extract, via
  adobe-cmap-parser) panics on them. Before this branch the upload handler did
  not catch that, and the client saw the connection close with no response.
- **10 × `413` and 1 broken pipe**: PDFs of 12.7–14.6 MB. Uploads are base64 JSON
  under the 16 MiB request-body limit, so the largest PDF an upload accepted was
  about 12 MB (the largest accepted was 12,145,421 bytes), while a watched
  folder reads files up to 64 MB. The 413 was axum's plain-text rejection, not a
  typed Camelid error. **Fixed on this branch after the run**; see *Upload size,
  after the fix*.

| Ingest | |
| --- | --- |
| Documents | **500** (450 uploaded, 50 from the watched folder) |
| Chunks | **99,888** |
| Upload time, 450 PDFs | 138.2 s; per PDF p50 182 ms, p95 657 ms, max 7.4 s |
| Watched-folder scan, 50 PDFs | 1.0 s |

## Every citation resolves to its exact span

`data/verify-before.json`: for each of the 500 documents the sweep fetched the
served text, then resolved **every chunk's citation** and checked it
independently.

- **99,888 / 99,888** citations resolve. Each returned span is byte-for-byte the
  served text at `[byte_start, byte_end)`, hashes to its `chunk_sha256`, and comes
  with the document's `doc_sha256`, which the served text hashes to.
- The library's `source_sha256` equals an independent SHA-256 of **every PDF's
  original bytes** (500 / 500).
- Every non-whitespace byte of every document lies inside some citable span.
- Resolve p50 16.7 ms, p95 29.6 ms.

## Questions find the planted passage, not the decoy

`data/retrieve.json`: each of the 50 questions searched the gate collection
(top 5).

- The passage holding the planted answer ranked **first 50 / 50**.
- The decoy ranked first **0 / 50**.
- Search p50 224 ms, p95 229 ms over 99,888 chunks.

## Answers with citations

`data/chat-v2.json`: each question went through the same steps as the chat UI
(`ChatWorkspace.jsx`): search, top 4 passages, the UI's exact prompt, then
`/v1/chat/completions`. Run twice (`data/chat-run1.json`), with identical answers
50 / 50.

| | |
| --- | --- |
| States the right amount | **50 / 50** |
| A cited passage resolves to a span holding that amount | **50 / 50** |
| Has a clickable citation pill, UI before this branch | 30 / 50 |
| Has a clickable citation pill, UI with this branch | **50 / 50** |
| Also states the decoy amount | 27 / 50 |
| Chat latency | p50 6.6 s, p95 7.7 s |

- **Citation pills.** The UI labels each excerpt `[Citation N from file]`, and
  in 20 answers the model copied that label back as `(Citation 1 from
  planted-05.pdf)` instead of `[1]`. The UI made no pill of that form, so those
  answers had nothing to click. This branch renders it as the same verified pill.
- **The decoy amount.** The decoy passage was among the four given to the model
  in 47 answers. In 27 the model listed its amount as a second cost, though the
  passage is about the *secondary* mirror (or *south* clarifier, *west*
  turntable). Its citation still resolves to that passage, so a reader who
  clicks it sees the difference; but the sentence misattributes it. This is the
  3B model reading two similar passages, not retrieval or citation; it is
  recorded, not fixed.

Clicking a pill opens the citation viewer, which calls the same
`POST /api/documents/citation/resolve` the sweep above checks and highlights
`span` between `before` and `after`; the viewer itself is covered by
`npm run smoke:citations-browser` and `npm run smoke:document-viewer-browser`.

## Corrupted sources refuse

`data/corrupt.json`, made with the server running:

1. **A real PDF's stored text altered** in the library database
   (`2610.10536v1.pdf`): every one of its citations is refused with
   `citation_source_corrupted`, and no refusal contains its text.
2. **A planted fact's document altered** (`planted-07.pdf`): search returns
   **no** passage from it. The top result for its question became the decoy
   passage in `planted-06.pdf`, which is verified and correct about what it says.
3. **A stored excerpt forged** while its source is intact (`planted-12.pdf`,
   amount changed to 999,999): the forged excerpt is **never** served; resolving
   that chunk returns the original, verified text.
4. **A watched source file changed on disk** (`planted-20.pdf`, 342,600 →
   342,700): the next scan re-indexed it (`updated: 1`). The citation shown before
   the change is refused with **`citation_document_changed`**, without text;
   search now returns the new amount and never the old.

`data/verify-after-corrupt.json`: **99,721** citations resolve and the **167**
of the two altered documents are refused as expected; 0 failures.

## After a restart

`data/verify-after-restart.json`: the same 99,721 resolve and 167 refuse.
`data/retrieve-after-restart.json`: 47 / 50 first; the three misses are exactly
facts 7, 12 and 20, the ones corrupted above.

## Upload size, after the fix

`data/large-uploads.json`, build `v0.7.8-31-g6f271e78`, fresh library,
`harness/large_uploads.py`. That commit's message was later corrected; its
source tree `a14190e1` is identical to the upload-limit commit `54f1a5bb` on
this branch.

- The **11** PDFs refused for size above (12.7–14.6 MB) **all ingest** now:
  3,408 chunks, and **3,408 / 3,408** of their citations resolve.
- A file of 64 MB + 1 byte is refused with **`413 document_too_large`**: "The
  file is larger than 64 MB, the most the Knowledge Library reads."

## Not covered

- **Search by meaning at this scale.** Ranking was keyword only. The encoder
  embeds about one chunk a second on a CPU like this one (#784), so the
  99,888 chunks here would take about a day to index; hybrid ranking for a
  library this size was not measured.
- **A screenshot of clicking a citation on this library.** The viewer is
  covered by the browser smokes above against fixtures; on this library it was
  checked through the API it calls.
- The 22 PDFs pdf-extract cannot read are reported above and not fixed here.

## Files

- `harness/` — `fetch_arxiv.py`, `make_planted_pdfs.py`, `gate.py` (all phases),
  `run_gate.sh` (the main run), `run_chat2.sh` (the chat re-run),
  `large_uploads.py` (the upload-size check).
- `data/` — one JSON per phase; `run2-ingest.json` is the re-run's ingest.
- `source/` — the arXiv manifest, the planted facts, and the planted PDFs' sha256.
