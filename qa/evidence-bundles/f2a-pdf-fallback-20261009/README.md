# F2a: reading the PDFs pdf-extract could not — 2026-10-09

Issue #799: in the 500-PDF gate, 22 of the real arXiv PDFs tried could not be
read. pdf-extract 0.12.0 panics on them while reading a font's character map.

Linux x86_64, 4 vCPU, CPU only. The corpus is the 500 arXiv PDFs of
`f2a-500-pdf-gate-20261008` (ids and sha256 in its `source/arxiv-manifest.json`).

## What changed

1. **pdf-extract 0.12.0 → 0.12.1.** The release's one change resets the font cache
   per page; 0.12.0 reused a font across pages that named different fonts the
   same, which panicked on some files and decoded others with the wrong font.
2. **A fallback:** when pdf-extract panics, errors, or finds no text, lopdf (which
   pdf-extract reads with, at the same version) extracts the text page by page,
   leaving out only a page it cannot read, and drops U+FFFD (glyphs it cannot map).
   A PDF pdf-extract reads gives exactly the text it did before.

## Measured

**Each extractor on the 22 PDFs** (`harness/probe-readable.rs`):

| | readable |
| --- | --- |
| pdf-extract 0.12.0 | 0 / 22 (all panic) |
| pdf-extract 0.12.1 | 12 / 22 |
| lopdf, page by page | 22 / 22 |

**0.12.0 against 0.12.1 on all 500 PDFs** (`data/pdf-extract-0.12.0-vs-0.12.1.tsv`,
`harness/probe-hash.rs`, a hash and word count of the whitespace-normalised text):

| | count |
| --- | --- |
| Read by both, same text | 308 |
| Read by both, text changed | 168 |
| Newly readable with 0.12.1 | 12 |
| Readable with 0.12.0, panics with 0.12.1 | 2 (`2610.02290v1`, `2610.09665v1`) |
| Unreadable by both | 10 |

The changed texts, checked by word diff (`harness/probe-words.rs`), are fixes of
the font mix-up rather than regressions. For example, `2608.26208v2` read
"Recivri lSndTolkzliz Masitkxt pLPIvi" with 0.12.0 and "Observed MKI67 in
endothelial DL-predicted Distance scaler" with 0.12.1, and `2609.11877v1`
"NF-k B" became "NF-kB". Some changes are spacing inside formulas.
The two newly panicking files, and the ten still unreadable by pdf-extract, are
read by the fallback.

A library keeps the text it stored: a document is re-read only when it is
uploaded again or its watched file changes, so existing citations are not
disturbed by the upgrade.

**The live engine** (`harness/run_live.sh`, release build of this branch): every
one of the 500 PDFs uploaded through `POST /api/documents/ingest`, plus the 50
planted PDFs through a watched folder (`data/ingest-500.json`):

- **500 / 500** real PDFs ingested, **0 refused** (before: 478 readable, and 11
  over the upload size limit #798 removes).
- 114,083 chunks; **every citation** — 114,083 of 114,083 — resolves to its exact
  bytes, every PDF's `source_sha256` matches an independent hash of the file, and
  every byte of text is citable (`data/verify-all.json`).
- Text from the fallback reads as prose with no replacement characters, e.g.
  `2610.02290v1`: "who uses observed returns to decide both which assets to hold
  and how much to invest in them".

## Not covered

- Layout: like pdf-extract, the fallback gives text in content-stream order;
  multi-column pages and tables come out as the PDF orders them.
- Scanned PDFs (images only) still have no text; that needs OCR.
