# F2a — verifiable citations: live evidence

Captured 2026-09-27 on `feat/f2-verifiable-citations`, build `v0.7.8-7-g1bd0bba9`
(the server's own `/v1/health` report).

Host: NVIDIA L4 (22 GiB), Ubuntu 22.04, debug build, all 28 layers resident on
the GPU. Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.

Everything here came from one `camelid serve` process with an empty document
library, driven through the shipped chat UI in headless Chrome. Nothing is
mocked or replayed. The document was uploaded through the UI's own file picker.
`api-transcript.json` is the byte-exact record of the API calls made alongside
the UI, so every number in the screenshots can be checked without them.

## screenshots/01-answer-with-citation.png

`support-policy.txt` (`source/support-policy.txt`, 1,535 bytes) is attached; the
chip reads **5 chunks**. The question is *What is the refund window for
enterprise customers?* The model's full answer, verbatim:

> According to the document excerpts, the refund window for Enterprise customers
> is within 60 days of the invoice date, and refunds are issued to the original
> payment method, taking 5 business days to process. [1]
>
> Additionally, it is mentioned that Team plans are refundable within 14 days of
> purchase, but Enterprise customers are not eligible for refunds under any
> circumstance. [1]
>
> It's worth noting that the document also mentions that refunds for full exports
> requested in writing are fulfilled within 30 days. [2]

The first paragraph is correct. The second and third misread the source: the
policy says **Trial** accounts are not eligible, and the 30 days concerns data
exports, not refunds. This is left in on purpose. A verified citation proves the
**cited passage is really in the source, unchanged**; it does not prove the model
paraphrased that passage correctly. Opening `[1]` (next screenshot) is how a
reader catches exactly this mistake.

The amber **VERIFIED** chip in the telemetry row is the existing support-contract
badge for the loaded model. It is not part of this change.

## screenshots/02-citation-verified.png

Clicking `[1]` sends the citation's binding to
`POST /api/documents/citation/resolve`. The server re-reads the stored source,
re-hashes it and re-derives the span before anything is shown. The viewer reads
**Verified · bytes 628–1006**. Only those bytes are highlighted; the surrounding
source text is dimmed context.

The highlight starts mid-sentence (*time of day, …*) because chunks are
fixed-size windows with overlap, not sentence-aligned. Nothing is trimmed or
adjusted for display.

Check it yourself:

```sh
head -c 1006 source/support-policy.txt | tail -c +629 | sha256sum
# 5f439e8ab837c9b2db1b08899d2b75306a9292c3369b03163233ed4319de57dc — the chunk_sha256 in api-transcript.json
sha256sum source/support-policy.txt
# equals doc_sha256 in api-transcript.json
```

## screenshots/03-citation-refused-after-corruption.png

The stored source text was then edited directly in the library's SQLite file
(`'60 days'` → `'30 days'`) to simulate corruption, with the server still
running. Opening **the same `[1]`** now shows **Citation refused** with
`citation_source_corrupted`. Neither the original nor the altered wording is
shown. The transcript records the resolve call going from `200` to `409`.

## screenshots/04-corrupted-source-not-served.png

A new chat, the same corrupted document attached, the same question. Search
withholds every passage from the corrupted document (the transcript's last
search returns `[]`). The model therefore receives only the question
(**tokens in 19**, against 416 in the first screenshot) and answers that it has
no information. There is nothing to cite, so no citation pill appears.

## Files

| File | What it is |
| --- | --- |
| `screenshots/*.png` | 1280×720 captures of the shipped UI |
| `source/support-policy.txt` | the exact document that was uploaded |
| `api-transcript.json` | search and resolve requests and responses, before and after the corruption |
| `SHA256SUMS` | checksums for every file above |
