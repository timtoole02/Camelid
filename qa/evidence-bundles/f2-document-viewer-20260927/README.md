# F2 — document viewer, message chips and files tray: live evidence

Captured 2026-09-27 at commit `52b307b7`, build `v0.7.8-10-g52b307b7` (the
server's own `/v1/health` report).

Host: NVIDIA L4 (22 GiB), Ubuntu 22.04, debug build, all 28 layers resident on
the GPU. Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.

One `camelid serve` process, driven through the shipped chat UI in headless
Chrome with a fresh browser profile. Nothing is mocked or replayed. The
server's library also held other documents from hands-on testing; the fresh
profile attaches only the uploaded file, and every library call in
`api-transcript.json` targets that one document. The document is
`support-policy.txt`, the same 1,535-byte file as
`../f2a-verifiable-citations-20260927/source/support-policy.txt`, uploaded
through the UI's own file picker. `capture-summary.json` records what the UI
showed at each step.

## screenshots/01-attachment-remove-control.png

The attached document as a chip in the composer, with the pointer on its remove
control. The control sits before the name, so it stays in place whatever the
filename length. It measures 24 × 24 px, is labelled
*Remove support-policy.txt*, and highlights on hover.

## screenshots/02-document-viewer-verified.png

Clicking the chip opens the viewer. The server serves the text only after
re-hashing it: in the transcript, `sha256_of_served_text` equals the recorded
`doc_sha256`, which is also the sha256 of the source file. The badge reads
**Verified · 1.5 KB** and the text scrolls inside the viewer.

## screenshots/03-message-chips-and-files-tray.png

The question is *What is the refund window for enterprise customers? Also give
it as a JSON object in a \`\`\`json code block.* The sent message keeps a chip
for the document it was sent with: **support-policy.txt · 4 passages used**. The
model's answer begins:

> According to the retrieved document excerpts, the refund window for Enterprise
> customers is 60 days from the invoice date.

The code block it produced is listed in the tray above the composer:
**1 file from this conversation**, `output.txt`, text/plain · 58 bytes. The
model left the fence untagged (the card reads **CODE**), so the existing
output-file naming falls back to `output.txt`; that naming is not part of this
change.

## screenshots/04-files-tray-opens-panel.png

Choosing the file in the tray opens the files panel on it, showing the model's
JSON.

## screenshots/05-document-viewer-refused-after-tampering.png

The stored text was then edited directly in the library's SQLite file
(`'60 days'` → `'30 days'`) with the server still running. Opening the chip on
the sent message now shows **Document not shown** with
`document_source_corrupted`. Neither wording is displayed. The transcript
records `200` → `409`. Putting the text back makes it verify again (`200`, same
hash), so the check is on the content, not on a stored flag.

## screenshots/06-document-viewer-deleted-document.png

After `DELETE /api/documents/:id` (`204`), the chip on the old message still
opens and says the document is no longer in the library (`404`,
`document_not_found`) instead of failing silently.

## Check it yourself

```sh
sha256sum ../f2a-verifiable-citations-20260927/source/support-policy.txt
# df85653c7c4940858c185c1c5235284df9f9c1ab96484e60e007154d18a32d1a
# equals doc_sha256 and sha256_of_served_text in api-transcript.json
```

## Files

`SHA256SUMS` lists every file in this directory except itself.
