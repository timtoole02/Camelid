# Section-aware chunking

Everything here was produced on one machine with two release builds: the
branch base `v0.7.8-25-gcfa1e191` (#786) and this change, `v0.7.8-26-g50570094`.
`harness/check_claims.py` re-derives every number in this README from the files
in this bundle, checks `SHA256SUMS`, and fails if any of them disagrees.

## The problem

The library cut every document into 512-character windows with 64 characters
of overlap, wherever the window happened to fall. A window could carry the end
of one section and the start of the next, and a passage about two topics scores
below a question about either one.

In the three-document probe library (`source/`: a support policy, an escalation
guide, and `PROJECT_CONTEXT.md`), the policy's chunk 3 ran from the end of
"3. Refunds" through "4. Data export" and "5. Incident handling" to the heading
"6. Changes to this policy" (`data/probe-chunks.json`). Asked "When will you
tell me about a data breach?", that chunk scored 0.6578 against the library
floor of 0.6603, so whole-library search did not serve the policy's answer
(seventy-two hours) at all. Asked "If hackers steal my information, when will
you tell me?", it scored 0.6344.

## The change

The chunker finds headings: a paragraph that starts after a blank line (or
starts the text) and is either a Markdown ATX heading (one to six `#` and a
space), or a single line of at most 80 characters that does not end like a
sentence or clause (`.`, `!`, `?`, `:`, `;`, `,`). A window ends at the first
heading that leaves at least a quarter window (128 characters) behind it, and
the next window starts exactly at that heading, with no overlap. Text without
headings is chunked exactly as before. Every chunk is still an exact byte
slice of the document's canonical text, which is what makes it citable.

Documents already in a library keep their chunks until they are ingested
again.

## Measured

`harness/chunk_eval.py` ingests the probe library into a fresh server, waits
for the pinned encoder (nomic-embed-text-v1.5) to index it, then records, for
nine questions, the top passages by meaning and what a whole-library search
serves at that library's floor (`data/chunk-eval-before.json`,
`data/chunk-eval-after.json`). The floor depends on how many chunks are
indexed, so it moves slightly: 0.6603 for 29 chunks before, 0.6609 for 32
after.

| Question | Served before (similarity) | Served after (similarity) |
| --- | --- | --- |
| If hackers steal my information, when will you tell me? | guide 1 (0.69) | guide "2. Security reports" (0.7459), policy "5. Incident handling" (0.6773) |
| When will you tell me about a data breach? | guide 1 (0.7031) | guide "2. Security reports" (0.7789), policy "5. Incident handling" (0.6826) |
| Who handles a billing dispute? | guide 1 (0.6642) | guide "3. Billing disputes" (0.7743) |
| How long do refunds take? | policy 2 (0.8344), policy 3 (0.7122), guide 1 (0.7021) | policy "3. Refunds" (0.8386) |
| hello, thanks!, a penguin joke, 17 times 23, an autumn haiku | nothing | nothing |

- Both breach questions now get the policy's incident section, which holds the
  answer, above the floor.
- The billing question's passage now holds only the billing section, and its
  margin over the floor grew from 0.0039 to 0.1134.
- The refund question is served one passage instead of three. The two it lost
  were mixed chunks: policy chunk 3 (the end of "3. Refunds" through "5.
  Incident handling") and guide chunk 1 (security reports and billing
  disputes, which mentions a refund being promised). The passage it kept is
  the whole refunds section, which holds the answer.
- The five off-topic messages still get nothing. Their best similarity after
  the change is 0.5875, before it 0.5942, against a floor of 0.66.

Chunk counts: the policy went from 5 chunks to 7, the guide from 2 to 4, and
`PROJECT_CONTEXT.md` from 22 to 21. Before, all 26 pairs of neighbouring chunks
overlapped. After, 16 of 29 do, all of them inside `PROJECT_CONTEXT.md`
sections longer than one window.

### The calibration libraries do not change

The library floor was calibrated on SciFact and FiQA libraries of 1000
documents each. `chunk_eval.py` re-ingests every one of those documents from
its stored text and compares the new chunk spans with the stored ones. With
this change, 0 of 1000 SciFact documents (4406 chunks) and 0 of 1000 FiQA
documents (2649 chunks) chunk differently, so the floor's calibration still
describes what this change stores. The same comparison on the branch base also
finds 0 of 1000 for both, which shows the comparison itself is exact.

### Edge suite

#786's whole-library edge suite (`harness/edge_library.py`, run on this
change's build with `harness/edge_chunks.sh`) passes 33 of 33 checks
(`data/edge-library.json`).

## Tests

`tests/unit.txt` holds the four new chunker tests and the `api::` unit tests
they ran with. The tests pin: one chunk per section with no overlap; sections
shorter than a quarter window sharing a chunk; prose without headings still
overlapping; and which lines count as headings (a `#[derive]` attribute, a
`#hashtag` line and seven `#`s do not). `tests/lint.txt` records `cargo fmt
--check` and `cargo clippy -D warnings` with and without `--all-features`.

## Limits

- The heading rule is a heuristic. A short line after a blank line that does not
  end in punctuation counts as a heading, so a signature line or a one-line
  list item can start a section. Such a split costs only overlap; every chunk
  is still an exact, citable slice.
- The probe is three documents and nine questions. It shows the failure and the
  fix; it is not a retrieval benchmark.
- Documents already in a library keep their old chunks until re-ingested.
