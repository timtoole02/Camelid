# F2a — knowledge collections: live evidence

Everything here was produced by real `camelid serve` processes, driven over
HTTP or through the shipped UI in headless Chrome. Nothing is mocked or
replayed. `harness/check_claims.py` re-derives every number and quoted result
below from the files in this bundle, checks `SHA256SUMS`, and fails if any of
them disagrees.

Host: NVIDIA L4 (22 GiB), 4-vCPU Intel Xeon @ 2.20 GHz, Ubuntu 22.04. Chat
model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf` on the GPU. Encoder: the pinned
`nomic-embed-text-v1.5.Q8_0.gguf` on the CPU. Builds use the release profile
with link-time optimization turned off so they compile in minutes; that does
not change what they compute.

| Run | Build (`/v1/health`) | Files |
| --- | --- | --- |
| Screenshots | `v0.7.8-12-gd5732301` | `screenshots/`, `capture-collections.json` |
| Edge cases | `v0.7.8-12-gd5732301`, and `v0.7.8-9-g0477691e` for the downgrade | `edge/edge-collections.json` |
| SciFact scale | `v0.7.8-12-gd5732301` | `scale/scale-collections.json` |
| UI smokes and deliberate regressions | commit `d5732301`, clean tree | `ui/smokes.txt`, `ui/mutations.txt` |
| Rust test suite | commit `031cad73` | `tests/cargo-test.txt` |

`d5732301` is the head of this pull request's code. `0477691e` is the head of
#784, the build this one replaces. `031cad73` differs from `d5732301` only in
`frontend/src/styles/knowledge.css` (a checkbox's size), so its Rust sources
are the same.

## Screenshots

The library starts with two documents ingested over the API: the repository
`README.md` and the same 1,535-byte `support-policy.txt` as
`../f2a-verifiable-citations-20260927/source/support-policy.txt`. The third,
`source/escalation-guide.txt`, is a 636-byte fixture written for this run;
it covers some of the same ground on purpose, so that a question has more
than one candidate passage.

### screenshots/01-knowledge-library.png

**Attach → Collections**. The collection **Customer support** was created in
this dialog, `support-policy.txt` was added from the library, and
`escalation-guide.txt` was uploaded straight into it. **Search this
collection in this chat** is on, and the collection is marked **In this
chat**. The rest of the library (`README.md`) is offered under **Add from
library (1)**.

### screenshots/02-composer-collection-chip.png

The composer shows what the next message will search:
**Customer support · 2 docs**.

### screenshots/03-answer-from-collection.png

The question is "If hackers steal my information, when will you tell me?".
The chat made one document search, and it sent the collection's id as
`collection_ids`, with no `doc_ids` (`capture-collections.json`,
`ui_search_requests`). It returned 4 passages, 2 from `escalation-guide.txt`
and 2 from `support-policy.txt`, every one from a member of the collection.
Replaying that request returned the same passages in the same order. The sent
message reads **Customer support · 4 passages used**.

The answer is Llama-3.2-3B's, unedited, and it is not all correct. It opens
"According to the Camelid Support Escalation Guide [1], if a security
incident affects customer data, the account owner will be notified within
seventy-two hours". That rule is in `support-policy.txt`, which the answer
cites correctly later, as citation [3]. The guide, citation [1], says nothing
about seventy-two hours. Which passages were found, and whether each one is
verified, is what this change affects. How the model attributes them is not.

### screenshots/04-citation-verified.png

Citation [3] opens as **Verified · bytes 942–1433**, **Found by meaning**,
and its highlighted span holds the sentence "Security incidents that affect
customer data are reported to the account owner within seventy-two hours of
confirmation".

### screenshots/05-project-editor.png

**Projects → New project** "Support desk", with **Customer support** chosen
under the project's knowledge collections.

### screenshots/06-project-chat-and-unavailable-chip.png

**New chat** on the project card starts a chat that already searches the
project's collection: **Customer support · 2 docs · project**. Then a second
collection, "Old drafts", was created, turned on for this chat and deleted
in the library. Its chip stays, reading **Collection unavailable**, and it is
not searched.

## Edge cases

`harness/edge_collections.py` drives a fresh library over HTTP. 51 of 51
checks pass:

- **Managing collections.** a new library has no collections; create trims
  the name and answers 201 with no members; a name differing only in ASCII
  case is taken (409); an empty name is rejected (422); a blank name is
  rejected (422); 81 characters is rejected (422); a control character is
  rejected (422); 80 characters (not bytes) are accepted; non-ASCII letters
  are compared exactly (SQLite NOCASE folds ASCII only); the list is ordered
  by name ignoring case; an upload into an unknown collection is refused and
  stores nothing; an upload can join several collections at once; adding
  documents keeps the order added and ignores repeats; adding with one
  unknown document adds none of them (404); adding to an unknown collection
  is a 404; rename answers the renamed collection with its members; rename to
  a taken name is a 409 and changes nothing; rename to a different case of
  its own name is allowed; rename of an unknown collection is a 404.
- **Searching collections.** the whole library holds matches outside any
  collection; keyword: a collection search returns only its members; keyword:
  it ranks exactly like naming its members as doc_ids; hybrid: a collection
  search returns only its members; hybrid: it ranks exactly like naming its
  members as doc_ids; doc_ids and collection_ids together search both; an
  empty collection searches nothing; an empty collection list searches
  nothing; an unknown collection in the scope is a 404, not a silent partial
  search; a document in two searched collections is searched once.
- **Membership.** removing a member is a 204, idempotent, and keeps the
  document; removing from an unknown collection is a 404; re-ingesting a
  member keeps its memberships and serves the new text; re-ingesting into
  another collection adds, never drops; deleting a document removes it from
  every collection; deleting a collection keeps its documents; a second
  delete is a 404; a deleted collection cannot be searched.
- **The scope limit.** keyword: a scope of exactly 30,000 documents is
  searched; hybrid: a scope of exactly 30,000 documents is searched; one more
  is a 422 search_scope_too_large, not a database error; deleting a
  30,001-member collection removes its memberships; the database is intact.
- **The previous build, this build, and back.** a library written by the
  previous build opens with no collections; its documents can be collected
  and searched; the previous build still searches the library; and ingests
  into it; a delete by the previous build also removes the membership (the
  cascade is in the schema); back on this build the collection is intact,
  less what the previous build deleted; the database is intact after the
  round trip.
- **The LAN chat surface** (a server started with `--lan-chat-only`). LAN
  surface: GET /api/collections is a 403 lan_chat_only; LAN surface: POST
  /api/collections is a 403 lan_chat_only; LAN surface: POST
  /api/documents/search is a 403 lan_chat_only.

One behaviour of the previous build is recorded as an observation rather than
a check: re-ingesting a document with the previous build drops it from its
collections. That build writes the document row with `INSERT OR REPLACE`,
which deletes the old row, and the membership goes with it. This change
writes the row with an upsert, which is why "re-ingesting a member keeps its
memberships" holds on this build. Going back to the previous build is
otherwise safe, and its deletes clean up after themselves.

## SciFact scale

`harness/scale_collections.py` copies the 1,000-document SciFact library
built for `../f2a-hybrid-retrieval-20260928` (1,000 documents, 4,406 chunks,
every one indexed) and never writes to the original. It collects every other document
id, 500 documents, into one collection with a single request, then runs the
300 SciFact test claims in hybrid mode with `top_k` 20, three ways: scoped by
the collection, scoped by the same 500 ids as `doc_ids`, and over the whole
library.

- 0 results from outside the collection, in hybrid or in keyword mode.
- Scoping by the collection and by the same ids as `doc_ids` ranked the same
  passages in the same order: identical for all 300 claims.
- For the 145 claims whose relevant documents are all in the collection,
  Recall@10 was 0.9483 inside the collection and 0.9379 over the whole
  library: taking the other 500 documents out of the running lets slightly
  more relevant ones into the top 10.

| Search | p50 | p95 |
| --- | --- | --- |
| collection | 438.8 ms | 742.8 ms |
| doc_ids | 439.3 ms | 744.4 ms |
| whole library | 448.0 ms | 752.5 ms |
| collection, keyword | 18.6 ms | 25.1 ms |

A collection costs nothing measurable over naming its documents. Most of a
hybrid search's time is the meaning half, which embeds the query on the CPU;
the same collection searched by keyword alone takes a small fraction of it.
Adding the 500 documents took 10.2 ms in one request, and listing
collections took 1.9 ms at the median of 20 calls. These are timings on
this host, not a portable service level.

## UI smokes and deliberate regressions

On the committed tree, the new knowledge collections browser smoke and the
five smokes nearest to the code it touches pass (`ui/smokes.txt`). Each of
the following regressions was then written into the source, the UI rebuilt,
and the smoke run; it failed every time, 8 of 8, and every file was restored
afterwards (`ui/mutations.txt`):

- the chat sends members instead of the collection
- a deleted collection is still searched
- the LAN surface asks for collections
- the LAN surface searches a chat's collections
- the sent message does not name its collections
- project collections do not reach the chat
- a per-chat opt-out of a project collection is ignored
- an upload ignores its collection

## Rust test suite

`cargo test --all-targets --all-features`: 3,095 passed, 0 failed, 159
ignored, across 57 test binaries. The run was stopped in `runnable_smoke`,
whose tinyllama case is documented as release-only and had run for over an
hour in this debug build; cargo does not reach the targets after a stopped
one, so they were run next (`tests/cargo-test.txt`). `runnable_smoke`'s other
case passed. Of the remaining targets, 80 passed and 1 failed:
`spm_tokenizer_matches_hf`, which compares the tinyllama tokenizer with
Hugging Face's and only runs where that model file is present. It does not
depend on this change: this change touches no Rust outside `src/api/`, and
the tokenizer does not use it.

## Reproducing

```text
harness/full_test_build.sh      # the Rust suite, then the release build
harness/run_evidence.sh         # edge cases, SciFact scale, screenshots
harness/ui_final.sh             # UI smokes, then harness/mutate_collections.py
python3 harness/check_claims.py
```

The harness expects the checkout at `~/Camelid`, the models under
`~/Camelid/models`, the SciFact library and BEIR files under
`/mnt/disks/data/camelid-eval`, and the previous build at
`/tmp/f2h/bin/camelid-head`.
