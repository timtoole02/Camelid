# Watched folders

Everything here was produced on one machine with the release build of this
change, `v0.7.8-27-gecee5ebf`, whose web UI was built from the same commit.
`harness/check_claims.py` re-derives every number in this README from the
files in this bundle, checks `SHA256SUMS`, and fails if any of them disagrees.

## What it does

A collection can watch a folder on this computer and holds every document in
it, kept current:

- A scan reads only files whose size or modification time changed since the
  last scan. A file whose bytes are unchanged is not re-indexed; a file whose
  content changed is re-indexed under the same document id, so its collection
  memberships survive. The documents of files that are gone are removed.
- Scans run when a folder is added, on **Check now**, and every 30 seconds
  while the server runs, starting when it starts.
- Only the library's document types are read (PDF, Word, Markdown, text, CSV,
  JSON, and `.rs`, `.py`, `.js`). Hidden entries and symbolic links are
  skipped. A file over 64 MB, an unreadable file, a file with no text and a
  file that cannot be parsed are listed as skipped, with the reason, and do not
  end the scan. A folder holding more than 10,000 documents is refused, not
  truncated.
- A folder that cannot be opened, or a subfolder that cannot be read, changes
  nothing: its documents are kept and the folder says why.
- Folders are read from this computer's disk, so `/api/folders` answers only
  the local web UI on a loopback listener (the rule the workspace folder picker
  already follows), and the LAN chat surface refuses it.
- Stopping a watch removes its documents from the library and leaves the files
  on disk. Deleting the collection stops its watches and keeps the documents.

Files and folders can also be dropped onto a collection in the knowledge
library; the readable ones are uploaded into it under their path inside the
drop.

## Live, through the HTTP API

`harness/folder_edge.py` drives a real server through `/api/folders` on a
real folder, and passes 46 of 46 checks (`data/folder-edge.json`):

- **Who may call it.** A request without the web UI's origin, or from another
  origin, is refused with 403, and so is stopping a watch. All four folder
  routes answer 403 `lan_chat_only` on the LAN chat surface.
- **What may be watched.** A relative path, a file, a missing folder and the
  filesystem root are refused with 422; an unknown collection is a 404; the
  same folder, a folder inside it and the folder around it are refused with
  409.
- **The first scan** adds the three documents, skips an empty file and a
  broken PDF with their reasons, and reads neither hidden files, another file
  type, nor a link to a file or a folder outside. Every document joins the
  collection, its source text is the file's text, and a search of the
  collection finds the answer in it.
- **Changes.** After a file is edited, one deleted and one added, the
  server's timer picked all three up 21.23 s later without being asked: the
  edited file keeps its document id and its new text is searchable, and the
  deleted file's document and source are gone. **Check now** re-indexes an
  edit at once.
- **A restart.** A file edited while the server was stopped had already been
  re-indexed by the startup check when the suite first asked, 0.0 s after the
  server answered.
- **A folder that disappears** reports that it cannot be opened and keeps its
  documents; once it is back, the error clears and nothing changes.
- **Stopping and deleting.** Stopping a watch removes its documents and
  leaves the files and the collection; deleting the collection stops the
  watch and keeps the documents.

## The UI, against the same build

`harness/capture_folders.mjs` drives the shipped web UI against a live server
with TinyLlama loaded (`data/capture-folders.json`, `screenshots/`):

1. Browse lists the folders the server reports, and choosing one fills the
   path (`1-browse.png`).
2. Watching it shows the three documents in the collection, with the two
   skipped files and their reasons (`2-watched.png`).
3. The folder is then changed on disk. The UI showed "Last check: 1 added,
   1 updated, 1 removed." and the new document list 18.8 s later, with nobody
   touching it (`3-changed-on-disk.png`).
4. Stopping the watch asks first and says the files on disk are not touched
   (`4-stop-watching.png`); the collection is then empty.

Every folder request the page made was answered 2xx, and the page raised no
errors.

The first run of this capture, against an earlier build of this branch,
timed out at step 3. The list then refreshed only when a poll caught a check
still running, and a timer check finishes between two polls. The UI now
refreshes whenever a poll shows a finished check that changed the library,
and the CI browser smoke covers exactly that case.

## Tests

- `tests/unit.txt`: the 13 folder unit tests, run with the `api::` unit tests
  (634 tests, all passing). They cover hidden entries, links and other types
  being skipped; re-indexing in place and removal; a new time on unchanged
  bytes re-indexing nothing; a deleted document returning; oversized and
  unparseable files; a file that loses its text; an unopenable folder; an
  unreadable subfolder; overlapping folders; path rules; stopping a watch;
  deleting the collection; and the local-only rule.
- `tests/mutations.txt`: 6 mutations of the scanner and the routes, each
  caught by a failing unit test: answering any caller, following links,
  reading hidden entries, dropping the documents of an unreadable subfolder,
  re-indexing unchanged bytes, and accepting overlapping folders.
- `tests/smoke.txt`: the new CI browser smoke,
  `npm run smoke:watched-folders-browser`, with 11 checks against an
  in-memory server with the same rules.
- `tests/lint.txt` and `tests/ci.txt`: fmt, clippy with and without
  `--all-features`, the frontend, public-scrub and validation-scripts jobs,
  and the 47 validation gates. On this machine the frontend job cannot pass
  in one run: with `PUPPETEER_EXECUTABLE_PATH` set, the project and
  conversation context browser smoke fails inside the job (it passes on its
  own, and fails the same way on #788, which has none of this change and is
  green in GitHub CI); without it, the divergence view smoke cannot find a
  browser. Run both ways, every step of the job passes in one run or the
  other, the new watched folders smoke in both.

The SmolLM3 runtime seal pins the git blob of `src/api/mod.rs`, which this
change edits, so it is re-pinned in the same commit.

## Limits

- Changes are found by polling every 30 seconds, not by file-system events.
  A file rewritten with the same size inside one modification-time tick of
  its file system is not seen until it changes again.
- A file skipped once (no text, unparseable) is not retried until it changes.
- Deleting a watched folder's document from the library brings it back on
  the next scan; the folder is the source of truth.
- Dropping a folder uploads its files once; it does not watch the folder.
  Browsers do not reveal a dropped folder's path, so watching needs the path.
