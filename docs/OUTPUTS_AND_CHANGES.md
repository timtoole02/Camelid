# Output previews, downloads, and file review

Generated code blocks now have **Preview** and **Download** actions. Structured
JSON replies have the same controls, and embedded MCP text resources and images
appear as downloadable output cards in tool results.

HTML and SVG render in an isolated static preview with scripts, network requests,
forms, and nested frames disabled. Markdown renders with Camelid's normal safe
Markdown renderer. JSON is formatted for inspection; CSV renders as a table with
up to 100 rows and 40 columns. Other text formats show their source. Downloads
retain the full output even when a table preview is limited. Preview is limited
to 256 KB. Unsupported MCP resource links are not fetched automatically.

## Review a generated file

1. Choose **Review file change** on a completed text output, or open
   **Changes → New review** to enter a proposal manually.
2. Choose a local workspace folder and a relative file path. The parent folder
   must already exist. The proposed content represents the complete destination
   file, rather than a patch fragment.
3. Choose **Prepare review**. Camelid reads the current file and saves its
   original and proposed contents in a review journal. The destination is not
   modified at this step.
4. Inspect **Diff summary**, **Before**, and **After**. The summary is bounded;
   the Before and After tabs retain the complete contents.
5. Choose **Approve & apply** or **Reject**. Each review can be decided once.
   Approval writes precisely the After contents. If the file changed since
   preparation, Camelid refuses to overwrite it; prepare a fresh review.
6. An applied review offers **Undo this change**. It restores the original text,
   or removes a file created by that review. Undo also refuses to overwrite
   subsequent edits. There is no force-undo option.

Saved reviews remain available after restarting Camelid. Removing a saved review
removes its history and undo snapshot, without modifying the destination file.
The UI asks for confirmation before discarding that history.

## Scope and storage

The Changes interface handles local UTF-8 file creation and replacement, up to
256 KB per file. It rejects path traversal, symbolic links, Git metadata, and
Camelid's workspace state directories. Existing file permissions are preserved.
It does not run commands or apply arbitrary patch text. Workspace's agent tool
profile remains read-only; a file output becomes a writable proposal only when
the user chooses its destination in Changes.

External MCP actions retain their separate per-call approval flow. They do not
receive an Undo button because Camelid cannot promise to reverse remote side
effects. Embedded output previews and downloads do not execute MCP calls.

The local engine keeps up to 128 review journals under `change-reviews` beside
its Workspace memory database. Each journal includes the original and proposed
file contents, so it can contain sensitive source text. On Unix, the directory
is owner-only and the journal files use owner-only permissions. A journal is
flushed before applying or undoing a file. Recovery after interruption inspects
the destination and resolves the recorded operation; it never repeats a write
automatically. Review journals are not included in conversation exports.

All `/api/changes` routes require a loopback-bound engine, same-origin browser
metadata, and `X-Camelid-Changes: 1`, in addition to configured API credentials.
The Changes view and its actions are unavailable on the LAN chat-only surface.
Previews and downloads remain available there.

| Route | Operation |
| --- | --- |
| `GET /api/changes` | List saved review metadata |
| `POST /api/changes` | Prepare `{ workspace, path, content, source? }` |
| `GET /api/changes/:id` | Read original/proposed content and diff summary |
| `POST /api/changes/:id/decision` | Approve or reject with `{ approved: boolean }` |
| `POST /api/changes/:id/undo` | Restore the original state if the file still matches |
| `DELETE /api/changes/:id` | Remove the saved review and undo snapshot |

## Validation

```sh
cargo test --lib api::changes::tests
cargo clippy --lib -- -D warnings
cd frontend
npm run smoke:outputs-changes
npm run build
npm run smoke:outputs-changes-browser
```

The Rust tests use temporary files to exercise preparation, rejection, apply,
replay refusal, durable undo, version conflicts, interrupted-operation recovery,
path confinement, and route authorization. The browser fixture checks isolated
previews, exact downloaded bytes, file review, approval, history after reload,
undo conflicts, undo, and desktop/mobile layout. It uses a deterministic chat
response; these checks do not certify any new model capabilities.
