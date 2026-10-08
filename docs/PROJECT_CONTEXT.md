# Project and conversation context

**Projects** groups related chats with reusable instructions and reference files.
Open **Projects → New project**, add a name, instructions, and optional text files,
then choose **New chat** on the project card. Its conversation list includes
archived chats; deleting a project leaves those conversations in Chat history.

Use **Conversation context** above the chat composer to assign an existing chat
to a project, choose which instructions and project files it inherits, or add
instructions and files just for that conversation. On short landscape screens,
the same panel opens from **Context** in the composer toolbar. Changes are explicit:
**Save context** applies them; **Cancel** leaves the saved context unchanged.

## What reaches the model

The panel's **Included in the next request** list shows the exact instruction and
reference messages, in order. Global instructions come from Generation controls;
Camelid's existing automatic code instructions are shown when the draft triggers
them. Project instructions follow, then conversation instructions, with a stated
preference for the conversation's instructions when preferences conflict. Turn off
inheritance to omit global or project instructions entirely.

Selected reference files follow as labelled user-role data messages, with their
names and contents quoted as JSON strings. They are not executed or rendered as
HTML and are not promoted into system instructions. The chat history and current
question follow the reference messages. As with other model instructions, actual
adherence depends on the selected model.

The context meter and send-time compaction use the composed messages, including
these sources and the newest image. Selected context is retained during compaction;
compaction only removes eligible earlier assistant/tool turns. The meter reports
estimates, not tokenizer measurements. Document retrieval and Web Auto can add
more context during send-time preflight, where the final request is checked again.

Context is composed in the shared send path, including regenerate, edit/resend,
and continue. An MCP run freezes its context for all approval and continuation
rounds. Context edits are disabled while a response or tool run is active.

## Knowledge collections

Uploads, library searches, collections, indexing status, and source verification use
the backend selected under **Settings → API base URL**, with its configured API key.
Changing that backend refreshes the library lists. A backend on another origin must
allow the UI's origin with `--cors-origin`; local folder access keeps its existing
same-origin loopback restrictions.

A knowledge collection is a named set of documents from the Knowledge Library.
Open **Attach → Collections** in the composer to create, rename, or delete a
collection, add library documents to it or remove them, or upload files straight
into it. A document can belong to any number of collections. Removing it from a
collection, or deleting the collection, leaves the document in the library;
deleting the document removes it from every collection.

A project searches the collections chosen in its editor, and its chats search
them too unless the chat turns one off: clear it under **Conversation context**,
or remove its chip in the composer. A chat can also search collections of its
own, from the same panel or with **Search this collection in this chat** in the
library. A chat or project searches at most 16 collections.

When a message is sent, the chat reads the collections again and runs one
document search over its attached documents and the members of every searched
collection together, so a message searches each document once however many
collections hold it. The results reach the model the same way as attached
documents' results, as quoted source excerpts with verifiable citations. The
composer shows a chip for each searched collection with its size, or its
indexing progress while its documents are embedded (indexing, waiting to index,
or indexing stopped, with the error, until the next upload or a restart); the
sent message names each collection and how many of the passages used came from
its documents.

A deleted collection shows as **Collection unavailable** and is not searched;
remove its chip to clear it. If the collections cannot be read, the message is
sent without them and says so.

## Whole library

**Attach → Whole library** makes a chat search every document in the library,
with nothing to attach or collect first. Only passages close enough in meaning
to the message are used, so a message about something the library does not
cover is sent without document context; attached documents and the chat's
collections are searched as well, in full. The composer shows a **Whole
library** chip with the library's size, or its indexing progress (indexing,
waiting to index, or indexing stopped, with the error), since a document is
found this way only once it is indexed. The sent message says how
many passages came from the library beyond what was attached or collected. The
switch is saved with the conversation, and a new chat starts with it off. It
needs search by meaning: without the embedding model the composer says so, and
a message is sent without the library, still searching what is attached. How
"close enough" was chosen is in
[embeddings](architecture/EMBEDDINGS.md#knowledge-library-integration).

## Files, storage, and scope

- The Knowledge Library reads PDF, Word (`.docx`), HTML (`.html`, `.htm`), plain
  text, Markdown, CSV, JSON, and source code (`.rs`, `.py`, `.js`, `.mjs`, `.cjs`,
  `.jsx`, `.ts`, `.tsx`, `.go`, `.java`, `.kt`, `.kts`, `.swift`, `.c`, `.h`, `.cc`,
  `.cpp`, `.cxx`, `.hpp`, `.hh`, `.cs`, `.rb`, `.php`, `.scala`, `.lua`, `.dart`,
  `.sh`, `.bash`, `.zsh`, `.sql`). A watched folder picks up only these types. An
  HTML page is read as its text: tags, scripts, styles, and comments are dropped,
  and its headings become Markdown headings, so each section is chunked and cited
  on its own. Its citations still bind the original file's bytes. A text or code
  file with a NUL byte in its first 8 KiB is binary, not text, and is skipped as
  having no text. A file over 64 MB is refused, whether uploaded or watched.
- Reference files are static UTF-8 copies: text, Markdown, code, CSV, or JSON.
  Re-add a file to update its contents. PDF and DOCX extraction continue to use
  the separate document attachment controls.
- Each project and conversation can have eight files, at most 32 KiB each, with
  a combined 96 KiB limit for its instructions and files. Instructions have a
  12,000-character limit. Up to 24 projects can be saved.
- Projects and unsent draft context use `camelid.projects` and
  `camelid.draftContext` in Camelid's UI storage. Saved conversations carry their
  context configuration. Desktop uses its existing native UI storage document;
  browser sessions use local storage for that origin. Browser quota failures
  during context saves appear as an error rather than a successful save.
- New chat clears conversation-specific context. New chat from a project card
  starts with that project's context. Older conversations without a context field
  keep global defaults and never inherit an unrelated new-chat draft.
- Updating a project affects future messages in its linked chats. Removing a
  project displays **Project unavailable** in those chats and omits its context.
  No conversation is deleted and no replacement project is chosen automatically.
- Context is sent to the selected chat endpoint. Instructions and reference
  files work on the LAN chat surface because they are entirely local to the UI
  and add no engine filesystem, tool-execution, indexing, or synchronization API.
- Knowledge collections live in the engine's library database, next to the
  documents they group; chats and projects store only collection ids. They are
  part of the library API, which the LAN chat surface does not serve, so that
  surface neither offers nor searches them and never requests them. A chat's
  saved collections apply again on the full surface. The same holds for
  whole-library search.
- Conversation exports retain their existing transcript-only field whitelist.
  Project and conversation context are excluded, and imports do not activate
  context from an imported file. Clearing UI storage removes saved context.

## Validation

Run `npm run smoke:project-context` and, after `npm run build`,
`npm run smoke:project-context-browser`,
`npm run smoke:knowledge-collections-browser` and
`npm run smoke:library-search-browser` from `frontend`. The browser suites use
local deterministic API fixtures to verify actual request payloads, inheritance,
file selection, persistence, project isolation, MCP continuation, the context
budget gate, project deletion, and desktop/mobile layouts, and that collections
are managed, searched per chat and per project, left out once deleted, and absent
from the LAN chat surface, and that whole-library search is saved per chat,
reported on each message, and falls back plainly without search by meaning. These checks verify
UI behavior and request composition; they are not model-quality evidence.
