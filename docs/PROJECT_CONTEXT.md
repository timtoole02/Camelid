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

## Files, storage, and scope

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
- Context is sent to the selected chat endpoint. The feature works on the LAN
  chat surface because its project management is entirely local to the UI.
  It adds no engine filesystem, tool-execution, indexing, or synchronization API.
- Conversation exports retain their existing transcript-only field whitelist.
  Project and conversation context are excluded, and imports do not activate
  context from an imported file. Clearing UI storage removes saved context.

## Validation

Run `npm run smoke:project-context` and, after `npm run build`,
`npm run smoke:project-context-browser` from `frontend`. The browser suite uses
local deterministic API fixtures to verify actual request payloads, inheritance,
file selection, persistence, project isolation, MCP continuation, the context
budget gate, project deletion, and desktop/mobile layouts. These checks verify
UI behavior and request composition; they are not model-quality evidence.
