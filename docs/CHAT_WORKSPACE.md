# Chat workspace

The composer keeps **Model**, **Tools**, **Attach**, **Options**, and **Send**
close to the message. Options contains Thinking, automatic web research,
verification receipts, structured output, token probabilities, and generation
controls. Available options follow the loaded model and engine capabilities.
Active settings appear as removable chips. Changes apply to the next message;
settings that affect an active tool turn remain locked until it finishes.

**Attach** offers documents for local retrieval, knowledge collections (named
sets of library documents a chat searches together), **Whole library** (every
library document, keeping only passages close in meaning to the message), and
images when a vision model is ready. Conversation instructions and reference files remain under
**Conversation context**. See [project context](PROJECT_CONTEXT.md).

## Context usage and trimming

The lower-right context bar is green below 80%, yellow from 80% through 94%, and
red from 95%. The solid portion estimates prompt usage; the hatched portion is
room reserved for a reply. An oversized reply reservation is clamped to the
space available and does not make a small prompt look full. Open the bar for
token estimates, the tested-context marker, and trimming controls.

Chat defaults to **Trim automatically at 80%**. **Trim what gets sent** manually
omits eligible older assistant/tool groups while preserving user messages,
instructions, recent messages, and complete call/result pairs. It does not
delete transcript history. **Send it all** restores the full send history.
Large user messages or reference files may still need to be shortened because
trimming preserves them. **New chat** starts fresh conversation history; global
instructions and other configured context can still apply.

## Tools and files

[Connected tools](MCP.md) have one card per call, with approval and results in
the same place. [Conversation files](OUTPUTS_AND_CHANGES.md) provides previews,
source inspection, filenames, downloads, and the existing file-review workflow.
