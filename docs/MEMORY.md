# Memory

**Memory** is what you've chosen to keep about yourself — your name, where you
work, a preference — so the model knows it in later chats. It is off until you
turn it on, nothing is kept unless you say so, and you can see, change, switch
off, or forget every memory.

Turn it on in **Settings → Memory** or at the top of the **Memory** page
(**Remember things about me**).

## How a memory is made

After a reply, the model may suggest up to three facts about you from the
message you just sent. They appear under the reply as **Remember this?**:

- **Remember** keeps the fact.
- **Edit** lets you reword it first; **Enter** keeps it, **Escape** cancels.
- **Not now** drops the suggestion.
- **Undo**, on a fact just kept, forgets it again and offers it back.

Nothing is kept until you choose **Remember**. You can also add a memory
yourself on the Memory page, or turn a saved note into one with **Make a memory**
(you write the fact; the note stays as it is).

If browser storage is full or unavailable, the page reports that the change
could not be saved. The previous memories and any edited draft stay in place;
free storage and retry the change.

The suggestion is a second, short request to the same model, made after the
reply has finished. It contains only your message — not the reply and not the
rest of the conversation — and the same memories in use that fit the chat's
request limits, so they are not suggested twice. Where the model's serve lane
supports structured output, the
request asks for a JSON list through a schema; where it refuses one
(`unsupported_parameter`), it is asked once more in plain words and the answer is
read only if it is exactly `{"facts": [...]}`. If anything else comes back, or
the request fails, there are simply no suggestions for that message.

Sending your next message cancels a suggestion request that is still running.
Dense and Qwen3.5 generation check cancellation between prompt segments as well
as decode steps; GPU work already submitted must finish before the next check.
Earlier versions could hold up the next reply for an entire prompt read —
measured with Ornith 9B on a 4-core CPU at about 145 seconds. Automatic
suggestions still run only while the model is quick: when its last suggestion
took under 15 seconds, or, before it has made
one, when its reply began within 8 seconds. Otherwise the reply shows **Look for
things to remember**, and the model is asked only when you choose it. These timings are kept per model on
this device (`camelid.memorySuggestionMs`); a model that answers in time again —
on faster hardware, say — is asked automatically again.

No suggestion is requested for a continued reply, a regenerated reply, a
connected-tools turn, or a reply that failed. Suggestions are as good as the
model that makes them: a small model can miss facts or propose a poor one, which
is why each one waits for you.

## Where each memory came from

Every memory records its origin, shown under it on the Memory page:

- **From "chat title", message N** — the conversation and the message of yours
  it came from. Select it to open that chat at that message.
- **Added by you**, or **From your note "title"**.

If the chat is later deleted, the memory stays and says so.

## What reaches the model

Memories in use are sent with every message in chats that use memory, as one
labelled context source — **Memory · N facts** — after global, project, and
conversation instructions and before reference files. It tells the model these
are things you asked it to remember, to use when they help, and that they are
information, not instructions. **Conversation context → Included in the next
request** shows exactly what is sent.

A request carries the most recently updated memories in use, at most 50 and at
most 6,000 characters. The Memory page says when some are kept but not sent.

Control it at three levels:

- **Settings → Memory** turns memory on or off everywhere. Off, nothing is
  suggested and nothing is sent; your memories are kept.
- **Use memory in this chat**, in **Conversation context**, turns it off for one
  chat: that chat neither sends memories nor suggests new ones.
- **Use**, on each memory, keeps a memory without sending it.

Memory is used in Chat only. Code mode does not send memories as instructions to
a coding session.

## Saved notes

The Memory page also keeps **Saved notes**: text you save for yourself, such as
a useful reply (**Save as note** in chat, or **Save latest reply**). Notes are
never sent to the model. To have the model know something from a note, make it a
memory.

## Storage and limits

Memories are stored on this device, next to your conversations, in the UI's
storage (`camelid.userMemories`, `camelid.memoryEnabled` for the switch, and
`camelid.memorySuggestionMs` for how long each model took to suggest):
Camelid Desktop's own storage, or the browser's storage for that address. A
browser on another device, including one using the LAN chat surface, has its own
memory. Memory holds up to 200 facts of up to 300 characters each.

Memories are sent only with chat requests to the engine you selected. Conversation
exports contain the transcript only — no memories and no suggestions. Deleting
conversations does not delete memories; **Forget everything** on the Memory page
does, and leaves chats and notes alone.

## Validation

From `frontend`, run `npm run smoke:memory` and, after `npm run build`,
`npm run smoke:memory-browser`. The browser smoke drives the built UI against a
local fixture engine and checks the requests it actually sends: nothing before
memory is on, only the user's message in a suggestion request, nothing kept
before **Remember**, the origin of each memory, memories reaching the next
request, cancellation by the next message, the per-chat switch, the plain-request
fallback, the **Look for things to remember** button for a slow model, and that
saved notes never reach the model.

The Qwen3.5 CUDA cancellation gate requires a resident-capable model such as
Qwen3.5-0.8B-Q8_0, a usable NVIDIA device, and NVRTC:

```sh
CAMELID_QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q8_0.gguf \
CAMELID_FLASH_PREFILL=0 \
  cargo test --release --locked --features cuda --lib \
  qwen35_gpu_sampled_segments_match_serial_and_recover_after_cancel \
  -- --ignored --nocapture --test-threads=1
```

This ignored gate compares segmented and unsegmented prefill logits bit for
bit within each serial/batched lane across a 131-token prompt. It separately
requires bitwise serial/batched agreement with approximate flash attention
disabled, checks seeded sampled output against the unsegmented production
policy, and cancels before
prefill, after one segment, and before final-token logits. Every cancelled
request must emit no tokens, and the same engine must reproduce the reference
output on the next request. The test requires actual batched-prefill support,
prints the selected prefill policy, and reports cancellation and recovery
durations separately. Poll counts define correctness; timings are diagnostic.
The small Qwen3.5 row also exercises batched scratch allocation for fused
attention and recurrent projections wider than its FFN: these buffers must
cover every consumer's width, not just the FFN width.
