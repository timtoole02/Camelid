# TOOLCALL_DIAG.md — diagnosing the malformed tool-call arguments

The observed failure (Llama 3.2 1B Q8_0, agent mode round-trip):

```
{"name":"read_file","parameters":{"properties":"{'path':'notes.txt'}","required":"[('path','notes.txt')]","type":"object"}}
```

i.e. the model echoed the **JSON-Schema** (`properties`/`required`/`type`) as its `parameters`
instead of producing `{"path":"notes.txt"}`. Before blaming model size, the cheaper hypotheses
were falsified (Phase 0).

## 1. The rendered prompt (offline, no model load)

The unit test `tool_render_nested_vs_flat_diagnostic` (`src/api/mod.rs`) renders the real Llama 3
metadata chat template with the tool defs. With the **OpenAI-nested** tools I was sending
(`{ "type":"function", "function":{…} }`), the template emits:

```
Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.

{
    "function": {
        "description": "Read a file",
        "name": "read_file",
        "parameters": {
            "properties": { "path": { "type": "string" } },
            "required": [ "path" ],
            "type": "object"
        }
    },
    "type": "function"
}
```

Two problems are now obvious in the prompt the model actually receives:
- The OpenAI **envelope leaks** (`"type":"function"`, nested `"function"`) — the tool's top-level
  keys are `type`/`function`, which do **not** match the `{"name":…, "parameters":…}` response
  format the template just asked for.
- The tool's `parameters` field IS the **JSON schema** (`properties`/`required`/`type`), and the
  model is asked to respond with a `parameters` field — so a weak model conflates "the parameters
  shape shown" with "the parameters I should emit" and copies the schema. The nested envelope
  makes this worse, not better.

This is the classic "tools rendered as a raw JSON-Schema object that the model mirrors" symptom
the spec called out.

## 2. Parser field mapping — correct

`tool_parse::call_from_obj` reads the call's arguments from `parameters` / `arguments` (and
unwraps a `function` envelope if the model emits one). It does **not** key off the schema's
`properties`. The 1B's output genuinely had `parameters = {properties,required,type}` — the
parser read `parameters` correctly; the model put the schema there. Not a parser bug.

## 3. A/B (nested vs flat) — render was the problem

Rendering the **flat function** form (`{ "name", "description", "parameters" }`, no `type`/
`function` envelope) produces the canonical Llama 3.1 tool prompt — what Meta's examples and
llama.cpp / vLLM actually feed the model, with `name`/`parameters` at the tool's top level
(mirroring the requested response format). The diagnostic test asserts the nested form leaks
`"type": "function"` and the flat form does not.

## Conclusion — **render bug, fixed**

The wire format stays OpenAI-standard (nested `tools`), but the server now **normalizes** each
tool to its flat `function` object before threading it into the model's chat template
(`render_chat_prompt_for_tokenization_with_tools`), matching llama.cpp / vLLM. The change is
localized to the **tools-present** render path; the shared render chain stays byte-identical for
`tools=None` (the 438 lib tests still pass), per the spec's stop-and-report boundary (not
triggered — no shared-chain refactor needed).

Whether even the 1B then emits *usable* arguments is a separate **model-capability** question,
decided by the `agent-eval` promotion harness (Phase 2) on a quiet box — not assumed here. What
is now proven: the template path is **canonical-correct**, so a future big-model failure cannot be
a leftover render bug.
