#!/usr/bin/env python3
"""Capture the Gemma 4 TOOLS chat-template shape pack.

Renders the GGUF-embedded Jinja chat template (the reference the weights were
trained on) over a fixed set of tool-calling conversation shapes and freezes
each rendering into qa/gemma4/tool_template_shapes_v1.json. Camelid's
`gemma4_chat_prompt_with_tools` marker renderer must reproduce every
`rendered_prompt` byte-for-byte (in-src pack-lock test in src/api/mod.rs).

The renderer is Jinja2 proper, not a re-implementation: the template string is
read live out of the GGUF header, so the pack cannot drift from the artifact.
BOS is deliberately rendered as '' — Camelid adds BOS at tokenization time
(same convention as qa/gemma4/template_shapes_v1.json).

Requires: python3 with jinja2 (any venv). Usage:
    python3 scripts/capture-gemma4-tool-template-shapes.py <model.gguf> [out.json]

Envelope notes (what the pack deliberately does NOT cover):
- tool results are given in call order with matching tool_call_id (Camelid's
  ChatMessage keeps neither ids nor names, so its renderer resolves response
  names positionally; in-order histories render identically to the reference)
- no `reasoning`/`reasoning_content` on history messages
- single leading system message (Camelid folds multiple leading system turns)
"""

import hashlib
import json
import struct
import sys


def read_gguf_chat_template(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))
    struct.unpack("<Q", f.read(8))
    (n_kv,) = struct.unpack("<Q", f.read(8))

    def rd_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def rd_val(t):
        scalars = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                   6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
        if t in scalars:
            fmt = scalars[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 7:
            return struct.unpack("<B", f.read(1))[0] != 0
        if t == 8:
            return rd_str()
        if t == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            return [rd_val(et) for _ in range(n)]
        raise ValueError(f"bad gguf value type {t}")

    for _ in range(n_kv):
        key = rd_str()
        (t,) = struct.unpack("<I", f.read(4))
        value = rd_val(t)
        if key == "tokenizer.chat_template":
            return value
    raise ValueError("no tokenizer.chat_template in GGUF header")


# The exact agent-battery tool definitions (src/chat/tools.rs specs, Full
# profile, shell enabled, no net, no subagents) — the schemas camelid
# agent-eval sends. Locking the declaration renderer over all of them covers
# string/integer/boolean/enum/array-of-object parameter shapes.
BATTERY_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read UTF-8 text. Optional start_line/max_lines select an excerpt; `N | ` prefixes are not file content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 200}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List directory entry names.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Find a literal substring in file contents; no regex or globs.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "update_plan",
        "description": "Record a task plan for a genuinely multi-step goal: an ordered list of short steps, each pending | in_progress | done. Never call this tool twice consecutively. Perform file/shell/delegation work between updates; the run permits at most two plan updates. The user sees it. It has no side effects.",
        "parameters": {"type": "object", "properties": {
            "steps": {"type": "array", "items": {"type": "object", "properties": {
                "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                "text": {"type": "string"}},
                "required": ["status", "text"]}}},
            "required": ["steps"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite one workspace file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace exact file text (`N | ` prefixes excluded); replace_all changes every match.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean"}},
            "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a workspace shell command. Put source in files; use this for builds, tests, apps, installs, git, or bulk work.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}},
            "required": ["command"]}}},
]

READ_FILE_TOOL = BATTERY_TOOLS[0]
WRITE_FILE_TOOL = BATTERY_TOOLS[4]
LIST_DIR_TOOL = BATTERY_TOOLS[1]

FIXTURE = "alpha\nbeta\ngamma\n"


def shapes():
    return [
        {
            "id": "tools-user-plain",
            "thinking": False,
            "tools": [READ_FILE_TOOL],
            "messages": [
                {"role": "user", "content": "Read the file notes.txt and tell me how many lines it has."},
            ],
        },
        {
            "id": "tools-system-user-plain",
            "thinking": False,
            "tools": [READ_FILE_TOOL, LIST_DIR_TOOL, WRITE_FILE_TOOL],
            "messages": [
                {"role": "system", "content": "You are a careful file assistant."},
                {"role": "user", "content": "List the current directory."},
            ],
        },
        {
            "id": "tools-user-thinking",
            "thinking": True,
            "tools": [READ_FILE_TOOL],
            "messages": [
                {"role": "user", "content": "Read notes.txt."},
            ],
        },
        {
            "id": "tools-roundtrip-single-call",
            "thinking": False,
            "tools": [READ_FILE_TOOL],
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "How many lines does notes.txt have?"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {
                        "name": "read_file",
                        "arguments": {"path": "notes.txt"}}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": FIXTURE},
            ],
        },
        {
            "id": "tools-roundtrip-two-rounds",
            "thinking": False,
            "tools": [READ_FILE_TOOL, WRITE_FILE_TOOL],
            "messages": [
                {"role": "user", "content": "Copy notes.txt to copy.txt."},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {
                        "name": "read_file",
                        "arguments": {"path": "notes.txt"}}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": FIXTURE},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {
                        "name": "write_file",
                        "arguments": {"path": "copy.txt", "content": "alpha\nbeta\ngamma\n"}}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "wrote 18 bytes"},
            ],
        },
        {
            "id": "tools-roundtrip-final-answer",
            "thinking": False,
            "tools": [READ_FILE_TOOL],
            "messages": [
                {"role": "user", "content": "How many lines does notes.txt have?"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {
                        "name": "read_file",
                        "arguments": {"path": "notes.txt"}}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": FIXTURE},
                {"role": "assistant", "content": "notes.txt has 3 lines."},
                {"role": "user", "content": "Thanks. What is the first line?"},
            ],
        },
        {
            "id": "tools-declaration-battery",
            "thinking": False,
            "tools": BATTERY_TOOLS,
            "messages": [
                {"role": "user", "content": "Create greeting.txt containing: hello there"},
            ],
        },
        {
            "id": "tools-dangling-call",
            "thinking": False,
            "tools": [READ_FILE_TOOL],
            "messages": [
                {"role": "user", "content": "Read notes.txt."},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {
                        "name": "read_file",
                        "arguments": {"path": "notes.txt"}}}]},
            ],
        },
        {
            "id": "tools-call-argument-types",
            "thinking": False,
            "tools": [READ_FILE_TOOL],
            "messages": [
                {"role": "user", "content": "Exercise every argument type."},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {
                        "name": "read_file",
                        "arguments": {
                            "path": "notes.txt",
                            "start_line": 2,
                            "verbose": True,
                            "tags": ["a", "b"],
                            "extra": {"depth": 1, "label": "x"},
                        }}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": "beta\ngamma\n"},
            ],
        },
    ]


def main():
    import jinja2

    gguf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "qa/gemma4/tool_template_shapes_v1.json"
    template_text = read_gguf_chat_template(gguf)
    env = jinja2.Environment()
    template = env.from_string(template_text)

    rendered_shapes = []
    for shape in shapes():
        rendered = template.render(
            messages=shape["messages"],
            tools=shape["tools"],
            enable_thinking=shape["thinking"],
            add_generation_prompt=True,
            bos_token="",
        )
        rendered_shapes.append({
            "id": shape["id"],
            "thinking": shape["thinking"],
            "tools": shape["tools"],
            "messages": shape["messages"],
            "rendered_prompt": rendered,
        })

    pack = {
        "schema": "camelid.gemma4.tool-template-shapes.v1",
        "pack_id": "gemma4-tool-template-shapes-v1",
        "description": (
            "Gemma 4 TOOLS chat-template shape pack: the reference rendering "
            "(GGUF-embedded Jinja template, rendered by Jinja2) for the tool-calling "
            "envelope — tool declarations in the system turn, <|tool_call> blocks, "
            "<|tool_response> blocks in a continued model turn, and the "
            "add_generation_prompt forms. Camelid's gemma4_chat_prompt_with_tools "
            "must reproduce rendered_prompt exactly. Envelope: tool results in call "
            "order; no history reasoning fields; single leading system message. "
            "Out-of-envelope shapes are NOT claimed."
        ),
        "reference": "GGUF tokenizer.chat_template rendered with Jinja2 (bos_token='')",
        "template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
        "shapes": rendered_shapes,
    }
    with open(out, "w") as fh:
        json.dump(pack, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {out} ({len(rendered_shapes)} shapes, template sha256 {pack['template_sha256'][:12]}…)")


if __name__ == "__main__":
    main()
