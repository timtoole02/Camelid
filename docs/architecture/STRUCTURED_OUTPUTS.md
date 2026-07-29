# Structured outputs (LLGuidance)

Status: shipped on `/v1/chat/completions` for non-streaming requests.

Camelid uses the Rust-native `llguidance` and `toktrie` crates, pinned to
`1.7.6`. Constraints compile at request time against a canonical byte tokenizer
and again against the loaded model's exact token pieces before inference. A
schema or grammar that cannot be enforced is a typed error; Camelid never drops
it and continues with unconstrained decoding.

## Request forms

OpenAI response formats:

```json
{"response_format":{"type":"json_object"}}
```

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
```

LLGuidance/Lark CFG extension:

```json
{
  "response_format": {
    "type": "grammar",
    "grammar": "start: \"yes\" | \"no\""
  }
}
```

For llama.cpp client compatibility, the schema or grammar may instead be sent
at the top level:

```json
{"json_schema":{"type":"string","pattern":"^[a-z]+$"}}
```

```json
{"grammar":"start: \"yes\" | \"no\""}
```

`response_format`, `json_schema`, and `grammar` are mutually exclusive.

## JSON Schema and CFG contract

Camelid accepts the JSON Schema surface LLGuidance 1.7.6 can compile, including
scalar roots, objects, arrays, unions/combinators, references, regular-expression
string constraints, numeric bounds, and enum/const constraints. LLGuidance's
own complexity limits are kept enabled. Unsupported, invalid, or over-complex
schemas are rejected with the upstream compiler diagnostic.

CFG input accepts LLGuidance's Lark dialect, `%llguidance` documents, and
serialized grammar lists. Invalid CFGs fail the same request-time gate.

## Decode invariants

1. Token bytes come directly from GGUF vocabulary pieces. Byte-level BPE and
   byte-fallback fragments are never decoded as standalone UTF-8 strings.
2. LLGuidance computes a compact token mask by walking its token trie. The
   sampler receives that mask before every constrained token.
3. The selected token is committed directly to the same parser state. Mask and
   commit cannot diverge through a decode/re-encode conversion.
4. Constraint compilation completes before the first model forward.
5. Speculative decoding, the resident-GPU sampling shortcut, and prompt-prefix
   caching remain disabled while a constraint is active.
6. Constrained output is not reclassified as a tool call.
7. Parity receipts normalize all accepted request forms into the existing
   `response_format` receipt field so replay compiles the same constraint.
8. `stream:true` plus any constraint remains a typed 400. Streaming parser
   integration is a separate feature and is never silently approximated.

## Error taxonomy

| Class | HTTP | `error.code` |
|---|---:|---|
| Missing/ambiguous request fields | 400 | `invalid_request_error` |
| Schema/CFG compile refusal | 400 | `unsupported_parameter` |
| Loaded tokenizer cannot represent the constraint | 400 | `unsupported_parameter` |
| Mask evaluation fails during decoding | 422 | `constraint_evaluation_failed` |
| No vocabulary token can extend the grammar | 422 | `constraint_unsatisfiable` |
| A sampled token cannot be committed | 422 | `constraint_commit_failed` |

The compile refusals are request facts. The 422 classes are model-tokenizer
facts and therefore remain distinct for callers that want to retry with another
model.
