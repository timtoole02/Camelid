# Sampling and API Reverse Engineering

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

Source: sampling/API behavior recon focused on common OpenAI-compatible local-runtime behavior and ForgeLocal's current integration needs.

## Summary

camelid should keep sampling independent from inference execution and HTTP serving. The inference engine should produce logits; the sampler chooses the next token from logits and token history; the HTTP/API layer translates OpenAI-compatible requests into generation settings and returns truthful responses.

Initial implementation should support deterministic greedy sampling first, then seeded random sampling with temperature, top-k, top-p, and repeat penalty. Streaming should be planned as SSE but should continue returning `501 stream_not_supported` until real token streaming exists.

## Key sampling concepts

Local inference runtimes commonly use a composable sampler chain. User-facing sampling parameters can map to penalties, grammar constraints, top-k, top-p, min-p, typical, temperature, mirostat, distribution sampling, or greedy sampling.

For camelid, do not implement the whole chain at once. Implement a small, deterministic subset with clear tests:

1. repeat penalty
2. temperature
3. top-k
4. top-p
5. seeded categorical sampling
6. greedy mode when temperature is zero or deterministic mode is requested

Advanced features to defer:

- grammar constraints
- JSON-schema constrained decoding
- mirostat
- DRY penalty
- XTC
- min-p / typical-p
- speculative decoding

## Sampling order

Recommended first order:

1. Start with logits for vocab.
2. Apply repeat penalty using recent token history.
3. If deterministic/greedy or `temperature == 0`, return argmax.
4. Divide logits by temperature.
5. Keep top-k if configured.
6. Apply softmax.
7. Keep nucleus/top-p if configured.
8. Renormalize.
9. Sample with seeded RNG.

Keep this order documented and tested. It does not need to perfectly match every advanced sampler interaction yet, but it should be stable and explicit.

## Sampler data model

Add `src/sampling/`:

```rust
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub seed: Option<u64>,
    pub deterministic: bool,
}

pub struct SamplingState {
    rng: StdRng,
    history: Vec<u32>,
}

pub trait Sampler {
    fn sample(&mut self, logits: &[f32], config: &SamplingConfig) -> Result<u32>;
    fn accept(&mut self, token: u32);
    fn reset(&mut self);
}
```

A simpler first implementation can combine config and state:

```rust
pub struct BasicSampler {
    config: SamplingConfig,
    rng: StdRng,
    history: Vec<u32>,
}
```

## Config defaults

Suggested defaults compatible with common local inference APIs:

```text
temperature = 0.8
top_p = 0.95
top_k = 40
repeat_penalty = 1.1
repeat_last_n = 64
seed = random unless deterministic requested
deterministic = false
```

For tests, use explicit seed and small logits.

## Repeat penalty behavior

For each token in the recent history window:

- if `logit[token] < 0`, multiply by penalty
- else divide by penalty

This matches common local-runtime penalty behavior.

Validation:

- penalty must be positive
- `repeat_last_n = 0` disables penalty window
- out-of-range token ids in history should be ignored or rejected internally; prefer avoiding invalid history insertion

## Top-k behavior

- `top_k = None` or `top_k == 0` disables top-k.
- Otherwise keep only the highest `k` logits.
- If `k >= vocab_size`, no filtering.
- Preserve deterministic tie-breaking by token id where possible.

## Top-p behavior

- Compute probabilities after temperature/top-k.
- Sort candidates by descending probability.
- Keep the smallest prefix whose cumulative probability >= `top_p`.
- Always keep at least one candidate.
- Renormalize before sampling.

## Deterministic behavior

Determinism requirements:

- Same seed + same logits + same history => same sampled token.
- Greedy mode always returns the lowest token id among tied max logits.
- Tests should not depend on platform floating point edge cases where avoidable.

## Stop conditions

The generation loop, not the sampler, should enforce stop conditions:

- EOS/EOT token
- max new tokens
- stop strings
- context length overflow
- client cancellation
- internal inference error

Represent finish reasons in a stable enum:

```rust
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}
```

Map to OpenAI-compatible strings:

- `Stop` -> `"stop"`
- `Length` -> `"length"`
- `Cancelled` -> `"stop"` or backend-specific metadata later
- `Error` -> error response, not a normal completion

## OpenAI-compatible API structs

Current camelid API has a minimal `ChatCompletionRequest`. Expand toward:

```rust
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stop: Option<StopSpec>,
    pub seed: Option<u64>,
}

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
```

Support roles initially:

- `system`
- `user`
- `assistant`

Reject tool/function/multimodal messages until supported.

Response shape:

```rust
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    pub camelid: BackendMetadata,
}
```

During no-inference phase, continue returning `501 not_implemented` instead of fake output.

## Error envelope

Use the existing OpenAI-like envelope:

```json
{
  "error": {
    "message": "Human-readable explanation.",
    "type": "invalid_request",
    "code": "not_implemented",
    "param": null
  }
}
```

Recommended codes:

- `invalid_request`
- `model_not_found`
- `model_not_loaded`
- `unsupported_message_role`
- `context_length_exceeded`
- `not_implemented`
- `stream_not_supported`
- `generation_error`
- `runtime_unavailable`

## Streaming plan

When streaming is implemented, use Server-Sent Events compatible with OpenAI-style clients:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Do not expose streaming until the generation loop can yield real token events and handle cancellation.

## ForgeLocal compatibility notes

ForgeLocal can use camelid through its external OpenAI-compatible provider flow if camelid exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`

ForgeLocal expects at least:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "total_tokens": 2
  }
}
```

It may provide an API key even for local providers. camelid should ignore authorization locally unless auth is explicitly added later.

## Phase sampling/API implementation tasks

1. Add `src/sampling/mod.rs`.
2. Add `SamplingConfig`, `BasicSampler`, and deterministic greedy path.
3. Add repeat penalty.
4. Add temperature scaling and stable softmax.
5. Add top-k filtering.
6. Add top-p filtering.
7. Add seeded categorical sampling.
8. Expand `ChatCompletionRequest` shape.
9. Add request validation for roles, stream flag, model id, max tokens, and stop fields.
10. Keep `/v1/chat/completions` returning `501` until inference exists.
11. Add streaming response structs but keep route returning `501` for `stream: true`.

## Recommended tests

Sampling tests:

- greedy returns argmax
- greedy tie breaks by lowest token id
- temperature zero uses greedy
- seeded sampling is repeatable
- top-k removes lower-ranked candidates
- top-p keeps smallest probability prefix
- repeat penalty lowers repeated positive logits
- repeat penalty makes repeated negative logits more negative
- invalid temperature rejects
- invalid top-p rejects

API tests:

- non-streaming chat returns `501 not_implemented` while inference is absent
- streaming chat returns `501 stream_not_supported`
- unsupported role returns `400 unsupported_message_role` once validation is added
- missing model returns `400 invalid_request` once model becomes required
- `/v1/models` returns loaded model id exactly

## Implementation boundary

Sampling should not know about HTTP, tokenizer internals, or tensor runtime. It should receive logits and history and return a token id. Stop-string handling requires decoded text, so keep it in the generation orchestration layer rather than the sampler itself.
