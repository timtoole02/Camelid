# Phi-3 hold evidence

Why `phi3_mini_4k_instruct_q8_0` is NOT advertised as supported.

## Cleared 2026-07-27

Prompt-token parity PASSES on the pinned artifact `Phi-3-mini-4k-instruct-Q8_0.gguf`:

- `prompt-token-parity-q8-20260727.json` - 8/8 raw prompts, `all_match=true`
- `chat-prompt-token-parity-q8-20260727.json` - 3/3 rendered chat prompts, `all_match=true`

This closed the tokenizer half of the hold. Three defects were fixed: no rstrip of the
whitespace following `<|...|>` markers; the SPM dummy prefix emitted as a standalone token
id instead of prepended as a character for SPM to merge; and raw text bypassing the SPM
algorithm entirely for a longest-match encoder. The superseded pre-fix captures
(`prompt-token-parity.json`, `phi3-chat-parity.json`) are retained for history.

## STILL BLOCKING: the engine does not repeat itself

`nondeterminism-q8-20260727.json` - camelid is NON-DETERMINISTIC on phi3 at temperature 0.
Six identical requests on the 9-token prefix `The capital of France is Paris.\n` produced
two different tokens, flipping between the reference `<|assistant|>` (32001) and `\n` (13).
The reference token wins on a majority of runs, so this row is CLOSER than the superseded
receipt implied - but generation parity cannot be certified while identical requests return
different answers.

Scoped by experiment: Llama-3.2-3B (metal resident) and Mistral-7B (the SAME
`cpu_reference` path) are both bit-identical across runs, so this is phi3-specific rather
than engine- or path-wide. It survives `--threads 1` (not a parallel-reduction race) and
`CAMELID_LAZY_Q8_0_LINEAR=0` (not lazy weight materialization). Single-threaded,
eager-weighted non-determinism is the signature of reading uninitialized or stale memory.
Open suspects, both phi3-only: `head_dim=96` (every other local row is 128; 96 is a
multiple of 32 but not 64, where a fixed-width kernel tail leaves a buffer partly
unwritten), and the fused `attn_qkv`/`ffn_up` sub-row descriptor expansion in
`src/model.rs`.

### Correction

`generation-divergence-q8-20260727.json` is SUPERSEDED and annotated in place. It read a
fresh prefill disagreeing with incremental decode as a structural prefill/decode split and
named the attention/KV path as the suspect. That was two samples of a noisy process with
no repeatability check, and it is wrong. It is kept, marked, rather than deleted.

Generation parity, the bounded-context packs, and API/WebUI evidence remain outstanding.
The row stays `active_validation_blocked_parity` and fail-closed in the frontend.
