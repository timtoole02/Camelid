# 2026-05-06 deep-support expansion set: Mistral, Mixtral, Qwen, Gemma

Owner context: Tim locked the next new-model-family set as Mistral 7B Instruct, Mixtral 8x7B Instruct, Qwen 2.5 7B Instruct, and Gemma 2 9B Instruct. This note coordinates real execution while keeping the public four-row support boundary frozen: TinyLlama current gate plus exact-row smoke for Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Q8_0 only, including only the checked bounded context packs already promoted in the support ledger.

## Current support boundary

Do not promote any of these four new lanes to supported/readiness/frontend-green from this note alone. New rows stay `acceptance_target` or `planned_exact_row_candidate` until row-specific artifacts exist.

## Lane 1: Mistral immediate closure

- Display row: `Mistral-7B-Instruct-v0.3.Q8_0.gguf`.
- Resolved public acquisition candidate: `albertodelazzari/Mistral-7B-Instruct-v0.3-Q8_0-GGUF`.
- Repository SHA observed via Hugging Face API on 2026-05-06: `6ef9098086fc9885224271e616699049e5c5fb33`.
- License metadata observed: `apache-2.0`.
- Repo rfilename observed: `mistral-7b-instruct-v0.3-q8_0.gguf`.
- Current execution state: Ubuntu load/serve evidence exists, but support is not promotion-ready until tokenizer/template parity passes.
- Local code-side closure started in this change: Mistral instruct chat rendering now emits metadata BOS as text without asking tokenizer encode to prepend an additional BOS, and keeps special parsing enabled for the rendered `<s>` control token. This is a parity blocker fix, not a support promotion.

Minimum next evidence before support wording can move:

1. Capture exact file checksum for the downloaded GGUF and record source repo SHA/license.
2. Capture GGUF metadata summary: architecture, tokenizer model, BOS/EOS IDs, chat template, context length, tensor types.
3. Produce independent prompt-token references for at least:
   - single user turn,
   - system + user turn,
   - completed assistant turn followed by next user turn.
4. Compare Camelid prompt-token IDs against the reference exactly.
5. Re-run Ubuntu bounded load/serve/readiness with command transcript, RSS notes, and scrubbed manifest.
6. Only after 1-5 are green: add deterministic generation parity, API smoke, WebUI smoke, bounded context packs, and performance/portability evidence one box at a time.

Suggested acquisition command template:

```bash
huggingface-cli download albertodelazzari/Mistral-7B-Instruct-v0.3-Q8_0-GGUF \
  mistral-7b-instruct-v0.3-q8_0.gguf \
  --local-dir "$CAMELID_MODEL_DIR/mistral-7b-instruct-v0.3-q8_0"
sha256sum "$CAMELID_MODEL_DIR/mistral-7b-instruct-v0.3-q8_0/mistral-7b-instruct-v0.3-q8_0.gguf"
```

## Lane 2: Mixtral active validation unsupported

- Display candidate row: `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf`.
- Resolved public acquisition candidate: `leserg/Mixtral-8x7B-Instruct-v0.1-Q8_0-GGUF`.
- Repository SHA observed via Hugging Face API and reconfirmed on 2026-05-09: `93c0492d1891b5147f42b2648d9fccc140910a2f`.
- License metadata observed: `apache-2.0`.
- Repo rfilename observed: `mixtral-8x7b-instruct-v0.1-q8_0.gguf`.
- GGUF ETag/size captured on 2026-05-09: `77b8ee314ae3e77cefaba7f33841235da3346c34171547fe10e8a85f127973a7`, `49626319776` bytes.
- Evidence bundle: `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-metadata-tokenizer-typed-unsupported-20260509/manifest.json`.
- Tokenizer/reference pack: `fixtures/tokenizer/mixtral-8x7b-instruct-v0.1-reference-pack.json`.

Architecture/template findings:

- Sparse GGUF metadata parses as `general.architecture=llama` with MoE metadata `llama.expert_count=8` and `llama.expert_used_count=2`.
- Expert tensors use `blk.N.ffn_gate_inp.weight` plus `blk.N.ffn_{gate,up,down}_exps.weight`; the dense LLaMA/Mistral FFN path must not bind these as ordinary dense tensors.
- Camelid now returns typed unsupported behavior for this MoE runtime path; `/api/models/load` exposes `unsupported_runtime.code=unsupported_model_architecture`, and generation requests fail closed with the Mixtral MoE routing message.
- Tokenizer/template prompt IDs match llama.cpp reference captures for the exact sparse GGUF header, but template parity does not prove MoE execution correctness.

Remaining bring-up order:

1. Implement and unit-test the smallest correct top-k expert-routing vertical slice.
2. Attempt deterministic one-prompt generation parity only after expert routing is known-good.
3. Add full-model bounded load/readiness, API/WebUI, RSS/timing, scrubbed manifest, and checksums only after generation parity exists.

## Lane 3: Qwen 2.5 first honest plan

- Display candidate row: `Qwen2.5-7B-Instruct-Q8_0.gguf`.
- Resolved public acquisition candidate: `paultimothymooney/Qwen2.5-7B-Instruct-Q8_0-GGUF`.
- Repository SHA observed via Hugging Face API on 2026-05-06: `4895b9bc220cd193d1dd93850547be999c50af65`.
- License metadata observed: `apache-2.0`.
- Repo rfilename observed: `qwen2.5-7b-instruct-q8_0.gguf`.

Architecture/template risks:

- Qwen tokenizer/template semantics may not match Camelid's current LLaMA SPM or Llama 3 BPE paths.
- Architecture/config mapping must be explicit; do not assume dense LLaMA field names are enough.
- Prompt-token parity is the first real gate, not generation.

Minimum bring-up evidence:

1. Acquisition/SHA/license transcript and GGUF metadata summary.
2. Tokenizer model/pre-tokenizer inventory and unsupported-path error if Camelid cannot parse it yet.
3. Chat-template fixture plus independent prompt-token references.
4. Camelid-vs-reference prompt-token comparison.
5. Bounded load/readiness result with RSS notes.
6. Deterministic generation/API/WebUI/bounded context only after tokenizer/template and architecture mapping are green.

## Lane 4: Gemma 2 first honest plan

- Display candidate row: `gemma-2-9b-it-Q8_0.gguf`.
- Resolved public acquisition candidate: `BenevolenceMessiah/gemma-2-9b-it-Q8_0-GGUF`.
- Repository SHA observed via Hugging Face API on 2026-05-06: `eb4741d7b2aece6d8519db62dc31d89029a40af8`.
- License metadata observed: `gemma`.
- Repo rfilename observed: `gemma-2-9b-it-q8_0.gguf`.

Architecture/template risks:

- Gemma2 architecture details and tensor names are not LLaMA-compatible by default.
- Tokenizer/control-token behavior and instruction template formatting need independent references.
- License/access metadata must stay visible in evidence because Gemma license differs from Apache rows.

Minimum bring-up evidence:

1. Acquisition/SHA/license transcript and GGUF metadata summary.
2. Typed unsupported architecture/tokenizer behavior if Camelid cannot parse/load it yet.
3. Tokenizer/chat-template fixtures and independent prompt-token references.
4. Bounded metadata/load/readiness result with RSS notes.
5. Deterministic generation/API/WebUI/bounded context only after architecture and tokenizer/template gates are green.

## Immediate command order

1. Keep `/api/capabilities`, `README.md`, `STATUS.md`, and `COMPATIBILITY.md` fail-closed for all four new lanes.
2. Run focused local tests for capability rows and Mistral renderer flags.
3. On Ubuntu, preserve dirty worktrees, then run acquisition/checksum/metadata commands for Mistral first.
4. Produce Mistral prompt-token reference output before any more serving claims.
5. Only after Mistral parity is closed, queue Mixtral/Qwen/Gemma acquisition metadata captures in that order, with no runtime promotion language.
