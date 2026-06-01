# Camelid Roadmap

Last updated: 2026-05-31

`ROADMAP.md` is Camelid's delivery plan of record. It is not a backlog and it is not a feature wish list. It answers one product question: **what must happen next for Camelid to widen its support boundary without weakening credibility?** The sequencing is intentional: protect the supported lane, remove the next exact blocker, and widen claims only when the resulting evidence can survive scrutiny.

[`COMPATIBILITY.md`](COMPATIBILITY.md) defines what Camelid can honestly support today. [`STATUS.md`](STATUS.md) records the artifacts, evidence boundaries, and blocker state behind that posture. Detailed completed-phase history lives in `ROADMAP_ARCHIVE.md` and `STATUS.md`. Read this file as operating sequence, not aspiration.

Executive summary: Camelid has one full verified gate, three bounded Llama exact-row lanes, and one Mistral active-validation lane. TinyLlama 1.1B Chat Q8_0 remains the trusted gate. Llama 3.2 1B Instruct Q8_0 is verified through checked 512/1024/2048/4096/8192 context packs. Llama 3.2 3B Instruct Q8_0 is supported as exact-row smoke with canonical Ubuntu API/WebUI evidence plus checked 512/1024/2048 context packs. Llama 3 8B Instruct Q8_0 is verified within checked 512/1024/2048 packs. Mistral 7B Instruct v0.3 Q8_0 remains evidence-only active validation with checked bounded 512/1024/2048/4096/8192 context packs plus a fail-closed API/WebUI/RSS bundle that has not been promoted into support. Mixtral remains partial backend runtime evidence only and is blocked by later-generation divergence plus a continuation HTTP hang. Ubuntu x86 Q8 acceleration is active default-off performance work, not a support, portability, or throughput claim.

Practical reading rule: if a task does not protect the current gate, remove the next exact blocker, or prepare aligned support-language updates, it is secondary to this roadmap.

## Program objective

Camelid is not pursuing breadth for its own sake. The roadmap exists to expand capability only when the product can expand claims just as responsibly and defend them with row-specific evidence.

Current program posture:

- **Supported generation gates:** TinyLlama 1.1B Chat Q8_0 remains the full supported gate; exact Llama 3.2 1B and Llama 3 8B rows have verified bounded support inside their checked envelopes, while the exact Llama 3.2 3B row remains supported as exact-row smoke inside its checked envelope.
- **Scope boundary:** the Llama support claim is exact-row only: model version/size, Instruct variant, Q8_0 quantization, loaded runtime readiness, and the checked smoke/parity/context envelope all matter.
- **1B promoted lane:** Llama 3.2 1B Instruct Q8_0 is verified through checked bounded 512/1024/2048/4096/8192 packs where row-specific PASS artifacts exist; this does not promote model-native/larger context, neighboring rows, production throughput, or portability.
- **3B promoted lane:** Llama 3.2 3B Instruct Q8_0 is supported as exact-row smoke with canonical Ubuntu API/WebUI refresh at source head `e9f926ed1a65`, compact/broader parity, bounded unique-chat RSS/perf, and checked 512/1024/2048 packs; broader/full support remains gated.
- **8B promoted lane:** Llama 3 8B Instruct Q8_0 has verified bounded support through compact parity, a three-prompt 50-token Ubuntu parity run, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and one bounded compact chat-template-shapes pack for the exact tracked Q8_0 GGUF; broader/full 8B support remains gated.
- **Next-family posture:** Mistral 7B Instruct v0.3 Q8_0 remains evidence-only active validation, Mixtral 8x7B Instruct v0.1 Q8_0 is blocked partial runtime evidence only, and Qwen/Gemma remain planned exact-row candidates.
- **Performance posture:** Ubuntu x86 Q8 work remains default-off and evidence-gated. It may guide runtime architecture, but it does not widen support language or promise user-visible speed until same-host parity and repeated timing evidence justify it.
- **Explicit non-claim:** no broad Llama-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Nothing inherits support from a nearby size, quantization, family, tokenizer lane, API surface, or UI state.

Near-term thesis: protect the trusted TinyLlama gate, the exact Llama 3.2 1B/3B and Llama 3 8B bounded rows, keep the Mistral 7B Instruct v0.3 Q8_0 lane fail-closed until explicit support promotion evidence exists, keep Mixtral fail-closed until its blockers are fixed, and advance Ubuntu x86 performance only through default-off, measured, parity-preserving slices.

## Roadmap operating rules

Four rules drive prioritization and sequencing:

- **Protect the current gate first.** TinyLlama Q8_0 remains the release anchor.
- **Remove the next honest blocker.** The highest-leverage work is the exact runtime seam that can create the next promotable artifact.
- **Move public surfaces together.** Documentation, API signals, and frontend readiness should change in the same change window.
- **Cite committed evidence anchors first.** The public bundle manifest/checksums, perf/portability envelope, reopened-lane API + frontend smoke manifest, 1B bounded 1024/2048/4096/8192 context bundles, 3B bounded 1024/2048 plus canonical API/WebUI bundle, the current-head 8B 1024/2048 bundle, Mistral validation bundles, Mixtral blocker bundles, and current-head per-row manifests are the roadmap-facing evidence layer; raw `target/` artifacts are drill-down only.

## What changed in the support line

Recent work moved the release ledger only where the evidence, API, frontend, and docs now agree.

- TinyLlama Q8_0 remains the trusted release gate.
- Llama 3.2 1B Q8_0 is now verified inside checked bounded 512/1024/2048/4096/8192 context packs; the 2048 pack turned green only after the RoPE frequency-factor fix, and the 4096/8192 packs are tied to their cited source/runtime heads.
- Llama 3.2 3B Q8_0 is now a supported exact-row smoke lane after exact-GGUF load, compact prompt-token/1-token/5-token/50-token parity, canonical Ubuntu API/WebUI refresh, frontend evidence, bounded unique-chat RSS/perf, and bounded 512/1024/2048 context-pack evidence aligned.
- Llama 3.2 3B no longer has the JSON-shaped broader prompt-pack blocker; the post-Q8-dot clean three-prompt 50-token rerun now passes against llama.cpp.
- Llama 3 8B Q8_0 moved from groundwork-only to verified bounded support after Ubuntu three-prompt parity, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and compact chat-template-shapes packs aligned for that exact row only.
- Mistral 7B Instruct v0.3 Q8_0 remains evidence-only active validation with tokenizer/template, 1-token generation, broader five-prompt/50-token parity, checked bounded 512/1024/2048/4096/8192 context, and a fail-closed API/WebUI/RSS bundle that still records the row as unsupported pending explicit support-contract promotion.
- Mixtral 8x7B Instruct v0.1 Q8_0 remains blocked partial runtime evidence only: bounded one-token MoE evidence exists, but Gate 9A later-generation divergence and a continuation backend HTTP hang block API/WebUI/frontend readiness and support promotion.
- Ubuntu x86 Q8 performance work has produced default-off route/control-plane/kernel slices and retained/rejected evidence, but the current roadmap treats it as evidence-gated performance work, not a support or throughput milestone.

Near-term objective: preserve the supported TinyLlama gate, exact Llama 3.2 1B/3B and Llama 3 8B bounded lanes, keep the Mistral 7B Instruct v0.3 Q8_0 lane fail-closed until support promotion is earned, fix Mixtral blockers before any support wording, and keep performance claims default-off until same-host evidence moves the whole-model result.

## Delivery sequence: now, next, later

This is the highest-level execution order. **Now** protects the current gate and clears the next blocker. **Next** is what Camelid may promote once bounded evidence exists. **Later** stays intentionally downstream of correctness and support-discipline work.

### Now

Protect the supported lanes and clear the next blocker before widening claims.

- Protect the validated TinyLlama Q8_0 gate.
- Protect the exact Llama 3.2 1B bounded 512/1024/2048/4096/8192 row.
- Protect the exact Llama 3.2 3B and Llama 3 8B bounded 512/1024/2048 rows.
- Preserve the Llama 3.2 1B/3B broader prompt-pack plus bounded context-pack wins while expanding only after model-native/larger-context, stronger performance/portability, and broader chat-template evidence land.
- Preserve the Llama 3 8B exact-row promotion through the checked 512/1024/2048-context packs on current `main`; older 1024/2048 bundles remain historical source-head evidence only.
- Protect the Mistral 7B Instruct v0.3 Q8_0 evidence-only active-validation lane across all support surfaces.
- Keep Mixtral fail-closed until later-generation divergence and the continuation hang are fixed and rerun through API/WebUI/RSS/frontend evidence.
- Keep Ubuntu x86 Q8 acceleration default-off while the team proves route hit, parity, repeated same-host timing, and whole-model impact.
- Keep README, `COMPATIBILITY.md`, `ROADMAP.md`, `STATUS.md`, `/api/capabilities`, and frontend readiness copy aligned.

### Next

Promote only what can be defended row by row.

- Close the active next-model bring-up set as exact-row evidence lanes first, never as family-wide support claims. **Mistral 7B Instruct v0.3 Q8_0** is the immediate validation lane; **Mixtral 8x7B Instruct v0.1 Q8_0** remains blocked until later-generation divergence and continuation hang are fixed and rerun.
- Widen Llama 3.2 3B Q8_0 beyond exact-row smoke only if broader prompt/chat-template, memory/performance, API, and WebUI evidence all land.
- Retain an Ubuntu x86 Q8 performance slice only when it is default-off, parity-preserving, measured on the canonical same-host lane, and improves the whole-model result or narrows the proven bottleneck.
- Broaden quantization support beyond Q8_0 with tests, docs, and exact-row evidence.
- Expand tokenizer and chat-template coverage for additional supported rows.
- Extend correctness checks into longer prompt and context buckets.

### Later

Broaden the product surface only after correctness and release discipline are stable.

- Richer OpenAI API completeness beyond the current supported subset.
- Measured performance optimization after correctness gates are stable.
- Packaging and portability work across non-primary platforms.
- Broader model-family expansion beyond current LLaMA-family priorities.
- First-class multi-model concurrency so Camelid can keep multiple local models loaded at once and serve agent/OpenClaw workloads that need different models simultaneously.
- For Qwen specifically, start with one exact GGUF target and do not schedule runtime-promotion work until tokenizer/chat-template fixtures, llama.cpp token-reference checks, and bounded load plus prompt-token parity are in place for that row.
- Treat Rust-coder and other specialized rows as validation candidates only until acquisition, tokenizer/runtime mapping, parity, API/WebUI, RSS, and throughput evidence exist for the exact row.

## Milestone table

| Milestone | Status | What must be true |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 supported gate | Complete | End-to-end generation parity artifacts exist and docs/API/frontend agree. |
| Llama 3.2 1B Instruct Q8_0 exact-row bounded support | Complete / bounded support | Compact parity, broader prompt-pack parity, API smoke, frontend smoke, exact-row metadata-Jinja row-template evidence, bounded template-shapes, unique-chat RSS/perf, and bounded 512/1024/2048/4096/8192 context packs agree for this exact 1B Q8_0 row. |
| Llama 3.2 3B Instruct Q8_0 exact-row smoke | Complete / narrow support | Exact GGUF load, compact prompt-token/1-token/5-token/50-token parity, broader three-prompt parity, canonical Ubuntu API/WebUI refresh, bounded unique-chat RSS/perf, and bounded 512/1024/2048 context packs agree for this exact 3B Q8_0 row. |
| Llama 3 8B Instruct Q8_0 exact-row bounded support | Complete / bounded support through checked 512/1024/2048 packs | Compact prompt-token/1-token/5-token/50-token parity, the three-prompt 50-token pack, API smoke, frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and the compact chat-template-shapes pack support this exact 8B Q8_0 row only. |
| Mistral 7B Instruct v0.3 Q8_0 exact-row validation | Active / evidence-only | Tokenizer/template, 1-token generation, broader five-prompt/50-token parity, bounded 512/1024/2048, and checked 4096/8192 context evidence exist for this exact row, but the fail-closed API/WebUI/RSS bundle still records `active_validation_unsupported`, so support is not yet promoted. |
| Mixtral 8x7B Instruct v0.1 Q8_0 runtime bring-up | Blocked / partial runtime evidence | Bounded one-token backend MoE evidence exists; Gate 9A later-generation divergence and continuation backend HTTP hang must be fixed before API/WebUI/frontend readiness or support promotion. |
| Quantization breadth beyond Q8_0 | Planned | Each quant format has loader/runtime tests, docs, and at least one row-specific real-model artifact. |
| Longer-context correctness | Planned | Context-length claims are backed by model-specific audits and documented limits. |
| API and sampling completeness | Planned | Newly supported fields have tests, honest docs, and typed unsupported errors removed only after implementation. |
| Ubuntu x86 Q8 performance and portability | Active / default-off evidence work | Optimizations stay default-off until route hit, parity, same-host repeated timing, and whole-model impact are proven without widening support claims. |

## Active roadmap lanes

### Compatibility matrix and support contract

`COMPATIBILITY.md` is the support ledger. This roadmap governs when rows are allowed to move.

Current required discipline:

- TinyLlama 1.1B Chat Q8_0 remains a supported generation gate.
- Llama 3.2 1B Q8_0 is verified for this exact row with compact/broader parity, API/WebUI evidence, exact-row metadata-Jinja row-template evidence, bounded template-shapes, unique-chat RSS/perf, and bounded 512/1024/2048/4096/8192 context-pack evidence; model-native/larger-context beyond checked packs, production throughput, portability, and broader arbitrary-template expansion remain gated.
- Llama 3.2 3B Q8_0 is supported as an exact-row smoke lane with compact and broader three-prompt parity, canonical Ubuntu API/WebUI refresh at source head `e9f926ed1a65`, bounded unique-chat RSS/perf, row-scoped metadata-Jinja/template-shape evidence, and bounded 512/1024/2048 context-pack evidence; model-native/larger-context and broader arbitrary-template expansion remain gated.
- Llama 3 8B Q8_0 has verified bounded support with compact parity, the three-prompt 50-token pass, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and one compact chat-template-shapes pack; model-native/larger context beyond checked packs, broader chat-template, production performance, and portability expansion remain gated.
- Mistral 7B Instruct v0.3 Q8_0 remains active exact-row validation only. Tokenizer/template, generation, and checked context evidence exist, but the fail-closed API/WebUI/RSS bundle still requires explicit support-contract promotion before README, compatibility, status, API, and frontend surfaces may call the row supported.
- Mixtral 8x7B Instruct v0.1 Q8_0 is active validation / partial backend runtime only. Later-generation divergence and the continuation hang block readiness.
- Qwen 2.5 7B and Gemma 2 9B remain planned exact-row candidates only.
- Frontend readiness must remain exact-row and exact-quant aware.
- Support-language updates should point first to the committed `qa/evidence-bundles/...` manifests/checksums and only then to raw `target/` drill-down artifacts.

Promotion evidence must update docs, API capability reporting, and frontend readiness language in the same change window.

### Q8 execution and Ubuntu x86 acceleration

This is the highest-leverage active performance engineering lane, but it is not a support claim.

What exists now:

- retained Q8_0 block loading
- serial `dot_row_f32`, `dot_all_rows_f32`, and single-input-row adapters
- CPU materialization-budget guardrails
- Llama 3 tokenizer, config, GQA, and RoPE groundwork
- a code-only chunked prefill slice (`CAMELID_PREFILL_CHUNK_TOKENS`, default `128`) that batches non-final prompt tokens through embedding, Q/K/V, RoPE, KV writes, causal attention context, attention output, and FFN while leaving the final logits token on the established single-token path
- Q8_0 file-backed batched matmul read reuse across input rows for bounded prefill chunks, plus a layer-major lazy-Q8 prefill schedule that reuses each layer's file-backed weights across all prefill chunks before moving to the next layer
- default-off Ubuntu x86 Q8 packed-runtime work under evidence-gated flags such as `CAMELID_X86_Q8_REPACK=on`
- selected default-off decode consumers, packed-rows4 matmul slices, GEMM4/VNNI experiments, and ExecutionPlan-managed cleanup for stale experimental flags
- retained/rejected evidence notes for Ubuntu x86 Q8 experiments, including explicit local-only versus same-host Ubuntu proof boundaries

What still needs to happen:

- measure chunked prefill and each optimized slice on approved row-specific runtime lanes before using it in support or throughput claims
- keep retained-Q8 linear execution wired through attention, FFN, and final output projection without unsafe eager dense materialization
- keep bounded scratch/output behavior explicit and measured
- verify first-token and longer-prompt generation with row-specific parity/RSS evidence before promoting any larger context box
- prove route hit, parity, repeated same-host timing, and whole-model TTFT/throughput movement before retaining any Ubuntu x86 speed claim
- keep local-only Rust/control-plane proof out of public performance language until canonical Ubuntu validation exists

What does **not** count as promotion evidence by itself:

- tokenizer freshness
- metadata load success
- standalone block benchmarks
- artifact presence on disk
- a default-off flag existing in code
- local Darwin compile/parity evidence for an Ubuntu x86 performance claim

### Quantization breadth

Camelid should broaden quant support only after the larger-model Q8 execution seam is trustworthy.

Priority shape:

- keep Q8_0 as the correctness baseline
- add the next real-world quant formats with the highest practical value
- require loader tests, runtime math checks, and at least one row-specific real-model artifact per supported quantization

No quant format is supported just because its metadata parses.

### Tokenizer and chat-template expansion

Tokenizer support remains part of the release contract, not a side detail.

Near-term expectations:

- preserve the current LLaMA/SPM and Llama 3 template behavior
- preserve and protect Mistral tokenizer/template evidence for the fail-closed exact-row validation lane
- keep Mixtral tokenizer/template and sparse-MoE evidence scoped as partial runtime evidence until later-generation and API/WebUI blockers close
- treat Qwen/Gemma tokenizer and chat-template work as planned exact-row fixture work, not readiness
- add fixtures for whitespace, multiline prompts, control tokens, EOS behavior, and prompt-shape edge cases
- keep unsupported tokenizer families as typed unsupported states until a full support lane exists

Tokenizer parity alone does not promote generation support.

### Longer-context correctness

Short-prompt success is not enough for broader support claims.

This lane should expand in bounded steps:

- validated short prompts
- 512-token bucket
- 1k-token bucket
- 2k-token bucket
- larger model-specific buckets only when memory/runtime evidence supports them

Current bucket posture:

- Llama 3.2 1B Q8_0 has checked bounded 512/1024/2048/4096/8192 context packs for the exact row.
- Llama 3.2 3B Q8_0 has checked bounded 512/1024/2048 context packs for the exact row.
- Llama 3 8B Q8_0 has checked bounded 512/1024/2048 context packs for the exact row.
- Mistral 7B Instruct v0.3 Q8_0 has checked bounded 512/1024/2048/4096/8192 context packs verified for this exact row only.
- Mixtral has no promoted context bucket; later-generation divergence and continuation hang block advancement.

For each promoted context bucket, Camelid should have:

- prompt-token evidence
- generation evidence where applicable
- clear model-specific documented limits
- no hidden inference from nearby rows

### OpenAI API and sampling completeness

Camelid already exposes a narrow but real OpenAI-compatible local surface. The roadmap here is to expand completeness without faking compatibility.

Active rule set:

- implement deterministic correctness first
- keep unsupported combinations as typed errors until behavior is real
- add richer fields only with tests and documentation

Near-term candidates include:

- richer logprob support
- broader streaming metadata completeness
- multi-choice generation
- stronger seeded sampling validation

### llama-server compatibility sequence

Tim's 2026-05-31 product decision is to do both: preserve Camelid's honest OpenAI-style subset as the stable user/developer path, and add llama-server-compatible routes/control-plane behavior incrementally where useful. llama-server is a reference target for API shape and client expectations only; Camelid does not copy source or claim full parity.

Sequenced plan:

1. Keep `/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming, cancellation behavior, errors, `/api/capabilities`, docs, and frontend readiness as the stable contract. Unsupported OpenAI fields stay typed errors until implemented and tested.
2. Grow read-only llama-server discovery first: `/props`, `/slots`, `/models`, health/model metadata, and WebUI probes may expose public readiness state, but local paths stay redacted and readiness fails closed unless the loaded exact row is contract-supported and generation-ready. `/models` is limited to currently loaded Camelid models until router-mode cache listing, reload/autoload, native load/unload, and model-source metadata are designed.
3. Keep tokenizer/control-plane utilities bounded: `/tokenize`, `/detokenize`, and `/apply-template` may work for loaded supported tokenizer/template lanes only; `/tokenize` may expose bounded `with_pieces=true` id/piece objects for supported tokenizers, while arbitrary template kwargs, prompt-cache metadata, and slot lifecycle actions remain unsupported until real semantics exist.
4. Add native generation compatibility only after request mapping is explicit: `/completion` must translate supported llama-server parameters onto Camelid's generation path without weakening the OpenAI subset, and unsupported sampler, cache, image, infill, and tool fields must remain typed errors.
5. Defer embeddings, reranking, Responses, Messages, multimodal routes, LoRA, metrics, router-mode model management, and full WebUI parity until backend support, route semantics, tests, capabilities text, docs, and frontend gates all move together.

Current gap sequence as of 2026-06-01:

1. Discovery parity: keep `/health`, `/v1/health`, `/props`, `/slots`, `/models`, `/v1/models`, `/v1/models/:model`, `/api/capabilities`, and frontend readiness fail-closed, privacy-safe, and tested before expanding writable control routes.
2. Tokenizer/template utilities: preserve loaded-model-only `/tokenize`, `/detokenize`, and `/apply-template`; `with_pieces=true` now has a bounded supported-tokenizer test, while template kwargs, richer error mapping, and tokenizer-piece edge cases outside supported lanes still need fixtures and capability text.
3. Native generation: map a narrow `/completion` request subset to the existing Camelid generation path only after sampler/stop/stream/timing/cancellation behavior is specified and unsupported fields stay typed.
4. WebUI control plane: add only probes that are read-only and support-contract-aware; keep slot cache actions, prompt-cache metadata, metrics, sleep/idle, router reload/autoload, and native model load/unload unsupported until semantics and tests exist.
5. Deferred runtime surfaces: embeddings, reranking, infill, multimodal inputs, LoRA adapters, router-mode model management, Responses/Messages, and broad llama-server WebUI parity remain blocked until backend support, docs, capabilities, and frontend gates move together.

### Performance, packaging, and portability

Performance work matters, but it should follow correctness and support honesty.

Execution order:

- preserve the validated baseline
- measure bottlenecks after each correctness milestone
- optimize only where evidence says it matters
- keep optimized kernels behind parity guardrails until proven

Portability and packaging should remain explicit:

- Ubuntu x86 Q8 has a narrow measured/default-off evidence lane; do not generalize it into broad Linux, x86, CPU, or production-throughput support
- no implied Mac/Windows/non-primary-platform support without matching validation
- no implied portable model-path assumptions without documentation
- no release packaging claim before reproducible setup instructions exist

## Promotion rules

A row may move forward only when all of the following are true:

1. Runtime behavior works for the exact row being claimed.
2. Evidence is captured for the exact scope being promoted.
3. Documentation says exactly what the evidence supports and nothing broader.
4. API capability reporting reflects the same boundary.
5. Frontend readiness and UI language reflect the same boundary.
6. Unsupported adjacent rows remain visibly unsupported.

Practical examples:

- A 1B row does not promote a 3B or 8B row.
- Metadata load does not promote generation support.
- Tokenizer parity does not promote runtime readiness.
- A first-token artifact does not automatically promote longer-context correctness.
- A benchmark does not promote portable packaging or production-readiness claims.

## Non-goals

For the current roadmap window, Camelid is **not** trying to:

- match every feature of mature inference runtimes
- claim broad LLaMA-family support from a narrow artifact set
- treat local artifact presence as runtime support
- infer readiness across neighboring sizes or quantizations
- advertise hosted/provider/catalog features that are not wired and tested
- prioritize GPU acceleration ahead of stable CPU correctness and evidence-backed model breadth

## Archived and completed phases

Early repo setup, backend skeleton, GGUF metadata parsing, tokenizer bring-up, tensor loading, and first-generation-lane work are complete enough that they no longer need full tactical detail here.

See:

- `ROADMAP_ARCHIVE.md` for concise completed-phase history
- `STATUS.md` for tactical runs, artifact paths, benchmark outputs, and diagnostic notes

The important completed milestone for current planning is simple: Camelid has one validated TinyLlama Q8_0 end-to-end generation gate, and every future milestone must preserve that trust.
Current planning also includes three bounded Llama exact-row lanes, but those are not a license to widen support language beyond their checked envelopes.
