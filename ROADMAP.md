# Camelid Roadmap

Last updated: 2026-08-09

`ROADMAP.md` is Camelid's delivery plan of record. It is not a backlog and it is not a feature wish list. It answers one product question: **what must happen next for Camelid to widen its support boundary without weakening credibility?** The sequencing is intentional: protect the supported lane, remove the next exact blocker, and widen claims only when the resulting evidence can survive scrutiny.

[`COMPATIBILITY.md`](COMPATIBILITY.md) defines what Camelid can honestly support today. [`STATUS.md`](./docs/reference/STATUS.md) records the artifacts, evidence boundaries, and blocker state behind that posture. Detailed completed-phase history lives in `ROADMAP_ARCHIVE.md` and `STATUS.md`. Read this file as operating sequence, not aspiration.

Executive summary: Camelid has one full verified gate and, at v0.6.1, more than two dozen supported exact rows in the [`COMPATIBILITY.md`](COMPATIBILITY.md) at-a-glance contract. TinyLlama 1.1B Chat Q8_0 remains the trusted gate. Llama 3.2 1B Instruct Q8_0 is verified through checked 512/1024/2048/4096/8192 context packs; Llama 3 8B Instruct Q8_0 within checked 512/1024/2048; Llama 3.2 3B Instruct Q8_0 as exact-row smoke on its anchored raw-decode ladder. Mistral 7B Instruct v0.3 Q8_0 is supported as exact-row smoke, with its support-promotion API/WebUI smoke bundle at `qa/evidence-bundles/mistral-7b-v0.3-q8-support-promotion-20260605T090914Z-head-d7b1699/manifest.json`. Since the four-row era the ledger has widened by exact rows, never by family: five dense Qwen3 rows, three Ornith `qwen35` rows (the `tool_capable` vehicle for agent mode), Gemma 3 1B and Gemma 4 E2B/E4B, seven PrismML Bonsai artifacts including two vision-capable 27B rows, four certified non-Q8_0 Llama requants, and the DiffusionGemma experimental lane. Windows x86_64 and Linux x86_64 are tracked platforms with CUDA lanes carrying their own per-GPU parity bundles. Agent mode is Supported (experimental) on receipted `tool_capable` rows. Mixtral remains partial backend runtime evidence only, blocked by later-generation divergence plus a continuation HTTP hang. LFM2.5 and the two distributed Gemma 4 rows are active validation, not support. Performance work stays evidence-gated and largely default-off: it is not a support, portability, or throughput claim.

Practical reading rule: if a task does not protect the current gate, remove the next exact blocker, or prepare aligned support-language updates, it is secondary to this roadmap.

## Program objective

Camelid is not pursuing breadth for its own sake. The roadmap exists to expand capability only when the product can expand claims just as responsibly and defend them with row-specific evidence.

Current program posture:

- **Supported generation gates:** TinyLlama 1.1B Chat Q8_0 remains the full supported gate; exact Llama 3.2 1B and Llama 3 8B rows have verified bounded support inside their checked envelopes, while the exact Llama 3.2 3B row remains supported as exact-row smoke inside its checked envelope.
- **Scope boundary:** the Llama support claim is exact-row only: model version/size, Instruct variant, Q8_0 quantization, loaded runtime readiness, and the checked smoke/parity/context envelope all matter.
- **1B promoted lane:** Llama 3.2 1B Instruct Q8_0 is verified through checked bounded 512/1024/2048/4096/8192 packs where row-specific PASS artifacts exist; this does not promote model-native/larger context, neighboring rows, production throughput, or portability.
- **3B promoted lane:** Llama 3.2 3B Instruct Q8_0 is supported as exact-row smoke with the anchored checked 512/1024/2048/4096/8192 raw-decode context ladder on the current canonical GGUF plus June capability receipts; the canonical Ubuntu API/WebUI refresh at source head `e9f926ed1a65`, compact/broader parity, and bounded unique-chat RSS/perf are retained as historical evidence for the prior upload; broader/full support remains gated.
- **8B promoted lane:** Llama 3 8B Instruct Q8_0 has verified bounded support through compact parity, a three-prompt 50-token Ubuntu parity run, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and one bounded compact chat-template-shapes pack for the exact tracked Q8_0 GGUF; broader/full 8B support remains gated.
- **Next-family posture:** Mistral 7B Instruct v0.3 Q8_0 is supported exact-row smoke, Mixtral 8x7B Instruct v0.1 Q8_0 is blocked partial runtime evidence only, and Qwen/Gemma 2 remain planned exact-row candidates; Gemma 4 E2B-It/E4B-It Q8_0 are promoted exact-row text-generation smoke lanes with 12B-it in active validation behind the two-node sharding lane.
- **Performance posture:** Ubuntu x86 Q8 work remains default-off and evidence-gated. It may guide runtime architecture, but it does not widen support language or promise user-visible speed until same-host parity and repeated timing evidence justify it.
- **Explicit non-claim:** no broad Llama-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Nothing inherits support from a nearby size, quantization, family, tokenizer lane, API surface, or UI state.

Near-term thesis: protect the trusted TinyLlama gate, the exact Llama 3.2 1B/3B and Llama 3 8B bounded rows, keep the Mistral 7B Instruct v0.3 Q8_0 exact-row smoke support language synchronized with its promotion evidence, keep Mixtral fail-closed until its blockers are fixed, and advance Ubuntu x86 performance only through default-off, measured, parity-preserving slices.

## Roadmap operating rules

Four rules drive prioritization and sequencing:

- **Protect the current gate first.** TinyLlama Q8_0 remains the release anchor.
- **Remove the next honest blocker.** The highest-leverage work is the exact runtime seam that can create the next promotable artifact.
- **Move public surfaces together.** Documentation, API signals, and frontend readiness should change in the same change window.
- **Cite committed evidence anchors first.** The public bundle manifest/checksums, perf/portability envelope, reopened-lane API + frontend smoke manifest, 1B bounded 1024/2048/4096/8192 context bundles, the 3B anchored 512-8192 ladder bundle plus historical canonical API/WebUI bundle, the current-head 8B 1024/2048 bundle, Mistral validation bundles, Mixtral blocker bundles, and current-head per-row manifests are the roadmap-facing evidence layer; raw `target/` artifacts are drill-down only.

## What changed in the support line

Recent work moved the release ledger only where the evidence, API, frontend, and docs now agree.

- TinyLlama Q8_0 remains the trusted release gate.
- Llama 3.2 1B Q8_0 is now verified inside checked bounded 512/1024/2048/4096/8192 context packs; the 2048 pack turned green only after the RoPE frequency-factor fix, and the 4096/8192 packs are tied to their cited source/runtime heads.
- Llama 3.2 3B Q8_0 is now a supported exact-row smoke lane after exact-GGUF load, compact prompt-token/1-token/5-token/50-token parity, canonical Ubuntu API/WebUI refresh, frontend evidence, bounded unique-chat RSS/perf, and bounded 512/1024/2048 context-pack evidence aligned (since re-anchored: the checked-context claim is now the 512/1024/2048/4096/8192 raw-decode ladder on the current canonical GGUF, and the May-era evidence is retained as historical for the prior upload).
- Llama 3.2 3B no longer has the JSON-shaped broader prompt-pack blocker; the post-Q8-dot clean three-prompt 50-token rerun now passes against llama.cpp.
- Llama 3 8B Q8_0 moved from groundwork-only to verified bounded support after Ubuntu three-prompt parity, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and compact chat-template-shapes packs aligned for that exact row only.
- Mistral 7B Instruct v0.3 Q8_0 is promoted to supported exact-row smoke with tokenizer/template, 1-token generation, broader five-prompt/50-token parity, checked bounded 512/1024/2048/4096/8192 context, GPU-vs-CPU greedy parity, and a support-promotion API/WebUI smoke bundle aligned on `supported_exact_row_smoke`.
- Mixtral 8x7B Instruct v0.1 Q8_0 remains blocked partial runtime evidence only: bounded one-token MoE evidence exists, but Gate 9A later-generation divergence and a continuation backend HTTP hang block API/WebUI/frontend readiness and support promotion.
- Ubuntu x86 Q8 performance work has produced default-off route/control-plane/kernel slices and retained/rejected evidence, but the current roadmap treats it as evidence-gated performance work, not a support or throughput milestone.
- Five dense Qwen3 rows (0.6B/1.7B/4B/8B `Q8_0` plus 4B `Q4_K_M`) are supported exact-row smoke for ChatML with thinking DISABLED. Thinking mode is available opt-in as a leading-trace lane; thinking-disabled remains the parity-locked mode.
- Three Ornith `qwen35` hybrid rows (`Q8_0`, and the in-house `Q4_K_M`/`Q3_K_M` requants) are supported exact-row smoke. The `Q8_0` row is the `tool_capable` promotion vehicle that made agent mode Supported (experimental).
- Gemma 3 1B-It `Q8_0` is supported exact-row smoke, defaulting to the Metal GPU-resident serve lane on a Metal host. Gemma 4 E2B-It and E4B-It `Q8_0` are supported exact-row smoke on CPU AND Metal GPU-resident.
- Seven hash-pinned PrismML Bonsai artifacts are supported exact-row smoke on macOS Metal and Windows x86_64 CUDA; the two 27B rows add single-image vision through the pinned Qwen3-VL projector.
- Four non-Q8_0 Llama requants are GPU-resident parity-certified exact-row smoke on raw greedy decode: Llama 3.2 1B `IQ4_XS` and `Q4_K_M`, Llama 3.2 3B `Q4_K_M` and `Q5_K_M`. Each stands on its own bundle and inherits no chat/context envelope from its parent row.
- Windows x86_64 became a tracked platform (CPU/MSVC and experimental CUDA), and the CUDA backend now compiles into the default build on Windows and x86_64 Linux. CUDA parity evidence is scoped to the recorded GPU, driver, and CUDA version — compiling the path in is a build-wiring fact, not a support claim.
- The API surface widened within the same evidence discipline: the stateless Responses API subset, embeddings and reranking on the exact Nomic v1.5 `Q8_0` row, structured output via LLGuidance, multi-choice and logprobs, and a privacy-safe Prometheus `/metrics` surface.
- `Phi-3-mini-4k-instruct-Q8_0` entered the curated catalog under a formal HOLD (`qa/muster/HOLD-phi3-mini-4k-instruct-q8_0.json`): prompt-token parity passes, but an SPM rstrip seam on chat specials and temperature-0 non-determinism on this row block certification. It is downloadable and NOT advertised as supported.
- LFM2.5-2.6B `Q8_0` opened the first `lfm2` lane as active validation, GPU-resident on Metal by default on Apple Silicon.

Near-term objective: preserve every supported exact row across all support surfaces without broadening any of them, close the active-validation lanes (Mixtral, LFM2.5, the two distributed Gemma 4 rows) or keep them visibly unpromoted, clear the Phi-3 HOLD blockers before any Phi wording, and keep performance and GPU claims scoped to the exact row, host, and device they were measured on.

## Delivery sequence: now, next, later

This is the highest-level execution order. **Now** protects the current gate and clears the next blocker. **Next** is what Camelid may promote once bounded evidence exists. **Later** stays intentionally downstream of correctness and support-discipline work.

### Now

Protect every supported lane and clear the next blocker before widening claims.

- Protect the validated TinyLlama Q8_0 gate.
- Protect the exact Llama 3.2 1B bounded 512/1024/2048/4096/8192 row, the Llama 3.2 3B anchored raw-decode ladder row, and the exact Llama 3 8B bounded 512/1024/2048 row.
- Protect the Mistral 7B Instruct v0.3 Q8_0, five dense Qwen3, three Ornith `qwen35`, Gemma 3 1B, Gemma 4 E2B/E4B, seven PrismML Bonsai, and four certified Llama requant rows across docs, `/api/capabilities`, and frontend readiness.
- Keep the parity-locked mode for Qwen3 rows thinking-DISABLED; keep opt-in thinking mode described as a leading-trace lane only.
- Keep Mixtral fail-closed until later-generation divergence and the continuation hang are fixed and rerun through API/WebUI/RSS/frontend evidence.
- Keep LFM2.5 and the two distributed Gemma 4 rows visibly active-validation, not supported.
- Keep the Phi-3 row downloadable but unadvertised until the SPM rstrip seam and temperature-0 non-determinism are resolved.
- Keep every CUDA claim scoped to its recorded GPU, driver, and CUDA version, and keep Ubuntu x86 Q8 acceleration default-off until route hit, parity, repeated same-host timing, and whole-model impact are proven.
- Keep README, `COMPATIBILITY.md`, `ROADMAP.md`, `STATUS.md`, `/api/capabilities`, the wiki, and frontend readiness copy aligned in the same change window.

### Next

Promote only what can be defended row by row.

- Close the active-validation set as exact-row lanes, never as family-wide claims: Mixtral 8x7B Instruct v0.1 Q8_0, LFM2.5-2.6B Q8_0, and the two distributed Gemma 4 rows.
- Move the bounded Llama rows toward full support: model-native/larger context beyond the checked packs, broader arbitrary/Jinja template coverage, production throughput, and portability. For Llama 3.2 3B specifically, re-anchor the May-era API/WebUI, template-shape, and perf/RSS evidence families to the canonical GGUF.
- Extend Windows coverage to the gaps its bundles name: WebUI/frontend smoke, longer/model-native context, thinking mode, and Windows parity bundles for the Llama 3.2 1B/3B rows.
- Take agent mode past macOS: Windows and Linux live-lane transcripts, so the host claim matches the CI-validated builds.
- Broaden quantization support only with per-quant tests, docs, and at least one row-specific real-model parity artifact. Metadata parsing alone never promotes a quant.
- Bring the next candidate families in as single exact rows — Qwen2.5-7B-Instruct Q8_0 and gemma-2-9b-it Q8_0 — with tokenizer/template fixtures and prompt-token parity before any runtime-support wording.
- Require separate evidence rows for any encoder, quant, GPU path, or classifier-head reranker beyond the exact Nomic embedding lane.

### Later

Broaden the product surface only after correctness and release discipline are stable.

- Richer OpenAI API completeness beyond the current supported subset: stateful Responses features, `/v1/messages`, and `/infill` stay fail-closed until their contracts exist.
- Measured performance optimization after correctness gates are stable.
- Packaging and portability work across non-primary platforms.
- Broader model-family expansion, still row by row.
- First-class multi-model concurrency so Camelid can keep multiple local models loaded at once and serve agent workloads that need different models simultaneously.
- Treat specialized rows as validation candidates only until acquisition, tokenizer/runtime mapping, parity, API/WebUI, RSS, and throughput evidence exist for the exact row.

## Milestone table

| Milestone | Status | What must be true |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 supported gate | Complete | End-to-end generation parity artifacts exist and docs/API/frontend agree. |
| Llama 3.2 1B Instruct Q8_0 exact-row bounded support | Complete / bounded support | Compact parity, broader prompt-pack parity, API smoke, frontend smoke, exact-row metadata-Jinja row-template evidence, bounded template-shapes, unique-chat RSS/perf, and bounded 512/1024/2048/4096/8192 context packs agree for this exact 1B Q8_0 row. |
| Llama 3.2 3B Instruct Q8_0 exact-row smoke | Complete / narrow support | The anchored 512/1024/2048/4096/8192 raw-decode context ladder and June capability receipts agree on the current canonical GGUF (sha256 `f34112a1…`); the May-era load/parity/API-WebUI/RSS evidence is retained as historical for the prior upload pending re-anchor. |
| Llama 3 8B Instruct Q8_0 exact-row bounded support | Complete / bounded support through checked 512/1024/2048 packs | Compact prompt-token/1-token/5-token/50-token parity, the three-prompt 50-token pack, API smoke, frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and the compact chat-template-shapes pack support this exact 8B Q8_0 row only. |
| Mistral 7B Instruct v0.3 Q8_0 exact-row smoke | Supported exact-row smoke | Tokenizer/template, 1-token generation, broader five-prompt/50-token parity, bounded 512/1024/2048, checked 4096/8192 context evidence, GPU-vs-CPU greedy parity, and the support-promotion API/WebUI smoke bundle agree for this exact row only. |
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
- Llama 3.2 3B Q8_0 is supported as an exact-row smoke lane with the anchored checked 512/1024/2048/4096/8192 raw-decode context ladder on the current canonical GGUF (sha256 `f34112a1…`) plus June capability receipts; the compact and broader three-prompt parity, canonical Ubuntu API/WebUI refresh at source head `e9f926ed1a65`, bounded unique-chat RSS/perf, and row-scoped metadata-Jinja/template-shape evidence were measured on the prior upload (sha256 `b5607b50…`) and are retained as historical evidence pending re-anchor; model-native/larger-context and broader arbitrary-template expansion remain gated.
- Llama 3 8B Q8_0 has verified bounded support with compact parity, the three-prompt 50-token pass, API/frontend smoke, bounded memory evidence, checked bounded 512/1024/2048 context packs, and one compact chat-template-shapes pack; model-native/larger context beyond checked packs, broader chat-template, production performance, and portability expansion remain gated.
- Mistral 7B Instruct v0.3 Q8_0 is supported exact-row smoke. Tokenizer/template, generation, checked context evidence, GPU-vs-CPU greedy parity, and the support-promotion API/WebUI smoke bundle agree on `supported_exact_row_smoke`; broader Mistral-family support, neighboring rows, other quants, model-native/larger context, and full support remain gated.
- Mixtral 8x7B Instruct v0.1 Q8_0 is active validation / partial backend runtime only. Later-generation divergence and the continuation hang block readiness.
- Qwen 2.5 7B and Gemma 2 9B remain planned exact-row candidates only; Gemma 4 exact rows (E2B-It/E4B-It Q8_0) carry their own committed parity packs and capability rows; Gemma 3 1B-It Q8_0 is supported exact-row chat smoke on the macOS Metal GPU-resident lane (its default lane there), with a committed above-sliding-window receipt to 2,403 prompt tokens and an explicit runnable-CPU fallback off Metal that the above-window claim does not travel to. No throughput claim is made for that lane.
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
- Llama 3.2 3B Q8_0 has the anchored checked bounded 512/1024/2048/4096/8192 raw-decode context ladder for the exact row's current canonical GGUF.
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

- streaming logprobs and multi-choice streaming (the non-streaming single-choice/multi-choice contracts are delivered)
- broader streaming metadata completeness (OpenAI `stream_options.include_usage` is delivered — a terminal usage chunk on chat-completions streaming, see `COMPATIBILITY.md`; further streaming metadata fields remain pending)
- stronger seeded sampling validation

### llama-server compatibility sequence

Tim's 2026-05-31 product decision is to do both: preserve Camelid's honest OpenAI-style subset as the stable user/developer path, and add llama-server-compatible routes/control-plane behavior incrementally where useful. llama-server is a reference target for API shape and client expectations only; Camelid does not copy source or claim full parity.

Sequenced plan:

1. Keep `/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming, cancellation behavior, errors, `/api/capabilities`, docs, and frontend readiness as the stable contract. Unsupported OpenAI fields stay typed errors until implemented and tested.
2. Grow read-only llama-server discovery first: `/props`, `/slots`, `/models`, health/model metadata, and WebUI probes may expose public readiness state, but local paths stay redacted and readiness fails closed unless the loaded exact row is contract-supported and generation-ready. `/models` is limited to currently loaded Camelid models until router-mode cache listing, reload/autoload, native load/unload, and model-source metadata are designed.
3. Keep tokenizer/control-plane utilities bounded: `/tokenize`, `/detokenize`, and `/apply-template` may work for loaded supported tokenizer/template lanes only; `/tokenize` may expose bounded `with_pieces=true` id/piece objects for supported tokenizers, while arbitrary template kwargs, prompt-cache metadata, and slot lifecycle actions remain unsupported until real semantics exist.
4. Add native generation compatibility only after request mapping is explicit: `/completion` must translate supported llama-server parameters onto Camelid's generation path without weakening the OpenAI subset, and unsupported sampler, cache, image, infill, and tool fields must remain typed errors.
5. Keep stateful Responses features, Messages, multimodal routes, LoRA, router-mode model management, and full WebUI parity deferred until backend support, route semantics, tests, capabilities text, docs, and frontend gates all move together. Embeddings/reranking, metrics, and the stateless Responses subset have moved only within their documented exact boundaries.

Current gap sequence as of 2026-06-01:

1. Discovery parity: keep `/health`, `/v1/health`, `/props`, `/slots`, `/models`, `/v1/models`, `/v1/models/:model`, `/api/capabilities`, and frontend readiness fail-closed, privacy-safe, and tested before expanding writable control routes.
2. Tokenizer/template utilities: preserve loaded-model-only `/tokenize`, `/detokenize`, and `/apply-template`; `with_pieces=true` now has a bounded supported-tokenizer test, while template kwargs, richer error mapping, and tokenizer-piece edge cases outside supported lanes still need fixtures and capability text.
3. Native generation: map a narrow `/completion` request subset to the existing Camelid generation path only after sampler/stop/stream/timing/cancellation behavior is specified and unsupported fields stay typed.
4. WebUI control plane: add only probes that are read-only and support-contract-aware; keep slot cache actions, prompt-cache metadata, metrics, sleep/idle, router reload/autoload, and native model load/unload unsupported until semantics and tests exist.
5. Deferred runtime surfaces: stateful Responses chaining/storage, Messages, infill, multimodal inputs, LoRA adapters, router-mode model management, broader encoder/reranker rows, and broad llama-server WebUI parity remain blocked until backend support, docs, capabilities, and frontend gates move together.

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

### Model fit advisor (capacity axis)

A **capacity** signal, deliberately separate from the support ledger: it estimates whether the *local machine* can load/run a catalog row, and never implies parity or support. See [`docs/architecture/MODEL_FIT_ADVISOR_PLAN.md`](docs/architecture/MODEL_FIT_ADVISOR_PLAN.md).

- The verdict (`fits_resident` / `fits_with_offload` / `cpu_only_ok` / `wont_fit` / `unknown`) is derived from the detected `HardwareProfile` and the row's on-disk weight footprint; it is advisory, never a support claim, and degrades to `unknown` (never a failure) on hosts whose memory cannot be probed.
- It informs but never enforces: catalog download stays un-gated, and the pre-load fail-fast (`model_too_large_for_host`) fires only on a `wont_fit` verdict from a probed host and is overridable with `CAMELID_SKIP_FIT_CHECK=1`.
- The authoritative capacity guards remain the runtime VRAM headroom check (mid-load) and the KV predict-and-abort budget (mid-generation); the advisor never relaxes them.

### Agent mode lane (DROVER)

The terminal coding agent (`camelid chat --agent`, headless `camelid agent exec`) is
**Supported (experimental)** as of 2026-07-22 — scope contract in `COMPATIBILITY.md`, evidence in
`qa/evidence-bundles/agent-mode-supported-experimental-20260722/`, decisions in `DECISIONS.md` D18.
Remaining lane work before a full (non-experimental) claim: per-OS live-lane transcripts (Windows /
Linux), a digest-verifiable receipt family with a `verify-receipt` arm, an agent smoke lane in CI,
re-minting the stale pre-hardening `tool_capable` receipts (or recording accepted staleness), and
an `api_features` row for the feature axis. `/compact` and `/model` in-session commands remain
open items.

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
