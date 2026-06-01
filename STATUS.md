# Camelid Status

Last updated: 2026-06-01

`STATUS.md` is Camelid's current release-evidence checkpoint. It records what Camelid can prove today, what moved recently, and what still blocks the next support change. Treat it as a briefing memo, not a diary. Detailed historical run logs, older validation slices, and superseded tactical notes now live in [`STATUS_ARCHIVE_2026-04.md`](STATUS_ARCHIVE_2026-04.md).

Use this file to answer three practical questions: what is supported now, what changed recently, and what still blocks the next support move?

War-room claim discipline: [`docs/WAR_ROOM_EVIDENCE_INDEX.md`](docs/WAR_ROOM_EVIDENCE_INDEX.md) records the public source-of-truth order and minimum evidence checklist for support, benchmark, API capability, and WebUI readiness wording. It does not promote any row or feature by itself; it keeps this status memo, `COMPATIBILITY.md`, `BENCHMARKS.md`, `/api/capabilities`, and frontend readiness copy tied to scrubbed evidence.

Current API capability guard: read-only llama-server model discovery may be described only as partial when focused tests prove `GET /models` returns currently loaded Camelid models with local paths redacted. Native `POST /completion` may be described only as partial non-streaming generation compatibility for the tested `prompt` plus `n_predict`/`max_tokens` subset mapped onto Camelid's existing generation path. Router-mode cache listing, reload/autoload, native `POST /models/load`, native `POST /models/unload`, multimodal architecture metadata, native `/completion` streaming/slot/cache/timing parity, and full llama-server model-management parity remain unsupported, and these compatibility routes do not change WebUI readiness or exact-row support status.

Executive summary: Camelid has API + frontend exact-row validation for the Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B Q8_0 rows within their checked envelopes, with TinyLlama Q8_0 still the baseline gate. Llama 3.2 1B retains bounded 512/1024/2048/4096/8192-context packs where row-specific PASS bundles are cited; Llama 3.2 3B is supported only as exact-row smoke, with canonical Ubuntu API/WebUI refresh at source head `e9f926ed1a65` plus bounded 512/1024/2048-context packs where row-specific PASS bundles are cited; Llama 3 8B is promoted through exact-row smoke plus checked bounded 512/1024/2048-context packs where row-specific PASS artifacts are cited; Mistral 7B Instruct v0.3 remains active validation only, with tokenizer/template, 1-token, broader 50-token, checked 512/1024/2048/4096/8192-context, and fail-closed API/WebUI/RSS evidence, but no support promotion. Mixtral-8x7B-Instruct-v0.1 Q8_0 has bounded one-token backend MoE runtime evidence only; Gate 9A later-generation evidence diverges and the continuation lane recorded a backend HTTP hang, so no Mixtral API/WebUI/frontend readiness or broad support claim is active. The public support boundary moves only for supported exact rows and only for validated local-chat/parity envelopes; broad-family support, model-native/larger contexts beyond checked packs, arbitrary template execution beyond row-scoped renderer/template evidence, production throughput, and portability remain outside the support claim.

## Milestone snapshot for reviewers

| Exact row | What is now worth showing off | Highest checked context story | Hard stop / missing proof |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | The full current gate is refreshed with parity, template-shape, context, API/WebUI, and RSS/perf evidence. | Bounded 512-context pack is green for this exact row. | No blocker for the current supported claim; keep rerunning on support-contract changes. |
| Llama 3.2 1B Instruct Q8_0 | Verified bounded support moved beyond a demo: API, WebUI, compact/broader parity, exact-row metadata-Jinja row-template parity, bounded template-shapes, unique-chat RSS/perf, and post-fix 2048 evidence all agree. | 512, 1024, 2048, 4096, and 8192 bounded packs are green; 2048 is green only after the RoPE frequency-factor fix, 4096 is green on source/runtime head `470388f8165b`, and 8192 is green on source/runtime head `aaf9207d1669` for the compact-template recall pack. | Broader/full support still needs model-native/larger context, broader arbitrary-template coverage beyond the supported 1B metadata-Jinja row template, production throughput, portability, and durable full-support normalization. |
| Llama 3.2 3B Instruct Q8_0 | Exact-row smoke support is refreshed by the canonical Ubuntu source-head API/WebUI artifact: load, `/v1/completions`, `/v1/chat/completions`, `/api/capabilities`, frontend smoke, compact/broader parity, five-prompt API smoke, row-scoped metadata-Jinja/template-shape evidence, bounded unique-chat perf/RSS, and the opt-in parallel Q8 first-token direction probe all agree inside the documented envelope. | 512, 1024, and 2048 bounded packs are green for this exact row; the canonical Ubuntu API/WebUI support-gate refresh is `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json`. | Broader/full support still needs model-native/larger context beyond checked packs, broader arbitrary/Jinja template coverage beyond the row-scoped renderer/template-shape evidence, production throughput beyond bounded perf/RSS and the direction probe, portability, and durable full-support normalization. |
| Llama 3 8B Instruct Q8_0 | Verified bounded support is now public: compact parity, broader 50-token parity, API/WebUI, checked 512/1024/2048-context packs, compact template-shapes, memory evidence, structured Q8 read counters, and hot-path measurements are public. | 512/1024/2048 bounded packs are green for this exact row where row-specific PASS artifacts are cited, with 1024/2048 tied to source/runtime head `8e26be0a73c0`. | Hard boundary: no model-native/larger context beyond checked packs, broad 8B/Llama support, arbitrary templates, production throughput, portability, or neighboring-row/context promotion without row-specific evidence. |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | Active exact-row validation only: exact-row load, tokenizer/template, 1-token generation, broader 50-token parity, checked 512/1024/2048/4096/8192 context packs, and fail-closed API/WebUI/RSS evidence exist, but the support contract is still blocked. | 512, 1024, 2048, 4096, and 8192 bounded packs are green for this exact row only. | Any Mistral support/readiness claim, Mistral-family support, neighboring variants, arbitrary templates beyond the row-scoped renderer and shapes, model-native/larger context beyond the checked packs, or production throughput. |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | Active validation / partial backend runtime only: bounded one-token MoE evidence exists, but Gate 9A 50-token evidence diverges and the longer-continuation backend HTTP hang remains unresolved. | One-token backend runtime only; no API/WebUI/RSS/frontend readiness or context bucket is promoted. | Hard blocker: fix later-generation divergence and continuation hang before any Mixtral support/readiness claim. |

This is the repo story in one sentence: Camelid preserves one trusted small-model gate and three bounded Llama exact-row lanes, while keeping Mistral, Mixtral, Qwen, and Gemma fail-closed for every unproven support, context, quantization, and runtime path.

Active work now splits into two explicit tracks:

- **Four-row hardening:** keep TinyLlama as the current full gate, keep Llama 3.2 1B/3B and Llama 3 8B labeled as exact-row verified support within validated bounds until they meet the harder normalized bar, and treat CI reliability as a release blocker rather than a best-effort signal.
- **Active next-model bring-up set:** Camelid is publicly working on exact next-family rows, with Mixtral/Qwen/Gemma as follow-on candidates.
  - `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` — active validation / partial backend runtime only; bounded one-token MoE evidence exists, but Gate 9A later-generation divergence and the continuation backend HTTP hang block API/WebUI/frontend readiness and support promotion.
  - `Qwen2.5-7B-Instruct-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and architecture mapping still need row-specific proof.
  - `gemma-2-9b-it-Q8_0.gguf` — planned exact-row candidate; tokenizer/template and Gemma2 runtime behavior still need row-specific proof.

Public readiness language is locked to row-by-row evidence: the Mistral row may say only “active exact-row validation, not supported yet”; the Mixtral row may say only “bounded one-token backend MoE runtime evidence; later-generation/API/WebUI/frontend readiness blocked”; Qwen and Gemma may say “planned exact-row candidate, not supported yet.” Any Mistral support claim still requires explicit support-contract promotion after the fail-closed API/WebUI/RSS slice, and any Mixtral support/readiness claim requires fixing later-generation divergence and rerunning row-specific API/WebUI/RSS/frontend evidence.

Recent Mixtral hardening note: continuation/long-generation work is active blocker work. Exact prompt-token continuation input is useful for repro/debugging, but the lane remains blocked by later-generation divergence and a backend HTTP hang; it does not promote Mixtral support.

README screenshot note: capture and add a frontend screenshot only after the UI is demo-ready and visibly communicates the exact support contract. It should show truthful runtime/readiness state, not become a substitute for CI or support evidence.

## Release ledger snapshot

Camelid follows the same four-lane release ledger across the README, compatibility matrix, API capability reporting, and frontend readiness copy. If another surface sounds broader, treat it as stale and bring it back to this ledger. The purpose of this file is simple: record exactly what the current evidence can defend, no more and no less.

Reading rule for the matrix: each row should answer three questions in plain English — what is validated now, what gates are still missing, and what exact blocker prevents promotion to the next release label.

For a fast read, the current answer is:

- **Verified support gates:** TinyLlama 1.1B Chat Q8_0 remains verified, and the exact Llama 3.2 1B plus Llama 3 8B Instruct Q8_0 rows have verified support within their validated bounds. Llama 3.2 3B Instruct Q8_0 is supported as exact-row smoke only after canonical Ubuntu source-head load, completion, chat-completion, frontend validation, API capability, and parity evidence aligned.
- **Scope boundary:** Llama support is exact-row only: model version/size, Instruct variant, Q8_0 quantization, loaded runtime readiness, and the tested smoke/parity envelope all matter.
- **8B promotion:** Llama 3 8B Instruct Q8_0 has end-to-end bounded generation parity artifacts for compact parity, a three-prompt 50-token Ubuntu parity run, the bounded compact chat-template-shapes pack, API/frontend smoke, bounded-memory evidence, and checked 512/1024/2048-context packs for the exact tracked Q8_0 GGUF. The current-head 1024/2048 canonical pass at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` matched prompt tokens, generated token IDs, and generated text (`CMLD-102` / `CMLD-204`) for source/runtime head `8e26be0a73c0`; earlier `b49034007f2e`, `9e3c64f2cfab`, `160348118d44`, `9f8588bb4a4e`, `844e8d4709d1`, `86081876d01e`, and `e203d3cf3ea5` passes remain historical source-head evidence.
- **Explicit non-claim:** no broad Llama-family support exists today; neighboring variants remain unsupported unless they have their own exact row and evidence.

Two standing rules apply to every row:

- **Support rule:** Nothing inherits support across model size, quantization, tokenizer lane, API surface, or frontend state.
- **Credit rule:** Visible llama.cpp / ggml acknowledgement and the MIT notice remain part of any parity-backed release claim.

For the formal support ledger, see [`COMPATIBILITY.md`](COMPATIBILITY.md). For sequencing, see [`ROADMAP.md`](ROADMAP.md).

Bottom line for reviewers: Camelid has the original TinyLlama verified support gate plus three exact Llama Q8_0 rows with verified support within validated bounds. Mixtral remains partial runtime evidence only until later-generation divergence and API/WebUI/frontend readiness blockers close.

## Apple Silicon Q8 speed snapshot

The Mac Q8 lane now has a narrow retained same-host speed result worth surfacing, while keeping the production-throughput boundary closed.

- The strongest retained Llama 3.2 3B short-request candidate measured Camelid at about `478 ms` total wall time against llama.cpp at about `530 ms`, improving from the prior Camelid baseline of about `857 ms` total against llama.cpp at about `609 ms`.
- That result is still default-off and narrow-envelope only. Camelid first-token latency in the same retained candidate was still slower than llama.cpp (`478 ms` vs `290 ms`), and decode-heavy evidence still shows residual first-content/TTFT gaps.
- Current source work adds deeper post-first-token timing, layer-role timing, output-logits profiling, FFN decode-chain telemetry, and default-off FFN decode-chain route experiments so the remaining gap can be attributed precisely. This is performance instrumentation and experimental routing, not a support-contract expansion.

Boundaries that remain in force:

- No default-on Apple Silicon Q8 acceleration claim.
- No broad production-throughput claim.
- No portability claim from Mac evidence.
- No support expansion for neighboring models, quantizations, contexts, or API/frontend readiness from speed work alone.

## Ubuntu x86 Q8 acceleration update

Recent Ubuntu x86 Q8 work has significantly improved the experimental accelerated path while keeping the default/reference path intact. The current work focuses on packed Q8 runtime storage, matrix-level execution, and AVX2 packed kernels. These paths remain default-off while validation continues; they are production-directional, not production-ready.

Current public takeaways:

- The retained Ubuntu x86 Q8 candidate remains a **default-off** acceleration path guarded by `CAMELID_X86_Q8_REPACK=on` plus the measured AVX2 kernel/runtime gates used in the current Ubuntu lane.
- Packed Q8 runtime storage now covers the dense attention projection family plus FFN gate/up/down rows for the measured Llama 3.2 3B Instruct Q8_0 lane, while preserving the safe fallback path when gates are absent or disabled.
- Evidence bundles now include parity checks, repeated timing discipline, perf counters, hot-symbol captures, explicit reject notes for non-retained experiments, and cold-vs-warm request evidence.
- Rejected Ubuntu x86 Q8 candidates are being documented instead of hidden; no claimed win is retained unless it survives repeated confirmation with checksum/text preservation on a clean host.
- The cold/warm split is now explicit in the evidence: `from_q8_0_bytes` is a cold/reload materialization cost on this lane, not the warm decode bottleneck.
- The active warm-path direction has shifted from leaf row-dot tuning toward matrix-level Q8 GEMM/MUL_MAT ownership, starting with deeper FFN ownership slices.
- Latest local-only follow-on work tightened default-off paired/triplet helpers and one-row packed-runtime decode output-group traversal so related projections reuse shared quantized inputs and wide rows4 decode projections can schedule independent output groups while consuming backend-owned packed runtime storage; Ubuntu x86_64 timing/profiling proof is still pending, so no measured-effect or support claim is added from those tweaks.
- Latest bounded output-slice work added a default-off multi-row `output.weight` PackedRows4 matmul consumer (`CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL`) that consumes backend-owned runtime storage only and is managed off by ExecutionPlan; local parity/gate tests pass, but no Ubuntu x86_64 timing/profiling validation is recorded for that local slice, so no retained measured-effect or support claim is added.
- Latest local-only packed-rows4 matmul follow-ons chunk parallel output-group traversal and reuse bounded quantized-input scratch for existing single, paired, and triplet multi-row helpers; they keep I8/matching-layout/backend-owned-storage guards and have local fmt/clippy/unit/timing-smoke coverage only, with no Ubuntu x86_64 timing/profiling validation recorded for those local slices.
- Latest docs/context guard keeps FFN-down GEMM4 AVX2 and output-route-resolver work in the evidence-needed lane: the same-host guard still rejects any Camelid speed promotion, and output route cleanup is implementation guidance only until local plus canonical Ubuntu gates prove it.
- Latest retained default-off hygiene slices are narrow: FFN-down GEMM4 row-group scheduling has a min-input-groups guard for the shallow-prefill synthetic surface, and ExecutionPlan now clears the FFN gate/up single-owner env gate. These are scheduler/control-plane guards only; they do not widen throughput, support, portability, API/frontend readiness, or default-on claims.
- Latest default-off VNNI decode slice adds llama.cpp-style Q8_0 tile16 packing and an `M == 1` FFN-down route behind `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE`. Local fallback/telemetry tests pass, and same-host Ubuntu unit validation proves the raw tile layout plus `ffn_down.x86_vnni_decode_consumer` route selection/counters. This is implementation groundwork only; no same-host timing win, default-on change, support expansion, or throughput claim is made until the retained benchmark/parity bundle exists.
- Latest retained same-host AVX2 baseline recheck (`qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260524T0535Z-avx2-register-sums/README.md`) keeps the Ubuntu x86 Q8 lane negative for Llama 3.2 3B Q8_0: exact-row one-token parity still matched llama.cpp, but the cleaned unique-prompt marker run measured Camelid at about `8837.65 ms` total against llama.cpp at about `498.45 ms` total, and Camelid was also slightly slower than the retained `214f733` baseline (`8823.37 ms`). Keep this as bottleneck evidence only; it does not justify any performance retention or promotion.
- The follow-on FFN decode-chain same-host artifacts on current `origin/main` keep the same boundary: the missing-gate run showed `ffn_decode_chain_taken=0`, and the complete-gate run proved route use plus one-token parity, but Camelid still trailed llama.cpp (`5703.86 ms` TTFT and `2008 ms` backend generate versus llama.cpp `162.56 ms` TTFT and `284.37 ms` total). These artifacts guide the next route/gate fix and do not promote throughput, support, portability, larger context, neighboring rows, broad Llama-family claims, or any default-on path.
- Latest docs host-reporting retained audit (`qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-5e4b0b83-20260520T1024Z-docs-host-reporting-retained-audit/README.md`) kept the canonical Ubuntu reporting rule green with a focused stale host-failure wording scan across public docs/source/status. Remote validation was not attempted in that docs-only run, so it makes no host availability or failure claim.

Boundaries that remain in force:

- No default-on acceleration claim.
- No broad production-throughput claim.
- No portability claim beyond the measured Ubuntu x86_64 lane.
- No broader model-family or neighboring-row support claim from this work alone.

Primary public evidence anchors for this lane are summarized in [`docs/performance/ubuntu-x86-q8.md`](docs/performance/ubuntu-x86-q8.md).

## Durable evidence anchors

- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json` plus `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS` are the committed carry-forward row bundles/checksums for the public smoke boundary.
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json` is the committed Ubuntu perf/portability summary for the current four-row sweep.
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/manifest.json` plus its per-row manifests/checksums are the durable current-head citation target for exact rerun tracks, blocker notes, and command files.
- `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json` is the sanitized API + frontend smoke summary for all four exact rows on a clean public checkout; `qa/evidence-bundles/four-row-api-only-20260504T230722Z-head-13a465608fbf/manifest.json` is the narrower API-only predecessor.
- `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/manifest.json` is the current-head normalized TinyLlama/1B/3B API/WebUI smoke bundle from the reopened Ubuntu lane; all three rows passed and the public bundle checksum file verifies with `SHA256SUMS` sha256 `ce87c02fba64fcd78efe10c01b030435d185bc785f06a4d9df4cbd04048da283`.
- `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json` is the canonical Ubuntu source-head refresh for the exact Llama 3.2 3B Instruct Q8_0 API/WebUI support gate at source head `e9f926ed1a65`: model load, `/v1/completions`, `/v1/chat/completions`, generation timing summary, `/api/capabilities`, and frontend smoke all passed with the expected `supported_exact_row_smoke` contract. This refreshes the already-supported exact-row API/WebUI box only; it does not promote larger context, arbitrary templates, production throughput, portability, neighboring rows, or broad-family support.
- `qa/evidence-bundles/supported-row-template-throughput-blocker-proof-20260513T2049Z-head-994569dbf995/manifest.json` records the current supported-row closeout proof for broad arbitrary/Jinja-template and production-throughput caveats: new runtime/API guardrails passed and show Camelid does not claim arbitrary Jinja templates as a broad supported renderer, does not expose production-throughput evidence for current supported rows, and has a shortest path for clearing the remaining blockers with broader template coverage plus production-throughput harness evidence. The exact 1B metadata-Jinja row-template path is now covered by focused renderer/API-contract evidence.
- `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/manifest.json` is the current-head TinyLlama final-normalization bundle: broader five-prompt/50-token marker-template parity, marker-template-shape parity, bounded 512-context parity, and backend RSS/perf sampling all passed; max backend RSS was `105120 KiB`, and bundle-local `SHA256SUMS` verification passes.
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json` is the clean-public-head Llama 3.2 1B/3B bounded compact chat-template-shapes bundle; both exact rows passed all four checked prompts with prompt-token, generated-token, and generated-text parity, closing only that compact template-shape box.
- `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json` is the clean-public-head Llama 3.2 1B/3B bounded unique-chat perf/RSS envelope bundle; both exact rows completed four measured unique chat requests after two warmups with hot weight-cache measured runs and RSS milestones, closing only that bounded memory/perf envelope box.
- `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json` is the exact Llama 3.2 3B opt-in parallel Q8 first-token runtime direction bundle; validation gates passed, bundle checksums verify, and the one-warmup/one-measured comparison moved generate time `13960 -> 12200 ms` with max sampled backend RSS `283.57 -> 282.97 MiB`. This closes only the **Llama 3.2 3B Instruct Q8_0 → performance lane → opt-in parallel Q8 first-token runtime direction sub-box**, not production-throughput or portability support.
- `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json` is the current-public-head normalized Llama 3 8B Instruct Q8_0 API/WebUI/RSS smoke bundle from the reopened Ubuntu lane; the exact row passed load/models/capabilities/completions/chat/timing-summary/frontend smoke, frontend chat generated `Hello`, max sampled backend RSS was `283372 KiB`, and `SHA256SUMS` sha256 is `83334a9083806081569322978db273044753515a195359d0b4326cf6352367da`.
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json` records the first bounded 512-context pack pass across TinyLlama 1.1B Chat Q8_0, Llama 3.2 1B/3B Instruct Q8_0, and Llama 3 8B Instruct Q8_0 from clean public head `b403884`; all four rows matched prompt tokens, generated token IDs, and generated text for the checked 5-token pack. This does not promote larger context buckets, arbitrary templates, or broad family support.
- `qa/evidence-bundles/llama32-1b-context-1024-20260505T081001Z-head-156ded6fc76b/manifest.json` records the second bounded 1024-context pack pass for the exact Llama 3.2 1B Instruct Q8_0 row from clean public head `156ded6fc76b`; the 881-token prompt matched prompt tokens, generated token IDs `[34,2735,35,12,4278]`, and generated text `CMLD-102` for the checked 5-token pack. This closes only that exact 1B row+box slice.
- `qa/evidence-bundles/llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/manifest.json` records the second bounded 1024-context pack pass for the exact Llama 3.2 3B Instruct Q8_0 row from clean public head `c14e5e7b5692`; the 881-token prompt matched prompt tokens, generated token IDs `[34,2735,35,12,4278]`, and generated text `CMLD-102` for the checked 5-token pack. This closes only that exact 3B row+box slice.
- `qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json` records the third bounded 2048-context pack pass for the exact Llama 3.2 3B Instruct Q8_0 row from clean public head `36ec8e492d65`; the 1910-token prompt matched prompt tokens, generated token IDs `[34,2735,35,12,7854]`, and generated text `CMLD-204` for the checked 5-token pack. This closes only that exact 3B row+box slice and does not lend 2048-context support to 8B, model-native context, or broad Llama-family claims.
- `qa/validation-notes/2026-05-05-llama32-1b-context-2048-blocker.md` records the earlier failed third bounded 2048-context attempt for the exact Llama 3.2 1B Instruct Q8_0 row before the RoPE frequency-factor fix. Keep it as superseded blocker history, not current support truth.
- `qa/validation-notes/2026-05-05-llama32-1b-q8-2048-current-main-watchdog.md` records a clean-current-main watchdog rerun before the RoPE frequency-factor fix; it is retained as superseded red-box history and no longer reflects current head.
- `qa/validation-notes/2026-05-05-llama32-1b-q8-2048-cache-probe.md` records the pre-fix Q8 file-cache probe: a diagnostic 256 MiB Q8 file cache did not change the then-divergent first token and raised RSS, so cache-based promotion was rejected. The later RoPE frequency-factor pass supersedes its red first-token result.
- `qa/validation-notes/2026-05-05-q8-borrowed-batch-read-reuse.md` records the follow-up structural Q8_0 read-reuse patch: borrowed/token-major file-backed Q8 matmul now routes eligible batches through the chunked Q8 block reader so each Q8 chunk is read once across input rows. This is local code evidence only and did not itself change any support row or 8B context status.
- `qa/validation-notes/2026-05-05-q8-layer-phase-read-trace.md` records the follow-up structured Q8_0 I/O tracing patch: layer memory timings now include phase-attributed Q8 file-read/cache-hit deltas so future 1B/2048 and Q8 architecture traces can identify the exact layer phase streaming file-backed Q8 payload. This is code-only instrumentation and does not change any row status.
- Current structural headroom work keeps layer-major prefill bounded while sizing scoped Q8 reuse carefully and bounding retained scratch. That code/test evidence does not by itself promote production throughput or broader/full support; the fresh current-head 8B 1024/2048 canonical PASS plus docs/API/frontend alignment closes only those exact checked bounded packs.
- `qa/evidence-bundles/llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/manifest.json` records the post-fix third bounded 2048-context pack pass for the exact Llama 3.2 1B Instruct Q8_0 row: the 1910-token prompt matched prompt tokens, generated token IDs `[34,2735,35,12,7854]`, and generated text `CMLD-204` for the checked 5-token pack. This closes only that exact 1B row+box slice after the RoPE frequency-factor fix.
- `qa/evidence-bundles/llama32-1b-context-4096-current-head-20260513T163426Z-head-470388f/manifest.json` records the latest fourth bounded 4096-context compact-template pack pass for the exact Llama 3.2 1B Instruct Q8_0 row on source/runtime head `470388f8165b`: the 3755-token prompt matched prompt tokens, generated token IDs `[34,2735,35,12,12378]`, and generated text `CMLD-409` for the checked 5-token pack. This closes only that exact 1B row+4096 box slice, not neighboring rows or model-native/larger context beyond checked packs; the older `qa/evidence-bundles/llama32-1b-context-4096-current-head-20260513T155455Z-head-039353c/manifest.json` remains historical source-head evidence.
- `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json` records the latest fifth bounded 8192-context compact-template pack pass for the exact Llama 3.2 1B Instruct Q8_0 row on source/runtime head `aaf9207d1669`: the 7650-token prompt matched prompt tokens, generated token IDs `[34,2735,35,12,18831]`, and generated text `CMLD-819` for the checked 5-token pack. This closes only that exact 1B row+8192 box slice, not neighboring rows, model-native context beyond checked packs, broader arbitrary templates beyond the supported 1B metadata-Jinja row template, production throughput, or portability.
- `qa/validation-notes/2026-05-05-llama3-8b-context-1024-blocker.md` records the earlier required 1024 diagnostic timeout for the exact Llama 3 8B Instruct Q8_0 row; keep it as superseded red-box history.
- `qa/validation-notes/2026-05-05-llama3-8b-context-2048-blocker.md` records the earlier failed 2048 attempt; keep it as superseded red-box history.
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json` plus `SHA256SUMS` is the current-head exact Llama 3 8B Q8_0 bounded 1024/2048 PASS bundle for source/runtime head `8e26be0a73c0`; it matched prompt tokens, generated token IDs, and generated text for `CMLD-102` and `CMLD-204`, closing only those exact bounded buckets.
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T024342Z-head-b49034007f2e/manifest.json` plus `SHA256SUMS` is the previous exact Llama 3 8B Q8_0 bounded 1024/2048 PASS bundle for source/runtime head `b49034007f2e`; it remains historical source-head evidence after later commits.
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T002636Z-head-9e3c64f2cfab/manifest.json` plus `SHA256SUMS` is the previous committed exact Llama 3 8B Q8_0 bounded 1024/2048 PASS bundle for source/runtime head `9e3c64f2cfab`; it remains historical source-head evidence after later commits. `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T210500Z-head-160348118d44/manifest.json` is the older predecessor for source/runtime head `160348118d44`.
- `qa/evidence-bundles/mistral-7b-v0.3-q8-broader-50tok-ubuntu-20260509T000633Z-head-d330e97ae992/manifest.json` plus `SHA256SUMS` records the exact Mistral 7B Instruct v0.3 Q8_0 broader five-prompt/50-token Ubuntu parity pass: prompt tokens, generated token IDs, and generated text all matched llama.cpp.
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-backend-parity-refresh-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-api-smoke-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-webui-readiness-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-rss-timing-runtime-20260511/`, and `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-manifest-checksum-20260511/` are retained historical promotion-candidate artifacts only; current support wording is superseded by later blocker evidence.
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-gate9a-50tok-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-longgen-continuation-20260511/`, and `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/` are the current Mixtral blockers: Gate 9A later-generation divergence and a backend HTTP hang prevent Mixtral API/WebUI/frontend readiness or support promotion.
- `qa/evidence-bundles/mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/manifest.json` plus `SHA256SUMS` records the exact Mistral 7B Instruct v0.3 Q8_0 checked 4096/8192-context Ubuntu parity pass: prompt tokens, generated token IDs, and generated text all matched llama.cpp; actual prompt-token counts were 3818 and 7818. This extends the row's active-validation evidence only; the current-head API/WebUI/RSS bundle still keeps the support contract fail-closed.
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T122749Z-head-9f8588bb4a4e/manifest.json` plus `SHA256SUMS` is historical source-head 8B 1024/2048 bounded-pack PASS evidence for source head `9f8588bb4a4e`; it matched prompt tokens, generated token IDs, and generated text for `CMLD-102` and `CMLD-204`.
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T062823Z-head-86081876d01e/manifest.json` plus `SHA256SUMS` is the immediate historical predecessor for source head `86081876d01e`; it matched the same bounded 1024/2048 packs on that source head only.
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T040523Z-head-e203d3cf3ea5/manifest.json` plus `SHA256SUMS` is historical 8B 1024/2048 bounded-pack PASS evidence for source head `e203d3cf3ea5`; it matched prompt tokens, generated token IDs, and generated text for `CMLD-102` and `CMLD-204` on that source head only.
- Older 8B 1024/2048 artifacts, including `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260507T194559Z-head-ab8e465b50c3/manifest.json`, `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260507T113718Z-head-b3633cbd6271/manifest.json`, `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260507T075407Z-head-4191ef9fcc28/manifest.json`, `qa/evidence-bundles/llama3-8b-context-1024-20260506T182100Z-head-e146d3b335d8/manifest.json`, and `qa/evidence-bundles/llama3-8b-context-2048-20260506T182534Z-head-e146d3b335d8/manifest.json`, remain review/superseded history for their source heads.
- Runtime/API alignment is now explicit: `/api/capabilities` exposes 512/1024/2048/4096/8192 context fields and promotes only the exact buckets with row-specific PASS bundles: Llama 3.2 1B through 8192, and Llama 3.2 3B plus Llama 3 8B through 2048. These remain exact bounded-pack claims only.
- `qa/evidence-bundles/8b-checkmark-current-head-20260505T052647Z-head-864e07b51f36/manifest.json` is the earlier public-main Llama 3 8B Instruct Q8_0 checkmark refresh at head `864e07b51f36`.
- `qa/evidence-bundles/8b-checkmark-current-main-20260505T084931Z-head-15bfc41d15d5/manifest.json` is the latest public-main Llama 3 8B Instruct Q8_0 checkmark refresh at head `15bfc41d15d5`; API/WebUI/RSS smoke passed load/models/capabilities/completions/chat/timing-summary/frontend smoke, frontend chat generated `Hello`, max sampled backend RSS was `286148 KiB`, and `SHA256SUMS` sha256 is `a2d7a7266bc1de9fec1f8143492661d29e68c2902ee70d3b1c60e2f2673892a6`. This preserves exact-row smoke support only.
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json` records the reopened-lane pass for the first bounded 8B 512-context pack.
- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json` records the reopened-lane pass for the bounded 8B broader three-prompt 50-token pack.
- `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json` records the reopened-lane pass for the bounded 8B compact chat-template-shapes pack.
- `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json` records the clean-main reopened-lane exact 8B API/WebUI smoke pass for the completion-diagnostics API path: `/v1/completions` carries response-local timing diagnostics, the promotion smoke timing summary passed, frontend chat generated `Hello`, and RSS was sampled around the smoke window. This is an exact-row API evidence slice, not a new broad support claim. The earlier patched-tree predecessor remains at `qa/evidence-bundles/llama3-8b-api-webui-rss-20260505T014408Z-head-8cef7af4d6c6/manifest.json` for audit history.
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json` records exact 8B retained-block lazy-Q8 hot-path cost probes for representative FFN tensors and `output.weight`, including the guarded swapped logical row/column interpretation. This is measurement evidence only; it does not promote broad 8B/full-context/arbitrary-template support or performance portability.
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json` validates the reusable hot-path bundle helper on clean public `main` at `e22307f2f90b` and repeats the exact-row retained-block measurements (`35.78 ms`, `35.73 ms`, `320.24 ms` for the checked tensors). This is helper/measurement evidence only.
- Raw `target/` paths below are drill-down artifacts only; they are not the sole public evidence anchor.

## What changed in this support line

Recent work moved the exact-row release ledger in a narrow, evidence-backed way:

- TinyLlama Q8_0 remains the trusted supported gate; its final-normalization bundle now closes the current-head broader parity, marker-template-shape, bounded 512-context, and backend RSS/perf box for the exact current-gate row.
- Llama 3.2 1B Q8_0 moved from evidence-only to verified bounded support after compact parity, broader prompt-pack parity, `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned. Its bounded compact chat-template-shapes box, bounded unique-chat perf/RSS envelope box, second bounded 1024-context parity box, third bounded 2048-context parity box after the RoPE frequency-factor fix, fourth bounded 4096-context parity box and fifth bounded 8192-context parity box on their cited source/runtime heads are now green for the exact row only.
- Llama 3.2 3B Q8_0 moved from acceptance target to supported exact-row smoke after exact-row load, compact prompt-token/1-token/5-token/50-token parity, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence aligned. Its bounded compact chat-template-shapes box, bounded unique-chat perf/RSS envelope box, second bounded 1024-context parity box, third bounded 2048-context parity box, and opt-in parallel Q8 first-token runtime direction sub-box are now green for the exact row only.
- The 3B broader JSON-shaped prompt divergence is now resolved: a post-Q8-dot rerun of the three-prompt, 50-token pack matched llama.cpp for prompt tokens, generated token IDs, and generated text.
- Llama 3 8B Q8_0 moved from groundwork-only to verified bounded support after Ubuntu three-prompt parity, API/frontend smoke, and bounded memory evidence aligned; the current public broader-pack rerun is the bounded three-prompt 50-token pass.
- Bounded 512-context evidence now exists for the original four Llama-family exact Q8_0 rows. This is checked bounded-pack evidence only, not a model-native context or portability claim. The exact Llama 3.2 1B, Llama 3.2 3B, and Llama 3 8B rows have checked support through 1024-context and 2048-context packs where listed; the exact Llama 3.2 1B row is also checked through 4096 and 8192 where cited.

Bottom line: the engineering seam and product surface now agree for exact 1B/3B/8B short chat/parity plus row-specific bounded-context blockers; the support language stays intentionally narrow.

## Repo-health verification pass

A fresh local repo-health pass ran on 2026-05-04 to keep the public tree and CI contract honest before heavier model work resumes.

Verified locally on the current tree:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all-targets --all-features`
- `cargo doc --no-deps --all-features`
- `bash scripts/check-public-scrub.sh`
- `cd frontend && npm ci && npm run build && npm run smoke:model-state`

Result: all of the above passed locally. The CI workflow was also tightened so the Rust job now enforces clippy and docs generation in addition to format and tests, and the frontend job now runs the support-contract/model-state smoke gate that protects exact-row chat unlock behavior. This keeps the GitHub gate aligned with the documented validation contract.

## Current support evidence

The sections below summarize the artifact-backed boundary for each tracked row. They are intentionally narrower than "what might be close." If a supporting artifact is not called out here or in the linked files, Camelid should not imply the claim elsewhere.

### TinyLlama 1.1B Chat Q8_0

Status: **supported current gate**

Current evidence boundary:

- Five-prompt, 50-token parity audit against known-good llama-server.
- Prompt token IDs, generated token arrays, and generated text match.
- The token-major `output.weight` interpretation remains a protected correctness guardrail.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/tinyllama_1_1b_chat_q8_0.bundle.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/tinyllama_1_1b_chat_q8_0/manifest.json`
- `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/manifest.json`

The older five 50-token source JSONs remain listed under that current-head row manifest's `broader-parity` carry-forward track instead of standing alone as the release citation. The current-head TinyLlama normalization bundle is now the durable citation for broader marker-template parity, compact template-shape behavior, bounded 512-context coverage, and backend RSS/perf sampling on this exact row.

### Llama 3.2 1B Instruct Q8_0

Status: **verified support (bounded)**

Current evidence boundary:

- Compact-header `hello` matches llama.cpp through the completed bounded run on Ubuntu.
- Prompt token IDs, generated token IDs, and generated text all match for the compact bounded response.
- The broader five-prompt parity pack also passed for this exact 1B row.
- The bounded compact chat-template-shapes pack passed all four checked prompt shapes for this exact 1B row.
- The second bounded 1024-context pack passed for this exact 1B row: actual reference prompt tokens `881`, generated tokens `[34,2735,35,12,4278]`, generated text `CMLD-102`, and timed-process max RSS `2897852 KiB`.
- The bounded unique-chat perf/RSS envelope passed for this exact 1B row: four measured unique chat requests after two warmups, hot weight-cache measured runs, no prompt-cache hits, average wall time `7379.73 ms`, average backend generate time `7065.25 ms`, and max sampled backend RSS `274.31 MiB`.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke evidence are aligned with `/api/capabilities`.
- The third bounded 2048-context pack is now green for this exact 1B row after the RoPE frequency-factor fix: prompt-token parity passed at 1910 prompt tokens, generated tokens matched `[34,2735,35,12,7854]`, and generated text matched `CMLD-204`.
- Runtime code now interprets GGUF `rope_freqs.weight` as llama.cpp-style RoPE frequency factors; the fresh Ubuntu 2048 prompt-pack artifact above promotes only this exact checked 2048 pack.
- The fourth bounded 4096-context pack and fifth bounded 8192-context pack are green for this exact 1B row: 4096 matched `CMLD-409` on source/runtime head `470388f8165b`, and 8192 matched `CMLD-819` on source/runtime head `aaf9207d1669` with 7650 prompt tokens and max timed-process RSS `3806876 KiB`.
- Follow-up structural Q8 code now reuses file-backed borrowed/token-major Q8 chunks across batched input rows when the weight layout can use the existing chunked Q8 block reader; this is covered by local guardrails only and does **not by itself** promote production throughput or broader/full support. The current-head 8B 1024/2048 canonical PASS plus docs/API/frontend alignment closes only those exact bounded packs.
- The support claim is limited to this exact 1B Instruct Q8_0 row, short local-chat smoke, the checked first bounded 512-context pack, the checked second bounded 1024-context pack, the checked third bounded 2048-context pack, the checked fourth bounded 4096-context pack, the checked fifth bounded 8192-context pack, the checked compact chat-template-shapes pack, the exact-row metadata-Jinja row-template path, and the bounded unique-chat perf/RSS envelope; neighboring Llama rows, other quantizations, model-native/larger contexts beyond checked packs, broader arbitrary GGUF/Jinja template execution, broad chat-template behavior, production throughput, and portability remain outside the claim.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama32_1b_instruct_q8_0.bundle.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_1b_instruct_q8_0/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-1024-20260505T081001Z-head-156ded6fc76b/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-4096-current-head-20260513T163426Z-head-470388f/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json`
- `qa/validation-notes/2026-05-05-llama32-1b-rope-freqs-runtime.md` (code/runtime slice only; not a 2048-context PASS artifact)

### Llama 3.2 3B Instruct Q8_0

Status: **supported exact-row smoke**

Current evidence boundary:

- The exact tracked GGUF is present locally.
- The exact tracked GGUF loads successfully through `/api/models/load` with low backend RSS after streaming metadata parsing.
- Recent file-backed lazy-Q8 recovery materially reduced the older eager dense-load spike.
- The Ubuntu compact-header `hello` harness matches llama.cpp for prompt tokens plus deterministic 1-token, 5-token, and bounded 50-token generation.
- The bounded compact chat-template-shapes pack passed all four checked prompt shapes for this exact 3B row.
- The bounded unique-chat perf/RSS envelope passed for this exact 3B row: four measured unique chat requests after two warmups, hot weight-cache measured runs, no prompt-cache hits, average wall time `19762.21 ms`, average backend generate time `19449.25 ms`, and max sampled backend RSS `287.21 MiB`.
- The canonical Ubuntu API/WebUI support-gate refresh at `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json` passed exact-row load, `/v1/completions`, `/v1/chat/completions`, generation timing summary, `/api/capabilities`, and frontend smoke with `compatibility_status=supported_exact_row_smoke`, `contract_supported=true`, and `webui_chat=enabled` for source head `e9f926ed1a65`.
- The second bounded 1024-context pack passed for this exact 3B row: actual reference prompt tokens `881`, generated tokens `[34,2735,35,12,4278]`, generated text `CMLD-102`, and timed-process max RSS `7331624 KiB`.
- The third bounded 2048-context pack passed for this exact 3B row: actual reference prompt tokens `1910`, generated tokens `[34,2735,35,12,7854]`, generated text `CMLD-204`, and timed-process max RSS `7450092 KiB`.
- The opt-in parallel Q8 first-token runtime direction sub-box passed for this exact 3B row: generate time moved `13960 -> 12200 ms`, layer time `13847.54 -> 12110.67 ms`, and max sampled backend RSS `283.57 -> 282.97 MiB` after the validation gates and public checksum verification.
- The support claim is limited to this exact 3B Instruct Q8_0 row, the validated local-chat/parity/API/frontend envelope, the checked first bounded 512-context pack, the checked second bounded 1024-context pack, the checked third bounded 2048-context pack, the row-scoped metadata-Jinja/template-shape evidence, the bounded unique-chat perf/RSS envelope, and the opt-in parallel Q8 first-token runtime direction sub-box; model-native/larger contexts beyond checked packs, broader arbitrary GGUF/Jinja template execution, production throughput beyond bounded perf/RSS and the direction probe, portability, neighboring rows, and broad-family behavior remain follow-up gates.

Representative durable evidence:

- `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama32_3b_instruct_q8_0.bundle.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama32_3b_instruct_q8_0/manifest.json`
- `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/manifest.json`
- `qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json`
- `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json`

Selected source artifacts recorded by those committed files:

- `target/ubuntu-followup-20260502T015231Z/llama32_3b-50tok.json` preserves compact bounded parity inside the carry-forward bundle.
- `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json` is the post-Q8-dot broader three-prompt clean rerun called out by the current-head row manifest notes.

Expansion beyond the current supported row remains blocked until Camelid has model-native/larger context coverage beyond the checked packs plus production performance, portability, and broader arbitrary-template evidence beyond the supported 1B metadata-Jinja row template.

### Llama 3 8B Instruct Q8_0

Status: **verified support (bounded)**

Current evidence boundary:

- Metadata, config, tokenizer, and chat-template handling are fixture-guarded.
- Independent tokenizer reference fixtures exist.
- Lazy/file-backed Q8 execution is now good enough for repeat bounded parity on the exact tracked Q8_0 GGUF.
- The Ubuntu compact-header `hello` harness matches llama.cpp for prompt tokens and deterministic generation at 1, 5, and bounded 50-token lengths on this exact row.
- The Ubuntu three-prompt 50-token parity run passed for `hello`, alpacas, and JSON with prompt tokens, generated token IDs, and generated text all matching the known-good reference.
- The first bounded 512-context pack now passes on the reopened Ubuntu lane for the exact 8B row at `58acf592345c69c1b684544124cd23804e2899f1`, and the later four-row public summary at `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json` records matching prompt tokens, generated token IDs, and generated text for TinyLlama plus the exact Llama 3.2 1B/3B and Llama 3 8B Q8_0 rows.
- The bounded compact chat-template-shapes pack now passes on the reopened Ubuntu lane: all 4 prompts in `qa/prompt-packs/llama3-chat-template-shapes.json` matched prompt tokens, generated token IDs, and generated text at `d13541ad8d7e87426cddd0d0a13e292f39c73f31`.
- `/api/models/load`, `/v1/completions`, `/v1/chat/completions`, and frontend smoke passed for this exact row.
- A clean-main reopened-lane API/WebUI/RSS smoke at `aee469b` validated that `/v1/completions` exposes `camelid.timings_ms` like chat completions, allowing the promotion smoke bundle to summarize response-local timings for both endpoints. The exact 8B smoke passed all API/frontend steps and generated `Hello` in the frontend smoke; this is recorded in `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`.
- The current-public-head watchdog refresh at `83c21f0cbf5a` repeats the normalized 8B API/WebUI/RSS smoke after the CI evidence-checksum gate landed; it is published at `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json` and remains exact-row smoke evidence only.
- The earlier public-main checkmark refresh at `864e07b51f36` repeated the exact 8B API/WebUI/RSS smoke; the latest public-main checkmark refresh at `15bfc41d15d5` repeats it again after the 1B 1024-context evidence landed. The latest bundle is published at `qa/evidence-bundles/8b-checkmark-current-main-20260505T084931Z-head-15bfc41d15d5/manifest.json` and remains exact-row smoke evidence only.
- A retained-block lazy-Q8 hot-path probe at `723a665` measured representative exact-row 8B costs without widening support: logical `[14336,4096]` FFN dots were about 36.7 ms each in serial microbench mode, and swapped logical `output.weight` `[128256,4096]` was about 328 ms while avoiding about 2.0 GiB f32 materialization. This narrows optimization targets; it is not production throughput, portability, or full-support evidence.
- The current 8B context support truth for this exact row includes checked bounded 512/1024/2048 packs. The current-head 1024/2048 pass for source/runtime head `8e26be0a73c0` matched the bounded prompt packs exactly and is published at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`; earlier `b49034007f2e`, `9e3c64f2cfab`, `160348118d44`, `9f8588bb4a4e`, `844e8d4709d1`, `86081876d01e`, and `e203d3cf3ea5` bundles remain historical source-head evidence.
- The remaining backend pivot is performance/memory architecture follow-up: `qa/validation-notes/2026-05-05-prefill-chunk-runtime.md` records the chunked-prefill/Q8-read-reuse runtime slice, structured prefill memory instrumentation, local gates, and an isolated Ubuntu code-validation pass. This is code evidence only until row-specific Ubuntu timing/parity artifacts exist.
- The support claim is limited to this exact Llama 3 8B Instruct Q8_0 row and tested smoke/parity envelope through the cited checked 512/1024/2048 bounded context packs. Neighboring Llama rows, other quantizations, larger/model-native contexts beyond checked packs, production throughput, portability, arbitrary templates, and broader chat-template behavior remain outside the claim until their own evidence exists.

Representative durable evidence:

- `qa/evidence-bundles/four-row-public-20260503T024327Z/llama3_8b_instruct_q8_0.bundle.json` (the committed pre-promotion guarded-WebUI carry-forward slice)
- `qa/evidence-bundles/four-row-public-20260503T024327Z/manifest.json`
- `qa/evidence-bundles/four-row-public-20260503T024327Z/SHA256SUMS`
- `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`
- `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/llama3_8b_instruct_q8_0/manifest.json`
- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T122749Z-head-9f8588bb4a4e/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T040523Z-head-e203d3cf3ea5/manifest.json` (historical source-head predecessor)
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260507T194559Z-head-ab8e465b50c3/manifest.json` (historical source-head predecessor)
- `qa/evidence-bundles/llama3-8b-context-1024-20260506T182100Z-head-e146d3b335d8/manifest.json` (historical/superseded for current-head promotion)
- `qa/evidence-bundles/llama3-8b-context-2048-20260506T182534Z-head-e146d3b335d8/manifest.json` (historical/superseded for current-head promotion)
- `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`
- `qa/evidence-bundles/llama3-8b-api-webui-rss-20260505T014408Z-head-8cef7af4d6c6/manifest.json` (historical patched-tree predecessor)
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`
- `qa/validation-notes/2026-05-05-lazy-q8-hotpath-costs.md`
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json`
- `qa/validation-notes/2026-05-05-lazy-q8-hotpath-helper-validation.md`
- `qa/validation-notes/2026-05-05-llama3-8b-context-2048-blocker.md`
- `qa/validation-notes/2026-05-05-8b-clean-main-api-webui-rss.md`
- `qa/validation-notes/2026-05-05-8b-broader-50tok.md`
- `qa/validation-notes/2026-05-04-8b-context-512-rerun.md`
- `qa/validation-notes/2026-05-05-8b-chat-template-shapes.md`
- `qa/validation-notes/2026-05-05-8b-api-webui-rss.md`
- `qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md`

Selected source artifacts recorded by those committed files:

- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json` is the public sanitized broader three-prompt 50-token parity pass for the exact 8B row; the older `target/acceptance-llama3-8b-broader-5tok-longtimeout-20260503T010536Z/summary.json` remains historical carry-forward evidence only.
- `target/ubuntu-llama3-8b-q8-current-head-20260502T000207Z/validation-summary.json` is the bounded-RSS short-slice summary carried forward beside the current-head blocker note.

## Latest promotion-relevant work

### Docs professionalism pass

The top-level documentation set was tightened for executive readability, hierarchy, and release consistency without changing support truth. `README.md`, `COMPATIBILITY.md`, `ROADMAP.md`, and `STATUS.md` remain the public sources of truth. The README now pairs the front-door support ledger with a clearer reading order, while visible llama.cpp / ggml acknowledgement and the MIT notice remain intact wherever reference tooling and parity evidence depend on them. Recon and planning docs continue to carry explicit note banners.


### Full frontend/API end-to-end smoke

Fresh end-to-end validation artifact: `target/e2e-docs-20260502T2130Z-r3/`.

- Llama 3.2 1B Instruct Q8_0 loaded as the exact supported compatibility row, reported `generation_ready=true`, unlocked WebUI chat as `contract_supported=true`, and returned `"Hello"` from `/v1/chat/completions` in 8.49s.
- Llama 3.2 3B Instruct Q8_0 loaded as the exact supported compatibility row, reported `generation_ready=true`, unlocked WebUI chat as `contract_supported=true`, and returned `"Hello"` from `/v1/chat/completions` in 24.24s.
- Llama 3 8B Instruct Q8_0 loaded and generated through the same frontend/API smoke path; after the later parity promotion it is now an exact supported compatibility row. The smoke returned `"Hello"` in 55.51s.

### Llama 3.2 3B exact-row smoke promotion

Recent backend and frontend work aligned the 3B execution seam with the user-visible support contract:

- streaming metadata parsing moved `/api/models/load` to low backend RSS for the exact 3B artifact
- file-backed Q8 linear handling reduced the older eager dense-load spike
- compact prompt-token, 1-token, 5-token, and bounded 50-token parity passed for the exact tracked 3B row
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke now pass under the exact supported compatibility row

This is a support promotion only for the exact 3B Instruct Q8_0 short-chat smoke row.

### Llama 3 8B exact-row smoke promotion

Recent backend work converted the 8B runtime artifacts into an exact-row support promotion:

- the exact tracked `Meta-Llama-3-8B-Instruct-Q8_0.gguf` loaded successfully on Ubuntu
- repeat bounded backend-only `/v1/completions` first-token probes returned `,` for prompt `hello`
- current-head raw `hello` prompt-token parity matched `[128000, 15339]` for the exact same model SHA
- a short deterministic 5-token backend slice returned `, I'm a new`
- the Ubuntu three-prompt 50-token parity run passed for `hello`, alpacas, and JSON: `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`
- the reopened-lane first 512-context pack passed with prompt-token, generated-token, and generated-text parity; public summary/checksums live at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`
- the current-head 1024/2048 bounded-context pack passed at source/runtime head `8e26be0a73c0` with prompt-token, generated-token, and generated-text parity; public summary/checksums live at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/`; older 1024/2048 bundles remain historical for their source heads
- the reopened-lane compact chat-template-shapes pack passed all 4 checked shapes with prompt-token, generated-token, and generated-text parity; public summary/checksums live at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/`
- `/v1/completions`, `/v1/chat/completions`, and frontend smoke are preserved in the sanitized carry-forward bundle at `qa/evidence-bundles/four-row-public-20260503T024327Z/llama3_8b_instruct_q8_0.bundle.json`; the newer reopened-lane API + frontend smoke summary at `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json` refreshes exact-row WebUI/API readiness for the four tracked rows, but broader/full-support evidence is still outstanding
- clean-main API timing diagnostics for `/v1/completions` passed the exact 8B API/WebUI smoke and response-local timing summary at `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`; sampled backend RSS for that smoke moved `6,316 -> 283,352 KiB`, which is useful smoke-window evidence but not peak memory proof
- retained-block lazy-Q8 hot-path measurement at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json` shows FFN logical `[14336,4096]` serial all-row dots around 36.7 ms and swapped `output.weight` logical `[128256,4096]` around 328 ms; this is optimization-grounding evidence only, not a wider support/performance claim
- the current-head memory gate stayed bounded: first-token sampled RSS roughly `6,220 -> 378,520 KiB`; 5-token sampled RSS roughly `6,076 -> 396,912 KiB`; no swap, OOM, timeout, or runaway retained-RSS signature appeared

This is a support promotion only for the exact Llama 3 8B Instruct Q8_0 row and tested smoke/parity envelope, including the cited checked bounded 512/1024/2048 context packs. Broad Llama-family, model-native/larger-context beyond checked packs, production-throughput, portability, arbitrary-template, and neighboring-row support remain unpromoted.

## Latest downloaded Llama-family matrix

Latest Ubuntu downloaded-model matrix: `target/downloaded-llama-matrix-20260502T231000Z/summary.json`.

Downloaded GGUF rows covered by this sweep:

- `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`
- `Llama-3.2-1B-Instruct-Q8_0.gguf`
- `Llama-3.2-3B-Instruct-Q8_0.gguf`
- `Meta-Llama-3-8B-Instruct-Q8_0.gguf`

Results:

- **TinyLlama 1.1B Chat Q8_0:** `hello` and the alpacas prompt matched llama.cpp; the JSON-shaped prompt diverged despite matching prompt tokens (`endpoint` vs `function` wording in the generated text).
- **Llama 3.2 1B Instruct Q8_0:** the three-prompt Llama 3 pack passed completely; prompt tokens, generated token IDs, and generated text all matched llama.cpp.
- **Llama 3.2 3B Instruct Q8_0:** the earlier downloaded matrix captured the now-fixed JSON-shaped prompt divergence; the post-Q8-dot clean rerun at `target/camelid-llama32-3b-broad-50-after-q8dot-clean-20260502T233427Z/pack/summary.json` supersedes that 3B result and passes all three prompts for prompt tokens, generated token IDs, and generated text.
- **Llama 3 8B Instruct Q8_0:** the reopened-lane rerun at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json` passed `hello`, alpacas, and JSON for prompt-token, generated-token, and generated-text parity at 50 tokens, clearing the earlier client-timeout blocker for the exact 8B row.

The downloaded-model matrix still disproves a broad inherited “perfect Llama-family parity” claim. Camelid should claim only the exact supported rows and envelopes backed by passing artifacts: TinyLlama Q8_0, Llama 3.2 1B Q8_0, Llama 3.2 3B Q8_0, and Llama 3 8B Q8_0.

Public evidence packaging note: sanitized carry-forward bundle manifests/checksums for the four-row smoke slices now live under `qa/evidence-bundles/four-row-public-20260503T024327Z/` and `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/`. They intentionally preserve the blocked public-head rerun state instead of overstating it.

Current-head durable execution note: the exact-row normalized rerun scaffold is now checked in at `qa/evidence-bundles/four-row-current-head-20260503T061958Z-head-34b954498a03/`. Its per-row manifests/commands give docs, API, and frontend a stable current-head citation target while broader context coverage and stronger perf/portability evidence are still outstanding. The TinyLlama final-normalization bundle is separately captured at `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/`, the exact Llama 3.2 1B/3B bounded compact chat-template-shapes bundle is separately captured at `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/`, the exact Llama 3.2 1B/3B bounded unique-chat perf/RSS envelope is separately captured at `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/`, the exact 8B broader three-prompt 50-token rerun is separately captured at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/`, the first four-row 512-context pack is separately captured at `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/`, the earlier exact 8B 512-context rerun is separately captured at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/`, the current-head exact 8B 1024/2048 rerun at source/runtime head `8e26be0a73c0` is captured at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`, the previous exact 8B 1024/2048 rerun at source/runtime head `b49034007f2e` is captured as historical evidence at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T024342Z-head-b49034007f2e/manifest.json`, the previous exact 8B 1024/2048 rerun at source/runtime head `9e3c64f2cfab` is captured as historical evidence at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T002636Z-head-9e3c64f2cfab/manifest.json`, the previous exact 8B 1024/2048 rerun at source/runtime head `160348118d44` is captured as historical evidence at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T210500Z-head-160348118d44/manifest.json`, the historical exact 8B source-head 1024/2048 rerun at `9f8588bb4a4e` is captured at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T122749Z-head-9f8588bb4a4e/manifest.json`, the older exact 8B source-head 1024/2048 rerun is separately captured as historical evidence at `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260508T040523Z-head-e203d3cf3ea5/manifest.json`, the bounded exact 8B compact chat-template-shapes rerun is separately captured at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/`, the historical patched-tree exact 8B API/WebUI/RSS timing smoke is separately captured at `qa/evidence-bundles/llama3-8b-api-webui-rss-20260505T014408Z-head-8cef7af4d6c6/`, the clean-main exact 8B API/WebUI/RSS timing rerun is captured at `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`, the earlier public-main exact 8B checkmark refresh is captured at `qa/evidence-bundles/8b-checkmark-current-head-20260505T052647Z-head-864e07b51f36/manifest.json`, the latest public-main exact 8B checkmark refresh is captured at `qa/evidence-bundles/8b-checkmark-current-main-20260505T084931Z-head-15bfc41d15d5/manifest.json`, the measurement-only retained-block lazy-Q8 hot-path probe is captured at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`, and the clean-main helper-validation repeat is captured at `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json`. 8B 1024/2048 is promoted only as exact bounded-pack current-head support via the fresh PASS bundle; historical PASS bundles remain audit evidence only for their source heads. Broader/full support still requires separate evidence.

## Next blocking work

In order of importance:

1. Preserve the TinyLlama Q8_0 supported gate and the exact Llama 3.2 1B/3B short-chat smoke gates.
2. Preserve and publish the Llama 3.2 1B broader prompt-pack win as exact-row evidence, without lending it to neighboring rows.
3. Preserve the Llama 3.2 3B broader prompt-pack win in docs, API, and regression evidence without lending it to neighboring rows.
4. Preserve the Llama 3 8B exact-row smoke plus checked-512/1024/2048 bounded-pack promotion in docs, API, frontend readiness, and regression evidence; do not lend any result to model-native/larger contexts or neighboring rows.
5. Harden the supported exact rows toward full support without changing labels until the missing normalized evidence exists: model-native/larger context beyond checked packs, broader arbitrary-template coverage beyond the supported 1B metadata-Jinja row template, production throughput, portability, and durable repeated current-head bundles.
6. Preserve and protect the Mistral-7B-Instruct-v0.3.Q8_0.gguf active exact-row validation lane across all surfaces without promoting it to supported status.
7. Preserve `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` as active validation / partial backend runtime only; next promotable work is fixing later-generation divergence and the continuation hang, then rerunning row-specific API/WebUI/RSS/frontend evidence before any support/readiness claim.
8. Keep docs, `/api/capabilities`, frontend readiness copy, and CI gates aligned with the exact-row support contract.
9. Plan the README frontend screenshot for the UI-demo-ready moment; do not use screenshot polish to block or distract from CI/support hardening.

Current operator update: public evidence anchors for the exact-row Llama lanes now include reopened-lane API-only and API + frontend smoke summaries at `qa/evidence-bundles/four-row-api-only-20260504T230722Z-head-13a465608fbf/manifest.json` and `qa/evidence-bundles/four-row-api-webui-20260505T003100Z-head-b403884/manifest.json`; the TinyLlama final-normalization bundle at `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/manifest.json`; the exact Llama 3.2 1B/3B compact chat-template-shapes pack at `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`; the exact Llama 3.2 1B/3B unique-chat perf/RSS pack at `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`; the exact 8B broader three-prompt 50-token pack at `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`; the first four-row bounded 512-context pack at `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`; the earlier exact 8B 512-context pack at `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`; and the bounded exact 8B compact chat-template-shapes pack at `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`. This does not widen support by itself: broader/full support still requires normalized parity, memory/perf, broader context/template coverage, and durable-bundle evidence. See `qa/validation-notes/2026-05-04-validation-lane-reopened.md`, `qa/validation-notes/2026-05-04-8b-context-512-rerun.md`, `qa/validation-notes/2026-05-05-four-row-context-512.md`, `qa/validation-notes/2026-05-05-llama32-1b-3b-chat-template-shapes.md`, `qa/validation-notes/2026-05-05-llama32-1b-3b-unique-chat-perf-rss.md`, `qa/validation-notes/2026-05-05-8b-broader-50tok.md`, and `qa/validation-notes/2026-05-05-8b-chat-template-shapes.md`.

### Mistral exact-row validation note

Mistral 7B Instruct v0.3 Q8_0 remains an active exact-row validation lane, not a supported row. The GGUF has exact-row load, tokenizer/template, 1-token generation, broader 50-token parity, checked 512/1024/2048/4096/8192 context packs, and a fail-closed current-head API/WebUI/RSS bundle, but the contract still records `active_validation_unsupported` with WebUI chat blocked. Next promotable work is explicit support-contract promotion for this exact row after a fresh evidence slice shows `supported_exact_row_smoke`, `contract_supported=true`, and `webui_chat=enabled`; broader Mistral-family quants/sizes, larger native contexts, and production throughput remain unsupported follow-up work.

### Next-family exact-row notes

`Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` is active validation / partial backend runtime only. Bounded one-token backend MoE evidence exists, but Gate 9A later-generation evidence diverges and the continuation bundle records a backend HTTP hang. No Mixtral API/WebUI/frontend readiness, neighboring-row, long-context, arbitrary-template, production-throughput, portability, or broader/full support claim is active.

Qwen and Gemma remain planned exact-row candidates.

## Validation note

This file is intentionally a snapshot, not a diary. When a change materially affects support or its blockers:

- add the current evidence summary here
- keep the detailed run log and older slices in `STATUS_ARCHIVE_2026-04.md` or later archives
- update `COMPATIBILITY.md`, `ROADMAP.md`, and user-visible readiness copy in the same change window when support language changes
