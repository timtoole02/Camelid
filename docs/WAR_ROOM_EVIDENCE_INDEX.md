# War Room Evidence Index and Claim Policy

Last updated: 2026-06-01

This file is the war-room guardrail for public Camelid claims. It does not create support, benchmark, API, or WebUI readiness claims by itself. It tells reviewers which evidence anchors control each public surface and how to keep docs, `/api/capabilities`, and frontend readiness aligned.

## Source-of-truth order

When surfaces disagree, use this order:

1. [`COMPATIBILITY.md`](../COMPATIBILITY.md) controls release support language, API feature status, and WebUI readiness wording.
2. [`STATUS.md`](../STATUS.md) records the current evidence snapshot, recent movements, and blockers.
3. [`BENCHMARKS.md`](../BENCHMARKS.md) controls public performance wording and same-host comparison boundaries.
4. [`frontend/README.md`](../frontend/README.md) controls WebUI readiness policy and smoke-test interpretation.
5. `/api/capabilities` must mirror the compatibility ledger; it must not promote a row, context bucket, API feature, or frontend state before the docs and evidence do.

Working notes, agent briefs, local validation notes, draft plans, and unreviewed logs are not public claim sources unless a public source above cites a scrubbed bundle and states the exact supported boundary.

If the source-of-truth files are dirty, treat the on-disk text as the review target and avoid widening claims until the diff is understood. If a dirty diff changes `COMPATIBILITY.md`, `/api/capabilities`, or frontend readiness behavior, update the other public surfaces in the same change or leave an explicit blocker instead of publishing a partial promotion.

## Evidence index

Use these anchors before changing public copy:

| Claim lane | Primary anchor | Required public boundary |
| --- | --- | --- |
| Exact-row support | [`COMPATIBILITY.md`](../COMPATIBILITY.md), [`STATUS.md`](../STATUS.md), row manifests under `qa/evidence-bundles/` | Exact model row, quantization, context buckets, tokenizer/template scope, API/WebUI status, and unsupported neighboring rows must be stated together. |
| API capabilities | [`COMPATIBILITY.md`](../COMPATIBILITY.md) API table plus `src/api/mod.rs` capability rows | New capability rows must use `supported`, `partial`, `planned`, or `unsupported` language that matches committed evidence and typed runtime behavior. |
| API model discovery metadata | [`COMPATIBILITY.md`](../COMPATIBILITY.md) API table plus `/v1/models`, `/v1/models/:model`, and read-only `/models` tests | Public model-discovery fields may expose only model-shape/count/size/quant metadata already available from the loaded GGUF/runtime config or currently loaded Camelid model IDs. They must not expose local paths and must not imply tokenizer parity, generation support, WebUI readiness, router-mode cache listing, native load/unload, broader llama-server parity, or neighboring-row support. |
| llama-server control-plane compatibility | [`COMPATIBILITY.md`](../COMPATIBILITY.md) API table plus focused API vertical-slice tests | `/props`, `/slots`, and similar discovery routes may be described only as read-only partial compatibility when tests prove privacy-safe public fields and typed unsupported behavior for lifecycle or write actions. They must not imply native `/completion`, slot lifecycle parity, prompt-cache semantics, cancellation metadata, embeddings, reranking, multimodal support, production throughput, or full WebUI parity. |
| WebUI readiness | [`frontend/README.md`](../frontend/README.md) plus `/api/capabilities` and `/v1/health` behavior | Chat readiness requires exact-row capability support and `loaded_now=true` plus `generation_ready=true`; filenames, catalog metadata, saved paths, or prior use are not evidence. |
| Benchmark/performance | [`BENCHMARKS.md`](../BENCHMARKS.md), `docs/performance/`, and scrubbed benchmark manifests | Performance claims require same-host, same-row, same-prompt, same-token-budget, same-thread evidence. Direction probes and local-only gates stay labeled as such. |
| llama.cpp comparison | [`BENCHMARKS.md`](../BENCHMARKS.md), [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), row parity manifests | Camelid may cite llama.cpp for parity/reference validation and credit ggml/llama.cpp work. Do not imply broad competitive superiority without a normalized same-host throughput bundle. |
| Next-family rows | [`COMPATIBILITY.md`](../COMPATIBILITY.md), [`STATUS.md`](../STATUS.md), blocker reconciliation manifests | Mistral, Mixtral, Qwen, and Gemma wording must remain exact-row, evidence-only, planned, or blocked until promotion artifacts close every named blocker. |
| Privacy/scrub state | `scripts/check-public-scrub.sh`, `scripts/check-public-evidence-claims.mjs`, evidence-bundle privacy audits | Public docs and manifests must not expose private hostnames, private IPs, key paths, home paths, model-library paths, raw operator commands, or raw failure logs. |

## Claim policy

- Do not promote a row from planned, evidence-only, active validation, or blocked status unless a scrubbed row-specific evidence bundle exists and `COMPATIBILITY.md`, `STATUS.md`, `/api/capabilities`, and WebUI readiness language are updated together.
- Do not infer support across neighboring model sizes, base/instruct variants, quantization formats, tokenizer families, context buckets, API surfaces, or frontend states.
- Do not turn local-only tests, dirty-tree experiments, direction probes, implementation scaffolding, or timing anecdotes into release, benchmark, or readiness claims.
- Do not publish local/private paths, hostnames, key paths, private IPs, raw operator commands, raw stderr, or model-library locations. Public evidence should use repo-relative paths, hashes, row IDs, timestamps, command names, and summarized pass/fail outcomes.
- Keep llama.cpp / ggml credit visible where parity or comparator evidence is cited. Phrase comparisons as bounded parity or measured same-host results only.
- If a capability is present for discovery or compatibility, label it precisely. For example, a read-only partial control-plane route must not imply native generation aliases, slot lifecycle parity, embeddings, reranking, multimodal support, production throughput, or full WebUI parity.
- For read-only `/slots` compatibility, public copy may say only that `GET /slots` exposes a privacy-safe slot snapshot and `fail_on_no_slot=1` behavior when the focused API test covers those fields. Keep `POST /slots`, slot save/restore/erase actions, prompt-cache metadata, cancellation metadata, continuous batching metrics, and full llama-server slot lifecycle parity explicitly unsupported until separate semantics and tests exist.
- Public model discovery metadata is descriptive only. Treat `/v1/models` `meta` values such as vocabulary size, training context, embedding size, parameter count, file size, or GGUF file type as inspection fields, not as support, benchmark, API-completeness, or WebUI-readiness evidence.
- Read-only llama-server `/models` compatibility may say only that `GET /models` lists currently loaded Camelid models with local paths redacted when focused tests cover that shape. Keep router-mode cache listing, reload/autoload, native `POST /models/load`, native `POST /models/unload`, multimodal architecture metadata, and full llama-server model-management parity explicitly unsupported until separate semantics and tests exist.
- API/WebUI readiness copy must state both sides of the gate: exact compatibility-row support from `/api/capabilities` and runtime readiness from `/v1/health` (`loaded_now=true` plus `generation_ready=true`). One without the other is not chat readiness evidence.
- Benchmark copy must keep negative, blocked, or slower same-host results when they are the current retained evidence. Do not replace an unfavorable retained result with a direction probe, local-only optimization, or future benchmark plan.

## Scheduled audit rule

War-room docs audits may update this policy, the evidence index, or blocker wording only when the edited surface is clean or the dirty diff has been read and does not itself promote support, benchmark, API capability, or WebUI readiness. A dirty diff in `COMPATIBILITY.md`, `/api/capabilities`, frontend readiness code, or public benchmark copy is a release-claim blocker unless the matching source-of-truth surfaces are updated in the same change.

Scheduled audits should record sanitized evidence only: repo-relative paths, row IDs, manifest or summary names, checksums when public, and pass/fail outcomes. They should not add operator-local commands, private network details, hostnames, user home paths, key paths, raw model-library locations, raw stderr, or local machine notes to public docs.

## Minimum safe update checklist

Before landing a support-sensitive docs/API/frontend change:

1. Read the current on-disk `README.md`, `STATUS.md`, `BENCHMARKS.md`, `COMPATIBILITY.md`, `/api/capabilities` implementation, and WebUI readiness docs.
2. Identify the exact row, API feature, benchmark lane, or WebUI state being changed.
3. Cite only scrubbed committed evidence or explicitly label the work as local-only, planned, partial, evidence-only, or blocked.
4. Run the public evidence and scrub guards when the change touches public claims.
5. Leave a blocker note instead of copyediting around missing evidence.
