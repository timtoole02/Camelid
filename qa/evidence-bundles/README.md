# Evidence bundles

This directory is for durable, reviewable evidence manifests and checksums.

## Reviewer shortcut

The support story is intentionally exact-row. `COMPATIBILITY.md` is the release contract, `STATUS.md` is the narrative evidence snapshot, and this directory is the public manifest/checksum map that keeps the claims auditable without exposing private raw artifact trees.

| Exact row | Public proof readers should start with | Current blocker, if any |
| --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | `tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/` plus the four-row API/WebUI and 512-context bundles. | None for the current supported gate; rerun on future support-contract changes. |
| Llama 3.2 1B Instruct Q8_0 | `full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/`, `llama32-1b-context-1024-20260505T081001Z-head-156ded6fc76b/`, `llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/`, `llama32-1b-context-4096-current-head-20260513T163426Z-head-470388f/`, `llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/`, and the shared 1B/3B template/perf bundles. | Broader/full support still needs model-native/larger context beyond the checked packs, arbitrary/Jinja template, production throughput, portability, and durable full-support normalization. |
| Llama 3.2 3B Instruct Q8_0 | `full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/`, `llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/`, `llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/`, `llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/`, the shared 1B/3B template/perf bundles, `llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/`, the scrubbed local streaming-responsiveness smoke at `llama3-3b-q8-mac-streaming-20260512T0713Z/`, and the v0.1 llama.cpp CPU comparator at `v0.1/20260531T184150Z-real-local/`. | Broader/full support still needs model-native/larger context, arbitrary/Jinja template, production throughput beyond the first-token direction probe, portability, durable full-support normalization, and a fix for the long first-content latency observed on the scrubbed local streaming smoke. |
| Llama 3 8B Instruct Q8_0 | `full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/`, `8b-checkmark-current-main-20260505T084931Z-head-15bfc41d15d5/`, `llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/`, `four-row-context-512-20260505T051510Z-head-b403884/`, `llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/`, `llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/`, and the lazy-Q8 hot-path bundles. | 1024/2048 context are now bounded exact-row PASS bundles for source/runtime head `8e26be0a73c0`; the checked packs remain bounded exact-row claims only. |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | `mistral-7b-v0.3-q8-broader-50tok-ubuntu-20260509T000633Z-head-d330e97ae992/`, `mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/`, and `mistral-7b-v0.3-q8-api-webui-rss-current-head-20260513T1935Z-head-9a296ea/`. | Active validation only; fail-closed API/WebUI/RSS evidence exists, but explicit contract promotion and synchronized support surfaces are still missing before any support claim. |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | `mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/` plus the original `mixtral-8x7b-v0.1-q8-support-probe-20260509/`. | Unsupported beyond bounded one-token backend MoE runtime evidence; 5-token/API/WebUI/RSS promotion-candidate artifacts are superseded by 50-token Gate 9A divergence and a long-continuation hang. |

Current public evidence map:
- `four-row-public-20260503T024327Z/` preserves the sanitized carry-forward smoke boundary.
- `four-row-perf-portability-public-20260503T025639Z/` preserves the compact perf/portability envelope.
- `four-row-current-head-20260503T061958Z-head-34b954498a03/` preserves the normalized current-head rerun scaffold and blocker notes.
- `four-row-api-only-20260504T230722Z-head-13a465608fbf/` is the reopened-lane API-only freshness slice with manifest and checksums.
- `four-row-api-webui-20260505T003100Z-head-b403884/` is the reopened-lane API + frontend smoke freshness slice for all four exact rows, with manifest and checksums.
- `full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/` is the current-head normalized TinyLlama/1B/3B API/WebUI smoke bundle from the reopened Ubuntu lane; it preserves manifest/checksum-verifiable evidence without broadening beyond exact-row smoke support.
- `llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/` is the canonical Ubuntu current-`main` refresh for the exact Llama 3.2 3B Q8_0 API/WebUI support gate: load, `/v1/completions`, `/v1/chat/completions`, timing summary, `/api/capabilities`, and frontend smoke passed with the expected supported exact-row contract.
- `tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/` closes the exact TinyLlama Q8_0 current-head broader five-prompt parity, marker chat-template-shapes parity, bounded 512-context parity, and perf/RSS durable-normalization slice; scope is exact-row only.
- `four-row-context-512-20260505T051510Z-head-b403884/` closes only the first bounded 512-context pack for the four exact supported Q8_0 rows; it does not promote larger context buckets or broad family support.
- `llama32-1b-context-1024-20260505T081001Z-head-156ded6fc76b/` closes only the second bounded 1024-context pack for the exact Llama 3.2 1B Instruct Q8_0 row.
- `llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/` closes only the third bounded 2048-context pack for the exact Llama 3.2 1B Instruct Q8_0 row after the RoPE frequency-factor fix.
- `llama32-1b-context-4096-current-head-20260513T163426Z-head-470388f/` closes only the fourth bounded 4096-context compact-template pack for the exact Llama 3.2 1B Instruct Q8_0 row at source/runtime head `470388f8165b`.
- `llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/` closes only the fifth bounded 8192-context compact-template pack for the exact Llama 3.2 1B Instruct Q8_0 row at source/runtime head `aaf9207d1669`.
- `llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/` closes only the second bounded 1024-context pack for the exact Llama 3.2 3B Instruct Q8_0 row.
- `llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/` closes only the third bounded 2048-context pack for the exact Llama 3.2 3B Instruct Q8_0 row; it does not promote neighboring rows, model-native context, or broad/full support.
- `llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/` closes only the bounded compact chat-template-shapes pack for the exact Llama 3.2 1B/3B Instruct Q8_0 rows.
- `llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/` closes only the bounded unique-chat memory/perf envelope for the exact Llama 3.2 1B/3B Instruct Q8_0 rows.
- `llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/` closes only the exact Llama 3.2 3B opt-in parallel Q8 first-token runtime direction sub-box; it is not production-throughput or portability support.
- `llama3-3b-q8-mac-streaming-20260512T0713Z/` records exact Llama 3.2 3B Q8_0 scrubbed local streaming-responsiveness evidence: pre-patch first content was about `253.5 s`, while the patched path emitted a role-only SSE chunk in about `91.8 ms` and kept `/health` responsive during generation. This is UI/API responsiveness evidence only; it does not close first-content latency or widen support.
- `v0.1/20260531T184150Z-real-local/` records the v0.1 clean-head same-host llama.cpp CPU comparator for the exact Llama 3.2 3B Q8_0 row. Camelid lost on bounded TTFT and total elapsed; the bundle is not Metal, Ollama, MLX, production-throughput, or broad-family evidence.
- `llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/` closes only the bounded 8B broader three-prompt 50-token pack.
- `llama3-8b-context-512-20260504T234625Z-head-58acf592345c/` closes only the first bounded 8B 512-context pack.
- `llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/` closes only the bounded 1024/2048-context packs for the exact Llama 3 8B Instruct Q8_0 row at source/runtime head `8e26be0a73c0`; it does not promote model-native/larger context, production throughput, portability, arbitrary templates, or broad 8B/Llama support. Older 1024/2048 bundles, including `llama3-8b-context-1024-20260506T144810Z-head-ae672d935a9d/` and `llama3-8b-context-2048-20260506T144037Z-head-ae672d935a9d/`, remain historical source-head evidence only.
- `llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/` closes only the bounded 8B compact chat-template-shapes pack.
- `llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/` is the clean-main exact 8B API/WebUI/RSS timing smoke for completion diagnostics; it does not widen support beyond the exact-row smoke envelope.
- `8b-checkmark-current-head-20260505T052647Z-head-864e07b51f36/` is the earlier public-main exact 8B API/WebUI/RSS checkmark refresh. It preserves `supported_exact_row_smoke` only and does not widen broader/full support.
- `8b-checkmark-current-main-20260505T084931Z-head-15bfc41d15d5/` is the latest public-main exact 8B API/WebUI/RSS checkmark refresh after the 1B 1024-context evidence landed. It preserves `supported_exact_row_smoke` only and does not widen broader/full support.
- `llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/` is the exact 8B retained-block lazy-Q8 hot-path cost probe; it is measurement evidence only, not a broader support/performance-portability promotion.
- `llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/` validates the reusable helper on clean public `main` and repeats the exact 8B retained-block Q8 measurements; it is still measurement evidence only.
- `mixtral-8x7b-v0.1-q8-support-probe-20260509/` is the original bounded Mixtral one-token backend MoE runtime anchor; it does not promote broad Mixtral or frontend/API support.
- `mixtral-8x7b-v0.1-q8-backend-parity-refresh-20260511/`, `mixtral-8x7b-v0.1-q8-api-smoke-20260511/`, `mixtral-8x7b-v0.1-q8-webui-readiness-20260511/`, and `mixtral-8x7b-v0.1-q8-rss-timing-runtime-20260511/` are exact-row promotion-candidate artifacts only; the WebUI bundle explicitly records fail-closed unsupported-contract behavior.
- `mixtral-8x7b-v0.1-q8-gate9a-50tok-20260511/`, `mixtral-8x7b-v0.1-q8-longgen-continuation-20260511/`, and `mixtral-8x7b-v0.1-q8-backend-hang-guard-20260511/` record the later Mixtral blocker evidence.
- `mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/` is the current Mixtral validation-boundary anchor: active validation partial runtime, unsupported beyond bounded one-token evidence, no broad Mixtral support.

Reproducibility helpers:
- `bash scripts/check-evidence-bundle-checksums.sh` verifies every committed `SHA256SUMS` under this directory, including older bundles that use bundle-local paths and newer bundles that use repo-relative paths.
- `node scripts/bench-q8-hotpath-bundle.mjs --model <model.gguf>` regenerates a sanitized retained-block Q8 hot-path bundle with per-tensor JSON, `manifest.json`, and `SHA256SUMS`. Use it for measurement staging only; pair results with production API/WebUI timing/RSS before making portability or throughput claims.

Rules:
- Commit only sanitized durable bundle content here.
- Keep raw/private staging copies out of git; they may contain private hostnames, home paths, or other operator-only details.
- Public bundles may point at `target/...` artifact roots, but they must not pretend those private raw trees are fetchable from GitHub.
- In committed manifests/checksums, prefer public-safe `qa/evidence-bundles/*-public-...` bundle paths over ignored raw bundle roots.
- Before citing or refreshing a durable bundle, run `node scripts/audit-evidence-bundle-privacy.mjs --root qa/evidence-bundles --out target/evidence-bundle-privacy-audit.json` and fix any findings.
