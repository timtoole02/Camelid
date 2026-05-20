# 2026-05-13 — Llama 3.2 1B metadata Jinja renderer slice

Scope: exact Llama 3.2 1B Instruct Q8_0 metadata-template renderer path for the recognized Llama 3 instruct chat-template shape. This is row-template renderer/API-contract evidence only; it does not promote broad arbitrary-template execution, neighboring rows, production throughput, portability, or model-native/larger context beyond checked packs.

Changes validated:

- Replaced the previous hand-written Llama 3 metadata-template subset with a real Jinja-compatible renderer backed by `minijinja` when `CAMELID_METADATA_CHAT_TEMPLATE=metadata` is enabled.
- Added a model-aware chat prompt path so the active exact Llama 3.2 1B Instruct Q8_0 row can use the metadata-Jinja renderer without the env opt-in when the tokenizer contains the recognized Llama 3 instruct template shape.
- Preserved fail-closed/default behavior: the shared `render_chat_prompt_for_tokenization(...)` helper and non-exact model IDs still use the compact renderer unless the metadata env opt-in is set.
- The renderer passes `messages`, `bos_token`, `eos_token`, `eot_token`, `eom_token`, `unk_token`, and `add_generation_prompt=true` into the metadata template.
- BOS de-duplication is handled by checking whether the rendered template already begins with the tokenizer BOS text before tokenizer-level special-token insertion.
- Unsupported/error branches are reported through a `raise_exception(...)` helper; the production chat-preparation path now returns an `unsupported_chat_template` error when the exact supported 1B metadata-Jinja renderer fails instead of silently falling back to the compact renderer.

Unit/API contract coverage:

- Exact 1B row metadata-Jinja rendering without env opt-in: system+user, user-only, multi-turn system/user/assistant/user, and assistant-final/continuation edge cases.
- Non-exact Llama 3.2 3B and non-Q8 1B model IDs preserve compact rendering without env opt-in.
- Existing metadata-Jinja opt-in tests cover user-only, system+user, multi-turn, assistant-final, templates that omit `bos_token`, simple loop-based Jinja execution, explicit unsupported `raise_exception(...)` branch handling, and exact-row required-renderer failure without silent fallback.
- `/api/capabilities` now reports `chat_template_renderer: "metadata_jinja_supported_for_exact_row"` for `llama32_1b_instruct_q8_0` only, while the blocker text keeps broad arbitrary-template/full-support gaps open.

Validation artifacts:

- Local artifact: `target/cron-95495a91-20260513T2149Z-jinja-required-error-head-1ae5e9a17aaf/`
- Local passes recorded there: `cargo fmt --check`; `cargo test metadata_jinja_renderer --lib -- --nocapture`; `cargo test`; `cargo test output_projection_q8_0_descriptor_shape_uses_storage_token_rows --lib -- --nocapture`.
- Earlier clean Ubuntu validation copied clean source head `1ae5e9a17aaf` plus the Jinja/API-contract patch to a scrubbed remote checkout and passed focused renderer/API-contract gates plus `cargo test -q --lib` under rustc/cargo 1.87.0; this follow-up required-renderer-error slice was validated locally because it is a prompt-construction/API error-path change, not a model-runtime parity change.
- Canonical Ubuntu host readiness was checked for this slice; `/` is currently 100% used with about 905 MiB free and the default `/usr/bin/cargo`/`rustc` is 1.75.0, so no fresh remote build/test was attempted. Readiness details are recorded in the local artifact as `ubuntu-host-readiness.log`.

Known blockers / not promoted:

- No new 4096/8192 model-runtime parity bundle was captured in this slice; the row continues to rely on the existing checked 4096 and 8192 compact-template context bundles cited in `STATUS.md`/`COMPATIBILITY.md`.
- This slice does not promote arbitrary-template, broad Llama-family, neighboring-row, production-throughput, portability, or full-support claims.

## 2026-05-13 22:53 UTC TPM follow-up on current `main`

Current-head audit: local `main` was `d15581beda28b4e37fc2871a4d2f4287bd75a436`; the working tree already contained unrelated edits in `src/inference.rs`, `src/main.rs`, and `src/metal.rs`, so this slice did not modify them. Cron/process health showed no active Camelid backend/cron runner locally.

Additional validation artifacts are under `target/cron-0719640b-20260513T2253Z-tpm-jinja-1b-current-head/`:

- `cargo-test-metadata-jinja.log`: 12/12 focused metadata-Jinja renderer tests passed on current head, including exact-row system+user, user-only, multi-turn, assistant-final/continuation, non-exact fallback, opt-in generic Jinja loop, no-BOS-template, and `raise_exception`/no-silent-fallback cases.
- `cargo-test-exact-llama32-1b.log`: 5/5 exact Llama 3.2 1B row renderer tests passed.
- `cargo-test-capabilities-llama32-boundaries.log` and `cargo-test-api-vertical-capabilities-contract.log`: API capability gates passed for the exact 1B metadata-Jinja renderer surface plus 512/1024/2048/4096/8192 bounded pack fields.
- `frontend-smoke-model-state-after-fixture-update.log` and `frontend-build-after-fixture-update.log`: frontend model-state smoke and Vite production build passed after updating the smoke fixture to enforce the 1B 4096/8192 boundary and latest `CMLD-819` surface.
- Canonical Ubuntu validation was attempted with a clean current-head source archive in a scrubbed remote checkout, but a fresh build failed with `No space left on device`; the attempted tree was removed. As a fallback, the existing near-head Ubuntu Jinja lane passed the focused metadata-Jinja/exact-row tests from its compiled target cache (`remote-existing-cargo-test-jinja-exact-1ae5.log`). Treat local current-head tests as the authoritative current-head evidence for this follow-up; the Ubuntu fallback is portability signal only, not a fresh current-head runtime promotion.

No new model-runtime 4096/8192 bundle was produced in this follow-up. The 1B row continues to cite the existing bounded compact-template runtime bundles in `STATUS.md`: 4096 at `qa/evidence-bundles/llama32-1b-context-4096-current-head-20260513T163426Z-head-470388f/manifest.json` and 8192 at `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json`.

## 2026-05-13 23:02 UTC backend resume on current `main`

Current-head audit: local `main` started at `d15581beda28b4e37fc2871a4d2f4287bd75a436` with pending runtime/frontend-gate edits in `src/inference.rs`, `src/main.rs`, `src/metal.rs`, and `frontend/scripts/model-state-smoke.mjs`. This resume preserved the committed exact-row metadata-Jinja renderer and validated it together with the pending 1B support-gate surface updates.

Validation artifacts are under `target/cron-58d09b5e-20260513T2302Z-backend-resume-head-d15581beda28/`:

- `repo-audit.txt`: recorded branch/head and the pre-validation dirty working tree.
- `cargo-fmt-check.log`: formatting passed.
- `cargo-test-metadata-jinja-renderer.log`: 12/12 focused metadata-Jinja tests passed, covering system+user, user-only, multi-turn, assistant-final/continuation, non-exact fallback, opt-in loop templates, no-BOS templates, and explicit unsupported `raise_exception(...)` behavior.
- `cargo-test-exact-llama32-1b.log`: 5/5 exact Llama 3.2 1B required-renderer tests passed.
- `cargo-test-api-vertical-capabilities-contract.log`: API capabilities contract passed for the current exact-row bounded-pack/support surface.
- `cargo-test-metal-q8.log`: macOS Metal Q8_0 focused kernels passed, including the new multi-row encoded Q8_0 kernel parity check.
- `cargo-clippy-all-targets-all-features.log`: clippy passed with `-D warnings`.
- `cargo-test-all-targets-all-features.log`: full Rust test suite passed.
- `frontend-smoke-model-state.log` and `frontend-build.log`: frontend support-surface smoke and production build passed.
- `ubuntu-readiness.log`: no fresh remote build was attempted in this resume; the note records historical disk/toolchain observations only and is not current Ubuntu host-status evidence.

This resume still does not promote arbitrary templates, neighboring rows, model-native/larger contexts beyond the checked 512/1024/2048/4096/8192 packs for the exact 1B row, production throughput, portability, or full Llama-family support.

## 2026-05-14 strict undefined-variable follow-up

Scope: exact Llama 3.2 1B Instruct Q8_0 metadata-Jinja renderer hardening. The supported renderer environment now uses MiniJinja strict undefined-variable behavior so unsupported template references fail as `UndefinedError` instead of silently rendering empty text. This preserves the existing exact-row system+user, user-only, multi-turn, assistant-continuation/final, no-BOS, loop-template, cache-reuse, and explicit `raise_exception(...)` behavior while making unsupported metadata-template cases more honest.

Validation artifact: `target/cron-95495a91-20260514T0450Z-jinja-strict-undefined-head-d927cef5dc2a/`

Recorded local gates:

- `cargo fmt --check`
- `cargo test metadata_jinja_renderer --lib -- --nocapture` — 15/15 focused metadata-Jinja tests passed, including the new undefined-variable unsupported cases.
- `cargo test exact_llama32_1b --lib -- --nocapture` — 6/6 exact-row required-renderer tests passed.
- `cargo test capabilities_report_support_contract_and_planned_lanes --test api_vertical_slice -- --nocapture` — API support-contract/capabilities gate passed, including the 8192 bounded-pack surface for the exact 1B row.
- `cargo test` — full local test suite passed, including runtime/multi-CPU unit gates already covered in the suite.
- `git diff --check`
- `scripts/check-public-scrub.sh`

No new model-runtime 8192 parity bundle was produced in this follow-up; the existing exact-row 8192 runtime bundle remains the current cited model-runtime evidence. No broad arbitrary-template, neighboring-row, production-throughput, portability, or broader/full-support claim is promoted here.

## 2026-05-14 current-head dot-access/API/runtime guard refresh

Scope: exact Llama 3.2 1B Instruct Q8_0 metadata-Jinja renderer hardening plus the adjacent 8192/API and focused multi-CPU runtime guardrails. Added a renderer execution test for MiniJinja `message.role` / `message.content` dot-field access and `loop.index0`, in addition to the existing bracket-access Llama 3 template shape coverage. This closes a small prompt-construction parity edge common in HuggingFace chat templates without widening the support claim beyond the recognized exact-row Llama 3.2 1B Q8_0 metadata-template path.

Validation artifact: `target/cron-95495a91-20260514T0704Z-jinja-dot-access-api-runtime-head-12784e8/`

Local gates recorded there:

- `cargo fmt --check` — passed.
- `cargo test metadata_jinja_renderer --lib -- --nocapture` — 16/16 focused metadata-Jinja tests passed, including dot-field access, system+user, user-only, multi-turn, assistant continuation/final, no-BOS templates, loop-template execution, cache reuse, strict undefined-variable errors, and explicit `raise_exception(...)` unsupported cases.
- `cargo test capabilities_report_support_contract_and_planned_lanes --test api_vertical_slice -- --nocapture` — passed; preserves the exact 1B metadata-Jinja renderer surface plus checked 512/1024/2048/4096/8192 bounded context fields and fail-closed broader-template wording.
- Focused runtime/multi-CPU guards passed: `q8_0_file_reader_parallelizes_wide_outputs_by_default`, `q8_0_block_reader_linear_matches_q8_path_with_parallel_chunks`, and `batch_attention_parallel_context_matches_serial`.
- `node scripts/test-chat-parity-harness.mjs` — passed.
- `scripts/check-public-scrub.sh` — passed.

Claim boundary: this is a renderer/API/runtime guard refresh for the existing exact-row 1B lane. It does not promote model-native/larger context beyond checked packs, arbitrary templates beyond the supported row-template path, production throughput, portability, neighboring rows, or broad/full support.

## 2026-05-14 10:00 UTC exact-row unsupported-template fail-closed refresh

Scope: exact Llama 3.2 1B Instruct Q8_0 metadata-Jinja renderer hardening. The model-aware renderer path now treats the exact 1B Q8_0 lane as a required metadata-template path: when that exact row is selected, chat prompt construction errors honestly if `tokenizer.chat_template` is missing or if the template is not the recognized Llama 3 instruct header/EOT shape. Generic/non-exact rows still keep the compact/fallback behavior unless `CAMELID_METADATA_CHAT_TEMPLATE=metadata` is explicitly set, and the env opt-in still executes arbitrary test templates for development coverage.

Validation artifact: `target/cron-95495a91-20260514T0959Z-jinja-unsupported-shape-api-runtime-head-83b1d00/`

Recorded gates:

- `cargo fmt --check` — passed.
- `cargo test exact_llama32_1b --lib -- --nocapture` — 8/8 exact-row renderer tests passed, including the new missing-template and unrecognized-template fail-closed cases.
- `cargo test metadata_jinja_renderer --lib -- --nocapture` — 16/16 focused metadata-Jinja tests passed, preserving system+user, user-only, multi-turn, assistant continuation/final, no-BOS, loop-template, dot-access, cache, strict undefined-variable, and explicit `raise_exception(...)` behavior.
- `cargo test capabilities_report_support_contract_and_planned_lanes --test api_vertical_slice -- --nocapture` — API/support contract passed, keeping the exact 1B metadata-Jinja renderer surface and checked 512/1024/2048/4096/8192 bounded-pack fields.
- Focused runtime/multi-CPU guards passed locally: `q8_0_file_reader_parallelizes_wide_outputs_by_default`, `q8_0_block_reader_linear_matches_q8_path_with_parallel_chunks`, and `batch_attention_parallel_context_matches_serial`.
- `node scripts/test-chat-parity-harness.mjs` — passed.
- `scripts/check-public-scrub.sh` — passed.
- `git diff --check` — passed.
- `cargo test` — full local Rust suite passed: 201 lib tests, 12 main tests, 59 API vertical tests, and all integration/doc-test targets.
- Canonical Ubuntu validation host readiness showed `/` at 54% used with ~91G available; focused current working-tree validation using the Rust 1.87 toolchain passed `cargo test exact_llama32_1b --lib -- --nocapture` and `cargo test capabilities_report_support_contract_and_planned_lanes --test api_vertical_slice -- --nocapture`. The temporary remote validation checkout was removed after the run.

Claim boundary: this is a fail-closed prompt-construction/API/runtime guard refresh for the existing exact-row 1B lane. It does not add a new model-runtime 8192 parity bundle and does not promote arbitrary templates beyond the supported row-template path, neighboring rows, model-native/larger contexts beyond checked packs, production throughput, portability, or broad/full support.

## 2026-05-14 12:18 UTC full Llama 3.2 GGUF-template execution refresh

Scope: exact Llama 3.2 1B Instruct Q8_0 metadata-Jinja prompt-construction parity. Added current-code coverage for the archived full Llama 3.2 GGUF `tokenizer.chat_template` shape, not just the compact/subset header template. The new tests exercise full-template parsing/execution with the built-in default date branch, system-message extraction/slicing, user-only/multi-turn rendering, assistant generation-prefix continuation, bracket and dot message access, and BOS de-duplication. The exact 1B Q8_0 model-aware path executes this full metadata template without `CAMELID_METADATA_CHAT_TEMPLATE` env opt-in; non-exact rows remain on the existing compact/fallback contract.

Validation artifact: `target/cron-95495a91-20260514T1218Z-full-llama32-jinja-head-2e77b02/`

Recorded local gates:

- `cargo fmt --check` — passed.
- `cargo test full_llama32_gguf_template --lib -- --nocapture` — passed.
- `cargo test metadata_jinja_renderer --lib -- --nocapture` — passed with the new full-template system+user case in the focused renderer suite.
- `cargo test exact_llama32_1b --lib -- --nocapture` — passed with the new exact-row full-template multi-turn case.
- `cargo test capabilities_report_support_contract_and_planned_lanes --test api_vertical_slice -- --nocapture` — passed, preserving the exact 1B metadata-Jinja renderer surface and checked 512/1024/2048/4096/8192 bounded-pack fields.
- Focused runtime/multi-CPU guards passed locally: `q8_0_file_reader_parallelizes_wide_outputs_by_default`, `q8_0_block_reader_linear_matches_q8_path_with_parallel_chunks`, and `batch_attention_parallel_context_matches_serial`.
- `node scripts/test-chat-parity-harness.mjs` — passed.
- `cargo test` — full local Rust suite passed.
- `git diff --check` — passed.
- `scripts/check-public-scrub.sh` — passed.

Canonical Ubuntu validation was not re-run for this prompt-template-only slice because the added coverage is deterministic metadata rendering and the local full suite plus focused API/runtime guards materially exercise the changed code. No new model-runtime 8192 parity bundle or production-throughput evidence is promoted. This remains exact-row metadata-template support only; arbitrary templates beyond the supported row path, neighboring rows, model-native/larger contexts beyond checked packs, portability, and broad/full support remain unpromoted.
