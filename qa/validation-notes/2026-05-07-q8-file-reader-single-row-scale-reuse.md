# 2026-05-07 — Q8 file-reader single-row scale reuse

Scope: backend/runtime structural hot-path cleanup only. This does not widen model support, API capabilities, frontend readiness, context support, or broader/full 8B claims.

Git/evidence posture:
- Initial lane check was clean `main` at `af537717f0913ebde29c1453dc3eeb9f571bd64a` (`Record current-head exact guardrails`) after preserving preexisting unrelated support-copy/API/doc edits in stash `pre-cron-camelid-existing-support-copy-edits-20260507T1513Z`.
- Before committing this patch, `origin/main` was fetched and local `main` was at `10a0a19c020bb18bedf0f545ef6a5c920b3f4b02` (`Publish bounded four-row parity claim`); this change stayed narrow to Q8 file-backed runtime code plus this validation note.
- Existing evidence remains bounded: TinyLlama is the live current gate; Llama 3.2 1B/3B are exact-row bounded through checked 2048 packs; Llama 3 8B has exact-row bounded 512/1024/2048 checked-pack evidence only, and any later runtime/source changes require fresh canonical PASS artifacts before being called current-head green.

Change:
- Updated the file-backed Q8_0 single-row matmul paths to decode per-block row scales once per chunk and reuse `dot_q8_0_encoded_row_with_scales`.
- Applied the same scale-reuse path to both the general block-reader matmul fast path and borrowed output-projection/file-backed linear path.
- This keeps the existing chunked read behavior and bounded scratch buffers while removing repeated f16 scale decode from the dot inner loop for single-token decode/output projection style calls.

Validation:
- `./scripts/with-rustup-cargo.sh fmt --all -- --check`
- `./scripts/with-rustup-cargo.sh test q8_0_file_backed -- --nocapture`
- `./scripts/with-rustup-cargo.sh test`
- `./scripts/with-rustup-cargo.sh clippy --all-targets -- -D warnings`

Canonical Ubuntu lane check:
- Verified only on the project-approved Ubuntu validation host.
- Host responded `host_ok` at `2026-05-07 15:16:36 UTC`.
- No additional long 8B context/parity remote validation was attempted for this local slice; no duplicate long 8B context/parity run was launched.

Claim boundary: code/test-only structural headroom. The existing exact-row bounded 8B checked-pack boundary is unchanged; model-native/larger context, arbitrary templates, production throughput, portability, neighboring rows, and broad/full Llama/8B support still require separate canonical PASS artifacts and aligned support surfaces.
