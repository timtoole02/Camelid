# Docs Support Contract / Host Audit

Time: 2026-05-21 01:39 UTC

Scope:

- `CONTEXT.md`
- `README.md`
- `COMPATIBILITY.md`
- `STATUS.md`
- `docs/**/*.md`
- `qa/validation-notes/**/*.md`

Purpose:

- Keep support-contract wording exact-row and evidence-scoped.
- Confirm public docs do not carry stale negative canonical Ubuntu host-access wording.
- Preserve the host-reporting rule: negative canonical Ubuntu host status is current only when the canonical SSH probe was run in the same slice and failure stderr is cited.

Commands:

```bash
rg -n 'auth-denial phrase|canonical Ubuntu host (blocked|down|unavailable)|Ubuntu host (blocked|down|unavailable)|host access (blocked|down|unavailable)' CONTEXT.md README.md COMPATIBILITY.md STATUS.md docs qa/validation-notes -g '*.md' -g '*.txt'

rg -n 'support(ed|)|active validation|not supported|bounded one-token|exact-row' README.md COMPATIBILITY.md STATUS.md docs/VALIDATION_MATRIX.md docs/CONTRIBUTOR_QUICKSTART.md -g '*.md'

git diff --check

bash scripts/check-public-scrub.sh

git status --short --branch
```

Results:

- Stale host-failure scan: PASS; no stale negative canonical-host access wording found in the scoped public docs and validation notes.
- Support-contract spot check: PASS; supported rows remain exact-row scoped, Mistral remains active validation without support promotion, and Mixtral remains partial backend-runtime evidence only.
- Public scrub guard: PASS.
- Diff whitespace check: PASS.
- Remote validation was not attempted in this run.

Retain / reject:

- Retain as a docs-only evidence slice.
- No support, throughput, Ubuntu timing/profiling, portability, or default-on runtime claim is added by this slice.
