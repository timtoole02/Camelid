# Docs Support-Contract And Host-Reporting Audit - 2026-05-22T10:14Z

## Scope

Docs/context/evidence-summary retained audit for support-contract honesty and canonical Ubuntu host reporting.

Remote validation was not attempted in this run. This audit makes no host availability, host failure, authentication, throughput, RSS, frontend, API, portability, support-row, or default-on claim.

## Feedback loop

Commands ran from the docs worktree at source head `d7d5036ed33d`:

```bash
focused stale host-status wording scan
retired Ubuntu host alias and tilde-key scan
guarded support-contract wording scan
bash scripts/check-public-scrub.sh
node scripts/check-public-evidence-claims.mjs
git diff --check
```

Focused outputs are preserved in `green-checks.txt`.

## Result

- Two older retained scan summaries repeated stale validation-lane status phrases inside their recorded regex patterns; this slice neutralized those summaries without changing their no-match result.
- The refreshed stale host-status scan returned no current stale host-failure wording matches after neutralization.
- Retired Ubuntu host aliases and the tilde key path were absent from public docs/context/status and retained docs-room evidence summaries.
- Guarded support-contract wording remains exact-row/default-off/evidence-needed scoped for supported rows and developer experiments.
- `bash scripts/check-public-scrub.sh` exited 0.
- `node scripts/check-public-evidence-claims.mjs` exited 0.
- `git diff --check` exited 0.

## Retain/reject

Retained as a safe docs/context/evidence-summary hygiene slice. No support rows, API behavior, frontend behavior, Ubuntu host status, portability, default-on behavior, RSS, or throughput claims were widened.
