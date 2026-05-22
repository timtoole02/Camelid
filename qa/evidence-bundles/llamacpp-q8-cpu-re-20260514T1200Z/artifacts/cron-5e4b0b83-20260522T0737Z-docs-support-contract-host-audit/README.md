# Docs Support-Contract And Host-Reporting Audit - 2026-05-22T07:37Z

## Scope

Docs/context/evidence-summary retained audit for support-contract honesty and canonical Ubuntu host reporting.

Remote validation was not attempted in this run. This audit makes no host availability, host failure, authentication, throughput, RSS, frontend, API, portability, support-row, or default-on claim.

## Feedback loop

Commands ran from the docs worktree at source head `6b92d6a7ab3b`:

```bash
focused stale host-status wording scan
refined stale host-status wording scan
guarded support-contract wording scan
bash scripts/check-public-scrub.sh
node scripts/check-public-evidence-claims.mjs
git diff --check
```

Focused outputs are preserved in `green-checks.txt`.

## Result

- The broad stale host-status scan found one false positive: `STATUS.md` contains `downloaded-model`, not stale host-down wording.
- The refined stale host-failure scan returned no matches.
- Guarded support-contract wording remains exact-row/default-off scoped for supported rows and developer experiments.
- `bash scripts/check-public-scrub.sh` exited 0.
- `node scripts/check-public-evidence-claims.mjs` exited 0.
- `git diff --check` exited 0.

## Retain/reject

Retained as a safe docs/context/evidence-summary hygiene slice. No support rows, API behavior, frontend behavior, Ubuntu host status, portability, default-on behavior, RSS, or throughput claims were widened.
