# Docs support-contract and host-reporting audit - 2026-05-21T15:15Z

## Scope

Docs/context-only retained audit for support-contract honesty and canonical Ubuntu host reporting.

Remote validation was not attempted in this run. This audit makes no host availability, host failure, authentication, throughput, RSS, frontend, API, portability, support-row, or default-on claim.

## Feedback loop

Commands ran from the docs worktree at source head `65c4f3317941`:

```bash
canonical host stale-failure phrase scan (pattern constructed at runtime; raw pattern intentionally omitted)
stale validation-lane availability phrase scan (raw pattern intentionally omitted)
guarded support-contract wording scan
./scripts/check-public-scrub.sh
node scripts/check-public-evidence-claims.mjs
git diff --check
```

Focused outputs are preserved in `stale-host-scan.txt`, `stale-validation-lane-scan.txt`, `support-contract-guarded-scan.txt`, `public-scrub.txt`, `public-evidence-claims.txt`, `diff-check.txt`, and `green-checks.txt`.

## Result

- The canonical host stale-failure phrase scan found no live docs/evidence-summary/status matches.
- The stale validation-lane availability scan found no public-doc matches.
- The literal auth-denial phrase scan found no docs/evidence-summary/status matches.
- The support-contract scan remained guarded; no row/support/default-on claim was widened.
- `scripts/check-public-scrub.sh` exited 0.
- `node scripts/check-public-evidence-claims.mjs` exited 0.
- `git diff --check` exited 0.
- Earlier audit summaries that embedded raw search patterns were neutralized so they no longer repeat stale host-failure wording.

## Retain/reject

Retained as a safe docs/context audit slice. No support rows, API behavior, frontend behavior, Ubuntu host status, portability, default-on behavior, RSS, or throughput claims were widened.
