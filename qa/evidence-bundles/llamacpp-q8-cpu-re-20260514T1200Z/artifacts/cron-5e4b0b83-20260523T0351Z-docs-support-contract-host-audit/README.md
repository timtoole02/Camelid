# Docs Support-Contract And Host-Reporting Audit - 2026-05-23T03:51Z

## Scope

Docs/context/evidence-summary retained audit for support-contract honesty and canonical Ubuntu host-reporting hygiene.

Remote validation was not attempted in this run. This audit makes no host access, authentication, throughput, RSS, frontend, API, portability, support-row, or default-on claim.

## Feedback loop

Commands ran from the docs worktree at source head `7d286e163e7de4e0c33ee92897d5784ac0646543`:

```bash
word-bound negative canonical-host status scan
retired Ubuntu host alias and key-shorthand scan
raw SSH auth-failure phrase scan
bash scripts/check-public-scrub.sh
node scripts/check-public-evidence-claims.mjs
git diff --check
```

Focused outputs are preserved in `green-checks.txt`.

## Result

- The word-bound negative canonical-host status scan found no current negative canonical-host status claim in public docs/context/status or docs-room evidence summaries.
- Retired Ubuntu host aliases and the disallowed key shorthand were absent from public docs/context/status and docs-room evidence summaries.
- No raw SSH auth-failure phrase was present in the audited public docs/context/status or docs-room evidence summaries.
- One prior retained audit transcript line that carried raw retired-host/key-shorthand search text was neutralized.
- Guarded support-contract wording remains exact-row/default-off/evidence-needed scoped for supported rows and developer experiments.
- `bash scripts/check-public-scrub.sh` exited 0.
- `node scripts/check-public-evidence-claims.mjs` exited 0.
- `git diff --check` exited 0.

## Retain/reject

Retained as a safe docs/context/evidence-summary hygiene slice. No support rows, API behavior, frontend behavior, Ubuntu host status, portability, default-on behavior, RSS, or throughput claims were widened.
