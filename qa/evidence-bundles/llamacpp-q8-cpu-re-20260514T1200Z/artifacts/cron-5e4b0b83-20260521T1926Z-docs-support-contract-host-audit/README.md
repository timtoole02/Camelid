# Docs support-contract and host-reporting audit - 2026-05-21T19:26Z

## Scope

Docs/context-only retained audit for support-contract honesty and canonical Ubuntu host reporting.

Remote validation was not attempted in this run. This audit makes no host availability, host failure, authentication, throughput, RSS, frontend, API, portability, support-row, or default-on claim.

## Feedback loop

Commands ran from the docs worktree at source head `afb3c20e528b`:

```bash
public docs/context host-reporting wording scan
public evidence-summary host-reporting wording scan with classification
guarded support-contract wording scan
./scripts/check-public-scrub.sh
node scripts/check-public-evidence-claims.mjs
git diff --check
```

Focused outputs are preserved in `green-checks.txt`.

## Result

- Public docs/context scan results were rule text, prior retained-audit references, or evidence-needed/default-off support-contract guard wording; no current Ubuntu host outage, authentication, availability, support-row, throughput, API, frontend, portability, or default-on claim was added.
- Evidence-summary scan results were classified as non-host-access failures, explicit historical success/evidence notes, or local-only validation caveats; no stale SSH-auth or current canonical Ubuntu host failure wording was found.
- Guarded support-contract wording remains exact-row/default-off scoped for supported rows and developer experiments.
- `scripts/check-public-scrub.sh` exited 0.
- `node scripts/check-public-evidence-claims.mjs` exited 0 with `public evidence claim check passed: 96 manifest(s), 49 summary file(s)`.
- `git diff --check` exited 0.

## Retain/reject

Retained as a safe docs/context audit slice. No support rows, API behavior, frontend behavior, Ubuntu host status, portability, default-on behavior, RSS, or throughput claims were widened.
