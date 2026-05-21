# Docs support-contract and host-reporting audit - 2026-05-21T04:26Z

## Scope

Docs/context-only retained audit for support-contract honesty and canonical Ubuntu host reporting.

Remote validation was not attempted in this run. No host-access failure, outage, or authentication failure is claimed.

## Feedback loop

Commands run from the docs worktree at source head `6c39e2998185`:

```bash
rg -n -S 'Permission denied \(publickey\)|Ubuntu (is )?(blocked|\bdown\b|unavailable)|ubuntu (is )?(blocked|\bdown\b|unavailable)|canonical Ubuntu host (is )?(blocked|\bdown\b|unavailable)|canonical host (is )?(blocked|\bdown\b|unavailable)|host (is )?(blocked|\bdown\b|unavailable).*Ubuntu|Ubuntu.*host (is )?(blocked|\bdown\b|unavailable)' CONTEXT.md docs README.md COMPATIBILITY.md STATUS.md ROADMAP.md FULL_SUPPORT_BLOCKER_MATRIX.md frontend/README.md frontend/src frontend/scripts tests qa/validation-notes qa/evidence-bundles/*/README.md qa/evidence-bundles/*/summary.json
rg -n -S 'remote validation (was|is) (blocked|down|unavailable)|remote runtime validation (was|is) (blocked|down|unavailable)|validation lane (was|is) (blocked|down|unavailable)|approved Ubuntu validation lane is reopened|remote validation is available again|remote runtime validation is available again' README.md COMPATIBILITY.md STATUS.md ROADMAP.md FULL_SUPPORT_BLOCKER_MATRIX.md docs frontend/README.md qa/validation-notes
rg -n -S 'support-contract supported|fully supported|production-ready|default-on acceleration|portability claim|broad-family support is active|broad support claim is active' README.md COMPATIBILITY.md STATUS.md ROADMAP.md FULL_SUPPORT_BLOCKER_MATRIX.md docs frontend/README.md frontend/src frontend/scripts tests/api_vertical_slice.rs
./scripts/check-public-scrub.sh
```

Focused outputs are preserved in `stale-host-scan.txt`, `stale-validation-lane-scan.txt`, `support-contract-guarded-scan.txt`, and `green-checks.txt`.

## Result

- The stale host-failure scan returned no matches.
- The stale validation-lane availability scan returned no public doc matches.
- The support-contract scan found only guarded wording: exact-row boundaries, negative caveats, default-off experiments, or blocker-matrix text.
- `scripts/check-public-scrub.sh` passed.

## Retain/reject

Retained as a safe docs/context audit slice. No support rows, API behavior, frontend behavior, Ubuntu host status, portability, default-on behavior, RSS, or throughput claims were widened.
