# Docs support-contract and host-reporting audit - 2026-05-21T10:17Z

## Scope

Docs/context-only retained audit for support-contract honesty and canonical Ubuntu host reporting.

Remote validation was not attempted in this run. No host-access failure, outage, or authentication failure is claimed.

## Feedback loop

Commands run from the docs worktree at source head `fdfbe70e6739`:

```bash
# canonical host stale-failure phrase scan (raw pattern intentionally omitted)
# stale validation-lane availability phrase scan (raw pattern intentionally omitted)
rg -n -S 'support-contract supported|fully supported|production-ready|default-on acceleration|portability claim|broad-family support is active|broad support claim is active' README.md COMPATIBILITY.md STATUS.md ROADMAP.md FULL_SUPPORT_BLOCKER_MATRIX.md docs frontend/README.md frontend/src frontend/scripts tests/api_vertical_slice.rs
./scripts/check-public-scrub.sh
```

Focused outputs are preserved in `stale-host-scan.txt`, `stale-validation-lane-scan.txt`, `support-contract-guarded-scan.txt`, `public-scrub.txt`, `green-checks.txt`, and `post-edit-checks.txt`.

## Result

- The stale host-failure scan returned no matches.
- The stale validation-lane availability scan returned no public doc matches.
- The support-contract scan found only guarded wording: exact-row boundaries, negative caveats, default-off experiments, or blocker-matrix text.
- `scripts/check-public-scrub.sh` passed with no stdout.
- After adding this evidence note and linking it from the public evidence list, the stale host-failure scan still returned no matches and `scripts/check-public-scrub.sh` still passed.

## Retain/reject

Retained as a safe docs/context audit slice. No support rows, API behavior, frontend behavior, Ubuntu host status, portability, default-on behavior, RSS, or throughput claims were widened.
