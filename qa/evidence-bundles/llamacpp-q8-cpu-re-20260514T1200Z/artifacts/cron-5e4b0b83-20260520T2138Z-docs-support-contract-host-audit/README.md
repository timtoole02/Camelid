# Docs support-contract and host-reporting audit - 2026-05-20T21:38Z

## Scope

Docs/context-only retained audit for support-contract honesty and canonical Ubuntu host reporting.

Remote validation was not attempted in this run. No host-access failure is claimed.

## Feedback loop

Commands run from the docs worktree:

```bash
# canonical host stale-failure phrase scan (raw pattern intentionally omitted)
# canonical host stale-failure phrase scan (raw pattern intentionally omitted)
rg -n -S '<private-host-reference-patterns>' CONTEXT.md docs README*
rg -n -S 'production-ready|fully supported|support-contract supported|default-on|throughput claim|RSS claim|broad platform claim' docs README* CONTEXT.md
```

Raw outputs are preserved in:

- `narrow-stale-host-scan.txt`
- `broad-stale-host-scan.txt`
- `private-host-reference-scan.txt` (sanitized summary; exact private host/key strings are intentionally not repeated in this artifact)
- `support-contract-scan.txt`

## Result

- The narrow stale-host-failure scan returned no matches.
- The broader scan returned only scoped historical/evidence wording, local-validation caveats, Node-version frontend failure wording, and guarded FFN-down/default-off references; it did not find a current canonical Ubuntu host-access failure claim.
- Public docs/context mention the canonical SSH probe only in the host-reporting rule; this is intentional rule text, not a host-state claim.
- Support-contract wording remains guarded as default-off, evidence-needed, not-production-ready, or not-production-throughput language.

## Retain/reject

Retained as a safe docs/context audit slice. No support rows, API behavior, frontend behavior, performance claims, or host-access claims were widened.
