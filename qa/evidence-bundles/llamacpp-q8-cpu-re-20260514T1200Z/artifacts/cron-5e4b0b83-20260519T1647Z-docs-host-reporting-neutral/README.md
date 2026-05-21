# Docs host-reporting neutral wording retained slice

## Target

Keep Camelid support-contract documentation honest by removing stale host-access phrasing from public docs and preserving evidence-scoped Ubuntu validation status language.

## Domain terms

- support contract
- evidence bundle
- retained slice
- same-host guard
- Ubuntu x86 Q8

## Feedback loop

Remote validation was not attempted in this run; this slice is docs/context only.

Green checks run locally:

```bash
rg -n -P "Ubuntu[^\\n]*(\\bblocked\\b|\\bunavailable\\b|auth-denial phrase|validation host[^\\n]*(\\bdown\\b|\\bblocked\\b|\\bunavailable\\b|failing SSH)" docs CONTEXT.md README.md || true
```

Result: no output.

```bash
git diff -- docs/runtime/cross-lane-sync.md docs/performance/ubuntu-x86-q8.md
```

Result: public docs now say to report evidence status and avoid negative host-access claims unless the canonical SSH probe ran in the same slice and stderr is captured.

## Files changed

- `docs/performance/ubuntu-x86-q8.md`
- `docs/runtime/cross-lane-sync.md`

## Retain/reject

Retain. The docs slice narrows host-status language without changing any support claim, model row, performance claim, or default-on/default-off behavior.

## Next tracer bullet

Keep future Ubuntu status updates evidence-scoped: either record current same-host validation evidence, or state plainly that remote validation was not attempted for the slice.
