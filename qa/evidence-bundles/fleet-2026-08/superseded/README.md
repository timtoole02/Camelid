# Superseded bundles

These two bundles were produced before a defect in the test harness was found
and fixed. They are retained rather than deleted, because the defect and its
correction are part of the record.

## What was wrong

Both probes waited for `GET /v1/models` to return HTTP 200 and treated that as
readiness. Camelid's serve answers 200 with an empty `{"data":[]}` while the
model is still loading, so on a host where loading outran the poll the probe
fired at a server that was up but had no model. Every request came back
`{"error":{"code":"model_not_loaded"}}`.

## Why it mattered more than a failed run

`cpu-thread-determinism-20260822T135532Z` reports `diverged_count: 0` while
every one of its seven runs has `status: no_completion`. Read without opening
the runs array, that says determinism held. Nothing was compared. Aggregating it
would have produced a determinism claim resting on zero measurements.

`r007-retest-20260822T135607Z` reports `no_choices_in_response`, which looks
like a platform difference on this host versus the others. It was not. The same
host returns `FIXED_tool_calls_populated` once readiness is checked correctly.

## The fix

Readiness now means the model is loaded: a shared `wait_for_model()` polls until
`data[]` is non-empty and returns the served model id from that same response,
so readiness and model identity cannot disagree. It also distinguishes a dead
server from a live one that never loaded, which the old code conflated.

The determinism receipt now carries `completed_runs` and an explicit `verdict`.
Fewer than two completed runs reports `inconclusive_insufficient_runs`, never a
pass.

## Replacements

- `cpu-thread-determinism-20260822T140258Z-head-db283b69` (7/7 completed)
- `r007-retest-20260822T140235Z-head-db283b69` (FIXED_tool_calls_populated)

Same host, same model file, same commit, corrected harness.
