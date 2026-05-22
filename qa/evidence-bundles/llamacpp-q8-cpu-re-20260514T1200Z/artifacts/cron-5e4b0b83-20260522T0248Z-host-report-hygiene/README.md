# Cron 5e4b0b83 Host-Report Hygiene

UTC: 2026-05-22T02:48Z

## Target

Keep Camelid docs aligned with the canonical Ubuntu host-reporting contract.

## Changes

- Updated `CONTEXT.md` so the canonical Ubuntu host report command includes `BatchMode=yes`, `ConnectTimeout=10`, `IdentitiesOnly=yes`, and the absolute key path.
- Updated `docs/performance/ubuntu-x86-q8.md` to show the same canonical probe.

## Validation

Remote validation was not attempted in this run. This bundle asserts no current remote reachability or authentication status.

Local documentation checks:

- Legacy host/path scan over `CONTEXT.md`, `docs`, and `README.md` returned no matches (`rc=1`).
- Canonical command scan over `CONTEXT.md`, `docs`, and `README.md` found the allowed reporting-rule entries in `CONTEXT.md` and `docs/performance/ubuntu-x86-q8.md` (`rc=0`).
- `git diff --check` passed.

## Retain/Reject

Retain as a docs/context hygiene slice. It changes no support contract rows and adds no performance, parity, RSS, portability, frontend, or default-on claim.
