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

- `rg -n "ssh -o IdentitiesOnly=yes -i /Users/timtoole/Documents/cert/ubuntu\\.pem|~/Documents/cert/ubuntu\\.pem|35\\.85\\.220\\.175|54\\.186\\.43\\.33|54\\.186\\.104\\.93" CONTEXT.md docs README.md` returned no matches (`rc=1`).
- `rg -n "ssh -o BatchMode=yes -o ConnectTimeout=10 -o IdentitiesOnly=yes -i /Users/timtoole/Documents/cert/ubuntu\\.pem ubuntu@16\\.146\\.143\\.184" CONTEXT.md docs README.md` found the canonical command in `CONTEXT.md` and `docs/performance/ubuntu-x86-q8.md` (`rc=0`).
- `git diff --check` passed.

## Retain/Reject

Retain as a docs/context hygiene slice. It changes no support contract rows and adds no performance, parity, RSS, portability, frontend, or default-on claim.
