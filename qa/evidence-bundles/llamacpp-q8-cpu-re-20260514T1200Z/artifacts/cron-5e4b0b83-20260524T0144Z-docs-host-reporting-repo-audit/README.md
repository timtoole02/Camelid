# Docs host-reporting repo audit — cron 5e4b0b83, 2026-05-24T01:44Z

CAMELID SLICE:
- Target: support-contract honesty and documentation accuracy for canonical Ubuntu host reporting.
- Domain terms used/updated: support contract, evidence bundle, canonical Ubuntu host report.
- Feedback loop: local repository wording scan for forbidden canonical Ubuntu failure phrasing plus `git diff --check`.
- Files changed: `docs/performance/ubuntu-x86-q8.md` and this evidence bundle.
- Gate/env: local macOS docs-only gate; remote validation was not attempted in this run.
- Baseline: prior retained docs host-reporting audits listed in `docs/performance/ubuntu-x86-q8.md`.
- Results: repository scan found no tracked public docs, evidence summaries, or status notes repeating `Permission denied (publickey)` or equivalent stale canonical Ubuntu failure wording; `git diff --check` passed after indexing this retained audit.
- Retain/reject: retain as a safe docs/context slice. This makes no host-availability claim because the canonical SSH probe was not executed in this run.
- Next tracer bullet: if a future docs or evidence summary cites a canonical Ubuntu failure state, require the exact canonical SSH command in that same run and cite exact stderr on failure; otherwise say remote validation was not attempted.

## Commands run

```sh
git grep -n "blocked by public-key auth\|canonical Ubuntu SSH was blocked\|probe was blocked\|Permission denied (publickey)" -- docs CONTEXT.md '*.md'
rg -n "Permission denied \(publickey\)|public-key auth|canonical Ubuntu SSH was blocked|canonical Ubuntu host probe was blocked" .
git diff --check
```
