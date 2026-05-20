# Camelid docs Ubuntu host-honesty guard — 2026-05-20T18:51Z

- Scope: documentation/evidence-summary wording only; no runtime/support-contract promotion.
- Remote validation was not attempted in this run; no current Ubuntu host failure is claimed.
- Purged stale host-status wording from docs/evidence notes encountered during the scan and kept local-only slices non-promotional.
- Scrubbed privacy-sensitive host/key/path literals in two newly present CI/QA guard evidence directories so strict privacy audit can pass.
- Gates: `node scripts/check-public-evidence-claims.mjs`; `node scripts/audit-evidence-bundle-privacy.mjs --root qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z --strict`; focused stale-host wording grep.
