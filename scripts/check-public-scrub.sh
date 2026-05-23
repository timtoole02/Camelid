#!/usr/bin/env bash
set -euo pipefail

# Public-repo privacy guard: keep private operator paths, key paths, host commands,
# raw SSH failures, and validation-host details out of tracked files.
patterns=(
  '/Users''/[^/]+'
  '/home/ubuntu'
  'Documents''/cert'
  'ssh ''-i'
  'ssh: connect to host'
  'Operation timed out'
  'rc=255'
  'exact stderr.*evidence bundle'
  'stderr is (captured|cited) in the evidence bundle'
  '[A-Za-z0-9._-]+@''[0-9]{1,3}([.][0-9]{1,3}){3}'
  '(^|[^0-9])10[.]([0-9]{1,3}[.]){2}[0-9]{1,3}([^0-9]|$)'
  '(^|[^0-9])192[.]168[.][0-9]{1,3}[.][0-9]{1,3}([^0-9]|$)'
  '(^|[^0-9])172[.](1[6-9]|2[0-9]|3[0-1])[.][0-9]{1,3}[.][0-9]{1,3}([^0-9]|$)'
  '54[.]218[.]217[.]232'
  '54[.]186[.]43[.]33'
  '35[.]91[.]125[.]30'
  '16[.]146[.]143[.]184'
  '[.]pem([^A-Za-z0-9_]|$)'
  '[$]HOME/Desktop/Code/backend|/Desktop/Code/backend'
  'StrictHostKeyChecking=accept-new'
  'target/model-promotion-host-[0-9TZ-]+'
)

status=0
for pattern in "${patterns[@]}"; do
  matches=$(git grep -n -I -E "$pattern" -- \
    ':!.git' \
    ':!target' \
    ':!frontend/dist' \
    ':!frontend/node_modules' \
    ':!tests/api_vertical_slice.rs' \
    ':!scripts/check-public-scrub.sh' \
    ':!scripts/test-audit-evidence-bundle-privacy.mjs' || true)
  if [[ -n "$matches" ]]; then
    printf 'public scrub guard failed for pattern: %s\n%s\n' "$pattern" "$matches" >&2
    status=1
  fi
done

branding_pattern='backendinference|BackendInference|backend inference'
branding_matches=$(git grep -n -I -E "$branding_pattern" -- \
  README.md \
  COMPATIBILITY.md \
  STATUS.md \
  ROADMAP.md \
  docs \
  frontend/README.md \
  qa/validation-notes \
  .github || true)
if [[ -n "$branding_matches" ]]; then
  printf 'public scrub guard failed for legacy backend branding: %s\n%s\n' "$branding_pattern" "$branding_matches" >&2
  status=1
fi

stale_validation_lane_pattern='remote validation is available again|remote runtime validation is available again|Current operator update: The approved Ubuntu validation lane is reopened|approved Ubuntu validation lane is reopened for Camelid promotion-grade'
stale_validation_lane_matches=$(git grep -n -I -E "$stale_validation_lane_pattern" -- \
  README.md \
  COMPATIBILITY.md \
  STATUS.md \
  ROADMAP.md \
  FULL_SUPPORT_BLOCKER_MATRIX.md \
  docs \
  frontend/README.md \
  qa/validation-notes \
  scripts ':!scripts/check-public-scrub.sh' || true)
if [[ -n "$stale_validation_lane_matches" ]]; then
  printf 'public scrub guard failed for stale validation-lane availability language: %s\n%s\n' "$stale_validation_lane_pattern" "$stale_validation_lane_matches" >&2
  status=1
fi

local_bundle_pattern='qa/evidence-bundles/(backend-local|local-|tpm-local-)'
local_bundle_matches=$(git grep -n -I -E "$local_bundle_pattern" -- \
  README.md \
  COMPATIBILITY.md \
  STATUS.md \
  ROADMAP.md \
  FULL_SUPPORT_BLOCKER_MATRIX.md \
  docs \
  frontend/README.md \
  frontend/src \
  frontend/scripts \
  qa/validation-notes \
  scripts ':!scripts/check-public-scrub.sh' ':!scripts/test-audit-evidence-bundle-privacy.mjs' || true)
if [[ -n "$local_bundle_matches" ]]; then
  printf 'public scrub guard failed for local-only evidence bundle citation: %s\n%s\n' "$local_bundle_pattern" "$local_bundle_matches" >&2
  status=1
fi

exit "$status"
