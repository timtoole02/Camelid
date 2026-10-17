#!/usr/bin/env bash
# =============================================================================
# scrub-bundle.sh: gate an evidence bundle before it leaves a controlled machine
# =============================================================================
# Some hosts in a validation fleet are not personal machines. They can run the
# harness perfectly safely, but anything produced on them has to be checked
# before it becomes a public deliverable.
#
# The bundle manifest is already written to be public-safe: it records OS, CPU,
# core counts, RAM, accelerator name/driver, and file hashes, and no hostname,
# address, port, or process name. The exposure is `raw/`, which stores the
# verbatim stdout and stderr of every step. Those streams contain absolute
# paths, and a checkout path on a work machine can carry an organisation name,
# a matter reference, or a user id.
#
# This script checks a bundle against that risk and can redact in place. It is
# deliberately noisy: a false positive costs a glance, a false negative costs a
# disclosure.
#
# One false-positive class is known and tuned for. `camelid inspect` dumps the
# model's tokenizer vocabulary, and real vocabularies contain token strings that
# look like secrets, for example a bare ".pem". Patterns that could match inside
# a vocabulary require surrounding path context so a bare token does not trip
# them. If a new pattern is added here, check it against an `inspect.out` before
# trusting it, or every bundle will report findings that are not there.
#
# The pattern set is a superset of Camelid's own scripts/check-public-scrub.sh,
# so a bundle that passes here also passes the repo's guard.
#
# Usage:
#   ./scrub-bundle.sh <bundle-dir> [--redact] [--extra 'REGEX']
#
#   --redact   rewrite offending files in place, replacing hits with [REDACTED]
#              and re-sealing SHA256SUMS afterwards. Without it, report only.
#   --extra    add an organisation-specific pattern. Repeatable.
#
# Exit: 0 clean, 1 findings present (or redacted), 2 usage error.
# =============================================================================

set -uo pipefail

BUNDLE=""; REDACT=0; declare -a EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --redact) REDACT=1; shift ;;
        --extra)  EXTRA+=("$2"); shift 2 ;;
        -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) BUNDLE="$1"; shift ;;
    esac
done
[[ -d "$BUNDLE" ]] || { echo "usage: $0 <bundle-dir> [--redact] [--extra REGEX]" >&2; exit 2; }

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# Mirrors Camelid's own guard, plus home-directory and UNC forms it does not
# need but a work host might produce.
PATTERNS=(
  '/Users/[^/[:space:]"]+'
  '/home/[^/[:space:]"]+'
  '[A-Za-z]:\\+Users\\+[^\\[:space:]"]+'
  '\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9._$-]+'
  '[A-Za-z0-9._-]+@[0-9]{1,3}([.][0-9]{1,3}){3}'
  '(^|[^0-9])10[.]([0-9]{1,3}[.]){2}[0-9]{1,3}([^0-9]|$)'
  '(^|[^0-9])192[.]168[.][0-9]{1,3}[.][0-9]{1,3}([^0-9]|$)'
  '(^|[^0-9])172[.](1[6-9]|2[0-9]|3[0-1])[.][0-9]{1,3}[.][0-9]{1,3}([^0-9]|$)'
  '[A-Za-z0-9_/.-]+[.]pem([^A-Za-z0-9_]|$)'
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}'
  'ssh -i'
  'BEGIN [A-Z ]*PRIVATE KEY'
)
for e in "${EXTRA[@]+"${EXTRA[@]}"}"; do PATTERNS+=("$e"); done

echo "Scrub gate: $BUNDLE"
echo "------------------------------------------------------------"

FOUND=0
declare -a DIRTY=()
while IFS= read -r -d '' f; do
    [[ "$(basename "$f")" == "SHA256SUMS" ]] && continue
    hits=""
    for pat in "${PATTERNS[@]}"; do
        m="$(grep -a -n -E "$pat" "$f" 2>/dev/null | head -3)"
        [[ -n "$m" ]] && hits="${hits}${m}"$'\n'
    done
    if [[ -n "$hits" ]]; then
        FOUND=1; DIRTY+=("$f")
        echo "  FINDING  ${f#$BUNDLE/}"
        printf '%s' "$hits" | sed 's/^/             /' | cut -c1-110
    fi
done < <(find "$BUNDLE" -type f -print0)

if (( ! FOUND )); then
    echo "  clean: no operator paths, addresses, emails, or key material found"
    echo "------------------------------------------------------------"
    echo "This bundle is safe to publish as-is."
    exit 0
fi

echo "------------------------------------------------------------"
if (( ! REDACT )); then
    echo "${#DIRTY[@]} file(s) carry identifying detail. Nothing was changed."
    echo "Re-run with --redact to rewrite them and re-seal the bundle."
    exit 1
fi

# Redact. Every pattern collapses to a fixed token so the surrounding text still
# reads, and the bundle is re-sealed so SHA256SUMS stays truthful afterwards.
for f in "${DIRTY[@]}"; do
    for pat in "${PATTERNS[@]}"; do
        perl -pi -e "s{$pat}{[REDACTED]}g" "$f" 2>/dev/null
    done
    echo "  redacted ${f#$BUNDLE/}"
done

( cd "$BUNDLE" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
  | while IFS= read -r -d '' f; do printf '%s  %s\n' "$(sha256_of "$f")" "${f#./}"; done \
  > SHA256SUMS )
echo "  re-sealed SHA256SUMS"
echo
echo "Redaction changes file contents, so the digests differ from the original"
echo "run. That is intended: the published bundle is self-consistent and states"
echo "it was redacted. Keep the unredacted original on the controlled machine if"
echo "it is needed for internal verification."
exit 1
