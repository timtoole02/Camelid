#!/usr/bin/env bash
# Fail when a restated Rust toolchain version disagrees with rust-toolchain.toml.
#
# rust-toolchain.toml is the single source of truth for the toolchain this repo
# builds with. rustup reads it as a *directory override*, which outranks the
# `rustup default` that dtolnay/rust-toolchain sets — so on a checkout of this
# repo the file decides what every `cargo` invocation runs, and the version
# named in ci.yml decides nothing. That asymmetry is exactly why the two drifted
# apart unnoticed for months (rust-toolchain.toml 1.87 -> 1.95 on 2026-05-14,
# ci.yml 1.87 -> 1.89 on 2026-06-16): the restatement was inert, so no run ever
# went red to report the disagreement.
#
# GitHub Actions cannot interpolate an expression into a `uses:` ref, so ci.yml
# has to spell the version out. This check is what keeps that spelling honest.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toml="$root/rust-toolchain.toml"
workflow="$root/.github/workflows/ci.yml"

for f in "$toml" "$workflow"; do
  [ -f "$f" ] || { echo "check-toolchain-pins: missing $f" >&2; exit 1; }
done

# `channel = "1.95.0"`, tolerating single quotes and surrounding whitespace.
want="$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*["'"'"']\([^"'"'"']*\)["'"'"'].*/\1/p' "$toml" | head -1)"
[ -n "$want" ] || { echo "check-toolchain-pins: no channel in $toml" >&2; exit 1; }
echo "rust-toolchain.toml channel: $want"

found=0
status=0
while IFS= read -r hit; do
  found=$((found + 1))
  line="${hit%%:*}"
  ref="$(printf '%s' "$hit" | sed 's/.*dtolnay\/rust-toolchain@\([^[:space:]"'"'"']*\).*/\1/')"
  if [ "$ref" = "$want" ]; then
    echo "  ok   ci.yml:$line  dtolnay/rust-toolchain@$ref"
  else
    echo "::error file=.github/workflows/ci.yml,line=$line::dtolnay/rust-toolchain@$ref disagrees with rust-toolchain.toml ($want). rust-toolchain.toml is the source of truth: update this ref, not the file."
    echo "  DRIFT ci.yml:$line  dtolnay/rust-toolchain@$ref != $want"
    status=1
  fi
done < <(grep -n 'dtolnay/rust-toolchain@' "$workflow" || true)

# A rename or refactor that drops every pin would otherwise pass vacuously.
if [ "$found" -eq 0 ]; then
  echo "::error file=.github/workflows/ci.yml::no dtolnay/rust-toolchain@<version> pin found; if the toolchain install moved, update scripts/check-toolchain-pins.sh to match." >&2
  exit 1
fi

if [ "$status" -eq 0 ]; then
  echo "check-toolchain-pins: $found pin(s) agree with rust-toolchain.toml."
fi
exit "$status"
