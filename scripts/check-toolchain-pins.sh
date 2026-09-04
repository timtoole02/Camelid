#!/usr/bin/env bash
# Fail when a restated Rust toolchain version disagrees with rust-toolchain.toml.
#
# rust-toolchain.toml is the single source of truth for the toolchain this repo
# builds with. rustup reads it as a *directory override*, which outranks the
# `rustup default` that dtolnay/rust-toolchain sets — so on a checkout of this
# repo the file decides what every `cargo` invocation runs, and the version
# named in a workflow decides nothing. That asymmetry is exactly why the two
# drifted apart unnoticed for months (rust-toolchain.toml 1.87 -> 1.95 on
# 2026-05-14, ci.yml 1.87 -> 1.89 on 2026-06-16): the restatement was inert, so
# no run ever went red to report the disagreement.
#
# GitHub Actions cannot interpolate an expression into a `uses:` ref, so the
# workflows have to spell the version out. This check keeps that spelling honest
# across ALL of them — release.yml pins the toolchain four more times, and those
# builds ship artifacts, so they are worth the same guard as CI.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toml="$root/rust-toolchain.toml"
workflow_dir="$root/.github/workflows"

[ -f "$toml" ] || { echo "check-toolchain-pins: missing $toml" >&2; exit 1; }
[ -d "$workflow_dir" ] || { echo "check-toolchain-pins: missing $workflow_dir" >&2; exit 1; }

# `channel = "1.95.0"`, tolerating single quotes and surrounding whitespace.
want="$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*["'"'"']\([^"'"'"']*\)["'"'"'].*/\1/p' "$toml" | head -1)"
[ -n "$want" ] || { echo "check-toolchain-pins: no channel in $toml" >&2; exit 1; }
echo "rust-toolchain.toml channel: $want"

found=0
status=0
for wf in "$workflow_dir"/*.yml "$workflow_dir"/*.yaml; do
  [ -e "$wf" ] || continue
  rel="${wf#"$root"/}"
  while IFS= read -r hit; do
    found=$((found + 1))
    line="${hit%%:*}"
    ref="$(printf '%s' "$hit" | sed 's/.*dtolnay\/rust-toolchain@\([^[:space:]"'"'"']*\).*/\1/')"
    if [ "$ref" = "$want" ]; then
      echo "  ok    $rel:$line  dtolnay/rust-toolchain@$ref"
    else
      echo "::error file=$rel,line=$line::dtolnay/rust-toolchain@$ref disagrees with rust-toolchain.toml ($want). rust-toolchain.toml is the source of truth: update this ref, not the file."
      echo "  DRIFT $rel:$line  dtolnay/rust-toolchain@$ref != $want"
      status=1
    fi
  done < <(grep -n 'dtolnay/rust-toolchain@' "$wf" || true)
done

# A rename or refactor that drops every pin would otherwise pass vacuously.
if [ "$found" -eq 0 ]; then
  echo "::error::no dtolnay/rust-toolchain@<version> pin found in $workflow_dir; if the toolchain install moved, update scripts/check-toolchain-pins.sh to match." >&2
  exit 1
fi

if [ "$status" -eq 0 ]; then
  echo "check-toolchain-pins: $found pin(s) across $(ls "$workflow_dir"/*.yml "$workflow_dir"/*.yaml 2>/dev/null | wc -l | tr -d ' ') workflow file(s) agree with rust-toolchain.toml."
fi
exit "$status"
