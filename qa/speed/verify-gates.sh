#!/usr/bin/env bash
# qa/speed/verify-gates.sh — ENFORCED PROOF LANE for the WIN2METAL byte-exact spec-verify gates.
#
# WHY THIS EXISTS (review M2/G1): the four bit-identity gates
#   metal_verify_gemv_batched_bit_identical   (Phase 3 C0 — the batched GEMV)
#   metal_verify_batch_rope_scatter_matches_row_dispatches (N<=8 dispatch collapse)
#   metal_spec_verify_bit_identical           (Phase 3 — linear verify_batch)
#   metal_tree_verify_bit_identical           (Phase 4 — tree verify_batch_tree)
# The GEMV and end-to-end gates guard on process-wide OnceLock env gates (f32y / wire / nsg8 /
# attn2 / split-K), which the CALLER must arm — a test may not arm them for itself (see the
# export block below). A plain `cargo test --all-targets` arms nothing, so those three take their
# SKIP branch, and a skipped #[test] counts as PASS. So a green `--all-targets` does NOT, by
# itself, exercise all byte-exact assertions; only this script does. The linear/tree gates
# additionally assert that the opt-in RoPE/scatter route actually encoded rather than silently
# falling back to the established row loops.
#
# This script runs each gate in ITS OWN cargo process (fresh OnceLocks) with the gates armed in
# the environment that process inherits — so the to_bits assertions actually run, and no sibling
# test is dragged onto the wire path. THIS is the byte-exactness proof of record; CI should run
# this, not rely on --all-targets.
#
# Usage:  qa/speed/verify-gates.sh
set -uo pipefail
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/Volumes/Untitled/camelid-target}"

# ARM THE GATES HERE, not inside the tests. They are process-wide OnceLocks, so an in-test
# `set_var` mutates the environment for every other Metal test in the binary — whichever have
# not read the gates yet then latch onto the wire path, where the standalone block helpers'
# 36-byte uploads are read as 34-byte wire blocks and come back NaN. The three gates below
# therefore SKIP unless the caller armed them, which is what this script is for.
export CAMELID_METAL_F32Y=1
export CAMELID_METAL_WIRE=1
export CAMELID_METAL_WIRE_NSG8=1
export CAMELID_METAL_ATTN2=1
# Split-K attention is default-ON (CAMELID_METAL_ATTN_SPLITK=0 opts out). Set it explicitly so an
# opt-out in the caller's environment cannot silently drop the straddle windows onto the v2 path —
# which would SKIP the linear/tree gates and fail this script rather than quietly prove less.
export CAMELID_METAL_ATTN_SPLITK=1
export CAMELID_METAL_VERIFY_BATCH_ROPE_SCATTER=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GATES=(
  metal_verify_gemv_batched_bit_identical
  metal_verify_batch_rope_scatter_matches_row_dispatches
  metal_spec_verify_bit_identical
  metal_tree_verify_bit_identical
)

rc=0
for g in "${GATES[@]}"; do
  echo "=== $g (isolated process) ==="
  log="$(mktemp "${TMPDIR:-/tmp}/verify_gate.XXXXXX")"
  if ! cargo test --release --lib "$g" -- --nocapture >"$log" 2>&1; then
    echo "  FAIL: $g returned non-zero"; tail -20 "$log"; rc=1; rm -f "$log"; continue
  fi
  # A SKIP here means the gates did not take effect even though this script armed them — a real
  # problem (Metal unavailable, or a gate dependency changed), NOT the benign --all-targets skip.
  if grep -q "SKIP $g" "$log"; then
    echo "  FAIL: $g SKIPPED with the gates armed (Metal device unavailable or a gate changed) — investigate"
    grep "SKIP $g" "$log" | sed 's/^/    /'; rc=1; rm -f "$log"; continue
  fi
  if ! grep -q "BIT-IDENTICAL" "$log"; then
    echo "  FAIL: $g produced no BIT-IDENTICAL line — did the assertions run?"; tail -20 "$log"; rc=1; rm -f "$log"; continue
  fi
  grep -E "BIT-IDENTICAL|PASS|test result:" "$log" | sed 's/^/  /'
  rm -f "$log"
done

echo
if [ "$rc" -eq 0 ]; then
  echo "PASS: all 4 byte-exact verify gates ran ENGAGED (including batched RoPE/scatter and split-K straddle 126 & 510) and are BIT-IDENTICAL."
else
  echo "FAIL: a byte-exact verify gate did not run/pass engaged — see above (this is the real proof, not --all-targets)."
fi
exit "$rc"
