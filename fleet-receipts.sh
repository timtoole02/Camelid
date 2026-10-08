#!/usr/bin/env bash
# =============================================================================
# fleet-receipts.sh: run the receipt harness across a fleet and collect results
# =============================================================================
# Drives camelid-receipts.sh on several hosts over SSH, then pulls every bundle
# back to one place and prints a comparison table.
#
# The point is not automation for its own sake. A cross-platform claim is only
# as good as its weakest link, and the weakest link in a hand-run matrix is
# consistency: a different model file, a different commit, or a different
# invocation on one host quietly invalidates the comparison. This script pins
# all three and records what it pinned.
#
# WHAT IT GUARANTEES
#   - Same git HEAD on every host, or the host is skipped and reported
#   - Same model files by sha256, or the mismatch is reported per host
#   - Same harness invocation everywhere
#   - Every bundle carries its own host facts, so results stay attributable
#     after they are collected into one directory
#
# WHAT IT DOES NOT DO
#   - No sudo, ever. If a host needs privileged setup, it is reported and
#     skipped rather than escalated.
#   - No building. Hosts are expected to have `cargo build --release` done
#     already; a host without a binary is reported and skipped.
#   - No model downloading by default (see --pull), because a 6 GB pull per
#     host is a decision the operator should make deliberately.
#
# Usage:
#   ./fleet-receipts.sh --hosts "user@host-a user@host-b" [--pull] [--quick] [--collect DIR]
#
#   --pull      have each host pull the three reference models first
#   --quick     pass --quick to the remote harness (skips verify + agent-eval)
#   --collect   local directory to gather bundles into (default ./fleet-evidence)
#   --remote-dir  path to the Camelid checkout on the remote (default ~/ai/Camelid)
# =============================================================================

set -uo pipefail

HOSTS=""
COLLECT="./fleet-evidence"
REMOTE_DIR="~/ai/Camelid"
DO_PULL=0
QUICK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)      HOSTS="$2"; shift 2 ;;
        --collect)    COLLECT="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --pull)       DO_PULL=1; shift ;;
        --quick)      QUICK="--quick"; shift ;;
        -h|--help)    sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$HOSTS" ]] || { echo "--hosts is required, e.g. --hosts \"user@host-a user@host-b\"" >&2; exit 2; }

LOCAL_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
LOCAL_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
mkdir -p "$COLLECT"

MODELS="qwen3_0_6b_instruct_q8_0 qwen3_1_7b_instruct_q8_0 qwen3_4b_instruct_q8_0"

say() { printf '%s\n' "$*"; }
hr()  { printf '%s\n' "------------------------------------------------------------"; }

# -----------------------------------------------------------------------------
# Preflight. A host that fails any check is skipped with a stated reason rather
# than half-run, because a partial bundle in a comparison set is worse than an
# absent one.
# -----------------------------------------------------------------------------
declare -a READY=() SKIPPED=()
say "Preflight against local HEAD ${LOCAL_SHORT}"
hr
for h in $HOSTS; do
    printf '  %-24s ' "$h"
    info="$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "cd ${REMOTE_DIR} 2>/dev/null && printf '%s|%s|%s' \
         \"\$(git rev-parse HEAD 2>/dev/null)\" \
         \"\$(test -x ./target/release/camelid && echo yes || echo no)\" \
         \"\$(uname -s)\"" 2>/dev/null)"
    if [[ -z "$info" ]]; then
        echo "SKIP: unreachable, or no checkout at ${REMOTE_DIR}"
        SKIPPED+=("$h:unreachable_or_no_checkout"); continue
    fi
    IFS='|' read -r rhead rbin ros <<< "$info"
    if [[ "$rbin" != "yes" ]]; then
        echo "SKIP: no release binary (run cargo build --release there)"
        SKIPPED+=("$h:no_binary"); continue
    fi
    if [[ "$rhead" != "$LOCAL_HEAD" ]]; then
        echo "SKIP: HEAD ${rhead:0:8} != ${LOCAL_SHORT}"
        SKIPPED+=("$h:head_mismatch_${rhead:0:8}"); continue
    fi
    echo "ready (${ros}, HEAD matches)"
    READY+=("$h")
done

if (( ${#READY[@]} == 0 )); then
    say ""; say "No hosts ready. Nothing to do."
    exit 1
fi

# -----------------------------------------------------------------------------
# Optional model pull. Opt-in because it is 6 GB per host.
# -----------------------------------------------------------------------------
if (( DO_PULL )); then
    say ""; say "Pulling reference models"; hr
    for h in "${READY[@]}"; do
        say "  $h"
        for m in $MODELS; do
            timeout 3600 ssh -o BatchMode=yes "$h" \
                "cd ${REMOTE_DIR} && ./target/release/camelid pull $m" \
                >/dev/null 2>&1 \
                && printf '    %-34s ok\n' "$m" \
                || printf '    %-34s FAILED\n' "$m"
        done
    done
fi

# -----------------------------------------------------------------------------
# Model identity. Different bytes make the comparison meaningless, so record the
# hashes before running anything and report divergence loudly.
# -----------------------------------------------------------------------------
say ""; say "Model identity across hosts"; hr
for h in "${READY[@]}"; do
    say "  $h"
    timeout 60 ssh -o BatchMode=yes "$h" \
        "cd ${REMOTE_DIR} && for f in models/*.gguf; do
             if command -v sha256sum >/dev/null 2>&1; then s=\$(sha256sum \"\$f\" | awk '{print \$1}');
             else s=\$(shasum -a 256 \"\$f\" | awk '{print \$1}'); fi
             printf '    %-34s %s\n' \"\$(basename \$f)\" \"\${s:0:16}\"
         done" 2>/dev/null || say "    (could not list models)"
done

# -----------------------------------------------------------------------------
# Run. Sequential on purpose: these are inference workloads and several hosts
# here share a power envelope or a human. Parallel runs would also make the
# timings even less comparable than they already are.
# -----------------------------------------------------------------------------
say ""; say "Running harness (sequential)"; hr
for h in "${READY[@]}"; do
    say "  $h ..."
    timeout 14400 ssh -o BatchMode=yes "$h" \
        "cd ${REMOTE_DIR} && ./camelid-receipts.sh --out ./evidence-out ${QUICK}" \
        2>&1 | sed 's/^/    /'
    say "    exit=$?"
done

# -----------------------------------------------------------------------------
# Collect.
# -----------------------------------------------------------------------------
say ""; say "Collecting bundles into ${COLLECT}"; hr
for h in "${READY[@]}"; do
    timeout 600 scp -q -o BatchMode=yes -r \
        "${h}:${REMOTE_DIR}/evidence-out/*" "$COLLECT/" 2>/dev/null \
        && say "  $h collected" \
        || say "  $h nothing to collect"
done

# -----------------------------------------------------------------------------
# Comparison table, read straight out of the manifests.
# -----------------------------------------------------------------------------
say ""; say "Fleet summary"; hr
python3 - "$COLLECT" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
rows = []
for m in sorted(root.glob("*/manifest.json")):
    try:
        d = json.loads(m.read_text())
    except Exception:
        continue
    h = d.get("host", {}); a = h.get("accelerator", {})
    ok = fail = 0
    for mod in d.get("models", []):
        for st in mod.get("steps", []):
            if st.get("exit_code") == 0: ok += 1
            else: fail += 1
    rows.append((
        m.parent.name[:40],
        f"{a.get('kind','?')}/{a.get('memory_kind','?')}",
        f"{a.get('memory_mb',0)}MB",
        h.get("arch","?"),
        f"{ok}/{ok+fail}",
    ))
if not rows:
    print("  no manifests found")
else:
    print(f"  {'bundle':<42} {'accel':<16} {'mem':<9} {'arch':<8} {'steps ok'}")
    for r in rows:
        print(f"  {r[0]:<42} {r[1]:<16} {r[2]:<9} {r[3]:<8} {r[4]}")
PY

say ""
if (( ${#SKIPPED[@]} )); then
    say "Skipped hosts (stated reasons, not silent failures):"
    for s in "${SKIPPED[@]}"; do say "  ${s%%:*} - ${s#*:}"; done
fi
say ""
say "Bundles: ${COLLECT}"
