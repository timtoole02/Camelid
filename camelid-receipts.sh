#!/usr/bin/env bash
# =============================================================================
# camelid-receipts.sh: portable cross-platform evidence-bundle generator
# =============================================================================
# Produces a Camelid-convention evidence bundle from an outside platform, so
# results from one machine can be compared against another without either party
# trusting the other's prose.
#
# WHY THIS EXISTS
#   Camelid's CUDA parity evidence is recorded on Windows x86_64 with an
#   RTX 3060 Laptop (6 GB), driver 576.83, CUDA 12.9. COMPATIBILITY.md states
#   plainly that results are "specific to the recorded GPU / driver / CUDA
#   version" because f32 reduction order is GPU-specific. Nothing there covers
#   Linux x86_64, a 4 GB card, or Apple Silicon at 8 GB unified.
#
#   This script runs the same subcommands on whatever host it is given and
#   emits a bundle in the repository's own manifest shape, so three machines
#   produce three comparable artifacts rather than three anecdotes.
#
# WHAT IT PROVES, AND WHAT IT DOES NOT
#   It records what these subcommands did on this exact host with these exact
#   files. runnable-smoke attests deterministic execution, NOT parity. verify
#   proves one request for one file. Neither promotes a support claim. Those
#   scope limits are Camelid's, and this script repeats them rather than
#   quietly widening them.
#
# PORTABILITY
#   Linux and macOS. Every platform-specific call is behind a helper below:
#   sha256, core count, RAM, and accelerator detection all differ between the
#   two. Tested on Gentoo Linux x86_64 + CUDA; written to run unmodified on
#   Apple Silicon + Metal.
#
# USAGE
#   ./camelid-receipts.sh [--models-dir DIR] [--out DIR] [--binary PATH]
#                         [--label TEXT] [--quick]
#
#   --quick   skip verify and agent-eval (the slow lanes); still emits a bundle
#
# The bundle is written to <out>/<label>-<UTC timestamp>-head-<short commit>/
# =============================================================================

set -uo pipefail

MODELS_DIR="./models"
OUT_ROOT="./evidence-out"
BINARY="./target/release/camelid"
LABEL=""
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --out)        OUT_ROOT="$2"; shift 2 ;;
        --binary)     BINARY="$2"; shift 2 ;;
        --label)      LABEL="$2"; shift 2 ;;
        --quick)      QUICK=1; shift ;;
        -h|--help)    sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -x "$BINARY" ]] || { echo "camelid binary not found or not executable: $BINARY" >&2; exit 2; }
[[ -d "$MODELS_DIR" ]] || { echo "models dir not found: $MODELS_DIR" >&2; exit 2; }

# -----------------------------------------------------------------------------
# Platform abstraction. Each of these differs between Linux and macOS, and
# getting one wrong silently produces a bundle that cannot be compared.
# -----------------------------------------------------------------------------
OS="$(uname -s)"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    elif command -v shasum   >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
    else echo "no-sha256-tool"; fi
}
core_count() {
    case "$OS" in
        Linux)  nproc ;;
        Darwin) sysctl -n hw.ncpu ;;
        *)      echo 0 ;;
    esac
}
perf_core_count() {
    case "$OS" in
        Darwin) sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu ;;
        Linux)  lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}' ;;
        *)      echo 0 ;;
    esac
}
ram_bytes() {
    case "$OS" in
        Linux)  awk '/MemTotal/{printf "%.0f", $2*1024}' /proc/meminfo ;;
        Darwin) sysctl -n hw.memsize ;;
        *)      echo 0 ;;
    esac
}
# -----------------------------------------------------------------------------
# Thermal and power envelope.
#
# A timing taken on a thermally constrained host is not comparable to one taken
# on a desktop, and a laptop's own numbers are not comparable across different
# power caps or cooling. Desktops in a fleet make this easy to forget, so the
# bundle records the envelope rather than leaving it to be reconstructed later
# from memory. Every field degrades to "unknown" rather than failing: these
# paths do not exist on Apple silicon and may not exist on older Intel parts.
#
# Cooling is deliberately NOT auto-detected. External cooling pads report
# nothing to the host, so an operator-supplied value is the only honest source.
# Set CAMELID_COOLING to describe it, e.g. CAMELID_COOLING="pad at 1000 rpm".
# -----------------------------------------------------------------------------
rapl_uw() {  # $1 = constraint index
    [[ "$OS" == "Linux" ]] || { echo "null"; return; }
    local v
    for base in /sys/class/powercap/intel-rapl:0 /sys/class/powercap/intel-rapl-mmio:0; do
        v=$(cat "${base}/constraint_${1}_power_limit_uw" 2>/dev/null) && \
            [[ -n "$v" ]] && { echo "$v"; return; }
    done
    echo "null"
}
cpu_governor() {
    [[ "$OS" == "Linux" ]] || { echo "unknown"; return; }
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown"
}
cpu_epp() {
    [[ "$OS" == "Linux" ]] || { echo "unknown"; return; }
    cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo "unknown"
}
chassis_kind() {
    [[ "$OS" == "Linux" ]] || { echo "unknown"; return; }
    if [[ -d /sys/class/power_supply/BAT0 || -d /sys/class/power_supply/BAT1 ]]; then
        echo "battery-capable"
    else
        echo "no-battery"
    fi
}

cpu_model() {
    case "$OS" in
        Linux)  awk -F: '/model name/{gsub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo ;;
        Darwin) sysctl -n machdep.cpu.brand_string ;;
        *)      echo unknown ;;
    esac
}
os_version() {
    case "$OS" in
        Linux)  printf '%s %s' "$(uname -s)" "$(uname -r)" ;;
        Darwin) printf 'macOS %s (%s)' "$(sw_vers -productVersion 2>/dev/null)" "$(uname -r)" ;;
        *)      uname -a ;;
    esac
}

# Accelerator: the axis this whole comparison turns on. Discrete VRAM is a hard
# wall; unified memory is not. Report both the kind and the number so a reader
# can tell which regime a bundle came from.
ACCEL_KIND="none"; ACCEL_NAME="none"; ACCEL_MEM_MB=0; ACCEL_DRIVER="none"; ACCEL_MEM_KIND="none"
detect_accel() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        ACCEL_KIND="cuda"
        ACCEL_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        ACCEL_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
        ACCEL_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        ACCEL_MEM_KIND="discrete"
    elif [[ "$OS" == "Darwin" ]]; then
        ACCEL_KIND="metal"
        ACCEL_NAME="$(system_profiler SPDisplaysDataType 2>/dev/null \
                      | awk -F': ' '/Chipset Model/{print $2; exit}')"
        [[ -z "$ACCEL_NAME" ]] && ACCEL_NAME="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
        # Apple Silicon shares one pool. Report total RAM and say so explicitly,
        # rather than inventing a VRAM figure that does not exist.
        ACCEL_MEM_MB=$(( $(ram_bytes) / 1048576 ))
        ACCEL_DRIVER="$(sw_vers -buildVersion 2>/dev/null || echo unknown)"
        ACCEL_MEM_KIND="unified"
    fi
}
detect_accel

# -----------------------------------------------------------------------------
# Bundle identity, following the repository's own naming convention:
#   <label>-<YYYYMMDDTHHMMSSZ>-head-<short commit>
# -----------------------------------------------------------------------------
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
CAMELID_VER="$($BINARY --version 2>/dev/null | head -1)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -z "$LABEL" ]]; then
    case "$ACCEL_KIND" in
        cuda)  LABEL="linux-x86_64-cuda-${ACCEL_MEM_MB}mb" ;;
        metal) LABEL="apple-silicon-metal-$(( $(ram_bytes) / 1073741824 ))gb-unified" ;;
        *)     LABEL="cpu-only-$(uname -m)" ;;
    esac
fi

BUNDLE="${OUT_ROOT}/${LABEL}-${STAMP}-head-${GIT_SHORT}"
mkdir -p "$BUNDLE/raw"
echo "bundle: $BUNDLE"

# -----------------------------------------------------------------------------
# Run one subcommand, capture stdout/stderr and exit status, never abort the run.
# A refusal is data: Camelid fails closed on purpose, and recording that is the
# point. Swallowing it would misrepresent the platform.
# -----------------------------------------------------------------------------
run_step() {
    local name="$1"; shift
    local out="$BUNDLE/raw/${name}.out" err="$BUNDLE/raw/${name}.err"
    local t0 t1 rc
    t0=$(date +%s)
    "$@" >"$out" 2>"$err"; rc=$?
    t1=$(date +%s)
    printf '  %-46s exit=%-3s %ss\n' "$name" "$rc" "$((t1-t0))"
    STEP_JSON="${STEP_JSON}$(printf '{"step":"%s","exit_code":%s,"seconds":%s,"stdout_bytes":%s,"stderr_bytes":%s},' \
        "$name" "$rc" "$((t1-t0))" "$(wc -c <"$out" | tr -d ' ')" "$(wc -c <"$err" | tr -d ' ')")"
    return $rc
}

MODEL_JSON=""
echo
echo "Running steps. A non-zero exit is recorded, not hidden."
for gguf in "$MODELS_DIR"/*.gguf; do
    [[ -e "$gguf" ]] || continue
    base="$(basename "$gguf" .gguf)"
    echo
    echo "model: $base"
    STEP_JSON=""
    sha="$(sha256_of "$gguf")"
    bytes=$(wc -c <"$gguf" | tr -d ' ')

    # Metadata only. Fast, and proves the file parses on this platform.
    run_step "${base}.inspect" "$BINARY" inspect "$gguf"

    # The discrete-vs-unified lever. Real detected budget, then forced budgets
    # so one machine's curve can be laid over another's.
    run_step "${base}.plan-offload.detected" "$BINARY" plan-offload "$gguf"
    for b in 4096 8192; do
        run_step "${base}.plan-offload.budget${b}" "$BINARY" plan-offload "$gguf" --budget-mb "$b"
    done

    # Admission + load + greedy forward + coherence. Attests deterministic
    # execution, NOT parity.
    run_step "${base}.runnable-smoke" "$BINARY" runnable-smoke "$gguf"

    if (( ! QUICK )); then
        # One request, one file, digest-sealed. Abstains without an exact-hash profile.
        run_step "${base}.verify" "$BINARY" verify "$gguf" \
            --output "$BUNDLE/raw/${base}.verify.json"
        # Tool-call round trip. INCONCLUSIVE on a contended box is a valid result.
        run_step "${base}.agent-eval" "$BINARY" agent-eval --model "$gguf"
    fi

    MODEL_JSON="${MODEL_JSON}$(printf '{"model_file":"%s","sha256":"%s","bytes":%s,"steps":[%s]},' \
        "$(basename "$gguf")" "$sha" "$bytes" "${STEP_JSON%,}")"
done

# -----------------------------------------------------------------------------
# manifest.json, in the repository's shape. scope and privacy are stated
# explicitly because an evidence-gated project treats an unstated limit as a
# claim.
# -----------------------------------------------------------------------------
cat > "$BUNDLE/manifest.json" <<JSON
{
  "schema": "camelid.external-platform-receipts.v1",
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_head": "${GIT_HEAD}",
  "git_head_short": "${GIT_SHORT}",
  "camelid_version": "${CAMELID_VER}",
  "validation_lane": "external_contributor_platform_run",
  "privacy": "Public summary only. Operator paths, hostnames, addresses, ports, and key material are intentionally omitted; only file basenames and hashes appear.",
  "scope": "Records what camelid inspect / plan-offload / runnable-smoke / verify / agent-eval did on this exact host with these exact files. runnable-smoke attests deterministic execution, NOT parity. verify proves one request for one file. agent-eval may return INCONCLUSIVE on a contended host and that is not a failure. This bundle promotes no support claim and extends no existing claim; it is a platform data point for comparison.",
  "host": {
    "os": "$(os_version)",
    "arch": "$(uname -m)",
    "cpu": "$(cpu_model)",
    "logical_cores": $(core_count),
    "physical_cores": $(perf_core_count),
    "ram_bytes": $(ram_bytes),
    "chassis": "$(chassis_kind)",
    "thermal": {
      "pl1_uw": $(rapl_uw 0),
      "pl2_uw": $(rapl_uw 1),
      "cpu_governor": "$(cpu_governor)",
      "energy_performance_preference": "$(cpu_epp)",
      "cooling": "${CAMELID_COOLING:-not stated by operator}",
      "note": "PL1/PL2 are the RAPL package limits in microwatts, read from the MSR interface with an MMIO fallback; null means the interface is absent (normal on Apple silicon). Cooling is operator-supplied because external cooling reports nothing to the host. A bundle from a battery-capable chassis with a low PL1 is thermally bounded and its timings are not comparable to a desktop's."
    },
    "accelerator": {
      "kind": "${ACCEL_KIND}",
      "name": "${ACCEL_NAME}",
      "memory_mb": ${ACCEL_MEM_MB},
      "memory_kind": "${ACCEL_MEM_KIND}",
      "driver": "${ACCEL_DRIVER}"
    }
  },
  "quick_mode": $( ((QUICK)) && echo true || echo false ),
  "models": [${MODEL_JSON%,}]
}
JSON

# -----------------------------------------------------------------------------
# SHA256SUMS over every artifact, so the bundle is tamper-evident on its own.
# -----------------------------------------------------------------------------
( cd "$BUNDLE" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | while IFS= read -r -d '' f; do printf '%s  %s\n' "$(sha256_of "$f")" "${f#./}"; done \
  > SHA256SUMS )

echo
echo "bundle written: $BUNDLE"
echo "  manifest.json  host + scope + per-step exit codes"
echo "  raw/           stdout and stderr of every step, verbatim"
echo "  SHA256SUMS     tamper evidence over all of the above"
