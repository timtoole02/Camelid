#!/usr/bin/env bash
# Capture hardware + software versions into an evidence bundle directory.
# Usage: capture-env.sh <bundle_dir>
set -euo pipefail
BUNDLE="${1:?usage: capture-env.sh <bundle_dir>}"
mkdir -p "$BUNDLE"

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -a)"
  echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  echo "mem_bytes: $(sysctl -n hw.memsize 2>/dev/null || echo unknown)"
} > "$BUNDLE/hardware.txt"
system_profiler SPHardwareDataType >> "$BUNDLE/hardware.txt" 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
{
  echo "camelid_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "camelid_commit_short: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "cargo: $(cargo --version 2>/dev/null || echo n/a)"
  echo "rustc: $(rustc --version 2>/dev/null || echo n/a)"
  echo "python3: $(python3 --version 2>&1 || echo n/a)"
  echo "llama-cli: $(llama-cli --version 2>&1 | head -1 || echo n/a)"
} > "$BUNDLE/versions.txt"

# MLX versions if the venv is active or discoverable
if command -v pip >/dev/null 2>&1; then
  pip show mlx 2>/dev/null | sed -n 's/^Version: /mlx: /p' >> "$BUNDLE/versions.txt" || true
  pip show mlx-lm 2>/dev/null | sed -n 's/^Version: /mlx-lm: /p' >> "$BUNDLE/versions.txt" || true
fi

echo "wrote $BUNDLE/hardware.txt and $BUNDLE/versions.txt"
