#!/usr/bin/env bash
# ABBA so linear thermal drift cancels instead of being attributed to the change.
# Sequential GPU A/Bs on this laptop have faked ~1.8x wins from drift alone, so the
# order and the cooldowns are part of the method, not ceremony.
set -uo pipefail
export SCRATCH="~/AppData/Local/Temp/claude/C--Users-timto/da34b761-d343-4f3f-9153-e0a13209f877/scratchpad"
cd ~/wt-prefixcont

cool() {
  for _ in $(seq 1 40); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -n "$t" ] && [ "$t" -le 55 ] && break
    sleep 3
  done
  echo "[cool] gpu=${t}C"
}

for arm in on-1:1 off-1:0 off-2:0 on-2:1; do
  label="${arm%%:*}"
  cont="${arm##*:}"
  cool
  echo "=== arm $label (CAMELID_CUDA_PREFIX_CONTINUATION=$cont) ==="
  bash "$SCRATCH/run-arm.sh" "$label" "$cont" 8099 5
done
echo "=== ALL ARMS DONE ==="
