#!/bin/zsh
set -euo pipefail

# One-click launcher for the exact Gemma 4 mapped-expert + MTP WebUI lane.
# It owns only the process it starts and refuses to replace an existing server.

readonly script_dir=${0:A:h}
readonly repo_root=${script_dir:h}
readonly source_commit=$(/usr/bin/git -C "$repo_root" rev-parse HEAD)
readonly port=${CAMELID_GEMMA4_WEBUI_PORT:-8181}
readonly model=${CAMELID_GEMMA4_WEBUI_MODEL:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf}
readonly cghost=${CAMELID_GEMMA4_WEBUI_CGHOST:-/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost}
readonly assistant=${CAMELID_GEMMA4_WEBUI_ASSISTANT:-/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors}
# Literal 1,408-slot Mini2 footprint used by the checked H71 descendant. The
# profile remains fixed so a launcher edit cannot silently widen resident
# memory. Context is separately exposed by health and enforced per request.
readonly profile=60,64,45,43,40,41,44,42,47,46,40,41,44,46,42,50,40,39,46,47,41,46,45,43,46,56,59,56,58,51

typeset binary=${CAMELID_GEMMA4_WEBUI_BINARY:-}
if [[ -z "$binary" ]]; then
  for candidate in \
    "$repo_root/target/release/camelid" \
    /Volumes/Untitled/cargo-targets/global/release/camelid; do
    if [[ -x "$candidate" && -f "$candidate" ]]; then
      binary=$candidate
      break
    fi
  done
fi

[[ "$port" == <1-65535> ]] || {
  print -u2 "Gemma 4 WebUI refused to start: invalid port $port"
  exit 64
}
[[ -n "$binary" && -x "$binary" && -f "$binary" ]] || {
  print -u2 "Gemma 4 WebUI refused to start: build the release binary first."
  exit 75
}
for input in "$model" "$cghost" "$assistant"; do
  [[ -f "$input" ]] || {
    print -u2 "Gemma 4 WebUI refused to start: missing $input"
    exit 75
  }
done
if /usr/sbin/lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "Gemma 4 WebUI is already using port $port; the existing server was left untouched."
  exit 76
fi

typeset free_percent
free_percent=$(/usr/bin/memory_pressure -Q | /usr/bin/awk '/free percentage/ {gsub(/%/, "", $5); print $5}')
(( free_percent >= 20 )) || {
  print -u2 "Gemma 4 WebUI refused to start: only ${free_percent}% memory is free."
  exit 75
}

typeset child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]] && /bin/kill -0 "$child_pid" 2>/dev/null; then
    /bin/kill -INT "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

print "Starting the exact Gemma 4 MTP WebUI on http://127.0.0.1:$port ..."
/usr/bin/env -i \
  HOME="${HOME:?}" \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  TMPDIR=/tmp \
  CAMELID_GHOST_ALLOW_LEGACY_SPARSE=0 \
  CAMELID_GEMMA4_GHOST_METAL=1 \
  CAMELID_GEMMA4_GHOST_METAL_SLOTS=1 \
  CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST=1 \
  CAMELID_GEMMA4_GHOST_METAL_TURBO=1 \
  CAMELID_GEMMA4_GHOST_METAL_COMMON=1 \
  CAMELID_GEMMA4_GHOST_METAL_CONTEXT=1024 \
  CAMELID_GEMMA4_KV_INIT=192 \
  CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT=0 \
  CAMELID_GEMMA4_GHOST_READ_THREADS=8 \
  CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY=1 \
  CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1 \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32 \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER="$profile" \
  CAMELID_GEMMA4_GHOST_METAL_DECODE_PROMOTION=0 \
  CAMELID_GEMMA4_SLOT_PIN=0 \
  CAMELID_GEMMA4_GHOST_METAL_HOT_PIN=0 \
  CAMELID_GEMMA4_VICTIM_CACHE=0 \
  CAMELID_GEMMA4_VICTIM_MB=0 \
  CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS=0 \
  CAMELID_GEMMA4_CHAINED_PREDICT=0 \
  CAMELID_SPEC_DECODE=off \
  CAMELID_GEMMA4_SPEC_K1_LANE=chained \
  CAMELID_GEMMA4_SPEC_CHUNK_MAX=15 \
  CAMELID_GEMMA4_SPEC_DRAFT_TOKENS=15 \
  CAMELID_GEMMA4_MTP_ADAPTIVE_WIDTH=1 \
  CAMELID_GEMMA4_MTP_ASSISTANT_PATH="$assistant" \
  CAMELID_GEMMA4_MTP_DEVICE_CHAIN=1 \
  CAMELID_GEMMA4_MTP_FULL_Q4=1 \
  CAMELID_GEMMA4_MTP_PREFILL_SEED_BOOTSTRAP=1 \
  CAMELID_GEMMA4_MTP_LOAD_WARMUP=1 \
  CAMELID_GEMMA4_MTP_BOUNDARY_ARBITRATION=1 \
  CAMELID_GEMMA4_DENSE_K8_GENERIC=1 \
  CAMELID_GEMMA4_HEAD_SPEC50_K8_COMPACT=1 \
  CAMELID_GEMMA4_MOE_MMA_K16=1 \
  CAMELID_GEMMA4_MTP_BF16_PRODUCER_FUSION=1 \
  CAMELID_GEMMA4_MTP_BF16_LATTICE_LOADS=1 \
  CAMELID_GEMMA4_GHOST_METAL_DEMAND_PROMOTION=1 \
  CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_PROFILE_FREE=1 \
  CAMELID_GEMMA4_GHOST_METAL_HOT_COLD_OVERLAP=1 \
  CAMELID_GEMMA4_GHOST_METAL_HOT_COLD_OVERLAP_PUBLISH=0 \
  CAMELID_GEMMA4_GHOST_METAL_HOT_COLD_SINGLE_DOWN=1 \
  CAMELID_GEMMA4_GHOST_METAL_DIRECT_STAGE_READ=1 \
  CAMELID_GEMMA4_GHOST_METAL_LIVE_SEQUENTIAL_PREDICT_PROBE=1 \
  CAMELID_GEMMA4_GHOST_METAL_LIVE_SEQUENTIAL_STAGE=1 \
  CAMELID_GEMMA4_GHOST_METAL_LIVE_SEQUENTIAL_FAST_PREDICT=1 \
  CAMELID_GEMMA4_GHOST_METAL_LIVE_SEQUENTIAL_STAGE_DUAL_READER=1 \
  CAMELID_GEMMA4_GHOST_METAL_PROMPT_RANKED_HOT_HANDOFF=1 \
  CAMELID_GEMMA4_GHOST_METAL_PROMPT_RANKED_HOT_HANDOFF_RESIDENT_FILL=1 \
  "$binary" serve \
    --addr "127.0.0.1:$port" \
    --model "$model" \
    --cghost "$cghost" \
    --expert-cache-mib 0 --gpu on --no-open &
child_pid=$!

typeset ready=0 health=""
for _ in {1..900}; do
  if health=$(/usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/health" 2>/dev/null); then
    if print -r -- "$health" | /usr/bin/jq -e --arg source_commit "$source_commit" '
      (.source_commit // "") != $source_commit
    ' >/dev/null 2>&1; then
      typeset running_commit
      running_commit=$(print -r -- "$health" | /usr/bin/jq -r '.source_commit // "missing"')
      print -u2 "Gemma 4 WebUI refused a stale build: binary=$running_commit source=$source_commit"
      exit 75
    fi
    if print -r -- "$health" | /usr/bin/jq -e --arg source_commit "$source_commit" '
        .source_commit == $source_commit and
        .generation_ready == true and
        .gemma4_serve_lane == "ghost_moe" and
        .gemma4_ghost_execution_mode == "full_common_metal" and
        .gemma4_ghost_common_metal_active == true and
        .gemma4_ghost_experts_metal_active == true and
        .gemma4_ghost_head_metal_active == true and
        .gemma4_ghost_common_metal_context_capacity == 1024 and
        .gemma4_ghost_exact_expert_policy_active == true and
        .gemma4_ghost_runtime_profile == "mini2-h71r-h58-h60-h62-1408-ctx1024-mtp15-adaptive-v1" and
        .gemma4_mtp_assistant_loaded == true and
        .gemma4_mtp_full_q4_active == true and
        .execution_plan.support_level == "supported_exact_row_smoke" and
        .execution_plan.selected_backend == "gemma4_ghost_moe_metal_runtime" and
        .execution_plan.prefill_path == "gemma4_ghost_moe_metal_prefill" and
        .execution_plan.decode_path == "gemma4_ghost_moe_metal_speculative_decode"
      ' >/dev/null 2>&1; then
      ready=1
      break
    fi
  fi
  /bin/kill -0 "$child_pid" 2>/dev/null || {
    print -u2 "Gemma 4 WebUI stopped before the model became ready."
    exit 75
  }
  /bin/sleep 1
done
(( ready == 1 )) || {
  print -u2 "Gemma 4 WebUI did not reach full-Metal + full-Q4 MTP readiness in 15 minutes."
  exit 75
}

/usr/bin/open "http://127.0.0.1:$port/"
print "Gemma 4 MTP is ready. Leave this window open; press Control-C to stop it."
wait "$child_pid"
child_pid=""
