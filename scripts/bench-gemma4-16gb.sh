#!/usr/bin/env bash
set -euo pipefail

# Benchmark Harness for Gemma 4 26B-A4B on 16GB Apple Silicon
# Outputs structured JSON metrics matching Phase 22 schema.

operator_home="${CAMELID_OPERATOR_HOME:-${HOME:?set CAMELID_OPERATOR_HOME or HOME}}"
model_root="${CAMELID_MODEL_ROOT:-${operator_home}/models}"
MODEL_PATH="${1:-${CAMELID_GEMMA4_MODEL:-${model_root}/gemma-4-26B_q4_0-it.gguf}}"
CGHOST_PATH="${2:-${CAMELID_GEMMA4_CGHOST:-${model_root}/gemma-4-26B_q4_0-it.cghost}}"
PROMPT="${3:-Write a Rust implementation of a concurrent lock-free queue.}"
MAX_TOKENS="${4:-64}"
CACHE_MIB="${5:-8192}"
SLOTS_PER_LAYER="${6:-96}"
OUT_DIR="qa/evidence-bundles/gemma4-26b-16gb"

mkdir -p "${OUT_DIR}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
OUT_JSON="${OUT_DIR}/bench-${TIMESTAMP}.json"

echo "=== Running Gemma 4 26B 16GB Benchmark ==="
echo "Model: ${MODEL_PATH}"
echo "Cache: ${CACHE_MIB} MiB | Slots/Layer: ${SLOTS_PER_LAYER} | Max Tokens: ${MAX_TOKENS}"

CAMELID_GHOST_ALLOW_LEGACY_SPARSE=1 \
CAMELID_GEMMA4_GHOST_METAL_SLOTS=1 \
CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST=1 \
CAMELID_GEMMA4_GHOST_METAL_TURBO=1 \
CAMELID_GEMMA4_GHOST_METAL_COMMON=0 \
CAMELID_GEMMA4_GHOST_METAL_CONTEXT=2048 \
CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER="${SLOTS_PER_LAYER}" \
CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT=1 \
CAMELID_GEMMA4_GHOST_READ_THREADS=8 \
/Volumes/Untitled/cargo-targets/global/release/camelid ghost-run \
  "${MODEL_PATH}" \
  --cghost "${CGHOST_PATH}" \
  --expert-cache-mib "${CACHE_MIB}" \
  --prompt "${PROMPT}" \
  --max-tokens "${MAX_TOKENS}" 2>&1 | tee "${OUT_DIR}/run-${TIMESTAMP}.log"

echo "Benchmark complete. Log saved to ${OUT_DIR}/run-${TIMESTAMP}.log"
