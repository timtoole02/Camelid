#!/usr/bin/env bash
# Orchestrate the Camelid-vs-MLX (+llama.cpp reference) benchmark across prompt sizes.
#
# Required env:
#   MODEL      path to the GGUF model for Camelid + llama.cpp
#   MLX_VENV   path to the mlx-lm virtualenv
# Optional env:
#   MLX_MODEL  default mlx-community/Llama-3.2-3B-Instruct-8bit
#   HF_HOME    Hugging Face cache dir for MLX
#   ITERS      measured iterations per lane (default 10)
#   MAX_TOKENS generated tokens (default 128)
#   PROMPTS    space list of prompt labels (default "128 512 2k 8k")
#   RUN_LLAMACPP 1 to include llama.cpp reference (default 1)
#   CAMELID_BIN path to the camelid release binary
#   OUT        output bundle dir (default qa/evidence-bundles/mlx-compare-<ts>)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

MODEL="${MODEL:?set MODEL to the gguf path}"
MLX_VENV="${MLX_VENV:?set MLX_VENV to the mlx venv dir}"
MLX_MODEL="${MLX_MODEL:-mlx-community/Llama-3.2-3B-Instruct-8bit}"
ITERS="${ITERS:-10}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPTS="${PROMPTS:-128 512 2k 8k}"
RUN_LLAMACPP="${RUN_LLAMACPP:-1}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO_ROOT/qa/evidence-bundles/mlx-compare-$TS}"

mkdir -p "$OUT"
bash "$HERE/lib/capture-env.sh" "$OUT"
python3 "$HERE/lib/gen_prompts.py" "$HERE/prompts"
rm -rf "$OUT/prompts"; cp -r "$HERE/prompts" "$OUT/prompts"
{
  echo "model=$MODEL"
  echo "mlx_model=$MLX_MODEL"
  echo "iters=$ITERS  max_tokens=$MAX_TOKENS  prompts=$PROMPTS"
} > "$OUT/commands.txt"

for p in $PROMPTS; do
  pf="$HERE/prompts/prompt-$p.txt"
  [ -f "$pf" ] || { echo "missing $pf" >&2; continue; }
  MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" ITERS="$ITERS" OUT="$OUT" \
    CAMELID_BIN="${CAMELID_BIN:-}" bash "$HERE/run-camelid.sh" "$pf" || true
  MLX_MODEL="$MLX_MODEL" MLX_VENV="$MLX_VENV" HF_HOME="${HF_HOME:-}" \
    MAX_TOKENS="$MAX_TOKENS" ITERS="$ITERS" OUT="$OUT" bash "$HERE/run-mlx-lm.sh" "$pf" || true
  if [ "$RUN_LLAMACPP" = "1" ]; then
    MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" OUT="$OUT" bash "$HERE/run-llamacpp.sh" "$pf" || true
  fi
done

python3 "$HERE/lib/aggregate.py" "$OUT"
echo "==> evidence bundle: $OUT"
