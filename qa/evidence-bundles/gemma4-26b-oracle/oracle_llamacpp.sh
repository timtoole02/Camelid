#!/bin/bash
# llama.cpp oracle for Gemma 4 26B-A4B greedy decode.
#
# MUST use the FULL GGUF on the T7. The copy under ~/models is the sparse hot
# shadow (1.5 GB physical of 13.4 GB logical; routed-expert ranges are holes),
# and llama.cpp would read zeros for every expert and emit gibberish.
set -uo pipefail

MODEL=${ORACLE_GGUF:-/Volumes/Untitled/models/gemma-4-26B_q4_0-it.gguf}
PORT=${ORACLE_PORT:-8899}
NPRED=${ORACLE_NPRED:-48}
OUT=${ORACLE_OUT:-/private/tmp/claude-501/-Users-timtoole/ad2b0c33-a5ea-46b6-8cf5-6cc063c5ad36/scratchpad/oracle_out.json}

phys=$(( $(stat -f %b "$MODEL") * 512 ))
log=$(stat -f %z "$MODEL")
if [ "$phys" -lt $(( log / 2 )) ]; then
  echo "REFUSING: $MODEL is sparse (${phys} of ${log} bytes resident) — that is the hot shadow, not an oracle." >&2
  exit 2
fi
echo "[oracle] model=$MODEL ($(echo "scale=2;$log/1073741824"|bc) GB, fully resident)"

pkill -f "llama-server.*$PORT" 2>/dev/null
sleep 1
# -ngl 0: keep the whole graph on the CPU. Full Metal offload of 13.4 GB of
# weights plus KV and compute buffers does not fit in 16 GB of unified memory
# (kIOGPUCommandBufferCallbackErrorOutOfMemory), and the CPU path is the better
# reference anyway: it removes llama.cpp's own Metal kernels from the
# comparison, so we arbitrate against the reference implementation rather than
# against a second GPU port. Small batch keeps the scratch buffers modest.
llama-server -m "$MODEL" -c 512 -b 256 -ub 256 -ngl 0 -t 8 \
  --host 127.0.0.1 --port "$PORT" \
  --no-warmup > "${OUT%.json}.server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

echo "[oracle] waiting for llama-server (pid $SRV) to load ~13.4 GB from the T7 (CPU graph)..."
for i in $(seq 1 300); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "[oracle] ready after ${i}s"; break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "[oracle] server died during load; tail of log:" >&2
    tail -25 "${OUT%.json}.server.log" >&2
    exit 3
  fi
  sleep 1
done
if ! curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
  echo "[oracle] server never became healthy" >&2; tail -25 "${OUT%.json}.server.log" >&2; exit 4
fi

python3 - "$PORT" "$NPRED" "$OUT" <<'PY'
import json, sys, urllib.request

port, npred, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base = f"http://127.0.0.1:{port}"

PROMPTS = {
 "greeting": "<|turn>user\nSay hello and name three colours.\n<turn|>\n<|turn>model\n",
 "json-yaml": "<|turn>user\nConvert this configuration payload to YAML:\n{\"cluster_id\": \"prod-1\", \"min_replicas\": 4}\n<turn|>\n<|turn>model\n",
 "code-edit": "<|turn>user\nAdd a `pub expires_at: u64,` field at the end of this struct and output the COMPLETE struct definition again, unchanged otherwise, with no explanation:\n\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n}\n<turn|>\n<|turn>model\n",
}

def post(path, payload):
    req = urllib.request.Request(base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise SystemExit(f"[oracle] {path} failed: HTTP {e.code}: {body}")

results = {}
for name, prompt in PROMPTS.items():
    tok = post("/tokenize", {"content": prompt, "add_special": True})
    prompt_ids = tok.get("tokens", [])
    # Greedy: temperature 0 with top_k 1 pins the argmax on every llama.cpp build.
    comp = post("/completion", {
        "prompt": prompt, "n_predict": npred,
        "temperature": 0.0, "top_k": 1, "seed": 0,
        "cache_prompt": False, "return_tokens": True,
    })
    gen_ids = comp.get("tokens") or []
    if not gen_ids:
        # Older builds omit return_tokens; recover ids by tokenizing the output
        # WITHOUT specials so the comparison stays id-level.
        t2 = post("/tokenize", {"content": comp.get("content", ""), "add_special": False})
        gen_ids = t2.get("tokens", [])
    results[name] = {"prompt_ids": prompt_ids, "gen_ids": gen_ids,
                     "text": comp.get("content", "")}
    print(f"\n=== ORACLE {name} ===")
    print(f"[oracle prompt ids] {prompt_ids}")
    print(f"[oracle gen ids   ] {gen_ids}")
    print(f"[oracle text      ] {results[name]['text']!r}")

with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\n[oracle] wrote {out}")
PY
