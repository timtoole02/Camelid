#!/usr/bin/env python3
"""MLX-LM generation benchmark emitting the same JSONL schema as `camelid bench-generate`.

One JSON object per measured iteration on stdout; a human summary on stderr.
Greedy (temperature 0) by default for deterministic, comparable output.
"""
import argparse
import json
import sys
import time

import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate

try:
    from mlx_lm.sample_utils import make_sampler
except Exception:  # pragma: no cover - older mlx_lm
    make_sampler = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file")
    ap.add_argument("--prompt")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()

    if args.prompt_file:
        prompt_text = open(args.prompt_file, encoding="utf-8").read()
    elif args.prompt:
        prompt_text = args.prompt
    else:
        print("need --prompt-file or --prompt", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_ms = (time.perf_counter() - t0) * 1000.0

    if "8bit" in args.model:
        quant = "8bit"
    elif "4bit" in args.model:
        quant = "4bit"
    else:
        quant = "unknown"
    commit = getattr(mlx_lm, "__version__", "unknown")

    sampler = None
    if make_sampler is not None:
        try:
            sampler = make_sampler(temp=args.temperature)
        except Exception:
            sampler = None

    def run_once():
        mx.reset_peak_memory()
        gen_start = time.perf_counter()
        first_t = None
        prompt_tokens = 0
        gen_tokens = 0
        prompt_tps = 0.0
        chunks = []
        kwargs = {"max_tokens": args.max_tokens}
        if sampler is not None:
            kwargs["sampler"] = sampler
        for resp in stream_generate(model, tokenizer, prompt_text, **kwargs):
            if first_t is None:
                first_t = time.perf_counter()
            chunks.append(resp.text)
            prompt_tokens = getattr(resp, "prompt_tokens", prompt_tokens) or prompt_tokens
            gen_tokens = getattr(resp, "generation_tokens", gen_tokens) or gen_tokens
            prompt_tps = getattr(resp, "prompt_tps", prompt_tps) or prompt_tps
        end = time.perf_counter()
        ttft_ms = (first_t - gen_start) * 1000.0 if first_t else 0.0
        decode_ms = (end - first_t) * 1000.0 if first_t else 0.0
        if not gen_tokens:
            gen_tokens = max(len(chunks), 1)
        prefill_ms = (prompt_tokens / prompt_tps * 1000.0) if prompt_tps > 0 else ttft_ms
        peak = int(mx.get_peak_memory())
        decode_tokens = max(gen_tokens - 1, 0)
        tps = decode_tokens / (decode_ms / 1000.0) if decode_ms > 0 and decode_tokens > 0 else 0.0
        return {
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(gen_tokens),
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "tokens_per_second": tps,
            "peak_memory_bytes": peak,
            "output_text": "".join(chunks),
        }

    if args.warmup:
        print("[mlx-lm] warmup iteration (unmeasured)...", file=sys.stderr)
        run_once()

    for i in range(args.iterations):
        r = run_once()
        rec = {
            "runtime": "mlx-lm",
            "commit": commit,
            "model": args.model,
            "quantization": quant,
            "iteration": i,
            "load_ms": load_ms,
            **r,
        }
        print(json.dumps(rec))
        sys.stdout.flush()
        print(
            f"[mlx-lm] iter {i} | prompt {r['prompt_tokens']} tok | gen {r['generated_tokens']} tok "
            f"| ttft {r['ttft_ms']:.1f} ms | decode {r['decode_ms']:.1f} ms "
            f"| {r['tokens_per_second']:.2f} tok/s | peak {r['peak_memory_bytes'] / 1.073741824e9:.2f} GB",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
