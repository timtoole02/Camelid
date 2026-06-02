#!/usr/bin/env python3
"""Aggregate raw per-iteration JSONL into summary stats + a markdown table.

Reads:  <bundle>/raw/{camelid,mlx-lm}/<label>.jsonl  (schema from bench-generate)
        <bundle>/raw/llamacpp/<label>.json          (llama-bench -o json, reference)
Writes: <bundle>/summaries/results.json
        <bundle>/summaries/results.md
"""
import glob
import json
import os
import statistics
import sys


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stat_block(values):
    if not values:
        return {"median": 0, "min": 0, "max": 0, "p95": 0, "n": 0}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": pct(values, 95),
        "n": len(values),
    }


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_runtime(bundle, runtime):
    out = {}
    for path in sorted(glob.glob(os.path.join(bundle, "raw", runtime, "*.jsonl"))):
        label = os.path.basename(path)[: -len(".jsonl")]
        rows = load_jsonl(path)
        if not rows:
            continue
        out[label] = {
            "prompt_tokens": rows[-1].get("prompt_tokens", 0),
            "generated_tokens": rows[-1].get("generated_tokens", 0),
            "load_ms": rows[0].get("load_ms", 0),
            "ttft_ms": stat_block([r["ttft_ms"] for r in rows]),
            "decode_ms": stat_block([r["decode_ms"] for r in rows]),
            "tokens_per_second": stat_block([r["tokens_per_second"] for r in rows]),
            "peak_memory_bytes": stat_block([r["peak_memory_bytes"] for r in rows]),
            "output_text": rows[-1].get("output_text", "")[:240],
        }
    return out


def parse_llamacpp(bundle):
    out = {}
    for path in sorted(glob.glob(os.path.join(bundle, "raw", "llamacpp", "*.json"))):
        label = os.path.basename(path)[: -len(".json")]
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        pp = tg = None
        for entry in data:
            t = entry.get("n_prompt", 0)
            if t and not entry.get("n_gen", 0):
                pp = entry.get("avg_ts")
            elif entry.get("n_gen", 0) and not entry.get("n_prompt", 0):
                tg = entry.get("avg_ts")
        out[label] = {"prefill_tps": pp, "decode_tps": tg}
    return out


def fmt_gb(b):
    return f"{b / 1.073741824e9:.2f}"


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "."
    summaries = os.path.join(bundle, "summaries")
    os.makedirs(summaries, exist_ok=True)

    result = {
        "camelid": summarize_runtime(bundle, "camelid"),
        "mlx-lm": summarize_runtime(bundle, "mlx-lm"),
        "llamacpp": parse_llamacpp(bundle),
    }
    with open(os.path.join(summaries, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    labels = sorted(
        set(result["camelid"]) | set(result["mlx-lm"]),
        key=lambda x: {"prompt-128": 0, "prompt-512": 1, "prompt-2k": 2, "prompt-8k": 3}.get(x, 9),
    )
    lines = []
    lines.append("| Runtime | Prompt | Prompt tok | Gen tok | TTFT ms (median) | Decode tok/s (median) | Decode tok/s (min–max) | Peak mem GB | Load ms |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label in labels:
        for runtime in ("camelid", "mlx-lm"):
            r = result[runtime].get(label)
            if not r:
                continue
            lines.append(
                f"| {runtime} | {label} | {r['prompt_tokens']} | {r['generated_tokens']} "
                f"| {r['ttft_ms']['median']:.1f} | {r['tokens_per_second']['median']:.2f} "
                f"| {r['tokens_per_second']['min']:.2f}–{r['tokens_per_second']['max']:.2f} "
                f"| {fmt_gb(r['peak_memory_bytes']['median'])} | {r['load_ms']:.0f} |"
            )
        ref = result["llamacpp"].get(label)
        if ref and ref.get("decode_tps"):
            lines.append(
                f"| llama.cpp (ref) | {label} | — | — | — | {ref['decode_tps']:.2f} | — | — | — |"
            )

    with open(os.path.join(summaries, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {summaries}/results.json and results.md")


if __name__ == "__main__":
    main()
