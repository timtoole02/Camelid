#!/usr/bin/env python3
"""Generate deterministic prompts targeting approximate token counts.

Exact token counts differ per tokenizer, so these target *approximate* sizes;
each runtime reports the actual prompt_tokens it tokenized. The text is a fixed
technical paragraph repeated to length so runs are reproducible.
"""
import os
import sys

BASE = (
    "The unified memory architecture on Apple Silicon lets the CPU and GPU share "
    "the same physical memory without explicit copies, which changes how a local "
    "inference runtime should schedule quantized matrix multiplications during "
    "decode. A memory-bandwidth-bound decoder reads every weight once per token, "
    "so the dominant cost is where the weights live rather than raw arithmetic "
    "throughput. Keeping eight-bit weights resident avoids streaming them from "
    "disk, and a clone-free projection path avoids copying multi-megabyte tensors "
    "on every step. Carefully measure prefill separately from decode, because "
    "prefill is compute-bound while decode is bandwidth-bound. "
)

# rough words-per-token ~ 0.75, so tokens ~ words / 0.75 = words * 1.33
TARGETS = {"128": 128, "512": 512, "2k": 2048, "8k": 8192}


def build(target_tokens: int) -> str:
    target_words = int(target_tokens * 0.75)
    words = []
    base_words = BASE.split()
    while len(words) < target_words:
        words.extend(base_words)
    text = " ".join(words[:target_words])
    return text.strip() + "\n"


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "prompts"
    )
    os.makedirs(out_dir, exist_ok=True)
    for label, toks in TARGETS.items():
        path = os.path.join(out_dir, f"prompt-{label}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build(toks))
        print(f"wrote {path} (~{toks} tokens target, {os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
