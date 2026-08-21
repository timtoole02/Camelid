#!/usr/bin/env python3
"""Generate a target-free official Gemma 4 26B-A4B QAT MTP oracle fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import safetensors
import torch
import transformers
from transformers import Gemma4AssistantForCausalLM


MODEL_REPOSITORY = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
MODEL_REVISION = "9537141506fe8875b3ed45b264af13580cb29166"
MODEL_SHA256 = "c082cc581c3ec90d70285c1a41c81544ff56cbc96650f16c900a280940655801"
CONFIG_SHA256 = "23d2bc4a8920f24c23653ff6871437bbd95e52527bf50007aaad05b0b6cab510"
TRANSFORMERS_REVISION = "0c92811846095910816a87aca50050d10c545270"
EXPECTED_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.16.0.dev0",
    "numpy": "2.5.2",
    "safetensors": "0.8.0",
}
EXPECTED_TRANSFORMERS_SOURCE_SHA256 = {
    "models/gemma4_assistant/modeling_gemma4_assistant.py":
        "a77e67673ce7d2248294e361c3c074c29d8d6a092655d7b27b95fc9777a2d0d1",
    "models/gemma4/modeling_gemma4.py":
        "51d9c119e7baa2ef342c3d7c6377d4b175930d00834e87ba59a123e68e4ac07b",
    "generation/candidate_generator.py":
        "f31975e27814f03f927359b2a2363932693b05c8c0d04d9a282801464f075c1b",
}

KV_LEN = 1_031
# The target has processed exactly KV_LEN tokens. The assistant input token is
# the target's still-unforwarded authoritative bonus token at the next position,
# paired with the final-normalized hidden row at KV_LEN - 1. Transformers calls
# the assistant with position_ids == input_ids.shape[1] - 1 == KV_LEN while its
# shared_kv_states contain only positions [0, KV_LEN).
POSITION_ID = KV_LEN
TARGET_HIDDEN = 2_816
SLIDING_KV_HEADS = 8
SLIDING_HEAD_DIM = 256
FULL_KV_HEADS = 2
FULL_HEAD_DIM = 512

# Direct, finite BF16 bit-pattern generator. Each value has a pseudorandom sign,
# exponent in [base, base + span), and mantissa. This is independent of libm,
# RNG implementation, and float-to-BF16 rounding behavior.
INPUT_SPECS = {
    "target_scaled_embedding": ((1, 1, TARGET_HIDDEN), 0x13579BDF, 120, 3),
    "target_final_normalized_hidden": ((1, 1, TARGET_HIDDEN), 0x2468ACE1, 125, 3),
    "sliding_key_layer28": ((1, SLIDING_KV_HEADS, KV_LEN, SLIDING_HEAD_DIM), 0x31415926, 125, 3),
    "sliding_value_layer28": ((1, SLIDING_KV_HEADS, KV_LEN, SLIDING_HEAD_DIM), 0x27182818, 125, 3),
    "full_key_layer29": ((1, FULL_KV_HEADS, KV_LEN, FULL_HEAD_DIM), 0x16180339, 125, 3),
    "full_value_layer29": ((1, FULL_KV_HEADS, KV_LEN, FULL_HEAD_DIM), 0x57721566, 125, 3),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_bf16(shape: tuple[int, ...], seed: int, exponent_base: int, exponent_span: int) -> torch.Tensor:
    count = math.prod(shape)
    index = np.arange(count, dtype=np.uint32)
    state = index + np.uint32(seed)
    state ^= state >> np.uint32(16)
    state *= np.uint32(0x7FEB352D)
    state ^= state >> np.uint32(15)
    state *= np.uint32(0x846CA68B)
    state ^= state >> np.uint32(16)
    sign = ((state >> np.uint32(31)) & np.uint32(1)) << np.uint32(15)
    exponent = (
        np.uint32(exponent_base)
        + ((state >> np.uint32(24)) % np.uint32(exponent_span))
    ) << np.uint32(7)
    mantissa = (state >> np.uint32(16)) & np.uint32(0x7F)
    bits = (sign | exponent | mantissa).astype(np.uint16, copy=False).reshape(shape)
    return torch.from_numpy(bits).view(torch.bfloat16)


def bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    assert tensor.device.type == "cpu" and tensor.dtype == torch.bfloat16
    return tensor.detach().contiguous().view(torch.uint16).numpy().astype("<u2", copy=False)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(bf16_bits(tensor).tobytes(order="C")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    require(platform.machine() == "arm64", f"expected arm64, got {platform.machine()}")
    require(torch.__version__ == EXPECTED_VERSIONS["torch"], f"torch version changed: {torch.__version__}")
    require(transformers.__version__ == EXPECTED_VERSIONS["transformers"], f"transformers version changed: {transformers.__version__}")
    require(np.__version__ == EXPECTED_VERSIONS["numpy"], f"numpy version changed: {np.__version__}")
    require(safetensors.__version__ == EXPECTED_VERSIONS["safetensors"], f"safetensors version changed: {safetensors.__version__}")

    tf_root = Path(transformers.__file__).resolve().parent
    source_hashes = {
        relative: sha256_file(tf_root / relative)
        for relative in EXPECTED_TRANSFORMERS_SOURCE_SHA256
    }
    require(source_hashes == EXPECTED_TRANSFORMERS_SOURCE_SHA256, "pinned Transformers source hashes changed")

    model_path = args.model_dir / "model.safetensors"
    config_path = args.model_dir / "config.json"
    require(sha256_file(model_path) == MODEL_SHA256, "assistant safetensors hash mismatch")
    require(sha256_file(config_path) == CONFIG_SHA256, "assistant config hash mismatch")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    load_started = time.perf_counter()
    model = Gemma4AssistantForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    cfg = model.config
    text_cfg = cfg.text_config
    require(cfg.backbone_hidden_size == TARGET_HIDDEN, "assistant backbone width mismatch")
    require(text_cfg.hidden_size == 1_024, "assistant hidden width mismatch")
    require(text_cfg.layer_types == ["sliding_attention"] * 3 + ["full_attention"], "assistant layer schedule mismatch")
    require(text_cfg.num_kv_shared_layers == 4, "assistant must share KV in all four layers")
    require(text_cfg.sliding_window == 1_024, "assistant sliding window mismatch")
    per_layer = list(text_cfg.per_layer_config)
    require(all(layer.num_key_value_heads == SLIDING_KV_HEADS for layer in per_layer[:3]), "sliding KV head count mismatch")
    require(all(layer.head_dim == SLIDING_HEAD_DIM for layer in per_layer[:3]), "sliding head width mismatch")
    require(per_layer[3].num_key_value_heads == FULL_KV_HEADS, "full KV head count mismatch")
    require(per_layer[3].head_dim == FULL_HEAD_DIM, "full head width mismatch")
    require(all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()), "non-BF16 model parameter")

    generated = {
        name: deterministic_bf16(*spec)
        for name, spec in INPUT_SPECS.items()
    }
    target_embedding = generated["target_scaled_embedding"]
    target_hidden = generated["target_final_normalized_hidden"]
    inputs_embeds = torch.cat((target_embedding, target_hidden), dim=-1)
    shared_kv_states = {
        "sliding_attention": (
            generated["sliding_key_layer28"],
            generated["sliding_value_layer28"],
        ),
        "full_attention": (
            generated["full_key_layer29"],
            generated["full_value_layer29"],
        ),
    }
    attention_mask = torch.ones((1, KV_LEN), dtype=torch.long)
    position_ids = torch.tensor([[POSITION_ID]], dtype=torch.long)

    forward_started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            use_cache=False,
        )
    forward_seconds = time.perf_counter() - forward_started

    logits = output.logits[0, 0].cpu().contiguous()
    recurrent_hidden = output.last_hidden_state[0, 0].cpu().contiguous()
    require(logits.dtype == torch.bfloat16 and tuple(logits.shape) == (262_144,), "unexpected logits")
    require(recurrent_hidden.dtype == torch.bfloat16 and tuple(recurrent_hidden.shape) == (TARGET_HIDDEN,), "unexpected recurrent hidden")

    top_values, top_tokens = torch.topk(logits.float(), k=16, largest=True, sorted=True)
    top_tokens_np = top_tokens.numpy().astype("<u4", copy=False)
    top_bits_np = bf16_bits(logits[top_tokens]).reshape(-1)
    top1 = int(torch.argmax(logits).item())
    require(top1 == int(top_tokens_np[0]), "argmax/topk disagreement")

    input_names = list(INPUT_SPECS)
    input_hashes = {name: tensor_sha256(generated[name]) for name in input_names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.array([1], dtype="<u4"),
        kv_len=np.array([KV_LEN], dtype="<u4"),
        position_id=np.array([POSITION_ID], dtype="<u4"),
        target_scaled_embedding_bf16_le=bf16_bits(target_embedding).reshape(-1),
        target_final_normalized_hidden_bf16_le=bf16_bits(target_hidden).reshape(-1),
        logits_bf16_le=bf16_bits(logits).reshape(-1),
        recurrent_hidden_bf16_le=bf16_bits(recurrent_hidden).reshape(-1),
        top16_token_ids=top_tokens_np,
        top16_logits_bf16_le=top_bits_np,
        top16_logits_f32=top_values.numpy().astype("<f4", copy=False),
        input_names=np.asarray(input_names, dtype="S40"),
        input_sha256=np.asarray([input_hashes[name] for name in input_names], dtype="S64"),
    )

    report = {
        "schema_version": 1,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "safetensors_sha256": MODEL_SHA256,
            "config_sha256": CONFIG_SHA256,
            "parameter_bytes": model.get_memory_footprint(return_buffers=True),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "transformers_revision": TRANSFORMERS_REVISION,
            "numpy": np.__version__,
            "safetensors": safetensors.__version__,
            "attention_implementation": "eager",
            "device": "cpu",
            "threads": 1,
        },
        "input": {
            "kv_len": KV_LEN,
            "position_id": POSITION_ID,
            "sliding_window": text_cfg.sliding_window,
            "tensor_sha256": input_hashes,
        },
        "output": {
            "top1_token_id": top1,
            "top16_token_ids": top_tokens_np.tolist(),
            "top16_logits_bf16_bits": [int(value) for value in top_bits_np],
            "logits_sha256": tensor_sha256(logits),
            "recurrent_hidden_sha256": tensor_sha256(recurrent_hidden),
            "fixture_sha256": sha256_file(args.output),
        },
        "timing": {
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "transformers_source_sha256": source_hashes,
    }
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
