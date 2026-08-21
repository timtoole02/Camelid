#!/usr/bin/env python3
"""Generate the schema-v2 target-free seven-proposal recurrence oracle.

The schema-v1 single-forward fixture remains the canonical arithmetic anchor.
This sibling adds the maximum K=8 assistant recurrence depth without loading a
target model.  After the canonical first input, a deterministic token-keyed
synthetic 2816-wide embedding stands in for the target embedding table; each
proposal consumes its own preceding recurrent hidden state.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
from pathlib import Path

import numpy as np
import safetensors
import torch
import transformers
from transformers import Gemma4AssistantForCausalLM

from generate_oracle import (
    CONFIG_SHA256,
    EXPECTED_TRANSFORMERS_SOURCE_SHA256,
    EXPECTED_VERSIONS,
    FULL_HEAD_DIM,
    FULL_KV_HEADS,
    INPUT_SPECS,
    KV_LEN,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SHA256,
    POSITION_ID,
    SLIDING_HEAD_DIM,
    SLIDING_KV_HEADS,
    TARGET_HIDDEN,
    TRANSFORMERS_REVISION,
    bf16_bits,
    deterministic_bf16,
    require,
    sha256_file,
    tensor_sha256,
)


SCHEMA_VERSION = 2
PROPOSAL_COUNT = 7
STOP_TOKEN_IDS = (1, 106)
FEEDBACK_SEED_XOR = 0x6A09_E667
FEEDBACK_TOKEN_MULTIPLIER = 0x9E37_79B9
FEEDBACK_EXPONENT_BASE = 120
FEEDBACK_EXPONENT_SPAN = 3
MIN_REFERENCE_MARGIN_BF16_ULP = 0
MIN_NATIVE_TOP16_OVERLAP = 15
NATIVE_MARGIN_CAP_BF16_ULP = 2
MIN_RECURRENT_COSINE = 0.99995
MAX_RECURRENT_RELATIVE_L2 = 0.01
RSS_LIMIT_BYTES = 3_000_000_000


def token_feedback_seed(token_id: int) -> int:
    require(0 <= token_id < 262_144, f"feedback token {token_id} is outside the vocabulary")
    product = (token_id * FEEDBACK_TOKEN_MULTIPLIER) & 0xFFFF_FFFF
    return (FEEDBACK_SEED_XOR ^ product) & 0xFFFF_FFFF


def token_feedback_embedding(token_id: int) -> torch.Tensor:
    return deterministic_bf16(
        (1, 1, TARGET_HIDDEN),
        token_feedback_seed(token_id),
        FEEDBACK_EXPONENT_BASE,
        FEEDBACK_EXPONENT_SPAN,
    )


def ordered_bf16(bits: int) -> int:
    return ((~bits) & 0xFFFF) if bits & 0x8000 else bits | 0x8000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    require(platform.machine() == "arm64", f"expected arm64, got {platform.machine()}")
    require(torch.__version__ == EXPECTED_VERSIONS["torch"], f"torch version changed: {torch.__version__}")
    require(
        transformers.__version__ == EXPECTED_VERSIONS["transformers"],
        f"transformers version changed: {transformers.__version__}",
    )
    require(np.__version__ == EXPECTED_VERSIONS["numpy"], f"numpy version changed: {np.__version__}")
    require(
        safetensors.__version__ == EXPECTED_VERSIONS["safetensors"],
        f"safetensors version changed: {safetensors.__version__}",
    )

    tf_root = Path(transformers.__file__).resolve().parent
    source_hashes = {
        relative: sha256_file(tf_root / relative)
        for relative in EXPECTED_TRANSFORMERS_SOURCE_SHA256
    }
    require(source_hashes == EXPECTED_TRANSFORMERS_SOURCE_SHA256, "pinned Transformers source hashes changed")
    require(sha256_file(args.model_dir / "model.safetensors") == MODEL_SHA256, "assistant weight hash mismatch")
    require(sha256_file(args.model_dir / "config.json") == CONFIG_SHA256, "assistant config hash mismatch")

    canonical = json.loads(args.canonical_manifest.read_text())
    require(canonical["schema_version"] == 1, "canonical oracle schema changed")
    require(canonical["scope"]["target_model_loaded"] is False, "canonical oracle loaded a target")
    require(canonical["scope"]["kv_len"] == KV_LEN, "canonical KV length changed")
    require(canonical["scope"]["position_id"] == POSITION_ID, "canonical position changed")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    model = Gemma4AssistantForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    require(all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()), "non-BF16 parameter")

    cfg = model.config
    text_cfg = cfg.text_config
    require(cfg.backbone_hidden_size == TARGET_HIDDEN, "assistant backbone width mismatch")
    require(text_cfg.hidden_size == 1_024, "assistant hidden width mismatch")
    require(text_cfg.num_kv_shared_layers == 4, "assistant shared-KV count mismatch")
    require(text_cfg.layer_types == ["sliding_attention"] * 3 + ["full_attention"], "layer schedule mismatch")
    require(text_cfg.sliding_window == 1_024, "sliding window mismatch")

    generated = {name: deterministic_bf16(*spec) for name, spec in INPUT_SPECS.items()}
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
    embedding = generated["target_scaled_embedding"]
    recurrent = generated["target_final_normalized_hidden"]
    steps: list[dict[str, object]] = []

    with torch.inference_mode():
        for step_index in range(PROPOSAL_COUNT):
            input_embedding_hash = tensor_sha256(embedding)
            recurrent_input_hash = tensor_sha256(recurrent)
            output = model(
                inputs_embeds=torch.cat((embedding, recurrent), dim=-1),
                position_ids=position_ids,
                attention_mask=attention_mask,
                shared_kv_states=shared_kv_states,
                use_cache=False,
            )
            logits = output.logits[0, 0].cpu().contiguous()
            next_recurrent = output.last_hidden_state[0, 0].cpu().contiguous()
            require(logits.dtype == torch.bfloat16 and tuple(logits.shape) == (262_144,), "unexpected logits")
            require(
                next_recurrent.dtype == torch.bfloat16
                and tuple(next_recurrent.shape) == (TARGET_HIDDEN,),
                "unexpected recurrent hidden",
            )

            top_values, top_tokens = torch.topk(logits.float(), k=16, largest=True, sorted=True)
            top_tokens_list = [int(token) for token in top_tokens.tolist()]
            top_bits = [int(value) for value in bf16_bits(logits[top_tokens]).reshape(-1)]
            top1 = int(torch.argmax(logits).item())
            require(top1 == top_tokens_list[0], "argmax/top-k disagreement")
            margin_ulp = ordered_bf16(top_bits[0]) - ordered_bf16(top_bits[1])
            require(margin_ulp >= MIN_REFERENCE_MARGIN_BF16_ULP, f"step {step_index} top-1 margin is only {margin_ulp} BF16 ULP")
            if margin_ulp == 0:
                tied_ids = torch.nonzero(logits == logits[top1], as_tuple=False).reshape(-1)
                require(top1 == int(tied_ids.min().item()), "PyTorch argmax tie policy is not lowest token ID")
            if step_index + 1 < PROPOSAL_COUNT:
                require(top1 not in STOP_TOKEN_IDS, f"step {step_index} stopped before seven proposals: {top1}")

            step = {
                "index": step_index,
                "input_embedding_bf16_sha256": input_embedding_hash,
                "recurrent_input_bf16_sha256": recurrent_input_hash,
                "top1_token_id": top1,
                "top16_token_ids": top_tokens_list,
                "top16_logits_bf16_bits": top_bits,
                "top16_logits_f32": [float(value) for value in top_values.tolist()],
                "top1_margin_bf16_ulp": margin_ulp,
                "logits_bf16_sha256": tensor_sha256(logits),
                "recurrent_hidden_bf16_sha256": tensor_sha256(next_recurrent),
                "recurrent_hidden_bf16_bits": [
                    int(value) for value in bf16_bits(next_recurrent).reshape(-1)
                ],
            }
            steps.append(step)

            if step_index == 0:
                expected = canonical["oracle_output"]
                require(top1 == expected["top1_token_id"], "schema-v2 step 0 top-1 changed")
                require(top_tokens_list == expected["top16_token_ids"], "schema-v2 step 0 top-16 changed")
                require(top_bits == expected["top16_logits_bf16_bits"], "schema-v2 step 0 top-16 logits changed")
                require(step["logits_bf16_sha256"] == expected["logits_bf16_sha256"], "schema-v2 step 0 logits changed")
                require(
                    step["recurrent_hidden_bf16_sha256"]
                    == expected["recurrent_hidden_bf16_sha256"],
                    "schema-v2 step 0 recurrent hidden changed",
                )

            recurrent = next_recurrent.reshape(1, 1, TARGET_HIDDEN)
            if step_index + 1 < PROPOSAL_COUNT:
                embedding = token_feedback_embedding(top1)

    for index, step in enumerate(steps):
        if index == 0:
            expected_input = canonical["input_tensor_sha256"]["target_final_normalized_hidden"]
        else:
            expected_input = steps[index - 1]["recurrent_hidden_bf16_sha256"]
        require(
            step["recurrent_input_bf16_sha256"] == expected_input,
            f"recurrent chain broke before step {index}",
        )

    max_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(max_rss_bytes <= RSS_LIMIT_BYTES, f"maximum RSS {max_rss_bytes} exceeds {RSS_LIMIT_BYTES}")
    document = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Target-free closed seven-proposal recurrence/top-k admission oracle",
        "authority": {
            "target_model_loaded": False,
            "assistant_repository": MODEL_REPOSITORY,
            "assistant_revision": MODEL_REVISION,
            "assistant_model_sha256": MODEL_SHA256,
            "assistant_config_sha256": CONFIG_SHA256,
            "transformers_revision": TRANSFORMERS_REVISION,
            "transformers_source_sha256": source_hashes,
            "canonical_manifest_sha256": sha256_file(args.canonical_manifest),
            "generator_sha256": sha256_file(Path(__file__)),
            "kv_len": KV_LEN,
            "position_id": POSITION_ID,
            "proposal_count": PROPOSAL_COUNT,
            "stop_token_ids": list(STOP_TOKEN_IDS),
            "argmax_tie_policy": "lowest_token_id",
            "minimum_reference_margin_bf16_ulp": MIN_REFERENCE_MARGIN_BF16_ULP,
        },
        "inputs": {
            "tensor_sha256": canonical["input_tensor_sha256"],
            "sliding_geometry": [SLIDING_KV_HEADS, KV_LEN, SLIDING_HEAD_DIM],
            "full_geometry": [FULL_KV_HEADS, KV_LEN, FULL_HEAD_DIM],
            "shared_kv_reused_for_every_step": True,
            "position_reused_for_every_step": True,
        },
        "feedback_embedding": {
            "rule": "existing deterministic_bf16 keyed only by preceding top1 token",
            "seed_formula": "(0x6A09E667 ^ ((token_id * 0x9E3779B9) mod 2^32)) mod 2^32",
            "seed_xor": FEEDBACK_SEED_XOR,
            "token_multiplier": FEEDBACK_TOKEN_MULTIPLIER,
            "shape": [1, 1, TARGET_HIDDEN],
            "exponent_base": FEEDBACK_EXPONENT_BASE,
            "exponent_span": FEEDBACK_EXPONENT_SPAN,
        },
        "admission": {
            "required_fresh_python_runs": 2,
            "required_native_repetitions": 2,
            "required_exact_top1_steps": PROPOSAL_COUNT,
            "minimum_top16_set_overlap_per_step": MIN_NATIVE_TOP16_OVERLAP,
            "native_margin_floor_rule": "min(reference_top1_margin_bf16_ulp, native_margin_cap_bf16_ulp)",
            "native_margin_cap_bf16_ulp": NATIVE_MARGIN_CAP_BF16_ULP,
            "minimum_recurrent_cosine_per_step": MIN_RECURRENT_COSINE,
            "maximum_recurrent_relative_l2_per_step": MAX_RECURRENT_RELATIVE_L2,
            "stop_on_first_top1_mismatch": True,
            "teacher_force_after_mismatch": False,
            "require_native_bf16_lattice": True,
            "require_native_repeat_bit_determinism": True,
        },
        "steps": steps,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "safetensors": safetensors.__version__,
            "device": "cpu",
            "attention_implementation": "eager",
            "threads": 1,
            "deterministic_algorithms": True,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(encoded)
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "proposal_count": len(steps),
        "top1_token_ids": [step["top1_token_id"] for step in steps],
        "max_rss_bytes": max_rss_bytes,
    }, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
