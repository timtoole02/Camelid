#!/usr/bin/env python3
"""Generate bounded BF16 checkpoints for the native Metal MTP diagnosis.

This is deliberately separate from the canonical oracle generator: it does
not rewrite the pinned fixture or manifest.  The output is an ephemeral JSON
file consumed only when CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON is set on the
ignored native test.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import OrderedDict
from pathlib import Path

import numpy as np
import safetensors
import torch
import transformers
from transformers import Gemma4AssistantForCausalLM
from transformers.models.gemma4 import modeling_gemma4 as gemma4_modeling

from generate_oracle import (
    CONFIG_SHA256,
    EXPECTED_TRANSFORMERS_SOURCE_SHA256,
    EXPECTED_VERSIONS,
    FULL_HEAD_DIM,
    FULL_KV_HEADS,
    INPUT_SPECS,
    KV_LEN,
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
from generate_recurrence_oracle import (
    SCHEMA_VERSION as RECURRENCE_SCHEMA_VERSION,
    token_feedback_embedding,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--recurrence-oracle", type=Path)
    parser.add_argument("--recurrence-step", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.recurrence_step < 0:
        parser.error("--recurrence-step must be non-negative")
    if args.recurrence_step != 0 and args.recurrence_oracle is None:
        parser.error("--recurrence-oracle is required when --recurrence-step is nonzero")

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
    require(canonical["scope"]["target_model_loaded"] is False, "canonical oracle unexpectedly loaded target")
    require(canonical["scope"]["kv_len"] == KV_LEN, "canonical KV length changed")
    require(canonical["scope"]["position_id"] == POSITION_ID, "canonical position changed")
    canonical_manifest_sha256 = sha256_file(args.canonical_manifest)

    recurrence = None
    selected_recurrence = None
    if args.recurrence_oracle is not None:
        recurrence = json.loads(args.recurrence_oracle.read_text())
        require(recurrence["schema_version"] == RECURRENCE_SCHEMA_VERSION, "recurrence oracle schema changed")
        require(recurrence["authority"]["target_model_loaded"] is False, "recurrence oracle loaded a target")
        require(
            recurrence["authority"]["assistant_model_sha256"] == MODEL_SHA256,
            "recurrence assistant weight hash changed",
        )
        require(
            recurrence["authority"]["assistant_config_sha256"] == CONFIG_SHA256,
            "recurrence assistant config hash changed",
        )
        require(
            recurrence["authority"]["transformers_revision"] == TRANSFORMERS_REVISION,
            "recurrence Transformers revision changed",
        )
        require(
            recurrence["authority"]["transformers_source_sha256"] == source_hashes,
            "recurrence Transformers source hashes changed",
        )
        require(
            recurrence["authority"]["canonical_manifest_sha256"] == canonical_manifest_sha256,
            "recurrence canonical manifest hash changed",
        )
        require(recurrence["authority"]["kv_len"] == KV_LEN, "recurrence KV length changed")
        require(recurrence["authority"]["position_id"] == POSITION_ID, "recurrence position changed")
        require(
            recurrence["inputs"]["tensor_sha256"] == canonical["input_tensor_sha256"],
            "recurrence synthetic inputs changed",
        )
        require(
            recurrence["inputs"]["shared_kv_reused_for_every_step"] is True,
            "recurrence shared-KV policy changed",
        )
        require(
            recurrence["inputs"]["position_reused_for_every_step"] is True,
            "recurrence position policy changed",
        )
        require(
            args.recurrence_step < len(recurrence["steps"]),
            f"recurrence step {args.recurrence_step} is outside the oracle",
        )
        selected_recurrence = recurrence["steps"][args.recurrence_step]
        require(
            selected_recurrence["index"] == args.recurrence_step,
            "recurrence step index/order changed",
        )

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

    checkpoints: OrderedDict[str, torch.Tensor] = OrderedDict()
    handles = []

    def store_checkpoint(name: str, value: torch.Tensor) -> None:
        require(isinstance(value, torch.Tensor), f"checkpoint {name} is not a tensor")
        require(value.dtype == torch.bfloat16, f"checkpoint {name} is {value.dtype}, not BF16")
        require(name not in checkpoints, f"checkpoint {name} executed more than once")
        checkpoints[name] = value.detach().cpu().contiguous().reshape(-1)

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            store_checkpoint(name, value)

        return hook

    checkpoint_modules = [("pre_projection", model.pre_projection)]
    expected_stage_names = ["pre_projection"]
    for layer_index, layer in enumerate(model.model.layers):
        prefix = f"layer.{layer_index}"
        expected_stage_names.extend(
            [
                f"{prefix}.input_norm",
                f"{prefix}.q_proj",
                f"{prefix}.q_norm",
                f"{prefix}.q_rope",
                f"{prefix}.attention_scores",
                f"{prefix}.attention_probs",
                f"{prefix}.attention_context",
                f"{prefix}.o_proj",
                f"{prefix}.post_attention_norm",
                f"{prefix}.pre_feedforward_norm",
                f"{prefix}.gate_proj",
                f"{prefix}.up_proj",
                f"{prefix}.down_proj",
                f"{prefix}.post_feedforward_norm",
                f"{prefix}.output",
            ]
        )
        checkpoint_modules.extend(
            [
                (f"{prefix}.input_norm", layer.input_layernorm),
                (f"{prefix}.q_proj", layer.self_attn.q_proj),
                (f"{prefix}.q_norm", layer.self_attn.q_norm),
                (f"{prefix}.o_proj", layer.self_attn.o_proj),
                (f"{prefix}.post_attention_norm", layer.post_attention_layernorm),
                (f"{prefix}.pre_feedforward_norm", layer.pre_feedforward_layernorm),
                (f"{prefix}.gate_proj", layer.mlp.gate_proj),
                (f"{prefix}.up_proj", layer.mlp.up_proj),
                (f"{prefix}.down_proj", layer.mlp.down_proj),
                (f"{prefix}.post_feedforward_norm", layer.post_feedforward_layernorm),
                (f"{prefix}.output", layer),
            ]
        )
    checkpoint_modules.extend(
        [
            ("final_norm", model.model.norm),
            ("post_projection", model.post_projection),
            ("lm_head", model.lm_head),
        ]
    )
    expected_stage_names.extend(["final_norm", "post_projection", "lm_head"])
    for name, module in checkpoint_modules:
        handles.append(module.register_forward_hook(capture(name)))

    # Reproduce the pinned eager attention function verbatim while retaining
    # only the four bounded tensors needed to distinguish layout/indexing from
    # reduction-order drift. Local scores/probabilities are cropped to the last
    # 1024 positions, exactly matching the native compact window buffer.
    original_eager_attention = gemma4_modeling.eager_attention_forward

    def diagnostic_eager_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        softcap=None,
        **kwargs,
    ):
        del kwargs
        prefix = f"layer.{module.layer_idx}"
        store_checkpoint(f"{prefix}.q_rope", query)
        if scaling is None:
            scaling = module.head_dim**-0.5
        key_states = gemma4_modeling.repeat_kv(key, module.num_key_value_groups)
        value_states = gemma4_modeling.repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        compact_count = min(KV_LEN, module.sliding_window) if module.sliding_window else KV_LEN
        store_checkpoint(f"{prefix}.attention_scores", attn_weights[..., -compact_count:])
        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        store_checkpoint(f"{prefix}.attention_probs", attn_weights[..., -compact_count:])
        attn_output = torch.matmul(attn_weights, value_states)
        store_checkpoint(f"{prefix}.attention_context", attn_output)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    gemma4_modeling.eager_attention_forward = diagnostic_eager_attention

    generated = {name: deterministic_bf16(*spec) for name, spec in INPUT_SPECS.items()}
    embedding = generated["target_scaled_embedding"]
    recurrent_input = generated["target_final_normalized_hidden"]
    if args.recurrence_step != 0:
        require(
            recurrence is not None and selected_recurrence is not None,
            "nonzero recurrence step was not resolved",
        )
        previous_recurrence = recurrence["steps"][args.recurrence_step - 1]
        require(
            previous_recurrence["index"] == args.recurrence_step - 1,
            "previous recurrence step index/order changed",
        )
        previous_recurrent_bits = previous_recurrence["recurrent_hidden_bf16_bits"]
        require(len(previous_recurrent_bits) == TARGET_HIDDEN, "previous recurrent width changed")
        recurrent_input = (
            torch.tensor(previous_recurrent_bits, dtype=torch.uint16)
            .view(torch.bfloat16)
            .reshape(1, 1, TARGET_HIDDEN)
        )
        require(
            tensor_sha256(recurrent_input) == previous_recurrence["recurrent_hidden_bf16_sha256"],
            "previous recurrent bits/hash disagree",
        )
        require(
            selected_recurrence["recurrent_input_bf16_sha256"]
            == previous_recurrence["recurrent_hidden_bf16_sha256"],
            "selected recurrence input does not chain from the preceding step",
        )
        embedding = token_feedback_embedding(int(previous_recurrence["top1_token_id"]))
        require(
            tensor_sha256(embedding) == selected_recurrence["input_embedding_bf16_sha256"],
            "selected recurrence feedback embedding changed",
        )
    inputs_embeds = torch.cat(
        (embedding, recurrent_input),
        dim=-1,
    )
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
    with torch.inference_mode():
        output = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            use_cache=False,
        )
    gemma4_modeling.eager_attention_forward = original_eager_attention
    for handle in handles:
        handle.remove()

    logits = output.logits[0, 0].cpu().contiguous()
    recurrent = output.last_hidden_state[0, 0].cpu().contiguous()
    expected_output = (
        canonical["oracle_output"] if args.recurrence_step == 0 else selected_recurrence
    )
    require(expected_output is not None, "selected oracle output was not resolved")
    require(
        tensor_sha256(logits) == expected_output["logits_bf16_sha256"],
        "diagnostic logits differ from the selected oracle step",
    )
    require(
        tensor_sha256(recurrent) == expected_output["recurrent_hidden_bf16_sha256"],
        "diagnostic recurrent hidden differs from the selected oracle step",
    )
    require(
        int(torch.argmax(logits).item()) == expected_output["top1_token_id"],
        "diagnostic top-1 differs from the selected oracle step",
    )
    require(
        list(checkpoints) == expected_stage_names,
        "checkpoint execution order changed",
    )
    require(checkpoints["post_projection"].shape == recurrent.reshape(-1).shape, "post-projection width changed")
    require(checkpoints["lm_head"].shape == logits.reshape(-1).shape, "LM-head width changed")

    serialized = []
    for name, tensor in checkpoints.items():
        bits = bf16_bits(tensor).reshape(-1)
        serialized.append(
            {
                "name": name,
                "elements": int(bits.size),
                "bf16_sha256": tensor_sha256(tensor),
                "bf16_bits": [int(value) for value in bits],
            }
        )
    authority = {
        "model_sha256": MODEL_SHA256,
        "transformers_revision": TRANSFORMERS_REVISION,
        "transformers_source_sha256": source_hashes,
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "kv_len": KV_LEN,
        "position_id": POSITION_ID,
        "target_hidden": TARGET_HIDDEN,
        "sliding_geometry": [SLIDING_KV_HEADS, KV_LEN, SLIDING_HEAD_DIM],
        "full_geometry": [FULL_KV_HEADS, KV_LEN, FULL_HEAD_DIM],
    }
    if args.recurrence_oracle is not None:
        authority.update(
            {
                "recurrence_oracle_sha256": sha256_file(args.recurrence_oracle),
                "recurrence_step": args.recurrence_step,
                "input_embedding_bf16_sha256": tensor_sha256(embedding),
                "recurrent_input_bf16_sha256": tensor_sha256(recurrent_input),
            }
        )
    document = {
        "schema_version": 1,
        "authority": authority,
        "stages": serialized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    summary = {
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "stage_count": len(serialized),
    }
    if args.recurrence_oracle is not None:
        summary["recurrence_step"] = args.recurrence_step
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
