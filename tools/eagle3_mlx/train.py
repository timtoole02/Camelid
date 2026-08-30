"""Warm-start Camelid's pinned SW512 EAGLE-3 head with exact-Q4 teachers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

import numpy as np

from .contract import (
    TENSOR_SPECS,
    read_and_validate_mapping,
    target_to_draft_index,
    validate_checkpoint,
)
from .features import FeatureStore, build_ttt_batch
from .model import (
    Eagle3Head,
    require_mlx,
    ttt_hard_ce_loss,
    ttt_hard_ce_metrics,
    ttt_soft_target_loss,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
except ModuleNotFoundError:
    mx = None
    nn = None
    optim = None
    tree_flatten = None
    tree_unflatten = None


def _json_line(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _load_mapping(contract):
    weights_path = contract.checkpoint_dir / "model.safetensors"
    draft_to_target, _mask, _fingerprint = read_and_validate_mapping(
        weights_path, contract.payload_start, contract.descriptors
    )
    return target_to_draft_index(draft_to_target), draft_to_target


def _evaluate(
    model,
    store: FeatureStore,
    target_to_draft: np.ndarray,
    *,
    max_length: int,
    ttt_length: int,
    max_samples: int | None,
) -> dict[str, Any]:
    model.eval()
    total_correct = 0
    total_rows = 0
    per_depth = [dict(correct=0, rows=0) for _ in range(ttt_length)]
    limit = len(store) if max_samples is None else min(len(store), max_samples)
    for index in range(limit):
        sample = store.load(index, max_length=max_length)
        batch = build_ttt_batch(sample, target_to_draft, ttt_length=ttt_length)
        metrics = ttt_hard_ce_metrics(model, batch)
        total_correct += metrics["correct"]
        total_rows += metrics["rows"]
        for depth in metrics["depths"]:
            target = per_depth[depth["depth"]]
            target["correct"] += depth["correct"]
            target["rows"] += depth["rows"]
    for depth, counts in enumerate(per_depth):
        counts["depth"] = depth
        counts["accuracy"] = counts["correct"] / max(counts["rows"], 1)
    model.train()
    return {
        "samples": limit,
        "correct": total_correct,
        "rows": total_rows,
        "accuracy": total_correct / max(total_rows, 1),
        "depths": per_depth,
    }


def _export_serving_checkpoint(
    model,
    source_contract,
    output_dir: Path,
    *,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "model.safetensors"
    config_path = output_dir / "config.json"
    tensors = model.serving_tensors()
    if set(tensors) != set(TENSOR_SPECS):
        raise RuntimeError(
            "export tensor set differs from serving contract: "
            f"missing={sorted(set(TENSOR_SPECS) - set(tensors))}, "
            f"extra={sorted(set(tensors) - set(TENSOR_SPECS))}"
        )
    mx.eval(tensors)
    mx.save_safetensors(
        str(tensor_path),
        tensors,
        metadata={
            "format": "camelid-eagle3-mlx-warmstart-v1",
            "source_weights_sha256": source_contract.weights_sha256,
            "source_mapping_sha256": source_contract.mapping_sha256,
        },
    )
    # Preserve the serving config byte-for-byte; changing it would silently
    # alter RoPE/window semantics even when every tensor stayed compatible.
    shutil.copyfile(source_contract.checkpoint_dir / "config.json", config_path)
    output_contract = validate_checkpoint(output_dir)
    if output_contract.config_sha256 != source_contract.config_sha256:
        raise RuntimeError("export changed the pinned config bytes")
    if output_contract.mapping_sha256 != source_contract.mapping_sha256:
        raise RuntimeError("export changed the fixed 32K vocabulary mapping")
    receipt = {
        **receipt,
        "output_weights_sha256": output_contract.weights_sha256,
        "output_config_sha256": output_contract.config_sha256,
        "output_mapping_sha256": output_contract.mapping_sha256,
        "tensor_count": len(output_contract.descriptors),
    }
    (output_dir / "training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _load_training_state(model, optimizer, state_dir: Path, source_contract) -> tuple[int, dict]:
    state_dir = state_dir.resolve()
    metadata_path = state_dir / "state.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load training state {metadata_path}: {error}") from error
    expected = {
        "schema": "camelid-eagle3-mlx-state-v1",
        "source_config_sha256": source_contract.config_sha256,
        "source_mapping_sha256": source_contract.mapping_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"training state {key}={metadata.get(key)!r}, expected {value!r}"
            )
    model_path = state_dir / "training-model.safetensors"
    optimizer_path = state_dir / "optimizer.safetensors"
    from .contract import sha256_file

    if sha256_file(model_path) != metadata.get("model_sha256"):
        raise RuntimeError("training model state SHA-256 mismatch")
    if sha256_file(optimizer_path) != metadata.get("optimizer_sha256"):
        raise RuntimeError("optimizer state SHA-256 mismatch")
    model.load_weights(str(model_path), strict=True)
    optimizer.state = tree_unflatten(mx.load(str(optimizer_path)))
    mx.eval(model.parameters(), optimizer.state)
    global_step = metadata.get("global_step")
    if not isinstance(global_step, int) or global_step < 0:
        raise RuntimeError("training state global_step is invalid")
    return global_step, metadata


def _save_training_state(
    model,
    optimizer,
    state_dir: Path,
    source_contract,
    *,
    global_step: int,
    rng_state: dict,
    training_contract: dict[str, Any],
) -> dict[str, Any]:
    state_dir = state_dir.resolve()
    if state_dir.exists() and any(state_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty training state {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    model_tmp = state_dir / f".training-model.{suffix}.safetensors"
    optimizer_tmp = state_dir / f".optimizer.{suffix}.safetensors"
    model_path = state_dir / "training-model.safetensors"
    optimizer_path = state_dir / "optimizer.safetensors"
    metadata_path = state_dir / "state.json"
    model.save_weights(str(model_tmp))
    optimizer_flat = tree_flatten(optimizer.state, destination={})
    mx.save_safetensors(str(optimizer_tmp), optimizer_flat)
    mx.eval(model.parameters(), optimizer.state)
    os.replace(model_tmp, model_path)
    os.replace(optimizer_tmp, optimizer_path)
    from .contract import sha256_file

    metadata = {
        "schema": "camelid-eagle3-mlx-state-v1",
        "global_step": global_step,
        "source_weights_sha256": source_contract.weights_sha256,
        "source_config_sha256": source_contract.config_sha256,
        "source_mapping_sha256": source_contract.mapping_sha256,
        "model_sha256": sha256_file(model_path),
        "optimizer_sha256": sha256_file(optimizer_path),
        "rng_state": rng_state,
        "training_contract": training_contract,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_tmp = state_dir / f".state.{suffix}.json"
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(metadata_tmp, metadata_path)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--eval-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--state-output",
        type=Path,
        help="new directory for resumable FP32/BF16 model and Adam state",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="state directory from an earlier serialized feature shard",
    )
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ttt-length", type=int, default=7)
    parser.add_argument(
        "--objective",
        choices=("soft_ce", "soft_kl", "hard_ce"),
        default="soft_ce",
        help="soft_ce matches SpecForge; soft_kl differs by teacher entropy; hard_ce is an ablation",
    )
    parser.add_argument(
        "--loss-row-chunk",
        type=int,
        default=16,
        help="maximum supervised [rows,32000] projection per loss chunk",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-max-samples", type=int)
    parser.add_argument("--parameter-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--skip-feature-hashes", action="store_true")
    parser.add_argument(
        "--allow-shard-reuse",
        action="store_true",
        help="permit more optimizer updates than feature records in this invocation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.max_steps <= 0
        or args.max_length <= args.ttt_length
        or args.loss_row_chunk <= 0
    ):
        raise ValueError("max_steps must be positive and max_length must exceed ttt_length")
    if args.warmup_steps < 0 or args.learning_rate <= 0:
        raise ValueError("warmup_steps must be non-negative and learning_rate positive")
    if (
        args.log_interval <= 0
        or args.eval_interval <= 0
        or args.max_grad_norm <= 0
        or args.weight_decay < 0
    ):
        raise ValueError("intervals/grad norm must be positive and weight decay non-negative")
    require_mlx()
    if not mx.metal.is_available():
        raise RuntimeError("this training path requires MLX Metal on Apple Silicon")

    source = validate_checkpoint(args.warm_start)
    if source.config.get("sliding_window") != 512 or source.config.get(
        "use_sliding_window"
    ) is not True:
        raise RuntimeError(
            "this campaign must warm-start the pinned SW512 head; the supplied "
            "checkpoint is not sliding_window=512/use_sliding_window=true"
        )
    train_store = FeatureStore(args.features, verify_hashes=not args.skip_feature_hashes)
    eval_store = FeatureStore(
        args.eval_features, verify_hashes=not args.skip_feature_hashes
    )
    target_to_draft, draft_to_target = _load_mapping(source)
    for name, store in (("training", train_store), ("evaluation", eval_store)):
        if store.manifest["eagle3_checkpoint_sha256"] != source.weights_sha256:
            raise RuntimeError(
                f"{name} features were not exported with the supplied SW512 checkpoint"
            )
        if not np.array_equal(store.draft_to_target, draft_to_target):
            raise RuntimeError(
                f"{name} feature draft_to_target rows do not match the warm checkpoint"
            )
    if (
        train_store.manifest["target_model_sha256"]
        != eval_store.manifest["target_model_sha256"]
    ):
        raise RuntimeError("training and evaluation stores use different Q4 target models")
    if args.max_steps > len(train_store) and not args.allow_shard_reuse:
        raise RuntimeError(
            f"max_steps={args.max_steps} would reuse a {len(train_store)}-sample feature "
            "shard; pass --allow-shard-reuse only for an intentional multi-epoch run"
        )
    model = Eagle3Head.from_checkpoint(source, parameter_dtype=args.parameter_dtype)
    model.train()
    mx.eval(model.parameters())
    rng = np.random.default_rng(args.seed)
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )
    global_step_start = 0
    resumed_metadata = None
    if args.resume_state:
        global_step_start, resumed_metadata = _load_training_state(
            model, optimizer, args.resume_state, source
        )
        prior_contract = resumed_metadata.get("training_contract", {})
        invariant = {
            "ttt_length": args.ttt_length,
            "parameter_dtype": args.parameter_dtype,
            "objective": args.objective,
            "loss_row_chunk": args.loss_row_chunk,
            "feature_draft_mapping_sha256": train_store.manifest[
                "draft_mapping_sha256"
            ],
        }
        for key, value in invariant.items():
            if prior_contract.get(key) != value:
                raise RuntimeError(
                    f"resume training contract {key}={prior_contract.get(key)!r}, "
                    f"expected {value!r}"
                )
        if isinstance(resumed_metadata.get("rng_state"), dict):
            rng.bit_generator.state = resumed_metadata["rng_state"]
    permutation = rng.permutation(len(train_store))

    def loss_fn(active_model, active_batch):
        if args.objective in ("soft_ce", "soft_kl"):
            return ttt_soft_target_loss(
                active_model,
                active_batch,
                row_chunk=args.loss_row_chunk,
                loss_kind=(
                    "cross_entropy" if args.objective == "soft_ce" else "kl"
                ),
            )
        return ttt_hard_ce_loss(active_model, active_batch)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    run_start = datetime.now(timezone.utc).isoformat()
    _json_line(
        "start",
        source_weights_sha256=source.weights_sha256,
        source_mapping_sha256=source.mapping_sha256,
        feature_draft_mapping_sha256=train_store.manifest["draft_mapping_sha256"],
        train_samples=len(train_store),
        eval_samples=len(eval_store),
        max_steps=args.max_steps,
        global_step_start=global_step_start,
        max_length=args.max_length,
        ttt_length=args.ttt_length,
        parameter_dtype=args.parameter_dtype,
        objective=args.objective,
        loss_row_chunk=args.loss_row_chunk,
    )
    baseline_eval = _evaluate(
        model,
        eval_store,
        target_to_draft,
        max_length=args.max_length,
        ttt_length=args.ttt_length,
        max_samples=args.eval_max_samples,
    )
    _json_line("eval_baseline", step=0, **baseline_eval)
    last_loss = None
    last_eval = None
    for local_step in range(1, args.max_steps + 1):
        step = global_step_start + local_step
        if (local_step - 1) % len(permutation) == 0 and local_step > 1:
            permutation = rng.permutation(len(train_store))
        sample_index = int(permutation[(local_step - 1) % len(permutation)])
        sample = train_store.load(sample_index, max_length=args.max_length)
        batch = build_ttt_batch(sample, target_to_draft, ttt_length=args.ttt_length)
        warmup_scale = min(1.0, step / max(args.warmup_steps, 1))
        optimizer.learning_rate = args.learning_rate * warmup_scale
        loss, gradients = loss_and_grad(model, batch)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, args.max_grad_norm)
        optimizer.update(model, gradients)
        mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
        last_loss = float(loss.item())
        if local_step == 1 or step % args.log_interval == 0:
            _json_line(
                "train",
                step=step,
                sample=sample.sample_id,
                rows=batch.row_count,
                loss=last_loss,
                gradient_norm=float(gradient_norm.item()),
                learning_rate=float(optimizer.learning_rate),
                active_memory_bytes=int(mx.get_active_memory()),
                peak_memory_bytes=int(mx.get_peak_memory()),
            )
        if step % args.eval_interval == 0 or local_step == args.max_steps:
            last_eval = _evaluate(
                model,
                eval_store,
                target_to_draft,
                max_length=args.max_length,
                ttt_length=args.ttt_length,
                max_samples=args.eval_max_samples,
            )
            _json_line("eval", step=step, **last_eval)

    baseline_accuracy = float(baseline_eval["accuracy"])
    final_accuracy = float(last_eval["accuracy"]) if last_eval else 0.0
    relative_accuracy_gain = (
        (final_accuracy - baseline_accuracy) / baseline_accuracy
        if baseline_accuracy > 0
        else None
    )
    receipt = {
        "schema": "camelid-eagle3-mlx-training-receipt-v1",
        "started_at": run_start,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "objective": args.objective,
        "teacher_distribution": "exact_q4_logits_renormalized_over_fixed_d2t_32000",
        "source_checkpoint": str(source.checkpoint_dir),
        "source_weights_sha256": source.weights_sha256,
        "source_config_sha256": source.config_sha256,
        "source_mapping_sha256": source.mapping_sha256,
        "train_feature_draft_mapping_sha256": train_store.manifest[
            "draft_mapping_sha256"
        ],
        "eval_feature_draft_mapping_sha256": eval_store.manifest[
            "draft_mapping_sha256"
        ],
        "feature_positional_contract": train_store.manifest["positional_contract"],
        "target_model_sha256": train_store.manifest.get("target_model_sha256"),
        "train_manifest": str((args.features / "manifest.json").resolve()),
        "eval_manifest": (
            str((args.eval_features / "manifest.json").resolve())
            if args.eval_features
            else None
        ),
        "max_steps": args.max_steps,
        "global_step_start": global_step_start,
        "global_step_end": global_step_start + args.max_steps,
        "max_length": args.max_length,
        "ttt_length": args.ttt_length,
        "loss_row_chunk": args.loss_row_chunk,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "parameter_dtype": args.parameter_dtype,
        "final_train_loss": last_loss,
        "final_eval": last_eval,
        "baseline_eval": baseline_eval,
        "eval_accuracy_absolute_gain": final_accuracy - baseline_accuracy,
        "eval_accuracy_relative_gain": relative_accuracy_gain,
        "pilot_quality_gate_passed": (
            relative_accuracy_gain is not None and relative_accuracy_gain >= 0.10
        ),
        "mlx_version": getattr(sys.modules.get("mlx"), "__version__", "unknown"),
        "pid": os.getpid(),
    }
    exported = _export_serving_checkpoint(
        model, source, args.output_dir.resolve(), receipt=receipt
    )
    if args.state_output:
        state_metadata = _save_training_state(
            model,
            optimizer,
            args.state_output,
            source,
            global_step=global_step_start + args.max_steps,
            rng_state=rng.bit_generator.state,
            training_contract={
                "ttt_length": args.ttt_length,
                "parameter_dtype": args.parameter_dtype,
                "objective": args.objective,
                "loss_row_chunk": args.loss_row_chunk,
                "feature_draft_mapping_sha256": train_store.manifest[
                    "draft_mapping_sha256"
                ],
            },
        )
        _json_line(
            "state",
            state_dir=str(args.state_output.resolve()),
            global_step=state_metadata["global_step"],
            model_sha256=state_metadata["model_sha256"],
            optimizer_sha256=state_metadata["optimizer_sha256"],
        )
    _json_line("export", output_dir=str(args.output_dir.resolve()), **exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
