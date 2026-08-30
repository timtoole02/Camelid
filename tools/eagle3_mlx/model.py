"""MLX implementation of Camelid's one-layer Llama EAGLE-3 head.

The public parameter tree deliberately spells the same 13 floating-point keys
as Camelid's Safetensors loader.  ``d2t`` and ``t2d`` remain immutable private
arrays and are reattached byte-for-byte at export, producing the full 15-key
serving artifact.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import (
    FLOAT_TENSOR_NAMES,
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    CheckpointContract,
    ContractError,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except ModuleNotFoundError:  # Allows format/unit tests without installing MLX.
    mx = None
    nn = None
    tree_flatten = None


def require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError(
            "MLX is required for training. Install a pinned mlx release in a dedicated "
            "environment on the Apple-Silicon training host."
        )


if nn is not None:

    class WeightOnlyLinear(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

        def __call__(self, values):
            return values @ self.weight.T


    class EagleRmsNorm(nn.Module):
        def __init__(self, weight, epsilon: float):
            super().__init__()
            self.weight = weight
            self.epsilon = epsilon

        def __call__(self, values):
            source_dtype = values.dtype
            values32 = values.astype(mx.float32)
            variance = mx.mean(values32 * values32, axis=-1, keepdims=True)
            normalized = values32 * mx.rsqrt(variance + self.epsilon)
            return normalized.astype(source_dtype) * self.weight


    class EagleSelfAttention(nn.Module):
        def __init__(self, weights: Mapping[str, Any], rope_theta: float):
            super().__init__()
            self.q_proj = WeightOnlyLinear(weights["midlayer.self_attn.q_proj.weight"])
            self.k_proj = WeightOnlyLinear(weights["midlayer.self_attn.k_proj.weight"])
            self.v_proj = WeightOnlyLinear(weights["midlayer.self_attn.v_proj.weight"])
            self.o_proj = WeightOnlyLinear(weights["midlayer.self_attn.o_proj.weight"])
            self.rope_theta = float(rope_theta)
            self.num_heads = NUM_ATTENTION_HEADS
            self.num_kv_heads = NUM_KEY_VALUE_HEADS
            self.groups = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
            self.head_dim = HEAD_DIM

        @staticmethod
        def _rotate_half(values):
            midpoint = values.shape[-1] // 2
            return mx.concatenate((-values[..., midpoint:], values[..., :midpoint]), axis=-1)

        def _rope(self, query, key, *, row_count: int, depth: int):
            frequency_index = mx.arange(0, self.head_dim, 2, dtype=mx.float32)
            inv_frequency = 1.0 / (
                self.rope_theta ** (frequency_index / float(self.head_dim))
            )
            positions = mx.arange(depth, depth + row_count, dtype=mx.float32)
            frequency = positions[:, None] * inv_frequency[None, :]
            phase = mx.concatenate((frequency, frequency), axis=-1)
            cos = mx.cos(phase)[None, None, :, :].astype(query.dtype)
            sin = mx.sin(phase)[None, None, :, :].astype(query.dtype)
            return (
                query * cos + self._rotate_half(query) * sin,
                key * cos + self._rotate_half(key) * sin,
            )

        def __call__(self, values, key_cache: list, value_cache: list, *, depth: int):
            batch, row_count, _ = values.shape
            query = self.q_proj(values).reshape(
                batch, row_count, self.num_heads, self.head_dim
            )
            key = self.k_proj(values).reshape(
                batch, row_count, self.num_kv_heads, self.head_dim
            )
            value = self.v_proj(values).reshape(
                batch, row_count, self.num_kv_heads, self.head_dim
            )
            query = mx.transpose(query, (0, 2, 1, 3))
            key = mx.transpose(key, (0, 2, 1, 3))
            value = mx.transpose(value, (0, 2, 1, 3))
            query, key = self._rope(query, key, row_count=row_count, depth=depth)
            key = mx.repeat(key, self.groups, axis=1)
            value = mx.repeat(value, self.groups, axis=1)
            key_cache.append(key)
            value_cache.append(value)

            scale = 1.0 / math.sqrt(self.head_dim)
            base_scores = (query @ mx.swapaxes(key_cache[0], -1, -2)) * scale
            causal = mx.triu(
                mx.full((row_count, row_count), -1.0e9, dtype=base_scores.dtype),
                k=1,
            )
            base_scores = base_scores + causal[None, None, :, :]
            if depth == 0:
                probabilities = mx.softmax(base_scores.astype(mx.float32), axis=-1).astype(
                    query.dtype
                )
                attention = probabilities @ value_cache[0]
            else:
                recurrent_scores = mx.stack(
                    [mx.sum(query * cached, axis=-1) * scale for cached in key_cache[1:]],
                    axis=-1,
                )
                scores = mx.concatenate((base_scores, recurrent_scores), axis=-1)
                probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
                    query.dtype
                )
                attention = probabilities[..., :row_count] @ value_cache[0]
                recurrent_p = probabilities[..., row_count:]
                for index, cached in enumerate(value_cache[1:]):
                    attention = attention + recurrent_p[..., index, None] * cached

            attention = mx.transpose(attention, (0, 2, 1, 3)).reshape(
                batch, row_count, HIDDEN_SIZE
            )
            return self.o_proj(attention)


    class EagleMlp(nn.Module):
        def __init__(self, weights: Mapping[str, Any]):
            super().__init__()
            self.down_proj = WeightOnlyLinear(weights["midlayer.mlp.down_proj.weight"])
            self.gate_proj = WeightOnlyLinear(weights["midlayer.mlp.gate_proj.weight"])
            self.up_proj = WeightOnlyLinear(weights["midlayer.mlp.up_proj.weight"])

        def __call__(self, values):
            gate = self.gate_proj(values)
            silu = gate * mx.sigmoid(gate)
            return self.down_proj(silu * self.up_proj(values))


    class EagleMidLayer(nn.Module):
        def __init__(self, weights: Mapping[str, Any], epsilon: float, rope_theta: float):
            super().__init__()
            self.hidden_norm = EagleRmsNorm(
                weights["midlayer.hidden_norm.weight"], epsilon
            )
            self.input_layernorm = EagleRmsNorm(
                weights["midlayer.input_layernorm.weight"], epsilon
            )
            self.self_attn = EagleSelfAttention(weights, rope_theta)
            self.post_attention_layernorm = EagleRmsNorm(
                weights["midlayer.post_attention_layernorm.weight"], epsilon
            )
            self.mlp = EagleMlp(weights)

        def __call__(
            self,
            input_embedding,
            hidden_state,
            key_cache: list,
            value_cache: list,
            *,
            depth: int,
        ):
            residual = hidden_state
            hidden_norm = self.hidden_norm(hidden_state)
            input_norm = self.input_layernorm(input_embedding)
            attention_input = mx.concatenate((input_norm, hidden_norm), axis=-1)
            hidden_state = residual + self.self_attn(
                attention_input, key_cache, value_cache, depth=depth
            )
            residual = hidden_state
            hidden_state = residual + self.mlp(self.post_attention_layernorm(hidden_state))
            return hidden_state


    class Eagle3Head(nn.Module):
        def __init__(
            self,
            weights: Mapping[str, Any],
            *,
            epsilon: float,
            rope_theta: float,
            sliding_window: int | None,
            mapping_tensors: Mapping[str, Any],
        ):
            super().__init__()
            self.fc = WeightOnlyLinear(weights["fc.weight"])
            self.lm_head = WeightOnlyLinear(weights["lm_head.weight"])
            self.midlayer = EagleMidLayer(weights, epsilon, rope_theta)
            self.norm = EagleRmsNorm(weights["norm.weight"], epsilon)
            # NumPy storage keeps these immutable serving tensors out of MLX's
            # trainable parameter tree. They are re-materialized only at export.
            self._mapping_tensors = {
                name: np.asarray(value) for name, value in mapping_tensors.items()
            }
            self._sliding_window = sliding_window

        @classmethod
        def from_checkpoint(
            cls,
            contract: CheckpointContract,
            *,
            parameter_dtype: str = "float32",
        ) -> "Eagle3Head":
            tensors = mx.load(str(contract.checkpoint_dir / "model.safetensors"))
            dtype = {"float32": mx.float32, "bfloat16": mx.bfloat16}.get(
                parameter_dtype
            )
            if dtype is None:
                raise ValueError("parameter_dtype must be 'float32' or 'bfloat16'")
            missing = set(FLOAT_TENSOR_NAMES) - set(tensors)
            if missing:
                raise ContractError(f"MLX load omitted tensors {sorted(missing)}")
            weights = {name: tensors[name].astype(dtype) for name in FLOAT_TENSOR_NAMES}
            model = cls(
                weights,
                epsilon=float(contract.config["rms_norm_eps"]),
                rope_theta=float(contract.config["rope_theta"]),
                sliding_window=contract.config.get("sliding_window"),
                mapping_tensors={"d2t": tensors["d2t"], "t2d": tensors["t2d"]},
            )
            actual = {name for name, _ in tree_flatten(model.parameters())}
            expected = set(FLOAT_TENSOR_NAMES)
            if actual != expected:
                raise ContractError(
                    "MLX model parameter tree does not match serving names; "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )
            return model

        def project_aux(self, aux_layer_inputs):
            return self.fc(aux_layer_inputs.astype(self.fc.weight.dtype))

        def recurrent_step(
            self,
            input_embedding,
            hidden_state,
            key_cache: list,
            value_cache: list,
            *,
            depth: int,
        ):
            row_count = int(hidden_state.shape[1])
            if self._sliding_window is not None and row_count > self._sliding_window:
                raise ValueError(
                    f"training rows {row_count} exceed checkpoint sliding window "
                    f"{self._sliding_window}; chunk the sample before training"
                )
            return self.midlayer(
                input_embedding,
                hidden_state,
                key_cache,
                value_cache,
                depth=depth,
            )

        def logits_at(self, hidden_state, positions):
            selected = mx.take(hidden_state[0], positions, axis=0)
            return self.lm_head(self.norm(selected))

        def serving_tensors(self) -> dict[str, Any]:
            tensors = {
                name: value.astype(mx.bfloat16)
                for name, value in tree_flatten(self.parameters())
            }
            tensors.update(
                {name: mx.array(value) for name, value in self._mapping_tensors.items()}
            )
            return tensors


else:

    class Eagle3Head:  # pragma: no cover - exercised only on hosts without MLX.
        @classmethod
        def from_checkpoint(cls, *_args, **_kwargs):
            require_mlx()


def hard_ce(logits, target_draft_ids):
    """Stable hard-label cross entropy over the fixed 32K draft vocabulary."""

    require_mlx()
    logits32 = logits.astype(mx.float32)
    selected = mx.take_along_axis(logits32, target_draft_ids[:, None], axis=-1)[:, 0]
    return mx.mean(mx.logsumexp(logits32, axis=-1) - selected)


def soft_target_kl_numpy(
    draft_logits: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    reduction: str = "mean",
) -> np.ndarray | float:
    """Stable reference KL(teacher || draft) over the fixed draft vocabulary.

    Both distributions are normalized over the supplied final dimension.  In
    production that dimension is the checkpoint-pinned 32K ``d2t`` order.  The
    NumPy implementation is intentionally independent of MLX and serves as the
    numerical parity oracle for the device kernel.
    """

    draft = np.asarray(draft_logits, dtype=np.float64)
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    if draft.shape != teacher.shape or draft.ndim != 2:
        raise ValueError("draft and teacher logits must have the same rank-2 shape")
    teacher_shifted = teacher - np.max(teacher, axis=-1, keepdims=True)
    teacher_exp = np.exp(teacher_shifted)
    teacher_log_z = np.log(np.sum(teacher_exp, axis=-1, keepdims=True))
    teacher_log_p = teacher_shifted - teacher_log_z
    teacher_p = teacher_exp / np.sum(teacher_exp, axis=-1, keepdims=True)
    draft_shifted = draft - np.max(draft, axis=-1, keepdims=True)
    draft_log_p = draft_shifted - np.log(
        np.sum(np.exp(draft_shifted), axis=-1, keepdims=True)
    )
    per_row = np.sum(teacher_p * (teacher_log_p - draft_log_p), axis=-1)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return float(np.sum(per_row))
    if reduction == "mean":
        return float(np.mean(per_row))
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def soft_target_cross_entropy_numpy(
    draft_logits: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    reduction: str = "mean",
) -> np.ndarray | float:
    """Stable NumPy reference for SpecForge's soft-target cross entropy."""

    draft = np.asarray(draft_logits, dtype=np.float64)
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    if draft.shape != teacher.shape or draft.ndim != 2:
        raise ValueError("draft and teacher logits must have the same rank-2 shape")
    teacher_shifted = teacher - np.max(teacher, axis=-1, keepdims=True)
    teacher_exp = np.exp(teacher_shifted)
    teacher_p = teacher_exp / np.sum(teacher_exp, axis=-1, keepdims=True)
    draft_shifted = draft - np.max(draft, axis=-1, keepdims=True)
    draft_log_p = draft_shifted - np.log(
        np.sum(np.exp(draft_shifted), axis=-1, keepdims=True)
    )
    per_row = -np.sum(teacher_p * draft_log_p, axis=-1)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return float(np.sum(per_row))
    if reduction == "mean":
        return float(np.mean(per_row))
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def soft_target_kl(logits, teacher_logits, *, reduction: str = "mean"):
    """Stable MLX KL against a teacher renormalized over the fixed 32K rows.

    This is gradient-equivalent to SpecForge's soft-target cross entropy, but
    subtracts teacher entropy so the reported scalar has true KL semantics.
    """

    require_mlx()
    draft32 = logits.astype(mx.float32)
    teacher32 = teacher_logits.astype(mx.float32)
    teacher_log_p = teacher32 - mx.logsumexp(teacher32, axis=-1, keepdims=True)
    teacher_p = mx.exp(teacher_log_p)
    draft_log_p = draft32 - mx.logsumexp(draft32, axis=-1, keepdims=True)
    per_row = mx.sum(teacher_p * (teacher_log_p - draft_log_p), axis=-1)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return mx.sum(per_row)
    if reduction == "mean":
        return mx.mean(per_row)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def soft_target_cross_entropy(logits, teacher_logits, *, reduction: str = "mean"):
    """Stable MLX form of SpecForge ``LogSoftmaxLoss`` for selected rows."""

    require_mlx()
    draft32 = logits.astype(mx.float32)
    teacher32 = teacher_logits.astype(mx.float32)
    teacher_p = mx.softmax(teacher32, axis=-1)
    draft_log_p = draft32 - mx.logsumexp(draft32, axis=-1, keepdims=True)
    per_row = -mx.sum(teacher_p * draft_log_p, axis=-1)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return mx.sum(per_row)
    if reduction == "mean":
        return mx.mean(per_row)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def ttt_soft_target_loss(
    model: Eagle3Head,
    batch,
    *,
    row_chunk: int = 16,
    loss_kind: str = "cross_entropy",
) -> Any:
    """Official seven-depth soft teacher objective with bounded logit rows.

    The Q4 exporter has already gathered teacher logits into the immutable
    checkpoint ``d2t`` row order.  Only supervised rows are decoded from BF16,
    and each 32K projection is chunked to bound unified-memory activations.
    """

    require_mlx()
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    if loss_kind not in ("cross_entropy", "kl"):
        raise ValueError("loss_kind must be 'cross_entropy' or 'kl'")
    compute_dtype = model.fc.weight.dtype
    hidden_state = model.project_aux(
        mx.array(batch.aux_layer_inputs).astype(compute_dtype)[None, ...]
    )
    key_cache: list = []
    value_cache: list = []
    losses = []
    weights = []
    for view in batch.depths:
        input_embedding = mx.array(view.embedding).astype(compute_dtype)[None, ...]
        hidden_state = model.recurrent_step(
            input_embedding,
            hidden_state,
            key_cache,
            value_cache,
            depth=view.depth,
        )
        positions = np.flatnonzero(view.supervised).astype(np.int32)
        if positions.size == 0:
            continue
        depth_sum = None
        for start in range(0, int(positions.size), row_chunk):
            selected_positions = positions[start : start + row_chunk]
            logits = model.logits_at(hidden_state, mx.array(selected_positions))
            teacher = mx.array(view.teacher_logits_at(selected_positions))
            if loss_kind == "cross_entropy":
                chunk_sum = soft_target_cross_entropy(
                    logits, teacher, reduction="sum"
                )
            else:
                chunk_sum = soft_target_kl(logits, teacher, reduction="sum")
            depth_sum = chunk_sum if depth_sum is None else depth_sum + chunk_sum
        # SpecForge's masked loss means over the full retained backbone length,
        # not only the supervised rows. Chunking must preserve that denominator.
        losses.append(depth_sum / float(batch.row_count))
        weights.append(0.8**view.depth)
    if not losses:
        raise ValueError(f"sample {batch.sample_id!r} has no supervised Q4 teacher rows")
    total = losses[0] * weights[0]
    for loss, weight in zip(losses[1:], weights[1:]):
        total = total + loss * weight
    return total


def ttt_hard_ce_loss(model: Eagle3Head, batch) -> Any:
    """Seven-depth, target-greedy EAGLE objective with SpecForge's 0.8 decay."""

    require_mlx()
    compute_dtype = model.fc.weight.dtype
    hidden_state = model.project_aux(
        mx.array(batch.aux_layer_inputs).astype(compute_dtype)[None, ...]
    )
    key_cache: list = []
    value_cache: list = []
    losses = []
    weights = []
    for view in batch.depths:
        input_embedding = mx.array(view.embedding).astype(compute_dtype)[None, ...]
        hidden_state = model.recurrent_step(
            input_embedding,
            hidden_state,
            key_cache,
            value_cache,
            depth=view.depth,
        )
        positions_np = np.flatnonzero(view.hard_supervised).astype(np.int32)
        if positions_np.size == 0:
            continue
        targets_np = view.target_draft_ids[positions_np].astype(np.int32)
        logits = model.logits_at(hidden_state, mx.array(positions_np))
        losses.append(
            hard_ce(logits, mx.array(targets_np))
            * (float(positions_np.size) / float(batch.row_count))
        )
        weights.append(0.8**view.depth)
    if not losses:
        raise ValueError(f"sample {batch.sample_id!r} has no mapped supervised Q4 targets")
    total = losses[0] * weights[0]
    for loss, weight in zip(losses[1:], weights[1:]):
        total = total + loss * weight
    return total


def ttt_hard_ce_metrics(model: Eagle3Head, batch) -> dict[str, Any]:
    """Evaluate mapped top-1 accuracy at every TTT depth."""

    require_mlx()
    compute_dtype = model.fc.weight.dtype
    hidden_state = model.project_aux(
        mx.array(batch.aux_layer_inputs).astype(compute_dtype)[None, ...]
    )
    key_cache: list = []
    value_cache: list = []
    depth_metrics: list[dict[str, float | int]] = []
    total_correct = 0
    total_rows = 0
    for view in batch.depths:
        hidden_state = model.recurrent_step(
            mx.array(view.embedding).astype(compute_dtype)[None, ...],
            hidden_state,
            key_cache,
            value_cache,
            depth=view.depth,
        )
        positions_np = np.flatnonzero(view.hard_supervised).astype(np.int32)
        if positions_np.size == 0:
            depth_metrics.append(
                {"depth": view.depth, "correct": 0, "rows": 0, "accuracy": 0.0}
            )
            continue
        targets_np = view.target_draft_ids[positions_np].astype(np.int32)
        logits = model.logits_at(hidden_state, mx.array(positions_np))
        predicted = mx.argmax(logits, axis=-1)
        correct_array = mx.sum(predicted == mx.array(targets_np))
        mx.eval(correct_array)
        correct = int(correct_array.item())
        rows = int(positions_np.size)
        total_correct += correct
        total_rows += rows
        depth_metrics.append(
            {
                "depth": view.depth,
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
            }
        )
    return {
        "depths": depth_metrics,
        "correct": total_correct,
        "rows": total_rows,
        "accuracy": total_correct / max(total_rows, 1),
    }
