"""Fail-closed checkpoint validation for Camelid's EAGLE-3 serving ABI.

This module intentionally has no MLX or Safetensors package dependency.  It
parses the small Safetensors header directly so a bad warm start is rejected
before the roughly 490 MB checkpoint is materialized on the accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np

HIDDEN_SIZE = 3_072
INTERMEDIATE_SIZE = 8_192
NUM_ATTENTION_HEADS = 24
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
TARGET_VOCAB_SIZE = 128_256
DRAFT_VOCAB_SIZE = 32_000
EXPECTED_TENSOR_COUNT = 15
MAX_HEADER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]


TENSOR_SPECS: dict[str, TensorSpec] = {
    "d2t": TensorSpec("I32", (DRAFT_VOCAB_SIZE,)),
    "fc.weight": TensorSpec("BF16", (HIDDEN_SIZE, 3 * HIDDEN_SIZE)),
    "lm_head.weight": TensorSpec("BF16", (DRAFT_VOCAB_SIZE, HIDDEN_SIZE)),
    "midlayer.hidden_norm.weight": TensorSpec("BF16", (HIDDEN_SIZE,)),
    "midlayer.input_layernorm.weight": TensorSpec("BF16", (HIDDEN_SIZE,)),
    "midlayer.mlp.down_proj.weight": TensorSpec(
        "BF16", (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    ),
    "midlayer.mlp.gate_proj.weight": TensorSpec(
        "BF16", (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    ),
    "midlayer.mlp.up_proj.weight": TensorSpec(
        "BF16", (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    ),
    "midlayer.post_attention_layernorm.weight": TensorSpec("BF16", (HIDDEN_SIZE,)),
    "midlayer.self_attn.k_proj.weight": TensorSpec(
        "BF16", (NUM_KEY_VALUE_HEADS * HEAD_DIM, 2 * HIDDEN_SIZE)
    ),
    "midlayer.self_attn.o_proj.weight": TensorSpec("BF16", (HIDDEN_SIZE, HIDDEN_SIZE)),
    "midlayer.self_attn.q_proj.weight": TensorSpec(
        "BF16", (NUM_ATTENTION_HEADS * HEAD_DIM, 2 * HIDDEN_SIZE)
    ),
    "midlayer.self_attn.v_proj.weight": TensorSpec(
        "BF16", (NUM_KEY_VALUE_HEADS * HEAD_DIM, 2 * HIDDEN_SIZE)
    ),
    "norm.weight": TensorSpec("BF16", (HIDDEN_SIZE,)),
    "t2d": TensorSpec("BOOL", (TARGET_VOCAB_SIZE,)),
}

FLOAT_TENSOR_NAMES = tuple(
    name for name, spec in TENSOR_SPECS.items() if spec.dtype == "BF16"
)

_DTYPE_BYTES = {"BOOL": 1, "BF16": 2, "I32": 4, "I64": 8}


class ContractError(ValueError):
    """The artifact cannot be served by Camelid's pinned EAGLE-3 loader."""


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class CheckpointContract:
    checkpoint_dir: Path
    config: Mapping[str, Any]
    descriptors: Mapping[str, TensorDescriptor]
    payload_start: int
    weights_sha256: str
    config_sha256: str
    mapping_sha256: str


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _elements(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension < 0:
            raise ContractError(f"negative Safetensors dimension {dimension}")
        result *= dimension
    return result


def parse_safetensors_header(
    path: Path,
) -> tuple[int, dict[str, TensorDescriptor], Mapping[str, str]]:
    file_bytes = path.stat().st_size
    with path.open("rb") as stream:
        length_raw = stream.read(8)
        if len(length_raw) != 8:
            raise ContractError(f"{path} is shorter than a Safetensors header prefix")
        (header_bytes,) = struct.unpack("<Q", length_raw)
        if header_bytes > MAX_HEADER_BYTES:
            raise ContractError(
                f"Safetensors header is {header_bytes} bytes, above {MAX_HEADER_BYTES}"
            )
        raw = stream.read(header_bytes)
        if len(raw) != header_bytes:
            raise ContractError("Safetensors header is truncated")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid Safetensors header JSON: {error}") from error
    if not isinstance(header, dict):
        raise ContractError("Safetensors header root must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ContractError("Safetensors metadata must be a string-to-string object")

    payload_start = 8 + header_bytes
    payload_bytes = file_bytes - payload_start
    descriptors: dict[str, TensorDescriptor] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, raw_descriptor in header.items():
        if not isinstance(name, str) or not isinstance(raw_descriptor, dict):
            raise ContractError("invalid Safetensors tensor descriptor")
        if set(raw_descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ContractError(f"tensor {name!r} has unknown or missing descriptor fields")
        dtype = raw_descriptor["dtype"]
        shape_raw = raw_descriptor["shape"]
        offsets = raw_descriptor["data_offsets"]
        if dtype not in _DTYPE_BYTES:
            raise ContractError(f"tensor {name!r} has unsupported dtype {dtype!r}")
        if not isinstance(shape_raw, list) or any(
            not isinstance(value, int) for value in shape_raw
        ):
            raise ContractError(f"tensor {name!r} shape is not an integer array")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
        ):
            raise ContractError(f"tensor {name!r} data_offsets are invalid")
        shape = tuple(shape_raw)
        start, end = offsets
        expected_bytes = _elements(shape) * _DTYPE_BYTES[dtype]
        if start < 0 or start > end or end > payload_bytes:
            raise ContractError(
                f"tensor {name!r} range [{start}, {end}] is outside payload {payload_bytes}"
            )
        if end - start != expected_bytes:
            raise ContractError(
                f"tensor {name!r} occupies {end - start} bytes, expected {expected_bytes}"
            )
        descriptors[name] = TensorDescriptor(dtype, shape, start, end)
        ranges.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ContractError(
                f"tensor {name!r} starts at {start}, expected dense payload offset {cursor}"
            )
        cursor = end
    if cursor != payload_bytes:
        raise ContractError(
            f"tensors cover {cursor} payload bytes, file contains {payload_bytes}"
        )
    return payload_start, descriptors, metadata


def _validate_descriptors(descriptors: Mapping[str, TensorDescriptor]) -> None:
    actual = set(descriptors)
    expected = set(TENSOR_SPECS)
    if actual != expected:
        raise ContractError(
            "EAGLE-3 tensor set differs from the pinned 15-tensor contract; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if len(descriptors) != EXPECTED_TENSOR_COUNT:
        raise ContractError(f"expected {EXPECTED_TENSOR_COUNT} tensors")
    for name, spec in TENSOR_SPECS.items():
        descriptor = descriptors[name]
        allowed_dtype = descriptor.dtype == spec.dtype or (
            name == "d2t" and descriptor.dtype == "I64"
        )
        if not allowed_dtype:
            suffix = " or I64" if name == "d2t" else ""
            raise ContractError(
                f"tensor {name!r} dtype is {descriptor.dtype}, expected {spec.dtype}{suffix}"
            )
        if descriptor.shape != spec.shape:
            raise ContractError(
                f"tensor {name!r} shape is {descriptor.shape}, expected {spec.shape}"
            )


def validate_config(config: Mapping[str, Any]) -> None:
    architecture = config.get("architectures")
    if architecture not in (["LlamaForCausalLMEagle3"], ["LlamaForCausalLM"]):
        raise ContractError(f"unsupported architectures={architecture!r}")
    expected = {
        "model_type": "llama",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": 1,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": TARGET_VOCAB_SIZE,
        "draft_vocab_size": DRAFT_VOCAB_SIZE,
        "rms_norm_eps": 1.0e-5,
        "tie_word_embeddings": False,
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ContractError(f"config {key}={config.get(key)!r}, expected {wanted!r}")
    dtype = config.get("torch_dtype", config.get("dtype"))
    if dtype != "bfloat16":
        raise ContractError(f"config dtype={dtype!r}, expected 'bfloat16'")
    if config.get("rope_theta") not in (10_000, 500_000, 10_000.0, 500_000.0):
        raise ContractError("config rope_theta must be 10000 or 500000")
    if config.get("rope_scaling") is not None:
        raise ContractError("this MLX path supports only the checkpoint's null rope_scaling")
    if config.get("hidden_act", "silu") != "silu":
        raise ContractError("config hidden_act must be silu")
    if config.get("fc_norm", False) is not False:
        raise ContractError("config fc_norm must be false/absent for the 15-tensor contract")
    if config.get("norm_output", True) is not True:
        raise ContractError("config norm_output must be true/absent")
    window = config.get("sliding_window")
    use_window = config.get("use_sliding_window")
    if window is None and use_window is None:
        return
    if architecture != ["LlamaForCausalLMEagle3"]:
        raise ContractError("sliding-window checkpoints require LlamaForCausalLMEagle3")
    if window not in (256, 512) or use_window is not True:
        raise ContractError(
            "sliding-window config must be absent or use_sliding_window=true with 256/512"
        )


def _read_tensor_bytes(
    weights_path: Path, payload_start: int, descriptor: TensorDescriptor
) -> bytes:
    with weights_path.open("rb") as stream:
        stream.seek(payload_start + descriptor.start)
        data = stream.read(descriptor.end - descriptor.start)
    if len(data) != descriptor.end - descriptor.start:
        raise ContractError("short Safetensors tensor read")
    return data


def read_and_validate_mapping(
    weights_path: Path,
    payload_start: int,
    descriptors: Mapping[str, TensorDescriptor],
) -> tuple[np.ndarray, np.ndarray, str]:
    d2t_desc = descriptors["d2t"]
    d2t_dtype = "<i4" if d2t_desc.dtype == "I32" else "<i8"
    offsets = np.frombuffer(
        _read_tensor_bytes(weights_path, payload_start, d2t_desc), dtype=d2t_dtype
    ).astype(np.int64)
    absolute = np.arange(DRAFT_VOCAB_SIZE, dtype=np.int64) + offsets
    if np.any(absolute < 0) or np.any(absolute >= TARGET_VOCAB_SIZE):
        raise ContractError("d2t resolves outside target vocabulary")
    if np.any(absolute[1:] <= absolute[:-1]):
        raise ContractError("d2t absolute target IDs must be strictly increasing")

    t2d_desc = descriptors["t2d"]
    raw_t2d = _read_tensor_bytes(weights_path, payload_start, t2d_desc)
    t2d_u8 = np.frombuffer(raw_t2d, dtype=np.uint8)
    if np.any((t2d_u8 != 0) & (t2d_u8 != 1)):
        raise ContractError("t2d contains a non-boolean byte")
    t2d = t2d_u8.astype(np.bool_)
    expected_mask = np.zeros(TARGET_VOCAB_SIZE, dtype=np.bool_)
    expected_mask[absolute] = True
    if not np.array_equal(t2d, expected_mask):
        raise ContractError("t2d is not the exact membership mask described by d2t")

    digest = hashlib.sha256()
    digest.update(d2t_desc.dtype.encode("ascii"))
    digest.update(_read_tensor_bytes(weights_path, payload_start, d2t_desc))
    digest.update(raw_t2d)
    return absolute.astype(np.uint32), t2d, digest.hexdigest()


def target_to_draft_index(draft_to_target: np.ndarray) -> np.ndarray:
    inverse = np.full(TARGET_VOCAB_SIZE, -1, dtype=np.int32)
    inverse[draft_to_target.astype(np.int64)] = np.arange(
        draft_to_target.size, dtype=np.int32
    )
    return inverse


def validate_checkpoint(checkpoint_dir: Path | str) -> CheckpointContract:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ContractError("config.json root must be an object")
    validate_config(config)
    payload_start, descriptors, _metadata = parse_safetensors_header(weights_path)
    _validate_descriptors(descriptors)
    _d2t, _t2d, mapping_sha256 = read_and_validate_mapping(
        weights_path, payload_start, descriptors
    )
    return CheckpointContract(
        checkpoint_dir=checkpoint_dir,
        config=config,
        descriptors=descriptors,
        payload_start=payload_start,
        weights_sha256=sha256_file(weights_path),
        config_sha256=sha256_file(config_path),
        mapping_sha256=mapping_sha256,
    )
