"""Reader and alignment logic for Camelid exact-Q4 EAGLE feature stores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .contract import DRAFT_VOCAB_SIZE, HIDDEN_SIZE, TARGET_VOCAB_SIZE

SCHEMA_ID = "camelid-eagle3-q4-features-v1"
POSITIONAL_CONTRACT = "eagle3-aux-p-next-token-teacher-p1-v1"
INVALID_TOKEN_ID = np.uint32(0xFFFF_FFFF)
AUX_WIDTH = 3 * HIDDEN_SIZE
TARGET_LAYER_INPUT_IDS = [2, 14, 25]

_REQUIRED_ARRAYS: dict[str, tuple[str, tuple[int | None, ...]]] = {
    "input_ids.u32le": ("uint32", (None,)),
    "labels.u32le": ("uint32", (None,)),
    "target_argmax.u32le": ("uint32", (None,)),
    "loss_mask.u8": ("uint8", (None,)),
    "aux_layer_inputs.bf16le": ("bfloat16", (None, AUX_WIDTH)),
    "hidden_state.bf16le": ("bfloat16", (None, HIDDEN_SIZE)),
    "input_embedding.bf16le": ("bfloat16", (None, HIDDEN_SIZE)),
    "next_token_embedding.bf16le": ("bfloat16", (None, HIDDEN_SIZE)),
    "teacher_draft_logits.bf16le": ("bfloat16", (None, DRAFT_VOCAB_SIZE)),
}
_OPTIONAL_ARRAYS: dict[str, tuple[str, tuple[int | None, ...]]] = {
    "teacher_logsumexp.f32le": ("float32", (None,)),
}


class FeatureFormatError(ValueError):
    pass


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    if raw.dtype != np.dtype("<u2"):
        raw = raw.astype("<u2", copy=False)
    widened = raw.astype(np.uint32) << np.uint32(16)
    return widened.view(np.float32)


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 using IEEE round-to-nearest-even.

    This helper is used by the synthetic format tests and matches the Rust
    exporter's conversion.  Training data normally arrives already encoded.
    """

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype("<u2")


@dataclass(frozen=True)
class FeatureSample:
    sample_id: str
    input_ids: np.ndarray
    labels: np.ndarray
    target_argmax: np.ndarray
    loss_mask: np.ndarray
    aux_layer_inputs: np.ndarray
    hidden_state: np.ndarray
    input_embedding: np.ndarray
    next_token_embedding: np.ndarray
    teacher_draft_logits_bf16: np.ndarray
    teacher_logsumexp: np.ndarray | None
    bootstrap_rows_masked: int

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass(frozen=True)
class TttDepthView:
    depth: int
    embedding: np.ndarray
    target_ids: np.ndarray
    target_draft_ids: np.ndarray
    supervised: np.ndarray
    hard_supervised: np.ndarray
    teacher_draft_logits_bf16: np.ndarray

    def teacher_logits_at(self, positions: np.ndarray) -> np.ndarray:
        """Decode only selected teacher rows, keeping the 32K file memory-mapped."""

        bits = np.asarray(self.teacher_draft_logits_bf16[positions], dtype="<u2")
        return np.ascontiguousarray(_bf16_to_float32(bits))


@dataclass(frozen=True)
class TttBatch:
    sample_id: str
    aux_layer_inputs: np.ndarray
    depths: tuple[TttDepthView, ...]

    @property
    def row_count(self) -> int:
        return int(self.aux_layer_inputs.shape[0])


def build_ttt_batch(
    sample: FeatureSample,
    target_to_draft: np.ndarray,
    *,
    ttt_length: int = 7,
) -> TttBatch:
    """Build the exact teacher-forced EAGLE-3 shift views.

    The exporter has already applied SpecForge's initial left shift.  Base row
    ``P`` is auxiliary state from target capture row ``P``, the embedding of
    token ``P+1``, and the teacher distribution captured after row ``P+1``
    (predicting token ``P+2``).  At unroll depth ``d``, row ``P`` therefore
    uses the base embedding, teacher and mask at ``P+d``.  A common prefix
    avoids padded tail rows without changing any retained causal computation.
    """

    if ttt_length <= 0:
        raise FeatureFormatError("ttt_length must be positive")
    if target_to_draft.shape != (TARGET_VOCAB_SIZE,):
        raise FeatureFormatError(
            f"target_to_draft shape is {target_to_draft.shape}, expected {(TARGET_VOCAB_SIZE,)}"
        )
    rows = sample.length - ttt_length
    if rows <= 0:
        raise FeatureFormatError(
            f"sample {sample.sample_id!r} has {sample.length} rows, not enough for "
            f"ttt_length={ttt_length}"
        )
    depths: list[TttDepthView] = []
    for depth in range(ttt_length):
        embedding = sample.next_token_embedding[depth : depth + rows]
        target_ids = sample.target_argmax[depth : depth + rows]
        source_mask = sample.loss_mask[depth : depth + rows].astype(np.bool_, copy=False)
        valid_target = target_ids != INVALID_TOKEN_ID
        safe_target = np.where(valid_target, target_ids, np.uint32(0)).astype(np.int64)
        if np.any(safe_target >= TARGET_VOCAB_SIZE):
            bad = int(safe_target[safe_target >= TARGET_VOCAB_SIZE][0])
            raise FeatureFormatError(f"target_argmax contains out-of-vocabulary id {bad}")
        draft_ids = target_to_draft[safe_target]
        supervised = source_mask & valid_target
        hard_supervised = supervised & (draft_ids >= 0)
        depths.append(
            TttDepthView(
                depth=depth,
                embedding=np.ascontiguousarray(embedding),
                target_ids=np.ascontiguousarray(target_ids),
                target_draft_ids=np.ascontiguousarray(draft_ids.astype(np.int32)),
                supervised=np.ascontiguousarray(supervised),
                hard_supervised=np.ascontiguousarray(hard_supervised),
                teacher_draft_logits_bf16=sample.teacher_draft_logits_bf16[
                    depth : depth + rows
                ],
            )
        )
    return TttBatch(
        sample_id=sample.sample_id,
        aux_layer_inputs=np.ascontiguousarray(sample.aux_layer_inputs[:rows]),
        depths=tuple(depths),
    )


class FeatureStore:
    def __init__(self, root: Path | str, *, verify_hashes: bool = True):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeatureFormatError(f"cannot load {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise FeatureFormatError("manifest.json root must be an object")
        expected_fields = {
            "schema": SCHEMA_ID,
            "positional_contract": POSITIONAL_CONTRACT,
            "endianness": "little",
            "hidden_size": HIDDEN_SIZE,
            "layer_input_ids": TARGET_LAYER_INPUT_IDS,
            "auxiliary_width": AUX_WIDTH,
            "draft_vocab_size": DRAFT_VOCAB_SIZE,
            "target_vocab_size": TARGET_VOCAB_SIZE,
        }
        for name, expected in expected_fields.items():
            if manifest.get(name) != expected:
                raise FeatureFormatError(
                    f"manifest {name}={manifest.get(name)!r}, expected {expected!r}"
                )
        mapping_sha = manifest.get("draft_mapping_sha256")
        if (
            not isinstance(mapping_sha, str)
            or len(mapping_sha) != 64
            or any(character not in "0123456789abcdef" for character in mapping_sha)
        ):
            raise FeatureFormatError(
                "manifest draft_mapping_sha256 must be a 64-character digest"
            )
        for digest_name in ("target_model_sha256", "eagle3_checkpoint_sha256"):
            digest = manifest.get(digest_name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise FeatureFormatError(
                    f"manifest {digest_name} must be a 64-character digest"
                )
        mapping_record = manifest.get("draft_mapping")
        if not isinstance(mapping_record, dict):
            raise FeatureFormatError("manifest draft_mapping must be an array record")
        if (
            mapping_record.get("file") != "draft_to_target.u32le"
            or mapping_record.get("dtype") != "uint32"
            or mapping_record.get("shape") != [DRAFT_VOCAB_SIZE]
            or mapping_record.get("sha256") != mapping_sha
        ):
            raise FeatureFormatError("manifest draft_mapping metadata is invalid")
        mapping_path = (self.root / "draft_to_target.u32le").resolve()
        if (
            self.root not in mapping_path.parents
            or mapping_path.stat().st_size != 4 * DRAFT_VOCAB_SIZE
        ):
            raise FeatureFormatError("draft_to_target.u32le has invalid path or size")
        if _sha256_file(mapping_path) != mapping_sha:
            raise FeatureFormatError("draft_to_target.u32le SHA-256 mismatch")
        draft_to_target = np.fromfile(mapping_path, dtype="<u4")
        if (
            draft_to_target.shape != (DRAFT_VOCAB_SIZE,)
            or np.any(draft_to_target >= TARGET_VOCAB_SIZE)
            or np.any(draft_to_target[1:] <= draft_to_target[:-1])
        ):
            raise FeatureFormatError("draft_to_target.u32le is not a strict 32K mapping")
        self.draft_to_target = draft_to_target
        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise FeatureFormatError("manifest samples must be a non-empty array")
        self.manifest = manifest
        self.sample_index = tuple(self._validate_index(samples))
        self.verify_hashes = verify_hashes

    def _validate_index(self, samples: list[Any]) -> Iterator[Mapping[str, Any]]:
        seen: set[str] = set()
        for ordinal, entry in enumerate(samples):
            if not isinstance(entry, dict):
                raise FeatureFormatError(f"manifest samples[{ordinal}] is not an object")
            sample_id = entry.get("id")
            relative = entry.get("path")
            length = entry.get("length")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise FeatureFormatError(f"invalid/duplicate sample id at index {ordinal}")
            if not isinstance(relative, str) or not relative:
                raise FeatureFormatError(f"sample {sample_id!r} has invalid path")
            if not isinstance(length, int) or length <= 0:
                raise FeatureFormatError(f"sample {sample_id!r} has invalid length")
            resolved = (self.root / relative).resolve()
            if self.root not in resolved.parents:
                raise FeatureFormatError(f"sample {sample_id!r} path escapes feature root")
            seen.add(sample_id)
            yield {"id": sample_id, "path": relative, "length": length}

    def __len__(self) -> int:
        return len(self.sample_index)

    def load(self, index: int, *, max_length: int | None = None) -> FeatureSample:
        entry = self.sample_index[index]
        sample_dir = (self.root / str(entry["path"])).resolve()
        meta_path = sample_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeatureFormatError(f"cannot load {meta_path}: {error}") from error
        if not isinstance(meta, dict):
            raise FeatureFormatError(f"{meta_path} root must be an object")
        if (
            meta.get("schema") != SCHEMA_ID
            or meta.get("positional_contract") != POSITIONAL_CONTRACT
            or meta.get("id") != entry["id"]
            or meta.get("draft_vocab_size") != DRAFT_VOCAB_SIZE
            or meta.get("draft_mapping_sha256")
            != self.manifest["draft_mapping_sha256"]
        ):
            raise FeatureFormatError(f"sample metadata identity mismatch in {meta_path}")
        length = meta.get("length")
        bootstrap = meta.get("bootstrap_rows_masked")
        if length != entry["length"] or bootstrap != 0:
            raise FeatureFormatError(f"sample metadata geometry mismatch in {meta_path}")
        records_raw = meta.get("arrays")
        if not isinstance(records_raw, list):
            raise FeatureFormatError(f"sample {entry['id']!r} arrays must be a list")
        records: dict[str, Mapping[str, Any]] = {}
        for record in records_raw:
            if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                raise FeatureFormatError(f"sample {entry['id']!r} has invalid array record")
            filename = record["file"]
            if filename in records:
                raise FeatureFormatError(f"sample {entry['id']!r} repeats {filename!r}")
            records[filename] = record
        missing = set(_REQUIRED_ARRAYS) - set(records)
        if missing:
            raise FeatureFormatError(
                f"sample {entry['id']!r} is missing arrays {sorted(missing)}"
            )
        array_specs = dict(_REQUIRED_ARRAYS)
        array_specs.update(
            {name: spec for name, spec in _OPTIONAL_ARRAYS.items() if name in records}
        )
        for filename, (dtype, shape_pattern) in array_specs.items():
            record = records[filename]
            shape = record.get("shape")
            expected_shape = [length if dim is None else dim for dim in shape_pattern]
            if record.get("dtype") != dtype or shape != expected_shape:
                raise FeatureFormatError(
                    f"sample {entry['id']!r} {filename} metadata is "
                    f"dtype={record.get('dtype')!r}, shape={shape!r}; expected "
                    f"dtype={dtype!r}, shape={expected_shape!r}"
                )
            path = (sample_dir / filename).resolve()
            if sample_dir not in path.parents:
                raise FeatureFormatError(f"array path {filename!r} escapes sample directory")
            item_bytes = {
                "bfloat16": 2,
                "uint32": 4,
                "float32": 4,
                "uint8": 1,
            }[dtype]
            expected_bytes = int(np.prod(expected_shape)) * item_bytes
            if path.stat().st_size != expected_bytes:
                raise FeatureFormatError(
                    f"sample {entry['id']!r} {filename} is {path.stat().st_size} bytes, "
                    f"expected {expected_bytes}"
                )
            if self.verify_hashes and _sha256_file(path) != record.get("sha256"):
                raise FeatureFormatError(
                    f"sample {entry['id']!r} {filename} SHA-256 mismatch"
                )

        limit = length if max_length is None else min(length, max_length)
        if limit <= 0:
            raise FeatureFormatError("max_length must be positive")

        def raw(filename: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
            mapped = np.memmap(sample_dir / filename, mode="r", dtype=dtype, shape=shape)
            return np.asarray(mapped[:limit]).copy()

        def mapped_raw(filename: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
            mapped = np.memmap(sample_dir / filename, mode="r", dtype=dtype, shape=shape)
            return mapped[:limit]

        input_ids = raw("input_ids.u32le", "<u4", (length,))
        labels = raw("labels.u32le", "<u4", (length,))
        target_argmax = raw("target_argmax.u32le", "<u4", (length,))
        loss_mask = raw("loss_mask.u8", "u1", (length,))
        aux = _bf16_to_float32(
            raw("aux_layer_inputs.bf16le", "<u2", (length, AUX_WIDTH))
        )
        hidden = _bf16_to_float32(
            raw("hidden_state.bf16le", "<u2", (length, HIDDEN_SIZE))
        )
        input_embedding = _bf16_to_float32(
            raw("input_embedding.bf16le", "<u2", (length, HIDDEN_SIZE))
        )
        next_embedding = _bf16_to_float32(
            raw("next_token_embedding.bf16le", "<u2", (length, HIDDEN_SIZE))
        )
        teacher_logits_bf16 = mapped_raw(
            "teacher_draft_logits.bf16le", "<u2", (length, DRAFT_VOCAB_SIZE)
        )
        teacher_logsumexp = (
            raw("teacher_logsumexp.f32le", "<f4", (length,))
            if "teacher_logsumexp.f32le" in records
            else None
        )

        if np.any(loss_mask > 1):
            raise FeatureFormatError(f"sample {entry['id']!r} loss_mask is not boolean")
        if np.any(input_ids >= TARGET_VOCAB_SIZE):
            raise FeatureFormatError(f"sample {entry['id']!r} input_ids exceed target vocab")
        valid_labels = labels != INVALID_TOKEN_ID
        valid_targets = target_argmax != INVALID_TOKEN_ID
        if np.any(labels[valid_labels] >= TARGET_VOCAB_SIZE):
            raise FeatureFormatError(f"sample {entry['id']!r} labels exceed target vocab")
        if np.any(target_argmax[valid_targets] >= TARGET_VOCAB_SIZE):
            raise FeatureFormatError(
                f"sample {entry['id']!r} target_argmax exceeds target vocab"
            )
        for name, values in (
            ("aux_layer_inputs", aux),
            ("hidden_state", hidden),
            ("input_embedding", input_embedding),
            ("next_token_embedding", next_embedding),
        ):
            if not np.all(np.isfinite(values)):
                raise FeatureFormatError(
                    f"sample {entry['id']!r} {name} contains non-finite values"
                )
        # Fail on NaN/Inf without inflating the whole [T, 32K] teacher to FP32.
        for start in range(0, limit, 32):
            bits = np.asarray(teacher_logits_bf16[start : start + 32], dtype=np.uint16)
            if np.any((bits & np.uint16(0x7F80)) == np.uint16(0x7F80)):
                raise FeatureFormatError(
                    f"sample {entry['id']!r} teacher_draft_logits contains non-finite values"
                )
        if teacher_logsumexp is not None:
            finite_prefix = teacher_logsumexp[:-1] if limit == length else teacher_logsumexp
            if not np.all(np.isfinite(finite_prefix)):
                raise FeatureFormatError(
                    f"sample {entry['id']!r} teacher_logsumexp contains invalid rows"
                )

        # The v1 positional contract is deliberately one row ahead of the raw
        # target capture: labels[P] is token[P+2], not token[P+1].
        if limit > 2 and not np.array_equal(labels[: limit - 2], input_ids[2:limit]):
            raise FeatureFormatError(
                f"sample {entry['id']!r} labels[P] do not equal input_ids[P+2]"
            )
        if limit > 1 and not np.array_equal(
            next_embedding[: limit - 1], input_embedding[1:limit]
        ):
            raise FeatureFormatError(
                f"sample {entry['id']!r} next embedding is not input_embedding[P+1]"
            )
        if limit == length:
            if (
                length < 2
                or np.any(labels[-2:] != INVALID_TOKEN_ID)
                or loss_mask[-1] != 0
                or target_argmax[-1] != INVALID_TOKEN_ID
                or np.any(np.asarray(teacher_logits_bf16[-1]) != 0)
                or np.any(next_embedding[-1] != 0)
                or (
                    teacher_logsumexp is not None
                    and not np.isnan(teacher_logsumexp[-1])
                )
            ):
                raise FeatureFormatError(
                    f"sample {entry['id']!r} shifted final rows must be sentinel/masked"
                )
            if bootstrap > length or np.any(loss_mask[:bootstrap] != 0):
                raise FeatureFormatError(
                    f"sample {entry['id']!r} does not mask its bootstrap rows"
                )
        return FeatureSample(
            sample_id=str(entry["id"]),
            input_ids=input_ids,
            labels=labels,
            target_argmax=target_argmax,
            loss_mask=loss_mask.astype(np.bool_),
            aux_layer_inputs=aux,
            hidden_state=hidden,
            input_embedding=input_embedding,
            next_token_embedding=next_embedding,
            teacher_draft_logits_bf16=teacher_logits_bf16,
            teacher_logsumexp=teacher_logsumexp,
            bootstrap_rows_masked=bootstrap,
        )
