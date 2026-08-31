"""Strict MLX-vs-Camelid Metal parity gate for one recurrent EAGLE cell.

The companion Rust example emits synthetic inputs and Camelid Metal outputs.
This process loads only the 15-tensor draft head; it never loads a target model
or consumes a prompt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from .contract import (
    DRAFT_VOCAB_SIZE,
    HIDDEN_SIZE,
    read_and_validate_mapping,
    sha256_file,
    validate_checkpoint,
)
from .features import AUX_WIDTH
from .model import Eagle3Head, require_mlx

try:
    import mlx.core as mx
except ModuleNotFoundError:
    mx = None


SCHEMA = "camelid-eagle3-cell-parity-v1"
TOP_K = 8


class CellParityError(ValueError):
    pass


@dataclass(frozen=True)
class ArraySpec:
    dtype_name: str
    numpy_dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Fixture:
    root: Path
    manifest: dict[str, Any]
    aux: np.ndarray
    embeddings: np.ndarray
    camelid_fused: np.ndarray
    camelid_depth0_hidden: np.ndarray
    camelid_recurrent_last_hidden: np.ndarray
    camelid_selected_draft_ids: np.ndarray
    camelid_top_draft_ids: np.ndarray
    camelid_top_target_ids: np.ndarray
    camelid_top_logits: np.ndarray


def _load_fixture(root: Path) -> Fixture:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CellParityError(f"cannot load {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise CellParityError(f"fixture must use schema {SCHEMA!r}")
    rows = manifest.get("rows")
    depths = manifest.get("depths")
    expected_scalars = {
        "generator": "splitmix64-f32-v1",
        "hidden_size": HIDDEN_SIZE,
        "auxiliary_width": AUX_WIDTH,
        "draft_vocab_size": DRAFT_VOCAB_SIZE,
        "top_k": TOP_K,
        "camelid_lane": "dense-bf16-weights-f32-activations-f16-kv",
    }
    for name, expected in expected_scalars.items():
        if manifest.get(name) != expected:
            raise CellParityError(
                f"fixture {name}={manifest.get(name)!r}, expected {expected!r}"
            )
    if not isinstance(manifest.get("camelid_version"), str):
        raise CellParityError("fixture camelid_version must be a string")
    for digest_name in (
        "checkpoint_weights_sha256",
        "checkpoint_config_sha256",
        "fixture_binary_sha256",
    ):
        digest = manifest.get(digest_name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CellParityError(f"fixture {digest_name} is not a lowercase SHA-256")
    if not isinstance(rows, int) or rows < 2:
        raise CellParityError("fixture rows must be an integer >=2")
    if not isinstance(depths, int) or not 2 <= depths <= 7:
        raise CellParityError("fixture depths must be in 2..=7")

    specs = {
        "input_aux.f32le": ArraySpec("float32", "<f4", (rows, AUX_WIDTH)),
        "input_embeddings.f32le": ArraySpec(
            "float32", "<f4", (depths, rows, HIDDEN_SIZE)
        ),
        "camelid_fused.f32le": ArraySpec(
            "float32", "<f4", (rows, HIDDEN_SIZE)
        ),
        "camelid_depth0_hidden.f32le": ArraySpec(
            "float32", "<f4", (rows, HIDDEN_SIZE)
        ),
        "camelid_recurrent_last_hidden.f32le": ArraySpec(
            "float32", "<f4", (depths - 1, HIDDEN_SIZE)
        ),
        "camelid_selected_draft_ids.u32le": ArraySpec(
            "uint32", "<u4", (depths,)
        ),
        "camelid_top_draft_ids.u32le": ArraySpec(
            "uint32", "<u4", (depths, TOP_K)
        ),
        "camelid_top_target_ids.u32le": ArraySpec(
            "uint32", "<u4", (depths, TOP_K)
        ),
        "camelid_top_logits.f32le": ArraySpec(
            "float32", "<f4", (depths, TOP_K)
        ),
    }
    records_raw = manifest.get("arrays")
    if not isinstance(records_raw, list):
        raise CellParityError("fixture arrays must be a list")
    records = {
        record.get("file"): record
        for record in records_raw
        if isinstance(record, dict) and isinstance(record.get("file"), str)
    }
    if set(records) != set(specs) or len(records_raw) != len(records):
        raise CellParityError(
            f"fixture array set mismatch: missing={sorted(set(specs) - set(records))}, "
            f"extra={sorted(set(records) - set(specs))}"
        )

    loaded: dict[str, np.ndarray] = {}
    for filename, spec in specs.items():
        record = records[filename]
        if record.get("dtype") != spec.dtype_name or record.get("shape") != list(
            spec.shape
        ):
            raise CellParityError(f"fixture metadata mismatch for {filename}")
        path = (root / filename).resolve()
        if root not in path.parents:
            raise CellParityError(f"fixture payload {filename} escapes its root")
        expected_bytes = int(np.prod(spec.shape)) * 4
        if path.stat().st_size != expected_bytes:
            raise CellParityError(
                f"fixture payload {filename} has {path.stat().st_size} bytes, "
                f"expected {expected_bytes}"
            )
        if sha256_file(path) != record.get("sha256"):
            raise CellParityError(f"fixture payload {filename} SHA-256 mismatch")
        loaded[filename] = np.fromfile(path, dtype=spec.numpy_dtype).reshape(
            spec.shape
        )

    for filename, values in loaded.items():
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise CellParityError(f"fixture payload {filename} contains NaN/Inf")

    selected = loaded["camelid_selected_draft_ids.u32le"]
    top_draft = loaded["camelid_top_draft_ids.u32le"]
    top_logits = loaded["camelid_top_logits.f32le"]
    if np.any(top_draft >= DRAFT_VOCAB_SIZE) or np.any(selected >= DRAFT_VOCAB_SIZE):
        raise CellParityError("Camelid fixture contains an out-of-range draft id")
    if np.any(selected != top_draft[:, 0]):
        raise CellParityError("Camelid selected draft id differs from retained top-1")
    for depth in range(depths):
        if np.unique(top_draft[depth]).size != TOP_K:
            raise CellParityError(f"Camelid candidates at depth {depth} contain duplicates")
        order = np.lexsort((top_draft[depth], -top_logits[depth]))
        if not np.array_equal(order, np.arange(TOP_K)):
            raise CellParityError(
                f"Camelid candidates at depth {depth} are not canonically ordered"
            )

    return Fixture(
        root=root,
        manifest=manifest,
        aux=loaded["input_aux.f32le"],
        embeddings=loaded["input_embeddings.f32le"],
        camelid_fused=loaded["camelid_fused.f32le"],
        camelid_depth0_hidden=loaded["camelid_depth0_hidden.f32le"],
        camelid_recurrent_last_hidden=loaded[
            "camelid_recurrent_last_hidden.f32le"
        ],
        camelid_selected_draft_ids=selected,
        camelid_top_draft_ids=top_draft,
        camelid_top_target_ids=loaded["camelid_top_target_ids.u32le"],
        camelid_top_logits=top_logits,
    )


def _numeric_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    difference = np.abs(actual64 - expected64)
    denominator = float(np.linalg.norm(actual64) * np.linalg.norm(expected64))
    cosine = (
        float(np.dot(actual64.ravel(), expected64.ravel()) / denominator)
        if denominator > 0.0
        else float(actual64.size == 0 or np.array_equal(actual64, expected64))
    )
    return {
        "max_abs": float(np.max(difference, initial=0.0)),
        "mean_abs": float(np.mean(difference)) if difference.size else 0.0,
        "cosine": cosine,
    }


def _top_k(logits: np.ndarray, count: int) -> np.ndarray:
    ids = np.arange(logits.size, dtype=np.int64)
    return np.lexsort((ids, -np.asarray(logits, dtype=np.float64)))[:count]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--parameter-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--max-fused-abs", type=float, default=2.0e-2)
    parser.add_argument("--max-hidden-abs", type=float, default=8.0e-2)
    parser.add_argument("--max-hidden-mean-abs", type=float, default=8.0e-3)
    parser.add_argument("--min-hidden-cosine", type=float, default=0.9999)
    parser.add_argument("--max-top-logit-abs", type=float, default=1.0e-1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "max_fused_abs",
        "max_hidden_abs",
        "max_hidden_mean_abs",
        "max_top_logit_abs",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.0 <= args.min_hidden_cosine <= 1.0:
        raise ValueError("--min-hidden-cosine must be in [0,1]")
    require_mlx()
    if not mx.metal.is_available():
        raise RuntimeError("cell parity requires MLX Metal on Apple Silicon")

    checkpoint = validate_checkpoint(args.warm_start)
    fixture = _load_fixture(args.fixture)
    if fixture.manifest.get("checkpoint_weights_sha256") != checkpoint.weights_sha256:
        raise CellParityError("fixture and MLX warm start use different checkpoint weights")
    if fixture.manifest.get("checkpoint_config_sha256") != checkpoint.config_sha256:
        raise CellParityError("fixture and MLX warm start use different checkpoint config")
    if fixture.manifest.get("rope_theta") != checkpoint.config.get("rope_theta"):
        raise CellParityError("fixture and checkpoint RoPE bases differ")
    if fixture.manifest.get("sliding_window") != checkpoint.config.get(
        "sliding_window"
    ):
        raise CellParityError("fixture and checkpoint attention windows differ")

    draft_to_target, _mask, mapping_sha = read_and_validate_mapping(
        checkpoint.checkpoint_dir / "model.safetensors",
        checkpoint.payload_start,
        checkpoint.descriptors,
    )
    if not np.array_equal(
        draft_to_target[fixture.camelid_top_draft_ids],
        fixture.camelid_top_target_ids,
    ):
        raise CellParityError("fixture target ids do not match the checkpoint d2t mapping")

    model = Eagle3Head.from_checkpoint(
        checkpoint, parameter_dtype=args.parameter_dtype
    )
    model.eval()
    compute_dtype = model.fc.weight.dtype
    fused = model.project_aux(mx.array(fixture.aux).astype(compute_dtype)[None, ...])
    mx.eval(fused)
    fused_np = np.asarray(fused[0], dtype=np.float32)
    fused_metrics = _numeric_metrics(fused_np, fixture.camelid_fused)

    hidden = fused
    key_cache: list = []
    value_cache: list = []
    hidden_metrics: list[dict[str, Any]] = []
    actual_top_ids: list[list[int]] = []
    actual_top_logits: list[list[float]] = []
    top_id_match: list[bool] = []
    top_logit_max_abs: list[float] = []
    for depth in range(int(fixture.manifest["depths"])):
        embeddings = mx.array(fixture.embeddings[depth]).astype(compute_dtype)[
            None, ...
        ]
        hidden = model.recurrent_step(
            embeddings, hidden, key_cache, value_cache, depth=depth
        )
        logits = model.logits_at(hidden, mx.array([fixture.manifest["rows"] - 1]))[
            0
        ]
        mx.eval(hidden, logits)
        hidden_np = np.asarray(hidden[0], dtype=np.float32)
        expected_hidden = (
            fixture.camelid_depth0_hidden
            if depth == 0
            else fixture.camelid_recurrent_last_hidden[depth - 1]
        )
        actual_hidden = hidden_np if depth == 0 else hidden_np[-1]
        hidden_metrics.append(
            {
                "depth": depth,
                **_numeric_metrics(actual_hidden, expected_hidden),
            }
        )
        logits_np = np.asarray(logits, dtype=np.float32)
        if not np.all(np.isfinite(logits_np)):
            raise CellParityError(f"MLX logits at depth {depth} contain NaN/Inf")
        top = _top_k(logits_np, TOP_K)
        top_ids = top.astype(np.uint32)
        top_values = logits_np[top]
        actual_top_ids.append([int(value) for value in top_ids])
        actual_top_logits.append([float(value) for value in top_values])
        top_id_match.append(
            bool(np.array_equal(top_ids, fixture.camelid_top_draft_ids[depth]))
        )
        top_logit_max_abs.append(
            float(np.max(np.abs(top_values - fixture.camelid_top_logits[depth])))
        )

    numeric_pass = (
        fused_metrics["max_abs"] <= args.max_fused_abs
        and all(
            metric["max_abs"] <= args.max_hidden_abs
            and metric["mean_abs"] <= args.max_hidden_mean_abs
            and metric["cosine"] >= args.min_hidden_cosine
            for metric in hidden_metrics
        )
        and all(value <= args.max_top_logit_abs for value in top_logit_max_abs)
    )
    ranking_pass = all(top_id_match) and all(
        actual_top_ids[depth][0]
        == int(fixture.camelid_selected_draft_ids[depth])
        for depth in range(len(actual_top_ids))
    )
    passed = bool(numeric_pass and ranking_pass)
    report = {
        "schema": "camelid-eagle3-cell-parity-receipt-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "checkpoint_config_sha256": checkpoint.config_sha256,
        "checkpoint_mapping_sha256": mapping_sha,
        "fixture_manifest_sha256": sha256_file(fixture.root / "manifest.json"),
        "fixture_binary_sha256": fixture.manifest["fixture_binary_sha256"],
        "camelid_version": fixture.manifest["camelid_version"],
        "parameter_dtype": args.parameter_dtype,
        "mlx_version": getattr(sys.modules.get("mlx"), "__version__", "unknown"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "thresholds": {
            "max_fused_abs": args.max_fused_abs,
            "max_hidden_abs": args.max_hidden_abs,
            "max_hidden_mean_abs": args.max_hidden_mean_abs,
            "min_hidden_cosine": args.min_hidden_cosine,
            "max_top_logit_abs": args.max_top_logit_abs,
            "ordered_top_k_required": True,
        },
        "fused": fused_metrics,
        "hidden": hidden_metrics,
        "top_id_match": top_id_match,
        "top_logit_max_abs": top_logit_max_abs,
        "camelid_top_draft_ids": fixture.camelid_top_draft_ids.tolist(),
        "mlx_top_draft_ids": actual_top_ids,
        "mlx_top_logits": actual_top_logits,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        receipt = args.receipt.resolve()
        if receipt.exists():
            raise FileExistsError(f"refusing to overwrite parity receipt {receipt}")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(rendered)
    print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
