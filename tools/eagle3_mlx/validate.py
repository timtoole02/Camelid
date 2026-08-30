"""Validate a warm start and exact-Q4 feature store without loading MLX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import (
    read_and_validate_mapping,
    target_to_draft_index,
    validate_checkpoint,
)
from .features import FeatureStore, build_ttt_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ttt-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-feature-hashes", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = validate_checkpoint(args.warm_start)
    weights_path = checkpoint.checkpoint_dir / "model.safetensors"
    draft_to_target, _mask, mapping_sha = read_and_validate_mapping(
        weights_path, checkpoint.payload_start, checkpoint.descriptors
    )
    inverse = target_to_draft_index(draft_to_target)
    store = FeatureStore(args.features, verify_hashes=not args.skip_feature_hashes)
    if store.manifest["eagle3_checkpoint_sha256"] != checkpoint.weights_sha256:
        raise RuntimeError(
            "features were not exported with the supplied SW512 checkpoint"
        )
    feature_mapping_sha = store.manifest["draft_mapping_sha256"]
    if not np.array_equal(store.draft_to_target, draft_to_target):
        raise RuntimeError(
            "feature draft_to_target rows do not match the warm checkpoint mapping"
        )
    limit = len(store) if args.limit is None else min(len(store), args.limit)
    mapped = np.zeros(args.ttt_length, dtype=np.int64)
    eligible = np.zeros(args.ttt_length, dtype=np.int64)
    rows = 0
    for index in range(limit):
        sample = store.load(index, max_length=args.max_length)
        batch = build_ttt_batch(sample, inverse, ttt_length=args.ttt_length)
        rows += batch.row_count
        for view in batch.depths:
            target_valid = view.target_ids != np.uint32(0xFFFF_FFFF)
            source_mask = sample.loss_mask[
                view.depth : view.depth + batch.row_count
            ]
            eligible[view.depth] += int(np.count_nonzero(source_mask & target_valid))
            mapped[view.depth] += int(np.count_nonzero(view.hard_supervised))
    report = {
        "schema": "camelid-eagle3-mlx-validation-v1",
        "checkpoint": str(checkpoint.checkpoint_dir),
        "weights_sha256": checkpoint.weights_sha256,
        "config_sha256": checkpoint.config_sha256,
        "mapping_sha256": mapping_sha,
        "tensor_count": len(checkpoint.descriptors),
        "feature_schema": store.manifest["schema"],
        "feature_positional_contract": store.manifest["positional_contract"],
        "feature_mapping_sha256": feature_mapping_sha,
        "target_model_sha256": store.manifest.get("target_model_sha256"),
        "samples": limit,
        "rows_per_depth": rows,
        "depths": [
            {
                "depth": depth,
                "eligible": int(eligible[depth]),
                "mapped": int(mapped[depth]),
                "mapping_coverage": float(mapped[depth] / max(eligible[depth], 1)),
            }
            for depth in range(args.ttt_length)
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
