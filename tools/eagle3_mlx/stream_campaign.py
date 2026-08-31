#!/usr/bin/env python3
"""Run bounded exact-Q4 feature export and resumable MLX EAGLE training shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence


CAMPAIGN_SCHEMA = "camelid-eagle3-stream-campaign-v1"
PROGRESS_SCHEMA = "camelid-eagle3-stream-progress-v1"
COMPLETE_SCHEMA = "camelid-eagle3-stream-complete-v1"
TARGET_SHA256 = "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff"
FEATURE_SCHEMA = "camelid-eagle3-q4-features-v1"
MATERIALIZATION_COMPLETE_SCHEMA = "camelid-eagle3-corpus-materialization-complete-v1"
MATERIALIZATION_RUN_SCHEMA = "camelid-eagle3-corpus-materialization-run-v1"
MATERIALIZATION_SHARD_SCHEMA = "camelid-eagle3-corpus-materialization-shard-v1"
TRAINING_RECEIPT_SCHEMA = "camelid-eagle3-mlx-training-receipt-v1"
STATE_SCHEMA = "camelid-eagle3-mlx-state-v2"

EXPORT_ENV = {
    "CAMELID_EAGLE3_BATCH_AUTHORITATIVE_KV": "1",
    "CAMELID_EAGLE3_BODY_Q8": "1",
    "CAMELID_EAGLE3_LM_HEAD_Q8": "1",
    "CAMELID_KQUANT_MMA": "1",
    "CAMELID_KQUANT_V2": "1",
    "CAMELID_KQUANT_V3": "1",
    "CAMELID_KQUANT_V4": "1",
    "CAMELID_KQUANT_V4_DIRECT_FRAGMENT": "1",
    "CAMELID_METAL_ATTN2": "1",
    "CAMELID_METAL_ATTN_BATCH_K": "1",
    "CAMELID_METAL_F32Y": "1",
    "CAMELID_METAL_KQUANT": "1",
    "CAMELID_METAL_KV_DTYPE": "f16",
    "CAMELID_METAL_LINEAR": "1",
    "CAMELID_METAL_NOCOPY": "1",
    "CAMELID_METAL_Q8": "1",
    "CAMELID_METAL_RESIDENT_DECODE": "1",
    "CAMELID_METAL_RESIDENT_PREFILL": "1",
    "CAMELID_METAL_WIRE": "1",
    "CAMELID_METAL_WIRE_NSG8": "1",
    "CAMELID_SPEC_TREE": "1",
}


class CampaignError(RuntimeError):
    """A sealed streaming-campaign invariant failed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise CampaignError(f"{label} SHA-256 is {actual}, expected {expected}")


def discover_materialized_shards(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete_path = root / "COMPLETE.json"
    run_path = root / "run.json"
    complete = load_json(complete_path)
    run = load_json(run_path)
    if complete.get("schema") != MATERIALIZATION_COMPLETE_SCHEMA:
        raise CampaignError("training materialization is not complete")
    if run.get("schema") != MATERIALIZATION_RUN_SCHEMA:
        raise CampaignError("unsupported materialization run schema")
    if complete.get("run_manifest_sha256") != sha256_file(run_path):
        raise CampaignError("materialization run hash differs from COMPLETE.json")
    if run.get("target", {}).get("gguf_sha256") != TARGET_SHA256:
        raise CampaignError("materialization did not use the pinned exact-Q4 target")

    shard_dirs = sorted(path for path in root.glob("shard-*") if path.is_dir())
    if len(shard_dirs) != complete.get("shards"):
        raise CampaignError("materialized shard count differs from COMPLETE.json")
    expected_manifest_hashes = complete.get("shard_manifest_sha256")
    if not isinstance(expected_manifest_hashes, list) or len(expected_manifest_hashes) != len(shard_dirs):
        raise CampaignError("COMPLETE.json has an invalid shard-manifest hash list")
    shards: list[dict[str, Any]] = []
    total_records = 0
    for index, (shard_dir, expected_manifest_hash) in enumerate(
        zip(shard_dirs, expected_manifest_hashes)
    ):
        manifest_path = shard_dir / "manifest.json"
        require_hash(manifest_path, expected_manifest_hash, f"materialized shard {index} manifest")
        manifest = load_json(manifest_path)
        if manifest.get("schema") != MATERIALIZATION_SHARD_SCHEMA or manifest.get("index") != index:
            raise CampaignError(f"invalid materialized shard manifest at index {index}")
        record_entry = manifest.get("records")
        if not isinstance(record_entry, dict) or record_entry.get("file") != "records.jsonl":
            raise CampaignError(f"materialized shard {index} does not seal records.jsonl")
        records_path = shard_dir / "records.jsonl"
        require_hash(records_path, record_entry.get("sha256"), f"materialized shard {index} records")
        records = int(record_entry.get("records", 0))
        if records <= 0:
            raise CampaignError(f"materialized shard {index} has no records")
        total_records += records
        shards.append(
            {
                "index": index,
                "dir": str(shard_dir.resolve()),
                "records_path": str(records_path.resolve()),
                "records": records,
                "manifest_sha256": expected_manifest_hash,
                "records_sha256": record_entry["sha256"],
            }
        )
    if total_records != complete.get("records") or total_records != run.get("source", {}).get("records"):
        raise CampaignError("materialized record totals disagree")
    seal = {
        "complete_sha256": sha256_file(complete_path),
        "run_sha256": sha256_file(run_path),
        "records": total_records,
        "shards": len(shards),
        "source_jobs_sha256": run["source"]["sha256"],
        "binary_sha256": run["runtime"]["binary_sha256"],
        "camelid_commit": run["runtime"]["camelid_commit"],
    }
    return shards, seal


def ensure_new_workspace(workspace: Path) -> None:
    if workspace.exists():
        if not workspace.is_dir() or any(workspace.iterdir()):
            raise CampaignError(f"workspace must be new or empty: {workspace}")
    else:
        workspace.mkdir(parents=True)
    for name in ("features", "states", "checkpoints", "logs", "receipts"):
        (workspace / name).mkdir()


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json({"command": command, "cwd": str(cwd)}) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(canonical_json({"exit_code": result.returncode}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode != 0:
        raise CampaignError(f"command failed with exit {result.returncode}; see {log_path}")


def safe_remove_regenerable(path: Path, workspace: Path) -> None:
    resolved = path.resolve()
    root = workspace.resolve()
    if resolved == root or root not in resolved.parents:
        raise CampaignError(f"refusing cleanup outside campaign workspace: {resolved}")
    if resolved.exists():
        if not resolved.is_dir():
            raise CampaignError(f"refusing to remove non-directory campaign artifact: {resolved}")
        shutil.rmtree(resolved)


def verify_feature_store(path: Path, *, expected_samples: int, warm_weights_sha256: str) -> str:
    manifest_path = path / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != FEATURE_SCHEMA:
        raise CampaignError("exported feature store has an unsupported schema")
    if manifest.get("target_model_sha256") != TARGET_SHA256:
        raise CampaignError("exported feature store used the wrong target")
    if manifest.get("eagle3_checkpoint_sha256") != warm_weights_sha256:
        raise CampaignError("exported feature store used the wrong warm checkpoint")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise CampaignError("exported feature-store sample count differs from its shard")
    return sha256_file(manifest_path)


def verify_training_artifacts(
    checkpoint: Path,
    state: Path,
    *,
    expected_start: int,
    expected_end: int,
    expected_eval_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = checkpoint / "training-receipt.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
        raise CampaignError("trainer emitted an unsupported receipt")
    if receipt.get("global_step_start") != expected_start or receipt.get("global_step_end") != expected_end:
        raise CampaignError("trainer receipt has the wrong global-step boundary")
    if receipt.get("target_model_sha256") != TARGET_SHA256:
        raise CampaignError("trainer receipt has the wrong target hash")
    if receipt.get("evaluation_overlap_count") != 0 or receipt.get("quality_gate_eligible") is not True:
        raise CampaignError("trainer receipt is not eligible for held-out checkpoint selection")
    if Path(receipt.get("eval_manifest", "")).resolve() != expected_eval_manifest.resolve():
        raise CampaignError("trainer receipt changed the locked evaluation manifest")
    require_hash(checkpoint / "model.safetensors", receipt["output_weights_sha256"], "serving weights")
    require_hash(checkpoint / "config.json", receipt["output_config_sha256"], "serving config")

    state_json = load_json(state / "state.json")
    if state_json.get("schema") != STATE_SCHEMA or state_json.get("global_step") != expected_end:
        raise CampaignError("serialized training state has the wrong schema or step")
    require_hash(state / "training-model.safetensors", state_json["model_sha256"], "training model")
    require_hash(state / "optimizer.safetensors", state_json["optimizer_sha256"], "optimizer state")
    return receipt, state_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True, type=Path)
    result.add_argument("--camelid-binary", required=True, type=Path)
    result.add_argument("--binary-sha256", required=True)
    result.add_argument("--target-model", required=True, type=Path)
    result.add_argument("--warm-start", required=True, type=Path)
    result.add_argument("--warm-weights-sha256", required=True)
    result.add_argument("--materialized-train", required=True, type=Path)
    result.add_argument("--eval-features", required=True, type=Path)
    result.add_argument("--eval-manifest-sha256", required=True)
    result.add_argument("--workspace", required=True, type=Path)
    result.add_argument("--lock", required=True, type=Path)
    result.add_argument("--python", required=True, type=Path)
    result.add_argument("--learning-rate", type=float, default=3.0e-5)
    result.add_argument("--warmup-steps", type=int, default=36)
    result.add_argument("--loss-row-chunk", type=int, default=4)
    result.add_argument("--max-length", type=int, default=512)
    result.add_argument("--ttt-length", type=int, default=7)
    result.add_argument("--seed", type=int, default=20260830)
    result.add_argument("--log-interval", type=int, default=16)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--stop-after-shards", type=int)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.learning_rate <= 0 or args.warmup_steps < 0 or args.loss_row_chunk <= 0:
            raise CampaignError("invalid training hyperparameters")
        for path, label in (
            (args.source_root, "source root"),
            (args.camelid_binary, "Camelid binary"),
            (args.target_model, "target model"),
            (args.warm_start, "warm checkpoint"),
            (args.materialized_train, "materialized training set"),
            (args.eval_features, "evaluation feature store"),
            (args.lock, "workload lock"),
            (args.python, "Python runtime"),
        ):
            if not path.exists():
                raise CampaignError(f"{label} does not exist: {path}")
        require_hash(args.camelid_binary, args.binary_sha256, "Camelid binary")
        require_hash(args.target_model, TARGET_SHA256, "target model")
        require_hash(args.warm_start / "model.safetensors", args.warm_weights_sha256, "warm weights")
        eval_manifest_path = args.eval_features / "manifest.json"
        require_hash(eval_manifest_path, args.eval_manifest_sha256, "locked eval manifest")
        eval_manifest = load_json(eval_manifest_path)
        if (
            eval_manifest.get("schema") != FEATURE_SCHEMA
            or eval_manifest.get("target_model_sha256") != TARGET_SHA256
            or eval_manifest.get("eagle3_checkpoint_sha256") != args.warm_weights_sha256
        ):
            raise CampaignError("locked evaluation store does not match the campaign target/head")

        shards, materialization_seal = discover_materialized_shards(args.materialized_train)
        total_steps = sum(int(shard["records"]) for shard in shards)
        config = {
            "schema": CAMPAIGN_SCHEMA,
            "created_by_sha256": sha256_file(Path(__file__).resolve()),
            "source_root": str(args.source_root.resolve()),
            "training_tool_sha256": sha256_file(args.source_root / "tools/eagle3_mlx/train.py"),
            "camelid_binary": str(args.camelid_binary.resolve()),
            "binary_sha256": args.binary_sha256,
            "target_model": str(args.target_model.resolve()),
            "target_model_sha256": TARGET_SHA256,
            "warm_start": str(args.warm_start.resolve()),
            "warm_weights_sha256": args.warm_weights_sha256,
            "materialized_train": str(args.materialized_train.resolve()),
            "materialization": materialization_seal,
            "eval_features": str(args.eval_features.resolve()),
            "eval_manifest_sha256": args.eval_manifest_sha256,
            "eval_samples": len(eval_manifest["samples"]),
            "export_environment": EXPORT_ENV,
            "training": {
                "objective": "soft_ce",
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "total_training_steps": total_steps,
                "loss_row_chunk": args.loss_row_chunk,
                "max_length": args.max_length,
                "ttt_length": args.ttt_length,
                "max_grad_norm": 0.5,
                "weight_decay": 0.0,
                "seed": args.seed,
            },
            "shards": shards,
        }
        if args.plan_only:
            print(json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2))
            return 0

        config_path = args.workspace / "campaign.json"
        progress_path = args.workspace / "progress.json"
        if args.resume:
            existing = load_json(config_path)
            if existing != config:
                raise CampaignError("resume campaign configuration differs from the sealed run")
            progress = load_json(progress_path)
            if progress.get("schema") != PROGRESS_SCHEMA:
                raise CampaignError("unsupported progress schema")
        else:
            ensure_new_workspace(args.workspace)
            atomic_json(config_path, config)
            progress = {
                "schema": PROGRESS_SCHEMA,
                "completed_shards": 0,
                "global_step": 0,
                "current_state": None,
                "best": None,
                "history": [],
                "cleanup_pending": [],
            }
            atomic_json(progress_path, progress)

        for pending in progress.get("cleanup_pending", []):
            safe_remove_regenerable(Path(pending), args.workspace)
        progress["cleanup_pending"] = []
        atomic_json(progress_path, progress)

        first_index = int(progress["completed_shards"])
        stop_index = len(shards)
        if args.stop_after_shards is not None:
            if args.stop_after_shards <= 0:
                raise CampaignError("--stop-after-shards must be positive")
            stop_index = min(stop_index, first_index + args.stop_after_shards)

        for shard in shards[first_index:stop_index]:
            index = int(shard["index"])
            tag = f"shard-{index:06d}"
            feature_dir = args.workspace / "features" / tag
            checkpoint_dir = args.workspace / "checkpoints" / tag
            state_dir = args.workspace / "states" / tag
            export_log = args.workspace / "logs" / f"{tag}-export.log"
            train_log = args.workspace / "logs" / f"{tag}-train.log"
            for partial in (feature_dir, checkpoint_dir, state_dir):
                safe_remove_regenerable(partial, args.workspace)
            for partial_log in (export_log, train_log):
                if partial_log.exists():
                    partial_log.unlink()

            export_command = [
                str(args.lock),
                "env",
                *[f"{key}={value}" for key, value in sorted(EXPORT_ENV.items())],
                str(args.camelid_binary),
                "export-eagle3-features",
                str(args.target_model),
                "--eagle3",
                str(args.warm_start),
                "--input",
                shard["records_path"],
                "--output",
                str(feature_dir),
            ]
            run_logged(export_command, cwd=args.source_root, log_path=export_log)
            feature_manifest_sha = verify_feature_store(
                feature_dir,
                expected_samples=int(shard["records"]),
                warm_weights_sha256=args.warm_weights_sha256,
            )

            start = int(progress["global_step"])
            end = start + int(shard["records"])
            train_command = [
                str(args.lock),
                str(args.python),
                "-u",
                "-m",
                "tools.eagle3_mlx.train",
                "--warm-start",
                str(args.warm_start),
                "--features",
                str(feature_dir),
                "--eval-features",
                str(args.eval_features),
                "--output-dir",
                str(checkpoint_dir),
                "--state-output",
                str(state_dir),
                "--max-steps",
                str(shard["records"]),
                "--total-training-steps",
                str(total_steps),
                "--max-length",
                str(args.max_length),
                "--ttt-length",
                str(args.ttt_length),
                "--objective",
                "soft_ce",
                "--loss-row-chunk",
                str(args.loss_row_chunk),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                "0",
                "--max-grad-norm",
                "0.5",
                "--warmup-steps",
                str(args.warmup_steps),
                "--log-interval",
                str(args.log_interval),
                "--eval-interval",
                str(total_steps),
                "--parameter-dtype",
                "float32",
                "--seed",
                str(args.seed),
            ]
            previous_state = progress.get("current_state")
            if previous_state:
                train_command.extend(["--resume-state", str(previous_state)])
            run_logged(train_command, cwd=args.source_root, log_path=train_log)
            receipt, state_json = verify_training_artifacts(
                checkpoint_dir,
                state_dir,
                expected_start=start,
                expected_end=end,
                expected_eval_manifest=eval_manifest_path,
            )

            receipt_copy = args.workspace / "receipts" / f"{tag}-training-receipt.json"
            state_copy = args.workspace / "receipts" / f"{tag}-state.json"
            feature_copy = args.workspace / "receipts" / f"{tag}-feature-manifest.json"
            shutil.copy2(checkpoint_dir / "training-receipt.json", receipt_copy)
            shutil.copy2(state_dir / "state.json", state_copy)
            shutil.copy2(feature_dir / "manifest.json", feature_copy)
            accuracy = float(receipt["final_eval"]["accuracy"])
            old_best = progress.get("best")
            new_is_best = old_best is None or accuracy > float(old_best["accuracy"])
            cleanup: list[str] = [str(feature_dir.resolve())]
            if previous_state:
                cleanup.append(str(Path(previous_state).resolve()))
            if new_is_best:
                if old_best is not None:
                    cleanup.append(str(Path(old_best["checkpoint"]).resolve()))
                best = {
                    "shard": index,
                    "global_step": end,
                    "accuracy": accuracy,
                    "checkpoint": str(checkpoint_dir.resolve()),
                    "weights_sha256": receipt["output_weights_sha256"],
                }
            else:
                cleanup.append(str(checkpoint_dir.resolve()))
                best = old_best
            history_entry = {
                "shard": index,
                "records": shard["records"],
                "global_step_start": start,
                "global_step_end": end,
                "feature_manifest_sha256": feature_manifest_sha,
                "state_model_sha256": state_json["model_sha256"],
                "state_optimizer_sha256": state_json["optimizer_sha256"],
                "serving_weights_sha256": receipt["output_weights_sha256"],
                "heldout_accuracy": accuracy,
                "heldout_correct": receipt["final_eval"]["correct"],
                "heldout_rows": receipt["final_eval"]["rows"],
                "selected_best": new_is_best,
            }
            progress["completed_shards"] = index + 1
            progress["global_step"] = end
            progress["current_state"] = str(state_dir.resolve())
            progress["best"] = best
            progress["history"].append(history_entry)
            progress["cleanup_pending"] = cleanup
            atomic_json(progress_path, progress)
            for cleanup_path in cleanup:
                safe_remove_regenerable(Path(cleanup_path), args.workspace)
            progress["cleanup_pending"] = []
            atomic_json(progress_path, progress)
            print(canonical_json({"event": "shard_complete", **history_entry}), flush=True)

        if int(progress["completed_shards"]) == len(shards):
            completion = {
                "schema": COMPLETE_SCHEMA,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "campaign_sha256": sha256_file(config_path),
                "progress_sha256": sha256_file(progress_path),
                "global_step": progress["global_step"],
                "shards": progress["completed_shards"],
                "best": progress["best"],
                "current_state": progress["current_state"],
            }
            atomic_json(args.workspace / "COMPLETE.json", completion)
            print(canonical_json({"event": "complete", **completion}), flush=True)
        else:
            print(
                canonical_json(
                    {
                        "event": "paused_at_requested_boundary",
                        "completed_shards": progress["completed_shards"],
                        "global_step": progress["global_step"],
                        "best": progress["best"],
                    }
                ),
                flush=True,
            )
        return 0
    except (CampaignError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"stream campaign failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
