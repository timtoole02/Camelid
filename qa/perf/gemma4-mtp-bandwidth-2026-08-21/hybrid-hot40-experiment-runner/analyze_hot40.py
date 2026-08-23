#!/usr/bin/env python3
"""Fail-closed receipt analyzer for the uniform Hot40 48-token experiment."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TOKENS = 48
EXPECTED_LAYERS = 30
LOGICAL_SLOTS_PER_LAYER = 128
HOT_SLOTS_PER_LAYER = 40
RECORD_PAYLOAD_BYTES = 3_345_408
SLOT_STRIDE_BYTES = 3_358_720
HOT_SLOTS_TOTAL = EXPECTED_LAYERS * HOT_SLOTS_PER_LAYER
HOT_CAPACITY_BYTES = HOT_SLOTS_TOTAL * SLOT_STRIDE_BYTES
LOGICAL_SLOTS_TOTAL = EXPECTED_LAYERS * LOGICAL_SLOTS_PER_LAYER
MAPPED_ADDRESS_SPAN_BYTES = LOGICAL_SLOTS_TOTAL * SLOT_STRIDE_BYTES
HOT_CAPACITY_BYTES_PER_LAYER = HOT_SLOTS_PER_LAYER * SLOT_STRIDE_BYTES
MAPPED_ADDRESS_SPAN_BYTES_PER_LAYER = LOGICAL_SLOTS_PER_LAYER * SLOT_STRIDE_BYTES

WATCHDOG_SCHEMA_VERSION = 3
WATCHDOG_SAMPLE_PERIOD_NS = 250_000_000
BASELINE_SOAK_SECONDS = 60
MIN_BASELINE_HEADROOM_BYTES = 7_680 * 1024**2
WARNING_MIN_BASELINE_HEADROOM_BYTES = 4_608 * 1024**2
MIN_RUNTIME_HEADROOM_BYTES = 2 * 1024**3
MAX_CHILD_FOOTPRINT_BYTES = 7_680 * 1024**2
MAX_HOST_WIRED_BYTES = 8 * 1024**3

REQUEST_SHA256 = "b2f1110079fc726699cc936a628a268a7ec5bf2076fa970899de39d4ea903939"
EXPECTED_IDS_SHA256 = "45e65ac09155d7627373c262f1edd1faf6188fb6dad26c5d5994fe5226a97975"
PROVENANCE_LIMITATION = (
    "Large model, cghost, and assistant contents are not hashed by this run. "
    "Historical SHA-256 claims are carried forward and bound only to live "
    "device/inode/size/mtime stat identity, avoiding pre-spawn page-cache "
    "pollution but providing weaker provenance than a fresh content hash."
)
FULL_Q4_MATRIX_BYTES = 236_077_056
FULL_Q4_BF16_MATRIX_BYTES = 839_385_088
F_RDAHEAD = 45
F_NOCACHE = 48
FULL_Q4_RESIDENCY_MARKER = (
    "[gemma4-mtp full-q4 residency] source_retained=false mapped_bytes=0 "
    "locked_bytes=0 resident_pages=0 total_pages=0 packed_bytes=236077056"
)

EXPERIMENT_ENV = "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT40_EXPERIMENT"
SPARSE_PREDICT_PROBE_ENV = "CAMELID_GEMMA4_GHOST_METAL_SPARSE_PREDICT_PROBE"
PER_LAYER_ENVIRONMENT = {
    "CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER",
    "CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER",
    "CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS",
    "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER",
}
EXPECTED_ENVIRONMENT = {
    "CAMELID_GHOST_ALLOW_LEGACY_SPARSE": "0",
    "CAMELID_GEMMA4_GHOST_METAL": "1",
    "CAMELID_GEMMA4_GHOST_METAL_SLOTS": "1",
    "CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST": "1",
    "CAMELID_GEMMA4_GHOST_METAL_TURBO": "1",
    "CAMELID_GEMMA4_GHOST_METAL_COMMON": "1",
    "CAMELID_GEMMA4_GHOST_METAL_CONTEXT": "1024",
    "CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT": "0",
    "CAMELID_GEMMA4_GHOST_READ_THREADS": "8",
    "CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY": "1",
    "CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS": "1",
    "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS": "32",
    EXPERIMENT_ENV: "1",
    "CAMELID_GEMMA4_GHOST_METAL_DECODE_PROMOTION": "0",
    "CAMELID_GEMMA4_SLOT_PIN": "0",
    "CAMELID_GEMMA4_GHOST_METAL_HOT_PIN": "0",
    "CAMELID_GEMMA4_VICTIM_CACHE": "0",
    "CAMELID_GEMMA4_VICTIM_MB": "0",
    "CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS": "0",
    "CAMELID_GEMMA4_CHAINED_PREDICT": "0",
    "CAMELID_SPEC_DECODE": "off",
    "CAMELID_GEMMA4_SPEC_K1_LANE": "chained",
    "CAMELID_GEMMA4_SPEC_CHUNK_MAX": "8",
    "CAMELID_GEMMA4_SPEC_DRAFT_TOKENS": "8",
    "CAMELID_GEMMA4_MTP_DEVICE_CHAIN": "1",
    "CAMELID_GEMMA4_MTP_FULL_Q4": "1",
    "CAMELID_GEMMA4_MTP_PREFILL_SEED_BOOTSTRAP": "1",
    "CAMELID_GEMMA4_DENSE_K8_GENERIC": "1",
    "CAMELID_GEMMA4_HEAD_SPEC50_K8_COMPACT": "1",
    "CAMELID_GEMMA4_SPEC_TIMING": "1",
    "CAMELID_GEMMA4_GHOST_METAL_TIMING": "1",
    "CAMELID_GEMMA4_ROUTE_TRACE": "1",
}

STAGE_PATTERN = re.compile(
    r"^\[metal chained stages\] split=([^ ]+) "
    r"qkv_o=([0-9]+(?:\.[0-9]+)?)ms "
    r"attn=([0-9]+(?:\.[0-9]+)?)ms "
    r"router=([0-9]+(?:\.[0-9]+)?)ms "
    r"shared=([0-9]+(?:\.[0-9]+)?)ms "
    r"gateup=([0-9]+(?:\.[0-9]+)?)ms "
    r"down=([0-9]+(?:\.[0-9]+)?)ms "
    r"resid=([0-9]+(?:\.[0-9]+)?)ms "
    r"gpu_total=([0-9]+(?:\.[0-9]+)?)ms$",
    re.MULTILINE,
)
FULL_Q4_PATTERN = re.compile(
    r"^\[gemma4-mtp full-q4\] enabled=true source_sha256=([0-9a-f]{64}) "
    r"matrices=(\d+) packed_bytes=(\d+) bf16_matrix_bytes=(\d+) "
    r"quantize_us=(\d+) norms_quantized=false fallback=false$",
    re.MULTILINE,
)
DEVICE_CHAIN_PATTERN = re.compile(
    r"^\[gemma4-mtp device-chain\] requested_drafts=(\d+) returned_drafts=(\d+) "
    r"command_buffers=1 commits=1 waits=1 cpu_embedding_callbacks=0 "
    r"linear_format=([^ ]+) matrix_bytes_per_draft=(\d+) "
    r"encode_us=(\d+) wait_us=(\d+) gpu_us=(\d+) kernel_us=(\d+) wall_us=(\d+)$",
    re.MULTILINE,
)
HOT40_STARTUP_PATTERN = re.compile(
    r"^\[gemma4-ghost-metal\] HYBRID HOT40 EXPERIMENT ACTIVE: .*"
    r"HYBRID_HOT_SLOTS=32 .*HYBRID_HOT40_EXPERIMENT=1, "
    r"layers=30 canonical_addressable=128/layer physical_hot=40/layer "
    r"hot_capacity_slots=1200 hot_capacity_bytes=4030464000 "
    r"mapped_cold_span_bytes=12897484800 verifier_k=1\.\.8 overflow=0 victim=0 "
    r"slot_pin=off prediction=off$",
    re.MULTILINE,
)


class ReceiptError(RuntimeError):
    """A required piece of evidence was absent or inconsistent."""


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite(value: Any, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0.0 if positive else number >= 0.0)


def _pressure_allowed(value: Any, maximum_pressure_level_raw: int) -> bool:
    allowed = {1} if maximum_pressure_level_raw == 1 else {1, 2}
    return maximum_pressure_level_raw in {1, 2} and value in allowed


def _effective_baseline_headroom(allow_warning_pressure: bool) -> int:
    return (
        WARNING_MIN_BASELINE_HEADROOM_BYTES
        if allow_warning_pressure
        else MIN_BASELINE_HEADROOM_BYTES
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReceiptError(f"could not read JSON {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ReceiptError(f"{path}:{line_number} is not a JSON object")
                events.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ReceiptError(f"could not read watchdog JSONL {path}: {error}") from error
    if not events:
        raise ReceiptError("watchdog receipt is empty")
    return events


def _sha256(path: Path) -> str:
    if sys.platform != "darwin":
        raise ReceiptError("F_NOCACHE receipt validation is supported only on macOS")
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"hashed input is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    descriptor = -1
    bytes_read = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        identity_before = os.fstat(descriptor)
        if not stat.S_ISREG(identity_before.st_mode):
            raise ReceiptError(f"hashed input is not a regular file: {path}")
        fcntl.fcntl(descriptor, F_RDAHEAD, 0)
        fcntl.fcntl(descriptor, F_NOCACHE, 1)
        while block := os.read(descriptor, 4 * 1024 * 1024):
            digest.update(block)
            bytes_read += len(block)
        identity_after = os.fstat(descriptor)
        stable_identity = (
            identity_before.st_dev,
            identity_before.st_ino,
            identity_before.st_size,
            identity_before.st_mtime_ns,
        ) == (
            identity_after.st_dev,
            identity_after.st_ino,
            identity_after.st_size,
            identity_after.st_mtime_ns,
        )
        if not stable_identity or bytes_read != identity_before.st_size:
            raise ReceiptError(f"hashed input identity changed while reading: {path}")
    except OSError as error:
        raise ReceiptError(f"could not hash {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _one(events: Iterable[dict[str, Any]], event_name: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("event") == event_name]
    if len(matches) != 1:
        raise ReceiptError(f"watchdog must contain exactly one {event_name!r} event")
    return matches[0]


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ReceiptError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_hashes(run_dir: Path, intent: dict[str, Any]) -> None:
    artifacts = intent.get("artifacts")
    required = {
        "binary",
        "request_source",
        "request_frozen",
        "expected_ids_source",
        "expected_ids_frozen",
        "runner",
        "analyzer",
        "host_sampler",
        "nocache_hasher",
        "hashing_memory_before",
        "hashing_memory_after",
        "watchdog",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ReceiptError("small-artifact hash manifest is incomplete or has extra entries")
    for label in sorted(required):
        record = artifacts[label]
        if not isinstance(record, dict):
            raise ReceiptError(f"intent artifact {label} is not an object")
        path_value = record.get("path")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if (
            set(record) != {"path", "sha256", "size_bytes", "verification"}
            or record.get("verification") != "sha256-this-run"
            or not isinstance(path_value, str)
            or not path_value.startswith("/")
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not _is_nonnegative_int(size)
        ):
            raise ReceiptError(f"intent artifact {label} has an invalid hashed identity")
        path = Path(path_value)
        try:
            actual_size = path.stat().st_size
        except OSError as error:
            raise ReceiptError(f"could not stat intent artifact {label}: {error}") from error
        if path.is_symlink() or not path.is_file() or actual_size != size:
            raise ReceiptError(f"intent artifact {label} file identity drifted")
        if _sha256(path) != digest:
            raise ReceiptError(f"intent artifact {label} SHA-256 drifted")

    if Path(artifacts["request_frozen"]["path"]) != run_dir / "request.json":
        raise ReceiptError("intent does not bind the run-local frozen request")
    if Path(artifacts["expected_ids_frozen"]["path"]) != run_dir / "expected-token-ids.json":
        raise ReceiptError("intent does not bind the run-local frozen token IDs")
    if Path(artifacts["hashing_memory_before"]["path"]) != run_dir / "hashing-memory-before.json":
        raise ReceiptError("intent does not bind the run-local preflight memory sample")
    if Path(artifacts["hashing_memory_after"]["path"]) != run_dir / "hashing-memory-after.json":
        raise ReceiptError("intent does not bind the run-local post-hash memory sample")
    for label in ("request_source", "request_frozen"):
        if artifacts[label]["sha256"] != REQUEST_SHA256:
            raise ReceiptError("request fixture differs from the frozen canonical request")
    for label in ("expected_ids_source", "expected_ids_frozen"):
        if artifacts[label]["sha256"] != EXPECTED_IDS_SHA256:
            raise ReceiptError("expected-token fixture differs from the frozen K1 IDs")

    large_inputs = intent.get("large_inputs")
    if not isinstance(large_inputs, dict) or set(large_inputs) != {
        "model",
        "cghost",
        "assistant",
    }:
        raise ReceiptError("large-input stat manifest is incomplete or has extra entries")
    for label in ("model", "cghost", "assistant"):
        record = large_inputs[label]
        if not isinstance(record, dict) or set(record) != {
            "path",
            "preverified_sha256",
            "size_bytes",
            "stat",
            "verification",
            "content_read_before_spawn",
        }:
            raise ReceiptError(f"large input {label} has an invalid metadata shape")
        path_value = record.get("path")
        digest = record.get("preverified_sha256")
        size = record.get("size_bytes")
        stat_record = record.get("stat")
        if (
            record.get("verification")
            != "historical-sha256-plus-live-stat-no-run-content-read"
            or record.get("content_read_before_spawn") is not False
            or not isinstance(path_value, str)
            or not path_value.startswith("/")
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not _is_nonnegative_int(size)
            or not isinstance(stat_record, dict)
            or set(stat_record) != {"device", "inode", "mtime_seconds"}
            or any(not _is_nonnegative_int(stat_record.get(key)) for key in stat_record)
        ):
            raise ReceiptError(f"large input {label} has invalid historical/stat metadata")
        path = Path(path_value)
        try:
            live = path.stat()
        except OSError as error:
            raise ReceiptError(f"could not stat large input {label}: {error}") from error
        if (
            path.is_symlink()
            or not path.is_file()
            or live.st_size != size
            or live.st_dev != stat_record["device"]
            or live.st_ino != stat_record["inode"]
            or int(live.st_mtime) != stat_record["mtime_seconds"]
        ):
            raise ReceiptError(f"large input {label} live stat identity drifted")
        # Deliberately do not call _sha256(path): large content remains unread
        # by the receipt harness and is bound only by historical SHA + stat.


def _validate_hashing_contract(run_dir: Path, intent: dict[str, Any]) -> None:
    contract = intent.get("hashing_contract")
    allow_warning_pressure = intent["allow_warning_pressure"]
    maximum_pressure_level_raw = 2 if allow_warning_pressure else 1
    minimum_baseline_headroom_bytes = _effective_baseline_headroom(
        allow_warning_pressure
    )
    expected_static = {
        "schema_version": 1,
        "algorithm": "sha256",
        "platform": "darwin",
        "f_rdahead_command": F_RDAHEAD,
        "f_rdahead_value": 0,
        "f_nocache_command": F_NOCACHE,
        "f_nocache_value": 1,
        "read_chunk_bytes": 4 * 1024 * 1024,
        "post_hash_cooldown_seconds": 0,
        "helper_artifact_label": "nocache_hasher",
        "host_sampler_artifact_label": "host_sampler",
        "telemetry_watchdog_artifact_label": "watchdog",
        "telemetry_source": "run_load_only_watchdog.NativeTelemetry.sample_host",
        "minimum_pre_hash_reclaimable_headroom_bytes": minimum_baseline_headroom_bytes,
        "minimum_post_hash_reclaimable_headroom_bytes": minimum_baseline_headroom_bytes,
        "maximum_host_wired_bytes": MAX_HOST_WIRED_BYTES,
        "allow_warning_pressure": allow_warning_pressure,
        "maximum_pressure_level_raw": maximum_pressure_level_raw,
        "reject_swapin_growth": True,
        "reject_swapout_growth": True,
        "large_inputs_content_hashed_this_run": False,
        "large_inputs_content_read_before_spawn": False,
        "large_input_binding": "historical-sha256-plus-live-stat",
        "provenance_limitation": PROVENANCE_LIMITATION,
    }
    provenance_reference = (
        contract.get("provenance_reference") if isinstance(contract, dict) else None
    )
    provenance_reference_sha = (
        contract.get("provenance_reference_sha256")
        if isinstance(contract, dict)
        else None
    )
    if (
        not isinstance(contract, dict)
        or set(contract)
        != set(expected_static)
        | {
            "memory_before",
            "memory_after",
            "provenance_reference",
            "provenance_reference_sha256",
        }
        or any(contract.get(key) != value for key, value in expected_static.items())
        or not isinstance(provenance_reference, str)
        or not provenance_reference
        or not isinstance(provenance_reference_sha, str)
        or (
            provenance_reference_sha != ""
            and re.fullmatch(r"[0-9a-f]{64}", provenance_reference_sha) is None
        )
    ):
        raise ReceiptError("intent hashing contract is absent or inexact")

    boot_identity = intent.get("boot_identity")
    if not isinstance(boot_identity, str) or not boot_identity:
        raise ReceiptError("intent has no boot identity for hash-phase telemetry")
    samples: list[dict[str, Any]] = []
    for phase in ("before", "after"):
        sample = contract.get(f"memory_{phase}")
        if (
            not isinstance(sample, dict)
            or set(sample) != {
                "schema_version",
                "telemetry_source",
                "boot_identity",
                "host",
            }
            or sample.get("schema_version") != 1
            or sample.get("telemetry_source") != expected_static["telemetry_source"]
            or sample.get("boot_identity") != boot_identity
            or not isinstance(sample.get("host"), dict)
        ):
            raise ReceiptError(f"hash-phase {phase} memory sample is invalid")
        host = sample["host"]
        required_nonnegative = {
            "sample_started_monotonic_ns",
            "observed_monotonic_ns",
            "sample_duration_ns",
            "unix_time_ns",
            "page_size_bytes",
            "free_bytes_strict",
            "active_bytes",
            "inactive_bytes",
            "reclaimable_headroom_bytes",
            "wired_bytes",
            "compressor_occupied_bytes",
            "compressed_logical_bytes",
            "swapins_pages",
            "swapouts_pages",
            "swapped_pages_current",
        }
        if (
            any(not _is_nonnegative_int(host.get(field)) for field in required_nonnegative)
            or host.get("page_size_bytes") != 16_384
            or not _pressure_allowed(
                host.get("pressure_level_raw"), maximum_pressure_level_raw
            )
            or host["reclaimable_headroom_bytes"] < minimum_baseline_headroom_bytes
            or host["wired_bytes"] > MAX_HOST_WIRED_BYTES
            or host["observed_monotonic_ns"] < host["sample_started_monotonic_ns"]
        ):
            raise ReceiptError(f"hash-phase {phase} memory safety gate failed")
        samples.append(sample)

    before_host, after_host = (sample["host"] for sample in samples)
    if (
        before_host["swapins_pages"] != after_host["swapins_pages"]
        or before_host["swapouts_pages"] != after_host["swapouts_pages"]
        # The system /usr/bin/python3 on macOS reports time.monotonic_ns()
        # relative to each sampler process.  It is valid only for the
        # start/finish ordering inside one sample, not across the two separate
        # capture_host_memory.py processes.  Wall-clock Unix time plus the
        # boot identity above provides the cross-process order.
        or before_host["unix_time_ns"] >= after_host["unix_time_ns"]
    ):
        raise ReceiptError("small-artifact hashing changed swap or reordered host samples")

    artifacts = intent["artifacts"]
    before_file = _read_json(Path(artifacts["hashing_memory_before"]["path"]))
    after_file = _read_json(Path(artifacts["hashing_memory_after"]["path"]))
    if before_file != samples[0] or after_file != samples[1]:
        raise ReceiptError("intent hash-phase telemetry differs from its bound artifacts")


def _validate_intent(run_dir: Path) -> dict[str, Any]:
    intent = _read_json(run_dir / "intent.json")
    source_commit = intent.get("source_commit") if isinstance(intent, dict) else None
    binary_source_commit = (
        intent.get("binary_source_commit") if isinstance(intent, dict) else None
    )
    harness_commit = intent.get("harness_commit") if isinstance(intent, dict) else None
    binary_source_contract = (
        intent.get("binary_source_contract") if isinstance(intent, dict) else None
    )
    allow_warning_pressure = (
        intent.get("allow_warning_pressure") if isinstance(intent, dict) else None
    )
    contract_expected = {
        "environment": "CAMELID_HOT40_BINARY_SOURCE_COMMIT",
        "canonical_full_commit": True,
        "ancestor_of_harness_commit": True,
        "runtime_source_diff_empty": True,
        "runtime_source_paths": ["src", "Cargo.toml", "Cargo.lock", "build.rs"],
    }
    expected_geometry = {
        "layers": EXPECTED_LAYERS,
        "logical_slots_per_layer": LOGICAL_SLOTS_PER_LAYER,
        "hot_slots_per_layer": HOT_SLOTS_PER_LAYER,
        "anonymous_hot_capacity_slots": HOT_SLOTS_TOTAL,
        "anonymous_hot_capacity_bytes": HOT_CAPACITY_BYTES,
        "file_mapped_addressable_slots": LOGICAL_SLOTS_TOTAL,
        "file_mapped_address_span_bytes": MAPPED_ADDRESS_SPAN_BYTES,
        "mapped_readahead_enabled": False,
        "mapped_readahead_max_inflight_records": 0,
        "overflow_slots": 0,
        "victim_slots": 0,
        "host_cache_budget_bytes": 0,
    }
    if (
        not isinstance(intent, dict)
        or intent.get("schema_version") != 1
        or intent.get("benchmark") != "gemma4-uniform-hot40-experiment"
        or intent.get("expected_tokens") != EXPECTED_TOKENS
        or intent.get("port") != 8189
        or intent.get("protected_port") != 8181
        or intent.get("protected_port_policy") != "never-bound-connected-or-signaled"
        or not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not isinstance(binary_source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", binary_source_commit) is None
        or not isinstance(harness_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", harness_commit) is None
        or source_commit != binary_source_commit
        or not isinstance(allow_warning_pressure, bool)
        or intent.get("source_worktree_clean") is not True
        or intent.get("harness_worktree_clean") is not True
        or intent.get("geometry") != expected_geometry
    ):
        raise ReceiptError("intent does not bind the exact Hot40 experiment")
    if (
        not isinstance(binary_source_contract, dict)
        or set(binary_source_contract)
        != set(contract_expected) | {"defaulted_to_harness_commit"}
        or any(
            binary_source_contract.get(key) != value
            for key, value in contract_expected.items()
        )
        or not isinstance(
            binary_source_contract.get("defaulted_to_harness_commit"), bool
        )
        or (
            binary_source_contract["defaulted_to_harness_commit"]
            and binary_source_commit != harness_commit
        )
    ):
        raise ReceiptError("intent binary/harness source contract is absent or inexact")
    disk = intent.get("disk")
    if (
        not isinstance(disk, dict)
        or disk.get("volume") != "/System/Volumes/Data"
        or disk.get("minimum_available_kib") != 20 * 1024 * 1024
        or not _is_nonnegative_int(disk.get("available_kib"))
        or disk["available_kib"] < disk["minimum_available_kib"]
    ):
        raise ReceiptError("intent does not prove at least 20 GiB Data-volume headroom")
    watchdog = intent.get("watchdog_contract")
    maximum_pressure_level_raw = 2 if allow_warning_pressure else 1
    minimum_baseline_headroom_bytes = _effective_baseline_headroom(
        allow_warning_pressure
    )
    if watchdog != {
        "schema_version": WATCHDOG_SCHEMA_VERSION,
        "sample_period_ns": WATCHDOG_SAMPLE_PERIOD_NS,
        "baseline_soak_seconds": BASELINE_SOAK_SECONDS,
        "minimum_baseline_reclaimable_headroom_bytes": minimum_baseline_headroom_bytes,
        "minimum_runtime_reclaimable_headroom_bytes": MIN_RUNTIME_HEADROOM_BYTES,
        "maximum_child_physical_footprint_bytes": MAX_CHILD_FOOTPRINT_BYTES,
        "maximum_host_wired_bytes": MAX_HOST_WIRED_BYTES,
        "maximum_pressure_level_raw": maximum_pressure_level_raw,
        "reject_swapin_growth": True,
        "reject_swapout_growth": True,
        "require_zero_current_swap": False,
    }:
        raise ReceiptError("intent watchdog contract is not exact")
    _validate_hashes(run_dir, intent)
    _validate_hashing_contract(run_dir, intent)
    return intent


def _validate_environment(environment: Any) -> None:
    if not isinstance(environment, dict):
        raise ReceiptError("watchdog did not receipt the experiment environment")
    if PER_LAYER_ENVIRONMENT.intersection(environment):
        raise ReceiptError("a forbidden per-layer slot override reached the child")
    if SPARSE_PREDICT_PROBE_ENV in environment:
        raise ReceiptError("the no-prediction Hot40 child received a sparse-predict probe")
    expected = dict(EXPECTED_ENVIRONMENT)
    assistant = environment.get("CAMELID_GEMMA4_MTP_ASSISTANT_PATH")
    if not isinstance(assistant, str) or not assistant.startswith("/"):
        raise ReceiptError("the child did not receive an absolute MTP assistant path")
    expected["CAMELID_GEMMA4_MTP_ASSISTANT_PATH"] = assistant
    if environment != expected:
        missing = sorted(set(expected).difference(environment))
        extra = sorted(set(environment).difference(expected))
        mismatched = sorted(
            key for key in set(expected).intersection(environment)
            if environment[key] != expected[key]
        )
        raise ReceiptError(
            f"child experiment environment drifted: missing={missing}, "
            f"extra={extra}, mismatched={mismatched}"
        )


def _validate_watchdog(run_dir: Path, intent: dict[str, Any]) -> dict[str, Any]:
    maximum_pressure_level_raw = 2 if intent["allow_warning_pressure"] else 1
    minimum_baseline_headroom_bytes = _effective_baseline_headroom(
        intent["allow_warning_pressure"]
    )
    events = _read_jsonl(run_dir / "watchdog.jsonl")
    if any(event.get("schema_version") != WATCHDOG_SCHEMA_VERSION for event in events):
        raise ReceiptError("every watchdog record must use schema version 3")
    forbidden = {
        "baseline_refused",
        "watchdog_abort",
        "parent_signal_abort",
        "unexpected_failure_abort",
    }
    seen_forbidden = sorted(
        {str(event.get("event")) for event in events if event.get("event") in forbidden}
    )
    if seen_forbidden:
        raise ReceiptError(f"watchdog contains safety refusal/abort events: {seen_forbidden}")

    clean = _one(events, "clean_parent_baseline")
    complete = _one(events, "baseline_soak_complete")
    started = _one(events, "child_started")
    final = _one(events, "final")
    if (
        complete.get("required_duration_seconds") != BASELINE_SOAK_SECONDS
        or complete.get("observed_duration_ns", 0) < BASELINE_SOAK_SECONDS * 1_000_000_000
        or complete.get("minimum_reclaimable_headroom_bytes")
        != minimum_baseline_headroom_bytes
        or complete.get("require_zero_current_swap") is not False
        or complete.get("maximum_pressure_level_raw") != maximum_pressure_level_raw
    ):
        raise ReceiptError("watchdog did not complete the exact 60 second nonzero-swap baseline")
    baseline_samples = [event for event in events if event.get("event") == "baseline_soak_sample"]
    if len(baseline_samples) < 240:
        raise ReceiptError("watchdog baseline contains fewer than 240 durable 250 ms samples")
    schedules = [event.get("scheduled_monotonic_ns") for event in baseline_samples]
    if any(not _is_nonnegative_int(value) for value in schedules) or any(
        later - earlier != WATCHDOG_SAMPLE_PERIOD_NS
        for earlier, later in zip(schedules, schedules[1:])
    ):
        raise ReceiptError("watchdog baseline is not on the exact 250 ms schedule")
    for sample in baseline_samples:
        lateness = sample.get("schedule_lateness_ns")
        duration = sample.get("telemetry_duration_ns")
        if (
            sample.get("violations") != []
            or not _is_nonnegative_int(lateness)
            or not _is_nonnegative_int(duration)
            or lateness + duration >= WATCHDOG_SAMPLE_PERIOD_NS
        ):
            raise ReceiptError("a baseline watchdog sample violated its 250 ms safety budget")

    host_events = [event for event in events if isinstance(event.get("host"), dict)]
    baseline_events = [clean, *baseline_samples]
    if not host_events or not isinstance(clean.get("host"), dict):
        raise ReceiptError("watchdog has no host memory evidence")
    first_host = clean["host"]
    baseline_swapins = first_host.get("swapins_pages")
    baseline_swapouts = first_host.get("swapouts_pages")
    if not _is_nonnegative_int(baseline_swapins) or not _is_nonnegative_int(baseline_swapouts):
        raise ReceiptError("watchdog baseline has invalid swap counters")
    preflight_after = intent["hashing_contract"]["memory_after"]["host"]
    if (
        baseline_swapins != preflight_after["swapins_pages"]
        or baseline_swapouts != preflight_after["swapouts_pages"]
    ):
        raise ReceiptError("swap changed between preflight and watchdog admission")

    for event in host_events:
        host = event["host"]
        if event.get("violations", []) != []:
            raise ReceiptError("a watchdog host sample recorded a safety violation")
        if (
            not _pressure_allowed(
                host.get("pressure_level_raw"), maximum_pressure_level_raw
            )
            or host.get("swapins_pages") != baseline_swapins
            or host.get("swapouts_pages") != baseline_swapouts
            or not _is_nonnegative_int(host.get("swapped_pages_current"))
            or host.get("wired_bytes", MAX_HOST_WIRED_BYTES + 1) > MAX_HOST_WIRED_BYTES
        ):
            raise ReceiptError("host pressure, swap growth, or wired-memory evidence failed")
        minimum = (
            minimum_baseline_headroom_bytes if event in baseline_events
            else MIN_RUNTIME_HEADROOM_BYTES
        )
        if host.get("reclaimable_headroom_bytes", -1) < minimum:
            raise ReceiptError("host reclaimable memory fell below its phase limit")

    expected_started = {
        "sample_period_ns": WATCHDOG_SAMPLE_PERIOD_NS,
        "minimum_reclaimable_headroom_bytes": MIN_RUNTIME_HEADROOM_BYTES,
        "maximum_child_physical_footprint_bytes": MAX_CHILD_FOOTPRINT_BYTES,
        "maximum_host_wired_bytes": MAX_HOST_WIRED_BYTES,
        "maximum_pressure_level_raw": maximum_pressure_level_raw,
        "require_zero_current_swap": False,
        "reject_swapin_growth": True,
        "report_producer": "external",
        "process_accounting_scope": "isolated_process_group_aggregate",
    }
    if any(started.get(key) != value for key, value in expected_started.items()):
        raise ReceiptError("watchdog child limits or accounting scope drifted")
    if (
        started.get("pid") != started.get("process_group")
        or started.get("report") != str(run_dir / "response.json")
        or started.get("baseline_swapins_pages") != baseline_swapins
        or started.get("baseline_swapouts_pages") != baseline_swapouts
    ):
        raise ReceiptError("watchdog child identity, report, or swap baseline drifted")
    _validate_environment(started.get("experiment_environment"))

    if (
        final.get("child_returncode") != 0
        or final.get("watchdog_aborted") is not False
        or final.get("abort_reasons") != []
        or final.get("process_group_empty") is not True
        or final.get("process_group") != started.get("process_group")
        or final.get("process_accounting_scope") != "isolated_process_group_aggregate"
        or final.get("report_exists") is not True
        or final.get("report_is_regular_file") is not True
        or final.get("report_is_symlink") is not False
        or final.get("report_size_bytes", 0) <= 0
        or final.get("peak_child_physical_footprint_bytes", MAX_CHILD_FOOTPRINT_BYTES + 1)
        > MAX_CHILD_FOOTPRINT_BYTES
        or final.get("peak_host_wired_bytes", MAX_HOST_WIRED_BYTES + 1)
        > MAX_HOST_WIRED_BYTES
        or final.get("minimum_reclaimable_headroom_bytes_observed", -1)
        < MIN_RUNTIME_HEADROOM_BYTES
    ):
        raise ReceiptError("watchdog final safety receipt failed")

    peak_rss = final.get("peak_child_rss_bytes")
    if not _is_nonnegative_int(peak_rss):
        raise ReceiptError("watchdog final receipt has no valid child RSS peak")
    swapped_current = [event["host"]["swapped_pages_current"] for event in host_events]
    pressure_levels = [event["host"]["pressure_level_raw"] for event in host_events]
    return {
        "baseline_swapins_pages": baseline_swapins,
        "baseline_swapouts_pages": baseline_swapouts,
        "swapins_growth_pages": 0,
        "swapouts_growth_pages": 0,
        "baseline_swapped_pages_current": first_host["swapped_pages_current"],
        "maximum_swapped_pages_current": max(swapped_current),
        "maximum_pressure_level_raw_observed": max(pressure_levels),
        "minimum_reclaimable_headroom_bytes": final[
            "minimum_reclaimable_headroom_bytes_observed"
        ],
        "peak_child_rss_bytes": peak_rss,
        "peak_child_physical_footprint_bytes": final[
            "peak_child_physical_footprint_bytes"
        ],
        "peak_host_wired_bytes": final["peak_host_wired_bytes"],
        "baseline_samples": len(baseline_samples),
        "sample_period_ms": WATCHDOG_SAMPLE_PERIOD_NS / 1_000_000,
    }


def _validate_health(run_dir: Path, binary_source_commit: str) -> None:
    health = _read_json(run_dir / "health.json")
    build = health.get("build") if isinstance(health, dict) else None
    expected = {
        "source_commit": binary_source_commit,
        "generation_ready": True,
        "gemma4_serve_lane": "ghost_moe",
        "gemma4_ghost_execution_mode": "full_common_metal",
        "gemma4_ghost_common_metal_active": True,
        "gemma4_ghost_experts_metal_active": True,
        "gemma4_ghost_head_metal_active": True,
        "gemma4_mtp_assistant_loaded": True,
        "gemma4_mtp_full_q4_active": True,
    }
    if (
        not isinstance(health, dict)
        or not isinstance(build, str)
        or not build
        or build.endswith("-dirty")
        or any(health.get(key) != value for key, value in expected.items())
    ):
        raise ReceiptError("health receipt did not prove full-Metal, full-Q4 readiness")


def _validate_geometry(geometry: Any) -> None:
    expected = {
        "layers": EXPECTED_LAYERS,
        "record_payload_bytes": RECORD_PAYLOAD_BYTES,
        "slot_stride_bytes": SLOT_STRIDE_BYTES,
        "logical_addressable_slots": LOGICAL_SLOTS_TOTAL,
        "anonymous_hot_capacity_slots": HOT_SLOTS_TOTAL,
        "anonymous_hot_capacity_bytes": HOT_CAPACITY_BYTES,
        "file_mapped_addressable_slots": LOGICAL_SLOTS_TOTAL,
        "file_mapped_address_span_bytes": MAPPED_ADDRESS_SPAN_BYTES,
        "overflow_slots": 0,
        "overflow_capacity_bytes": 0,
        "victim_record_capacity": 0,
        "victim_capacity_bytes": 0,
        "host_cache_budget_bytes": 0,
        "mapped_readahead_enabled": False,
        "mapped_readahead_max_inflight_records": 0,
        "mapped_readahead_anonymous_capacity_bytes": 0,
    }
    if not isinstance(geometry, dict) or any(geometry.get(key) != value for key, value in expected.items()):
        raise ReceiptError("telemetry aggregate geometry is not exact uniform Hot40")
    per_layer = geometry.get("per_layer")
    if not isinstance(per_layer, list) or len(per_layer) != EXPECTED_LAYERS:
        raise ReceiptError("telemetry does not contain exactly 30 layer geometries")
    for layer_index, layer in enumerate(per_layer):
        expected_layer = {
            "layer": layer_index,
            "logical_addressable_slots": LOGICAL_SLOTS_PER_LAYER,
            "anonymous_hot_capacity_slots": HOT_SLOTS_PER_LAYER,
            "anonymous_hot_capacity_bytes": HOT_CAPACITY_BYTES_PER_LAYER,
            "file_mapped_addressable_slots": LOGICAL_SLOTS_PER_LAYER,
            "file_mapped_address_span_bytes": MAPPED_ADDRESS_SPAN_BYTES_PER_LAYER,
            "overflow_slots": 0,
            "victim_slots": 0,
        }
        if not isinstance(layer, dict) or layer != expected_layer:
            raise ReceiptError(f"layer {layer_index} is not exact 40-hot/128-mapped geometry")


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ReceiptError("cannot summarize an empty timing series")
    return {
        "count": len(values),
        "minimum_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "maximum_ms": max(values),
    }


def _validate_response(
    run_dir: Path,
) -> tuple[dict[str, Any], list[int]]:
    response = _read_json(run_dir / "response.json")
    expected_ids = _read_json(run_dir / "expected-token-ids.json")
    if not isinstance(expected_ids, list) or len(expected_ids) != EXPECTED_TOKENS:
        raise ReceiptError("expected-token fixture does not contain exactly 48 IDs")
    usage = response.get("usage") if isinstance(response, dict) else None
    choices = response.get("choices") if isinstance(response, dict) else None
    camelid = response.get("camelid") if isinstance(response, dict) else None
    generated = camelid.get("generated_token_ids") if isinstance(camelid, dict) else None
    if (
        not isinstance(usage, dict)
        or usage.get("completion_tokens") != EXPECTED_TOKENS
        or generated != expected_ids
        or not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "length"
    ):
        raise ReceiptError("response token count, finish reason, or exact token IDs drifted")
    telemetry = camelid.get("hybrid_telemetry")
    if (
        not isinstance(telemetry, dict)
        or telemetry.get("schema_version") != 2
        or telemetry.get("scope") != "single_completed_measured_request"
    ):
        raise ReceiptError("response has no exact schema-v2 hybrid telemetry")
    _validate_geometry(telemetry.get("geometry"))

    route_interval = telemetry.get("route_interval")
    if not isinstance(route_interval, dict):
        raise ReceiptError("telemetry has no route interval")
    routed = route_interval.get("routed_expert_ids_per_layer")
    unique = route_interval.get("routed_unique_per_layer")
    if (
        not isinstance(routed, list)
        or len(routed) != EXPECTED_LAYERS
        or not isinstance(unique, list)
        or len(unique) != EXPECTED_LAYERS
    ):
        raise ReceiptError("route interval is not attributed across exactly 30 layers")
    for layer_ids, layer_unique in zip(routed, unique):
        if (
            not isinstance(layer_ids, list)
            or any(not _is_nonnegative_int(value) or value >= LOGICAL_SLOTS_PER_LAYER for value in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)
            or layer_unique != len(layer_ids)
        ):
            raise ReceiptError("route interval contains invalid expert IDs or uniqueness counts")

    rounds = telemetry.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ReceiptError("telemetry has no completed measured rounds")
    committed_ids: list[int] = []
    proposed = accepted = 0
    round_wall_values: list[float] = []
    verifier_wall_values: list[float] = []
    verifier_gpu_values: list[float] = []
    assistant_exposed_values: list[float] = []
    assistant_gpu_values: list[float] = []
    assistant_rounds = 0
    assistant_proposals: list[int] = []
    seen_round_sequences: set[int] = set()
    full_k8_rounds = 0
    for index, receipt in enumerate(rounds):
        if not isinstance(receipt, dict) or receipt.get("round_index") != index:
            raise ReceiptError(f"round {index} is absent or reordered")
        if receipt.get("success") is not True or receipt.get("bootstrap") is not False:
            raise ReceiptError(f"round {index} failed or entered a separate bootstrap lane")
        for field in (
            "selected_dropped",
            "missing_failclose",
            "slot_capacity_overflow",
            "overflow_slots",
            "overflow_bytes",
            "overflow_layers",
            "overflow_experts",
            "victim_hits",
            "mapped_readahead_enqueued_records",
            "mapped_readahead_enqueued_bytes",
            "mapped_readahead_enqueue_ms",
            "mapped_readahead_previous_union_enqueued_records",
            "mapped_readahead_previous_union_enqueued_bytes",
            "mapped_readahead_previous_union_enqueue_ms",
        ):
            if receipt.get(field) != 0:
                raise ReceiptError(f"round {index} has nonzero safety field {field}")
        proposed_k = receipt.get("proposed_k")
        accepted_k = receipt.get("accepted_drafts")
        useful = receipt.get("useful_accepted_drafts")
        verifier_k = receipt.get("verifier_k")
        round_sequence = receipt.get("chained_round_sequence")
        start_pos = receipt.get("prefix_tokens_before")
        committed = receipt.get("committed_tokens")
        if (
            not _is_nonnegative_int(proposed_k)
            or not _is_nonnegative_int(accepted_k)
            or not _is_nonnegative_int(useful)
            or not _is_nonnegative_int(verifier_k)
            or receipt.get("k") != verifier_k
            or verifier_k != proposed_k + 1
            or receipt.get("requested_k") != 8
            or accepted_k > proposed_k
            or useful != accepted_k
            or not isinstance(committed, list)
            or len(committed) != useful + 1
            or any(not _is_nonnegative_int(token) or token > 0xFFFF_FFFF for token in committed)
            or not _is_nonnegative_int(round_sequence)
            or round_sequence == 0
            or round_sequence in seen_round_sequences
            or not _is_nonnegative_int(start_pos)
        ):
            raise ReceiptError(f"round {index} has inconsistent K or commit accounting")
        seen_round_sequences.add(round_sequence)
        per_layer = receipt.get("per_layer")
        if not isinstance(per_layer, list) or len(per_layer) != EXPECTED_LAYERS:
            raise ReceiptError(f"round {index} lacks exact per-layer routing evidence")
        for layer_index, layer in enumerate(per_layer):
            if not isinstance(layer, dict) or layer.get("layer_index") != layer_index:
                raise ReceiptError(f"round {index} layer routing is absent or reordered")
            active = layer.get("active_unique")
            hot = layer.get("hot_bound")
            mapped = layer.get("mapped_bound")
            bound = layer.get("bound_records")
            if (
                not all(_is_nonnegative_int(value) for value in (active, hot, mapped, bound))
                or hot > HOT_SLOTS_PER_LAYER
                or mapped > LOGICAL_SLOTS_PER_LAYER
                or active != bound
                or bound != hot + mapped
                or bound > LOGICAL_SLOTS_PER_LAYER
            ):
                raise ReceiptError(f"round {index} layer {layer_index} has invalid bounds")
        timing_fields = (
            ("receipt_round_wall_ms", round_wall_values),
            ("verifier_wall_ms", verifier_wall_values),
            ("verifier_gpu_ms", verifier_gpu_values),
            ("assistant_exposed_ms", assistant_exposed_values),
            ("assistant_gpu_ms", assistant_gpu_values),
        )
        for field, values in timing_fields:
            value = receipt.get(field)
            if not _is_finite(value, positive=field in {"receipt_round_wall_ms", "verifier_wall_ms", "verifier_gpu_ms"}):
                raise ReceiptError(f"round {index} has invalid {field}")
            values.append(float(value))
        if proposed_k > 0:
            assistant_rounds += 1
            assistant_proposals.append(proposed_k)
        if proposed_k == 7 and verifier_k == 8 and receipt.get("budget_truncated") is False:
            full_k8_rounds += 1
        proposed += proposed_k
        accepted += accepted_k
        committed_ids.extend(committed)
    if committed_ids != expected_ids:
        raise ReceiptError("round-committed token IDs do not reconcile the frozen response IDs")
    if assistant_rounds == 0 or full_k8_rounds == 0:
        raise ReceiptError("the K8 benchmark did not execute a full assistant/verifier round")

    aggregate = telemetry.get("aggregate")
    metrics = telemetry.get("metrics")
    if not isinstance(aggregate, dict) or not isinstance(metrics, dict):
        raise ReceiptError("telemetry aggregate or metrics are absent")
    for field in ("host_fills", "direct_read_failures", "overflow_experts", "victim_hits"):
        if aggregate.get(field) != 0:
            raise ReceiptError(f"telemetry aggregate has nonzero safety field {field}")
    round_wall_ms = sum(round_wall_values)
    if (
        metrics.get("response_completion_tokens") != EXPECTED_TOKENS
        or metrics.get("proposed_drafts") != proposed
        or metrics.get("accepted_drafts") != accepted
        or metrics.get("outer_lookahead_nonzero_count") != 0
        or not _is_finite(metrics.get("receipt_round_wall_ms"), positive=True)
        or not math.isclose(
            float(metrics["receipt_round_wall_ms"]), round_wall_ms, rel_tol=1e-9, abs_tol=1e-6
        )
    ):
        raise ReceiptError("structured metrics do not reconcile completed rounds")
    effective_tps = EXPECTED_TOKENS / (round_wall_ms / 1000.0)
    return (
        {
            "tokens": EXPECTED_TOKENS,
            "receipt_round_wall_ms": round_wall_ms,
            "effective_decode_tokens_per_second": effective_tps,
            "reported_decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
            "proposed_drafts": proposed,
            "accepted_drafts": accepted,
            "acceptance_probability": accepted / proposed if proposed else 0.0,
            "rounds": len(rounds),
            "verifier_wall": _summary(verifier_wall_values),
            "verifier_gpu": _summary(verifier_gpu_values),
            "assistant_exposed": _summary(assistant_exposed_values),
            "assistant_gpu": _summary(assistant_gpu_values),
        },
        assistant_proposals,
    )


def _validate_log(
    run_dir: Path,
    assistant_proposals: list[int],
    assistant_sha256: str,
) -> dict[str, Any]:
    try:
        log = (run_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ReceiptError(f"could not read server log: {error}") from error
    if len(HOT40_STARTUP_PATTERN.findall(log)) != 1:
        raise ReceiptError("server log does not contain one exact uniform Hot40 admission")
    if re.search(r"(?:thread '.+' panicked|safety[ _-]?abort|fatal error)", log, re.IGNORECASE):
        raise ReceiptError("server log contains a panic or safety-abort marker")
    if "[gemma4 sparse-predict probe]" in log:
        raise ReceiptError("no-prediction Hot40 log contains a sparse-predict probe receipt")
    if "[gemma4-ghost-metal] mapped-cold MADV_WILLNEED refused:" in log:
        raise ReceiptError("the kernel refused mapped-cold page advice")
    full_q4 = FULL_Q4_PATTERN.findall(log)
    if len(full_q4) != 1:
        raise ReceiptError("server did not admit exactly one full-Q4 assistant")
    source_sha, matrices, packed_bytes, bf16_bytes, quantize_us = full_q4[0]
    if (
        source_sha != assistant_sha256
        or int(matrices) != 23
        or int(packed_bytes) != FULL_Q4_MATRIX_BYTES
        or int(bf16_bytes) != FULL_Q4_BF16_MATRIX_BYTES
        or int(quantize_us) <= 0
    ):
        raise ReceiptError("full-Q4 assistant receipt is incomplete")
    if log.splitlines().count(FULL_Q4_RESIDENCY_MARKER) != 1:
        raise ReceiptError("full-Q4 assistant did not release its BF16 source mapping")
    chains = DEVICE_CHAIN_PATTERN.findall(log)
    if len(chains) != len(assistant_proposals):
        raise ReceiptError("device-chain receipt count does not match assistant rounds")
    for chain, proposed in zip(chains, assistant_proposals):
        requested, returned, linear_format, matrix_bytes, *_ = chain
        if (
            linear_format != "q4_0_all"
            or int(matrix_bytes) != FULL_Q4_MATRIX_BYTES
            or int(requested) < proposed
            or int(returned) != proposed
        ):
            raise ReceiptError("a device-chain receipt is not bounded full-Q4 execution")
    stage_matches = STAGE_PATTERN.findall(log)
    if not stage_matches:
        raise ReceiptError("server log has no Metal chained-stage timings")
    split_modes = sorted({match[0] for match in stage_matches})
    if split_modes != ["per-cb"]:
        raise ReceiptError(f"unexpected Metal stage split modes: {split_modes}")
    names = ("qkv_o", "attn", "router", "shared", "gateup", "down", "resid", "gpu_total")
    values = {name: [] for name in names}
    for match in stage_matches:
        for name, raw in zip(names, match[1:]):
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ReceiptError("Metal stage receipt contains an invalid duration")
            values[name].append(value)
    return {
        "receipt_count": len(stage_matches),
        "split_mode": "per-cb",
        "milliseconds": {name: _summary(series) for name, series in values.items()},
        "full_q4": {
            "source_sha256": source_sha,
            "matrices": int(matrices),
            "packed_bytes": int(packed_bytes),
            "bf16_matrix_bytes": int(bf16_bytes),
        },
        "device_chain_receipts": len(chains),
    }


def _validate_ports(run_dir: Path) -> None:
    receipt = _read_json(run_dir / "port-clear.json")
    if receipt != {
        "schema_version": 1,
        "ports": {
            "8181": {
                "clear": True,
                "policy": "never-bound-connected-or-signaled",
            },
            "8189": {"clear": True, "policy": "benchmark-only"},
        },
    }:
        raise ReceiptError("final port-clear receipt does not prove ports 8181 and 8189 clear")


def analyze(run_dir: Path) -> dict[str, Any]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ReceiptError(f"run directory is missing or symlinked: {run_dir}")
    intent = _validate_intent(run_dir)
    _validate_health(run_dir, intent["binary_source_commit"])
    memory = _validate_watchdog(run_dir, intent)
    performance, assistant_proposals = _validate_response(run_dir)
    assistant_sha256 = intent["large_inputs"]["assistant"]["preverified_sha256"]
    stages = _validate_log(run_dir, assistant_proposals, assistant_sha256)
    _validate_ports(run_dir)
    hashing_contract = intent["hashing_contract"]
    hashing_before = hashing_contract["memory_before"]["host"]
    hashing_after = hashing_contract["memory_after"]["host"]
    return {
        "schema_version": 1,
        "pass": True,
        "benchmark": "gemma4-uniform-hot40-experiment",
        "source_commit": intent["source_commit"],
        "binary_source_commit": intent["binary_source_commit"],
        "harness_commit": intent["harness_commit"],
        "gates": {
            "exact_48_token_ids": True,
            "uniform_40_by_30_geometry": True,
            "full_q4_device_chain": True,
            "no_safety_abort": True,
            "bounded_memory": True,
            "no_swapin_or_swapout_growth": True,
            "existing_swap_allowed": True,
            "ports_clear_after_run": True,
            "large_inputs_not_read_by_receipt_harness_before_spawn": True,
            "historical_large_input_hashes_bound_to_live_stat": True,
            "small_artifact_f_nocache_hashing": True,
            "preflight_bounded_memory": True,
            "preflight_no_swap_growth": True,
            "no_sparse_prediction_probe": True,
            "protected_port_never_bound_connected_or_signaled": True,
            "binary_source_is_ancestor_of_harness": True,
            "runtime_source_diff_empty_between_binary_and_harness": True,
            "pressure_never_exceeded_configured_maximum": True,
            "mapped_readahead_disabled_with_zero_counters": True,
        },
        "pressure_policy": {
            "allow_warning_pressure": intent["allow_warning_pressure"],
            "maximum_pressure_level_raw": (
                2 if intent["allow_warning_pressure"] else 1
            ),
            "maximum_pressure_level_raw_observed": memory[
                "maximum_pressure_level_raw_observed"
            ],
        },
        "geometry": intent["geometry"],
        "performance": {
            **performance,
            "throughput_contaminated_by_sparse_predict_probe": False,
            "measurement_scope": "isolated-hot40-k8-full-q4",
        },
        "stages": stages,
        "hashing": {
            "algorithm": "sha256",
            "f_rdahead_command": F_RDAHEAD,
            "f_nocache_command": F_NOCACHE,
            "read_chunk_bytes": 4 * 1024 * 1024,
            "post_hash_cooldown_seconds": 0,
            "hashed_scope": "binary-fixtures-tooling-only",
            "large_inputs_content_hashed_this_run": False,
            "large_inputs_content_read_before_spawn": False,
            "large_input_binding": "historical-sha256-plus-live-stat",
            "provenance_reference": hashing_contract["provenance_reference"],
            "provenance_reference_sha256": hashing_contract[
                "provenance_reference_sha256"
            ],
            "provenance_limitation": hashing_contract["provenance_limitation"],
            "pre_hash_reclaimable_headroom_bytes": hashing_before[
                "reclaimable_headroom_bytes"
            ],
            "post_hash_reclaimable_headroom_bytes": hashing_after[
                "reclaimable_headroom_bytes"
            ],
            "pre_hash_wired_bytes": hashing_before["wired_bytes"],
            "post_hash_wired_bytes": hashing_after["wired_bytes"],
            "swapins_growth_pages": 0,
            "swapouts_growth_pages": 0,
        },
        "memory": memory,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verdict = analyze(args.run_dir.resolve(strict=True))
        _write_atomic(args.output, verdict)
    except ReceiptError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 75
    print(
        "HOT40_PASS "
        f"observed_tps={verdict['performance']['effective_decode_tokens_per_second']:.3f} "
        f"wall_ms={verdict['performance']['receipt_round_wall_ms']:.3f} "
        f"peak_footprint_gib={verdict['memory']['peak_child_physical_footprint_bytes'] / 1024**3:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
