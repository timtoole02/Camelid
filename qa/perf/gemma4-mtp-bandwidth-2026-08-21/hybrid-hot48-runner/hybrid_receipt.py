#!/usr/bin/env python3
"""Fail-closed analyzer for fixed-budget hot/mapped-cold Gemma 4 receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


GIB = 1024**3
LAYERS = 30
CANONICAL_PER_LAYER = 128
HOT_PER_LAYER = 32
UNIFORM_HOT_PROFILE = (HOT_PER_LAYER,) * LAYERS
V5_HOT_PROFILE = (
    39, 40, 33, 30, 30, 31, 31, 30, 34, 30,
    26, 28, 30, 31, 28, 37, 31, 30, 31, 32,
    31, 32, 30, 31, 32, 35, 32, 34, 34, 37,
)
PROFILE_SELECTOR_ENV = "CAMELID_HYBRID_RESIDENCY_PROFILE"
PROFILE_SELECTOR_V5 = "v5-k8-max-960"
HOT_PROFILE_ENV = "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER"
V5_HOT_PROFILE_ENCODED = ",".join(str(slots) for slots in V5_HOT_PROFILE)
CANONICAL_TOTAL = LAYERS * CANONICAL_PER_LAYER
HOT_TOTAL = LAYERS * HOT_PER_LAYER
SLOT_STRIDE_BYTES = 3_358_720
RECORD_PAYLOAD_BYTES = 3_345_408
HOT_CAPACITY_BYTES = HOT_TOTAL * SLOT_STRIDE_BYTES
MAPPED_COLD_SPAN_BYTES = CANONICAL_TOTAL * SLOT_STRIDE_BYTES
MAPPED_COLD_SPAN_BYTES_PER_LAYER = CANONICAL_PER_LAYER * SLOT_STRIDE_BYTES
MIN_BASELINE_HEADROOM_BYTES = 8 * GIB
MIN_RUNTIME_HEADROOM_BYTES = 2 * GIB
MAX_CHILD_FOOTPRINT_BYTES = 8_053_063_680  # 7.5 GiB
MAX_HOST_WIRED_BYTES = 8 * GIB
MIN_BASELINE_SOAK_NS = 60_000_000_000
WATCHDOG_SAMPLE_PERIOD_NS = 250_000_000
EXPECTED_REQUEST_SHA256 = {
    9: "a612ca079082b32a1cf80cd51f76d41ffe6f26cf22266089e148b9aed966a0d4",
    48: "b2f1110079fc726699cc936a628a268a7ec5bf2076fa970899de39d4ea903939",
}

FORBIDDEN_ENVIRONMENT = {
    "CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER",
    "CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER",
    "CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS",
}

COMMON_ENVIRONMENT = {
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
    "CAMELID_GEMMA4_SLOT_PIN": "0",
    "CAMELID_GEMMA4_GHOST_METAL_HOT_PIN": "0",
    "CAMELID_GEMMA4_VICTIM_CACHE": "0",
    "CAMELID_GEMMA4_VICTIM_MB": "0",
    "CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS": "0",
    "CAMELID_GEMMA4_CHAINED_PREDICT": "0",
    "CAMELID_SPEC_DECODE": "off",
    "CAMELID_GEMMA4_SPEC_K1_LANE": "chained",
    "CAMELID_GEMMA4_SPEC_TIMING": "1",
    "CAMELID_GEMMA4_GHOST_METAL_TIMING": "1",
    "CAMELID_GEMMA4_ROUTE_TRACE": "1",
}

STARTUP_PATTERN = re.compile(
    r"HYBRID ACTIVE: .*FILE_MAPPED_EXPERTS=1 .*HYBRID_HOT_SLOTS=32, "
    r"layers=30 canonical_addressable=128/layer physical_hot=32/layer "
    r"hot_capacity_bytes=3224371200 mapped_cold_span_bytes=12897484800 "
    r"overflow=0 victim=0 slot_pin=off prediction=off"
)
PROFILED_STARTUP_PATTERN = re.compile(
    r"HYBRID ACTIVE: .*FILE_MAPPED_EXPERTS=1 .*HYBRID_HOT_SLOTS=32 "
    + re.escape(HOT_PROFILE_ENV)
    + r"="
    + re.escape(V5_HOT_PROFILE_ENCODED)
    + r", layers=30 canonical_addressable=128/layer "
    r"physical_hot=per-layer-profile hot_capacity_slots=960 "
    r"hot_capacity_bytes=3224371200 mapped_cold_span_bytes=12897484800 "
    r"overflow=0 victim=0 slot_pin=off prediction=off"
)
FILE_PAGER_PATTERN = re.compile(
    r"clean file-pager Q4_0 experts enabled: layers=30 "
    r"logical_addressable_slots/layer=128 "
    r"anonymous_expert_capacity_bytes=3224371200 "
    r"mapped_address_span_bytes=12897484800 mapped_address_span=12\.01GiB "
    r"mode=(?:fused-fast|CPU-GeGLU parity)"
)
DEMAND_PREWARM_MARKER = (
    "CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY=1: arbitrary cold-start "
    "expert prewarm skipped; persistent slots will populate from routed demand"
)


class ReceiptError(RuntimeError):
    pass


def sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"hashed input is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReceiptError(f"hash input {path}: {error}") from error
    return digest.hexdigest()


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0.0 if positive else number >= 0.0)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReceiptError(f"read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ReceiptError(f"{path}:{line_number} is not an object")
                events.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ReceiptError(f"read JSONL {path}: {error}") from error
    if not events:
        raise ReceiptError(f"{path} has no events")
    return events


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
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


def _last(events: Iterable[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("event") == name]
    if not matches:
        raise ReceiptError(f"watchdog has no {name!r} event")
    return matches[-1]


def validate_environment(
    environment: dict[str, Any],
    lane: str,
    hot_profile: tuple[int, ...] = UNIFORM_HOT_PROFILE,
    *,
    include_selector: bool = True,
    require_assistant: bool = True,
) -> None:
    expected = dict(COMMON_ENVIRONMENT)
    expected.update(
        {
            "CAMELID_GEMMA4_SPEC_CHUNK_MAX": "1" if lane == "k1" else "8",
            "CAMELID_GEMMA4_SPEC_DRAFT_TOKENS": "1" if lane == "k1" else "8",
        }
    )
    if lane == "k1":
        expected["CAMELID_GEMMA4_CHAINED_K1"] = "1"
    profiled = hot_profile == V5_HOT_PROFILE
    if profiled:
        expected[HOT_PROFILE_ENV] = V5_HOT_PROFILE_ENCODED
        if include_selector:
            expected[PROFILE_SELECTOR_ENV] = PROFILE_SELECTOR_V5
    elif hot_profile != UNIFORM_HOT_PROFILE:
        raise ReceiptError("environment validator received an unadmitted hot profile")
    for key, expected_value in expected.items():
        if environment.get(key) != expected_value:
            raise ReceiptError(
                f"environment {key}={environment.get(key)!r}, expected {expected_value!r}"
            )
    present_forbidden = sorted(FORBIDDEN_ENVIRONMENT.intersection(environment))
    if present_forbidden:
        raise ReceiptError(f"forbidden environment keys present: {present_forbidden}")
    for key in (HOT_PROFILE_ENV, PROFILE_SELECTOR_ENV):
        if key not in expected and key in environment:
            raise ReceiptError(f"uniform Hot32 environment unexpectedly contains {key}")
    if lane == "k1" and "CAMELID_GEMMA4_MTP_ASSISTANT_PATH" in environment:
        raise ReceiptError("K1 lane unexpectedly received an assistant path")
    if lane != "k1" and "CAMELID_GEMMA4_CHAINED_K1" in environment:
        raise ReceiptError(f"{lane} lane unexpectedly enabled chained K1")
    if lane != "k1" and require_assistant and not str(
        environment.get("CAMELID_GEMMA4_MTP_ASSISTANT_PATH", "")
    ).startswith("/"):
        raise ReceiptError(f"{lane} lane has no absolute assistant path")


def validate_effective_load_environment(
    environment: dict[str, Any],
    hot_profile: tuple[int, ...] = UNIFORM_HOT_PROFILE,
) -> None:
    """Validate the target process after the load-only harness scrubs tuning vars.

    The watchdog receipts the complete outer runner environment.  The harness
    deliberately removes proposal-only controls before constructing the target,
    so the effective target receipt has a smaller, separately frozen contract.
    """

    validate_environment(
        environment,
        "k8",
        hot_profile,
        include_selector=False,
        require_assistant=False,
    )
    if "CAMELID_GEMMA4_MTP_ASSISTANT_PATH" in environment:
        raise ReceiptError("effective load-only target unexpectedly received an assistant")


def validate_watchdog(
    path: Path,
    lane: str,
    hot_profile: tuple[int, ...] = UNIFORM_HOT_PROFILE,
) -> dict[str, Any]:
    events = read_jsonl(path)
    if any(event.get("schema_version") != 3 for event in events):
        raise ReceiptError("hybrid receipts require watchdog schema version 3 throughout")
    for singleton in (
        "clean_parent_baseline",
        "baseline_soak_complete",
        "child_started",
        "final",
    ):
        if sum(event.get("event") == singleton for event in events) != 1:
            raise ReceiptError(f"watchdog must contain exactly one {singleton!r} event")
    forbidden_events = {
        "baseline_refused",
        "watchdog_abort",
        "parent_signal_abort",
        "unexpected_failure_abort",
    }
    present_forbidden = sorted(
        {str(event.get("event")) for event in events if event.get("event") in forbidden_events}
    )
    if present_forbidden:
        raise ReceiptError(f"watchdog contains refusal/abort events: {present_forbidden}")
    clean = _last(events, "clean_parent_baseline")
    complete = _last(events, "baseline_soak_complete")
    final = _last(events, "final")
    started = _last(events, "child_started")
    if clean.get("schema_version") != 3 or final.get("schema_version") != 3:
        raise ReceiptError("hybrid receipts require watchdog schema version 3")
    if complete.get("required_duration_seconds") != 60:
        raise ReceiptError("watchdog baseline soak must be configured to exactly 60 seconds")
    if (
        complete.get("minimum_reclaimable_headroom_bytes")
        != MIN_BASELINE_HEADROOM_BYTES
        or complete.get("require_zero_current_swap") is not True
    ):
        raise ReceiptError("watchdog baseline soak limits do not match the hybrid gate")
    if complete.get("observed_duration_ns", 0) < MIN_BASELINE_SOAK_NS:
        raise ReceiptError("watchdog observed less than 60 seconds of baseline soak")
    soak_samples = [
        event for event in events if event.get("event") == "baseline_soak_sample"
    ]
    required_samples = (
        MIN_BASELINE_SOAK_NS + WATCHDOG_SAMPLE_PERIOD_NS - 1
    ) // WATCHDOG_SAMPLE_PERIOD_NS
    if len(soak_samples) < required_samples:
        raise ReceiptError("watchdog baseline soak has fewer than 240 durable samples")
    for event in soak_samples:
        if event.get("violations") != []:
            raise ReceiptError("watchdog baseline sample recorded a violation")
        lateness = event.get("schedule_lateness_ns")
        duration = event.get("telemetry_duration_ns")
        if (
            not isinstance(lateness, int)
            or lateness < 0
            or not isinstance(duration, int)
            or duration < 0
            or lateness + duration >= WATCHDOG_SAMPLE_PERIOD_NS
        ):
            raise ReceiptError("watchdog baseline sample exceeded its 250 ms budget")
    schedules = [event["scheduled_monotonic_ns"] for event in soak_samples]
    if any(
        later - earlier != WATCHDOG_SAMPLE_PERIOD_NS
        for earlier, later in zip(schedules, schedules[1:])
    ):
        raise ReceiptError("watchdog baseline samples are not on the exact 250 ms schedule")
    baseline_events = [
        event
        for event in events
        if event.get("event") in {"clean_parent_baseline", "baseline_soak_sample"}
    ]
    host_events = [event for event in events if isinstance(event.get("host"), dict)]
    if not baseline_events or not host_events:
        raise ReceiptError("watchdog is missing host baseline/runtime samples")
    first_host = baseline_events[0]["host"]
    baseline_swapins = first_host.get("swapins_pages")
    baseline_swapouts = first_host.get("swapouts_pages")
    for event in baseline_events:
        host = event["host"]
        if host.get("reclaimable_headroom_bytes", 0) < MIN_BASELINE_HEADROOM_BYTES:
            raise ReceiptError("baseline reclaimable headroom fell below 8 GiB")
    for event in host_events:
        host = event["host"]
        if event.get("violations", []) != []:
            raise ReceiptError("watchdog host sample recorded a violation")
        if host.get("pressure_level_raw") != 1:
            raise ReceiptError("watchdog observed non-normal memory pressure")
        if host.get("swapped_pages_current") != 0:
            raise ReceiptError("watchdog observed nonzero current swap")
        if host.get("swapins_pages") != baseline_swapins:
            raise ReceiptError("watchdog observed swap-in growth")
        if host.get("swapouts_pages") != baseline_swapouts:
            raise ReceiptError("watchdog observed swapout growth")
        if host.get("wired_bytes", 0) > MAX_HOST_WIRED_BYTES:
            raise ReceiptError("watchdog observed more than 8 GiB host wired memory")
        if event.get("event") not in {
            "clean_parent_baseline",
            "baseline_soak_sample",
        } and host.get("reclaimable_headroom_bytes", 0) < MIN_RUNTIME_HEADROOM_BYTES:
            raise ReceiptError("runtime reclaimable headroom fell below 2 GiB")
    if final.get("child_returncode") != 0:
        raise ReceiptError(f"child returned {final.get('child_returncode')!r}")
    if final.get("watchdog_aborted") is not False or final.get("abort_reasons") != []:
        raise ReceiptError("watchdog aborted or recorded abort reasons")
    if final.get("process_group_empty") is not True:
        raise ReceiptError("watchdog did not prove the saved process group is empty")
    if started.get("pid") != started.get("process_group"):
        raise ReceiptError("watchdog child PID does not equal its isolated process group")
    if final.get("process_group") != started.get("process_group"):
        raise ReceiptError("watchdog final process-group identity drifted")
    if (
        started.get("process_accounting_scope")
        != "isolated_process_group_aggregate"
        or final.get("process_accounting_scope")
        != "isolated_process_group_aggregate"
    ):
        raise ReceiptError("watchdog did not use aggregate process-group accounting")
    expected_report = path.parent / (
        "load-only-report.json" if lane == "load-only" else "response.json"
    )
    if started.get("report") != str(expected_report):
        raise ReceiptError("watchdog report path does not match the frozen lane layout")
    expected_producer = "child" if lane == "load-only" else "external"
    if started.get("report_producer") != expected_producer:
        raise ReceiptError("watchdog report producer mode does not match the lane")
    expected_limits = {
        "sample_period_ns": WATCHDOG_SAMPLE_PERIOD_NS,
        "minimum_reclaimable_headroom_bytes": MIN_RUNTIME_HEADROOM_BYTES,
        "maximum_child_physical_footprint_bytes": MAX_CHILD_FOOTPRINT_BYTES,
        "maximum_host_wired_bytes": MAX_HOST_WIRED_BYTES,
        "require_zero_current_swap": True,
        "reject_swapin_growth": True,
    }
    if any(started.get(key) != value for key, value in expected_limits.items()):
        raise ReceiptError("watchdog child_started limits do not match the hybrid gate")
    if (
        final.get("report_exists") is not True
        or final.get("report_is_regular_file") is not True
        or final.get("report_is_symlink") is not False
        or final.get("report_size_bytes", 0) <= 0
    ):
        raise ReceiptError("watchdog final report is absent, empty, or not a regular file")
    if final.get("peak_child_physical_footprint_bytes", 0) > MAX_CHILD_FOOTPRINT_BYTES:
        raise ReceiptError("child physical footprint exceeded 7.5 GiB")
    if final.get("peak_host_wired_bytes", 0) > MAX_HOST_WIRED_BYTES:
        raise ReceiptError("peak host wired memory exceeded 8 GiB")
    environment = started.get("experiment_environment")
    if not isinstance(environment, dict):
        raise ReceiptError("watchdog child_started event has no environment receipt")
    validate_environment(environment, lane, hot_profile)
    return final


def validate_request_identity(
    lane_dir: Path, intent: dict[str, Any], expected_tokens: int
) -> str:
    request = intent.get("request")
    if not isinstance(request, dict):
        raise ReceiptError("intent has no frozen request identity")
    if expected_tokens == 0:
        if request != {"source_path": "", "frozen_path": "", "sha256": ""}:
            raise ReceiptError("load-only intent unexpectedly binds a request fixture")
        return ""

    source_path = request.get("source_path")
    frozen_path = request.get("frozen_path")
    recorded_sha = request.get("sha256")
    expected_frozen_path = lane_dir / "request.json"
    if (
        not isinstance(source_path, str)
        or Path(source_path).name != f"request-{expected_tokens}.json"
        or frozen_path != str(expected_frozen_path)
        or not isinstance(recorded_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_sha)
        or recorded_sha != EXPECTED_REQUEST_SHA256.get(expected_tokens)
    ):
        raise ReceiptError("intent request identity does not match the canonical lane contract")
    if sha256_regular_file(expected_frozen_path) != recorded_sha:
        raise ReceiptError("frozen request bytes do not match the intent hash")
    fixture = read_json(expected_frozen_path)
    if (
        fixture.get("max_tokens") != expected_tokens
        or fixture.get("temperature") != 0
        or fixture.get("top_k") != 1
        or fixture.get("seed") != 0
        or fixture.get("stream") is not False
        or fixture.get("camelid_receipt") is not True
        or fixture.get("camelid_enable_thinking") is not False
    ):
        raise ReceiptError("frozen request does not match the deterministic token/receipt contract")
    return recorded_sha


def validate_auxiliary_receipts(
    lane_dir: Path, stage: str, lane: str, expected_tokens: int
) -> tuple[int, ...]:
    intent = read_json(lane_dir / "intent.json")
    if (
        intent.get("schema_version") != 1
        or intent.get("stage") != stage
        or intent.get("lane") != lane
        or intent.get("expected_tokens") != expected_tokens
    ):
        raise ReceiptError("intent identity does not match the requested stage/lane")
    disk = intent.get("disk")
    if (
        not isinstance(disk, dict)
        or disk.get("available_kib", 0) < 20 * 1024 * 1024
        or disk.get("used_percent", 101) > 90
    ):
        raise ReceiptError("intent does not prove the 20 GiB/90% disk gate")
    profile = intent.get("profile")
    expected_profile = {
        "demand_load_only": 1,
        "file_mapped_experts": 1,
        "hybrid_hot_slots": 32,
        "slot_pin": 0,
        "assistant_residency_policy": "observed_from_assistant_ledger",
        "physical_slots_per_layer": "unset",
    }
    if not isinstance(profile, dict) or any(
        profile.get(key) != value for key, value in expected_profile.items()
    ):
        raise ReceiptError("intent does not bind the exact hybrid profile")
    residency_profile = profile.get("residency_profile")
    per_layer = profile.get("hybrid_hot_slots_per_layer")
    if residency_profile is None and per_layer is None:
        hot_profile = UNIFORM_HOT_PROFILE
    elif (
        residency_profile == PROFILE_SELECTOR_V5
        and per_layer == list(V5_HOT_PROFILE)
    ):
        hot_profile = V5_HOT_PROFILE
    else:
        raise ReceiptError("intent contains an unadmitted per-layer residency profile")
    executable = intent.get("executable")
    tooling = intent.get("tooling")
    integration = intent.get("integration_contract")
    hashes = []
    if isinstance(executable, dict):
        hashes.append(executable.get("sha256"))
    if isinstance(tooling, dict):
        hashes.extend(
            tooling.get(key)
            for key in ("runner_sha256", "watchdog_sha256", "analyzer_sha256")
        )
    if isinstance(integration, dict):
        hashes.append(integration.get("sha256"))
    if len(hashes) != 5 or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in hashes
    ):
        raise ReceiptError("intent is missing frozen executable/tooling hashes")
    validate_request_identity(lane_dir, intent, expected_tokens)
    port = read_json(lane_dir / "port-clear.json")
    if port != {"schema_version": 1, "port": 8189, "clear": True}:
        raise ReceiptError("lane does not contain the exact port-clear receipt")
    return hot_profile


def validate_startup_log(
    path: Path,
    lane: str,
    hot_profile: tuple[int, ...] = UNIFORM_HOT_PROFILE,
) -> None:
    try:
        log = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ReceiptError(f"read server log {path}: {error}") from error
    startup_pattern = (
        PROFILED_STARTUP_PATTERN if hot_profile == V5_HOT_PROFILE else STARTUP_PATTERN
    )
    if hot_profile not in (UNIFORM_HOT_PROFILE, V5_HOT_PROFILE):
        raise ReceiptError("startup validator received an unadmitted hot profile")
    if len(startup_pattern.findall(log)) != 1:
        raise ReceiptError("server log must contain one exact HYBRID ACTIVE admission line")
    if len(FILE_PAGER_PATTERN.findall(log)) != 1:
        raise ReceiptError("server log must contain one exact clean file-pager line")
    if log.count(DEMAND_PREWARM_MARKER) != 1:
        raise ReceiptError("server log must contain one exact demand-prewarm-skip line")
    forbidden = [
        "CPU Ghost experts",
        "selected experts may be skipped",
        "slot_capacity_overflow=1",
        "missing_expert_failclose=1",
        "F_NOCACHE",
        "thread 'main' panicked",
    ]
    for marker in forbidden:
        if marker in log:
            raise ReceiptError(f"server log contains forbidden marker {marker!r}")
    if lane == "k8" and "lm_head=q4_0" not in log:
        raise ReceiptError("K8 server log does not prove the Q4_0 assistant head")


def validate_response(path: Path, expected_tokens: int) -> dict[str, Any]:
    response = read_json(path)
    usage = response.get("usage")
    camelid = response.get("camelid")
    choices = response.get("choices")
    if not isinstance(usage, dict) or usage.get("completion_tokens") != expected_tokens:
        raise ReceiptError(f"response did not emit exactly {expected_tokens} tokens")
    if not isinstance(camelid, dict) or not isinstance(
        camelid.get("generated_token_ids"), list
    ):
        raise ReceiptError("response has no generated token ID receipt")
    if len(camelid["generated_token_ids"]) != expected_tokens:
        raise ReceiptError("generated token ID count differs from completion count")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or choices[0].get("finish_reason") != "length"
        or not isinstance(choices[0].get("message", {}).get("content"), str)
    ):
        raise ReceiptError("response has no single length-finished text choice")
    return response


def _geometry(
    telemetry: dict[str, Any],
    hot_profile: tuple[int, ...] = UNIFORM_HOT_PROFILE,
) -> None:
    if hot_profile not in (UNIFORM_HOT_PROFILE, V5_HOT_PROFILE):
        raise ReceiptError("geometry validator received an unadmitted hot profile")
    geometry = telemetry.get("geometry")
    expected = {
        "layers": LAYERS,
        "record_payload_bytes": RECORD_PAYLOAD_BYTES,
        "slot_stride_bytes": SLOT_STRIDE_BYTES,
        "logical_addressable_slots": CANONICAL_TOTAL,
        "anonymous_hot_capacity_slots": HOT_TOTAL,
        "anonymous_hot_capacity_bytes": HOT_CAPACITY_BYTES,
        "file_mapped_addressable_slots": CANONICAL_TOTAL,
        "file_mapped_address_span_bytes": MAPPED_COLD_SPAN_BYTES,
        "overflow_slots": 0,
        "overflow_capacity_bytes": 0,
        "victim_record_capacity": 0,
        "victim_capacity_bytes": 0,
        "host_cache_budget_bytes": 0,
    }
    if not isinstance(geometry, dict):
        raise ReceiptError("hybrid telemetry has no geometry object")
    drift = {
        key: {"actual": geometry.get(key), "expected": value}
        for key, value in expected.items()
        if geometry.get(key) != value
    }
    if drift:
        raise ReceiptError(f"hybrid geometry drift: {drift}")
    per_layer = geometry.get("per_layer")
    if not isinstance(per_layer, list) or len(per_layer) != LAYERS:
        raise ReceiptError("hybrid geometry must contain exactly 30 layer records")
    for layer_index, layer in enumerate(per_layer):
        if not isinstance(layer, dict) or layer.get("layer") != layer_index:
            raise ReceiptError(f"hybrid geometry layer {layer_index} is absent or reordered")
        layer_slots = hot_profile[layer_index]
        layer_expected = {
            "logical_addressable_slots": CANONICAL_PER_LAYER,
            "anonymous_hot_capacity_slots": layer_slots,
            "anonymous_hot_capacity_bytes": layer_slots * SLOT_STRIDE_BYTES,
            "file_mapped_addressable_slots": CANONICAL_PER_LAYER,
            "file_mapped_address_span_bytes": MAPPED_COLD_SPAN_BYTES_PER_LAYER,
            "overflow_slots": 0,
            "victim_slots": 0,
        }
        drift = {
            key: {"actual": layer.get(key), "expected": value}
            for key, value in layer_expected.items()
            if layer.get(key) != value
        }
        if drift:
            raise ReceiptError(f"hybrid geometry layer {layer_index} drift: {drift}")


def validate_hybrid_telemetry(
    telemetry: dict[str, Any],
    lane: str,
    expected_tokens: int | None = None,
    expected_token_ids: list[int] | None = None,
    hot_profile: tuple[int, ...] = UNIFORM_HOT_PROFILE,
) -> dict[str, Any]:
    if lane not in ("k1", "k8"):
        raise ReceiptError(f"unknown hybrid lane {lane!r}")
    if not _nonnegative_int(expected_tokens) or expected_tokens == 0:
        raise ReceiptError("hybrid telemetry validation requires a positive token budget")
    if telemetry.get("schema_version") != 1:
        raise ReceiptError("hybrid telemetry schema_version must be 1")
    if telemetry.get("scope") != "single_completed_measured_request":
        raise ReceiptError("hybrid telemetry does not cover one completed measured request")
    route_interval = telemetry.get("route_interval")
    if (
        not isinstance(route_interval, dict)
        or route_interval.get("scope") != "measured_request_prefill_plus_generation"
    ):
        raise ReceiptError("hybrid route interval does not cover measured prefill+generation")
    _geometry(telemetry, hot_profile)
    rounds = telemetry.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ReceiptError("hybrid telemetry contains no completed rounds")
    observed_k8 = False
    observed_full_k8 = False
    hot_bound_total = 0
    cold_bound_total = 0
    forwarded_total = 0
    proposed_total = 0
    accepted_total = 0
    round_wall_total = 0.0
    full_round_zero_accepts = 0
    full_round_assistant_exposed: list[float] = []
    committed_forwarded: list[int] = []
    previous_prefix: int | None = None
    previous_sequence: int | None = None
    for round_index, round_receipt in enumerate(rounds):
        if not isinstance(round_receipt, dict):
            raise ReceiptError(f"hybrid round {round_index} is not an object")
        if round_receipt.get("round_index") != round_index:
            raise ReceiptError(f"hybrid round {round_index} has a reordered round_index")
        prefix = round_receipt.get("prefix_tokens_before")
        sequence = round_receipt.get("chained_round_sequence")
        if not _nonnegative_int(prefix) or not _nonnegative_int(sequence):
            raise ReceiptError(f"hybrid round {round_index} has invalid sequence identity")
        if round_index > 0 and (
            prefix != previous_prefix or sequence != previous_sequence
        ):
            raise ReceiptError(f"hybrid round {round_index} sequence continuity drifted")
        remaining = round_receipt.get("remaining_budget_before")
        expected_remaining = expected_tokens - forwarded_total
        if remaining != expected_remaining or expected_remaining <= 0:
            raise ReceiptError(
                f"hybrid round {round_index} remaining budget does not reconcile"
            )
        k = round_receipt.get("k")
        requested = round_receipt.get("requested_k")
        proposed = round_receipt.get("proposed_k")
        verifier = round_receipt.get("verifier_k")
        accepted = round_receipt.get("accepted_drafts")
        useful = round_receipt.get("useful_accepted_drafts")
        if not all(
            _nonnegative_int(value)
            for value in (k, requested, proposed, verifier, accepted, useful)
        ):
            raise ReceiptError(f"hybrid round {round_index} has invalid K/draft counts")
        if (
            k != verifier
            or verifier != proposed + 1
            or accepted > proposed
            or useful != accepted
        ):
            raise ReceiptError(f"hybrid round {round_index} has inconsistent K/draft counts")
        committed = round_receipt.get("committed_tokens")
        if (
            not isinstance(committed, list)
            or len(committed) != 1 + accepted
            or any(
                not _nonnegative_int(token) or token > 0xFFFF_FFFF
                for token in committed
            )
        ):
            raise ReceiptError(f"hybrid round {round_index} has invalid committed tokens")
        assistant_exposed = round_receipt.get("assistant_exposed_ms")
        assistant_gpu = round_receipt.get("assistant_gpu_ms")
        round_wall = round_receipt.get("receipt_round_wall_ms")
        if (
            not _finite_number(assistant_exposed)
            or not _finite_number(assistant_gpu)
            or not _finite_number(round_wall, positive=True)
        ):
            raise ReceiptError(f"hybrid round {round_index} has invalid timing")

        observed_k8 |= k == 8
        if lane == "k8":
            if round_index == 0:
                if (
                    round_receipt.get("bootstrap") is not True
                    or (k, requested, proposed, verifier, accepted, useful)
                    != (1, 1, 0, 1, 0, 0)
                    or round_receipt.get("budget_truncated") is not False
                    or assistant_exposed != 0
                    or assistant_gpu != 0
                ):
                    raise ReceiptError("hybrid K8 telemetry has no exact round-zero K1 bootstrap")
            else:
                if (
                    round_receipt.get("bootstrap") is not False
                    or remaining < 2
                    or requested != 8
                    or verifier > min(8, remaining)
                    or proposed > min(7, remaining - 1)
                    or round_receipt.get("budget_truncated") is not (remaining < 8)
                ):
                    raise ReceiptError(
                        f"hybrid K8 round {round_index} does not match its remaining budget"
                    )
                if (
                    verifier == 8
                    and proposed == 7
                    and round_receipt.get("budget_truncated") is False
                ):
                    observed_full_k8 = True
                    full_round_assistant_exposed.append(float(assistant_exposed))
                    if accepted == 0:
                        full_round_zero_accepts += 1
        elif (
            round_receipt.get("bootstrap") is not False
            or (k, requested, proposed, verifier, accepted, useful)
            != (1, 1, 0, 1, 0, 0)
            or round_receipt.get("budget_truncated") is not False
            or assistant_exposed != 0
            or assistant_gpu != 0
        ):
            raise ReceiptError(f"hybrid K1 round {round_index} is not a zero-draft K1 forward")

        if round_receipt.get("success") is not True:
            raise ReceiptError(f"hybrid round {round_index} did not complete successfully")
        for key in (
            "selected_dropped",
            "missing_failclose",
            "slot_capacity_overflow",
            "overflow_experts",
        ):
            if round_receipt.get(key) != 0:
                raise ReceiptError(
                    f"hybrid round {round_index} has {key}={round_receipt.get(key)!r}"
                )
        layers = round_receipt.get("per_layer")
        if not isinstance(layers, list) or len(layers) != LAYERS:
            raise ReceiptError(f"hybrid round {round_index} does not contain 30 layers")
        for layer_index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ReceiptError(f"round {round_index} layer {layer_index} is invalid")
            if layer.get("layer_index") != layer_index:
                raise ReceiptError(
                    f"round {round_index} layer {layer_index} is absent or reordered"
                )
            unique = layer.get("active_unique")
            hot = layer.get("hot_bound")
            cold = layer.get("mapped_bound")
            bound = layer.get("bound_records")
            if not all(
                isinstance(value, int) and value >= 0
                for value in (unique, hot, cold, bound)
            ):
                raise ReceiptError(f"round {round_index} layer {layer_index} has invalid counts")
            if unique != hot + cold or bound != unique:
                raise ReceiptError(
                    f"round {round_index} layer {layer_index} tier partition drift"
                )
            if isinstance(k, int) and 1 <= k <= 8 and not (8 <= unique <= k * 8):
                raise ReceiptError(
                    f"round {round_index} layer {layer_index} is outside 8..K×8"
                )
            if unique > CANONICAL_PER_LAYER or hot > hot_profile[layer_index]:
                raise ReceiptError(
                    f"round {round_index} layer {layer_index} exceeds hybrid geometry"
                )
            hot_bound_total += hot
            cold_bound_total += cold
        forwarded_total += 1 + useful
        committed_forwarded.extend(committed)
        proposed_total += proposed
        accepted_total += accepted
        round_wall_total += float(round_wall)
        previous_prefix = prefix + len(committed)
        previous_sequence = sequence + 1
    if lane == "k8" and not observed_k8:
        raise ReceiptError("K8 telemetry contains no K=8 round")
    if lane == "k8" and not observed_full_k8:
        raise ReceiptError("K8 telemetry contains no full requested=8/proposed=7 verifier round")
    aggregate = telemetry.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ReceiptError("hybrid telemetry has no aggregate object")
    if aggregate.get("scope") != "single_completed_measured_request":
        raise ReceiptError("hybrid aggregate does not cover one completed measured request")
    if lane == "k8":
        if aggregate.get("hot_hits", 0) <= 0 or hot_bound_total <= 0:
            raise ReceiptError("K8 telemetry does not prove hot-tier use")
        if aggregate.get("mapped_cold_selections", 0) <= 0 or cold_bound_total <= 0:
            raise ReceiptError("K8 telemetry does not prove mapped-cold use")
    for field in (
        "route_lookups",
        "hot_hits",
        "mapped_cold_selections",
        "direct_reads",
        "direct_read_bytes",
        "host_cache_hits",
        "host_cache_misses",
        "host_cache_evictions",
        "overflow_experts",
        "victim_hits",
        "chained_promotion_loads",
        "chained_promotion_read_bytes",
    ):
        if not _nonnegative_int(aggregate.get(field)):
            raise ReceiptError(f"hybrid aggregate has invalid {field}")
    if aggregate["route_lookups"] != (
        aggregate["hot_hits"] + aggregate["mapped_cold_selections"]
    ):
        raise ReceiptError("hybrid aggregate route lookup partition is inconsistent")
    for field in (
        "direct_reads",
        "direct_read_bytes",
        "host_cache_hits",
        "host_cache_misses",
        "host_cache_evictions",
        "overflow_experts",
        "victim_hits",
    ):
        if aggregate[field] != 0:
            raise ReceiptError(f"hybrid aggregate requires {field}=0")
    if aggregate["chained_promotion_read_bytes"] != (
        aggregate["chained_promotion_loads"] * RECORD_PAYLOAD_BYTES
    ):
        raise ReceiptError("hybrid promotion byte/load delta is inconsistent")
    if aggregate["chained_promotion_loads"] > aggregate["mapped_cold_selections"]:
        raise ReceiptError("hybrid promotion loads exceed mapped-cold selections")
    metrics = telemetry.get("metrics")
    if not isinstance(metrics, dict):
        raise ReceiptError("hybrid telemetry has no structured metrics")
    forwarded = metrics.get("forwarded_decode_tokens")
    terminal_unforwarded = metrics.get("terminal_unforwarded_tokens")
    response_tokens = metrics.get("response_completion_tokens")
    proposed_metric = metrics.get("proposed_drafts")
    accepted_metric = metrics.get("accepted_drafts")
    receipt_round_wall = metrics.get("receipt_round_wall_ms")
    decode_tokens_per_second = metrics.get("decode_tokens_per_second")
    reported_zero_accepts = metrics.get("full_round_zero_accepts")
    reported_max_exposed = metrics.get("max_full_assistant_exposed_ms")
    outer_lookahead = metrics.get("outer_lookahead_nonzero_count")
    expected_decode_tps = forwarded_total / (round_wall_total / 1_000.0)
    expected_max_exposed = max(full_round_assistant_exposed, default=0.0)
    if (
        not all(
            _nonnegative_int(value)
            for value in (
                forwarded,
                terminal_unforwarded,
                response_tokens,
                proposed_metric,
                accepted_metric,
                reported_zero_accepts,
                outer_lookahead,
            )
        )
        or forwarded != forwarded_total
        or terminal_unforwarded not in (0, 1)
        or response_tokens != expected_tokens
        or forwarded + terminal_unforwarded != response_tokens
        or proposed_metric != proposed_total
        or accepted_metric != accepted_total
        or not _finite_number(receipt_round_wall, positive=True)
        or not math.isclose(
            float(receipt_round_wall), round_wall_total, rel_tol=1e-9, abs_tol=1e-6
        )
        or not _finite_number(decode_tokens_per_second, positive=True)
        or not math.isclose(
            float(decode_tokens_per_second), expected_decode_tps, rel_tol=1e-9, abs_tol=1e-6
        )
        or reported_zero_accepts != full_round_zero_accepts
        or not _finite_number(reported_max_exposed)
        or not math.isclose(
            float(reported_max_exposed), expected_max_exposed, rel_tol=1e-9, abs_tol=1e-6
        )
        or outer_lookahead != 0
    ):
        raise ReceiptError("hybrid metrics do not reconcile the measured rounds")
    if lane == "k1":
        if (
            terminal_unforwarded != 1
            or forwarded != expected_tokens - 1
            or len(rounds) != expected_tokens - 1
            or proposed_metric != 0
            or accepted_metric != 0
            or reported_zero_accepts != 0
            or reported_max_exposed != 0
        ):
            raise ReceiptError(
                "K1 metrics do not reconcile length-finished zero-draft forwards"
            )
    if expected_token_ids is not None:
        if (
            len(expected_token_ids) != expected_tokens
            or any(
                not _nonnegative_int(token) or token > 0xFFFF_FFFF
                for token in expected_token_ids
            )
            or committed_forwarded != expected_token_ids[:forwarded_total]
        ):
            raise ReceiptError("hybrid committed tokens do not match the API response")
    return metrics


def _load_hybrid_telemetry(response: dict[str, Any]) -> dict[str, Any]:
    camelid = response.get("camelid")
    if isinstance(camelid, dict) and isinstance(camelid.get("hybrid_telemetry"), dict):
        return camelid["hybrid_telemetry"]
    raise ReceiptError(
        "response.camelid.hybrid_telemetry is missing; runtime/API receipt integration has not landed"
    )


def validate_load_only(lane_dir: Path) -> dict[str, Any]:
    hot_profile = validate_auxiliary_receipts(lane_dir, "load-only", "load-only", 0)
    validate_watchdog(lane_dir / "watchdog.jsonl", "load-only", hot_profile)
    validate_startup_log(lane_dir / "child.log", "load-only", hot_profile)
    report = read_json(lane_dir / "load-only-report.json")
    if report.get("schema_version") != 4:
        raise ReceiptError("hybrid load-only report schema_version must be 4")
    if report.get("completed") is not True or report.get("failure") is not None:
        raise ReceiptError("load-only child did not publish a completed success report")
    for field in (
        "assistant_warmups",
        "assistant_proposals",
        "tokenizer_calls",
        "target_prefills",
        "target_steps",
        "target_generations",
        "target_kv_borrows",
    ):
        if report.get(field) != 0:
            raise ReceiptError(f"load-only operation counter {field} is nonzero")
    assistant = report.get("assistant_ledger")
    if not isinstance(assistant, dict):
        raise ReceiptError("load-only report has no assistant residency ledger")
    mapped_bytes = assistant.get("mapped_bytes")
    locked_bytes = assistant.get("locked_bytes")
    if not isinstance(mapped_bytes, int) or mapped_bytes <= 0:
        raise ReceiptError("load-only assistant residency ledger has no mapped bytes")
    if not isinstance(locked_bytes, int) or locked_bytes <= 0 or locked_bytes != mapped_bytes:
        raise ReceiptError("load-only assistant residency ledger has invalid locked bytes")
    resident_pages = assistant.get("resident_pages")
    total_pages = assistant.get("total_pages")
    if (
        not isinstance(resident_pages, int)
        or resident_pages <= 0
        or resident_pages != total_pages
    ):
        raise ReceiptError("load-only assistant residency ledger is not fully resident")
    target = report.get("target_final_ledger")
    if not isinstance(target, dict):
        raise ReceiptError("load-only report has no final target ledger")
    expected = {
        "expert_layer_count": LAYERS,
        "expert_logical_slot_count": CANONICAL_TOTAL,
        "expert_slot_count": HOT_TOTAL,
        "expert_slot_capacity_bytes": HOT_CAPACITY_BYTES,
        "expert_file_mapped_slot_count": CANONICAL_TOTAL,
        "expert_file_mapped_address_span_bytes": MAPPED_COLD_SPAN_BYTES,
        "expert_table_directory_slot_count": HOT_TOTAL,
        "expert_table_directory_capacity_bytes": HOT_CAPACITY_BYTES,
        "expert_table_bound_active_slot_count": LAYERS * 8,
        "overflow_slot_count": 0,
        "overflow_capacity_bytes": 0,
        "victim_record_capacity": 0,
        "victim_capacity_bytes": 0,
        "host_cache_budget_bytes": 0,
        "host_cache_resident_bytes": 0,
        "host_cache_explicitly_touched_bytes": 0,
        "expert_slot_explicitly_touched_bytes": 0,
        "overflow_explicitly_touched_bytes": 0,
        "victim_explicitly_touched_bytes": 0,
        "planned_prewarm_records": 0,
        "planned_prewarm_bytes": 0,
        "touched_prewarm_records": 0,
        "touched_prewarm_bytes": 0,
    }
    if hot_profile == V5_HOT_PROFILE:
        expected["expert_anonymous_slots_per_layer"] = list(hot_profile)
    elif target.get("expert_anonymous_slots_per_layer") not in (
        None,
        list(UNIFORM_HOT_PROFILE),
    ):
        raise ReceiptError("uniform load-only ledger has a non-uniform anonymous profile")
    drift = {
        key: {"actual": target.get(key), "expected": value}
        for key, value in expected.items()
        if target.get(key) != value
    }
    if drift:
        raise ReceiptError(f"load-only hybrid ledger drift: {drift}")
    if target.get("expert_tables_compute_bound") is not True:
        raise ReceiptError("load-only report did not bind the production expert tables")
    if target.get("arbitrary_slot_prewarm_skipped") is not True:
        raise ReceiptError("load-only report did not prove arbitrary prewarm was skipped")
    if (
        not isinstance(target.get("cghost_logical_bytes"), int)
        or target["cghost_logical_bytes"] < MAPPED_COLD_SPAN_BYTES
        or not isinstance(target.get("cghost_mapped_bytes"), int)
        or target["cghost_mapped_bytes"] < MAPPED_COLD_SPAN_BYTES
    ):
        raise ReceiptError("load-only .cghost mapping does not cover the canonical span")
    effective = report.get("target_runtime", {}).get("environment")
    if not isinstance(effective, dict):
        raise ReceiptError("load-only report has no effective target environment")
    validate_effective_load_environment(effective, hot_profile)
    return {
        "pass": True,
        "stage": "load-only",
        "geometry": expected,
        "assistant_residency": {
            "mapped_bytes": mapped_bytes,
            "locked_bytes": locked_bytes,
            "resident_pages": resident_pages,
            "total_pages": total_pages,
            "policy": "native_observed_hard_lock",
        },
    }


def validate_lane(lane_dir: Path, lane: str, expected_tokens: int) -> dict[str, Any]:
    stage = {
        ("k8", 9): "smoke-k8",
        ("k1", 9): "smoke-k1",
        ("k8", 48): "promotion-k8",
        ("k1", 48): "promotion-k1",
    }.get((lane, expected_tokens))
    if stage is None:
        raise ReceiptError(
            f"unsupported lane/token contract: lane={lane!r} tokens={expected_tokens!r}"
        )
    hot_profile = validate_auxiliary_receipts(lane_dir, stage, lane, expected_tokens)
    final = validate_watchdog(lane_dir / "watchdog.jsonl", lane, hot_profile)
    validate_startup_log(lane_dir / "server.log", lane, hot_profile)
    response = validate_response(lane_dir / "response.json", expected_tokens)
    telemetry = _load_hybrid_telemetry(response)
    metrics = validate_hybrid_telemetry(
        telemetry,
        lane,
        expected_tokens,
        response["camelid"]["generated_token_ids"],
        hot_profile,
    )
    return {
        "pass": True,
        "lane": lane,
        "expected_tokens": expected_tokens,
        "watchdog": final,
        "metrics": metrics,
    }


def validate_parity(root: Path, stage: str) -> dict[str, Any]:
    tokens = 9 if stage == "smoke" else 48
    pair = root / ("02-smoke-9t" if stage == "smoke" else "03-promotion-48t")
    k8_verdict = read_json(pair / "k8" / "verdict.json")
    k1_verdict = read_json(pair / "k1" / "verdict.json")
    if k8_verdict.get("pass") is not True or k1_verdict.get("pass") is not True:
        raise ReceiptError("both lane verdicts must pass before parity analysis")
    k8_request_sha = validate_request_identity(
        pair / "k8", read_json(pair / "k8" / "intent.json"), tokens
    )
    k1_request_sha = validate_request_identity(
        pair / "k1", read_json(pair / "k1" / "intent.json"), tokens
    )
    if k8_request_sha != k1_request_sha:
        raise ReceiptError("K1/K8 frozen request hashes differ")
    k8 = validate_response(pair / "k8" / "response.json", tokens)
    k1 = validate_response(pair / "k1" / "response.json", tokens)
    k8_ids = k8["camelid"]["generated_token_ids"]
    k1_ids = k1["camelid"]["generated_token_ids"]
    k8_text = k8["choices"][0]["message"]["content"]
    k1_text = k1["choices"][0]["message"]["content"]
    if k8_ids != k1_ids or k8_text != k1_text:
        raise ReceiptError("K1/K8 token IDs or decoded text differ")
    result: dict[str, Any] = {
        "pass": True,
        "stage": stage,
        "tokens": tokens,
        "request_sha256": k8_request_sha,
        "exact_token_id_parity": True,
        "exact_text_parity": True,
    }
    if stage == "promotion":
        metrics = k8_verdict.get("metrics")
        if not isinstance(metrics, dict):
            raise ReceiptError("promotion K8 verdict has no structured metrics")
        proposed = metrics.get("proposed_drafts")
        accepted = metrics.get("accepted_drafts")
        decode_tokens_per_second = metrics.get("decode_tokens_per_second")
        zero_accepts = metrics.get("full_round_zero_accepts")
        assistant_exposed_ms = metrics.get("max_full_assistant_exposed_ms")
        outer_lookahead_count = metrics.get("outer_lookahead_nonzero_count")
        if (
            not isinstance(proposed, int)
            or proposed <= 0
            or not isinstance(accepted, int)
            or accepted < 0
            or accepted > proposed
            or not isinstance(decode_tokens_per_second, (int, float))
            or isinstance(decode_tokens_per_second, bool)
            or not isinstance(zero_accepts, int)
            or zero_accepts < 0
            or not isinstance(assistant_exposed_ms, (int, float))
            or isinstance(assistant_exposed_ms, bool)
            or assistant_exposed_ms < 0
            or not isinstance(outer_lookahead_count, int)
            or outer_lookahead_count < 0
        ):
            raise ReceiptError("promotion metrics have invalid draft counts")
        acceptance = accepted / proposed
        gates = {
            "decode_tokens_per_second": decode_tokens_per_second >= 28.0,
            "acceptance_probability": acceptance >= 0.85,
            "no_zero_accept_full_round": zero_accepts == 0,
            "assistant_exposed_ms": assistant_exposed_ms <= 35.0,
            "outer_lookahead_off": outer_lookahead_count == 0,
        }
        result.update(
            {
                "pass": all(gates.values()),
                "performance_gates": gates,
                "acceptance_probability": acceptance,
            }
        )
    return result


def run_with_verdict(output: Path, operation: Any) -> int:
    try:
        result = operation()
    except ReceiptError as error:
        result = {"pass": False, "error": str(error)}
    except Exception as error:  # Keep malformed evidence fail-closed and durable.
        result = {
            "pass": False,
            "error": f"internal analyzer refusal: {type(error).__name__}: {error}",
        }
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("pass") is True else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    load = commands.add_parser("load-only")
    load.add_argument("--lane-dir", type=Path, required=True)
    load.add_argument("--output", type=Path, required=True)
    lane = commands.add_parser("lane")
    lane.add_argument("--lane-dir", type=Path, required=True)
    lane.add_argument("--lane", choices=("k1", "k8"), required=True)
    lane.add_argument("--expected-tokens", type=int, choices=(9, 48), required=True)
    lane.add_argument("--output", type=Path, required=True)
    parity = commands.add_parser("parity")
    parity.add_argument("--receipt-root", type=Path, required=True)
    parity.add_argument("--stage", choices=("smoke", "promotion"), required=True)
    parity.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output.exists():
        print(f"REFUSED: verdict already exists: {args.output}", file=sys.stderr)
        return 77
    if args.command == "load-only":
        return run_with_verdict(args.output, lambda: validate_load_only(args.lane_dir))
    if args.command == "lane":
        return run_with_verdict(
            args.output,
            lambda: validate_lane(args.lane_dir, args.lane, args.expected_tokens),
        )
    return run_with_verdict(
        args.output,
        lambda: validate_parity(args.receipt_root, args.stage),
    )


if __name__ == "__main__":
    raise SystemExit(main())
