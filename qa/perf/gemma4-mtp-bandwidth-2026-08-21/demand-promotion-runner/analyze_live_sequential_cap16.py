#!/usr/bin/env python3
"""Fail-closed analyzer for one exact H69 cap-4/8/16 observation request."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import stat
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


PROBE_PREFIX = "[metal live-sequential-predict probe] "
STAGE_PREFIX = "[gemma4 live-sequential stage] "
EXPECTED_ROUNDS = ((104, 14), (118, 13), (131, 14), (145, 7))
TARGET_LAYERS = tuple(range(1, 30))
CAPS = (4, 8, 16)
MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES = 8_053_063_679
MAX_HOST_WIRED_BYTES = 8_589_934_591
GO_SAVED_US = 50_000
GO_PRECISION_NUMERATOR = 30
GO_PRECISION_DENOMINATOR = 100

INT_RE = re.compile(r"0|[1-9][0-9]*")
JSON_INT_RE = re.compile(r"-?(?:0|[1-9][0-9]*)")
JSON_DECIMAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]+")
MS_RE = re.compile(r"(0|[1-9][0-9]*)\.([0-9]{3})")
RATIO_RE = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{6}")
ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")

CAP_FIELD_STEMS = {
    "hits",
    "actual_cold",
    "predicted_cold",
    "recall",
    "precision",
    "read_wall_ms",
    "projected_saved_ms",
    "read_wall_weighted_recall",
}
PROBE_FIELDS = {
    "schema",
    "admitted",
    "profile",
    "start_pos",
    "K",
    "source",
    "target",
    "first_target_layer",
    "eligible",
    "attempts",
    "failures",
    "truth_valid",
    "predict_us",
    "predictor_top_k_per_row",
    "probe_caps",
    *(f"cap{cap}_{stem}" for cap in CAPS for stem in CAP_FIELD_STEMS),
    "cap16_incremental_hits_vs_cap8",
    "cap16_incremental_predicted_vs_cap8",
    "cap16_incremental_precision_vs_cap8",
    "cap16_incremental_saved_ms_vs_cap8",
    "total_wave_load_ms",
    "projection",
    "stage_admitted",
    "launches",
    "launch_failures",
    "stage_candidates",
    "readiness_measured",
    "contention_measured",
    "output_mutation",
    "io_mutation",
    "slot_policy_mutation",
    "routing_authority",
    "cap8_candidates",
    "cap16_candidates",
    "cap4_layers",
    "cap8_layers",
    "cap16_layers",
}
INVALID_PROBE_FIELDS = {"schema", "admitted", "truth_valid", "reason"}
STAGE_FIELDS = {
    "schema",
    "round_seq",
    "start_pos",
    "K",
    "ok",
    "cap",
    "workers",
    "workers_started",
    "workers_done",
    "predict_impl",
    "launches",
    "candidates",
    "reads_started",
    "reads_succeeded",
    "reads_failed",
    "reads_in_flight",
    "speculative_read_ms",
    "seals",
    "exact_cold",
    "ready_hits",
    "previous_ready_hits",
    "direct_fallback",
    "ready_unused",
    "ready_malformed",
    "ready_copy_ms",
    "ready_returned",
    "worker_unused_ready",
    "late_discarded",
    "worker_done",
    "cancelled",
    "snapshot_terminal",
    "shared_demand_pool",
    "consumer_waits",
    "late_model_publish",
    "routing_authority",
}

PROFILE_NAME = "H49-live-hidden-sequential-fast-predict-dual-reader-kv192-control"
PROFILE_FILE = Path(__file__).resolve().parent / "env" / PROFILE_NAME
EXPECTED_TOKEN_FILE = (
    Path(__file__).resolve().parent.parent
    / "hybrid-hot48-runner"
    / "expected-48-token-ids.json"
)
COMMON_ENV_METADATA = {
    "manifest_format",
    "HOME",
    "PATH",
    "TMPDIR",
    "binary",
    "cache_mib",
    "expected_token_ids_sha256",
}
PROVENANCE_ENV_METADATA = {
    "supervision_mode",
    "source_commit",
    "source_tree_clean",
    "binary_sha256",
    "binary_size",
    "binary_version",
    "runner_sha256",
    "manual_safety_sampler_sha256",
}


class ReceiptError(RuntimeError):
    """The run cannot support an H69 discriminator result."""


def _expect_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ReceiptError(
            f"{label} field set drifted: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReceiptError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _json_int(raw: str) -> int:
    if JSON_INT_RE.fullmatch(raw) is None:
        raise ReceiptError(f"non-canonical JSON integer {raw!r}")
    return int(raw)


def _json_decimal(raw: str) -> Decimal:
    if JSON_DECIMAL_RE.fullmatch(raw) is None:
        raise ReceiptError(f"non-canonical JSON decimal {raw!r}")
    return Decimal(raw)


def _json_constant(raw: str) -> Any:
    raise ReceiptError(f"non-finite JSON number {raw!r}")


def _loads_json(raw: str, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_json_pairs,
            parse_int=_json_int,
            parse_float=_json_decimal,
            parse_constant=_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ReceiptError(f"malformed {label}: {error}") from error


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReceiptError(f"cannot stat {label}: {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReceiptError(f"{label} is not a regular nonsymlink file: {path}")


def _read_text(path: Path, label: str) -> str:
    _regular_file(path, label)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReceiptError(f"cannot read {label}: {path}: {error}") from error


def _load_json(path: Path, label: str) -> Any:
    return _loads_json(_read_text(path, label), label)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_json_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise ReceiptError(f"{label} is not an integer >= {minimum}")
    return value


def _exact_json_scalar(value: Any, expected: Any) -> bool:
    return type(value) is type(expected) and value == expected


def _parse_fields(payload: str, label: str) -> dict[str, str]:
    if not payload or payload.startswith(" ") or payload.endswith(" ") or "  " in payload:
        raise ReceiptError(f"{label} has non-canonical field spacing")
    fields: dict[str, str] = {}
    for encoded in payload.split(" "):
        key, separator, value = encoded.partition("=")
        if not separator or not key or not value or key in fields:
            raise ReceiptError(f"{label} has a malformed or duplicate field")
        fields[key] = value
    return fields


def _parse_uint(raw: str, label: str, *, maximum: int | None = None) -> int:
    if INT_RE.fullmatch(raw) is None:
        raise ReceiptError(f"{label} is not a canonical unsigned decimal")
    value = int(raw)
    if maximum is not None and value > maximum:
        raise ReceiptError(f"{label} exceeds {maximum}")
    return value


def _parse_ms_us(raw: str, label: str) -> int:
    match = MS_RE.fullmatch(raw)
    if match is None:
        raise ReceiptError(f"{label} is not canonical millisecond precision M.mmm")
    return int(match.group(1)) * 1_000 + int(match.group(2))


def _format_ms_us(value: int) -> str:
    return f"{value // 1_000}.{value % 1_000:03d}"


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "na"
    scaled, remainder = divmod(numerator * 1_000_000, denominator)
    if remainder * 2 >= denominator:
        scaled += 1
    return f"{scaled // 1_000_000}.{scaled % 1_000_000:06d}"


def _check_ratio(raw: str, numerator: int, denominator: int, label: str) -> None:
    expected = _format_ratio(numerator, denominator)
    if raw != expected:
        raise ReceiptError(f"{label} was {raw!r}, recomputed {expected!r}")
    if raw != "na" and RATIO_RE.fullmatch(raw) is None:
        raise ReceiptError(f"{label} is not a canonical six-place ratio")


def _parse_layers(raw: str, cap: int) -> list[dict[str, int]]:
    records = raw.split(",")
    if len(records) != len(TARGET_LAYERS):
        raise ReceiptError(f"cap{cap}_layers does not contain exactly 29 records")
    parsed: list[dict[str, int]] = []
    pattern = re.compile(
        r"L(0|[1-9][0-9]*):(0|[1-9][0-9]*)/(0|[1-9][0-9]*)/"
        r"(0|[1-9][0-9]*)@((?:0|[1-9][0-9]*)\.[0-9]{3})"
    )
    for expected_layer, encoded in zip(TARGET_LAYERS, records):
        match = pattern.fullmatch(encoded)
        if match is None:
            raise ReceiptError(f"cap{cap} layer record is malformed: {encoded!r}")
        layer, hits, actual, predicted = (int(value) for value in match.groups()[:4])
        if layer != expected_layer:
            raise ReceiptError(
                f"cap{cap} layer order drifted: expected L{expected_layer}, got L{layer}"
            )
        parsed.append(
            {
                "layer": layer,
                "hits": hits,
                "actual": actual,
                "predicted": predicted,
                "wall_us": _parse_ms_us(match.group(5), f"cap{cap} L{layer} wall"),
            }
        )
    return parsed


def _parse_candidates(raw: str, cap: int) -> list[list[int]]:
    records = raw.split("/")
    if len(records) != len(TARGET_LAYERS):
        raise ReceiptError(f"cap{cap}_candidates does not contain exactly 29 layers")
    parsed: list[list[int]] = []
    for expected_layer, encoded in zip(TARGET_LAYERS, records):
        prefix = f"L{expected_layer}:"
        if not encoded.startswith(prefix):
            raise ReceiptError(f"cap{cap} candidate layer order drifted at {encoded!r}")
        body = encoded[len(prefix) :]
        ids = [] if body == "" else [
            _parse_uint(item, f"cap{cap} L{expected_layer} candidate", maximum=127)
            for item in body.split(",")
        ]
        if len(ids) > cap or len(set(ids)) != len(ids):
            raise ReceiptError(f"cap{cap} L{expected_layer} candidates are duplicated or over cap")
        parsed.append(ids)
    return parsed


def _validate_cap(fields: dict[str, str], cap: int) -> dict[str, Any]:
    layers = _parse_layers(fields[f"cap{cap}_layers"], cap)
    hits = sum(item["hits"] for item in layers)
    actual = sum(item["actual"] for item in layers)
    predicted = sum(item["predicted"] for item in layers)
    wall_us = sum(item["wall_us"] for item in layers)
    saved_us = sum(
        item["wall_us"] * item["hits"] // item["actual"] if item["actual"] else 0
        for item in layers
    )
    expected_ints = {
        "hits": hits,
        "actual_cold": actual,
        "predicted_cold": predicted,
    }
    for stem, expected in expected_ints.items():
        observed = _parse_uint(fields[f"cap{cap}_{stem}"], f"cap{cap}_{stem}")
        if observed != expected:
            raise ReceiptError(
                f"cap{cap}_{stem} was {observed}, per-layer recomputation was {expected}"
            )
    observed_wall = _parse_ms_us(fields[f"cap{cap}_read_wall_ms"], f"cap{cap}_read_wall_ms")
    observed_saved = _parse_ms_us(
        fields[f"cap{cap}_projected_saved_ms"], f"cap{cap}_projected_saved_ms"
    )
    if observed_wall != wall_us or observed_saved != saved_us:
        raise ReceiptError(f"cap{cap} wall/savings aggregate disagrees with its layers")
    _check_ratio(fields[f"cap{cap}_recall"], hits, actual, f"cap{cap}_recall")
    _check_ratio(fields[f"cap{cap}_precision"], hits, predicted, f"cap{cap}_precision")
    _check_ratio(
        fields[f"cap{cap}_read_wall_weighted_recall"],
        saved_us,
        wall_us,
        f"cap{cap}_read_wall_weighted_recall",
    )
    for item in layers:
        if (
            item["predicted"] > cap
            or item["actual"] > 128
            or item["hits"] > item["actual"]
            or item["hits"] > item["predicted"]
        ):
            raise ReceiptError(f"cap{cap} L{item['layer']} violates tally bounds")
    if not 0 <= saved_us <= wall_us:
        raise ReceiptError(f"cap{cap} projected savings lie outside read wall")
    return {
        "hits": hits,
        "actual_cold": actual,
        "predicted_cold": predicted,
        "read_wall_us": wall_us,
        "projected_saved_us": saved_us,
        "layers": layers,
    }


def parse_probe_line(line: str) -> dict[str, Any]:
    if not line.startswith(PROBE_PREFIX):
        raise ReceiptError("not an H69 probe line")
    fields = _parse_fields(line[len(PROBE_PREFIX) :], "H69 probe")
    if set(fields) == INVALID_PROBE_FIELDS:
        if (
            fields["schema"] == "2"
            and fields["admitted"] == "1"
            and fields["truth_valid"] == "0"
        ):
            raise ReceiptError(f"runtime invalidated H69 truth: {fields['reason']}")
        raise ReceiptError("malformed compact invalid H69 receipt")
    _expect_fields(fields, PROBE_FIELDS, "H69 probe")

    exact_strings = {
        "schema": "2",
        "admitted": "1",
        "profile": "exact-live-sequential-hot-shape",
        "source": "post-attention-slab-b",
        "target": "next-layer-exact-union",
        "first_target_layer": "1",
        "eligible": "29",
        "attempts": "29",
        "failures": "0",
        "truth_valid": "1",
        "predictor_top_k_per_row": "8",
        "probe_caps": "4,8,16",
        "projection": "record-linear-per-layer-wave-load-ceiling",
        "stage_admitted": "1",
        "launches": "29",
        "launch_failures": "0",
        "readiness_measured": "1",
        "contention_measured": "1",
        "output_mutation": "0",
        "io_mutation": "1",
        "slot_policy_mutation": "0",
        "routing_authority": "exact-router",
    }
    for key, expected in exact_strings.items():
        if fields[key] != expected:
            raise ReceiptError(f"H69 probe requires {key}={expected}, got {fields[key]!r}")
    start_pos = _parse_uint(fields["start_pos"], "probe start_pos")
    k_tokens = _parse_uint(fields["K"], "probe K")
    predict_us = _parse_uint(fields["predict_us"], "probe predict_us")
    if predict_us == 0:
        raise ReceiptError("probe predict_us was zero")

    caps = {cap: _validate_cap(fields, cap) for cap in CAPS}
    cap8_candidates = _parse_candidates(fields["cap8_candidates"], 8)
    cap16_candidates = _parse_candidates(fields["cap16_candidates"], 16)
    for index, layer in enumerate(TARGET_LAYERS):
        c4, c8, c16 = (caps[cap]["layers"][index] for cap in CAPS)
        if c4["actual"] != c8["actual"] or c8["actual"] != c16["actual"]:
            raise ReceiptError(f"L{layer} actual-cold truth differs across caps")
        if c4["wall_us"] != c8["wall_us"] or c8["wall_us"] != c16["wall_us"]:
            raise ReceiptError(f"L{layer} read wall differs across caps")
        if not (c4["hits"] <= c8["hits"] <= c16["hits"]):
            raise ReceiptError(f"L{layer} hit lattice is not monotone")
        if not (c4["predicted"] <= c8["predicted"] <= c16["predicted"]):
            raise ReceiptError(f"L{layer} prediction lattice is not monotone")
        if c8["hits"] - c4["hits"] > c8["predicted"] - c4["predicted"]:
            raise ReceiptError(f"L{layer} cap4-to-cap8 incremental hits are impossible")
        if c16["hits"] - c8["hits"] > c16["predicted"] - c8["predicted"]:
            raise ReceiptError(f"L{layer} cap8-to-cap16 incremental hits are impossible")
        ids8 = cap8_candidates[index]
        ids16 = cap16_candidates[index]
        if len(ids8) != c8["predicted"] or len(ids16) != c16["predicted"]:
            raise ReceiptError(f"L{layer} candidate identities disagree with predicted counts")
        if len(ids8) != min(8, len(ids16)):
            raise ReceiptError(
                f"L{layer} cap8 candidates are not the exact cap16 prefix width"
            )
        if ids8 != ids16[: len(ids8)]:
            raise ReceiptError(f"L{layer} cap8 candidates are not a cap16 prefix")
        if c4["predicted"] != min(4, len(ids8)):
            raise ReceiptError(
                f"L{layer} cap4 candidates are not the exact cap8 prefix width"
            )

    if len({caps[cap]["actual_cold"] for cap in CAPS}) != 1:
        raise ReceiptError("aggregate actual-cold truth differs across caps")
    if len({caps[cap]["read_wall_us"] for cap in CAPS}) != 1:
        raise ReceiptError("aggregate read wall differs across caps")
    if not (caps[4]["hits"] <= caps[8]["hits"] <= caps[16]["hits"]):
        raise ReceiptError("aggregate hits are not monotone")
    if not (
        caps[4]["predicted_cold"]
        <= caps[8]["predicted_cold"]
        <= caps[16]["predicted_cold"]
    ):
        raise ReceiptError("aggregate predicted counts are not monotone")
    if not (
        caps[4]["projected_saved_us"]
        <= caps[8]["projected_saved_us"]
        <= caps[16]["projected_saved_us"]
    ):
        raise ReceiptError("aggregate projected savings are not monotone")

    incremental_hits = caps[16]["hits"] - caps[8]["hits"]
    incremental_predicted = caps[16]["predicted_cold"] - caps[8]["predicted_cold"]
    incremental_saved_us = (
        caps[16]["projected_saved_us"] - caps[8]["projected_saved_us"]
    )
    if incremental_hits > incremental_predicted or incremental_saved_us < 0:
        raise ReceiptError("cap16 incremental values are impossible")
    observed_incremental_hits = _parse_uint(
        fields["cap16_incremental_hits_vs_cap8"], "incremental hits"
    )
    observed_incremental_predicted = _parse_uint(
        fields["cap16_incremental_predicted_vs_cap8"], "incremental predicted"
    )
    observed_incremental_saved = _parse_ms_us(
        fields["cap16_incremental_saved_ms_vs_cap8"], "incremental saved wall"
    )
    if (
        observed_incremental_hits != incremental_hits
        or observed_incremental_predicted != incremental_predicted
        or observed_incremental_saved != incremental_saved_us
    ):
        raise ReceiptError("runtime incremental fields disagree with recomputation")
    _check_ratio(
        fields["cap16_incremental_precision_vs_cap8"],
        incremental_hits,
        incremental_predicted,
        "cap16 incremental precision",
    )
    total_wave_load_us = _parse_ms_us(fields["total_wave_load_ms"], "total_wave_load_ms")
    if total_wave_load_us < caps[8]["read_wall_us"]:
        raise ReceiptError("total wave load is smaller than the L1-L29 read wall")
    stage_candidates = _parse_uint(fields["stage_candidates"], "stage_candidates")
    if stage_candidates != caps[8]["predicted_cold"]:
        raise ReceiptError("probe stage candidate count differs from cap8 prediction total")
    return {
        "start_pos": start_pos,
        "K": k_tokens,
        "predict_us": predict_us,
        "caps": caps,
        "incremental_hits": incremental_hits,
        "incremental_predicted": incremental_predicted,
        "incremental_saved_us": incremental_saved_us,
        "total_wave_load_us": total_wave_load_us,
        "layer0_wave_load_us": total_wave_load_us - caps[8]["read_wall_us"],
        "stage_candidates": stage_candidates,
    }


def parse_stage_line(line: str, probe: dict[str, Any]) -> dict[str, Any]:
    if not line.startswith(STAGE_PREFIX):
        raise ReceiptError("not a live-sequential stage line")
    fields = _parse_fields(line[len(STAGE_PREFIX) :], "live-sequential stage")
    _expect_fields(fields, STAGE_FIELDS, "live-sequential stage")
    values = {
        key: _parse_uint(fields[key], f"stage {key}")
        for key in STAGE_FIELDS
        if key not in {
            "predict_impl",
            "speculative_read_ms",
            "ready_copy_ms",
            "routing_authority",
        }
    }
    exact = {
        "schema": 1,
        "start_pos": probe["start_pos"],
        "K": probe["K"],
        "ok": 1,
        "cap": 8,
        "workers": 2,
        "workers_started": 2,
        "workers_done": 0,
        "launches": 29,
        "reads_failed": 0,
        "reads_in_flight": 0,
        "previous_ready_hits": 0,
        "ready_malformed": 0,
        "worker_done": 0,
        "cancelled": 0,
        "snapshot_terminal": 0,
        "shared_demand_pool": 0,
        "consumer_waits": 0,
        "late_model_publish": 0,
    }
    for key, expected in exact.items():
        if values[key] != expected:
            raise ReceiptError(f"stage requires {key}={expected}, got {values[key]}")
    if fields["predict_impl"] != "accelerate-sgemm":
        raise ReceiptError("stage predictor implementation is not accelerate-sgemm")
    if fields["routing_authority"] != "exact-router":
        raise ReceiptError("stage routing authority is not exact-router")
    _parse_ms_us(fields["speculative_read_ms"], "stage speculative_read_ms")
    _parse_ms_us(fields["ready_copy_ms"], "stage ready_copy_ms")

    if values["candidates"] != probe["caps"][8]["predicted_cold"]:
        raise ReceiptError("stage candidates disagree with cap8 predicted total")
    if values["candidates"] != probe["stage_candidates"]:
        raise ReceiptError("runtime probe and stage candidate accounting disagree")
    if values["reads_started"] != (
        values["reads_succeeded"] + values["reads_failed"] + values["reads_in_flight"]
    ):
        raise ReceiptError("stage read start/outcome accounting disagrees")
    if values["reads_started"] > values["candidates"]:
        raise ReceiptError("stage started more reads than candidates")
    if values["ready_returned"] != (
        values["ready_hits"] + values["ready_unused"] + values["ready_malformed"]
    ):
        raise ReceiptError("stage returned-ready accounting disagrees")
    if values["reads_succeeded"] != (
        values["ready_returned"]
        + values["worker_unused_ready"]
        + values["late_discarded"]
    ):
        raise ReceiptError("stage successful-read accounting disagrees")
    if values["ready_hits"] + values["previous_ready_hits"] + values["direct_fallback"] != values["exact_cold"]:
        raise ReceiptError("stage exact-cold accounting disagrees")
    if values["ready_hits"] > probe["caps"][8]["hits"]:
        raise ReceiptError("measured ready hits exceed cap8 truth hits")
    # Probe truth covers L1-L29, while stage accounting also covers L0 and is
    # sampled at a different point in the request.  Keep the signed scope delta:
    # a negative value is observed in valid H49 traces and is not an underflow.
    exact_cold_scope_delta = (
        values["exact_cold"] - probe["caps"][8]["actual_cold"]
    )
    # Stage admission can fall back independently for any layer, so probe
    # cold-positive layers do not determine the number of StageCold seals.
    if values["seals"] > 30:
        raise ReceiptError("stage seal count exceeds the 30-layer request")
    return {
        **values,
        "exact_cold_minus_probe_l1_l29_actual": exact_cold_scope_delta,
    }


def _parse_profile(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _read_text(path, "H49 environment profile").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or ENV_KEY_RE.fullmatch(key) is None or key in values:
            raise ReceiptError(f"malformed or duplicate H49 profile line {line!r}")
        values[key] = value
    return values


def parse_env_manifest(path: Path) -> dict[str, str]:
    lines = _read_text(path, "environment manifest").splitlines()
    if not lines or lines[0] != "manifest_format=base64-v1":
        raise ReceiptError("environment manifest is not strict base64-v1")
    if any(not line or line.startswith("#") for line in lines):
        raise ReceiptError("environment manifest contains blank/comment lines")
    values = {"manifest_format": "base64-v1"}
    encoded_re = re.compile(rf"({ENV_KEY_RE.pattern})@BASE64=([A-Za-z0-9+/]*={{0,2}})")
    for line in lines[1:]:
        match = encoded_re.fullmatch(line)
        if match is None:
            raise ReceiptError(f"malformed base64-v1 environment line {line!r}")
        key, encoded = match.groups()
        if key in values:
            raise ReceiptError(f"duplicate environment field {key}")
        try:
            raw = base64.b64decode(encoded, validate=True)
            value = raw.decode("utf-8")
        except (binascii.Error, UnicodeError) as error:
            raise ReceiptError(f"invalid base64 environment value for {key}") from error
        if base64.b64encode(raw).decode("ascii") != encoded:
            raise ReceiptError(f"non-canonical base64 environment value for {key}")
        values[key] = value
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_text(path, "expected token fixture").encode("utf-8")).hexdigest()


def validate_environment(run_dir: Path, safety_mode: str) -> dict[str, str]:
    expected = _parse_profile(PROFILE_FILE)
    observed = parse_env_manifest(run_dir / "env.txt")
    metadata = COMMON_ENV_METADATA | PROVENANCE_ENV_METADATA
    _expect_fields(observed, set(expected) | metadata, "environment manifest")
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if key == "CAMELID_GEMMA4_MTP_ASSISTANT_PATH":
            if not observed_value.startswith("/") or not observed_value.endswith(
                "/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors"
            ):
                raise ReceiptError("H49 assistant path is absent or unexpected")
        elif observed_value != expected_value:
            raise ReceiptError(
                f"H49 environment drifted at {key}: expected {expected_value!r}, "
                f"observed {observed_value!r}"
            )
    slots = [
        _parse_uint(item, "H49 per-layer slot count")
        for item in observed["CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER"].split(",")
    ]
    if len(slots) != 30 or sum(slots) != 1_408:
        raise ReceiptError("H49 environment is not the literal 1,408-slot profile")
    if observed["CAMELID_GEMMA4_KV_INIT"] != "192":
        raise ReceiptError("H49 environment is not KV_INIT=192")
    exact_metadata = {
        "manifest_format": "base64-v1",
        "HOME": "/Users/timtoole",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/tmp",
        "cache_mib": "0",
        "expected_token_ids_sha256": _sha256(EXPECTED_TOKEN_FILE),
    }
    for key, expected_value in exact_metadata.items():
        if observed[key] != expected_value:
            raise ReceiptError(f"environment metadata requires {key}={expected_value!r}")
    if not observed["binary"].startswith("/") or "\n" in observed["binary"]:
        raise ReceiptError("environment binary path is not an absolute single line")
    expected_supervision = (
        "manual-no-watchdog" if safety_mode == "manual-no-watchdog" else "strict-watchdog"
    )
    if observed["supervision_mode"] != expected_supervision:
        raise ReceiptError("environment supervision mode disagrees with safety receipt")
    if COMMIT_RE.fullmatch(observed["source_commit"]) is None:
        raise ReceiptError("source_commit is not a full lowercase commit")
    if observed["source_tree_clean"] != "1":
        raise ReceiptError("source tree was not recorded clean")
    for key in (
        "binary_sha256",
        "runner_sha256",
        "manual_safety_sampler_sha256",
    ):
        if SHA256_RE.fullmatch(observed[key]) is None:
            raise ReceiptError(f"{key} is not a lowercase SHA-256")
    if _parse_uint(observed["binary_size"], "binary_size") == 0:
        raise ReceiptError("binary_size is zero")
    version = observed["binary_version"]
    if not version or "\n" in version or f"g{observed['source_commit'][:8]}" not in version:
        raise ReceiptError("binary_version does not bind the source commit")
    return observed


def _expected_token_ids() -> list[int]:
    _regular_file(EXPECTED_TOKEN_FILE, "expected token fixture")
    value = _load_json(EXPECTED_TOKEN_FILE, "expected token fixture")
    if (
        type(value) is not list
        or len(value) != 48
        or any(not _is_int(item) or item < 0 for item in value)
    ):
        raise ReceiptError("expected token fixture is not exactly 48 integer IDs")
    return value


def validate_response(run_dir: Path) -> None:
    response = _load_json(run_dir / "response.json", "response JSON")
    if type(response) is not dict:
        raise ReceiptError("response JSON is not an object")
    _expect_fields(
        response,
        {"id", "object", "created", "model", "choices", "usage", "camelid"},
        "response",
    )
    if (
        response["id"] != "chatcmpl-gemma4"
        or response["object"] != "chat.completion"
        or response["model"] != "26B_dequant_it_hf"
        or not _is_int(response["created"])
        or response["created"] < 0
    ):
        raise ReceiptError("response identity is malformed")
    choices = response["choices"]
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ReceiptError("response does not have exactly one choice")
    choice = choices[0]
    _expect_fields(choice, {"index", "message", "finish_reason"}, "response choice")
    if not _exact_json_scalar(choice["index"], 0) or choice["finish_reason"] != "length":
        raise ReceiptError("response choice identity drifted")
    if type(choice["message"]) is not dict:
        raise ReceiptError("response message is not an object")
    _expect_fields(choice["message"], {"role", "content"}, "response message")
    if choice["message"]["role"] != "assistant" or type(choice["message"]["content"]) is not str:
        raise ReceiptError("response message is malformed")
    usage = response["usage"]
    if type(usage) is not dict:
        raise ReceiptError("response usage is not an object")
    _expect_fields(usage, {"prompt_tokens", "completion_tokens", "total_tokens"}, "response usage")
    expected_usage = {"prompt_tokens": 104, "completion_tokens": 48, "total_tokens": 152}
    if any(not _exact_json_scalar(usage[key], value) for key, value in expected_usage.items()):
        raise ReceiptError("response token accounting is not exact 104+48")
    camelid = response["camelid"]
    if type(camelid) is not dict:
        raise ReceiptError("response camelid receipt is not an object")
    _expect_fields(
        camelid,
        {"generated_token_ids", "timings_ms", "exact_match_expected", "exact_match_count"},
        "response camelid receipt",
    )
    if (
        camelid["exact_match_expected"] is not True
        or not _exact_json_scalar(camelid["exact_match_count"], 48)
        or camelid["generated_token_ids"] != _expected_token_ids()
    ):
        raise ReceiptError("response does not exactly match the frozen 48-token fixture")
    timings = camelid["timings_ms"]
    if type(timings) is not dict:
        raise ReceiptError("response timings receipt is not an object")
    _expect_fields(timings, {"generate", "generation", "prompt_evaluation", "lane"}, "response timings")
    if type(timings["generation"]) is not dict or type(timings["prompt_evaluation"]) is not dict:
        raise ReceiptError("response timing subobjects are malformed")
    _expect_fields(timings["generation"], {"forward_total"}, "response generation timing")
    _expect_fields(timings["prompt_evaluation"], set(), "response prompt timing")
    for key, value in (("generate", timings["generate"]), ("forward_total", timings["generation"]["forward_total"])):
        if type(value) not in (int, Decimal) or value <= 0:
            raise ReceiptError(f"response timing {key} is not finite and positive")
    if timings["generate"] != timings["generation"]["forward_total"]:
        raise ReceiptError("response generation timing totals disagree")
    if timings["lane"] != "gemma4_wall_clock_total_only":
        raise ReceiptError("response timing lane drifted")


def validate_health(run_dir: Path, environment: dict[str, str]) -> None:
    health = _load_json(run_dir / "health.json", "health JSON")
    if type(health) is not dict:
        raise ReceiptError("health JSON is not an object")
    required = {
        "ok": True,
        "engine": "camelid",
        "generation_ready": True,
        "model_family": "gemma4",
        "gemma4_available": True,
        "gemma4_serve_lane": "ghost_moe",
        "gemma4_ghost_common_metal_active": True,
        "gemma4_ghost_execution_mode": "full_common_metal",
        "gemma4_mtp_assistant_loaded": True,
        "gemma4_mtp_full_q4_active": True,
        "gemma4_ghost_experts_metal_active": True,
        "gemma4_ghost_head_metal_active": True,
        "gemma4_ghost_backend": "metal",
    }
    missing = sorted(set(required) - set(health))
    if missing:
        raise ReceiptError(f"health JSON lacks readiness fields: {missing}")
    for key, expected in required.items():
        if not _exact_json_scalar(health[key], expected):
            raise ReceiptError(f"health readiness requires {key}={expected!r}")
    if "source_commit" in environment:
        if health.get("source_commit") != environment["source_commit"]:
            raise ReceiptError("health source commit disagrees with environment provenance")
        if "dirty" in str(health.get("build", "")):
            raise ReceiptError("health endpoint reports a dirty binary build")


MANUAL_TOP_FIELDS = {
    "schema_version",
    "mode",
    "qualifying",
    "point_samples_only",
    "configured_caps",
    "samples",
    "sampled_peaks",
    "child",
    "report",
    "checks",
}
MANUAL_HOST_FIELDS = {
    "page_size_bytes",
    "swapped_pages_current",
    "swapped_bytes_current",
    "swapins_pages",
    "swapouts_pages",
    "pressure_level_raw",
    "wired_bytes",
}
MANUAL_PROCESS_FIELDS = {
    "pid",
    "process_group",
    "member_pids",
    "rss_bytes",
    "physical_footprint_bytes",
}
MANUAL_CHECK_FIELDS = {
    "current_swap_zero",
    "swapins_unchanged",
    "swapouts_unchanged",
    "pressure_within_cap",
    "host_wired_within_cap",
    "child_physical_footprint_within_cap",
    "child_returned_zero",
    "process_group_empty",
    "report_valid",
    "all_passed",
}


def validate_manual_safety(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manual-safety.json"
    receipt = _load_json(path, "manual safety receipt")
    if type(receipt) is not dict:
        raise ReceiptError("manual safety receipt is not an object")
    _expect_fields(receipt, MANUAL_TOP_FIELDS, "manual safety receipt")
    if (
        not _exact_json_scalar(receipt["schema_version"], 1)
        or receipt["mode"] != "manual-no-watchdog"
        or receipt["qualifying"] is not False
        or receipt["point_samples_only"] is not True
    ):
        raise ReceiptError("manual safety identity fields drifted")
    caps = receipt["configured_caps"]
    if type(caps) is not dict:
        raise ReceiptError("manual configured_caps is not an object")
    expected_caps = {
        "maximum_child_physical_footprint_bytes": MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES,
        "maximum_host_wired_bytes": MAX_HOST_WIRED_BYTES,
        "maximum_pressure_level_raw": 1,
        "require_zero_current_swap": True,
        "reject_swapin_growth": True,
        "reject_swapout_growth": True,
    }
    _expect_fields(caps, set(expected_caps), "manual configured_caps")
    if any(not _exact_json_scalar(caps[key], value) for key, value in expected_caps.items()):
        raise ReceiptError("manual safety caps are not the exact Mini2 treaty")

    samples = receipt["samples"]
    if type(samples) is not dict:
        raise ReceiptError("manual samples is not an object")
    _expect_fields(samples, {"pre", "ready", "post"}, "manual samples")
    parsed_samples: dict[str, dict[str, Any]] = {}
    for phase in ("pre", "ready", "post"):
        sample = samples[phase]
        if type(sample) is not dict:
            raise ReceiptError(f"manual {phase} sample is not an object")
        _expect_fields(sample, {"schema_version", "phase", "host", "process"}, f"manual {phase} sample")
        if not _exact_json_scalar(sample["schema_version"], 1) or sample["phase"] != phase:
            raise ReceiptError(f"manual {phase} sample identity drifted")
        host = sample["host"]
        if type(host) is not dict:
            raise ReceiptError(f"manual {phase} host sample is not an object")
        _expect_fields(host, MANUAL_HOST_FIELDS, f"manual {phase} host")
        for key, value in host.items():
            _require_json_int(value, f"manual {phase} host {key}")
        if host["page_size_bytes"] != 16_384:
            raise ReceiptError("manual safety sample is not from a 16 KiB-page Mini2 host")
        if (
            host["swapped_pages_current"] != 0
            or host["swapped_bytes_current"] != 0
            or host["swapins_pages"] != 0
            or host["swapouts_pages"] != 0
        ):
            raise ReceiptError(f"manual {phase} sample has absolute swap activity")
        if host["pressure_level_raw"] != 1:
            raise ReceiptError(f"manual {phase} sample is outside pressure level 1")
        if not 0 < host["wired_bytes"] <= MAX_HOST_WIRED_BYTES:
            raise ReceiptError(f"manual {phase} wired memory exceeds the cap")
        process = sample["process"]
        if phase == "pre":
            if process is not None:
                raise ReceiptError("manual pre sample unexpectedly has a child process")
        else:
            if type(process) is not dict:
                raise ReceiptError(f"manual {phase} process sample is absent")
            _expect_fields(process, MANUAL_PROCESS_FIELDS, f"manual {phase} process")
            for key in ("pid", "process_group", "rss_bytes", "physical_footprint_bytes"):
                _require_json_int(process[key], f"manual {phase} process {key}", minimum=1)
            member_pids = process["member_pids"]
            if (
                type(member_pids) is not list
                or not member_pids
                or any(not _is_int(pid) or pid <= 0 for pid in member_pids)
                or len(member_pids) != len(set(member_pids))
                or process["pid"] not in member_pids
                or process["pid"] != process["process_group"]
            ):
                raise ReceiptError(f"manual {phase} process-group identity is malformed")
            if process["physical_footprint_bytes"] > MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES:
                raise ReceiptError(f"manual {phase} child footprint exceeds the cap")
        parsed_samples[phase] = sample

    ready_process = parsed_samples["ready"]["process"]
    post_process = parsed_samples["post"]["process"]
    if (
        ready_process["pid"] != post_process["pid"]
        or ready_process["process_group"] != post_process["process_group"]
    ):
        raise ReceiptError("manual ready/post process identity drifted")
    peaks = receipt["sampled_peaks"]
    if type(peaks) is not dict:
        raise ReceiptError("manual sampled_peaks is not an object")
    expected_peaks = {
        "child_rss_bytes": max(ready_process["rss_bytes"], post_process["rss_bytes"]),
        "child_physical_footprint_bytes": max(
            ready_process["physical_footprint_bytes"], post_process["physical_footprint_bytes"]
        ),
        "host_wired_bytes": max(sample["host"]["wired_bytes"] for sample in parsed_samples.values()),
    }
    _expect_fields(peaks, set(expected_peaks), "manual sampled_peaks")
    if peaks != expected_peaks:
        raise ReceiptError("manual sampled peaks disagree with point samples")

    child = receipt["child"]
    if type(child) is not dict:
        raise ReceiptError("manual child final is not an object")
    _expect_fields(child, {"pid", "process_group", "returncode", "process_group_empty"}, "manual child")
    expected_child = {
        "pid": ready_process["pid"],
        "process_group": ready_process["process_group"],
        "returncode": 0,
        "process_group_empty": True,
    }
    if any(not _exact_json_scalar(child[key], value) for key, value in expected_child.items()):
        raise ReceiptError("manual child final is not a clean zero/empty exit")
    report = receipt["report"]
    if type(report) is not dict:
        raise ReceiptError("manual report metadata is not an object")
    _expect_fields(report, {"exists", "is_regular_file", "is_symlink", "size_bytes"}, "manual report")
    response_path = run_dir / "response.json"
    response_size = response_path.stat().st_size
    expected_report = {
        "exists": True,
        "is_regular_file": True,
        "is_symlink": False,
        "size_bytes": response_size,
    }
    if any(not _exact_json_scalar(report[key], value) for key, value in expected_report.items()):
        raise ReceiptError("manual report metadata disagrees with response.json")
    checks = receipt["checks"]
    if type(checks) is not dict:
        raise ReceiptError("manual checks is not an object")
    _expect_fields(checks, MANUAL_CHECK_FIELDS, "manual checks")
    if any(value is not True for value in checks.values()):
        raise ReceiptError("manual safety checks did not all pass")
    return {
        "mode": "manual-no-watchdog",
        "qualifying": False,
        "point_samples_only": True,
        "peak_child_physical_footprint_bytes": peaks["child_physical_footprint_bytes"],
        "peak_host_wired_bytes": peaks["host_wired_bytes"],
    }


WATCHDOG_HOST_FIELDS = {
    "sample_started_monotonic_ns",
    "observed_monotonic_ns",
    "sample_duration_ns",
    "unix_time_ns",
    "page_size_bytes",
    "free_pages_raw_including_speculative",
    "speculative_pages",
    "free_pages_strict",
    "free_bytes_strict",
    "active_bytes",
    "inactive_bytes",
    "reclaimable_headroom_bytes",
    "wired_bytes",
    "compressor_occupied_bytes",
    "compressed_logical_bytes",
    "pageins_bytes",
    "pageouts_bytes",
    "swapins_pages",
    "swapins_bytes",
    "swapouts_pages",
    "swapouts_bytes",
    "swapped_pages_current",
    "swapped_bytes_current",
    "pressure_level_raw",
}


def _validate_watchdog_host(host: Any, label: str) -> None:
    if type(host) is not dict:
        raise ReceiptError(f"{label} host sample is not an object")
    _expect_fields(host, WATCHDOG_HOST_FIELDS, f"{label} host")
    for key, value in host.items():
        _require_json_int(value, f"{label} host {key}")
    if (
        host["swapped_pages_current"] != 0
        or host["swapped_bytes_current"] != 0
        or host["swapins_pages"] != 0
        or host["swapouts_pages"] != 0
        or host["pressure_level_raw"] != 1
        or not 0 < host["wired_bytes"] <= MAX_HOST_WIRED_BYTES
    ):
        raise ReceiptError(f"{label} host violates swap, pressure, or wired-memory limits")


def validate_watchdog_safety(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "memory-watchdog.jsonl"
    raw_lines = _read_text(path, "watchdog JSONL").splitlines()
    if not raw_lines or any(not line for line in raw_lines):
        raise ReceiptError("watchdog JSONL is empty or contains blank lines")
    events = [_loads_json(line, f"watchdog JSONL line {index}") for index, line in enumerate(raw_lines, 1)]
    if any(type(event) is not dict for event in events):
        raise ReceiptError("watchdog JSONL contains a non-object")
    if any(event.get("schema_version") != 3 for event in events):
        raise ReceiptError("watchdog JSONL is not uniformly schema 3")
    allowed = {
        "clean_parent_baseline",
        "baseline_soak_sample",
        "baseline_soak_complete",
        "child_started",
        "sample",
        "post_exit_sample",
        "final",
    }
    kinds = [event.get("event") for event in events]
    if any(kind not in allowed for kind in kinds):
        raise ReceiptError("watchdog JSONL contains a refusal or abort event")
    for required in ("clean_parent_baseline", "baseline_soak_complete", "child_started", "post_exit_sample", "final"):
        if kinds.count(required) != 1:
            raise ReceiptError(f"watchdog requires exactly one {required} event")
    if kinds[-1] != "final" or kinds.index("clean_parent_baseline") > kinds.index("child_started"):
        raise ReceiptError("watchdog event order is incomplete")
    if kinds.index("child_started") > kinds.index("post_exit_sample"):
        raise ReceiptError("watchdog start/post-exit order is invalid")

    baseline = events[kinds.index("clean_parent_baseline")]
    post = events[kinds.index("post_exit_sample")]
    start = events[kinds.index("child_started")]
    final = events[-1]
    for label, event in (("baseline", baseline), ("post-exit", post)):
        if event.get("violations") != []:
            raise ReceiptError(f"watchdog {label} event contains violations")
        _validate_watchdog_host(event.get("host"), label)
    for index, event in enumerate(events):
        if event.get("event") in {"baseline_soak_sample", "sample"}:
            if event.get("violations") != []:
                raise ReceiptError(f"watchdog sample {index} contains violations")
            _validate_watchdog_host(event.get("host"), f"sample {index}")
            process = event.get("process")
            if process is not None:
                if type(process) is not dict:
                    raise ReceiptError("watchdog process sample is malformed")
                footprint = _require_json_int(
                    process.get("physical_footprint_bytes"), "watchdog process footprint", minimum=1
                )
                if footprint > MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES:
                    raise ReceiptError("watchdog process sample exceeds child footprint cap")
    pid = _require_json_int(start.get("pid"), "watchdog child PID", minimum=1)
    pgid = _require_json_int(start.get("process_group"), "watchdog child process group", minimum=1)
    expected_start = {
        "process_accounting_scope": "isolated_process_group_aggregate",
        "maximum_child_physical_footprint_bytes": MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES,
        "maximum_host_wired_bytes": MAX_HOST_WIRED_BYTES,
        "require_zero_current_swap": True,
        "maximum_pressure_level_raw": 1,
        "reject_swapin_growth": True,
        "baseline_swapins_pages": 0,
        "baseline_swapouts_pages": 0,
        "report_producer": "external",
    }
    if pid != pgid or any(start.get(key) != value for key, value in expected_start.items()):
        raise ReceiptError("watchdog child start does not carry the exact safety contract")
    report_path = start.get("report")
    if type(report_path) is not str or Path(report_path).name != "response.json":
        raise ReceiptError("watchdog report path is not response.json")
    expected_final = {
        "pid": pid,
        "child_returncode": 0,
        "watchdog_aborted": False,
        "abort_reasons": [],
        "baseline_swapins_pages": 0,
        "baseline_swapouts_pages": 0,
        "process_group": pgid,
        "process_group_empty": True,
        "process_accounting_scope": "isolated_process_group_aggregate",
        "report_exists": True,
        "report_is_regular_file": True,
        "report_is_symlink": False,
    }
    if any(final.get(key) != value for key, value in expected_final.items()):
        raise ReceiptError("watchdog final event is not a clean completed receipt")
    peak_child = _require_json_int(
        final.get("peak_child_physical_footprint_bytes"), "watchdog peak child footprint", minimum=1
    )
    peak_wired = _require_json_int(final.get("peak_host_wired_bytes"), "watchdog peak wired memory", minimum=1)
    if peak_child > MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES or peak_wired > MAX_HOST_WIRED_BYTES:
        raise ReceiptError("watchdog final peaks exceed configured caps")
    if final.get("report_size_bytes") != (run_dir / "response.json").stat().st_size:
        raise ReceiptError("watchdog final report size disagrees with response.json")
    return {
        "mode": "strict-watchdog",
        "qualifying": True,
        "point_samples_only": False,
        "peak_child_physical_footprint_bytes": peak_child,
        "peak_host_wired_bytes": peak_wired,
    }


def validate_safety(run_dir: Path) -> dict[str, Any]:
    manual = run_dir / "manual-safety.json"
    watchdog = run_dir / "memory-watchdog.jsonl"
    manual_exists = manual.exists() or manual.is_symlink()
    watchdog_exists = watchdog.exists() or watchdog.is_symlink()
    if manual_exists == watchdog_exists:
        raise ReceiptError("run must contain exactly one manual or strict-watchdog safety receipt")
    return validate_manual_safety(run_dir) if manual_exists else validate_watchdog_safety(run_dir)


def parse_observations(log: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probe_lines = [line for line in log.splitlines() if line.startswith(PROBE_PREFIX)]
    stage_lines = [line for line in log.splitlines() if line.startswith(STAGE_PREFIX)]
    if len(probe_lines) != 4 or len(stage_lines) != 4:
        raise ReceiptError(
            f"request requires exactly four probes/stages; got {len(probe_lines)}/{len(stage_lines)}"
        )
    probes = [parse_probe_line(line) for line in probe_lines]
    observed_rounds = tuple((probe["start_pos"], probe["K"]) for probe in probes)
    if observed_rounds != EXPECTED_ROUNDS:
        raise ReceiptError(f"probe round schedule drifted: {observed_rounds!r}")
    if len(set(observed_rounds)) != 4:
        raise ReceiptError("probe request contains duplicate rounds")
    stages = [parse_stage_line(line, probe) for line, probe in zip(stage_lines, probes)]
    sequences = [stage["round_seq"] for stage in stages]
    if sequences != list(range(sequences[0], sequences[0] + 4)):
        raise ReceiptError("stage round_seq values are not an arbitrary consecutive sequence")
    return probes, stages


def _number(numerator: int, denominator: int = 1) -> float:
    return float(Decimal(numerator) / Decimal(denominator))


def analyze(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ReceiptError("run_dir is not a real directory")
    safety = validate_safety(run_dir)
    environment = validate_environment(run_dir, safety["mode"])
    validate_health(run_dir, environment)
    validate_response(run_dir)
    log = _read_text(run_dir / "server.log", "server log")
    probes, stages = parse_observations(log)

    cap_totals: dict[int, dict[str, int]] = {}
    for cap in CAPS:
        cap_totals[cap] = {
            key: sum(probe["caps"][cap][key] for probe in probes)
            for key in (
                "hits",
                "actual_cold",
                "predicted_cold",
                "read_wall_us",
                "projected_saved_us",
            )
        }
    incremental_hits = cap_totals[16]["hits"] - cap_totals[8]["hits"]
    incremental_predicted = (
        cap_totals[16]["predicted_cold"] - cap_totals[8]["predicted_cold"]
    )
    incremental_saved_us = (
        cap_totals[16]["projected_saved_us"] - cap_totals[8]["projected_saved_us"]
    )
    if incremental_predicted <= 0:
        raise ReceiptError("request cap16 incremental prediction denominator is not positive")
    if incremental_hits < 0 or incremental_hits > incremental_predicted or incremental_saved_us < 0:
        raise ReceiptError("request incremental projection is impossible")
    ready_hits = sum(stage["ready_hits"] for stage in stages)
    cap8_hits = cap_totals[8]["hits"]
    if cap8_hits <= 0 or ready_hits > cap8_hits:
        raise ReceiptError("request cannot compute a bounded measured cap8 readiness")
    saved_threshold_passed = incremental_saved_us >= GO_SAVED_US
    precision_threshold_passed = (
        incremental_hits * GO_PRECISION_DENOMINATOR
        >= incremental_predicted * GO_PRECISION_NUMERATOR
    )
    verdict = "GO" if saved_threshold_passed and precision_threshold_passed else "NO-GO"

    rounds = []
    for probe, stage in zip(probes, stages):
        rounds.append(
            {
                "start_pos": probe["start_pos"],
                "K": probe["K"],
                "stage_round_seq": stage["round_seq"],
                "predict_us": probe["predict_us"],
                "cap8_ready_hits_measured": stage["ready_hits"],
                "stage_exact_cold_minus_probe_l1_l29_actual": stage[
                    "exact_cold_minus_probe_l1_l29_actual"
                ],
                "layer0_wave_load_ms": _number(probe["layer0_wave_load_us"], 1_000),
                "caps": {
                    str(cap): {
                        "hits": probe["caps"][cap]["hits"],
                        "actual_cold": probe["caps"][cap]["actual_cold"],
                        "predicted_cold": probe["caps"][cap]["predicted_cold"],
                        "recall": _number(
                            probe["caps"][cap]["hits"], probe["caps"][cap]["actual_cold"]
                        ) if probe["caps"][cap]["actual_cold"] else None,
                        "precision": _number(
                            probe["caps"][cap]["hits"], probe["caps"][cap]["predicted_cold"]
                        ) if probe["caps"][cap]["predicted_cold"] else None,
                        "read_wall_ms": _number(probe["caps"][cap]["read_wall_us"], 1_000),
                        "projected_saved_ms": _number(
                            probe["caps"][cap]["projected_saved_us"], 1_000
                        ),
                    }
                    for cap in CAPS
                },
            }
        )
    request_caps = {
        str(cap): {
            "hits": values["hits"],
            "actual_cold": values["actual_cold"],
            "predicted_cold": values["predicted_cold"],
            "recall": _number(values["hits"], values["actual_cold"])
            if values["actual_cold"] else None,
            "precision": _number(values["hits"], values["predicted_cold"])
            if values["predicted_cold"] else None,
            "read_wall_ms": _number(values["read_wall_us"], 1_000),
            "projected_saved_ms": _number(values["projected_saved_us"], 1_000),
        }
        for cap, values in cap_totals.items()
    }
    readiness_adjusted_cap16_us = Decimal(cap_totals[16]["projected_saved_us"]) * Decimal(
        ready_hits
    ) / Decimal(cap8_hits)
    readiness_adjusted_incremental_us = Decimal(incremental_saved_us) * Decimal(ready_hits) / Decimal(
        cap8_hits
    )
    return {
        "schema_version": 1,
        "run": run_dir.name,
        "integrity": {
            "exact_h49_kv192_1408_environment": True,
            "exact_48_token_response": True,
            "four_ordered_probe_and_stage_receipts": True,
            "cap_lattice_recomputed": True,
            "stage_accounting_recomputed": True,
            "safety_mode": safety["mode"],
            "qualifying_safety_receipt": safety["qualifying"],
            "point_samples_only": safety["point_samples_only"],
            "observation_only": True,
            "projected_savings_are_measured": False,
            "throughput_promotion_allowed": False,
        },
        "safety": safety,
        "rounds": rounds,
        "request_projection": {
            "caps": request_caps,
            "cap16_incremental_hits_vs_cap8": incremental_hits,
            "cap16_incremental_predicted_vs_cap8": incremental_predicted,
            "cap16_incremental_precision_vs_cap8": _number(
                incremental_hits, incremental_predicted
            ),
            "cap16_incremental_projected_saved_ms_vs_cap8": _number(
                incremental_saved_us, 1_000
            ),
            "measured_cap8_readiness": _number(ready_hits, cap8_hits),
            "readiness_adjusted_cap16_projected_saved_ms": float(
                readiness_adjusted_cap16_us / Decimal(1_000)
            ),
            "readiness_adjusted_cap16_incremental_projected_saved_ms_vs_cap8": float(
                readiness_adjusted_incremental_us / Decimal(1_000)
            ),
        },
        "decision": {
            "verdict": verdict,
            "saved_threshold_passed": saved_threshold_passed,
            "precision_threshold_passed": precision_threshold_passed,
            "thresholds": {
                "cap16_incremental_projected_saved_ms_vs_cap8": 50.0,
                "cap16_incremental_precision_vs_cap8": 0.30,
            },
            "qualification": (
                "QUALIFYING_STRICT_WATCHDOG"
                if safety["qualifying"]
                else "NON_QUALIFYING_MANUAL_POINT_SAMPLES"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.run_dir)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (OSError, ReceiptError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
