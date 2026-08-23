#!/usr/bin/env python3
"""Fail-closed analyzer for the exact Gemma 4 48-token / 960 ms gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


EXPECTED_TOKENS = 48
MAX_DECODE_WALL_MS = 960.0
REQUEST_SHA256 = "b2f1110079fc726699cc936a628a268a7ec5bf2076fa970899de39d4ea903939"
EXPECTED_TOKEN_IDS_SHA256 = (
    "45e65ac09155d7627373c262f1edd1faf6188fb6dad26c5d5994fe5226a97975"
)
PROFILE = (
    39, 40, 33, 30, 30, 31, 31, 30, 34, 30,
    26, 28, 30, 31, 28, 37, 31, 30, 31, 32,
    31, 32, 30, 31, 32, 35, 32, 34, 34, 37,
)
PROFILE_CSV = ",".join(str(value) for value in PROFILE)
MAPPED_READAHEAD_MAX_INFLIGHT_RECORDS = 64
FULL_Q4_MATRIX_BYTES = 236_077_056
FULL_Q4_SOURCE_SHA256 = "c082cc581c3ec90d70285c1a41c81544ff56cbc96650f16c900a280940655801"
BF16_MATRIX_BYTES = 839_385_088
MAPPED_READAHEAD_MARKER = (
    "[gemma4-ghost-metal] mapped-cold readahead policy: "
    "CAMELID_GEMMA4_GHOST_METAL_MAPPED_READAHEAD effective=1 "
    "scope=selected-cold-only advice=MADV_WILLNEED dispatch=async-read-pool"
)
PREVIOUS_UNION_READAHEAD_MARKER = (
    "[gemma4-ghost-metal] mapped-cold previous-union policy: "
    "source=previous-target-exact-routed-union timing=before-assistant "
    "scope=selected-cold-only advice=MADV_WILLNEED dispatch=async-read-pool "
    "correctness_dependency=0"
)
MAPPED_READAHEAD_REFUSAL_MARKER = (
    "[gemma4-ghost-metal] mapped-cold MADV_WILLNEED refused:"
)
PACKED_K8_GATEUP_MARKER = (
    "[gemma4 exact partition] packed_k8_gateup=row_complete "
    "runtime_width_oracle=raw_bit_exact"
)
COMPACT_K8_HEAD_MARKER = (
    "[metal] CAMELID_GEMMA4_HEAD_SPEC50_K8_COMPACT=1 "
    "exact RB2/SG4 dispatch active"
)
FULL_Q4_RESIDENCY_MARKER = (
    "[gemma4-mtp full-q4 residency] source_retained=false mapped_bytes=0 "
    "locked_bytes=0 resident_pages=0 total_pages=0 packed_bytes=236077056"
)
FULL_Q4_MARKER_PATTERN = re.compile(
    r"^\[gemma4-mtp full-q4\] enabled=true source_sha256=([0-9a-f]{64}) "
    r"matrices=23 packed_bytes=(\d+) bf16_matrix_bytes=(\d+) quantize_us=(\d+) "
    r"norms_quantized=false fallback=false$",
    re.MULTILINE,
)
DEVICE_CHAIN_PATTERN = re.compile(
    r"^\[gemma4-mtp device-chain\] requested_drafts=(\d+) returned_drafts=(\d+) "
    r"command_buffers=1 commits=1 waits=1 cpu_embedding_callbacks=0 "
    r"linear_format=([^ ]+) matrix_bytes_per_draft=(\d+) "
    r"encode_us=(\d+) wait_us=(\d+) gpu_us=(\d+) kernel_us=(\d+) wall_us=(\d+)$",
    re.MULTILINE,
)


class GateError(RuntimeError):
    pass


def require_sha256(path: Path, expected: str, label: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise GateError(f"could not read {label} {path}: {error}") from error
    if actual != expected:
        raise GateError(f"frozen {label} SHA-256 drifted")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"could not read JSON {path}: {error}") from error


def finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def analyze(response_path: Path, log_path: Path, expected_ids_path: Path) -> dict[str, Any]:
    require_sha256(
        expected_ids_path,
        EXPECTED_TOKEN_IDS_SHA256,
        "expected-token fixture",
    )
    response = read_json(response_path)
    expected_ids = read_json(expected_ids_path)
    if not isinstance(expected_ids, list) or len(expected_ids) != EXPECTED_TOKENS:
        raise GateError("expected-token fixture must contain exactly 48 token IDs")
    usage = response.get("usage")
    camelid = response.get("camelid")
    choices = response.get("choices")
    if not isinstance(usage, dict) or usage.get("completion_tokens") != EXPECTED_TOKENS:
        raise GateError("response did not return exactly 48 completion tokens")
    generated_ids = camelid.get("generated_token_ids") if isinstance(camelid, dict) else None
    if generated_ids != expected_ids:
        raise GateError("generated token IDs differ from the frozen target-authoritative K1 baseline")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "length"
    ):
        raise GateError("response did not terminate at the exact 48-token length boundary")

    telemetry = camelid.get("hybrid_telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("schema_version") != 2:
        raise GateError("response has no schema-v2 exact-hybrid telemetry")
    geometry = telemetry.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("mapped_readahead_enabled") is not True
        or geometry.get("mapped_readahead_max_inflight_records")
        != MAPPED_READAHEAD_MAX_INFLIGHT_RECORDS
        or geometry.get("mapped_readahead_anonymous_capacity_bytes") != 0
        or not isinstance(geometry.get("record_payload_bytes"), int)
        or isinstance(geometry.get("record_payload_bytes"), bool)
        or geometry["record_payload_bytes"] <= 0
    ):
        raise GateError("telemetry did not admit bounded mapped-cold readahead")
    record_payload_bytes = geometry["record_payload_bytes"]
    rounds = telemetry.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise GateError("response has no completed verifier rounds")

    proposed = 0
    accepted = 0
    committed = 0
    full_k8_rounds = 0
    zero_accept_full_k8 = 0
    assistant_rounds = 0
    round_wall_ms = 0.0
    mapped_readahead_records = 0
    mapped_readahead_bytes = 0
    mapped_readahead_enqueue_ms = 0.0
    previous_union_readahead_records = 0
    previous_union_readahead_bytes = 0
    previous_union_readahead_enqueue_ms = 0.0
    committed_ids: list[int] = []
    previous_prefix: int | None = None
    previous_sequence: int | None = None
    assistant_round_expectations: list[tuple[int, int]] = []
    for index, receipt in enumerate(rounds):
        if not isinstance(receipt, dict) or receipt.get("round_index") != index:
            raise GateError(f"round {index} is missing or reordered")
        if receipt.get("bootstrap") is not False:
            raise GateError("the 50 tok/s lane executed a separate K1 bootstrap")
        k = receipt.get("k")
        requested = receipt.get("requested_k")
        proposed_k = receipt.get("proposed_k")
        verifier_k = receipt.get("verifier_k")
        accepted_k = receipt.get("accepted_drafts")
        useful = receipt.get("useful_accepted_drafts")
        committed_tokens = receipt.get("committed_tokens")
        prefix = receipt.get("prefix_tokens_before")
        sequence = receipt.get("chained_round_sequence")
        remaining = receipt.get("remaining_budget_before")
        wall = receipt.get("receipt_round_wall_ms")
        enqueued_records = receipt.get("mapped_readahead_enqueued_records")
        enqueued_bytes = receipt.get("mapped_readahead_enqueued_bytes")
        enqueue_ms = receipt.get("mapped_readahead_enqueue_ms")
        previous_enqueued_records = receipt.get(
            "mapped_readahead_previous_union_enqueued_records"
        )
        previous_enqueued_bytes = receipt.get(
            "mapped_readahead_previous_union_enqueued_bytes"
        )
        previous_enqueue_ms = receipt.get(
            "mapped_readahead_previous_union_enqueue_ms"
        )
        per_layer = receipt.get("per_layer")
        if (
            not all(isinstance(value, int) and value >= 0 for value in (
                k, requested, proposed_k, verifier_k, accepted_k, useful
            ))
            or receipt.get("success") is not True
            or k != verifier_k
            or verifier_k != proposed_k + 1
            or requested != 8
            or accepted_k > proposed_k
            or useful != accepted_k
            or not isinstance(committed_tokens, list)
            or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or token < 0
                or token > 0xFFFF_FFFF
                for token in committed_tokens
            )
            or len(committed_tokens) != 1 + useful
            or not isinstance(prefix, int)
            or isinstance(prefix, bool)
            or prefix < 0
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
            or remaining != EXPECTED_TOKENS - committed
            or remaining <= 0
            or proposed_k > min(7, remaining - 1)
            or receipt.get("budget_truncated") is not (remaining < 8)
            or not finite_number(wall, positive=True)
        ):
            raise GateError(f"round {index} has inconsistent K, commit, or timing fields")
        if previous_prefix is not None and (
            prefix != previous_prefix or sequence != previous_sequence
        ):
            raise GateError(f"round {index} prefix/sequence continuity drifted")
        if not isinstance(per_layer, list) or len(per_layer) != len(PROFILE):
            raise GateError(f"round {index} has no exact per-layer mapped receipt")
        mapped_bound = 0
        for layer in per_layer:
            layer_mapped = layer.get("mapped_bound") if isinstance(layer, dict) else None
            if (
                not isinstance(layer_mapped, int)
                or isinstance(layer_mapped, bool)
                or layer_mapped < 0
            ):
                raise GateError(f"round {index} has an invalid per-layer mapped bound")
            mapped_bound += layer_mapped
        if (
            not isinstance(enqueued_records, int)
            or isinstance(enqueued_records, bool)
            or enqueued_records < 0
            or enqueued_records > mapped_bound
            or not isinstance(enqueued_bytes, int)
            or isinstance(enqueued_bytes, bool)
            or enqueued_bytes != enqueued_records * record_payload_bytes
            or not finite_number(enqueue_ms)
            or float(enqueue_ms) < 0.0
        ):
            raise GateError(f"round {index} has an invalid mapped-cold readahead receipt")
        if (
            not isinstance(previous_enqueued_records, int)
            or isinstance(previous_enqueued_records, bool)
            or previous_enqueued_records < 0
            or not isinstance(previous_enqueued_bytes, int)
            or isinstance(previous_enqueued_bytes, bool)
            or previous_enqueued_bytes != previous_enqueued_records * record_payload_bytes
            or not finite_number(previous_enqueue_ms)
            or float(previous_enqueue_ms) < 0.0
        ):
            raise GateError(
                f"round {index} has an invalid previous-union readahead receipt"
            )
        for field in ("selected_dropped", "missing_failclose", "slot_capacity_overflow", "overflow_experts"):
            if receipt.get(field) != 0:
                raise GateError(f"round {index} has nonzero {field}")
        if verifier_k == 8 and proposed_k == 7 and receipt.get("budget_truncated") is False:
            full_k8_rounds += 1
            if accepted_k == 0:
                zero_accept_full_k8 += 1
        if proposed_k:
            assistant_rounds += 1
            assistant_round_expectations.append((min(7, remaining - 1), proposed_k))
        proposed += proposed_k
        accepted += accepted_k
        committed += len(committed_tokens)
        round_wall_ms += float(wall)
        mapped_readahead_records += enqueued_records
        mapped_readahead_bytes += enqueued_bytes
        mapped_readahead_enqueue_ms += float(enqueue_ms)
        previous_union_readahead_records += previous_enqueued_records
        previous_union_readahead_bytes += previous_enqueued_bytes
        previous_union_readahead_enqueue_ms += float(previous_enqueue_ms)
        committed_ids.extend(committed_tokens)
        previous_prefix = prefix + len(committed_tokens)
        previous_sequence = sequence + 1

    metrics = telemetry.get("metrics")
    if not isinstance(metrics, dict):
        raise GateError("telemetry has no structured metrics")
    forwarded = metrics.get("forwarded_decode_tokens")
    terminal = metrics.get("terminal_unforwarded_tokens")
    reported_wall = metrics.get("receipt_round_wall_ms")
    if (
        not isinstance(forwarded, int)
        or not isinstance(terminal, int)
        or forwarded + terminal != EXPECTED_TOKENS
        or committed != forwarded
        or metrics.get("response_completion_tokens") != EXPECTED_TOKENS
        or metrics.get("proposed_drafts") != proposed
        or metrics.get("accepted_drafts") != accepted
        or not finite_number(reported_wall, positive=True)
        or not math.isclose(float(reported_wall), round_wall_ms, rel_tol=1e-9, abs_tol=1e-6)
        or metrics.get("outer_lookahead_nonzero_count") != 0
        or committed_ids != generated_ids[:forwarded]
        or generated_ids[forwarded:] != expected_ids[forwarded:forwarded + terminal]
    ):
        raise GateError("structured metrics do not reconcile the completed rounds")
    if full_k8_rounds == 0 or zero_accept_full_k8 != 0:
        raise GateError("the run did not contain a productive full K8 verifier round")
    acceptance = accepted / proposed if proposed else 0.0
    effective_tps = EXPECTED_TOKENS / (round_wall_ms / 1000.0)

    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise GateError(f"could not read server log {log_path}: {error}") from error
    profile_marker = (
        "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER=" + PROFILE_CSV
    )
    if log.count(profile_marker) != 1:
        raise GateError("server did not admit the exact budget-neutral per-layer hot profile")
    promotion_marker = (
        "hybrid decode promotion policy: CAMELID_GEMMA4_GHOST_METAL_DECODE_PROMOTION "
        "effective=0 terminal_decode_promotion=off final_prefill_hot_handoff=on"
    )
    if log.count(promotion_marker) != 1:
        raise GateError("server did not disable terminal decode promotion")
    if log.splitlines().count(MAPPED_READAHEAD_MARKER) != 1:
        raise GateError("server did not admit exact selected-cold mapped readahead")
    if log.splitlines().count(PREVIOUS_UNION_READAHEAD_MARKER) != 1:
        raise GateError("server did not admit pre-assistant previous-union readahead")
    if MAPPED_READAHEAD_REFUSAL_MARKER in log:
        raise GateError("the kernel refused at least one mapped-cold page advisory")
    bootstrap_marker = (
        "[gemma4-mtp bootstrap] prefill_seed_attempted=1 used=1 fallback=none"
    )
    if log.count(bootstrap_marker) != 1:
        raise GateError("generation did not consume the exact final-prefill seed")
    partition_marker = (
        "[gemma4 exact partition] CAMELID_GEMMA4_DENSE_K8_GENERIC=1 "
        "static_k8_dense=off runtime_k_dense=on"
    )
    if log.count(partition_marker) != 1:
        raise GateError("generation did not use the K1/K8 partition-parity dense path")
    if log.splitlines().count(PACKED_K8_GATEUP_MARKER) != 1:
        raise GateError("generation did not admit the exact packed K8 GateUp path")
    if log.splitlines().count(COMPACT_K8_HEAD_MARKER) != 1:
        raise GateError("generation did not admit the compact exact K8 tied head")
    if previous_union_readahead_records == 0:
        raise GateError("generation did not dispatch any previous-union page advice")
    full_q4_markers = FULL_Q4_MARKER_PATTERN.findall(log)
    if len(full_q4_markers) != 1:
        raise GateError("server did not admit one exact full-Q4 assistant")
    full_q4_hash, packed_bytes, bf16_bytes, quantize_us = full_q4_markers[0]
    if (
        full_q4_hash != FULL_Q4_SOURCE_SHA256
        or int(packed_bytes) != FULL_Q4_MATRIX_BYTES
        or int(bf16_bytes) != BF16_MATRIX_BYTES
        or int(quantize_us) <= 0
    ):
        raise GateError("full-Q4 assistant admission receipt is inconsistent")
    if log.splitlines().count(FULL_Q4_RESIDENCY_MARKER) != 1:
        raise GateError("full-Q4 assistant did not release its BF16 source mapping")
    chain_receipts = DEVICE_CHAIN_PATTERN.findall(log)
    if len(chain_receipts) != assistant_rounds:
        raise GateError(
            f"device-chain receipts={len(chain_receipts)} do not match assistant rounds={assistant_rounds}"
        )
    for receipt, expected_round in zip(chain_receipts, assistant_round_expectations):
        requested, returned, *_ = receipt
        expected_requested, expected_returned = expected_round
        if (int(requested), int(returned)) != (expected_requested, expected_returned):
            raise GateError("device-chain drafts do not reconcile their structured round")
    if any(
        linear_format != "q4_0_all" or int(matrix_bytes) != FULL_Q4_MATRIX_BYTES
        for _, _, linear_format, matrix_bytes, *_ in chain_receipts
    ):
        raise GateError("a device-chain receipt did not execute the full-Q4 assistant")

    gates = {
        "exact_k1_token_identity": True,
        "no_k1_bootstrap": True,
        "device_resident_assistant_chain": True,
        "decode_wall_le_960_ms": round_wall_ms <= MAX_DECODE_WALL_MS,
        "effective_decode_tps_ge_50": effective_tps >= 50.0,
        "acceptance_ge_85_percent": acceptance >= 0.85,
        "no_zero_accept_full_k8": zero_accept_full_k8 == 0,
        "terminal_decode_promotion_off": True,
        "mapped_cold_readahead_active": True,
        "previous_union_readahead_active": True,
        "exact_packed_k8_gateup_active": True,
        "compact_exact_k8_head_active": True,
        "full_q4_assistant_active": True,
        "full_q4_bf16_source_released": True,
    }
    return {
        "schema_version": 1,
        "pass": all(gates.values()),
        "gate": "gemma4_48_tokens_le_960ms",
        "tokens": EXPECTED_TOKENS,
        "receipt_round_wall_ms": round_wall_ms,
        "effective_decode_tokens_per_second": effective_tps,
        "reported_forwarded_decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
        "proposed_drafts": proposed,
        "accepted_drafts": accepted,
        "acceptance_probability": acceptance,
        "rounds": len(rounds),
        "full_k8_rounds": full_k8_rounds,
        "device_chain_receipts": len(chain_receipts),
        "full_q4_matrix_bytes_per_draft": FULL_Q4_MATRIX_BYTES,
        "mapped_readahead_enqueued_records": mapped_readahead_records,
        "mapped_readahead_enqueued_bytes": mapped_readahead_bytes,
        "mapped_readahead_enqueue_ms": mapped_readahead_enqueue_ms,
        "previous_union_readahead_enqueued_records": previous_union_readahead_records,
        "previous_union_readahead_enqueued_bytes": previous_union_readahead_bytes,
        "previous_union_readahead_enqueue_ms": previous_union_readahead_enqueue_ms,
        "performance_gates": gates,
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--expected-token-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_sha256(args.request, REQUEST_SHA256, "request fixture")
        verdict = analyze(args.response, args.server_log, args.expected_token_ids)
    except GateError as error:
        verdict = {"schema_version": 1, "pass": False, "error": str(error)}
    atomic_write(args.output, verdict)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
