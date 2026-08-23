#!/usr/bin/env python3
"""Fail-closed analyzer for the exact Gemma 4 48-token / 960 ms gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


EXPECTED_TOKENS = 48
MAX_DECODE_WALL_MS = 960.0
PROFILE = (
    39, 40, 33, 30, 30, 31, 31, 30, 34, 30,
    26, 28, 30, 31, 28, 37, 31, 30, 31, 32,
    31, 32, 30, 31, 32, 35, 32, 34, 34, 37,
)
PROFILE_CSV = ",".join(str(value) for value in PROFILE)
DEVICE_CHAIN_PATTERN = re.compile(
    r"^\[gemma4-mtp device-chain\] requested_drafts=(\d+) returned_drafts=(\d+) "
    r"command_buffers=1 commits=1 waits=1 cpu_embedding_callbacks=0 "
    r"encode_us=(\d+) wait_us=(\d+) gpu_us=(\d+) kernel_us=(\d+) wall_us=(\d+)$",
    re.MULTILINE,
)


class GateError(RuntimeError):
    pass


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
    response = read_json(response_path)
    expected_ids = read_json(expected_ids_path)
    if not isinstance(expected_ids, list) or len(expected_ids) != EXPECTED_TOKENS:
        raise GateError("expected-token fixture must contain exactly 48 token IDs")
    usage = response.get("usage")
    camelid = response.get("camelid")
    choices = response.get("choices")
    if not isinstance(usage, dict) or usage.get("completion_tokens") != EXPECTED_TOKENS:
        raise GateError("response did not return exactly 48 completion tokens")
    if not isinstance(camelid, dict) or camelid.get("generated_token_ids") != expected_ids:
        raise GateError("generated token IDs differ from the frozen target-authoritative K1 baseline")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "length"
    ):
        raise GateError("response did not terminate at the exact 48-token length boundary")

    telemetry = camelid.get("hybrid_telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("schema_version") != 1:
        raise GateError("response has no schema-v1 exact-hybrid telemetry")
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
        wall = receipt.get("receipt_round_wall_ms")
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
            or len(committed_tokens) != 1 + useful
            or not finite_number(wall, positive=True)
        ):
            raise GateError(f"round {index} has inconsistent K, commit, or timing fields")
        for field in ("selected_dropped", "missing_failclose", "slot_capacity_overflow", "overflow_experts"):
            if receipt.get(field) != 0:
                raise GateError(f"round {index} has nonzero {field}")
        if verifier_k == 8 and proposed_k == 7 and receipt.get("budget_truncated") is False:
            full_k8_rounds += 1
            if accepted_k == 0:
                zero_accept_full_k8 += 1
        if proposed_k:
            assistant_rounds += 1
        proposed += proposed_k
        accepted += accepted_k
        committed += len(committed_tokens)
        round_wall_ms += float(wall)

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
    chain_receipts = DEVICE_CHAIN_PATTERN.findall(log)
    if len(chain_receipts) != assistant_rounds:
        raise GateError(
            f"device-chain receipts={len(chain_receipts)} do not match assistant rounds={assistant_rounds}"
        )
    if any(int(requested) < int(returned) for requested, returned, *_ in chain_receipts):
        raise GateError("a device-chain receipt returned more drafts than it encoded")

    gates = {
        "exact_k1_token_identity": True,
        "no_k1_bootstrap": True,
        "device_resident_assistant_chain": True,
        "decode_wall_le_960_ms": round_wall_ms <= MAX_DECODE_WALL_MS,
        "effective_decode_tps_ge_50": effective_tps >= 50.0,
        "acceptance_ge_85_percent": acceptance >= 0.85,
        "no_zero_accept_full_k8": zero_accept_full_k8 == 0,
        "terminal_decode_promotion_off": True,
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
    parser.add_argument("--expected-token-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        verdict = analyze(args.response, args.server_log, args.expected_token_ids)
    except GateError as error:
        verdict = {"schema_version": 1, "pass": False, "error": str(error)}
    atomic_write(args.output, verdict)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
