#!/usr/bin/env python3
"""Validate and summarize one observation-only H2 predictor-pressure receipt."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


LAYERS = 30
MASK128 = (1 << 128) - 1
PROBE_PREFIX = "[gemma4 residual-predict probe] "
TRACE_RE = re.compile(
    r"^\[hybrid fill trace\] layer=(\d+) selected=(\d+) slots=(\d+) "
    r"occupied=(\d+) hits=(\d+) loads=(\d+) evicted=(\d+) "
    r"cold_fallback=(\d+)$"
)
MTP_RE = re.compile(
    r"^\[mtp round\] #\d+ wall=[\d.]+ms "
    r"\(assistant=[\d.]+ms, verifier=[\d.]+ms\) accepted=\d+/8$"
)
DEVICE_RE = re.compile(
    r"^\[gemma4-mtp device-chain\] requested_drafts=(\d+) "
    r"returned_drafts=(\d+) .+$"
)
LEDGER_RE = re.compile(
    r"^\[metal chained ledger\] start_pos=(\d+) K=(\d+) "
    r"ok=(true|false)\b"
)
EXPECTED_PROBE_FIELDS = {
    "schema",
    "round_seq",
    "start_pos",
    "K",
    "predictor_top_k",
    "predict_us",
    "truth_valid",
    "residual_pairs",
    "predicted_cold_pairs",
    "predicted_residual_hits",
    "actual_sizes",
    "hot_sizes",
    "approx_sizes",
    "actual_masks",
    "hot_masks",
    "approx_masks",
    "approx_ranked_ids",
}
H2_SLOTS = [
    74, 78, 57, 55, 52, 53, 56, 54, 60, 58,
    52, 53, 56, 59, 54, 63, 52, 51, 58, 60,
    53, 59, 57, 55, 59, 70, 73, 69, 72, 64,
]
ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RUN_METADATA_FIELDS = {
    "manifest_format",
    "HOME",
    "PATH",
    "TMPDIR",
    "binary",
    "cache_mib",
}
REQUIRED_PROBE_ENV = {
    "CAMELID_GEMMA4_GHOST_METAL_SPARSE_PREDICT_PROBE": "1",
    "CAMELID_GEMMA4_GHOST_METAL_DEMAND_PROMOTION_TRACE": "1",
}


class ReceiptError(RuntimeError):
    pass


def popcount(value: int) -> int:
    """Return a population count on Mini2's pre-3.10 system Python."""
    if value < 0:
        raise ReceiptError("cannot count bits in a negative mask")
    return bin(value).count("1")


def parse_env(path: Path) -> dict[str, str]:
    """Read a profile, a legacy receipt, or a strict base64-v1 manifest."""
    values: dict[str, str] = {}
    lines = path.read_text().splitlines()
    meaningful = [line for line in lines if line and not line.startswith("#")]
    strict = bool(meaningful and meaningful[0] == "manifest_format=base64-v1")
    for line in meaningful:
        if line == "manifest_format=base64-v1":
            key, value = "manifest_format", "base64-v1"
        else:
            encoded = re.fullmatch(
                rf"({ENV_KEY_RE.pattern})@BASE64=([A-Za-z0-9+/]*={{0,2}})", line
            )
            if encoded:
                key = encoded.group(1)
                try:
                    value = base64.b64decode(
                        encoded.group(2), validate=True
                    ).decode("utf-8")
                except (binascii.Error, UnicodeDecodeError) as error:
                    raise ReceiptError(
                        f"invalid base64 environment manifest value for {key}"
                    ) from error
            else:
                plain = re.fullmatch(rf"({ENV_KEY_RE.pattern})=(.*)", line)
                if strict or plain is None:
                    if strict:
                        raise ReceiptError(
                            f"malformed base64-v1 environment manifest line: {line!r}"
                        )
                    # Legacy multiline KEY@FILE receipts are not round-trippable;
                    # ignore continuation lines while parsing their ordinary keys.
                    continue
                key, value = plain.groups()
        if key in values:
            raise ReceiptError(f"duplicate environment field {key}")
        values[key] = value
    return values


def validate_h2_environment(run_dir: Path) -> None:
    here = Path(__file__).resolve().parent
    expected = parse_env(here / "env" / "H2-proportional")
    observed = parse_env(run_dir / "env.txt")
    allowed = set(expected) | set(REQUIRED_PROBE_ENV) | RUN_METADATA_FIELDS
    extras = sorted(set(observed) - allowed)
    if extras:
        raise ReceiptError(
            f"probe environment has unapproved extra fields: {extras}"
        )
    for key, value in expected.items():
        if key == "CAMELID_GEMMA4_MTP_ASSISTANT_PATH":
            if not observed.get(key, "").endswith("model.safetensors"):
                raise ReceiptError("H2 assistant path is absent or unexpected")
        elif observed.get(key) != value:
            raise ReceiptError(
                f"H2 environment drifted at {key}: "
                f"expected {value!r}, observed {observed.get(key)!r}"
            )
    for key, value in REQUIRED_PROBE_ENV.items():
        if observed.get(key) != value:
            raise ReceiptError(f"probe environment lacks exact {key}={value}")
    if observed.get("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT48_EXPERIMENT") == "1":
        raise ReceiptError("receipt is Hot48, not proportional H2")


def parse_int(fields: dict[str, str], key: str) -> int:
    raw = fields[key]
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ReceiptError(f"probe {key} is not a canonical decimal")
    return int(raw)


def parse_decimal_list(raw: str, field: str) -> list[int]:
    parts = raw.split(",")
    if len(parts) != LAYERS or any(re.fullmatch(r"[0-9]+", x) is None for x in parts):
        raise ReceiptError(f"probe {field} is not exactly 30 canonical decimals")
    return [int(x) for x in parts]


def parse_mask_list(raw: str, field: str) -> list[int]:
    parts = raw.split(",")
    if len(parts) != LAYERS or any(
        re.fullmatch(r"[0-9a-f]{32}", x) is None for x in parts
    ):
        raise ReceiptError(f"probe {field} is not exactly 30 canonical masks")
    return [int(x, 16) for x in parts]


def parse_ranked_ids(raw: str) -> list[list[int]]:
    layers = raw.split("/")
    if len(layers) != LAYERS:
        raise ReceiptError("probe approx_ranked_ids does not have exactly 30 layers")
    parsed: list[list[int]] = []
    for layer, encoded in enumerate(layers):
        parts = encoded.split(",")
        if not encoded or any(re.fullmatch(r"[0-9]+", item) is None for item in parts):
            raise ReceiptError(f"layer {layer} ranked IDs are malformed")
        ids = [int(item) for item in parts]
        if len(set(ids)) != len(ids) or any(expert >= 128 for expert in ids):
            raise ReceiptError(f"layer {layer} ranked IDs are non-canonical")
        parsed.append(ids)
    return parsed


def parse_probe(line: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for encoded in line[len(PROBE_PREFIX) :].split(" "):
        key, separator, value = encoded.partition("=")
        if not separator or not key or not value or key in fields:
            raise ReceiptError("malformed or duplicate probe field")
        fields[key] = value
    if set(fields) != EXPECTED_PROBE_FIELDS:
        raise ReceiptError(
            "probe field set drifted: "
            f"missing={sorted(EXPECTED_PROBE_FIELDS - set(fields))}, "
            f"extra={sorted(set(fields) - EXPECTED_PROBE_FIELDS)}"
        )
    values = {
        key: parse_int(fields, key)
        for key in (
            "schema", "round_seq", "start_pos", "K", "predictor_top_k",
            "predict_us", "truth_valid", "residual_pairs",
            "predicted_cold_pairs", "predicted_residual_hits",
        )
    }
    if (
        values["schema"] != 1
        or values["K"] != 8
        or values["predictor_top_k"] != 8
        or values["predict_us"] <= 0
        or values["truth_valid"] != 1
    ):
        raise ReceiptError("probe schema, K8/top8 admission, timing, or truth failed")

    sizes = {
        name: parse_decimal_list(fields[name], name)
        for name in ("actual_sizes", "hot_sizes", "approx_sizes")
    }
    masks = {
        name: parse_mask_list(fields[name], name)
        for name in ("actual_masks", "hot_masks", "approx_masks")
    }
    approx_ranked_ids = parse_ranked_ids(fields["approx_ranked_ids"])
    for stem in ("actual", "hot", "approx"):
        if any(
            size != popcount(mask)
            for size, mask in zip(sizes[f"{stem}_sizes"], masks[f"{stem}_masks"])
        ):
            raise ReceiptError(f"{stem} size/mask popcount drifted")
    if any(not 8 <= size <= 64 for size in sizes["actual_sizes"]):
        raise ReceiptError("actual union lies outside exact K8 bounds")
    if any(not 8 <= size <= 64 for size in sizes["approx_sizes"]):
        raise ReceiptError("approximate union lies outside top8/K8 bounds")
    for layer, (ids, size, mask) in enumerate(
        zip(approx_ranked_ids, sizes["approx_sizes"], masks["approx_masks"])
    ):
        ranked_mask = sum(1 << expert for expert in ids)
        if len(ids) != size or ranked_mask != mask:
            raise ReceiptError(f"layer {layer} ranked IDs disagree with approximate mask")

    residual_masks = [
        actual & (~hot & MASK128)
        for actual, hot in zip(masks["actual_masks"], masks["hot_masks"])
    ]
    predicted_cold_masks = [
        approx & (~hot & MASK128)
        for approx, hot in zip(masks["approx_masks"], masks["hot_masks"])
    ]
    predicted_hit_masks = [
        residual & approx
        for residual, approx in zip(residual_masks, masks["approx_masks"])
    ]
    derived = {
        "residual_pairs": sum(popcount(mask) for mask in residual_masks),
        "predicted_cold_pairs": sum(popcount(mask) for mask in predicted_cold_masks),
        "predicted_residual_hits": sum(popcount(mask) for mask in predicted_hit_masks),
    }
    if any(values[key] != value for key, value in derived.items()):
        raise ReceiptError("probe aggregate counters disagree with exact masks")
    return {
        **values,
        **sizes,
        **masks,
        "approx_ranked_ids": approx_ranked_ids,
        "residual_masks": residual_masks,
        "predicted_cold_masks": predicted_cold_masks,
        "predicted_hit_masks": predicted_hit_masks,
    }


def parse_trace(match: re.Match[str]) -> dict[str, int]:
    names = (
        "layer", "selected", "slots", "occupied", "hits", "loads",
        "evicted", "cold_fallback",
    )
    return dict(zip(names, (int(value) for value in match.groups())))


def validate_trace(probe: dict[str, Any], traces: list[dict[str, int]]) -> None:
    if len(traces) != LAYERS or sorted(item["layer"] for item in traces) != list(range(LAYERS)):
        raise ReceiptError("probe is not preceded by one fill trace for every layer")
    by_layer = {item["layer"]: item for item in traces}
    for layer in range(LAYERS):
        item = by_layer[layer]
        actual = probe["actual_masks"][layer]
        hot = probe["hot_masks"][layer]
        residual = popcount(probe["residual_masks"][layer])
        if item["slots"] != H2_SLOTS[layer]:
            raise ReceiptError(f"layer {layer} is not exact H2 slot geometry")
        # `plan_hot_overrides` removes planned evictions before the trace
        # samples `occupied`, while the probe froze its hot mask before planning.
        # Reconstruct start occupancy rather than comparing unlike timepoints.
        if probe["hot_sizes"][layer] != item["occupied"] + item["evicted"]:
            raise ReceiptError(
                f"layer {layer} hot mask disagrees with post-plan occupancy plus evictions"
            )
        if item["occupied"] > item["slots"] or probe["hot_sizes"][layer] > item["slots"]:
            raise ReceiptError(f"layer {layer} hot residency exceeds slot capacity")
        if item["selected"] != popcount(actual):
            raise ReceiptError(f"layer {layer} selected count disagrees with route truth")
        if item["hits"] != popcount(actual & hot):
            raise ReceiptError(f"layer {layer} hit count disagrees with start residency")
        if residual != item["loads"] + item["cold_fallback"]:
            raise ReceiptError(f"layer {layer} residual demand does not reconcile")
        if item["evicted"] > item["loads"]:
            raise ReceiptError(f"layer {layer} fill pressure counters are impossible")


def exact_response(run_dir: Path) -> bool:
    expected_path = (
        Path(__file__).resolve().parent.parent
        / "hybrid-hot48-runner"
        / "expected-48-token-ids.json"
    )
    expected = json.loads(expected_path.read_text())
    response = json.loads((run_dir / "response.json").read_text())
    ids = response.get("camelid", {}).get("generated_token_ids")
    return response.get("usage", {}).get("completion_tokens") == 48 and ids == expected


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ReceiptError("cannot take percentile of empty series")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def cap_round_robin(probe: dict[str, Any], cap: int) -> dict[str, int]:
    """Select ranked, currently-cold proposals one/layer/pass up to a global cap."""
    cold_ranked = [
        [
            expert
            for expert in probe["approx_ranked_ids"][layer]
            if not (probe["hot_masks"][layer] & (1 << expert))
        ]
        for layer in range(LAYERS)
    ]
    cursors = [0] * LAYERS
    selected_masks = [0] * LAYERS
    selected = 0
    while selected < cap:
        progressed = False
        for layer in range(LAYERS):
            if selected == cap:
                break
            cursor = cursors[layer]
            if cursor >= len(cold_ranked[layer]):
                continue
            expert = cold_ranked[layer][cursor]
            cursors[layer] += 1
            selected_masks[layer] |= 1 << expert
            selected += 1
            progressed = True
        if not progressed:
            break
    hits = sum(
        popcount(selected_masks[layer] & probe["residual_masks"][layer])
        for layer in range(LAYERS)
    )
    return {"selected": selected, "hits": hits}


def parse_observations(
    log: str,
) -> tuple[
    list[tuple[dict[str, Any], list[dict[str, int]]]],
    int,
    int,
    int,
    int,
    int,
]:
    """Bind probe receipts only to full K8 MTP verifier calls.

    The MTP summary keeps an `/8` offer denominator for its K5/K3 generation
    tail, so it cannot define probe cardinality. A device-chain request of seven
    drafts followed by an exact K8 chained ledger is the full verifier binding.
    """
    pending_traces: list[dict[str, int]] = []
    rounds: list[tuple[dict[str, Any], list[dict[str, int]]]] = []
    mtp_rounds = 0
    full_k8_verifiers = 0
    tail_verifiers = 0
    refused_trace_lines = 0
    pending_device_drafts: int | None = None
    probes_before_device = 0

    for line in log.splitlines():
        trace_match = TRACE_RE.fullmatch(line)
        if trace_match:
            pending_traces.append(parse_trace(trace_match))
            continue
        if line.startswith("[hybrid fill trace]") and (
            "REFUSED:" in line or "SKIPPED:" in line
        ):
            refused_trace_lines += 1

        device_match = DEVICE_RE.fullmatch(line)
        if device_match:
            if pending_device_drafts is not None:
                raise ReceiptError("device-chain verifier binding overlapped")
            requested, returned = (int(value) for value in device_match.groups())
            if requested != returned or not 1 <= requested <= 7:
                raise ReceiptError("device-chain draft receipt is not exact")
            pending_device_drafts = requested
            probes_before_device = len(rounds)
            continue

        if line.startswith(PROBE_PREFIX):
            if pending_device_drafts != 7:
                raise ReceiptError("residual probe is not inside a seven-draft verifier")
            probe = parse_probe(line)
            traces = pending_traces[-LAYERS:]
            validate_trace(probe, traces)
            rounds.append((probe, traces))
            pending_traces = []
            continue

        ledger_match = LEDGER_RE.match(line)
        if ledger_match:
            start_pos, k, ok_raw = ledger_match.groups()
            if pending_device_drafts is not None:
                expected_k = pending_device_drafts + 1
                if ok_raw != "true" or int(k) != expected_k:
                    raise ReceiptError("device-chain draft count and verifier K disagree")
                if expected_k == 8:
                    full_k8_verifiers += 1
                    if len(rounds) != probes_before_device + 1:
                        raise ReceiptError("full K8 verifier lacks exactly one residual probe")
                    probe = rounds[-1][0]
                    if probe["start_pos"] != int(start_pos) or probe["K"] != int(k):
                        raise ReceiptError("residual probe does not bind its K8 verifier ledger")
                else:
                    tail_verifiers += 1
                    if len(rounds) != probes_before_device:
                        raise ReceiptError("short tail verifier unexpectedly emitted a probe")
                pending_device_drafts = None
            pending_traces = []
            continue

        if MTP_RE.fullmatch(line):
            mtp_rounds += 1

    if pending_device_drafts is not None:
        raise ReceiptError("unterminated device-chain verifier binding")
    if len(rounds) != full_k8_verifiers:
        raise ReceiptError(
            f"probe/full-K8 mismatch: probe={len(rounds)} full_k8={full_k8_verifiers}"
        )
    if mtp_rounds != full_k8_verifiers + tail_verifiers:
        raise ReceiptError(
            "MTP summaries do not reconcile with full-K8 and short-tail verifiers"
        )
    fill_failures = len(re.findall(r"hot-slot refill failed", log))
    return (
        rounds,
        mtp_rounds,
        full_k8_verifiers,
        tail_verifiers,
        refused_trace_lines,
        fill_failures,
    )


def analyze(run_dir: Path) -> dict[str, Any]:
    validate_h2_environment(run_dir)
    log = (run_dir / "server.log").read_text(errors="replace")
    if not exact_response(run_dir):
        raise ReceiptError("response does not exactly reproduce the frozen 48-token fixture")

    (
        rounds,
        mtp_rounds,
        full_k8_verifiers,
        tail_verifiers,
        refused_trace_lines,
        fill_failures,
    ) = parse_observations(log)
    if not rounds:
        raise ReceiptError("receipt has no full-K8 residual-prediction observations")
    keys = {(probe["round_seq"], probe["start_pos"], probe["K"]) for probe, _ in rounds}
    if len(keys) != len(rounds):
        raise ReceiptError("duplicate probe round binding")
    if refused_trace_lines or fill_failures:
        raise ReceiptError(
            f"fill trace contains refusals/failures: refused={refused_trace_lines} "
            f"failures={fill_failures}"
        )

    residual_total = sum(probe["residual_pairs"] for probe, _ in rounds)
    predicted_cold_total = sum(probe["predicted_cold_pairs"] for probe, _ in rounds)
    predicted_hits_total = sum(probe["predicted_residual_hits"] for probe, _ in rounds)
    recall = ratio(predicted_hits_total, residual_total)
    precision = ratio(predicted_hits_total, predicted_cold_total)
    predicted_cold_per_round = [probe["predicted_cold_pairs"] for probe, _ in rounds]
    capped: dict[int, dict[str, Any]] = {}
    for cap in (64, 96):
        cap_rounds = [cap_round_robin(probe, cap) for probe, _ in rounds]
        selected_total = sum(item["selected"] for item in cap_rounds)
        hits_total = sum(item["hits"] for item in cap_rounds)
        capped[cap] = {
            "cap_records_per_round": cap,
            "selection": "confidence-ranked-within-layer/layer-round-robin-global",
            "selected_records": selected_total,
            "predicted_residual_hits": hits_total,
            "residual_miss_recall": ratio(hits_total, residual_total),
            "predicted_cold_precision": ratio(hits_total, selected_total),
            "records_per_round_mean": statistics.fmean(
                item["selected"] for item in cap_rounds
            ),
            "records_per_round_maximum": max(item["selected"] for item in cap_rounds),
        }
    loads_per_round = [sum(item["loads"] for item in traces) for _, traces in rounds]
    evictions_per_round = [sum(item["evicted"] for item in traces) for _, traces in rounds]
    fallback_total = sum(
        item["cold_fallback"] for _, traces in rounds for item in traces
    )

    per_layer: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        layer_residual = sum(
            popcount(probe["residual_masks"][layer]) for probe, _ in rounds
        )
        layer_predicted_cold = sum(
            popcount(probe["predicted_cold_masks"][layer]) for probe, _ in rounds
        )
        layer_hits = sum(
            popcount(probe["predicted_hit_masks"][layer]) for probe, _ in rounds
        )
        traces = [
            {item["layer"]: item for item in items}[layer]
            for _, items in rounds
        ]
        per_layer.append(
            {
                "layer": layer,
                "slots": H2_SLOTS[layer],
                "mean_selected": statistics.fmean(item["selected"] for item in traces),
                "mean_loads": statistics.fmean(item["loads"] for item in traces),
                "maximum_loads": max(item["loads"] for item in traces),
                "mean_evictions": statistics.fmean(item["evicted"] for item in traces),
                "maximum_selected_to_slots": max(
                    item["selected"] / item["slots"] for item in traces
                ),
                "residual_pairs": layer_residual,
                "predicted_cold_pairs": layer_predicted_cold,
                "predicted_residual_hits": layer_hits,
                "residual_recall": ratio(layer_hits, layer_residual),
                "predicted_cold_precision": ratio(layer_hits, layer_predicted_cold),
            }
        )

    max_predicted_cold = max(predicted_cold_per_round)
    cap96 = capped[96]
    if (
        cap96["residual_miss_recall"] >= 0.40
        and cap96["predicted_cold_precision"] >= 0.50
    ):
        decision = "GO"
        reason = "prototype capped asynchronous staging without directory mutation"
    elif cap96["residual_miss_recall"] >= 0.30:
        decision = "CONDITIONAL"
        reason = "capped prediction is useful, but needs better precision before staging"
    else:
        decision = "NO-GO"
        reason = "residual-miss recall is below the 30% implementation floor"

    return {
        "schema_version": 1,
        "run": run_dir.name,
        "integrity": {
            "exact_match_expected": True,
            "exact_h2_environment": True,
            "truth_valid_every_round": True,
            "one_trace_per_layer_per_round": True,
            "trace_mask_reconciliation": True,
            "throughput_contaminated_by_probe": True,
            "throughput_promotion_allowed": False,
        },
        "rounds": len(rounds),
        "mtp_rounds": mtp_rounds,
        "full_k8_verifier_rounds": full_k8_verifiers,
        "short_tail_verifier_rounds": tail_verifiers,
        "predict_microseconds": {
            "mean": statistics.fmean(probe["predict_us"] for probe, _ in rounds),
            "maximum": max(probe["predict_us"] for probe, _ in rounds),
        },
        "prediction": {
            "residual_pairs": residual_total,
            "predicted_cold_pairs": predicted_cold_total,
            "predicted_residual_hits": predicted_hits_total,
            "residual_miss_recall": recall,
            "predicted_cold_precision": precision,
            "predicted_cold_records_per_round_mean": statistics.fmean(
                predicted_cold_per_round
            ),
            "predicted_cold_records_per_round_p95": percentile(
                predicted_cold_per_round, 0.95
            ),
            "predicted_cold_records_per_round_maximum": max_predicted_cold,
            "global_caps": {
                "64": capped[64],
                "96": capped[96],
            },
        },
        "demand_pressure": {
            "loads_per_round_mean": statistics.fmean(loads_per_round),
            "loads_per_round_p95": percentile(loads_per_round, 0.95),
            "evictions_per_round_mean": statistics.fmean(evictions_per_round),
            "evictions_per_round_p95": percentile(evictions_per_round, 0.95),
            "cold_fallback_total": fallback_total,
        },
        "per_layer": per_layer,
        "decision": {
            "verdict": decision,
            "reason": reason,
            "thresholds": {
                "hard_no_go_below_residual_recall": 0.30,
                "go_residual_recall": 0.40,
                "go_predicted_cold_precision": 0.50,
                "decision_metric": "global_cap96",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.run_dir)
    except (OSError, ValueError, json.JSONDecodeError, ReceiptError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
