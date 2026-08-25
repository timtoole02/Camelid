#!/usr/bin/env python3
"""Validate exact decode-route traces and solve the fixed 1,408-slot profile."""

import hashlib
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import analyze_live_sequential_cap16 as h69


PREFIX = "[gemma4 exact-residency trace] "
LAYERS = 30
EXPERTS = 128
EXPECTED_SCHEDULE = ((104, 14), (118, 13), (131, 14), (145, 7))
EXPECTED_CAPACITIES = (
    60, 64, 45, 43, 40, 41, 44, 42, 47, 46,
    40, 41, 44, 46, 42, 50, 40, 39, 46, 47,
    41, 46, 45, 43, 46, 56, 59, 56, 58, 51,
)
FIELDS = {
    "schema", "requested", "admitted", "truth_valid", "scope", "profile",
    "round_seq", "start_pos", "K", "layers", "experts", "stage_cap",
    "capacity_total", "resident_total", "exact_unique_records", "resident_hits", "cold_records",
    "total_wave_load_ms", "capacities", "route_sizes", "resident_sizes", "cold_sizes",
    "wave_load_ms_per_layer", "route_masks", "resident_masks", "cold_masks", "timing",
    "output_mutation", "io_mutation", "expert_read_mutation", "slot_policy_mutation",
    "table_mutation", "route_mutation", "routing_authority", "throughput_eligible",
}
MTP_ROUND_RE = re.compile(
    r"\[mtp round\] #(\d+) wall=((?:0|[1-9][0-9]*)\.[0-9]{2})ms "
    r"\(assistant=(?:0|[1-9][0-9]*)\.[0-9]{2}ms, "
    r"verifier=(?:0|[1-9][0-9]*)\.[0-9]{2}ms\) "
    r"accepted=(\d+)/(\d+)"
)


class TraceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise TraceError(message)


def parse_fields(payload):
    require(payload and payload == payload.strip() and "  " not in payload,
            "trace has noncanonical field spacing")
    fields = {}
    for item in payload.strip().split(" "):
        require(item and "=" in item, "trace contains an empty or malformed field")
        key, value = item.split("=", 1)
        require(key and value and key not in fields, f"duplicate or empty trace field {key!r}")
        fields[key] = value
    require(set(fields) == FIELDS,
            f"trace field set drifted: missing={sorted(FIELDS - set(fields))}, "
            f"extra={sorted(set(fields) - FIELDS)}")
    return fields


def parse_uint(value, label):
    require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
            f"{label} is not a canonical unsigned integer")
    return int(value)


def parse_ms_us(value, label):
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.([0-9]{3})", value)
    require(match is not None, f"{label} is not canonical M.mmm")
    return int(match.group(1)) * 1_000 + int(match.group(2))


def parse_uint_list(value, label):
    items = value.split(",")
    require(len(items) == LAYERS, f"{label} does not contain exactly {LAYERS} values")
    return tuple(parse_uint(item, f"{label}[{index}]") for index, item in enumerate(items))


def parse_mask_layers(value, label):
    parts = value.split("/")
    require(len(parts) == LAYERS, f"{label} does not contain exactly {LAYERS} layers")
    decoded = []
    rendered = []
    for layer, part in enumerate(parts):
        match = re.fullmatch(rf"L{layer}:([0-9a-f]{{32}})", part)
        require(match is not None, f"{label} layer {layer} is noncanonical")
        mask_hex = match.group(1)
        mask = int(mask_hex, 16)
        experts = frozenset(expert for expert in range(EXPERTS) if mask & (1 << expert))
        decoded.append(experts)
        rendered.append(mask_hex)
    return tuple(decoded), tuple(rendered)


def parse_trace_line(line):
    require(line.startswith(PREFIX), "not an exact-residency trace line")
    fields = parse_fields(line[len(PREFIX):])
    exact_strings = {
        "schema": "1",
        "requested": "1",
        "admitted": "1",
        "truth_valid": "1",
        "scope": "successful-target-decode-round",
        "profile": "mini2-h49-1408",
        "layers": "30",
        "experts": "128",
        "stage_cap": "8",
        "capacity_total": "1408",
        "timing": "post-tied-head-output-fixed",
        "output_mutation": "0",
        "io_mutation": "0",
        "expert_read_mutation": "0",
        "slot_policy_mutation": "0",
        "table_mutation": "0",
        "route_mutation": "0",
        "routing_authority": "exact-router",
        "throughput_eligible": "0",
    }
    for key, expected in exact_strings.items():
        require(fields[key] == expected, f"trace requires {key}={expected}")

    capacities = parse_uint_list(fields["capacities"], "capacities")
    require(capacities == EXPECTED_CAPACITIES, "trace capacity vector is not exact H49 1,408")
    route_sizes = parse_uint_list(fields["route_sizes"], "route_sizes")
    resident_sizes = parse_uint_list(fields["resident_sizes"], "resident_sizes")
    cold_sizes = parse_uint_list(fields["cold_sizes"], "cold_sizes")
    wave_load_us = tuple(
        parse_ms_us(item, f"wave_load_ms_per_layer[{index}]")
        for index, item in enumerate(fields["wave_load_ms_per_layer"].split(","))
    )
    require(len(wave_load_us) == LAYERS, "wave_load_ms_per_layer is not exactly 30 values")
    routes, route_hex = parse_mask_layers(fields["route_masks"], "route_masks")
    residents, resident_hex = parse_mask_layers(fields["resident_masks"], "resident_masks")
    cold, cold_hex = parse_mask_layers(fields["cold_masks"], "cold_masks")

    k_tokens = parse_uint(fields["K"], "K")
    require(1 <= k_tokens <= 16, "K is outside 1..16")
    for layer in range(LAYERS):
        require(8 <= len(routes[layer]) <= min(EXPERTS, 8 * k_tokens),
                f"L{layer} route width is outside the exact K-dependent domain")
        require(len(routes[layer]) == route_sizes[layer], f"L{layer} route size disagrees")
        require(len(residents[layer]) == resident_sizes[layer] == capacities[layer],
                f"L{layer} resident occupancy is not the complete H49 capacity")
        require(cold[layer] == routes[layer] - residents[layer],
                f"L{layer} cold mask is not route minus residency")
        require(len(cold[layer]) == cold_sizes[layer], f"L{layer} cold size disagrees")

    exact_unique = sum(route_sizes)
    resident_total = sum(resident_sizes)
    require(resident_total == sum(EXPECTED_CAPACITIES) == 1_408,
            "resident occupancy is not the complete 1,408-slot H49 profile")
    require(parse_uint(fields["resident_total"], "resident_total") == resident_total,
            "resident occupancy aggregate disagrees")
    resident_hits = sum(len(routes[layer] & residents[layer]) for layer in range(LAYERS))
    cold_records = sum(cold_sizes)
    require(parse_uint(fields["exact_unique_records"], "exact_unique_records") == exact_unique,
            "exact unique aggregate disagrees")
    require(parse_uint(fields["resident_hits"], "resident_hits") == resident_hits,
            "resident hit aggregate disagrees")
    require(parse_uint(fields["cold_records"], "cold_records") == cold_records,
            "cold aggregate disagrees")
    require(resident_hits + cold_records == exact_unique, "identity aggregates do not close")
    total_wave_us = parse_ms_us(fields["total_wave_load_ms"], "total_wave_load_ms")
    require(sum(wave_load_us) == total_wave_us, "per-layer wave wall does not sum to total")

    return {
        "round_seq": parse_uint(fields["round_seq"], "round_seq"),
        "start_pos": parse_uint(fields["start_pos"], "start_pos"),
        "K": k_tokens,
        "capacities": capacities,
        "routes": routes,
        "residents": residents,
        "cold": cold,
        "route_hex": route_hex,
        "resident_hex": resident_hex,
        "cold_hex": cold_hex,
        "wave_load_us": wave_load_us,
        "total_wave_load_us": total_wave_us,
        "exact_unique_records": exact_unique,
        "resident_hits": resident_hits,
        "cold_records": cold_records,
    }


def decode_wall_us(log):
    round_lines = [line for line in log.splitlines() if line.startswith("[mtp round] ")]
    require(len(round_lines) == 4, "server log does not contain exactly four MTP rounds")
    matches = [MTP_ROUND_RE.fullmatch(line) for line in round_lines]
    require(all(match is not None for match in matches), "MTP round line is noncanonical")
    rounds = [match.groups() for match in matches]
    require(tuple(int(item[0]) for item in rounds) == (0, 1, 2, 3),
            "MTP round indices are not 0,1,2,3")
    require(tuple(int(item[3]) for item in rounds) == (14, 13, 14, 7),
            "MTP widths are not 14,13,14,7")
    require(all(int(item[2]) == int(item[3]) - 1 for item in rounds),
            "every MTP round must accept exactly K-1 drafts")
    emitted = sum(int(item[2]) for item in rounds) + len(rounds)
    require(emitted == 48, f"MTP accounting emitted {emitted}, not 48")
    walls_us = [
        int(raw.split(".", 1)[0]) * 1_000 + int(raw.split(".", 1)[1]) * 10
        for _, raw, _, _ in rounds
    ]
    require(all(wall > 0 for wall in walls_us), "MTP round wall must be positive")
    return sum(walls_us)


def solve_profile(traces, measured_decode_wall_us):
    require(type(measured_decode_wall_us) is int and measured_decode_wall_us > 0,
            "clean control decode wall must be a positive integer number of microseconds")
    require(len(traces) == 4, "expected exactly four exact-residency traces")
    require(tuple((trace["start_pos"], trace["K"]) for trace in traces) == EXPECTED_SCHEDULE,
            "exact-residency traces do not match the 104/14,118/13,131/14,145/7 schedule")
    sequences = tuple(trace["round_seq"] for trace in traces)
    require(sequences == tuple(range(sequences[0], sequences[0] + 4)),
            "trace round_seq values are not consecutive")
    baseline_residents = traces[0]["residents"]
    require(all(trace["residents"] == baseline_residents for trace in traces[1:]),
            "round-start residency changed within the frozen H49 request")

    per_record_cost = [[Fraction(0) for _ in range(LAYERS)] for _ in traces]
    for layer in range(LAYERS):
        observed = [
            Fraction(trace["wave_load_us"][layer], len(trace["cold"][layer]))
            for trace in traces if trace["cold"][layer]
        ]
        fallback = sum(observed, Fraction(0)) / len(observed) if observed else Fraction(0)
        for round_index, trace in enumerate(traces):
            cold_count = len(trace["cold"][layer])
            wall = trace["wave_load_us"][layer]
            require(cold_count != 0 or wall == 0,
                    f"L{layer} has wave load without a cold identity")
            per_record_cost[round_index][layer] = (
                Fraction(wall, cold_count) if cold_count else fallback
            )

    selected = []
    layer_scores = []
    for layer, capacity in enumerate(EXPECTED_CAPACITIES):
        scores = []
        for expert in range(EXPERTS):
            score = sum(
                (per_record_cost[round_index][layer]
                 if expert in trace["routes"][layer] else Fraction(0))
                for round_index, trace in enumerate(traces)
            )
            scores.append((expert, score))
        scores.sort(key=lambda item: (
            -item[1],
            -int(item[0] in baseline_residents[layer]),
            item[0],
        ))
        chosen = frozenset(expert for expert, _ in scores[:capacity])
        require(len(chosen) == capacity, f"L{layer} solver did not fill its exact capacity")
        selected.append(chosen)
        layer_scores.append(scores)
    selected = tuple(selected)

    baseline_wave_us = sum(trace["total_wave_load_us"] for trace in traces)
    require(0 <= baseline_wave_us <= measured_decode_wall_us,
            "observed wave wall is outside the clean control decode wall")
    projected_residual = Fraction(0)
    current_coverage = 0
    selected_coverage = 0
    route_occurrences = 0
    residual_occurrences = 0
    for round_index, trace in enumerate(traces):
        for layer in range(LAYERS):
            routes = trace["routes"][layer]
            route_occurrences += len(routes)
            current_coverage += len(routes & baseline_residents[layer])
            selected_coverage += len(routes & selected[layer])
            uncovered = routes - selected[layer]
            residual_occurrences += len(uncovered)
            projected_residual += per_record_cost[round_index][layer] * len(uncovered)

    projected_saved = Fraction(baseline_wave_us) - projected_residual
    require(Fraction(0) <= projected_saved <= baseline_wave_us,
            "projected residency saving is outside the observed wave ceiling")
    required_saved = max(0, measured_decode_wall_us - 960_000)
    projected_decode_wall = Fraction(measured_decode_wall_us) - projected_saved
    require(projected_decode_wall > 0,
            "projected decode wall is not positive")
    profile_lines = [
        f"L{layer}:" + ",".join(str(expert) for expert in sorted(selected[layer]))
        for layer in range(LAYERS)
    ]
    profile_text = "/".join(profile_lines)
    digest = hashlib.sha256(profile_text.encode("ascii")).hexdigest()
    projected_decode_ms = float(projected_decode_wall / 1_000)

    return {
        "schema_version": 1,
        "valid": True,
        "profile_kind": "offline-exact-route-weighted-residency-ceiling",
        "fixture_specialized": True,
        "throughput_promotion_allowed": False,
        "capacity_total": sum(EXPECTED_CAPACITIES),
        "capacities": list(EXPECTED_CAPACITIES),
        "route_occurrences": route_occurrences,
        "current_resident_occurrences": current_coverage,
        "profile_resident_occurrences": selected_coverage,
        "projected_residual_cold_occurrences": residual_occurrences,
        "observed_wave_load_ms": round(baseline_wave_us / 1_000, 3),
        "projected_residual_wave_load_ms": round(float(projected_residual / 1_000), 3),
        "projected_wave_saved_ms": round(float(projected_saved / 1_000), 3),
        "measured_decode_wall_ms": round(measured_decode_wall_us / 1_000, 3),
        "required_saved_for_50_tok_s_ms": round(required_saved / 1_000, 3),
        "optimistic_decode_wall_floor_ms": round(projected_decode_ms, 3),
        "optimistic_decode_tok_s_ceiling": round(48_000 / projected_decode_ms, 6),
        "linear_ceiling_at_least_300ms_saved": projected_saved >= 300_000,
        "linear_ceiling_covers_required_50_tok_s_saving": projected_saved >= required_saved,
        "linear_ceiling_allows_at_least_50_tok_s": projected_decode_wall <= 960_000,
        "profile_sha256": digest,
        "profile": profile_lines,
    }


def reconcile_h69_rounds(traces, probes, stages):
    require(len(probes) == len(stages) == len(traces) == 4,
            "H69 analysis did not expose exactly four joined rounds")
    for index, (trace, probe, stage) in enumerate(zip(traces, probes, stages)):
        require(stage["cap"] == 8, f"H69 round {index} is not cap8")
        require(
            (trace["round_seq"], trace["start_pos"], trace["K"])
            == (stage["round_seq"], probe["start_pos"], probe["K"]),
            f"trace round {index} does not join exact H69 stage truth",
        )
        cap8 = probe["caps"][8]
        for layer, observed in zip(h69.TARGET_LAYERS, cap8["layers"]):
            require(len(trace["cold"][layer]) == observed["actual"],
                    f"trace round {index} L{layer} cold identities disagree with H69")
            require(trace["wave_load_us"][layer] == observed["wall_us"],
                    f"trace round {index} L{layer} wave wall disagrees with H69")
        # H69 formats the raw 30-layer sum once, while this trace sums 30
        # individually rounded microsecond values. Their only legal drift is
        # bounded decimal rounding, never a layer-shifted accounting change.
        require(abs(trace["total_wave_load_us"] - probe["total_wave_load_us"]) <= LAYERS,
                f"trace round {index} total wave wall exceeds rounding tolerance")


def analyze_run(run_dir, control_run_dir):
    raw_run_path = Path(run_dir)
    raw_control_path = Path(control_run_dir)
    require(raw_run_path.is_dir() and not raw_run_path.is_symlink(),
            "trace run directory is missing or a symlink")
    require(raw_control_path.is_dir() and not raw_control_path.is_symlink(),
            "control run directory is missing or a symlink")
    run_path = raw_run_path.resolve()
    control_path = raw_control_path.resolve()
    h69.analyze(run_path)
    h69.analyze(control_path)
    try:
        log = (run_path / "server.log").read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise TraceError(f"cannot read strict server log: {error}") from error
    lines = [line for line in log.splitlines() if line.startswith(PREFIX)]
    traces = [parse_trace_line(line) for line in lines]
    probes, stages = h69.parse_observations(log)
    reconcile_h69_rounds(traces, probes, stages)
    try:
        control_log = (control_path / "server.log").read_text(
            encoding="utf-8", errors="strict"
        )
    except (OSError, UnicodeError) as error:
        raise TraceError(f"cannot read strict control server log: {error}") from error
    require(not any(line.startswith(PREFIX) for line in control_log.splitlines()),
            "clean control log contains an exact-residency trace")
    control_decode_wall_us = decode_wall_us(control_log)
    result = solve_profile(traces, control_decode_wall_us)
    environment = h69.parse_env_manifest(run_path / "env.txt")
    control_environment = h69.parse_env_manifest(control_path / "env.txt")
    require(
        environment.get(h69.EXACT_RESIDENCY_TRACE_SELECTOR) == "1",
        "environment did not select the literal exact-residency trace",
    )
    require(
        h69.EXACT_RESIDENCY_TRACE_SELECTOR not in control_environment
        and h69.STAGE_CAP16_SELECTOR not in control_environment,
        "clean control selected a trace or cap16 descendant",
    )
    comparable_trace_environment = dict(environment)
    comparable_trace_environment.pop(h69.EXACT_RESIDENCY_TRACE_SELECTOR)
    require(comparable_trace_environment == control_environment,
            "trace/control environment or provenance differs beyond the trace selector")
    result["run"] = run_path.name
    result["clean_control_run"] = control_path.name
    result["source_commit"] = environment["source_commit"]
    result["binary_sha256"] = environment["binary_sha256"]
    result["trace_observation_decode_wall_ms"] = round(decode_wall_us(log) / 1_000, 3)
    result["clean_control_decode_wall_ms"] = round(control_decode_wall_us / 1_000, 3)
    result["trace_h69_valid"] = True
    result["control_h69_valid"] = True
    result["trace_lines_sha256"] = hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()
    result["trace_server_log_sha256"] = hashlib.sha256(log.encode("utf-8")).hexdigest()
    result["control_server_log_sha256"] = hashlib.sha256(
        control_log.encode("utf-8")
    ).hexdigest()
    result["trace_rounds"] = [
        {
            "round_seq": trace["round_seq"],
            "start_pos": trace["start_pos"],
            "K": trace["K"],
            "exact_unique_records": trace["exact_unique_records"],
            "resident_hits": trace["resident_hits"],
            "cold_records": trace["cold_records"],
            "wave_load_ms": round(trace["total_wave_load_us"] / 1_000, 3),
        }
        for trace in traces
    ]
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 4) or (len(args) == 4 and args[2] != "--output"):
        print(
            f"usage: {os.path.basename(sys.argv[0])} TRACE_RUN_DIR "
            "CLEAN_CONTROL_RUN_DIR [--output FILE]",
            file=sys.stderr,
        )
        return 64
    try:
        result = analyze_run(args[0], args[1])
    except (h69.ReceiptError, TraceError, OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 75
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if len(args) == 4:
        Path(args[3]).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
