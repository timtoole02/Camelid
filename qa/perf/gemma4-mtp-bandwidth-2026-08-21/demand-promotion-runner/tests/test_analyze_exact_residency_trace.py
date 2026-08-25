#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
RUNNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER))
SCRIPT = RUNNER / "analyze_exact_residency_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_exact_residency_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def mask(experts):
    value = sum(1 << expert for expert in experts)
    return f"{value:032x}"


def trace_line(round_seq, start_pos, k_tokens):
    capacities = ANALYZER.EXPECTED_CAPACITIES
    routes = [set(range(120, 128)) for _ in range(30)]
    residents = [set(range(capacity)) for capacity in capacities]
    cold = [route - resident for route, resident in zip(routes, residents)]
    masks = lambda values: "/".join(
        f"L{layer}:{mask(experts)}" for layer, experts in enumerate(values)
    )
    fields = {
        "schema": "1",
        "requested": "1",
        "admitted": "1",
        "truth_valid": "1",
        "scope": "successful-target-decode-round",
        "profile": "mini2-h49-1408",
        "round_seq": str(round_seq),
        "start_pos": str(start_pos),
        "K": str(k_tokens),
        "layers": "30",
        "experts": "128",
        "stage_cap": "8",
        "capacity_total": "1408",
        "resident_total": "1408",
        "exact_unique_records": "240",
        "resident_hits": "0",
        "cold_records": "240",
        "total_wave_load_ms": "30.000",
        "capacities": ",".join(map(str, capacities)),
        "route_sizes": ",".join(["8"] * 30),
        "resident_sizes": ",".join(map(str, capacities)),
        "cold_sizes": ",".join(["8"] * 30),
        "wave_load_ms_per_layer": ",".join(["1.000"] * 30),
        "route_masks": masks(routes),
        "resident_masks": masks(residents),
        "cold_masks": masks(cold),
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
    assert set(fields) == ANALYZER.FIELDS
    return ANALYZER.PREFIX + " ".join(f"{key}={value}" for key, value in fields.items())


class ExactResidencyTraceTest(unittest.TestCase):
    def traces(self):
        return [
            ANALYZER.parse_trace_line(trace_line(41 + index, start_pos, k_tokens))
            for index, (start_pos, k_tokens) in enumerate(ANALYZER.EXPECTED_SCHEDULE)
        ]

    def test_valid_trace_and_solver_fill_exact_profile(self):
        traces = self.traces()
        result = ANALYZER.solve_profile(traces, 1_200_000)
        self.assertEqual(result["capacity_total"], 1408)
        self.assertEqual(result["route_occurrences"], 960)
        self.assertEqual(result["profile_resident_occurrences"], 960)
        self.assertEqual(result["projected_residual_cold_occurrences"], 0)
        self.assertEqual(result["projected_wave_saved_ms"], 120.0)
        self.assertFalse(result["linear_projection_at_least_300ms_saved"])
        self.assertFalse(result["physical_demand_identity_bound"])
        self.assertEqual(len(result["profile"]), 30)

    def test_solver_prefers_recurrent_weighted_identities_over_low_ids(self):
        traces = self.traces()
        route_sets = (
            frozenset(range(0, 64)),
            frozenset(range(0, 64)),
            frozenset(range(32, 96)),
            frozenset(range(40, 96)),
        )
        layer = 4
        for trace, routes in zip(traces, route_sets):
            residents = trace["residents"][layer]
            cold = routes - residents
            old_wave = trace["wave_load_us"][layer]
            wave = list(trace["wave_load_us"])
            wave[layer] = len(cold) * 100
            all_routes = list(trace["routes"])
            all_cold = list(trace["cold"])
            all_routes[layer] = routes
            all_cold[layer] = cold
            trace.update(
                routes=tuple(all_routes),
                cold=tuple(all_cold),
                wave_load_us=tuple(wave),
                total_wave_load_us=trace["total_wave_load_us"] - old_wave + wave[layer],
            )
        result = ANALYZER.solve_profile(traces, 1_200_000)
        selected = {int(value) for value in result["profile"][layer].split(":", 1)[1].split(",")}
        self.assertEqual(len(selected), ANALYZER.EXPECTED_CAPACITIES[layer])
        self.assertTrue(set(range(40, 64)) <= selected)

    def test_cold_mask_or_total_wall_forgery_is_rejected(self):
        valid = trace_line(41, 104, 14)
        forged_mask = valid.replace(
            "cold_masks=L0:ff000000000000000000000000000000",
            "cold_masks=L0:7f000000000000000000000000000000",
            1,
        )
        forged_wall = valid.replace("total_wave_load_ms=30.000", "total_wave_load_ms=29.999", 1)
        for line in (forged_mask, forged_wall):
            with self.subTest(line=line), self.assertRaises(ANALYZER.TraceError):
                ANALYZER.parse_trace_line(line)

    def test_schedule_and_frozen_residency_are_required(self):
        traces = self.traces()
        traces[2] = dict(traces[2], start_pos=132)
        with self.assertRaises(ANALYZER.TraceError):
            ANALYZER.solve_profile(traces, 1_200_000)
        traces = self.traces()
        altered = list(traces[2]["residents"])
        altered[0] = frozenset((altered[0] - {0}) | {100})
        traces[2] = dict(traces[2], residents=tuple(altered))
        with self.assertRaises(ANALYZER.TraceError):
            ANALYZER.solve_profile(traces, 1_200_000)

    def test_underfilled_residency_is_rejected_for_profile_solving(self):
        valid = trace_line(41, 104, 14)
        underfilled = valid.replace(
            "resident_total=1408",
            "resident_total=1407",
            1,
        ).replace(
            "resident_sizes=60,64",
            "resident_sizes=59,64",
            1,
        ).replace(
            f"resident_masks=L0:{mask(range(60))}",
            f"resident_masks=L0:{mask(range(59))}",
            1,
        )
        with self.assertRaises(ANALYZER.TraceError):
            ANALYZER.parse_trace_line(underfilled)

    def test_solver_rejects_nonpositive_or_sub_wave_control_wall(self):
        traces = self.traces()
        for wall_us in (0, -1, 119_999):
            with self.subTest(wall_us=wall_us), self.assertRaises(ANALYZER.TraceError):
                ANALYZER.solve_profile(traces, wall_us)

    def test_decode_wall_requires_canonical_exact_acceptance_rounds(self):
        lines = [
            f"[mtp round] #{index} wall=100.00ms (assistant=20.00ms, "
            f"verifier=80.00ms) accepted={width - 1}/{width}"
            for index, width in enumerate((14, 13, 14, 7))
        ]
        self.assertEqual(ANALYZER.decode_wall_us("\n".join(lines)), 400_000)
        forged = lines.copy()
        forged[0] = forged[0].replace("accepted=13/14", "accepted=12/14")
        forged[1] = forged[1].replace("accepted=12/13", "accepted=13/13")
        with self.assertRaises(ANALYZER.TraceError):
            ANALYZER.decode_wall_us("\n".join(forged))
        malformed_extra = lines + ["[mtp round] malformed"]
        with self.assertRaises(ANALYZER.TraceError):
            ANALYZER.decode_wall_us("\n".join(malformed_extra))

    def test_trace_reconciles_every_h69_layer_and_bounded_total_rounding(self):
        trace = self.traces()[0]
        probe = {
            "start_pos": trace["start_pos"],
            "K": trace["K"],
            "total_wave_load_us": trace["total_wave_load_us"] + 30,
            "caps": {8: {"layers": [{"actual": 8, "wall_us": 1_000}] * 29}},
        }
        stage = {"cap": 8, "round_seq": trace["round_seq"], "exact_cold": 240}
        traces = [dict(trace, round_seq=41 + index) for index in range(4)]
        probes = [dict(probe) for _ in range(4)]
        stages = [dict(stage, round_seq=41 + index) for index in range(4)]
        ANALYZER.reconcile_h69_rounds(traces, probes, stages)
        probes[2] = dict(probes[2], caps={8: {"layers": [
            {"actual": 7, "wall_us": 1_000},
            *([{"actual": 8, "wall_us": 1_000}] * 28),
        ]}})
        with self.assertRaises(ANALYZER.TraceError):
            ANALYZER.reconcile_h69_rounds(traces, probes, stages)

    def test_h70_h71_h72_are_exact_selector_descendants_of_h49(self):
        env_dir = RUNNER / "env"
        baseline = ANALYZER.h69._parse_profile(
            env_dir / "H49-live-hidden-sequential-fast-predict-dual-reader-kv192-control"
        )
        expected_selectors = {
            "H70-exact-residency-trace-kv192-control": {
                ANALYZER.h69.EXACT_RESIDENCY_TRACE_SELECTOR: "1",
            },
            "H71-prompt-ranked-hot-handoff-kv192": {
                ANALYZER.h69.PROMPT_RANKED_HOT_HANDOFF_SELECTOR: "1",
            },
            "H72-prompt-ranked-hot-handoff-trace-kv192": {
                ANALYZER.h69.PROMPT_RANKED_HOT_HANDOFF_SELECTOR: "1",
                ANALYZER.h69.EXACT_RESIDENCY_TRACE_SELECTOR: "1",
            },
        }
        for profile_name, selectors in expected_selectors.items():
            with self.subTest(profile=profile_name):
                observed = ANALYZER.h69._parse_profile(env_dir / profile_name)
                for key, value in selectors.items():
                    self.assertEqual(observed.pop(key), value)
                self.assertEqual(observed, baseline)


if __name__ == "__main__":
    unittest.main()
