#!/usr/bin/env python3
"""Model-free boundary tests for the hybrid runner and receipt analyzer."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock


RUNNER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = next(parent for parent in RUNNER_DIR.parents if (parent / ".git").exists())
ANALYZER_PATH = RUNNER_DIR / "hybrid_receipt.py"
RUNNER_PATH = RUNNER_DIR / "run_stage.zsh"
WATCHDOG_PATH = (
    REPO_ROOT
    / "qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/run_load_only_watchdog.py"
)


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receipt = load_module("hybrid_receipt_under_test", ANALYZER_PATH)
watchdog = load_module("hybrid_watchdog_under_test", WATCHDOG_PATH)


def host_sample(**changes: int) -> dict[str, int]:
    sample = {
        "swapins_pages": 20,
        "swapouts_pages": 10,
        "swapped_pages_current": 0,
        "pressure_level_raw": 1,
        "reclaimable_headroom_bytes": 8 * receipt.GIB,
        "wired_bytes": 8 * receipt.GIB,
    }
    sample.update(changes)
    return sample


def lane_environment(lane: str) -> dict[str, str]:
    environment = dict(receipt.COMMON_ENVIRONMENT)
    environment.update(
        {
            "CAMELID_GEMMA4_SPEC_CHUNK_MAX": "1" if lane == "k1" else "8",
            "CAMELID_GEMMA4_SPEC_DRAFT_TOKENS": "1" if lane == "k1" else "8",
        }
    )
    if lane == "k1":
        environment["CAMELID_GEMMA4_CHAINED_K1"] = "1"
    if lane != "k1":
        environment["CAMELID_GEMMA4_MTP_ASSISTANT_PATH"] = "/internal/assistant"
    return environment


def hybrid_telemetry(unique: int = 64) -> dict[str, object]:
    hot = min(unique, receipt.HOT_PER_LAYER)
    cold = unique - hot
    per_layer_geometry = [
        {
            "layer": layer,
            "logical_addressable_slots": receipt.CANONICAL_PER_LAYER,
            "anonymous_hot_capacity_slots": receipt.HOT_PER_LAYER,
            "file_mapped_addressable_slots": receipt.CANONICAL_PER_LAYER,
            "file_mapped_address_span_bytes": receipt.MAPPED_COLD_SPAN_BYTES_PER_LAYER,
            "overflow_slots": 0,
            "victim_slots": 0,
        }
        for layer in range(receipt.LAYERS)
    ]
    per_round_layer = [
        {
            "layer_index": layer,
            "active_unique": unique,
            "hot_bound": hot,
            "mapped_bound": cold,
            "bound_records": unique,
        }
        for layer in range(receipt.LAYERS)
    ]
    bootstrap_layer = [
        {
            "layer_index": layer,
            "active_unique": 8,
            "hot_bound": 8,
            "mapped_bound": 0,
            "bound_records": 8,
        }
        for layer in range(receipt.LAYERS)
    ]
    hot_total = (8 + hot) * receipt.LAYERS
    cold_total = cold * receipt.LAYERS
    return {
        "schema_version": 1,
        "scope": "single_completed_measured_request",
        "route_interval": {"scope": "measured_request_prefill_plus_generation"},
        "geometry": {
            "layers": receipt.LAYERS,
            "record_payload_bytes": receipt.RECORD_PAYLOAD_BYTES,
            "slot_stride_bytes": receipt.SLOT_STRIDE_BYTES,
            "logical_addressable_slots": receipt.CANONICAL_TOTAL,
            "anonymous_hot_capacity_slots": receipt.HOT_TOTAL,
            "anonymous_hot_capacity_bytes": receipt.HOT_CAPACITY_BYTES,
            "file_mapped_addressable_slots": receipt.CANONICAL_TOTAL,
            "file_mapped_address_span_bytes": receipt.MAPPED_COLD_SPAN_BYTES,
            "overflow_slots": 0,
            "overflow_capacity_bytes": 0,
            "victim_record_capacity": 0,
            "victim_capacity_bytes": 0,
            "host_cache_budget_bytes": 0,
            "per_layer": per_layer_geometry,
        },
        "rounds": [
            {
                "round_index": 0,
                "chained_round_sequence": 10,
                "prefix_tokens_before": 100,
                "bootstrap": True,
                "remaining_budget_before": 9,
                "k": 1,
                "requested_k": 1,
                "proposed_k": 0,
                "verifier_k": 1,
                "budget_truncated": False,
                "success": True,
                "accepted_drafts": 0,
                "useful_accepted_drafts": 0,
                "committed_tokens": [10],
                "assistant_exposed_ms": 0.0,
                "assistant_gpu_ms": 0.0,
                "receipt_round_wall_ms": 10.0,
                "selected_dropped": 0,
                "missing_failclose": 0,
                "slot_capacity_overflow": 0,
                "overflow_experts": 0,
                "per_layer": bootstrap_layer,
            },
            {
                "round_index": 1,
                "chained_round_sequence": 11,
                "prefix_tokens_before": 101,
                "bootstrap": False,
                "remaining_budget_before": 8,
                "k": 8,
                "requested_k": 8,
                "proposed_k": 7,
                "verifier_k": 8,
                "budget_truncated": False,
                "success": True,
                "accepted_drafts": 7,
                "useful_accepted_drafts": 7,
                "committed_tokens": list(range(11, 19)),
                "assistant_exposed_ms": 20.0,
                "assistant_gpu_ms": 15.0,
                "receipt_round_wall_ms": 290.0,
                "selected_dropped": 0,
                "missing_failclose": 0,
                "slot_capacity_overflow": 0,
                "overflow_experts": 0,
                "per_layer": per_round_layer,
            }
        ],
        "aggregate": {
            "scope": "single_completed_measured_request",
            "route_lookups": hot_total + cold_total,
            "hot_hits": hot_total,
            "mapped_cold_selections": cold_total,
            "direct_reads": 0,
            "direct_read_bytes": 0,
            "host_cache_hits": 0,
            "host_cache_misses": 0,
            "host_cache_evictions": 0,
            "overflow_experts": 0,
            "victim_hits": 0,
            "chained_promotion_loads": 1,
            "chained_promotion_read_bytes": receipt.RECORD_PAYLOAD_BYTES,
        },
        "metrics": {
            "forwarded_decode_tokens": 9,
            "terminal_unforwarded_tokens": 0,
            "response_completion_tokens": 9,
            "receipt_round_wall_ms": 300.0,
            "proposed_drafts": 7,
            "accepted_drafts": 7,
            "decode_tokens_per_second": 30.0,
            "full_round_zero_accepts": 0,
            "max_full_assistant_exposed_ms": 20.0,
            "outer_lookahead_nonzero_count": 0,
        },
    }


def watchdog_events(lane_dir: Path, lane: str = "k8") -> list[dict[str, object]]:
    host = host_sample()
    events: list[dict[str, object]] = [
        {
            "schema_version": 3,
            "event": "clean_parent_baseline",
            "sequence": 0,
            "host": host,
            "violations": [],
        }
    ]
    for index in range(1, 241):
        events.append(
            {
                "schema_version": 3,
                "event": "baseline_soak_sample",
                "sequence": index,
                "scheduled_monotonic_ns": index * receipt.WATCHDOG_SAMPLE_PERIOD_NS,
                "schedule_lateness_ns": 0,
                "telemetry_duration_ns": 1,
                "host": host,
                "violations": [],
            }
        )
    events.extend(
        [
            {
                "schema_version": 3,
                "event": "baseline_soak_complete",
                "sequence": 240,
                "required_duration_seconds": 60.0,
                "observed_duration_ns": receipt.MIN_BASELINE_SOAK_NS,
                "minimum_reclaimable_headroom_bytes": receipt.MIN_BASELINE_HEADROOM_BYTES,
                "require_zero_current_swap": True,
            },
            {
                "schema_version": 3,
                "event": "child_started",
                "sequence": 240,
                "pid": 1234,
                "process_group": 1234,
                "process_accounting_scope": "isolated_process_group_aggregate",
                "sample_period_ns": receipt.WATCHDOG_SAMPLE_PERIOD_NS,
                "minimum_reclaimable_headroom_bytes": receipt.MIN_RUNTIME_HEADROOM_BYTES,
                "maximum_child_physical_footprint_bytes": receipt.MAX_CHILD_FOOTPRINT_BYTES,
                "maximum_host_wired_bytes": receipt.MAX_HOST_WIRED_BYTES,
                "require_zero_current_swap": True,
                "reject_swapin_growth": True,
                "report_producer": "child" if lane == "load-only" else "external",
                "experiment_environment": lane_environment(lane),
                "report": str(
                    lane_dir
                    / ("load-only-report.json" if lane == "load-only" else "response.json")
                ),
            },
            {
                "schema_version": 3,
                "event": "sample",
                "sequence": 241,
                "host": host,
                "process": {"physical_footprint_bytes": 1024},
                "violations": [],
            },
            {
                "schema_version": 3,
                "event": "post_exit_sample",
                "sequence": 242,
                "host": host,
                "violations": [],
            },
            {
                "schema_version": 3,
                "event": "final",
                "sequence": 243,
                "pid": 1234,
                "process_group": 1234,
                "process_group_empty": True,
                "process_accounting_scope": "isolated_process_group_aggregate",
                "child_returncode": 0,
                "watchdog_aborted": False,
                "abort_reasons": [],
                "peak_child_physical_footprint_bytes": 1024,
                "peak_host_wired_bytes": receipt.MAX_HOST_WIRED_BYTES,
                "report_exists": True,
                "report_size_bytes": 1,
                "report_is_regular_file": True,
                "report_is_symlink": False,
            },
        ]
    )
    return events


def startup_log(k8: bool = False) -> str:
    lines = [
        "[gemma4-ghost-metal] HYBRID ACTIVE: CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1 CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32, layers=30 canonical_addressable=128/layer physical_hot=32/layer hot_capacity_bytes=3224371200 mapped_cold_span_bytes=12897484800 overflow=0 victim=0 slot_pin=off prediction=off",
        "[gemma4-ghost-metal] " + receipt.DEMAND_PREWARM_MARKER,
        "[gemma4-ghost-metal] clean file-pager Q4_0 experts enabled: layers=30 logical_addressable_slots/layer=128 anonymous_expert_capacity_bytes=3224371200 mapped_address_span_bytes=12897484800 mapped_address_span=12.01GiB mode=fused-fast",
    ]
    if k8:
        lines.extend(
            [
                "lm_head=q4_0",
            ]
        )
    return "\n".join(lines)


class WatchdogBoundaryTests(unittest.TestCase):
    @staticmethod
    def _process_sample(pid: int, multiplier: int = 1) -> dict[str, int]:
        return {
            "pid": pid,
            "rss_bytes": 10 * multiplier,
            "physical_footprint_bytes": 20 * multiplier,
            "wired_bytes": 30 * multiplier,
            "pageins": 40 * multiplier,
            "disk_read_bytes": 50 * multiplier,
            "disk_written_bytes": 60 * multiplier,
        }

    def test_host_threshold_boundaries_and_counter_regression(self) -> None:
        baseline = host_sample()
        self.assertEqual(
            watchdog._host_violation_reasons(
                baseline,
                10,
                baseline_swapins_pages=20,
                minimum_reclaimable_headroom_bytes=8 * receipt.GIB,
                maximum_host_wired_bytes=8 * receipt.GIB,
                require_zero_current_swap=True,
            ),
            [],
        )
        mutations = (
            {"swapins_pages": 21},
            {"swapins_pages": 19},
            {"swapouts_pages": 11},
            {"swapouts_pages": 9},
            {"swapped_pages_current": 1},
            {"pressure_level_raw": 2},
            {"reclaimable_headroom_bytes": 8 * receipt.GIB - 1},
            {"wired_bytes": 8 * receipt.GIB + 1},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertTrue(
                    watchdog._host_violation_reasons(
                        host_sample(**mutation),
                        10,
                        baseline_swapins_pages=20,
                        minimum_reclaimable_headroom_bytes=8 * receipt.GIB,
                        maximum_host_wired_bytes=8 * receipt.GIB,
                        require_zero_current_swap=True,
                    )
                )

    def test_group_signal_does_not_depend_on_leader_poll(self) -> None:
        with mock.patch.object(watchdog.os, "killpg") as killpg:
            self.assertTrue(watchdog._send_group_signal(4321, 15, 4321))
        killpg.assert_called_once_with(4321, 15)

    def test_departed_nonleader_restarts_complete_group_sample(self) -> None:
        leader = 4321
        helper = 4322
        telemetry = mock.Mock()
        telemetry.process_group_pids.side_effect = [
            [leader, helper],
            [leader],
        ]
        telemetry.sample_process.side_effect = [
            self._process_sample(leader),
            watchdog.ProcessDisappeared(helper, "helper vanished"),
            self._process_sample(leader, multiplier=2),
        ]

        with mock.patch.object(watchdog, "_pid_exists", return_value=False):
            sample = watchdog.NativeTelemetry.sample_process_group(
                telemetry, leader, leader
            )

        self.assertEqual(sample["member_pids"], [leader])
        self.assertEqual(sample["initial_member_pids"], [leader, helper])
        self.assertEqual(sample["departed_member_pids"], [helper])
        self.assertEqual(
            sample["membership_rechecks"],
            [
                {
                    "sampling_attempt": 1,
                    "failed_pid": helper,
                    "telemetry_error_type": "ProcessDisappeared",
                    "telemetry_error": "helper vanished",
                    "member_pids": [leader],
                }
            ],
        )
        self.assertEqual(sample["sampling_attempts"], 2)
        self.assertEqual(sample["member_count"], 1)
        self.assertEqual(sample["rss_bytes"], 20)
        self.assertEqual(sample["physical_footprint_bytes"], 40)
        self.assertEqual(
            telemetry.sample_process.call_args_list,
            [mock.call(leader), mock.call(helper), mock.call(leader)],
        )

    def test_live_nonleader_failure_immediately_fails_closed(self) -> None:
        leader = 4321
        helper = 4322
        telemetry = mock.Mock()
        telemetry.process_group_pids.return_value = [leader, helper]
        telemetry.sample_process.side_effect = [
            self._process_sample(leader),
            watchdog.ProcessAccountingError(helper, "live helper failure"),
        ]

        with self.assertRaisesRegex(
            watchdog.ProcessAccountingError, "live helper failure"
        ):
            watchdog.NativeTelemetry.sample_process_group(
                telemetry, leader, leader
            )
        self.assertEqual(telemetry.sample_process.call_count, 2)
        telemetry.process_group_pids.assert_called_once_with(leader)

    def test_group_absence_does_not_tolerate_live_escaped_member(self) -> None:
        leader = 4321
        helper = 4322
        telemetry = mock.Mock()
        telemetry.process_group_pids.side_effect = [
            [leader, helper],
            [leader],
        ]
        telemetry.sample_process.side_effect = [
            self._process_sample(leader),
            watchdog.ProcessDisappeared(helper, "helper accounting vanished"),
        ]

        with mock.patch.object(watchdog, "_pid_exists", return_value=True):
            with self.assertRaisesRegex(
                watchdog.TelemetryError, "remains live or escaped"
            ):
                watchdog.NativeTelemetry.sample_process_group(
                    telemetry, leader, leader
                )

    def test_zero_rusage_is_a_typed_live_accounting_failure(self) -> None:
        pid = 4322
        telemetry = object.__new__(watchdog.NativeTelemetry)
        telemetry.lib = mock.Mock()
        telemetry.lib.proc_pid_rusage.return_value = 0

        with mock.patch.object(watchdog.os, "kill") as kill:
            with self.assertRaisesRegex(
                watchdog.ProcessAccountingError,
                "returned zero resident/footprint bytes",
            ):
                telemetry.sample_process(pid)
        kill.assert_called_once_with(pid, 0)

    def test_zero_rusage_with_esrch_is_typed_as_disappeared(self) -> None:
        pid = 4322
        telemetry = object.__new__(watchdog.NativeTelemetry)
        telemetry.lib = mock.Mock()
        telemetry.lib.proc_pid_rusage.return_value = 0

        with mock.patch.object(
            watchdog.os, "kill", side_effect=ProcessLookupError
        ):
            with self.assertRaisesRegex(
                watchdog.ProcessDisappeared,
                "returned zero resident/footprint bytes",
            ):
                telemetry.sample_process(pid)

    def test_pid_liveness_permission_error_is_fail_closed_live(self) -> None:
        with mock.patch.object(
            watchdog.os, "kill", side_effect=PermissionError
        ):
            self.assertTrue(watchdog._pid_exists(4322))

    def test_required_leader_failure_is_never_skipped(self) -> None:
        leader = 4321
        telemetry = mock.Mock()
        telemetry.process_group_pids.return_value = [leader]
        telemetry.sample_process.side_effect = watchdog.ProcessDisappeared(
            leader, "leader accounting failed"
        )

        with self.assertRaisesRegex(
            watchdog.TelemetryError, "leader accounting failed"
        ):
            watchdog.NativeTelemetry.sample_process_group(
                telemetry, leader, leader
            )
        telemetry.process_group_pids.assert_called_once_with(leader)

    def test_second_departure_fails_closed_instead_of_churning(self) -> None:
        leader = 4321
        first_helper = 4322
        second_helper = 4323
        telemetry = mock.Mock()
        telemetry.process_group_pids.side_effect = [
            [leader, first_helper],
            [leader, second_helper],
            [leader],
        ]
        telemetry.sample_process.side_effect = [
            self._process_sample(leader),
            watchdog.ProcessDisappeared(first_helper, "first helper vanished"),
            self._process_sample(leader),
            watchdog.ProcessDisappeared(second_helper, "second helper vanished"),
        ]

        with mock.patch.object(watchdog, "_pid_exists", return_value=False):
            with self.assertRaisesRegex(
                watchdog.TelemetryError,
                "changed during both telemetry attempts",
            ):
                watchdog.NativeTelemetry.sample_process_group(
                    telemetry, leader, leader
                )



class ReceiptTests(unittest.TestCase):
    def test_hot32_geometry_and_valid_33_through_64_cold_spill(self) -> None:
        self.assertEqual(receipt.HOT_PER_LAYER, 32)
        self.assertEqual(receipt.HOT_TOTAL, 960)
        self.assertEqual(receipt.HOT_CAPACITY_BYTES, 3_224_371_200)
        for unique in (33, 64):
            with self.subTest(unique=unique):
                metrics = receipt.validate_hybrid_telemetry(
                    hybrid_telemetry(unique), "k8", 9
                )
                self.assertEqual(metrics["decode_tokens_per_second"], 30.0)

    def test_drop_or_partition_drift_fails_closed(self) -> None:
        telemetry = hybrid_telemetry(64)
        telemetry["rounds"][0]["selected_dropped"] = 1
        with self.assertRaises(receipt.ReceiptError):
            receipt.validate_hybrid_telemetry(telemetry, "k8", 9)

        telemetry = hybrid_telemetry(64)
        telemetry["rounds"][1]["per_layer"][0].update(
            {
                "active_unique": 1,
                "hot_bound": 0,
                "mapped_bound": 1,
                "bound_records": 1,
            }
        )
        with self.assertRaisesRegex(receipt.ReceiptError, "outside 8..K×8"):
            receipt.validate_hybrid_telemetry(telemetry, "k8", 9)

    def test_nine_token_smoke_proves_bootstrap_plus_full_k8(self) -> None:
        receipt.validate_hybrid_telemetry(
            hybrid_telemetry(64), "k8", 9, list(range(10, 19))
        )

        with self.assertRaisesRegex(receipt.ReceiptError, "API response"):
            receipt.validate_hybrid_telemetry(
                hybrid_telemetry(64), "k8", 9, list(range(10, 18)) + [999]
            )

        actual_eight_token_shape = hybrid_telemetry(56)
        bootstrap = actual_eight_token_shape["rounds"][0]
        bootstrap["remaining_budget_before"] = 8
        truncated = actual_eight_token_shape["rounds"][1]
        truncated.update(
            {
                "remaining_budget_before": 7,
                "k": 7,
                "proposed_k": 6,
                "verifier_k": 7,
                "budget_truncated": True,
                "accepted_drafts": 6,
                "useful_accepted_drafts": 6,
                "committed_tokens": list(range(11, 18)),
                "assistant_exposed_ms": 65.668,
                "assistant_gpu_ms": 40.727,
                "receipt_round_wall_ms": 290.0,
            }
        )
        actual_eight_token_shape["metrics"].update(
            {
                "forwarded_decode_tokens": 8,
                "response_completion_tokens": 8,
                "proposed_drafts": 6,
                "accepted_drafts": 6,
                "decode_tokens_per_second": 8 / 0.3,
                "max_full_assistant_exposed_ms": 0.0,
            }
        )
        with self.assertRaisesRegex(receipt.ReceiptError, "no K=8 round"):
            receipt.validate_hybrid_telemetry(actual_eight_token_shape, "k8", 8)

    def test_k8_bootstrap_is_mandatory_exact_and_first(self) -> None:
        cases = (
            ("missing", lambda telemetry: telemetry["rounds"].pop(0)),
            (
                "reordered",
                lambda telemetry: telemetry["rounds"].reverse(),
            ),
            (
                "assistant exposure",
                lambda telemetry: telemetry["rounds"][0].__setitem__(
                    "assistant_exposed_ms", 0.01
                ),
            ),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                telemetry = hybrid_telemetry(64)
                mutate(telemetry)
                with self.assertRaises(receipt.ReceiptError):
                    receipt.validate_hybrid_telemetry(telemetry, "k8", 9)

    def test_k1_rounds_are_zero_draft_and_reconcile_response_tokens(self) -> None:
        telemetry = hybrid_telemetry(8)
        round_receipt = telemetry["rounds"][0]
        round_receipt.update(
            {
                "bootstrap": False,
                "remaining_budget_before": 2,
                "k": 1,
                "requested_k": 1,
                "proposed_k": 0,
                "verifier_k": 1,
                "accepted_drafts": 0,
                "useful_accepted_drafts": 0,
                "assistant_exposed_ms": 0.0,
                "assistant_gpu_ms": 0.0,
                "committed_tokens": [10],
            }
        )
        telemetry["rounds"] = [round_receipt]
        telemetry["aggregate"]["chained_promotion_loads"] = 0
        telemetry["aggregate"]["chained_promotion_read_bytes"] = 0
        telemetry["metrics"].update(
            {
                "forwarded_decode_tokens": 1,
                "terminal_unforwarded_tokens": 1,
                "response_completion_tokens": 2,
                "receipt_round_wall_ms": 10.0,
                "proposed_drafts": 0,
                "accepted_drafts": 0,
                "decode_tokens_per_second": 100.0,
                "full_round_zero_accepts": 0,
                "max_full_assistant_exposed_ms": 0.0,
                "outer_lookahead_nonzero_count": 0,
            }
        )
        receipt.validate_hybrid_telemetry(telemetry, "k1", 2)
        telemetry["rounds"][0]["assistant_exposed_ms"] = 0.01
        with self.assertRaises(receipt.ReceiptError):
            receipt.validate_hybrid_telemetry(telemetry, "k1", 2)
        telemetry = hybrid_telemetry(64)
        telemetry["rounds"][1]["per_layer"][0]["mapped_bound"] = 15
        with self.assertRaises(receipt.ReceiptError):
            receipt.validate_hybrid_telemetry(telemetry, "k8", 9)

    def test_environment_requires_hybrid_knob_and_rejects_legacy_physical(self) -> None:
        environment = lane_environment("k8")
        receipt.validate_environment(environment, "k8")
        environment["CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER"] = "32"
        with self.assertRaises(receipt.ReceiptError):
            receipt.validate_environment(environment, "k8")

    def test_unknown_lane_token_contract_fails_before_receipt_io(self) -> None:
        with self.assertRaisesRegex(receipt.ReceiptError, "unsupported lane/token contract"):
            receipt.validate_lane(Path("/definitely/absent"), "k8", 8)

    def test_nine_token_k1_length_finish_has_eight_forwards_and_one_terminal(self) -> None:
        telemetry = hybrid_telemetry(8)
        template = telemetry["rounds"][0]
        template.update(
            {
                "bootstrap": False,
                "k": 1,
                "requested_k": 1,
                "proposed_k": 0,
                "verifier_k": 1,
                "accepted_drafts": 0,
                "useful_accepted_drafts": 0,
                "committed_tokens": [10],
                "assistant_exposed_ms": 0.0,
                "assistant_gpu_ms": 0.0,
            }
        )
        rounds = []
        for index in range(8):
            round_receipt = copy.deepcopy(template)
            round_receipt["round_index"] = index
            round_receipt["chained_round_sequence"] = 10 + index
            round_receipt["prefix_tokens_before"] = 100 + index
            round_receipt["remaining_budget_before"] = 9 - index
            round_receipt["committed_tokens"] = [10 + index]
            rounds.append(round_receipt)
        telemetry["rounds"] = rounds
        telemetry["aggregate"]["chained_promotion_loads"] = 0
        telemetry["aggregate"]["chained_promotion_read_bytes"] = 0
        telemetry["metrics"].update(
            {
                "forwarded_decode_tokens": 8,
                "terminal_unforwarded_tokens": 1,
                "response_completion_tokens": 9,
                "receipt_round_wall_ms": 80.0,
                "proposed_drafts": 0,
                "accepted_drafts": 0,
                "decode_tokens_per_second": 100.0,
                "full_round_zero_accepts": 0,
                "max_full_assistant_exposed_ms": 0.0,
                "outer_lookahead_nonzero_count": 0,
            }
        )
        receipt.validate_hybrid_telemetry(telemetry, "k1", 9)
        telemetry["metrics"]["terminal_unforwarded_tokens"] = 0
        with self.assertRaises(receipt.ReceiptError):
            receipt.validate_hybrid_telemetry(telemetry, "k1", 9)

    def test_smoke_parity_compares_ninth_token_and_frozen_request_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pair = root / "02-smoke-9t"
            ids = list(range(9))
            request_bytes = (RUNNER_DIR / "request-9.json").read_bytes()
            for lane in ("k8", "k1"):
                lane_dir = pair / lane
                lane_dir.mkdir(parents=True)
                request_path = lane_dir / "request.json"
                request_path.write_bytes(request_bytes)
                request_sha = receipt.sha256_regular_file(request_path)
                (lane_dir / "intent.json").write_text(
                    json.dumps(
                        {
                            "request": {
                                "source_path": str(RUNNER_DIR / "request-9.json"),
                                "frozen_path": str(request_path),
                                "sha256": request_sha,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                (lane_dir / "verdict.json").write_text(
                    json.dumps({"pass": True}), encoding="utf-8"
                )
                (lane_dir / "response.json").write_text(
                    json.dumps(
                        {
                            "usage": {"completion_tokens": 9},
                            "camelid": {"generated_token_ids": ids},
                            "choices": [
                                {
                                    "finish_reason": "length",
                                    "message": {"content": "nine tokens"},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            self.assertTrue(receipt.validate_parity(root, "smoke")["pass"])
            k1_response_path = pair / "k1" / "response.json"
            k1_response = json.loads(k1_response_path.read_text(encoding="utf-8"))
            k1_response["camelid"]["generated_token_ids"][-1] = 999
            k1_response_path.write_text(json.dumps(k1_response), encoding="utf-8")
            with self.assertRaisesRegex(receipt.ReceiptError, "token IDs"):
                receipt.validate_parity(root, "smoke")

            k1_response["camelid"]["generated_token_ids"][-1] = 8
            k1_response_path.write_text(json.dumps(k1_response), encoding="utf-8")
            k1_request_path = pair / "k1" / "request.json"
            changed_request = json.loads(k1_request_path.read_text(encoding="utf-8"))
            changed_request["audit_nonce"] = 1
            k1_request_path.write_text(json.dumps(changed_request), encoding="utf-8")
            k1_intent_path = pair / "k1" / "intent.json"
            k1_intent = json.loads(k1_intent_path.read_text(encoding="utf-8"))
            k1_intent["request"]["sha256"] = receipt.sha256_regular_file(k1_request_path)
            k1_intent_path.write_text(json.dumps(k1_intent), encoding="utf-8")
            with self.assertRaisesRegex(receipt.ReceiptError, "canonical lane contract"):
                receipt.validate_parity(root, "smoke")

    def test_exact_startup_markers_are_singletons(self) -> None:
        log = startup_log(k8=True)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "server.log"
            path.write_text(log, encoding="utf-8")
            receipt.validate_startup_log(path, "k8")
            path.write_text(log + "\n" + log.splitlines()[0], encoding="utf-8")
            with self.assertRaises(receipt.ReceiptError):
                receipt.validate_startup_log(path, "k8")

    def test_observed_startup_requires_each_admission_marker(self) -> None:
        log = startup_log() + (
            "\n[gemma4-ghost-common] OBSERVED: Q4 common + KV/scratch + empty slots "
            "active, context cap=1024 KV=0.11GiB mode=fused-fast "
            "q4-row=simdgroup-ordered"
        )
        cases = [
            ("HYBRID ACTIVE", "one exact HYBRID ACTIVE admission line"),
            (receipt.DEMAND_PREWARM_MARKER, "one exact demand-prewarm-skip line"),
            ("clean file-pager Q4_0 experts enabled", "one exact clean file-pager line"),
        ]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "child.log"
            for marker, expected_error in cases:
                with self.subTest(marker=marker):
                    filtered = "\n".join(
                        line for line in log.splitlines() if marker not in line
                    )
                    path.write_text(filtered, encoding="utf-8")
                    with self.assertRaisesRegex(receipt.ReceiptError, expected_error):
                        receipt.validate_startup_log(path, "load-only")

    def test_watchdog_receipt_requires_all_240_soak_samples(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lane_dir = Path(raw)
            path = lane_dir / "watchdog.jsonl"
            events = watchdog_events(lane_dir)
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            receipt.validate_watchdog(path, "k8")
            del events[100]
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaises(receipt.ReceiptError):
                receipt.validate_watchdog(path, "k8")

    def test_load_only_v4_binds_actual_hybrid_and_assistant_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lane = Path(raw)
            sha = "a" * 64
            (lane / "intent.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "load-only",
                        "lane": "load-only",
                        "expected_tokens": 0,
                        "disk": {"available_kib": 20 * 1024 * 1024, "used_percent": 90},
                        "profile": {
                            "demand_load_only": 1,
                            "file_mapped_experts": 1,
                            "hybrid_hot_slots": 32,
                            "slot_pin": 0,
                            "assistant_residency_policy": "observed_from_assistant_ledger",
                            "physical_slots_per_layer": "unset",
                        },
                        "executable": {"sha256": sha},
                        "tooling": {
                            "runner_sha256": sha,
                            "watchdog_sha256": sha,
                            "analyzer_sha256": sha,
                        },
                        "integration_contract": {"sha256": sha},
                        "request": {
                            "source_path": "",
                            "frozen_path": "",
                            "sha256": "",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (lane / "port-clear.json").write_text(
                json.dumps({"schema_version": 1, "port": 8189, "clear": True}),
                encoding="utf-8",
            )
            (lane / "child.log").write_text(startup_log(), encoding="utf-8")
            (lane / "watchdog.jsonl").write_text(
                "".join(
                    json.dumps(event) + "\n"
                    for event in watchdog_events(lane, "load-only")
                ),
                encoding="utf-8",
            )
            effective = dict(receipt.COMMON_ENVIRONMENT)
            effective.update(
                {
                    "CAMELID_GEMMA4_SPEC_CHUNK_MAX": "8",
                    "CAMELID_GEMMA4_SPEC_DRAFT_TOKENS": "8",
                }
            )
            report = {
                "schema_version": 4,
                "completed": True,
                "failure": None,
                "assistant_warmups": 0,
                "assistant_proposals": 0,
                "tokenizer_calls": 0,
                "target_prefills": 0,
                "target_steps": 0,
                "target_generations": 0,
                "target_kv_borrows": 0,
                "assistant_ledger": {
                    "mapped_bytes": 1024,
                    "locked_bytes": 1024,
                    "resident_pages": 1,
                    "total_pages": 1,
                },
                "target_runtime": {"environment": effective},
                "target_final_ledger": {
                    "cghost_logical_bytes": receipt.MAPPED_COLD_SPAN_BYTES,
                    "cghost_mapped_bytes": receipt.MAPPED_COLD_SPAN_BYTES,
                    "expert_layer_count": receipt.LAYERS,
                    "expert_logical_slot_count": receipt.CANONICAL_TOTAL,
                    "expert_slot_count": receipt.HOT_TOTAL,
                    "expert_slot_capacity_bytes": receipt.HOT_CAPACITY_BYTES,
                    "expert_file_mapped_slot_count": receipt.CANONICAL_TOTAL,
                    "expert_file_mapped_address_span_bytes": receipt.MAPPED_COLD_SPAN_BYTES,
                    "expert_table_directory_slot_count": receipt.HOT_TOTAL,
                    "expert_table_directory_capacity_bytes": receipt.HOT_CAPACITY_BYTES,
                    "expert_table_bound_active_slot_count": receipt.LAYERS * 8,
                    "expert_tables_compute_bound": True,
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
                    "arbitrary_slot_prewarm_skipped": True,
                },
            }
            report_path = lane / "load-only-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = receipt.validate_load_only(lane)
            self.assertEqual(result["assistant_residency"]["locked_bytes"], 1024)

            report["target_final_ledger"]["expert_file_mapped_slot_count"] -= 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(receipt.ReceiptError):
                receipt.validate_load_only(lane)


class RunnerPreflightTests(unittest.TestCase):
    def test_request_fixtures_explicitly_opt_in_to_hybrid_receipts(self) -> None:
        for name, tokens in (("request-9.json", 9), ("request-48.json", 48)):
            request_path = RUNNER_DIR / name
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertIs(request.get("camelid_receipt"), True, name)
            self.assertIs(request.get("stream"), False, name)
            self.assertEqual(request.get("temperature"), 0, name)
            self.assertEqual(request.get("top_k"), 1, name)
            self.assertEqual(request.get("seed"), 0, name)
            self.assertIs(request.get("camelid_enable_thinking"), False, name)
            self.assertEqual(request.get("max_tokens"), tokens, name)
            self.assertEqual(
                receipt.sha256_regular_file(request_path),
                receipt.EXPECTED_REQUEST_SHA256[tokens],
                name,
            )

    def run_stage(self, stage: str, root: Path, extra: dict[str, str] | None = None):
        model = root / "model.gguf"
        cghost = root / "model.cghost"
        assistant = root / "assistant.safetensors"
        server = root / "camelid"
        load = root / "load-test"
        cam_lock = root / "cam-lock.sh"
        for path in (model, cghost, assistant):
            path.write_bytes(b"fixture")
        for path in (server, load, cam_lock):
            path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            path.chmod(0o700)
        environment = os.environ.copy()
        environment.update(
            {
                "CAMELID_HYBRID_RECEIPT_ROOT": str(root),
                "CAMELID_HYBRID_SERVER_BINARY": str(server),
                "CAMELID_HYBRID_LOAD_BINARY": str(load),
                "CAMELID_HYBRID_MODEL": str(model),
                "CAMELID_HYBRID_CGHOST": str(cghost),
                "CAMELID_HYBRID_ASSISTANT": str(assistant),
                "CAMELID_CAM_LOCK": str(cam_lock),
            }
        )
        if extra:
            environment.update(extra)
        return subprocess.run(
            ["/bin/zsh", str(RUNNER_PATH), stage],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_shell_parses_and_contains_no_unused_zero_controls(self) -> None:
        subprocess.run(["/bin/zsh", "-n", str(RUNNER_PATH)], check=True)
        source = RUNNER_PATH.read_text(encoding="utf-8")
        for key in (
            "CAMELID_STARTUP_WARMUP=",
            "CAMELID_GEMMA4_MTP_PIN=",
            "CAMELID_GEMMA4_PREFILL_SLOT_RESERVE=",
            "CAMELID_GEMMA4_MTP_OUTER_PIPELINE=",
            "CAMELID_GEMMA4_MTP_PREFETCH=",
            "CAMELID_GEMMA4_OPTION_B=",
        ):
            self.assertNotIn(key, source)
        self.assertIn("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32", source)
        self.assertIn("CAMELID_GEMMA4_SLOT_PIN=0", source)
        self.assertIn("CAMELID_GEMMA4_CHAINED_K1=1", source)
        self.assertIn('lane_dir="$receipt_root/02-smoke-9t/k8"', source)
        self.assertIn('lane_dir="$receipt_root/02-smoke-9t/k1"', source)
        self.assertNotIn("02-smoke-8t", source)
        self.assertNotIn("request-8.json", source)

    def test_inherited_legacy_physical_key_refuses_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.run_stage(
                "load-only",
                root,
                {"CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER": "32"},
            )
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertIn("legacy physical-slot", result.stderr)
            self.assertFalse((root / "01-load-only").exists())

    def test_missing_predecessor_never_creates_or_starts_k1(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.run_stage("smoke-k1", root)
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertIn("predecessor PASS", result.stderr)
            self.assertFalse((root / "02-smoke-9t/k1").exists())

    def test_missing_load_integration_contract_refuses_before_lane_creation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.run_stage("load-only", root)
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertIn("integration contract is absent", result.stderr)
            self.assertFalse((root / "01-load-only").exists())


if __name__ == "__main__":
    unittest.main()
