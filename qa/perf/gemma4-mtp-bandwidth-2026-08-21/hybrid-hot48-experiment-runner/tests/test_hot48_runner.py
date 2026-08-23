from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ANALYZER_PATH = BASE_DIR / "analyze_hot48.py"
RUNNER_PATH = BASE_DIR / "run_hot48.zsh"
REQUEST_PATH = BASE_DIR / "request-48.json"
EXPECTED_IDS_PATH = BASE_DIR / "expected-48-token-ids.json"

SPEC = importlib.util.spec_from_file_location("hot48_analyzer", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
hot48 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hot48)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


class SyntheticRun:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(mode=0o700)
        self.source_commit = "a" * 40
        self.ids = json.loads(EXPECTED_IDS_PATH.read_text(encoding="utf-8"))
        shutil.copyfile(REQUEST_PATH, self.root / "request.json")
        shutil.copyfile(EXPECTED_IDS_PATH, self.root / "expected-token-ids.json")

        self.inputs: dict[str, Path] = {}
        for label in ("binary", "model", "cghost", "assistant", "runner", "analyzer", "watchdog"):
            path = self.root.parent / f"{self.root.name}-{label}.bin"
            path.write_bytes(f"synthetic-{label}\n".encode())
            self.inputs[label] = path
        os.chmod(self.inputs["binary"], 0o700)

        self.response = self._response()
        write_json(self.root / "response.json", self.response)
        write_json(self.root / "health.json", self._health())
        self.intent = self._intent()
        write_json(self.root / "intent.json", self.intent)
        self.events = self._events()
        self.write_events()
        (self.root / "server.log").write_text(self._server_log(), encoding="utf-8")
        write_json(
            self.root / "port-clear.json",
            {
                "schema_version": 1,
                "ports": {"8181": {"clear": True}, "8189": {"clear": True}},
            },
        )

    @staticmethod
    def host(*, headroom: int, swapins: int = 100, swapped: int = 4096) -> dict[str, int]:
        return {
            "pressure_level_raw": 1,
            "reclaimable_headroom_bytes": headroom,
            "wired_bytes": 4 * 1024**3,
            "swapins_pages": swapins,
            "swapouts_pages": 25,
            "swapped_pages_current": swapped,
            "sample_duration_ns": 1_000_000,
        }

    def _environment(self) -> dict[str, str]:
        environment = dict(hot48.EXPECTED_ENVIRONMENT)
        environment["CAMELID_GEMMA4_MTP_ASSISTANT_PATH"] = str(self.inputs["assistant"])
        return environment

    def _health(self) -> dict[str, object]:
        return {
            "build": "v0.6.1-999-gaaaaaaaaa",
            "source_commit": self.source_commit,
            "generation_ready": True,
            "gemma4_serve_lane": "ghost_moe",
            "gemma4_ghost_execution_mode": "full_common_metal",
            "gemma4_ghost_common_metal_active": True,
            "gemma4_ghost_experts_metal_active": True,
            "gemma4_ghost_head_metal_active": True,
            "gemma4_mtp_assistant_loaded": True,
            "gemma4_mtp_full_q4_active": True,
        }

    def _intent(self) -> dict[str, object]:
        artifacts = {label: artifact(path) for label, path in self.inputs.items()}
        artifacts.update(
            {
                "request_source": artifact(REQUEST_PATH),
                "request_frozen": artifact(self.root / "request.json"),
                "expected_ids_source": artifact(EXPECTED_IDS_PATH),
                "expected_ids_frozen": artifact(self.root / "expected-token-ids.json"),
            }
        )
        return {
            "schema_version": 1,
            "benchmark": "gemma4-uniform-hot48-experiment",
            "source_commit": self.source_commit,
            "source_worktree_clean": True,
            "nonce": "synthetic",
            "boot_identity": "synthetic",
            "expected_tokens": 48,
            "port": 8189,
            "protected_port": 8181,
            "disk": {
                "available_kib": 24 * 1024 * 1024,
                "minimum_available_kib": 20 * 1024 * 1024,
                "volume": "/System/Volumes/Data",
            },
            "geometry": {
                "layers": 30,
                "logical_slots_per_layer": 128,
                "hot_slots_per_layer": 48,
                "anonymous_hot_capacity_slots": 1440,
                "anonymous_hot_capacity_bytes": 4_836_556_800,
                "file_mapped_addressable_slots": 3840,
                "file_mapped_address_span_bytes": 12_897_484_800,
                "overflow_slots": 0,
                "victim_slots": 0,
                "host_cache_budget_bytes": 0,
            },
            "watchdog_contract": {
                "schema_version": 3,
                "sample_period_ns": 250_000_000,
                "baseline_soak_seconds": 60,
                "minimum_baseline_reclaimable_headroom_bytes": 8 * 1024**3,
                "minimum_runtime_reclaimable_headroom_bytes": 2 * 1024**3,
                "maximum_child_physical_footprint_bytes": 7_680 * 1024**2,
                "maximum_host_wired_bytes": 8 * 1024**3,
                "reject_swapin_growth": True,
                "require_zero_current_swap": False,
            },
            "artifacts": artifacts,
        }

    def _events(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = [
            {
                "schema_version": 3,
                "event": "clean_parent_baseline",
                "sequence": 0,
                "host": self.host(headroom=9 * 1024**3),
                "violations": [],
            }
        ]
        scheduled = 1_000_000_000
        for index in range(240):
            events.append(
                {
                    "schema_version": 3,
                    "event": "baseline_soak_sample",
                    "sequence": index + 1,
                    "scheduled_monotonic_ns": scheduled + index * 250_000_000,
                    "schedule_lateness_ns": 1_000,
                    "telemetry_duration_ns": 1_000_000,
                    "host": self.host(headroom=9 * 1024**3),
                    "violations": [],
                }
            )
        events.extend(
            [
                {
                    "schema_version": 3,
                    "event": "baseline_soak_complete",
                    "sequence": 240,
                    "required_duration_seconds": 60,
                    "observed_duration_ns": 60_000_000_001,
                    "minimum_reclaimable_headroom_bytes": 8 * 1024**3,
                    "require_zero_current_swap": False,
                },
                {
                    "schema_version": 3,
                    "event": "child_started",
                    "sequence": 240,
                    "pid": 12345,
                    "process_group": 12345,
                    "process_accounting_scope": "isolated_process_group_aggregate",
                    "sample_period_ns": 250_000_000,
                    "minimum_reclaimable_headroom_bytes": 2 * 1024**3,
                    "maximum_child_physical_footprint_bytes": 7_680 * 1024**2,
                    "maximum_host_wired_bytes": 8 * 1024**3,
                    "require_zero_current_swap": False,
                    "reject_swapin_growth": True,
                    "baseline_swapins_pages": 100,
                    "baseline_swapouts_pages": 25,
                    "report_producer": "external",
                    "experiment_environment": self._environment(),
                    "report": str(self.root / "response.json"),
                },
                {
                    "schema_version": 3,
                    "event": "sample",
                    "sequence": 241,
                    "host": self.host(headroom=3 * 1024**3),
                    "process": {
                        "rss_bytes": 5 * 1024**3,
                        "physical_footprint_bytes": 6 * 1024**3,
                    },
                    "violations": [],
                },
                {
                    "schema_version": 3,
                    "event": "post_exit_sample",
                    "sequence": 242,
                    "host": self.host(headroom=3 * 1024**3),
                    "violations": [],
                },
                {
                    "schema_version": 3,
                    "event": "final",
                    "sequence": 243,
                    "pid": 12345,
                    "child_returncode": 0,
                    "watchdog_aborted": False,
                    "abort_reasons": [],
                    "minimum_free_bytes_observed": 1024**3,
                    "minimum_reclaimable_headroom_bytes_observed": 3 * 1024**3,
                    "peak_child_rss_bytes": 5 * 1024**3,
                    "peak_child_physical_footprint_bytes": 6 * 1024**3,
                    "peak_host_wired_bytes": 4 * 1024**3,
                    "baseline_swapins_pages": 100,
                    "baseline_swapouts_pages": 25,
                    "process_group": 12345,
                    "process_group_empty": True,
                    "process_accounting_scope": "isolated_process_group_aggregate",
                    "report_exists": True,
                    "report_size_bytes": (self.root / "response.json").stat().st_size,
                    "report_is_regular_file": True,
                    "report_is_symlink": False,
                },
            ]
        )
        return events

    def _geometry(self) -> dict[str, object]:
        return {
            "layers": 30,
            "record_payload_bytes": 3_345_408,
            "slot_stride_bytes": 3_358_720,
            "logical_addressable_slots": 3840,
            "anonymous_hot_capacity_slots": 1440,
            "anonymous_hot_capacity_bytes": 4_836_556_800,
            "file_mapped_addressable_slots": 3840,
            "file_mapped_address_span_bytes": 12_897_484_800,
            "overflow_slots": 0,
            "overflow_capacity_bytes": 0,
            "victim_record_capacity": 0,
            "victim_capacity_bytes": 0,
            "host_cache_budget_bytes": 0,
            "mapped_readahead_enabled": True,
            "mapped_readahead_max_inflight_records": 64,
            "mapped_readahead_anonymous_capacity_bytes": 0,
            "per_layer": [
                {
                    "layer": layer,
                    "logical_addressable_slots": 128,
                    "anonymous_hot_capacity_slots": 48,
                    "anonymous_hot_capacity_bytes": 161_218_560,
                    "file_mapped_addressable_slots": 128,
                    "file_mapped_address_span_bytes": 429_916_160,
                    "overflow_slots": 0,
                    "victim_slots": 0,
                }
                for layer in range(30)
            ],
        }

    def _response(self) -> dict[str, object]:
        rounds = []
        for index in range(6):
            committed = self.ids[index * 8 : (index + 1) * 8]
            rounds.append(
                {
                    "round_index": index,
                    "chained_round_sequence": index + 1,
                    "prefix_tokens_before": index * 8,
                    "bootstrap": False,
                    "remaining_budget_before": 48 - index * 8,
                    "k": 8,
                    "requested_k": 8,
                    "proposed_k": 7,
                    "verifier_k": 8,
                    "budget_truncated": False,
                    "success": True,
                    "accepted_drafts": 7,
                    "useful_accepted_drafts": 7,
                    "committed_tokens": committed,
                    "assistant_exposed_ms": 10.0,
                    "assistant_gpu_ms": 8.0,
                    "verifier_wall_ms": 80.0,
                    "verifier_gpu_ms": 70.0,
                    "receipt_round_wall_ms": 100.0,
                    "selected_dropped": 0,
                    "missing_failclose": 0,
                    "slot_capacity_overflow": 0,
                    "overflow_slots": 0,
                    "overflow_bytes": 0,
                    "overflow_layers": 0,
                    "overflow_experts": 0,
                    "victim_hits": 0,
                    "mapped_readahead_enqueued_records": 10,
                    "mapped_readahead_enqueued_bytes": 33_454_080,
                    "mapped_readahead_enqueue_ms": 0.1,
                    "mapped_readahead_previous_union_enqueued_records": 10,
                    "mapped_readahead_previous_union_enqueued_bytes": 33_454_080,
                    "mapped_readahead_previous_union_enqueue_ms": 0.1,
                    "per_layer": [
                        {
                            "layer_index": layer,
                            "active_unique": 8,
                            "hot_bound": 8,
                            "mapped_bound": 0,
                            "bound_records": 8,
                        }
                        for layer in range(30)
                    ],
                }
            )
        return {
            "usage": {"completion_tokens": 48},
            "choices": [{"finish_reason": "length", "message": {"content": "synthetic"}}],
            "camelid": {
                "generated_token_ids": self.ids,
                "hybrid_telemetry": {
                    "schema_version": 2,
                    "scope": "single_completed_measured_request",
                    "geometry": self._geometry(),
                    "route_interval": {
                        "scope": "measured_request_prefill_plus_generation",
                        "epoch": 1,
                        "routed_expert_ids_per_layer": [[] for _ in range(30)],
                        "routed_unique_per_layer": [0 for _ in range(30)],
                        "routed_unique_experts_sum": 0,
                        "routed_unique_experts_max": 0,
                    },
                    "rounds": rounds,
                    "aggregate": {
                        "scope": "request_delta",
                        "host_fills": 0,
                        "direct_read_failures": 0,
                        "overflow_experts": 0,
                        "victim_hits": 0,
                    },
                    "metrics": {
                        "response_completion_tokens": 48,
                        "proposed_drafts": 42,
                        "accepted_drafts": 42,
                        "outer_lookahead_nonzero_count": 0,
                        "receipt_round_wall_ms": 600.0,
                        "decode_tokens_per_second": 80.0,
                    },
                },
            },
        }

    def _server_log(self) -> str:
        assistant_sha = sha256(self.inputs["assistant"])
        lines = [
            "[gemma4-ghost-metal] HYBRID HOT48 EXPERIMENT ACTIVE: "
            "CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1 "
            "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32 "
            "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT48_EXPERIMENT=1, "
            "layers=30 canonical_addressable=128/layer physical_hot=48/layer "
            "hot_capacity_slots=1440 hot_capacity_bytes=4836556800 "
            "mapped_cold_span_bytes=12897484800 overflow=0 victim=0 "
            "slot_pin=off prediction=off",
            f"[gemma4-mtp full-q4] enabled=true source_sha256={assistant_sha} "
            "matrices=23 packed_bytes=236077056 bf16_matrix_bytes=839385088 "
            "quantize_us=123 norms_quantized=false fallback=false",
            hot48.FULL_Q4_RESIDENCY_MARKER,
        ]
        for _ in range(6):
            lines.append(
                "[gemma4-mtp device-chain] requested_drafts=7 returned_drafts=7 "
                "command_buffers=1 commits=1 waits=1 cpu_embedding_callbacks=0 "
                "linear_format=q4_0_all matrix_bytes_per_draft=236077056 "
                "encode_us=10 wait_us=20 gpu_us=30 kernel_us=25 wall_us=40"
            )
            lines.append(
                "[metal chained stages] split=per-cb qkv_o=10.0ms attn=5.0ms "
                "router=2.0ms shared=3.0ms gateup=20.0ms down=15.0ms "
                "resid=1.0ms gpu_total=56.0ms"
            )
        return "\n".join(lines) + "\n"

    def write_events(self) -> None:
        with (self.root / "watchdog.jsonl").open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def write_response(self) -> None:
        write_json(self.root / "response.json", self.response)


class Hot48AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.run = SyntheticRun(self.base / "run")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_receipt_allows_existing_swap_and_reports_measurements(self) -> None:
        verdict = hot48.analyze(self.run.root)
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["geometry"]["anonymous_hot_capacity_slots"], 1440)
        self.assertAlmostEqual(
            verdict["performance"]["effective_decode_tokens_per_second"], 80.0
        )
        self.assertEqual(verdict["memory"]["baseline_swapped_pages_current"], 4096)
        self.assertEqual(verdict["memory"]["swapins_growth_pages"], 0)
        self.assertEqual(verdict["stages"]["receipt_count"], 6)

    def test_rejects_one_layer_with_47_hot_slots(self) -> None:
        geometry = self.run.response["camelid"]["hybrid_telemetry"]["geometry"]
        geometry["per_layer"][9]["anonymous_hot_capacity_slots"] = 47
        self.run.write_response()
        with self.assertRaisesRegex(hot48.ReceiptError, "layer 9"):
            hot48.analyze(self.run.root)

    def test_rejects_wrong_token_id(self) -> None:
        self.run.response["camelid"]["generated_token_ids"][3] += 1
        self.run.write_response()
        with self.assertRaisesRegex(hot48.ReceiptError, "exact token IDs"):
            hot48.analyze(self.run.root)

    def test_rejects_round_safety_counter(self) -> None:
        rounds = self.run.response["camelid"]["hybrid_telemetry"]["rounds"]
        rounds[2]["missing_failclose"] = 1
        self.run.write_response()
        with self.assertRaisesRegex(hot48.ReceiptError, "missing_failclose"):
            hot48.analyze(self.run.root)

    def test_rejects_watchdog_safety_abort(self) -> None:
        self.run.events.insert(
            -1,
            {
                "schema_version": 3,
                "event": "watchdog_abort",
                "sequence": 242,
                "violations": ["synthetic"],
            },
        )
        self.run.write_events()
        with self.assertRaisesRegex(hot48.ReceiptError, "refusal/abort"):
            hot48.analyze(self.run.root)

    def test_rejects_swapin_growth_but_not_current_swap(self) -> None:
        sample = next(event for event in self.run.events if event["event"] == "sample")
        sample["host"]["swapins_pages"] = 101
        self.run.write_events()
        with self.assertRaisesRegex(hot48.ReceiptError, "swap growth"):
            hot48.analyze(self.run.root)

    def test_rejects_per_layer_override_in_child_environment(self) -> None:
        started = next(event for event in self.run.events if event["event"] == "child_started")
        started["experiment_environment"][
            "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER"
        ] = ",".join(["48"] * 30)
        self.run.write_events()
        with self.assertRaisesRegex(hot48.ReceiptError, "per-layer"):
            hot48.analyze(self.run.root)

    def test_rejects_dirty_binary_build(self) -> None:
        health = self.run._health()
        health["build"] = "v0.6.1-999-gaaaaaaaaa-dirty"
        write_json(self.run.root / "health.json", health)
        with self.assertRaisesRegex(hot48.ReceiptError, "readiness"):
            hot48.analyze(self.run.root)


class Hot48SourceContractTests(unittest.TestCase):
    def test_frozen_fixture_hashes(self) -> None:
        self.assertEqual(sha256(REQUEST_PATH), hot48.REQUEST_SHA256)
        self.assertEqual(sha256(EXPECTED_IDS_PATH), hot48.EXPECTED_IDS_SHA256)
        self.assertEqual(len(json.loads(EXPECTED_IDS_PATH.read_text())), 48)

    def test_runner_is_exact_and_does_not_reintroduce_zero_swap_gate(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT48_EXPERIMENT=1", source)
        self.assertIn("CAMELID_GEMMA4_MTP_DEVICE_CHAIN=1", source)
        self.assertIn("CAMELID_GEMMA4_MTP_FULL_Q4=1", source)
        self.assertIn("--baseline-soak-seconds 60", source)
        self.assertIn("--reject-swapin-growth", source)
        self.assertNotIn("--require-zero-current-swap", source)
        self.assertIn("readonly PORT=8189", source)
        self.assertIn("readonly PROTECTED_PORT=8181", source)
        self.assertIn("MIN_DISK_AVAILABLE_KIB=20971520", source)
        self.assertIn("parameters[CAMELID_HOT48_BINARY]", source)
        self.assertNotIn("target/release/camelid", source)
        self.assertIn("data_volume_device", source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertIn("status --porcelain=v1 --untracked-files=all", source)
        self.assertIn('endswith("-dirty")', source)
        for key in hot48.PER_LAYER_ENVIRONMENT:
            self.assertIn(key, source)
            self.assertNotIn(f"{key}=", source)

    def test_sources_parse(self) -> None:
        subprocess.run(["/bin/zsh", "-n", str(RUNNER_PATH)], check=True)
        with tempfile.TemporaryDirectory() as directory:
            py_compile.compile(
                str(ANALYZER_PATH), cfile=str(Path(directory) / "analyzer.pyc"), doraise=True
            )


if __name__ == "__main__":
    unittest.main()
