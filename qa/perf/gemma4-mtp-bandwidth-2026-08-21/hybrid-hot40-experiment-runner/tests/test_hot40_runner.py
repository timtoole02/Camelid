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
ANALYZER_PATH = BASE_DIR / "analyze_hot40.py"
HOST_SAMPLER_PATH = BASE_DIR / "capture_host_memory.py"
HASHER_PATH = BASE_DIR / "sha256_nocache.py"
RUNNER_PATH = BASE_DIR / "run_hot40.zsh"
REQUEST_PATH = BASE_DIR / "request-48.json"
EXPECTED_IDS_PATH = BASE_DIR / "expected-48-token-ids.json"
WATCHDOG_PATH = (
    BASE_DIR.parents[2]
    / "evidence-bundles/gemma4-26b-mtp-assistant-oracle/run_load_only_watchdog.py"
)

SPEC = importlib.util.spec_from_file_location("hot40_analyzer", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
hot40 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hot40)

WATCHDOG_SPEC = importlib.util.spec_from_file_location("hot40_watchdog", WATCHDOG_PATH)
assert WATCHDOG_SPEC is not None and WATCHDOG_SPEC.loader is not None
watchdog_module = importlib.util.module_from_spec(WATCHDOG_SPEC)
WATCHDOG_SPEC.loader.exec_module(watchdog_module)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "verification": "sha256-this-run",
    }


def large_input(path: Path) -> dict[str, object]:
    identity = path.stat()
    return {
        "path": str(path),
        "preverified_sha256": sha256(path),
        "size_bytes": identity.st_size,
        "stat": {
            "device": identity.st_dev,
            "inode": identity.st_ino,
            "mtime_seconds": int(identity.st_mtime),
        },
        "verification": "historical-sha256-plus-live-stat-no-run-content-read",
        "content_read_before_spawn": False,
    }


class SyntheticRun:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(mode=0o700)
        self.binary_source_commit = "a" * 40
        self.harness_commit = "b" * 40
        self.source_commit = self.binary_source_commit
        self.ids = json.loads(EXPECTED_IDS_PATH.read_text(encoding="utf-8"))
        shutil.copyfile(REQUEST_PATH, self.root / "request.json")
        shutil.copyfile(EXPECTED_IDS_PATH, self.root / "expected-token-ids.json")

        self.inputs: dict[str, Path] = {}
        for label in (
            "binary",
            "model",
            "cghost",
            "assistant",
            "runner",
            "analyzer",
            "host_sampler",
            "nocache_hasher",
            "watchdog",
        ):
            path = self.root.parent / f"{self.root.name}-{label}.bin"
            path.write_bytes(f"synthetic-{label}\n".encode())
            self.inputs[label] = path
        os.chmod(self.inputs["binary"], 0o700)

        self.hashing_memory_before = {
            "schema_version": 1,
            "telemetry_source": "run_load_only_watchdog.NativeTelemetry.sample_host",
            "boot_identity": "synthetic",
            "host": self.host(headroom=9 * 1024**3),
        }
        self.hashing_memory_before["host"]["sample_started_monotonic_ns"] = 100
        self.hashing_memory_before["host"]["observed_monotonic_ns"] = 110
        self.hashing_memory_after = copy.deepcopy(self.hashing_memory_before)
        self.hashing_memory_after["host"]["sample_started_monotonic_ns"] = 200
        self.hashing_memory_after["host"]["observed_monotonic_ns"] = 210
        write_json(self.root / "hashing-memory-before.json", self.hashing_memory_before)
        write_json(self.root / "hashing-memory-after.json", self.hashing_memory_after)

        self.response = self._response()
        write_json(self.root / "response.json", self.response)
        write_json(self.root / "health.json", self._health())
        self.intent = self._intent()
        write_json(self.root / "intent.json", self.intent)
        self.events = self._events()
        self.write_events()
        self.server_log = self._server_log()
        self.write_server_log()
        write_json(
            self.root / "port-clear.json",
            {
                "schema_version": 1,
                "ports": {
                    "8181": {
                        "clear": True,
                        "policy": "never-bound-connected-or-signaled",
                    },
                    "8189": {"clear": True, "policy": "benchmark-only"},
                },
            },
        )

    @staticmethod
    def host(
        *,
        headroom: int,
        swapins: int = 100,
        swapouts: int = 25,
        swapped: int = 4096,
    ) -> dict[str, int]:
        return {
            "sample_started_monotonic_ns": 1,
            "observed_monotonic_ns": 2,
            "sample_duration_ns": 1_000_000,
            "unix_time_ns": 1,
            "page_size_bytes": 16_384,
            "free_bytes_strict": headroom // 2,
            "active_bytes": 3 * 1024**3,
            "inactive_bytes": headroom // 2,
            "pressure_level_raw": 1,
            "reclaimable_headroom_bytes": headroom,
            "wired_bytes": 4 * 1024**3,
            "compressor_occupied_bytes": 1024**3,
            "compressed_logical_bytes": 2 * 1024**3,
            "swapins_pages": swapins,
            "swapouts_pages": swapouts,
            "swapped_pages_current": swapped,
        }

    def _environment(self) -> dict[str, str]:
        environment = dict(hot40.EXPECTED_ENVIRONMENT)
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
        artifacts = {
            label: artifact(self.inputs[label])
            for label in (
                "binary",
                "runner",
                "analyzer",
                "host_sampler",
                "nocache_hasher",
                "watchdog",
            )
        }
        artifacts.update(
            {
                "request_source": artifact(REQUEST_PATH),
                "request_frozen": artifact(self.root / "request.json"),
                "expected_ids_source": artifact(EXPECTED_IDS_PATH),
                "expected_ids_frozen": artifact(self.root / "expected-token-ids.json"),
                "hashing_memory_before": artifact(
                    self.root / "hashing-memory-before.json"
                ),
                "hashing_memory_after": artifact(
                    self.root / "hashing-memory-after.json"
                ),
            }
        )
        return {
            "schema_version": 1,
            "benchmark": "gemma4-uniform-hot40-experiment",
            "source_commit": self.source_commit,
            "binary_source_commit": self.binary_source_commit,
            "harness_commit": self.harness_commit,
            "source_worktree_clean": True,
            "harness_worktree_clean": True,
            "binary_source_contract": {
                "environment": "CAMELID_HOT40_BINARY_SOURCE_COMMIT",
                "defaulted_to_harness_commit": False,
                "canonical_full_commit": True,
                "ancestor_of_harness_commit": True,
                "runtime_source_diff_empty": True,
                "runtime_source_paths": [
                    "src",
                    "Cargo.toml",
                    "Cargo.lock",
                    "build.rs",
                ],
            },
            "nonce": "synthetic",
            "boot_identity": "synthetic",
            "expected_tokens": 48,
            "port": 8189,
            "protected_port": 8181,
            "protected_port_policy": "never-bound-connected-or-signaled",
            "allow_warning_pressure": False,
            "disk": {
                "available_kib": 24 * 1024 * 1024,
                "minimum_available_kib": 20 * 1024 * 1024,
                "volume": "/System/Volumes/Data",
            },
            "geometry": {
                "layers": 30,
                "logical_slots_per_layer": 128,
                "hot_slots_per_layer": 40,
                "anonymous_hot_capacity_slots": 1200,
                "anonymous_hot_capacity_bytes": 4_030_464_000,
                "file_mapped_addressable_slots": 3840,
                "file_mapped_address_span_bytes": 12_897_484_800,
                "mapped_readahead_enabled": False,
                "mapped_readahead_max_inflight_records": 0,
                "overflow_slots": 0,
                "victim_slots": 0,
                "host_cache_budget_bytes": 0,
            },
            "watchdog_contract": {
                "schema_version": 3,
                "sample_period_ns": 250_000_000,
                "baseline_soak_seconds": 60,
                "minimum_baseline_reclaimable_headroom_bytes": 7_680 * 1024**2,
                "minimum_runtime_reclaimable_headroom_bytes": 2 * 1024**3,
                "maximum_child_physical_footprint_bytes": 7_680 * 1024**2,
                "maximum_host_wired_bytes": 8 * 1024**3,
                "maximum_pressure_level_raw": 1,
                "reject_swapin_growth": True,
                "reject_swapout_growth": True,
                "require_zero_current_swap": False,
            },
            "hashing_contract": {
                "schema_version": 1,
                "algorithm": "sha256",
                "platform": "darwin",
                "f_rdahead_command": 45,
                "f_rdahead_value": 0,
                "f_nocache_command": 48,
                "f_nocache_value": 1,
                "read_chunk_bytes": 4 * 1024 * 1024,
                "post_hash_cooldown_seconds": 0,
                "helper_artifact_label": "nocache_hasher",
                "host_sampler_artifact_label": "host_sampler",
                "telemetry_watchdog_artifact_label": "watchdog",
                "telemetry_source": "run_load_only_watchdog.NativeTelemetry.sample_host",
                "minimum_pre_hash_reclaimable_headroom_bytes": 7_680 * 1024**2,
                "minimum_post_hash_reclaimable_headroom_bytes": 7_680 * 1024**2,
                "maximum_host_wired_bytes": 8 * 1024**3,
                "allow_warning_pressure": False,
                "maximum_pressure_level_raw": 1,
                "reject_swapin_growth": True,
                "reject_swapout_growth": True,
                "large_inputs_content_hashed_this_run": False,
                "large_inputs_content_read_before_spawn": False,
                "large_input_binding": "historical-sha256-plus-live-stat",
                "provenance_reference": "synthetic prior receipt",
                "provenance_reference_sha256": "",
                "provenance_limitation": hot40.PROVENANCE_LIMITATION,
                "memory_before": self.hashing_memory_before,
                "memory_after": self.hashing_memory_after,
            },
            "large_inputs": {
                label: large_input(self.inputs[label])
                for label in ("model", "cghost", "assistant")
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
                    "minimum_reclaimable_headroom_bytes": 7_680 * 1024**2,
                    "require_zero_current_swap": False,
                    "maximum_pressure_level_raw": 1,
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
                    "maximum_pressure_level_raw": 1,
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
            "anonymous_hot_capacity_slots": 1200,
            "anonymous_hot_capacity_bytes": 4_030_464_000,
            "file_mapped_addressable_slots": 3840,
            "file_mapped_address_span_bytes": 12_897_484_800,
            "overflow_slots": 0,
            "overflow_capacity_bytes": 0,
            "victim_record_capacity": 0,
            "victim_capacity_bytes": 0,
            "host_cache_budget_bytes": 0,
            "mapped_readahead_enabled": False,
            "mapped_readahead_max_inflight_records": 0,
            "mapped_readahead_anonymous_capacity_bytes": 0,
            "per_layer": [
                {
                    "layer": layer,
                    "logical_addressable_slots": 128,
                    "anonymous_hot_capacity_slots": 40,
                    "anonymous_hot_capacity_bytes": 134_348_800,
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
                    "mapped_readahead_enqueued_records": 0,
                    "mapped_readahead_enqueued_bytes": 0,
                    "mapped_readahead_enqueue_ms": 0.0,
                    "mapped_readahead_previous_union_enqueued_records": 0,
                    "mapped_readahead_previous_union_enqueued_bytes": 0,
                    "mapped_readahead_previous_union_enqueue_ms": 0.0,
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
        assistant_sha = self.intent["large_inputs"]["assistant"]["preverified_sha256"]
        lines = [
            "[gemma4-ghost-metal] HYBRID HOT40 EXPERIMENT ACTIVE: "
            "CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS=1 "
            "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32 "
            "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT40_EXPERIMENT=1, "
            "layers=30 canonical_addressable=128/layer physical_hot=40/layer "
            "hot_capacity_slots=1200 hot_capacity_bytes=4030464000 "
            "mapped_cold_span_bytes=12897484800 verifier_k=1..8 "
            "overflow=0 victim=0 slot_pin=off prediction=off",
            f"[gemma4-mtp full-q4] enabled=true source_sha256={assistant_sha} "
            "matrices=23 packed_bytes=236077056 bf16_matrix_bytes=839385088 "
            "quantize_us=123 norms_quantized=false fallback=false",
            hot40.FULL_Q4_RESIDENCY_MARKER,
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

    def enable_warning_pressure(self) -> None:
        self.intent["allow_warning_pressure"] = True
        watchdog = self.intent["watchdog_contract"]
        watchdog["maximum_pressure_level_raw"] = 2
        watchdog["minimum_baseline_reclaimable_headroom_bytes"] = 4_608 * 1024**2
        hashing = self.intent["hashing_contract"]
        hashing["allow_warning_pressure"] = True
        hashing["maximum_pressure_level_raw"] = 2
        hashing["minimum_pre_hash_reclaimable_headroom_bytes"] = 4_608 * 1024**2
        hashing["minimum_post_hash_reclaimable_headroom_bytes"] = 4_608 * 1024**2
        for sample in (self.hashing_memory_before, self.hashing_memory_after):
            sample["host"]["pressure_level_raw"] = 2
        write_json(self.root / "hashing-memory-before.json", self.hashing_memory_before)
        write_json(self.root / "hashing-memory-after.json", self.hashing_memory_after)
        self.intent["artifacts"]["hashing_memory_before"] = artifact(
            self.root / "hashing-memory-before.json"
        )
        self.intent["artifacts"]["hashing_memory_after"] = artifact(
            self.root / "hashing-memory-after.json"
        )
        for event in self.events:
            if isinstance(event.get("host"), dict):
                event["host"]["pressure_level_raw"] = 2
            if event["event"] == "baseline_soak_complete":
                event["minimum_reclaimable_headroom_bytes"] = 4_608 * 1024**2
                event["maximum_pressure_level_raw"] = 2
            if event["event"] == "child_started":
                event["maximum_pressure_level_raw"] = 2
        self.write_intent()
        self.write_events()

    def write_intent(self) -> None:
        write_json(self.root / "intent.json", self.intent)

    def write_server_log(self) -> None:
        (self.root / "server.log").write_text(self.server_log, encoding="utf-8")


class Hot40AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.run = SyntheticRun(self.base / "run")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_receipt_reports_exact_hot40_measurement(self) -> None:
        verdict = hot40.analyze(self.run.root)
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["geometry"]["anonymous_hot_capacity_slots"], 1200)
        self.assertEqual(verdict["geometry"]["anonymous_hot_capacity_bytes"], 4_030_464_000)
        self.assertAlmostEqual(
            verdict["performance"]["effective_decode_tokens_per_second"], 80.0
        )
        self.assertFalse(
            verdict["performance"]["throughput_contaminated_by_sparse_predict_probe"]
        )
        self.assertEqual(verdict["memory"]["baseline_swapped_pages_current"], 4096)
        self.assertEqual(verdict["memory"]["swapins_growth_pages"], 0)
        self.assertFalse(verdict["hashing"]["large_inputs_content_hashed_this_run"])
        self.assertFalse(verdict["hashing"]["large_inputs_content_read_before_spawn"])
        self.assertEqual(verdict["stages"]["receipt_count"], 6)
        self.assertEqual(verdict["source_commit"], self.run.binary_source_commit)
        self.assertEqual(
            verdict["binary_source_commit"], self.run.binary_source_commit
        )
        self.assertEqual(verdict["harness_commit"], self.run.harness_commit)
        self.assertTrue(
            verdict["gates"]["runtime_source_diff_empty_between_binary_and_harness"]
        )
        self.assertFalse(verdict["pressure_policy"]["allow_warning_pressure"])
        self.assertEqual(verdict["pressure_policy"]["maximum_pressure_level_raw"], 1)

    def test_explicit_warning_pressure_receipt_passes_at_four_and_a_half_gib_baseline(self) -> None:
        self.run.enable_warning_pressure()
        verdict = hot40.analyze(self.run.root)
        self.assertTrue(verdict["pass"])
        self.assertTrue(verdict["pressure_policy"]["allow_warning_pressure"])
        self.assertEqual(verdict["pressure_policy"]["maximum_pressure_level_raw"], 2)
        self.assertEqual(
            verdict["pressure_policy"]["maximum_pressure_level_raw_observed"], 2
        )

    def test_default_policy_rejects_warning_pressure(self) -> None:
        self.run.hashing_memory_before["host"]["pressure_level_raw"] = 2
        write_json(
            self.run.root / "hashing-memory-before.json", self.run.hashing_memory_before
        )
        self.run.intent["hashing_contract"]["memory_before"] = (
            self.run.hashing_memory_before
        )
        self.run.intent["artifacts"]["hashing_memory_before"] = artifact(
            self.run.root / "hashing-memory-before.json"
        )
        self.run.write_intent()
        with self.assertRaisesRegex(hot40.ReceiptError, "hash-phase before"):
            hot40.analyze(self.run.root)

    def test_rejects_mapped_readahead_or_nonzero_counter(self) -> None:
        geometry = self.run.response["camelid"]["hybrid_telemetry"]["geometry"]
        geometry["mapped_readahead_enabled"] = True
        self.run.write_response()
        with self.assertRaisesRegex(hot40.ReceiptError, "geometry"):
            hot40.analyze(self.run.root)

        geometry["mapped_readahead_enabled"] = False
        rounds = self.run.response["camelid"]["hybrid_telemetry"]["rounds"]
        rounds[0]["mapped_readahead_enqueued_records"] = 1
        self.run.write_response()
        with self.assertRaisesRegex(hot40.ReceiptError, "mapped_readahead"):
            hot40.analyze(self.run.root)

    def test_rejects_binary_source_alias_or_contract_drift(self) -> None:
        self.run.intent["source_commit"] = "c" * 40
        self.run.write_intent()
        with self.assertRaisesRegex(hot40.ReceiptError, "exact Hot40"):
            hot40.analyze(self.run.root)

        self.run.intent["source_commit"] = self.run.binary_source_commit
        self.run.intent["binary_source_contract"]["runtime_source_diff_empty"] = False
        self.run.write_intent()
        with self.assertRaisesRegex(hot40.ReceiptError, "binary/harness"):
            hot40.analyze(self.run.root)

    def test_health_must_report_binary_not_harness_commit(self) -> None:
        health = self.run._health()
        health["source_commit"] = self.run.harness_commit
        write_json(self.run.root / "health.json", health)
        with self.assertRaisesRegex(hot40.ReceiptError, "readiness"):
            hot40.analyze(self.run.root)

    def test_rejects_one_layer_with_39_hot_slots(self) -> None:
        geometry = self.run.response["camelid"]["hybrid_telemetry"]["geometry"]
        geometry["per_layer"][9]["anonymous_hot_capacity_slots"] = 39
        self.run.write_response()
        with self.assertRaisesRegex(hot40.ReceiptError, "layer 9"):
            hot40.analyze(self.run.root)

    def test_rejects_wrong_token_id(self) -> None:
        self.run.response["camelid"]["generated_token_ids"][3] += 1
        self.run.write_response()
        with self.assertRaisesRegex(hot40.ReceiptError, "exact token IDs"):
            hot40.analyze(self.run.root)

    def test_rejects_round_safety_counter(self) -> None:
        rounds = self.run.response["camelid"]["hybrid_telemetry"]["rounds"]
        rounds[2]["missing_failclose"] = 1
        self.run.write_response()
        with self.assertRaisesRegex(hot40.ReceiptError, "missing_failclose"):
            hot40.analyze(self.run.root)

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
        with self.assertRaisesRegex(hot40.ReceiptError, "refusal/abort"):
            hot40.analyze(self.run.root)

    def test_rejects_runtime_swapin_or_swapout_growth(self) -> None:
        sample = next(event for event in self.run.events if event["event"] == "sample")
        sample["host"]["swapouts_pages"] = 26
        self.run.write_events()
        with self.assertRaisesRegex(hot40.ReceiptError, "swap growth"):
            hot40.analyze(self.run.root)

    def test_rejects_swap_change_between_preflight_and_watchdog(self) -> None:
        clean = next(
            event for event in self.run.events if event["event"] == "clean_parent_baseline"
        )
        clean["host"]["swapins_pages"] = 101
        for event in self.run.events:
            if isinstance(event.get("host"), dict):
                event["host"]["swapins_pages"] = 101
        started = next(
            event for event in self.run.events if event["event"] == "child_started"
        )
        started["baseline_swapins_pages"] = 101
        self.run.write_events()
        with self.assertRaisesRegex(hot40.ReceiptError, "between preflight"):
            hot40.analyze(self.run.root)

    def test_rejects_preflight_swap_growth(self) -> None:
        after = self.run.intent["hashing_contract"]["memory_after"]
        after["host"]["swapins_pages"] += 1
        write_json(self.run.root / "hashing-memory-after.json", after)
        self.run.intent["artifacts"]["hashing_memory_after"] = artifact(
            self.run.root / "hashing-memory-after.json"
        )
        self.run.write_intent()
        with self.assertRaisesRegex(hot40.ReceiptError, "changed swap"):
            hot40.analyze(self.run.root)

    def test_rejects_large_input_live_stat_drift_without_hashing_it(self) -> None:
        with self.inputs_open("model", "ab") as handle:
            handle.write(b"drift")
        with self.assertRaisesRegex(hot40.ReceiptError, "live stat identity drifted"):
            hot40.analyze(self.run.root)

    def inputs_open(self, label: str, mode: str):
        return self.run.inputs[label].open(mode)

    def test_rejects_invalid_historical_large_input_hash_metadata(self) -> None:
        self.run.intent["large_inputs"]["assistant"]["preverified_sha256"] = "not-a-sha"
        self.run.write_intent()
        with self.assertRaisesRegex(hot40.ReceiptError, "historical/stat"):
            hot40.analyze(self.run.root)

    def test_rejects_sparse_probe_environment_or_log(self) -> None:
        started = next(
            event for event in self.run.events if event["event"] == "child_started"
        )
        started["experiment_environment"][hot40.SPARSE_PREDICT_PROBE_ENV] = "1"
        self.run.write_events()
        with self.assertRaisesRegex(hot40.ReceiptError, "sparse-predict"):
            hot40.analyze(self.run.root)

        del started["experiment_environment"][hot40.SPARSE_PREDICT_PROBE_ENV]
        self.run.write_events()
        self.run.server_log += "[gemma4 sparse-predict probe] forbidden\n"
        self.run.write_server_log()
        with self.assertRaisesRegex(hot40.ReceiptError, "sparse-predict"):
            hot40.analyze(self.run.root)

    def test_rejects_per_layer_override(self) -> None:
        started = next(
            event for event in self.run.events if event["event"] == "child_started"
        )
        started["experiment_environment"][
            "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER"
        ] = ",".join(["40"] * 30)
        self.run.write_events()
        with self.assertRaisesRegex(hot40.ReceiptError, "per-layer"):
            hot40.analyze(self.run.root)


class Hot40SourceContractTests(unittest.TestCase):
    def test_baseline_gate_is_exactly_seven_and_a_half_gib(self) -> None:
        self.assertEqual(hot40.MIN_BASELINE_HEADROOM_BYTES, 7_680 * 1024**2)
        self.assertEqual(hot40.WARNING_MIN_BASELINE_HEADROOM_BYTES, 4_608 * 1024**2)

    def test_watchdog_warning_opt_in_preserves_hard_memory_and_swap_gates(self) -> None:
        sample = SyntheticRun.host(headroom=5 * 1024**3)
        sample["pressure_level_raw"] = 2
        self.assertTrue(
            any(
                "pressure_level" in reason
                for reason in watchdog_module._host_violation_reasons(sample, 25)
            )
        )
        self.assertEqual(
            watchdog_module._host_violation_reasons(
                sample,
                25,
                baseline_swapins_pages=100,
                minimum_reclaimable_headroom_bytes=4_608 * 1024**2,
                maximum_host_wired_bytes=8 * 1024**3,
                maximum_pressure_level_raw=2,
            ),
            [],
        )
        sample["swapouts_pages"] += 1
        sample["reclaimable_headroom_bytes"] = 2 * 1024**3 - 1
        sample["wired_bytes"] = 8 * 1024**3 + 1
        reasons = watchdog_module._host_violation_reasons(
            sample,
            25,
            baseline_swapins_pages=100,
            minimum_reclaimable_headroom_bytes=2 * 1024**3,
            maximum_host_wired_bytes=8 * 1024**3,
            maximum_pressure_level_raw=2,
        )
        self.assertTrue(any("swapouts_changed" in reason for reason in reasons))
        self.assertTrue(any("reclaimable_headroom" in reason for reason in reasons))
        self.assertTrue(any("host_wired" in reason for reason in reasons))

    def test_frozen_fixture_hashes(self) -> None:
        self.assertEqual(sha256(REQUEST_PATH), hot40.REQUEST_SHA256)
        self.assertEqual(sha256(EXPECTED_IDS_PATH), hot40.EXPECTED_IDS_SHA256)
        self.assertEqual(len(json.loads(EXPECTED_IDS_PATH.read_text())), 48)

    def test_runner_is_prebuilt_exact_hot40_and_never_reads_large_inputs(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT40_EXPERIMENT=1", source)
        self.assertNotIn("CAMELID_GEMMA4_GHOST_METAL_SPARSE_PREDICT_PROBE=1", source)
        self.assertIn("CAMELID_GEMMA4_MTP_DEVICE_CHAIN=1", source)
        self.assertIn("CAMELID_GEMMA4_MTP_FULL_Q4=1", source)
        self.assertNotIn("CAMELID_GEMMA4_MOE_MMA_K8", source)
        self.assertIn("hot_slots_per_layer: 40", source)
        self.assertIn("anonymous_hot_capacity_slots: 1200", source)
        self.assertIn("anonymous_hot_capacity_bytes: 4030464000", source)
        self.assertIn("MIN_BASELINE_HEADROOM_BYTES=8053063680", source)
        self.assertIn("WARNING_MIN_BASELINE_HEADROOM_BYTES=4831838208", source)
        self.assertIn("CAMELID_HOT40_ALLOW_WARNING_PRESSURE", source)
        self.assertIn('--maximum-pressure-level-raw "$maximum_pressure_level_raw"', source)
        self.assertNotIn("CAMELID_GEMMA4_GHOST_METAL_MAPPED_READAHEAD=", source)
        self.assertIn("mapped_readahead_enabled: false", source)
        self.assertIn("mapped_readahead_max_inflight_records: 0", source)
        self.assertIn("--baseline-soak-seconds 60", source)
        self.assertIn("--reject-swapin-growth", source)
        self.assertNotIn("--require-zero-current-swap", source)
        self.assertIn("readonly PORT=8189", source)
        self.assertIn("readonly PROTECTED_PORT=8181", source)
        self.assertNotIn("http://127.0.0.1:8181", source)
        self.assertIn('[[ "$input" == /* ]]', source)
        self.assertIn('while /bin/kill -0 "$request_pid"', source)
        self.assertIn("became active during the request", source)
        self.assertIn("parameters[CAMELID_HOT40_BINARY]", source)
        self.assertIn("CAMELID_HOT40_BINARY_SOURCE_COMMIT", source)
        self.assertIn("rev-parse --verify --end-of-options", source)
        self.assertIn("merge-base --is-ancestor", source)
        self.assertIn("diff --quiet", source)
        self.assertIn("src Cargo.toml Cargo.lock build.rs", source)
        self.assertIn('--arg binary_source_commit "$binary_source_commit"', source)
        self.assertIn('--arg harness_commit "$harness_commit"', source)
        self.assertNotIn("target/release/camelid", source)
        self.assertNotIn('sha256_file "$model"', source)
        self.assertNotIn('sha256_file "$cghost"', source)
        self.assertNotIn('sha256_file "$assistant"', source)
        self.assertNotIn("POST_HASH_COOLDOWN_SECONDS", source)
        self.assertIn("large_inputs_content_read_before_spawn: false", source)
        self.assertIn("historical-sha256-plus-live-stat", source)
        self.assertIn("status --porcelain=v1 --untracked-files=all", source)
        for key in hot40.PER_LAYER_ENVIRONMENT:
            self.assertIn(key, source)
            self.assertNotIn(f"{key}=", source)

    def test_sources_parse(self) -> None:
        subprocess.run(["/bin/zsh", "-n", str(RUNNER_PATH)], check=True)
        with tempfile.TemporaryDirectory() as directory:
            for name, path in (
                ("analyzer", ANALYZER_PATH),
                ("hasher", HASHER_PATH),
                ("host_sampler", HOST_SAMPLER_PATH),
                ("watchdog", WATCHDOG_PATH),
            ):
                py_compile.compile(
                    str(path),
                    cfile=str(Path(directory) / f"{name}.pyc"),
                    doraise=True,
                )

    def test_nocache_hasher_matches_canonical_fixture_hashes(self) -> None:
        for path in (REQUEST_PATH, EXPECTED_IDS_PATH):
            result = subprocess.run(
                ["/usr/bin/python3", str(HASHER_PATH), str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), sha256(path))

    def test_nocache_hasher_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "request-link.json"
            link.symlink_to(REQUEST_PATH)
            result = subprocess.run(
                ["/usr/bin/python3", str(HASHER_PATH), str(link)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 75)
            self.assertIn("REFUSED:", result.stderr)


if __name__ == "__main__":
    unittest.main()
