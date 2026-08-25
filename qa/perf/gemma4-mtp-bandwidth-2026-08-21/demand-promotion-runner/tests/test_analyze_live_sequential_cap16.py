#!/usr/bin/env python3

import base64
import contextlib
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve().parents[1] / "analyze_live_sequential_cap16.py"
SPEC = importlib.util.spec_from_file_location("analyze_live_sequential_cap16", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)

SOURCE_COMMIT = "1" * 40


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(path: Path, values: dict[str, str]) -> None:
    lines = ["manifest_format=base64-v1"]
    for key, value in values.items():
        encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
        lines.append(f"{key}@BASE64={encoded}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_values() -> dict[str, str]:
    expected_text = ANALYZER.EXPECTED_TOKEN_FILE.read_bytes()
    values = {
        "HOME": "/Users/timtoole",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/tmp",
    }
    for raw in ANALYZER.PROFILE_FILE.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        key, value = raw.split("=", 1)
        values[key] = value.replace(
            "__ASSISTANT__",
            "/Users/timtoole/models/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors",
        )
    values.update(
        {
            "binary": "/Users/timtoole/camelid-h69-bin/camelid",
            "cache_mib": "0",
            "expected_token_ids_sha256": hashlib.sha256(expected_text).hexdigest(),
            "supervision_mode": "manual-no-watchdog",
            "source_commit": SOURCE_COMMIT,
            "source_tree_clean": "1",
            "binary_sha256": "2" * 64,
            "binary_size": "26710848",
            "binary_version": f"camelid v0.6.1-1-g{SOURCE_COMMIT[:8]}",
            "runner_sha256": "3" * 64,
            "manual_safety_sampler_sha256": "4" * 64,
        }
    )
    return values


def cap_values(cap: int) -> dict[str, object]:
    hits = {4: 2, 8: 4, 16: 8}[cap]
    predicted = cap
    actual = 8
    wall_us = 10_000
    saved_us = wall_us * hits // actual
    layers = ",".join(
        f"L{layer}:{hits}/{actual}/{predicted}@10.000"
        for layer in ANALYZER.TARGET_LAYERS
    )
    count = len(ANALYZER.TARGET_LAYERS)
    return {
        "hits": hits * count,
        "actual": actual * count,
        "predicted": predicted * count,
        "wall_us": wall_us * count,
        "saved_us": saved_us * count,
        "layers": layers,
    }


def probe_line(start_pos: int, k_tokens: int) -> str:
    caps = {cap: cap_values(cap) for cap in ANALYZER.CAPS}
    fields: dict[str, str] = {
        "schema": "2",
        "admitted": "1",
        "profile": "exact-live-sequential-hot-shape",
        "start_pos": str(start_pos),
        "K": str(k_tokens),
        "source": "post-attention-slab-b",
        "target": "next-layer-exact-union",
        "first_target_layer": "1",
        "eligible": "29",
        "attempts": "29",
        "failures": "0",
        "truth_valid": "1",
        "predict_us": "100",
        "predictor_top_k_per_row": "8",
        "probe_caps": "4,8,16",
    }
    for cap, values in caps.items():
        fields.update(
            {
                f"cap{cap}_hits": str(values["hits"]),
                f"cap{cap}_actual_cold": str(values["actual"]),
                f"cap{cap}_predicted_cold": str(values["predicted"]),
                f"cap{cap}_recall": ANALYZER._format_ratio(values["hits"], values["actual"]),
                f"cap{cap}_precision": ANALYZER._format_ratio(
                    values["hits"], values["predicted"]
                ),
                f"cap{cap}_read_wall_ms": ANALYZER._format_ms_us(values["wall_us"]),
                f"cap{cap}_projected_saved_ms": ANALYZER._format_ms_us(values["saved_us"]),
                f"cap{cap}_read_wall_weighted_recall": ANALYZER._format_ratio(
                    values["saved_us"], values["wall_us"]
                ),
            }
        )
    incremental_hits = caps[16]["hits"] - caps[8]["hits"]
    incremental_predicted = caps[16]["predicted"] - caps[8]["predicted"]
    incremental_saved = caps[16]["saved_us"] - caps[8]["saved_us"]
    candidates8 = "/".join(
        f"L{layer}:" + ",".join(str(expert) for expert in range(8))
        for layer in ANALYZER.TARGET_LAYERS
    )
    candidates16 = "/".join(
        f"L{layer}:" + ",".join(str(expert) for expert in range(16))
        for layer in ANALYZER.TARGET_LAYERS
    )
    fields.update(
        {
            "cap16_incremental_hits_vs_cap8": str(incremental_hits),
            "cap16_incremental_predicted_vs_cap8": str(incremental_predicted),
            "cap16_incremental_precision_vs_cap8": ANALYZER._format_ratio(
                incremental_hits, incremental_predicted
            ),
            "cap16_incremental_saved_ms_vs_cap8": ANALYZER._format_ms_us(
                incremental_saved
            ),
            "total_wave_load_ms": "300.000",
            "projection": "record-linear-per-layer-wave-load-ceiling",
            "stage_admitted": "1",
            "launches": "29",
            "launch_failures": "0",
            "stage_candidates": str(caps[8]["predicted"]),
            "readiness_measured": "1",
            "contention_measured": "1",
            "output_mutation": "0",
            "io_mutation": "1",
            "slot_policy_mutation": "0",
            "routing_authority": "exact-router",
            "cap8_candidates": candidates8,
            "cap16_candidates": candidates16,
            "cap4_layers": str(caps[4]["layers"]),
            "cap8_layers": str(caps[8]["layers"]),
            "cap16_layers": str(caps[16]["layers"]),
        }
    )
    assert set(fields) == ANALYZER.PROBE_FIELDS
    return ANALYZER.PROBE_PREFIX + " ".join(f"{key}={value}" for key, value in fields.items())


def stage_line(round_seq: int, start_pos: int, k_tokens: int) -> str:
    fields = {
        "schema": "1",
        "round_seq": str(round_seq),
        "start_pos": str(start_pos),
        "K": str(k_tokens),
        "ok": "1",
        "cap": "8",
        "workers": "2",
        "workers_started": "2",
        "workers_done": "0",
        "predict_impl": "accelerate-sgemm",
        "launches": "29",
        "candidates": "232",
        "reads_started": "200",
        "reads_succeeded": "200",
        "reads_failed": "0",
        "reads_in_flight": "0",
        "speculative_read_ms": "100.000",
        "seals": "30",
        "exact_cold": "240",
        "ready_hits": "100",
        "previous_ready_hits": "0",
        "direct_fallback": "140",
        "ready_unused": "10",
        "ready_malformed": "0",
        "ready_copy_ms": "5.000",
        "ready_returned": "110",
        "worker_unused_ready": "50",
        "late_discarded": "40",
        "worker_done": "0",
        "cancelled": "0",
        "snapshot_terminal": "0",
        "shared_demand_pool": "0",
        "consumer_waits": "0",
        "late_model_publish": "0",
        "routing_authority": "exact-router",
    }
    assert set(fields) == ANALYZER.STAGE_FIELDS
    return ANALYZER.STAGE_PREFIX + " ".join(f"{key}={value}" for key, value in fields.items())


def response() -> dict[str, object]:
    expected_ids = json.loads(ANALYZER.EXPECTED_TOKEN_FILE.read_text(encoding="utf-8"))
    return {
        "id": "chatcmpl-gemma4",
        "object": "chat.completion",
        "created": 1,
        "model": "26B_dequant_it_hf",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "fixture"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 104, "completion_tokens": 48, "total_tokens": 152},
        "camelid": {
            "generated_token_ids": expected_ids,
            "timings_ms": {
                "generate": 1000.0,
                "generation": {"forward_total": 1000.0},
                "prompt_evaluation": {},
                "lane": "gemma4_wall_clock_total_only",
            },
            "exact_match_expected": True,
            "exact_match_count": 48,
        },
    }


def health() -> dict[str, object]:
    return {
        "ok": True,
        "engine": "camelid",
        "build": f"v0.6.1-1-g{SOURCE_COMMIT[:8]}",
        "source_commit": SOURCE_COMMIT,
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


def host_sample(wired: int) -> dict[str, int]:
    return {
        "page_size_bytes": 16_384,
        "swapped_pages_current": 0,
        "swapped_bytes_current": 0,
        "swapins_pages": 0,
        "swapouts_pages": 0,
        "pressure_level_raw": 1,
        "wired_bytes": wired,
    }


def process_sample(footprint: int, rss: int) -> dict[str, object]:
    return {
        "pid": 123,
        "process_group": 123,
        "member_pids": [123],
        "rss_bytes": rss,
        "physical_footprint_bytes": footprint,
    }


def manual_safety(response_size: int) -> dict[str, object]:
    checks = {key: True for key in ANALYZER.MANUAL_CHECK_FIELDS}
    return {
        "schema_version": 1,
        "mode": "manual-no-watchdog",
        "qualifying": False,
        "point_samples_only": True,
        "configured_caps": {
            "maximum_child_physical_footprint_bytes": ANALYZER.MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES,
            "maximum_host_wired_bytes": ANALYZER.MAX_HOST_WIRED_BYTES,
            "maximum_pressure_level_raw": 1,
            "require_zero_current_swap": True,
            "reject_swapin_growth": True,
            "reject_swapout_growth": True,
        },
        "samples": {
            "pre": {"schema_version": 1, "phase": "pre", "host": host_sample(1000), "process": None},
            "ready": {
                "schema_version": 1,
                "phase": "ready",
                "host": host_sample(2000),
                "process": process_sample(4000, 3000),
            },
            "post": {
                "schema_version": 1,
                "phase": "post",
                "host": host_sample(1500),
                "process": process_sample(3500, 2500),
            },
        },
        "sampled_peaks": {
            "child_rss_bytes": 3000,
            "child_physical_footprint_bytes": 4000,
            "host_wired_bytes": 2000,
        },
        "child": {"pid": 123, "process_group": 123, "returncode": 0, "process_group_empty": True},
        "report": {
            "exists": True,
            "is_regular_file": True,
            "is_symlink": False,
            "size_bytes": response_size,
        },
        "checks": checks,
    }


def create_fixture(run_dir: Path) -> None:
    write_manifest(run_dir / "env.txt", manifest_values())
    write_json(run_dir / "health.json", health())
    write_json(run_dir / "response.json", response())
    lines = []
    for index, (start_pos, k_tokens) in enumerate(ANALYZER.EXPECTED_ROUNDS):
        lines.append(probe_line(start_pos, k_tokens))
        lines.append(stage_line(41 + index, start_pos, k_tokens))
    (run_dir / "server.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        run_dir / "manual-safety.json",
        manual_safety((run_dir / "response.json").stat().st_size),
    )


def refresh_report_size(run_dir: Path) -> None:
    path = run_dir / "manual-safety.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["report"]["size_bytes"] = (run_dir / "response.json").stat().st_size
    write_json(path, value)


@contextlib.contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "h69-fixture"
        run_dir.mkdir()
        create_fixture(run_dir)
        yield run_dir


def mutate_log_field(run_dir: Path, prefix: str, key: str, value: str, occurrence: int = 0) -> None:
    lines = (run_dir / "server.log").read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    index = matches[occurrence]
    encoded = lines[index].split(" ")
    old = next(item for item in encoded if item.startswith(f"{key}="))
    encoded[encoded.index(old)] = f"{key}={value}"
    lines[index] = " ".join(encoded)
    (run_dir / "server.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


class LiveSequentialCap16AnalyzerTest(unittest.TestCase):
    def test_valid_manual_fixture_is_go_but_non_qualifying(self) -> None:
        with fixture() as run_dir:
            result = ANALYZER.analyze(run_dir)
        self.assertEqual(result["decision"]["verdict"], "GO")
        self.assertEqual(result["decision"]["qualification"], "NON_QUALIFYING_MANUAL_POINT_SAMPLES")
        self.assertFalse(result["integrity"]["qualifying_safety_receipt"])
        self.assertEqual([item["stage_round_seq"] for item in result["rounds"]], [41, 42, 43, 44])
        self.assertAlmostEqual(result["request_projection"]["measured_cap8_readiness"], 100 / 116)

    def test_probe_rejects_unknown_missing_and_duplicate_fields(self) -> None:
        for mutation in ("unknown", "missing", "duplicate"):
            with self.subTest(mutation=mutation), fixture() as run_dir:
                lines = (run_dir / "server.log").read_text(encoding="utf-8").splitlines()
                index = next(i for i, line in enumerate(lines) if line.startswith(ANALYZER.PROBE_PREFIX))
                if mutation == "unknown":
                    lines[index] += " surprise=1"
                elif mutation == "missing":
                    lines[index] = lines[index].replace(" predict_us=100", "", 1)
                else:
                    lines[index] += " predict_us=100"
                (run_dir / "server.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaises(ANALYZER.ReceiptError):
                    ANALYZER.analyze(run_dir)

    def test_probe_rejects_noncanonical_numbers_and_forged_aggregate(self) -> None:
        cases = (
            ("start_pos", "0104", "canonical"),
            ("cap4_read_wall_ms", "290.00", "canonical"),
            ("cap16_hits", "233", "recomputation"),
        )
        for key, value, message in cases:
            with self.subTest(key=key), fixture() as run_dir:
                mutate_log_field(run_dir, ANALYZER.PROBE_PREFIX, key, value)
                with self.assertRaisesRegex(ANALYZER.ReceiptError, message):
                    ANALYZER.analyze(run_dir)

    def test_probe_rejects_compact_invalid_truth_receipt(self) -> None:
        with fixture() as run_dir:
            lines = (run_dir / "server.log").read_text(encoding="utf-8").splitlines()
            index = next(i for i, line in enumerate(lines) if line.startswith(ANALYZER.PROBE_PREFIX))
            lines[index] = ANALYZER.PROBE_PREFIX + "schema=2 admitted=1 truth_valid=0 reason=LATTICE"
            (run_dir / "server.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ANALYZER.ReceiptError, "invalidated"):
                ANALYZER.analyze(run_dir)

    def test_probe_rejects_layer_lattice_and_candidate_identity_mutations(self) -> None:
        cases = (
            (
                "cap4_layers",
                cap_values(4)["layers"].replace("L1:2/8/4", "L1:2/8/3", 1),
            ),
            (
                "cap16_layers",
                cap_values(16)["layers"].replace("L1:8/8/16", "L1:3/8/16", 1),
            ),
            (
                "cap16_candidates",
                probe_line(104, 14).split(" cap16_candidates=", 1)[1].split(" ", 1)[0].replace(
                    "L1:0,1", "L1:16,1", 1
                ),
            ),
            (
                "cap8_candidates",
                probe_line(104, 14).split(" cap8_candidates=", 1)[1].split(" ", 1)[0].replace(
                    "L1:0,1", "L1:0,0", 1
                ),
            ),
        )
        for key, value in cases:
            with self.subTest(key=key), fixture() as run_dir:
                mutate_log_field(run_dir, ANALYZER.PROBE_PREFIX, key, str(value))
                with self.assertRaises(ANALYZER.ReceiptError):
                    ANALYZER.analyze(run_dir)

    def test_exact_round_schedule_and_consecutive_arbitrary_stage_base_are_required(self) -> None:
        with fixture() as run_dir:
            mutate_log_field(run_dir, ANALYZER.PROBE_PREFIX, "start_pos", "105")
            with self.assertRaisesRegex(ANALYZER.ReceiptError, "schedule"):
                ANALYZER.analyze(run_dir)
        with fixture() as run_dir:
            mutate_log_field(run_dir, ANALYZER.STAGE_PREFIX, "round_seq", "99", occurrence=2)
            with self.assertRaisesRegex(ANALYZER.ReceiptError, "consecutive"):
                ANALYZER.analyze(run_dir)

    def test_stage_rejects_candidate_and_internal_accounting_forgery(self) -> None:
        for key, value in (
            ("candidates", "231"),
            ("reads_succeeded", "199"),
            ("ready_returned", "109"),
            ("direct_fallback", "139"),
            ("seals", "31"),
            ("consumer_waits", "1"),
            ("ready_malformed", "1"),
        ):
            with self.subTest(key=key), fixture() as run_dir:
                mutate_log_field(run_dir, ANALYZER.STAGE_PREFIX, key, value)
                with self.assertRaises(ANALYZER.ReceiptError):
                    ANALYZER.analyze(run_dir)

    def test_stage_and_probe_cold_scopes_are_not_equated(self) -> None:
        with fixture() as run_dir:
            result = ANALYZER.analyze(run_dir)
            self.assertEqual(
                result["rounds"][0][
                    "stage_exact_cold_minus_probe_l1_l29_actual"
                ],
                8,
            )
        with fixture() as run_dir:
            mutate_log_field(run_dir, ANALYZER.STAGE_PREFIX, "exact_cold", "200")
            mutate_log_field(run_dir, ANALYZER.STAGE_PREFIX, "direct_fallback", "100")
            mutate_log_field(run_dir, ANALYZER.STAGE_PREFIX, "seals", "29")
            result = ANALYZER.analyze(run_dir)
            self.assertEqual(
                result["rounds"][0][
                    "stage_exact_cold_minus_probe_l1_l29_actual"
                ],
                -32,
            )

    def test_response_rejects_duplicate_json_keys_nan_and_wrong_token(self) -> None:
        with fixture() as run_dir:
            path = run_dir / "response.json"
            raw = path.read_text(encoding="utf-8").replace(
                '"id": "chatcmpl-gemma4"', '"id": "duplicate", "id": "chatcmpl-gemma4"'
            )
            path.write_text(raw, encoding="utf-8")
            refresh_report_size(run_dir)
            with self.assertRaisesRegex(ANALYZER.ReceiptError, "duplicate JSON"):
                ANALYZER.analyze(run_dir)
        with fixture() as run_dir:
            path = run_dir / "response.json"
            raw = path.read_text(encoding="utf-8").replace("1000.0", "NaN", 1)
            path.write_text(raw, encoding="utf-8")
            refresh_report_size(run_dir)
            with self.assertRaisesRegex(ANALYZER.ReceiptError, "non-finite"):
                ANALYZER.analyze(run_dir)
        with fixture() as run_dir:
            value = response()
            value["camelid"]["generated_token_ids"][0] += 1
            write_json(run_dir / "response.json", value)
            safety = manual_safety((run_dir / "response.json").stat().st_size)
            write_json(run_dir / "manual-safety.json", safety)
            with self.assertRaisesRegex(ANALYZER.ReceiptError, "frozen"):
                ANALYZER.analyze(run_dir)

    def test_environment_rejects_profile_drift_and_unknown_fields(self) -> None:
        for mutation in ("slots", "kv", "extra"):
            with self.subTest(mutation=mutation), fixture() as run_dir:
                values = manifest_values()
                if mutation == "slots":
                    values["CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER"] = values[
                        "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER"
                    ].replace("60", "61", 1)
                elif mutation == "kv":
                    values["CAMELID_GEMMA4_KV_INIT"] = "512"
                else:
                    values["CAMELID_UNDECLARED"] = "1"
                write_manifest(run_dir / "env.txt", values)
                with self.assertRaises(ANALYZER.ReceiptError):
                    ANALYZER.analyze(run_dir)

    def test_manual_safety_rejects_swap_pressure_footprint_and_false_check(self) -> None:
        mutations = ("swap", "pressure", "footprint", "check")
        for mutation in mutations:
            with self.subTest(mutation=mutation), fixture() as run_dir:
                path = run_dir / "manual-safety.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                if mutation == "swap":
                    value["samples"]["post"]["host"]["swapouts_pages"] = 1
                elif mutation == "pressure":
                    value["samples"]["ready"]["host"]["pressure_level_raw"] = 2
                elif mutation == "footprint":
                    value["samples"]["ready"]["process"]["physical_footprint_bytes"] = (
                        ANALYZER.MAX_CHILD_PHYSICAL_FOOTPRINT_BYTES + 1
                    )
                else:
                    value["checks"]["all_passed"] = False
                write_json(path, value)
                with self.assertRaises(ANALYZER.ReceiptError):
                    ANALYZER.analyze(run_dir)

    def test_health_must_be_generation_ready(self) -> None:
        with fixture() as run_dir:
            value = health()
            value["generation_ready"] = False
            write_json(run_dir / "health.json", value)
            with self.assertRaisesRegex(ANALYZER.ReceiptError, "generation_ready"):
                ANALYZER.analyze(run_dir)


if __name__ == "__main__":
    unittest.main()
