from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gate_50tps", SCRIPT_DIR / "gate_50tps.py")
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class Gate50TpsTests(unittest.TestCase):
    def fixture(self, root: Path, wall_per_round: float = 150.0) -> tuple[Path, Path, Path]:
        expected = list(range(48))
        rounds = []
        for index in range(6):
            per_layer = [{"mapped_bound": 1}]
            per_layer.extend({"mapped_bound": 0} for _ in gate.PROFILE[1:])
            rounds.append(
                {
                    "round_index": index,
                    "bootstrap": False,
                    "k": 8,
                    "requested_k": 8,
                    "proposed_k": 7,
                    "verifier_k": 8,
                    "accepted_drafts": 7,
                    "useful_accepted_drafts": 7,
                    "budget_truncated": False,
                    "committed_tokens": expected[index * 8 : (index + 1) * 8],
                    "receipt_round_wall_ms": wall_per_round,
                    "success": True,
                    "selected_dropped": 0,
                    "missing_failclose": 0,
                    "slot_capacity_overflow": 0,
                    "overflow_experts": 0,
                    "mapped_readahead_advised_records": 1,
                    "mapped_readahead_advised_bytes": 3_345_408,
                    "mapped_readahead_enqueue_ms": 0.01,
                    "mapped_readahead_previous_union_advised_records": 1,
                    "mapped_readahead_previous_union_advised_bytes": 3_345_408,
                    "mapped_readahead_previous_union_enqueue_ms": 0.01,
                    "per_layer": per_layer,
                }
            )
        total_wall = wall_per_round * len(rounds)
        response = {
            "usage": {"completion_tokens": 48},
            "choices": [{"finish_reason": "length"}],
            "camelid": {
                "generated_token_ids": expected,
                "hybrid_telemetry": {
                    "schema_version": 1,
                    "geometry": {
                        "record_payload_bytes": 3_345_408,
                        "mapped_readahead_enabled": True,
                        "mapped_readahead_max_inflight_records": 64,
                        "mapped_readahead_anonymous_capacity_bytes": 0,
                    },
                    "rounds": rounds,
                    "metrics": {
                        "forwarded_decode_tokens": 48,
                        "terminal_unforwarded_tokens": 0,
                        "response_completion_tokens": 48,
                        "proposed_drafts": 42,
                        "accepted_drafts": 42,
                        "receipt_round_wall_ms": total_wall,
                        "decode_tokens_per_second": 48 / (total_wall / 1000.0),
                        "outer_lookahead_nonzero_count": 0,
                    },
                },
            },
        }
        response_path = root / "response.json"
        expected_path = root / "expected.json"
        log_path = root / "server.log"
        response_path.write_text(json.dumps(response), encoding="utf-8")
        expected_path.write_text(json.dumps(expected), encoding="utf-8")
        chain = (
            "[gemma4-mtp device-chain] requested_drafts=7 returned_drafts=7 "
            "command_buffers=1 commits=1 waits=1 cpu_embedding_callbacks=0 "
            "linear_format=q4_0_all matrix_bytes_per_draft=236077056 "
            "encode_us=1 wait_us=2 gpu_us=3 kernel_us=4 wall_us=5\n"
        )
        log_path.write_text(
            "CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER="
            + gate.PROFILE_CSV
            + "\n"
            + "hybrid decode promotion policy: CAMELID_GEMMA4_GHOST_METAL_DECODE_PROMOTION "
            "effective=0 terminal_decode_promotion=off final_prefill_hot_handoff=on\n"
            + gate.MAPPED_READAHEAD_MARKER
            + "\n"
            + gate.PREVIOUS_UNION_READAHEAD_MARKER
            + "\n"
            + "[gemma4-mtp bootstrap] prefill_seed_attempted=1 used=1 fallback=none\n"
            + "[gemma4 exact partition] CAMELID_GEMMA4_DENSE_K8_GENERIC=1 "
            "static_k8_dense=off runtime_k_dense=on\n"
            + gate.PACKED_K8_GATEUP_MARKER
            + "\n"
            + "[gemma4-mtp full-q4] enabled=true source_sha256="
            + gate.FULL_Q4_SOURCE_SHA256
            + " matrices=23 packed_bytes=236077056 bf16_matrix_bytes=839385088 "
            "quantize_us=1 norms_quantized=false fallback=false\n"
            + chain * 6,
            encoding="utf-8",
        )
        return response_path, log_path, expected_path

    def test_exact_48_token_900ms_receipt_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            verdict = gate.analyze(*paths)
            self.assertTrue(verdict["pass"])
            self.assertEqual(verdict["effective_decode_tokens_per_second"], 48 / 0.9)

    def test_receipt_over_960ms_fails_the_performance_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary), wall_per_round=161.0)
            verdict = gate.analyze(*paths)
            self.assertFalse(verdict["pass"])
            self.assertFalse(verdict["performance_gates"]["decode_wall_le_960_ms"])

    def test_k1_bootstrap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response["camelid"]["hybrid_telemetry"]["rounds"][0]["bootstrap"] = True
            response_path.write_text(json.dumps(response), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "separate K1 bootstrap"):
                gate.analyze(response_path, log_path, expected_path)

    def test_target_token_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response["camelid"]["generated_token_ids"][17] = 999
            response_path.write_text(json.dumps(response), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "target-authoritative K1 baseline"):
                gate.analyze(response_path, log_path, expected_path)

    def test_missing_partition_parity_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            log = log_path.read_text(encoding="utf-8")
            log = log.replace(
                "[gemma4 exact partition] CAMELID_GEMMA4_DENSE_K8_GENERIC=1 "
                "static_k8_dense=off runtime_k_dense=on\n",
                "",
            )
            log_path.write_text(log, encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "partition-parity"):
                gate.analyze(response_path, log_path, expected_path)

    def test_missing_mapped_readahead_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            log = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                log.replace(gate.MAPPED_READAHEAD_MARKER + "\n", ""),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate.GateError, "selected-cold mapped readahead"):
                gate.analyze(response_path, log_path, expected_path)

    def test_duplicate_mapped_readahead_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(gate.MAPPED_READAHEAD_MARKER + "\n")
            with self.assertRaisesRegex(gate.GateError, "selected-cold mapped readahead"):
                gate.analyze(response_path, log_path, expected_path)

    def test_inconsistent_mapped_readahead_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response["camelid"]["hybrid_telemetry"]["rounds"][0][
                "mapped_readahead_advised_bytes"
            ] += 1
            response_path.write_text(json.dumps(response), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "mapped-cold readahead receipt"):
                gate.analyze(response_path, log_path, expected_path)

    def test_missing_previous_union_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            log = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                log.replace(gate.PREVIOUS_UNION_READAHEAD_MARKER + "\n", ""),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate.GateError, "previous-union readahead"):
                gate.analyze(response_path, log_path, expected_path)

    def test_inconsistent_previous_union_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response["camelid"]["hybrid_telemetry"]["rounds"][0][
                "mapped_readahead_previous_union_advised_bytes"
            ] += 1
            response_path.write_text(json.dumps(response), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "previous-union readahead receipt"):
                gate.analyze(response_path, log_path, expected_path)

    def test_missing_exact_packed_gateup_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            log = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                log.replace(gate.PACKED_K8_GATEUP_MARKER + "\n", ""),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate.GateError, "exact packed K8 GateUp"):
                gate.analyze(response_path, log_path, expected_path)

    def test_missing_full_q4_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            log = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                gate.FULL_Q4_MARKER_PATTERN.sub("", log),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate.GateError, "exact full-Q4 assistant"):
                gate.analyze(response_path, log_path, expected_path)

    def test_non_full_q4_device_chain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            response_path, log_path, expected_path = self.fixture(Path(temporary))
            log = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                log.replace("linear_format=q4_0_all", "linear_format=bf16_q4_0_head"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate.GateError, "full-Q4 assistant"):
                gate.analyze(response_path, log_path, expected_path)


if __name__ == "__main__":
    unittest.main()
