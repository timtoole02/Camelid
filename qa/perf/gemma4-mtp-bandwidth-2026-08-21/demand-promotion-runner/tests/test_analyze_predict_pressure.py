#!/usr/bin/env python3

import base64
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve().parents[1] / "analyze_predict_pressure.py"
SPEC = importlib.util.spec_from_file_location("analyze_predict_pressure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def expert_mask(experts: range) -> str:
    return f"{sum(1 << expert for expert in experts):032x}"


def probe_line(round_seq: int, start_pos: int) -> str:
    actual_mask = expert_mask(range(8))
    hot_mask = expert_mask(range(4))
    sizes = ",".join(["8"] * PROBE.LAYERS)
    hot_sizes = ",".join(["4"] * PROBE.LAYERS)
    actual_masks = ",".join([actual_mask] * PROBE.LAYERS)
    hot_masks = ",".join([hot_mask] * PROBE.LAYERS)
    ranked = "/".join(["0,1,2,3,4,5,6,7"] * PROBE.LAYERS)
    return (
        PROBE.PROBE_PREFIX
        + f"schema=1 round_seq={round_seq} start_pos={start_pos} K=8 "
        "predictor_top_k=8 predict_us=10 truth_valid=1 "
        "residual_pairs=120 predicted_cold_pairs=120 "
        "predicted_residual_hits=120 "
        f"actual_sizes={sizes} hot_sizes={hot_sizes} approx_sizes={sizes} "
        f"actual_masks={actual_masks} hot_masks={hot_masks} "
        f"approx_masks={actual_masks} approx_ranked_ids={ranked}"
    )


def fill_traces() -> list[str]:
    return [
        "[hybrid fill trace] "
        f"layer={layer} selected=8 slots={slots} occupied=4 "
        "hits=4 loads=4 evicted=0 cold_fallback=0"
        for layer, slots in enumerate(PROBE.H2_SLOTS)
    ]


def device_line(drafts: int) -> str:
    return (
        "[gemma4-mtp device-chain] "
        f"requested_drafts={drafts} returned_drafts={drafts} command_buffers=1"
    )


def ledger_line(start_pos: int, k: int) -> str:
    return (
        f"[metal chained ledger] start_pos={start_pos} K={k} "
        "ok=true predicted=false slot_wait=1.0ms"
    )


def mtp_line(round_index: int) -> str:
    return (
        f"[mtp round] #{round_index} wall=10.0ms "
        "(assistant=1.0ms, verifier=9.0ms) accepted=1/8"
    )


def cap_probe(
    ranked: list[list[int]], hot_masks: list[int], residual_masks: list[int]
) -> dict[str, object]:
    return {
        "approx_ranked_ids": ranked,
        "hot_masks": hot_masks,
        "residual_masks": residual_masks,
    }


class PredictPressureTailBindingTest(unittest.TestCase):
    def synthetic_log(self) -> str:
        lines: list[str] = []
        for round_index in range(6):
            start_pos = 100 + round_index * 8
            lines.append(device_line(7))
            lines.extend(fill_traces())
            lines.append(probe_line(round_index + 1, start_pos))
            lines.append(ledger_line(start_pos, 8))
            lines.append(mtp_line(round_index))
        for round_index, (drafts, k, start_pos) in enumerate(
            ((4, 5, 148), (2, 3, 150)), start=6
        ):
            lines.append(device_line(drafts))
            lines.extend(fill_traces())
            lines.append(ledger_line(start_pos, k))
            lines.append(mtp_line(round_index))
        return "\n".join(lines)

    def test_k8_probes_ignore_k5_k3_generation_tail(self) -> None:
        rounds, mtp, full_k8, tails, refused, failures = PROBE.parse_observations(
            self.synthetic_log()
        )
        self.assertEqual(len(rounds), 6)
        self.assertEqual(mtp, 8)
        self.assertEqual(full_k8, 6)
        self.assertEqual(tails, 2)
        self.assertEqual(refused, 0)
        self.assertEqual(failures, 0)

    def test_global_caps_use_ranked_cold_round_robin(self) -> None:
        parsed = PROBE.parse_probe(probe_line(1, 100))
        self.assertEqual(PROBE.cap_round_robin(parsed, 64), {"selected": 64, "hits": 64})
        self.assertEqual(PROBE.cap_round_robin(parsed, 96), {"selected": 96, "hits": 96})

    def test_global_cap_skips_hot_and_counts_false_candidates_before_hits(self) -> None:
        parsed = cap_probe(
            ranked=[[0, 1, 2] for _ in range(PROBE.LAYERS)],
            hot_masks=[1 << 0 for _ in range(PROBE.LAYERS)],
            residual_masks=[1 << 2 for _ in range(PROBE.LAYERS)],
        )
        self.assertEqual(PROBE.cap_round_robin(parsed, 30), {"selected": 30, "hits": 0})
        self.assertEqual(PROBE.cap_round_robin(parsed, 60), {"selected": 60, "hits": 30})

    def test_global_cap_advances_layers_round_robin_not_one_layer_deep(self) -> None:
        parsed = cap_probe(
            ranked=[[10, 11]] + [[20] for _ in range(PROBE.LAYERS - 1)],
            hot_masks=[0] * PROBE.LAYERS,
            residual_masks=[1 << 11] + [0] * (PROBE.LAYERS - 1),
        )
        self.assertEqual(PROBE.cap_round_robin(parsed, 30), {"selected": 30, "hits": 0})
        self.assertEqual(PROBE.cap_round_robin(parsed, 31), {"selected": 31, "hits": 1})

    def test_trace_reconstructs_start_hot_as_post_plan_occupied_plus_evicted(self) -> None:
        parsed = PROBE.parse_probe(probe_line(1, 100))
        traces = [PROBE.parse_trace(PROBE.TRACE_RE.fullmatch(line)) for line in fill_traces()]
        traces[0]["occupied"] -= 1
        traces[0]["evicted"] += 1
        PROBE.validate_trace(parsed, traces)
        traces[0]["occupied"] += 1
        with self.assertRaisesRegex(PROBE.ReceiptError, "occupancy plus evictions"):
            PROBE.validate_trace(parsed, traces)

    def test_full_k8_verifier_without_probe_is_refused(self) -> None:
        log = "\n".join((device_line(7), ledger_line(100, 8), mtp_line(0)))
        with self.assertRaisesRegex(PROBE.ReceiptError, "lacks exactly one"):
            PROBE.parse_observations(log)

    def test_base64_manifest_preserves_trailing_newlines(self) -> None:
        value = "line one\nline two\n\n"
        encoded = base64.b64encode(value.encode()).decode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.txt"
            path.write_text(
                "manifest_format=base64-v1\n"
                f"CAMELID_TEST_VALUE@BASE64={encoded}\n"
            )
            self.assertEqual(PROBE.parse_env(path)["CAMELID_TEST_VALUE"], value)

    def test_h2_environment_rejects_unapproved_behavior_variable(self) -> None:
        profile = (SCRIPT.parent / "env" / "H2-proportional").read_text()
        profile = profile.replace("__ASSISTANT__", "/tmp/model.safetensors")
        profile += (
            "CAMELID_GEMMA4_GHOST_METAL_SPARSE_PREDICT_PROBE=1\n"
            "CAMELID_GEMMA4_GHOST_METAL_DEMAND_PROMOTION_TRACE=1\n"
            "CAMELID_GEMMA4_GHOST_METAL_HOT_COLD_OVERLAP=1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "env.txt").write_text(profile)
            with self.assertRaisesRegex(PROBE.ReceiptError, "unapproved extra"):
                PROBE.validate_h2_environment(run_dir)


if __name__ == "__main__":
    unittest.main()
