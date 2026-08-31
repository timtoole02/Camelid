from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.eagle3_mlx.cell_parity import (
    SCHEMA,
    CellParityError,
    _load_fixture,
    _numeric_metrics,
)
from tools.eagle3_mlx.contract import DRAFT_VOCAB_SIZE, HIDDEN_SIZE
from tools.eagle3_mlx.features import AUX_WIDTH


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CellParityFixtureTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        rows = 2
        depths = 2
        top_k = 8
        payloads = {
            "input_aux.f32le": (
                "float32",
                [rows, AUX_WIDTH],
                np.zeros((rows, AUX_WIDTH), dtype="<f4"),
            ),
            "input_embeddings.f32le": (
                "float32",
                [depths, rows, HIDDEN_SIZE],
                np.zeros((depths, rows, HIDDEN_SIZE), dtype="<f4"),
            ),
            "camelid_fused.f32le": (
                "float32",
                [rows, HIDDEN_SIZE],
                np.zeros((rows, HIDDEN_SIZE), dtype="<f4"),
            ),
            "camelid_depth0_hidden.f32le": (
                "float32",
                [rows, HIDDEN_SIZE],
                np.zeros((rows, HIDDEN_SIZE), dtype="<f4"),
            ),
            "camelid_recurrent_last_hidden.f32le": (
                "float32",
                [depths - 1, HIDDEN_SIZE],
                np.zeros((depths - 1, HIDDEN_SIZE), dtype="<f4"),
            ),
            "camelid_selected_draft_ids.u32le": (
                "uint32",
                [depths],
                np.zeros(depths, dtype="<u4"),
            ),
            "camelid_top_draft_ids.u32le": (
                "uint32",
                [depths, top_k],
                np.tile(np.arange(top_k, dtype="<u4"), (depths, 1)),
            ),
            "camelid_top_target_ids.u32le": (
                "uint32",
                [depths, top_k],
                np.tile(np.arange(100, 100 + top_k, dtype="<u4"), (depths, 1)),
            ),
            "camelid_top_logits.f32le": (
                "float32",
                [depths, top_k],
                np.tile(np.arange(top_k, 0, -1, dtype="<f4"), (depths, 1)),
            ),
        }
        arrays = []
        for filename, (dtype, shape, values) in payloads.items():
            path = root / filename
            path.write_bytes(values.tobytes(order="C"))
            arrays.append(
                {
                    "file": filename,
                    "dtype": dtype,
                    "shape": shape,
                    "sha256": _sha(path),
                }
            )
        manifest = {
            "schema": SCHEMA,
            "generator": "splitmix64-f32-v1",
            "seed": 7,
            "rows": rows,
            "depths": depths,
            "hidden_size": HIDDEN_SIZE,
            "auxiliary_width": AUX_WIDTH,
            "draft_vocab_size": DRAFT_VOCAB_SIZE,
            "top_k": top_k,
            "rope_theta": 10000.0,
            "sliding_window": 512,
            "checkpoint_weights_sha256": "11" * 32,
            "checkpoint_config_sha256": "22" * 32,
            "fixture_binary_sha256": "33" * 32,
            "camelid_version": "test",
            "camelid_lane": "dense-bf16-weights-f32-activations-f16-kv",
            "arrays": arrays,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_fixture_loader_is_strict_and_numeric_metrics_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            fixture = _load_fixture(root)
            self.assertEqual(fixture.embeddings.shape, (2, 2, HIDDEN_SIZE))
            self.assertEqual(fixture.camelid_top_draft_ids[1, 7], 7)
            metrics = _numeric_metrics(
                np.array([1.0, -2.0]), np.array([1.0, -2.0])
            )
            self.assertEqual(metrics["max_abs"], 0.0)
            self.assertEqual(metrics["mean_abs"], 0.0)
            self.assertAlmostEqual(metrics["cosine"], 1.0)

    def test_fixture_payload_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            payload = root / "camelid_top_logits.f32le"
            data = bytearray(payload.read_bytes())
            data[0] ^= 1
            payload.write_bytes(data)
            with self.assertRaisesRegex(CellParityError, "SHA-256 mismatch"):
                _load_fixture(root)


if __name__ == "__main__":
    unittest.main()
