from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from tools.eagle3_mlx.contract import (
    DRAFT_VOCAB_SIZE,
    EXPECTED_TENSOR_COUNT,
    FLOAT_TENSOR_NAMES,
    TARGET_VOCAB_SIZE,
    TENSOR_SPECS,
    ContractError,
    parse_safetensors_header,
    target_to_draft_index,
    validate_config,
)
from tools.eagle3_mlx.features import (
    AUX_WIDTH,
    HIDDEN_SIZE,
    INVALID_TOKEN_ID,
    POSITIONAL_CONTRACT,
    SCHEMA_ID,
    FeatureFormatError,
    FeatureStore,
    build_ttt_batch,
    float32_to_bf16_bits,
)
from tools.eagle3_mlx.model import (
    soft_target_cross_entropy,
    soft_target_cross_entropy_numpy,
    soft_target_kl,
    soft_target_kl_numpy,
)
from tools.eagle3_mlx.train import _scheduled_learning_rate

try:
    import mlx.core as mx
except ModuleNotFoundError:
    mx = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContractTests(unittest.TestCase):
    def test_train_module_imports_without_optional_mlx(self):
        train_module = importlib.import_module("tools.eagle3_mlx.train")
        self.assertTrue(callable(train_module.main))

    def test_serving_contract_is_exactly_fifteen_tensors(self):
        self.assertEqual(len(TENSOR_SPECS), EXPECTED_TENSOR_COUNT)
        self.assertEqual(len(FLOAT_TENSOR_NAMES), 13)
        self.assertEqual(TENSOR_SPECS["fc.weight"].shape, (3072, 9216))
        self.assertEqual(TENSOR_SPECS["lm_head.weight"].shape, (32000, 3072))
        self.assertEqual(TENSOR_SPECS["d2t"].shape, (32000,))
        self.assertEqual(TENSOR_SPECS["t2d"].shape, (128256,))

    def test_target_to_draft_inverse_uses_mapping_rows_not_offsets(self):
        absolute = np.arange(DRAFT_VOCAB_SIZE, dtype=np.uint32) * 2
        inverse = target_to_draft_index(absolute)
        self.assertEqual(inverse.shape, (TARGET_VOCAB_SIZE,))
        self.assertEqual(int(inverse[0]), 0)
        self.assertEqual(int(inverse[12]), 6)
        self.assertEqual(int(inverse[13]), -1)

    def test_config_geometry_fails_closed(self):
        config = {
            "architectures": ["LlamaForCausalLMEagle3"],
            "model_type": "llama",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 1,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 128256,
            "draft_vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "rope_theta": 10000.0,
            "sliding_window": 512,
            "use_sliding_window": True,
        }
        validate_config(config)
        config["draft_vocab_size"] = 32001
        with self.assertRaisesRegex(ContractError, "draft_vocab_size"):
            validate_config(config)

    def test_header_parser_requires_dense_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.safetensors"
            header = {
                "a": {"dtype": "I32", "shape": [1], "data_offsets": [1, 5]}
            }
            raw = json.dumps(header, separators=(",", ":")).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 5)
            with self.assertRaisesRegex(ContractError, "dense payload offset"):
                parse_safetensors_header(path)

    def test_warmup_cosine_schedule_is_global_and_bounded(self):
        values = [
            _scheduled_learning_rate(
                step=step, base=1.0, warmup_steps=2, total_steps=6
            )
            for step in range(1, 7)
        ]
        self.assertEqual(values[:2], [0.5, 1.0])
        self.assertGreater(values[2], values[3])
        self.assertGreater(values[3], values[4])
        self.assertEqual(values[-1], 0.0)
        with self.assertRaises(ValueError):
            _scheduled_learning_rate(
                step=7, base=1.0, warmup_steps=2, total_steps=6
            )


class FeatureStoreTests(unittest.TestCase):
    def _write_store(self, root: Path, rows: int = 12) -> Path:
        sample_dir = root / "samples" / "00000000"
        sample_dir.mkdir(parents=True)
        input_ids = np.arange(100, 100 + rows, dtype="<u4")
        labels = np.full(rows, INVALID_TOKEN_ID, dtype="<u4")
        labels[:-2] = input_ids[2:]
        target = np.arange(200, 200 + rows, dtype="<u4")
        target[-1] = INVALID_TOKEN_ID
        mask = np.ones(rows, dtype="u1")
        mask[:3] = 0
        mask[-2:] = 0
        aux_values = np.repeat(np.arange(rows, dtype=np.float32)[:, None], AUX_WIDTH, axis=1)
        hidden_values = np.repeat(
            np.arange(rows, dtype=np.float32)[:, None], HIDDEN_SIZE, axis=1
        )
        embedding_values = np.repeat(
            (10 + np.arange(rows, dtype=np.float32))[:, None], HIDDEN_SIZE, axis=1
        )
        embedding_values[-1] = 0
        teacher_values = np.repeat(
            (20 + np.arange(rows, dtype=np.float32))[:, None],
            DRAFT_VOCAB_SIZE,
            axis=1,
        )
        teacher_values[-1] = 0
        payloads = {
            "input_ids.u32le": ("uint32", [rows], input_ids),
            "labels.u32le": ("uint32", [rows], labels),
            "target_argmax.u32le": ("uint32", [rows], target),
            "loss_mask.u8": ("uint8", [rows], mask),
            "aux_layer_inputs.bf16le": (
                "bfloat16",
                [rows, AUX_WIDTH],
                float32_to_bf16_bits(aux_values),
            ),
            "hidden_state.bf16le": (
                "bfloat16",
                [rows, HIDDEN_SIZE],
                float32_to_bf16_bits(hidden_values),
            ),
            "input_embedding.bf16le": (
                "bfloat16",
                [rows, HIDDEN_SIZE],
                float32_to_bf16_bits(
                    np.repeat(
                        (9 + np.arange(rows, dtype=np.float32))[:, None],
                        HIDDEN_SIZE,
                        axis=1,
                    )
                ),
            ),
            "next_token_embedding.bf16le": (
                "bfloat16",
                [rows, HIDDEN_SIZE],
                float32_to_bf16_bits(embedding_values),
            ),
            "teacher_draft_logits.bf16le": (
                "bfloat16",
                [rows, DRAFT_VOCAB_SIZE],
                float32_to_bf16_bits(teacher_values),
            ),
        }
        arrays = []
        for filename, (dtype, shape, values) in payloads.items():
            path = sample_dir / filename
            path.write_bytes(values.tobytes(order="C"))
            arrays.append(
                {
                    "file": filename,
                    "dtype": dtype,
                    "shape": shape,
                    "sha256": _sha(path),
                }
            )
        (sample_dir / "meta.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA_ID,
                    "positional_contract": POSITIONAL_CONTRACT,
                    "id": "synthetic",
                    "length": rows,
                    "bootstrap_rows_masked": 0,
                    "draft_vocab_size": DRAFT_VOCAB_SIZE,
                    "draft_mapping_sha256": "",
                    "arrays": arrays,
                }
            )
        )
        draft_mapping = np.arange(DRAFT_VOCAB_SIZE, dtype="<u4")
        mapping_path = root / "draft_to_target.u32le"
        mapping_path.write_bytes(draft_mapping.tobytes())
        mapping_sha = _sha(mapping_path)
        meta_path = sample_dir / "meta.json"
        metadata = json.loads(meta_path.read_text())
        metadata["draft_mapping_sha256"] = mapping_sha
        meta_path.write_text(json.dumps(metadata))
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA_ID,
                    "positional_contract": POSITIONAL_CONTRACT,
                    "endianness": "little",
                    "target_model": "synthetic.gguf",
                    "target_model_sha256": "00" * 32,
                    "target_quantization": "Q4_K_M",
                    "eagle3_checkpoint_sha256": "22" * 32,
                    "hidden_size": HIDDEN_SIZE,
                    "layer_input_ids": [2, 14, 25],
                    "auxiliary_width": AUX_WIDTH,
                    "target_vocab_size": TARGET_VOCAB_SIZE,
                    "draft_vocab_size": DRAFT_VOCAB_SIZE,
                    "draft_mapping_sha256": mapping_sha,
                    "draft_mapping": {
                        "file": "draft_to_target.u32le",
                        "dtype": "uint32",
                        "shape": [DRAFT_VOCAB_SIZE],
                        "sha256": mapping_sha,
                    },
                    "samples": [
                        {"id": "synthetic", "path": "samples/00000000", "length": rows}
                    ],
                }
            )
        )
        return sample_dir

    def test_reader_and_ttt_alignment_match_runtime_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_store(root)
            store = FeatureStore(root)
            sample = store.load(0)
            inverse = np.full(TARGET_VOCAB_SIZE, -1, dtype=np.int32)
            inverse[:DRAFT_VOCAB_SIZE] = np.arange(DRAFT_VOCAB_SIZE, dtype=np.int32)
            batch = build_ttt_batch(sample, inverse, ttt_length=3)
            self.assertEqual(batch.row_count, 9)
            self.assertEqual(float(batch.depths[0].embedding[0, 0]), 10.0)
            self.assertEqual(float(batch.depths[2].embedding[0, 0]), 12.0)
            self.assertEqual(int(batch.depths[0].target_ids[0]), 200)
            self.assertEqual(int(batch.depths[0].target_ids[3]), 203)
            self.assertEqual(int(batch.depths[2].target_ids[1]), 203)
            self.assertEqual(
                float(batch.depths[2].teacher_logits_at(np.array([1]))[0, 0]),
                23.0,
            )
            # The exported mask is already aligned to the token predicted by
            # each base teacher row; TTT shifts that runtime-aligned mask once
            # per recurrent depth.
            self.assertFalse(bool(batch.depths[0].supervised[2]))
            self.assertTrue(bool(batch.depths[0].supervised[3]))
            self.assertTrue(bool(batch.depths[2].supervised[1]))
            self.assertEqual(int(batch.depths[2].target_draft_ids[1]), 203)

    def test_unmapped_target_is_eligible_but_never_trainable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_store(root)
            store = FeatureStore(root)
            sample = store.load(0)
            inverse = np.full(TARGET_VOCAB_SIZE, -1, dtype=np.int32)
            inverse[:DRAFT_VOCAB_SIZE] = np.arange(DRAFT_VOCAB_SIZE, dtype=np.int32)
            inverse[203] = -1
            batch = build_ttt_batch(sample, inverse, ttt_length=3)
            self.assertTrue(bool(batch.depths[0].eligible[3]))
            self.assertFalse(bool(batch.depths[0].supervised[3]))
            self.assertFalse(bool(batch.depths[0].hard_supervised[3]))

    def test_reader_requires_two_runtime_mask_sentinels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = self._write_store(root)
            path = sample_dir / "loss_mask.u8"
            data = bytearray(path.read_bytes())
            data[-2] = 1
            path.write_bytes(data)
            store = FeatureStore(root, verify_hashes=False)
            with self.assertRaisesRegex(FeatureFormatError, "sentinel/masked"):
                store.load(0)

    def test_checksum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = self._write_store(root)
            path = sample_dir / "loss_mask.u8"
            data = bytearray(path.read_bytes())
            data[4] ^= 1
            path.write_bytes(data)
            store = FeatureStore(root)
            with self.assertRaisesRegex(FeatureFormatError, "SHA-256 mismatch"):
                store.load(0)

    def test_missing_positional_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_store(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["positional_contract"]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FeatureFormatError, "positional_contract"):
                FeatureStore(root)

    def test_bf16_round_to_nearest_even(self):
        values = np.array([1.0, -2.5, 0.0, np.pi], dtype=np.float32)
        restored = (float32_to_bf16_bits(values).astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_allclose(restored, values, rtol=8e-3, atol=1e-4)

    def test_soft_target_kl_matches_independent_numpy_reference(self):
        teacher = np.array([[1000.0, 1001.0, 999.0], [-2.0, 4.0, 0.5]])
        draft = np.array([[1000.5, 999.0, 998.0], [1.0, 2.0, -3.0]])
        shifted_teacher = teacher - teacher.max(axis=1, keepdims=True)
        teacher_p = np.exp(shifted_teacher)
        teacher_p /= teacher_p.sum(axis=1, keepdims=True)
        shifted_draft = draft - draft.max(axis=1, keepdims=True)
        draft_p = np.exp(shifted_draft)
        draft_p /= draft_p.sum(axis=1, keepdims=True)
        reference = np.mean(
            np.sum(teacher_p * (np.log(teacher_p) - np.log(draft_p)), axis=1)
        )
        actual = soft_target_kl_numpy(draft, teacher)
        self.assertAlmostEqual(actual, float(reference), places=12)
        ce_reference = np.mean(-np.sum(teacher_p * np.log(draft_p), axis=1))
        ce_actual = soft_target_cross_entropy_numpy(draft, teacher)
        self.assertAlmostEqual(ce_actual, float(ce_reference), places=12)

    @unittest.skipIf(mx is None, "MLX is not installed on this development host")
    def test_mlx_soft_target_kl_matches_numpy_reference(self):
        teacher = np.array([[8.0, -3.0, 1.5], [0.25, 0.5, -0.75]], dtype=np.float32)
        draft = np.array([[4.0, 2.0, -1.0], [-2.0, 1.0, 3.0]], dtype=np.float32)
        reference = soft_target_kl_numpy(draft, teacher)
        actual = soft_target_kl(mx.array(draft), mx.array(teacher))
        ce_actual = soft_target_cross_entropy(mx.array(draft), mx.array(teacher))
        ce_reference = soft_target_cross_entropy_numpy(draft, teacher)
        mx.eval(actual, ce_actual)
        self.assertAlmostEqual(float(actual.item()), float(reference), places=5)
        self.assertAlmostEqual(float(ce_actual.item()), float(ce_reference), places=5)


if __name__ == "__main__":
    unittest.main()
