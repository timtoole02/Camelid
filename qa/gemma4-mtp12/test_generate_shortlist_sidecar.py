#!/usr/bin/env python3
import contextlib
import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import generate_shortlist_sidecar as sidecar


class GeneratorTest(unittest.TestCase):
    def test_stable_grouped_sums_match_original_mask_reduction_bits(self):
        rng = np.random.default_rng(8439)
        values = rng.standard_normal((401, 1024)).astype(np.float32)
        values[::11] *= 1e9
        assignment = rng.integers(0, 19, size=len(values))
        expected = rng.standard_normal((23, 1024)).astype(np.float32)
        actual = expected.copy()
        counts = np.zeros(23, dtype=np.int64)
        for cluster in range(23):
            mask = assignment == cluster
            if mask.any():
                expected[cluster] += values[mask].sum(axis=0)
        sidecar.accumulate_stable_sums(values, assignment, actual, counts)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(counts, np.bincount(assignment, minlength=23))

    def test_seeded_training_matches_original_generator(self):
        weights = np.random.default_rng(89).standard_normal((97, 32)).astype(np.float32)
        normalized = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-8)
        clusters, chunk = 11, 23
        rng = np.random.default_rng(12345)
        centroids = normalized[rng.choice(len(weights), clusters, replace=False)].copy()
        for _ in range(3):
            sums = np.zeros_like(centroids)
            counts = np.zeros(clusters, dtype=np.int64)
            for first in range(0, len(weights), chunk):
                rows = normalized[first:first + chunk]
                assignment = (rows @ centroids.T).argmax(axis=1)
                for cluster in range(clusters):
                    mask = assignment == cluster
                    if mask.any():
                        sums[cluster] += rows[mask].sum(axis=0)
                        counts[cluster] += mask.sum()
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
            if (~nonempty).any():
                centroids[~nonempty] = normalized[rng.choice(len(weights), int((~nonempty).sum()), replace=False)]
            centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
        assignment = np.zeros((len(weights), 4), dtype=np.uint16)
        for first in range(0, len(weights), chunk):
            scores = normalized[first:first + chunk] @ centroids.T
            candidates = np.argpartition(-scores, 3, axis=1)[:, :3]
            for row, ids in enumerate(candidates):
                assignment[first + row, :3] = ids[np.argsort(-scores[row, ids])]
        raw_mean = np.zeros_like(centroids)
        for cluster in range(clusters):
            mask = assignment[:, 0] == cluster
            if mask.any():
                raw_mean[cluster] = weights[mask].sum(axis=0) / mask.sum()
        with contextlib.redirect_stdout(io.StringIO()):
            actual_mean, actual_assignment = sidecar.train(weights, clusters, 3, 12345, chunk)
        np.testing.assert_array_equal(actual_assignment, assignment)
        np.testing.assert_array_equal(actual_mean.view(np.uint32), raw_mean.view(np.uint32))

    def test_source_hash_is_checked_before_tensor_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.safetensors"
            path.write_bytes(b"not the official model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                sidecar.read_embedding(path)

    def test_header_length_endianness_and_invalid_write_preserves_destination(self):
        means = np.zeros((sidecar.K, sidecar.HID), dtype=np.float32)
        means[0, 0] = 1.25
        assignment = np.tile(np.array([0, 1, 2047, 0], dtype=np.uint16), (sidecar.VOCAB, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sidecar.c4sl"
            with contextlib.redirect_stdout(io.StringIO()):
                sidecar.write_sidecar(path, means, assignment)
            payload = path.read_bytes()
            self.assertEqual(len(payload), 10485816)
            self.assertEqual(payload[:8], b"C4SL\x01\x00\x00\x00")
            self.assertEqual(payload[8:40].hex(), sidecar.ASSISTANT_SHA256)
            self.assertEqual(struct.unpack_from("<IIII", payload, 40), (2048, 1024, 262144, 3))
            self.assertEqual(struct.unpack_from("<f", payload, 56), (1.25,))
            self.assertEqual(struct.unpack_from("<HHHH", payload, 56 + 2048 * 1024 * 4), (0, 1, 2047, 0))
            assignment[0, 1] = 0
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sidecar.write_sidecar(path, means, assignment)
            self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
