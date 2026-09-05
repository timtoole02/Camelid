#!/usr/bin/env python3
"""Generate the pinned Gemma 4 12B assistant overlapping head shortlist.

Usage: generate_shortlist_sidecar.py model.safetensors output.c4sl

Version 1: C4SL magic, u32 version, 32-byte assistant SHA-256, u32 K/D/V/M,
[K,D] little-endian f32 raw-mean centroids, [V,4] little-endian u16 token
cluster IDs (three distinct IDs plus zero padding). Header length is 56 bytes.
The official source hash and fixed geometry are checked before training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time

import numpy as np

ASSISTANT_SHA256 = "67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6"
VOCAB, HID, K, M = 262144, 1024, 2048, 3
CHUNK = 16384


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def read_embedding(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != ASSISTANT_SHA256:
            raise ValueError(f"assistant SHA-256 {digest} != {ASSISTANT_SHA256}")
        stream.seek(0)
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len))
        tensor = header["model.embed_tokens.weight"]
        if tensor["dtype"] != "BF16" or tensor["shape"] != [VOCAB, HID]:
            raise ValueError(f"unexpected embedding geometry: {tensor}")
        start, end = tensor["data_offsets"]
        if end - start != VOCAB * HID * 2 or start < 0:
            raise ValueError("invalid embedding byte span")
        stream.seek(8 + header_len + start)
        raw = stream.read(end - start)
    if len(raw) != VOCAB * HID * 2:
        raise ValueError("truncated embedding")
    words = np.frombuffer(raw, dtype="<u2").reshape(VOCAB, HID)
    return (words.astype(np.uint32) << 16).view(np.float32)


def accumulate_stable_sums(values, assignment, sums, counts):
    """Accumulate each cluster in source row order, matching mask.sum(axis=0).

    Stable sorting replaces K separate full-chunk boolean scans. Keep NumPy's
    axis-zero sum on each same-shaped segment: reduceat can change float32
    addition order, which would change the seeded artifact on later iterations.
    """
    order = np.argsort(assignment, kind="stable")
    sorted_values = values[order]
    bounds = np.searchsorted(assignment[order], np.arange(len(sums) + 1))
    for cluster in np.flatnonzero(np.diff(bounds)):
        first, last = bounds[cluster:cluster + 2]
        sums[cluster] += sorted_values[first:last].sum(axis=0)
        counts[cluster] += last - first


def train(weights, clusters=K, iterations=15, seed=12345, chunk=CHUNK):
    rng = np.random.default_rng(seed)
    normalized = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-8)
    centroids = normalized[rng.choice(len(normalized), clusters, replace=False)].copy()
    for iteration in range(iterations):
        started = time.monotonic()
        sums = np.zeros_like(centroids)
        counts = np.zeros(clusters, dtype=np.int64)
        for first in range(0, len(normalized), chunk):
            rows = normalized[first:first + chunk]
            assignment = (rows @ centroids.T).argmax(axis=1)
            accumulate_stable_sums(rows, assignment, sums, counts)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            centroids[~nonempty] = normalized[rng.choice(len(normalized), int((~nonempty).sum()), replace=False)]
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
        print(f"iteration {iteration}: empty={int((~nonempty).sum())} max_cluster={counts.max()} seconds={time.monotonic()-started:.1f}", flush=True)
    assignment = np.zeros((len(weights), 4), dtype=np.uint16)
    for first in range(0, len(normalized), chunk):
        scores = normalized[first:first + chunk] @ centroids.T
        # Keep the original kth=3 / default quicksort policy for reproducibility.
        candidates = np.argpartition(-scores, M, axis=1)[:, :M]
        order = np.argsort(-np.take_along_axis(scores, candidates, axis=1), axis=1)
        assignment[first:first + len(scores), :M] = np.take_along_axis(candidates, order, axis=1)
    sums = np.zeros_like(centroids)
    counts = np.zeros(clusters, dtype=np.int64)
    # The original raw centroid pass sums each cluster over the whole matrix,
    # so do not split this pass into chunks and change its addition grouping.
    accumulate_stable_sums(weights, assignment[:, 0], sums, counts)
    raw_mean = np.zeros_like(centroids)
    nonempty = counts > 0
    raw_mean[nonempty] = sums[nonempty] / counts[nonempty, None]
    return raw_mean, assignment


def write_sidecar(path, centroids, assignment):
    if centroids.shape != (K, HID) or assignment.shape != (VOCAB, 4):
        raise ValueError("sidecar geometry does not match the pinned assistant")
    if not np.isfinite(centroids).all() or (assignment[:, :M] >= K).any():
        raise ValueError("nonfinite centroid or out-of-range cluster")
    ordered = np.sort(assignment[:, :M], axis=1)
    if (np.diff(ordered, axis=1) == 0).any() or assignment[:, 3].any():
        raise ValueError("duplicate cluster or nonzero padding")
    header = b"C4SL" + struct.pack("<I", 1) + bytes.fromhex(ASSISTANT_SHA256)
    header += struct.pack("<IIII", K, HID, VOCAB, M)
    assert len(header) == 56
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(header)
            stream.write(centroids.astype("<f4", copy=False).tobytes())
            stream.write(assignment.astype("<u2", copy=False).tobytes())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"wrote {path}: {path.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=positive_int, default=15)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    weights = read_embedding(args.model)
    centroids, assignment = train(weights, iterations=args.iterations, seed=args.seed)
    write_sidecar(args.output, centroids, assignment)


if __name__ == "__main__":
    main()
