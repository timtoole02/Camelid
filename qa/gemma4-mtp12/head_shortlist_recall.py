#!/usr/bin/env python3
"""Offline recall experiment for a k-means head shortlist on the Gemma 4 12B
MTP assistant tied head.

  python3 head_shortlist_recall.py <model.safetensors> <draft-query dump> [k] [iters]

The dump is what CAMELID_MTP12_DUMP_DRAFT_QUERIES writes: per draft, u32 token
(the full-head argmax the GPU produced from Q4_0 rows), u32 step, u32 position,
u32 draft_k, then the 1024 f32 head query. Reports top-1 recall of the true
argmax token when only the rows of the top-T clusters (by centroid score) are
scored, for T in {8, 16, 32, 64, 128}, for both spherical and raw k-means and
both centroid scorings.

Result on mini2 (2026-09-04, 1,587 W8 draft queries from bench_fresh_40 --suite
full; the tied rows are near-unit norm 0.965..0.992, bf16 argmax == GPU Q4_0
argmax 97.9%): best variant (spherical k-means, raw-mean centroid score)
top-8 0.743 / top-16 0.810 / top-32 0.866 / top-64 0.912 / top-128 0.948 recall;
raw k-means 0.717 / 0.785 / 0.847 / 0.894 / 0.936. All far below the 97%
break-even, so the shortlisted assistant head was NOT implemented.
"""
import json, struct, sys, time
import numpy as np

path_st, dump = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 15
VOCAB, HID = 262144, 1024

with open(path_st, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    t = hdr["model.embed_tokens.weight"]
    assert t["dtype"] == "BF16" and t["shape"] == [VOCAB, HID], t
    a, b = t["data_offsets"]
    f.seek(8 + n + a)
    raw = f.read(b - a)
W16 = np.frombuffer(raw, dtype=np.uint16).reshape(VOCAB, HID)
W = (W16.astype(np.uint32) << 16).view(np.float32)  # exact bf16 -> f32
del raw, W16
norms = np.linalg.norm(W, axis=1)
print(f"embedding rows {W.shape}, row norm min/med/max {norms.min():.3f}/{np.median(norms):.3f}/{norms.max():.3f}", flush=True)

rec = np.dtype([("token", "<u4"), ("step", "<u4"), ("position", "<u4"), ("draft_k", "<u4"), ("h", "<f4", (HID,))])
Q = np.fromfile(dump, dtype=rec)
H = np.ascontiguousarray(Q["h"].astype(np.float32))
tok = Q["token"].astype(np.int64)
print(f"queries {len(Q)} (steps {np.bincount(Q['step']).tolist()})", flush=True)


def argmax_rows(H, W, chunk=64):
    out = np.empty(len(H), dtype=np.int64)
    for i in range(0, len(H), chunk):
        out[i:i + chunk] = (H[i:i + chunk] @ W.T).argmax(axis=1)
    return out


t0 = time.time()
am = argmax_rows(H, W)
print(f"bf16 full-head argmax == GPU Q4_0 argmax token: {(am == tok).mean():.4f} ({time.time() - t0:.1f}s)", flush=True)


def kmeans(X, k, iters, seed, spherical):
    rng = np.random.default_rng(seed)
    if spherical:
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    else:
        Xn = X
    C = Xn[rng.choice(len(Xn), k, replace=False)].copy()
    assign = np.zeros(len(Xn), dtype=np.int64)
    for it in range(iters):
        t0 = time.time()
        half = 0.5 * (C * C).sum(1)
        for i in range(0, len(Xn), 16384):
            S = Xn[i:i + 16384] @ C.T
            if not spherical:
                S -= half[None, :]
            assign[i:i + 16384] = S.argmax(1)
        order = np.argsort(assign, kind="stable")
        sorted_assign = assign[order]
        bounds = np.searchsorted(sorted_assign, np.arange(k))
        cnt = np.bincount(assign, minlength=k)
        sums = np.add.reduceat(Xn[order], np.minimum(bounds, len(Xn) - 1), axis=0)
        Cn = C.copy()
        nz = cnt > 0
        Cn[nz] = sums[nz] / cnt[nz, None]
        if (~nz).any():
            Cn[~nz] = Xn[rng.choice(len(Xn), int((~nz).sum()), replace=False)]
        if spherical:
            Cn /= np.maximum(np.linalg.norm(Cn, axis=1, keepdims=True), 1e-8)
        shift = float(np.abs(Cn - C).max())
        C = Cn
        print(f"  {'spherical' if spherical else 'raw'} k={k} iter {it}: shift {shift:.4g} empty {(~nz).sum()} max cluster {cnt.max()} ({time.time() - t0:.1f}s)", flush=True)
    # final assignment with the last centroids
    half = 0.5 * (C * C).sum(1)
    for i in range(0, len(Xn), 16384):
        S = Xn[i:i + 16384] @ C.T
        if not spherical:
            S -= half[None, :]
        assign[i:i + 16384] = S.argmax(1)
    cnt = np.bincount(assign, minlength=k)
    # raw-row mean per cluster (for the alternative centroid scoring)
    order = np.argsort(assign, kind="stable")
    bounds = np.searchsorted(assign[order], np.arange(k))
    raw_sums = np.add.reduceat(X[order], np.minimum(bounds, len(X) - 1), axis=0)
    raw_mean = np.zeros_like(C)
    nz = cnt > 0
    raw_mean[nz] = raw_sums[nz] / cnt[nz, None]
    return C, assign, cnt, raw_mean


def recall(name, Cscore, assign, cnt, targets):
    S = H @ Cscore.T  # (queries, k)
    order = np.argsort(-S, axis=1)
    res = {}
    for T in (8, 16, 32, 64, 128):
        top = order[:, :T]
        hit = (assign[targets][:, None] == top).any(axis=1)
        rows = cnt[top].sum(axis=1)
        res[T] = (float(hit.mean()), float(rows.mean()), int(rows.max()))
    line = "  ".join(f"top{T}: recall {r:.4f} rows {rows:.0f} (max {mx})" for T, (r, rows, mx) in res.items())
    print(f"{name}: {line}", flush=True)
    return res


results = {}
for spherical in (True, False):
    C, assign, cnt, raw_mean = kmeans(W, K, ITERS, 12345, spherical)
    label = "spherical" if spherical else "raw"
    print(f"{label} k-means: cluster sizes min/med/max {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}", flush=True)
    results[f"{label}/unit-centroid vs GPU token"] = recall(f"{label:9} unit-centroid score, GPU Q4_0 token ", C if spherical else C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-8), assign, cnt, tok)
    results[f"{label}/mean-centroid vs GPU token"] = recall(f"{label:9} raw-mean-centroid score, GPU token  ", raw_mean, assign, cnt, tok)
    results[f"{label}/mean-centroid vs bf16 argmax"] = recall(f"{label:9} raw-mean-centroid score, bf16 argmax", raw_mean, assign, cnt, am)
    # Oracle-ish variant: score clusters by the best row norm-weighted direction
    # (max over the cluster is not cheap online; skip) -- instead the row-count
    # weighted centroid is the mean above.
print(json.dumps({k: {str(T): v for T, v in r.items()} for k, r in results.items()}, indent=1))
