"""Independent NumPy FP32 oracle for the pinned BF16 Qwen EAGLE checkpoint.

Usage: python scripts/eagle3-cuda-reference.py CHECKPOINT_DIRECTORY OUTPUT_JSON
No CUDA or Camelid implementation code is used to compute the reference.
Pass --q8 for the production Q8/128 draft-weight format.
"""
import hashlib
import json
import pathlib
import struct
import sys
import numpy as np

arguments = sys.argv[1:]
q8 = "--q8" in arguments
if q8:
    arguments.remove("--q8")
root, output = map(pathlib.Path, arguments)
path = root / "model.safetensors"
assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == "58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e"
with path.open("rb") as f:
    size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(size))
data = np.memmap(path, mode="r", dtype=np.uint8, offset=8 + size)

def tensor(name):
    spec = header[name]
    start, end = spec["data_offsets"]
    raw = data[start:end]
    if spec["dtype"] == "BF16":
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(spec["shape"])
    return raw.view({"I64": "<i8", "I32": "<i4", "F32": "<f4", "BOOL": "?"}[spec["dtype"]]).reshape(spec["shape"])

w = {name: tensor(name) for name in header if name != "__metadata__"}
if q8:
    for name, value in w.items():
        if value.ndim != 2:
            continue
        blocks = value.reshape(-1, 128)
        scales = np.max(np.abs(blocks), axis=1) / np.float32(127)
        safe_scales = np.where(scales == 0, np.float32(1), scales)
        quantized = np.clip(np.rint(blocks / safe_scales[:, None]), -127, 127).astype(np.int8)
        w[name] = (quantized.astype(np.float32) * scales[:, None]).reshape(value.shape)


def norm(x, weight):
    return x * np.float32(1 / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(1e-6))) * weight

def rope(x, position):
    angles = np.float32(position) * np.power(np.float32(1000000), -np.arange(64, dtype=np.float32) / np.float32(64))
    c, s = np.cos(angles), np.sin(angles)
    return np.concatenate((x[:, :64] * c - x[:, 64:] * s, x[:, 64:] * c + x[:, :64] * s), axis=1)

keys, values, rows = [], [], []
for row in range(2):
    features = np.sin((np.arange(7680, dtype=np.float64) + row * 7) * .013).astype(np.float32) * np.float32(.1)
    embedding = np.cos((np.arange(2560, dtype=np.float64) + row * 11) * .017).astype(np.float32) * np.float32(.09)
    g = w["fc.weight"] @ features
    combined = np.concatenate((norm(embedding, w["midlayer.input_layernorm.weight"]), norm(g, w["midlayer.hidden_norm.weight"])))
    q = rope((w["midlayer.self_attn.q_proj.weight"] @ combined).reshape(32, 128), row)
    keys.append(rope((w["midlayer.self_attn.k_proj.weight"] @ combined).reshape(8, 128), row))
    values.append((w["midlayer.self_attn.v_proj.weight"] @ combined).reshape(8, 128))
    context = np.empty((32, 128), dtype=np.float32)
    for h in range(32):
        scores = np.array([q[h] @ k[h // 4] for k in keys], dtype=np.float32) / np.float32(np.sqrt(128))
        probs = np.exp(scores - scores.max()); probs /= probs.sum()
        context[h] = sum(p * v[h // 4] for p, v in zip(probs, values))
    residual = g + w["midlayer.self_attn.o_proj.weight"] @ context.flatten()
    normalized = norm(residual, w["midlayer.post_attention_layernorm.weight"])
    gate = w["midlayer.mlp.gate_proj.weight"] @ normalized
    up = w["midlayer.mlp.up_proj.weight"] @ normalized
    raw = residual + w["midlayer.mlp.down_proj.weight"] @ ((gate / (1 + np.exp(-gate))) * up)
    logits = w["lm_head.weight"] @ norm(raw, w["norm.weight"])
    draft = int(logits.argmax())
    rows.append({"logits": logits.tolist(), "target_token": draft + int(w["d2t"][draft])})
output.write_text(json.dumps({"schema": "camelid.eagle3.cuda.numpy-reference.v1", "weight_format": "q8_128" if q8 else "bf16", "rows": rows}), encoding="utf-8")
print(json.dumps({"rows": len(rows), "target_tokens": [r["target_token"] for r in rows]}))
