# Tensor Runtime Reverse Engineering

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

Source: GGUF/GGML-format recon focused on tensor descriptors, tensor type metadata, tensor byte-size validation, and the first CPU reference runtime needed by camelid.

## Summary

camelid should treat tensor work as a separate lane from GGUF metadata parsing and model inference. The immediate goal is not fast matrix kernels; it is a correct, validated CPU tensor representation that can load tensor payloads, recognize GGML dtypes, convert simple stored formats to f32, and reject unsupported quantization clearly.

The current camelid GGUF reader already parses tensor descriptors and validates byte ranges for common GGML types. Phase 4 should extend this into actual payload access and CPU tensor loading.

## Key GGUF/GGML concepts

Important source areas:

- `ggml/include/ggml.h` — tensor type enum, tensor struct concepts, max dims.
- `ggml/src/ggml.cpp` — type traits, block sizes, type sizes, tensor byte-size helpers.
- `ggml/src/gguf.cpp` — GGUF tensor descriptor parsing and data offsets.
- `src/llama-model-loader.cpp` — model tensor lookup, mmap/direct I/O, split file handling, backend buffer loading.
- backend kernels under `ggml/src/ggml-*` — dequantization and matmul implementations for accelerated backends.

Core concept: GGML stores tensors in typed blocks. For quantized types, dimensions are not simply `elements * scalar_size`; byte size is:

```text
nbytes = (element_count / block_size(type)) * type_size(type)
```

The first dimension must be divisible by the type block size for block-quantized types.

## Tensor descriptor validation rules

Mirror the important GGUF/GGML validation rules:

- tensor names must be unique
- tensor name length should stay below GGML max name length (`64`)
- dimensions count must be `1..=4`
- dimensions must be non-negative
- element count must not overflow
- tensor type id must be known and have nonzero block/type size
- first dimension must be divisible by block size
- tensor descriptor offsets should be contiguous in descriptor order
- tensor byte range must fit inside the file
- tensor absolute offset = aligned data start offset + descriptor relative offset

camelid already implements much of this in `src/gguf/reader.rs`; keep improving it with malformed fixture tests.

## DType mapping table

Initial mapping for descriptor validation and future loading:

| GGML id | camelid type | block size | type size bytes | Phase 4 action |
|---:|---|---:|---:|---|
| 0 | F32 | 1 | 4 | load directly |
| 1 | F16 | 1 | 2 | convert to f32 |
| 2 | Q4_0 | 32 | 18 | recognize; dequantize later |
| 3 | Q4_1 | 32 | 18 | recognize; dequantize later |
| 6 | Q5_0 | 32 | 22 | recognize; dequantize later |
| 7 | Q5_1 | 32 | 22 | recognize; dequantize later |
| 8 | Q8_0 | 32 | 34 | first quantized dequant candidate |
| 9 | Q8_1 | 32 | 36 | recognize; dequantize later |
| 10 | Q2_K | 256 | 84 | recognize; unsupported initially |
| 11 | Q3_K | 256 | 110 | recognize; unsupported initially |
| 12 | Q4_K | 256 | 144 | common; dequantize after f16/q8 path |
| 13 | Q5_K | 256 | 176 | recognize; unsupported initially |
| 14 | Q6_K | 256 | 210 | recognize; unsupported initially |
| 15 | Q8_K | 256 | 292 | recognize; unsupported initially |
| 24 | I8 | 1 | 1 | load only if needed |
| 25 | I16 | 1 | 2 | load only if needed |
| 26 | I32 | 1 | 4 | load only if needed |
| 27 | I64 | 1 | 8 | load only if needed |
| 28 | F64 | 1 | 8 | convert or reject initially |
| 30 | BF16 | 1 | 2 | convert to f32 |

Removed/deprecated GGML type ids should be rejected, not treated as unknown-but-loadable.

## Recommended Rust data model

Add `src/tensor/`:

```rust
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
}

pub struct TensorShape {
    pub dims: Vec<usize>,
}

pub struct CpuTensor {
    pub shape: TensorShape,
    pub dtype: RuntimeDType,
    pub data: Vec<f32>,
}

pub enum RuntimeDType {
    F32,
}
```

Keep stored dtype (`GgufTensorType`) distinct from runtime dtype (`RuntimeDType`). The first runtime should eagerly convert supported storage types to f32.

## Tensor storage access

Current parser reads the full GGUF file into memory for metadata. Phase 4 should not eagerly copy full model files. Add one of these paths:

1. Simple first pass: reopen file, seek to `absolute_offset`, read `n_bytes` for the requested tensor.
2. Better next pass: memory-map the file and borrow tensor byte slices.

Recommended API:

```rust
pub struct TensorStore {
    file: File,
    descriptors: HashMap<String, GgufTensorDescriptor>,
}

impl TensorStore {
    pub fn open(path: &Path, gguf: &GgufFile) -> Result<Self>;
    pub fn tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>>;
    pub fn load_cpu_f32(&mut self, name: &str) -> Result<CpuTensor>;
}
```

Later, switch `Vec<u8>` to borrowed mmap slices.

## Minimal conversion paths

### F32

Read little-endian `f32` values directly.

Validation:

- byte length is divisible by 4
- number of f32 values equals shape element count

### F16

Convert IEEE fp16 to f32. Use a small crate like `half` unless avoiding dependencies is important.

### BF16

Convert BF16 to f32 by shifting bits into the high 16 bits of an f32 representation.

### Q8_0 first quantized candidate

Q8_0 blocks are relatively simple compared with K-quants:

- block size: 32 values
- stored block size: 34 bytes
- scale: fp16
- quantized values: 32 signed i8 values
- dequantized value: `scale * q`

Implement Q8_0 dequantization before Q4_K/Q6_K. It is easier to test by hand.

### Q4_0 next candidate

Q4_0 is common and still simpler than K-quants:

- block size: 32 values
- scale: fp16
- 16 bytes of packed 4-bit values
- values are unpacked to signed-ish nibble range according to GGML behavior

Add only after Q8_0 and f16 tests are solid.

## CPU runtime primitives needed before inference

Start with shape-safe primitives:

- `matmul(a, b)` for 2D f32 tensors
- `add(a, b)` with exact-shape validation
- `mul(a, b)` elementwise
- `silu(x)`
- `rms_norm(x, weight, eps)`
- `softmax_last_dim(x)`
- `embedding_lookup(weight, token_ids)`
- `transpose_2d`
- reshape/view helpers or explicit indexing helpers

Do not optimize first. Keep implementations clear and heavily tested.

## Error handling

Add tensor-specific errors:

```rust
TensorNotFound(String)
UnsupportedTensorType(String)
InvalidTensorShape { name: String, expected: String, actual: Vec<u64> }
InvalidTensorData(String)
TensorOutOfBounds(String)
RuntimeShapeMismatch(String)
```

## Phase 4 implementation tasks

1. Add `src/tensor/mod.rs` and export it.
2. Move/bridge `GgufTensorType` into a tensor dtype abstraction.
3. Add `TensorShape` and `CpuTensor`.
4. Add `TensorStore` for payload reads by descriptor.
5. Add direct F32 loader.
6. Add F16/BF16 conversion.
7. Add Q8_0 dequantization.
8. Add tensor malformed tests:
   - byte-size mismatch
   - missing tensor
   - unsupported dtype
   - invalid shape/block divisibility
9. Add CPU primitive tests:
   - matmul
   - RMSNorm
   - SiLU
   - softmax
   - embedding lookup
10. Keep quantized model inference blocked until dequantized tensor loading is proven.

## Recommended tests

Payload/loading tests:

- load f32 tensor from synthetic GGUF fixture
- load f16 tensor with known values
- load bf16 tensor with known values
- q8_0 dequantizes one block correctly
- missing tensor returns `TensorNotFound`
- unsupported dtype returns `UnsupportedTensorType`

Runtime primitive tests:

- `matmul_2x2`
- `rms_norm_known_vector`
- `silu_known_values`
- `softmax_is_stable_for_large_logits`
- `embedding_lookup_returns_expected_rows`

## Performance notes for later

- Replace full-file reads with mmap before real model loading.
- Keep CPU reference runtime as correctness oracle even after SIMD/GPU paths exist.
- Add feature flags for accelerated backends later.
- Avoid `unsafe` until profiling proves it is needed and tests cover the path.
