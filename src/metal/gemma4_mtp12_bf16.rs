//! Default-off higher-precision assistant dense projections. The Q4 tied head
//! and all target arithmetic remain outside this module.
use super::*;

pub(super) const ENV: &str = "CAMELID_GEMMA4_MTP12_DENSE_BF16";

fn policy(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(invalid(format!(
            "{ENV} must be exactly 0 or 1; got {other:?}"
        ))),
    }
}

pub(super) fn enabled() -> Result<bool> {
    let value = std::env::var_os(ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| invalid(format!("{ENV} must be UTF-8 and exactly 0 or 1")))
        })
        .transpose()?;
    policy(value.as_deref())
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
inline float dense_bf16_round(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return as_type<float>((bits & 0xffff0000u) | 0x00400000u);
    return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
}
kernel void mtp12_dense_bf16_gemv(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& byte_offset [[buffer(5)]],
    constant uint& round_output [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4u + sg;
    if (row >= rows) return;
    device const ushort4* w = reinterpret_cast<device const ushort4*>(weights + byte_offset) + ulong(row) * (cols / 4u);
    device const float4* x = reinterpret_cast<device const float4*>(input);
    float4 acc = 0.0f;
    for (uint c = lane; c < cols / 4u; c += 32u) {
        const float4 values = as_type<float4>(uint4(w[c]) << 16u);
        acc = fma(values, x[c], acc);
    }
    const float sum = simd_sum(((acc.x + acc.y) + acc.z) + acc.w);
    if (lane == 0u) output[row] = round_output ? dense_bf16_round(sum) : sum;
}
"#;

pub(super) struct Bf16Dense {
    weights: Buffer,
    pipeline: ComputePipelineState,
    // The established Q4 descriptor identifies a dense matrix without changing
    // every caller's layout. Values point into the separate raw-BF16 buffer.
    offsets: BTreeMap<u64, u64>,
    pub(super) upload_us: u128,
    pub(super) pipeline_compile_us: u128,
}

impl Bf16Dense {
    pub(super) fn load(
        device: &Device,
        mapping: &GgufWireMmap,
        pairs: impl IntoIterator<Item = (TensorRef, Q4TensorRef)>,
        embedding: Q4TensorRef,
    ) -> Result<Self> {
        let started = Instant::now();
        let mut sources = Vec::new();
        let mut offsets = BTreeMap::new();
        let mut bytes = 0u64;
        for (source, packed) in pairs {
            if packed.byte_offset == embedding.byte_offset {
                continue;
            }
            if source.rows == 0
                || source.cols == 0
                || source.cols % 4 != 0
                || (source.rows, source.cols) != (packed.rows, packed.cols)
            {
                return Err(invalid("BF16 dense matrix shape is invalid"));
            }
            let length = u64::from(source.rows)
                .checked_mul(u64::from(source.cols))
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| invalid("BF16 dense matrix byte length overflow"))?;
            if offsets.insert(packed.byte_offset, bytes).is_some() {
                return Err(invalid("BF16 dense layout contains a duplicate matrix"));
            }
            sources.push((source, bytes, length));
            bytes = bytes
                .checked_add(length)
                .ok_or_else(|| invalid("BF16 dense layout overflow"))?;
        }
        if sources.is_empty() || bytes > usize::MAX as u64 {
            return Err(invalid(
                "BF16 dense layout is empty or exceeds address space",
            ));
        }
        let weights = shared_buffer(device, bytes as usize);
        for (source, destination, length) in sources {
            let raw = mapping.bytes(source.absolute_offset, length as usize)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    weights.contents().cast::<u8>().add(destination as usize),
                    length as usize,
                );
            }
        }
        let upload_us = started.elapsed().as_micros();
        let compile_started = Instant::now();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(SHADER, &options)
            .map_err(|e| invalid(format!("BF16 dense shader compilation failed: {e}")))?;
        let function = library
            .get_function("mtp12_dense_bf16_gemv", None)
            .map_err(|e| invalid(format!("BF16 dense shader function missing: {e}")))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| invalid(format!("BF16 dense pipeline failed: {e}")))?;
        Ok(Self {
            weights,
            pipeline,
            offsets,
            upload_us,
            pipeline_compile_us: compile_started.elapsed().as_micros(),
        })
    }

    pub(super) fn byte_len(&self) -> u64 {
        self.weights.length()
    }

    /// `output_byte_offset` lands the row vector inside a larger buffer.
    pub(super) fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        output_byte_offset: u64,
        matrix: Q4TensorRef,
        round_output_bf16: bool,
    ) {
        let offset = self
            .offsets
            .get(&matrix.byte_offset)
            .expect("every non-embedding matrix was admitted into the BF16 dense layout");
        let round = u32::from(round_output_bf16);
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), output_byte_offset);
        encoder.set_bytes(3, 4, &matrix.cols as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &matrix.rows as *const u32 as *const c_void);
        encoder.set_bytes(5, 8, offset as *const u64 as *const c_void);
        encoder.set_bytes(6, 4, &round as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(matrix.rows).div_ceil(4),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn dense_bf16_gate_is_default_off_and_fails_closed() {
        assert!(!policy(None).unwrap());
        assert!(!policy(Some("0")).unwrap());
        assert!(policy(Some(" 1 ")).unwrap());
        for invalid in ["", "true", "on", "2", "01"] {
            assert!(policy(Some(invalid)).is_err());
        }
    }

    #[test]
    fn dense_bf16_preserves_source_bits_shapes_and_output_rounding() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();
        // Four production input widths plus a short/tail fixture. Rows=5
        // exercises the final partial four-row threadgroup.
        for cols in [32usize, 1024, 4096, 7680, 8192] {
            let rows = 5usize;
            let input = (0..cols)
                .map(|i| ((i * 3 % 15) as f32 - 7.0) / 64.0)
                .collect::<Vec<_>>();
            let values = (0..rows * cols)
                .map(|i| ((i * 7 % 15) as f32 - 7.0) / 128.0)
                .collect::<Vec<_>>();
            let raw = values
                .iter()
                .flat_map(|&v| f32_to_bf16_rne_bits(v).to_le_bytes())
                .collect::<Vec<_>>();
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(&[0u8; 64]).unwrap();
            file.write_all(&raw).unwrap();
            file.write_all(&raw).unwrap();
            file.flush().unwrap();
            let mapping = GgufWireMmap::map(file.path()).unwrap();
            let embedding = Q4TensorRef {
                byte_offset: 0,
                byte_len: 18,
                rows: 1,
                cols: 32,
            };
            let packed = Q4TensorRef {
                byte_offset: 18,
                byte_len: (rows * cols / 32 * 18) as u64,
                rows: rows as u32,
                cols: cols as u32,
            };
            let second = Q4TensorRef {
                byte_offset: packed.byte_offset + packed.byte_len,
                ..packed
            };
            let source = TensorRef {
                absolute_offset: 64,
                rows: rows as u32,
                cols: cols as u32,
            };
            let dense = Bf16Dense::load(
                &device,
                &mapping,
                [
                    (
                        TensorRef {
                            absolute_offset: 0,
                            rows: 1,
                            cols: 32,
                        },
                        embedding,
                    ),
                    (source, packed),
                    (
                        TensorRef {
                            absolute_offset: 64 + raw.len() as u64,
                            ..source
                        },
                        second,
                    ),
                ],
                embedding,
            )
            .unwrap();
            assert_eq!(dense.byte_len(), (raw.len() * 2) as u64);
            assert!(!dense.offsets.contains_key(&embedding.byte_offset));
            assert_eq!(dense.offsets[&packed.byte_offset], 0);
            assert_eq!(dense.offsets[&second.byte_offset], raw.len() as u64);
            let copied = unsafe {
                std::slice::from_raw_parts(dense.weights.contents().cast::<u8>(), raw.len() * 2)
            };
            assert_eq!(&copied[..raw.len()], raw);
            assert_eq!(&copied[raw.len()..], raw);
            drop(mapping);
            drop(file); // Resident projections must not borrow the mmap.
            let input_buffer = f32_buffer(&device, &input).unwrap();
            let output_buffer = f32_buffer(&device, &vec![f32::NAN; rows + 3]).unwrap();
            for matrix in [packed, second] {
                for round in [false, true] {
                    let cb = queue.new_command_buffer();
                    let encoder = cb.new_compute_command_encoder();
                    dense.encode(encoder, &input_buffer, &output_buffer, 0, matrix, round);
                    encoder.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
                    let mut actual = vec![0.0f32; rows + 3];
                    read_buffer_f32(&output_buffer, &mut actual).unwrap();
                    for row in 0..rows {
                        // Dyadic inputs keep the complete dot exactly
                        // representable; this is an arithmetic oracle without
                        // duplicating the shader's accumulation order.
                        let expected = values[row * cols..(row + 1) * cols]
                            .iter()
                            .zip(&input)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum::<f64>() as f32;
                        let expected = if round {
                            bf16_bits_to_f32(f32_to_bf16_rne_bits(expected))
                        } else {
                            expected
                        };
                        assert_eq!(
                            actual[row].to_bits(),
                            expected.to_bits(),
                            "cols={cols}, row={row}, round={round}"
                        );
                    }
                    assert!(
                        actual[rows..].iter().all(|x| x.is_nan()),
                        "tail rows must not write outside output"
                    );
                }
            }
        }
    }
}
