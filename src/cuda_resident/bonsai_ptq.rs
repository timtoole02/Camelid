//! Packed Bonsai 2 projections and device-side folded-weight transforms.
use super::*;
use crate::runnable::hadamard::Transform;
use std::sync::Mutex;

struct ResidentTransform {
    metadata: Arc<Transform>,
    width: usize,
    signs: CudaSlice<f32>,
    scratch: Mutex<CudaSlice<f32>>,
}

pub(super) struct BonsaiKernels {
    transform: CudaFunction,
    matmul: CudaFunction,
    ptq_tiles: [CudaFunction; 5],
    f32_tiles: [CudaFunction; 3],
    bf16_tiles: [CudaFunction; 3],
    legacy_projection: bool,
    pub(super) embed: CudaFunction,
    // Destroy captured graphs before the transform buffers they reference.
    body_graphs: Vec<(u32, u32, SendCudaGraph)>,
    transforms: Vec<ResidentTransform>,
}

impl BonsaiKernels {
    pub(super) fn has_graphs(&self) -> bool {
        !self.body_graphs.is_empty()
    }

    pub(super) fn new(
        load: &impl Fn(&str) -> Result<CudaFunction, String>,
    ) -> Result<Self, String> {
        Ok(Self {
            transform: load("bonsai_hadamard")?,
            matmul: load("bonsai_ptq_matmul")?,
            ptq_tiles: [
                load("bonsai_ptq_t1_r2")?,
                load("bonsai_ptq_t1_r4")?,
                load("bonsai_ptq_t2_r4")?,
                load("bonsai_ptq_t4_r4")?,
                load("bonsai_ptq_t8_r4")?,
            ],
            f32_tiles: [
                load("bonsai_f32_t1")?,
                load("bonsai_f32_t4")?,
                load("bonsai_f32_t8")?,
            ],
            bf16_tiles: [
                load("bonsai_bf16_t1")?,
                load("bonsai_bf16_t4")?,
                load("bonsai_bf16_t8")?,
            ],
            // Retain a reproducible A/B control without changing model loading,
            // graph capture, sampling, or placement between measurements.
            legacy_projection: std::env::var("CAMELID_BONSAI2_CUDA_PROJECTION").as_deref()
                == Ok("baseline"),
            embed: load("embed_gather_ptq1_0")?,
            body_graphs: Vec::new(),
            transforms: Vec::new(),
        })
    }

    fn register(
        &mut self,
        stream: &Arc<CudaStream>,
        metadata: Arc<Transform>,
        width: usize,
    ) -> Result<usize, String> {
        if let Some(id) = self
            .transforms
            .iter()
            .position(|t| t.width == width && t.metadata == metadata)
        {
            return Ok(id);
        }
        let (block, signs, _) = metadata.cuda_parts();
        if block > 4096
            || width == 0
            || !width.is_multiple_of(block)
            || (!signs.is_empty() && signs.len() != width)
        {
            return Err("unsupported Bonsai CUDA Hadamard geometry".into());
        }
        let sign_values: &[f32] = if signs.is_empty() { &[1.0] } else { signs };
        let signs = stream
            .clone_htod(sign_values)
            .map_err(|e| format!("Bonsai CUDA signs: {e}"))?;
        // Scratch is shared only by this engine's ordered compute stream. The
        // offload stream transfers weights and never touches activations.
        let scratch = stream
            .alloc_zeros::<f32>(width * MAX_PRISM_PREFILL_K)
            .map_err(|e| format!("Bonsai CUDA transform scratch: {e}"))?;
        let id = self.transforms.len();
        self.transforms.push(ResidentTransform {
            metadata,
            width,
            signs,
            scratch: Mutex::new(scratch),
        });
        Ok(id)
    }

    #[allow(clippy::too_many_arguments)]
    fn rotate(
        &self,
        stream: &Arc<CudaStream>,
        transform: &ResidentTransform,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        tokens: usize,
        inverse: bool,
    ) -> Result<(), cudarc::driver::DriverError> {
        let (block, signs, permutation) = transform.metadata.cuda_parts();
        let (hd, nk, rep) = permutation.unwrap_or((0, 0, 0));
        let args = [
            transform.width as i32,
            block as i32,
            signs.len() as i32,
            hd as i32,
            nk as i32,
            rep as i32,
            i32::from(inverse),
        ];
        let mut launch = stream.launch_builder(&self.transform);
        launch
            .arg(input)
            .arg(&transform.signs)
            .arg(output)
            .arg(&args[0])
            .arg(&args[1])
            .arg(&args[2])
            .arg(&args[3])
            .arg(&args[4])
            .arg(&args[5])
            .arg(&args[6]);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((transform.width / block) as u32, tokens as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (block * 4) as u32,
            })
        }
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(
        &self,
        stream: &Arc<CudaStream>,
        lane: ProjQuant,
        input: &CudaSlice<f32>,
        weights: &CudaView<u8>,
        rows: usize,
        cols: usize,
        tokens: usize,
        output: &mut CudaSlice<f32>,
        residual: i32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let (transform_id, dense) = match lane {
            ProjQuant::Ptq1_0(id) => (Some(id), 0_i32),
            ProjQuant::F32 => (None, 1),
            ProjQuant::BF16 => (None, 2),
            _ => unreachable!("Bonsai projection received another quant family"),
        };
        let mut scratch = transform_id.map(|id| {
            self.transforms[id]
                .scratch
                .lock()
                .expect("Bonsai transform scratch poisoned")
        });
        if let Some(id) = transform_id {
            let transform = &self.transforms[id];
            debug_assert_eq!(cols, transform.width);
            debug_assert!(tokens <= MAX_PRISM_PREFILL_K);
            self.rotate(
                stream,
                transform,
                input,
                scratch.as_mut().unwrap(),
                tokens,
                false,
            )?;
        }
        let activation = scratch.as_deref().unwrap_or(input);
        let args = [rows as i32, cols as i32, residual, dense, tokens as i32];
        let (kernel, token_tile, row_tile) = if self.legacy_projection {
            (&self.matmul, 8, 4)
        } else if dense == 0 {
            match tokens {
                1 if rows >= 32768 => (&self.ptq_tiles[1], 1, 16),
                1 => (&self.ptq_tiles[0], 1, 8),
                2 => (&self.ptq_tiles[2], 2, 16),
                3..=4 => (&self.ptq_tiles[3], 4, 16),
                _ => (&self.ptq_tiles[4], 8, 16),
            }
        } else {
            let tiles = if dense == 1 {
                &self.f32_tiles
            } else {
                &self.bf16_tiles
            };
            match tokens {
                1 => (&tiles[0], 1, 4),
                2..=4 => (&tiles[1], 4, 4),
                _ => (&tiles[2], 8, 4),
            }
        };
        let mut launch = stream.launch_builder(kernel);
        launch
            .arg(activation)
            .arg(weights)
            .arg(output)
            .arg(&args[0])
            .arg(&args[1])
            .arg(&args[2])
            .arg(&args[3])
            .arg(&args[4]);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    (rows as u32).div_ceil(row_tile),
                    (tokens as u32).div_ceil(token_tile),
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
    }

    pub(super) fn inverse_embedding(
        &self,
        stream: &Arc<CudaStream>,
        id: usize,
        embedding: &mut CudaSlice<f32>,
        width: usize,
    ) -> Result<(), cudarc::driver::DriverError> {
        let transform = &self.transforms[id];
        let mut scratch = transform
            .scratch
            .lock()
            .expect("Bonsai transform scratch poisoned");
        self.rotate(stream, transform, embedding, &mut scratch, 1, true)?;
        stream.memcpy_dtod(&scratch.slice(..width), &mut embedding.slice_mut(..width))
    }
}

impl CudaResidentDecode {
    pub(super) fn bonsai_graph_eligible(&self, position: usize) -> bool {
        let disabled = matches!(
            std::env::var("CAMELID_BONSAI2_CUDA_NO_GRAPH")
                .ok()
                .as_deref(),
            Some("1" | "true" | "on" | "yes")
        );
        !disabled
            && qwen35_device_graphs_enabled()
            && cuda_manual_stream_order_enabled()
            && matches!(self.output_quant, ProjQuant::Ptq1_0(_))
            && self.qwen35.is_some()
            && position < SPLITK_THRESHOLD
            && self.kv_slot_count() == 1
            && self.offload.is_none()
            && self.overlap.is_none()
            && self.gemma3.is_none()
            && std::env::var_os("CAMELID_LAYER_DUMP").is_none()
    }

    /// Capture the unchanged PTQ forward with the live attention group count.
    /// The physical KV position remains a device input. At the split-K boundary
    /// eligibility returns false, so scalar and speculative verification keep
    /// exactly the same reduction and recurrent-state arithmetic.
    pub(super) fn forward_bonsai_graphed(
        &mut self,
        host_inputs: Option<(&[f32], &[f32], &[f32])>,
        position: usize,
        scale: f32,
    ) -> Result<(), String> {
        use cudarc::driver::sys;
        debug_assert!(self.bonsai_graph_eligible(position));
        let map = |step: &'static str| {
            move |error: cudarc::driver::DriverError| format!("Bonsai CUDA graph {step}: {error}")
        };
        let stream = self.k.stream.clone();
        if let Some((embedding, cos, sin)) = host_inputs {
            stream
                .memcpy_htod(embedding, &mut self.d_hidden)
                .map_err(map("embedding"))?;
            stream
                .memcpy_htod(cos, &mut self.d_cos)
                .map_err(map("cos"))?;
            stream
                .memcpy_htod(sin, &mut self.d_sin)
                .map_err(map("sin"))?;
            stream
                .memcpy_htod(&[position as i32], &mut self.d_position)
                .map_err(map("position"))?;
        }
        let groups = attention_decode_groups(self.head_dim, position + 1);
        let scale_bits = scale.to_bits();
        let index = self
            .k
            .bonsai
            .body_graphs
            .iter()
            .position(|(g, s, _)| *g == groups && *s == scale_bits);
        let index = if let Some(index) = index {
            index
        } else {
            stream.synchronize().map_err(map("pre-capture sync"))?;
            stream
                .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(map("begin"))?;
            // Inputs have already been uploaded or gathered. graph_capture=true
            // skips staging; device_inputs=false retains LIVE attention geometry
            // instead of the older Q1 device graph's max_pos geometry.
            let recorded = self.forward_pass(&[], &[], &[], position, scale, true, true, false);
            let captured = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            recorded?;
            let graph = captured
                .map_err(map("end"))?
                .ok_or("Bonsai graph capture produced no graph")?;
            graph.upload().map_err(map("upload"))?;
            if self.k.bonsai.body_graphs.is_empty() {
                eprintln!(
                    "[qwen35] Bonsai 2 CUDA graph active (live attention groups, packed PTQ)"
                );
            }
            self.k
                .bonsai
                .body_graphs
                .push((groups, scale_bits, SendCudaGraph(graph)));
            if std::env::var_os("CAMELID_RESIDENT_TRACE").is_some() {
                eprintln!("[cuda-graph] Bonsai 2 PTQ forward captured with {groups} live attention groups");
            }
            self.k.bonsai.body_graphs.len() - 1
        };
        self.k.bonsai.body_graphs[index]
            .2
             .0
            .launch()
            .map_err(map("replay"))
    }

    pub(crate) fn register_bonsai_transform(
        &mut self,
        transform: Arc<Transform>,
        width: usize,
    ) -> Result<ProjQuant, String> {
        // Transform scratch may feed several projections, so preserve the
        // engine's single ordered compute stream even with a global override.
        self.overlap = None;
        let id = self.k.bonsai.register(&self.k.stream, transform, width)?;
        Ok(ProjQuant::Ptq1_0(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonsai_upload_borrows_unchanged_packed_bytes() {
        for (lane, length) in [
            (ProjQuant::Ptq1_0(0), 28),
            (ProjQuant::BF16, 256),
            (ProjQuant::F32, 512),
        ] {
            let bytes: Vec<u8> = (0..length).map(|value| (value * 31) as u8).collect();
            let repacked = repack_for_lane(&bytes, lane, 1, 128, false).unwrap();
            assert!(matches!(repacked, std::borrow::Cow::Borrowed(_)));
            assert_eq!(repacked.as_ptr(), bytes.as_ptr());
            assert_eq!(repacked.as_ref(), bytes);
        }
    }

    fn warp_dot(weights: &[f32], input: &[f32]) -> f32 {
        let mut partial = [0.0_f32; 32];
        for (lane, sum) in partial.iter_mut().enumerate() {
            for column in (lane..weights.len()).step_by(32) {
                *sum += weights[column] * input[column];
            }
        }
        for step in [16, 8, 4, 2, 1] {
            for lane in 0..32 - step {
                partial[lane] += partial[lane + step];
            }
        }
        partial[0]
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn bonsai_cuda_projection_tiles_match_legacy_bits() {
        let mut kernels = CudaResidentKernels::new().expect("CUDA kernel compilation");
        let stream = kernels.stream.clone();
        // Include partial row/token tiles, the vocabulary dispatch threshold,
        // representative widths, residual addition, and finite half-scale extremes.
        for (rows, width, dense) in [
            (1, 128, 0),
            (7, 2048, 0),
            (33, 4096, 0),
            (7, 6144, 0),
            (7, 8192, 0),
            (32769, 128, 0),
            (7, 96, 1),
            (7, 5120, 1),
            (7, 96, 2),
            (7, 5120, 2),
        ] {
            let (lane, weights) = if dense == 0 {
                let metadata = Arc::new(Transform::test_rotation_geometry(width, 128, None));
                let id = kernels.bonsai.register(&stream, metadata, width).unwrap();
                let scales = [0_u16, 0x8000, 1, 0x03ff, 0x0400, 0x3000, 0xb555, 0x7bff];
                let mut bytes = Vec::with_capacity(rows * width / 128 * 28);
                for b in 0..rows * width / 128 {
                    bytes.extend((0..26).map(|i| (b * 17 + i * 31) as u8));
                    bytes.extend_from_slice(&scales[b % scales.len()].to_le_bytes());
                }
                (ProjQuant::Ptq1_0(id), bytes)
            } else {
                let values = (0..rows * width).map(|i| ((i * 29 % 257) as f32 - 128.0) / 37.0);
                if dense == 1 {
                    (ProjQuant::F32, values.flat_map(f32::to_le_bytes).collect())
                } else {
                    (
                        ProjQuant::BF16,
                        values
                            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                            .collect(),
                    )
                }
            };
            let device_weights = stream.clone_htod(&weights).unwrap();
            for tokens in [1, 2, 3, 4, 5, 8, 9] {
                let input: Vec<_> = (0..width * tokens)
                    .map(|i| ((i * 13 % 113) as f32 - 56.0) / 31.0)
                    .collect();
                let device_input = stream.clone_htod(&input).unwrap();
                let initial: Vec<_> = (0..rows * tokens).map(|i| (i % 53) as f32 / 19.0).collect();
                for residual in [0, 1] {
                    let mut results = Vec::new();
                    for legacy in [true, false] {
                        kernels.bonsai.legacy_projection = legacy;
                        let mut output = stream.clone_htod(&initial).unwrap();
                        kernels
                            .bonsai
                            .project(
                                &stream,
                                lane,
                                &device_input,
                                &device_weights.as_view(),
                                rows,
                                width,
                                tokens,
                                &mut output,
                                residual,
                            )
                            .unwrap();
                        results.push(
                            stream
                                .clone_dtoh(&output)
                                .unwrap()
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>(),
                        );
                    }
                    assert_eq!(results[0], results[1], "rows={rows} width={width} dense={dense} tokens={tokens} residual={residual}");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn bonsai_cuda_bf16_projection_matches_warp_reference() {
        let kernels = CudaResidentKernels::new().expect("CUDA kernel compilation");
        let stream = kernels.stream.clone();
        for width in [96, 5120] {
            let rows = 7;
            let tokens = 9; // Covers a full tile and the partial final tile.
            let input: Vec<f32> = (0..width * tokens)
                .map(|i| ((i * 13 % 113) as f32 - 56.0) / 31.0)
                .collect();
            let bits: Vec<u16> = (0..rows * width)
                .map(|i| ((((i * 29 % 257) as f32 - 128.0) / 37.0).to_bits() >> 16) as u16)
                .collect();
            let weights: Vec<u8> = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
            let decoded: Vec<f32> = bits
                .iter()
                .map(|&value| f32::from_bits(u32::from(value) << 16))
                .collect();
            let device_input = stream.clone_htod(&input).unwrap();
            let device_weights = stream.clone_htod(&weights).unwrap();
            let initial: Vec<f32> = (0..rows * tokens).map(|i| i as f32 / 19.0).collect();
            for residual in [0, 1] {
                let mut output = stream.clone_htod(&initial).unwrap();
                kernels
                    .bonsai
                    .project(
                        &stream,
                        ProjQuant::BF16,
                        &device_input,
                        &device_weights.as_view(),
                        rows,
                        width,
                        tokens,
                        &mut output,
                        residual,
                    )
                    .unwrap();
                let batched = stream.clone_dtoh(&output).unwrap();
                for token in 0..tokens {
                    let single_input = stream
                        .clone_htod(&input[token * width..(token + 1) * width])
                        .unwrap();
                    let mut single_output = stream
                        .clone_htod(&initial[token * rows..(token + 1) * rows])
                        .unwrap();
                    kernels
                        .bonsai
                        .project(
                            &stream,
                            ProjQuant::BF16,
                            &single_input,
                            &device_weights.as_view(),
                            rows,
                            width,
                            1,
                            &mut single_output,
                            residual,
                        )
                        .unwrap();
                    let single = stream.clone_dtoh(&single_output).unwrap();
                    for row in 0..rows {
                        let dot = warp_dot(
                            &decoded[row * width..(row + 1) * width],
                            &input[token * width..(token + 1) * width],
                        );
                        let expected = if residual == 0 {
                            dot
                        } else {
                            initial[token * rows + row] + dot
                        };
                        assert_eq!(
                            single[row].to_bits(),
                            expected.to_bits(),
                            "BF16 width={width}, row={row}, token={token}, residual={residual}"
                        );
                        assert_eq!(
                            single[row].to_bits(),
                            batched[token * rows + row].to_bits(),
                            "BF16 single/batch"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn bonsai_cuda_transform_projection_and_embedding() {
        let mut kernels = CudaResidentKernels::new().expect("CUDA kernel compilation");
        let stream = kernels.stream.clone();
        for (block, width, permutation) in [
            (512, 2048, None),
            (512, 4096, Some((128, 8, 4))),
            (1024, 6144, Some((128, 16, 3))),
            (4096, 8192, None),
        ] {
            let metadata = Arc::new(Transform::test_rotation_geometry(width, block, permutation));
            let id = kernels
                .bonsai
                .register(&stream, metadata.clone(), width)
                .unwrap();
            let rows = 7;
            let tokens = 9;
            let input: Vec<f32> = (0..width * tokens)
                .map(|i| ((i * 13 % 113) as f32 - 56.0) / 32.0)
                .collect();
            let device_input = stream.clone_htod(&input).unwrap();
            let transform = &kernels.bonsai.transforms[id];
            let mut rotated = stream.alloc_zeros::<f32>(input.len()).unwrap();
            kernels
                .bonsai
                .rotate(
                    &stream,
                    transform,
                    &device_input,
                    &mut rotated,
                    tokens,
                    false,
                )
                .unwrap();
            let got_rotation = stream.clone_dtoh(&rotated).unwrap();
            let expected_rotation: Vec<_> = input
                .chunks_exact(width)
                .flat_map(|row| metadata.forward(row))
                .collect();
            assert_eq!(
                got_rotation, expected_rotation,
                "forward FWHT block={block}"
            );

            // Exercise every packed byte and each interleaving group with
            // independent CPU decoding, including all-positive/all-negative codes.
            let mut weights = Vec::with_capacity(rows * width / 128 * 28);
            let mut decoded = Vec::with_capacity(rows * width);
            for index in 0..rows * width / 128 {
                let bytes: Vec<_> = (0..26).map(|i| (index * 17 + i * 31) as u8).collect();
                weights.extend_from_slice(&bytes);
                weights.extend_from_slice(&0x3000_u16.to_le_bytes()); // 0.125
                for (start, count, powers) in [(0, 16, 5), (16, 8, 5), (24, 2, 4)] {
                    let mut power = 1_u8;
                    for _ in 0..powers {
                        for &byte in &bytes[start..start + count] {
                            let digit = ((u16::from(byte.wrapping_mul(power)) * 3) >> 8) as i32 - 1;
                            decoded.push(digit as f32 * 0.125);
                        }
                        power = power.wrapping_mul(3);
                    }
                }
            }
            let device_weights = stream.clone_htod(&weights).unwrap();
            let mut output = stream.alloc_zeros::<f32>(rows * tokens).unwrap();
            kernels
                .bonsai
                .project(
                    &stream,
                    ProjQuant::Ptq1_0(id),
                    &device_input,
                    &device_weights.as_view(),
                    rows,
                    width,
                    tokens,
                    &mut output,
                    0,
                )
                .unwrap();
            let batched = stream.clone_dtoh(&output).unwrap();
            for token in 0..tokens {
                let single_input = stream
                    .clone_htod(&input[token * width..(token + 1) * width])
                    .unwrap();
                let mut single_output = stream.alloc_zeros::<f32>(rows).unwrap();
                kernels
                    .bonsai
                    .project(
                        &stream,
                        ProjQuant::Ptq1_0(id),
                        &single_input,
                        &device_weights.as_view(),
                        rows,
                        width,
                        1,
                        &mut single_output,
                        0,
                    )
                    .unwrap();
                let single = stream.clone_dtoh(&single_output).unwrap();
                assert_eq!(
                    single,
                    batched[token * rows..(token + 1) * rows],
                    "single/batch PTQ"
                );
                for row in 0..rows {
                    let expected = warp_dot(
                        &decoded[row * width..(row + 1) * width],
                        &expected_rotation[token * width..(token + 1) * width],
                    );
                    assert_eq!(single[row].to_bits(), expected.to_bits(), "PTQ row {row}");
                }
            }
            let token = stream.clone_htod(&[3_u32]).unwrap();
            let mut embedding = stream.alloc_zeros::<f32>(width).unwrap();
            launch_embed_gather(
                &stream,
                &kernels.bonsai.embed,
                &device_weights,
                &token.as_view(),
                width,
                &mut embedding,
            )
            .unwrap();
            kernels
                .bonsai
                .inverse_embedding(&stream, id, &mut embedding, width)
                .unwrap();
            let actual = stream.clone_dtoh(&embedding).unwrap();
            let mut expected = decoded[3 * width..4 * width].to_vec();
            metadata.inverse(&mut expected);
            assert_eq!(actual, expected, "embedding inverse FWHT block={block}");
        }
    }
}
