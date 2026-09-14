// Isolated Llama-3.2-3B diagnostic. Deliberately not part of ResidentDecodeState:
// normal decode/verify caps, arithmetic, weight caches, and tree paths are untouched.
// Defined by include! in metal.rs; uses its existing native kernels and helpers.

include!("metal_fp16_sg.rs");
include!("metal_fp16_attention_mm.rs");
include!("metal_fp16_gate_up.rs");

const FP16_PROBE_HIDDEN: usize = 3072;
const FP16_PROBE_KV: usize = 1024;
const FP16_PROBE_VOCAB: usize = 128256;
const FP16_PROBE_LAYERS: usize = 28;
const FP16_PROBE_MAX_ROWS: usize = 64;
const FP16_PROBE_MAX_CONTEXT: usize = 2048;

struct Fp16ProbeLayer {
    projections: [Buffer; 7],
    attn_norm: Buffer,
    ffn_norm: Buffer,
}

/// Experimental target: canonical Q4_K/Q6_K weights rounded once to half,
/// half activations, native MMA with f32 accumulation, f32 residual/norm/SiLU,
/// and the existing per-row attention with its own f16 KV. This is NOT the V4
/// arithmetic target. All prompt, serial-reference, and wide calls use this
/// same object. Construction prewarms mirrors outside timed forwards.
///
/// Admission is deliberately restricted to the measured 3B geometry and a
/// 2048-position context. Mirrors use 6,425,149,440 bytes; f16 KV at the maximum
/// uses 234,881,024 bytes, in addition to canonical model storage. Attention's
/// retained per-row partials can add roughly 716 MB at width64/context2048.
/// There is no serving route or automatic acceptance. Diagnostic callers own
/// acceptance and explicitly truncate rejected KV suffixes.
pub struct ResidentFp16Probe {
    sg: Option<&'static Fp16SgKernels>,
    gate_up_silu: Option<&'static ComputePipelineState>,
    layers: Vec<Fp16ProbeLayer>,
    output: Buffer,
    final_norm: Buffer,
    cache_k: Vec<Buffer>,
    cache_v: Vec<Buffer>,
    max_positions: usize,
    filled: usize,
    source_key: usize,
    eps: f32,
    split_half_pairing: bool,
    projection_count: u64,
    actual_matrix_dispatch_count: u64,
    mirror_bytes: u64,
    setup_ms: f64,
    attention_chunk: usize,
    attention_batch_chunks: u64,
    attention_batch_rows: u64,
    attention_mm: bool,
    attention_mm_layers: u64,
    attention_mm_peak_scratch_bytes: u64,
    attention_mm_peak_physical_rows: usize,
}

fn fp16_probe_attention_chunk() -> usize {
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(
        || match std::env::var("CAMELID_BENCH_FP16_ATTN_BATCH").as_deref() {
            Ok("1" | "8") => 8,
            Ok("16") => 16,
            _ => 0,
        },
    )
}

pub(crate) fn fp16_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_BENCH_FP16_VERIFY")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

impl ResidentFp16Probe {
    /// Reset logical position without rebuilding weights. All calls are
    /// synchronous, so no outstanding GPU work references a previous epoch.
    /// Later attention reads only the rewritten prefix below `filled`.
    pub fn reset(&mut self) {
        self.filled = 0;
    }
    /// Roll back completed speculative input rows. Forward calls are synchronous;
    /// stale suffix slots remain unread until a later forward overwrites them.
    /// A truncation may never advance the logical KV watermark.
    pub fn truncate(&mut self, filled: usize) -> bool {
        if filled > self.filled {
            return false;
        }
        self.filled = filled;
        true
    }

    pub fn filled(&self) -> usize {
        self.filled
    }
    pub fn projection_count(&self) -> u64 {
        self.projection_count
    }
    /// Logical weighted projections; fused gate/up still represents two.
    pub fn logical_projection_count(&self) -> u64 {
        self.projection_count
    }
    /// Completed weighted-projection GEMM dispatches. Attention GEMMs are
    /// accounted separately by attention route counters.
    pub fn actual_matrix_dispatch_count(&self) -> u64 {
        self.actual_matrix_dispatch_count
    }
    pub fn gate_up_fused(&self) -> bool {
        self.gate_up_silu.is_some()
    }
    pub fn mirror_bytes(&self) -> u64 {
        self.mirror_bytes
    }
    pub fn setup_ms(&self) -> f64 {
        self.setup_ms
    }
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }
    pub fn projection_backend(&self) -> &'static str {
        if self.sg.is_some() {
            "sg-packed-8xnt"
        } else {
            "native-64x64"
        }
    }
    pub fn attention_chunk(&self) -> usize {
        self.attention_chunk
    }
    pub fn attention_batch_chunks(&self) -> u64 {
        self.attention_batch_chunks
    }
    pub fn attention_batch_rows(&self) -> u64 {
        self.attention_batch_rows
    }
    pub fn attention_backend(&self) -> &'static str {
        if self.attention_mm {
            "absolute-aligned-half-mma"
        } else if self.attention_chunk > 0 {
            "existing-splitk-chunks-with-row-fallback"
        } else {
            "existing-per-row"
        }
    }
    pub fn attention_mm_layers(&self) -> u64 {
        self.attention_mm_layers
    }
    pub fn attention_mm_peak_scratch_bytes(&self) -> u64 {
        self.attention_mm_peak_scratch_bytes
    }
    pub fn attention_mm_peak_physical_rows(&self) -> usize {
        self.attention_mm_peak_physical_rows
    }
    pub(crate) fn has_source(&self, key: usize) -> bool {
        self.source_key == key
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        layers: &[ResidentLayerWeights],
        logits: &LogitsStage,
        max_positions: usize,
        eps: f32,
        split_half_pairing: bool,
        source_key: usize,
    ) -> Option<Self> {
        if !fp16_probe_enabled()
            || layers.len() != FP16_PROBE_LAYERS
            || !(1..=FP16_PROBE_MAX_CONTEXT).contains(&max_positions)
            || logits.vocab_size != FP16_PROBE_VOCAB
            || logits.final_norm.len() != FP16_PROBE_HIDDEN
            || !eps.is_finite()
            || eps <= 0.0
        {
            return None;
        }
        let shapes = [
            (3072usize, 3072usize),
            (3072, 1024),
            (3072, 1024),
            (3072, 3072),
            (3072, 8192),
            (3072, 8192),
            (8192, 3072),
        ];
        let eligible = |w: &ResidentWeightBytes, width, rows| {
            matches!(
                w.format(),
                ResidentWeightFormat::Q4K | ResidentWeightFormat::Q6K
            ) && w.matches_shape(width, rows)
        };
        for layer in layers {
            if layer.attn_norm.len() != FP16_PROBE_HIDDEN
                || layer.ffn_norm.len() != FP16_PROBE_HIDDEN
                || layer.q_norm.is_some()
                || layer.k_norm.is_some()
                || layer.post_attn_norm.is_some()
                || layer.post_ffw_norm.is_some()
                || layer.ffn_geglu
            {
                return None;
            }
            for (w, &(width, rows)) in [
                &layer.q_weight_blocks,
                &layer.k_weight_blocks,
                &layer.v_weight_blocks,
                &layer.o_weight_blocks,
                &layer.gate_weight_blocks,
                &layer.up_weight_blocks,
                &layer.down_weight_blocks,
            ]
            .into_iter()
            .zip(&shapes)
            {
                if !eligible(w, width, rows) {
                    return None;
                }
            }
        }
        if !eligible(
            &logits.output_weight_blocks,
            FP16_PROBE_HIDDEN,
            FP16_PROBE_VOCAB,
        ) {
            return None;
        }
        // A 16 GB host is the measured target. Refuse this full-model mirror on
        // smaller hosts instead of relying on allocation pressure or swap.
        let mut physical_bytes = 0u64;
        let mut len = std::mem::size_of::<u64>();
        let name = std::ffi::CString::new("hw.memsize").ok()?;
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut physical_bytes as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || physical_bytes < 16 * 1024 * 1024 * 1024 {
            return None;
        }
        let started = std::time::Instant::now();
        let k = metal_linear_kernel()?;
        let gate_up_fused = fp16_probe_gate_up_fused_enabled();
        if gate_up_fused && fp16_sg_enabled() {
            eprintln!("[fp16-probe] fused gate/up requires native row-major half projections");
            return None;
        }
        let gate_up_silu = if gate_up_fused {
            Some(fp16_probe_gate_up_silu_pipeline(k)?)
        } else {
            None
        };
        // Refuse an unavailable requested backend: packed weights must never be
        // consumed by the row-major native kernel as a silent fallback.
        let sg = if fp16_sg_enabled() {
            Some(fp16_sg_kernels(k)?)
        } else {
            None
        };
        let attention_chunk = fp16_probe_attention_chunk();
        let attention_mm = fp16_probe_attention_mm_enabled();
        if !threadgroup_alloc_fits(&k.device, 8192)
            || k.half_mm_batched_pipeline.thread_execution_width() != 32
            || k.half_mm_batched_pipeline
                .max_total_threads_per_threadgroup()
                < 128
        {
            return None;
        }
        let mut canonical = Vec::with_capacity(layers.len() * 7 + 1);
        let mut norms = Vec::with_capacity(layers.len());
        let final_norm;
        {
            let mut cache = metal_linear_cache().lock().ok()?;
            for layer in layers {
                for w in [
                    &layer.q_weight_blocks,
                    &layer.k_weight_blocks,
                    &layer.v_weight_blocks,
                    &layer.o_weight_blocks,
                    &layer.gate_weight_blocks,
                    &layer.up_weight_blocks,
                    &layer.down_weight_blocks,
                ] {
                    canonical.push(resolve_resident_weight(&mut cache, &k.device, w, true)?);
                }
                norms.push((
                    cache.weight_buffer(&k.device, layer.attn_norm),
                    cache.weight_buffer(&k.device, layer.ffn_norm),
                ));
            }
            canonical.push(resolve_resident_weight(
                &mut cache,
                &k.device,
                &logits.output_weight_blocks,
                true,
            )?);
            final_norm = cache.weight_buffer(&k.device, logits.final_norm);
        }
        let cb = k.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        let mut keep = Vec::new();
        let mut mirrors: Vec<Buffer> = Vec::with_capacity(canonical.len());
        let mut mirror_bytes = 0u64;
        for (i, weight) in canonical.iter().enumerate() {
            let (width, rows) = if i < layers.len() * 7 {
                shapes[i % 7]
            } else {
                (FP16_PROBE_HIDDEN, FP16_PROBE_VOCAB)
            };
            let bytes = (width * rows * 2) as u64;
            mirror_bytes += bytes;
            let fused_gate = gate_up_fused && i < layers.len() * 7 && i % 7 == 4;
            let fused_up = gate_up_fused && i < layers.len() * 7 && i % 7 == 5;
            let (mirror, output_offset) = if fused_up {
                // The preceding gate already allocated both matrices. Keep
                // the seven-slot layer shape, but slot5 aliases slot4 and is
                // never dispatched separately when fusion is selected.
                let mirror = mirrors.last().expect("validated gate precedes up").clone();
                assert_eq!(mirror.length(), bytes * 2);
                (mirror, bytes)
            } else {
                (
                    k.device.new_buffer(
                        bytes * if fused_gate { 2 } else { 1 },
                        MTLResourceOptions::StorageModeShared,
                    ),
                    0,
                )
            };
            let scalar = pool_get(k, 8);
            unsafe {
                let p = scalar.contents().cast::<u32>();
                *p = (width / 256) as u32;
                *p.add(1) = rows as u32;
            }
            let pipeline = match (sg, weight.format == ResidentWeightFormat::Q4K) {
                (Some(sg), true) => &sg.dequant_q4,
                (Some(sg), false) => &sg.dequant_q6,
                (None, true) => &k.q4k_dequant_to_half_pipeline,
                (None, false) => &k.q6k_dequant_to_half_pipeline,
            };
            e.set_compute_pipeline_state(pipeline);
            e.set_buffer(0, Some(&weight.buffer), 0);
            e.set_buffer(1, Some(&mirror), output_offset);
            e.set_buffer(2, Some(&scalar), 0);
            e.set_buffer(3, Some(&scalar), 4);
            dispatch_1d(e, pipeline, rows * (width / 256));
            keep.push(scalar);
            mirrors.push(mirror);
        }
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != metal::MTLCommandBufferStatus::Completed {
            eprintln!("[fp16-probe] mirror setup GPU status={:?}", cb.status());
            pool_recycle(k, keep);
            return None;
        }
        pool_recycle(k, keep);
        if mirror_bytes != 6_425_149_440 {
            return None;
        }
        let output = mirrors.pop()?;
        let mut mirror_iter = mirrors.into_iter();
        let mut prepared_layers = Vec::with_capacity(layers.len());
        for (attn_norm, ffn_norm) in norms {
            let projections: [Buffer; 7] = std::array::from_fn(|_| {
                mirror_iter.next().expect("prevalidated seven projections")
            });
            prepared_layers.push(Fp16ProbeLayer {
                projections,
                attn_norm,
                ffn_norm,
            });
        }
        let kv_bytes = (FP16_PROBE_KV * max_positions * 2) as u64;
        let allocate_kv = || {
            let buffer = k
                .device
                .new_buffer(kv_bytes, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, kv_bytes as usize);
            }
            buffer
        };
        let cache_k = (0..layers.len()).map(|_| allocate_kv()).collect();
        let cache_v = (0..layers.len()).map(|_| allocate_kv()).collect();
        let setup_ms = started.elapsed().as_secs_f64() * 1000.0;
        let backend = if sg.is_some() {
            "sg-packed-8xnt"
        } else {
            "native-64x64"
        };
        eprintln!("[fp16-probe] ready arithmetic=canonical-half-mma projection_backend={backend} gate_up_fused={gate_up_fused} attention_chunk={attention_chunk} attention_mm={attention_mm} mirror_bytes={mirror_bytes} setup_ms={setup_ms:.3} max_rows=64 max_positions={max_positions}");
        Some(Self {
            sg,
            gate_up_silu,
            layers: prepared_layers,
            output,
            final_norm,
            cache_k,
            cache_v,
            max_positions,
            filled: 0,
            source_key,
            eps,
            split_half_pairing,
            projection_count: 0,
            actual_matrix_dispatch_count: 0,
            mirror_bytes,
            setup_ms,
            attention_chunk,
            attention_batch_chunks: 0,
            attention_batch_rows: 0,
            attention_mm,
            attention_mm_layers: 0,
            attention_mm_peak_scratch_bytes: 0,
            attention_mm_peak_physical_rows: 0,
        })
    }

    /// Teacher-forced forward from this probe's own prefix. The caller may feed
    /// one prior prediction to obtain serial greedy generation. Every active
    /// row advances the probe; this does not perform speculative acceptance.
    pub(crate) fn forward(
        &mut self,
        embeddings: &[f32],
        cos: &[f32],
        sin: &[f32],
        scale: f32,
    ) -> Option<Vec<u32>> {
        let n = embeddings.len().checked_div(FP16_PROBE_HIDDEN)?;
        if !(1..=FP16_PROBE_MAX_ROWS).contains(&n)
            || embeddings.len() != n * FP16_PROBE_HIDDEN
            || self.filled + n > self.max_positions
            || cos.len() != n * 64
            || sin.len() != cos.len()
            || !scale.is_finite()
            || scale <= 0.0
        {
            return None;
        }
        let k = metal_linear_kernel()?;
        // This is a separate arithmetic target, selected for every call on the
        // object including prompt and serial calls. An inadmissible MM forward
        // fails here, before encoding, rather than switching to row arithmetic.
        let attention_mm = if self.attention_mm {
            Some(Fp16ProbeAttentionMm::new(
                k,
                self.filled,
                n,
                self.max_positions,
                scale,
            )?)
        } else {
            None
        };
        let attention_mm_stats = attention_mm
            .as_ref()
            .map(|mm| (mm.scratch_bytes(), mm.physical_rows()));
        let nb = |bytes: usize| pool_get(k, bytes as u64);
        let mut keep = Vec::new();
        let a = nb(n * 3072 * 4);
        let b = nb(n * 3072 * 4);
        let mid = nb(n * 3072 * 4);
        let norm = nb(n * 3072 * 4);
        let query = nb(n * 3072 * 4);
        let key = nb(n * 1024 * 4);
        let value = nb(n * 1024 * 4);
        let context = nb(n * 3072 * 4);
        let o = nb(n * 3072 * 4);
        let gate = nb(n * if self.gate_up_fused() { 16384 } else { 8192 } * 4);
        let up = nb(if self.gate_up_fused() {
            4
        } else {
            n * 8192 * 4
        });
        let silu = nb(n * 8192 * 4);
        let down = nb(n * 3072 * 4);
        let logits = nb(n * FP16_PROBE_VOCAB * 4);
        let predictions = nb(n * 4);
        let cosine = nb(cos.len() * 4);
        let sine = nb(sin.len() * 4);
        write_buffer_f32(&a, embeddings);
        write_buffer_f32(&cosine, cos);
        write_buffer_f32(&sine, sin);
        // The general MMA unconditionally loads complete 64-row B tiles. Keep
        // physical padding zero, including on reuse from a larger previous n.
        let half_hidden = nb(64 * 3072 * 2);
        let half_ffn = nb(64 * 8192 * 2);
        for panel in [&half_hidden, &half_ffn] {
            unsafe {
                std::ptr::write_bytes(panel.contents().cast::<u8>(), 0, panel.length() as usize);
            }
        }
        let norms = nb(8);
        let counts = nb(12);
        let rope_q = nb(16);
        let rope_k = nb(16);
        unsafe {
            let p = norms.contents().cast::<u32>();
            *p = 3072;
            *(p.add(1).cast::<f32>()) = self.eps;
            let p = counts.contents().cast::<u32>();
            *p = (n * 3072) as u32;
            *p.add(1) = (n * 8192) as u32;
            *p.add(2) = FP16_PROBE_VOCAB as u32;
            for (scalar, heads) in [(&rope_q, 24u32), (&rope_k, 8)] {
                let p = scalar.contents().cast::<u32>();
                *p = heads;
                *p.add(1) = 128;
                *p.add(2) = 64;
                *p.add(3) = u32::from(self.split_half_pairing);
            }
        }
        let scores = nb(24 * (self.filled + n) * 4);
        let scatter = nb(n * 16);
        let mut attention_scalars = Vec::with_capacity(n);
        for row in 0..n {
            unsafe {
                let p = scatter.contents().cast::<u32>().add(row * 4);
                *p = 128;
                *p.add(1) = self.max_positions as u32;
                *p.add(2) = (self.filled + row) as u32;
                *p.add(3) = 1024;
            }
            let scalar = nb(32);
            unsafe {
                let p = scalar.contents().cast::<u32>();
                *p = 24;
                *p.add(1) = 128;
                *p.add(2) = (self.filled + row + 1) as u32;
                *p.add(3) = 3;
                *(p.add(4).cast::<f32>()) = scale;
                *p.add(5) = 128;
                *p.add(6) = (self.max_positions * 128) as u32;
                *p.add(7) = 0;
            }
            attention_scalars.push(scalar);
        }
        let cb = k.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        let mut encoded_matrix_dispatches = 0u64;
        let mut project = |e: &metal::ComputeCommandEncoderRef,
                           keep: &mut Vec<Buffer>,
                           input: &Buffer,
                           weight: &Buffer,
                           out: &Buffer,
                           width: usize,
                           rows: usize| {
            encoded_matrix_dispatches += 1;
            let (panel, count_offset) = if width == 8192 {
                (&half_ffn, 4u64)
            } else {
                (&half_hidden, 0u64)
            };
            if let Some(sg) = self.sg {
                encode_fp16_sg_projection(
                    e, k, sg, keep, input, weight, panel, out, width, rows, n,
                );
                return;
            }
            e.set_compute_pipeline_state(&k.f32_to_f16_pipeline);
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(panel), 0);
            e.set_buffer(2, Some(&counts), count_offset);
            dispatch_1d(e, &k.f32_to_f16_pipeline, n * width);
            encode_fp16_probe_mm(e, k, keep, weight, panel, out, width, rows, n);
        };
        let mut from_a = true;
        let mut attention_batch_chunks = 0u64;
        let mut attention_batch_rows = 0u64;
        for (index, layer) in self.layers.iter().enumerate() {
            let (cur, next) = if from_a { (&a, &b) } else { (&b, &a) };
            encode_rms_norm_batch(k, e, cur, &layer.attn_norm, &norm, &norms, n);
            project(
                e,
                &mut keep,
                &norm,
                &layer.projections[0],
                &query,
                3072,
                3072,
            );
            project(e, &mut keep, &norm, &layer.projections[1], &key, 3072, 1024);
            project(
                e,
                &mut keep,
                &norm,
                &layer.projections[2],
                &value,
                3072,
                1024,
            );
            // All candidate KV rows are written first, but each row's attention
            // position_count remains filled+row+1, so future tokens are masked.
            for row in 0..n {
                encode_rope(
                    e,
                    k,
                    &query,
                    &cosine,
                    &sine,
                    &rope_q,
                    24,
                    64,
                    (row * 3072 * 4) as u64,
                    (row * 64 * 4) as u64,
                );
                encode_rope(
                    e,
                    k,
                    &key,
                    &cosine,
                    &sine,
                    &rope_k,
                    8,
                    64,
                    (row * 1024 * 4) as u64,
                    (row * 64 * 4) as u64,
                );
                e.set_compute_pipeline_state(&k.kv_scatter_kv16_pipeline);
                e.set_buffer(0, Some(&key), (row * 1024 * 4) as u64);
                e.set_buffer(1, Some(&value), (row * 1024 * 4) as u64);
                e.set_buffer(2, Some(&self.cache_k[index]), 0);
                e.set_buffer(3, Some(&self.cache_v[index]), 0);
                for j in 0..4u64 {
                    e.set_buffer(4 + j, Some(&scatter), (row * 16) as u64 + j * 4);
                }
                dispatch_1d(e, &k.kv_scatter_kv16_pipeline, 1024);
            }
            if let Some(mm) = &attention_mm {
                // The probe allocated these exact admitted shapes above. A
                // violated size invariant must never silently change targets.
                assert!(
                    mm.encode(
                        k,
                        e,
                        &query,
                        &self.cache_k[index],
                        &self.cache_v[index],
                        &context
                    ),
                    "prevalidated FP16 attention-MM buffers changed shape"
                );
            } else {
                let mut row = 0;
                while row < n {
                    if self.attention_chunk > 0 {
                        let end = (row + self.attention_chunk).min(n);
                        let positions: Vec<usize> =
                            (row..end).map(|i| self.filled + i + 1).collect();
                        let route = encode_attention_splitk_kv16_batch_offset(
                            e,
                            k,
                            &mut keep,
                            &query,
                            &self.cache_k[index],
                            &self.cache_v[index],
                            &context,
                            &attention_scalars[row],
                            24,
                            8,
                            128,
                            &positions,
                            None,
                            (row * 3072 * 4) as u64,
                            (row * 3072 * 4) as u64,
                        );
                        if route.encoded {
                            attention_batch_chunks += 1;
                            attention_batch_rows += (end - row) as u64;
                            row = end;
                            continue;
                        }
                    }
                    encode_attention(
                        e,
                        k,
                        &mut keep,
                        &query,
                        &self.cache_k[index],
                        &self.cache_v[index],
                        None,
                        true,
                        false,
                        &scores,
                        &context,
                        &attention_scalars[row],
                        24,
                        8,
                        128,
                        self.filled + row + 1,
                        (row * 3072 * 4) as u64,
                        (row * 3072 * 4) as u64,
                    );
                    row += 1;
                }
            }
            project(
                e,
                &mut keep,
                &context,
                &layer.projections[3],
                &o,
                3072,
                3072,
            );
            encode_binary(
                e,
                &k.residual_add_pipeline,
                cur,
                &o,
                &mid,
                &counts,
                n * 3072,
            );
            encode_rms_norm_batch(k, e, &mid, &layer.ffn_norm, &norm, &norms, n);
            if let Some(pipeline) = self.gate_up_silu {
                project(
                    e,
                    &mut keep,
                    &norm,
                    &layer.projections[4],
                    &gate,
                    3072,
                    16384,
                );
                encode_fp16_probe_gate_up_silu(e, pipeline, &gate, &silu, &counts, 4, n);
            } else {
                project(
                    e,
                    &mut keep,
                    &norm,
                    &layer.projections[4],
                    &gate,
                    3072,
                    8192,
                );
                project(e, &mut keep, &norm, &layer.projections[5], &up, 3072, 8192);
                encode_binary_off(
                    e,
                    &k.silu_mul_pipeline,
                    &gate,
                    &up,
                    &silu,
                    &counts,
                    4,
                    n * 8192,
                );
            }
            project(
                e,
                &mut keep,
                &silu,
                &layer.projections[6],
                &down,
                8192,
                3072,
            );
            encode_binary(
                e,
                &k.residual_add_pipeline,
                &mid,
                &down,
                next,
                &counts,
                n * 3072,
            );
            from_a = !from_a;
        }
        let final_state = if from_a { &a } else { &b };
        encode_rms_norm_batch(k, e, final_state, &self.final_norm, &norm, &norms, n);
        project(
            e,
            &mut keep,
            &norm,
            &self.output,
            &logits,
            3072,
            FP16_PROBE_VOCAB,
        );
        let chunks = verify_batch_argmax_chunks(FP16_PROBE_VOCAB);
        let values = nb(n * chunks * 4);
        let ids = nb(n * chunks * 4);
        let chunk_scalar = nb(4);
        unsafe {
            *chunk_scalar.contents().cast::<u32>() = chunks as u32;
        }
        // Existing argmax helper expects a count at offset0; do not alias the
        // activation count used by earlier kernels in this command buffer.
        let vocab_scalar = nb(4);
        unsafe {
            *vocab_scalar.contents().cast::<u32>() = FP16_PROBE_VOCAB as u32;
        }
        encode_verify_batch_argmax(
            k,
            e,
            &logits,
            &predictions,
            &vocab_scalar,
            &chunk_scalar,
            &values,
            &ids,
            n,
            chunks,
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ok = cb.status() == metal::MTLCommandBufferStatus::Completed;
        let result = if ok {
            Some(unsafe {
                std::slice::from_raw_parts(predictions.contents().cast::<u32>(), n).to_vec()
            })
        } else {
            eprintln!(
                "[fp16-probe] forward GPU status={:?} position={} rows={n}",
                cb.status(),
                self.filled
            );
            None
        };
        keep.extend([
            a,
            b,
            mid,
            norm,
            query,
            key,
            value,
            context,
            o,
            gate,
            up,
            silu,
            down,
            logits,
            predictions,
            cosine,
            sine,
            half_hidden,
            half_ffn,
            norms,
            counts,
            rope_q,
            rope_k,
            scores,
            scatter,
            values,
            ids,
            chunk_scalar,
            vocab_scalar,
        ]);
        keep.extend(attention_scalars);
        if let Some(mm) = attention_mm {
            keep.extend(mm.into_keep());
        }
        pool_recycle(k, keep);
        if ok {
            self.filled += n;
            self.projection_count += (self.layers.len() * 7 + 1) as u64;
            self.actual_matrix_dispatch_count += encoded_matrix_dispatches;
            self.attention_batch_chunks += attention_batch_chunks;
            self.attention_batch_rows += attention_batch_rows;
            if let Some((scratch_bytes, physical_rows)) = attention_mm_stats {
                self.attention_mm_layers += self.layers.len() as u64;
                self.attention_mm_peak_scratch_bytes =
                    self.attention_mm_peak_scratch_bytes.max(scratch_bytes);
                self.attention_mm_peak_physical_rows =
                    self.attention_mm_peak_physical_rows.max(physical_rows);
            }
        }
        result
    }
}

/// Native half GEMM with row-major A[rows,width], physically padded B[64,width],
/// and f32 C[n,rows]. Every scalar is immutable until command completion.
#[allow(clippy::too_many_arguments)]
fn encode_fp16_probe_mm(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    keep: &mut Vec<Buffer>,
    weight: &Buffer,
    panel: &Buffer,
    out: &Buffer,
    width: usize,
    rows: usize,
    n: usize,
) {
    debug_assert!((1..=64).contains(&n) && width.is_multiple_of(32));
    debug_assert!(
        panel.length() >= (64 * width * 2) as u64 && weight.length() >= (rows * width * 2) as u64
    );
    debug_assert!(out.length() >= (n * rows * 4) as u64);
    let scalar = pool_get(k, 52);
    let fields = [
        width as u32,
        rows as u32,
        n as u32,
        0,
        0,
        0,
        width as u32,
        width as u32,
        rows as u32,
        1,
        1,
        0,
        0,
    ];
    unsafe {
        std::ptr::copy_nonoverlapping(
            fields.as_ptr(),
            scalar.contents().cast::<u32>(),
            fields.len(),
        );
    }
    e.set_compute_pipeline_state(&k.half_mm_batched_pipeline);
    e.set_buffer(0, Some(weight), 0);
    e.set_buffer(1, Some(panel), 0);
    e.set_buffer(2, Some(out), 0);
    for j in 0..13u64 {
        e.set_buffer(3 + j, Some(&scalar), j * 4);
    }
    e.set_threadgroup_memory_length(0, 8192);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: rows.div_ceil(64) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    keep.push(scalar);
}

#[cfg(test)]
include!("metal_fp16_probe_tests.rs");

#[cfg(test)]
include!("metal_fp16_sg_tests.rs");

#[cfg(test)]
include!("metal_fp16_attention_tests.rs");
