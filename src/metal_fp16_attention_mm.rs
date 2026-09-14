// Isolated diagnostic attention target, included in metal.rs on macOS only.
// Uses existing prefill MMA/softmax kernels, with absolute-position query tiles.
// No production attention or verifier admission changes are made by this file.

fn fp16_probe_attention_mm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_BENCH_FP16_ATTN_MM")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

static FP16_PROBE_ATTN_MM_ENCODES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counts encoded layer-attention chains, not completed GPU work. Callers must
/// also require command-buffer success before describing a completed route.
pub fn metal_fp16_probe_attention_mm_encode_count() -> u64 {
    FP16_PROBE_ATTN_MM_ENCODES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Scratch for one isolated forward, reused across its layers on one encoder.
/// Q and P padding is zeroed once before any commands are encoded. Each layer
/// overwrites precisely the same active row spans, so that padding stays zero.
/// Every scalar remains immutable through command completion.
struct Fp16ProbeAttentionMm {
    active_rows: usize,
    leading_rows: usize,
    physical_rows: usize,
    max_positions: usize,
    position_count: usize,
    padded_positions: usize,
    query_half: Buffer,
    scores_half: Buffer,
    probabilities_half: Buffer,
    transposed_values_half: Buffer,
    context_full: Buffer,
    score_geometry: Buffer,
    pv_geometry: Buffer,
    softmax_geometry: Buffer,
    common_geometry: Buffer,
}

impl Fp16ProbeAttentionMm {
    fn physical_rows(&self) -> usize {
        self.physical_rows
    }

    fn leading_rows(&self) -> usize {
        self.leading_rows
    }

    fn padded_positions(&self) -> usize {
        self.padded_positions
    }

    fn scratch_bytes(&self) -> u64 {
        [
            &self.query_half,
            &self.scores_half,
            &self.probabilities_half,
            &self.transposed_values_half,
            &self.context_full,
            &self.score_geometry,
            &self.pv_geometry,
            &self.softmax_geometry,
            &self.common_geometry,
        ]
        .iter()
        .map(|buffer| buffer.length())
        .sum()
    }

    /// Does not consult the environment: the host chooses this target before
    /// constructing scratch. Tests can exercise the same encoder explicitly.
    fn new(
        k: &MetalLinearKernel,
        base: usize,
        active_rows: usize,
        max_positions: usize,
        scale: f32,
    ) -> Option<Self> {
        let position_count = base.checked_add(active_rows)?;
        if !(1..=64).contains(&active_rows)
            || !(1..=2048).contains(&max_positions)
            || position_count > max_positions
            || !scale.is_finite()
            || scale <= 0.0
            || !threadgroup_alloc_fits(&k.device, 8192)
            || k.half_mm_batched_pipeline.thread_execution_width() != 32
            || k.half_mm_batched_f16o_pipeline.thread_execution_width() != 32
            || k.softmax_causal_rows_pipeline.thread_execution_width() != 32
            || k.half_mm_batched_pipeline
                .max_total_threads_per_threadgroup()
                < 128
            || k.half_mm_batched_f16o_pipeline
                .max_total_threads_per_threadgroup()
                < 128
            || k.softmax_causal_rows_pipeline
                .max_total_threads_per_threadgroup()
                < 256
        {
            return None;
        }
        let aligned_base = base / 64 * 64;
        let leading_rows = base - aligned_base;
        let physical_rows = (leading_rows + active_rows).next_multiple_of(64);
        let padded_positions = position_count.next_multiple_of(64);
        debug_assert!(physical_rows == 64 || physical_rows == 128);
        let nb = |bytes: usize| pool_get(k, bytes as u64);
        let query_half = nb(physical_rows * 3072 * 2);
        let scores_half = nb(24 * physical_rows * padded_positions * 2);
        let probabilities_half = nb(24 * physical_rows * padded_positions * 2);
        let transposed_values_half = nb(8 * 128 * padded_positions * 2);
        let context_full = nb(physical_rows * 3072 * 4);
        // S contains some culled future tiles. Softmax never reads them, but
        // zeroing S as well makes the scratch contract explicit and testable.
        for buffer in [&query_half, &scores_half, &probabilities_half] {
            unsafe {
                std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, buffer.length() as usize);
            }
        }
        let scalar = |fields: &[u32]| {
            let buffer = nb(fields.len() * 4);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    fields.as_ptr(),
                    buffer.contents().cast::<u32>(),
                    fields.len(),
                );
            }
            buffer
        };
        // S[token,key] = Q[token,dim] * K[key,dim]. The native kernel describes
        // this as C = B * A^T, so A is the original f16 K cache, B is Q.
        let score_geometry = scalar(&[
            128,
            position_count as u32,
            physical_rows as u32,
            (max_positions * 128) as u32,
            128,
            (physical_rows * padded_positions) as u32,
            128,
            3072,
            padded_positions as u32,
            1,
            3,
            1,
            aligned_base as u32,
        ]);
        // O[token,dim] = P[token,key] * V[key,dim]. Transposed V gives the
        // native kernel contiguous A[dim,key] rows; C stays f32 for the probe.
        let pv_geometry = scalar(&[
            padded_positions as u32,
            128,
            physical_rows as u32,
            (128 * padded_positions) as u32,
            (physical_rows * padded_positions) as u32,
            128,
            padded_positions as u32,
            padded_positions as u32,
            3072,
            1,
            3,
            2,
            aligned_base as u32,
        ]);
        let softmax_geometry = scalar(&[
            padded_positions as u32,
            position_count as u32,
            scale.to_bits(),
            aligned_base as u32,
            physical_rows as u32,
            0, // window disabled for dense Llama
        ]);
        let common_geometry = scalar(&[
            (active_rows * 3072) as u32,
            128,
            max_positions as u32,
            padded_positions as u32,
            position_count as u32,
        ]);
        Some(Self {
            active_rows,
            leading_rows,
            physical_rows,
            max_positions,
            position_count,
            padded_positions,
            query_half,
            scores_half,
            probabilities_half,
            transposed_values_half,
            context_full,
            score_geometry,
            pv_geometry,
            softmax_geometry,
            common_geometry,
        })
    }

    /// Call only after all active rows' RoPE and KV scatter have been encoded.
    /// The same target must be selected for prompt, serial, and wide forwards.
    /// Arithmetic differs from row attention: Q, scores and probabilities are
    /// half, with f32 MMA accumulation and an f32 final context panel.
    fn encode(
        &self,
        k: &MetalLinearKernel,
        e: &metal::ComputeCommandEncoderRef,
        query_f32: &Buffer,
        keys_half: &Buffer,
        values_half: &Buffer,
        context_f32: &Buffer,
    ) -> bool {
        let active_bytes = (self.active_rows * 3072 * 4) as u64;
        let cache_bytes = (8 * self.max_positions * 128 * 2) as u64;
        if query_f32.length() < active_bytes
            || context_f32.length() < active_bytes
            || keys_half.length() < cache_bytes
            || values_half.length() < cache_bytes
        {
            return false;
        }
        // Absolute query alignment makes every active row belong to the same
        // 64-query MMA tile regardless of serial/wide batching. In particular,
        // softmax's per-row write_end and PV's per-tile k_end then agree.
        e.set_compute_pipeline_state(&k.f32_to_f16_pipeline);
        e.set_buffer(0, Some(query_f32), 0);
        e.set_buffer(
            1,
            Some(&self.query_half),
            (self.leading_rows * 3072 * 2) as u64,
        );
        e.set_buffer(2, Some(&self.common_geometry), 0);
        dispatch_1d(e, &k.f32_to_f16_pipeline, self.active_rows * 3072);

        e.set_compute_pipeline_state(&k.transpose_v16_pipeline);
        e.set_buffer(0, Some(values_half), 0);
        e.set_buffer(1, Some(&self.transposed_values_half), 0);
        for j in 0..4u64 {
            e.set_buffer(2 + j, Some(&self.common_geometry), 4 + j * 4);
        }
        e.dispatch_threads(
            metal::MTLSize {
                width: self.padded_positions as u64,
                height: 128,
                depth: 8,
            },
            metal::MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );

        let mm = |pipeline: &ComputePipelineState,
                  a: &Buffer,
                  b: &Buffer,
                  out: &Buffer,
                  geometry: &Buffer,
                  rows: usize,
                  half_output: bool| {
            e.set_compute_pipeline_state(pipeline);
            e.set_buffer(0, Some(a), 0);
            e.set_buffer(1, Some(b), 0);
            e.set_buffer(2, Some(out), 0);
            for j in 0..13u64 {
                e.set_buffer(3 + j, Some(geometry), j * 4);
            }
            if half_output {
                e.set_buffer(16, Some(&self.softmax_geometry), 20);
            }
            e.set_threadgroup_memory_length(0, 8192);
            e.dispatch_thread_groups(
                metal::MTLSize {
                    width: rows.div_ceil(64) as u64,
                    height: (self.physical_rows / 64) as u64,
                    depth: 24,
                },
                metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        };
        mm(
            &k.half_mm_batched_f16o_pipeline,
            keys_half,
            &self.query_half,
            &self.scores_half,
            &self.score_geometry,
            self.position_count,
            true,
        );
        e.set_compute_pipeline_state(&k.softmax_causal_rows_pipeline);
        e.set_buffer(0, Some(&self.scores_half), 0);
        e.set_buffer(1, Some(&self.probabilities_half), 0);
        for j in 0..6u64 {
            e.set_buffer(2 + j, Some(&self.softmax_geometry), j * 4);
        }
        e.dispatch_thread_groups(
            metal::MTLSize {
                width: 24,
                height: (self.physical_rows / 8) as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        mm(
            &k.half_mm_batched_pipeline,
            &self.transposed_values_half,
            &self.probabilities_half,
            &self.context_full,
            &self.pv_geometry,
            128,
            false,
        );
        e.set_compute_pipeline_state(&k.copy_f32_pipeline);
        e.set_buffer(
            0,
            Some(&self.context_full),
            (self.leading_rows * 3072 * 4) as u64,
        );
        e.set_buffer(1, Some(context_f32), 0);
        e.set_buffer(2, Some(&self.common_geometry), 0);
        dispatch_1d(e, &k.copy_f32_pipeline, self.active_rows * 3072);
        FP16_PROBE_ATTN_MM_ENCODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Move these into the forward's keep list. Recycle only after its command
    /// buffer completes; dropping/recycling during encoding would break reuse.
    fn into_keep(self) -> Vec<Buffer> {
        vec![
            self.query_half,
            self.scores_half,
            self.probabilities_half,
            self.transposed_values_half,
            self.context_full,
            self.score_geometry,
            self.pv_geometry,
            self.softmax_geometry,
            self.common_geometry,
        ]
    }
}

#[cfg(test)]
include!("metal_fp16_attention_mm_tests.rs");
