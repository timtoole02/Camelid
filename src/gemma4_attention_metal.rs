//! Experimental Gemma 4 12B verifier attention for Metal.
//!
//! This module is intentionally not wired into the resident runtime.  It owns a
//! strict-math Metal library for the two exact 12B attention geometries and exposes
//! an opt-in encoder for the future width-K verifier:
//!
//! * sliding: 16 Q heads, 8 KV heads, head dimension 256, window 1,024;
//! * global V-less: 16 Q heads, 1 KV head, head dimension 512.
//!
//! "V-less" describes the projection graph, not the attention cache layout.  The
//! caller must put the weightless-normalized raw K projection in the separate V
//! cache before RoPE mutates K.  Keeping K and V as separate inputs is what prevents
//! the verifier from accidentally attending to roped K as V.

#[cfg(test)]
use metal::CommandQueue;
use metal::{Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions};
use std::sync::OnceLock;

const GEMMA4_12B_HEADS: usize = 16;
const GEMMA4_12B_SLIDING_KV_HEADS: usize = 8;
const GEMMA4_12B_SLIDING_HEAD_DIM: usize = 256;
const GEMMA4_12B_SLIDING_WINDOW: usize = 1_024;
const GEMMA4_12B_GLOBAL_KV_HEADS: usize = 1;
const GEMMA4_12B_GLOBAL_HEAD_DIM: usize = 512;
const SPLIT_SPAN: usize = 64;
const MAX_SPLITS: usize = 64;

/// The only two geometries admitted by the experimental encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gemma4VerifierAttentionKind {
    Sliding,
    GlobalVless,
}

impl Gemma4VerifierAttentionKind {
    fn code(self) -> u32 {
        match self {
            Self::Sliding => 0,
            Self::GlobalVless => 1,
        }
    }

    fn n_kv_heads(self) -> usize {
        match self {
            Self::Sliding => GEMMA4_12B_SLIDING_KV_HEADS,
            Self::GlobalVless => GEMMA4_12B_GLOBAL_KV_HEADS,
        }
    }

    fn head_dim(self) -> usize {
        match self {
            Self::Sliding => GEMMA4_12B_SLIDING_HEAD_DIM,
            Self::GlobalVless => GEMMA4_12B_GLOBAL_HEAD_DIM,
        }
    }

    fn heads_per_threadgroup(self) -> usize {
        match self {
            // Both local query heads sharing one KV head reuse each staged K/V tile.
            Self::Sliding => 2,
            // Eight query heads share a staged global K/V tile.  Two threadgroups
            // cover the 16-query-head GQA group, reducing K/V reads from 16x to 2x.
            Self::GlobalVless => 8,
        }
    }

    fn position_range(self, position: usize) -> (usize, usize) {
        let filled = position + 1;
        match self {
            Self::Sliding => {
                let count = filled.min(GEMMA4_12B_SLIDING_WINDOW);
                (filled - count, count)
            }
            Self::GlobalVless => (0, filled),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Gemma4AttentionArgs {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    group: u32,
    cache_capacity: u32,
    max_splits: u32,
    kind: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4AttentionRow {
    window_start: u32,
    position_count: u32,
    n_splits: u32,
    query_position: u32,
}

#[derive(Debug, Clone)]
struct Gemma4AttentionPlan {
    kind: Gemma4VerifierAttentionKind,
    rows: Vec<Gemma4AttentionRow>,
    cache_capacity: usize,
    max_splits: usize,
    scale: f32,
}

impl Gemma4AttentionPlan {
    fn new(
        kind: Gemma4VerifierAttentionKind,
        positions: &[usize],
        cache_capacity: usize,
        scale: f32,
    ) -> Option<Self> {
        if !matches!(positions.len(), 1 | 2 | 4 | 8)
            || cache_capacity == 0
            || cache_capacity > u32::MAX as usize
            || !scale.is_finite()
            || scale <= 0.0
            || positions
                .iter()
                .any(|&position| position >= u32::MAX as usize)
            || positions.windows(2).any(|pair| pair[1] != pair[0] + 1)
        {
            return None;
        }

        // A width-K local verifier scatters every candidate before attention.  The
        // window therefore needs K slack slots so later candidates cannot overwrite
        // a prefix slot still needed by an earlier row in the same batch.
        if kind == Gemma4VerifierAttentionKind::Sliding
            && cache_capacity < GEMMA4_12B_SLIDING_WINDOW + positions.len()
        {
            return None;
        }
        if kind == Gemma4VerifierAttentionKind::GlobalVless
            && positions.last().copied()? >= cache_capacity
        {
            return None;
        }

        let mut rows = Vec::with_capacity(positions.len());
        let mut max_splits = 0usize;
        for &position in positions {
            let (window_start, position_count) = kind.position_range(position);
            let n_splits = position_count.div_ceil(SPLIT_SPAN);
            if position_count == 0
                || n_splits == 0
                || n_splits > MAX_SPLITS
                || window_start > u32::MAX as usize
                || position_count > u32::MAX as usize
                || position > u32::MAX as usize
            {
                return None;
            }
            max_splits = max_splits.max(n_splits);
            rows.push(Gemma4AttentionRow {
                window_start: window_start as u32,
                position_count: position_count as u32,
                n_splits: n_splits as u32,
                query_position: position as u32,
            });
        }
        Some(Self {
            kind,
            rows,
            cache_capacity,
            max_splits,
            scale,
        })
    }

    fn args(&self) -> Gemma4AttentionArgs {
        let n_kv_heads = self.kind.n_kv_heads();
        Gemma4AttentionArgs {
            n_heads: GEMMA4_12B_HEADS as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: self.kind.head_dim() as u32,
            group: (GEMMA4_12B_HEADS / n_kv_heads) as u32,
            cache_capacity: self.cache_capacity as u32,
            max_splits: self.max_splits as u32,
            kind: self.kind.code(),
            scale: self.scale,
        }
    }

    fn threadgroups_x(&self) -> usize {
        let group = GEMMA4_12B_HEADS / self.kind.n_kv_heads();
        self.kind.n_kv_heads() * group.div_ceil(self.kind.heads_per_threadgroup())
    }

    fn threads_per_threadgroup(&self) -> usize {
        self.kind.heads_per_threadgroup() * 32
    }
}

/// Buffers allocated by [`encode_gemma4_verifier_attention_batch`].  They must
/// outlive the command buffer that owns the supplied encoder.
#[allow(dead_code)]
pub(crate) struct Gemma4VerifierAttentionScratch {
    _partials: Buffer,
    _args: Buffer,
    _rows: Buffer,
}

/// Explicit production admission gate.  No current runtime calls this module.
#[allow(dead_code)]
pub(crate) fn gemma4_verifier_attention_batch_enabled() -> bool {
    std::env::var("CAMELID_GEMMA4_ATTN_BATCH_K")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Encode strict-F16, row-dimensional Gemma 4 12B attention into an existing
/// command encoder.  The caller must pre-scatter K and V for every `position`.
///
/// Queries and output are `[row][16][head_dim]` f32.  K and V are independent
/// half caches `[kv_head][cache_capacity][head_dim]`.  For `GlobalVless`, V must
/// contain the weightless-normalized raw K projection (pre-RoPE).  This function
/// is opt-in and remains unwired until the dense verifier owns F16 caches.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn encode_gemma4_verifier_attention_batch(
    encoder: &metal::ComputeCommandEncoderRef,
    query: &Buffer,
    cache_k: &Buffer,
    cache_v: &Buffer,
    output: &Buffer,
    kind: Gemma4VerifierAttentionKind,
    positions: &[usize],
    cache_capacity: usize,
    scale: f32,
) -> Option<Gemma4VerifierAttentionScratch> {
    if !gemma4_verifier_attention_batch_enabled() {
        return None;
    }
    let plan = Gemma4AttentionPlan::new(kind, positions, cache_capacity, scale)?;
    encode_batch(encoder, query, cache_k, cache_v, output, &plan)
}

fn upload_struct<T>(device: &Device, value: &T) -> Buffer {
    device.new_buffer_with_data(
        value as *const T as *const _,
        std::mem::size_of::<T>() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn upload_slice<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr() as *const _,
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn encode_batch(
    encoder: &metal::ComputeCommandEncoderRef,
    query: &Buffer,
    cache_k: &Buffer,
    cache_v: &Buffer,
    output: &Buffer,
    plan: &Gemma4AttentionPlan,
) -> Option<Gemma4VerifierAttentionScratch> {
    let kernels = gemma4_attention_kernels()?;
    let rows = plan.rows.len();
    let head_dim = plan.kind.head_dim();
    let query_bytes = rows
        .checked_mul(GEMMA4_12B_HEADS)?
        .checked_mul(head_dim)?
        .checked_mul(4)?;
    let cache_bytes = plan
        .kind
        .n_kv_heads()
        .checked_mul(plan.cache_capacity)?
        .checked_mul(head_dim)?
        .checked_mul(2)?;
    if query.length() < query_bytes as u64
        || output.length() < query_bytes as u64
        || cache_k.length() < cache_bytes as u64
        || cache_v.length() < cache_bytes as u64
    {
        return None;
    }
    let partial_values = rows
        .checked_mul(GEMMA4_12B_HEADS)?
        .checked_mul(plan.max_splits)?
        .checked_mul(head_dim + 2)?;
    let partials = kernels.device.new_buffer(
        partial_values.checked_mul(4)? as u64,
        MTLResourceOptions::StorageModePrivate,
    );
    let args = upload_struct(&kernels.device, &plan.args());
    let row_meta = upload_slice(&kernels.device, &plan.rows);

    encoder.set_compute_pipeline_state(&kernels.batch_pipeline);
    encoder.set_buffer(0, Some(query), 0);
    encoder.set_buffer(1, Some(cache_k), 0);
    encoder.set_buffer(2, Some(cache_v), 0);
    encoder.set_buffer(3, Some(&partials), 0);
    encoder.set_buffer(5, Some(&args), 0);
    encoder.set_buffer(6, Some(&row_meta), 0);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: plan.threadgroups_x() as u64,
            height: plan.max_splits as u64,
            depth: rows as u64,
        },
        metal::MTLSize {
            width: plan.threads_per_threadgroup() as u64,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&kernels.merge_batch_pipeline);
    encoder.set_buffer(0, Some(&partials), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(5, Some(&args), 0);
    encoder.set_buffer(6, Some(&row_meta), 0);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: GEMMA4_12B_HEADS as u64,
            height: rows as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Some(Gemma4VerifierAttentionScratch {
        _partials: partials,
        _args: args,
        _rows: row_meta,
    })
}

struct Gemma4AttentionKernels {
    device: Device,
    #[cfg(test)]
    queue: CommandQueue,
    #[cfg(test)]
    row_pipeline: ComputePipelineState,
    batch_pipeline: ComputePipelineState,
    #[cfg(test)]
    merge_row_pipeline: ComputePipelineState,
    merge_batch_pipeline: ComputePipelineState,
}

static GEMMA4_ATTENTION_KERNELS: OnceLock<Option<Gemma4AttentionKernels>> = OnceLock::new();

fn gemma4_attention_kernels() -> Option<&'static Gemma4AttentionKernels> {
    GEMMA4_ATTENTION_KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            // This lane is a target-authoritative verifier.  Reassociation and
            // approximate transcendental lowering are deliberately disabled.
            options.set_fast_math_enabled(false);
            let library = device
                .new_library_with_source(GEMMA4_ATTENTION_SHADER, &options)
                .map_err(|error| {
                    eprintln!("[gemma4-attn-metal] strict shader compile failed: {error}")
                })
                .ok()?;
            let pipeline = |name: &str| {
                let function = library.get_function(name, None).ok()?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .ok()
            };
            Some(Gemma4AttentionKernels {
                #[cfg(test)]
                queue: device.new_command_queue(),
                #[cfg(test)]
                row_pipeline: pipeline("gemma4_attention_splitk_kv16_strict_row")?,
                batch_pipeline: pipeline("gemma4_attention_splitk_kv16_batch")?,
                #[cfg(test)]
                merge_row_pipeline: pipeline("gemma4_attention_splitk_merge_strict_row")?,
                merge_batch_pipeline: pipeline("gemma4_attention_splitk_merge_batch")?,
                device,
            })
        })
        .as_ref()
}

const GEMMA4_ATTENTION_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Gemma4AttentionArgs {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint group;
    uint cache_capacity;
    uint max_splits;
    uint kind; // 0 = sliding ring, 1 = global V-less
    float scale;
};

inline uint gemma4_cache_slot(uint absolute_position, constant Gemma4AttentionArgs& args) {
    return (args.kind == 0) ? absolute_position % args.cache_capacity : absolute_position;
}

// Chosen strict oracle: one row per host dispatch.  Its split boundaries, four-score
// online-softmax recurrence, SIMD dot reduction and split merge order define the exact
// arithmetic contract for the row-dimensional sibling below.
kernel void gemma4_attention_splitk_kv16_strict_row(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    constant Gemma4AttentionArgs& args [[buffer(5)]],
    device const uint4* row_ptr [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint MAX_DPL = 16;
    constexpr uint STAGE_HALF4 = 512;
    const uint4 row = row_ptr[0];
    const uint position_count = row.y;
    const uint n_splits = row.z;
    const uint split = tg.y;
    if (split >= n_splits) return;

    const uint heads_per_tg = (args.kind == 0) ? 2 : 8;
    const uint head_blocks = args.group / heads_per_tg;
    const uint kvh = tg.x / head_blocks;
    const uint head_block = tg.x - kvh * head_blocks;
    const uint qh = kvh * args.group + head_block * heads_per_tg + sg;
    const bool active = sg < heads_per_tg && qh < args.n_heads;
    const uint dpl = args.head_dim / 32;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);
    const uint kv_base = kvh * args.cache_capacity * args.head_dim;

    float q[MAX_DPL];
    if (active) {
        const uint q_base = qh * args.head_dim;
        for (uint i = 0; i < dpl; ++i) {
            q[i] = query[q_base + lane + i * 32] * args.scale;
        }
    }
    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_DPL];
    for (uint i = 0; i < MAX_DPL; ++i) acc[i] = 0.0f;

    threadgroup half4 k_stage[STAGE_HALF4];
    threadgroup half4 v_stage[STAGE_HALF4];
    threadgroup half* ks = reinterpret_cast<threadgroup half*>(k_stage);
    threadgroup half* vs = reinterpret_cast<threadgroup half*>(v_stage);
    const uint pt_size = (args.kind == 0) ? 8 : 4;
    const uint tg_width = heads_per_tg * 32;
    for (uint pt = p0; pt < p1; pt += pt_size) {
        const uint count = min(pt_size, p1 - pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint half4_count = count * args.head_dim / 4;
        for (uint idx4 = tid; idx4 < half4_count; idx4 += tg_width) {
            const uint element = idx4 * 4;
            const uint p = element / args.head_dim;
            const uint d = element - p * args.head_dim;
            const uint absolute_position = row.x + pt + p;
            const uint slot = gemma4_cache_slot(absolute_position, args);
            const uint source = kv_base + slot * args.head_dim + d;
            k_stage[idx4] = *reinterpret_cast<device const half4*>(keys + source);
            v_stage[idx4] = *reinterpret_cast<device const half4*>(values + source);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint j0 = 0; j0 < count; j0 += 4) {
                float scores[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    const uint j = j0 + jj;
                    if (j < count) {
                        float score = 0.0f;
                        for (uint i = 0; i < dpl; ++i) {
                            score += q[i] * float(ks[j * args.head_dim + lane + i * 32]);
                        }
                        scores[jj] = simd_sum(score);
                    } else {
                        scores[jj] = -INFINITY;
                    }
                }
                const float block_max = max(max(scores[0], scores[1]), max(scores[2], scores[3]));
                const float next_m = max(m, block_max);
                const float correction = exp(m - next_m);
                float weights[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    weights[jj] = (scores[jj] == -INFINITY) ? 0.0f : exp(scores[jj] - next_m);
                }
                for (uint i = 0; i < dpl; ++i) {
                    float value = acc[i] * correction;
                    for (uint jj = 0; jj < 4; ++jj) {
                        if (j0 + jj < count) {
                            value += weights[jj]
                                * float(vs[(j0 + jj) * args.head_dim + lane + i * 32]);
                        }
                    }
                    acc[i] = value;
                }
                l = l * correction + weights[0] + weights[1] + weights[2] + weights[3];
                m = next_m;
            }
        }
    }
    if (active) {
        device float* dst = partials
            + ((ulong(qh) * n_splits + split) * (args.head_dim + 2));
        for (uint i = 0; i < dpl; ++i) dst[lane + i * 32] = acc[i];
        if (lane == 0) {
            dst[args.head_dim] = m;
            dst[args.head_dim + 1] = l;
        }
    }
}

// Row-dimensional verifier sibling.  Grid Z is the verifier row; uint4 metadata
// preserves every row's exact window and split count.
kernel void gemma4_attention_splitk_kv16_batch(
    device const float* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    constant Gemma4AttentionArgs& args [[buffer(5)]],
    device const uint4* rows [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint MAX_DPL = 16;
    constexpr uint STAGE_HALF4 = 512;
    const uint row_index = tg.z;
    const uint4 row = rows[row_index];
    const uint position_count = row.y;
    const uint n_splits = row.z;
    const uint split = tg.y;
    if (split >= n_splits) return;

    const uint heads_per_tg = (args.kind == 0) ? 2 : 8;
    const uint head_blocks = args.group / heads_per_tg;
    const uint kvh = tg.x / head_blocks;
    const uint head_block = tg.x - kvh * head_blocks;
    const uint qh = kvh * args.group + head_block * heads_per_tg + sg;
    const bool active = sg < heads_per_tg && qh < args.n_heads;
    const uint dpl = args.head_dim / 32;
    const uint chunk = (position_count + n_splits - 1) / n_splits;
    const uint p0 = min(split * chunk, position_count);
    const uint p1 = min(p0 + chunk, position_count);
    const uint kv_base = kvh * args.cache_capacity * args.head_dim;

    float q[MAX_DPL];
    if (active) {
        const ulong q_base = (ulong(row_index) * args.n_heads + qh) * args.head_dim;
        for (uint i = 0; i < dpl; ++i) {
            q[i] = query[q_base + lane + i * 32] * args.scale;
        }
    }
    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_DPL];
    for (uint i = 0; i < MAX_DPL; ++i) acc[i] = 0.0f;

    threadgroup half4 k_stage[STAGE_HALF4];
    threadgroup half4 v_stage[STAGE_HALF4];
    threadgroup half* ks = reinterpret_cast<threadgroup half*>(k_stage);
    threadgroup half* vs = reinterpret_cast<threadgroup half*>(v_stage);
    const uint pt_size = (args.kind == 0) ? 8 : 4;
    const uint tg_width = heads_per_tg * 32;
    for (uint pt = p0; pt < p1; pt += pt_size) {
        const uint count = min(pt_size, p1 - pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint half4_count = count * args.head_dim / 4;
        for (uint idx4 = tid; idx4 < half4_count; idx4 += tg_width) {
            const uint element = idx4 * 4;
            const uint p = element / args.head_dim;
            const uint d = element - p * args.head_dim;
            const uint absolute_position = row.x + pt + p;
            const uint slot = gemma4_cache_slot(absolute_position, args);
            const uint source = kv_base + slot * args.head_dim + d;
            k_stage[idx4] = *reinterpret_cast<device const half4*>(keys + source);
            v_stage[idx4] = *reinterpret_cast<device const half4*>(values + source);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint j0 = 0; j0 < count; j0 += 4) {
                float scores[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    const uint j = j0 + jj;
                    if (j < count) {
                        float score = 0.0f;
                        for (uint i = 0; i < dpl; ++i) {
                            score += q[i] * float(ks[j * args.head_dim + lane + i * 32]);
                        }
                        scores[jj] = simd_sum(score);
                    } else {
                        scores[jj] = -INFINITY;
                    }
                }
                const float block_max = max(max(scores[0], scores[1]), max(scores[2], scores[3]));
                const float next_m = max(m, block_max);
                const float correction = exp(m - next_m);
                float weights[4];
                for (uint jj = 0; jj < 4; ++jj) {
                    weights[jj] = (scores[jj] == -INFINITY) ? 0.0f : exp(scores[jj] - next_m);
                }
                for (uint i = 0; i < dpl; ++i) {
                    float value = acc[i] * correction;
                    for (uint jj = 0; jj < 4; ++jj) {
                        if (j0 + jj < count) {
                            value += weights[jj]
                                * float(vs[(j0 + jj) * args.head_dim + lane + i * 32]);
                        }
                    }
                    acc[i] = value;
                }
                l = l * correction + weights[0] + weights[1] + weights[2] + weights[3];
                m = next_m;
            }
        }
    }
    if (active) {
        device float* dst = partials
            + ((((ulong)row_index * args.n_heads + qh) * args.max_splits + split)
                * (args.head_dim + 2));
        for (uint i = 0; i < dpl; ++i) dst[lane + i * 32] = acc[i];
        if (lane == 0) {
            dst[args.head_dim] = m;
            dst[args.head_dim + 1] = l;
        }
    }
}

kernel void gemma4_attention_splitk_merge_strict_row(
    device const float* partials [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Gemma4AttentionArgs& args [[buffer(5)]],
    device const uint4* row_ptr [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint n_splits = row_ptr[0].z;
    const uint stride = args.head_dim + 2;
    device const float* base = partials + ulong(head) * n_splits * stride;
    float total_max = -INFINITY;
    for (uint split = 0; split < n_splits; ++split) {
        total_max = max(total_max, base[split * stride + args.head_dim]);
    }
    float total_l = 0.0f;
    for (uint split = 0; split < n_splits; ++split) {
        const float split_max = base[split * stride + args.head_dim];
        total_l += base[split * stride + args.head_dim + 1] * exp(split_max - total_max);
    }
    const float inverse_l = 1.0f / total_l;
    for (uint d = tid; d < args.head_dim; d += 256) {
        float value = 0.0f;
        for (uint split = 0; split < n_splits; ++split) {
            const float split_max = base[split * stride + args.head_dim];
            value += base[split * stride + d] * exp(split_max - total_max);
        }
        output[head * args.head_dim + d] = value * inverse_l;
    }
}

kernel void gemma4_attention_splitk_merge_batch(
    device const float* partials [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Gemma4AttentionArgs& args [[buffer(5)]],
    device const uint4* rows [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint head = tg.x;
    const uint row = tg.y;
    const uint n_splits = rows[row].z;
    const uint stride = args.head_dim + 2;
    device const float* base = partials
        + ((ulong(row) * args.n_heads + head) * args.max_splits * stride);
    float total_max = -INFINITY;
    for (uint split = 0; split < n_splits; ++split) {
        total_max = max(total_max, base[split * stride + args.head_dim]);
    }
    float total_l = 0.0f;
    for (uint split = 0; split < n_splits; ++split) {
        const float split_max = base[split * stride + args.head_dim];
        total_l += base[split * stride + args.head_dim + 1] * exp(split_max - total_max);
    }
    const float inverse_l = 1.0f / total_l;
    device float* destination = output + (ulong(row) * args.n_heads + head) * args.head_dim;
    for (uint d = tid; d < args.head_dim; d += 256) {
        float value = 0.0f;
        for (uint split = 0; split < n_splits; ++split) {
            const float split_max = base[split * stride + args.head_dim];
            value += base[split * stride + d] * exp(split_max - total_max);
        }
        destination[d] = value * inverse_l;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{f16_bits_to_f32, f32_to_f16_bits};

    fn upload_f32(device: &Device, values: &[f32]) -> Buffer {
        device.new_buffer_with_data(
            values.as_ptr() as *const _,
            std::mem::size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn upload_f16(device: &Device, values: &[u16]) -> Buffer {
        device.new_buffer_with_data(
            values.as_ptr() as *const _,
            std::mem::size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn deterministic_inputs(plan: &Gemma4AttentionPlan) -> (Vec<f32>, Vec<u16>, Vec<u16>) {
        let head_dim = plan.kind.head_dim();
        let n_kv_heads = plan.kind.n_kv_heads();
        let query: Vec<f32> = (0..plan.rows.len() * GEMMA4_12B_HEADS * head_dim)
            .map(|index| ((index * 17 % 31) as f32 - 15.0) * (1.0 / 128.0))
            .collect();
        let mut keys = vec![f32_to_f16_bits(0.0); n_kv_heads * plan.cache_capacity * head_dim];
        let mut values = keys.clone();
        let last = plan.rows.last().unwrap().query_position as usize;
        for absolute_position in 0..=last {
            let slot = if plan.kind == Gemma4VerifierAttentionKind::Sliding {
                absolute_position % plan.cache_capacity
            } else {
                absolute_position
            };
            for head in 0..n_kv_heads {
                for d in 0..head_dim {
                    let index = (head * plan.cache_capacity + slot) * head_dim + d;
                    let key = (((absolute_position * 13 + head * 7 + d * 3) % 37) as f32 - 18.0)
                        * (1.0 / 256.0);
                    // Deliberately not equal to K.  On a V-less global layer this models
                    // the separately cached, weightless-normalized raw K projection,
                    // which must remain pre-RoPE.
                    let value = (((absolute_position * 19 + head * 11 + d * 5 + 3) % 41) as f32
                        - 20.0)
                        * (1.0 / 128.0);
                    keys[index] = f32_to_f16_bits(key);
                    values[index] = f32_to_f16_bits(value);
                }
            }
        }
        (query, keys, values)
    }

    fn run_gpu(
        plan: &Gemma4AttentionPlan,
        query: &[f32],
        keys: &[u16],
        values: &[u16],
        batched: bool,
    ) -> Option<Vec<f32>> {
        let kernels = gemma4_attention_kernels()?;
        let q = upload_f32(&kernels.device, query);
        let k = upload_f16(&kernels.device, keys);
        let v = upload_f16(&kernels.device, values);
        let output_len = plan.rows.len() * GEMMA4_12B_HEADS * plan.kind.head_dim();
        let output = kernels.device.new_buffer(
            (output_len * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let args = upload_struct(&kernels.device, &plan.args());
        let rows = upload_slice(&kernels.device, &plan.rows);
        let command_buffer = kernels.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let mut batch_scratch = None;
        let mut row_partials = Vec::new();

        if batched {
            batch_scratch = Some(encode_batch(encoder, &q, &k, &v, &output, plan)?);
        } else {
            let head_dim = plan.kind.head_dim();
            for row in 0..plan.rows.len() {
                let n_splits = plan.rows[row].n_splits as usize;
                let partials = kernels.device.new_buffer(
                    (GEMMA4_12B_HEADS * n_splits * (head_dim + 2) * 4) as u64,
                    MTLResourceOptions::StorageModePrivate,
                );
                encoder.set_compute_pipeline_state(&kernels.row_pipeline);
                encoder.set_buffer(0, Some(&q), (row * GEMMA4_12B_HEADS * head_dim * 4) as u64);
                encoder.set_buffer(1, Some(&k), 0);
                encoder.set_buffer(2, Some(&v), 0);
                encoder.set_buffer(3, Some(&partials), 0);
                encoder.set_buffer(5, Some(&args), 0);
                encoder.set_buffer(
                    6,
                    Some(&rows),
                    (row * std::mem::size_of::<Gemma4AttentionRow>()) as u64,
                );
                encoder.dispatch_thread_groups(
                    metal::MTLSize {
                        width: plan.threadgroups_x() as u64,
                        height: n_splits as u64,
                        depth: 1,
                    },
                    metal::MTLSize {
                        width: plan.threads_per_threadgroup() as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.set_compute_pipeline_state(&kernels.merge_row_pipeline);
                encoder.set_buffer(0, Some(&partials), 0);
                encoder.set_buffer(
                    1,
                    Some(&output),
                    (row * GEMMA4_12B_HEADS * head_dim * 4) as u64,
                );
                encoder.set_buffer(5, Some(&args), 0);
                encoder.set_buffer(
                    6,
                    Some(&rows),
                    (row * std::mem::size_of::<Gemma4AttentionRow>()) as u64,
                );
                encoder.dispatch_thread_groups(
                    metal::MTLSize {
                        width: GEMMA4_12B_HEADS as u64,
                        height: 1,
                        depth: 1,
                    },
                    metal::MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                row_partials.push(partials);
            }
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != metal::MTLCommandBufferStatus::Completed {
            return None;
        }
        let mut result = vec![0.0f32; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output.contents() as *const f32,
                result.as_mut_ptr(),
                output_len,
            );
        }
        drop(batch_scratch);
        drop(row_partials);
        Some(result)
    }

    fn cpu_probe(
        plan: &Gemma4AttentionPlan,
        query: &[f32],
        keys: &[u16],
        values: &[u16],
        row: usize,
        head: usize,
        output_dim: usize,
    ) -> f32 {
        let head_dim = plan.kind.head_dim();
        let kv_head = head / (GEMMA4_12B_HEADS / plan.kind.n_kv_heads());
        let meta = plan.rows[row];
        let q_base = (row * GEMMA4_12B_HEADS + head) * head_dim;
        let mut scores = Vec::with_capacity(meta.position_count as usize);
        for offset in 0..meta.position_count as usize {
            let absolute = meta.window_start as usize + offset;
            let slot = if plan.kind == Gemma4VerifierAttentionKind::Sliding {
                absolute % plan.cache_capacity
            } else {
                absolute
            };
            let cache_base = (kv_head * plan.cache_capacity + slot) * head_dim;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += query[q_base + d] * plan.scale * f16_bits_to_f32(keys[cache_base + d]);
            }
            scores.push(score);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0f32;
        let mut numerator = 0.0f32;
        for (offset, score) in scores.into_iter().enumerate() {
            let weight = (score - max).exp();
            let absolute = meta.window_start as usize + offset;
            let slot = if plan.kind == Gemma4VerifierAttentionKind::Sliding {
                absolute % plan.cache_capacity
            } else {
                absolute
            };
            let cache_index = (kv_head * plan.cache_capacity + slot) * head_dim + output_dim;
            denominator += weight;
            numerator += weight * f16_bits_to_f32(values[cache_index]);
        }
        numerator / denominator
    }

    #[test]
    fn plan_admits_only_exact_widths_and_keeps_sliding_batch_slack() {
        for rows in [1usize, 2, 4, 8] {
            let positions: Vec<usize> = (1_023..1_023 + rows).collect();
            let plan = Gemma4AttentionPlan::new(
                Gemma4VerifierAttentionKind::Sliding,
                &positions,
                GEMMA4_12B_SLIDING_WINDOW + rows,
                1.0,
            )
            .expect("exact Gemma sliding plan");
            assert_eq!(plan.rows[0].window_start, 0);
            assert_eq!(plan.rows[0].position_count, 1_024);
            if rows >= 2 {
                assert_eq!(plan.rows[1].window_start, 1);
                assert_eq!(plan.rows[1].position_count, 1_024);
            }
            if rows >= 4 {
                assert_eq!(plan.rows[2].window_start, 2);
                assert_eq!(plan.rows[2].query_position, 1_025);
            }
        }
        assert!(Gemma4AttentionPlan::new(
            Gemma4VerifierAttentionKind::Sliding,
            &[1_023, 1_024, 1_025],
            1_027,
            1.0,
        )
        .is_none());
        assert!(Gemma4AttentionPlan::new(
            Gemma4VerifierAttentionKind::Sliding,
            &[1_023, 1_024, 1_025, 1_026],
            1_027,
            1.0,
        )
        .is_none());
    }

    #[test]
    fn strict_f16_batch_matches_row_oracle_at_1024_boundary() {
        if Device::system_default().is_none() {
            return;
        }
        for kind in [
            Gemma4VerifierAttentionKind::Sliding,
            Gemma4VerifierAttentionKind::GlobalVless,
        ] {
            for width in [1usize, 2, 4, 8] {
                let positions: Vec<usize> = (1_023..1_023 + width).collect();
                let capacity = match kind {
                    Gemma4VerifierAttentionKind::Sliding => GEMMA4_12B_SLIDING_WINDOW + width,
                    Gemma4VerifierAttentionKind::GlobalVless => {
                        positions.last().copied().unwrap() + 1
                    }
                };
                let plan = Gemma4AttentionPlan::new(kind, &positions, capacity, 1.0 / 16.0)
                    .expect("exact Gemma verifier plan");
                let (query, keys, values) = deterministic_inputs(&plan);
                let rowwise =
                    run_gpu(&plan, &query, &keys, &values, false).expect("strict row oracle");
                let batched =
                    run_gpu(&plan, &query, &keys, &values, true).expect("row-dimensional batch");
                assert_eq!(rowwise.len(), batched.len());
                for (index, (&expected, &actual)) in rowwise.iter().zip(&batched).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "kind={kind:?} width={width} element={index}: batch={actual} ({:#010x}) != strict-row={expected} ({:#010x})",
                        actual.to_bits(),
                        expected.to_bits(),
                    );
                }

                // Independent sequential CPU probe confirms the selected row metadata,
                // F16 cache interpretation, and V cache semantics.  It is an envelope
                // check because Metal's SIMD dot/exp implementation is intentionally the
                // bit-authoritative oracle above.
                for row in 0..width {
                    let head = row % GEMMA4_12B_HEADS;
                    let dim = (row * 29 + 7) % kind.head_dim();
                    let want = cpu_probe(&plan, &query, &keys, &values, row, head, dim);
                    let got = batched[(row * GEMMA4_12B_HEADS + head) * kind.head_dim() + dim];
                    assert!(
                        (got - want).abs() <= 2.0e-3,
                        "kind={kind:?} width={width} row={row} pos={} CPU probe {want} != Metal {got}",
                        positions[row],
                    );
                }
            }
        }
    }

    #[test]
    fn global_vless_reads_pre_rope_value_cache_not_key_cache() {
        if Device::system_default().is_none() {
            return;
        }
        let positions = [1_023usize, 1_024, 1_025, 1_026];
        let plan = Gemma4AttentionPlan::new(
            Gemma4VerifierAttentionKind::GlobalVless,
            &positions,
            1_027,
            1.0 / 16.0,
        )
        .unwrap();
        let (query, keys, mut values) = deterministic_inputs(&plan);
        // Give the pre-RoPE V-less cache a visible positive bias.  K remains
        // centered near zero, so binding K as V cannot accidentally pass because
        // both weighted averages happen to cancel around zero.
        for value in &mut values {
            *value = f32_to_f16_bits(0.5 + 0.25 * f16_bits_to_f32(*value));
        }
        let got = run_gpu(&plan, &query, &keys, &values, true).expect("global V-less batch");
        let row = 2usize; // explicit position 1,025 boundary receipt
        let head = 0usize;
        let dim = 11usize;
        let expected = cpu_probe(&plan, &query, &keys, &values, row, head, dim);
        let wrong_k_as_v = cpu_probe(&plan, &query, &keys, &keys, row, head, dim);
        let actual = got[(row * GEMMA4_12B_HEADS + head) * GEMMA4_12B_GLOBAL_HEAD_DIM + dim];
        assert!(
            (actual - expected).abs() <= 2.0e-3,
            "{actual} != {expected}"
        );
        assert!(
            (actual - wrong_k_as_v).abs() > 0.25,
            "test fixture did not distinguish pre-RoPE V from roped K: actual={actual}, K-as-V={wrong_k_as_v}"
        );
    }
}
