//! spec50 router lane: a batched, bit-exact replacement for the gemma4 router GEMV.
//!
//! The shipping kernel (`gemma4_router_batch_k_f32`) launches one 256-thread
//! threadgroup per token, leaves 128 of those threads idle, and re-reads the whole
//! `[num_experts][hidden]` f32 router matrix once per token. At K=8 that is eight
//! passes over 1.44 MB per layer.
//!
//! This module keeps the arithmetic byte-for-byte and only changes the layout:
//!
//! * `spec50_router_rms_factor` computes the per-token RMS factor once, in exactly
//!   the operand order of the shipping kernel's phase 1 (256-thread strided
//!   sum-of-squares, halving tree, `rms_inv * (1/sqrt(hidden))`), and parks the K
//!   factors in a tiny scratch buffer.
//! * `spec50_router_gemv_exact` then owns one `(token, expert)` pair per lane
//!   (`lane = expert_sub * 8 + token_sub`), so a 32-lane simdgroup streams four
//!   expert rows for up to eight tokens at once and the matrix is read exactly
//!   once per K-token chunk instead of once per token.
//!
//! The per-lane accumulation is the shipping kernel's dot loop copied character for
//! character — same source text, same compile options, therefore the same contraction
//! and reassociation decisions and the same bits. This is load-bearing, not
//! decoration: widening those loads to `float4` and unrolling the body by hand is
//! enough for fast math to vectorise the accumulator, which moves 4374 of 9216 logits
//! by up to 38 ULP. `spec50_router_gemv_vec4` keeps that variant alive purely so the
//! benchmark can price it; it must never reach the runtime.
//!
//! Everything here is additive: its own Metal library behind a `OnceLock`, free
//! encode functions with explicit buffers, no existing dispatch site touched.
#![allow(dead_code)]

use super::*;

/// Tokens one router-GEMV threadgroup handles (one chunk); K=8 is the whole scope.
pub(crate) const SPEC50_ROUTER_CHUNK_TOKENS: usize = 8;
/// Threads per router-GEMV threadgroup: one simdgroup = 4 experts x 8 tokens.
pub(crate) const SPEC50_ROUTER_TG_THREADS: u64 = 32;
/// Experts resolved by one router-GEMV threadgroup.
pub(crate) const SPEC50_ROUTER_EXPERTS_PER_TG: usize = 4;
/// Threads per RMS-factor threadgroup. MUST be 256: the halving tree below is the
/// shipping kernel's tree, and that tree's rounding depends on the thread count.
pub(crate) const SPEC50_ROUTER_RMS_TG_THREADS: u64 = 256;

/// Bytes the integrator must allocate for the RMS-factor scratch buffer: one f32 per
/// token in the batch. Metal buffers are page-backed, so this is a 16-byte floor.
pub(crate) fn spec50_router_factor_bytes(k_tokens: usize) -> u64 {
    ((k_tokens.max(1) * 4) as u64).max(16)
}

/// Compiled with DEFAULT options (fast math ON) on purpose: the shipping twin lives
/// in `ELEMENTWISE_SHADER`, which is compiled with `CompileOptions::new()`. Same
/// source text plus same compiler flags is what makes the contraction decisions —
/// and therefore the bits — come out the same.
const SPEC50_ROUTER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define SPEC50_CHUNK 8u

// Phase 1 of gemma4_router_batch_k_f32, verbatim, hoisted out of the per-expert loop
// so the K factors are computed once per batch instead of once per (token, expert)
// threadgroup. Must be dispatched with exactly 256 threads per threadgroup.
kernel void spec50_router_rms_factor(
    device const float* input [[buffer(0)]],
    device float* out_factor [[buffer(1)]],
    constant uint& hidden [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint& k_tokens [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tgsize [[threads_per_threadgroup]]
) {
    if (token_idx >= k_tokens) return;
    device const float* in_tok = input + token_idx * hidden;

    threadgroup float partial[256];
    float local_ss = 0.0f;
    for (uint i = tid; i < hidden; i += tgsize) {
        float v = in_tok[i];
        local_ss += v * v;
    }
    partial[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = 1.0f / sqrt(partial[0] / float(hidden) + eps);
    float factor = rms_inv * (1.0f / sqrt(float(hidden)));
    if (tid == 0) {
        out_factor[token_idx] = factor;
    }
}

// Phase 2: one simdgroup = 4 expert rows x 8 tokens, one serial dot per lane.
// lane = expert_sub * 8 + token_sub, so the eight lanes sharing an expert broadcast
// the same weight address and the four lanes sharing a token broadcast the same
// activation address. No constraint on `hidden`.
kernel void spec50_router_gemv_exact(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    // Inactive lanes still walk the loop (they are in the same simdgroup anyway);
    // clamp their pointers so they stay inside the allocations.
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    // Textually the shipping kernel's dot loop, character for character. That is the
    // whole bit-exactness argument: same source text + same compile options => the
    // compiler makes the same contraction and reassociation decisions. Widening these
    // loads to float4 measurably breaks it (fast math then vectorises the accumulator
    // itself: 4374/9216 logits move, up to 38 ULP) — see spec50_router_gemv_vec4.
    float dot_val = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        float r_i = in_tok[i] * factor * gate_inp_scale[i];
        dot_val += w_row[i] * r_i;
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}

// NOT BIT-EXACT. Identical layout to spec50_router_gemv_exact, but with float4 loads
// and a hand-unrolled body. Kept only so the benchmark can price what exactness costs;
// it must never be wired into the runtime. `hidden` must be a multiple of 4.
kernel void spec50_router_gemv_vec4(
    device const float* input [[buffer(0)]],
    device const float* gate_inp_scale [[buffer(1)]],
    device const float* gate_inp_weights [[buffer(2)]],
    device float* out_logits [[buffer(3)]],
    device const float* factors [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k_tokens [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint tok0 = tg.y * SPEC50_CHUNK;
    if (tok0 >= k_tokens) return;
    const uint kt = min(k_tokens - tok0, SPEC50_CHUNK);

    const uint tok_sub = lane & 7u;
    const uint e_sub = lane >> 3;
    const uint e = tg.x * 4u + e_sub;
    const bool active = (tok_sub < kt) && (e < num_experts);
    const uint tc = min(tok_sub, kt - 1u);
    const uint ec = min(e, num_experts - 1u);

    device const float* in_tok = input + (tok0 + tc) * hidden;
    device const float* w_row = gate_inp_weights + ec * hidden;
    const float factor = factors[tok0 + tc];

    float dot_val = 0.0f;
    for (uint i = 0; i < hidden; i += 4u) {
        float4 x4 = *reinterpret_cast<device const float4*>(in_tok + i);
        float4 s4 = *reinterpret_cast<device const float4*>(gate_inp_scale + i);
        float4 w4 = *reinterpret_cast<device const float4*>(w_row + i);
        dot_val += w4.x * (x4.x * factor * s4.x);
        dot_val += w4.y * (x4.y * factor * s4.y);
        dot_val += w4.z * (x4.z * factor * s4.z);
        dot_val += w4.w * (x4.w * factor * s4.w);
    }

    if (active) {
        out_logits[(tok0 + tok_sub) * num_experts + e] = dot_val;
    }
}
"#;

#[cfg(target_os = "macos")]
pub(crate) struct Spec50RouterKernels {
    pub(crate) rms_factor_pipeline: ComputePipelineState,
    pub(crate) router_gemv_pipeline: ComputePipelineState,
    /// NOT bit-exact; benchmark-only. See `encode_spec50_router_gemv_vec4`.
    pub(crate) router_gemv_vec4_pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
static SPEC50_ROUTER_KERNELS: OnceLock<Option<Spec50RouterKernels>> = OnceLock::new();

/// Compiles (once) and returns the spec50 router pipelines, or `None` — with a stderr
/// line saying why — if the Metal compile fails. Unlike `metal_linear_kernel()`, a
/// failure here disables only this lane.
#[cfg(target_os = "macos")]
pub(crate) fn spec50_router_kernels() -> Option<&'static Spec50RouterKernels> {
    SPEC50_ROUTER_KERNELS
        .get_or_init(|| {
            let device = Device::system_default()?;
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(SPEC50_ROUTER_SHADER, &options)
                .map_err(|err| eprintln!("[metal] SPEC50_ROUTER_SHADER compile failed: {err}"))
                .ok()?;
            let pipeline = |name: &str| -> Option<ComputePipelineState> {
                let function = library
                    .get_function(name, None)
                    .map_err(|err| eprintln!("[metal] spec50 function {name} missing: {err}"))
                    .ok()?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| eprintln!("[metal] spec50 pipeline {name} failed: {err}"))
                    .ok()
            };
            Some(Spec50RouterKernels {
                rms_factor_pipeline: pipeline("spec50_router_rms_factor")?,
                router_gemv_pipeline: pipeline("spec50_router_gemv_exact")?,
                router_gemv_vec4_pipeline: pipeline("spec50_router_gemv_vec4")?,
            })
        })
        .as_ref()
}

/// Batched router GEMV, bit-identical twin of `encode_gemma4_router_batch_k`.
///
/// Encodes TWO dispatches into `encoder` (a compute encoder is serial, so the
/// ordering is guaranteed without an explicit barrier):
///   1. `spec50_router_rms_factor` -> `factors[0..k_tokens]`
///   2. `spec50_router_gemv_exact` -> `out_logits[k_tokens][num_experts]`
///
/// Buffers:
///   0 `input`            `[k_tokens][hidden]` f32
///   1 `gate_inp_scale`   `[hidden]` f32
///   2 `gate_inp_weights` `[num_experts][hidden]` f32
///   3 `out_logits`       `[k_tokens][num_experts]` f32, at `out_logits_offset`
///   4 `factors`          NEW scratch, `spec50_router_factor_bytes(k_tokens)` bytes
///
/// Returns `None` (encode nothing) unless `num_experts % 4 == 0` — the caller should
/// fall back to the shipping encode then. `hidden` is unconstrained.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_router_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50RouterKernels,
    input: &Buffer,
    gate_inp_scale: &Buffer,
    gate_inp_weights: &Buffer,
    out_logits: &Buffer,
    factors: &Buffer,
    hidden: usize,
    eps: f32,
    num_experts: usize,
    k_tokens: usize,
    out_logits_offset: u64,
) -> Option<()> {
    if k_tokens == 0
        || num_experts == 0
        || hidden == 0
        || num_experts % SPEC50_ROUTER_EXPERTS_PER_TG != 0
    {
        return None;
    }
    let hidden_u32 = hidden as u32;
    let num_experts_u32 = num_experts as u32;
    let k_tokens_u32 = k_tokens as u32;

    encoder.set_compute_pipeline_state(&kernels.rms_factor_pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(factors), 0);
    encoder.set_bytes(2, 4, &hidden_u32 as *const u32 as *const _);
    encoder.set_bytes(3, 4, &eps as *const f32 as *const _);
    encoder.set_bytes(4, 4, &k_tokens_u32 as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: k_tokens as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: SPEC50_ROUTER_RMS_TG_THREADS,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&kernels.router_gemv_pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(gate_inp_scale), 0);
    encoder.set_buffer(2, Some(gate_inp_weights), 0);
    encoder.set_buffer(3, Some(out_logits), out_logits_offset);
    encoder.set_buffer(4, Some(factors), 0);
    encoder.set_bytes(5, 4, &hidden_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &num_experts_u32 as *const u32 as *const _);
    encoder.set_bytes(7, 4, &k_tokens_u32 as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (num_experts / SPEC50_ROUTER_EXPERTS_PER_TG) as u64,
            height: k_tokens.div_ceil(SPEC50_ROUTER_CHUNK_TOKENS) as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: SPEC50_ROUTER_TG_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Some(())
}

/// NOT BIT-EXACT — benchmark and diagnostics only, never the runtime.
///
/// Same buffers and same grid as [`encode_spec50_router_gemv`], but dispatches the
/// `float4` variant of the dot loop. It exists to measure what the bit-exactness
/// constraint costs; it disagrees with `gemma4_router_batch_k_f32` by up to 38 ULP.
/// Requires `hidden % 4 == 0` (the vector loads have no scalar tail).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_spec50_router_gemv_vec4_inexact(
    encoder: &metal::ComputeCommandEncoderRef,
    kernels: &Spec50RouterKernels,
    input: &Buffer,
    gate_inp_scale: &Buffer,
    gate_inp_weights: &Buffer,
    out_logits: &Buffer,
    factors: &Buffer,
    hidden: usize,
    eps: f32,
    num_experts: usize,
    k_tokens: usize,
    out_logits_offset: u64,
) -> Option<()> {
    if k_tokens == 0
        || num_experts == 0
        || hidden == 0
        || num_experts % SPEC50_ROUTER_EXPERTS_PER_TG != 0
        || hidden % 4 != 0
    {
        return None;
    }
    let hidden_u32 = hidden as u32;
    let num_experts_u32 = num_experts as u32;
    let k_tokens_u32 = k_tokens as u32;

    encoder.set_compute_pipeline_state(&kernels.rms_factor_pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(factors), 0);
    encoder.set_bytes(2, 4, &hidden_u32 as *const u32 as *const _);
    encoder.set_bytes(3, 4, &eps as *const f32 as *const _);
    encoder.set_bytes(4, 4, &k_tokens_u32 as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: k_tokens as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: SPEC50_ROUTER_RMS_TG_THREADS,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&kernels.router_gemv_vec4_pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(gate_inp_scale), 0);
    encoder.set_buffer(2, Some(gate_inp_weights), 0);
    encoder.set_buffer(3, Some(out_logits), out_logits_offset);
    encoder.set_buffer(4, Some(factors), 0);
    encoder.set_bytes(5, 4, &hidden_u32 as *const u32 as *const _);
    encoder.set_bytes(6, 4, &num_experts_u32 as *const u32 as *const _);
    encoder.set_bytes(7, 4, &k_tokens_u32 as *const u32 as *const _);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: (num_experts / SPEC50_ROUTER_EXPERTS_PER_TG) as u64,
            height: k_tokens.div_ceil(SPEC50_ROUTER_CHUNK_TOKENS) as u64,
            depth: 1,
        },
        metal::MTLSize {
            width: SPEC50_ROUTER_TG_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Some(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    const HIDDEN: usize = 2816;
    const EXPERTS: usize = 128;
    const EPS: f32 = 1e-6;
    const LAYERS: usize = 30;

    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next_u32() as f64 / u32::MAX as f64) as f32
        }
        fn sym(&mut self) -> f32 {
            self.unit() * 2.0 - 1.0
        }
    }

    fn shared(device: &Device, bytes: usize) -> Buffer {
        device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
    }

    fn buf_f32(device: &Device, values: &[f32]) -> Buffer {
        let b = shared(device, values.len() * 4);
        write_buffer_f32(&b, values);
        b
    }

    fn read_f32(buffer: &Buffer, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        read_buffer_f32(buffer, &mut out);
        out
    }

    /// Runs one command buffer and returns its hardware GPU-busy window in ms.
    fn run<F: FnOnce(&metal::ComputeCommandEncoderRef)>(queue: &CommandQueue, f: F) -> f64 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        f(e);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let (us, _) = command_buffer_gpu_times_us(cb);
        us as f64 / 1000.0
    }

    struct RouterFixture {
        device: Device,
        queue: CommandQueue,
        input: Vec<f32>,
        scale: Vec<f32>,
        weights: Vec<f32>,
        input_buf: Buffer,
        scale_buf: Buffer,
        weights_buf: Buffer,
        factors: Buffer,
        old_out: Buffer,
        new_out: Buffer,
    }

    fn router_fixture(seed: u64, k_max: usize) -> RouterFixture {
        let device = Device::system_default().expect("metal device");
        let queue = device.new_command_queue();
        let mut rng = Lcg(seed);
        let input: Vec<f32> = (0..k_max * HIDDEN).map(|_| rng.sym() * 3.0).collect();
        let scale: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + rng.sym() * 0.25).collect();
        let weights: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.sym() * 0.05).collect();
        let input_buf = buf_f32(&device, &input);
        let scale_buf = buf_f32(&device, &scale);
        let weights_buf = buf_f32(&device, &weights);
        let factors = shared(&device, spec50_router_factor_bytes(k_max) as usize);
        let old_out = shared(&device, k_max * EXPERTS * 4);
        let new_out = shared(&device, k_max * EXPERTS * 4);
        RouterFixture {
            device,
            queue,
            input,
            scale,
            weights,
            input_buf,
            scale_buf,
            weights_buf,
            factors,
            old_out,
            new_out,
        }
    }

    /// f64 oracle: the exact logits plus the L1 norm of the summed terms, so the f32
    /// error can be judged against the dot product's own backward-error scale.
    fn cpu_router_ref_f64(fx: &RouterFixture, k: usize) -> (Vec<f64>, Vec<f64>) {
        let mut out = vec![0f64; k * EXPERTS];
        let mut l1 = vec![0f64; k * EXPERTS];
        for t in 0..k {
            let x = &fx.input[t * HIDDEN..(t + 1) * HIDDEN];
            let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let rms_inv = 1.0 / (ss / HIDDEN as f64 + EPS as f64).sqrt();
            let factor = rms_inv * (1.0 / (HIDDEN as f64).sqrt());
            for e in 0..EXPERTS {
                let w = &fx.weights[e * HIDDEN..(e + 1) * HIDDEN];
                let mut acc = 0f64;
                let mut abs = 0f64;
                for (i, &wi) in w.iter().enumerate() {
                    let term = wi as f64 * (x[i] as f64 * factor * fx.scale[i] as f64);
                    acc += term;
                    abs += term.abs();
                }
                out[t * EXPERTS + e] = acc;
                l1[t * EXPERTS + e] = abs;
            }
        }
        (out, l1)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_vec4(
        e: &metal::ComputeCommandEncoderRef,
        kernels: &Spec50RouterKernels,
        input: &Buffer,
        fx: &RouterFixture,
        weights: &Buffer,
        out: &Buffer,
        k: usize,
        offset: u64,
    ) {
        encode_spec50_router_gemv_vec4_inexact(
            e,
            kernels,
            input,
            &fx.scale_buf,
            weights,
            out,
            &fx.factors,
            HIDDEN,
            EPS,
            EXPERTS,
            k,
            offset,
        )
        .expect("spec50 router vec4 encode rejected a supported shape");
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_new(
        e: &metal::ComputeCommandEncoderRef,
        kernels: &Spec50RouterKernels,
        input: &Buffer,
        fx: &RouterFixture,
        weights: &Buffer,
        out: &Buffer,
        k: usize,
        offset: u64,
    ) {
        encode_spec50_router_gemv(
            e,
            kernels,
            input,
            &fx.scale_buf,
            weights,
            out,
            &fx.factors,
            HIDDEN,
            EPS,
            EXPERTS,
            k,
            offset,
        )
        .expect("spec50 router encode rejected a supported shape");
    }

    /// The contract: bit-identical to `gemma4_router_batch_k_f32` for every K in
    /// 1..=8, and every token's logits identical whether it is evaluated alone
    /// (K=1) or inside a batch (per-token batch independence).
    #[test]
    fn spec50_router_gemv_bitwise_vs_old_and_batch_independent() {
        if !detect_metal_device().available {
            eprintln!("[spec50 router] no Metal device, skipping");
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels failed to compile");
        let old = metal_linear_kernel().expect("metal linear kernel");
        let fx = router_fixture(0x5eed_0001, 8);

        // Each token evaluated completely alone, as its own K=1 dispatch.
        let mut alone = vec![0f32; 8 * EXPERTS];
        for t in 0..8 {
            let tok_buf = buf_f32(&fx.device, &fx.input[t * HIDDEN..(t + 1) * HIDDEN]);
            run(&fx.queue, |e| {
                encode_new(e, kernels, &tok_buf, &fx, &fx.weights_buf, &fx.new_out, 1, 0);
            });
            alone[t * EXPERTS..(t + 1) * EXPERTS].copy_from_slice(&read_f32(&fx.new_out, EXPERTS));
        }

        let mut mismatches_total = 0usize;
        let mut max_abs_diff = 0f32;
        let mut max_ulp: u32 = 0;
        for k in 1..=8usize {
            run(&fx.queue, |e| {
                encode_gemma4_router_batch_k(
                    e,
                    old,
                    &fx.input_buf,
                    &fx.scale_buf,
                    &fx.weights_buf,
                    &fx.old_out,
                    HIDDEN,
                    EPS,
                    EXPERTS,
                    k,
                    0,
                )
                .unwrap();
                encode_new(
                    e,
                    kernels,
                    &fx.input_buf,
                    &fx,
                    &fx.weights_buf,
                    &fx.new_out,
                    k,
                    0,
                );
            });
            let got_old = read_f32(&fx.old_out, k * EXPERTS);
            let got_new = read_f32(&fx.new_out, k * EXPERTS);
            for i in 0..k * EXPERTS {
                let (a, b) = (got_old[i], got_new[i]);
                if a.to_bits() != b.to_bits() {
                    mismatches_total += 1;
                    max_abs_diff = max_abs_diff.max((a - b).abs());
                    max_ulp = max_ulp.max(a.to_bits().abs_diff(b.to_bits()));
                }
                assert_eq!(
                    b.to_bits(),
                    alone[i].to_bits(),
                    "K={k} token {} expert {}: batched result differs from its K=1 result",
                    i / EXPERTS,
                    i % EXPERTS
                );
            }
        }
        eprintln!(
            "[spec50 router] K=1..8 vs gemma4_router_batch_k_f32: {mismatches_total} mismatching \
             logits (max abs diff {max_abs_diff:.3e}, max ULP delta {max_ulp})"
        );
        assert_eq!(
            mismatches_total, 0,
            "new router logits are NOT bit-identical to gemma4_router_batch_k_f32 \
             (max ULP delta {max_ulp}, max abs diff {max_abs_diff:.3e})"
        );

        // Sanity floor: the f32 chain must still track the f64 oracle.
        let (reference, l1) = cpu_router_ref_f64(&fx, 8);
        run(&fx.queue, |e| {
            encode_new(
                e,
                kernels,
                &fx.input_buf,
                &fx,
                &fx.weights_buf,
                &fx.new_out,
                8,
                0,
            );
        });
        let got = read_f32(&fx.new_out, 8 * EXPERTS);
        let mut worst_rel_l1 = 0f64;
        for i in 0..8 * EXPERTS {
            worst_rel_l1 = worst_rel_l1.max((got[i] as f64 - reference[i]).abs() / l1[i]);
        }
        eprintln!("[spec50 router] vs f64 oracle: max err / term-L1 = {worst_rel_l1:.3e}");
        assert!(
            worst_rel_l1 <= 1e-5,
            "router f32 error {worst_rel_l1:.3e} exceeds 1e-5 of the term L1 norm"
        );
    }

    /// Documents WHY the shipping kernel keeps a scalar dot loop: the same layout
    /// with `float4` loads is not bit-exact. If this test ever reports 0 mismatches
    /// the toolchain changed, and the vec4 variant becomes eligible for promotion.
    #[test]
    fn spec50_router_gemv_vec4_is_not_bit_exact() {
        if !detect_metal_device().available {
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels");
        let old = metal_linear_kernel().expect("metal linear kernel");
        let fx = router_fixture(0x5eed_0001, 8);
        let mut mismatches = 0usize;
        let mut max_ulp: u32 = 0;
        let mut max_abs = 0f32;
        let mut total = 0usize;
        for k in 1..=8usize {
            run(&fx.queue, |e| {
                encode_gemma4_router_batch_k(
                    e,
                    old,
                    &fx.input_buf,
                    &fx.scale_buf,
                    &fx.weights_buf,
                    &fx.old_out,
                    HIDDEN,
                    EPS,
                    EXPERTS,
                    k,
                    0,
                )
                .unwrap();
                encode_vec4(
                    e,
                    kernels,
                    &fx.input_buf,
                    &fx,
                    &fx.weights_buf,
                    &fx.new_out,
                    k,
                    0,
                );
            });
            let a = read_f32(&fx.old_out, k * EXPERTS);
            let b = read_f32(&fx.new_out, k * EXPERTS);
            for i in 0..k * EXPERTS {
                total += 1;
                if a[i].to_bits() != b[i].to_bits() {
                    mismatches += 1;
                    max_ulp = max_ulp.max(a[i].to_bits().abs_diff(b[i].to_bits()));
                    max_abs = max_abs.max((a[i] - b[i]).abs());
                }
            }
        }
        eprintln!(
            "[spec50 router] vec4 variant vs gemma4_router_batch_k_f32: {mismatches}/{total} \
             logits differ, max ULP delta {max_ulp}, max abs diff {max_abs:.3e} \
             -- this is why the shipping kernel stays scalar"
        );
        assert!(
            mismatches > 0,
            "vec4 variant is now bit-exact; re-evaluate promoting it"
        );
    }

    /// Ragged K (not a multiple of the 8-token chunk) and a non-zero output offset.
    #[test]
    fn spec50_router_gemv_ragged_k_and_offset() {
        if !detect_metal_device().available {
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels");
        let old = metal_linear_kernel().expect("metal linear kernel");
        for k in [3usize, 5, 7, 8] {
            let fx = router_fixture(0x5eed_0002 + k as u64, k);
            let old_out = shared(&fx.device, (k + 4) * EXPERTS * 4);
            let new_out = shared(&fx.device, (k + 4) * EXPERTS * 4);
            let offset = (4 * EXPERTS * 4) as u64;
            run(&fx.queue, |e| {
                encode_gemma4_router_batch_k(
                    e,
                    old,
                    &fx.input_buf,
                    &fx.scale_buf,
                    &fx.weights_buf,
                    &old_out,
                    HIDDEN,
                    EPS,
                    EXPERTS,
                    k,
                    offset,
                )
                .unwrap();
                encode_new(
                    e,
                    kernels,
                    &fx.input_buf,
                    &fx,
                    &fx.weights_buf,
                    &new_out,
                    k,
                    offset,
                );
            });
            let a = read_f32(&old_out, (k + 4) * EXPERTS);
            let b = read_f32(&new_out, (k + 4) * EXPERTS);
            for i in 4 * EXPERTS..(k + 4) * EXPERTS {
                assert_eq!(a[i].to_bits(), b[i].to_bits(), "K={k}: mismatch at {i}");
            }
            // Nothing may be written before the offset.
            for i in 0..4 * EXPERTS {
                assert_eq!(b[i].to_bits(), 0, "K={k}: wrote below out_logits_offset");
            }
        }
    }

    /// 30 layers' worth of router GEMV per command buffer with a distinct 1.44 MB
    /// matrix per layer, old vs new, at K = 1 / 4 / 8. `--nocapture` to see it.
    #[test]
    fn spec50_router_gemv_bench_30_layers() {
        if !detect_metal_device().available {
            return;
        }
        let kernels = spec50_router_kernels().expect("spec50 kernels");
        let old = metal_linear_kernel().expect("metal linear kernel");
        let fx = router_fixture(0x5eed_0003, 8);
        let mut rng = Lcg(77);
        // 30 x 1.44 MB = 43 MB of router matrices, so no layer is served from L2.
        let layer_weights: Vec<Buffer> = (0..LAYERS)
            .map(|_| {
                let w: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.sym() * 0.05).collect();
                buf_f32(&fx.device, &w)
            })
            .collect();
        let matrix_bytes = (EXPERTS * HIDDEN * 4 * LAYERS) as f64;
        eprintln!(
            "[spec50 router bench] {LAYERS} layers/cb, hidden {HIDDEN}, experts {EXPERTS}, \
             {:.1} MB of router matrix per measurement",
            matrix_bytes / 1e6
        );
        eprintln!(
            "| K | old ms | new ms | old GB/s | new GB/s | speedup | vec4 ms (INEXACT) |"
        );
        eprintln!("|---|--------|--------|----------|----------|---------|-------------------|");
        for &k in &[1usize, 4, 8] {
            let mut best_old = f64::MAX;
            let mut best_new = f64::MAX;
            let mut best_vec4 = f64::MAX;
            for _round in 0..5 {
                let t_old = run(&fx.queue, |e| {
                    for w in &layer_weights {
                        encode_gemma4_router_batch_k(
                            e,
                            old,
                            &fx.input_buf,
                            &fx.scale_buf,
                            w,
                            &fx.old_out,
                            HIDDEN,
                            EPS,
                            EXPERTS,
                            k,
                            0,
                        )
                        .unwrap();
                    }
                });
                let t_new = run(&fx.queue, |e| {
                    for w in &layer_weights {
                        encode_new(e, kernels, &fx.input_buf, &fx, w, &fx.new_out, k, 0);
                    }
                });
                let t_vec4 = run(&fx.queue, |e| {
                    for w in &layer_weights {
                        encode_vec4(e, kernels, &fx.input_buf, &fx, w, &fx.new_out, k, 0);
                    }
                });
                best_old = best_old.min(t_old);
                best_new = best_new.min(t_new);
                best_vec4 = best_vec4.min(t_vec4);
            }
            // GB/s counts the router matrix only: that is the byte floor the lane
            // is being measured against (1.44 MB/layer, read once).
            let gbs = |ms: f64| matrix_bytes / (ms * 1e-3) / 1e9;
            eprintln!(
                "| {k} | {best_old:.3} | {best_new:.3} | {:.1} | {:.1} | {:.2}x | {best_vec4:.3} |",
                gbs(best_old),
                gbs(best_new),
                best_old / best_new
            );
        }
    }
}
