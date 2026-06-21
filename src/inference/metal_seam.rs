//! Metal offload seam.
//!
//! Keeps the per-matmul Metal (Apple GPU) offload attempts out of the x86/CUDA
//! CPU hot path. On macOS each wrapper runs the original opt-in Metal logic
//! verbatim; on every other target it is a no-op that returns the "not handled"
//! sentinel, so the shared matmul functions carry zero `metal::` references and
//! surgical x86/CUDA work cannot touch a Metal branch.
//!
//! Behaviour is byte-identical on all targets: the bodies below are the exact
//! expressions that previously lived inline in `inference.rs`, only relocated.
//! The runtime `q8_flags.metal*` gates are preserved.

// The macOS bodies reference inference.rs helpers via the glob; the non-macOS
// stubs need only the explicit type imports below for their signatures.
use super::q8_runtime::Q8RuntimeFlags;
#[cfg(target_os = "macos")]
use super::*;
use crate::tensor::Q8_0Block;

// ---- Inference session lifecycle (no-ops off macOS) ----

#[cfg(target_os = "macos")]
#[inline]
pub(super) fn start_inference_session() {
    crate::metal::start_inference_session();
}
#[cfg(not(target_os = "macos"))]
#[inline]
pub(super) fn start_inference_session() {}

#[cfg(target_os = "macos")]
#[inline]
pub(super) fn end_inference_session() {
    crate::metal::end_inference_session();
}
#[cfg(not(target_os = "macos"))]
#[inline]
pub(super) fn end_inference_session() {}

#[cfg(target_os = "macos")]
#[inline]
pub(super) fn synchronize_active_session() {
    crate::metal::synchronize_active_session();
}
#[cfg(not(target_os = "macos"))]
#[inline]
pub(super) fn synchronize_active_session() {}

#[cfg(target_os = "macos")]
#[inline]
pub(super) fn wire_mode_active() -> bool {
    crate::metal::wire_mode_active()
}
#[cfg(not(target_os = "macos"))]
#[inline]
pub(super) fn wire_mode_active() -> bool {
    false
}

// ---- Per-matmul Q8_0 offload (block reader, encoded single row) ----
// Was inline in `matmul_rhs_transposed_q8_0_block_reader_with_flags` (~15868).

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_encoded_row(
    q8_flags: &Q8RuntimeFlags,
    use_q8_0_block_dot: bool,
    quantized_input_blocks: &[Q8_0Block],
    chunk: &[u8],
    scales: &[f32],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    if use_q8_0_block_dot && q8_flags.metal {
        // Slice INSIDE the gate: when block-dot is off, quantized_input_blocks is
        // empty, so this slice would panic if hoisted to the (unconditional) call site.
        let (input_scales, input_quants) =
            q8_0_block_scales_and_quants(&quantized_input_blocks[..blocks_per_row]);
        crate::metal::try_q8_0_encoded_linear_row(
            &input_scales,
            &input_quants,
            chunk,
            scales,
            rows,
            blocks_per_row,
            output,
        )
    } else {
        false
    }
}
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_encoded_row(
    _q8_flags: &Q8RuntimeFlags,
    _use_q8_0_block_dot: bool,
    _quantized_input_blocks: &[Q8_0Block],
    _chunk: &[u8],
    _scales: &[f32],
    _rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

// ---- Per-matmul Q8_0 offload (block reader, encoded multi-row) ----
// Was inline in `matmul_rhs_transposed_q8_0_block_reader_with_flags` (~15974).

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_encoded_rows(
    q8_flags: &Q8RuntimeFlags,
    use_q8_0_block_dot: bool,
    quantized_input_blocks: &[Q8_0Block],
    chunk: &[u8],
    scales: &[f32],
    rows: usize,
    rows_this_chunk: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    if use_q8_0_block_dot && q8_flags.metal {
        let (input_scales, input_quants) = q8_0_block_scales_and_quants(quantized_input_blocks);
        crate::metal::try_q8_0_encoded_linear_rows(
            &input_scales,
            &input_quants,
            chunk,
            scales,
            rows,
            rows_this_chunk,
            blocks_per_row,
            output,
        )
    } else {
        false
    }
}
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_encoded_rows(
    _q8_flags: &Q8RuntimeFlags,
    _use_q8_0_block_dot: bool,
    _quantized_input_blocks: &[Q8_0Block],
    _chunk: &[u8],
    _scales: &[f32],
    _rows: usize,
    _rows_this_chunk: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

// ---- Retained-block transposed decode offload (hybrid + metal_retained) ----
// Was inline in `accumulate_transposed_linear_row_q8_0_block_dot_quantized_with_flags`
// (~17406-17450). Returns true if Metal handled the projection (caller returns early).

#[cfg(target_os = "macos")]
pub(super) fn try_transposed_block_offload(
    q8_flags: &Q8RuntimeFlags,
    quantized_input: &[Q8_0Block],
    weight_blocks: &[Q8_0Block],
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    if q8_flags.hybrid_retained {
        let gpu_rows = q8_flags.hybrid_gpu_rows_for_output(output.len());
        if gpu_rows > 0 && gpu_rows < output.len() {
            let cpu_rows = output.len() - gpu_rows;
            let gpu_block_start = cpu_rows * blocks_per_row;
            let (cpu_output, gpu_output) = output.split_at_mut(cpu_rows);
            let cpu_weight_blocks = &weight_blocks[..gpu_block_start];
            let gpu_weight_blocks = &weight_blocks[gpu_block_start..];
            let gpu_weight_bytes = q8_0_blocks_as_bytes(gpu_weight_blocks);
            if with_q8_0_block_scales_and_quants(quantized_input, |input_scales, input_quants| {
                crate::metal::try_q8_0_block_linear_row_with_cpu(
                    input_scales,
                    input_quants,
                    gpu_weight_bytes,
                    gpu_rows,
                    blocks_per_row,
                    gpu_output,
                    || {
                        accumulate_q8_0_block_dot_quantized_cpu(
                            quantized_input,
                            cpu_weight_blocks,
                            cpu_output,
                        )
                    },
                )
            }) {
                trace_q8_0_hybrid_retained_success(cpu_rows, gpu_rows, blocks_per_row);
                return true;
            }
        }
    }
    if q8_flags.metal_retained {
        let weight_bytes = q8_0_blocks_as_bytes(weight_blocks);
        if with_q8_0_block_scales_and_quants(quantized_input, |input_scales, input_quants| {
            crate::metal::try_q8_0_block_linear_row(
                input_scales,
                input_quants,
                weight_bytes,
                output.len(),
                blocks_per_row,
                output,
            )
        }) {
            return true;
        }
    }
    false
}
#[cfg(not(target_os = "macos"))]
pub(super) fn try_transposed_block_offload(
    _q8_flags: &Q8RuntimeFlags,
    _quantized_input: &[Q8_0Block],
    _weight_blocks: &[Q8_0Block],
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}

// ---- FFN gate/up hybrid (Metal GPU rows + CPU rows) ----
// 1:1 passthrough of the Metal call inside `try_gated_ffn_gate_up_hybrid_q8_0`
// (~8637). Mirrors `metal::try_q8_0_block_two_linear_rows_with_cpu`: on non-macOS
// the CPU work is NOT run here (the caller's hybrid_retained gate is off), exactly
// like the previous stub.

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_block_two_linear_rows_with_cpu<F: FnOnce()>(
    input_scales: &[f32],
    input_quants: &[i8],
    first_weight_blocks: &[u8],
    second_weight_blocks: &[u8],
    rows: usize,
    blocks_per_row: usize,
    first_output: &mut [f32],
    second_output: &mut [f32],
    cpu_work: F,
) -> bool {
    crate::metal::try_q8_0_block_two_linear_rows_with_cpu(
        input_scales,
        input_quants,
        first_weight_blocks,
        second_weight_blocks,
        rows,
        blocks_per_row,
        first_output,
        second_output,
        cpu_work,
    )
}
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_block_two_linear_rows_with_cpu<F: FnOnce()>(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _first_weight_blocks: &[u8],
    _second_weight_blocks: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
    _first_output: &mut [f32],
    _second_output: &mut [f32],
    _cpu_work: F,
) -> bool {
    false
}

// ---- Retained-block single-row decode offload (file-reader block-dot path) ----
// 1:1 passthrough of metal::try_q8_0_block_linear_row, runtime-gated by metal_retained.

#[cfg(target_os = "macos")]
pub(super) fn try_block_linear_row(
    input_scales: &[f32],
    input_quants: &[i8],
    weight_bytes: &[u8],
    rows: usize,
    blocks_per_row: usize,
    output: &mut [f32],
) -> bool {
    crate::metal::try_q8_0_block_linear_row(
        input_scales,
        input_quants,
        weight_bytes,
        rows,
        blocks_per_row,
        output,
    )
}
#[cfg(not(target_os = "macos"))]
pub(super) fn try_block_linear_row(
    _input_scales: &[f32],
    _input_quants: &[i8],
    _weight_bytes: &[u8],
    _rows: usize,
    _blocks_per_row: usize,
    _output: &mut [f32],
) -> bool {
    false
}
