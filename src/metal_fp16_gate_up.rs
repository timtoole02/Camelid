// Native FP16 diagnostic only. Gate/up weights occupy one contiguous mirror;
// the combined GEMM produces token rows [gate8192,up8192].
fn fp16_probe_gate_up_fused_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_BENCH_FP16_GATE_UP_FUSED")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

fn fp16_probe_gate_up_silu_pipeline(
    k: &MetalLinearKernel,
) -> Option<&'static ComputePipelineState> {
    static PIPELINE: OnceLock<Option<ComputePipelineState>> = OnceLock::new();
    PIPELINE
        .get_or_init(|| {
            // Match ELEMENTWISE_SHADER's default compile options and preserve the
            // original silu_mul_f32 expression; only operand addresses differ.
            let options = CompileOptions::new();
            let source = r#"
#include <metal_stdlib>
using namespace metal;
kernel void fp16_probe_gate_up_silu(
    device const float* combined [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const uint token = gid / 8192;
    const uint col = gid % 8192;
    device const float* gate = combined + token * 16384;
    device const float* up = gate + 8192;
    float g = gate[col];
    output[gid] = (g / (1.0 + exp(-g))) * up[col];
}
"#;
            let library = k
                .device
                .new_library_with_source(source, &options)
                .map_err(|error| eprintln!("[fp16-gate-up] SiLU compile failed: {error}"))
                .ok()?;
            let function = library.get_function("fp16_probe_gate_up_silu", None).ok()?;
            let pipeline = k
                .device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            admitted_32_lane_pipeline(Some(&pipeline))?;
            Some(pipeline)
        })
        .as_ref()
}

fn encode_fp16_probe_gate_up_silu(
    e: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    combined: &Buffer,
    output: &Buffer,
    count: &Buffer,
    count_offset: u64,
    rows: usize,
) {
    assert!((1..=64).contains(&rows));
    assert!(combined.length() >= (rows * 16384 * 4) as u64);
    assert!(output.length() >= (rows * 8192 * 4) as u64);
    e.set_compute_pipeline_state(pipeline);
    e.set_buffer(0, Some(combined), 0);
    e.set_buffer(1, Some(output), 0);
    e.set_buffer(2, Some(count), count_offset);
    dispatch_1d(e, pipeline, rows * 8192);
}

#[cfg(test)]
include!("metal_fp16_gate_up_tests.rs");
