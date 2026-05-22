use std::collections::HashMap;

use serde::Serialize;

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8ScheduleTelemetry {
    pub rayon_fanout_boundaries: u64,
    pub i8mm_single_projection_calls: u64,
    pub i8mm_fused_gate_up_calls: u64,
    pub i8mm_single_projection_by_role: HashMap<String, LlamaQ8ScheduleRoleTelemetry>,
    pub output_projection_calls: u64,
    pub output_projection_by_route: HashMap<String, LlamaQ8OutputProjectionRouteTelemetry>,
    pub output_projection_by_layer_route:
        HashMap<String, LlamaQ8OutputProjectionLayerRouteTelemetry>,
    pub projection_route_denials: HashMap<String, LlamaQ8ProjectionRouteDenialTelemetry>,
    pub ffn_gate_up_decode_consumer_activation_us: u64,
    pub ffn_gate_up_decode_consumer_tensor_us: u64,
    pub activation_pack_calls: u64,
    pub activation_pack_rows: u64,
    pub activation_pack_bytes_requested: u64,
    pub scratch_allocation_count: u64,
    pub scratch_bytes_allocated: u64,
    pub scratch_bytes_reused: u64,
    pub scratch_peak_capacity_bytes: u64,
    pub activation_quantize_pack_us: u64,
    pub q8_gemm_compute_us: u64,
    pub conservative_tail_rows: u64,
    pub ffn_down_gemm4_prefill_candidates: u64,
    pub ffn_down_gemm4_prefill_reject_plan_off: u64,
    pub ffn_down_gemm4_prefill_reject_rows_lt4: u64,
    pub ffn_down_gemm4_prefill_reject_bad_input_width: u64,
    pub ffn_down_gemm4_prefill_reject_no_runtime_packed: u64,
    pub ffn_down_gemm4_prefill_reject_non_i8_interleave: u64,
    pub ffn_down_decode_consumer_taken: u64,
    pub ffn_down_vnni_decode_candidates: u64,
    pub ffn_down_vnni_decode_taken: u64,
    pub ffn_down_vnni_decode_quantize_us: u64,
    pub ffn_down_vnni_decode_kernel_us: u64,
    pub ffn_down_vnni_decode_reject_gate_off: u64,
    pub ffn_down_vnni_decode_reject_cpu_feature: u64,
    pub ffn_down_vnni_decode_reject_no_vnni_pack: u64,
    pub ffn_down_vnni_decode_reject_bad_input_width: u64,
    pub ffn_down_vnni_decode_reject_bad_output_width: u64,
    pub ffn_down_vnni_decode_reject_shape_or_role: u64,
    pub prefill_single_token_fallbacks: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8ScheduleRoleTelemetry {
    pub calls: u64,
    pub rows: u64,
    pub pack_us: u64,
    pub gemm_us: u64,
    pub tail_rows: u64,
    pub rayon_fanout_boundaries: u64,
    pub scheduler_chunk_calls: u64,
    pub scheduler_output_groups: u64,
    pub scheduler_row_groups: u64,
    pub scheduler_groups_per_chunk: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8OutputProjectionRouteTelemetry {
    pub role: String,
    pub route: String,
    pub calls: u64,
    pub rows: u64,
    pub input_width: u64,
    pub output_width: u64,
    pub elapsed_us: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8OutputProjectionLayerRouteTelemetry {
    pub layer_index: usize,
    pub role: String,
    pub route: String,
    pub calls: u64,
    pub rows: u64,
    pub input_width: u64,
    pub output_width: u64,
    pub elapsed_us: u64,
}

#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaQ8ProjectionRouteDenialTelemetry {
    pub role: String,
    pub route: String,
    pub reason: String,
    pub denials: u64,
    pub rows: u64,
    pub input_width: u64,
    pub output_width: u64,
}
