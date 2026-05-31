use std::{
    collections::HashMap,
    env, mem,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    },
    time::Instant,
};

use serde::Serialize;

use crate::tensor::Q8_0PackedRows4Block;

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
    pub ffn_gate_up_decode_consumer_taken: u64,
    pub ffn_gate_up_decode_fused_activation_taken: u64,
    pub ffn_gate_up_decode_consumer_activation_us: u64,
    pub ffn_gate_up_decode_consumer_tensor_us: u64,
    pub ffn_decode_chain_taken: u64,
    pub ffn_decode_chain_total_us: u64,
    pub ffn_decode_chain_input_quantize_us: u64,
    pub ffn_decode_chain_activation_quantize_us: u64,
    pub ffn_decode_chain_down_us: u64,
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

pub(super) const Q8_SCHEDULE_TELEMETRY_ENV: &str = "CAMELID_Q8_SCHED_TELEMETRY";

pub(super) static Q8_SCHED_RAYON_FANOUT_BOUNDARIES: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_OUTPUT_PROJECTION_CALLS: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TAKEN: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_GATE_UP_DECODE_FUSED_ACTIVATION_TAKEN: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DECODE_CHAIN_TAKEN: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DECODE_CHAIN_TOTAL_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DECODE_CHAIN_INPUT_QUANTIZE_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DECODE_CHAIN_ACTIVATION_QUANTIZE_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DECODE_CHAIN_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_ACTIVATION_PACK_CALLS: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_ACTIVATION_PACK_ROWS: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_SCRATCH_ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_SCRATCH_BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_SCRATCH_BYTES_REUSED: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_Q8_GEMM_COMPUTE_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_CONSERVATIVE_TAIL_ROWS: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH: AtomicU64 =
    AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED: AtomicU64 =
    AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE: AtomicU64 =
    AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH: AtomicU64 =
    AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH: AtomicU64 =
    AtomicU64::new(0);
pub(super) static Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS: AtomicU64 = AtomicU64::new(0);
pub(super) static Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE: OnceLock<
    Mutex<HashMap<String, LlamaQ8ScheduleRoleTelemetry>>,
> = OnceLock::new();
pub(super) static Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE: OnceLock<
    Mutex<HashMap<String, LlamaQ8OutputProjectionRouteTelemetry>>,
> = OnceLock::new();
pub(super) static Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE: OnceLock<
    Mutex<HashMap<String, LlamaQ8OutputProjectionLayerRouteTelemetry>>,
> = OnceLock::new();
pub(super) static Q8_SCHED_PROJECTION_ROUTE_DENIALS: OnceLock<
    Mutex<HashMap<String, LlamaQ8ProjectionRouteDenialTelemetry>>,
> = OnceLock::new();
#[cfg(not(test))]
pub(super) static Q8_SCHED_TELEMETRY_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn q8_schedule_telemetry_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(Q8_SCHEDULE_TELEMETRY_ENV)
    }
    #[cfg(not(test))]
    *Q8_SCHED_TELEMETRY_ENABLED.get_or_init(|| env_flag_enabled(Q8_SCHEDULE_TELEMETRY_ENV))
}

pub fn reset_q8_schedule_telemetry() {
    Q8_SCHED_RAYON_FANOUT_BOUNDARIES.store(0, Ordering::Relaxed);
    Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_OUTPUT_PROJECTION_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TAKEN.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_GATE_UP_DECODE_FUSED_ACTIVATION_TAKEN.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DECODE_CHAIN_TAKEN.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DECODE_CHAIN_TOTAL_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DECODE_CHAIN_INPUT_QUANTIZE_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DECODE_CHAIN_ACTIVATION_QUANTIZE_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DECODE_CHAIN_DOWN_US.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_CALLS.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_ROWS.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_BYTES_ALLOCATED.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_BYTES_REUSED.store(0, Ordering::Relaxed);
    Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES.store(0, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US.store(0, Ordering::Relaxed);
    Q8_SCHED_Q8_GEMM_COMPUTE_US.store(0, Ordering::Relaxed);
    Q8_SCHED_CONSERVATIVE_TAIL_ROWS.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH.store(0, Ordering::Relaxed);
    Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE.store(0, Ordering::Relaxed);
    Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS.store(0, Ordering::Relaxed);
    if let Some(by_role) = Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE.get() {
        by_role.lock().expect("q8 role telemetry mutex").clear();
    }
    if let Some(by_route) = Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE.get() {
        by_route
            .lock()
            .expect("q8 output projection telemetry mutex")
            .clear();
    }
    if let Some(by_layer_route) = Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE.get() {
        by_layer_route
            .lock()
            .expect("q8 output projection layer route telemetry mutex")
            .clear();
    }
    if let Some(denials) = Q8_SCHED_PROJECTION_ROUTE_DENIALS.get() {
        denials
            .lock()
            .expect("q8 projection route denial telemetry mutex")
            .clear();
    }
}

pub fn snapshot_q8_schedule_telemetry() -> LlamaQ8ScheduleTelemetry {
    LlamaQ8ScheduleTelemetry {
        rayon_fanout_boundaries: Q8_SCHED_RAYON_FANOUT_BOUNDARIES.load(Ordering::Relaxed),
        i8mm_single_projection_calls: Q8_SCHED_I8MM_SINGLE_PROJECTION_CALLS.load(Ordering::Relaxed),
        i8mm_fused_gate_up_calls: Q8_SCHED_I8MM_FUSED_GATE_UP_CALLS.load(Ordering::Relaxed),
        i8mm_single_projection_by_role: Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE
            .get()
            .map(|by_role| by_role.lock().expect("q8 role telemetry mutex").clone())
            .unwrap_or_default(),
        output_projection_calls: Q8_SCHED_OUTPUT_PROJECTION_CALLS.load(Ordering::Relaxed),
        output_projection_by_route: Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE
            .get()
            .map(|by_route| {
                by_route
                    .lock()
                    .expect("q8 output projection telemetry mutex")
                    .clone()
            })
            .unwrap_or_default(),
        output_projection_by_layer_route: Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE
            .get()
            .map(|by_layer_route| {
                by_layer_route
                    .lock()
                    .expect("q8 output projection layer route telemetry mutex")
                    .clone()
            })
            .unwrap_or_default(),
        projection_route_denials: Q8_SCHED_PROJECTION_ROUTE_DENIALS
            .get()
            .map(|denials| {
                denials
                    .lock()
                    .expect("q8 projection route denial telemetry mutex")
                    .clone()
            })
            .unwrap_or_default(),
        ffn_gate_up_decode_consumer_taken: Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TAKEN
            .load(Ordering::Relaxed),
        ffn_gate_up_decode_fused_activation_taken:
            Q8_SCHED_FFN_GATE_UP_DECODE_FUSED_ACTIVATION_TAKEN.load(Ordering::Relaxed),
        ffn_gate_up_decode_consumer_activation_us:
            Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_ACTIVATION_US.load(Ordering::Relaxed),
        ffn_gate_up_decode_consumer_tensor_us: Q8_SCHED_FFN_GATE_UP_DECODE_CONSUMER_TENSOR_US
            .load(Ordering::Relaxed),
        ffn_decode_chain_taken: Q8_SCHED_FFN_DECODE_CHAIN_TAKEN.load(Ordering::Relaxed),
        ffn_decode_chain_total_us: Q8_SCHED_FFN_DECODE_CHAIN_TOTAL_US.load(Ordering::Relaxed),
        ffn_decode_chain_input_quantize_us: Q8_SCHED_FFN_DECODE_CHAIN_INPUT_QUANTIZE_US
            .load(Ordering::Relaxed),
        ffn_decode_chain_activation_quantize_us: Q8_SCHED_FFN_DECODE_CHAIN_ACTIVATION_QUANTIZE_US
            .load(Ordering::Relaxed),
        ffn_decode_chain_down_us: Q8_SCHED_FFN_DECODE_CHAIN_DOWN_US.load(Ordering::Relaxed),
        activation_pack_calls: Q8_SCHED_ACTIVATION_PACK_CALLS.load(Ordering::Relaxed),
        activation_pack_rows: Q8_SCHED_ACTIVATION_PACK_ROWS.load(Ordering::Relaxed),
        activation_pack_bytes_requested: Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED
            .load(Ordering::Relaxed),
        scratch_allocation_count: Q8_SCHED_SCRATCH_ALLOCATION_COUNT.load(Ordering::Relaxed),
        scratch_bytes_allocated: Q8_SCHED_SCRATCH_BYTES_ALLOCATED.load(Ordering::Relaxed),
        scratch_bytes_reused: Q8_SCHED_SCRATCH_BYTES_REUSED.load(Ordering::Relaxed),
        scratch_peak_capacity_bytes: Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES.load(Ordering::Relaxed),
        activation_quantize_pack_us: Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US.load(Ordering::Relaxed),
        q8_gemm_compute_us: Q8_SCHED_Q8_GEMM_COMPUTE_US.load(Ordering::Relaxed),
        conservative_tail_rows: Q8_SCHED_CONSERVATIVE_TAIL_ROWS.load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_candidates: Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_CANDIDATES
            .load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_plan_off: Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_PLAN_OFF
            .load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_rows_lt4: Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_ROWS_LT4
            .load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_bad_input_width:
            Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_BAD_INPUT_WIDTH.load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_no_runtime_packed:
            Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NO_RUNTIME_PACKED.load(Ordering::Relaxed),
        ffn_down_gemm4_prefill_reject_non_i8_interleave:
            Q8_SCHED_FFN_DOWN_GEMM4_PREFILL_REJECT_NON_I8_INTERLEAVE.load(Ordering::Relaxed),
        ffn_down_decode_consumer_taken: Q8_SCHED_FFN_DOWN_DECODE_CONSUMER_TAKEN
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_candidates: Q8_SCHED_FFN_DOWN_VNNI_DECODE_CANDIDATES
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_taken: Q8_SCHED_FFN_DOWN_VNNI_DECODE_TAKEN.load(Ordering::Relaxed),
        ffn_down_vnni_decode_quantize_us: Q8_SCHED_FFN_DOWN_VNNI_DECODE_QUANTIZE_US
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_kernel_us: Q8_SCHED_FFN_DOWN_VNNI_DECODE_KERNEL_US
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_gate_off: Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_GATE_OFF
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_cpu_feature: Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_CPU_FEATURE
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_no_vnni_pack: Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_NO_VNNI_PACK
            .load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_bad_input_width:
            Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_INPUT_WIDTH.load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_bad_output_width:
            Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_BAD_OUTPUT_WIDTH.load(Ordering::Relaxed),
        ffn_down_vnni_decode_reject_shape_or_role:
            Q8_SCHED_FFN_DOWN_VNNI_DECODE_REJECT_SHAPE_OR_ROLE.load(Ordering::Relaxed),
        prefill_single_token_fallbacks: Q8_SCHED_PREFILL_SINGLE_TOKEN_FALLBACKS
            .load(Ordering::Relaxed),
    }
}

#[allow(dead_code)]
pub(super) fn add_q8_schedule_counter(counter: &AtomicU64, value: u64) {
    if q8_schedule_telemetry_enabled() && value > 0 {
        counter.fetch_add(value, Ordering::Relaxed);
    }
}

#[allow(dead_code)]
pub(super) fn update_q8_schedule_peak(counter: &AtomicU64, value: u64) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    let mut current = counter.load(Ordering::Relaxed);
    while value > current {
        match counter.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_activation_pack(
    packed_inputs: &mut Vec<Q8_0PackedRows4Block>,
    before_capacity: usize,
    packed_rows: usize,
    blocks_per_row: usize,
    elapsed_us: u128,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    let requested_blocks = packed_rows / 4 * blocks_per_row;
    let requested_bytes = requested_blocks.saturating_mul(mem::size_of::<Q8_0PackedRows4Block>());
    let before_capacity_bytes =
        before_capacity.saturating_mul(mem::size_of::<Q8_0PackedRows4Block>());
    let after_capacity_bytes = packed_inputs
        .capacity()
        .saturating_mul(mem::size_of::<Q8_0PackedRows4Block>());
    Q8_SCHED_ACTIVATION_PACK_CALLS.fetch_add(1, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_ROWS.fetch_add(packed_rows as u64, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_PACK_BYTES_REQUESTED.fetch_add(requested_bytes as u64, Ordering::Relaxed);
    Q8_SCHED_ACTIVATION_QUANTIZE_PACK_US.fetch_add(elapsed_us as u64, Ordering::Relaxed);
    if before_capacity < requested_blocks {
        Q8_SCHED_SCRATCH_ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        Q8_SCHED_SCRATCH_BYTES_ALLOCATED.fetch_add(
            after_capacity_bytes.saturating_sub(before_capacity_bytes) as u64,
            Ordering::Relaxed,
        );
    } else {
        Q8_SCHED_SCRATCH_BYTES_REUSED.fetch_add(requested_bytes as u64, Ordering::Relaxed);
    }
    update_q8_schedule_peak(
        &Q8_SCHED_SCRATCH_PEAK_CAPACITY_BYTES,
        after_capacity_bytes as u64,
    );
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_i8mm_single_projection_role_call(
    role: &str,
    rows: u64,
    rayon_fanout_boundaries: u64,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.calls = entry.calls.saturating_add(1);
        entry.rows = entry.rows.saturating_add(rows);
        entry.rayon_fanout_boundaries = entry
            .rayon_fanout_boundaries
            .saturating_add(rayon_fanout_boundaries);
    });
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_i8mm_single_projection_role_pack(role: &str, elapsed_us: u128) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.pack_us = entry.pack_us.saturating_add(elapsed_us as u64);
    });
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_i8mm_single_projection_role_gemm(role: &str, elapsed_us: u128) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.gemm_us = entry.gemm_us.saturating_add(elapsed_us as u64);
    });
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_i8mm_single_projection_role_tail(role: &str, tail_rows: u64) {
    if !q8_schedule_telemetry_enabled() || tail_rows == 0 {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.tail_rows = entry.tail_rows.saturating_add(tail_rows);
    });
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_i8mm_single_projection_role_scheduler(
    role: &str,
    output_groups: u64,
    row_groups: u64,
    groups_per_chunk: u64,
) {
    if !q8_schedule_telemetry_enabled() || output_groups == 0 {
        return;
    }
    update_q8_schedule_role(role, |entry| {
        entry.scheduler_chunk_calls = entry.scheduler_chunk_calls.saturating_add(1);
        entry.scheduler_output_groups = entry.scheduler_output_groups.saturating_add(output_groups);
        entry.scheduler_row_groups = entry.scheduler_row_groups.saturating_add(row_groups);
        entry.scheduler_groups_per_chunk = entry
            .scheduler_groups_per_chunk
            .saturating_add(groups_per_chunk.max(1));
    });
}

#[allow(dead_code)]
fn update_q8_schedule_role(role: &str, update: impl FnOnce(&mut LlamaQ8ScheduleRoleTelemetry)) {
    let by_role =
        Q8_SCHED_I8MM_SINGLE_PROJECTION_BY_ROLE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut by_role = by_role.lock().expect("q8 role telemetry mutex");
    update(by_role.entry(role.to_string()).or_default());
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_output_projection_route_call(
    role: &str,
    route: &str,
    projection_name: Option<&str>,
    rows: usize,
    input_width: usize,
    output_width: usize,
    elapsed_us: u128,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    Q8_SCHED_OUTPUT_PROJECTION_CALLS.fetch_add(1, Ordering::Relaxed);
    let key = format!("{role}.{route}");
    let by_route = Q8_SCHED_OUTPUT_PROJECTION_BY_ROUTE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut by_route = by_route
        .lock()
        .expect("q8 output projection telemetry mutex");
    let entry = by_route
        .entry(key)
        .or_insert_with(|| LlamaQ8OutputProjectionRouteTelemetry {
            role: role.to_string(),
            route: route.to_string(),
            ..LlamaQ8OutputProjectionRouteTelemetry::default()
        });
    entry.calls = entry.calls.saturating_add(1);
    entry.rows = entry.rows.saturating_add(rows as u64);
    entry.input_width = input_width as u64;
    entry.output_width = output_width as u64;
    entry.elapsed_us = entry.elapsed_us.saturating_add(elapsed_us as u64);

    let Some(layer_index) = projection_name.and_then(q8_schedule_layer_index_for_projection_name)
    else {
        return;
    };
    let key = format!("layer_{layer_index}.{role}.{route}");
    let by_layer_route =
        Q8_SCHED_OUTPUT_PROJECTION_BY_LAYER_ROUTE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut by_layer_route = by_layer_route
        .lock()
        .expect("q8 output projection layer route telemetry mutex");
    let entry =
        by_layer_route
            .entry(key)
            .or_insert_with(|| LlamaQ8OutputProjectionLayerRouteTelemetry {
                layer_index,
                role: role.to_string(),
                route: route.to_string(),
                ..LlamaQ8OutputProjectionLayerRouteTelemetry::default()
            });
    entry.calls = entry.calls.saturating_add(1);
    entry.rows = entry.rows.saturating_add(rows as u64);
    entry.input_width = input_width as u64;
    entry.output_width = output_width as u64;
    entry.elapsed_us = entry.elapsed_us.saturating_add(elapsed_us as u64);
}

pub(super) fn q8_schedule_layer_index_for_projection_name(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("layer_")?;
    let (digits, _) = rest.split_once('_')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

pub(super) fn record_q8_schedule_projection_route_elapsed(
    role: &str,
    route: &str,
    projection_name: &str,
    rows: usize,
    input_width: usize,
    output_width: usize,
    started: Option<Instant>,
) {
    if let Some(started) = started {
        record_q8_schedule_output_projection_route_call(
            role,
            route,
            Some(projection_name),
            rows,
            input_width,
            output_width,
            started.elapsed().as_micros(),
        );
    }
}

#[allow(dead_code)]
pub(super) fn record_q8_schedule_projection_route_denial(
    role: &str,
    route: &str,
    reason: &str,
    rows: usize,
    input_width: usize,
    output_width: usize,
) {
    if !q8_schedule_telemetry_enabled() {
        return;
    }
    let key = format!("{role}.{route}.{reason}");
    let denials = Q8_SCHED_PROJECTION_ROUTE_DENIALS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut denials = denials
        .lock()
        .expect("q8 projection route denial telemetry mutex");
    let entry = denials
        .entry(key)
        .or_insert_with(|| LlamaQ8ProjectionRouteDenialTelemetry {
            role: role.to_string(),
            route: route.to_string(),
            reason: reason.to_string(),
            ..LlamaQ8ProjectionRouteDenialTelemetry::default()
        });
    entry.denials = entry.denials.saturating_add(1);
    entry.rows = entry.rows.saturating_add(rows as u64);
    entry.input_width = input_width as u64;
    entry.output_width = output_width as u64;
}

fn env_flag_enabled(key: &str) -> bool {
    matches!(
        env::var(key).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") | Ok("on") | Ok("ON")
    )
}
