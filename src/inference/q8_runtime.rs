use std::env;

use super::{diagnostic_linear_accumulation_precision, LinearAccumulationPrecision};
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Q8RuntimeFlags {
    pub(super) block_dot: bool,
    pub(super) file_reader_block_dot: bool,
    pub(super) attention_projection_decode_consumer: bool,
    pub(super) attention_output_decode_consumer: bool,
    pub(super) attention_output_packed_rows4_matmul: bool,
    pub(super) attention_qkv_decode_consumer: bool,
    pub(super) attention_qkv_decode_group_chunking: bool,
    pub(super) attention_qkv_packed_rows4_matmul: bool,
    pub(super) output_packed_rows4_matmul: bool,
    pub(super) output_decode_owner: bool,
    pub(super) ffn_gate_up_decode_consumer: bool,
    pub(super) ffn_gate_up_decode_group_chunking: bool,
    pub(super) ffn_gate_up_decode_fused_activation: bool,
    pub(super) ffn_gate_up_decode_paired_dot: bool,
    pub(super) ffn_gate_up_packed_rows4_matmul: bool,
    pub(super) ffn_gate_up_single_owner: bool,
    pub(super) ffn_down_decode_consumer: bool,
    pub(super) ffn_down_packed_rows4_matmul: bool,
    pub(super) ffn_down_gemm4_prefill: bool,
    pub(super) ffn_down_gemm4_row_group_schedule: bool,
    pub(super) ffn_down_gemm4_avx2: bool,
    pub(super) ffn_down_amx_prefill: bool,
    pub(super) ffn_down_single_owner: bool,
    pub(super) ffn_down_vnni_decode: bool,
    pub(super) metal: bool,
    pub(super) metal_retained: bool,
    pub(super) hybrid_retained: bool,
    pub(super) hybrid_gpu_rows: Option<usize>,
    pub(super) hybrid_gpu_percent: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ResolvedRuntimePlan {
    pub(super) linear_accumulation_precision: LinearAccumulationPrecision,
    pub(super) q8: Q8RuntimeFlags,
}

impl ResolvedRuntimePlan {
    pub(super) fn from_env() -> Result<Self> {
        Ok(Self {
            linear_accumulation_precision: diagnostic_linear_accumulation_precision()?,
            q8: Q8RuntimeFlags::from_env(),
        })
    }
}

impl Q8RuntimeFlags {
    pub(super) fn from_env() -> Self {
        Self {
            block_dot: q8_0_env_flag_enabled_default_on_fail_closed("CAMELID_Q8_0_BLOCK_DOT"),
            file_reader_block_dot: q8_0_env_flag_enabled_default_on_fail_closed(
                "CAMELID_Q8_0_FILE_READER_BLOCK_DOT",
            ),
            attention_projection_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
            ),
            attention_output_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
            ),
            attention_output_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
            ),
            attention_qkv_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
            ),
            attention_qkv_decode_group_chunking: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
            ),
            attention_qkv_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
            ),
            output_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
            ),
            output_decode_owner: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
            ),
            ffn_gate_up_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            ) || q8_0_env_flag_enabled_default_off(
                "CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            ),
            ffn_gate_up_decode_group_chunking: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
            ),
            ffn_gate_up_decode_fused_activation: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
            ),
            ffn_gate_up_decode_paired_dot: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
            ),
            ffn_gate_up_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
            ),
            ffn_gate_up_single_owner: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
            ),
            ffn_down_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
            ) || q8_0_env_flag_enabled_default_off(
                "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
            ),
            ffn_down_packed_rows4_matmul: x86_q8_ffn_down_packed_rows4_matmul_enabled(),
            ffn_down_gemm4_prefill: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
            ),
            ffn_down_gemm4_row_group_schedule: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
            ),
            ffn_down_gemm4_avx2: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
            ),
            ffn_down_amx_prefill: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
            ),
            ffn_down_single_owner: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
            ),
            ffn_down_vnni_decode: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
            ),
            metal: q8_0_env_flag_enabled_default_off("CAMELID_METAL_Q8"),
            metal_retained: q8_0_env_flag_enabled_default_off("CAMELID_METAL_Q8_RETAINED"),
            hybrid_retained: q8_0_env_flag_enabled_default_off("CAMELID_HYBRID_Q8_RETAINED"),
            hybrid_gpu_rows: env::var("CAMELID_HYBRID_Q8_GPU_ROWS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok()),
            hybrid_gpu_percent: env::var("CAMELID_HYBRID_Q8_GPU_PERCENT")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(10)
                .min(90),
        }
    }

    pub(super) fn hybrid_gpu_rows_for_output(self, output_rows: usize) -> usize {
        if output_rows < 2 {
            return 0;
        }
        if let Some(rows) = self.hybrid_gpu_rows {
            return rows.min(output_rows.saturating_sub(1));
        }
        ((output_rows * self.hybrid_gpu_percent).div_ceil(100))
            .max(1)
            .min(output_rows.saturating_sub(1))
    }
}

pub(super) fn q8_0_env_flag_enabled_default_on_fail_closed(key: &str) -> bool {
    match env::var(key) {
        Ok(value) => {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        }
        Err(_) => true,
    }
}

pub(super) fn q8_0_env_flag_enabled_default_off(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

pub(super) fn q8_0_env_flag_disabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("disabled")
                || value.eq_ignore_ascii_case("dequantized")
                || value.eq_ignore_ascii_case("f32")
        })
        .unwrap_or(false)
}

fn x86_q8_ffn_down_packed_rows4_matmul_enabled() -> bool {
    q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL")
        || q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL")
}
