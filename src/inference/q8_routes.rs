use super::{
    q8_0_selected_borrowed_packed_rows4, BorrowedLinearWeight, GgufTensorType, Q8_0_BLOCK_VALUES,
};

#[allow(dead_code)]
pub(super) fn q8_schedule_output_projection_route_kind(
    weight: BorrowedLinearWeight<'_>,
    input_width: usize,
    output_width: usize,
) -> &'static str {
    if weight.source_type == Some(GgufTensorType::Q8_0) {
        if q8_0_selected_borrowed_packed_rows4(weight)
            .filter(|(packed, _)| {
                packed.rows == output_width
                    && packed.blocks_per_row == input_width / Q8_0_BLOCK_VALUES
            })
            .is_some()
        {
            "q8_0_borrowed_packed_rows4"
        } else if weight.q8_0_blocks.is_some() {
            "q8_0_retained_blocks"
        } else if weight.q8_0_file_backing.is_some()
            && weight.cols == input_width
            && weight.rows == output_width
            && input_width.is_multiple_of(Q8_0_BLOCK_VALUES)
        {
            "q8_0_file_reader"
        } else {
            "q8_0_f32_fallback"
        }
    } else {
        "f32"
    }
}

#[allow(dead_code)]
pub(super) fn q8_schedule_role_for_output_name(name: &str) -> &'static str {
    if name.contains("attention_q") || name.contains("attn_q") {
        "attention_q"
    } else if name.contains("attention_k") || name.contains("attn_k") {
        "attention_k"
    } else if name.contains("attention_v") || name.contains("attn_v") {
        "attention_v"
    } else if name.contains("attention_output") || name.contains("attn_output") {
        "attention_output"
    } else if name.contains("ffn_gate") {
        "ffn_gate"
    } else if name.contains("ffn_up") {
        "ffn_up"
    } else if name.contains("ffn_down") {
        "ffn_down"
    } else if name.contains("logits") {
        "logits"
    } else {
        "unknown"
    }
}
