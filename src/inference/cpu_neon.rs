//! Apple-Silicon (aarch64 NEON / dotprod) Q8 leaf kernels, relocated out of
//! inference.rs so the matmul region carries only the cfg-gated dispatch seams.
//! Module is gated on `target_arch = "aarch64"` (not macOS) so it compiles on
//! aarch64-linux for CI verification; the dispatch sites stay macOS-gated, so off
//! macOS these are unused (dead_code allowed). Bodies are byte-for-byte the
//! originals — reduction order and accumulation are unchanged.

use super::q8_runtime::q8_0_env_flag_disabled;
use super::*;
use crate::tensor::Q8_0Block;

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_dot_rows_neon_dotprod(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;

    let mut total_sum = 0.0_f32;

    for (idx, (w_block, i_block)) in weight.iter().zip(input).enumerate() {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if let Some(next_block) = weight.get(idx + 2) {
                asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) next_block.quants.as_ptr(),
                    options(nostack, preserves_flags, readonly)
                );
            }
        }
        let weight_lo = vld1q_s8(w_block.quants.as_ptr());
        let input_lo = vld1q_s8(i_block.quants.as_ptr());
        let weight_hi = vld1q_s8(w_block.quants.as_ptr().add(16));
        let input_hi = vld1q_s8(i_block.quants.as_ptr().add(16));

        let mut acc = vdupq_n_s32(0);
        asm!(
            "sdot {acc:v}.4s, {weight_lo:v}.16b, {input_lo:v}.16b",
            "sdot {acc:v}.4s, {weight_hi:v}.16b, {input_hi:v}.16b",
            acc = inout(vreg) acc,
            weight_lo = in(vreg) weight_lo,
            input_lo = in(vreg) input_lo,
            weight_hi = in(vreg) weight_hi,
            input_hi = in(vreg) input_hi,
            options(nostack, preserves_flags)
        );

        // Keep horizontal sum exactly identical to existing register horizontal sum
        let int_sum = horizontal_sum_i32x4(acc);
        total_sum += int_sum as f32 * w_block.scale * i_block.scale;
    }

    total_sum
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_wire_row_dot_neon_dotprod(
    weight_wire: &[u8],
    input: &[Q8_0Block],
) -> f32 {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;
    const WIRE: usize = 34;

    let mut total = 0.0_f32;
    for (b, i_block) in input.iter().enumerate() {
        let base = b * WIRE;
        let scale = f16_bits_to_f32(u16::from_le_bytes([
            weight_wire[base],
            weight_wire[base + 1],
        ]));
        let qptr = weight_wire.as_ptr().add(base + 2) as *const i8;

        if let Some(next) = input.get(b + 2) {
            let _ = next; // prefetch the weight bytes two blocks ahead
            asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) weight_wire.as_ptr().add((b + 2) * WIRE),
                options(nostack, preserves_flags, readonly)
            );
        }

        let weight_lo = vld1q_s8(qptr);
        let weight_hi = vld1q_s8(qptr.add(16));
        let input_lo = vld1q_s8(i_block.quants.as_ptr());
        let input_hi = vld1q_s8(i_block.quants.as_ptr().add(16));

        let mut acc = vdupq_n_s32(0);
        asm!(
            "sdot {acc:v}.4s, {weight_lo:v}.16b, {input_lo:v}.16b",
            "sdot {acc:v}.4s, {weight_hi:v}.16b, {input_hi:v}.16b",
            acc = inout(vreg) acc,
            weight_lo = in(vreg) weight_lo,
            input_lo = in(vreg) input_lo,
            weight_hi = in(vreg) weight_hi,
            input_hi = in(vreg) input_hi,
            options(nostack, preserves_flags)
        );

        let int_sum = horizontal_sum_i32x4(acc);
        total += int_sum as f32 * scale * i_block.scale;
    }

    total
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q4_0_wire_row_dot_neon_dotprod(
    weight_wire: &[u8],
    input: &[Q8_0Block],
) -> f32 {
    use std::arch::aarch64::{
        vandq_u8, vdupq_n_s32, vdupq_n_s8, vdupq_n_u8, vld1q_s8, vld1q_u8, vreinterpretq_s8_u8,
        vshrq_n_u8, vsubq_s8,
    };
    use std::arch::asm;
    const WIRE: usize = Q4_0_WIRE_BYTES_PER_BLOCK;

    let mask = vdupq_n_u8(0x0F);
    let bias = vdupq_n_s8(8);
    let mut total = 0.0_f32;
    for (b, i_block) in input.iter().enumerate() {
        let base = b * WIRE;
        let scale = f16_bits_to_f32(u16::from_le_bytes([
            weight_wire[base],
            weight_wire[base + 1],
        ]));

        if input.get(b + 2).is_some() {
            asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) weight_wire.as_ptr().add((b + 2) * WIRE),
                options(nostack, preserves_flags, readonly)
            );
        }

        let packed = vld1q_u8(weight_wire.as_ptr().add(base + 2));
        // 4-bit -> 8-bit with the -8 bias, low nibbles = weights 0..16,
        // high nibbles = weights 16..32 (the GGUF q4_0 packing).
        let weight_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask)), bias);
        let weight_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), bias);
        let input_lo = vld1q_s8(i_block.quants.as_ptr());
        let input_hi = vld1q_s8(i_block.quants.as_ptr().add(16));

        let mut acc = vdupq_n_s32(0);
        asm!(
            "sdot {acc:v}.4s, {weight_lo:v}.16b, {input_lo:v}.16b",
            "sdot {acc:v}.4s, {weight_hi:v}.16b, {input_hi:v}.16b",
            acc = inout(vreg) acc,
            weight_lo = in(vreg) weight_lo,
            input_lo = in(vreg) input_lo,
            weight_hi = in(vreg) weight_hi,
            input_hi = in(vreg) input_hi,
            options(nostack, preserves_flags)
        );

        let int_sum = horizontal_sum_i32x4(acc);
        total += int_sum as f32 * scale * i_block.scale;
    }

    total
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_two_dot_rows_neon_dotprod(
    first_weight: &[Q8_0Block],
    second_weight: &[Q8_0Block],
    input: &[Q8_0Block],
) -> (f32, f32) {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;

    let mut first_sum = 0.0_f32;
    let mut second_sum = 0.0_f32;

    for (idx, ((first_block, second_block), input_block)) in first_weight
        .iter()
        .zip(second_weight)
        .zip(input)
        .enumerate()
    {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if let Some(next_block) = first_weight.get(idx + 2) {
                asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) next_block.quants.as_ptr(),
                    options(nostack, preserves_flags, readonly)
                );
            }
            if let Some(next_block) = second_weight.get(idx + 2) {
                asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) next_block.quants.as_ptr(),
                    options(nostack, preserves_flags, readonly)
                );
            }
        }
        let input_lo = vld1q_s8(input_block.quants.as_ptr());
        let input_hi = vld1q_s8(input_block.quants.as_ptr().add(16));

        let w1_lo = vld1q_s8(first_block.quants.as_ptr());
        let w1_hi = vld1q_s8(first_block.quants.as_ptr().add(16));

        let w2_lo = vld1q_s8(second_block.quants.as_ptr());
        let w2_hi = vld1q_s8(second_block.quants.as_ptr().add(16));

        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);

        asm!(
            "sdot {acc1:v}.4s, {w1_lo:v}.16b, {input_lo:v}.16b",
            "sdot {acc1:v}.4s, {w1_hi:v}.16b, {input_hi:v}.16b",
            "sdot {acc2:v}.4s, {w2_lo:v}.16b, {input_lo:v}.16b",
            "sdot {acc2:v}.4s, {w2_hi:v}.16b, {input_hi:v}.16b",
            acc1 = inout(vreg) acc1,
            acc2 = inout(vreg) acc2,
            w1_lo = in(vreg) w1_lo,
            w1_hi = in(vreg) w1_hi,
            w2_lo = in(vreg) w2_lo,
            w2_hi = in(vreg) w2_hi,
            input_lo = in(vreg) input_lo,
            input_hi = in(vreg) input_hi,
            options(nostack, preserves_flags)
        );

        let int_sum1 = horizontal_sum_i32x4(acc1);
        let int_sum2 = horizontal_sum_i32x4(acc2);

        first_sum += int_sum1 as f32 * first_block.scale * input_block.scale;
        second_sum += int_sum2 as f32 * second_block.scale * input_block.scale;
    }

    (first_sum, second_sum)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_dot_rows_dotprod(weight: &[Q8_0Block], input: &[Q8_0Block]) -> f32 {
    weight
        .iter()
        .zip(input)
        .map(|(weight_block, input_block)| {
            // SAFETY: each Q8_0Block contains exactly 32 contiguous i8 values.
            let int_sum = unsafe {
                q8_0_i8_block_dotprod(weight_block.quants.as_ptr(), input_block.quants.as_ptr())
            };
            int_sum as f32 * weight_block.scale * input_block.scale
        })
        .sum()
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_two_dot_rows_dotprod(
    first_weight: &[Q8_0Block],
    second_weight: &[Q8_0Block],
    input: &[Q8_0Block],
) -> (f32, f32) {
    let mut first_sum = 0.0_f32;
    let mut second_sum = 0.0_f32;
    for ((first_block, second_block), input_block) in
        first_weight.iter().zip(second_weight).zip(input)
    {
        // SAFETY: each Q8_0Block contains exactly 32 contiguous i8 values.
        let first_int_sum = unsafe {
            q8_0_i8_block_dotprod(first_block.quants.as_ptr(), input_block.quants.as_ptr())
        };
        // SAFETY: each Q8_0Block contains exactly 32 contiguous i8 values.
        let second_int_sum = unsafe {
            q8_0_i8_block_dotprod(second_block.quants.as_ptr(), input_block.quants.as_ptr())
        };
        first_sum += first_int_sum as f32 * first_block.scale * input_block.scale;
        second_sum += second_int_sum as f32 * second_block.scale * input_block.scale;
    }
    (first_sum, second_sum)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_packed_4x4_block_dotprod(
    packed_quants: *const i8,
    input_quants: *const i8,
) -> [i32; 4] {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;

    // SAFETY: callers provide a packed 128-i8 block and a 32-i8 input block.
    let b0 = unsafe { vld1q_s8(packed_quants) };
    let b1 = unsafe { vld1q_s8(packed_quants.add(16)) };
    let b2 = unsafe { vld1q_s8(packed_quants.add(32)) };
    let b3 = unsafe { vld1q_s8(packed_quants.add(48)) };
    let b4 = unsafe { vld1q_s8(packed_quants.add(64)) };
    let b5 = unsafe { vld1q_s8(packed_quants.add(80)) };
    let b6 = unsafe { vld1q_s8(packed_quants.add(96)) };
    let b7 = unsafe { vld1q_s8(packed_quants.add(112)) };
    let a0 = unsafe { vld1q_s8(input_quants) };
    let a1 = unsafe { vld1q_s8(input_quants.add(16)) };

    let mut acc = vdupq_n_s32(0);
    // SAFETY: target_feature(dotprod) enables SDOT. This mirrors llama.cpp's q8_0 4x4
    // GEMV lane-dot shape: one output row per accumulator lane.
    unsafe {
        asm!(
            "sdot {acc:v}.4s, {b0:v}.16b, {a0:v}.4b[0]",
            "sdot {acc:v}.4s, {b1:v}.16b, {a0:v}.4b[1]",
            "sdot {acc:v}.4s, {b2:v}.16b, {a0:v}.4b[2]",
            "sdot {acc:v}.4s, {b3:v}.16b, {a0:v}.4b[3]",
            "sdot {acc:v}.4s, {b4:v}.16b, {a1:v}.4b[0]",
            "sdot {acc:v}.4s, {b5:v}.16b, {a1:v}.4b[1]",
            "sdot {acc:v}.4s, {b6:v}.16b, {a1:v}.4b[2]",
            "sdot {acc:v}.4s, {b7:v}.16b, {a1:v}.4b[3]",
            acc = inout(vreg) acc,
            b0 = in(vreg) b0,
            b1 = in(vreg) b1,
            b2 = in(vreg) b2,
            b3 = in(vreg) b3,
            b4 = in(vreg) b4,
            b5 = in(vreg) b5,
            b6 = in(vreg) b6,
            b7 = in(vreg) b7,
            a0 = in(vreg) a0,
            a1 = in(vreg) a1,
            options(nostack, preserves_flags)
        );
    }
    // SAFETY: int32x4_t is a four-lane i32 vector; lane order is output-row order.
    unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc) }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_packed_4x8_block_dotprod(
    packed_quants: *const i8,
    input_quants: *const i8,
) -> [i32; 4] {
    use std::arch::aarch64::{vcombine_s8, vdupq_n_s32, vld1_s8, vld1q_s8};
    use std::arch::asm;

    // SAFETY: callers provide a packed 128-i8 block and a 32-i8 input block.
    let b0 = unsafe { vld1q_s8(packed_quants) };
    let b1 = unsafe { vld1q_s8(packed_quants.add(16)) };
    let b2 = unsafe { vld1q_s8(packed_quants.add(32)) };
    let b3 = unsafe { vld1q_s8(packed_quants.add(48)) };
    let b4 = unsafe { vld1q_s8(packed_quants.add(64)) };
    let b5 = unsafe { vld1q_s8(packed_quants.add(80)) };
    let b6 = unsafe { vld1q_s8(packed_quants.add(96)) };
    let b7 = unsafe { vld1q_s8(packed_quants.add(112)) };
    let a0_half = unsafe { vld1_s8(input_quants) };
    let a1_half = unsafe { vld1_s8(input_quants.add(8)) };
    let a2_half = unsafe { vld1_s8(input_quants.add(16)) };
    let a3_half = unsafe { vld1_s8(input_quants.add(24)) };
    let a0 = vcombine_s8(a0_half, a0_half);
    let a1 = vcombine_s8(a1_half, a1_half);
    let a2 = vcombine_s8(a2_half, a2_half);
    let a3 = vcombine_s8(a3_half, a3_half);

    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    // SAFETY: target_feature(dotprod) enables SDOT. This mirrors llama.cpp's q8_0 4x8
    // GEMV dot shape; pairwise lane sums below mirror vpaddq_s32(ret0, ret1).
    unsafe {
        asm!(
            "sdot {acc0:v}.4s, {b0:v}.16b, {a0:v}.16b",
            "sdot {acc1:v}.4s, {b1:v}.16b, {a0:v}.16b",
            "sdot {acc0:v}.4s, {b2:v}.16b, {a1:v}.16b",
            "sdot {acc1:v}.4s, {b3:v}.16b, {a1:v}.16b",
            "sdot {acc0:v}.4s, {b4:v}.16b, {a2:v}.16b",
            "sdot {acc1:v}.4s, {b5:v}.16b, {a2:v}.16b",
            "sdot {acc0:v}.4s, {b6:v}.16b, {a3:v}.16b",
            "sdot {acc1:v}.4s, {b7:v}.16b, {a3:v}.16b",
            acc0 = inout(vreg) acc0,
            acc1 = inout(vreg) acc1,
            b0 = in(vreg) b0,
            b1 = in(vreg) b1,
            b2 = in(vreg) b2,
            b3 = in(vreg) b3,
            b4 = in(vreg) b4,
            b5 = in(vreg) b5,
            b6 = in(vreg) b6,
            b7 = in(vreg) b7,
            a0 = in(vreg) a0,
            a1 = in(vreg) a1,
            a2 = in(vreg) a2,
            a3 = in(vreg) a3,
            options(nostack, preserves_flags)
        );
    }
    // SAFETY: int32x4_t is a four-lane i32 vector.
    let lanes0 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc0) };
    let lanes1 = unsafe { std::mem::transmute::<std::arch::aarch64::int32x4_t, [i32; 4]>(acc1) };
    [
        lanes0[0] + lanes0[1],
        lanes0[2] + lanes0[3],
        lanes1[0] + lanes1[1],
        lanes1[2] + lanes1[3],
    ]
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn dot_q8_0_encoded_row_with_scales_dotprod(
    input: &[Q8_0Block],
    row_bytes: &[u8],
    scales: &[f32],
) -> f32 {
    let mut sum = 0.0_f32;
    for ((input_block, block), scale) in input
        .iter()
        .zip(row_bytes.chunks_exact(Q8BlockReader::BLOCK_SIZE_BYTES))
        .zip(scales)
    {
        // SAFETY: each encoded Q8_0 block stores 32 contiguous signed quant bytes after
        // the two-byte f16 scale header.
        let int_sum = unsafe {
            q8_0_i8_block_dotprod(
                block[2..].as_ptr().cast::<i8>(),
                input_block.quants.as_ptr(),
            )
        };
        sum += int_sum as f32 * *scale * input_block.scale;
    }
    sum
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
pub(super) fn aarch64_dotprod_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !q8_0_env_flag_disabled("CAMELID_AARCH64_DOTPROD")
            && std::arch::is_aarch64_feature_detected!("dotprod")
    })
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
pub(super) unsafe fn q8_0_i8_block_neon(weight: *const i8, input: *const i8) -> i32 {
    if aarch64_dotprod_enabled() {
        // SAFETY: feature detection above guarantees the dot-product instructions are
        // available, and callers pass pointers to at least 32 contiguous i8 values.
        return unsafe { q8_0_i8_block_dotprod(weight, input) };
    }

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    unsafe { q8_0_i8_block_neon_mul(weight, input) }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_0_i8_block_dotprod(weight: *const i8, input: *const i8) -> i32 {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    use std::arch::asm;

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    let weight_lo = unsafe { vld1q_s8(weight) };
    let input_lo = unsafe { vld1q_s8(input) };
    let weight_hi = unsafe { vld1q_s8(weight.add(16)) };
    let input_hi = unsafe { vld1q_s8(input.add(16)) };

    let mut acc = vdupq_n_s32(0);
    // SAFETY: target_feature(dotprod) enables SDOT for this function. The operands are full
    // 128-bit vector registers loaded above, and the instruction only updates `acc`.
    unsafe {
        asm!(
            "sdot {acc:v}.4s, {weight_lo:v}.16b, {input_lo:v}.16b",
            "sdot {acc:v}.4s, {weight_hi:v}.16b, {input_hi:v}.16b",
            acc = inout(vreg) acc,
            weight_lo = in(vreg) weight_lo,
            input_lo = in(vreg) input_lo,
            weight_hi = in(vreg) weight_hi,
            input_hi = in(vreg) input_hi,
            options(nostack, preserves_flags)
        );
    }
    horizontal_sum_i32x4(acc)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[inline(always)]
pub(super) unsafe fn q8_0_i8_block_neon_mul(weight: *const i8, input: *const i8) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8, vpaddlq_s16,
    };

    // SAFETY: callers pass pointers to at least 32 contiguous i8 values.
    let weight_lo = unsafe { vld1q_s8(weight) };
    let input_lo = unsafe { vld1q_s8(input) };
    let weight_hi = unsafe { vld1q_s8(weight.add(16)) };
    let input_hi = unsafe { vld1q_s8(input.add(16)) };

    let mut acc = vdupq_n_s32(0);
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_low_s8(weight_lo), vget_low_s8(input_lo))),
    );
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_high_s8(weight_lo), vget_high_s8(input_lo))),
    );
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_low_s8(weight_hi), vget_low_s8(input_hi))),
    );
    acc = vaddq_s32(
        acc,
        vpaddlq_s16(vmull_s8(vget_high_s8(weight_hi), vget_high_s8(input_hi))),
    );
    horizontal_sum_i32x4(acc)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code, unused_variables))]
#[inline(always)]
pub(super) fn horizontal_sum_i32x4(acc: std::arch::aarch64::int32x4_t) -> i32 {
    unsafe { std::arch::aarch64::vaddvq_s32(acc) }
}
