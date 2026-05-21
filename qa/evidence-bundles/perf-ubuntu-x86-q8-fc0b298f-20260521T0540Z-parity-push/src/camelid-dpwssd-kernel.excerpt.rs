            .for_each(|(group_idx, output_chunk)| compute_group(group_idx, output_chunk));
        return;
    }

    for (group_idx, output_chunk) in output.chunks_mut(4).enumerate() {
        compute_group(group_idx, output_chunk);
    }
}

fn q8_0_packed_rows4_dot_i8_matmul(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
    use_hoisted_avx2: bool,
) -> [f32; 4] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled() {
            // SAFETY: runtime feature detection in
            // `x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled` confirms support.
            return unsafe { q8_0_packed_rows4_dot_i8_avx512vnni_dpwssd(packed_blocks, input) };
        }
        if use_hoisted_avx2 {
            // SAFETY: `use_hoisted_avx2` is only true after runtime AVX2 detection.
            return unsafe { q8_0_packed_rows4_dot_i8_avx2(packed_blocks, input) };
        }
    }
    let _ = use_hoisted_avx2;
    q8_0_packed_rows4_dot(packed_blocks, input, Q8_0PackedRows4Interleave::I8)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_packed_rows4_dot_i8_avx512vnni_dpwssd(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
) -> [f32; 4] {
    debug_assert_eq!(packed_blocks.len(), input.len());
    let mut sums = [0.0_f32; 4];
    for (packed_block, input_block) in packed_blocks.iter().zip(input) {
        let int_sums = unsafe {
            q8_0_packed_4x8_block_avx512vnni_dpwssd(
                packed_block.quants.as_ptr(),
                input_block.quants.as_ptr(),
            )
        };
        let input_scale = input_block.scale;
        for lane in 0..4 {
            sums[lane] += int_sums[lane] as f32 * packed_block.scales[lane] * input_scale;
        }
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_packed_rows4_dot_i8_avx2(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
) -> [f32; 4] {
    debug_assert_eq!(packed_blocks.len(), input.len());
    let mut sums = [0.0_f32; 4];
    for (packed_block, input_block) in packed_blocks.iter().zip(input) {
        let int_sums = unsafe {
            q8_0_packed_4x8_block_avx2(packed_block.quants.as_ptr(), input_block.quants.as_ptr())
        };
        let input_scale = input_block.scale;
        for lane in 0..4 {
            sums[lane] += int_sums[lane] as f32 * packed_block.scales[lane] * input_scale;
        }
    }
    sums
}

fn q8_0_packed_rows4_dot(
    packed_blocks: &[Q8_0PackedRows4Block],
    input: &[Q8_0Block],
    interleave: Q8_0PackedRows4Interleave,
) -> [f32; 4] {
    debug_assert_eq!(packed_blocks.len(), input.len());
    let mut sums = [0.0_f32; 4];
    for (packed_block, input_block) in packed_blocks.iter().zip(input) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let int_sums = if aarch64_dotprod_enabled() {
            // SAFETY: runtime feature detection confirms dot-product support; packed quants
            // contain 128 i8 values and input quants contain 32 contiguous i8 values.
            unsafe {
                match interleave {
                    Q8_0PackedRows4Interleave::I4 => q8_0_packed_4x4_block_dotprod(
                        packed_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                    Q8_0PackedRows4Interleave::I8 => q8_0_packed_4x8_block_dotprod(
                        packed_block.quants.as_ptr(),
                        input_block.quants.as_ptr(),
                    ),
                }
            }
        } else {
            q8_0_packed_rows4_block_dot_scalar(
                &packed_block.quants,
                &input_block.quants,
                interleave,
            )
        };
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let int_sums = {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if interleave == Q8_0PackedRows4Interleave::I8
                    && x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled()
                {
                    // SAFETY: runtime feature detection confirms AVX512F/BW/VNNI and AVX2
                    // support; packed quants contain one complete rows4/I8 block and input
                    // quants contain one Q8_0 block.
                    unsafe {
                        q8_0_packed_4x8_block_avx512vnni_dpwssd(
                            packed_block.quants.as_ptr(),
                            input_block.quants.as_ptr(),
                        )
                    }
                } else if interleave == Q8_0PackedRows4Interleave::I8
                    && (x86_q8_packed_rows4_avx2_dot_enabled() || x86_q8_kernel_avx2_enabled())
                    && std::arch::is_x86_feature_detected!("avx2")
                {
                    // SAFETY: runtime feature detection confirms AVX2 support; packed quants
                    // contain one complete rows4/I8 block and input quants contain one Q8_0 block.
                    unsafe {
                        q8_0_packed_4x8_block_avx2(
                            packed_block.quants.as_ptr(),
                            input_block.quants.as_ptr(),
                        )
                    }
                } else {
                    q8_0_packed_rows4_block_dot_scalar(
                        &packed_block.quants,
                        &input_block.quants,
                        interleave,
                    )
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                q8_0_packed_rows4_block_dot_scalar(
                    &packed_block.quants,
                    &input_block.quants,
                    interleave,
                )
            }
        };
        for lane in 0..4 {
            sums[lane] += int_sums[lane] as f32 * packed_block.scales[lane] * input_block.scale;
        }
    }
    sums
}

fn q8_0_packed_rows4_block_dot_scalar(
    packed: &[i8; 128],
    input: &[i8; 32],
    interleave: Q8_0PackedRows4Interleave,
) -> [i32; 4] {
    let block_len = interleave.block_len();
    let chunks = 32 / block_len;
    let mut sums = [0_i32; 4];
    for chunk in 0..chunks {
        for lane in 0..4 {
            for idx in 0..block_len {
                sums[lane] += i32::from(packed[chunk * 4 * block_len + lane * block_len + idx])
                    * i32::from(input[chunk * block_len + idx]);
            }
        }
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
unsafe fn q8_0_packed_4x8_block_avx512vnni_dpwssd(packed: *const i8, input: *const i8) -> [i32; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_broadcastsi128_si256, _mm256_loadu_si256, _mm512_cvtepi8_epi16, _mm512_dpwssd_epi32,
        _mm512_setzero_si512, _mm512_storeu_si512, _mm_loadl_epi64, _mm_unpacklo_epi64,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_broadcastsi128_si256, _mm256_loadu_si256, _mm512_cvtepi8_epi16, _mm512_dpwssd_epi32,
        _mm512_setzero_si512, _mm512_storeu_si512, _mm_loadl_epi64, _mm_unpacklo_epi64,
    };

    let mut acc = _mm512_setzero_si512();
    for chunk in 0..4usize {
        let packed32 = unsafe { _mm256_loadu_si256(packed.add(chunk * 32).cast()) };
        let packed_i16 = _mm512_cvtepi8_epi16(packed32);

        let input8 = unsafe { _mm_loadl_epi64(input.add(chunk * 8).cast()) };
        let input16 = _mm_unpacklo_epi64(input8, input8);
