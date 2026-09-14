#[cfg(test)]
mod fp16_attention_tests {
    use super::*;

    /// Exercises real nonzero query/output offsets, causal counts across split
    /// boundaries, and untouched guards. Run separately under prefetch and
    /// exact rowshare environments; the existing per-row route is the reference.
    #[test]
    fn diagnostic_attention_chunks_match_serial_offsets_and_masks() {
        assert!(
            detect_metal_device().available,
            "run this diagnostic on mini2"
        );
        assert!(
            attn2_enabled() && splitk_attention_enabled(),
            "existing attention gates required"
        );
        let k = metal_linear_kernel().expect("Metal");
        let alloc = |bytes: usize| {
            k.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        };
        let stride = 2048usize;
        let hidden = 3072usize;
        let guard = 16usize;
        let sentinel = 0x7fa12345u32;
        let kv_values: Vec<u16> = (0..8 * stride * 128)
            .map(|i| f32_to_f16_bits((((i * 43 + i / 128) % 251) as f32 - 125.0) * 0.003))
            .collect();
        let keys = alloc(kv_values.len() * 2);
        let values = alloc(kv_values.len() * 2);
        unsafe {
            std::ptr::copy_nonoverlapping(
                kv_values.as_ptr(),
                keys.contents().cast::<u16>(),
                kv_values.len(),
            );
            std::ptr::copy_nonoverlapping(
                kv_values.as_ptr(),
                values.contents().cast::<u16>(),
                kv_values.len(),
            );
        }
        for base in [128usize, 504, 1020, 2032] {
            for n in [2usize, 8, 15, 16] {
                let offset_rows = 7usize;
                let total_rows = offset_rows + n;
                let queries: Vec<f32> = (0..total_rows * hidden)
                    .map(|i| (((i * 17 + i / hidden * 73) % 257) as f32 - 128.0) * 0.004)
                    .collect();
                let query = alloc(queries.len() * 4);
                write_buffer_f32(&query, &queries);
                let output = alloc((total_rows * hidden + guard) * 4);
                let reference = alloc((total_rows * hidden + guard) * 4);
                unsafe {
                    for b in [&output, &reference] {
                        std::slice::from_raw_parts_mut(
                            b.contents().cast::<u32>(),
                            total_rows * hidden + guard,
                        )
                        .fill(sentinel);
                    }
                }
                let positions: Vec<usize> = (0..n).map(|i| base + i + 1).collect();
                let scalars: Vec<Buffer> = positions
                    .iter()
                    .map(|&pc| {
                        let b = alloc(32);
                        let fields = [
                            24u32,
                            128,
                            pc as u32,
                            3,
                            (1.0f32 / 128.0f32.sqrt()).to_bits(),
                            128,
                            (stride * 128) as u32,
                            0,
                        ];
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                fields.as_ptr(),
                                b.contents().cast::<u32>(),
                                8,
                            );
                        }
                        b
                    })
                    .collect();
                let scores = alloc(24 * stride * 4);
                let cb = k.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                let mut keep = Vec::new();
                let offset = (offset_rows * hidden * 4) as u64;
                // A rejected offset must encode nothing or allocate scratch.
                let invalid = encode_attention_splitk_kv16_batch_offset(
                    e,
                    k,
                    &mut keep,
                    &query,
                    &keys,
                    &values,
                    &output,
                    &scalars[0],
                    24,
                    8,
                    128,
                    &positions,
                    None,
                    query.length(),
                    offset,
                );
                assert!(!invalid.encoded && keep.is_empty());
                let route = encode_attention_splitk_kv16_batch_offset(
                    e,
                    k,
                    &mut keep,
                    &query,
                    &keys,
                    &values,
                    &output,
                    &scalars[0],
                    24,
                    8,
                    128,
                    &positions,
                    None,
                    offset,
                    offset,
                );
                assert!(route.encoded, "batch must be exercised base={base} n={n}");
                for (row, &pc) in positions.iter().enumerate() {
                    let row_offset = ((offset_rows + row) * hidden * 4) as u64;
                    encode_attention(
                        e,
                        k,
                        &mut keep,
                        &query,
                        &keys,
                        &values,
                        None,
                        true,
                        false,
                        &scores,
                        &reference,
                        &scalars[row],
                        24,
                        8,
                        128,
                        pc,
                        row_offset,
                        row_offset,
                    );
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                let read = |b: &Buffer| unsafe {
                    std::slice::from_raw_parts(
                        b.contents().cast::<u32>(),
                        total_rows * hidden + guard,
                    )
                    .to_vec()
                };
                let actual = read(&output);
                let expected = read(&reference);
                assert_eq!(
                    actual, expected,
                    "chunk offset or arithmetic differs base={base} n={n}"
                );
                assert!(actual[..offset_rows * hidden]
                    .iter()
                    .all(|&v| v == sentinel));
                assert!(actual[total_rows * hidden..].iter().all(|&v| v == sentinel));
                assert!(actual[offset_rows * hidden..total_rows * hidden]
                    .iter()
                    .all(|&v| f32::from_bits(v).is_finite()));
                pool_recycle(k, keep);
            }
        }
    }
}
