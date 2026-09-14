#[cfg(test)]
mod fp16_attention_mm_tests {
    use super::*;

    /// Ragged absolute bases must be serial-identical in this attention target,
    /// including both sides of a 64-position boundary and padded output guards.
    /// Future cache rows are deliberately populated; a missing causal bound
    /// cannot hide behind initially zero KV.
    #[test]
    fn absolute_query_tiles_are_serial_identical_and_guarded() {
        if !detect_metal_device().available {
            return;
        }
        let k = metal_linear_kernel().expect("Metal available");
        let max_positions = 2048;
        let cache_elements = 8 * max_positions * 128;
        let half_buffer = |salt: usize| {
            let values: Vec<u16> = (0..cache_elements)
                .map(|i| {
                    f32_to_f16_bits((((i * 31 + i / 128 * 17 + salt) % 251) as f32 - 125.0) * 0.003)
                })
                .collect();
            let buffer = k.device.new_buffer(
                (values.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().cast::<u16>(),
                    values.len(),
                );
            }
            buffer
        };
        let keys = half_buffer(19);
        let values = half_buffer(71);
        let sentinel = 0xff123456u32;
        let run = |base: usize, n: usize, query_values: &[f32]| {
            let scratch =
                Fp16ProbeAttentionMm::new(k, base, n, max_positions, 1.0 / 128.0f32.sqrt())
                    .expect("admitted attention shape");
            let query = k
                .device
                .new_buffer((n * 3072 * 4) as u64, MTLResourceOptions::StorageModeShared);
            write_buffer_f32(&query, query_values);
            let output = k.device.new_buffer(
                ((n * 3072 + 16) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::slice::from_raw_parts_mut(output.contents().cast::<u32>(), n * 3072 + 16)
                    .fill(sentinel);
            }
            let cb = k.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            assert!(scratch.encode(k, e, &query, &keys, &values, &output));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
            let actual = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<u32>(), n * 3072 + 16).to_vec()
            };
            assert!(actual[n * 3072..].iter().all(|&word| word == sentinel));
            assert!(actual[..n * 3072]
                .iter()
                .all(|&word| word != sentinel && f32::from_bits(word).is_finite()));
            pool_recycle(k, scratch.into_keep());
            actual[..n * 3072].to_vec()
        };
        for base in [7usize, 743, 1023] {
            for n in [1usize, 17, 64] {
                let query_values: Vec<f32> = (0..n * 3072)
                    .map(|i| (((i * 13 + (base + i / 3072) * 73) % 257) as f32 - 128.0) * 0.009)
                    .collect();
                let wide = run(base, n, &query_values);
                for row in 0..n {
                    let serial = run(base + row, 1, &query_values[row * 3072..(row + 1) * 3072]);
                    assert_eq!(
                        &wide[row * 3072..(row + 1) * 3072],
                        serial.as_slice(),
                        "attention base={base} n={n} row={row}"
                    );
                }
                // Sparse independent attention reference catches a shared
                // head/stride/scale defect that serial equality alone misses.
                // Reproduce half Q, scores and probabilities, allowing the
                // f32-MMA/Metal-exp difference from this CPU f64 calculation.
                let key_words = unsafe {
                    std::slice::from_raw_parts(keys.contents().cast::<u16>(), cache_elements)
                };
                let value_words = unsafe {
                    std::slice::from_raw_parts(values.contents().cast::<u16>(), cache_elements)
                };
                let scale = (1.0 / 128.0f32.sqrt()) as f64;
                for row in [0, n - 1] {
                    for head in [0usize, 23] {
                        let positions = base + row + 1;
                        let kv_head = head / 3;
                        let mut scores = Vec::with_capacity(positions);
                        for position in 0..positions {
                            let mut dot = 0.0f64;
                            for dim in 0..128 {
                                let q = f16_bits_to_f32(f32_to_f16_bits(
                                    query_values[row * 3072 + head * 128 + dim],
                                )) as f64;
                                let key = f16_bits_to_f32(
                                    key_words[(kv_head * max_positions + position) * 128 + dim],
                                ) as f64;
                                dot += q * key;
                            }
                            scores
                                .push(f16_bits_to_f32(f32_to_f16_bits(dot as f32)) as f64 * scale);
                        }
                        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let denominator: f64 =
                            scores.iter().map(|score| (score - maximum).exp()).sum();
                        let probabilities: Vec<f64> = scores
                            .iter()
                            .map(|score| {
                                f16_bits_to_f32(f32_to_f16_bits(
                                    ((score - maximum).exp() / denominator) as f32,
                                )) as f64
                            })
                            .collect();
                        for dim in [0usize, 127] {
                            let expected: f64 = probabilities
                                .iter()
                                .enumerate()
                                .map(|(position, probability)| {
                                    probability
                                        * f16_bits_to_f32(
                                            value_words
                                                [(kv_head * max_positions + position) * 128 + dim],
                                        ) as f64
                                })
                                .sum();
                            let actual = f32::from_bits(wide[row * 3072 + head * 128 + dim]) as f64;
                            assert!((actual - expected).abs() < 4e-4,
                                "attention independent reference base={base} n={n} row={row} head={head} dim={dim}: {actual} vs {expected}");
                        }
                    }
                }
            }
        }
        assert!(Fp16ProbeAttentionMm::new(k, 2000, 64, max_positions, 0.1).is_none());
        assert!(Fp16ProbeAttentionMm::new(k, 0, 0, max_positions, 0.1).is_none());
        assert!(Fp16ProbeAttentionMm::new(k, 0, 1, max_positions, f32::NAN).is_none());
    }
}
