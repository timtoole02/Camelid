#[cfg(test)]
mod fp16_probe_tests {
    use super::*;

    /// The diagnostic's actual encoder must reproduce repeated native-MMA
    /// single rows, including padding on both the token and output-row axes.
    /// This deliberately compares within FP16 arithmetic, never against V4.
    #[test]
    fn native_fp16_probe_projection_is_serial_identical_and_guarded() {
        if !detect_metal_device().available {
            return;
        }
        let k = metal_linear_kernel().expect("Metal available");
        let sentinel = 0xff123456u32;
        let read = |b: &Buffer, len: usize| unsafe {
            std::slice::from_raw_parts(b.contents().cast::<u32>(), len).to_vec()
        };
        for width in [3072usize, 8192] {
            let rows = 19usize;
            let weight_values: Vec<u16> = (0..rows * width)
                .map(|i| f32_to_f16_bits((((i * 31 + i / 17) % 127) as f32 - 63.0) * 0.001))
                .collect();
            let weight = k.device.new_buffer(
                (weight_values.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    weight_values.as_ptr(),
                    weight.contents().cast::<u16>(),
                    weight_values.len(),
                );
            }
            for n in [1usize, 9, 16, 32, 63, 64] {
                let physical: Vec<u16> = (0..64 * width)
                    .map(|i| {
                        if i / width >= n {
                            0
                        } else {
                            f32_to_f16_bits(
                                (((i * 17 + i / width * 73) % 257) as f32 - 128.0) * 0.01,
                            )
                        }
                    })
                    .collect();
                let panel = k.device.new_buffer(
                    (physical.len() * 2) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        physical.as_ptr(),
                        panel.contents().cast::<u16>(),
                        physical.len(),
                    );
                }
                let output = k.device.new_buffer(
                    ((n * rows + 16) * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                unsafe {
                    std::slice::from_raw_parts_mut(output.contents().cast::<u32>(), n * rows + 16)
                        .fill(sentinel);
                }
                let cb = k.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                let mut keep = Vec::new();
                encode_fp16_probe_mm(e, k, &mut keep, &weight, &panel, &output, width, rows, n);
                let mut references = Vec::new();
                for token in 0..n {
                    let one = k.device.new_buffer(
                        (64 * width * 2) as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    unsafe {
                        std::ptr::write_bytes(one.contents().cast::<u8>(), 0, 64 * width * 2);
                        std::ptr::copy_nonoverlapping(
                            physical[token * width..].as_ptr(),
                            one.contents().cast::<u16>(),
                            width,
                        );
                    }
                    let out = k.device.new_buffer(
                        ((rows + 16) * 4) as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    unsafe {
                        std::slice::from_raw_parts_mut(out.contents().cast::<u32>(), rows + 16)
                            .fill(sentinel);
                    }
                    encode_fp16_probe_mm(e, k, &mut keep, &weight, &one, &out, width, rows, 1);
                    references.push(out);
                    keep.push(one);
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                let actual = read(&output, n * rows + 16);
                assert!(actual[n * rows..].iter().all(|&x| x == sentinel));
                assert!(actual[..n * rows]
                    .iter()
                    .all(|&x| x != sentinel && f32::from_bits(x).is_finite()));
                for (token, reference) in references.iter().enumerate() {
                    let expected = read(reference, rows + 16);
                    assert!(expected[rows..].iter().all(|&x| x == sentinel));
                    assert_eq!(
                        &actual[token * rows..(token + 1) * rows],
                        &expected[..rows],
                        "K={width} n={n} token={token}"
                    );
                }
                // Sparse independent f64 dot reference catches a shared binding
                // error that could otherwise pass the serial-equivalence check.
                for token in [0, n - 1] {
                    for row in [0, 7, 18] {
                        let expected: f64 = (0..width)
                            .map(|i| {
                                f16_bits_to_f32(weight_values[row * width + i]) as f64
                                    * f16_bits_to_f32(physical[token * width + i]) as f64
                            })
                            .sum();
                        let got = f32::from_bits(actual[token * rows + row]) as f64;
                        assert!(
                            (got - expected).abs() < 1e-3,
                            "K={width} n={n} ({token},{row}): {got} vs {expected}"
                        );
                    }
                }
                pool_recycle(k, keep);
            }
        }
    }
}
