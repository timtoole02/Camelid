#[cfg(test)]
mod fp16_sg_tests {
    use super::*;

    fn words(buffer: &Buffer, count: usize) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u32>(), count).to_vec() }
    }
    fn halves(buffer: &Buffer, count: usize) -> Vec<u16> {
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), count).to_vec() }
    }
    fn fill_guard(buffer: &Buffer) {
        unsafe {
            std::slice::from_raw_parts_mut(
                buffer.contents().cast::<u32>(),
                buffer.length() as usize / 4,
            )
            .fill(0x7fa12345);
        }
    }

    /// Both formats must keep every canonical half bit, then every activation
    /// half bit, before testing SG arithmetic against repeated SG serial rows.
    #[test]
    fn packed_fp16_operands_and_serial_rows_are_exact() {
        assert!(
            detect_metal_device().available,
            "run this diagnostic on mini2"
        );
        let k = metal_linear_kernel().expect("Metal");
        let sg = fp16_sg_kernels(k).expect("SG library");
        let buffer = |bytes: usize| {
            k.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        };
        let rows = 24usize;
        let guard = 16usize;
        for width in [3072usize, 8192] {
            for format in [ResidentWeightFormat::Q4K, ResidentWeightFormat::Q6K] {
                let block_bytes = format.wire_bytes_per_block();
                let mut wire: Vec<u8> = (0..rows * (width / 256) * block_bytes)
                    .map(|i| ((i * 61 + i / 19 + width) & 255) as u8)
                    .collect();
                for (i, block) in wire.chunks_exact_mut(block_bytes).enumerate() {
                    let d = 0.003 + (i % 29) as f32 * 0.00007;
                    if format == ResidentWeightFormat::Q4K {
                        block[..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                        block[2..4].copy_from_slice(&f32_to_f16_bits(d * 0.375).to_le_bytes());
                    } else {
                        block[208..210].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    }
                }
                let source = buffer(wire.len());
                write_buffer_u8(&source, &wire);
                let plain = buffer((rows * width + guard) * 2);
                let packed = buffer((rows * width + guard) * 2);
                fill_guard(&plain);
                fill_guard(&packed);
                let scalar = buffer(8);
                unsafe {
                    *scalar.contents().cast::<u32>() = (width / 256) as u32;
                    *scalar.contents().cast::<u32>().add(1) = rows as u32;
                }
                let cb = k.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                let (native, candidate) = if format == ResidentWeightFormat::Q4K {
                    (&k.q4k_dequant_to_half_pipeline, &sg.dequant_q4)
                } else {
                    (&k.q6k_dequant_to_half_pipeline, &sg.dequant_q6)
                };
                for (pipeline, output) in [(native, &plain), (candidate, &packed)] {
                    e.set_compute_pipeline_state(pipeline);
                    e.set_buffer(0, Some(&source), 0);
                    e.set_buffer(1, Some(output), 0);
                    e.set_buffer(2, Some(&scalar), 0);
                    e.set_buffer(3, Some(&scalar), 4);
                    dispatch_1d(e, pipeline, rows * width / 256);
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                let a = halves(&plain, rows * width + guard);
                let p = halves(&packed, rows * width + guard);
                assert_eq!(&a[rows * width..], &p[rows * width..]);
                assert!(
                    words(&packed, (rows * width + guard) / 2)[rows * width / 2..]
                        .iter()
                        .all(|&v| v == 0x7fa12345)
                );
                for row in 0..rows {
                    for col in 0..width {
                        let at = ((row / 8 * (width / 8) + col / 8) * 64) + col % 8 * 8 + row % 8;
                        assert_eq!(
                            a[row * width + col],
                            p[at],
                            "canonical half format={format:?} K={width} row={row} col={col}"
                        );
                    }
                }
                for n in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64] {
                    let values: Vec<f32> = (0..n * width)
                        .map(|i| (((i * 17 + i / width * 73) % 257) as f32 - 128.0) * 0.01)
                        .collect();
                    let input = buffer(values.len() * 4);
                    write_buffer_f32(&input, &values);
                    let physical = fp16_sg_physical_rows(n);
                    let panel = buffer((physical * width + guard) * 2);
                    let output = buffer((n * rows + guard) * 4);
                    // Compare the same canonical half operands through the
                    // pre-existing native encoder, independently of SG packing.
                    let native_panel = buffer(64 * width * 2);
                    let native_output = buffer((n * rows + guard) * 4);
                    unsafe {
                        let dst = std::slice::from_raw_parts_mut(
                            native_panel.contents().cast::<u16>(),
                            64 * width,
                        );
                        dst.fill(0);
                        for (out, &value) in dst.iter_mut().zip(&values) {
                            *out = f32_to_f16_bits(value);
                        }
                    }
                    fill_guard(&panel);
                    fill_guard(&output);
                    fill_guard(&native_output);
                    let cb = k.queue.new_command_buffer();
                    let e = cb.new_compute_command_encoder();
                    let mut keep = Vec::new();
                    encode_fp16_sg_projection(
                        e, k, sg, &mut keep, &input, &packed, &panel, &output, width, rows, n,
                    );
                    encode_fp16_probe_mm(
                        e,
                        k,
                        &mut keep,
                        &plain,
                        &native_panel,
                        &native_output,
                        width,
                        rows,
                        n,
                    );
                    let mut serial = Vec::new();
                    for token in 0..n {
                        let one = buffer(width * 4);
                        write_buffer_f32(&one, &values[token * width..(token + 1) * width]);
                        let one_panel = buffer(8 * width * 2);
                        let one_output = buffer((rows + guard) * 4);
                        fill_guard(&one_output);
                        encode_fp16_sg_projection(
                            e,
                            k,
                            sg,
                            &mut keep,
                            &one,
                            &packed,
                            &one_panel,
                            &one_output,
                            width,
                            rows,
                            1,
                        );
                        keep.extend([one, one_panel]);
                        serial.push(one_output);
                    }
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                    let actual = words(&output, n * rows + guard);
                    assert!(actual[n * rows..].iter().all(|&x| x == 0x7fa12345));
                    assert!(actual[..n * rows]
                        .iter()
                        .all(|&x| f32::from_bits(x).is_finite()));
                    let x = halves(&panel, physical * width + guard);
                    assert!(
                        words(&panel, (physical * width + guard) / 2)[physical * width / 2..]
                            .iter()
                            .all(|&v| v == 0x7fa12345)
                    );
                    for token in 0..physical {
                        for col in 0..width {
                            let at = ((token / 8 * (width / 8) + col / 8) * 64)
                                + token % 8 * 8
                                + col % 8;
                            let expected = if token < n {
                                f32_to_f16_bits(values[token * width + col])
                            } else {
                                0
                            };
                            assert_eq!(
                                x[at], expected,
                                "activation K={width} n={n} token={token} col={col}"
                            );
                        }
                    }
                    for (token, reference) in serial.iter().enumerate() {
                        let want = words(reference, rows + guard);
                        assert!(want[rows..].iter().all(|&v| v == 0x7fa12345));
                        assert_eq!(
                            &actual[token * rows..(token + 1) * rows],
                            &want[..rows],
                            "SG serial format={format:?} K={width} n={n} token={token}"
                        );
                    }
                    let native_words = words(&native_output, n * rows + guard);
                    assert!(native_words[n * rows..].iter().all(|&x| x == 0x7fa12345));
                    assert_eq!(&actual[..n * rows], &native_words[..n * rows],
                        "same canonical half operands differ: native versus SG format={format:?} K={width} n={n}");
                    let mut max_absolute_error = 0.0f64;
                    let mut max_bound_fraction = 0.0f64;
                    let mut max_condition = 0.0f64;
                    let mut sequential_f32_mismatches = 0;
                    let mut worst = None;
                    for token in [0, n - 1] {
                        for row in [0, 7, rows - 1] {
                            let mut exact = 0.0f64;
                            let mut sum_abs = 0.0f64;
                            let mut sequential_f32 = 0.0f32;
                            for i in 0..width {
                                let w = f16_bits_to_f32(a[row * width + i]);
                                let x = f16_bits_to_f32(f32_to_f16_bits(values[token * width + i]));
                                let product = w as f64 * x as f64;
                                exact += product;
                                sum_abs += product.abs();
                                sequential_f32 = w.mul_add(x, sequential_f32);
                            }
                            let got = f32::from_bits(actual[token * rows + row]) as f64;
                            // A small dot result can hide cancellation of large
                            // signed terms: scaling tolerance by |dot| is unsound.
                            // Half products fit exactly in f32; these finite fixture
                            // terms neither overflow nor underflow f32. Use the
                            // canonical dot-product bound gamma_K * sum|w*x|,
                            // gamma_K = K*u/(1-K*u), u=2^-24. See equation2.1:
                            // https://www.cs.utsa.edu/faculty/atc/pub/J42.pdf
                            // This is conservative. Exact native and SG-serial bit
                            // comparisons above remain the primary kernel gates.
                            let ku = width as f64 * 2.0f64.powi(-24);
                            let bound = ku / (1.0 - ku) * sum_abs;
                            let error = (got - exact).abs();
                            let condition = if exact == 0.0 {
                                f64::INFINITY
                            } else {
                                sum_abs / exact.abs()
                            };
                            if error >= max_absolute_error {
                                worst =
                                    Some((token, row, got, exact, sequential_f32, sum_abs, bound));
                            }
                            max_absolute_error = max_absolute_error.max(error);
                            max_bound_fraction = max_bound_fraction.max(if bound > 0.0 {
                                error / bound
                            } else {
                                0.0
                            });
                            max_condition = max_condition.max(condition);
                            sequential_f32_mismatches +=
                                usize::from(sequential_f32.to_bits() != actual[token * rows + row]);
                            assert!(
                                error <= bound,
                                "independent dot format={format:?} K={width} n={n} token={token} row={row} got={got} exact={exact} sequential_f32={sequential_f32} sum_abs={sum_abs} error={error} bound={bound}"
                            );
                        }
                    }
                    eprintln!("[fp16-sg-numerics] format={format:?} K={width} n={n} native_mismatches=0 sequential_f32_spot_mismatches={sequential_f32_mismatches} max_abs_error={max_absolute_error} max_bound_fraction={max_bound_fraction} max_condition={max_condition} worst={worst:?}");
                    pool_recycle(k, keep);
                }
            }
        }
    }
}
