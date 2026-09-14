// Included in metal::tests so this test uses the established strict V2 fixtures.
// Run on mini2; it exercises actual f32 input quantization and both panel offsets.
#[test]
fn metal_kquant_v4_stream16_production_route_is_bit_identical() {
    if !detect_metal_device().available {
        return;
    }
    let k = metal_linear_kernel().expect("Metal kernel");
    let v4 = kquant_v2_kernels().expect("strict V2 library");
    let read_words = |buffer: &Buffer, len: usize| unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u32>(), len).to_vec()
    };
    let sentinel = KQUANT_TEST_SENTINEL.to_bits();
    let guard = 16usize;
    for input_width in [3072usize, 8192] {
        let n_sb = input_width / 256;
        let specs = [
            (ResidentWeightFormat::Q4K, 19usize),
            (ResidentWeightFormat::Q6K, 8),
            (ResidentWeightFormat::Q4K, 33),
        ];
        let weights: Vec<_> = specs
            .iter()
            .map(|&(format, rows)| {
                let block_bytes = format.wire_bytes_per_block();
                let mut wire: Vec<u8> = (0..rows * n_sb * block_bytes)
                    .map(|i| ((i * 61 + i / 19 + n_sb) & 255) as u8)
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
                let buffer = k
                    .device
                    .new_buffer(wire.len() as u64, MTLResourceOptions::StorageModeShared);
                write_buffer_u8(&buffer, &wire);
                ResidentLinearWeight {
                    format,
                    buffer,
                    soa8_buffer: None,
                    q8_wire: false,
                }
            })
            .collect();
        for n_tokens in [9usize, 15, 16] {
            let mut input: Vec<f32> = (0..n_tokens * input_width)
                .map(|i| {
                    let token = i / input_width;
                    let column = i % input_width;
                    (((column * 43 + token * 137) % 257) as f32 - 128.0) * 0.005
                })
                .collect();
            for token in 0..n_tokens {
                for sb in 0..n_sb {
                    let base = token * input_width + sb * 256;
                    if (token + sb) % 11 == 0 {
                        input[base..base + 256].fill(0.0);
                    } else {
                        input[base + 7] = if token % 2 == 0 { 1.25 } else { -1.25 };
                        input[base + 200] = -input[base + 7];
                    }
                }
            }
            let input_buffer = k.device.new_buffer(
                (input.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            write_buffer_f32(&input_buffer, &input);
            // Three mixed-format outputs prove immutable per-projection rows;
            // the single Q6 group also exercises the no-ysums staging path.
            for indices in [&[0usize, 1, 2][..], &[1usize][..]] {
                let outputs: Vec<_> = indices
                    .iter()
                    .map(|&i| {
                        let words = n_tokens * specs[i].1 + guard;
                        let buffer = k
                            .device
                            .new_buffer((words * 4) as u64, MTLResourceOptions::StorageModeShared);
                        fill_buffer_sentinel(&buffer, words);
                        buffer
                    })
                    .collect();
                let scalar = k
                    .device
                    .new_buffer(12, MTLResourceOptions::StorageModeShared);
                let projections: Vec<_> = indices
                    .iter()
                    .enumerate()
                    .map(|(j, &i)| (&weights[i], &outputs[j], &scalar, specs[i].1))
                    .collect();
                let cb = k.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                let mut keep = Vec::new();
                let before = metal_kquant_v4_stream16_projection_count();
                for invalid_tokens in [0, 1, 8, 17] {
                    assert!(!encode_kquant_v4_stream16_group(
                        e,
                        k,
                        &mut keep,
                        &input_buffer,
                        &projections,
                        input_width,
                        invalid_tokens
                    ));
                }
                assert!(!encode_kquant_v4_stream16_group(
                    e,
                    k,
                    &mut keep,
                    &input_buffer,
                    &[],
                    input_width,
                    n_tokens
                ));
                assert_eq!(before, metal_kquant_v4_stream16_projection_count());
                assert!(
                    keep.is_empty(),
                    "rejected route must not allocate or encode"
                );
                assert!(encode_kquant_v4_stream16_group(
                    e,
                    k,
                    &mut keep,
                    &input_buffer,
                    &projections,
                    input_width,
                    n_tokens
                ));
                let mut reference_outputs = Vec::new();
                for t0 in [0usize, 8] {
                    let gn = (n_tokens - t0).min(8);
                    let chunk = k.device.new_buffer(
                        (gn * input_width * 4) as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    write_buffer_f32(&chunk, &input[t0 * input_width..(t0 + gn) * input_width]);
                    let scales = pool_get(k, (gn * n_sb * 4) as u64);
                    let quants = pool_get(k, (gn * input_width) as u64);
                    let stage = allocate_kquant_v4_activation_stage(k, input_width, gn, true, true);
                    e.set_compute_pipeline_state(&k.quantize_q8k_rows_pipeline);
                    e.set_buffer(0, Some(&chunk), 0);
                    e.set_buffer(1, Some(&scales), 0);
                    e.set_buffer(2, Some(&quants), 0);
                    e.set_buffer(3, Some(&stage.scalar), 0);
                    e.set_buffer(4, Some(&stage.scalar), 8);
                    dispatch_1d(e, &k.quantize_q8k_rows_pipeline, gn * n_sb);
                    encode_kquant_v4_activation_stage(e, v4, &quants, &stage);
                    for (j, &i) in indices.iter().enumerate() {
                        let rows = specs[i].1;
                        let geometry = pool_get(k, 12);
                        unsafe {
                            let p = geometry.contents().cast::<u32>();
                            *p = n_sb as u32;
                            *p.add(1) = rows as u32;
                            *p.add(2) = gn as u32;
                        }
                        let output = k.device.new_buffer(
                            ((gn * rows + guard) * 4) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        fill_buffer_sentinel(&output, gn * rows + guard);
                        assert!(encode_kquant_v4_prepared_projection_route(
                            e,
                            v4,
                            &scales,
                            &stage,
                            &weights[i],
                            &output,
                            &geometry,
                            rows,
                            KquantV4ProjectionRoute::RegisterExact
                        ));
                        reference_outputs.push((output, j, t0, gn, rows));
                        keep.push(geometry);
                    }
                    keep.extend([chunk, scales, quants]);
                    stage.recycle_into(&mut keep);
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
                let actual: Vec<_> = outputs
                    .iter()
                    .enumerate()
                    .map(|(j, output)| {
                        let rows = specs[indices[j]].1;
                        let words = read_words(output, n_tokens * rows + guard);
                        assert!(words[n_tokens * rows..]
                            .iter()
                            .all(|&word| word == sentinel));
                        assert!(words[..n_tokens * rows]
                            .iter()
                            .all(|&word| word != sentinel && f32::from_bits(word).is_finite()));
                        words
                    })
                    .collect();
                for (output, j, t0, gn, rows) in reference_outputs {
                    let expected = read_words(&output, gn * rows + guard);
                    assert!(expected[gn * rows..].iter().all(|&word| word == sentinel));
                    assert_eq!(
                        &expected[..gn * rows],
                        &actual[j][t0 * rows..(t0 + gn) * rows],
                        "width={input_width} N={n_tokens} projection={j} tile={t0}"
                    );
                }
            }
        }
    }
}
