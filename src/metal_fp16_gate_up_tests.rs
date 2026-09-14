#[cfg(test)]
mod fp16_gate_up_tests {
    use super::*;

    /// Actual3B gate/up geometry. The fused matrix and two independent matrices
    /// contain exactly the same half operands. Check every projection and SiLU
    /// word, including partial token tiles and all output guards.
    #[test]
    fn fused_gate_up_projection_and_silu_match_separate() {
        assert!(
            detect_metal_device().available,
            "run this diagnostic on mini2"
        );
        let k = metal_linear_kernel().expect("Metal");
        let pipeline = fp16_probe_gate_up_silu_pipeline(k).expect("combined-layout SiLU");
        let alloc = |bytes: usize| {
            k.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        };
        let hidden = 3072usize;
        let ffn = 8192usize;
        let half_words = hidden * ffn;
        let guard = 16usize;
        let sentinel = 0x7fa12345u32;
        let combined_weights = alloc(half_words * 2 * 2);
        let gate_weights = alloc(half_words * 2);
        let up_weights = alloc(half_words * 2);
        unsafe {
            let gate =
                std::slice::from_raw_parts_mut(gate_weights.contents().cast::<u16>(), half_words);
            let up =
                std::slice::from_raw_parts_mut(up_weights.contents().cast::<u16>(), half_words);
            for i in 0..half_words {
                gate[i] = f32_to_f16_bits((((i * 31 + i / 17) % 127) as f32 - 63.0) * 0.001);
                up[i] = f32_to_f16_bits((((i * 43 + i / 23) % 251) as f32 - 125.0) * 0.0003);
            }
            std::ptr::copy_nonoverlapping(
                gate.as_ptr(),
                combined_weights.contents().cast::<u16>(),
                half_words,
            );
            std::ptr::copy_nonoverlapping(
                up.as_ptr(),
                combined_weights.contents().cast::<u16>().add(half_words),
                half_words,
            );
        }
        let read = |b: &Buffer, count: usize| unsafe {
            std::slice::from_raw_parts(b.contents().cast::<u32>(), count).to_vec()
        };
        let guarded = |count: usize| {
            let b = alloc((count + guard) * 4);
            unsafe {
                std::slice::from_raw_parts_mut(b.contents().cast::<u32>(), count + guard)
                    .fill(sentinel);
            }
            b
        };
        for n in [1usize, 17, 64] {
            let panel = alloc(64 * hidden * 2);
            unsafe {
                let p = std::slice::from_raw_parts_mut(panel.contents().cast::<u16>(), 64 * hidden);
                p.fill(0);
                for (i, value) in p[..n * hidden].iter_mut().enumerate() {
                    *value =
                        f32_to_f16_bits((((i * 17 + i / hidden * 73) % 257) as f32 - 128.0) * 0.01);
                }
            }
            let combined = guarded(n * 2 * ffn);
            let gate = guarded(n * ffn);
            let up = guarded(n * ffn);
            let actual_silu = guarded(n * ffn);
            let expected_silu = guarded(n * ffn);
            let count = alloc(8);
            unsafe {
                *count.contents().cast::<u32>() = (n * hidden) as u32;
                *count.contents().cast::<u32>().add(1) = (n * ffn) as u32;
            }
            let cb = k.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            let mut keep = Vec::new();
            encode_fp16_probe_mm(
                e,
                k,
                &mut keep,
                &combined_weights,
                &panel,
                &combined,
                hidden,
                2 * ffn,
                n,
            );
            encode_fp16_probe_gate_up_silu(e, pipeline, &combined, &actual_silu, &count, 4, n);
            encode_fp16_probe_mm(
                e,
                k,
                &mut keep,
                &gate_weights,
                &panel,
                &gate,
                hidden,
                ffn,
                n,
            );
            encode_fp16_probe_mm(e, k, &mut keep, &up_weights, &panel, &up, hidden, ffn, n);
            encode_binary_off(
                e,
                &k.silu_mul_pipeline,
                &gate,
                &up,
                &expected_silu,
                &count,
                4,
                n * ffn,
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), metal::MTLCommandBufferStatus::Completed);
            let combined_words = read(&combined, n * 2 * ffn + guard);
            let gate_words = read(&gate, n * ffn + guard);
            let up_words = read(&up, n * ffn + guard);
            let got = read(&actual_silu, n * ffn + guard);
            let expected = read(&expected_silu, n * ffn + guard);
            for (words, len) in [
                (&combined_words, n * 2 * ffn),
                (&gate_words, n * ffn),
                (&up_words, n * ffn),
                (&got, n * ffn),
                (&expected, n * ffn),
            ] {
                assert!(words[len..].iter().all(|&v| v == sentinel), "guard n={n}");
                assert!(
                    words[..len].iter().all(|&v| f32::from_bits(v).is_finite()),
                    "finite n={n}"
                );
            }
            for token in 0..n {
                assert_eq!(
                    &combined_words[token * 2 * ffn..token * 2 * ffn + ffn],
                    &gate_words[token * ffn..(token + 1) * ffn],
                    "gate n={n} token={token}"
                );
                assert_eq!(
                    &combined_words[token * 2 * ffn + ffn..(token + 1) * 2 * ffn],
                    &up_words[token * ffn..(token + 1) * ffn],
                    "up n={n} token={token}"
                );
            }
            assert_eq!(got, expected, "combined-layout SiLU n={n}");
            eprintln!(
                "[fp16-gate-up] n={n} projection_words={} silu_words={} exact=true guards=true",
                n * 2 * ffn,
                n * ffn
            );
            pool_recycle(k, keep);
        }
    }
}
