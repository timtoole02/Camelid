//! Per-kernel parity tests for the resident-decode kernels. Each test runs one
//! kernel on the GPU and compares to a small CPU reference, so a divergence is
//! isolated to a single kernel. All require a CUDA device (`#[ignore]`d in
//! GPU-less CI); run with `cargo test --features cuda -- --ignored`.

use super::{CudaResidentDecode, CudaResidentKernels, ProjQuant, ResidentCudaArtifact};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

#[test]
fn q8_resident_cache_rejects_partial_blocks_before_cuda_initialization() {
    let result = CudaResidentDecode::new_for_artifact_with_kv_quant(
        1,
        1,
        1,
        48,
        48,
        64,
        48,
        8,
        32,
        1e-5,
        false,
        ResidentCudaArtifact::Generic,
        crate::model::KvCacheQuantization::Q8_0,
    );
    let err = match result {
        Ok(_) => panic!("a partial Q8_0 head block must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("multiple of 32"), "unexpected error: {err}");
}

fn fill_q1_wire(wire: &mut [u8], seed: u8) {
    for (block_index, block) in wire.chunks_exact_mut(18).enumerate() {
        let scale = 0.0025 + (block_index % 29) as f32 * 0.000_19;
        block[..2].copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
        for (byte_index, value) in block[2..].iter_mut().enumerate() {
            *value = seed
                .wrapping_add((block_index as u8).wrapping_mul(37))
                .wrapping_add((byte_index as u8).wrapping_mul(19));
        }
    }
}

// Pure predicate (no GPU): the device-decode embed-gather allowlist must stay in
// lockstep with the `embed_gather_*` dispatch in `forward_token_device`. Families
// without a kernel (Q5_K/Q2_K/IQ4_XS) must be refused at `set_device_decode_tables`
// so the engine falls back to the host-fed loop instead of failing mid-forward.
#[test]
fn device_embed_gather_allowlist_matches_the_gather_dispatch() {
    for q in [
        ProjQuant::Q8_0,
        ProjQuant::Q1_0,
        ProjQuant::Q2_0G64,
        ProjQuant::Q2_0G128,
        ProjQuant::Q4K,
        ProjQuant::Q6K,
        ProjQuant::Q3K,
    ] {
        assert!(
            q.has_device_embed_gather(),
            "{q:?} has an embed_gather_* kernel and must be device-decode eligible"
        );
    }
    for q in [ProjQuant::Q5K, ProjQuant::Q2K, ProjQuant::IQ4XS] {
        assert!(
            !q.has_device_embed_gather(),
            "{q:?} has no embed_gather_* kernel and must fall back to the host-fed loop"
        );
    }
}

#[test]
fn gemma4_expert_q8_warp_policy_accepts_only_supported_cta_shapes() {
    for (raw, expected) in [
        (None, 2),
        (Some("1"), 1),
        (Some(" 2 "), 2),
        (Some("4"), 4),
        (Some("8"), 8),
        (Some(""), 2),
        (Some("0"), 2),
        (Some("3"), 2),
        (Some("16"), 2),
        (Some("-1"), 2),
        (Some("not-a-number"), 2),
    ] {
        assert_eq!(
            super::parse_gemma4_expert_q8_warps(raw),
            expected,
            "unexpected policy for {raw:?}"
        );
    }
}

#[test]
fn gemma4_mtp_routed_q4_chunked_policy_is_strict() {
    assert!(!super::parse_gemma4_mtp_routed_q4_chunked(None).unwrap());
    assert!(!super::parse_gemma4_mtp_routed_q4_chunked(Some("0")).unwrap());
    assert!(super::parse_gemma4_mtp_routed_q4_chunked(Some("1")).unwrap());
    for invalid in ["", " 1", "1 ", "true", "2", "-1"] {
        let error = super::parse_gemma4_mtp_routed_q4_chunked(Some(invalid)).unwrap_err();
        assert!(error.contains("must be exactly 0 or 1"), "{error}");
    }
}

#[test]
fn gemma4_mtp_dense_q4_zero_shared_policy_defaults_to_shared_and_is_strict() {
    assert!(!super::parse_gemma4_mtp_dense_q4_zero_shared(None).unwrap());
    assert!(!super::parse_gemma4_mtp_dense_q4_zero_shared(Some("0")).unwrap());
    assert!(super::parse_gemma4_mtp_dense_q4_zero_shared(Some("1")).unwrap());
    for invalid in ["", " 1", "1 ", "true", "2", "-1"] {
        let error = super::parse_gemma4_mtp_dense_q4_zero_shared(Some(invalid)).unwrap_err();
        assert!(error.contains("must be exactly 0 or 1"), "{error}");
    }
}

#[test]
fn gemma4_mtp_dense_q4_imma_policy_defaults_off_and_is_strict() {
    assert!(!super::parse_gemma4_mtp_dense_q4_imma(None).unwrap());
    assert!(!super::parse_gemma4_mtp_dense_q4_imma(Some("0")).unwrap());
    assert!(super::parse_gemma4_mtp_dense_q4_imma(Some("1")).unwrap());
    for invalid in ["", " 1", "1 ", "true", "2", "-1"] {
        let error = super::parse_gemma4_mtp_dense_q4_imma(Some(invalid)).unwrap_err();
        assert!(error.contains("must be exactly 0 or 1"), "{error}");
    }
}

#[test]
fn gemma4_mtp_dense_q6_anchor_dp4a_policy_defaults_off_and_is_strict() {
    assert!(!super::parse_gemma4_mtp_dense_q6_anchor_dp4a(None).unwrap());
    assert!(!super::parse_gemma4_mtp_dense_q6_anchor_dp4a(Some("0")).unwrap());
    assert!(super::parse_gemma4_mtp_dense_q6_anchor_dp4a(Some("1")).unwrap());
    for invalid in ["", " 1", "1 ", "true", "2", "-1"] {
        let error = super::parse_gemma4_mtp_dense_q6_anchor_dp4a(Some(invalid)).unwrap_err();
        assert!(error.contains("must be exactly 0 or 1"), "{error}");
    }
}

#[test]
fn q1t128_roundtrip_is_same_size_and_tail_exact() {
    for &(rows, cols) in &[
        (1usize, 128usize),
        (8, 256),
        (127, 384),
        (128, 256),
        (129, 256),
        (255, 512),
        (257, 128),
    ] {
        let blocks = cols / 128;
        let mut wire = vec![0u8; rows * blocks * 18];
        for (index, byte) in wire.iter_mut().enumerate() {
            *byte = index
                .wrapping_mul(73)
                .wrapping_add(index / 17)
                .wrapping_add(29) as u8;
        }
        let tiled = super::repack_q1_t128(&wire, rows, cols).unwrap();
        assert_eq!(tiled.len(), wire.len(), "Q1T128 changed VRAM size");
        let roundtrip = super::unpack_q1_t128(&tiled, rows, cols).unwrap();
        assert_eq!(
            roundtrip, wire,
            "Q1T128 tail roundtrip failed at {rows}x{cols}"
        );

        // Pin the group shape directly: first K block in each row tile is all
        // signs followed by all scales, including a short final tile.
        let mut group = 0usize;
        for row0 in (0..rows).step_by(128) {
            let nr = (rows - row0).min(128);
            for block in 0..blocks {
                for rr in 0..nr {
                    let raw = ((row0 + rr) * blocks + block) * 18;
                    assert_eq!(
                        &tiled[group + rr * 16..group + (rr + 1) * 16],
                        &wire[raw + 2..raw + 18]
                    );
                    assert_eq!(
                        &tiled[group + nr * 16 + rr * 2..group + nr * 16 + (rr + 1) * 2],
                        &wire[raw..raw + 2]
                    );
                }
                group += nr * 18;
            }
        }
        assert_eq!(group, tiled.len());
    }
    assert!(super::repack_q1_t128(&[0; 18], 1, 64).is_err());
    assert!(super::repack_q1_t128(&[0; 17], 1, 128).is_err());
}

#[test]
#[ignore = "requires a CUDA device; validates every Q1T128 reader"]
fn prism_q1t128_readers_match_raw_wire_with_row_and_token_tails() {
    let Some(kernels) = kernels() else {
        return;
    };
    let (rows, cols, k_tokens) = (131usize, 256usize, 19usize);
    let mut rng = Lcg(0x51_31_74_12_80);
    let mut wire = vec![0u8; rows * (cols / 128) * 18];
    for block in wire.chunks_exact_mut(18) {
        let scale = 0.002 + rng.next_f32().abs() * 0.035;
        block[..2].copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
        for byte in &mut block[2..] {
            *byte = rng.next_u8();
        }
    }
    let tiled = super::repack_q1_t128(&wire, rows, cols).unwrap();
    let activation = (0..k_tokens * cols)
        .map(|_| rng.next_f32() * 1.75)
        .collect::<Vec<_>>();
    let d_wire = kernels.stream.clone_htod(&wire).unwrap();
    let d_tiled = kernels.stream.clone_htod(&tiled).unwrap();
    let d_activation = kernels.stream.clone_htod(&activation).unwrap();

    // Strict f32 decode and K<=2 prompt paths must remain bitwise identical:
    // Q1T128 changes addresses only, never the parity-locked reduction order.
    let mut d_strict_raw = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    let mut d_strict_tiled = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_prism_low_bit_f32_gemv(
        &kernels.stream,
        &kernels.prism_low_bit_f32_gemv,
        &d_activation,
        &d_wire.slice(0..wire.len()),
        rows,
        cols,
        1,
        128,
        false,
        &mut d_strict_raw,
        0,
    )
    .unwrap();
    super::launch_prism_low_bit_f32_gemv(
        &kernels.stream,
        &kernels.prism_low_bit_f32_gemv,
        &d_activation,
        &d_tiled.slice(0..tiled.len()),
        rows,
        cols,
        1,
        128,
        true,
        &mut d_strict_tiled,
        0,
    )
    .unwrap();
    let mut d_prompt_raw = kernels.stream.alloc_zeros::<f32>(2 * rows).unwrap();
    let mut d_prompt_tiled = kernels.stream.alloc_zeros::<f32>(2 * rows).unwrap();
    super::launch_prism_q1_f32_gemm_batched(
        &kernels.stream,
        &kernels.prism_q1_f32_gemm_batched,
        &d_activation,
        &d_wire.slice(0..wire.len()),
        rows,
        cols,
        2,
        false,
        &mut d_prompt_raw,
    )
    .unwrap();
    super::launch_prism_q1_f32_gemm_batched(
        &kernels.stream,
        &kernels.prism_q1_f32_gemm_batched,
        &d_activation,
        &d_tiled.slice(0..tiled.len()),
        rows,
        cols,
        2,
        true,
        &mut d_prompt_tiled,
    )
    .unwrap();

    let mut d_quants = kernels.stream.alloc_zeros::<i8>(k_tokens * cols).unwrap();
    let mut d_scales = kernels
        .stream
        .alloc_zeros::<f32>(k_tokens * cols / 32)
        .unwrap();
    super::launch_quantize(
        &kernels.stream,
        &kernels.quantize,
        &d_activation,
        &mut d_quants,
        &mut d_scales,
        k_tokens * cols / 32,
    )
    .unwrap();

    // Decode uses a deliberately different four-lane subgroup mapping to make
    // the tiled sign slab coalesced. It may reassociate block sums by a few ULP,
    // while the batched readers below retain bitwise output parity.
    let mut d_fast_raw = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    let mut d_fast_tiled = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_prism_q1_q8_gemv(
        &kernels.stream,
        &kernels.prism_q1_q8_gemv,
        &d_quants,
        &d_scales,
        &d_wire.slice(0..wire.len()),
        rows,
        cols,
        &mut d_fast_raw,
        0,
    )
    .unwrap();
    super::launch_prism_q1_q8_gemv(
        &kernels.stream,
        &kernels.prism_q1t128_q8_gemv,
        &d_quants,
        &d_scales,
        &d_tiled.slice(0..tiled.len()),
        rows,
        cols,
        &mut d_fast_tiled,
        0,
    )
    .unwrap();

    let mut d_dp4a_raw = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
    let mut d_dp4a_tiled = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
    super::launch_prism_q1_q8_gemm_batched(
        &kernels.stream,
        &kernels.prism_q1_q8_gemm_batched,
        &d_quants,
        &d_scales,
        &d_wire.slice(0..wire.len()),
        rows,
        cols,
        k_tokens,
        false,
        &mut d_dp4a_raw,
        0,
    )
    .unwrap();
    super::launch_prism_q1_q8_gemm_batched(
        &kernels.stream,
        &kernels.prism_q1_q8_gemm_batched,
        &d_quants,
        &d_scales,
        &d_tiled.slice(0..tiled.len()),
        rows,
        cols,
        k_tokens,
        true,
        &mut d_dp4a_tiled,
        0,
    )
    .unwrap();

    let mut tensor_core_outputs = None;
    if let Some(imma) = kernels.prism_q1_q8_wmma_gemm_batched.as_ref() {
        let mut raw = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        let mut tiled_out = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        super::launch_prism_q1_q8_wmma_gemm_batched(
            &kernels.stream,
            imma,
            &d_quants,
            &d_scales,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            k_tokens,
            false,
            &mut raw,
            0,
        )
        .unwrap();
        super::launch_prism_q1_q8_wmma_gemm_batched(
            &kernels.stream,
            imma,
            &d_quants,
            &d_scales,
            &d_tiled.slice(0..tiled.len()),
            rows,
            cols,
            k_tokens,
            true,
            &mut tiled_out,
            0,
        )
        .unwrap();
        tensor_core_outputs = Some((raw, tiled_out));
    }

    let mut bmma_outputs = None;
    if let (Some(pack), Some(bmma)) = (
        kernels.prism_q8_b128_bitpack.as_ref(),
        kernels.prism_q1_q8_b128_bmma_gemm_batched.as_ref(),
    ) {
        let mut bits = kernels
            .stream
            .alloc_zeros::<u32>(k_tokens * cols / 4)
            .unwrap();
        let mut scales = kernels
            .stream
            .alloc_zeros::<f32>(k_tokens * cols / 128)
            .unwrap();
        super::launch_prism_q8_b128_bitpack(
            &kernels.stream,
            pack,
            &d_activation,
            &mut bits,
            &mut scales,
            cols,
            k_tokens,
        )
        .unwrap();
        let mut raw = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        let mut tiled_out = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        super::launch_prism_q1_q8_b128_bmma_gemm_batched(
            &kernels.stream,
            bmma,
            &bits,
            &scales,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            k_tokens,
            false,
            &mut raw,
            0,
        )
        .unwrap();
        super::launch_prism_q1_q8_b128_bmma_gemm_batched(
            &kernels.stream,
            bmma,
            &bits,
            &scales,
            &d_tiled.slice(0..tiled.len()),
            rows,
            cols,
            k_tokens,
            true,
            &mut tiled_out,
            0,
        )
        .unwrap();
        bmma_outputs = Some((raw, tiled_out));
    }

    let mut strict_raw = vec![0.0f32; rows];
    let mut strict_tiled = vec![0.0f32; rows];
    let mut prompt_raw = vec![0.0f32; 2 * rows];
    let mut prompt_tiled = vec![0.0f32; 2 * rows];
    let mut fast_raw = vec![0.0f32; rows];
    let mut fast_tiled = vec![0.0f32; rows];
    let mut dp4a_raw = vec![0.0f32; k_tokens * rows];
    let mut dp4a_tiled = vec![0.0f32; k_tokens * rows];
    kernels
        .stream
        .memcpy_dtoh(&d_strict_raw, &mut strict_raw)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_strict_tiled, &mut strict_tiled)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_prompt_raw, &mut prompt_raw)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_prompt_tiled, &mut prompt_tiled)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_fast_raw, &mut fast_raw)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_fast_tiled, &mut fast_tiled)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_dp4a_raw, &mut dp4a_raw)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_dp4a_tiled, &mut dp4a_tiled)
        .unwrap();
    kernels.ctx.synchronize().unwrap();

    assert_same_bits("strict Q1 decode", &strict_raw, &strict_tiled);
    assert_same_bits("strict Q1 prompt", &prompt_raw, &prompt_tiled);
    assert_same_bits("Q1 DP4A prompt", &dp4a_raw, &dp4a_tiled);
    let (cosine, relative_l2, max_abs) = vector_error(&fast_tiled, &fast_raw);
    assert!(
        cosine > 0.999_999 && relative_l2 < 3e-6,
        "Q1T128 subgroup decode diverged: cosine={cosine:.9} relative_l2={relative_l2:.9} max_abs={max_abs:.9}"
    );

    if let Some((raw, tiled_out)) = tensor_core_outputs {
        let mut raw_host = vec![0.0f32; k_tokens * rows];
        let mut tiled_host = vec![0.0f32; k_tokens * rows];
        kernels.stream.memcpy_dtoh(&raw, &mut raw_host).unwrap();
        kernels
            .stream
            .memcpy_dtoh(&tiled_out, &mut tiled_host)
            .unwrap();
        kernels.ctx.synchronize().unwrap();
        assert_same_bits("Q1 IMMA prompt", &raw_host, &tiled_host);
    }
    if let Some((raw, tiled_out)) = bmma_outputs {
        let mut raw_host = vec![0.0f32; k_tokens * rows];
        let mut tiled_host = vec![0.0f32; k_tokens * rows];
        kernels.stream.memcpy_dtoh(&raw, &mut raw_host).unwrap();
        kernels
            .stream
            .memcpy_dtoh(&tiled_out, &mut tiled_host)
            .unwrap();
        kernels.ctx.synchronize().unwrap();
        assert_same_bits("Q1 BMMA prompt", &raw_host, &tiled_host);
    }
}

#[test]
#[ignore = "requires a CUDA device; exact Bonsai-27B fused-projection parity"]
fn prism_q1t128_fused_bonsai27b_projections_match_separate_launches_bitwise() {
    let Some(kernels) = kernels() else {
        return;
    };
    const COLS: usize = 5_120;
    let mut rng = Lcg(0x000f_027b_05a1);
    let activation = (0..COLS).map(|_| rng.next_f32() * 1.75).collect::<Vec<_>>();
    let d_activation = kernels.stream.clone_htod(&activation).unwrap();
    let mut d_quants = kernels.stream.alloc_zeros::<i8>(COLS).unwrap();
    let mut d_scales = kernels.stream.alloc_zeros::<f32>(COLS / 32).unwrap();
    super::launch_quantize(
        &kernels.stream,
        &kernels.quantize,
        &d_activation,
        &mut d_quants,
        &mut d_scales,
        COLS / 32,
    )
    .unwrap();
    let mut d_bitplanes = kernels.stream.alloc_zeros::<u32>((COLS / 32) * 8).unwrap();
    let mut d_qsums = kernels.stream.alloc_zeros::<i32>(COLS / 32).unwrap();

    // Full attention: qgate + K + V.
    {
        const QGATE: usize = 12_288;
        const KV: usize = 1_024;
        let qgate = q1t_fixture(QGATE, COLS, &mut rng);
        let k = q1t_fixture(KV, COLS, &mut rng);
        let v = q1t_fixture(KV, COLS, &mut rng);
        let d_qgate = kernels.stream.clone_htod(&qgate).unwrap();
        let d_k = kernels.stream.clone_htod(&k).unwrap();
        let d_v = kernels.stream.clone_htod(&v).unwrap();
        let mut separate_qgate = kernels.stream.alloc_zeros::<f32>(QGATE).unwrap();
        let mut separate_k = kernels.stream.alloc_zeros::<f32>(KV).unwrap();
        let mut separate_v = kernels.stream.alloc_zeros::<f32>(KV).unwrap();
        let mut fused_qgate = kernels.stream.alloc_zeros::<f32>(QGATE).unwrap();
        let mut fused_k = kernels.stream.alloc_zeros::<f32>(KV).unwrap();
        let mut fused_v = kernels.stream.alloc_zeros::<f32>(KV).unwrap();
        let mut popc_qgate = kernels.stream.alloc_zeros::<f32>(QGATE).unwrap();
        let mut popc_k = kernels.stream.alloc_zeros::<f32>(KV).unwrap();
        let mut popc_v = kernels.stream.alloc_zeros::<f32>(KV).unwrap();
        super::launch_prism_q1_q8_gemv(
            &kernels.stream,
            &kernels.prism_q1t128_q8_gemv,
            &d_quants,
            &d_scales,
            &d_qgate.slice(0..qgate.len()),
            QGATE,
            COLS,
            &mut separate_qgate,
            0,
        )
        .unwrap();
        super::launch_prism_q1_q8_gemv(
            &kernels.stream,
            &kernels.prism_q1t128_q8_gemv,
            &d_quants,
            &d_scales,
            &d_k.slice(0..k.len()),
            KV,
            COLS,
            &mut separate_k,
            0,
        )
        .unwrap();
        super::launch_prism_q1_q8_gemv(
            &kernels.stream,
            &kernels.prism_q1t128_q8_gemv,
            &d_quants,
            &d_scales,
            &d_v.slice(0..v.len()),
            KV,
            COLS,
            &mut separate_v,
            0,
        )
        .unwrap();
        super::launch_prism_q1t128_fused_full_bonsai27b(
            &kernels.stream,
            &kernels.prism_q1t128_fused_full_bonsai27b,
            &d_quants,
            &d_scales,
            &d_qgate.slice(0..qgate.len()),
            &d_k.slice(0..k.len()),
            &d_v.slice(0..v.len()),
            &mut fused_qgate,
            &mut fused_k,
            &mut fused_v,
        )
        .unwrap();
        super::launch_prism_q8_32_bitplanes_qsum(
            &kernels.stream,
            &kernels.prism_q8_32_bitplanes_qsum,
            &d_quants,
            &mut d_bitplanes,
            &mut d_qsums,
            COLS / 32,
        )
        .unwrap();
        super::launch_prism_q1t128_q8_popc_fused_full_bonsai27b(
            &kernels.stream,
            &kernels.prism_q1t128_q8_popc_fused_full_bonsai27b,
            &d_bitplanes,
            &d_qsums,
            &d_scales,
            &d_qgate.slice(0..qgate.len()),
            &d_k.slice(0..k.len()),
            &d_v.slice(0..v.len()),
            &mut popc_qgate,
            &mut popc_k,
            &mut popc_v,
        )
        .unwrap();
        let mut separate_qgate_host = vec![0.0f32; QGATE];
        let mut separate_k_host = vec![0.0f32; KV];
        let mut separate_v_host = vec![0.0f32; KV];
        let mut fused_qgate_host = vec![0.0f32; QGATE];
        let mut fused_k_host = vec![0.0f32; KV];
        let mut fused_v_host = vec![0.0f32; KV];
        let mut popc_qgate_host = vec![0.0f32; QGATE];
        let mut popc_k_host = vec![0.0f32; KV];
        let mut popc_v_host = vec![0.0f32; KV];
        kernels
            .stream
            .memcpy_dtoh(&separate_qgate, &mut separate_qgate_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&separate_k, &mut separate_k_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&separate_v, &mut separate_v_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_qgate, &mut fused_qgate_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_k, &mut fused_k_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_v, &mut fused_v_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&popc_qgate, &mut popc_qgate_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&popc_k, &mut popc_k_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&popc_v, &mut popc_v_host)
            .unwrap();
        kernels.ctx.synchronize().unwrap();
        assert_same_bits("fused qgate", &fused_qgate_host, &separate_qgate_host);
        assert_same_bits("fused K", &fused_k_host, &separate_k_host);
        assert_same_bits("fused V", &fused_v_host, &separate_v_host);
        assert_same_bits("POPC fused qgate", &popc_qgate_host, &fused_qgate_host);
        assert_same_bits("POPC fused K", &popc_k_host, &fused_k_host);
        assert_same_bits("POPC fused V", &popc_v_host, &fused_v_host);

        let mut popc_ms = Vec::new();
        let mut dp4a_ms = Vec::new();
        for _ in 0..11 {
            let started = std::time::Instant::now();
            super::launch_prism_q8_32_bitplanes_qsum(
                &kernels.stream,
                &kernels.prism_q8_32_bitplanes_qsum,
                &d_quants,
                &mut d_bitplanes,
                &mut d_qsums,
                COLS / 32,
            )
            .unwrap();
            super::launch_prism_q1t128_q8_popc_fused_full_bonsai27b(
                &kernels.stream,
                &kernels.prism_q1t128_q8_popc_fused_full_bonsai27b,
                &d_bitplanes,
                &d_qsums,
                &d_scales,
                &d_qgate.slice(0..qgate.len()),
                &d_k.slice(0..k.len()),
                &d_v.slice(0..v.len()),
                &mut popc_qgate,
                &mut popc_k,
                &mut popc_v,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
            popc_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = std::time::Instant::now();
            super::launch_prism_q1t128_fused_full_bonsai27b(
                &kernels.stream,
                &kernels.prism_q1t128_fused_full_bonsai27b,
                &d_quants,
                &d_scales,
                &d_qgate.slice(0..qgate.len()),
                &d_k.slice(0..k.len()),
                &d_v.slice(0..v.len()),
                &mut fused_qgate,
                &mut fused_k,
                &mut fused_v,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
            dp4a_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let popc_ms = mean(&popc_ms);
        let dp4a_ms = mean(&dp4a_ms);
        eprintln!(
            "Q1T full fused qgate/K/V: POPC+pack={popc_ms:.4}ms DP4A={dp4a_ms:.4}ms speedup={:.3}x",
            dp4a_ms / popc_ms,
        );
    }

    // SSM input: wqkv + z + the short beta/alpha row tile.
    {
        const WQKV: usize = 10_240;
        const Z: usize = 6_144;
        const TAIL: usize = 48;
        let wqkv = q1t_fixture(WQKV, COLS, &mut rng);
        let z = q1t_fixture(Z, COLS, &mut rng);
        let beta = q1t_fixture(TAIL, COLS, &mut rng);
        let alpha = q1t_fixture(TAIL, COLS, &mut rng);
        let d_wqkv = kernels.stream.clone_htod(&wqkv).unwrap();
        let d_z = kernels.stream.clone_htod(&z).unwrap();
        let d_beta = kernels.stream.clone_htod(&beta).unwrap();
        let d_alpha = kernels.stream.clone_htod(&alpha).unwrap();
        let mut separate_wqkv = kernels.stream.alloc_zeros::<f32>(WQKV).unwrap();
        let mut separate_z = kernels.stream.alloc_zeros::<f32>(Z).unwrap();
        let mut separate_beta = kernels.stream.alloc_zeros::<f32>(TAIL).unwrap();
        let mut separate_alpha = kernels.stream.alloc_zeros::<f32>(TAIL).unwrap();
        let mut fused_wqkv = kernels.stream.alloc_zeros::<f32>(WQKV).unwrap();
        let mut fused_z = kernels.stream.alloc_zeros::<f32>(Z).unwrap();
        let mut fused_beta = kernels.stream.alloc_zeros::<f32>(TAIL).unwrap();
        let mut fused_alpha = kernels.stream.alloc_zeros::<f32>(TAIL).unwrap();
        let mut popc_wqkv = kernels.stream.alloc_zeros::<f32>(WQKV).unwrap();
        let mut popc_z = kernels.stream.alloc_zeros::<f32>(Z).unwrap();
        let mut popc_beta = kernels.stream.alloc_zeros::<f32>(TAIL).unwrap();
        let mut popc_alpha = kernels.stream.alloc_zeros::<f32>(TAIL).unwrap();
        for (weight, rows, out) in [
            (&d_wqkv, WQKV, &mut separate_wqkv),
            (&d_z, Z, &mut separate_z),
            (&d_beta, TAIL, &mut separate_beta),
            (&d_alpha, TAIL, &mut separate_alpha),
        ] {
            super::launch_prism_q1_q8_gemv(
                &kernels.stream,
                &kernels.prism_q1t128_q8_gemv,
                &d_quants,
                &d_scales,
                &weight.slice(0..weight.len()),
                rows,
                COLS,
                out,
                0,
            )
            .unwrap();
        }
        super::launch_prism_q1t128_fused_ssm_bonsai27b(
            &kernels.stream,
            &kernels.prism_q1t128_fused_ssm_bonsai27b,
            &d_quants,
            &d_scales,
            &d_wqkv.slice(0..wqkv.len()),
            &d_z.slice(0..z.len()),
            &d_beta.slice(0..beta.len()),
            &d_alpha.slice(0..alpha.len()),
            &mut fused_wqkv,
            &mut fused_z,
            &mut fused_beta,
            &mut fused_alpha,
        )
        .unwrap();
        super::launch_prism_q8_32_bitplanes_qsum(
            &kernels.stream,
            &kernels.prism_q8_32_bitplanes_qsum,
            &d_quants,
            &mut d_bitplanes,
            &mut d_qsums,
            COLS / 32,
        )
        .unwrap();
        super::launch_prism_q1t128_q8_popc_fused_ssm_bonsai27b(
            &kernels.stream,
            &kernels.prism_q1t128_q8_popc_fused_ssm_bonsai27b,
            &d_bitplanes,
            &d_qsums,
            &d_scales,
            &d_wqkv.slice(0..wqkv.len()),
            &d_z.slice(0..z.len()),
            &d_beta.slice(0..beta.len()),
            &d_alpha.slice(0..alpha.len()),
            &mut popc_wqkv,
            &mut popc_z,
            &mut popc_beta,
            &mut popc_alpha,
        )
        .unwrap();
        let mut separate_wqkv_host = vec![0.0f32; WQKV];
        let mut separate_z_host = vec![0.0f32; Z];
        let mut separate_beta_host = vec![0.0f32; TAIL];
        let mut separate_alpha_host = vec![0.0f32; TAIL];
        let mut fused_wqkv_host = vec![0.0f32; WQKV];
        let mut fused_z_host = vec![0.0f32; Z];
        let mut fused_beta_host = vec![0.0f32; TAIL];
        let mut fused_alpha_host = vec![0.0f32; TAIL];
        let mut popc_wqkv_host = vec![0.0f32; WQKV];
        let mut popc_z_host = vec![0.0f32; Z];
        let mut popc_beta_host = vec![0.0f32; TAIL];
        let mut popc_alpha_host = vec![0.0f32; TAIL];
        kernels
            .stream
            .memcpy_dtoh(&separate_wqkv, &mut separate_wqkv_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&separate_z, &mut separate_z_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&separate_beta, &mut separate_beta_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&separate_alpha, &mut separate_alpha_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_wqkv, &mut fused_wqkv_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_z, &mut fused_z_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_beta, &mut fused_beta_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_alpha, &mut fused_alpha_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&popc_wqkv, &mut popc_wqkv_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&popc_z, &mut popc_z_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&popc_beta, &mut popc_beta_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&popc_alpha, &mut popc_alpha_host)
            .unwrap();
        kernels.ctx.synchronize().unwrap();
        assert_same_bits("fused wqkv", &fused_wqkv_host, &separate_wqkv_host);
        assert_same_bits("fused z", &fused_z_host, &separate_z_host);
        assert_same_bits("fused beta", &fused_beta_host, &separate_beta_host);
        assert_same_bits("fused alpha", &fused_alpha_host, &separate_alpha_host);
        assert_same_bits("POPC fused wqkv", &popc_wqkv_host, &fused_wqkv_host);
        assert_same_bits("POPC fused z", &popc_z_host, &fused_z_host);
        assert_same_bits("POPC fused beta", &popc_beta_host, &fused_beta_host);
        assert_same_bits("POPC fused alpha", &popc_alpha_host, &fused_alpha_host);

        let mut popc_ms = Vec::new();
        let mut dp4a_ms = Vec::new();
        for _ in 0..11 {
            let started = std::time::Instant::now();
            super::launch_prism_q8_32_bitplanes_qsum(
                &kernels.stream,
                &kernels.prism_q8_32_bitplanes_qsum,
                &d_quants,
                &mut d_bitplanes,
                &mut d_qsums,
                COLS / 32,
            )
            .unwrap();
            super::launch_prism_q1t128_q8_popc_fused_ssm_bonsai27b(
                &kernels.stream,
                &kernels.prism_q1t128_q8_popc_fused_ssm_bonsai27b,
                &d_bitplanes,
                &d_qsums,
                &d_scales,
                &d_wqkv.slice(0..wqkv.len()),
                &d_z.slice(0..z.len()),
                &d_beta.slice(0..beta.len()),
                &d_alpha.slice(0..alpha.len()),
                &mut popc_wqkv,
                &mut popc_z,
                &mut popc_beta,
                &mut popc_alpha,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
            popc_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = std::time::Instant::now();
            super::launch_prism_q1t128_fused_ssm_bonsai27b(
                &kernels.stream,
                &kernels.prism_q1t128_fused_ssm_bonsai27b,
                &d_quants,
                &d_scales,
                &d_wqkv.slice(0..wqkv.len()),
                &d_z.slice(0..z.len()),
                &d_beta.slice(0..beta.len()),
                &d_alpha.slice(0..alpha.len()),
                &mut fused_wqkv,
                &mut fused_z,
                &mut fused_beta,
                &mut fused_alpha,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
            dp4a_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let popc_ms = mean(&popc_ms);
        let dp4a_ms = mean(&dp4a_ms);
        eprintln!(
            "Q1T SSM fused wqkv/z/beta/alpha: POPC+pack={popc_ms:.4}ms DP4A={dp4a_ms:.4}ms speedup={:.3}x",
            dp4a_ms / popc_ms,
        );
    }

    // FFN gate + up.
    {
        const FFN: usize = 17_408;
        let gate = q1t_fixture(FFN, COLS, &mut rng);
        let up = q1t_fixture(FFN, COLS, &mut rng);
        let d_gate = kernels.stream.clone_htod(&gate).unwrap();
        let d_up = kernels.stream.clone_htod(&up).unwrap();
        let mut separate_gate = kernels.stream.alloc_zeros::<f32>(FFN).unwrap();
        let mut separate_up = kernels.stream.alloc_zeros::<f32>(FFN).unwrap();
        let mut fused_gate = kernels.stream.alloc_zeros::<f32>(FFN).unwrap();
        let mut fused_up = kernels.stream.alloc_zeros::<f32>(FFN).unwrap();
        super::launch_prism_q1_q8_gemv(
            &kernels.stream,
            &kernels.prism_q1t128_q8_gemv,
            &d_quants,
            &d_scales,
            &d_gate.slice(0..gate.len()),
            FFN,
            COLS,
            &mut separate_gate,
            0,
        )
        .unwrap();
        super::launch_prism_q1_q8_gemv(
            &kernels.stream,
            &kernels.prism_q1t128_q8_gemv,
            &d_quants,
            &d_scales,
            &d_up.slice(0..up.len()),
            FFN,
            COLS,
            &mut separate_up,
            0,
        )
        .unwrap();
        super::launch_prism_q1t128_fused_ffn_bonsai27b(
            &kernels.stream,
            &kernels.prism_q1t128_fused_ffn_bonsai27b,
            &d_quants,
            &d_scales,
            &d_gate.slice(0..gate.len()),
            &d_up.slice(0..up.len()),
            &mut fused_gate,
            &mut fused_up,
        )
        .unwrap();
        let mut separate_gate_host = vec![0.0f32; FFN];
        let mut separate_up_host = vec![0.0f32; FFN];
        let mut fused_gate_host = vec![0.0f32; FFN];
        let mut fused_up_host = vec![0.0f32; FFN];
        kernels
            .stream
            .memcpy_dtoh(&separate_gate, &mut separate_gate_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&separate_up, &mut separate_up_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_gate, &mut fused_gate_host)
            .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&fused_up, &mut fused_up_host)
            .unwrap();
        kernels.ctx.synchronize().unwrap();
        assert_same_bits("fused FFN gate", &fused_gate_host, &separate_gate_host);
        assert_same_bits("fused FFN up", &fused_up_host, &separate_up_host);
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn prism_low_bit_f32_gemv_matches_packed_wire_oracle() {
    let Some(kernels) = kernels() else {
        return;
    };
    let rows = 79usize; // crosses several 8-warp thread blocks
    let kdim = 256usize;
    let mut rng = Lcg(0x51_52_53_54);
    let activation = (0..kdim).map(|_| rng.next_f32()).collect::<Vec<_>>();

    for (lane, bits, block_elements) in [
        (ProjQuant::Q1_0, 1usize, 128usize),
        (ProjQuant::Q2_0G64, 2usize, 64usize),
        (ProjQuant::Q2_0G128, 2usize, 128usize),
    ] {
        let block_bytes = 2 + block_elements * bits / 8;
        let weight_blocks_per_row = kdim / block_elements;
        let mut wire = vec![0u8; rows * weight_blocks_per_row * block_bytes];
        for block in wire.chunks_exact_mut(block_bytes) {
            let scale = rng.next_f32().abs() * 0.04 + 0.001;
            block[..2].copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
            for byte in &mut block[2..] {
                *byte = rng.next_u8();
            }
        }

        let mut expected = vec![0.0f32; rows];
        for (row, slot) in expected.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for wb in 0..weight_blocks_per_row {
                let offset = (row * weight_blocks_per_row + wb) * block_bytes;
                let block = &wire[offset..offset + block_bytes];
                let scale =
                    crate::tensor::f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for index in 0..block_elements {
                    let weight = if bits == 1 {
                        if (block[2 + index / 8] >> (index % 8)) & 1 == 1 {
                            1.0
                        } else {
                            -1.0
                        }
                    } else {
                        f32::from((block[2 + index / 4] >> ((index % 4) * 2)) & 3) - 1.0
                    };
                    acc += activation[wb * block_elements + index] * weight * scale;
                }
            }
            *slot = acc;
        }

        let d_input = kernels.stream.clone_htod(&activation).unwrap();
        let d_wire = kernels.stream.clone_htod(&wire).unwrap();
        let mut d_out = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
        super::launch_prism_low_bit_f32_gemv(
            &kernels.stream,
            &kernels.prism_low_bit_f32_gemv,
            &d_input,
            &d_wire.slice(0..wire.len()),
            rows,
            kdim,
            bits,
            block_elements,
            false,
            &mut d_out,
            0,
        )
        .unwrap();
        let mut got = vec![0.0f32; rows];
        kernels.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
        kernels.ctx.synchronize().unwrap();
        assert!(
            close(&got, &expected, 2e-5),
            "{lane:?} packed f32 GEMV diverged from the wire oracle"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn prism_q1_f32_gemm_batched_matches_decode_gemv_bitwise() {
    let Some(kernels) = kernels() else {
        return;
    };
    let (rows, kdim, k_tokens) = (79usize, 256usize, 2usize);
    let blocks_per_row = kdim / 128;
    let mut rng = Lcg(0x71_31_ba_7c);
    let activation = (0..k_tokens * kdim)
        .map(|_| rng.next_f32())
        .collect::<Vec<_>>();
    let mut wire = vec![0u8; rows * blocks_per_row * 18];
    for block in wire.chunks_exact_mut(18) {
        let scale = rng.next_f32().abs() * 0.04 + 0.001;
        block[..2].copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
        for byte in &mut block[2..] {
            *byte = rng.next_u8();
        }
    }

    let d_wire = kernels.stream.clone_htod(&wire).unwrap();
    let mut expected = vec![0.0f32; k_tokens * rows];
    for token in 0..k_tokens {
        let row = &activation[token * kdim..(token + 1) * kdim];
        let d_input = kernels.stream.clone_htod(row).unwrap();
        let mut d_out = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
        super::launch_prism_low_bit_f32_gemv(
            &kernels.stream,
            &kernels.prism_low_bit_f32_gemv,
            &d_input,
            &d_wire.slice(0..wire.len()),
            rows,
            kdim,
            1,
            128,
            false,
            &mut d_out,
            0,
        )
        .unwrap();
        kernels
            .stream
            .memcpy_dtoh(&d_out, &mut expected[token * rows..(token + 1) * rows])
            .unwrap();
    }

    let d_input = kernels.stream.clone_htod(&activation).unwrap();
    let mut d_out = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
    super::launch_prism_q1_f32_gemm_batched(
        &kernels.stream,
        &kernels.prism_q1_f32_gemm_batched,
        &d_input,
        &d_wire.slice(0..wire.len()),
        rows,
        kdim,
        k_tokens,
        false,
        &mut d_out,
    )
    .unwrap();
    let mut got = vec![0.0f32; k_tokens * rows];
    kernels.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    kernels.ctx.synchronize().unwrap();
    assert!(
        got.iter()
            .zip(&expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "Q1 batched prompt GEMM changed the parity-locked decode reduction"
    );
}

#[test]
#[ignore = "requires a CUDA device; performance diagnostic"]
fn prism_q1_f32_gemm_batched_real_shape_speed_probe() {
    let Some(kernels) = kernels() else {
        return;
    };
    let (rows, cols, k_tokens) = (17_408usize, 5_120usize, 2usize);
    let mut wire = vec![0u8; rows * (cols / 128) * 18];
    for (block_index, block) in wire.chunks_exact_mut(18).enumerate() {
        // f16 1.0 scale plus a deterministic, nontrivial Q1 sign pattern.
        block[0] = 0x00;
        block[1] = 0x3c;
        for (byte_index, value) in block[2..].iter_mut().enumerate() {
            *value = (block_index as u8)
                .wrapping_mul(17)
                .wrapping_add((byte_index as u8).wrapping_mul(29));
        }
    }
    let activation = vec![0.25f32; k_tokens * cols];
    let d_wire = kernels.stream.clone_htod(&wire).unwrap();
    let d_batch = kernels.stream.clone_htod(&activation).unwrap();
    let single = vec![0.25f32; cols];
    let d_single = kernels.stream.clone_htod(&single).unwrap();
    let mut d_serial_out = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    let mut d_batch_out = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();

    let mut serial_tile = || {
        for _ in 0..k_tokens {
            super::launch_prism_low_bit_f32_gemv(
                &kernels.stream,
                &kernels.prism_low_bit_f32_gemv,
                &d_single,
                &d_wire.slice(0..wire.len()),
                rows,
                cols,
                1,
                128,
                false,
                &mut d_serial_out,
                0,
            )
            .unwrap();
        }
        kernels.ctx.synchronize().unwrap();
    };
    let mut batched_tile = || {
        super::launch_prism_q1_f32_gemm_batched(
            &kernels.stream,
            &kernels.prism_q1_f32_gemm_batched,
            &d_batch,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            k_tokens,
            false,
            &mut d_batch_out,
        )
        .unwrap();
        kernels.ctx.synchronize().unwrap();
    };
    serial_tile();
    batched_tile();

    let mut serial_ms = Vec::new();
    let mut batched_ms = Vec::new();
    for _ in 0..5 {
        let started = std::time::Instant::now();
        serial_tile();
        serial_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        batched_tile();
        batched_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    eprintln!(
        "Q1 real-shape 2-token tile: serial={:.3}ms batched={:.3}ms speedup={:.3}x",
        mean(&serial_ms),
        mean(&batched_ms),
        mean(&serial_ms) / mean(&batched_ms),
    );
}

#[test]
#[ignore = "requires a CUDA device; performance diagnostic"]
// These closures mutably capture device outputs. Explicitly consuming them
// before DtoH copies is the intended borrow boundary in this diagnostic.
#[allow(clippy::drop_non_drop)]
fn prism_q1_q8_gemm_batched_real_shape_speed_probe() {
    let kernels = CudaResidentKernels::new().expect("CUDA resident kernels for Q1 speed probe");
    let (rows, cols, k_tokens) = (17_408usize, 5_120usize, 8usize);
    let wire = vec![0u8; rows * (cols / 128) * 18];
    let tiled_wire = super::repack_q1_t128(&wire, rows, cols).unwrap();
    let activation = (0..k_tokens * cols)
        .map(|index| ((index % 251) as f32 - 125.0) / 127.0)
        .collect::<Vec<_>>();
    let d_wire = kernels.stream.clone_htod(&wire).unwrap();
    let d_tiled_wire = kernels.stream.clone_htod(&tiled_wire).unwrap();
    let d_activation = kernels.stream.clone_htod(&activation).unwrap();
    let mut d_quants = kernels.stream.alloc_zeros::<i8>(k_tokens * cols).unwrap();
    let mut d_scales = kernels
        .stream
        .alloc_zeros::<f32>(k_tokens * cols / 32)
        .unwrap();
    super::launch_quantize(
        &kernels.stream,
        &kernels.quantize,
        &d_activation,
        &mut d_quants,
        &mut d_scales,
        k_tokens * cols / 32,
    )
    .unwrap();
    let mut d_decode_exact = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    let mut d_decode_fast = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    let mut d_decode_tiled = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    let mut decode_exact = || {
        super::launch_prism_low_bit_f32_gemv(
            &kernels.stream,
            &kernels.prism_low_bit_f32_gemv,
            &d_activation,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            1,
            128,
            false,
            &mut d_decode_exact,
            0,
        )
        .unwrap();
        kernels.ctx.synchronize().unwrap();
    };
    let mut decode_fast = || {
        super::launch_prism_q1_q8_gemv(
            &kernels.stream,
            &kernels.prism_q1_q8_gemv,
            &d_quants,
            &d_scales,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            &mut d_decode_fast,
            0,
        )
        .unwrap();
        kernels.ctx.synchronize().unwrap();
    };
    let mut decode_tiled = || {
        super::launch_prism_q1_q8_gemv(
            &kernels.stream,
            &kernels.prism_q1t128_q8_gemv,
            &d_quants,
            &d_scales,
            &d_tiled_wire.slice(0..tiled_wire.len()),
            rows,
            cols,
            &mut d_decode_tiled,
            0,
        )
        .unwrap();
        kernels.ctx.synchronize().unwrap();
    };
    decode_exact();
    decode_fast();
    decode_tiled();
    let mut decode_exact_ms = Vec::new();
    let mut decode_fast_ms = Vec::new();
    let mut decode_tiled_ms = Vec::new();
    for _ in 0..15 {
        let started = std::time::Instant::now();
        decode_exact();
        decode_exact_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        decode_fast();
        decode_fast_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        decode_tiled();
        decode_tiled_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    drop(decode_exact);
    drop(decode_fast);
    drop(decode_tiled);
    let mut decode_exact_out = vec![0.0f32; rows];
    let mut decode_fast_out = vec![0.0f32; rows];
    let mut decode_tiled_out = vec![0.0f32; rows];
    kernels
        .stream
        .memcpy_dtoh(&d_decode_exact, &mut decode_exact_out)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_decode_fast, &mut decode_fast_out)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_decode_tiled, &mut decode_tiled_out)
        .unwrap();
    let decode_max_abs = decode_exact_out
        .iter()
        .zip(&decode_fast_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (_, tiled_relative_l2, tiled_max_abs) = vector_error(&decode_tiled_out, &decode_fast_out);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    eprintln!(
        "Q1 real-shape decode: exact-f32={:.3}ms raw-q8={:.3}ms q1t128-q8={:.3}ms exact/tiled={:.3}x raw/tiled={:.3}x exact-vs-raw-max={:.6} tiled-vs-raw-rel-l2={:.3e} tiled-vs-raw-max={:.6}",
        mean(&decode_exact_ms),
        mean(&decode_fast_ms),
        mean(&decode_tiled_ms),
        mean(&decode_exact_ms) / mean(&decode_tiled_ms),
        mean(&decode_fast_ms) / mean(&decode_tiled_ms),
        decode_max_abs,
        tiled_relative_l2,
        tiled_max_abs,
    );
    let mut d_exact_out = kernels.stream.alloc_zeros::<f32>(2 * rows).unwrap();
    let mut d_fast_out = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
    let mut d_wmma_out = kernels.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();

    let mut exact_tiles = || {
        for _ in 0..(k_tokens / 2) {
            super::launch_prism_q1_f32_gemm_batched(
                &kernels.stream,
                &kernels.prism_q1_f32_gemm_batched,
                &d_activation,
                &d_wire.slice(0..wire.len()),
                rows,
                cols,
                2,
                false,
                &mut d_exact_out,
            )
            .unwrap();
        }
        kernels.ctx.synchronize().unwrap();
    };
    let mut fast_tile = || {
        super::launch_prism_q1_q8_gemm_batched(
            &kernels.stream,
            &kernels.prism_q1_q8_gemm_batched,
            &d_quants,
            &d_scales,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            k_tokens,
            false,
            &mut d_fast_out,
            0,
        )
        .unwrap();
        kernels.ctx.synchronize().unwrap();
    };
    let wmma = kernels
        .prism_q1_q8_wmma_gemm_batched
        .as_ref()
        .expect("SM 7.5+ tensor-core kernel");
    let mut wmma_tile = || {
        super::launch_prism_q1_q8_wmma_gemm_batched(
            &kernels.stream,
            wmma,
            &d_quants,
            &d_scales,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            k_tokens,
            false,
            &mut d_wmma_out,
            0,
        )
        .unwrap();
        kernels.ctx.synchronize().unwrap();
    };
    exact_tiles();
    fast_tile();
    wmma_tile();

    let mut exact_ms = Vec::new();
    let mut fast_ms = Vec::new();
    let mut wmma_ms = Vec::new();
    for _ in 0..7 {
        let started = std::time::Instant::now();
        exact_tiles();
        exact_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        fast_tile();
        fast_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        wmma_tile();
        wmma_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    drop(fast_tile);
    drop(wmma_tile);
    let mut fast_out = vec![0.0f32; k_tokens * rows];
    let mut wmma_out = vec![0.0f32; k_tokens * rows];
    kernels
        .stream
        .memcpy_dtoh(&d_fast_out, &mut fast_out)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_wmma_out, &mut wmma_out)
        .unwrap();
    let max_abs = fast_out
        .iter()
        .zip(&wmma_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!(
        "Q1 real-shape 8-token tile: exact-k2={:.3}ms q8-dp4a={:.3}ms q8-wmma={:.3}ms exact/wmma={:.3}x dp4a/wmma={:.3}x max_abs={:.6}",
        mean(&exact_ms),
        mean(&fast_ms),
        mean(&wmma_ms),
        mean(&exact_ms) / mean(&wmma_ms),
        mean(&fast_ms) / mean(&wmma_ms),
        max_abs,
    );

    // The tensor-core CTA is designed for a full prompt tile. At K=128 all
    // eight warps reuse the same decoded Q1 tile; compare it with the 64 exact
    // K2 launches Camelid currently needs for the same amount of prompt work.
    let big_k = 128usize;
    let big_activation = (0..big_k * cols)
        .map(|index| ((index % 251) as f32 - 125.0) / 127.0)
        .collect::<Vec<_>>();
    let d_big_activation = kernels.stream.clone_htod(&big_activation).unwrap();
    let mut d_big_quants = kernels.stream.alloc_zeros::<i8>(big_k * cols).unwrap();
    let mut d_big_scales = kernels
        .stream
        .alloc_zeros::<f32>(big_k * cols / 32)
        .unwrap();
    super::launch_quantize(
        &kernels.stream,
        &kernels.quantize,
        &d_big_activation,
        &mut d_big_quants,
        &mut d_big_scales,
        big_k * cols / 32,
    )
    .unwrap();
    let mut d_big_exact = kernels.stream.alloc_zeros::<f32>(2 * rows).unwrap();
    let mut d_big_wmma = kernels.stream.alloc_zeros::<f32>(big_k * rows).unwrap();
    let mut exact_128 = || {
        for _ in 0..(big_k / 2) {
            super::launch_prism_q1_f32_gemm_batched(
                &kernels.stream,
                &kernels.prism_q1_f32_gemm_batched,
                &d_big_activation,
                &d_wire.slice(0..wire.len()),
                rows,
                cols,
                2,
                false,
                &mut d_big_exact,
            )
            .unwrap();
        }
        kernels.ctx.synchronize().unwrap();
    };
    let mut wmma_128 = || {
        super::launch_prism_q1_q8_wmma_gemm_batched(
            &kernels.stream,
            wmma,
            &d_big_quants,
            &d_big_scales,
            &d_wire.slice(0..wire.len()),
            rows,
            cols,
            big_k,
            false,
            &mut d_big_wmma,
            0,
        )
        .unwrap();
        kernels.ctx.synchronize().unwrap();
    };
    exact_128();
    wmma_128();
    let mut exact_128_ms = Vec::new();
    let mut wmma_128_ms = Vec::new();
    for _ in 0..5 {
        let started = std::time::Instant::now();
        exact_128();
        exact_128_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        wmma_128();
        wmma_128_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!(
        "Q1 real-shape 128-token tile: exact-k2x64={:.3}ms q8-wmma={:.3}ms speedup={:.3}x",
        mean(&exact_128_ms),
        mean(&wmma_128_ms),
        mean(&exact_128_ms) / mean(&wmma_128_ms),
    );
}

// CPU oracle for the experimental BMMA activation format. The payload is one
// byte per activation arranged as [K/128][plane][token][u32 word], while the
// returned q values remain token-major solely for the scalar dot reference.
fn quantize_q8_b128_cpu(
    input: &[f32],
    cols: usize,
    k_tokens: usize,
) -> (Vec<u32>, Vec<f32>, Vec<i8>) {
    assert_eq!(cols % 128, 0);
    assert_eq!(input.len(), cols * k_tokens);
    let blocks = cols / 128;
    let mut bitplanes = vec![0u32; k_tokens * cols / 4];
    let mut scales = vec![0.0f32; blocks * k_tokens];
    let mut quants = vec![0i8; input.len()];
    for block in 0..blocks {
        for token in 0..k_tokens {
            let base = token * cols + block * 128;
            let values = &input[base..base + 128];
            let max_abs = values.iter().fold(0.0f32, |m, value| m.max(value.abs()));
            let unrounded = max_abs / 127.0;
            scales[block * k_tokens + token] = f16rt(unrounded);
            let inv = if unrounded == 0.0 {
                0.0
            } else {
                1.0 / unrounded
            };
            for (index, &value) in values.iter().enumerate() {
                let q = (value * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
                quants[base + index] = q;
                let uq = q as u8;
                let word = index / 32;
                let bit = index % 32;
                for plane in 0..8 {
                    if ((uq >> plane) & 1) != 0 {
                        let dst = (((block * 8 + plane) * k_tokens + token) * 4) + word;
                        bitplanes[dst] |= 1u32 << bit;
                    }
                }
            }
        }
    }
    (bitplanes, scales, quants)
}

fn q1_q8_b128_cpu(
    wire: &[u8],
    quants: &[i8],
    scales: &[f32],
    rows: usize,
    cols: usize,
    k_tokens: usize,
) -> Vec<f32> {
    let blocks = cols / 128;
    let mut output = vec![0.0f32; rows * k_tokens];
    for token in 0..k_tokens {
        for row in 0..rows {
            let mut acc = 0.0f32;
            for block in 0..blocks {
                let wb = &wire[(row * blocks + block) * 18..(row * blocks + block + 1) * 18];
                let dw = crate::tensor::f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
                let mut dot = 0i32;
                for index in 0..128 {
                    let sign = if ((wb[2 + index / 8] >> (index % 8)) & 1) != 0 {
                        1
                    } else {
                        -1
                    };
                    dot += sign * i32::from(quants[token * cols + block * 128 + index]);
                }
                let term = (dot as f32 * dw) * scales[block * k_tokens + token];
                acc += term;
            }
            output[token * rows + row] = acc;
        }
    }
    output
}

fn vector_error(actual: &[f32], reference: &[f32]) -> (f64, f64, f32) {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut rr = 0.0f64;
    let mut dd = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&a, &r) in actual.iter().zip(reference) {
        dot += f64::from(a) * f64::from(r);
        aa += f64::from(a) * f64::from(a);
        rr += f64::from(r) * f64::from(r);
        let delta = a - r;
        dd += f64::from(delta) * f64::from(delta);
        max_abs = max_abs.max(delta.abs());
    }
    let cosine = dot / (aa.sqrt() * rr.sqrt()).max(f64::MIN_POSITIVE);
    let relative_l2 = (dd / rr.max(f64::MIN_POSITIVE)).sqrt();
    (cosine, relative_l2, max_abs)
}

#[test]
fn prism_bmma_dispatch_policy_is_strict_and_shape_safe() {
    let allow = |fast, enabled, available, cols, tokens, threshold| {
        super::prism_bmma_dispatch_policy(fast, enabled, available, cols, tokens, threshold)
    };
    assert!(allow(true, true, true, 5_120, 32, 32));
    assert!(allow(true, true, true, 5_120, 128, 32));
    assert!(!allow(false, true, true, 5_120, 128, 32)); // strict mode
    assert!(!allow(true, false, true, 5_120, 128, 32)); // force-disable
    assert!(!allow(true, true, false, 5_120, 128, 32)); // pre-SM80
    assert!(!allow(true, true, true, 5_121, 128, 32)); // incomplete Q1 block
    assert!(!allow(true, true, true, 5_120, 31, 32));
    assert!(!allow(true, true, true, 5_120, 129, 32));

    assert_eq!(super::parse_prism_bmma_min_tokens(None), 32);
    assert_eq!(super::parse_prism_bmma_min_tokens(Some("14")), 14);
    assert_eq!(super::parse_prism_bmma_min_tokens(Some("0")), 1);
    assert_eq!(super::parse_prism_bmma_min_tokens(Some("999")), 128);
    assert_eq!(super::parse_prism_bmma_min_tokens(Some("invalid")), 32);
}

#[test]
fn prism_q1_model_policy_defaults_only_for_exact_windows_bonsai27b() {
    let policy = |fast, windows, artifact, exact, q1_on, q1_off, fused_on, fused_off| {
        super::prism_q1_model_policy(
            fast, windows, artifact, exact, q1_on, q1_off, fused_on, fused_off,
        )
    };
    let prism = super::ResidentCudaArtifact::PrismBonsai27bQ1;
    let generic = super::ResidentCudaArtifact::Generic;

    let production = policy(true, true, prism, true, false, false, false, false);
    assert!(production.q1_tiled);
    assert!(production.fused_projections);

    assert_eq!(
        policy(true, false, prism, true, false, false, false, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: false,
            fused_projections: false,
        }
    );
    assert_eq!(
        policy(true, true, generic, true, false, false, false, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: false,
            fused_projections: false,
        }
    );
    assert_eq!(
        policy(true, true, prism, false, false, false, false, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: false,
            fused_projections: false,
        }
    );
    assert_eq!(
        policy(false, true, prism, true, false, false, false, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: false,
            fused_projections: false,
        }
    );

    assert!(super::is_bonsai27b_geometry(
        64, 5_120, 17_408, 6_144, 1_024
    ));
    assert!(!super::is_bonsai27b_geometry(
        63, 5_120, 17_408, 6_144, 1_024
    ));
    assert!(!super::is_bonsai27b_geometry(
        64, 5_120, 17_409, 6_144, 1_024
    ));
}

#[test]
fn prism_q1_model_policy_keeps_opt_ins_and_negative_escapes_ordered() {
    let policy = |fast, windows, artifact, exact, q1_on, q1_off, fused_on, fused_off| {
        super::prism_q1_model_policy(
            fast, windows, artifact, exact, q1_on, q1_off, fused_on, fused_off,
        )
    };
    let prism = super::ResidentCudaArtifact::PrismBonsai27bQ1;
    let generic = super::ResidentCudaArtifact::Generic;

    // Q1T128 remains available to other shapes and to the strict f32 reader as
    // an explicit bring-up lane. Fusion still requires exact artifact identity.
    assert_eq!(
        policy(false, false, generic, false, true, false, true, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: true,
            fused_projections: false,
        }
    );
    assert_eq!(
        policy(true, false, generic, true, true, false, true, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: true,
            fused_projections: false,
        }
    );
    assert_eq!(
        policy(true, false, prism, true, true, false, true, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: true,
            fused_projections: true,
        }
    );

    // A negative layout escape disables both layout and its consumers even if
    // every positive/default source is also present. Fusion has its own escape.
    assert_eq!(
        policy(true, true, prism, true, true, true, true, false),
        super::PrismQ1ModelPolicy {
            q1_tiled: false,
            fused_projections: false,
        }
    );
    assert_eq!(
        policy(true, true, prism, true, false, false, true, true),
        super::PrismQ1ModelPolicy {
            q1_tiled: true,
            fused_projections: false,
        }
    );
}

#[test]
fn prism_cuda_fast_policy_is_per_construction_not_process_cached() {
    assert!(super::prism_cuda_fast_policy(None));
    assert!(super::prism_cuda_fast_policy(Some("0")));
    assert!(!super::prism_cuda_fast_policy(Some("1")));
    assert!(!super::prism_cuda_fast_policy(Some("true")));
    assert!(!super::prism_cuda_fast_policy(Some("on")));
    assert!(!super::prism_cuda_fast_policy(Some("yes")));

    // Sequential evaluations intentionally disagree. Engine construction reads
    // this pure policy once, so model reloads cannot inherit a OnceLock decision.
    assert_ne!(
        super::prism_cuda_fast_policy(None),
        super::prism_cuda_fast_policy(Some("1"))
    );
}

#[test]
fn prism_cuda_popc_policy_is_exact_artifact_sm86_and_escape_safe() {
    let prism = super::ResidentCudaArtifact::PrismBonsai27bQ1;
    let generic = super::ResidentCudaArtifact::Generic;
    let allow = |fast, windows, artifact, exact, tiled, sm86, off| {
        super::prism_cuda_popc_policy(fast, windows, artifact, exact, tiled, sm86, off)
    };
    assert!(allow(true, true, prism, true, true, true, false));
    assert!(!allow(false, true, prism, true, true, true, false));
    assert!(!allow(true, false, prism, true, true, true, false));
    assert!(!allow(true, true, generic, true, true, true, false));
    assert!(!allow(true, true, prism, false, true, true, false));
    assert!(!allow(true, true, prism, true, false, true, false));
    assert!(!allow(true, true, prism, true, true, false, false));
    assert!(!allow(true, true, prism, true, true, true, true));
}

#[test]
fn resident_cuda_artifact_identity_is_sha_pinned() {
    assert_eq!(
        super::resident_cuda_artifact_from_sha256(
            "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
        ),
        super::ResidentCudaArtifact::PrismBonsai27bQ1
    );
    assert_eq!(
        super::resident_cuda_artifact_from_sha256(
            "868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757"
        ),
        super::ResidentCudaArtifact::Generic
    );
    assert_eq!(
        super::resident_cuda_artifact_from_sha256("not-a-promoted-artifact"),
        super::ResidentCudaArtifact::Generic
    );
}

#[test]
#[ignore = "requires CUDA; exact fused POPC versus production fused DP4A"]
fn prism_q1t_q8_popc_fused_ffn_matches_and_times_production() {
    let Some(kernels) = kernels() else {
        return;
    };
    const COLS: usize = 5_120;
    const CHUNKS: usize = COLS / 32;
    const ROWS: usize = 17_408;
    let edge = [-128i8, -127, -64, -1, 0, 1, 63, 126, 127];
    let quants = (0..COLS)
        .map(|index| edge[(index * 11 + index / 32) % edge.len()])
        .collect::<Vec<_>>();
    let scales = (0..CHUNKS)
        .map(|index| 0.000_75 + (index % 31) as f32 * 0.000_11)
        .collect::<Vec<_>>();
    let mut expected_bits = vec![0u32; CHUNKS * 8];
    let mut expected_qsums = vec![0i32; CHUNKS];
    for (chunk, values) in quants.chunks_exact(32).enumerate() {
        expected_qsums[chunk] = values.iter().map(|&q| i32::from(q)).sum();
        for (lane, &q) in values.iter().enumerate() {
            for plane in 0..8 {
                expected_bits[chunk * 8 + plane] |= u32::from(((q as u8) >> plane) & 1) << lane;
            }
        }
    }
    let d_quants = kernels.stream.clone_htod(&quants).unwrap();
    let d_scales = kernels.stream.clone_htod(&scales).unwrap();
    let mut d_bits = kernels
        .stream
        .alloc_zeros::<u32>(expected_bits.len())
        .unwrap();
    let mut d_qsums = kernels.stream.alloc_zeros::<i32>(CHUNKS).unwrap();
    super::launch_prism_q8_32_bitplanes_qsum(
        &kernels.stream,
        &kernels.prism_q8_32_bitplanes_qsum,
        &d_quants,
        &mut d_bits,
        &mut d_qsums,
        CHUNKS,
    )
    .unwrap();
    let mut got_bits = vec![0u32; expected_bits.len()];
    let mut got_qsums = vec![0i32; CHUNKS];
    kernels.stream.memcpy_dtoh(&d_bits, &mut got_bits).unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_qsums, &mut got_qsums)
        .unwrap();
    kernels.ctx.synchronize().unwrap();
    assert_eq!(got_bits, expected_bits);
    assert_eq!(got_qsums, expected_qsums);

    // Tail/residual parity for the reusable M16 primitive.
    for &rows in &[48usize, 131] {
        let blocks = COLS / 128;
        let mut wire = vec![0u8; rows * blocks * 18];
        fill_q1_wire(&mut wire, 0x73);
        let tiled = super::repack_q1_t128(&wire, rows, COLS).unwrap();
        let d_weight = kernels.stream.clone_htod(&tiled).unwrap();
        for residual in [0i32, 1] {
            let initial = (0..rows)
                .map(|index| (index % 37) as f32 * 0.001_25 - 0.02)
                .collect::<Vec<_>>();
            let mut d_dp4a = kernels.stream.clone_htod(&initial).unwrap();
            let mut d_popc = kernels.stream.clone_htod(&initial).unwrap();
            super::launch_prism_q1_q8_gemv(
                &kernels.stream,
                &kernels.prism_q1t128_q8_gemv,
                &d_quants,
                &d_scales,
                &d_weight.slice(0..d_weight.len()),
                rows,
                COLS,
                &mut d_dp4a,
                residual,
            )
            .unwrap();
            super::launch_prism_q1t128_q8_popc_gemv_m16(
                &kernels.stream,
                &kernels.prism_q1t128_q8_popc_gemv_m16,
                &d_bits,
                &d_qsums,
                &d_scales,
                &d_weight.slice(0..d_weight.len()),
                rows,
                COLS,
                &mut d_popc,
                residual,
            )
            .unwrap();
            let mut dp4a = vec![0.0f32; rows];
            let mut popc = vec![0.0f32; rows];
            kernels.stream.memcpy_dtoh(&d_dp4a, &mut dp4a).unwrap();
            kernels.stream.memcpy_dtoh(&d_popc, &mut popc).unwrap();
            kernels.ctx.synchronize().unwrap();
            assert_same_bits(
                &format!("Q1T POPC M16 rows={rows} residual={residual}"),
                &popc,
                &dp4a,
            );
        }
    }

    let blocks = COLS / 128;
    let mut gate_wire = vec![0u8; ROWS * blocks * 18];
    let mut up_wire = vec![0u8; ROWS * blocks * 18];
    fill_q1_wire(&mut gate_wire, 0x25);
    fill_q1_wire(&mut up_wire, 0xb7);
    let gate_tiled = super::repack_q1_t128(&gate_wire, ROWS, COLS).unwrap();
    let up_tiled = super::repack_q1_t128(&up_wire, ROWS, COLS).unwrap();
    let d_gate_weight = kernels.stream.clone_htod(&gate_tiled).unwrap();
    let d_up_weight = kernels.stream.clone_htod(&up_tiled).unwrap();
    let mut d_gate_dp4a = kernels.stream.alloc_zeros::<f32>(ROWS).unwrap();
    let mut d_up_dp4a = kernels.stream.alloc_zeros::<f32>(ROWS).unwrap();
    let mut d_gate_popc = kernels.stream.alloc_zeros::<f32>(ROWS).unwrap();
    let mut d_up_popc = kernels.stream.alloc_zeros::<f32>(ROWS).unwrap();

    macro_rules! pack {
        () => {{
            super::launch_prism_q8_32_bitplanes_qsum(
                &kernels.stream,
                &kernels.prism_q8_32_bitplanes_qsum,
                &d_quants,
                &mut d_bits,
                &mut d_qsums,
                CHUNKS,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    macro_rules! popc_fused {
        () => {{
            super::launch_prism_q1t128_q8_popc_fused_ffn_bonsai27b(
                &kernels.stream,
                &kernels.prism_q1t128_q8_popc_fused_ffn_bonsai27b,
                &d_bits,
                &d_qsums,
                &d_scales,
                &d_gate_weight.slice(0..d_gate_weight.len()),
                &d_up_weight.slice(0..d_up_weight.len()),
                &mut d_gate_popc,
                &mut d_up_popc,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    macro_rules! dp4a_fused {
        () => {{
            super::launch_prism_q1t128_fused_ffn_bonsai27b(
                &kernels.stream,
                &kernels.prism_q1t128_fused_ffn_bonsai27b,
                &d_quants,
                &d_scales,
                &d_gate_weight.slice(0..d_gate_weight.len()),
                &d_up_weight.slice(0..d_up_weight.len()),
                &mut d_gate_dp4a,
                &mut d_up_dp4a,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    pack!();
    popc_fused!();
    dp4a_fused!();
    let mut pack_ms = Vec::new();
    let mut popc_ms = Vec::new();
    let mut dp4a_ms = Vec::new();
    for _ in 0..11 {
        let started = std::time::Instant::now();
        pack!();
        pack_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        let started = std::time::Instant::now();
        popc_fused!();
        popc_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        let started = std::time::Instant::now();
        dp4a_fused!();
        dp4a_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let pack_ms = mean(&pack_ms);
    let popc_ms = mean(&popc_ms);
    let dp4a_ms = mean(&dp4a_ms);
    eprintln!(
        "Q1T fused gate/up 17408x5120: POPC pack={pack_ms:.4}ms kernel={popc_ms:.4}ms total={:.4}ms DP4A-fused={dp4a_ms:.4}ms speedup={:.3}x",
        pack_ms + popc_ms,
        dp4a_ms / (pack_ms + popc_ms),
    );
    let mut gate_dp4a = vec![0.0f32; ROWS];
    let mut up_dp4a = vec![0.0f32; ROWS];
    let mut gate_popc = vec![0.0f32; ROWS];
    let mut up_popc = vec![0.0f32; ROWS];
    kernels
        .stream
        .memcpy_dtoh(&d_gate_dp4a, &mut gate_dp4a)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_up_dp4a, &mut up_dp4a)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_gate_popc, &mut gate_popc)
        .unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_up_popc, &mut up_popc)
        .unwrap();
    kernels.ctx.synchronize().unwrap();
    assert_same_bits("Q1T fused POPC gate", &gate_popc, &gate_dp4a);
    assert_same_bits("Q1T fused POPC up", &up_popc, &up_dp4a);
}

#[test]
#[ignore = "requires CUDA; exact single-projection POPC production shape timings"]
fn prism_q1t_q8_popc_single_projection_real_shapes_match_and_gate() {
    let Some(kernels) = kernels() else {
        return;
    };
    for &(label, rows, cols, seed) in &[
        ("O/SSM-out", 5_120usize, 6_144usize, 0x31u8),
        ("FFN-down", 5_120, 17_408, 0x57),
        ("lm-head", 151_936, 5_120, 0x9b),
    ] {
        let chunks = cols / 32;
        let edge = [-128i8, -127, -64, -1, 0, 1, 63, 126, 127];
        let quants = (0..cols)
            .map(|index| edge[(index * 13 + index / 32) % edge.len()])
            .collect::<Vec<_>>();
        let scales = (0..chunks)
            .map(|index| 0.000_6 + (index % 29) as f32 * 0.000_13)
            .collect::<Vec<_>>();
        let d_quants = kernels.stream.clone_htod(&quants).unwrap();
        let d_scales = kernels.stream.clone_htod(&scales).unwrap();
        let mut d_bitplanes = kernels.stream.alloc_zeros::<u32>(chunks * 8).unwrap();
        let mut d_qsums = kernels.stream.alloc_zeros::<i32>(chunks).unwrap();

        let mut wire = vec![0u8; rows * (cols / 128) * 18];
        fill_q1_wire(&mut wire, seed);
        let tiled = super::repack_q1_t128(&wire, rows, cols).unwrap();
        drop(wire);
        let d_weight = kernels.stream.clone_htod(&tiled).unwrap();
        drop(tiled);
        let mut d_dp4a = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
        let mut d_popc = kernels.stream.alloc_zeros::<f32>(rows).unwrap();

        let launch_pack_popc = |d_bitplanes: &mut CudaSlice<u32>,
                                d_qsums: &mut CudaSlice<i32>,
                                d_popc: &mut CudaSlice<f32>| {
            super::launch_prism_q8_32_bitplanes_qsum(
                &kernels.stream,
                &kernels.prism_q8_32_bitplanes_qsum,
                &d_quants,
                d_bitplanes,
                d_qsums,
                chunks,
            )
            .unwrap();
            super::launch_prism_q1t128_q8_popc_gemv_m16(
                &kernels.stream,
                &kernels.prism_q1t128_q8_popc_gemv_m16,
                d_bitplanes,
                d_qsums,
                &d_scales,
                &d_weight.slice(0..d_weight.len()),
                rows,
                cols,
                d_popc,
                0,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        };
        let launch_dp4a = |d_dp4a: &mut CudaSlice<f32>| {
            super::launch_prism_q1_q8_gemv(
                &kernels.stream,
                &kernels.prism_q1t128_q8_gemv,
                &d_quants,
                &d_scales,
                &d_weight.slice(0..d_weight.len()),
                rows,
                cols,
                d_dp4a,
                0,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        };

        launch_pack_popc(&mut d_bitplanes, &mut d_qsums, &mut d_popc);
        launch_dp4a(&mut d_dp4a);
        let mut popc = vec![0.0f32; rows];
        let mut dp4a = vec![0.0f32; rows];
        kernels.stream.memcpy_dtoh(&d_popc, &mut popc).unwrap();
        kernels.stream.memcpy_dtoh(&d_dp4a, &mut dp4a).unwrap();
        kernels.ctx.synchronize().unwrap();
        assert_same_bits(&format!("Q1T POPC {label}"), &popc, &dp4a);

        let mut popc_ms = Vec::new();
        let mut dp4a_ms = Vec::new();
        for _ in 0..11 {
            let started = std::time::Instant::now();
            launch_pack_popc(&mut d_bitplanes, &mut d_qsums, &mut d_popc);
            popc_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            let started = std::time::Instant::now();
            launch_dp4a(&mut d_dp4a);
            dp4a_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let popc_ms = mean(&popc_ms);
        let dp4a_ms = mean(&dp4a_ms);
        eprintln!(
            "Q1T {label} {rows}x{cols}: POPC+pack={popc_ms:.4}ms DP4A={dp4a_ms:.4}ms speedup={:.3}x",
            dp4a_ms / popc_ms,
        );
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device; BMMA parity/performance diagnostic"]
fn prism_q1_q8_b128_bmma_parity_and_real_shape_speed_probe() {
    let Some(kernels) = kernels() else {
        return;
    };
    let Some(pack) = kernels.prism_q8_b128_bitpack.as_ref() else {
        eprintln!("Q8/128 BMMA probe skipped: device is below SM80");
        return;
    };
    let Some(bmma) = kernels.prism_q1_q8_b128_bmma_gemm_batched.as_ref() else {
        eprintln!("Q8/128 BMMA probe skipped: BMMA kernel unavailable");
        return;
    };

    // First validate every seam: non-multiple row/token tails, two K=128
    // blocks, exact packed bits/scales, fragment mapping, and XOR reconstruction.
    let (rows, cols, k_tokens) = (131usize, 256usize, 19usize);
    let blocks = cols / 128;
    let mut rng = Lcg(0xb1_80_12_80);
    let activation = (0..cols * k_tokens)
        .map(|_| rng.next_f32() * 1.75)
        .collect::<Vec<_>>();
    let mut wire = vec![0u8; rows * blocks * 18];
    for block in wire.chunks_exact_mut(18) {
        let scale = 0.0025 + rng.next_f32().abs() * 0.03;
        block[..2].copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
        for byte in &mut block[2..] {
            *byte = rng.next_u8();
        }
    }
    let (expected_bits, expected_scales, expected_quants) =
        quantize_q8_b128_cpu(&activation, cols, k_tokens);
    let expected = q1_q8_b128_cpu(
        &wire,
        &expected_quants,
        &expected_scales,
        rows,
        cols,
        k_tokens,
    );

    let d_activation = kernels.stream.clone_htod(&activation).unwrap();
    let d_wire = kernels.stream.clone_htod(&wire).unwrap();
    let mut d_bits = kernels
        .stream
        .alloc_zeros::<u32>(expected_bits.len())
        .unwrap();
    let mut d_scales = kernels
        .stream
        .alloc_zeros::<f32>(expected_scales.len())
        .unwrap();
    super::launch_prism_q8_b128_bitpack(
        &kernels.stream,
        pack,
        &d_activation,
        &mut d_bits,
        &mut d_scales,
        cols,
        k_tokens,
    )
    .unwrap();
    let mut d_out = kernels.stream.alloc_zeros::<f32>(rows * k_tokens).unwrap();
    super::launch_prism_q1_q8_b128_bmma_gemm_batched(
        &kernels.stream,
        bmma,
        &d_bits,
        &d_scales,
        &d_wire.slice(0..wire.len()),
        rows,
        cols,
        k_tokens,
        false,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got_bits = vec![0u32; expected_bits.len()];
    let mut got_scales = vec![0.0f32; expected_scales.len()];
    let mut got = vec![0.0f32; expected.len()];
    kernels.stream.memcpy_dtoh(&d_bits, &mut got_bits).unwrap();
    kernels
        .stream
        .memcpy_dtoh(&d_scales, &mut got_scales)
        .unwrap();
    kernels.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    kernels.ctx.synchronize().unwrap();
    assert_eq!(
        got_bits, expected_bits,
        "Q8/128 bit-slice pack changed bits"
    );
    assert!(
        got_scales
            .iter()
            .zip(&expected_scales)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Q8/128 bit-slice pack changed f16-rounded scales"
    );
    let (cpu_cosine, cpu_relative_l2, cpu_max_abs) = vector_error(&got, &expected);
    assert!(
        cpu_cosine > 0.999_999 && cpu_relative_l2 < 2e-6,
        "BMMA XOR reconstruction diverged: cosine={cpu_cosine:.9} relative_l2={cpu_relative_l2:.9} max_abs={cpu_max_abs:.9}"
    );

    // The production attention/down projections use the in-kernel residual
    // epilogue. Exercise it independently so enabling BMMA cannot silently
    // overwrite the hidden-state residual.
    let residual = (0..expected.len())
        .map(|index| (index % 37) as f32 * 0.002 - 0.03)
        .collect::<Vec<_>>();
    let mut d_residual = kernels.stream.clone_htod(&residual).unwrap();
    super::launch_prism_q1_q8_b128_bmma_gemm_batched(
        &kernels.stream,
        bmma,
        &d_bits,
        &d_scales,
        &d_wire.slice(0..wire.len()),
        rows,
        cols,
        k_tokens,
        false,
        &mut d_residual,
        1,
    )
    .unwrap();
    let mut got_residual = vec![0.0f32; expected.len()];
    kernels
        .stream
        .memcpy_dtoh(&d_residual, &mut got_residual)
        .unwrap();
    kernels.ctx.synchronize().unwrap();
    let expected_residual = expected
        .iter()
        .zip(&residual)
        .map(|(value, residual)| value + residual)
        .collect::<Vec<_>>();
    let (residual_cosine, residual_relative_l2, _) =
        vector_error(&got_residual, &expected_residual);
    assert!(
        residual_cosine > 0.999_999 && residual_relative_l2 < 2e-6,
        "BMMA residual epilogue diverged: cosine={residual_cosine:.9} relative_l2={residual_relative_l2:.9}"
    );

    // Real Bonsai FFN-up shape. Sweep the image/prompt tail sizes that matter in
    // production and compare against both existing Q8/32 implementations. This
    // is the dispatch-crossover probe, not a toy square GEMM benchmark.
    let (rows, cols, capacity) = (17_408usize, 5_120usize, 128usize);
    let blocks = cols / 128;
    let mut wire = vec![0u8; rows * blocks * 18];
    for (block_index, block) in wire.chunks_exact_mut(18).enumerate() {
        let scale = 0.003 + (block_index % 29) as f32 * 0.0002;
        block[..2].copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
        for (byte_index, value) in block[2..].iter_mut().enumerate() {
            *value = (block_index as u8)
                .wrapping_mul(37)
                .wrapping_add((byte_index as u8).wrapping_mul(19));
        }
    }
    let mut rng = Lcg(0x80_12_80_b1);
    let activation = (0..cols * capacity)
        .map(|_| rng.next_f32() * 2.0)
        .collect::<Vec<_>>();
    let d_wire = kernels.stream.clone_htod(&wire).unwrap();
    let d_activation = kernels.stream.clone_htod(&activation).unwrap();

    let mut d_b128_bits = kernels
        .stream
        .alloc_zeros::<u32>(capacity * cols / 4)
        .unwrap();
    let mut d_b128_scales = kernels
        .stream
        .alloc_zeros::<f32>(capacity * blocks)
        .unwrap();
    let mut d_b128_out = kernels.stream.alloc_zeros::<f32>(capacity * rows).unwrap();
    let mut d_q8_quants = kernels.stream.alloc_zeros::<i8>(capacity * cols).unwrap();
    let mut d_q8_scales = kernels
        .stream
        .alloc_zeros::<f32>(capacity * cols / 32)
        .unwrap();
    let mut d_imma_out = kernels.stream.alloc_zeros::<f32>(capacity * rows).unwrap();
    let mut d_dp4a_out = kernels.stream.alloc_zeros::<f32>(capacity * rows).unwrap();
    let imma = kernels
        .prism_q1_q8_wmma_gemm_batched
        .as_ref()
        .expect("SM80 device should expose the Q8/32 IMMA comparison kernel");

    macro_rules! pack_b128 {
        ($k_tokens:expr) => {{
            super::launch_prism_q8_b128_bitpack(
                &kernels.stream,
                pack,
                &d_activation,
                &mut d_b128_bits,
                &mut d_b128_scales,
                cols,
                $k_tokens,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    macro_rules! run_bmma {
        ($k_tokens:expr) => {{
            super::launch_prism_q1_q8_b128_bmma_gemm_batched(
                &kernels.stream,
                bmma,
                &d_b128_bits,
                &d_b128_scales,
                &d_wire.slice(0..wire.len()),
                rows,
                cols,
                $k_tokens,
                false,
                &mut d_b128_out,
                0,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    macro_rules! pack_q8 {
        ($k_tokens:expr) => {{
            super::launch_quantize(
                &kernels.stream,
                &kernels.quantize,
                &d_activation,
                &mut d_q8_quants,
                &mut d_q8_scales,
                $k_tokens * cols / 32,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    macro_rules! run_imma {
        ($k_tokens:expr) => {{
            super::launch_prism_q1_q8_wmma_gemm_batched(
                &kernels.stream,
                imma,
                &d_q8_quants,
                &d_q8_scales,
                &d_wire.slice(0..wire.len()),
                rows,
                cols,
                $k_tokens,
                false,
                &mut d_imma_out,
                0,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    macro_rules! run_dp4a {
        ($k_tokens:expr) => {{
            super::launch_prism_q1_q8_gemm_batched(
                &kernels.stream,
                &kernels.prism_q1_q8_gemm_batched,
                &d_q8_quants,
                &d_q8_scales,
                &d_wire.slice(0..wire.len()),
                rows,
                cols,
                $k_tokens,
                false,
                &mut d_dp4a_out,
                0,
            )
            .unwrap();
            kernels.ctx.synchronize().unwrap();
        }};
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    for &k_tokens in &[8usize, 14, 16, 32, 64, 128] {
        pack_b128!(k_tokens);
        run_bmma!(k_tokens);
        pack_q8!(k_tokens);
        run_dp4a!(k_tokens);
        run_imma!(k_tokens);

        let mut b128_pack_ms = Vec::new();
        let mut bmma_ms = Vec::new();
        let mut q8_pack_ms = Vec::new();
        let mut dp4a_ms = Vec::new();
        let mut imma_ms = Vec::new();
        for _ in 0..5 {
            let started = std::time::Instant::now();
            pack_b128!(k_tokens);
            b128_pack_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let started = std::time::Instant::now();
            run_bmma!(k_tokens);
            bmma_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let started = std::time::Instant::now();
            pack_q8!(k_tokens);
            q8_pack_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let started = std::time::Instant::now();
            run_dp4a!(k_tokens);
            dp4a_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let started = std::time::Instant::now();
            run_imma!(k_tokens);
            imma_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }

        let mut b128_out = vec![0.0f32; capacity * rows];
        let mut q8_out = vec![0.0f32; capacity * rows];
        kernels
            .stream
            .memcpy_dtoh(&d_b128_out, &mut b128_out)
            .unwrap();
        // DP4A and IMMA implement the same Q8/32 math; use DP4A as the quality
        // reference while timing both independently.
        kernels
            .stream
            .memcpy_dtoh(&d_dp4a_out, &mut q8_out)
            .unwrap();
        kernels.ctx.synchronize().unwrap();
        let live = k_tokens * rows;
        let (cosine, relative_l2, max_abs) = vector_error(&b128_out[..live], &q8_out[..live]);
        let b128_total = mean(&b128_pack_ms) + mean(&bmma_ms);
        let q8_pack = mean(&q8_pack_ms);
        let dp4a_total = q8_pack + mean(&dp4a_ms);
        let imma_total = q8_pack + mean(&imma_ms);
        let baseline_total = dp4a_total.min(imma_total);
        let baseline = if dp4a_total <= imma_total {
            "DP4A"
        } else {
            "IMMA"
        };
        eprintln!(
            "Q1 17408x5120 N{k_tokens}: q8/128-pack={:.3}ms BMMA={:.3}ms total={:.3}ms; q8/32-pack={:.3}ms DP4A={:.3}ms total={:.3}ms IMMA={:.3}ms total={:.3}ms; best={baseline} BMMA-speedup={:.3}x cosine={:.8} rel_l2={:.6} max_abs={:.6}",
            mean(&b128_pack_ms),
            mean(&bmma_ms),
            b128_total,
            q8_pack,
            mean(&dp4a_ms),
            dp4a_total,
            mean(&imma_ms),
            imma_total,
            baseline_total / b128_total,
            cosine,
            relative_l2,
            max_abs,
        );
        assert!(
            cosine > 0.99,
            "Q8/128 quality scout cosine is too low at N={k_tokens}: {cosine}"
        );
        assert!(
            relative_l2 < 0.15,
            "Q8/128 quality scout relative L2 is too high at N={k_tokens}: {relative_l2}"
        );
    }
}

// f16 round-trip matching the engine.
fn f16rt(x: f32) -> f32 {
    crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(x))
}

// Quantize an f32 weight tensor [rows*k] to rows*(k/32) Q8_0 36-byte blocks
// (f32 scale LE + 32 i8 quants), the layout the GPU GEMV reads.
fn quantize_blocks(w: &[f32], k: usize) -> Vec<u8> {
    let n_blocks = w.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 36);
    let _ = k;
    for b in 0..n_blocks {
        let blk = &w[b * 32..b * 32 + 32];
        let max_abs = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let unrounded = max_abs / 127.0;
        let scale = f16rt(unrounded);
        let inv = if unrounded == 0.0 {
            0.0
        } else {
            1.0 / unrounded
        };
        out.extend_from_slice(&scale.to_le_bytes());
        for &x in blk {
            let q = (x * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

#[test]
fn q8_soa_repack_restores_wire_scale_footprint() {
    let weights = [
        -0.75f32, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, -1.0, 0.33, -0.66, 0.1,
        -0.2, 0.3, -0.4, 0.6, -0.8, 0.9, -0.95, 0.42, -0.37, 0.73, -0.81, 0.12, -0.13, 0.14, -0.15,
        0.16, -0.17, 0.18, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.1, -1.2, 0.23, -0.34,
        0.45, -0.56, 0.67, -0.78, 0.89, -1.0, 0.11, -0.22, 0.44, -0.55, 0.77, -0.88, 0.99, -1.01,
        0.31, -0.41, 0.51, -0.61, 0.71, -0.91,
    ];
    let aos = quantize_blocks(&weights, 64);
    let soa = super::repack_q8_soa(&aos);
    assert_eq!(aos.len(), 2 * 36);
    assert_eq!(soa.len(), 2 * 34);
    for b in 0..2 {
        assert_eq!(&soa[b * 32..b * 32 + 32], &aos[b * 36 + 4..b * 36 + 36]);
        let f32_scale = f32::from_le_bytes(aos[b * 36..b * 36 + 4].try_into().expect("scale"));
        assert_eq!(
            &soa[64 + b * 2..64 + b * 2 + 2],
            &crate::inference::f32_to_f16_bits(f32_scale).to_le_bytes()
        );
    }
}

// Quantize an activation row to per-block (scale, quants).
fn quantize_row(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
    let nb = x.len() / 32;
    let mut scales = vec![0f32; nb];
    let mut quants = vec![0i8; x.len()];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let max_abs = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let unrounded = max_abs / 127.0;
        scales[b] = f16rt(unrounded);
        let inv = if unrounded == 0.0 {
            0.0
        } else {
            1.0 / unrounded
        };
        for (j, &xv) in blk.iter().enumerate() {
            quants[b * 32 + j] = (xv * inv).round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    }
    (scales, quants)
}

// CPU reference Q8 matmul: quantized input row dotted against Q8 weight blocks,
// rows outputs. Sequential block accumulation (the CPU engine's order).
fn cpu_q8_dot(in_s: &[f32], in_q: &[i8], wblocks: &[u8], rows: usize, bpr: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows];
    for (r, slot) in out.iter_mut().enumerate() {
        let mut sum = 0f32;
        for b in 0..bpr {
            let blk = (r * bpr + b) * 36;
            let ws = f32::from_le_bytes(wblocks[blk..blk + 4].try_into().unwrap());
            let mut int_sum = 0i32;
            for j in 0..32 {
                let wq = wblocks[blk + 4 + j] as i8;
                int_sum += i32::from(wq) * i32::from(in_q[b * 32 + j]);
            }
            sum += int_sum as f32 * ws * in_s[b];
        }
        *slot = sum;
    }
    out
}

fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(v, wv)| v * scale * wv).collect()
}

fn cpu_rope(vec: &mut [f32], cos: &[f32], sin: &[f32], n_heads: usize, head_dim: usize) {
    let pairs = cos.len();
    for head in 0..n_heads {
        for p in 0..pairs {
            let (c, s) = (cos[p], sin[p]);
            let d0 = head * head_dim + 2 * p;
            let (x0, x1) = (vec[d0], vec[d0 + 1]);
            vec[d0] = x0 * c - x1 * s;
            vec[d0 + 1] = x0 * s + x1 * c;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    q: &[f32],
    ck: &[f32],
    cv: &[f32],
    pos_count: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    max_pos: usize,
    scale: f32,
) -> Vec<f32> {
    let repeats = n_heads / n_kv;
    let mut out = vec![0f32; n_heads * head_dim];
    for head in 0..n_heads {
        let kv = head / repeats;
        let qh = &q[head * head_dim..head * head_dim + head_dim];
        let mut scores = vec![0f32; pos_count];
        for (p, sc) in scores.iter_mut().enumerate() {
            let base = (kv * max_pos + p) * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += qh[d] * ck[base + d];
            }
            *sc = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for d in 0..head_dim {
            let mut acc = 0f32;
            for (p, sc) in scores.iter().enumerate() {
                acc += (sc * inv) * cv[(kv * max_pos + p) * head_dim + d];
            }
            out[head * head_dim + d] = acc;
        }
    }
    out
}

#[test]
#[ignore = "requires a CUDA device"]
fn full_forward_token_matches_cpu() {
    let Some(_k) = kernels() else {
        return;
    };
    // Tiny Llama-shaped model.
    let n_layers = 2usize;
    let hidden = 64usize;
    let n_heads = 2usize;
    let n_kv = 1usize;
    let head_dim = 32usize;
    let rope_dim = 32usize;
    let ffn = 128usize;
    let vocab = 96usize;
    let max_pos = 16usize;
    let eps = 1e-5f32;
    let base = 10000f32;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut rng = Lcg(0xabcdef);
    let rand = |rng: &mut Lcg, n: usize| (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();

    // Per-layer f32 weights, quantized to blocks (the same blocks feed CPU + GPU).
    struct LayerF {
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        o: Vec<u8>,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
        an: Vec<f32>,
        fnv: Vec<f32>,
    }
    let mut layers = Vec::new();
    for _ in 0..n_layers {
        layers.push(LayerF {
            q: quantize_blocks(&rand(&mut rng, q_width * hidden), hidden),
            k: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            v: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            o: quantize_blocks(&rand(&mut rng, hidden * q_width), q_width),
            gate: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            up: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            down: quantize_blocks(&rand(&mut rng, hidden * ffn), ffn),
            an: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
            fnv: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
        });
    }
    let final_norm: Vec<f32> = rand(&mut rng, hidden)
        .iter()
        .map(|v| v * 0.2 + 1.0)
        .collect();
    let output_w = quantize_blocks(&rand(&mut rng, vocab * hidden), hidden);

    // Build the GPU engine.
    let mut engine = CudaResidentDecode::new(
        n_layers, n_heads, n_kv, head_dim, hidden, ffn, rope_dim, max_pos, vocab, eps, false,
    )
    .unwrap();
    for l in &layers {
        engine
            .set_layer(
                &l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down, &l.an, &l.fnv,
            )
            .unwrap();
    }
    engine
        .set_output(&final_norm, &output_w, ProjQuant::Q8_0)
        .unwrap();

    // CPU reference KV cache, layout [kv_head][position][head_dim] per layer.
    let mut cpu_k = vec![vec![0f32; kv_width * max_pos]; n_layers];
    let mut cpu_v = vec![vec![0f32; kv_width * max_pos]; n_layers];

    let pairs = rope_dim / 2;
    for position in 0..4usize {
        let emb = rand(&mut rng, hidden);
        let cos: Vec<f32> = (0..pairs)
            .map(|p| {
                let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
                (position as f32 * theta).cos()
            })
            .collect();
        let sin: Vec<f32> = (0..pairs)
            .map(|p| {
                let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
                (position as f32 * theta).sin()
            })
            .collect();

        // CPU reference forward
        let mut hidden_v = emb.clone();
        for (li, l) in layers.iter().enumerate() {
            let normed = cpu_rmsnorm(&hidden_v, &l.an, eps);
            let (is, iq) = quantize_row(&normed);
            let mut q = cpu_q8_dot(&is, &iq, &l.q, q_width, hidden / 32);
            let mut kv_k = cpu_q8_dot(&is, &iq, &l.k, kv_width, hidden / 32);
            let kv_v = cpu_q8_dot(&is, &iq, &l.v, kv_width, hidden / 32);
            cpu_rope(&mut q, &cos, &sin, n_heads, head_dim);
            cpu_rope(&mut kv_k, &cos, &sin, n_kv, head_dim);
            for kv in 0..n_kv {
                for d in 0..head_dim {
                    cpu_k[li][(kv * max_pos + position) * head_dim + d] =
                        f16rt(kv_k[kv * head_dim + d]);
                    cpu_v[li][(kv * max_pos + position) * head_dim + d] =
                        f16rt(kv_v[kv * head_dim + d]);
                }
            }
            let ctx = cpu_attention(
                &q,
                &cpu_k[li],
                &cpu_v[li],
                position + 1,
                n_heads,
                n_kv,
                head_dim,
                max_pos,
                scale,
            );
            let (cs, cq) = quantize_row(&ctx);
            let o = cpu_q8_dot(&cs, &cq, &l.o, hidden, q_width / 32);
            for i in 0..hidden {
                hidden_v[i] += o[i];
            }
            let fnormed = cpu_rmsnorm(&hidden_v, &l.fnv, eps);
            let (fs, fq) = quantize_row(&fnormed);
            let gate = cpu_q8_dot(&fs, &fq, &l.gate, ffn, hidden / 32);
            let up = cpu_q8_dot(&fs, &fq, &l.up, ffn, hidden / 32);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let (as_, aq) = quantize_row(&act);
            let down = cpu_q8_dot(&as_, &aq, &l.down, hidden, ffn / 32);
            for i in 0..hidden {
                hidden_v[i] += down[i];
            }
        }
        let fnormed = cpu_rmsnorm(&hidden_v, &final_norm, eps);
        let (s, qq) = quantize_row(&fnormed);
        let logits = cpu_q8_dot(&s, &qq, &output_w, vocab, hidden / 32);
        let cpu_tok = logits
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                },
            )
            .0 as u32;

        let gpu_tok = engine
            .forward_token(&emb, &cos, &sin, position, scale, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            gpu_tok, cpu_tok,
            "token mismatch at position {position}: gpu={gpu_tok} cpu={cpu_tok}"
        );
    }
}

// The GPU `prefill` loop (no per-token sync) must build exactly the same KV
// cache as running `forward_token` sequentially per position. The real decode
// seam prefills the first n-1 prompt tokens, then decodes the last token at
// position n-1 — so the token produced at position n-1 must be identical
// whether the earlier KV came from `prefill` or from sequential forwards.
#[test]
#[ignore = "requires a CUDA device"]
fn prefill_then_decode_matches_sequential() {
    let Some(_k) = kernels() else {
        return;
    };
    // Real TinyLlama-shaped dims (GQA n_kv=4, head_dim=64, hidden=2048, ffn=5632,
    // vocab=32000) — the prefill bug is dimension-specific and does not show at
    // toy sizes. Two layers and a short context keep the test fast.
    let n_layers = 3usize;
    let hidden = 2048usize;
    let n_heads = 32usize;
    let n_kv = 4usize;
    let head_dim = 64usize;
    let rope_dim = 64usize;
    let ffn = 5632usize;
    let vocab = 32000usize;
    let max_pos = 64usize;
    let eps = 1e-5f32;
    let base = 10000f32;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut rng = Lcg(0x1234_5678);
    let rand = |rng: &mut Lcg, n: usize| (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();

    // Identical weights feed both engines.
    struct LayerF {
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        o: Vec<u8>,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
        an: Vec<f32>,
        fnv: Vec<f32>,
    }
    let build_engine = |layers: &[LayerF], final_norm: &[f32], output_w: &[u8]| {
        let mut engine = CudaResidentDecode::new(
            n_layers, n_heads, n_kv, head_dim, hidden, ffn, rope_dim, max_pos, vocab, eps, false,
        )
        .unwrap();
        for l in layers {
            engine
                .set_layer(
                    &l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down, &l.an, &l.fnv,
                )
                .unwrap();
        }
        engine
            .set_output(final_norm, output_w, ProjQuant::Q8_0)
            .unwrap();
        engine
    };

    let layers: Vec<LayerF> = (0..n_layers)
        .map(|_| LayerF {
            q: quantize_blocks(&rand(&mut rng, q_width * hidden), hidden),
            k: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            v: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            o: quantize_blocks(&rand(&mut rng, hidden * q_width), q_width),
            gate: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            up: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            down: quantize_blocks(&rand(&mut rng, hidden * ffn), ffn),
            an: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
            fnv: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
        })
        .collect();
    let final_norm: Vec<f32> = rand(&mut rng, hidden)
        .iter()
        .map(|v| v * 0.2 + 1.0)
        .collect();
    let output_w = quantize_blocks(&rand(&mut rng, vocab * hidden), hidden);

    // A short prompt of n tokens (random embeddings) plus per-position RoPE tables.
    let n = 10usize;
    let half = rope_dim / 2;
    let embeddings: Vec<Vec<f32>> = (0..n).map(|_| rand(&mut rng, hidden)).collect();
    let mut cos_all = vec![0f32; n * half];
    let mut sin_all = vec![0f32; n * half];
    for pos in 0..n {
        for p in 0..half {
            let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
            cos_all[pos * half + p] = (pos as f32 * theta).cos();
            sin_all[pos * half + p] = (pos as f32 * theta).sin();
        }
    }

    // Sequential reference: forward every position through forward_token_logits.
    let mut seq = build_engine(&layers, &final_norm, &output_w);
    let mut seq_logits = Vec::new();
    for pos in 0..n {
        seq_logits = seq
            .forward_token_logits(
                &embeddings[pos],
                &cos_all[pos * half..(pos + 1) * half],
                &sin_all[pos * half..(pos + 1) * half],
                pos,
                scale,
            )
            .unwrap();
    }

    // Prefill the first n-1 tokens in one batched loop, then decode the last.
    let mut pre = build_engine(&layers, &final_norm, &output_w);
    let flat_emb: Vec<f32> = embeddings[..n - 1].iter().flatten().copied().collect();
    pre.prefill(
        &flat_emb,
        &cos_all[..(n - 1) * half],
        &sin_all[..(n - 1) * half],
        n - 1,
        scale,
    )
    .unwrap();
    let pre_logits = pre
        .forward_token_logits(
            &embeddings[n - 1],
            &cos_all[(n - 1) * half..n * half],
            &sin_all[(n - 1) * half..n * half],
            n - 1,
            scale,
        )
        .unwrap();

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
            )
            .0
    };
    assert_eq!(
        argmax(&pre_logits),
        argmax(&seq_logits),
        "prefill+decode produced a different token than sequential forwards"
    );
    assert!(
        close(&pre_logits, &seq_logits, 1e-3),
        "prefill logits diverged from sequential logits at position {}",
        n - 1
    );

    // Batched prefill must build the SAME KV (hence the same next-token logits) as the
    // serial prefill — it is the identical math run in MAX_VERIFY_K-token chunks. With
    // n-1 = 9 tokens it exercises full chunks, a short final chunk, and cross-chunk
    // causal attention.
    let mut preb = build_engine(&layers, &final_norm, &output_w);
    preb.prefill_batched(
        &flat_emb,
        &cos_all[..(n - 1) * half],
        &sin_all[..(n - 1) * half],
        n - 1,
        scale,
    )
    .unwrap();
    let preb_logits = preb
        .forward_token_logits(
            &embeddings[n - 1],
            &cos_all[(n - 1) * half..n * half],
            &sin_all[(n - 1) * half..n * half],
            n - 1,
            scale,
        )
        .unwrap();
    assert_eq!(
        argmax(&preb_logits),
        argmax(&seq_logits),
        "batched prefill+decode produced a different token than sequential forwards"
    );
    assert!(
        close(&preb_logits, &pre_logits, 1e-4),
        "batched prefill logits diverged from serial prefill logits"
    );
}

// Deterministic LCG so the tests need no rand dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 56) as u8
    }
}

fn q1t_fixture(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<u8> {
    let mut wire = vec![0u8; rows * (cols / 128) * 18];
    for block in wire.chunks_exact_mut(18) {
        let scale = 0.001 + rng.next_f32().abs() * 0.04;
        block[..2].copy_from_slice(&crate::inference::f32_to_f16_bits(scale).to_le_bytes());
        for byte in &mut block[2..] {
            *byte = rng.next_u8();
        }
    }
    super::repack_q1_t128(&wire, rows, cols).unwrap()
}

/// Bring up the resident kernels, or skip when this machine has no CUDA device.
///
/// The distinction matters. `CudaResidentKernels::new().ok()` collapsed "there is no GPU
/// here" and "the GPU is here and the module failed to build" into the same `None`, and
/// every `#[ignore]`d parity test in this file then returned early and reported **ok**. A
/// bitwise gate that passes in 0.3 s because the kernels never compiled is worse than no
/// gate: it is a green tick over an untested claim, and it took a suspiciously fast test
/// run to notice.
///
/// So: no device -> skip, which is the honest outcome on a CI box without a GPU. Device
/// present but bring-up failed -> panic with the driver's own message, because on a
/// machine that can run these tests, not running them is a failure.
fn kernels() -> Option<CudaResidentKernels> {
    match CudaResidentKernels::new() {
        Ok(kernels) => Some(kernels),
        Err(error) => {
            let ordinal = crate::cuda::selected_device_ordinal();
            match cudarc::driver::CudaContext::new(ordinal) {
                Ok(_) => panic!(
                    "CUDA device {ordinal} is present but the resident kernel module failed to \
                     build, so this test would otherwise have reported a pass without running: \
                     {error}"
                ),
                Err(_) => {
                    eprintln!("skip: no usable CUDA device (ordinal {ordinal}): {error}");
                    None
                }
            }
        }
    }
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        let d = (x - y).abs() / y.abs().max(1.0);
        d < tol
    })
}

fn assert_same_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length changed");
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
    {
        panic!(
            "{label} changed at {index}: actual={actual:?} ({:#010x}) expected={expected:?} ({:#010x})",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn rms_norm_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 2048usize;
    let eps = 1e-5f32;
    let mut rng = Lcg(1);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.5 + 1.0).collect();
    // CPU reference
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = x.iter().zip(&w).map(|(v, wv)| v * scale * wv).collect();
    // GPU
    let dx = k.stream.clone_htod(&x).unwrap();
    let dw = k.stream.clone_htod(&w).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n).unwrap();
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        // Stages the full row in shared for the in-order sum (matches launch_rmsnorm).
        shared_mem_bytes: (n as u32) * 4,
    };
    let n_i = n as i32;
    let mut b = k.stream.launch_builder(&k.rms_norm);
    b.arg(&dx).arg(&dw).arg(&mut dout).arg(&n_i).arg(&eps);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-4), "rms_norm diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn gemm_batched_matches_per_token() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let bpr = 4usize; // K_dim = 128
    let kdim = bpr * 32;
    let ktok = 4usize;
    let mut rng = Lcg(7);
    // Weight [rows*kdim] -> 36-byte Q8 blocks -> SoA layout the kernel reads.
    let w: Vec<f32> = (0..rows * kdim).map(|_| rng.next_f32()).collect();
    let wblocks = quantize_blocks(&w, kdim);
    let wsoa = super::repack_q8_soa(&wblocks);
    // K inputs laid out [token][block]; CPU reference per token.
    let mut in_s = vec![0f32; ktok * bpr];
    let mut in_q = vec![0i8; ktok * kdim];
    let mut cpu_out = vec![0f32; ktok * rows];
    for t in 0..ktok {
        let x: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let (s, q) = quantize_row(&x);
        in_s[t * bpr..(t + 1) * bpr].copy_from_slice(&s);
        in_q[t * kdim..(t + 1) * kdim].copy_from_slice(&q);
        let r = cpu_q8_dot(&s, &q, &wblocks, rows, bpr);
        cpu_out[t * rows..(t + 1) * rows].copy_from_slice(&r);
    }
    let d_is = k.stream.clone_htod(&in_s).unwrap();
    let d_iq = k.stream.clone_htod(&in_q).unwrap();
    let d_w = k.stream.clone_htod(&wsoa).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(ktok * rows).unwrap();
    super::launch_gemm_batched(
        &k.stream,
        &k.gemm_batched,
        &d_is,
        &d_iq,
        &d_w,
        rows,
        bpr,
        ktok,
        &mut d_out,
    )
    .unwrap();
    let mut got = vec![0f32; ktok * rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    // Tree reduction vs the CPU's sequential block sum -> close, not bit-exact.
    assert!(close(&got, &cpu_out, 1e-3), "batched gemm diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn verify_batch_matches_sequential() {
    if kernels().is_none() {
        return;
    }
    let n_layers = 2usize;
    let hidden = 64usize;
    let n_heads = 2usize;
    let n_kv = 1usize;
    let head_dim = 32usize;
    let rope_dim = 32usize;
    let ffn = 128usize;
    let vocab = 96usize;
    let max_pos = 16usize;
    let eps = 1e-5f32;
    let base = 10000f32;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut rng = Lcg(0x13579);
    let rand = |rng: &mut Lcg, n: usize| (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();
    struct L {
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        o: Vec<u8>,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
        an: Vec<f32>,
        fnv: Vec<f32>,
    }
    let layers: Vec<L> = (0..n_layers)
        .map(|_| L {
            q: quantize_blocks(&rand(&mut rng, q_width * hidden), hidden),
            k: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            v: quantize_blocks(&rand(&mut rng, kv_width * hidden), hidden),
            o: quantize_blocks(&rand(&mut rng, hidden * q_width), q_width),
            gate: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            up: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
            down: quantize_blocks(&rand(&mut rng, hidden * ffn), ffn),
            an: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
            fnv: rand(&mut rng, hidden)
                .iter()
                .map(|v| v * 0.2 + 1.0)
                .collect(),
        })
        .collect();
    let final_norm: Vec<f32> = rand(&mut rng, hidden)
        .iter()
        .map(|v| v * 0.2 + 1.0)
        .collect();
    let output_w = quantize_blocks(&rand(&mut rng, vocab * hidden), hidden);
    let build = || {
        let mut e = CudaResidentDecode::new(
            n_layers, n_heads, n_kv, head_dim, hidden, ffn, rope_dim, max_pos, vocab, eps, false,
        )
        .unwrap();
        for l in &layers {
            e.set_layer(
                &l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down, &l.an, &l.fnv,
            )
            .unwrap();
        }
        e.set_output(&final_norm, &output_w, ProjQuant::Q8_0)
            .unwrap();
        e
    };
    let ktok = 4usize;
    let pairs = rope_dim / 2;
    let mut embs = vec![0f32; ktok * hidden];
    let mut cos_all = vec![0f32; ktok * pairs];
    let mut sin_all = vec![0f32; ktok * pairs];
    let mut per_emb = Vec::new();
    for t in 0..ktok {
        let emb = rand(&mut rng, hidden);
        embs[t * hidden..(t + 1) * hidden].copy_from_slice(&emb);
        per_emb.push(emb);
        for p in 0..pairs {
            let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
            cos_all[t * pairs + p] = (t as f32 * theta).cos();
            sin_all[t * pairs + p] = (t as f32 * theta).sin();
        }
    }
    // Sequential forward_token at positions 0..ktok (the proven single-token path).
    let mut seq = build();
    let mut expected = Vec::new();
    for t in 0..ktok {
        let cos = &cos_all[t * pairs..(t + 1) * pairs];
        let sin = &sin_all[t * pairs..(t + 1) * pairs];
        let tok = seq
            .forward_token(&per_emb[t], cos, sin, t, scale, true)
            .unwrap()
            .unwrap();
        expected.push(tok);
    }
    // Batched verify over the same K tokens.
    let mut bat = build();
    let got = bat
        .verify_batch(&embs, &cos_all, &sin_all, 0, ktok, scale)
        .unwrap();
    assert_eq!(
        got, expected,
        "verify_batch tokens != sequential forward_token"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn quantize_q8_0_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_blocks = 64usize;
    let n = n_blocks * 32;
    let mut rng = Lcg(7);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
    // CPU reference (quantize_q8_0_block): f16-rounded scale, unrounded inverse,
    // round-half-to-even, clamp [-128,127].
    let mut exp_scales = vec![0f32; n_blocks];
    let mut exp_quants = vec![0i8; n];
    for bidx in 0..n_blocks {
        let blk = &x[bidx * 32..bidx * 32 + 32];
        let max_abs = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let unrounded = max_abs / 127.0;
        exp_scales[bidx] =
            crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(unrounded));
        let inv = if unrounded == 0.0 {
            0.0
        } else {
            1.0 / unrounded
        };
        for j in 0..32 {
            let v = (blk[j] * inv).round_ties_even().clamp(-128.0, 127.0);
            exp_quants[bidx * 32 + j] = v as i8;
        }
    }
    // GPU
    let dx = k.stream.clone_htod(&x).unwrap();
    let mut dq = k.stream.alloc_zeros::<i8>(n).unwrap();
    let mut ds = k.stream.alloc_zeros::<f32>(n_blocks).unwrap();
    let block = 64u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_blocks as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nb_i = n_blocks as i32;
    let mut b = k.stream.launch_builder(&k.quantize);
    b.arg(&dx).arg(&mut dq).arg(&mut ds).arg(&nb_i);
    unsafe { b.launch(cfg).unwrap() };
    let mut gq = vec![0i8; n];
    let mut gs = vec![0f32; n_blocks];
    k.stream.memcpy_dtoh(&dq, &mut gq).unwrap();
    k.stream.memcpy_dtoh(&ds, &mut gs).unwrap();
    k.ctx.synchronize().unwrap();
    assert_eq!(gq, exp_quants, "quantize quants diverged");
    assert!(close(&gs, &exp_scales, 1e-6), "quantize scales diverged");
}

// The Gemma 4 decode fusion must be an exact replacement for its two source
// launches at the production hidden width. This covers all 88 Q8_0 blocks,
// including exact-half inputs that exercise the generic kernel's rintf rounding.
#[test]
#[ignore = "requires a CUDA device"]
fn rms_norm_quantize_matches_composed_at_gemma4_hidden() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 2816usize;

    let run_case = |label: &str, x: &[f32], weight: &[f32], eps: f32| {
        let d_x = k.stream.clone_htod(x).unwrap();
        let d_weight = k.stream.clone_htod(weight).unwrap();

        let mut d_normed = k.stream.alloc_zeros::<f32>(n).unwrap();
        let mut d_expected_quants = k.stream.alloc_zeros::<i8>(n).unwrap();
        let mut d_expected_scales = k.stream.alloc_zeros::<f32>(n / 32).unwrap();
        super::launch_rmsnorm(
            &k.stream,
            &k.rms_norm,
            &d_x,
            &d_weight,
            &mut d_normed,
            n,
            eps,
        )
        .unwrap();
        super::launch_quantize(
            &k.stream,
            &k.quantize,
            &d_normed,
            &mut d_expected_quants,
            &mut d_expected_scales,
            n / 32,
        )
        .unwrap();

        let mut d_actual_quants = k.stream.alloc_zeros::<i8>(n).unwrap();
        let mut d_actual_scales = k.stream.alloc_zeros::<f32>(n / 32).unwrap();
        super::launch_rmsnorm_quantize(
            &k.stream,
            &k.rms_norm_quantize,
            &d_x,
            &d_weight,
            &mut d_actual_quants,
            &mut d_actual_scales,
            n,
            eps,
        )
        .unwrap();

        let mut expected_quants = vec![0i8; n];
        let mut expected_scales = vec![0f32; n / 32];
        let mut actual_quants = vec![0i8; n];
        let mut actual_scales = vec![0f32; n / 32];
        k.stream
            .memcpy_dtoh(&d_expected_quants, &mut expected_quants)
            .unwrap();
        k.stream
            .memcpy_dtoh(&d_expected_scales, &mut expected_scales)
            .unwrap();
        k.stream
            .memcpy_dtoh(&d_actual_quants, &mut actual_quants)
            .unwrap();
        k.stream
            .memcpy_dtoh(&d_actual_scales, &mut actual_scales)
            .unwrap();
        k.ctx.synchronize().unwrap();

        assert_eq!(
            actual_quants, expected_quants,
            "{label}: fused RMSNorm Q8_0 quant bytes changed"
        );
        assert_same_bits(
            &format!("{label}: fused RMSNorm Q8_0 scales"),
            &actual_scales,
            &expected_scales,
        );
    };

    let mut rng = Lcg(0x4655_5345_445f_5138);
    let x = (0..n).map(|_| rng.next_f32() * 3.0).collect::<Vec<_>>();
    let weight = (0..n)
        .map(|_| 1.0 + rng.next_f32() * 0.25)
        .collect::<Vec<_>>();
    run_case("nontrivial hidden row", &x, &weight, 1e-6);

    // x=1 and eps=0 make the RMS scale exactly one. Every block has maxima
    // +/-127 and exact +/-0.5 values, pinning ties-to-even in all 88 blocks.
    let x = vec![1.0f32; n];
    let mut weight = vec![0.0f32; n];
    for block in weight.chunks_exact_mut(32) {
        block[0] = 127.0;
        block[1] = -127.0;
        block[2] = 0.5;
        block[3] = -0.5;
    }
    run_case("rounding ties across every block", &x, &weight, 0.0);
}

// Gemma's routed-expert input keeps the CPU's sequential RMS reduction/powf
// scalar, but applies the weighted normalization and quantization to d_hidden.
// Compare every resulting scale bit and quant byte with the Windows CPU path.
#[test]
#[ignore = "requires a CUDA device"]
fn rms_inv_norm_quantizers_match_windows_cpu_bytes() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 2816usize; // Gemma 4 hidden size: 88 Q8_0 blocks / 11 Q8_K blocks.

    let run_case = |label: &str, x: &[f32], weight: &[f32], eps: f32| {
        // Deliberately spell this exactly like gemma4_runtime::rms_norm. The
        // scalar is the only host result passed to the new kernels.
        let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let rms_inv = (mss + eps).powf(-0.5);
        let normalized = x
            .iter()
            .zip(weight)
            .map(|(&v, &w)| v * rms_inv * w)
            .collect::<Vec<_>>();
        let expected_q8_0 = crate::inference::quantize_q8_0_blocks(&normalized);
        let expected_q8_0_scales = expected_q8_0
            .iter()
            .map(|block| block.scale)
            .collect::<Vec<_>>();
        let expected_q8_0_quants = expected_q8_0
            .iter()
            .flat_map(|block| block.quants.iter().copied())
            .collect::<Vec<_>>();
        let expected_q8k = crate::inference::quantize_q8_k_blocks(&normalized);
        let expected_q8k_scales = expected_q8k.iter().map(|block| block.d).collect::<Vec<_>>();
        let expected_q8k_quants = expected_q8k
            .iter()
            .flat_map(|block| block.qs.iter().copied())
            .collect::<Vec<_>>();

        let d_x = k.stream.clone_htod(x).unwrap();
        let d_weight = k.stream.clone_htod(weight).unwrap();
        let mut d_q8_0_quants = k.stream.alloc_zeros::<i8>(n).unwrap();
        let mut d_q8_0_scales = k.stream.alloc_zeros::<f32>(n / 32).unwrap();
        super::launch_rms_inv_norm_quantize_q8_0(
            &k.stream,
            &k.rms_inv_norm_quantize_q8_0,
            &d_x,
            &d_weight,
            &mut d_q8_0_quants,
            &mut d_q8_0_scales,
            n,
            rms_inv,
        )
        .unwrap();

        let mut d_q8k_quants = k.stream.alloc_zeros::<i8>(n).unwrap();
        let mut d_q8k_scales = k.stream.alloc_zeros::<f32>(n / 256).unwrap();
        super::launch_rms_inv_norm_quantize_q8k(
            &k.stream,
            &k.rms_inv_norm_quantize_q8k,
            &d_x,
            &d_weight,
            &mut d_q8k_quants,
            &mut d_q8k_scales,
            n,
            rms_inv,
        )
        .unwrap();

        let mut got_q8_0_quants = vec![0i8; n];
        let mut got_q8_0_scales = vec![0f32; n / 32];
        let mut got_q8k_quants = vec![0i8; n];
        let mut got_q8k_scales = vec![0f32; n / 256];
        k.stream
            .memcpy_dtoh(&d_q8_0_quants, &mut got_q8_0_quants)
            .unwrap();
        k.stream
            .memcpy_dtoh(&d_q8_0_scales, &mut got_q8_0_scales)
            .unwrap();
        k.stream
            .memcpy_dtoh(&d_q8k_quants, &mut got_q8k_quants)
            .unwrap();
        k.stream
            .memcpy_dtoh(&d_q8k_scales, &mut got_q8k_scales)
            .unwrap();
        k.ctx.synchronize().unwrap();

        assert_eq!(
            got_q8_0_quants, expected_q8_0_quants,
            "{label}: Q8_0 quant bytes changed"
        );
        assert_same_bits(
            &format!("{label}: Q8_0 scales"),
            &got_q8_0_scales,
            &expected_q8_0_scales,
        );
        assert_eq!(
            got_q8k_quants, expected_q8k_quants,
            "{label}: Q8_K quant bytes changed"
        );
        assert_same_bits(
            &format!("{label}: Q8_K scales"),
            &got_q8k_scales,
            &expected_q8k_scales,
        );
    };

    let mut rng = Lcg(0x524d_5349_4e56_5138);
    let x = (0..n).map(|_| rng.next_f32() * 2.5).collect::<Vec<_>>();
    let weight = (0..n)
        .map(|_| 1.0 + rng.next_f32() * 0.25)
        .collect::<Vec<_>>();
    run_case("nontrivial RMS inverse", &x, &weight, 1e-6);

    // With x=1 and eps=0, rms_inv is exactly 1. The first Q8_0 block's
    // scale is exactly 1, so +/-0.5 must quantize to +/-1 (half away), not 0
    // as rintf would. Equal +/-127 maxima also pin Q8_K's first-max sign.
    let x = vec![1.0f32; n];
    let mut weight = vec![0.0f32; n];
    weight[0] = 127.0;
    weight[1] = -127.0;
    weight[2] = 0.5;
    weight[3] = -0.5;
    let tie_q8_0 = crate::inference::quantize_q8_0_blocks(&weight);
    assert_eq!(tie_q8_0[0].quants[2], 1);
    assert_eq!(tie_q8_0[0].quants[3], -1);
    let tie_q8k = crate::inference::quantize_q8_k_blocks(&weight);
    assert_eq!(tie_q8k[0].d.to_bits(), (-1.0f32).to_bits());
    run_case("round/sign ties", &x, &weight, 0.0);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rope_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 4usize;
    let head_dim = 64usize;
    let rope_dim = 64usize;
    let base = 10000f32;
    let position = 13usize;
    let mut rng = Lcg(3);
    let vec: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.next_f32()).collect();
    // cos/sin tables per pair
    let pairs = rope_dim / 2;
    let cos_t: Vec<f32> = (0..pairs)
        .map(|p| {
            let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
            (position as f32 * theta).cos()
        })
        .collect();
    let sin_t: Vec<f32> = (0..pairs)
        .map(|p| {
            let theta = base.powf(-(2.0 * p as f32) / rope_dim as f32);
            (position as f32 * theta).sin()
        })
        .collect();
    // CPU reference: adjacent-even-odd forward
    let mut expected = vec.clone();
    for head in 0..n_heads {
        for p in 0..pairs {
            let (c, s) = (cos_t[p], sin_t[p]);
            let d0 = head * head_dim + 2 * p;
            let d1 = d0 + 1;
            let x0 = vec[d0];
            let x1 = vec[d1];
            expected[d0] = x0 * c - x1 * s;
            expected[d1] = x0 * s + x1 * c;
        }
    }
    // GPU
    let mut dvec = k.stream.clone_htod(&vec).unwrap();
    let dcos = k.stream.clone_htod(&cos_t).unwrap();
    let dsin = k.stream.clone_htod(&sin_t).unwrap();
    let (nh, hd, rd) = (n_heads as i32, head_dim as i32, rope_dim as i32);
    // pairing=0 → adjacent-even-odd, matching the CPU reference above. (rope_rotate
    // gained this 7th param in e08cffae; this direct-launch test must pass it too —
    // the production `launch_rope` wrapper always does.)
    let pairing = 0i32;
    let total = (n_heads * pairs) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = k.stream.launch_builder(&k.rope);
    b.arg(&mut dvec)
        .arg(&dcos)
        .arg(&dsin)
        .arg(&nh)
        .arg(&hd)
        .arg(&rd)
        .arg(&pairing);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n_heads * head_dim];
    k.stream.memcpy_dtoh(&dvec, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-5), "rope diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn silu_mul_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 5632usize;
    let mut rng = Lcg(5);
    let gate: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
    let up: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let expected: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let dg = k.stream.clone_htod(&gate).unwrap();
    let du = k.stream.clone_htod(&up).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n).unwrap();
    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = k.stream.launch_builder(&k.silu_mul);
    b.arg(&dg).arg(&du).arg(&mut dout).arg(&n_i);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-5), "silu_mul diverged");
}

// Gemma GeGLU parity: out = gelu_pytorch_tanh(gate) * up. The expected values
// replicate inference::gemma4::gelu_tanh exactly (same constants + f32 order);
// tanhf's last-bit transcendental rounding makes this tolerance-, not bit-, exact.
#[test]
#[ignore = "requires a CUDA device"]
fn geglu_mul_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 5632usize;
    let mut rng = Lcg(11);
    let gate: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 8.0).collect();
    let up: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let expected: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            // gelu coefficient sqrt(2/pi), matched to the kernel's literal for parity.
            #[allow(clippy::excessive_precision)]
            let inner = 0.79788456f32 * (g + 0.044715f32 * g * g * g);
            (0.5f32 * g * (1.0f32 + inner.tanh())) * u
        })
        .collect();
    let dg = k.stream.clone_htod(&gate).unwrap();
    let du = k.stream.clone_htod(&up).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n).unwrap();
    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = k.stream.launch_builder(&k.geglu_mul);
    b.arg(&dg).arg(&du).arg(&mut dout).arg(&n_i);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-5), "geglu_mul diverged");
}

// The routed fusion must be bit-identical to the two kernels it replaces:
// `geglu_mul` followed by the serial-reference `quantize_q8_0`. Besides parity,
// nff=704 exercises the production 22-block shape and its six-warp final CTA;
// reordered route IDs verify that outputs land in router-positioned scratch.
#[test]
#[ignore = "requires a CUDA device"]
fn geglu_quantize_routed_matches_composed_kernels_and_tail() {
    let Some(k) = kernels() else {
        return;
    };
    let nff = 704usize;
    let blocks = nff / 32;
    let route_capacity = 4usize;
    let routes = [2i32, 0, 3];
    let experts = routes.len();
    let mut rng = Lcg(0x4745_474c_5551_3830);
    let mut gate_up = (0..route_capacity * 2 * nff)
        .map(|_| rng.next_f32() * 4.0)
        .collect::<Vec<_>>();
    // Pin one all-zero Q8 block as well as the non-zero random blocks.
    for value in &mut gate_up[2 * 2 * nff..2 * 2 * nff + 32] {
        *value = 0.0;
    }

    let d_gate_up = k.stream.clone_htod(&gate_up).unwrap();
    let d_routes = k.stream.clone_htod(&routes).unwrap();
    let mut d_quants = k.stream.alloc_zeros::<i8>(route_capacity * nff).unwrap();
    let mut d_scales = k
        .stream
        .alloc_zeros::<f32>(route_capacity * blocks)
        .unwrap();
    let block = 256u32;
    let warps = block / 32;
    let routed_cfg = LaunchConfig {
        grid_dim: ((blocks as u32).div_ceil(warps), experts as u32, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nff_i = nff as i32;
    let blocks_i = blocks as i32;
    let experts_i = experts as i32;
    let mut launch = k.stream.launch_builder(&k.geglu_quantize_routed);
    launch
        .arg(&d_gate_up)
        .arg(&d_routes)
        .arg(&mut d_quants)
        .arg(&mut d_scales)
        .arg(&nff_i)
        .arg(&blocks_i)
        .arg(&experts_i);
    unsafe { launch.launch(routed_cfg).unwrap() };

    let mut got_quants = vec![0i8; route_capacity * nff];
    let mut got_scales = vec![0f32; route_capacity * blocks];
    k.stream.memcpy_dtoh(&d_quants, &mut got_quants).unwrap();
    k.stream.memcpy_dtoh(&d_scales, &mut got_scales).unwrap();

    for &route in &routes {
        let route = route as usize;
        let base = route * 2 * nff;
        let gate = d_gate_up.slice(base..base + nff);
        let up = d_gate_up.slice(base + nff..base + 2 * nff);
        let mut d_values = k.stream.alloc_zeros::<f32>(nff).unwrap();
        let geglu_cfg = LaunchConfig {
            grid_dim: ((nff as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut geglu = k.stream.launch_builder(&k.geglu_mul);
        geglu.arg(&gate).arg(&up).arg(&mut d_values).arg(&nff_i);
        unsafe { geglu.launch(geglu_cfg).unwrap() };

        let mut d_ref_quants = k.stream.alloc_zeros::<i8>(nff).unwrap();
        let mut d_ref_scales = k.stream.alloc_zeros::<f32>(blocks).unwrap();
        let quant_cfg = LaunchConfig {
            grid_dim: ((blocks as u32).div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut quantize = k.stream.launch_builder(&k.quantize);
        quantize
            .arg(&d_values)
            .arg(&mut d_ref_quants)
            .arg(&mut d_ref_scales)
            .arg(&blocks_i);
        unsafe { quantize.launch(quant_cfg).unwrap() };

        let mut ref_quants = vec![0i8; nff];
        let mut ref_scales = vec![0f32; blocks];
        k.stream
            .memcpy_dtoh(&d_ref_quants, &mut ref_quants)
            .unwrap();
        k.stream
            .memcpy_dtoh(&d_ref_scales, &mut ref_scales)
            .unwrap();
        k.ctx.synchronize().unwrap();
        assert_eq!(
            &got_quants[route * nff..(route + 1) * nff],
            ref_quants,
            "routed GeGLU quants diverged at route {route}"
        );
        assert_same_bits(
            &format!("routed GeGLU scales at route {route}"),
            &got_scales[route * blocks..(route + 1) * blocks],
            &ref_scales,
        );
    }

    // Route 1 was not selected and must remain untouched.
    assert!(got_quants[nff..2 * nff].iter().all(|&q| q == 0));
    assert!(got_scales[blocks..2 * blocks]
        .iter()
        .all(|&scale| scale == 0.0));
}

// Gemma final-logit soft-cap parity: x = cap*tanh(x/cap), cap=30, in place.
// Matches inference::gemma4::soft_cap_in_place (tolerance for tanhf).
#[test]
#[ignore = "requires a CUDA device"]
fn soft_cap_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 4096usize;
    let cap = 30.0f32;
    let mut rng = Lcg(7);
    let x: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 200.0).collect();
    let expected: Vec<f32> = x.iter().map(|&v| cap * (v / cap).tanh()).collect();
    let mut dx = k.stream.clone_htod(&x).unwrap();
    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = k.stream.launch_builder(&k.soft_cap);
    b.arg(&mut dx).arg(&n_i).arg(&cap);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n];
    k.stream.memcpy_dtoh(&dx, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-5), "soft_cap diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn argmax_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 32000usize;
    let mut rng = Lcg(9);
    let mut logits: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    logits[12345] = 5.0; // clear winner
                         // CPU strict-> scan
    let mut best = logits[0];
    let mut besti = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > best {
            best = *v;
            besti = i;
        }
    }
    let dl = k.stream.clone_htod(&logits).unwrap();
    let mut didx = k.stream.alloc_zeros::<u32>(1).unwrap();
    let block = 256u32;
    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 8, // f32 val + i32 idx per thread
    };
    let mut b = k.stream.launch_builder(&k.argmax);
    b.arg(&dl).arg(&n_i).arg(&mut didx);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0u32; 1];
    k.stream.memcpy_dtoh(&didx, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert_eq!(got[0] as usize, besti, "argmax diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn gumbel_sampler_matches_stateless_reference() {
    let Some(k) = kernels() else {
        return;
    };
    // With identical logits Gumbel-max selects the largest uniform draw. Compare
    // the exact 24-bit SplitMix64 value instead of CPU/GPU log implementations,
    // which proves the RNG, seed plumbing, reduction, and tie-break independently
    // of libm approximation differences.
    let n = 4096usize;
    let logits = vec![0.0f32; n];
    let dl = k.stream.clone_htod(&logits).unwrap();
    let mut didx = k.stream.alloc_zeros::<u32>(1).unwrap();
    let mut sampled = Vec::new();
    for seed in [0u64, 1, 7, 42, u32::MAX as u64, u64::MAX] {
        let mut best_uniform = 0u32;
        let mut expected = 0usize;
        for idx in 0..n {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(idx as u64 + 1));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let uniform = (z >> 40) as u32;
            if uniform > best_uniform {
                best_uniform = uniform;
                expected = idx;
            }
        }
        super::launch_sample_gumbel(&k.stream, &k.sample_gumbel, &dl, n, 1.0, seed, &mut didx)
            .unwrap();
        let mut got = [0u32; 1];
        k.stream.memcpy_dtoh(&didx, &mut got).unwrap();
        k.ctx.synchronize().unwrap();
        assert_eq!(got[0] as usize, expected, "seed {seed}");
        sampled.push(got[0]);
    }
    sampled.sort_unstable();
    sampled.dedup();
    assert!(
        sampled.len() > 1,
        "different seeds should not collapse to one sampled token"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn attention_decode_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 32usize;
    let n_kv = 4usize;
    let head_dim = 64usize;
    let max_pos = 128usize;
    let position_count = 40usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let repeats = n_heads / n_kv;
    let mut rng = Lcg(11);
    let q: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.next_f32()).collect();
    let mut cache_k = vec![0f32; n_kv * max_pos * head_dim];
    let mut cache_v = vec![0f32; n_kv * max_pos * head_dim];
    for kv in 0..n_kv {
        for p in 0..position_count {
            for d in 0..head_dim {
                cache_k[(kv * max_pos + p) * head_dim + d] = rng.next_f32();
                cache_v[(kv * max_pos + p) * head_dim + d] = rng.next_f32();
            }
        }
    }
    // The GPU KV cache stores f16 bits, so round the reference K/V through f16 (the real path
    // does this in kv_scatter) and upload the bits — then GPU and CPU read identical values.
    for x in cache_k.iter_mut() {
        *x = crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(*x));
    }
    for x in cache_v.iter_mut() {
        *x = crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(*x));
    }
    let cache_k_bits: Vec<u16> = cache_k
        .iter()
        .map(|&x| crate::inference::f32_to_f16_bits(x))
        .collect();
    let cache_v_bits: Vec<u16> = cache_v
        .iter()
        .map(|&x| crate::inference::f32_to_f16_bits(x))
        .collect();
    // CPU reference
    let mut expected = vec![0f32; n_heads * head_dim];
    for head in 0..n_heads {
        let kv_head = head / repeats;
        let qh = &q[head * head_dim..head * head_dim + head_dim];
        let mut scores = vec![0f32; position_count];
        for (p, score) in scores.iter_mut().enumerate() {
            let kbase = (kv_head * max_pos + p) * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += qh[d] * cache_k[kbase + d];
            }
            *score = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for d in 0..head_dim {
            let mut acc = 0f32;
            for p in 0..position_count {
                acc += (scores[p] * inv) * cache_v[(kv_head * max_pos + p) * head_dim + d];
            }
            expected[head * head_dim + d] = acc;
        }
    }
    // GPU
    let dq = k.stream.clone_htod(&q).unwrap();
    let cache_k_bytes: Vec<u8> = cache_k_bits
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let cache_v_bytes: Vec<u8> = cache_v_bits
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let dk = k.stream.clone_htod(&cache_k_bytes).unwrap();
    let dv = k.stream.clone_htod(&cache_v_bytes).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n_heads * head_dim).unwrap();
    let mut dscores = k.stream.alloc_zeros::<f32>(n_heads * max_pos).unwrap();
    let (nh, nkv, hd, mp) = (n_heads as i32, n_kv as i32, head_dim as i32, max_pos as i32);
    // The kernel reads position from device memory and uses position_count = pos+1.
    let dpos = k.stream.clone_htod(&[(position_count - 1) as i32]).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: (2 * head_dim * 4) as u32,
    };
    let mut b = k.stream.launch_builder(&k.attention);
    b.arg(&dq)
        .arg(&dk)
        .arg(&dv)
        .arg(&mut dout)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(&dpos)
        .arg(&mp)
        .arg(&scale)
        .arg(&mut dscores);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n_heads * head_dim];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-4), "attention diverged");
}

// GATE OF RECORD for the split-K spec-verify parity fix: the spec-verify kernels
// (attention_batched / attention_tree_batched, splitk_active=1) must be BYTE-IDENTICAL to
// whatever plain greedy decode dispatches at that position_count -- split-K above
// SPLITK_THRESHOLD, G-group at/below it. Deterministic (asserts on the u32 bit-casts, no
// epsilon and no near-tie luck): FAILS pre-fix (G-group != split-K for every pc > 512) and
// passes post-fix. Sweeps n_splits steps and both clamp edges. Linear tree only (count==pc,
// slots[i]==i): the committed path is the only one held to a decode reference.
#[test]
#[ignore = "requires a CUDA device"]
fn splitk_spec_verify_bit_identical() {
    let Some(k) = kernels() else {
        return;
    };
    if k.attn_coalesced {
        eprintln!(
            "skip splitk_spec_verify_bit_identical: CAMELID_ATTN_COALESCED re-associates the \
             split-K per-position dot, which this emulation does not reproduce; the >512 lossless \
             guarantee is scoped to the default non-coalesced path."
        );
        return;
    }
    let n_heads = 8usize;
    let n_kv = 2usize;
    let head_dim = 128usize;
    // SIROCCO Lane K: raised 4096 -> 5632 so the sweep can cross the old SPLITK_MAX=16 cap and
    // exercise the new 17..=22 split regime. The ceiling is the TREE emulation, not the real path:
    // attention_tree_batched holds all scores+slots in dynamic shared = (head_dim + 2*(pc-1+k) +
    // max_groups*head_dim)*4 bytes, which crosses the 48KB opt-out limit at pc>=5569 (n_splits 23+).
    // The real split-K decode (launch_attention_splitk) streams positions and has no such limit --
    // it runs to 32k. Split counts 23..=32 and the ceil>32->32 clamp are code-identical by
    // construction (n_splits only bounds the grid.y/reduction loop; no value-dependent branch) and
    // are covered empirically by the end-to-end greedy-token-parity check at ctx>=6602 / >=8193.
    let max_pos = 5632usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut rng = Lcg(20240626);
    let q: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.next_f32()).collect();
    let mut cache_k = vec![0f32; n_kv * max_pos * head_dim];
    let mut cache_v = vec![0f32; n_kv * max_pos * head_dim];
    for x in cache_k.iter_mut() {
        *x = rng.next_f32();
    }
    for x in cache_v.iter_mut() {
        *x = rng.next_f32();
    }
    let cache_k_bits: Vec<u16> = cache_k
        .iter()
        .map(|&x| crate::inference::f32_to_f16_bits(x))
        .collect();
    let cache_v_bits: Vec<u16> = cache_v
        .iter()
        .map(|&x| crate::inference::f32_to_f16_bits(x))
        .collect();
    let dq = k.stream.clone_htod(&q).unwrap();
    let cache_k_bytes: Vec<u8> = cache_k_bits
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let cache_v_bytes: Vec<u8> = cache_v_bits
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let dk = k.stream.clone_htod(&cache_k_bytes).unwrap();
    let dv = k.stream.clone_htod(&cache_v_bytes).unwrap();

    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    let outlen = n_heads * head_dim;
    // 512/513: strict-`>` boundary. 768/769, 1024: n_splits = ceil(pc/256) steps. 3840/3841: the
    // OLD ceil==15/16 step. 4096/4097: crosses the OLD SPLITK_MAX=16 cap (4097 -> ceil 17, first
    // split count the old build could never reach). 4864 -> 19, 5504 -> 22: the new 17..=22 regime,
    // topping out just under the tree emulation's 48KB shared ceiling (pc<=5568). Higher split
    // counts are argued by construction + the end-to-end token-parity gate (see max_pos note).
    let sweep = [
        512usize, 513, 768, 769, 1024, 2000, 3840, 3841, 4096, 4097, 4864, 5504,
    ];

    for &pc in &sweep {
        let dpos = k.stream.clone_htod(&[(pc - 1) as i32]).unwrap();
        // Reference = exactly what plain decode dispatches at this position_count (the
        // `!graph_capture && attn_shared > SPLITK_THRESHOLD` branch in forward_pass).
        let mut dref = k.stream.alloc_zeros::<f32>(outlen).unwrap();
        let mut d_scores = k.stream.alloc_zeros::<f32>(n_heads * max_pos).unwrap();
        if pc > super::SPLITK_THRESHOLD {
            let mut sc = k.stream.alloc_zeros::<f32>(n_heads * max_pos).unwrap();
            let mut cm = k
                .stream
                .alloc_zeros::<f32>(n_heads * super::SPLITK_MAX)
                .unwrap();
            let mut ls = k
                .stream
                .alloc_zeros::<f32>(n_heads * super::SPLITK_MAX)
                .unwrap();
            let mut ac = k
                .stream
                .alloc_zeros::<f32>(n_heads * super::SPLITK_MAX * head_dim)
                .unwrap();
            super::launch_attention_splitk(
                &k.stream, &k, &dq, &dk, &dv, &mut dref, &mut sc, &mut cm, &mut ls, &mut ac,
                n_heads, n_kv, head_dim, &dpos, pc, max_pos, scale, false,
            )
            .unwrap();
        } else {
            super::launch_attention(
                &k.stream,
                &k.attention,
                &dq,
                &dk,
                &dv,
                &mut dref,
                n_heads,
                n_kv,
                head_dim,
                &dpos,
                pc,
                max_pos,
                scale,
                &mut d_scores,
            )
            .unwrap();
        }
        let mut ref_out = vec![0f32; outlen];
        k.stream.memcpy_dtoh(&dref, &mut ref_out).unwrap();

        // Linear verify: attention_batched, single token at absolute position pc-1, splitk_active=1.
        let mut dver = k.stream.alloc_zeros::<f32>(outlen).unwrap();
        let mut d_scores = k.stream.alloc_zeros::<f32>(n_heads * max_pos).unwrap();
        super::launch_attention_batched(
            &k.stream,
            &k.attention_batched,
            &dq,
            &dk,
            &dv,
            &mut dver,
            n_heads,
            n_kv,
            head_dim,
            pc - 1, // base_position => position_count = base + 0 + 1 = pc
            max_pos,
            scale,
            n_heads * head_dim, // q_per_token
            1,                  // k
            1,                  // splitk_active
            &mut d_scores,
        )
        .unwrap();
        let mut ver_out = vec![0f32; outlen];
        k.stream.memcpy_dtoh(&dver, &mut ver_out).unwrap();

        // Tree verify: linear single node attending [0, base) + itself (slot base), so count==pc
        // and slots[i]==i -- bit-identical to the linear/decode path on the committed branch.
        let mut dtree = k.stream.alloc_zeros::<f32>(outlen).unwrap();
        let anc: Vec<u32> = vec![1u32]; // node 0: ancestor bit 0 set => attends slot base+0
        let danc = k.stream.clone_htod(&anc).unwrap();
        super::launch_attention_tree_batched(
            &k.stream,
            &k.attention_tree_batched,
            &dq,
            &dk,
            &dv,
            &mut dtree,
            &danc,
            1, // words
            n_heads,
            n_kv,
            head_dim,
            pc - 1,
            max_pos,
            scale,
            n_heads * head_dim,
            1,
            1,
            &mut d_scores,
        )
        .unwrap();
        let mut tree_out = vec![0f32; outlen];
        k.stream.memcpy_dtoh(&dtree, &mut tree_out).unwrap();

        k.ctx.synchronize().unwrap();
        assert_eq!(
            bits(&ref_out),
            bits(&ver_out),
            "linear verify != plain decode at pc={pc}"
        );
        assert_eq!(
            bits(&ref_out),
            bits(&tree_out),
            "tree verify != plain decode at pc={pc}"
        );
    }
}

// Sliding-window attention parity (gemma4 sliding layers): only the last `window`
// keys are attended. Same setup as attention_decode_matches_cpu but the CPU ref
// masks to [start, position_count) with start = position_count - window. Validates
// the window masking; weighted-V is FP-reassociated so this is tolerance-, not
// bit-, exact (1e-4, same as the full-causal test).
#[test]
#[ignore = "requires a CUDA device"]
fn attention_decode_sw_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 32usize;
    let n_kv = 4usize;
    let head_dim = 64usize;
    let max_pos = 128usize;
    let position_count = 40usize;
    let window = 16usize; // start = 40 - 16 = 24
    let start = if window > 0 && position_count > window {
        position_count - window
    } else {
        0
    };
    let scale = 1.0 / (head_dim as f32).sqrt();
    let repeats = n_heads / n_kv;
    let mut rng = Lcg(13);
    let q: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.next_f32()).collect();
    let mut cache_k = vec![0f32; n_kv * max_pos * head_dim];
    let mut cache_v = vec![0f32; n_kv * max_pos * head_dim];
    for kv in 0..n_kv {
        for p in 0..position_count {
            for d in 0..head_dim {
                cache_k[(kv * max_pos + p) * head_dim + d] = rng.next_f32();
                cache_v[(kv * max_pos + p) * head_dim + d] = rng.next_f32();
            }
        }
    }
    // Round K/V through f16 (the real path does this in kv_scatter), upload the bits.
    for x in cache_k.iter_mut() {
        *x = crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(*x));
    }
    for x in cache_v.iter_mut() {
        *x = crate::inference::f16_bits_to_f32(crate::inference::f32_to_f16_bits(*x));
    }
    let cache_k_bits: Vec<u16> = cache_k
        .iter()
        .map(|&x| crate::inference::f32_to_f16_bits(x))
        .collect();
    let cache_v_bits: Vec<u16> = cache_v
        .iter()
        .map(|&x| crate::inference::f32_to_f16_bits(x))
        .collect();
    // CPU reference: windowed [start, position_count).
    let mut expected = vec![0f32; n_heads * head_dim];
    for head in 0..n_heads {
        let kv_head = head / repeats;
        let qh = &q[head * head_dim..head * head_dim + head_dim];
        let mut scores = vec![0f32; position_count];
        // `p` indexes scores AND computes the cache_k base offset, so a range loop is clearest.
        #[allow(clippy::needless_range_loop)]
        for p in start..position_count {
            let kbase = (kv_head * max_pos + p) * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += qh[d] * cache_k[kbase + d];
            }
            scores[p] = dot * scale;
        }
        let m = scores[start..position_count]
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for s in scores[start..position_count].iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for d in 0..head_dim {
            let mut acc = 0f32;
            for p in start..position_count {
                acc += (scores[p] * inv) * cache_v[(kv_head * max_pos + p) * head_dim + d];
            }
            expected[head * head_dim + d] = acc;
        }
    }
    // GPU
    let dq = k.stream.clone_htod(&q).unwrap();
    let cache_k_bytes: Vec<u8> = cache_k_bits
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let cache_v_bytes: Vec<u8> = cache_v_bits
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let dk = k.stream.clone_htod(&cache_k_bytes).unwrap();
    let dv = k.stream.clone_htod(&cache_v_bytes).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n_heads * head_dim).unwrap();
    let mut dscores = k.stream.alloc_zeros::<f32>(n_heads * max_pos).unwrap();
    let (nh, nkv, hd, mp) = (n_heads as i32, n_kv as i32, head_dim as i32, max_pos as i32);
    let win = window as i32;
    let dpos = k.stream.clone_htod(&[(position_count - 1) as i32]).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: (2 * head_dim * 4) as u32,
    };
    let mut b = k.stream.launch_builder(&k.attention_sw);
    b.arg(&dq)
        .arg(&dk)
        .arg(&dv)
        .arg(&mut dout)
        .arg(&nh)
        .arg(&nkv)
        .arg(&hd)
        .arg(&dpos)
        .arg(&mp)
        .arg(&scale)
        .arg(&win)
        .arg(&mut dscores);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n_heads * head_dim];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 1e-4), "attention_decode_sw diverged");
}

// ---- QK-norm per-head parity ----

fn cpu_rms_norm_per_head(
    input: &[f32],
    weight: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let base = h * head_dim;
        let slice = &input[base..base + head_dim];
        let sum: f32 = slice.iter().map(|v| v * v).sum::<f32>();
        let scale = 1.0 / (sum / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            out[base + i] = slice[i] * scale * weight[i];
        }
    }
    out
}

#[test]
#[ignore] // requires CUDA device
fn rms_norm_per_head_parity() {
    let k = CudaResidentKernels::new().unwrap();
    let n_heads = 4usize;
    let head_dim = 64usize;
    let eps = 1e-6f32;
    let total = n_heads * head_dim;

    let input: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.01 - 1.28).sin())
        .collect();
    let weight: Vec<f32> = (0..head_dim).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let expected = cpu_rms_norm_per_head(&input, &weight, n_heads, head_dim, eps);

    let mut d_buf = k.stream.clone_htod(&input).unwrap();
    let d_weight = k.stream.clone_htod(&weight).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (head_dim as u32) * 4,
    };
    let (hd_i, uw) = (head_dim as i32, 1i32);
    let mut b = k.stream.launch_builder(&k.rms_norm_per_head);
    b.arg(&mut d_buf)
        .arg(&d_weight)
        .arg(&hd_i)
        .arg(&eps)
        .arg(&uw);
    unsafe { b.launch(cfg).unwrap() };

    let mut got = vec![0f32; total];
    k.stream.memcpy_dtoh(&d_buf, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(
        close(&got, &expected, 1e-5),
        "rms_norm_per_head diverged\ngot: {:?}\nexp: {:?}",
        &got[..8],
        &expected[..8]
    );
}

/// Gemma 4 applies a weightless RMS norm to V for every KV head. The K-wide
/// launcher flattens [token][head] into one grid; pin it bit-for-bit against the
/// scalar-token launch used by decode, at both production head widths.
#[test]
#[ignore = "requires a CUDA device"]
fn rms_norm_per_head_weightless_batched_heads_match_scalar_tokens_bitwise() {
    let Some(k) = kernels() else {
        return;
    };
    let eps = 1e-6f32;
    let k_tokens = crate::gemma4_runtime::MAX_MTP_VERIFY_ROWS;
    for (kv_heads, head_dim, seed) in [(8usize, 256usize, 0x56_25u64), (2, 512, 0x56_51)] {
        let mut rng = Lcg(seed);
        let per_token = kv_heads * head_dim;
        let input: Vec<f32> = (0..k_tokens * per_token)
            .map(|_| rng.next_f32() * 3.0)
            .collect();
        let dummy_weight = k.stream.clone_htod(&vec![1.0f32; head_dim]).unwrap();

        // Reference: the existing scalar-token dispatch, including its unused
        // but valid weight pointer and use_weight=0 branch.
        let mut scalar = vec![0f32; input.len()];
        for token in 0..k_tokens {
            let mut d_one = k
                .stream
                .clone_htod(&input[token * per_token..(token + 1) * per_token])
                .unwrap();
            let cfg = LaunchConfig {
                grid_dim: (kv_heads as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (head_dim as u32) * 4,
            };
            let (head_dim_i, use_weight) = (head_dim as i32, 0i32);
            let mut b = k.stream.launch_builder(&k.rms_norm_per_head);
            b.arg(&mut d_one)
                .arg(&dummy_weight)
                .arg(&head_dim_i)
                .arg(&eps)
                .arg(&use_weight);
            unsafe { b.launch(cfg) }.unwrap();
            k.stream
                .memcpy_dtoh(
                    &d_one,
                    &mut scalar[token * per_token..(token + 1) * per_token],
                )
                .unwrap();
        }

        let mut d_all = k.stream.clone_htod(&input).unwrap();
        super::launch_rms_norm_per_head_weightless(
            &k.stream,
            &k.rms_norm_per_head,
            &mut d_all,
            k_tokens * kv_heads,
            head_dim,
            eps,
        )
        .unwrap();
        let mut batched = vec![0f32; input.len()];
        k.stream.memcpy_dtoh(&d_all, &mut batched).unwrap();
        k.ctx.synchronize().unwrap();

        assert_same_bits(
            &format!("weightless V norm K={k_tokens}, heads={kv_heads}, dim={head_dim}"),
            &batched,
            &scalar,
        );
    }
}

// ---- Split-half RoPE parity ----

fn cpu_rope_split_half(
    vec: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
) {
    let pairs = rope_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for p in 0..pairs {
            let d0 = p;
            let d1 = p + pairs;
            let x0 = vec[base + d0];
            let x1 = vec[base + d1];
            let c = cos[p];
            let s = sin[p];
            vec[base + d0] = x0 * c - x1 * s;
            vec[base + d1] = x0 * s + x1 * c;
        }
    }
}

#[test]
#[ignore] // requires CUDA device
fn rope_split_half_parity() {
    let k = CudaResidentKernels::new().unwrap();
    let n_heads = 4usize;
    let head_dim = 64usize;
    let rope_dim = 64usize;
    let pairs = rope_dim / 2;
    let total = n_heads * head_dim;

    let input: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.1).sin()).collect();
    let cos: Vec<f32> = (0..pairs).map(|p| ((p as f32) * 0.05).cos()).collect();
    let sin: Vec<f32> = (0..pairs).map(|p| ((p as f32) * 0.05).sin()).collect();

    let mut expected = input.clone();
    cpu_rope_split_half(&mut expected, &cos, &sin, n_heads, head_dim, rope_dim);

    let mut d_vec = k.stream.clone_htod(&input).unwrap();
    let d_cos = k.stream.clone_htod(&cos).unwrap();
    let d_sin = k.stream.clone_htod(&sin).unwrap();

    let grid_total = (n_heads * pairs) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_total.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nh, hd, rd, pairing) = (n_heads as i32, head_dim as i32, rope_dim as i32, 1i32);
    let mut b = k.stream.launch_builder(&k.rope);
    b.arg(&mut d_vec)
        .arg(&d_cos)
        .arg(&d_sin)
        .arg(&nh)
        .arg(&hd)
        .arg(&rd)
        .arg(&pairing);
    unsafe { b.launch(cfg).unwrap() };

    let mut got = vec![0f32; total];
    k.stream.memcpy_dtoh(&d_vec, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(
        close(&got, &expected, 1e-6),
        "rope_split_half diverged\ngot: {:?}\nexp: {:?}",
        &got[..8],
        &expected[..8]
    );
}

// ---- Adjacent-even-odd RoPE still works (regression check) ----

#[test]
#[ignore] // requires CUDA device
fn rope_adjacent_parity() {
    let k = CudaResidentKernels::new().unwrap();
    let n_heads = 4usize;
    let head_dim = 64usize;
    let rope_dim = 64usize;
    let pairs = rope_dim / 2;
    let total = n_heads * head_dim;

    let input: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.1).cos()).collect();
    let cos: Vec<f32> = (0..pairs).map(|p| ((p as f32) * 0.03).cos()).collect();
    let sin: Vec<f32> = (0..pairs).map(|p| ((p as f32) * 0.03).sin()).collect();

    let mut expected = input.clone();
    for h in 0..n_heads {
        let base = h * head_dim;
        for p in 0..pairs {
            let d0 = 2 * p;
            let d1 = d0 + 1;
            let x0 = expected[base + d0];
            let x1 = expected[base + d1];
            let c = cos[p];
            let s = sin[p];
            expected[base + d0] = x0 * c - x1 * s;
            expected[base + d1] = x0 * s + x1 * c;
        }
    }

    let mut d_vec = k.stream.clone_htod(&input).unwrap();
    let d_cos = k.stream.clone_htod(&cos).unwrap();
    let d_sin = k.stream.clone_htod(&sin).unwrap();

    let grid_total = (n_heads * pairs) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_total.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (nh, hd, rd, pairing) = (n_heads as i32, head_dim as i32, rope_dim as i32, 0i32);
    let mut b = k.stream.launch_builder(&k.rope);
    b.arg(&mut d_vec)
        .arg(&d_cos)
        .arg(&d_sin)
        .arg(&nh)
        .arg(&hd)
        .arg(&rd)
        .arg(&pairing);
    unsafe { b.launch(cfg).unwrap() };

    let mut got = vec![0f32; total];
    k.stream.memcpy_dtoh(&d_vec, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(
        close(&got, &expected, 1e-6),
        "rope_adjacent diverged\ngot: {:?}\nexp: {:?}",
        &got[..8],
        &expected[..8]
    );
}

// ---- Tree-verify parity (lossless GPU tree speculation, Lane A) -------------

/// A tiny synthetic Llama-shaped model on the GPU, built deterministically so the
/// linear verify, the tree verify, and the sequential single-token path all run
/// on identical weights. Returns a fresh engine each call (own KV cache).
struct SynthModel {
    n_layers: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    hidden: usize,
    ffn: usize,
    rope_dim: usize,
    max_pos: usize,
    vocab: usize,
    eps: f32,
    scale: f32,
    layers: Vec<SynthLayer>,
    final_norm: Vec<f32>,
    output_w: Vec<u8>,
    base: f32,
}
struct SynthLayer {
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
    o: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    an: Vec<f32>,
    fnv: Vec<f32>,
}

impl SynthModel {
    fn new() -> Self {
        let (n_layers, hidden, n_heads, n_kv, head_dim, ffn, vocab, max_pos) = (
            2usize, 64usize, 2usize, 1usize, 32usize, 128usize, 96usize, 64usize,
        );
        let rope_dim = 32usize;
        let eps = 1e-5f32;
        let base = 10000f32;
        let q_width = n_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Lcg(0x5eed_cafe);
        let rand = |rng: &mut Lcg, n: usize| (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();
        let layers = (0..n_layers)
            .map(|_| SynthLayer {
                q: quantize_blocks(&rand(&mut rng, q_width * hidden), hidden),
                k: quantize_blocks(&rand(&mut rng, n_kv * head_dim * hidden), hidden),
                v: quantize_blocks(&rand(&mut rng, n_kv * head_dim * hidden), hidden),
                o: quantize_blocks(&rand(&mut rng, hidden * q_width), q_width),
                gate: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
                up: quantize_blocks(&rand(&mut rng, ffn * hidden), hidden),
                down: quantize_blocks(&rand(&mut rng, hidden * ffn), ffn),
                an: rand(&mut rng, hidden)
                    .iter()
                    .map(|v| v * 0.2 + 1.0)
                    .collect(),
                fnv: rand(&mut rng, hidden)
                    .iter()
                    .map(|v| v * 0.2 + 1.0)
                    .collect(),
            })
            .collect();
        let final_norm: Vec<f32> = rand(&mut rng, hidden)
            .iter()
            .map(|v| v * 0.2 + 1.0)
            .collect();
        let output_w = quantize_blocks(&rand(&mut rng, vocab * hidden), hidden);
        SynthModel {
            n_layers,
            n_heads,
            n_kv,
            head_dim,
            hidden,
            ffn,
            rope_dim,
            max_pos,
            vocab,
            eps,
            scale,
            layers,
            final_norm,
            output_w,
            base,
        }
    }

    fn build(&self) -> CudaResidentDecode {
        let mut e = CudaResidentDecode::new(
            self.n_layers,
            self.n_heads,
            self.n_kv,
            self.head_dim,
            self.hidden,
            self.ffn,
            self.rope_dim,
            self.max_pos,
            self.vocab,
            self.eps,
            false,
        )
        .unwrap();
        for l in &self.layers {
            e.set_layer(
                &l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down, &l.an, &l.fnv,
            )
            .unwrap();
        }
        e.set_output(&self.final_norm, &self.output_w, ProjQuant::Q8_0)
            .unwrap();
        e
    }

    /// Deterministic embedding for a token id (no real embedding table needed —
    /// any fixed function works as long as it's the same everywhere).
    fn embed(&self, tok: u32) -> Vec<f32> {
        let mut rng = Lcg(0xE3B0_0000u64 ^ (tok as u64).wrapping_mul(0x9E3779B97F4A7C15));
        (0..self.hidden).map(|_| rng.next_f32()).collect()
    }

    /// RoPE tables (cos, sin) for an absolute position, matching the test's other
    /// RoPE construction (theta = base^(-2p/rope_dim)).
    fn rope(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let pairs = self.rope_dim / 2;
        let mut cos = vec![0f32; pairs];
        let mut sin = vec![0f32; pairs];
        for p in 0..pairs {
            let theta = self.base.powf(-(2.0 * p as f32) / self.rope_dim as f32);
            cos[p] = (pos as f32 * theta).cos();
            sin[p] = (pos as f32 * theta).sin();
        }
        (cos, sin)
    }
}

/// LINEAR TREE == LINEAR VERIFY: on a single-branch tree, `verify_tree`'s argmax
/// per node must equal `verify_batch`'s argmax — i.e. the tree kernels reduce
/// bit-identically to the batched kernels (the losslessness anchor). Both run on
/// fresh copies of the same synthetic model with the same KV seeded by a prefill.
#[test]
#[ignore = "requires a CUDA device"]
fn tree_linear_matches_verify_batch() {
    if kernels().is_none() {
        return;
    }
    use crate::inference::spec_tree::TokenTree;
    let m = SynthModel::new();
    let pairs = m.rope_dim / 2;
    let prefix = 5usize; // committed prefix so base_position > 0 (exercises dense prefix)
    let prefix_tokens: Vec<u32> = (0..prefix as u32)
        .map(|t| (t * 7 + 3) % m.vocab as u32)
        .collect();
    // The linear chain: anchor + drafts.
    let anchor = 11u32;
    let drafts = [13u32, 17, 23, 29, 31];
    let k = drafts.len() + 1;

    // Seed both engines with the SAME prefix via the sequential path.
    let seed = |e: &mut CudaResidentDecode| {
        for (i, &tok) in prefix_tokens.iter().enumerate() {
            let (cos, sin) = m.rope(i);
            e.forward_token(&m.embed(tok), &cos, &sin, i, m.scale, false)
                .unwrap();
        }
        e.set_filled(prefix);
    };

    // Build the K-token chunk inputs (embeddings + per-token RoPE at base+i).
    let mut embs = vec![0f32; k * m.hidden];
    let mut cos_all = vec![0f32; k * pairs];
    let mut sin_all = vec![0f32; k * pairs];
    let chain: Vec<u32> = std::iter::once(anchor)
        .chain(drafts.iter().copied())
        .collect();
    for (i, &tok) in chain.iter().enumerate() {
        embs[i * m.hidden..(i + 1) * m.hidden].copy_from_slice(&m.embed(tok));
        let (cos, sin) = m.rope(prefix + i);
        cos_all[i * pairs..(i + 1) * pairs].copy_from_slice(&cos);
        sin_all[i * pairs..(i + 1) * pairs].copy_from_slice(&sin);
    }

    // Linear verify.
    let mut e_lin = m.build();
    seed(&mut e_lin);
    let lin = e_lin
        .verify_batch(&embs, &cos_all, &sin_all, prefix, k, m.scale)
        .unwrap();

    // Tree verify on the equivalent linear() tree.
    let tree = TokenTree::linear(anchor, &drafts);
    let node_kvslot = tree.node_kvslot(prefix);
    let (anc, words) = tree.ancestor_bitset();
    let mut e_tree = m.build();
    seed(&mut e_tree);
    let tre = e_tree
        .verify_tree(
            &embs,
            &cos_all,
            &sin_all,
            &node_kvslot,
            &anc,
            words,
            prefix,
            k,
            m.scale,
        )
        .unwrap();

    assert_eq!(
        lin, tre,
        "tree verify on a linear tree != linear verify_batch"
    );
}

/// THE CRITICAL ONE: drive a multi-round decode with a BRANCHING drafter and
/// assert the emitted token-id stream is IDENTICAL to plain greedy decode. This
/// exercises COMPACT-BY-RESCATTER every round a non-first branch is taken; an
/// off-by-one in the KV compaction corrupts the NEXT round silently, so we span
/// many rounds and compare the whole stream. Pure synthetic model — no download.
#[test]
#[ignore = "requires a CUDA device"]
fn tree_verify_multiround_lossless() {
    if kernels().is_none() {
        return;
    }
    use crate::inference::spec_tree::TokenTree;
    let m = SynthModel::new();
    let pairs = m.rope_dim / 2;
    let prompt: Vec<u32> = vec![3, 8, 1, 6, 2];
    let count = 40usize;

    // --- Ground truth: plain greedy decode via the proven single-token path. ---
    let truth: Vec<u32> = {
        let mut e = m.build();
        let mut pos = 0usize;
        let mut last = 0u32;
        for (i, &tok) in prompt.iter().enumerate() {
            let (cos, sin) = m.rope(i);
            last = e
                .forward_token(&m.embed(tok), &cos, &sin, i, m.scale, true)
                .unwrap()
                .unwrap();
            pos = i + 1;
        }
        let mut out = vec![last];
        for _ in 1..count {
            let (cos, sin) = m.rope(pos);
            last = e
                .forward_token(&m.embed(last), &cos, &sin, pos, m.scale, true)
                .unwrap()
                .unwrap();
            pos += 1;
            out.push(last);
        }
        out
    };

    // --- Tree-driven decode. A deterministic branching drafter builds a tree of
    // candidate continuations from the running history; whichever branch (if any)
    // the model confirms is accepted + compacted, the rest discarded. ---
    let mut e = m.build();
    // Prefill the prompt; the first emitted token is the argmax after the last prompt token.
    let mut pos = 0usize;
    let mut last = 0u32;
    for (i, &tok) in prompt.iter().enumerate() {
        let (cos, sin) = m.rope(i);
        last = e
            .forward_token(&m.embed(tok), &cos, &sin, i, m.scale, true)
            .unwrap()
            .unwrap();
        pos = i + 1;
    }
    e.set_filled(pos);
    let mut emitted: Vec<u32> = vec![last];
    let mut history: Vec<u32> = prompt.clone();
    history.push(last);

    // A drafter that proposes a BRANCHING tree: from the anchor it offers a few
    // candidate next tokens (some deliberately wrong so branches diverge and the
    // accepted path is rarely node 1 — forcing real compaction), and from the
    // first candidate, a couple of grandchildren. Tokens are derived from history
    // so they vary round to round.
    let draft = |anchor: u32, hist: &[u32]| -> TokenTree {
        let h = hist.len() as u32;
        // children of the anchor (nodes 1..=3)
        let c1 = (anchor.wrapping_mul(5).wrapping_add(h)) % m.vocab as u32;
        let c2 = (anchor.wrapping_mul(3).wrapping_add(7)) % m.vocab as u32;
        let c3 = (anchor.wrapping_add(h).wrapping_mul(2)) % m.vocab as u32;
        // grandchildren of c1 (nodes 4,5) and c2 (node 6)
        let g1 = (c1.wrapping_mul(11).wrapping_add(1)) % m.vocab as u32;
        let g2 = (c1.wrapping_add(13)) % m.vocab as u32;
        let g3 = (c2.wrapping_mul(7)) % m.vocab as u32;
        TokenTree {
            tokens: vec![anchor, c1, c2, c3, g1, g2, g3],
            parent: vec![-1, 0, 0, 0, 1, 1, 2],
            depth: vec![0, 1, 1, 1, 2, 2, 2],
        }
    };

    let mut rounds = 0usize;
    let mut accepted_total = 0usize;
    let mut compaction_rounds = 0usize; // rounds where the accepted path was NOT node-1-prefixed
    while emitted.len() < count {
        rounds += 1;
        assert!(rounds < 10_000, "tree decode did not terminate");
        let anchor = *history.last().unwrap();
        let tree = draft(anchor, &history);
        let n = tree.nodes();
        // Stage node-order inputs.
        let mut embs = vec![0f32; n * m.hidden];
        let mut cos_all = vec![0f32; n * pairs];
        let mut sin_all = vec![0f32; n * pairs];
        let node_depth = tree.node_depth();
        for i in 0..n {
            embs[i * m.hidden..(i + 1) * m.hidden].copy_from_slice(&m.embed(tree.tokens[i]));
            let (cos, sin) = m.rope(pos + node_depth[i] as usize);
            cos_all[i * pairs..(i + 1) * pairs].copy_from_slice(&cos);
            sin_all[i * pairs..(i + 1) * pairs].copy_from_slice(&sin);
        }
        let node_kvslot = tree.node_kvslot(pos);
        let (anc, words) = tree.ancestor_bitset();
        let predicted = e
            .verify_tree(
                &embs,
                &cos_all,
                &sin_all,
                &node_kvslot,
                &anc,
                words,
                pos,
                n,
                m.scale,
            )
            .unwrap();
        let (round_emit, leaf) = tree.accept_longest_path(&predicted);
        let path = tree.path_to(leaf);
        // A path is "compacting" when some accepted node's BFS index != its path rank
        // (i.e. a non-first branch was taken) — exactly the rescatter off-by-one risk.
        if path.iter().enumerate().any(|(r, &node)| node != r) {
            compaction_rounds += 1;
        }
        e.compact_tree_kv_path(&path, pos).unwrap();
        accepted_total += round_emit.len();
        pos += round_emit.len();
        e.set_filled(pos);
        for t in round_emit {
            emitted.push(t);
            history.push(t);
        }
    }
    emitted.truncate(count);
    assert_eq!(
        emitted, truth,
        "tree-verify decode diverged from plain greedy (lossless violated)"
    );
    eprintln!(
        "tree_verify_multiround_lossless: {} tokens over {} rounds, {:.2} accepted/round, {} compacting rounds",
        count, rounds, accepted_total as f64 / rounds as f64, compaction_rounds
    );
}

/// Sibling of the multi-round test that GUARANTEES the rescatter fires: a drafter
/// whose FIRST child is always a deliberately-wrong token, so any accepted draft
/// must come from a non-node-1 branch (path rank != BFS index) — forcing the
/// compact-by-rescatter copy. Steered so the model's own argmax (mined live from a
/// throwaway probe forward) is planted at node 2's grandchild, making the accepted
/// path [0,2,6] and the compaction non-trivial on real rounds.
#[test]
#[ignore = "requires a CUDA device"]
fn tree_verify_forced_compaction_lossless() {
    if kernels().is_none() {
        return;
    }
    use crate::inference::spec_tree::TokenTree;
    let m = SynthModel::new();
    let pairs = m.rope_dim / 2;
    let prompt: Vec<u32> = vec![2, 9, 4, 1, 7, 3];
    let count = 32usize;

    // Ground truth: plain greedy.
    let truth: Vec<u32> = {
        let mut e = m.build();
        let mut pos = 0usize;
        let mut last = 0u32;
        for (i, &tok) in prompt.iter().enumerate() {
            let (cos, sin) = m.rope(i);
            last = e
                .forward_token(&m.embed(tok), &cos, &sin, i, m.scale, true)
                .unwrap()
                .unwrap();
            pos = i + 1;
        }
        let mut out = vec![last];
        for _ in 1..count {
            let (cos, sin) = m.rope(pos);
            last = e
                .forward_token(&m.embed(last), &cos, &sin, pos, m.scale, true)
                .unwrap()
                .unwrap();
            pos += 1;
            out.push(last);
        }
        out
    };

    // Tree decode. To force the accepted path off node 1, build the tree so the
    // model's actual next token (taken from `truth`) is planted at node 2 (a
    // sibling of node 1), and node 1's token is something else. The accepted path
    // then becomes [0, 2] every time the model confirms a token here — exercising
    // the rescatter (slot pos+2 -> pos+1) on EVERY accepted round.
    let mut e = m.build();
    let mut pos = 0usize;
    let mut last = 0u32;
    for (i, &tok) in prompt.iter().enumerate() {
        let (cos, sin) = m.rope(i);
        last = e
            .forward_token(&m.embed(tok), &cos, &sin, i, m.scale, true)
            .unwrap()
            .unwrap();
        pos = i + 1;
    }
    e.set_filled(pos);
    let mut emitted: Vec<u32> = vec![last];
    let mut history: Vec<u32> = prompt.clone();
    history.push(last);

    let mut rounds = 0usize;
    let mut compaction_rounds = 0usize;
    let mut accepted_total = 0usize;
    while emitted.len() < count {
        rounds += 1;
        assert!(rounds < 10_000, "decode did not terminate");
        let anchor = *history.last().unwrap();
        // The token the model WILL pick next (from ground truth) goes at node 2,
        // never node 1. Node 1 gets a deliberately different token.
        let want = truth[emitted.len().min(truth.len() - 1)];
        let wrong = (want + 1) % m.vocab as u32;
        let tree = TokenTree {
            //          0
            //        / | \
            //       1  2  3       (wrong, want, other)
            tokens: vec![
                anchor,
                wrong,
                want,
                (anchor.wrapping_add(5)) % m.vocab as u32,
            ],
            parent: vec![-1, 0, 0, 0],
            depth: vec![0, 1, 1, 1],
        };
        let n = tree.nodes();
        let mut embs = vec![0f32; n * m.hidden];
        let mut cos_all = vec![0f32; n * pairs];
        let mut sin_all = vec![0f32; n * pairs];
        let node_depth = tree.node_depth();
        for i in 0..n {
            embs[i * m.hidden..(i + 1) * m.hidden].copy_from_slice(&m.embed(tree.tokens[i]));
            let (cos, sin) = m.rope(pos + node_depth[i] as usize);
            cos_all[i * pairs..(i + 1) * pairs].copy_from_slice(&cos);
            sin_all[i * pairs..(i + 1) * pairs].copy_from_slice(&sin);
        }
        let node_kvslot = tree.node_kvslot(pos);
        let (anc, words) = tree.ancestor_bitset();
        let predicted = e
            .verify_tree(
                &embs,
                &cos_all,
                &sin_all,
                &node_kvslot,
                &anc,
                words,
                pos,
                n,
                m.scale,
            )
            .unwrap();
        let (round_emit, leaf) = tree.accept_longest_path(&predicted);
        let path = tree.path_to(leaf);
        if path.iter().enumerate().any(|(r, &node)| node != r) {
            compaction_rounds += 1;
        }
        e.compact_tree_kv_path(&path, pos).unwrap();
        accepted_total += round_emit.len();
        pos += round_emit.len();
        e.set_filled(pos);
        for t in round_emit {
            emitted.push(t);
            history.push(t);
        }
    }
    emitted.truncate(count);
    assert_eq!(
        emitted, truth,
        "forced-compaction tree decode diverged from greedy"
    );
    assert!(
        compaction_rounds > 0,
        "test did not exercise the rescatter — no compacting rounds occurred"
    );
    eprintln!(
        "tree_verify_forced_compaction_lossless: {} tokens, {} rounds, {} COMPACTING rounds, {:.2} accepted/round",
        count, rounds, compaction_rounds, accepted_total as f64 / rounds as f64
    );
}

// Build `rows*n_sb` synthetic Q4_K_M super-blocks (144 bytes each, row-major).
// The bytes need not be a "real" quantization — the kernel and the oracle interpret
// the SAME bytes, so any pattern exercises bit-parity. d/dmin are small positive
// f16 values; scales (12 bytes) and quants (128 bytes) are random.
fn synth_q4k_wire(rows: usize, n_sb: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 144;
    let mut out = vec![0u8; rows * n_sb * WIRE];
    for sb in 0..rows * n_sb {
        let blk = &mut out[sb * WIRE..(sb + 1) * WIRE];
        // small positive f16 super-scales so the products stay in a sane f32 range.
        let d = (rng.next_f32().abs() * 0.05 + 0.001).min(0.2);
        let dmin = (rng.next_f32().abs() * 0.05 + 0.001).min(0.2);
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        let dmb = crate::inference::f32_to_f16_bits(dmin).to_le_bytes();
        blk[0] = db[0];
        blk[1] = db[1];
        blk[2] = dmb[0];
        blk[3] = dmb[1];
        for b in blk.iter_mut().take(144).skip(4) {
            *b = rng.next_u8();
        }
    }
    out
}

// Bit-parity receipt for the Q4_K_M fused-dequant decode GEMV. Generates synthetic
// Q4_K super-block weight bytes + a Q8_K-quantized activation, runs q4k_gemv on the
// GPU, and asserts each output row reproduces the validated CPU oracle
// `q4_k_wire_row_dot` on the SAME bytes. The kernel mirrors the oracle's ordered
// f32 accumulation (8 main lanes + scalar mins, summed left-to-right per row), so
// the result is expected BIT-IDENTICAL — but we accept the same tiny ordered-f32
// tolerance the q8 parity tests use to stay robust across compilers.
#[test]
#[ignore = "requires a CUDA device"]
fn q4k_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let n_sb = 3usize; // contraction dim = 3*256 = 768
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x4b_4b_4b);

    // Synthetic Q4_K weight wire bytes (rows*n_sb super-blocks). The GPU-side
    // layout is the upload-time quant-byte swizzle (swz_q4k_blocks, exactly as
    // repack_for_lane applies it); the CPU oracle reads the stock wire.
    let wire = synth_q4k_wire(rows, n_sb, &mut rng);
    let wsoa = super::swz_q4k_blocks(&wire);

    // Q8_K activation: quantize a random f32 row, then split into per-superblock
    // scales (y.d) and the concatenated 256-wide i8 quants (y.qs).
    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8k = crate::inference::quantize_q8_k_blocks(&act);
    assert_eq!(q8k.len(), n_sb);
    let in_scales: Vec<f32> = q8k.iter().map(|b| b.d).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8k.iter().enumerate() {
        in_quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
    }

    // CPU oracle per output row.
    const WIRE: usize = 144;
    let row_bytes = n_sb * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::q4_k_wire_row_dot(row_wire, &q8k);
    }

    // GPU.
    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wsoa).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q4k_gemv(
        &k.stream,
        &k.q4k_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wsoa.len()),
        rows,
        n_sb,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    // The kernel reproduces the oracle's exact ordered f32 sum, so this should be
    // bit-identical; report the worst lane and assert within the q8 ordered-f32 tol.
    let mut worst = 0f32;
    let mut exact = 0usize;
    for (g, e) in got.iter().zip(&expected) {
        if g.to_bits() == e.to_bits() {
            exact += 1;
        }
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    eprintln!(
        "q4k_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        exact, rows, worst
    );
    assert!(
        close(&got, &expected, 1e-4),
        "q4k_gemv diverged from q4_k_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

/// Batched Q4_K receipt: every token-major output must match the same CPU
/// oracle used to validate the single-token GEMV. This specifically exercises
/// the upload-swizzled weights against natural-order batched Q8_K activations.
#[test]
#[ignore = "requires a CUDA device"]
fn q4k_gemm_batched_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let (rows, n_sb, k_tokens) = (64usize, 3usize, 4usize);
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x4b_47_45_4d_4d);
    let wire = synth_q4k_wire(rows, n_sb, &mut rng);
    let weights = super::swz_q4k_blocks(&wire);
    let mut in_scales = Vec::with_capacity(k_tokens * n_sb);
    let mut in_quants = Vec::with_capacity(k_tokens * kdim);
    let mut expected = vec![0f32; k_tokens * rows];
    for t in 0..k_tokens {
        let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let q8k = crate::inference::quantize_q8_k_blocks(&act);
        for block in &q8k {
            in_scales.push(block.d);
            in_quants.extend_from_slice(&block.qs);
        }
        for row in 0..rows {
            let lo = row * n_sb * 144;
            expected[t * rows + row] =
                crate::inference::q4_k_wire_row_dot(&wire[lo..lo + n_sb * 144], &q8k);
        }
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&weights).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
    super::launch_kquant_gemm_batched(
        &k.stream,
        &k.q4k_gemm_batched,
        &d_is,
        &d_iq,
        &d_w,
        rows,
        n_sb,
        k_tokens,
        9,
        &mut d_out,
    )
    .unwrap();
    let mut got = vec![0f32; k_tokens * rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let exact = got
        .iter()
        .zip(&expected)
        .filter(|(g, e)| g.to_bits() == e.to_bits())
        .count();
    eprintln!(
        "q4k_gemm_batched_matches_oracle: {exact}/{} outputs bit-identical",
        got.len()
    );
    assert!(
        close(&got, &expected, 1e-4),
        "batched Q4_K diverged from q4_k_wire_row_dot"
    );
}

// Build `rows*n_sb` synthetic Q5_K_M super-blocks (176 bytes each, row-major):
// d(f16), dmin(f16), scales[12], qh[32], qs[128]. Like synth_q4k_wire the bytes need
// not be a real quantization — the kernel and the oracle read the SAME bytes — so the
// full qh[32]+qs[128] weight region is random (exercises every fifth-bit combination).
fn synth_q5k_wire(rows: usize, n_sb: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 176;
    let mut out = vec![0u8; rows * n_sb * WIRE];
    for sb in 0..rows * n_sb {
        let blk = &mut out[sb * WIRE..(sb + 1) * WIRE];
        let d = (rng.next_f32().abs() * 0.05 + 0.001).min(0.2);
        let dmin = (rng.next_f32().abs() * 0.05 + 0.001).min(0.2);
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        let dmb = crate::inference::f32_to_f16_bits(dmin).to_le_bytes();
        blk[0] = db[0];
        blk[1] = db[1];
        blk[2] = dmb[0];
        blk[3] = dmb[1];
        // scales[12] + qh[32] + qs[128] = bytes 4..176, fully random.
        for b in blk.iter_mut().take(176).skip(4) {
            *b = rng.next_u8();
        }
    }
    out
}

// Bit-parity receipt for the Q5_K_M fused-dequant decode GEMV. Generates synthetic
// Q5_K super-block weight bytes + a Q8_K-quantized activation, runs q5k_gemv on the
// GPU, and asserts each output row reproduces the validated CPU oracle
// `q5_k_wire_row_dot` on the SAME bytes. Q5_K is Q4_K plus a fifth (qh) bit, so the
// kernel mirrors the same ordered f32 accumulation (8 main lanes + scalar mins) as
// q4k_gemv — the result is expected BIT-IDENTICAL, within the same tiny ordered-f32
// tolerance the q4k/q8 parity tests use.
#[test]
#[ignore = "requires a CUDA device"]
fn q5k_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let n_sb = 3usize; // contraction dim = 3*256 = 768
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x5b_5b_5b);

    // Synthetic Q5_K weight wire bytes (rows*n_sb super-blocks). The kernel reads the
    // RAW 176-byte wire layout directly (low nibbles + qh fifth bit + kmask scales
    // expanded on the fly), so no host repack — the same bytes the resident upload
    // passes through.
    let wire = synth_q5k_wire(rows, n_sb, &mut rng);
    let wsoa = wire.clone();

    // Q8_K activation: quantize a random f32 row, then split into per-superblock
    // scales (y.d) and the concatenated 256-wide i8 quants (y.qs).
    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8k = crate::inference::quantize_q8_k_blocks(&act);
    assert_eq!(q8k.len(), n_sb);
    let in_scales: Vec<f32> = q8k.iter().map(|b| b.d).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8k.iter().enumerate() {
        in_quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
    }

    // CPU oracle per output row.
    const WIRE: usize = 176;
    let row_bytes = n_sb * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::q5_k_wire_row_dot(row_wire, &q8k);
    }

    // GPU.
    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wsoa).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q5k_gemv(
        &k.stream,
        &k.q5k_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wsoa.len()),
        rows,
        n_sb,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let mut worst = 0f32;
    let mut exact = 0usize;
    for (g, e) in got.iter().zip(&expected) {
        if g.to_bits() == e.to_bits() {
            exact += 1;
        }
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    eprintln!(
        "q5k_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        exact, rows, worst
    );
    assert!(
        close(&got, &expected, 1e-4),
        "q5k_gemv diverged from q5_k_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

// Synthetic Q2_K weight wire bytes: rows*n_sb super-blocks of 84 bytes each
fn synth_q2k_wire(rows: usize, n_sb: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 84;
    let mut out = vec![0u8; rows * n_sb * WIRE];
    for sb in 0..rows * n_sb {
        let blk = &mut out[sb * WIRE..(sb + 1) * WIRE];
        for b in blk.iter_mut().take(80) {
            *b = rng.next_u8(); // scales[16] + qs[64]
        }
        let d = (rng.next_f32().abs() * 0.05 + 0.001).min(0.2);
        let dmin = (rng.next_f32().abs() * 0.05 + 0.001).min(0.2);
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        let dmb = crate::inference::f32_to_f16_bits(dmin).to_le_bytes();
        blk[80] = db[0];
        blk[81] = db[1];
        blk[82] = dmb[0];
        blk[83] = dmb[1];
    }
    out
}

// Bit-parity receipt for the Q2_K fused-dequant decode GEMV. Generates synthetic
// Q2_K super-block weight bytes + a Q8_K-quantized activation, runs q2k_gemv on the
// GPU, and asserts each output row reproduces the CPU oracle `q2_k_wire_row_dot` on
// the SAME bytes. The kernel mirrors the oracle's ordered f32 reduction (per
// super-block `dall*isum - dmin*summs`, summed in order), so the result is expected
// BIT-IDENTICAL — within the same tiny ordered-f32 tolerance the q8/q4k tests use.
#[test]
#[ignore = "requires a CUDA device"]
fn q2k_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let n_sb = 3usize; // contraction dim = 3*256 = 768
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x2b_2b_2b);

    let wire = synth_q2k_wire(rows, n_sb, &mut rng);

    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8k = crate::inference::quantize_q8_k_blocks(&act);
    assert_eq!(q8k.len(), n_sb);
    let in_scales: Vec<f32> = q8k.iter().map(|b| b.d).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8k.iter().enumerate() {
        in_quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
    }

    const WIRE: usize = 84;
    let row_bytes = n_sb * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::q2_k_wire_row_dot(row_wire, &q8k);
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q2k_gemv(
        &k.stream,
        &k.q2k_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        n_sb,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let mut worst = 0f32;
    let mut exact = 0usize;
    for (g, e) in got.iter().zip(&expected) {
        if g.to_bits() == e.to_bits() {
            exact += 1;
        }
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    eprintln!(
        "q2k_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        exact, rows, worst
    );
    assert!(
        close(&got, &expected, 1e-4),
        "q2k_gemv diverged from q2_k_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

// Bit-parity receipt for the ROUTED Q2_K expert GEMV against the dense kernel it
// was ported from. Builds a multi-slot arena of synthetic Q2_K expert slabs, runs
// q2k_gemv_routed once over a shuffled route (slot_ids out of order, route_ids a
// permutation), and asserts every expert's output block is BIT-IDENTICAL to a
// dense q2k_gemv launch over that expert's slab. This scopes the certificate to
// what the routed port added — arena addressing, slot/route indirection, output
// placement — on top of the dense kernel's own oracle receipt above.
#[test]
#[ignore = "requires a CUDA device"]
fn q2k_gemv_routed_matches_dense() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 64usize;
    let n_sb = 11usize; // the 26B gate_up geometry: hidden 2816 = 11*256
    let kdim = n_sb * 256;
    let slots = 6usize;
    let route_count = 4usize;
    let mut rng = Lcg(0x2b_2b_2c);

    const WIRE: usize = 84;
    let stride = rows * n_sb * WIRE;
    let mut arena = vec![0u8; slots * stride];
    for slot in 0..slots {
        let wire = synth_q2k_wire(rows, n_sb, &mut rng);
        arena[slot * stride..(slot + 1) * stride].copy_from_slice(&wire);
    }

    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8k = crate::inference::quantize_q8_k_blocks(&act);
    let in_scales: Vec<f32> = q8k.iter().map(|b| b.d).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8k.iter().enumerate() {
        in_quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
    }

    // Deliberately non-identity mappings: routed expert e reads arena slot
    // slot_ids[e] and writes output block route_ids[e].
    let slot_ids: Vec<i32> = vec![4, 1, 5, 2];
    let route_ids: Vec<i32> = vec![2, 0, 3, 1];

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_arena = k.stream.clone_htod(&arena).unwrap();
    let d_slots = k.stream.clone_htod(&slot_ids).unwrap();
    let d_routes = k.stream.clone_htod(&route_ids).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(route_count * rows).unwrap();

    {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let block = 256u32;
        let warps = block / 32;
        let cfg = LaunchConfig {
            grid_dim: ((rows as u32).div_ceil(warps), route_count as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: n_sb as u32 * 256 + n_sb as u32 * 4 + warps * n_sb as u32 * 2 * 4,
        };
        let stride_u64 = stride as u64;
        let rows_i = rows as i32;
        let n_sb_i = n_sb as i32;
        let experts_i = route_count as i32;
        let mut b = k.stream.launch_builder(&k.q2k_gemv_routed);
        b.arg(&d_is)
            .arg(&d_iq)
            .arg(&d_arena)
            .arg(&d_slots)
            .arg(&d_routes)
            .arg(&stride_u64)
            .arg(&rows_i)
            .arg(&n_sb_i)
            .arg(&mut d_out)
            .arg(&experts_i);
        unsafe { b.launch(cfg) }.unwrap();
    }
    let mut routed = vec![0f32; route_count * rows];
    k.stream.memcpy_dtoh(&d_out, &mut routed).unwrap();

    // Dense reference: one q2k_gemv per selected slot over the same arena bytes.
    let mut mismatches = 0usize;
    for e in 0..route_count {
        let slot = slot_ids[e] as usize;
        let route = route_ids[e] as usize;
        let mut d_dense = k.stream.alloc_zeros::<f32>(rows).unwrap();
        super::launch_q2k_gemv(
            &k.stream,
            &k.q2k_gemv,
            &d_is,
            &d_iq,
            &d_arena.slice(slot * stride..(slot + 1) * stride),
            rows,
            n_sb,
            &mut d_dense,
            0,
        )
        .unwrap();
        let mut dense = vec![0f32; rows];
        k.stream.memcpy_dtoh(&d_dense, &mut dense).unwrap();
        k.ctx.synchronize().unwrap();
        for r in 0..rows {
            if routed[route * rows + r].to_bits() != dense[r].to_bits() {
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "q2k_gemv_routed must be bit-identical to per-slot dense q2k_gemv"
    );
}

// Synthetic Q3_K weight wire bytes: rows*n_sb super-blocks of 110 bytes each
// (hmask[32] + qs[64] + scales[12] + d f16). Small positive f16 super-scale keeps
// the dequant products in a sane f32 range; hmask/qs/scales fully random.
fn synth_q3k_wire(rows: usize, n_sb: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 110;
    let mut out = vec![0u8; rows * n_sb * WIRE];
    for sb in 0..rows * n_sb {
        let blk = &mut out[sb * WIRE..(sb + 1) * WIRE];
        for b in blk.iter_mut().take(108) {
            *b = rng.next_u8(); // hmask[32] + qs[64] + scales[12]
        }
        let d = (rng.next_f32().abs() * 0.05 + 0.001).min(0.2);
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        blk[108] = db[0];
        blk[109] = db[1];
    }
    out
}

// Bit-parity receipt for the Q3_K fused-dequant decode GEMV. Asserts each output row
// reproduces the CPU oracle `q3_k_wire_row_dot` on the SAME bytes. The kernel mirrors
// the oracle's ordered f32 reduction (per super-block `d*isum`), expected BIT-IDENTICAL.
#[test]
#[ignore = "requires a CUDA device"]
fn q3k_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let n_sb = 3usize;
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x3b_3b_3b);

    let wire = synth_q3k_wire(rows, n_sb, &mut rng);

    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8k = crate::inference::quantize_q8_k_blocks(&act);
    assert_eq!(q8k.len(), n_sb);
    let in_scales: Vec<f32> = q8k.iter().map(|b| b.d).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8k.iter().enumerate() {
        in_quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
    }

    const WIRE: usize = 110;
    let row_bytes = n_sb * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::q3_k_wire_row_dot(row_wire, &q8k);
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q3k_gemv(
        &k.stream,
        &k.q3k_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        n_sb,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let mut worst = 0f32;
    let mut exact = 0usize;
    for (g, e) in got.iter().zip(&expected) {
        if g.to_bits() == e.to_bits() {
            exact += 1;
        }
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    eprintln!(
        "q3k_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        exact, rows, worst
    );
    assert!(
        close(&got, &expected, 1e-4),
        "q3k_gemv diverged from q3_k_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

// Synthetic Q4_0 weight wire bytes: rows*bpr blocks of 18 bytes each (f16 scale +
// 16 nibble bytes). Small positive f16 scale keeps the dequant products in range.
fn synth_q4_0_wire(rows: usize, bpr: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 18;
    let mut out = vec![0u8; rows * bpr * WIRE];
    for blk_idx in 0..rows * bpr {
        let blk = &mut out[blk_idx * WIRE..(blk_idx + 1) * WIRE];
        let d = (rng.next_f32().abs() * 0.03 + 0.001).min(0.1);
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        blk[0] = db[0];
        blk[1] = db[1];
        for b in blk.iter_mut().skip(2) {
            *b = rng.next_u8();
        }
    }
    out
}

/// Reference the native-f32 Q4_0 SoA lane with exactly the device kernel's
/// rounding points: ordered low nibbles, ordered high nibbles, one scale
/// multiply per block, then an increasing-block fold.
fn q4_0_f32_soa_ordered_row_dot(
    soa: &[u8],
    rows: usize,
    bpr: usize,
    row: usize,
    input: &[f32],
) -> f32 {
    let blocks = rows * bpr;
    let quant_bytes = blocks * 16;
    assert_eq!(soa.len(), blocks * 18);
    assert!(row < rows);
    assert_eq!(input.len(), bpr * 32);

    let mut acc = 0.0f32;
    for block in 0..bpr {
        let index = row * bpr + block;
        let quants = &soa[index * 16..index * 16 + 16];
        let x = &input[block * 32..block * 32 + 32];
        let mut block_dot = 0.0f32;
        for column in 0..16 {
            let quant = (quants[column] & 0x0f) as i32 - 8;
            let product = quant as f32 * x[column];
            block_dot += product;
        }
        for column in 0..16 {
            let quant = (quants[column] >> 4) as i32 - 8;
            let product = quant as f32 * x[16 + column];
            block_dot += product;
        }
        let scale_offset = quant_bytes + index * 2;
        let scale = crate::inference::f16_bits_to_f32(u16::from_le_bytes([
            soa[scale_offset],
            soa[scale_offset + 1],
        ]));
        let term = block_dot * scale;
        acc += term;
    }
    acc
}

fn run_q4_0_f32_gemv_soa(
    kernels: &CudaResidentKernels,
    soa: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let d_input = kernels.stream.clone_htod(input).unwrap();
    let d_weight = kernels.stream.clone_htod(soa).unwrap();
    let mut d_output = kernels.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q4_0_f32_gemv_soa(
        &kernels.stream,
        &kernels.q4_0_f32_gemv_soa,
        &d_input,
        &d_weight.slice(0..d_weight.len()),
        rows,
        cols,
        &mut d_output,
        0,
    )
    .unwrap();
    let mut output = vec![0.0f32; rows];
    kernels.stream.memcpy_dtoh(&d_output, &mut output).unwrap();
    kernels.ctx.synchronize().unwrap();
    output
}

// Bit-parity receipt for the Q4_0 resident GEMV. Generates synthetic Q4_0 wire bytes
// + a Q8_0 activation, runs q4_0_gemv on the GPU, and asserts each output row
// reproduces the validated CPU oracle `q4_0_wire_row_dot` on the SAME bytes. The
// kernel mirrors the oracle's exact per-block integer dot + ordered f32 accumulation
// (the same contract as q8_gemv), so the result is expected bit-identical.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let bpr = 24usize; // contraction dim = 24*32 = 768
    let kdim = bpr * 32;
    let mut rng = Lcg(0x40_40_40);

    let wire = synth_q4_0_wire(rows, bpr, &mut rng);

    // Q8_0 activation: quantize a random f32 row to Q8_0 blocks (the oracle format),
    // then split into per-block scales + concatenated i8 quants for the GPU.
    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8 = crate::inference::quantize_q8_0_blocks(&act);
    assert_eq!(q8.len(), bpr);
    let in_scales: Vec<f32> = q8.iter().map(|b| b.scale).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8.iter().enumerate() {
        in_quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
    }

    // CPU oracle per output row.
    const WIRE: usize = 18;
    let row_bytes = bpr * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::q4_0_wire_row_dot(row_wire, &q8);
    }

    // GPU.
    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q4_0_gemv(
        &k.stream,
        &k.q4_0_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        bpr,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let mut worst = 0f32;
    let mut exact = 0usize;
    for (g, e) in got.iter().zip(&expected) {
        if g.to_bits() == e.to_bits() {
            exact += 1;
        }
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    eprintln!(
        "q4_0_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        exact, rows, worst
    );
    assert!(
        close(&got, &expected, 1e-4),
        "q4_0_gemv diverged from q4_0_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

// Synthetic Q4_1 wire bytes: rows*bpr blocks of 20 bytes (f16 d, f16 m, 16 nibbles).
fn synth_q4_1_wire(rows: usize, bpr: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 20;
    let mut out = vec![0u8; rows * bpr * WIRE];
    for blk_idx in 0..rows * bpr {
        let blk = &mut out[blk_idx * WIRE..(blk_idx + 1) * WIRE];
        let d = (rng.next_f32().abs() * 0.03 + 0.001).min(0.1);
        let m = (rng.next_f32() - 0.5) * 0.05;
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        let mb = crate::inference::f32_to_f16_bits(m).to_le_bytes();
        blk[0] = db[0];
        blk[1] = db[1];
        blk[2] = mb[0];
        blk[3] = mb[1];
        for b in blk.iter_mut().skip(4) {
            *b = rng.next_u8();
        }
    }
    out
}

// Bit-parity receipt for the Q4_1 resident GEMV vs the CPU oracle `q4_1_wire_row_dot`
// on the same synthetic bytes (the gemma4 mixed-Q4_0 ffn_down lane).
#[test]
#[ignore = "requires a CUDA device"]
fn q4_1_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let bpr = 24usize; // contraction dim = 24*32 = 768
    let kdim = bpr * 32;
    let mut rng = Lcg(0x41_41_41);

    let wire = synth_q4_1_wire(rows, bpr, &mut rng);

    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8 = crate::inference::quantize_q8_0_blocks(&act);
    assert_eq!(q8.len(), bpr);
    let in_scales: Vec<f32> = q8.iter().map(|b| b.scale).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8.iter().enumerate() {
        in_quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
    }

    const WIRE: usize = 20;
    let row_bytes = bpr * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::q4_1_wire_row_dot(row_wire, &q8);
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q4_1_gemv(
        &k.stream,
        &k.q4_1_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        bpr,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let mut worst = 0f32;
    let mut exact = 0usize;
    for (g, e) in got.iter().zip(&expected) {
        if g.to_bits() == e.to_bits() {
            exact += 1;
        }
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    eprintln!(
        "q4_1_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        exact, rows, worst
    );
    assert!(
        close(&got, &expected, 1e-4),
        "q4_1_gemv diverged from q4_1_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

fn run_q4_0_gemm_batched_case(ktok: usize, bpr: usize, seed: u64) {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let kdim = bpr * 32;
    let mut rng = Lcg(seed);
    let wire = synth_q4_0_wire(rows, bpr, &mut rng);
    let mut in_scales = vec![0f32; ktok * bpr];
    let mut in_quants = vec![0i8; ktok * kdim];
    let mut cpu = vec![0f32; ktok * rows];
    let mut scalar_gpu = vec![0f32; ktok * rows];
    const WIRE: usize = 18;
    let row_bytes = bpr * WIRE;

    for t in 0..ktok {
        let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let q8 = crate::inference::quantize_q8_0_blocks(&act);
        let scales: Vec<f32> = q8.iter().map(|b| b.scale).collect();
        let mut quants = vec![0i8; kdim];
        for (b, blk) in q8.iter().enumerate() {
            quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
        }
        in_scales[t * bpr..(t + 1) * bpr].copy_from_slice(&scales);
        in_quants[t * kdim..(t + 1) * kdim].copy_from_slice(&quants);
        for r in 0..rows {
            cpu[t * rows + r] =
                crate::inference::q4_0_wire_row_dot(&wire[r * row_bytes..(r + 1) * row_bytes], &q8);
        }

        let d_s = k.stream.clone_htod(&scales).unwrap();
        let d_q = k.stream.clone_htod(&quants).unwrap();
        let d_w = k.stream.clone_htod(&wire).unwrap();
        let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
        super::launch_q4_0_gemv(
            &k.stream,
            &k.q4_0_gemv,
            &d_s,
            &d_q,
            &d_w.slice(0..wire.len()),
            rows,
            bpr,
            &mut d_out,
            0,
        )
        .unwrap();
        k.stream
            .memcpy_dtoh(&d_out, &mut scalar_gpu[t * rows..(t + 1) * rows])
            .unwrap();
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(ktok * rows).unwrap();
    super::launch_q4_0_gemm_batched(
        &k.stream,
        &k.q4_0_gemm_batched,
        &d_is,
        &d_iq,
        &d_w,
        rows,
        bpr,
        ktok,
        &mut d_out,
    )
    .unwrap();
    let mut batched = vec![0f32; ktok * rows];
    k.stream.memcpy_dtoh(&d_out, &mut batched).unwrap();
    k.ctx.synchronize().unwrap();

    assert!(
        batched
            .iter()
            .zip(&scalar_gpu)
            .all(|(batch, scalar)| batch.to_bits() == scalar.to_bits()),
        "Q4_0 batched GEMM must be bit-identical to K scalar GEMVs"
    );
    assert!(
        close(&batched, &cpu, 1e-4),
        "Q4_0 batched GEMM diverged from the CPU wire oracle"
    );
}

/// `q4_0_gemm_routed` must be BITWISE identical to `q4_0_gemv_routed` run once per
/// (expert, token) pair.
///
/// The GEMM hoists a weight block into registers and reuses it across the tokens
/// routed to that expert, instead of re-fetching it per token. Nothing else moves:
/// same `q4_0_dot32_dp4a_packed`, same per-block float term, same single-lane
/// increasing-block fold. This test is what lets that ship as exact-parity.
///
/// Exercised with a RAGGED assignment (experts with 0, 1, and many tokens), a
/// permuted slot map, and a tile smaller than the largest token count so the
/// `blockIdx.z` tiling path is covered — those are exactly what a routed GEMM
/// gets wrong and a uniform test would miss.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemm_routed_matches_gemv() {
    let Some(k) = kernels() else {
        return;
    };
    for (rows, bpr, seed) in [(1408usize, 88usize, 0x6e_01u64), (2816, 22, 0x6e_02)] {
        let experts = 4usize;
        let n_tokens = 7usize;
        let kdim = bpr * 32;
        let mut rng = Lcg(seed);

        let per_slot = rows * bpr * 18;
        let mut arena = Vec::with_capacity(experts * per_slot);
        for _ in 0..experts {
            arena.extend_from_slice(&synth_q4_0_wire(rows, bpr, &mut rng));
        }
        // Activations for every token.
        let mut in_s = vec![0f32; n_tokens * bpr];
        let mut in_q = vec![0i8; n_tokens * kdim];
        for t in 0..n_tokens {
            let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
            let q8 = crate::inference::quantize_q8_0_blocks(&act);
            for (b, blk) in q8.iter().enumerate() {
                in_s[t * bpr + b] = blk.scale;
                in_q[t * kdim + b * 32..t * kdim + (b + 1) * 32].copy_from_slice(&blk.quants);
            }
        }
        // Ragged CSR: expert 0 -> 3 tokens, 1 -> 0, 2 -> 1, 3 -> 4. Total 8.
        let token_offsets: Vec<i32> = vec![0, 3, 3, 4, 8];
        let token_ids: Vec<i32> = vec![5, 0, 3, 6, 2, 4, 1, 5];
        let slots: Vec<i32> = vec![2, 0, 3, 1];
        let assignments = *token_offsets.last().unwrap() as usize;

        let d_s = k.stream.clone_htod(&in_s).unwrap();
        let d_q = k.stream.clone_htod(&in_q).unwrap();
        let d_w = k.stream.clone_htod(&arena).unwrap();
        let d_slots = k.stream.clone_htod(&slots).unwrap();
        let d_off = k.stream.clone_htod(&token_offsets).unwrap();
        let d_tok = k.stream.clone_htod(&token_ids).unwrap();

        // Reference: the shipped GEMV, once per assignment, with a single-token
        // activation view — exactly what prefill does today.
        let mut reference = vec![0f32; assignments * rows];
        for e in 0..experts {
            let (lo, hi) = (token_offsets[e] as usize, token_offsets[e + 1] as usize);
            for a in lo..hi {
                let t = token_ids[a] as usize;
                let d_s1 = k.stream.clone_htod(&in_s[t * bpr..(t + 1) * bpr]).unwrap();
                let d_q1 = k
                    .stream
                    .clone_htod(&in_q[t * kdim..(t + 1) * kdim])
                    .unwrap();
                let one_slot: Vec<i32> = vec![slots[e]];
                let one_route: Vec<i32> = vec![0];
                let d_s1s = k.stream.clone_htod(&one_slot).unwrap();
                let d_r1 = k.stream.clone_htod(&one_route).unwrap();
                let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
                use cudarc::driver::{LaunchConfig, PushKernelArg};
                let block = 256u32;
                let warps = block / 32;
                let cfg = LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(warps), 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: bpr as u32 * 32 + bpr as u32 * 4 + warps * bpr as u32 * 4,
                };
                let (stride, rows_i, bpr_i, one, zero) =
                    (per_slot as u64, rows as i32, bpr as i32, 1i32, 0i32);
                let mut b = k.stream.launch_builder(&k.q4_0_gemv_routed);
                b.arg(&d_s1)
                    .arg(&d_q1)
                    .arg(&d_w)
                    .arg(&d_s1s)
                    .arg(&d_r1)
                    .arg(&stride)
                    .arg(&rows_i)
                    .arg(&bpr_i)
                    .arg(&mut d_out)
                    .arg(&one)
                    .arg(&zero);
                unsafe { b.launch(cfg) }.unwrap();
                k.stream
                    .memcpy_dtoh(&d_out, &mut reference[a * rows..(a + 1) * rows])
                    .unwrap();
                k.ctx.synchronize().unwrap();
            }
        }

        // Under test: one GEMM launch, tile 2 so the blockIdx.z path runs.
        let tile = 2usize;
        let max_count = (0..experts)
            .map(|e| token_offsets[e + 1] - token_offsets[e])
            .max()
            .unwrap() as usize;
        let mut d_out = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
        {
            use cudarc::driver::{LaunchConfig, PushKernelArg};
            let block = 256u32;
            let warps = block / 32;
            let cfg = LaunchConfig {
                grid_dim: (
                    (rows as u32).div_ceil(warps),
                    experts as u32,
                    max_count.div_ceil(tile) as u32,
                ),
                block_dim: (block, 1, 1),
                shared_mem_bytes: warps * tile as u32 * bpr as u32 * 4,
            };
            let (stride, rows_i, bpr_i, experts_i, tile_i) = (
                per_slot as u64,
                rows as i32,
                bpr as i32,
                experts as i32,
                tile as i32,
            );
            let mut b = k.stream.launch_builder(&k.q4_0_gemm_routed);
            b.arg(&d_s)
                .arg(&d_q)
                .arg(&d_w)
                .arg(&d_slots)
                .arg(&d_off)
                .arg(&d_tok)
                .arg(&stride)
                .arg(&rows_i)
                .arg(&bpr_i)
                .arg(&mut d_out)
                .arg(&experts_i)
                .arg(&tile_i);
            unsafe { b.launch(cfg) }.unwrap();
        }
        let mut got = vec![0f32; assignments * rows];
        k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
        k.ctx.synchronize().unwrap();

        // The production helper owns launch geometry selection. Keep the
        // explicit tile-2 launch above to cover grid.z, and independently pin
        // the helper to the same exact assignment output.
        let mut d_helper = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
        super::launch_q4_0_gemm_routed(
            &k.stream,
            &k.q4_0_gemm_routed,
            &d_s,
            &d_q,
            &d_w,
            &d_slots,
            &d_off,
            &d_tok,
            per_slot,
            rows,
            bpr,
            experts,
            max_count,
            false,
            &mut d_helper,
        )
        .unwrap();
        let mut helper = vec![0f32; assignments * rows];
        k.stream.memcpy_dtoh(&d_helper, &mut helper).unwrap();
        k.ctx.synchronize().unwrap();
        assert_same_bits("Q4_0 routed launch helper", &helper, &reference);

        // Two-pass range launch — the resident-first split's mechanism: experts
        // [0,2) in one launch, [2,4) in a second, same CSR arrays, same output
        // buffer. Each expert's per-row fold still happens whole inside exactly
        // one launch, so the split must be BITWISE identical to the full launch.
        let mut d_split = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
        super::launch_q4_0_gemm_routed_range(
            &k.stream,
            &k.q4_0_gemm_routed,
            &d_s,
            &d_q,
            &d_w,
            &d_slots,
            &d_off,
            &d_tok,
            per_slot,
            rows,
            bpr,
            0,
            2,
            max_count,
            false,
            &mut d_split,
        )
        .unwrap();
        super::launch_q4_0_gemm_routed_range(
            &k.stream,
            &k.q4_0_gemm_routed,
            &d_s,
            &d_q,
            &d_w,
            &d_slots,
            &d_off,
            &d_tok,
            per_slot,
            rows,
            bpr,
            2,
            experts - 2,
            max_count,
            false,
            &mut d_split,
        )
        .unwrap();
        let mut split = vec![0f32; assignments * rows];
        k.stream.memcpy_dtoh(&d_split, &mut split).unwrap();
        k.ctx.synchronize().unwrap();
        assert_same_bits("Q4_0 routed two-pass range launch", &split, &reference);

        let mismatches = got
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "q4_0_gemm_routed must be BITWISE identical to per-token q4_0_gemv_routed \
             (rows {rows}, blocks/row {bpr}): {mismatches}/{} outputs differ",
            got.len()
        );
        assert!(
            reference.iter().any(|v| *v != 0.0),
            "degenerate fixture: the reference produced all zeros"
        );
    }
}

/// The draft-chain gather must decode exactly the padded-Q6_K element map that
/// `q6k_gemv` reads, from a device-resident token id. Pinned BITWISE against a
/// CPU mirror of the kernel's own expression (`d * (sc*a) * scale`) over
/// synthesized blocks that exercise every quadrant, both halves, and all scale
/// groups, at the real head geometry (2816 = 11 super-blocks per row).
#[test]
#[ignore = "requires a CUDA device"]
fn q6k_row_gather_scale_matches_cpu_decode() {
    let Some(k) = kernels() else {
        return;
    };
    let hidden = 2816usize;
    let rows = 5usize;
    let n_sb = hidden / 256;
    let mut rng = Lcg(0x6e_77);
    // Q6_K wire super-block: ql[128] qh[64] scales[16] d(f16) = 210 bytes.
    let mut wire = vec![0u8; rows * n_sb * 210];
    for chunk in wire.chunks_mut(210) {
        for byte in chunk[..208].iter_mut() {
            *byte = rng.next_u8();
        }
        // A modest positive normal f16 keeps every product finite and nonzero.
        let d_bits = 0x2C00u16 | (u16::from(rng.next_u8()) & 0x03FF);
        chunk[208] = (d_bits & 0xFF) as u8;
        chunk[209] = (d_bits >> 8) as u8;
    }
    let padded = super::pad_q6k_blocks(&wire);
    let d_head = k.stream.clone_htod(&padded).unwrap();
    let scale = (hidden as f32).sqrt();
    for token in [0u32, 2, 4] {
        let d_token = k.stream.clone_htod(&[token]).unwrap();
        let mut d_out = k.stream.alloc_zeros::<f32>(hidden).unwrap();
        super::launch_q6k_row_gather_scale(
            &k.stream,
            &k.q6k_row_gather_scale,
            &d_head,
            &d_token,
            0,
            &mut d_out,
            hidden,
            scale,
        )
        .unwrap();
        let mut got = vec![0f32; hidden];
        k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
        k.ctx.synchronize().unwrap();

        let mut nonzero = 0usize;
        for (i, &got_bits) in got.iter().enumerate() {
            let sb = i >> 8;
            let r = i & 255;
            let block = &padded[(token as usize * n_sb + sb) * 224..][..224];
            let h = r >> 7;
            let rem = r & 127;
            let q = rem >> 5;
            let s = (rem >> 4) & 1;
            let l = rem & 15;
            let albyte = i32::from(block[h * 64 + if q & 1 == 1 { 32 } else { 0 } + s * 16 + l]);
            let hbyte = i32::from(block[128 + h * 32 + s * 16 + l]);
            let a = match q {
                0 => ((albyte & 0xF) | ((hbyte & 3) << 4)) - 32,
                1 => ((albyte & 0xF) | (((hbyte >> 2) & 3) << 4)) - 32,
                2 => ((albyte >> 4) | (((hbyte >> 4) & 3) << 4)) - 32,
                _ => ((albyte >> 4) | (((hbyte >> 6) & 3) << 4)) - 32,
            };
            let sc = i32::from(block[192 + 8 * h + s + 2 * q] as i8);
            let d = crate::inference::f16_bits_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let expect = d * ((sc * a) as f32) * scale;
            assert_eq!(
                got_bits.to_bits(),
                expect.to_bits(),
                "element {i} of row {token}: got {got_bits} expected {expect}"
            );
            if expect != 0.0 {
                nonzero += 1;
            }
        }
        assert!(
            nonzero > hidden / 2,
            "degenerate fixture: row {token} decoded mostly to zeros"
        );
    }
}

/// The routed-record SoA repack is a pure byte permutation, region by region.
///
/// Asserted with explicit index arithmetic rather than by calling the helper
/// back, so this still fails if `q4_0_record_wire_to_soa` is ever changed to
/// agree with a broken kernel — the same convention as
/// `q4_0_wire_to_soa_is_a_pure_permutation`. Also pins the mixed-record
/// contract (a non-Q4_0 down region stays byte-identical wire) and exercises
/// the real 3,345,408-byte record geometry. No GPU required.
#[test]
fn q4_0_record_wire_to_soa_is_a_pure_permutation() {
    const WIRE: usize = 18;
    let assert_region_soa = |wire: &[u8], soa: &[u8]| {
        assert_eq!(soa.len(), wire.len(), "SoA must not change the region size");
        let n = wire.len() / WIRE;
        let (quants, scales) = soa.split_at(n * 16);
        for b in 0..n {
            assert_eq!(
                &scales[b * 2..b * 2 + 2],
                &wire[b * WIRE..b * WIRE + 2],
                "block {b}: f16 scale bits must survive verbatim into the scale plane"
            );
            assert_eq!(
                &quants[b * 16..b * 16 + 16],
                &wire[b * WIRE + 2..b * WIRE + WIRE],
                "block {b}: the 16 nibble bytes must survive verbatim into the quant plane"
            );
        }
    };
    let invert_region = |soa: &[u8]| -> Vec<u8> {
        let n = soa.len() / WIRE;
        let (quants, scales) = soa.split_at(n * 16);
        let mut wire = vec![0u8; soa.len()];
        for b in 0..n {
            wire[b * WIRE..b * WIRE + 2].copy_from_slice(&scales[b * 2..b * 2 + 2]);
            wire[b * WIRE + 2..b * WIRE + WIRE].copy_from_slice(&quants[b * 16..b * 16 + 16]);
        }
        wire
    };

    // Small geometry with explicit arithmetic, then the real routed record
    // (gate_up 1408x88 + down 2816x22 = 3,345,408 bytes). One reused scratch
    // pins that a dirty scratch from a previous record cannot leak through.
    let mut scratch = Vec::new();
    for (gu_rows, gu_bpr, down_rows, down_bpr) in
        [(64usize, 2usize, 64usize, 1usize), (1408, 88, 2816, 22)]
    {
        let mut rng = Lcg(0x50_a2_01 ^ ((gu_rows as u64) << 16));
        let gu_wire = synth_q4_0_wire(gu_rows, gu_bpr, &mut rng);
        let down_wire = synth_q4_0_wire(down_rows, down_bpr, &mut rng);
        let gate_up = 0..gu_wire.len();
        let down = gu_wire.len()..gu_wire.len() + down_wire.len();
        let mut record = [gu_wire.as_slice(), down_wire.as_slice()].concat();
        if (gu_rows, gu_bpr) == (1408, 88) {
            assert_eq!(record.len(), 3_345_408, "real routed record byte length");
        }

        super::q4_0_record_wire_to_soa(
            &mut record,
            gate_up.clone(),
            true,
            down.clone(),
            true,
            &mut scratch,
        );
        assert_ne!(
            record[gate_up.clone()],
            gu_wire[..],
            "raw passthrough is the defect this repack exists to remove"
        );
        assert_region_soa(&gu_wire, &record[gate_up.clone()]);
        assert_region_soa(&down_wire, &record[down.clone()]);
        // Every input byte is accounted for exactly once: inverting both
        // regions reproduces the wire record byte for byte.
        assert_eq!(
            invert_region(&record[gate_up.clone()]),
            gu_wire,
            "gate_up permutation must be exactly invertible"
        );
        assert_eq!(
            invert_region(&record[down.clone()]),
            down_wire,
            "down permutation must be exactly invertible"
        );

        // The mixed layout (Q4_1 down, layers 0..=6 of the tracked artifact):
        // only the Q4_0 gate_up region moves; down stays byte-identical wire.
        let mut mixed = [gu_wire.as_slice(), down_wire.as_slice()].concat();
        super::q4_0_record_wire_to_soa(
            &mut mixed,
            gate_up.clone(),
            true,
            down.clone(),
            false,
            &mut scratch,
        );
        assert_eq!(mixed[gate_up.clone()], record[gate_up.clone()]);
        assert_eq!(
            mixed[down.clone()],
            down_wire[..],
            "a non-Q4_0 region must never be touched"
        );
    }
}

/// `q4_0_gemv_routed_soa` on per-slot repacked arenas must be BITWISE identical
/// to `q4_0_gemv_routed` on the raw-wire arenas.
///
/// Same four-step argument as the dense `q4_0_gemv_soa`: identical dp4a bytes,
/// identical per-block term, identical lane-0 increasing-b fold — only the load
/// instructions change. Exercised on both real routed geometries plus a small
/// synthetic one, with a permuted slot map and both `batched_input` forms
/// (shared gate/up activation and per-route down activation), because per-slot
/// plane bases and the `route * rows + row` output index are exactly what an
/// arena repack gets wrong.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemv_routed_soa_matches_wire() {
    let Some(k) = kernels() else {
        return;
    };
    for (rows, bpr, seed) in [
        (1408usize, 88usize, 0x50_a3_01u64),
        (2816, 22, 0x50_a3_02),
        (96, 8, 0x50_a3_03),
    ] {
        let experts = 4usize;
        let kdim = bpr * 32;
        let mut rng = Lcg(seed);

        let per_slot = rows * bpr * 18;
        assert_eq!(
            per_slot % 16,
            0,
            "slot stride must keep uint4 loads aligned"
        );
        let mut arena = Vec::with_capacity(experts * per_slot);
        for _ in 0..experts {
            arena.extend_from_slice(&synth_q4_0_wire(rows, bpr, &mut rng));
        }
        let mut arena_soa = vec![0u8; arena.len()];
        for slot in 0..experts {
            super::q4_0_wire_to_soa_into(
                &arena[slot * per_slot..(slot + 1) * per_slot],
                &mut arena_soa[slot * per_slot..(slot + 1) * per_slot],
            );
        }

        // Per-route activations; batched_input=0 reads only route 0's view.
        let mut in_s = vec![0f32; experts * bpr];
        let mut in_q = vec![0i8; experts * kdim];
        for t in 0..experts {
            let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
            let q8 = crate::inference::quantize_q8_0_blocks(&act);
            for (b, blk) in q8.iter().enumerate() {
                in_s[t * bpr + b] = blk.scale;
                in_q[t * kdim + b * 32..t * kdim + (b + 1) * 32].copy_from_slice(&blk.quants);
            }
        }
        let slots: Vec<i32> = vec![2, 0, 3, 1];
        let routes: Vec<i32> = vec![1, 3, 0, 2];

        let d_s = k.stream.clone_htod(&in_s).unwrap();
        let d_q = k.stream.clone_htod(&in_q).unwrap();
        let d_slots = k.stream.clone_htod(&slots).unwrap();
        let d_routes = k.stream.clone_htod(&routes).unwrap();

        for batched_input in [0i32, 1] {
            let run = |weights: &[u8], func: &cudarc::driver::CudaFunction| -> Vec<f32> {
                use cudarc::driver::{LaunchConfig, PushKernelArg};
                let d_w = k.stream.clone_htod(weights).unwrap();
                let mut d_out = k.stream.alloc_zeros::<f32>(experts * rows).unwrap();
                let block = 256u32;
                let warps = block / 32;
                let cfg = LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(warps), experts as u32, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: bpr as u32 * 32 + bpr as u32 * 4 + warps * bpr as u32 * 4,
                };
                let (stride, rows_i, bpr_i, experts_i) =
                    (per_slot as u64, rows as i32, bpr as i32, experts as i32);
                let mut b = k.stream.launch_builder(func);
                b.arg(&d_s)
                    .arg(&d_q)
                    .arg(&d_w)
                    .arg(&d_slots)
                    .arg(&d_routes)
                    .arg(&stride)
                    .arg(&rows_i)
                    .arg(&bpr_i)
                    .arg(&mut d_out)
                    .arg(&experts_i)
                    .arg(&batched_input);
                unsafe { b.launch(cfg) }.unwrap();
                let mut host = vec![0f32; experts * rows];
                k.stream.memcpy_dtoh(&d_out, &mut host).unwrap();
                k.ctx.synchronize().unwrap();
                host
            };

            let from_wire = run(&arena, &k.q4_0_gemv_routed);
            let from_soa = run(&arena_soa, &k.q4_0_gemv_routed_soa);
            assert_same_bits(
                &format!("q4_0_gemv_routed_soa {rows}x{bpr} batched_input={batched_input}"),
                &from_soa,
                &from_wire,
            );
            assert!(
                from_wire.iter().any(|v| *v != 0.0),
                "degenerate fixture: the wire reference produced all zeros"
            );
        }
    }
}

/// The SoA routed GEMMs (plain and chunked32) on per-slot repacked arenas must
/// be BITWISE identical to `q4_0_gemm_routed` on the raw-wire arena.
///
/// Modeled on `q4_0_gemm_routed_matches_gemv`: RAGGED CSR (experts with 0, 1,
/// and many tokens), a permuted slot map, both real routed shapes (1408x88 and
/// 2816x22) plus a synthetic one, an explicit tile-2 launch so the `blockIdx.z`
/// tiling path runs, and the production launch helper for both SoA kernels.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemm_routed_soa_matches_wire() {
    let Some(k) = kernels() else {
        return;
    };
    for (rows, bpr, seed) in [
        (1408usize, 88usize, 0x50_a4_01u64),
        (2816, 22, 0x50_a4_02),
        (96, 8, 0x50_a4_03),
    ] {
        let experts = 4usize;
        let n_tokens = 7usize;
        let kdim = bpr * 32;
        let mut rng = Lcg(seed);

        let per_slot = rows * bpr * 18;
        assert_eq!(
            per_slot % 16,
            0,
            "slot stride must keep uint4 loads aligned"
        );
        let mut arena = Vec::with_capacity(experts * per_slot);
        for _ in 0..experts {
            arena.extend_from_slice(&synth_q4_0_wire(rows, bpr, &mut rng));
        }
        let mut arena_soa = vec![0u8; arena.len()];
        for slot in 0..experts {
            super::q4_0_wire_to_soa_into(
                &arena[slot * per_slot..(slot + 1) * per_slot],
                &mut arena_soa[slot * per_slot..(slot + 1) * per_slot],
            );
        }
        let mut in_s = vec![0f32; n_tokens * bpr];
        let mut in_q = vec![0i8; n_tokens * kdim];
        for t in 0..n_tokens {
            let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
            let q8 = crate::inference::quantize_q8_0_blocks(&act);
            for (b, blk) in q8.iter().enumerate() {
                in_s[t * bpr + b] = blk.scale;
                in_q[t * kdim + b * 32..t * kdim + (b + 1) * 32].copy_from_slice(&blk.quants);
            }
        }
        // Ragged CSR: expert 0 -> 3 tokens, 1 -> 0, 2 -> 1, 3 -> 4. Total 8.
        let token_offsets: Vec<i32> = vec![0, 3, 3, 4, 8];
        let token_ids: Vec<i32> = vec![5, 0, 3, 6, 2, 4, 1, 5];
        let slots: Vec<i32> = vec![2, 0, 3, 1];
        let assignments = *token_offsets.last().unwrap() as usize;
        let max_count = (0..experts)
            .map(|e| token_offsets[e + 1] - token_offsets[e])
            .max()
            .unwrap() as usize;

        let d_s = k.stream.clone_htod(&in_s).unwrap();
        let d_q = k.stream.clone_htod(&in_q).unwrap();
        let d_w = k.stream.clone_htod(&arena).unwrap();
        let d_w_soa = k.stream.clone_htod(&arena_soa).unwrap();
        let d_slots = k.stream.clone_htod(&slots).unwrap();
        let d_off = k.stream.clone_htod(&token_offsets).unwrap();
        let d_tok = k.stream.clone_htod(&token_ids).unwrap();

        // Reference: the shipped wire GEMM (itself pinned bitwise against
        // per-token q4_0_gemv_routed by q4_0_gemm_routed_matches_gemv).
        let mut d_ref = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
        super::launch_q4_0_gemm_routed(
            &k.stream,
            &k.q4_0_gemm_routed,
            &d_s,
            &d_q,
            &d_w,
            &d_slots,
            &d_off,
            &d_tok,
            per_slot,
            rows,
            bpr,
            experts,
            max_count,
            false,
            &mut d_ref,
        )
        .unwrap();
        let mut reference = vec![0f32; assignments * rows];
        k.stream.memcpy_dtoh(&d_ref, &mut reference).unwrap();
        k.ctx.synchronize().unwrap();
        assert!(
            reference.iter().any(|v| *v != 0.0),
            "degenerate fixture: the wire reference produced all zeros"
        );

        // Explicit tile-2 launch of the SoA kernel so blockIdx.z tiling runs.
        let tile = 2usize;
        let mut d_tiled = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
        {
            use cudarc::driver::{LaunchConfig, PushKernelArg};
            let block = 256u32;
            let warps = block / 32;
            let cfg = LaunchConfig {
                grid_dim: (
                    (rows as u32).div_ceil(warps),
                    experts as u32,
                    max_count.div_ceil(tile) as u32,
                ),
                block_dim: (block, 1, 1),
                shared_mem_bytes: warps * tile as u32 * bpr as u32 * 4,
            };
            let (stride, rows_i, bpr_i, experts_i, tile_i) = (
                per_slot as u64,
                rows as i32,
                bpr as i32,
                experts as i32,
                tile as i32,
            );
            let mut b = k.stream.launch_builder(&k.q4_0_gemm_routed_soa);
            b.arg(&d_s)
                .arg(&d_q)
                .arg(&d_w_soa)
                .arg(&d_slots)
                .arg(&d_off)
                .arg(&d_tok)
                .arg(&stride)
                .arg(&rows_i)
                .arg(&bpr_i)
                .arg(&mut d_tiled)
                .arg(&experts_i)
                .arg(&tile_i);
            unsafe { b.launch(cfg) }.unwrap();
        }
        let mut tiled = vec![0f32; assignments * rows];
        k.stream.memcpy_dtoh(&d_tiled, &mut tiled).unwrap();
        k.ctx.synchronize().unwrap();
        assert_same_bits(
            &format!("q4_0_gemm_routed_soa tile-2 {rows}x{bpr}"),
            &tiled,
            &reference,
        );

        // Production launch helper, both SoA kernels: the pairing the runtime's
        // (chunked policy x arena-SoA gate) dispatch actually selects.
        for (label, func, chunked) in [
            (
                "q4_0_gemm_routed_soa helper",
                &k.q4_0_gemm_routed_soa,
                false,
            ),
            (
                "q4_0_gemm_routed_chunked_soa helper",
                &k.q4_0_gemm_routed_chunked_soa,
                true,
            ),
        ] {
            let mut d_out = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
            super::launch_q4_0_gemm_routed(
                &k.stream, func, &d_s, &d_q, &d_w_soa, &d_slots, &d_off, &d_tok, per_slot, rows,
                bpr, experts, max_count, chunked, &mut d_out,
            )
            .unwrap();
            let mut got = vec![0f32; assignments * rows];
            k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
            k.ctx.synchronize().unwrap();
            assert_same_bits(&format!("{label} {rows}x{bpr}"), &got, &reference);
        }
    }
}

/// `q4_0_gemv_routed_rows` must be BITWISE identical to `q4_0_gemv_routed`.
///
/// Step 08 moves the tail fold from one lane to R lanes, but each row is still
/// summed by a single lane in increasing block order, so every row's f32
/// association is unchanged. That is the whole parity argument, and this is what
/// holds it. Exercised with >= 3 slots and a permuted route order, because the
/// per-slot expert base and the `route * rows + row` output index are exactly
/// what a rows-per-warp rewrite gets wrong, and a slot-0-only test cannot see it.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemv_routed_rows_matches_scalar() {
    let Some(k) = kernels() else {
        return;
    };
    // Both real routed geometries: gate_up (1408 rows x 88 blocks) and down (2816 x 22).
    for (rows, bpr, seed) in [(1408usize, 88usize, 0x08_a1u64), (2816, 22, 0x08_a2)] {
        let experts = 4usize;
        let kdim = bpr * 32;
        let mut rng = Lcg(seed);

        // One arena of `experts` slots, each a full independent weight block.
        let per_slot = rows * bpr * 18;
        let mut arena = Vec::with_capacity(experts * per_slot);
        for _ in 0..experts {
            arena.extend_from_slice(&synth_q4_0_wire(rows, bpr, &mut rng));
        }
        let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let q8 = crate::inference::quantize_q8_0_blocks(&act);
        let scales: Vec<f32> = q8.iter().map(|b| b.scale).collect();
        let mut quants = vec![0i8; kdim];
        for (b, blk) in q8.iter().enumerate() {
            quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
        }
        // Permuted, non-identity slot/route mapping.
        let slots: Vec<i32> = vec![2, 0, 3, 1];
        let routes: Vec<i32> = vec![1, 3, 0, 2];

        let d_s = k.stream.clone_htod(&scales).unwrap();
        let d_q = k.stream.clone_htod(&quants).unwrap();
        let d_w = k.stream.clone_htod(&arena).unwrap();
        let d_slots = k.stream.clone_htod(&slots).unwrap();
        let d_routes = k.stream.clone_htod(&routes).unwrap();

        let run = |func: &cudarc::driver::CudaFunction, rows_per_warp: u32| -> Vec<f32> {
            use cudarc::driver::{LaunchConfig, PushKernelArg};
            let block = 256u32;
            let warps = block / 32;
            let mut d_out = k.stream.alloc_zeros::<f32>(experts * rows).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (
                    (rows as u32).div_ceil(warps * rows_per_warp),
                    experts as u32,
                    1,
                ),
                block_dim: (block, 1, 1),
                shared_mem_bytes: bpr as u32 * 32
                    + bpr as u32 * 4
                    + if rows_per_warp > 1 {
                        warps * rows_per_warp * 32
                    } else {
                        warps * bpr as u32
                    } * 4,
            };
            let stride_u64 = per_slot as u64;
            let (rows_i, bpr_i, experts_i, batched) =
                (rows as i32, bpr as i32, experts as i32, 0i32);
            let rows_per_warp_i = rows_per_warp as i32;
            let mut b = k.stream.launch_builder(func);
            b.arg(&d_s)
                .arg(&d_q)
                .arg(&d_w)
                .arg(&d_slots)
                .arg(&d_routes)
                .arg(&stride_u64)
                .arg(&rows_i)
                .arg(&bpr_i)
                .arg(&mut d_out)
                .arg(&experts_i)
                .arg(&batched);
            if rows_per_warp > 1 {
                b.arg(&rows_per_warp_i);
            }
            unsafe { b.launch(cfg) }.unwrap();
            let mut host = vec![0f32; experts * rows];
            k.stream.memcpy_dtoh(&d_out, &mut host).unwrap();
            k.ctx.synchronize().unwrap();
            host
        };

        let scalar = run(&k.q4_0_gemv_routed, 1);
        let rowsk = run(&k.q4_0_gemv_routed_rows, 8);

        let mismatches = scalar
            .iter()
            .zip(&rowsk)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "q4_0_gemv_routed_rows must be BITWISE identical to q4_0_gemv_routed \
             (rows {rows}, blocks/row {bpr}): {mismatches}/{} outputs differ",
            scalar.len()
        );
        // Guard against both kernels being trivially zero.
        assert!(
            scalar.iter().any(|v| *v != 0.0),
            "the scalar routed GEMV produced all zeros; the test proves nothing"
        );
    }
}

/// The Q4_0 quants-first SoA repack is a pure byte permutation.
///
/// Asserted with explicit index arithmetic rather than by calling the helper
/// back, so this still fails if `q4_0_wire_to_soa` is ever changed to agree with
/// a broken kernel — the same convention as
/// `gemma4_head_upload_matches_each_lane_gemv_layout`. No GPU required.
#[test]
fn q4_0_wire_to_soa_is_a_pure_permutation() {
    const WIRE: usize = 18;
    let (rows, bpr) = (7usize, 5usize);
    let n = rows * bpr;
    let mut rng = Lcg(0x40_50_a1);
    let wire = synth_q4_0_wire(rows, bpr, &mut rng);
    let soa = super::q4_0_wire_to_soa(&wire);

    assert_eq!(
        soa.len(),
        wire.len(),
        "SoA must not change size: residency and every slot budget depend on 18 B/block"
    );
    assert_ne!(
        soa, wire,
        "raw passthrough is the defect this repack exists to remove"
    );

    let (quants, scales) = soa.split_at(n * 16);
    for b in 0..n {
        assert_eq!(
            &scales[b * 2..b * 2 + 2],
            &wire[b * WIRE..b * WIRE + 2],
            "block {b}: f16 scale bits must survive verbatim into the scale plane"
        );
        assert_eq!(
            &quants[b * 16..b * 16 + 16],
            &wire[b * WIRE + 2..b * WIRE + WIRE],
            "block {b}: the 16 nibble bytes must survive verbatim into the quant plane"
        );
        assert_eq!(
            (b * 16) % 16,
            0,
            "block {b}: quant plane offset must be 16-byte aligned for the uint4 load"
        );
    }

    // Every input byte is accounted for exactly once.
    let mut back = vec![0u8; wire.len()];
    for b in 0..n {
        back[b * WIRE..b * WIRE + 2].copy_from_slice(&scales[b * 2..b * 2 + 2]);
        back[b * WIRE + 2..b * WIRE + WIRE].copy_from_slice(&quants[b * 16..b * 16 + 16]);
    }
    assert_eq!(back, wire, "the permutation must be exactly invertible");
}

/// `q4_0_gemv_soa` on repacked weights must be BITWISE identical to `q4_0_gemv`
/// on the raw wire — not merely close.
///
/// This is the gate that lets the SoA repack ship as an exact-parity change: the
/// integer `__dp4a` chain is exact regardless of how the 16 bytes were loaded,
/// the per-block float term is unchanged, and the tail fold is still lane 0
/// summing in increasing block order. If any of that stops being true, the
/// greedy token stream moves and this test is what catches it.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemv_soa_matches_wire() {
    let Some(k) = kernels() else {
        return;
    };
    for (rows, bpr, seed) in [(96usize, 8usize, 0x50_a1_01u64), (257, 88, 0x50_a1_02)] {
        let kdim = bpr * 32;
        let mut rng = Lcg(seed);
        let wire = synth_q4_0_wire(rows, bpr, &mut rng);
        let soa = super::q4_0_wire_to_soa(&wire);

        let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let q8 = crate::inference::quantize_q8_0_blocks(&act);
        let scales: Vec<f32> = q8.iter().map(|b| b.scale).collect();
        let mut quants = vec![0i8; kdim];
        for (b, blk) in q8.iter().enumerate() {
            quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
        }

        let d_s = k.stream.clone_htod(&scales).unwrap();
        let d_q = k.stream.clone_htod(&quants).unwrap();

        let run = |weights: &[u8], func: &cudarc::driver::CudaFunction| -> Vec<f32> {
            let d_w = k.stream.clone_htod(weights).unwrap();
            let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
            super::launch_q4_0_gemv(
                &k.stream,
                func,
                &d_s,
                &d_q,
                &d_w.slice(0..weights.len()),
                rows,
                bpr,
                &mut d_out,
                0,
            )
            .unwrap();
            let mut host = vec![0f32; rows];
            k.stream.memcpy_dtoh(&d_out, &mut host).unwrap();
            k.ctx.synchronize().unwrap();
            host
        };

        let from_wire = run(&wire, &k.q4_0_gemv);
        let from_soa = run(&soa, &k.q4_0_gemv_soa);

        let mismatches = from_wire
            .iter()
            .zip(&from_soa)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches, 0,
            "q4_0_gemv_soa must be BITWISE identical to q4_0_gemv \
             (rows {rows}, blocks/row {bpr}): {mismatches}/{rows} rows differ"
        );

        // Second opinion: both must also track the CPU wire oracle.
        const WIRE: usize = 18;
        let row_bytes = bpr * WIRE;
        let cpu: Vec<f32> = (0..rows)
            .map(|r| {
                crate::inference::q4_0_wire_row_dot(&wire[r * row_bytes..(r + 1) * row_bytes], &q8)
            })
            .collect();
        assert!(
            close(&from_soa, &cpu, 1e-4),
            "q4_0_gemv_soa diverged from the CPU q4_0_wire_row_dot oracle"
        );
    }
}

/// Every contraction width used by the tracked Gemma 4 MTP assistant must
/// reproduce the ordered native-f32 oracle bit for bit. The odd row count also
/// exercises the final partially populated CTA for every launch geometry.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_f32_gemv_soa_matches_ordered_oracle_all_assistant_widths_bitwise() {
    let Some(kernels) = kernels() else {
        return;
    };
    const ROWS: usize = 257;
    for (case, cols) in [1024usize, 4096, 5632, 8192].into_iter().enumerate() {
        let bpr = cols / 32;
        let mut rng = Lcg(0x40_f3_20_00u64 + case as u64);
        let wire = synth_q4_0_wire(ROWS, bpr, &mut rng);
        let soa = super::q4_0_wire_to_soa(&wire);
        let input: Vec<f32> = (0..cols).map(|_| rng.next_f32() * 1.75).collect();
        let expected: Vec<f32> = (0..ROWS)
            .map(|row| q4_0_f32_soa_ordered_row_dot(&soa, ROWS, bpr, row, &input))
            .collect();
        let actual = run_q4_0_f32_gemv_soa(&kernels, &soa, &input, ROWS, cols);
        assert_same_bits(
            &format!("q4_0_f32_gemv_soa {ROWS}x{cols}"),
            &actual,
            &expected,
        );
    }
}

/// Covers the seven distinct matrix geometries shared by all 23 matrices in
/// the tracked assistant pack. The GPU executes every output row; sampling the
/// first, middle, and last CPU rows keeps this production-size receipt useful
/// on a 16 GiB host while still pinning row addressing and the tail CTA.
#[test]
#[ignore = "requires a CUDA device and allocates the production Q4_0 matrices"]
fn q4_0_f32_gemv_soa_matches_ordered_oracle_production_shapes() {
    let Some(kernels) = kernels() else {
        return;
    };
    for (case, (rows, cols)) in [
        (262_144usize, 1_024usize),
        (4_096, 1_024),
        (1_024, 4_096),
        (8_192, 1_024),
        (1_024, 8_192),
        (1_024, 5_632),
        (2_816, 1_024),
    ]
    .into_iter()
    .enumerate()
    {
        let bpr = cols / 32;
        let mut rng = Lcg(0x40_f3_70_00u64 + case as u64);
        let wire = synth_q4_0_wire(rows, bpr, &mut rng);
        let soa = super::q4_0_wire_to_soa(&wire);
        let input: Vec<f32> = (0..cols).map(|_| rng.next_f32() * 1.75).collect();
        let actual = run_q4_0_f32_gemv_soa(&kernels, &soa, &input, rows, cols);
        for row in [0usize, rows / 2, rows - 1] {
            let expected = q4_0_f32_soa_ordered_row_dot(&soa, rows, bpr, row, &input);
            assert_same_bits(
                &format!("q4_0_f32_gemv_soa {rows}x{cols} row {row}"),
                &actual[row..=row],
                &[expected],
            );
        }
    }
}

/// The verifier-width Q4_0 GEMM must consume the exact SoA layout used by
/// Gemma 4's resident common projections and reproduce repeated scalar
/// `q4_0_gemv_soa` launches bit for bit. Exercise the narrow, scheduled, and
/// maximum verifier widths against the production hidden=2816 contraction.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemm_batched_soa_variants_match_scalar_soa_bitwise() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 2816usize;
    let bpr = 88usize;
    let max_tokens = crate::gemma4_runtime::MAX_MTP_VERIFY_ROWS;
    assert_eq!(max_tokens, 14, "test cases pin the production verifier cap");
    let kdim = bpr * 32;
    let mut rng = Lcg(0x40_50_b9);
    let wire = synth_q4_0_wire(rows, bpr, &mut rng);
    let soa = super::q4_0_wire_to_soa(&wire);
    let d_w = k.stream.clone_htod(&soa).unwrap();

    let mut in_scales = vec![0f32; max_tokens * bpr];
    let mut in_quants = vec![0i8; max_tokens * kdim];
    for t in 0..max_tokens {
        let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let q8 = crate::inference::quantize_q8_0_blocks(&act);
        for (b, block) in q8.iter().enumerate() {
            in_scales[t * bpr + b] = block.scale;
            in_quants[t * kdim + b * 32..t * kdim + (b + 1) * 32].copy_from_slice(&block.quants);
        }
    }

    let mut scalar = vec![0f32; max_tokens * rows];
    for t in 0..max_tokens {
        let d_s = k
            .stream
            .clone_htod(&in_scales[t * bpr..(t + 1) * bpr])
            .unwrap();
        let d_q = k
            .stream
            .clone_htod(&in_quants[t * kdim..(t + 1) * kdim])
            .unwrap();
        let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
        super::launch_q4_0_gemv(
            &k.stream,
            &k.q4_0_gemv_soa,
            &d_s,
            &d_q,
            &d_w.slice(0..d_w.len()),
            rows,
            bpr,
            &mut d_out,
            0,
        )
        .unwrap();
        k.stream
            .memcpy_dtoh(&d_out, &mut scalar[t * rows..(t + 1) * rows])
            .unwrap();
    }

    let d_s = k.stream.clone_htod(&in_scales).unwrap();
    let d_q = k.stream.clone_htod(&in_quants).unwrap();
    if k.sm86 {
        assert!(
            k.q4_0_gemm_batched_soa_imma.is_some(),
            "SM86 must load the exact Q4_0 IMMA comparison kernel"
        );
    }
    for k_tokens in [1usize, 7, max_tokens] {
        let mut d_shared = k.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        super::launch_q4_0_gemm_batched_soa_shared(
            &k.stream,
            &k.q4_0_gemm_batched_soa_shared,
            &d_s,
            &d_q,
            &d_w,
            rows,
            bpr,
            k_tokens,
            &mut d_shared,
        )
        .unwrap();
        let mut d_zero_shared = k.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        super::launch_q4_0_gemm_batched_soa(
            &k.stream,
            &k.q4_0_gemm_batched_soa,
            &d_s,
            &d_q,
            &d_w,
            rows,
            bpr,
            k_tokens,
            &mut d_zero_shared,
        )
        .unwrap();
        let mut d_imma = k
            .q4_0_gemm_batched_soa_imma
            .as_ref()
            .map(|_| k.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap());
        if let (Some(kernel), Some(output)) =
            (k.q4_0_gemm_batched_soa_imma.as_ref(), d_imma.as_mut())
        {
            super::launch_q4_0_gemm_batched_soa_imma(
                &k.stream, kernel, &d_s, &d_q, &d_w, rows, bpr, k_tokens, output,
            )
            .unwrap();
        }
        let mut shared = vec![0f32; k_tokens * rows];
        let mut zero_shared = vec![0f32; k_tokens * rows];
        let mut imma = d_imma.as_ref().map(|_| vec![0f32; k_tokens * rows]);
        k.stream.memcpy_dtoh(&d_shared, &mut shared).unwrap();
        k.stream
            .memcpy_dtoh(&d_zero_shared, &mut zero_shared)
            .unwrap();
        if let (Some(device), Some(host)) = (d_imma.as_ref(), imma.as_mut()) {
            k.stream.memcpy_dtoh(device, host).unwrap();
        }
        k.ctx.synchronize().unwrap();

        let scalar_prefix = &scalar[..k_tokens * rows];
        for (variant, batched) in [("shared", &shared), ("zero-shared", &zero_shared)] {
            let mismatches = batched
                .iter()
                .zip(scalar_prefix)
                .filter(|(batch, one)| batch.to_bits() != one.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "Q4_0 SoA {variant} K={k_tokens} GEMM must be BITWISE identical to repeated \
                 scalar SoA GEMV: {mismatches}/{} outputs differ",
                batched.len()
            );
        }
        assert_same_bits(
            &format!("Q4_0 SoA shared vs zero-shared K={k_tokens}"),
            &shared,
            &zero_shared,
        );
        if let Some(imma) = imma.as_ref() {
            let mismatches = imma
                .iter()
                .zip(scalar_prefix)
                .filter(|(batch, one)| batch.to_bits() != one.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "Q4_0 SoA IMMA K={k_tokens} GEMM must be BITWISE identical to repeated \
                 scalar SoA GEMV: {mismatches}/{} outputs differ",
                imma.len()
            );
            assert_same_bits(
                &format!("Q4_0 SoA shared vs IMMA K={k_tokens}"),
                &shared,
                imma,
            );
        }
    }
    assert!(
        scalar.iter().any(|value| *value != 0.0),
        "degenerate Q4_0 SoA fixture produced only zeros"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_gemm_batched_matches_scalar_bitwise() {
    // Odd BPR exercises Q4_0's alternating 18-byte row/block alignment; BPR=88
    // is the installed Gemma-4 target's hidden-width contraction geometry.
    run_q4_0_gemm_batched_case(2, 23, 0x40_ba_72);
    run_q4_0_gemm_batched_case(4, 88, 0x40_ba_74);
}

fn run_q4_1_gemm_batched_case(ktok: usize, bpr: usize, seed: u64) {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let kdim = bpr * 32;
    let mut rng = Lcg(seed);
    let wire = synth_q4_1_wire(rows, bpr, &mut rng);
    let mut in_scales = vec![0f32; ktok * bpr];
    let mut in_quants = vec![0i8; ktok * kdim];
    let mut cpu = vec![0f32; ktok * rows];
    let mut scalar_gpu = vec![0f32; ktok * rows];
    const WIRE: usize = 20;
    let row_bytes = bpr * WIRE;

    for t in 0..ktok {
        let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let q8 = crate::inference::quantize_q8_0_blocks(&act);
        let scales: Vec<f32> = q8.iter().map(|b| b.scale).collect();
        let mut quants = vec![0i8; kdim];
        for (b, blk) in q8.iter().enumerate() {
            quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
        }
        in_scales[t * bpr..(t + 1) * bpr].copy_from_slice(&scales);
        in_quants[t * kdim..(t + 1) * kdim].copy_from_slice(&quants);
        for r in 0..rows {
            cpu[t * rows + r] =
                crate::inference::q4_1_wire_row_dot(&wire[r * row_bytes..(r + 1) * row_bytes], &q8);
        }

        let d_s = k.stream.clone_htod(&scales).unwrap();
        let d_q = k.stream.clone_htod(&quants).unwrap();
        let d_w = k.stream.clone_htod(&wire).unwrap();
        let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
        super::launch_q4_1_gemv(
            &k.stream,
            &k.q4_1_gemv,
            &d_s,
            &d_q,
            &d_w.slice(0..wire.len()),
            rows,
            bpr,
            &mut d_out,
            0,
        )
        .unwrap();
        k.stream
            .memcpy_dtoh(&d_out, &mut scalar_gpu[t * rows..(t + 1) * rows])
            .unwrap();
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(ktok * rows).unwrap();
    super::launch_q4_1_gemm_batched(
        &k.stream,
        &k.q4_1_gemm_batched,
        &d_is,
        &d_iq,
        &d_w,
        rows,
        bpr,
        ktok,
        &mut d_out,
    )
    .unwrap();
    let mut batched = vec![0f32; ktok * rows];
    k.stream.memcpy_dtoh(&d_out, &mut batched).unwrap();
    k.ctx.synchronize().unwrap();

    assert!(
        batched
            .iter()
            .zip(&scalar_gpu)
            .all(|(batch, scalar)| batch.to_bits() == scalar.to_bits()),
        "Q4_1 batched GEMM must be bit-identical to K scalar GEMVs"
    );
    assert!(
        close(&batched, &cpu, 1e-4),
        "Q4_1 batched GEMM diverged from the CPU wire oracle"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn q4_1_gemm_batched_matches_scalar_bitwise() {
    run_q4_1_gemm_batched_case(2, 23, 0x41_ba_72);
    run_q4_1_gemm_batched_case(4, 88, 0x41_ba_74);
}

// Synthetic NVFP4 weight wire: rows*n_sb superblocks of 36 bytes (d[4] UE4M3 scales
// then qs[32] packed E2M1 nibbles). Scale bytes cycle a deliberately adversarial set
// — 0x00 zero, 0x01 subnormal, 0x08 min-normal, interior values, 0x7E max-normal
// (224), 0x7F flush->0.0, and 0xFF ->240.0. The two sentinels (0x7F/0xFF) appear ONLY
// here, below the load-time refusal seam: admitted files never carry them, but the
// kernel must still decode them PIN-CPU-BITWISE (0x7F->0, 0xFF->240 — NOT the pin's
// CUDA-intrinsic double-flush), and since kernel and oracle read the SAME bytes this
// wire exercises exactly that. qs bytes are random, so the full codebook (codes +-12)
// appears across blocks.
fn synth_nvfp4_wire(rows: usize, n_sb: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 36;
    const SCALES: [u8; 10] = [0x00, 0x01, 0x08, 0x2C, 0x40, 0x51, 0x66, 0x7E, 0x7F, 0xFF];
    let mut wire = vec![0u8; rows * n_sb * WIRE];
    for sb in 0..rows * n_sb {
        let blk = &mut wire[sb * WIRE..(sb + 1) * WIRE];
        for (s, b) in blk.iter_mut().take(4).enumerate() {
            *b = SCALES[(sb + s) % SCALES.len()];
        }
        for b in blk.iter_mut().skip(4) {
            *b = rng.next_u8();
        }
    }
    wire
}

// Split a Q8_0 activation into the (scales, concatenated i8 quants) buffers the GPU
// GEMV reads — the oracle format, quantized by the CPU `quantize_q8_0_blocks` which is
// bit-paired with the device `quantize_q8_0`, so kernel and oracle see identical
// integers/scales and the GEMV kernel alone is under test.
fn q8_activation_buffers(act: &[f32]) -> (Vec<crate::tensor::Q8_0Block>, Vec<f32>, Vec<i8>) {
    let q8 = crate::inference::quantize_q8_0_blocks(act);
    let scales: Vec<f32> = q8.iter().map(|b| b.scale).collect();
    let mut quants = vec![0i8; act.len()];
    for (b, blk) in q8.iter().enumerate() {
        quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
    }
    (q8, scales, quants)
}

// BASALT Phase 4 bit-parity GATE for the NVFP4 resident GEMV. Generates synthetic
// NVFP4 wire (incl. crafted sentinel/subnormal/max scales) + a Q8_0 activation, runs
// `nvfp4_gemv` on the GPU across several shapes (in_dim 64/128/2560/10240 — the last
// is the ffn_down worst case at bpr=320 — with odd row counts), and asserts each
// output row reproduces the validated CPU oracle `nvfp4_wire_row_dot` on the SAME
// bytes. The kernel mirrors the oracle's exact per-sub-block integer dot + ordered f32
// accumulation (superblock-major / sub-block-minor, the same ordered-sum contract as
// q8/q4_0/q4_1), so the result is EXPECTED 100% bit-identical; the 1e-4 close() is a
// compiler-robustness backstop only (no looser than Q4_K per conductor §6).
#[test]
#[ignore = "requires a CUDA device"]
fn nvfp4_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    // (rows, n_sb): 1/2/40/160 NVFP4 superblocks per row == in_dim 64/128/2560/10240.
    let cases: [(usize, usize); 4] = [(1, 1), (3, 2), (37, 40), (5, 160)];
    let mut total_rows = 0usize;
    let mut total_exact = 0usize;
    let mut worst = 0f32;
    for (ci, &(rows, n_sb)) in cases.iter().enumerate() {
        let bpr = n_sb * 2; // Q8_0 activation blocks per row (in_dim/32)
        let kdim = bpr * 32; // in_dim
        let mut rng = Lcg(0x4E_F4_00 + ci as u64);
        let wire = synth_nvfp4_wire(rows, n_sb, &mut rng);

        let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
        let (q8, in_scales, in_quants) = q8_activation_buffers(&act);
        assert_eq!(q8.len(), bpr);

        let row_bytes = n_sb * 36;
        let mut expected = vec![0f32; rows];
        for (r, slot) in expected.iter_mut().enumerate() {
            let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
            *slot = crate::inference::nvfp4_wire_row_dot(row_wire, &q8);
        }

        let d_is = k.stream.clone_htod(&in_scales).unwrap();
        let d_iq = k.stream.clone_htod(&in_quants).unwrap();
        let d_w = k.stream.clone_htod(&wire).unwrap();
        let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
        super::launch_nvfp4_gemv(
            &k.stream,
            &k.nvfp4_gemv,
            &d_is,
            &d_iq,
            &d_w.slice(0..wire.len()),
            rows,
            bpr,
            &mut d_out,
            0,
        )
        .unwrap();
        let mut got = vec![0f32; rows];
        k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
        k.ctx.synchronize().unwrap();

        for (g, e) in got.iter().zip(&expected) {
            if g.to_bits() == e.to_bits() {
                total_exact += 1;
            }
            let d = (g - e).abs() / e.abs().max(1.0);
            if d > worst {
                worst = d;
            }
        }
        total_rows += rows;
        assert!(
            close(&got, &expected, 1e-4),
            "nvfp4_gemv shape (rows={rows}, n_sb={n_sb}) diverged from oracle (worst rel {worst:.3e})"
        );
    }
    eprintln!(
        "nvfp4_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        total_exact, total_rows, worst
    );
    assert_eq!(
        total_exact, total_rows,
        "NVFP4 GEMV is an ordered-sum lane: every row must be BIT-identical to nvfp4_wire_row_dot"
    );
}

// BASALT Phase 4 — L3 I-nan-scale (below-the-refusal-seam decode semantics on the
// KERNEL): a crafted one-row wire whose sub-block 0 scales are exactly [0x7F, 0xFF,
// 0x00, 0x7E] with nonzero qs. The kernel must decode them pin-CPU-bitwise — raw 0x7F
// and 0x00 flush to 0.0 (their terms vanish), raw 0xFF decodes to 240.0 (D17/T5, NOT
// the pin CUDA-intrinsic 0xFF->0 double-flush) — which is proven by bit-matching the
// oracle on the same bytes: had the kernel copied the double-flush, the 240-scaled
// sub-block-1 term would vanish on the GPU and diverge from the oracle here.
#[test]
#[ignore = "requires a CUDA device"]
fn nvfp4_gemv_decodes_ue4m3_sentinels_pin_cpu_bitwise() {
    let Some(k) = kernels() else {
        return;
    };
    let (rows, n_sb) = (1usize, 2usize);
    let bpr = n_sb * 2;
    let kdim = bpr * 32;
    let mut rng = Lcg(0x5E_71_7E);
    let mut wire = synth_nvfp4_wire(rows, n_sb, &mut rng);
    // Superblock 0 sub-block scales, and nonzero codes so the 0xFF term is real.
    wire[0] = 0x7F;
    wire[1] = 0xFF;
    wire[2] = 0x00;
    wire[3] = 0x7E;
    for b in wire.iter_mut().take(36).skip(4) {
        *b = 0x77; // both nibbles = 7 => code +12
    }

    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let (q8, in_scales, in_quants) = q8_activation_buffers(&act);
    let want = crate::inference::nvfp4_wire_row_dot(&wire, &q8);

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_nvfp4_gemv(
        &k.stream,
        &k.nvfp4_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        bpr,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert_eq!(
        got[0].to_bits(),
        want.to_bits(),
        "kernel UE4M3 sentinel decode must be pin-CPU-bitwise (0x7F->0.0, 0xFF->240.0): \
         got {} want {want}",
        got[0]
    );
}

// BASALT Phase 4 — residual-fusion twin (F2 contract): `residual=1` must equal
// gemv-then-add. Seeds `output` with a base vector, runs the fused launch, and asserts
// each row bit-equals base + nvfp4_wire_row_dot.
#[test]
#[ignore = "requires a CUDA device"]
fn nvfp4_gemv_fuses_residual_add() {
    let Some(k) = kernels() else {
        return;
    };
    let (rows, n_sb) = (7usize, 3usize);
    let bpr = n_sb * 2;
    let kdim = bpr * 32;
    let mut rng = Lcg(0x4E_F4_AD);
    let wire = synth_nvfp4_wire(rows, n_sb, &mut rng);
    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let (q8, in_scales, in_quants) = q8_activation_buffers(&act);

    let base: Vec<f32> = (0..rows).map(|_| rng.next_f32() * 5.0).collect();
    let row_bytes = n_sb * 36;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = base[r] + crate::inference::nvfp4_wire_row_dot(row_wire, &q8);
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.clone_htod(&base).unwrap(); // residual=1 adds onto this
    super::launch_nvfp4_gemv(
        &k.stream,
        &k.nvfp4_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        bpr,
        &mut d_out,
        1,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    for (r, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "nvfp4_gemv residual fusion row {r}: got {g} want {e}"
        );
    }
}

// BASALT Phase 4 — L3 I-k-div lane-native guard: the launcher refuses an odd Q8_0-block
// count (in_dim % 64 != 0) with a typed `Nvfp4LaunchError::OddBlocksPerRow` BEFORE
// touching the GPU, because one 64-value NVFP4 superblock needs a whole pair of
// 32-value activation blocks. Fail-closed in every build profile (not a debug panic).
#[test]
#[ignore = "requires a CUDA device"]
fn nvfp4_gemv_requires_even_q8_blocks() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 4usize;
    let odd_bpr = 3usize; // 3 Q8_0 blocks = 96 values; not a whole 64-superblock
    let d_is = k.stream.clone_htod(&vec![0f32; odd_bpr]).unwrap();
    let d_iq = k.stream.clone_htod(&vec![0i8; odd_bpr * 32]).unwrap();
    let d_w = k.stream.clone_htod(&vec![0u8; rows * 2 * 36]).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    match super::launch_nvfp4_gemv(
        &k.stream,
        &k.nvfp4_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..d_w.len()),
        rows,
        odd_bpr,
        &mut d_out,
        0,
    ) {
        Err(super::Nvfp4LaunchError::OddBlocksPerRow(bpr)) => assert_eq!(bpr, odd_bpr),
        Err(super::Nvfp4LaunchError::Driver(e)) => panic!("expected OddBlocksPerRow, got {e}"),
        Ok(()) => panic!("odd blocks_per_row must refuse typed (I-k-div)"),
    }
}

// Synthetic Q6_K weight wire bytes: rows*n_sb super-blocks of 210 bytes each
// (ql[128] + qh[64] + scales(i8)[16] + d(f16)). Random payload with a small
// positive f16 super-scale so the products stay in a sane f32 range.
fn synth_q6k_wire(rows: usize, n_sb: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 210;
    let mut out = vec![0u8; rows * n_sb * WIRE];
    for sb in 0..rows * n_sb {
        let blk = &mut out[sb * WIRE..(sb + 1) * WIRE];
        for b in blk.iter_mut().take(208) {
            *b = rng.next_u8();
        }
        let d = (rng.next_f32().abs() * 0.03 + 0.001).min(0.1);
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        blk[208] = db[0];
        blk[209] = db[1];
    }
    out
}

// Bit-parity receipt for the Q6_K resident decode GEMV. Generates synthetic Q6_K
// wire bytes + a Q8_K activation, runs q6k_gemv on the GPU, and asserts each output
// row reproduces the validated CPU oracle `q6_k_wire_row_dot` on the SAME bytes. The
// kernel mirrors the oracle's ordered 8-lane f32 accumulation (weights pre-minus-32,
// no mins term), so the result is expected bit-identical within the q8 ordered-f32 tol.
#[test]
#[ignore = "requires a CUDA device"]
fn q6k_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let n_sb = 3usize; // contraction dim = 3*256 = 768
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x6b_6b_6b);

    let wire = synth_q6k_wire(rows, n_sb, &mut rng);
    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8k = crate::inference::quantize_q8_k_blocks(&act);
    assert_eq!(q8k.len(), n_sb);
    let in_scales: Vec<f32> = q8k.iter().map(|b| b.d).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8k.iter().enumerate() {
        in_quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
    }

    const WIRE: usize = 210;
    let row_bytes = n_sb * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::q6_k_wire_row_dot(row_wire, &q8k);
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    // The GPU-side q6k layout is the 224 B-padded wire (pad_q6k_blocks, as
    // repack_for_lane uploads it); the CPU oracle reads the raw 210 B wire.
    let d_w = k.stream.clone_htod(&super::pad_q6k_blocks(&wire)).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_q6k_gemv(
        &k.stream,
        &k.q6k_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        n_sb,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let mut worst = 0f32;
    let mut exact = 0usize;
    for (g, e) in got.iter().zip(&expected) {
        if g.to_bits() == e.to_bits() {
            exact += 1;
        }
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    eprintln!(
        "q6k_gemv_matches_oracle: {}/{} rows bit-identical, worst rel diff {:.3e}",
        exact, rows, worst
    );
    assert!(
        close(&got, &expected, 1e-4),
        "q6k_gemv diverged from q6_k_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

/// Batched Q6_K receipt over several independent Q8_K activation rows.
#[test]
#[ignore = "requires a CUDA device"]
fn q6k_gemm_batched_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    for (rows, n_sb, k_tokens, seed) in [
        (64usize, 3usize, 4usize, 0x6b_47_45_4d_4d),
        // Production Gemma 4 tied-head contraction: hidden=2816 => 11
        // Q8_K super-blocks, at the maximum 14-row verifier width.
        (
            64,
            11,
            crate::gemma4_runtime::MAX_MTP_VERIFY_ROWS,
            0x6b_31_31_4b_31_34,
        ),
    ] {
        let kdim = n_sb * 256;
        let mut rng = Lcg(seed);
        let wire = synth_q6k_wire(rows, n_sb, &mut rng);
        let weights = super::pad_q6k_blocks(&wire);
        let mut in_scales = Vec::with_capacity(k_tokens * n_sb);
        let mut in_quants = Vec::with_capacity(k_tokens * kdim);
        let mut expected = vec![0f32; k_tokens * rows];
        for t in 0..k_tokens {
            let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
            let q8k = crate::inference::quantize_q8_k_blocks(&act);
            for block in &q8k {
                in_scales.push(block.d);
                in_quants.extend_from_slice(&block.qs);
            }
            for row in 0..rows {
                let lo = row * n_sb * 210;
                expected[t * rows + row] =
                    crate::inference::q6_k_wire_row_dot(&wire[lo..lo + n_sb * 210], &q8k);
            }
        }

        let d_is = k.stream.clone_htod(&in_scales).unwrap();
        let d_iq = k.stream.clone_htod(&in_quants).unwrap();
        let d_w = k.stream.clone_htod(&weights).unwrap();
        let mut d_out = k.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        super::launch_kquant_gemm_batched(
            &k.stream,
            &k.q6k_gemm_batched,
            &d_is,
            &d_iq,
            &d_w,
            rows,
            n_sb,
            k_tokens,
            8,
            &mut d_out,
        )
        .unwrap();
        let mut got = vec![0f32; k_tokens * rows];
        k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
        k.ctx.synchronize().unwrap();

        let exact = got
            .iter()
            .zip(&expected)
            .filter(|(g, e)| g.to_bits() == e.to_bits())
            .count();
        eprintln!(
            "q6k_gemm_batched_matches_oracle: {exact}/{} outputs bit-identical",
            got.len()
        );
        assert!(
            close(&got, &expected, 1e-4),
            "batched Q6_K diverged from q6_k_wire_row_dot"
        );
    }
}

/// Exact A/B for the anchor-major Q6_K DP4A verifier kernel. The established
/// batched kernel is the production bit-pattern anchor; the CPU wire oracle is
/// retained as an independent numeric check because its test historically
/// permits the same 1e-4 tolerance as scalar q6k_gemv.
#[test]
#[ignore = "requires an SM61+ CUDA device"]
fn q6k_gemm_batched_anchor_dp4a_matches_production_bitwise() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 64usize;
    let n_sb = 11usize; // Gemma 4 hidden=2816 production contraction.
    let kdim = n_sb * 256;
    for (k_tokens, seed) in [
        (1usize, 0x6b_44_50_34_01u64),
        (7usize, 0x6b_44_50_34_07u64),
        (
            crate::gemma4_runtime::MAX_MTP_VERIFY_ROWS,
            0x6b_44_50_34_14u64,
        ),
    ] {
        let mut rng = Lcg(seed);
        let wire = synth_q6k_wire(rows, n_sb, &mut rng);
        let weights = super::pad_q6k_blocks(&wire);
        let mut in_scales = Vec::with_capacity(k_tokens * n_sb);
        let mut in_quants = Vec::with_capacity(k_tokens * kdim);
        let mut expected = vec![0f32; k_tokens * rows];
        for t in 0..k_tokens {
            let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
            let q8k = crate::inference::quantize_q8_k_blocks(&act);
            for block in &q8k {
                in_scales.push(block.d);
                in_quants.extend_from_slice(&block.qs);
            }
            for row in 0..rows {
                let lo = row * n_sb * 210;
                expected[t * rows + row] =
                    crate::inference::q6_k_wire_row_dot(&wire[lo..lo + n_sb * 210], &q8k);
            }
        }

        let d_is = k.stream.clone_htod(&in_scales).unwrap();
        let d_iq = k.stream.clone_htod(&in_quants).unwrap();
        let d_w = k.stream.clone_htod(&weights).unwrap();
        let mut d_production = k.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        let mut d_anchor = k.stream.alloc_zeros::<f32>(k_tokens * rows).unwrap();
        super::launch_kquant_gemm_batched(
            &k.stream,
            &k.q6k_gemm_batched,
            &d_is,
            &d_iq,
            &d_w,
            rows,
            n_sb,
            k_tokens,
            8,
            &mut d_production,
        )
        .unwrap();
        super::launch_kquant_gemm_batched(
            &k.stream,
            &k.q6k_gemm_batched_anchor_dp4a,
            &d_is,
            &d_iq,
            &d_w,
            rows,
            n_sb,
            k_tokens,
            0,
            &mut d_anchor,
        )
        .unwrap();

        let mut production = vec![0f32; k_tokens * rows];
        let mut anchor = vec![0f32; k_tokens * rows];
        k.stream
            .memcpy_dtoh(&d_production, &mut production)
            .unwrap();
        k.stream.memcpy_dtoh(&d_anchor, &mut anchor).unwrap();
        k.ctx.synchronize().unwrap();
        assert_same_bits(
            &format!("Q6_K production vs anchor DP4A K={k_tokens}"),
            &anchor,
            &production,
        );
        assert!(
            close(&anchor, &expected, 1e-4),
            "anchor-major Q6_K DP4A K={k_tokens} diverged from the CPU wire oracle"
        );
    }
}

fn synth_iq4xs_wire(rows: usize, n_sb: usize, rng: &mut Lcg) -> Vec<u8> {
    const WIRE: usize = 136;
    let mut out = vec![0u8; rows * n_sb * WIRE];
    for sb in 0..rows * n_sb {
        let blk = &mut out[sb * WIRE..(sb + 1) * WIRE];
        // Random scales_h (+2), scales_l (+4), and qs (+8) exercise every sub-block
        // scale split and codebook index; d (+0) is a sane f16 super-block scale.
        for b in blk.iter_mut().skip(2) {
            *b = rng.next_u8();
        }
        let d = (rng.next_f32().abs() * 0.03 + 0.001).min(0.1);
        let db = crate::inference::f32_to_f16_bits(d).to_le_bytes();
        blk[0] = db[0];
        blk[1] = db[1];
    }
    out
}

// Parity receipt for the IQ4_XS resident decode GEMV. Generates synthetic IQ4_XS
// wire bytes + a Q8_K activation, runs iq4xs_gemv on the GPU, and asserts each output
// row reproduces the validated CPU oracle `iq4_xs_wire_row_dot` on the SAME bytes,
// within the same 1e-4 relative tolerance as the other resident K-quant lanes (the
// integer sub-block dots are exact; only the warp-reduce f32 order differs).
#[test]
#[ignore = "requires a CUDA device"]
fn iq4xs_gemv_matches_oracle() {
    let Some(k) = kernels() else {
        return;
    };
    let rows = 96usize;
    let n_sb = 3usize; // contraction dim = 3*256 = 768
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x1a4_5eed_u64);

    let wire = synth_iq4xs_wire(rows, n_sb, &mut rng);
    let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
    let q8k = crate::inference::quantize_q8_k_blocks(&act);
    assert_eq!(q8k.len(), n_sb);
    let in_scales: Vec<f32> = q8k.iter().map(|b| b.d).collect();
    let mut in_quants = vec![0i8; kdim];
    for (b, blk) in q8k.iter().enumerate() {
        in_quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
    }

    const WIRE: usize = 136;
    let row_bytes = n_sb * WIRE;
    let mut expected = vec![0f32; rows];
    for (r, slot) in expected.iter_mut().enumerate() {
        let row_wire = &wire[r * row_bytes..(r + 1) * row_bytes];
        *slot = crate::inference::iq4_xs_wire_row_dot(row_wire, &q8k);
    }

    let d_is = k.stream.clone_htod(&in_scales).unwrap();
    let d_iq = k.stream.clone_htod(&in_quants).unwrap();
    // IQ4_XS uploads the RAW 136-byte wire (repack_for_lane passes it through).
    let d_w = k.stream.clone_htod(&wire).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
    super::launch_iq4xs_gemv(
        &k.stream,
        &k.iq4xs_gemv,
        &d_is,
        &d_iq,
        &d_w.slice(0..wire.len()),
        rows,
        n_sb,
        &mut d_out,
        0,
    )
    .unwrap();
    let mut got = vec![0f32; rows];
    k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    k.ctx.synchronize().unwrap();

    let mut worst = 0f32;
    for (g, e) in got.iter().zip(&expected) {
        let d = (g - e).abs() / e.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    assert!(
        close(&got, &expected, 1e-4),
        "iq4xs_gemv diverged from iq4_xs_wire_row_dot oracle (worst rel {worst:.3e})"
    );
}

// ---- qwen35 (Ornith) gated-delta-net SSM kernels ---------------------------

#[test]
#[ignore = "requires a CUDA device"]
fn ssm_l2_norm_per_head_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let nk = 4usize;
    let hd = 128usize;
    let eps = 1e-6f32;
    let mut rng = Lcg(11);
    let buf: Vec<f32> = (0..nk * hd).map(|_| rng.next_f32()).collect();
    // CPU: double-precision sum, fmax(eps) — matches l2_norm_inplace.
    let mut expected = buf.clone();
    for h in 0..nk {
        let s = &mut expected[h * hd..(h + 1) * hd];
        let ss: f64 = s.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let scale = 1.0f32 / (ss as f32).sqrt().max(eps);
        for v in s.iter_mut() {
            *v *= scale;
        }
    }
    let mut dbuf = k.stream.clone_htod(&buf).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (nk as u32, 1, 1),
        block_dim: (hd as u32, 1, 1),
        shared_mem_bytes: (hd as u32) * 4,
    };
    let hdi = hd as i32;
    let mut b = k.stream.launch_builder(&k.ssm_l2_norm_per_head);
    b.arg(&mut dbuf).arg(&hdi).arg(&eps);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; nk * hd];
    k.stream.memcpy_dtoh(&dbuf, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(
        close(&got, &expected, 2e-3),
        "ssm_l2_norm_per_head diverged"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn ssm_conv1d_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let conv_dim = 256usize;
    let d_conv = 4usize;
    let cm1 = d_conv - 1;
    let mut rng = Lcg(23);
    let w: Vec<f32> = (0..conv_dim * d_conv).map(|_| rng.next_f32()).collect();
    let x: Vec<f32> = (0..conv_dim).map(|_| rng.next_f32()).collect();
    let st0: Vec<f32> = (0..conv_dim * cm1).map(|_| rng.next_f32()).collect();
    let silu = |v: f32| v / (1.0 + (-v).exp());
    // CPU reference (matches qwen35_ssm_compute conv loop).
    let mut exp_out = vec![0f32; conv_dim];
    let mut exp_st = st0.clone();
    for c in 0..conv_dim {
        let mut acc = 0.0f32;
        for t in 0..cm1 {
            acc += w[c * d_conv + t] * exp_st[c * cm1 + t];
        }
        acc += w[c * d_conv + cm1] * x[c];
        exp_out[c] = silu(acc);
        for t in 0..cm1 - 1 {
            exp_st[c * cm1 + t] = exp_st[c * cm1 + t + 1];
        }
        exp_st[c * cm1 + (cm1 - 1)] = x[c];
    }
    let dw = k.stream.clone_htod(&w).unwrap();
    let dx = k.stream.clone_htod(&x).unwrap();
    let mut dst = k.stream.clone_htod(&st0).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(conv_dim).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (conv_dim.div_ceil(128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let cdi = conv_dim as i32;
    let dci = d_conv as i32;
    let mut b = k.stream.launch_builder(&k.ssm_conv1d);
    b.arg(&dw)
        .arg(&dx)
        .arg(&mut dst)
        .arg(&mut dout)
        .arg(&cdi)
        .arg(&dci);
    unsafe { b.launch(cfg).unwrap() };
    let mut got_out = vec![0f32; conv_dim];
    let mut got_st = vec![0f32; conv_dim * cm1];
    k.stream.memcpy_dtoh(&dout, &mut got_out).unwrap();
    k.stream.memcpy_dtoh(&dst, &mut got_st).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got_out, &exp_out, 2e-3), "ssm_conv1d out diverged");
    assert!(
        close(&got_st, &exp_st, 2e-3),
        "ssm_conv1d ring-state diverged"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn ssm_delta_rule_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let ds = 128usize;
    let nk = 16usize;
    let nv = 32usize;
    let eps = 1e-6f32;
    let mut rng = Lcg(0x5511);
    let state0: Vec<f32> = (0..nv * ds * ds).map(|_| rng.next_f32()).collect();
    let kc: Vec<f32> = (0..nk * ds).map(|_| rng.next_f32()).collect();
    let qc: Vec<f32> = (0..nk * ds).map(|_| rng.next_f32()).collect();
    let vc: Vec<f32> = (0..nv * ds).map(|_| rng.next_f32()).collect();
    let z: Vec<f32> = (0..nv * ds).map(|_| rng.next_f32()).collect();
    let beta: Vec<f32> = (0..nv).map(|_| rng.next_f32() * 0.5 + 0.5).collect(); // (0,1)
    let decay: Vec<f32> = (0..nv)
        .map(|_| (-(rng.next_f32() * 0.5 + 0.5) * 0.1).exp())
        .collect(); // (0,1]
    let norm: Vec<f32> = (0..ds).map(|_| rng.next_f32() * 0.5 + 1.0).collect();
    let silu = |v: f32| v / (1.0 + (-v).exp());
    // CPU reference (matches qwen35_ssm_compute recurrence + gated RMSNorm).
    let mut exp_state = state0.clone();
    let mut exp_out = vec![0f32; nv * ds];
    let qscale = 1.0f32 / (ds as f32).sqrt();
    for h in 0..nv {
        let hk = h % nk;
        let g = decay[h];
        let bh = beta[h];
        let st = &mut exp_state[h * ds * ds..(h + 1) * ds * ds];
        for s in st.iter_mut() {
            *s *= g;
        }
        let mut sk = vec![0f32; ds];
        for i in 0..ds {
            let ki = kc[hk * ds + i];
            for j in 0..ds {
                sk[j] += st[i * ds + j] * ki;
            }
        }
        let mut dvec = vec![0f32; ds];
        for j in 0..ds {
            dvec[j] = (vc[h * ds + j] - sk[j]) * bh;
        }
        for i in 0..ds {
            let ki = kc[hk * ds + i];
            for j in 0..ds {
                st[i * ds + j] += ki * dvec[j];
            }
        }
        let mut o = vec![0f32; ds];
        for i in 0..ds {
            let qi = qc[hk * ds + i] * qscale;
            for j in 0..ds {
                o[j] += st[i * ds + j] * qi;
            }
        }
        let ss: f32 = o.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / ds as f32 + eps).sqrt();
        for j in 0..ds {
            exp_out[h * ds + j] = (o[j] * inv * norm[j]) * silu(z[h * ds + j]);
        }
    }
    let mut dstate = k.stream.clone_htod(&state0).unwrap();
    let dk = k.stream.clone_htod(&kc).unwrap();
    let dq = k.stream.clone_htod(&qc).unwrap();
    let dv = k.stream.clone_htod(&vc).unwrap();
    let dz = k.stream.clone_htod(&z).unwrap();
    let dbeta = k.stream.clone_htod(&beta).unwrap();
    let ddecay = k.stream.clone_htod(&decay).unwrap();
    let dnorm = k.stream.clone_htod(&norm).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(nv * ds).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (nv as u32, 1, 1),
        block_dim: (ds as u32, 1, 1),
        shared_mem_bytes: (3 * ds as u32) * 4,
    };
    let dsi = ds as i32;
    let nki = nk as i32;
    let mut b = k.stream.launch_builder(&k.ssm_delta_rule);
    b.arg(&mut dstate)
        .arg(&dk)
        .arg(&dq)
        .arg(&dv)
        .arg(&dz)
        .arg(&dbeta)
        .arg(&ddecay)
        .arg(&dnorm)
        .arg(&mut dout)
        .arg(&dsi)
        .arg(&nki)
        .arg(&eps);
    unsafe { b.launch(cfg).unwrap() };
    let mut got_out = vec![0f32; nv * ds];
    let mut got_state = vec![0f32; nv * ds * ds];
    k.stream.memcpy_dtoh(&dout, &mut got_out).unwrap();
    k.stream.memcpy_dtoh(&dstate, &mut got_state).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(
        close(&got_out, &exp_out, 3e-3),
        "ssm_delta_rule output diverged"
    );
    assert!(
        close(&got_state, &exp_state, 3e-3),
        "ssm_delta_rule carried state diverged"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn qwen35_ssm_register_sharded_d128_matches_cpu_and_does_not_spill() {
    let Some(k) = kernels() else {
        return;
    };
    let (ds, nk, nv, k_tokens) = (128usize, 2usize, 4usize, 8usize);
    let key_dim = nk * ds;
    let value_dim = nv * ds;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;
    let mut rng = Lcg(0xB0_55_A1_27);
    let mut conv: Vec<f32> = (0..k_tokens * conv_dim)
        .map(|_| rng.next_f32() * 0.2)
        .collect();
    // The recurrence consumes already-normalized q/k vectors.
    for token in 0..k_tokens {
        for head in 0..nk {
            for offset in [0usize, key_dim] {
                let base = token * conv_dim + offset + head * ds;
                let row = &mut conv[base..base + ds];
                let ss: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
                let scale = 1.0f32 / (ss as f32).sqrt().max(eps);
                for v in row {
                    *v *= scale;
                }
            }
        }
    }
    let z: Vec<f32> = (0..k_tokens * value_dim).map(|_| rng.next_f32()).collect();
    let beta: Vec<f32> = (0..k_tokens * nv)
        .map(|_| 0.45 + rng.next_f32() * 0.1)
        .collect();
    let decay: Vec<f32> = (0..k_tokens * nv)
        .map(|_| 0.985 + rng.next_f32() * 0.005)
        .collect();
    let norm: Vec<f32> = (0..ds).map(|_| 1.0 + rng.next_f32() * 0.1).collect();
    let state0: Vec<f32> = (0..nv * ds * ds).map(|_| rng.next_f32() * 0.01).collect();

    // CPU row-major oracle.
    let mut expected_state = state0.clone();
    let mut expected_mix = vec![0.0f32; k_tokens * value_dim];
    let qscale = 1.0f32 / (ds as f32).sqrt();
    for token in 0..k_tokens {
        let token_conv = &conv[token * conv_dim..(token + 1) * conv_dim];
        for h in 0..nv {
            let hk = h % nk;
            let qv = &token_conv[hk * ds..(hk + 1) * ds];
            let kv = &token_conv[key_dim + hk * ds..key_dim + (hk + 1) * ds];
            let vv = &token_conv[2 * key_dim + h * ds..2 * key_dim + (h + 1) * ds];
            let st = &mut expected_state[h * ds * ds..(h + 1) * ds * ds];
            let g = decay[token * nv + h];
            let bh = beta[token * nv + h];
            let mut sk = [0.0f32; 128];
            for i in 0..ds {
                for j in 0..ds {
                    st[i * ds + j] *= g;
                    sk[j] += st[i * ds + j] * kv[i];
                }
            }
            let mut delta = [0.0f32; 128];
            for j in 0..ds {
                delta[j] = (vv[j] - sk[j]) * bh;
            }
            let mut raw = [0.0f32; 128];
            for i in 0..ds {
                for j in 0..ds {
                    st[i * ds + j] += kv[i] * delta[j];
                    raw[j] += st[i * ds + j] * (qv[i] * qscale);
                }
            }
            let ss: f32 = raw.iter().map(|v| v * v).sum();
            let scale = 1.0 / (ss / ds as f32 + eps).sqrt();
            for j in 0..ds {
                let zj = z[token * value_dim + h * ds + j];
                let silu = zj / (1.0 + (-zj).exp());
                expected_mix[token * value_dim + h * ds + j] = raw[j] * scale * norm[j] * silu;
            }
        }
    }

    // Fast state persists as [head][column][row].
    let mut state_t = vec![0.0f32; state0.len()];
    for h in 0..nv {
        for row in 0..ds {
            for col in 0..ds {
                state_t[(h * ds + col) * ds + row] = state0[(h * ds + row) * ds + col];
            }
        }
    }
    let mut dstate = k.stream.clone_htod(&state_t).unwrap();
    let dconv = k.stream.clone_htod(&conv).unwrap();
    let dz = k.stream.clone_htod(&z).unwrap();
    let dbeta = k.stream.clone_htod(&beta).unwrap();
    let ddecay = k.stream.clone_htod(&decay).unwrap();
    let dnorm = k.stream.clone_htod(&norm).unwrap();
    let mut draw = k.stream.alloc_zeros::<f32>(k_tokens * value_dim).unwrap();
    let mut dquants = k.stream.alloc_zeros::<i8>(k_tokens * value_dim).unwrap();
    let mut dscales = k
        .stream
        .alloc_zeros::<f32>(k_tokens * value_dim / 32)
        .unwrap();
    super::launch_qwen35_ssm_delta_rule_d128_batched(
        &k.stream,
        &k.qwen35_ssm_delta_rule_d128_batched,
        &mut dstate,
        &dconv,
        &dbeta,
        &ddecay,
        &mut draw,
        nk,
        nv,
        key_dim,
        value_dim,
        conv_dim,
        k_tokens,
    )
    .unwrap();
    super::launch_qwen35_ssm_rmsnorm_gate_q8_d128_batched(
        &k.stream,
        &k.qwen35_ssm_rmsnorm_gate_q8_d128_batched,
        &draw,
        &dz,
        &dnorm,
        &mut dquants,
        &mut dscales,
        nv,
        value_dim,
        k_tokens,
        eps,
    )
    .unwrap();
    let mut got_state_t = vec![0.0f32; state_t.len()];
    let mut got_quants = vec![0i8; k_tokens * value_dim];
    let mut got_scales = vec![0.0f32; k_tokens * value_dim / 32];
    k.stream.memcpy_dtoh(&dstate, &mut got_state_t).unwrap();
    k.stream.memcpy_dtoh(&dquants, &mut got_quants).unwrap();
    k.stream.memcpy_dtoh(&dscales, &mut got_scales).unwrap();
    k.ctx.synchronize().unwrap();

    let mut got_state = vec![0.0f32; state0.len()];
    for h in 0..nv {
        for row in 0..ds {
            for col in 0..ds {
                got_state[(h * ds + row) * ds + col] = got_state_t[(h * ds + col) * ds + row];
            }
        }
    }
    let mut got_mix = vec![0.0f32; k_tokens * value_dim];
    for (block, &scale) in got_scales.iter().enumerate() {
        for lane in 0..32 {
            got_mix[block * 32 + lane] = f32::from(got_quants[block * 32 + lane]) * scale;
        }
    }
    let state_max_abs = got_state
        .iter()
        .zip(&expected_state)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dot: f64 = got_mix
        .iter()
        .zip(&expected_mix)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let an: f64 = got_mix.iter().map(|v| f64::from(*v).powi(2)).sum();
    let bn: f64 = expected_mix.iter().map(|v| f64::from(*v).powi(2)).sum();
    let cosine = dot / (an.sqrt() * bn.sqrt()).max(f64::MIN_POSITIVE);
    let regs = k.qwen35_ssm_delta_rule_d128_batched.num_regs().unwrap();
    let local = k
        .qwen35_ssm_delta_rule_d128_batched
        .local_size_bytes()
        .unwrap();
    eprintln!(
        "Bonsai D128 register SSM: regs/thread={regs} local_bytes={local} state_max_abs={state_max_abs:.6} output_cosine={cosine:.9}"
    );
    assert!(
        local <= 16,
        "register SSM spilled {local} local bytes/thread"
    );
    assert!(
        state_max_abs < 2e-4,
        "register SSM state drift {state_max_abs}"
    );
    assert!(cosine > 0.9998, "register SSM output cosine {cosine}");
}

#[test]
#[ignore = "requires a CUDA device; performance diagnostic"]
fn qwen35_ssm_register_sharded_real_27b_speed_probe() {
    let Some(k) = kernels() else {
        return;
    };
    let (ds, nk, nv, k_tokens) = (128usize, 4usize, 48usize, 128usize);
    let key_dim = nk * ds;
    let value_dim = nv * ds;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;
    let conv = vec![0.0f32; k_tokens * conv_dim];
    let z = vec![0.0f32; k_tokens * value_dim];
    let beta = vec![0.5f32; k_tokens * nv];
    let decay = vec![0.99f32; k_tokens * nv];
    let norm = vec![1.0f32; ds];
    let dconv = k.stream.clone_htod(&conv).unwrap();
    let dz = k.stream.clone_htod(&z).unwrap();
    let dbeta = k.stream.clone_htod(&beta).unwrap();
    let ddecay = k.stream.clone_htod(&decay).unwrap();
    let dnorm = k.stream.clone_htod(&norm).unwrap();
    let mut generic_state = k.stream.alloc_zeros::<f32>(nv * ds * ds).unwrap();
    let mut fast_state = k.stream.alloc_zeros::<f32>(nv * ds * ds).unwrap();
    let mut generic_out = k.stream.alloc_zeros::<f32>(k_tokens * value_dim).unwrap();
    let mut raw = k.stream.alloc_zeros::<f32>(k_tokens * value_dim).unwrap();
    let mut quants = k.stream.alloc_zeros::<i8>(k_tokens * value_dim).unwrap();
    let mut scales = k
        .stream
        .alloc_zeros::<f32>(k_tokens * value_dim / 32)
        .unwrap();
    let mut generic = || {
        super::launch_ssm_delta_rule_batched(
            &k.stream,
            &k.ssm_delta_rule_batched,
            &mut generic_state,
            &dconv,
            &dz,
            &dbeta,
            &ddecay,
            &dnorm,
            &mut generic_out,
            ds,
            nk,
            nv,
            key_dim,
            value_dim,
            conv_dim,
            k_tokens,
            eps,
        )
        .unwrap();
        k.ctx.synchronize().unwrap();
    };
    let mut fast = || {
        super::launch_qwen35_ssm_delta_rule_d128_batched(
            &k.stream,
            &k.qwen35_ssm_delta_rule_d128_batched,
            &mut fast_state,
            &dconv,
            &dbeta,
            &ddecay,
            &mut raw,
            nk,
            nv,
            key_dim,
            value_dim,
            conv_dim,
            k_tokens,
        )
        .unwrap();
        super::launch_qwen35_ssm_rmsnorm_gate_q8_d128_batched(
            &k.stream,
            &k.qwen35_ssm_rmsnorm_gate_q8_d128_batched,
            &raw,
            &dz,
            &dnorm,
            &mut quants,
            &mut scales,
            nv,
            value_dim,
            k_tokens,
            eps,
        )
        .unwrap();
        k.ctx.synchronize().unwrap();
    };
    generic();
    fast();
    let mut generic_ms = Vec::new();
    let mut fast_ms = Vec::new();
    for _ in 0..5 {
        let start = std::time::Instant::now();
        generic();
        generic_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        let start = std::time::Instant::now();
        fast();
        fast_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    eprintln!(
        "Bonsai-27B SSM K128: generic={:.3}ms register+Q8={:.3}ms speedup={:.3}x",
        mean(&generic_ms),
        mean(&fast_ms),
        mean(&generic_ms) / mean(&fast_ms),
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn sigmoid_mul_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n = 512usize;
    let mut rng = Lcg(77);
    let out0: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let gate: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
    let expected: Vec<f32> = out0
        .iter()
        .zip(&gate)
        .map(|(o, g)| o * (1.0 / (1.0 + (-g).exp())))
        .collect();
    let mut dout = k.stream.clone_htod(&out0).unwrap();
    let dgate = k.stream.clone_htod(&gate).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let ni = n as i32;
    let mut b = k.stream.launch_builder(&k.sigmoid_mul);
    b.arg(&mut dout).arg(&dgate).arg(&ni);
    unsafe { b.launch(cfg).unwrap() };
    let mut got = vec![0f32; n];
    k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got, &expected, 2e-3), "sigmoid_mul diverged");
}

#[test]
#[ignore = "requires a CUDA device"]
fn ssm_gates_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let nv = 32usize;
    let mut rng = Lcg(31);
    let beta_raw: Vec<f32> = (0..nv).map(|_| rng.next_f32() * 3.0).collect();
    let alpha_raw: Vec<f32> = (0..nv).map(|_| rng.next_f32() * 3.0).collect();
    let dt_bias: Vec<f32> = (0..nv).map(|_| rng.next_f32()).collect();
    let a: Vec<f32> = (0..nv).map(|_| -(rng.next_f32() * 0.5 + 0.5)).collect(); // a = -exp(.) <= 0
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let exp_beta: Vec<f32> = beta_raw.iter().map(|&v| sigmoid(v)).collect();
    let exp_decay: Vec<f32> = (0..nv)
        .map(|h| (softplus(alpha_raw[h] + dt_bias[h]) * a[h]).exp())
        .collect();
    let dbr = k.stream.clone_htod(&beta_raw).unwrap();
    let dar = k.stream.clone_htod(&alpha_raw).unwrap();
    let ddt = k.stream.clone_htod(&dt_bias).unwrap();
    let da = k.stream.clone_htod(&a).unwrap();
    let mut dbeta = k.stream.alloc_zeros::<f32>(nv).unwrap();
    let mut ddecay = k.stream.alloc_zeros::<f32>(nv).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (nv as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nvi = nv as i32;
    let mut b = k.stream.launch_builder(&k.ssm_gates);
    b.arg(&dbr)
        .arg(&dar)
        .arg(&ddt)
        .arg(&da)
        .arg(&mut dbeta)
        .arg(&mut ddecay)
        .arg(&nvi);
    unsafe { b.launch(cfg).unwrap() };
    let mut got_beta = vec![0f32; nv];
    let mut got_decay = vec![0f32; nv];
    k.stream.memcpy_dtoh(&dbeta, &mut got_beta).unwrap();
    k.stream.memcpy_dtoh(&ddecay, &mut got_decay).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got_beta, &exp_beta, 2e-3), "ssm_gates beta diverged");
    assert!(
        close(&got_decay, &exp_decay, 2e-3),
        "ssm_gates decay diverged"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn deinterleave_qgate_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 16usize;
    let hd = 256usize;
    let mut rng = Lcg(91);
    let qg: Vec<f32> = (0..n_heads * 2 * hd).map(|_| rng.next_f32()).collect();
    let mut exp_q = vec![0f32; n_heads * hd];
    let mut exp_gate = vec![0f32; n_heads * hd];
    for h in 0..n_heads {
        let b = h * hd * 2;
        exp_q[h * hd..(h + 1) * hd].copy_from_slice(&qg[b..b + hd]);
        exp_gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[b + hd..b + 2 * hd]);
    }
    let dqg = k.stream.clone_htod(&qg).unwrap();
    let mut dq = k.stream.alloc_zeros::<f32>(n_heads * hd).unwrap();
    let mut dgate = k.stream.alloc_zeros::<f32>(n_heads * hd).unwrap();
    let cfg = LaunchConfig {
        grid_dim: ((n_heads * hd).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let nh = n_heads as i32;
    let hdi = hd as i32;
    let mut b = k.stream.launch_builder(&k.deinterleave_qgate);
    b.arg(&dqg).arg(&mut dq).arg(&mut dgate).arg(&nh).arg(&hdi);
    unsafe { b.launch(cfg).unwrap() };
    let mut got_q = vec![0f32; n_heads * hd];
    let mut got_gate = vec![0f32; n_heads * hd];
    k.stream.memcpy_dtoh(&dq, &mut got_q).unwrap();
    k.stream.memcpy_dtoh(&dgate, &mut got_gate).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got_q, &exp_q, 1e-6), "deinterleave_qgate q diverged");
    assert!(
        close(&got_gate, &exp_gate, 1e-6),
        "deinterleave_qgate gate diverged"
    );
}

// Composition test: the 4 SSM kernels (gates -> conv1d -> l2norm q/k -> delta_rule)
// chained into one SSM-layer `mix`, vs the CPU qwen35_ssm_compute core (fed the same
// post-projection qkv/z/beta_raw/alpha_raw). Proves the GPU orchestration — conv_out
// sub-range views, in-place per-head L2 norm, launch sequencing — matches the
// reference. This is the SSM half of the qwen35 forward branch, validated before the
// engine wiring.
#[test]
#[ignore = "requires a CUDA device"]
fn ssm_layer_chain_matches_cpu() {
    let Some(k) = kernels() else {
        return;
    };
    let ds = 128usize;
    let nk = 16usize;
    let nv = 32usize;
    let d_conv = 4usize;
    let cm1 = d_conv - 1;
    let key_dim = nk * ds; // 2048
    let value_dim = nv * ds; // 4096
    let conv_dim = 2 * key_dim + value_dim; // 8192
    let eps = 1e-6f32;
    let mut rng = Lcg(0xC0FFEE);
    let qkv: Vec<f32> = (0..conv_dim).map(|_| rng.next_f32()).collect();
    let z: Vec<f32> = (0..value_dim).map(|_| rng.next_f32()).collect();
    let beta_raw: Vec<f32> = (0..nv).map(|_| rng.next_f32() * 3.0).collect();
    let alpha_raw: Vec<f32> = (0..nv).map(|_| rng.next_f32() * 3.0).collect();
    let dt_bias: Vec<f32> = (0..nv).map(|_| rng.next_f32()).collect();
    let a_vec: Vec<f32> = (0..nv).map(|_| -(rng.next_f32() * 0.5 + 0.5)).collect();
    let conv_w: Vec<f32> = (0..conv_dim * d_conv).map(|_| rng.next_f32()).collect();
    let ssm_norm: Vec<f32> = (0..ds).map(|_| rng.next_f32() * 0.5 + 1.0).collect();
    let conv_state0: Vec<f32> = (0..conv_dim * cm1).map(|_| rng.next_f32()).collect();
    let state0: Vec<f32> = (0..nv * ds * ds).map(|_| rng.next_f32() * 0.1).collect();
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

    // ---- CPU reference (qwen35_ssm_compute core, post-projection) ----
    let mut beta = vec![0f32; nv];
    let mut decay = vec![0f32; nv];
    for h in 0..nv {
        beta[h] = sigmoid(beta_raw[h]);
        decay[h] = (softplus(alpha_raw[h] + dt_bias[h]) * a_vec[h]).exp();
    }
    let mut exp_cs = conv_state0.clone();
    let mut conv_out = vec![0f32; conv_dim];
    for c in 0..conv_dim {
        let mut acc = 0f32;
        for t in 0..cm1 {
            acc += conv_w[c * d_conv + t] * exp_cs[c * cm1 + t];
        }
        acc += conv_w[c * d_conv + cm1] * qkv[c];
        conv_out[c] = silu(acc);
        for t in 0..cm1 - 1 {
            exp_cs[c * cm1 + t] = exp_cs[c * cm1 + t + 1];
        }
        exp_cs[c * cm1 + (cm1 - 1)] = qkv[c];
    }
    let mut q_conv = conv_out[0..key_dim].to_vec();
    let mut k_conv = conv_out[key_dim..2 * key_dim].to_vec();
    let v_conv = conv_out[2 * key_dim..].to_vec();
    for hk in 0..nk {
        for buf in [&mut q_conv, &mut k_conv] {
            let s = &mut buf[hk * ds..(hk + 1) * ds];
            let ss: f64 = s.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let sc = 1.0f32 / (ss as f32).sqrt().max(eps);
            for v in s.iter_mut() {
                *v *= sc;
            }
        }
    }
    let qscale = 1.0f32 / (ds as f32).sqrt();
    let mut exp_state = state0.clone();
    let mut exp_final = vec![0f32; value_dim];
    for h in 0..nv {
        let hk = h % nk;
        let qh = &q_conv[hk * ds..(hk + 1) * ds];
        let kh = &k_conv[hk * ds..(hk + 1) * ds];
        let vh = &v_conv[h * ds..(h + 1) * ds];
        let st = &mut exp_state[h * ds * ds..(h + 1) * ds * ds];
        let g = decay[h];
        for s in st.iter_mut() {
            *s *= g;
        }
        let mut sk = vec![0f32; ds];
        for i in 0..ds {
            let ki = kh[i];
            for j in 0..ds {
                sk[j] += st[i * ds + j] * ki;
            }
        }
        let mut dvec = vec![0f32; ds];
        for j in 0..ds {
            dvec[j] = (vh[j] - sk[j]) * beta[h];
        }
        for i in 0..ds {
            let ki = kh[i];
            for j in 0..ds {
                st[i * ds + j] += ki * dvec[j];
            }
        }
        let mut o = vec![0f32; ds];
        for i in 0..ds {
            let qi = qh[i] * qscale;
            for j in 0..ds {
                o[j] += st[i * ds + j] * qi;
            }
        }
        let ssn: f32 = o.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ssn / ds as f32 + eps).sqrt();
        for j in 0..ds {
            exp_final[h * ds + j] = (o[j] * inv * ssm_norm[j]) * silu(z[h * ds + j]);
        }
    }

    // ---- GPU chain ----
    let dqkv = k.stream.clone_htod(&qkv).unwrap();
    let dz = k.stream.clone_htod(&z).unwrap();
    let dbr = k.stream.clone_htod(&beta_raw).unwrap();
    let dar = k.stream.clone_htod(&alpha_raw).unwrap();
    let ddt = k.stream.clone_htod(&dt_bias).unwrap();
    let da = k.stream.clone_htod(&a_vec).unwrap();
    let dcw = k.stream.clone_htod(&conv_w).unwrap();
    let dnorm = k.stream.clone_htod(&ssm_norm).unwrap();
    let mut dcs = k.stream.clone_htod(&conv_state0).unwrap();
    let mut dstate = k.stream.clone_htod(&state0).unwrap();
    let mut dbeta = k.stream.alloc_zeros::<f32>(nv).unwrap();
    let mut ddecay = k.stream.alloc_zeros::<f32>(nv).unwrap();
    let mut dconv_out = k.stream.alloc_zeros::<f32>(conv_dim).unwrap();
    let mut dfinal = k.stream.alloc_zeros::<f32>(value_dim).unwrap();
    let nvi = nv as i32;
    let dsi = ds as i32;
    let nki = nk as i32;
    // gates
    {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (nv as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = k.stream.launch_builder(&k.ssm_gates);
        b.arg(&dbr)
            .arg(&dar)
            .arg(&ddt)
            .arg(&da)
            .arg(&mut dbeta)
            .arg(&mut ddecay)
            .arg(&nvi);
        unsafe { b.launch(cfg).unwrap() };
    }
    // conv1d
    {
        let cfg = LaunchConfig {
            grid_dim: (conv_dim.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let cdi = conv_dim as i32;
        let dci = d_conv as i32;
        let mut b = k.stream.launch_builder(&k.ssm_conv1d);
        b.arg(&dcw)
            .arg(&dqkv)
            .arg(&mut dcs)
            .arg(&mut dconv_out)
            .arg(&cdi)
            .arg(&dci);
        unsafe { b.launch(cfg).unwrap() };
    }
    // l2-norm q (conv_out[0..key_dim]) and k (conv_out[key_dim..2*key_dim]), in place
    for lo in [0usize, key_dim] {
        let cfg = LaunchConfig {
            grid_dim: (nk as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: (ds as u32) * 4,
        };
        let mut view = dconv_out.slice_mut(lo..lo + key_dim);
        let mut b = k.stream.launch_builder(&k.ssm_l2_norm_per_head);
        b.arg(&mut view).arg(&dsi).arg(&eps);
        unsafe { b.launch(cfg).unwrap() };
    }
    // delta rule + gated RMSNorm (reads q/k/v sub-ranges of conv_out)
    {
        let cfg = LaunchConfig {
            grid_dim: (nv as u32, 1, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: (3 * ds as u32) * 4,
        };
        let qv = dconv_out.slice(0..key_dim);
        let kv = dconv_out.slice(key_dim..2 * key_dim);
        let vv = dconv_out.slice(2 * key_dim..2 * key_dim + value_dim);
        let mut b = k.stream.launch_builder(&k.ssm_delta_rule);
        b.arg(&mut dstate)
            .arg(&kv)
            .arg(&qv)
            .arg(&vv)
            .arg(&dz)
            .arg(&dbeta)
            .arg(&ddecay)
            .arg(&dnorm)
            .arg(&mut dfinal)
            .arg(&dsi)
            .arg(&nki)
            .arg(&eps);
        unsafe { b.launch(cfg).unwrap() };
    }
    let mut got_final = vec![0f32; value_dim];
    let mut got_state = vec![0f32; nv * ds * ds];
    let mut got_cs = vec![0f32; conv_dim * cm1];
    k.stream.memcpy_dtoh(&dfinal, &mut got_final).unwrap();
    k.stream.memcpy_dtoh(&dstate, &mut got_state).unwrap();
    k.stream.memcpy_dtoh(&dcs, &mut got_cs).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(
        close(&got_final, &exp_final, 3e-3),
        "ssm chain final_out diverged"
    );
    assert!(
        close(&got_state, &exp_state, 3e-3),
        "ssm chain state diverged"
    );
    assert!(
        close(&got_cs, &exp_cs, 3e-3),
        "ssm chain conv_state diverged"
    );
}

// The specialized head_dim kernels must keep `o_acc` in registers: a non-zero stack frame means
// the accumulator spilled to local memory, which is the failure mode the compile-time HEAD_DIM
// specialization exists to prevent. Folding all four entry points into one kernel with a runtime
// branch reintroduces it (measured on sm_89: 224 bytes charged to every launch).
#[test]
#[ignore = "requires a CUDA device"]
fn flash_prefill_specialized_kernels_have_no_local_memory_frame() {
    let Some(k) = kernels() else {
        return;
    };
    for (name, f) in [
        (
            "flash_attention_prefill_tiled_d64",
            &k.flash_attention_prefill_tiled_d64,
        ),
        (
            "flash_attention_prefill_tiled_d128",
            &k.flash_attention_prefill_tiled_d128,
        ),
        (
            "flash_attention_prefill_tiled_d256",
            &k.flash_attention_prefill_tiled_d256,
        ),
    ] {
        let local = f.local_size_bytes().unwrap();
        let regs = f.num_regs().unwrap();
        let max_threads = f.max_threads_per_block().unwrap();
        println!("{name}: num_regs={regs} local_size_bytes={local} max_threads={max_threads}");
        assert_eq!(local, 0, "{name} spilled {local} bytes to local memory");
        assert!(
            max_threads >= 256,
            "{name} cannot launch the 256-thread block the launcher uses (max {max_threads})"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn flash_attention_prefill_tiled_parity() {
    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 8usize;
    let n_kv = 2usize;
    let max_pos = 2048usize;

    // 64/128/256 take the compile-time-specialized kernels; 96 routes to the runtime-head_dim
    // twin, which is a different entry point and would otherwise be shipped untested.
    let head_dims = [64usize, 96usize, 128usize, 256usize];
    let base_positions = [0usize, 64usize, 128usize, 512usize, 1024usize];
    let k_tokens_list = [1usize, 8usize, 16usize, 32usize, 64usize];

    for &head_dim in &head_dims {
        let scale = 1.0 / (head_dim as f32).sqrt();
        for &base_position in &base_positions {
            for &k_tokens in &k_tokens_list {
                let mut rng = Lcg(0xCA_FE_12_34
                    + (head_dim as u64) * 1000
                    + (base_position as u64)
                    + (k_tokens as u64));
                let q: Vec<f32> = (0..k_tokens * n_heads * head_dim)
                    .map(|_| rng.next_f32() * 2.0 - 1.0)
                    .collect();
                let total_kv_pos = base_position + k_tokens;
                let mut cache_k_f32 = vec![0f32; n_kv * max_pos * head_dim];
                let mut cache_v_f32 = vec![0f32; n_kv * max_pos * head_dim];
                for x in cache_k_f32[..n_kv * total_kv_pos * head_dim].iter_mut() {
                    *x = rng.next_f32() * 2.0 - 1.0;
                }
                for x in cache_v_f32[..n_kv * total_kv_pos * head_dim].iter_mut() {
                    *x = rng.next_f32() * 2.0 - 1.0;
                }

                let cache_k_bits: Vec<u16> = cache_k_f32
                    .iter()
                    .map(|&x| crate::inference::f32_to_f16_bits(x))
                    .collect();
                let cache_v_bits: Vec<u16> = cache_v_f32
                    .iter()
                    .map(|&x| crate::inference::f32_to_f16_bits(x))
                    .collect();
                let cache_k_bytes: Vec<u8> = cache_k_bits
                    .iter()
                    .flat_map(|bits| bits.to_le_bytes())
                    .collect();
                let cache_v_bytes: Vec<u8> = cache_v_bits
                    .iter()
                    .flat_map(|bits| bits.to_le_bytes())
                    .collect();

                let dq = k.stream.clone_htod(&q).unwrap();
                let dk = k.stream.clone_htod(&cache_k_bytes).unwrap();
                let dv = k.stream.clone_htod(&cache_v_bytes).unwrap();
                let mut dout = k
                    .stream
                    .alloc_zeros::<f32>(k_tokens * n_heads * head_dim)
                    .unwrap();

                super::launch_attention_flash_prefill(
                    &k.stream,
                    &k,
                    &dq,
                    &dk,
                    &dv,
                    &mut dout,
                    n_heads,
                    n_kv,
                    head_dim,
                    base_position,
                    k_tokens,
                    n_heads * head_dim,
                    max_pos,
                    scale,
                )
                .unwrap();

                let mut got = vec![0f32; k_tokens * n_heads * head_dim];
                k.stream.memcpy_dtoh(&dout, &mut got).unwrap();
                k.ctx.synchronize().unwrap();

                // CPU reference causal attention
                let mut expected = vec![0f32; k_tokens * n_heads * head_dim];
                let repeats = n_heads / n_kv;
                for t in 0..k_tokens {
                    let global_q_pos = base_position + t;
                    for h in 0..n_heads {
                        let kv_h = h / repeats;
                        let q_offset = (t * n_heads + h) * head_dim;
                        let q_slice = &q[q_offset..q_offset + head_dim];

                        // Compute scores
                        let mut scores = Vec::with_capacity(global_q_pos + 1);
                        for p in 0..=global_q_pos {
                            let k_offset = (kv_h * max_pos + p) * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                let k_val =
                                    crate::inference::f16_bits_to_f32(cache_k_bits[k_offset + d]);
                                dot += q_slice[d] * k_val;
                            }
                            scores.push(dot * scale);
                        }

                        // Softmax
                        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exp_scores: Vec<f32> =
                            scores.iter().map(|&s| (s - max_s).exp()).collect();
                        let sum_exp: f32 = exp_scores.iter().sum();
                        let inv_sum = 1.0 / sum_exp;

                        // Weighted V
                        let out_offset = (t * n_heads + h) * head_dim;
                        for d in 0..head_dim {
                            let mut acc = 0.0f32;
                            for (p, &exp_score) in
                                exp_scores.iter().enumerate().take(global_q_pos + 1)
                            {
                                let v_offset = (kv_h * max_pos + p) * head_dim;
                                let v_val =
                                    crate::inference::f16_bits_to_f32(cache_v_bits[v_offset + d]);
                                acc += exp_score * v_val;
                            }
                            expected[out_offset + d] = acc * inv_sum;
                        }
                    }
                }

                assert!(
                    close(&got, &expected, 5e-4),
                    "flash_attention_prefill_tiled diverged at head_dim={head_dim}, base_position={base_position}, k_tokens={k_tokens}"
                );
            }
        }
    }
}

// `attention_sw_batched` must be BITWISE identical to the scalar `attention_decode_sw` as
// the gemma4 runtime launches it -- blockDim.x == head_dim, i.e. G == 1 -- for every token
// in a verify batch. Anything less than bitwise here is not a rounding detail: gemma4 folds
// the attention scale (scale == 1.0), so its softmax is extremely peaked and a one-ulp
// score difference can flip an argmax, which would make speculative verify disagree with
// plain decode and silently break losslessness.
//
// Both gemma4 layer types are covered: sliding (window crops the prefix, and the KV ring
// wraps modulo its window-plus-verifier-slack capacity) and full (window == 0, no crop). The batch sweep
// includes K == 1 because that is the structural gate the H40 harness runs, and widths
// where `base + K` straddles the window edge so some tokens in one batch crop and others
// do not.
#[test]
#[ignore = "requires a CUDA device"]
fn attention_sw_batched_matches_gemma4_scalar_decode() {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let Some(k) = kernels() else {
        return;
    };
    // gemma4 26B-A4B geometry: sliding layers 16 q-heads x 256 over 8 KV heads, the full
    // layer 16 x 512 over 2. `scale` is 1.0 on both -- gemma folds it into q_norm.
    //
    // `kv_capacity` and `context` are deliberately separate. A sliding cache is a ring of
    // only `window + MAX_MTP_VERIFY_ROWS` slots, while absolute positions can advance to the context limit.
    // The score scratch is ring-sized and indexed relative to the active window; keeping
    // the limits separate exercises both score bounds and `p % max_pos` after wraparound.
    let cases: [(usize, usize, usize, usize, usize, usize); 2] = [
        // (n_heads, n_kv, head_dim, window, kv_capacity, context)
        (
            16,
            8,
            256,
            1024,
            1024 + crate::gemma4_runtime::MAX_MTP_VERIFY_ROWS,
            4096,
        ),
        (16, 2, 512, 0, 4096, 4096),
    ];
    let scale = 1.0f32;

    for (n_heads, n_kv, head_dim, window, max_pos, context) in cases {
        let mut rng = Lcg(0x5eed_1234 ^ (head_dim as u64));
        let mut cache_k = vec![0f32; n_kv * max_pos * head_dim];
        let mut cache_v = vec![0f32; n_kv * max_pos * head_dim];
        for x in cache_k.iter_mut() {
            *x = rng.next_f32();
        }
        for x in cache_v.iter_mut() {
            *x = rng.next_f32();
        }
        let cache_k_bits: Vec<u16> = cache_k
            .iter()
            .map(|&x| crate::inference::f32_to_f16_bits(x))
            .collect();
        let cache_v_bits: Vec<u16> = cache_v
            .iter()
            .map(|&x| crate::inference::f32_to_f16_bits(x))
            .collect();
        let cache_k_bytes: Vec<u8> = cache_k_bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let cache_v_bytes: Vec<u8> = cache_v_bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let dk = k.stream.clone_htod(&cache_k_bytes).unwrap();
        let dv = k.stream.clone_htod(&cache_v_bytes).unwrap();

        // Inside the window; straddling its edge (so tokens in ONE batch disagree about
        // whether the prefix is cropped); and past the ring's capacity, where `p % max_pos`
        // starts doing real work. That last group matters most: a sliding cache holds only
        // `window + MAX_MTP_VERIFY_ROWS` slots, so every later verify batch reads wrapped
        // history, and it is where a relative-indexed batched kernel is most likely to part
        // company with an absolute-indexed scalar one.
        let mut bases: Vec<usize> = vec![7, 64];
        if window > 0 {
            bases.extend([window - 3, window, max_pos + 475, 2 * max_pos + 13]);
        } else {
            bases.extend([512, 2000]);
        }
        for base in bases {
            for kt in [1usize, 2, 4, 8, 9, 13, 14] {
                // Bounded by the CONTEXT, not the ring: the scalar kernel's shared
                // `scores[]` is indexed by absolute position.
                if base + kt >= context {
                    continue;
                }
                let q_per_token = n_heads * head_dim;
                let q: Vec<f32> = (0..kt * q_per_token).map(|_| rng.next_f32()).collect();
                let dq = k.stream.clone_htod(&q).unwrap();

                // Reference: the scalar kernel run once per token, launched EXACTLY the way
                // `Gemma4CudaResident::forward_token_impl` launches it.
                let mut reference = vec![0f32; kt * q_per_token];
                for t in 0..kt {
                    let position = base + t;
                    let dpos = k.stream.clone_htod(&[position as i32]).unwrap();
                    let dq_t = k
                        .stream
                        .clone_htod(&q[t * q_per_token..][..q_per_token])
                        .unwrap();
                    let mut dout = k.stream.alloc_zeros::<f32>(q_per_token).unwrap();
                    let mut global_scores = k.stream.alloc_zeros::<f32>(n_heads * max_pos).unwrap();
                    let cfg = LaunchConfig {
                        grid_dim: (n_heads as u32, 1, 1),
                        block_dim: (head_dim as u32, 1, 1),
                        shared_mem_bytes: ((2 * head_dim) as u32) * 4,
                    };
                    let (nh, nkv, hd, mp, win) = (
                        n_heads as i32,
                        n_kv as i32,
                        head_dim as i32,
                        max_pos as i32,
                        window as i32,
                    );
                    let mut b = k.stream.launch_builder(&k.attention_sw);
                    b.arg(&dq_t)
                        .arg(&dk)
                        .arg(&dv)
                        .arg(&mut dout)
                        .arg(&nh)
                        .arg(&nkv)
                        .arg(&hd)
                        .arg(&dpos)
                        .arg(&mp)
                        .arg(&scale)
                        .arg(&win)
                        .arg(&mut global_scores);
                    unsafe { b.launch(cfg) }.unwrap();
                    k.stream
                        .memcpy_dtoh(&dout, &mut reference[t * q_per_token..][..q_per_token])
                        .unwrap();
                }

                let mut dbatched = k.stream.alloc_zeros::<f32>(kt * q_per_token).unwrap();
                super::launch_attention_sw_batched(
                    &k.stream,
                    &k.attention_sw_batched,
                    &dq,
                    &dk,
                    &dv,
                    &mut dbatched,
                    n_heads,
                    n_kv,
                    head_dim,
                    base,
                    max_pos,
                    scale,
                    window,
                    q_per_token,
                    kt,
                )
                .unwrap();
                let mut batched = vec![0f32; kt * q_per_token];
                k.stream.memcpy_dtoh(&dbatched, &mut batched).unwrap();

                for (i, (got, want)) in batched.iter().zip(&reference).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "head_dim {head_dim} window {window} base {base} K {kt}: element {i} \
                         (token {}, head {}, dim {}) is {got} but scalar decode gives {want}",
                        i / q_per_token,
                        (i % q_per_token) / head_dim,
                        i % head_dim,
                    );
                }
            }
        }
    }
}

/// Drive the actual f16 batched scatter and sliding-attention kernels through
/// more than two complete ring wraps, comparing every fourteen-row verifier batch
/// against token-by-token Gemma decode. The cache comparison pins modulo writes;
/// the output comparison pins the stronger lifetime rule: scattering later rows
/// must not overwrite history still needed by an earlier row in the same batch.
#[test]
#[ignore = "requires a CUDA device"]
fn kv_scatter_batched_ring_wrap_preserves_attention_history() {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let Some(k) = kernels() else {
        return;
    };
    let n_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 32usize;
    let window = 7usize;
    let context = 128usize;
    let k_tokens = crate::gemma4_runtime::MAX_MTP_VERIFY_ROWS;
    let max_pos = crate::gemma4_runtime::gemma4_kv_capacity(Some(window), context);
    assert_eq!(
        max_pos,
        window + k_tokens,
        "sliding KV must retain one full verifier width beyond the active window"
    );

    let kv_per_token = n_kv_heads * head_dim;
    let q_per_token = n_heads * head_dim;
    let cache_bytes = n_kv_heads * max_pos * head_dim * 2;
    let mut scalar_k = k.stream.alloc_zeros::<u8>(cache_bytes).unwrap();
    let mut scalar_v = k.stream.alloc_zeros::<u8>(cache_bytes).unwrap();
    let mut batched_k = k.stream.alloc_zeros::<u8>(cache_bytes).unwrap();
    let mut batched_v = k.stream.alloc_zeros::<u8>(cache_bytes).unwrap();
    let scale = 1.0f32;
    let rounds = 6usize;
    let mut rng = Lcg(0x4b_56_72_39);

    for round in 0..rounds {
        let base = round * k_tokens;
        let src_k: Vec<f32> = (0..k_tokens * kv_per_token)
            .map(|_| rng.next_f32())
            .collect();
        let src_v: Vec<f32> = (0..k_tokens * kv_per_token)
            .map(|_| rng.next_f32())
            .collect();
        let q: Vec<f32> = (0..k_tokens * q_per_token)
            .map(|_| rng.next_f32())
            .collect();

        // Reference: exactly the scalar Gemma order -- scatter one position,
        // attend that position, then advance to the next token.
        let mut reference = vec![0f32; k_tokens * q_per_token];
        for t in 0..k_tokens {
            let position = base + t;
            let d_position = k.stream.clone_htod(&[position as i32]).unwrap();
            let d_src_k = k
                .stream
                .clone_htod(&src_k[t * kv_per_token..(t + 1) * kv_per_token])
                .unwrap();
            let d_src_v = k
                .stream
                .clone_htod(&src_v[t * kv_per_token..(t + 1) * kv_per_token])
                .unwrap();
            super::launch_kv_scatter(
                &k.stream,
                &k.kv_scatter,
                &d_src_k,
                &mut scalar_k,
                &d_position,
                n_kv_heads,
                head_dim,
                max_pos,
            )
            .unwrap();
            super::launch_kv_scatter(
                &k.stream,
                &k.kv_scatter,
                &d_src_v,
                &mut scalar_v,
                &d_position,
                n_kv_heads,
                head_dim,
                max_pos,
            )
            .unwrap();

            let d_q = k
                .stream
                .clone_htod(&q[t * q_per_token..(t + 1) * q_per_token])
                .unwrap();
            let mut d_out = k.stream.alloc_zeros::<f32>(q_per_token).unwrap();
            let mut global_scores = k.stream.alloc_zeros::<f32>(n_heads * max_pos).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (n_heads as u32, 1, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: ((2 * head_dim) as u32) * 4,
            };
            let (nh, nkv, hd, mp, win) = (
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                max_pos as i32,
                window as i32,
            );
            let mut launch = k.stream.launch_builder(&k.attention_sw);
            launch
                .arg(&d_q)
                .arg(&scalar_k)
                .arg(&scalar_v)
                .arg(&mut d_out)
                .arg(&nh)
                .arg(&nkv)
                .arg(&hd)
                .arg(&d_position)
                .arg(&mp)
                .arg(&scale)
                .arg(&win)
                .arg(&mut global_scores);
            unsafe { launch.launch(cfg) }.unwrap();
            k.stream
                .memcpy_dtoh(
                    &d_out,
                    &mut reference[t * q_per_token..(t + 1) * q_per_token],
                )
                .unwrap();
        }

        // Under test: scatter every speculative row first, then run the exact
        // K-wide sliding-attention kernel over the same logical positions.
        let d_src_k = k.stream.clone_htod(&src_k).unwrap();
        let d_src_v = k.stream.clone_htod(&src_v).unwrap();
        let d_q = k.stream.clone_htod(&q).unwrap();
        super::launch_kv_scatter_batched(
            &k.stream,
            &k.kv_scatter_batched,
            &d_src_k,
            &mut batched_k,
            base,
            n_kv_heads,
            head_dim,
            max_pos,
            kv_per_token,
            k_tokens,
        )
        .unwrap();
        super::launch_kv_scatter_batched(
            &k.stream,
            &k.kv_scatter_batched,
            &d_src_v,
            &mut batched_v,
            base,
            n_kv_heads,
            head_dim,
            max_pos,
            kv_per_token,
            k_tokens,
        )
        .unwrap();
        let mut d_batched_out = k.stream.alloc_zeros::<f32>(k_tokens * q_per_token).unwrap();
        super::launch_attention_sw_batched(
            &k.stream,
            &k.attention_sw_batched,
            &d_q,
            &batched_k,
            &batched_v,
            &mut d_batched_out,
            n_heads,
            n_kv_heads,
            head_dim,
            base,
            max_pos,
            scale,
            window,
            q_per_token,
            k_tokens,
        )
        .unwrap();

        let mut batched = vec![0f32; k_tokens * q_per_token];
        let mut scalar_k_host = vec![0u8; cache_bytes];
        let mut scalar_v_host = vec![0u8; cache_bytes];
        let mut batched_k_host = vec![0u8; cache_bytes];
        let mut batched_v_host = vec![0u8; cache_bytes];
        k.stream.memcpy_dtoh(&d_batched_out, &mut batched).unwrap();
        k.stream.memcpy_dtoh(&scalar_k, &mut scalar_k_host).unwrap();
        k.stream.memcpy_dtoh(&scalar_v, &mut scalar_v_host).unwrap();
        k.stream
            .memcpy_dtoh(&batched_k, &mut batched_k_host)
            .unwrap();
        k.stream
            .memcpy_dtoh(&batched_v, &mut batched_v_host)
            .unwrap();
        k.ctx.synchronize().unwrap();

        assert_eq!(
            batched_k_host, scalar_k_host,
            "K cache differs after ring round {round} (base {base})"
        );
        assert_eq!(
            batched_v_host, scalar_v_host,
            "V cache differs after ring round {round} (base {base})"
        );
        for (i, (got, want)) in batched.iter().zip(&reference).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "ring round {round} base {base}: output element {i} differs after batched scatter"
            );
        }
    }

    let last_base = (rounds - 1) * k_tokens;
    assert!(
        last_base > 2 * max_pos,
        "fixture must cross more than two complete KV-ring wraps"
    );
}

/// `q4_1_gemm_routed` must be BITWISE identical to per-assignment `q4_1_gemv_routed`.
///
/// The Q4_1 twin of `q4_0_gemm_routed_matches_gemv`, and it needs its own gate rather than
/// riding on that one: the two kernels agree structurally but decode blocks differently,
/// and the Q4_1 term `(w_d*isum + w_m*asum)` has a second scale and a second integer sum
/// that the Q4_0 form simply does not have. The reference spells that block out as a
/// scalar 16-iteration nibble loop while this kernel uses the packed dp4a helper, so what
/// is really being pinned here is that the two produce the same integers.
///
/// Real geometries (the 26B-A4B `down_exps` shapes), a ragged CSR including an expert with
/// zero tokens and one with more than a tile, a permuted slot map, and `tile` small enough
/// that the `blockIdx.z` path runs.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_1_gemm_routed_matches_gemv() {
    let Some(k) = kernels() else {
        return;
    };
    for (rows, bpr, seed) in [(2816usize, 22usize, 0x6e_11u64), (1408, 88, 0x6e_12)] {
        let experts = 4usize;
        let n_tokens = 7usize;
        let kdim = bpr * 32;
        let mut rng = Lcg(seed);

        let per_slot = rows * bpr * 20;
        let mut arena = Vec::with_capacity(experts * per_slot);
        for _ in 0..experts {
            arena.extend_from_slice(&synth_q4_1_wire(rows, bpr, &mut rng));
        }
        let mut in_s = vec![0f32; n_tokens * bpr];
        let mut in_q = vec![0i8; n_tokens * kdim];
        for t in 0..n_tokens {
            let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
            let q8 = crate::inference::quantize_q8_0_blocks(&act);
            for (b, blk) in q8.iter().enumerate() {
                in_s[t * bpr + b] = blk.scale;
                in_q[t * kdim + b * 32..t * kdim + (b + 1) * 32].copy_from_slice(&blk.quants);
            }
        }
        // Ragged: expert 0 -> 3 tokens, 1 -> none, 2 -> 1, 3 -> 4. A repeated token id
        // (5 appears twice) checks that output is keyed by ASSIGNMENT, not by token.
        let token_offsets: Vec<i32> = vec![0, 3, 3, 4, 8];
        let token_ids: Vec<i32> = vec![5, 0, 3, 6, 2, 4, 1, 5];
        let slots: Vec<i32> = vec![2, 0, 3, 1];
        let assignments = *token_offsets.last().unwrap() as usize;

        let d_s = k.stream.clone_htod(&in_s).unwrap();
        let d_q = k.stream.clone_htod(&in_q).unwrap();
        let d_w = k.stream.clone_htod(&arena).unwrap();
        let d_slots = k.stream.clone_htod(&slots).unwrap();
        let d_off = k.stream.clone_htod(&token_offsets).unwrap();
        let d_tok = k.stream.clone_htod(&token_ids).unwrap();

        // Reference: the shipped GEMV, once per assignment, single-token activation view.
        let mut reference = vec![0f32; assignments * rows];
        for e in 0..experts {
            let (lo, hi) = (token_offsets[e] as usize, token_offsets[e + 1] as usize);
            for a in lo..hi {
                let t = token_ids[a] as usize;
                let d_s1 = k.stream.clone_htod(&in_s[t * bpr..(t + 1) * bpr]).unwrap();
                let d_q1 = k
                    .stream
                    .clone_htod(&in_q[t * kdim..(t + 1) * kdim])
                    .unwrap();
                let one_slot: Vec<i32> = vec![slots[e]];
                let one_route: Vec<i32> = vec![0];
                let d_s1s = k.stream.clone_htod(&one_slot).unwrap();
                let d_r1 = k.stream.clone_htod(&one_route).unwrap();
                let mut d_out = k.stream.alloc_zeros::<f32>(rows).unwrap();
                use cudarc::driver::{LaunchConfig, PushKernelArg};
                let block = 256u32;
                let warps = block / 32;
                let cfg = LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(warps), 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: bpr as u32 * 32 + bpr as u32 * 4 + warps * bpr as u32 * 4,
                };
                let (stride, rows_i, bpr_i, one, zero) =
                    (per_slot as u64, rows as i32, bpr as i32, 1i32, 0i32);
                let mut b = k.stream.launch_builder(&k.q4_1_gemv_routed);
                b.arg(&d_s1)
                    .arg(&d_q1)
                    .arg(&d_w)
                    .arg(&d_s1s)
                    .arg(&d_r1)
                    .arg(&stride)
                    .arg(&rows_i)
                    .arg(&bpr_i)
                    .arg(&mut d_out)
                    .arg(&one)
                    .arg(&zero);
                unsafe { b.launch(cfg) }.unwrap();
                k.stream
                    .memcpy_dtoh(&d_out, &mut reference[a * rows..(a + 1) * rows])
                    .unwrap();
                k.ctx.synchronize().unwrap();
            }
        }

        let tile = 2usize;
        let max_count = (0..experts)
            .map(|e| token_offsets[e + 1] - token_offsets[e])
            .max()
            .unwrap() as usize;
        let mut d_out = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
        {
            use cudarc::driver::{LaunchConfig, PushKernelArg};
            let block = 256u32;
            let warps = block / 32;
            let cfg = LaunchConfig {
                grid_dim: (
                    (rows as u32).div_ceil(warps),
                    experts as u32,
                    max_count.div_ceil(tile) as u32,
                ),
                block_dim: (block, 1, 1),
                shared_mem_bytes: warps * tile as u32 * bpr as u32 * 4,
            };
            let (stride, rows_i, bpr_i, experts_i, tile_i) = (
                per_slot as u64,
                rows as i32,
                bpr as i32,
                experts as i32,
                tile as i32,
            );
            let mut b = k.stream.launch_builder(&k.q4_1_gemm_routed);
            b.arg(&d_s)
                .arg(&d_q)
                .arg(&d_w)
                .arg(&d_slots)
                .arg(&d_off)
                .arg(&d_tok)
                .arg(&stride)
                .arg(&rows_i)
                .arg(&bpr_i)
                .arg(&mut d_out)
                .arg(&experts_i)
                .arg(&tile_i);
            unsafe { b.launch(cfg) }.unwrap();
        }
        let mut got = vec![0f32; assignments * rows];
        k.stream.memcpy_dtoh(&d_out, &mut got).unwrap();
        k.ctx.synchronize().unwrap();

        let mut d_helper = k.stream.alloc_zeros::<f32>(assignments * rows).unwrap();
        super::launch_q4_1_gemm_routed(
            &k.stream,
            &k.q4_1_gemm_routed,
            &d_s,
            &d_q,
            &d_w,
            &d_slots,
            &d_off,
            &d_tok,
            per_slot,
            rows,
            bpr,
            experts,
            max_count,
            false,
            &mut d_helper,
        )
        .unwrap();
        let mut helper = vec![0f32; assignments * rows];
        k.stream.memcpy_dtoh(&d_helper, &mut helper).unwrap();
        k.ctx.synchronize().unwrap();
        assert_same_bits("Q4_1 routed launch helper", &helper, &reference);

        let mismatches = got
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "q4_1_gemm_routed must be BITWISE identical to per-assignment q4_1_gemv_routed \
             (rows {rows}, blocks/row {bpr}): {mismatches}/{} outputs differ",
            got.len()
        );
        assert!(
            reference.iter().any(|v| *v != 0.0),
            "degenerate fixture: the reference produced all zeros"
        );
    }
}

/// The 32-block low-shared A/B must remain bitwise identical to the current
/// production routed GEMM at narrow, common, and maximum verifier widths. Cover
/// Q4_0 gate_up and both down formats at their real 26B-A4B contractions; K14
/// also crosses the launcher's nine-assignment tile boundary.
#[test]
#[ignore = "requires a CUDA device"]
fn q4_gemm_routed_chunked_matches_production_k1_k8_k14_bitwise() {
    let Some(k) = kernels() else {
        return;
    };
    let max_tokens = crate::gemma4_runtime::MAX_MTP_VERIFY_ROWS;
    assert_eq!(max_tokens, 14, "test cases pin the production verifier cap");

    for (q4_1, rows, bpr, seed, label) in [
        (false, 1408usize, 88usize, 0x40_c8_14u64, "Q4_0 gate_up"),
        (false, 2816usize, 22usize, 0x40_d0_14u64, "Q4_0 down"),
        (true, 2816usize, 22usize, 0x41_d0_14u64, "Q4_1 down"),
    ] {
        let kdim = bpr * 32;
        let mut rng = Lcg(seed);
        let wire = if q4_1 {
            synth_q4_1_wire(rows, bpr, &mut rng)
        } else {
            synth_q4_0_wire(rows, bpr, &mut rng)
        };
        let mut in_scales = vec![0f32; max_tokens * bpr];
        let mut in_quants = vec![0i8; max_tokens * kdim];
        for t in 0..max_tokens {
            let act: Vec<f32> = (0..kdim).map(|_| rng.next_f32()).collect();
            let q8 = crate::inference::quantize_q8_0_blocks(&act);
            for (b, block) in q8.iter().enumerate() {
                in_scales[t * bpr + b] = block.scale;
                in_quants[t * kdim + b * 32..t * kdim + (b + 1) * 32]
                    .copy_from_slice(&block.quants);
            }
        }

        let d_s = k.stream.clone_htod(&in_scales).unwrap();
        let d_q = k.stream.clone_htod(&in_quants).unwrap();
        let d_w = k.stream.clone_htod(&wire).unwrap();
        let slots = [0i32];
        let d_slots = k.stream.clone_htod(&slots).unwrap();
        for width in [1usize, 8, max_tokens] {
            let token_offsets = [0i32, width as i32];
            let token_ids = (0..width as i32).collect::<Vec<_>>();
            let d_offsets = k.stream.clone_htod(&token_offsets).unwrap();
            let d_tokens = k.stream.clone_htod(&token_ids).unwrap();
            let mut d_production = k.stream.alloc_zeros::<f32>(width * rows).unwrap();
            let mut d_chunked = k.stream.alloc_zeros::<f32>(width * rows).unwrap();
            if q4_1 {
                super::launch_q4_1_gemm_routed(
                    &k.stream,
                    &k.q4_1_gemm_routed,
                    &d_s,
                    &d_q,
                    &d_w,
                    &d_slots,
                    &d_offsets,
                    &d_tokens,
                    wire.len(),
                    rows,
                    bpr,
                    1,
                    width,
                    false,
                    &mut d_production,
                )
                .unwrap();
                super::launch_q4_1_gemm_routed(
                    &k.stream,
                    &k.q4_1_gemm_routed_chunked,
                    &d_s,
                    &d_q,
                    &d_w,
                    &d_slots,
                    &d_offsets,
                    &d_tokens,
                    wire.len(),
                    rows,
                    bpr,
                    1,
                    width,
                    true,
                    &mut d_chunked,
                )
                .unwrap();
            } else {
                super::launch_q4_0_gemm_routed(
                    &k.stream,
                    &k.q4_0_gemm_routed,
                    &d_s,
                    &d_q,
                    &d_w,
                    &d_slots,
                    &d_offsets,
                    &d_tokens,
                    wire.len(),
                    rows,
                    bpr,
                    1,
                    width,
                    false,
                    &mut d_production,
                )
                .unwrap();
                super::launch_q4_0_gemm_routed(
                    &k.stream,
                    &k.q4_0_gemm_routed_chunked,
                    &d_s,
                    &d_q,
                    &d_w,
                    &d_slots,
                    &d_offsets,
                    &d_tokens,
                    wire.len(),
                    rows,
                    bpr,
                    1,
                    width,
                    true,
                    &mut d_chunked,
                )
                .unwrap();
            }
            let mut production = vec![0f32; width * rows];
            let mut chunked = vec![0f32; width * rows];
            k.stream
                .memcpy_dtoh(&d_production, &mut production)
                .unwrap();
            k.stream.memcpy_dtoh(&d_chunked, &mut chunked).unwrap();
            k.ctx.synchronize().unwrap();
            assert_same_bits(
                &format!("{label} chunked vs production K={width}"),
                &chunked,
                &production,
            );
            assert!(
                production.iter().any(|value| *value != 0.0),
                "degenerate {label} K={width} fixture produced only zeros"
            );
        }
    }
}

/// Routed GEMMs emit expert-major CSR assignment rows, but Gemma's exact MoE
/// sum is token-major and router-rank ordered. Exercise the maximum planned
/// verifier width with experts reused across tokens and a deliberately scrambled
/// mapping, then compare bit-for-bit with the shipped rank-by-rank scaled-AXPY.
#[test]
#[ignore = "requires a CUDA device"]
fn moe_weighted_sum_batched_maps_csr_assignments_in_router_order_bitwise() {
    let Some(k) = kernels() else {
        return;
    };
    let k_tokens = 15usize;
    let route_count = 8usize;
    let expert_count = 32usize;
    let hidden = 2816usize;
    let assignments = k_tokens * route_count;

    // Every token selects eight distinct experts, while the same experts recur
    // across tokens. Grouping by expert produces the CSR assignment order that
    // the routed GEMMs write, which is intentionally not router order.
    let mut experts_by_route = vec![0usize; assignments];
    for token in 0..k_tokens {
        for route in 0..route_count {
            experts_by_route[token * route_count + route] = (route * 7 + token * 3) % expert_count;
        }
        let mut one_token =
            experts_by_route[token * route_count..(token + 1) * route_count].to_vec();
        one_token.sort_unstable();
        one_token.dedup();
        assert_eq!(
            one_token.len(),
            route_count,
            "fixture must not repeat an expert within one token"
        );
    }
    assert!(
        (0..expert_count).any(|expert| {
            (0..k_tokens)
                .filter(|token| {
                    experts_by_route[*token * route_count..(*token + 1) * route_count]
                        .contains(&expert)
                })
                .count()
                > 1
        }),
        "fixture must reuse experts across different tokens"
    );

    let mut route_to_assignment = vec![-1i32; assignments];
    let mut assignment_experts = Vec::with_capacity(assignments);
    for expert in 0..expert_count {
        for token in 0..k_tokens {
            for route in 0..route_count {
                let router_index = token * route_count + route;
                if experts_by_route[router_index] == expert {
                    route_to_assignment[router_index] = assignment_experts.len() as i32;
                    assignment_experts.push(expert);
                }
            }
        }
    }
    assert_eq!(assignment_experts.len(), assignments);
    let mut checked_map = route_to_assignment.clone();
    checked_map.sort_unstable();
    assert_eq!(
        checked_map,
        (0..assignments as i32).collect::<Vec<_>>(),
        "every router position must name one unique, in-range assignment row"
    );
    assert_ne!(
        route_to_assignment,
        (0..assignments as i32).collect::<Vec<_>>(),
        "fixture must exercise a non-identity assignment mapping"
    );

    let mut rng = Lcg(0x4d_4f_45_4b_31_35);
    let mut expert_y = vec![0f32; assignments * hidden];
    for (assignment, expert) in assignment_experts.iter().copied().enumerate() {
        for i in 0..hidden {
            expert_y[assignment * hidden + i] = rng.next_f32()
                * (1.0 + expert as f32 / expert_count as f32)
                + (i % 17) as f32 * 0.000_31;
        }
    }
    let route_scales: Vec<f32> = (0..assignments)
        .map(|index| 0.0125 + (index % route_count) as f32 * 0.017 + rng.next_f32().abs() * 0.01)
        .collect();

    let d_y = k.stream.clone_htod(&expert_y).unwrap();
    let mut scalar = vec![0f32; k_tokens * hidden];
    let cfg = LaunchConfig {
        grid_dim: ((hidden as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let hidden_i = hidden as i32;
    for token in 0..k_tokens {
        let mut d_acc = k.stream.alloc_zeros::<f32>(hidden).unwrap();
        for route in 0..route_count {
            let router_index = token * route_count + route;
            let assignment = route_to_assignment[router_index] as usize;
            let y = d_y.slice(assignment * hidden..(assignment + 1) * hidden);
            let scale = route_scales[router_index];
            let mut b = k.stream.launch_builder(&k.scaled_axpy);
            b.arg(&mut d_acc).arg(&y).arg(&scale).arg(&hidden_i);
            unsafe { b.launch(cfg) }.unwrap();
        }
        k.stream
            .memcpy_dtoh(&d_acc, &mut scalar[token * hidden..(token + 1) * hidden])
            .unwrap();
    }

    let d_map = k.stream.clone_htod(&route_to_assignment).unwrap();
    let d_scales = k.stream.clone_htod(&route_scales).unwrap();
    let mut d_out = k.stream.alloc_zeros::<f32>(k_tokens * hidden).unwrap();
    super::launch_moe_weighted_sum_batched(
        &k.stream,
        &k.moe_weighted_sum_batched,
        &d_y,
        &d_map,
        &d_scales,
        &mut d_out,
        hidden,
        k_tokens,
        route_count,
    )
    .unwrap();
    let mut batched = vec![0f32; k_tokens * hidden];
    k.stream.memcpy_dtoh(&d_out, &mut batched).unwrap();
    k.ctx.synchronize().unwrap();

    assert_same_bits(
        "mapped K-wide MoE weighted sum vs router-order scaled AXPY",
        &batched,
        &scalar,
    );
    assert!(
        scalar.iter().any(|value| *value != 0.0),
        "degenerate mapped MoE fixture produced only zeros"
    );
}

/// Regression: `attention_batched_q8_0` dropped most of the KV once a prompt exceeded
/// `head_dim` tokens.
///
/// The kernel splits the weighted-V accumulation into `G = ceil(position_count /
/// head_dim)` groups but derived `gid` straight from `tid`, while the launcher fixes
/// `block_dim` at 128. For `head_dim == 128` that made `gid` always 0, so only
/// `1/G` of the positions were accumulated and the `g >= 1` slots of the shared
/// `vpart` buffer were summed back **uninitialised**. The f16 sibling grid-strides
/// over all `G * head_dim` partials and was always correct.
///
/// It is invisible below `head_dim` positions (`G == 1`) and wrong above, which is why
/// it shipped: short smoke prompts pass. Measured against a CPU f32 reference on
/// Llama 3.2 3B, the worst single-token logit error was 8.6 before the fix and 0.195
/// after — the latter matching the CPU implementation of the same format (0.170).
///
/// This test drives the kernel either side of that boundary and compares to a CPU
/// reference computed from the same dequantized blocks.
#[test]
#[ignore = "requires a CUDA device; q8_0 batched attention parity across the G boundary"]
fn q8_0_batched_attention_matches_cpu_reference_past_the_head_dim_boundary() {
    let k = match CudaResidentKernels::new() {
        Ok(k) => k,
        Err(e) => panic!("cuda kernels: {e}"),
    };
    let s = k.stream.clone();

    let (n_heads, n_kv_heads) = (1usize, 1usize);
    let max_pos = 512usize;

    // head_dim is swept because the defect took a different shape at each width, and
    // `launch_attention_batched` pins blockDim at 128 regardless:
    //   128 / 100 pos -> G == 1, the one regime the old kernel got right
    //   128 / 200 pos -> G == 2, over-subscribed: only half the positions accumulated
    //    64 /  40 pos -> G == 1 but UNDER-subscribed: threads 64..127 raced on vpart
    //    64 / 200 pos -> G == 4
    // head_dim 64 is a shipped shape (the TinyLlama-shaped decode tests above use it),
    // so pinning 128 alone would let a regression that only broke 64 through.
    for &(head_dim, n_pos) in &[(128usize, 100usize), (128, 200), (64, 40), (64, 200)] {
        let blocks_per_head = head_dim / 32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv: Vec<f32> = (0..n_pos * head_dim)
            .map(|i| ((i * 31 % 97) as f32 / 97.0 - 0.5) * 2.0)
            .collect();
        let vv: Vec<f32> = (0..n_pos * head_dim)
            .map(|i| ((i * 53 % 89) as f32 / 89.0 - 0.5) * 2.0)
            .collect();
        let q: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 17 % 41) as f32 / 41.0 - 0.5) * 2.0)
            .collect();

        // Stage the KV through the device scatter so the test exercises the same
        // quantization the engine uses.
        let d_kv = s.clone_htod(&kv).expect("htod k");
        let d_vv = s.clone_htod(&vv).expect("htod v");
        let mut cache_k: CudaSlice<u8> = s
            .alloc_zeros::<u8>(n_kv_heads * max_pos * blocks_per_head * 34)
            .expect("alloc k");
        let mut cache_v: CudaSlice<u8> = s
            .alloc_zeros::<u8>(n_kv_heads * max_pos * blocks_per_head * 34)
            .expect("alloc v");
        super::launch_kv_scatter_batched_q8_0(
            &s,
            &k.kv_scatter_batched_q8_0,
            &d_kv,
            &mut cache_k,
            0,
            n_kv_heads,
            head_dim,
            max_pos,
            head_dim,
            n_pos,
        )
        .expect("scatter k");
        super::launch_kv_scatter_batched_q8_0(
            &s,
            &k.kv_scatter_batched_q8_0,
            &d_vv,
            &mut cache_v,
            0,
            n_kv_heads,
            head_dim,
            max_pos,
            head_dim,
            n_pos,
        )
        .expect("scatter v");

        // One query at the last position, attending over all n_pos keys.
        let d_q = s.clone_htod(&q).expect("htod q");
        let mut d_out: CudaSlice<f32> = s.alloc_zeros::<f32>(head_dim).expect("alloc out");
        let mut d_scores: CudaSlice<f32> = s
            .alloc_zeros::<f32>(n_heads * max_pos)
            .expect("alloc scores");
        super::launch_attention_batched(
            &s,
            &k.attention_batched_q8_0,
            &d_q,
            &cache_k,
            &cache_v,
            &mut d_out,
            n_heads,
            n_kv_heads,
            head_dim,
            n_pos - 1,
            max_pos,
            scale,
            head_dim,
            1,
            0,
            &mut d_scores,
        )
        .expect("launch attention_batched_q8_0");
        k.ctx.synchronize().expect("sync");
        let got = s.clone_dtoh(&d_out).expect("dtoh out");

        // CPU reference over the SAME dequantized blocks, so the only thing under test
        // is the kernel's attention arithmetic, not the quantizer.
        let deq = |src: &[f32]| -> Vec<f32> {
            let mut blocks = vec![crate::tensor::kv_quant::BlockQ8_0::default(); blocks_per_head];
            let mut out = vec![0.0f32; src.len()];
            for p in 0..n_pos {
                crate::tensor::kv_quant::quantize_row_q8_0(
                    &src[p * head_dim..(p + 1) * head_dim],
                    &mut blocks,
                );
                crate::tensor::kv_quant::dequantize_row_q8_0(
                    &blocks,
                    &mut out[p * head_dim..(p + 1) * head_dim],
                );
            }
            out
        };
        let kd = deq(&kv);
        let vd = deq(&vv);

        let mut scores = vec![0.0f32; n_pos];
        for p in 0..n_pos {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * kd[p * head_dim + d];
            }
            scores[p] = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in scores.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let mut want = vec![0.0f32; head_dim];
        for p in 0..n_pos {
            let w = scores[p] / sum;
            for d in 0..head_dim {
                want[d] += w * vd[p * head_dim + d];
            }
        }

        let worst = (0..head_dim).fold(0.0f32, |a, d| a.max((got[d] - want[d]).abs()));
        assert!(
            worst < 1e-3,
            "q8_0 batched attention diverged from the CPU reference: head_dim {head_dim}, \
             {n_pos} positions (G = {}), worst |delta| {worst}",
            n_pos.div_ceil(head_dim)
        );
    }
}
