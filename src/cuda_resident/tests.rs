//! Per-kernel parity tests for the resident-decode kernels. Each test runs one
//! kernel on the GPU and compares to a small CPU reference, so a divergence is
//! isolated to a single kernel. All require a CUDA device (`#[ignore]`d in
//! GPU-less CI); run with `cargo test --features cuda -- --ignored`.

use super::{CudaResidentDecode, CudaResidentKernels, ProjQuant};
use cudarc::driver::{LaunchConfig, PushKernelArg};

// Pure predicate (no GPU): the device-decode embed-gather allowlist must stay in
// lockstep with the `embed_gather_*` dispatch in `forward_token_device`. Families
// without a kernel (Q5_K/Q2_K/IQ4_XS) must be refused at `set_device_decode_tables`
// so the engine falls back to the host-fed loop instead of failing mid-forward.
#[test]
fn device_embed_gather_allowlist_matches_the_gather_dispatch() {
    for q in [
        ProjQuant::Q8_0,
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

fn kernels() -> Option<CudaResidentKernels> {
    CudaResidentKernels::new().ok()
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        let d = (x - y).abs() / y.abs().max(1.0);
        d < tol
    })
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
    let dk = k.stream.clone_htod(&cache_k_bits).unwrap();
    let dv = k.stream.clone_htod(&cache_v_bits).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n_heads * head_dim).unwrap();
    let (nh, nkv, hd, mp) = (n_heads as i32, n_kv as i32, head_dim as i32, max_pos as i32);
    // The kernel reads position from device memory and uses position_count = pos+1.
    let dpos = k.stream.clone_htod(&[(position_count - 1) as i32]).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: ((head_dim + position_count) * 4) as u32,
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
        .arg(&scale);
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
    let dk = k.stream.clone_htod(&cache_k_bits).unwrap();
    let dv = k.stream.clone_htod(&cache_v_bits).unwrap();

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
                n_heads, n_kv, head_dim, &dpos, pc, max_pos, scale,
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
            )
            .unwrap();
        }
        let mut ref_out = vec![0f32; outlen];
        k.stream.memcpy_dtoh(&dref, &mut ref_out).unwrap();

        // Linear verify: attention_batched, single token at absolute position pc-1, splitk_active=1.
        let mut dver = k.stream.alloc_zeros::<f32>(outlen).unwrap();
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
    let dk = k.stream.clone_htod(&cache_k_bits).unwrap();
    let dv = k.stream.clone_htod(&cache_v_bits).unwrap();
    let mut dout = k.stream.alloc_zeros::<f32>(n_heads * head_dim).unwrap();
    let (nh, nkv, hd, mp) = (n_heads as i32, n_kv as i32, head_dim as i32, max_pos as i32);
    let win = window as i32;
    let dpos = k.stream.clone_htod(&[(position_count - 1) as i32]).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: ((head_dim + position_count) * 4) as u32,
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
        .arg(&win);
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
    let (rows, n_sb, k_tokens) = (64usize, 3usize, 4usize);
    let kdim = n_sb * 256;
    let mut rng = Lcg(0x6b_47_45_4d_4d);
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
    let glog: Vec<f32> = (0..nv)
        .map(|_| -(rng.next_f32() * 0.5 + 0.5) * 0.1)
        .collect(); // <= 0
    let norm: Vec<f32> = (0..ds).map(|_| rng.next_f32() * 0.5 + 1.0).collect();
    let silu = |v: f32| v / (1.0 + (-v).exp());
    // CPU reference (matches qwen35_ssm_compute recurrence + gated RMSNorm).
    let mut exp_state = state0.clone();
    let mut exp_out = vec![0f32; nv * ds];
    let qscale = 1.0f32 / (ds as f32).sqrt();
    for h in 0..nv {
        let hk = h % nk;
        let g = glog[h].exp();
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
    let dglog = k.stream.clone_htod(&glog).unwrap();
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
        .arg(&dglog)
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
    let exp_glog: Vec<f32> = (0..nv)
        .map(|h| softplus(alpha_raw[h] + dt_bias[h]) * a[h])
        .collect();
    let dbr = k.stream.clone_htod(&beta_raw).unwrap();
    let dar = k.stream.clone_htod(&alpha_raw).unwrap();
    let ddt = k.stream.clone_htod(&dt_bias).unwrap();
    let da = k.stream.clone_htod(&a).unwrap();
    let mut dbeta = k.stream.alloc_zeros::<f32>(nv).unwrap();
    let mut dglog = k.stream.alloc_zeros::<f32>(nv).unwrap();
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
        .arg(&mut dglog)
        .arg(&nvi);
    unsafe { b.launch(cfg).unwrap() };
    let mut got_beta = vec![0f32; nv];
    let mut got_glog = vec![0f32; nv];
    k.stream.memcpy_dtoh(&dbeta, &mut got_beta).unwrap();
    k.stream.memcpy_dtoh(&dglog, &mut got_glog).unwrap();
    k.ctx.synchronize().unwrap();
    assert!(close(&got_beta, &exp_beta, 2e-3), "ssm_gates beta diverged");
    assert!(close(&got_glog, &exp_glog, 2e-3), "ssm_gates glog diverged");
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
    let mut glog = vec![0f32; nv];
    for h in 0..nv {
        beta[h] = sigmoid(beta_raw[h]);
        glog[h] = softplus(alpha_raw[h] + dt_bias[h]) * a_vec[h];
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
        let g = glog[h].exp();
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
    let mut dglog = k.stream.alloc_zeros::<f32>(nv).unwrap();
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
            .arg(&mut dglog)
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
            .arg(&dglog)
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
