//! Explicit mini2-only live-checkpoint proof of the asymmetric Qwen EAGLE cell.
use camelid::eagle3::{Eagle3Bf16Matrix, Eagle3DraftModel, Eagle3Geometry};
use camelid::metal::{Eagle3MetalState, Eagle3MetalWeights};

fn matvec(matrix: &Eagle3Bf16Matrix, input: &[f32]) -> Vec<f32> {
    let [rows, cols] = matrix.shape;
    assert_eq!(input.len(), cols);
    (0..rows)
        .map(|row| {
            matrix.bytes[row * cols * 2..(row + 1) * cols * 2]
                .chunks_exact(2)
                .zip(input)
                .map(|(b, &x)| {
                    f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16) * x
                })
                .sum()
        })
        .collect()
}

fn norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    let scale = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    x.iter().zip(weight).map(|(x, w)| x * scale * w).collect()
}

fn rope(x: &mut [f32], position: usize, theta: f32) {
    for head in x.chunks_exact_mut(128) {
        for j in 0..64 {
            let angle = position as f32 / theta.powf(2.0 * j as f32 / 128.0);
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (head[j], head[j + 64]);
            head[j] = a * cos - b * sin;
            head[j + 64] = b * cos + a * sin;
        }
    }
}

// Round onto the IEEE half lattice independently of the production converter.
fn round_half(value: f32) -> f32 {
    assert!(value.is_finite() && value.abs() <= 65504.0);
    if value == 0.0 {
        return value;
    }
    let exponent = ((value.to_bits() >> 23) & 255) as i32 - 127;
    let quantum = 2.0f32.powi((exponent - 10).max(-24));
    (value / quantum).round_ties_even() * quantum
}

fn close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} width");
    assert!(actual.iter().all(|x| x.is_finite()), "{label} finite");
    let diff: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let magnitude: f64 = expected.iter().map(|x| f64::from(*x).powi(2)).sum();
    let relative = (diff / magnitude.max(1e-20)).sqrt();
    eprintln!("{label}: relative L2={relative:.6e}");
    assert!(
        relative < 0.002,
        "{label}: BF16 cell differs from independent CPU oracle"
    );
}

#[test]
#[ignore = "loads pinned checkpoint and runs Metal; only run sequentially on mini2"]
fn qwen_eagle_cell_matches_independent_cpu_attention() {
    let path = std::env::var("CAMELID_QWEN_EAGLE_CHECKPOINT").expect("checkpoint path");
    let model = Eagle3DraftModel::load(std::path::Path::new(&path)).unwrap();
    assert_eq!(model.config.geometry(), Eagle3Geometry::QWEN);
    for flag in [
        "CAMELID_EAGLE3_BODY_Q4",
        "CAMELID_EAGLE3_LM_HEAD_Q4",
        "CAMELID_EAGLE3_BODY_Q8",
        "CAMELID_EAGLE3_LM_HEAD_Q8",
    ] {
        assert_ne!(
            std::env::var(flag).ok().as_deref(),
            Some("1"),
            "oracle requires original BF16 head"
        );
    }
    let m = &model.matrices;
    let n = &model.norms;
    let weights = Eagle3MetalWeights {
        fc_bf16: &m.feature_fusion.bytes,
        q_proj_bf16: &m.attention_q.bytes,
        k_proj_bf16: &m.attention_k.bytes,
        v_proj_bf16: &m.attention_v.bytes,
        o_proj_bf16: &m.attention_o.bytes,
        gate_proj_bf16: &m.mlp_gate.bytes,
        up_proj_bf16: &m.mlp_up.bytes,
        down_proj_bf16: &m.mlp_down.bytes,
        lm_head_bf16: &m.lm_head.bytes,
        input_layernorm: &n.input,
        hidden_norm: &n.hidden,
        post_attention_layernorm: &n.post_attention,
        output_norm: &n.output,
        d2t_offsets: &model.d2t_offsets,
        rope_theta: model.config.rope_theta,
        sliding_window: None,
    };
    let mut head = Eagle3MetalState::new_with_geometry(weights, 8, Eagle3Geometry::QWEN).unwrap();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let eps = model.config.rms_norm_eps;
    for position in 0..3 {
        let feature: Vec<f32> = (0..7680)
            .map(|i| ((i + position * 71) as f32 * 0.017).sin() * 0.1)
            .collect();
        let g = matvec(&m.feature_fusion, &feature);
        close("feature fusion", &head.fuse_features(&feature).unwrap(), &g);
        let embedding: Vec<f32> = (0..2560)
            .map(|i| ((i + position * 43) as f32 * 0.013).cos() * 0.1)
            .collect();
        let mut combined = norm(&embedding, &n.input, eps);
        combined.extend(norm(&g, &n.hidden, eps));
        let mut q = matvec(&m.attention_q, &combined);
        let mut k = matvec(&m.attention_k, &combined);
        let v = matvec(&m.attention_v, &combined);
        rope(&mut q, position, model.config.rope_theta);
        rope(&mut k, position, model.config.rope_theta);
        // The production cell stores its one-layer KV in half precision.
        let half = |xs: Vec<f32>| xs.into_iter().map(round_half).collect::<Vec<_>>();
        keys.push(half(k));
        values.push(half(v));
        let mut context = vec![0.0; 4096];
        for q_head in 0..32 {
            let kv = (q_head / 4) * 128;
            let query = &q[q_head * 128..(q_head + 1) * 128];
            let mut scores: Vec<f32> = keys
                .iter()
                .map(|key| {
                    query
                        .iter()
                        .zip(&key[kv..kv + 128])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / 128.0f32.sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for score in &mut scores {
                *score = (*score - max).exp();
            }
            let denominator: f32 = scores.iter().sum();
            for (row, score) in scores.into_iter().enumerate() {
                for d in 0..128 {
                    context[q_head * 128 + d] += score / denominator * values[row][kv + d];
                }
            }
        }
        let attention = matvec(&m.attention_o, &context);
        let residual: Vec<f32> = g.iter().zip(attention).map(|(a, b)| a + b).collect();
        let post = norm(&residual, &n.post_attention, eps);
        let gate = matvec(&m.mlp_gate, &post);
        let up = matvec(&m.mlp_up, &post);
        let activation: Vec<f32> = gate
            .iter()
            .zip(up)
            .map(|(a, b)| a / (1.0 + (-a).exp()) * b)
            .collect();
        let down = matvec(&m.mlp_down, &activation);
        let raw: Vec<f32> = residual.iter().zip(down).map(|(a, b)| a + b).collect();
        let logits = matvec(&m.lm_head, &norm(&raw, &n.output, eps));
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        let actual = head.forward_token(&embedding, &g, position).unwrap();
        close("recurrent hidden", &actual.raw_hidden, &raw);
        assert_eq!(
            actual.draft_token as usize, expected,
            "CPU/GPU argmax at position {position}"
        );
        assert_eq!(actual.target_token, model.draft_to_target[expected]);
    }
}
