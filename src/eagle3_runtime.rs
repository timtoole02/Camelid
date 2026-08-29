//! Correctness-first recurrent EAGLE-3 drafting orchestration.
//!
//! The target remains authoritative. This module owns only the learned draft head and its
//! private one-layer KV cache; every proposed token is still checked by the target model's
//! existing greedy speculative verifier before it can be emitted.

use crate::eagle3::{Eagle3DraftModel, HIDDEN_SIZE, TARGET_LAYER_INPUT_IDS};
use crate::error::{BackendError, Result};
use crate::inference::LlamaLoadedWeights;
use crate::metal::{Eagle3MetalOutput, Eagle3MetalState, Eagle3MetalWeights, EAGLE3_AUX_WIDTH};
use crate::tensor::CpuTensor;

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::RuntimeShapeMismatch(message.into())
}

fn metal<T>(result: std::result::Result<T, String>) -> Result<T> {
    result.map_err(|message| invalid(format!("EAGLE-3 Metal runtime: {message}")))
}

/// Interleave three target layer-input captures from `[tap][row][hidden]` into the
/// checkpoint encoder's required `[row][low || middle || high]` layout.
pub fn interleave_target_layer_inputs(captures: &[CpuTensor]) -> Result<Vec<f32>> {
    if captures.len() != TARGET_LAYER_INPUT_IDS.len() {
        return Err(invalid(format!(
            "EAGLE-3 requires {} target layer captures, got {}",
            TARGET_LAYER_INPUT_IDS.len(),
            captures.len()
        )));
    }
    let rows = captures[0].dim(0)?;
    for (slot, capture) in captures.iter().enumerate() {
        let capture_rows = capture.dim(0)?;
        let width = capture.dim(1)?;
        if capture_rows != rows || width != HIDDEN_SIZE {
            return Err(invalid(format!(
                "EAGLE-3 target capture {} (layer input {}) has shape {:?}, expected [{rows}, {HIDDEN_SIZE}]",
                slot, TARGET_LAYER_INPUT_IDS[slot], capture.shape.dims
            )));
        }
    }
    let mut features = vec![0.0f32; rows * EAGLE3_AUX_WIDTH];
    for row in 0..rows {
        for (tap, capture) in captures.iter().enumerate() {
            let source = row * HIDDEN_SIZE;
            let destination = row * EAGLE3_AUX_WIDTH + tap * HIDDEN_SIZE;
            features[destination..destination + HIDDEN_SIZE]
                .copy_from_slice(&capture.data[source..source + HIDDEN_SIZE]);
        }
    }
    Ok(features)
}

/// Linear top-1 EAGLE-3 drafter. `stable_seed` is the output of the newest
/// authoritative head-cache row and therefore predicts the first token of the next round.
pub struct Eagle3Drafter {
    head: Eagle3MetalState,
    stable_seed: Option<Eagle3MetalOutput>,
}

impl Eagle3Drafter {
    fn forward_authoritative_last_output(
        &mut self,
        embeddings: &[f32],
        fused: &[f32],
        start: usize,
    ) -> Result<Eagle3MetalOutput> {
        if std::env::var("CAMELID_EAGLE3_FULL_AUTHORITATIVE")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes" | "enabled"
                )
            })
        {
            return metal(self.head.forward_batch(embeddings, fused, start))?
                .pop()
                .ok_or_else(|| invalid("EAGLE-3 authoritative batch produced no output"));
        }
        metal(
            self.head
                .forward_batch_last_output(embeddings, fused, start),
        )
    }

    /// Upload the validated head to Metal. The large host checkpoint buffers are released
    /// before this returns; `Eagle3MetalState` owns its uploaded copies.
    pub fn new(model: Eagle3DraftModel, max_positions: usize) -> Result<Self> {
        let matrices = &model.matrices;
        let norms = &model.norms;
        let weights = Eagle3MetalWeights {
            fc_bf16: &matrices.feature_fusion.bytes,
            q_proj_bf16: &matrices.attention_q.bytes,
            k_proj_bf16: &matrices.attention_k.bytes,
            v_proj_bf16: &matrices.attention_v.bytes,
            o_proj_bf16: &matrices.attention_o.bytes,
            gate_proj_bf16: &matrices.mlp_gate.bytes,
            up_proj_bf16: &matrices.mlp_up.bytes,
            down_proj_bf16: &matrices.mlp_down.bytes,
            lm_head_bf16: &matrices.lm_head.bytes,
            input_layernorm: &norms.input,
            hidden_norm: &norms.hidden,
            post_attention_layernorm: &norms.post_attention,
            output_norm: &norms.output,
            d2t_offsets: &model.d2t_offsets,
        };
        let head = metal(Eagle3MetalState::new(weights, max_positions))?;
        Ok(Self {
            head,
            stable_seed: None,
        })
    }

    pub fn filled(&self) -> usize {
        self.head.filled()
    }

    /// Seed the stable draft cache from every authoritative prompt row. At head position
    /// `P`, EAGLE consumes `(token[P+1], target_features[P])`; the final prompt row is paired
    /// with the target's freshly sampled, still-unconsumed anchor token.
    pub fn seed_prompt(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        prompt_tokens: &[u32],
        anchor: u32,
        captures: &[CpuTensor],
    ) -> Result<()> {
        if prompt_tokens.is_empty() {
            return Err(invalid("EAGLE-3 prompt must contain at least one token"));
        }
        if self.head.filled() != 0 || self.stable_seed.is_some() {
            return Err(invalid(
                "EAGLE-3 prompt seed may only run on a fresh drafter",
            ));
        }
        let features = interleave_target_layer_inputs(captures)?;
        let rows = features.len() / EAGLE3_AUX_WIDTH;
        if rows != prompt_tokens.len() {
            return Err(invalid(format!(
                "EAGLE-3 prompt capture rows {rows} do not match prompt tokens {}",
                prompt_tokens.len()
            )));
        }
        let fused = metal(self.head.fuse_features(&features))?;
        let mut paired_tokens = Vec::with_capacity(prompt_tokens.len());
        paired_tokens.extend_from_slice(&prompt_tokens[1..]);
        paired_tokens.push(anchor);
        let embeddings = target_weights
            .token_embedding
            .embedding_lookup(&paired_tokens, "eagle3_prompt_next_token_embeddings")?;
        let output = self.forward_authoritative_last_output(&embeddings.data, &fused, 0)?;
        self.stable_seed = Some(output);
        Ok(())
    }

    /// Propose a top-1 linear chain. Recursive rows are ephemeral: the head watermark is
    /// restored to the authoritative stable prefix before this method returns.
    pub fn draft(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let seed = self
            .stable_seed
            .as_ref()
            .ok_or_else(|| invalid("EAGLE-3 must be seeded before drafting"))?;
        let stable = self.head.filled();
        let mut drafts = Vec::with_capacity(max_tokens);
        drafts.push(seed.target_token);
        let mut recurrent = seed.raw_hidden.clone();
        let result = (|| -> Result<()> {
            while drafts.len() < max_tokens {
                let token = *drafts.last().expect("first draft was pushed above");
                let embedding = target_weights
                    .token_embedding
                    .embedding_lookup(&[token], "eagle3_recursive_token_embedding")?;
                let output = metal(self.head.forward_token(
                    &embedding.data,
                    &recurrent,
                    self.head.filled(),
                ))?;
                recurrent = output.raw_hidden;
                drafts.push(output.target_token);
            }
            Ok(())
        })();
        let rollback = metal(self.head.rollback_to_position(stable));
        result?;
        rollback?;
        Ok(drafts)
    }

    /// Extend the stable draft cache using target-verified rows only. `captures` is the
    /// whole verify batch; its first `emitted.len()` rows correspond one-for-one to the
    /// emitted target tokens and are the only rows allowed to survive rejection rollback.
    pub fn accept_authoritative(
        &mut self,
        target_weights: &LlamaLoadedWeights,
        captures: &[CpuTensor],
        emitted: &[u32],
    ) -> Result<()> {
        if emitted.is_empty() {
            return Err(invalid(
                "an EAGLE-3 verify round must emit at least one token",
            ));
        }
        let features = interleave_target_layer_inputs(captures)?;
        let rows = features.len() / EAGLE3_AUX_WIDTH;
        if emitted.len() > rows {
            return Err(invalid(format!(
                "EAGLE-3 verify emitted {} tokens but captured only {rows} target rows",
                emitted.len()
            )));
        }
        let fused = metal(
            self.head
                .fuse_features(&features[..emitted.len() * EAGLE3_AUX_WIDTH]),
        )?;
        let embeddings = target_weights
            .token_embedding
            .embedding_lookup(emitted, "eagle3_authoritative_next_token_embeddings")?;
        let start = self.head.filled();
        let output = self.forward_authoritative_last_output(&embeddings.data, &fused, start)?;
        self.stable_seed = Some(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(name: &str, rows: usize, base: f32) -> CpuTensor {
        let data = (0..rows * HIDDEN_SIZE)
            .map(|index| base + index as f32)
            .collect();
        CpuTensor::from_f32(name, vec![rows, HIDDEN_SIZE], data).unwrap()
    }

    #[test]
    fn target_taps_are_interleaved_row_major_in_trained_order() {
        let captures = vec![
            capture("low", 2, 1.0),
            capture("middle", 2, 10_000.0),
            capture("high", 2, 20_000.0),
        ];
        let fused = interleave_target_layer_inputs(&captures).unwrap();
        assert_eq!(&fused[..HIDDEN_SIZE], &captures[0].data[..HIDDEN_SIZE]);
        assert_eq!(
            &fused[HIDDEN_SIZE..2 * HIDDEN_SIZE],
            &captures[1].data[..HIDDEN_SIZE]
        );
        assert_eq!(
            &fused[2 * HIDDEN_SIZE..3 * HIDDEN_SIZE],
            &captures[2].data[..HIDDEN_SIZE]
        );
        assert_eq!(
            &fused[EAGLE3_AUX_WIDTH..EAGLE3_AUX_WIDTH + HIDDEN_SIZE],
            &captures[0].data[HIDDEN_SIZE..]
        );
    }

    #[test]
    fn target_tap_shape_mismatch_fails_closed() {
        let captures = vec![
            capture("low", 2, 0.0),
            capture("middle", 1, 0.0),
            capture("high", 2, 0.0),
        ];
        assert!(interleave_target_layer_inputs(&captures).is_err());
    }
}
