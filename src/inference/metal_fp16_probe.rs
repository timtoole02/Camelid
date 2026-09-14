// Included in metal_resident.rs. Diagnostic bridge from the original Q4 GGUF
// weights/config to an independent FP16 target, with no ordinary session mutation.
impl super::LlamaInferenceSession {
    /// Create an isolated, default-off 3B FP16 diagnostic target. Mirror setup is
    /// synchronous and belongs outside forward timings. The ordinary session's
    /// resident engine and KV position are not accessed or changed.
    pub fn create_fp16_probe(
        &self,
        max_sequence_length: usize,
    ) -> Result<Option<metal::ResidentFp16Probe>> {
        if !metal::fp16_probe_enabled()
            || self.config.architecture != "llama"
            || self.weights.layer_range.is_some()
            || max_sequence_length == 0
            || max_sequence_length > 2048
            || max_sequence_length > self.config.context_length as usize
            || !self.resident_decode_eligible(true)?
        {
            return Ok(None);
        }
        let dims = DenseLlamaDims::from_config(&self.config)?;
        if (
            dims.block_count,
            dims.embedding_length,
            dims.feed_forward_length,
            dims.vocab_size,
            self.config.attention_head_count as usize,
            dims.attention_head_count_kv,
            dims.head_dim,
        ) != (28, 3072, 8192, 128256, 24, 8, 128)
        {
            return Ok(None);
        }
        let weights = &self.weights;
        let Some(rope_tables) = rope::resident_decode_rope_tables(
            0,
            dims.head_dim,
            &self.config,
            weights.rope_freqs.as_ref(),
        )?
        else {
            return Ok(None);
        };
        if rope_tables.cos.len() != 64 || rope_tables.sin.len() != 64 {
            return Ok(None);
        }
        let layer_views: Vec<metal::ResidentLayerWeights> = weights
            .layers
            .iter()
            .map(|layer| metal::ResidentLayerWeights {
                attn_norm: &layer.attention_norm.data,
                ffn_norm: &layer.ffn_norm.data,
                q_norm: layer.attention_q_norm.as_ref().map(|t| t.data.as_slice()),
                k_norm: layer.attention_k_norm.as_ref().map(|t| t.data.as_slice()),
                post_attn_norm: layer
                    .post_attention_norm
                    .as_ref()
                    .map(|t| t.data.as_slice()),
                post_ffw_norm: layer.post_ffw_norm.as_ref().map(|t| t.data.as_slice()),
                ffn_geglu: false,
                q_weight_blocks: resident_weight_bytes(&layer.attention_q),
                k_weight_blocks: resident_weight_bytes(&layer.attention_k),
                v_weight_blocks: resident_weight_bytes(&layer.attention_v),
                o_weight_blocks: resident_weight_bytes(&layer.attention_output),
                gate_weight_blocks: resident_weight_bytes(&layer.ffn_gate),
                up_weight_blocks: resident_weight_bytes(&layer.ffn_up),
                down_weight_blocks: resident_weight_bytes(&layer.ffn_down),
            })
            .collect();
        let logits = metal::LogitsStage {
            final_norm: &weights.output_norm.data,
            output_weight_blocks: resident_weight_bytes(weights.output_projection()),
            vocab_size: dims.vocab_size,
            output_is_tied_embedding: weights.output.is_none(),
        };
        Ok(metal::ResidentFp16Probe::new(
            &layer_views,
            &logits,
            max_sequence_length,
            diagnostic_rms_norm_epsilon(self.config.rms_norm_epsilon)?,
            rope_tables.split_half_pairing,
            Arc::as_ptr(weights) as usize,
        ))
    }

    /// Teacher-force one chunk (1..64 inputs) against the probe's own causal
    /// prefix and return one greedy prediction per input. Feed the prediction
    /// from a single input back here for an independent serial FP16 reference.
    /// Always use this method for the prompt too; it never imports V4 KV.
    pub fn forward_fp16_probe_tokens(
        &self,
        probe: &mut metal::ResidentFp16Probe,
        input_tokens: &[u32],
    ) -> Result<Option<Vec<u32>>> {
        if !metal::fp16_probe_enabled()
            || !probe.has_source(Arc::as_ptr(&self.weights) as usize)
            || !(1..=64).contains(&input_tokens.len())
            || probe.filled() + input_tokens.len() > probe.max_positions()
        {
            return Ok(None);
        }
        let embeddings = self
            .weights
            .token_embedding
            .embedding_lookup(input_tokens, "fp16_diagnostic_token_embedding")?;
        let mut cos = Vec::with_capacity(input_tokens.len() * 64);
        let mut sin = Vec::with_capacity(input_tokens.len() * 64);
        for row in 0..input_tokens.len() {
            let Some(tables) = rope::resident_decode_rope_tables(
                probe.filled() + row,
                128,
                &self.config,
                self.weights.rope_freqs.as_ref(),
            )?
            else {
                return Ok(None);
            };
            cos.extend_from_slice(&tables.cos);
            sin.extend_from_slice(&tables.sin);
        }
        let scale = attention_score_scale_value(128, diagnostic_attention_score_scale()?);
        Ok(probe.forward(&embeddings.data, &cos, &sin, scale))
    }
}
