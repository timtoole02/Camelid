//! Qwen3.5 draft verification. Rejection restores attention, convolution and
//! recurrent state together; truncating KV alone is not sufficient for this graph.
use super::*;

impl RunnableModel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_qwen35_with_draft(
        &self,
        draft: &Self,
        window: usize,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        is_cancelled: &dyn Fn() -> bool,
        should_stop: &dyn Fn(&[u32]) -> bool,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        let (mut cache, logits) = self.prefill_qwen35(prompt, Some(is_cancelled))?;
        let (mut draft_cache, _) = draft.prefill_qwen35(prompt, Some(is_cancelled))?;
        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let mut history = prompt.to_vec();
        let select = |logits: Vec<f32>, history: &[u32]| match &sampler {
            Some(sampler) => qwen35_sample_logits(logits, sampler, history),
            None => Ok(argmax(&logits)),
        };
        let mut next = select(logits, &history)?;
        let mut out = Vec::with_capacity(max_new);
        let mut proposed = 0;
        let mut accepted = 0;
        let started = std::time::Instant::now();
        while out.len() < max_new && !stop.contains(&next) {
            if is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let position = history.len();
            let target_before = cache.clone();
            let draft_before = draft_cache.clone();
            // The first token already belongs to the target. Draft continuations
            // after it, then verify the entire chain with shared weight reads.
            let mut candidates = vec![next];
            let mut draft_consumed = 0;
            let count = (window + 1).min(max_new - out.len());
            while candidates.len() < count {
                if is_cancelled() {
                    return Err(BackendError::Cancelled);
                }
                let logits = draft.decode_token_qwen35(
                    *candidates.last().unwrap(),
                    position + candidates.len() - 1,
                    &mut draft_cache,
                    true,
                )?;
                let token = argmax(&logits);
                draft_consumed += 1;
                if stop.contains(&token) {
                    break;
                }
                candidates.push(token);
            }
            proposed += candidates.len() - 1;
            let hidden =
                self.qwen35_batch_hidden(&candidates, position, &mut cache, Some(is_cancelled))?;
            let normed: Vec<_> = hidden
                .iter()
                .map(|h| self.apply_norm(h, &self.output_norm))
                .collect();
            let logits = self.output.par_matmul(&normed)?;
            let mut committed = 0;
            for (i, row) in logits.into_iter().enumerate() {
                let token = candidates[i];
                out.push(token);
                history.push(token);
                on_token(token);
                committed += 1;
                if is_cancelled() {
                    return Err(BackendError::Cancelled);
                }
                if out.len() == max_new || should_stop(&out) || qwen35_repetition_loop(&out) {
                    eprintln!("[bonsai2-spec] accepted {accepted}/{proposed} drafts; {} tokens in {:.3}s decode", out.len(), started.elapsed().as_secs_f64());
                    return Ok(out);
                }
                // Sample only the target, once per committed position. This is
                // exact target sampling even when the drafter is greedy: its
                // proposals merely predict the target's next draw.
                next = select(row, &history)?;
                if candidates.get(i + 1) != Some(&next) {
                    break;
                }
                accepted += 1;
            }
            if committed != candidates.len() {
                cache = target_before;
                self.qwen35_batch_hidden(
                    &candidates[..committed],
                    position,
                    &mut cache,
                    Some(is_cancelled),
                )?;
            }
            if committed < draft_consumed {
                draft_cache = draft_before;
                draft_consumed = 0;
            }
            if committed > draft_consumed {
                // On full acceptance, retain the drafter's work and only consume
                // its final proposal. On rejection, replay the accepted prefix.
                draft.qwen35_batch_hidden(
                    &candidates[draft_consumed..committed],
                    position + draft_consumed,
                    &mut draft_cache,
                    Some(is_cancelled),
                )?;
            }
        }
        eprintln!(
            "[bonsai2-spec] accepted {accepted}/{proposed} drafts; {} tokens in {:.3}s decode",
            out.len(),
            started.elapsed().as_secs_f64()
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformed_packed_batches_match_individual_tokens() {
        let mut bytes = Vec::new();
        for row in 0..5 {
            for block in 0..2 {
                bytes.extend((0..26).map(|i| ((i * 17 + row * 13 + block * 7) % 256) as u8));
                bytes.extend(0x3800_u16.to_le_bytes());
            }
        }
        let mut mat = RawMat {
            bytes: RawMatBytes::owned(bytes),
            tt: GgufTensorType::Ptq1_0,
            in_features: 256,
            out_features: 5,
            forward_transform: Some(Arc::new(
                super::super::super::hadamard::Transform::test_rotation(256),
            )),
            inverse_transform: None,
            draft_quantized_ptq: false,
        };
        let inputs: Vec<Vec<f32>> = (0..4)
            .map(|p| (0..256).map(|i| ((i * 7 + p * 11) as f32).sin()).collect())
            .collect();
        let expected: Vec<_> = inputs
            .iter()
            .map(|x| mat.par_matvec(x, "test").unwrap())
            .collect();
        assert_eq!(mat.par_matmul(&inputs).unwrap(), expected);
        mat.draft_quantized_ptq = true;
        let expected: Vec<_> = inputs
            .iter()
            .map(|x| mat.par_matvec(x, "draft test").unwrap())
            .collect();
        assert_eq!(mat.par_matmul(&inputs).unwrap(), expected);
    }

    /// Run with CAMELID_BONSAI_TARGET and CAMELID_BONSAI_DRAFT set to real packs.
    #[test]
    #[ignore = "requires official Bonsai model files"]
    fn bonsai2_batch_cache_and_draft_parity() {
        let target_path = std::env::var("CAMELID_BONSAI_TARGET").unwrap();
        let draft_path = std::env::var("CAMELID_BONSAI_DRAFT").unwrap();
        let target = RunnableModel::load(&target_path).unwrap();
        let draft = RunnableModel::load_with_draft_activations(&draft_path, true).unwrap();
        let tokenizer =
            crate::tokenizer::Tokenizer::from_gguf(&read_metadata(&target_path).unwrap()).unwrap();
        let prompt = tokenizer
            .encode(
                "<|im_start|>user\nCount from one to ten.<|im_end|>\n<|im_start|>assistant\n",
                false,
                true,
            )
            .unwrap();
        let (initial, _) = target.prefill_qwen35(&prompt, None).unwrap();
        let suffix = tokenizer.encode("One, two, three", false, true).unwrap();
        let mut serial = initial.clone();
        let mut serial_logits = Vec::new();
        for (i, &token) in suffix.iter().enumerate() {
            serial_logits.push(
                target
                    .decode_token_qwen35(token, prompt.len() + i, &mut serial, true)
                    .unwrap(),
            );
        }
        let mut batch = initial.clone();
        let hidden = target
            .qwen35_batch_hidden(&suffix, prompt.len(), &mut batch, None)
            .unwrap();
        let normed: Vec<_> = hidden
            .iter()
            .map(|h| target.apply_norm(h, &target.output_norm))
            .collect();
        assert_eq!(target.output.par_matmul(&normed).unwrap(), serial_logits);
        assert_eq!(batch.k, serial.k);
        assert_eq!(batch.v, serial.v);
        assert_eq!(batch.conv, serial.conv);
        assert_eq!(batch.state, serial.state);
        let mut expected = Vec::new();
        let (mut cache, logits) = target.prefill_qwen35(&prompt, None).unwrap();
        let mut next = argmax(&logits);
        for i in 0..24 {
            expected.push(next);
            next = argmax(
                &target
                    .decode_token_qwen35(next, prompt.len() + i, &mut cache, true)
                    .unwrap(),
            );
        }
        let actual = target
            .generate_qwen35_with_draft(
                &draft,
                3,
                &prompt,
                24,
                &[],
                None,
                &|| false,
                &|_| false,
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(actual, expected);
    }
}
