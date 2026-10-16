//! Qwen3.5 draft verification. Rejection restores attention, convolution and
//! recurrent state together; truncating KV alone is not sufficient for this graph.
use super::*;

#[cfg(feature = "cuda")]
pub(super) struct CachedCudaDraft {
    path: std::path::PathBuf,
    len: u64,
    modified: Option<std::time::SystemTime>,
    max_pos: usize,
    model: Box<RunnableModel>,
}

fn draft_configuration() -> Result<(String, usize)> {
    let path = std::env::var("CAMELID_SPEC_DRAFT_MODEL").map_err(|_| {
        BackendError::InvalidModelMetadata("draft mode requires CAMELID_SPEC_DRAFT_MODEL".into())
    })?;
    let window = std::env::var("CAMELID_SPEC_DRAFT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3)
        .clamp(1, 8);
    Ok((path, window))
}

impl RunnableModel {
    pub(super) fn load_qwen35_draft(&self, integer_activations: bool) -> Result<(Self, usize)> {
        let (path, window) = draft_configuration()?;
        Ok((
            self.load_qwen35_draft_path(&path, integer_activations)?,
            window,
        ))
    }

    fn load_qwen35_draft_path(&self, path: &str, integer_activations: bool) -> Result<Self> {
        let target_tokens =
            crate::tokenizer::Tokenizer::from_gguf(&read_metadata(&self.source_path)?)?;
        let draft_tokens = crate::tokenizer::Tokenizer::from_gguf(&read_metadata(path)?)?;
        if target_tokens.model != draft_tokens.model
            || target_tokens.tokens.len() != draft_tokens.tokens.len()
            || !target_tokens
                .tokens
                .iter()
                .zip(&draft_tokens.tokens)
                .all(|(a, b)| a.text == b.text)
        {
            return Err(BackendError::InvalidModelMetadata(
                "draft and target token mappings differ".into(),
            ));
        }
        let draft = Self::load_with_draft_activations(path, integer_activations)?;
        if draft.qwen35.is_none() {
            return Err(BackendError::InvalidModelMetadata(
                "Bonsai draft must use the qwen35 runtime".into(),
            ));
        }
        Ok(draft)
    }

    /// Both graphs execute on CUDA. Only token selection and the small token
    /// embedding/RoPE staging buffers cross the CPU boundary. The cached drafter
    /// is built before the target so the target's offload budget accounts for it.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_qwen35_cuda_with_draft(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        is_cancelled: &dyn Fn() -> bool,
        should_stop: &dyn Fn(&[u32]) -> bool,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        if max_new == 0 {
            return Ok(Vec::new());
        }
        if is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let (path, window) = draft_configuration()?;
        let canonical_path = std::fs::canonicalize(&path).map_err(|error| {
            BackendError::InvalidModelMetadata(format!("draft model path: {error}"))
        })?;
        let metadata = std::fs::metadata(&canonical_path).map_err(|error| {
            BackendError::InvalidModelMetadata(format!("draft model metadata: {error}"))
        })?;
        let modified = metadata.modified().ok();
        let max_pos = std::env::var("CAMELID_QWEN35_CUDA_MAXPOS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8192usize);
        if prompt.is_empty() || prompt.len() >= max_pos {
            return Err(BackendError::InvalidTensorData(format!(
                "CUDA draft prompt length {} is outside 1..{max_pos}",
                prompt.len()
            )));
        }
        let mut cached = self.cuda_draft.lock().map_err(|_| {
            BackendError::InvalidTensorData("qwen35 CUDA draft mutex poisoned".into())
        })?;
        let mut target_guard = self.cuda.lock().map_err(|_| {
            BackendError::InvalidTensorData("qwen35 CUDA target mutex poisoned".into())
        })?;
        // A queued request may have been cancelled while another request held
        // these graphs. Do not start loading or allocating its replacement.
        if is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let replace_draft = cached.as_ref().is_none_or(|cached| {
            cached.path != canonical_path
                || cached.len != metadata.len()
                || cached.modified != modified
                || cached.max_pos != max_pos
        });
        if replace_draft {
            // Validate before replacing a usable cached graph. Rebuilding the
            // target is necessary only when its proposal model changes: the old
            // residency plan may have consumed the drafter's required VRAM.
            let model = self.load_qwen35_draft_path(&path, false)?;
            if is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            if let Some(engine) = target_guard.as_ref() {
                engine
                    .synchronize()
                    .map_err(BackendError::InvalidTensorData)?;
            }
            *target_guard = None;
            *cached = Some(CachedCudaDraft {
                path: canonical_path,
                len: metadata.len(),
                modified,
                max_pos,
                model: Box::new(model),
            });
        }
        let draft = &cached.as_ref().expect("initialized above").model;
        let mut draft_guard = draft.cuda.lock().map_err(|_| {
            BackendError::InvalidTensorData("qwen35 CUDA proposal mutex poisoned".into())
        })?;
        if draft_guard.is_none() {
            *draft_guard = Some(
                draft
                    .build_qwen35_resident(max_pos)
                    .map_err(BackendError::InvalidTensorData)?,
            );
        }
        if is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        if target_guard.is_none() {
            *target_guard = Some(
                self.build_qwen35_resident(max_pos)
                    .map_err(BackendError::InvalidTensorData)?,
            );
        }
        if is_cancelled() {
            return Err(BackendError::Cancelled);
        }
        let target = target_guard.as_mut().expect("initialized above");
        let proposal = draft_guard.as_mut().expect("initialized above");
        target
            .reset_qwen35_state()
            .map_err(BackendError::InvalidTensorData)?;
        proposal
            .reset_qwen35_state()
            .map_err(BackendError::InvalidTensorData)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let draft_scale = 1.0f32 / (draft.head_dim as f32).sqrt();
        let logits = self.prefill_qwen35_cuda_draft(target, prompt, scale, is_cancelled)?;
        // The proposal graph only needs prompt state. Its next prediction comes
        // after consuming the already-selected target token, so projecting and
        // downloading the prompt's unused vocabulary row would waste work.
        draft.prefill_qwen35_cuda_prefix(proposal, prompt, draft_scale, is_cancelled)?;
        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let select = |logits: Vec<f32>, history: &[u32]| match &sampler {
            Some(sampler) => qwen35_sample_logits(logits, sampler, history),
            None => Ok(argmax(&logits)),
        };
        let mut history = prompt.to_vec();
        let mut next = select(logits, &history)?;
        let mut out = Vec::with_capacity(max_new);
        let mut proposed = 0usize;
        let mut accepted = 0usize;
        let started = std::time::Instant::now();
        let mut target_before = target
            .snapshot_qwen35_state()
            .map_err(BackendError::InvalidTensorData)?;
        let mut draft_before = proposal
            .snapshot_qwen35_state()
            .map_err(BackendError::InvalidTensorData)?;
        while out.len() < max_new && !stop.contains(&next) {
            if is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let position = history.len();
            let count = (window + 1)
                .min(max_new - out.len())
                .min(max_pos.saturating_sub(position));
            if count == 0 {
                break;
            }
            target
                .refresh_qwen35_state(&mut target_before)
                .map_err(BackendError::InvalidTensorData)?;
            proposal
                .refresh_qwen35_state(&mut draft_before)
                .map_err(BackendError::InvalidTensorData)?;
            let mut candidates = vec![next];
            let mut draft_consumed = 0usize;
            if proposal.device_decode_ready() && count > 1 {
                // Keep the proposal chain on the device and synchronize once.
                // Its bounded window can run past a draft EOS; if that happens,
                // the existing rollback below replays only committed inputs.
                for offset in 0..count - 1 {
                    if is_cancelled() {
                        return Err(BackendError::Cancelled);
                    }
                    let pos = position + offset;
                    proposal
                        .forward_token_device(
                            offset.checked_sub(1),
                            (offset == 0).then_some(next),
                            pos,
                            pos,
                            draft_scale,
                            Some(offset),
                        )
                        .map_err(BackendError::InvalidTensorData)?;
                    draft_consumed += 1;
                }
                let ids = proposal
                    .read_out_tokens(0, draft_consumed)
                    .map_err(BackendError::InvalidTensorData)?;
                proposal.set_filled(position + draft_consumed);
                candidates.extend(ids.into_iter().take_while(|token| !stop.contains(token)));
            } else {
                // CUDA graphs with an embedding format lacking device gather
                // still run every layer on CUDA; only the embedding row is staged.
                while candidates.len() < count {
                    if is_cancelled() {
                        return Err(BackendError::Cancelled);
                    }
                    let token = *candidates.last().expect("first target token exists");
                    let pos = position + candidates.len() - 1;
                    let embedding = draft.token_embd.dequant_row(token as usize, "token_embd")?;
                    let (cos, sin) = qwen35_rope_tables(pos, draft.rope_base, draft.rope_dim);
                    let next_draft = proposal
                        .forward_token(&embedding, &cos, &sin, pos, draft_scale, true)
                        .map_err(BackendError::InvalidTensorData)?
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData("CUDA drafter returned no token".into())
                        })?;
                    proposal.set_filled(pos + 1);
                    draft_consumed += 1;
                    if stop.contains(&next_draft) {
                        break;
                    }
                    candidates.push(next_draft);
                }
            }
            proposed += candidates.len() - 1;
            if is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let (embeddings, cos, sin) = self.qwen35_cuda_draft_rows(&candidates, position)?;
            let logits = target
                .verify_qwen35_batch_logits(
                    &embeddings,
                    &cos,
                    &sin,
                    position,
                    candidates.len(),
                    scale,
                )
                .map_err(BackendError::InvalidTensorData)?;
            // Verification synchronizes after the device finishes the chain.
            // Honor a cancellation that arrived while it was in flight before
            // publishing any of that chain to the GUI.
            if is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let mut committed = 0usize;
            for (index, row) in logits.into_iter().enumerate() {
                let token = candidates[index];
                out.push(token);
                history.push(token);
                on_token(token);
                committed += 1;
                if is_cancelled() {
                    return Err(BackendError::Cancelled);
                }
                if out.len() == max_new || should_stop(&out) || qwen35_repetition_loop(&out) {
                    eprintln!("[bonsai2-spec-cuda] accepted {accepted}/{proposed} drafts; {} tokens in {:.3}s decode", out.len(), started.elapsed().as_secs_f64());
                    return Ok(out);
                }
                next = select(row, &history)?;
                if candidates.get(index + 1) != Some(&next) {
                    break;
                }
                accepted += 1;
            }
            if committed != candidates.len() {
                target
                    .restore_qwen35_state(&target_before)
                    .map_err(BackendError::InvalidTensorData)?;
                self.consume_qwen35_cuda_draft(
                    target,
                    &candidates[..committed],
                    position,
                    scale,
                    is_cancelled,
                )?;
            }
            if committed < draft_consumed {
                proposal
                    .restore_qwen35_state(&draft_before)
                    .map_err(BackendError::InvalidTensorData)?;
                draft_consumed = 0;
            }
            if committed > draft_consumed {
                draft.consume_qwen35_cuda_draft(
                    proposal,
                    &candidates[draft_consumed..committed],
                    position + draft_consumed,
                    draft_scale,
                    is_cancelled,
                )?;
            }
        }
        eprintln!(
            "[bonsai2-spec-cuda] accepted {accepted}/{proposed} drafts; {} tokens in {:.3}s decode",
            out.len(),
            started.elapsed().as_secs_f64()
        );
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn qwen35_cuda_draft_rows(
        &self,
        tokens: &[u32],
        position: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut embeddings = Vec::with_capacity(tokens.len() * self.d_model);
        let mut cos_all = Vec::with_capacity(tokens.len() * self.rope_dim / 2);
        let mut sin_all = Vec::with_capacity(tokens.len() * self.rope_dim / 2);
        for (offset, &token) in tokens.iter().enumerate() {
            embeddings.extend(self.token_embd.dequant_row(token as usize, "token_embd")?);
            let (cos, sin) = qwen35_rope_tables(position + offset, self.rope_base, self.rope_dim);
            cos_all.extend(cos);
            sin_all.extend(sin);
        }
        Ok((embeddings, cos_all, sin_all))
    }

    #[cfg(feature = "cuda")]
    fn consume_qwen35_cuda_draft(
        &self,
        engine: &mut crate::cuda_resident::CudaResidentDecode,
        tokens: &[u32],
        position: usize,
        scale: f32,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<()> {
        // Replays are short accepted prefixes after a rejection. The ordinary
        // serial CUDA forward is the authoritative state transition here.
        for (offset, &token) in tokens.iter().enumerate() {
            if is_cancelled() {
                return Err(BackendError::Cancelled);
            }
            let embedding = self.token_embd.dequant_row(token as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(position + offset, self.rope_base, self.rope_dim);
            engine
                .forward_token(&embedding, &cos, &sin, position + offset, scale, false)
                .map_err(BackendError::InvalidTensorData)?;
        }
        engine.set_filled(position + tokens.len());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn prefill_qwen35_cuda_prefix(
        &self,
        engine: &mut crate::cuda_resident::CudaResidentDecode,
        prompt: &[u32],
        scale: f32,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<()> {
        let mut embeddings = Vec::with_capacity(prompt.len() * self.d_model);
        let mut cos_all = Vec::with_capacity(prompt.len() * self.rope_dim / 2);
        let mut sin_all = Vec::with_capacity(prompt.len() * self.rope_dim / 2);
        qwen35_cuda_prefill_segments(prompt, is_cancelled, &mut |base, tokens| {
            let (rows, cos, sin) = self.qwen35_cuda_draft_rows(tokens, base)?;
            embeddings.extend(rows);
            cos_all.extend(cos);
            sin_all.extend(sin);
            // Both lanes preserve per-token arithmetic and synchronize at the
            // existing cancellation boundary. Packed PTQ batches additionally
            // amortize projection launches and streamed weights across the chunk.
            if engine.prefers_batched_prefill() {
                engine.prefill_batched_from(
                    &embeddings,
                    &cos_all,
                    &sin_all,
                    base + tokens.len(),
                    scale,
                    base,
                )
            } else {
                engine.prefill_from(
                    &embeddings,
                    &cos_all,
                    &sin_all,
                    base + tokens.len(),
                    scale,
                    base,
                )
            }
            .map_err(BackendError::InvalidTensorData)?;
            engine.set_filled(base + tokens.len());
            Ok(())
        })
    }

    #[cfg(feature = "cuda")]
    fn prefill_qwen35_cuda_draft(
        &self,
        engine: &mut crate::cuda_resident::CudaResidentDecode,
        prompt: &[u32],
        scale: f32,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        let (&last, prior) = prompt
            .split_last()
            .ok_or_else(|| BackendError::InvalidTensorData("empty draft prompt".into()))?;
        self.prefill_qwen35_cuda_prefix(engine, prior, scale, is_cancelled)?;
        let embedding = self.token_embd.dequant_row(last as usize, "token_embd")?;
        let (cos, sin) = qwen35_rope_tables(prior.len(), self.rope_base, self.rope_dim);
        let logits = engine
            .forward_token_logits(&embedding, &cos, &sin, prior.len(), scale)
            .map_err(BackendError::InvalidTensorData)?;
        engine.set_filled(prompt.len());
        Ok(logits)
    }

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

    #[cfg(feature = "cuda")]
    fn assert_bits_equal(actual: &[u32], expected: &[u32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length differs");
        if let Some((index, (&actual, &expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            panic!("{label}: first mismatch at {index}: {actual:08x} != {expected:08x}");
        }
    }

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

    /// Score fixed float-path continuations without feeding Q8 choices back in.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA and an official Bonsai model"]
    fn bonsai2_cuda_teacher_forced_quality() {
        // Score a fixed continuation under each arithmetic policy. Greedy text
        // alone can hide near ties or accumulated probability drift.
        let target_path = std::env::var("CAMELID_BONSAI_TARGET").unwrap();
        let cases_path = std::env::var("CAMELID_BONSAI_QUALITY_CASES").unwrap();
        let output_path = std::env::var("CAMELID_BONSAI_QUALITY_OUTPUT").unwrap();
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(cases_path).unwrap()).unwrap();
        let target = RunnableModel::load(&target_path).unwrap();
        let tokenizer =
            crate::tokenizer::Tokenizer::from_gguf(&read_metadata(&target_path).unwrap()).unwrap();
        let scale = 1.0_f32 / (target.head_dim as f32).sqrt();
        let mut engine = target.build_qwen35_resident(2048).unwrap();
        let mut report = Vec::new();
        for (name, prompt) in [
            ("count", include_str!("../../qa/bonsai2/count.chatml")),
            ("code", include_str!("../../qa/bonsai2/code.chatml")),
            ("explain", include_str!("../../qa/bonsai2/explain.chatml")),
        ] {
            let tokens = tokenizer.encode(prompt, false, true).unwrap();
            let case = cases["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["name"] == name)
                .unwrap();
            let continuation: Vec<u32> = case["baseline"]["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(continuation.len(), 32);
            engine.reset_qwen35_state().unwrap();
            let mut logits = target
                .prefill_qwen35_cuda_draft(&mut engine, &tokens, scale, &|| false)
                .unwrap();
            let mut scores = Vec::new();
            for (offset, &token) in continuation.iter().enumerate() {
                assert!(logits.iter().all(|v| v.is_finite()));
                let (argmax, &maximum) = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                let normalizer = logits
                    .iter()
                    .map(|&v| (f64::from(v) - f64::from(maximum)).exp())
                    .sum::<f64>()
                    .ln()
                    + f64::from(maximum);
                scores.push(serde_json::json!({ "token": token, "argmax": argmax,
                    "log_probability": f64::from(logits[token as usize]) - normalizer,
                    "top1_log_probability": f64::from(maximum) - normalizer }));
                if offset + 1 < continuation.len() {
                    let position = tokens.len() + offset;
                    let embedding = target
                        .token_embd
                        .dequant_row(token as usize, "token_embd")
                        .unwrap();
                    let (cos, sin) =
                        qwen35_rope_tables(position, target.rope_base, target.rope_dim);
                    logits = engine
                        .forward_token_logits(&embedding, &cos, &sin, position, scale)
                        .unwrap();
                    engine.set_filled(position + 1);
                }
            }
            report.push(serde_json::json!({ "name": name, "scores": scores }));
        }
        std::fs::write(output_path, serde_json::to_vec_pretty(&serde_json::json!({
            "model": target_path,
            "projection_policy": std::env::var("CAMELID_BONSAI2_CUDA_PROJECTION").unwrap_or_else(|_| "tiled".into()),
            "cases": report,
        })).unwrap()).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA and an official Bonsai model"]
    fn bonsai2_cuda_batch_cache_and_rollback_parity() {
        let target_path = std::env::var("CAMELID_BONSAI_TARGET").unwrap();
        let target = RunnableModel::load(&target_path).unwrap();
        let tokenizer =
            crate::tokenizer::Tokenizer::from_gguf(&read_metadata(&target_path).unwrap()).unwrap();
        let mut prompt = tokenizer
            .encode(
                "<|im_start|>user\nExplain why the sky is blue.<|im_end|>\n<|im_start|>assistant\n",
                false,
                true,
            )
            .unwrap();
        if let Ok(context) = std::env::var("CAMELID_BONSAI_TEST_CONTEXT") {
            // Four verified inputs straddle either the first graph-cache
            // attention-group transition or the graph/split-K boundary.
            let context = match context.as_str() {
                "group-boundary" => target.head_dim.saturating_sub(2),
                "splitk-boundary" => 510,
                value => value.parse::<usize>().expect("invalid CUDA test context"),
            };
            let token = *prompt.last().unwrap();
            prompt.resize(context.max(prompt.len()), token);
        }
        let suffix = tokenizer
            .encode("The sky is blue because", false, true)
            .unwrap();
        assert!(suffix.len() >= 4);
        let suffix = &suffix[..4];
        let scale = 1.0f32 / (target.head_dim as f32).sqrt();
        let mut engine = target
            .build_qwen35_resident((prompt.len() + 64).max(256))
            .unwrap();
        engine.reset_qwen35_state().unwrap();
        let batch_prefill_logits = target
            .prefill_qwen35_cuda_draft(&mut engine, &prompt, scale, &|| false)
            .unwrap();
        let batch_prefill_state = engine.qwen35_state_bits().unwrap();
        engine.reset_qwen35_state().unwrap();
        let (&last_prompt, prior_prompt) = prompt.split_last().unwrap();
        target
            .consume_qwen35_cuda_draft(&mut engine, prior_prompt, 0, scale, &|| false)
            .unwrap();
        let last_embedding = target
            .token_embd
            .dequant_row(last_prompt as usize, "token_embd")
            .unwrap();
        let (cos, sin) = qwen35_rope_tables(prior_prompt.len(), target.rope_base, target.rope_dim);
        let serial_prefill_logits = engine
            .forward_token_logits(&last_embedding, &cos, &sin, prior_prompt.len(), scale)
            .unwrap();
        engine.set_filled(prompt.len());
        assert_bits_equal(
            &batch_prefill_logits
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            &serial_prefill_logits
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            "batched CUDA prefill logits",
        );
        assert_bits_equal(
            &batch_prefill_state,
            &engine.qwen35_state_bits().unwrap(),
            "batched CUDA prefill recurrent/KV state",
        );
        let initial = engine.snapshot_qwen35_state().unwrap();
        let mut serial_logits = Vec::new();
        for (offset, &token) in suffix.iter().enumerate() {
            let position = prompt.len() + offset;
            let embedding = target
                .token_embd
                .dequant_row(token as usize, "token_embd")
                .unwrap();
            let (cos, sin) = qwen35_rope_tables(position, target.rope_base, target.rope_dim);
            serial_logits.push(
                engine
                    .forward_token_logits(&embedding, &cos, &sin, position, scale)
                    .unwrap(),
            );
            engine.set_filled(position + 1);
        }
        let serial_state = engine.qwen35_state_bits().unwrap();
        engine.restore_qwen35_state(&initial).unwrap();
        let (embeddings, cos, sin) = target.qwen35_cuda_draft_rows(suffix, prompt.len()).unwrap();
        let batch_logits = engine
            .verify_qwen35_batch_logits(&embeddings, &cos, &sin, prompt.len(), suffix.len(), scale)
            .unwrap();
        let to_bits = |rows: Vec<Vec<f32>>| {
            rows.into_iter()
                .flatten()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_bits_equal(
            &to_bits(batch_logits),
            &to_bits(serial_logits),
            "batched CUDA logits",
        );
        assert_bits_equal(
            &engine.qwen35_state_bits().unwrap(),
            &serial_state,
            "batched CUDA recurrent/KV state",
        );

        // Reject the final two proposals. Rewinding KV alone would leave both
        // recurrent state machines polluted; replay must match an ordinary prefix.
        engine.restore_qwen35_state(&initial).unwrap();
        target
            .consume_qwen35_cuda_draft(&mut engine, &suffix[..2], prompt.len(), scale, &|| false)
            .unwrap();
        let expected_prefix = engine.qwen35_state_bits().unwrap();
        engine.restore_qwen35_state(&initial).unwrap();
        engine
            .verify_qwen35_batch_logits(&embeddings, &cos, &sin, prompt.len(), suffix.len(), scale)
            .unwrap();
        engine.restore_qwen35_state(&initial).unwrap();
        target
            .consume_qwen35_cuda_draft(&mut engine, &suffix[..2], prompt.len(), scale, &|| false)
            .unwrap();
        assert_bits_equal(
            &engine.qwen35_state_bits().unwrap(),
            &expected_prefix,
            "rejected suffix CUDA state",
        );
    }

    /// Requires CAMELID_BONSAI_TARGET, CAMELID_SPEC_DRAFT_MODEL, and CUDA. No
    /// environment mutation occurs inside the test, so its hardware/configuration
    /// provenance remains visible in the invocation.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA and official target/draft models"]
    fn bonsai2_cuda_draft_greedy_sampled_stop_and_cancel_parity() {
        let target_path = std::env::var("CAMELID_BONSAI_TARGET").unwrap();
        let target = RunnableModel::load(&target_path).unwrap();
        let tokenizer =
            crate::tokenizer::Tokenizer::from_gguf(&read_metadata(&target_path).unwrap()).unwrap();
        let prompt = tokenizer.encode(
            "<|im_start|>user\nExplain in three sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n",
            false, true,
        ).unwrap();
        let mut streamed = Vec::new();
        let initially_cancelled = target.generate_qwen35_cuda_with_draft(
            &prompt,
            8,
            &[],
            None,
            &|| true,
            &|_| false,
            &mut |token| streamed.push(token),
        );
        assert!(matches!(initially_cancelled, Err(BackendError::Cancelled)));
        assert!(streamed.is_empty());
        assert!(target.cuda_draft.lock().unwrap().is_none());
        assert!(!target.cuda_language_active());
        assert!(target
            .generate_qwen35_cuda(&prompt, 0, &[], None, &|| false, &|_| false, &mut |token| {
                streamed.push(token)
            },)
            .unwrap()
            .is_empty());
        assert!(streamed.is_empty());
        assert!(!target.cuda_language_active());
        let max_pos = std::env::var("CAMELID_QWEN35_CUDA_MAXPOS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8192usize);
        for invalid_prompt in [&[][..], &vec![prompt[0]; max_pos][..]] {
            assert!(matches!(
                target.generate_qwen35_cuda(
                    invalid_prompt,
                    1,
                    &[],
                    None,
                    &|| false,
                    &|_| false,
                    &mut |token| streamed.push(token),
                ),
                Err(BackendError::InvalidTensorData(_))
            ));
        }
        assert!(streamed.is_empty());
        assert!(!target.cuda_language_active());
        let sampling = SamplingConfig {
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            seed: Some(42),
            repeat_penalty: 1.05,
            ..SamplingConfig::default()
        };
        for config in [None, Some(&sampling)] {
            let expected = target
                .generate_qwen35_cuda(&prompt, 32, &[], config, &|| false, &|_| false, &mut |_| {})
                .unwrap();
            let mut streamed = Vec::new();
            let actual = target
                .generate_qwen35_cuda_with_draft(
                    &prompt,
                    32,
                    &[],
                    config,
                    &|| false,
                    &|_| false,
                    &mut |token| streamed.push(token),
                )
                .unwrap();
            assert_eq!(actual, expected, "CUDA draft changed target sampling");
            assert_eq!(
                streamed, actual,
                "GUI stream exposed an unverified proposal"
            );
        }
        let expected = target
            .generate_qwen35_cuda(&prompt, 8, &[], None, &|| false, &|_| false, &mut |_| {})
            .unwrap();
        let mut streamed = Vec::new();
        let ordinary_stopped = target
            .generate_qwen35_cuda(
                &prompt,
                8,
                &[],
                None,
                &|| false,
                &|tokens| tokens.len() == 3,
                &mut |token| streamed.push(token),
            )
            .unwrap();
        assert_eq!(ordinary_stopped, expected[..3]);
        assert_eq!(streamed, ordinary_stopped);
        let cancel_requested = std::cell::Cell::new(false);
        let mut streamed = Vec::new();
        let ordinary_cancelled = target.generate_qwen35_cuda(
            &prompt,
            8,
            &[],
            None,
            &|| cancel_requested.get(),
            &|_| false,
            &mut |token| {
                streamed.push(token);
                cancel_requested.set(true);
            },
        );
        assert!(matches!(ordinary_cancelled, Err(BackendError::Cancelled)));
        assert_eq!(streamed, expected[..1]);
        let actual = target
            .generate_qwen35_cuda_with_draft(
                &prompt,
                8,
                &[],
                None,
                &|| false,
                &|tokens| tokens.len() == 3,
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(actual, expected[..3]);
        let stopped = target
            .generate_qwen35_cuda_with_draft(
                &prompt,
                8,
                &expected[..1],
                None,
                &|| false,
                &|_| false,
                &mut |_| {},
            )
            .unwrap();
        assert!(stopped.is_empty());
        let mut streamed = Vec::new();
        let cancelled = target.generate_qwen35_cuda_with_draft(
            &prompt,
            8,
            &[],
            None,
            &|| true,
            &|_| false,
            &mut |token| streamed.push(token),
        );
        assert!(matches!(cancelled, Err(BackendError::Cancelled)));
        assert!(streamed.is_empty());
    }
}
