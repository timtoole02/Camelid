//! Experimental dense Q8 CPU-prefix/GPU-suffix execution.
//!
//! `CAMELID_CUDA_CPU_PREFIX_LAYERS=N` explicitly selects the first N layers
//! for CPU computation. The remaining layers and output head must fit on the
//! GPU without weight streaming. CPU prefill and an eager mirror of each new
//! GPU KV row keep the ordinary session's history authoritative. This trades
//! one batched row readback for simple, safe cloning and rollback semantics.

use super::*;

const PREFIX_ENV: &str = "CAMELID_CUDA_CPU_PREFIX_LAYERS";

pub(super) struct State {
    prefix_layers: usize,
    weights_identity: usize,
    config: LlamaModelConfig,
    engine: crate::cuda_resident::CudaResidentDecode,
    poisoned: bool,
}

fn invalid(message: impl std::fmt::Display) -> BackendError {
    BackendError::RuntimeShapeMismatch(format!("experimental CUDA CPU-prefix mode: {message}"))
}

fn parse_prefix(value: Option<&str>, layer_count: usize) -> Result<Option<usize>> {
    let Some(value) = value else { return Ok(None) };
    let count = value
        .trim()
        .parse::<usize>()
        .map_err(|_| invalid(format!("{PREFIX_ENV} must be 0 or a positive layer count")))?;
    if count == 0 {
        return Ok(None);
    }
    if count >= layer_count {
        return Err(invalid(format!(
            "CPU prefix {count} must leave at least one GPU layer out of {layer_count}"
        )));
    }
    Ok(Some(count))
}

impl LlamaInferenceSession {
    pub(super) fn invalidate_cuda_cpu_prefix_kv(&mut self) -> Result<()> {
        if let Some(state) = self.cuda_cpu_prefix.as_mut() {
            let state = state
                .get_mut()
                .map_err(|_| invalid("GPU suffix state mutex was poisoned; start a new session"))?;
            // CPU work after a rollback can reach the old GPU cursor with a
            // different history. Cursor equality alone would then accept stale
            // rows. Retain weights and poison state, but require a full reseed.
            state.engine.set_filled(0);
        }
        Ok(())
    }

    pub(super) fn cuda_cpu_prefix_layers(&self) -> Result<Option<usize>> {
        let value = std::env::var(PREFIX_ENV)
            .map(Some)
            .or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(None),
                _ => Err(invalid(format!("{PREFIX_ENV} must be valid UTF-8"))),
            })?;
        let Some(prefix) = parse_prefix(value.as_deref(), self.weights.layers.len())? else {
            if self.cuda_cpu_prefix.is_some() {
                return Err(invalid(
                    "CPU/GPU split was disabled within an active session",
                ));
            }
            return Ok(None);
        };
        // Shared QKV helpers reload runtime flags instead of consuming the
        // per-layer plan. Reject this independent switch before either half
        // runs so the CPU prefix cannot silently upload projection weights.
        if crate::cuda::runtime_enabled() {
            return Err(invalid(
                "incompatible with per-projection CUDA Q8; disable CAMELID_CUDA_Q8 and its runtime toggle",
            ));
        }
        if !resident_decode_cuda_enabled()
            || self.resident_paths_disabled
            || self.is_drafter
            || spec_coexist_reserve_bytes() != 0
            || self.cuda_sequence_capacity() != 1
            || self.cuda_paged_kv_enabled()
            || self.weights.layer_range.is_some()
            || self.cuda_resident_pin.is_some()
        {
            return Err(invalid(
                "requires one unsharded CUDA sequence with no speculative, paged, or previously resident session",
            ));
        }
        if !matches!(
            self.config.architecture.as_str(),
            "llama" | "mistral" | "qwen3"
        ) || self.config.gemma3.is_some()
            || self.config.moe.is_some()
            || self.config.kv_quant != crate::model::KvCacheQuantization::F16
            || !matches!(self.kv_cache.dtype, KvDtype::F32 | KvDtype::F16)
        {
            return Err(invalid(
                "requires dense Llama/Mistral/Qwen3, F16 GPU KV, and F16 or F32 host KV",
            ));
        }
        let is_q8 = |tensor: &CpuTensor| tensor.source_type == Some(GgufTensorType::Q8_0);
        if !is_q8(self.weights.output_projection())
            || self.weights.layers.iter().any(|layer| {
                [
                    &layer.attention_q,
                    &layer.attention_k,
                    &layer.attention_v,
                    &layer.attention_output,
                    &layer.ffn_gate,
                    &layer.ffn_up,
                    &layer.ffn_down,
                ]
                .into_iter()
                .any(|tensor| !is_q8(tensor))
            })
            || !self.resident_decode_eligible(true)?
        {
            return Err(invalid(
                "requires supported Q8_0 projections in every layer and the output head",
            ));
        }
        Ok(Some(prefix))
    }

    pub(super) fn forward_cuda_cpu_prefix(
        &mut self,
        embedding: &CpuTensor,
        prefix_layers: usize,
        gpu_sample_token: Option<u32>,
        sample: Option<(f32, u64)>,
    ) -> Result<ResidentForward> {
        let position = self.kv_cache.position;
        if !self.kv_cache.history_materialized(position) {
            return Err(invalid(
                "CPU prefill/history is not authoritative; refusing to seed missing KV",
            ));
        }
        let dims = DenseLlamaDims::from_config(&self.config)?;
        let kv_cap = (self.config.context_length as usize)
            .min(self.kv_cache.plan.max_sequence_length)
            .min(self.resident_cuda_context_cap());
        if position >= kv_cap || embedding.data.len() != dims.embedding_length {
            return Err(invalid(
                "context capacity or hidden-state geometry is invalid",
            ));
        }
        let tables = rope::resident_decode_rope_tables(
            position,
            dims.head_dim,
            &self.config,
            self.weights.rope_freqs.as_ref(),
        )?
        .ok_or_else(|| invalid("RoPE tables are unavailable"))?;
        let rms_eps = diagnostic_rms_norm_epsilon(self.config.rms_norm_epsilon)?;
        let scale = attention_score_scale_value(dims.head_dim, diagnostic_attention_score_scale()?);
        let weights_identity = Arc::as_ptr(&self.weights) as usize;
        let suffix_layers = dims.block_count - prefix_layers;

        if self.cuda_cpu_prefix.is_none() {
            // A prior CLI iteration may have dropped its session-owned engine.
            // Return its unused async allocations before probing the next fit.
            crate::cuda::release_async_pool();
            let free_vram_bytes = crate::cuda::probe_capability()
                .map(|cap| cap.vram_free_bytes)
                .unwrap_or(0);
            let engine = build_resident_cuda_engine(
                &self.weights,
                prefix_layers..dims.block_count,
                suffix_layers,
                self.config.attention_head_count as usize,
                dims.attention_head_count_kv,
                dims.head_dim,
                dims.embedding_length,
                dims.feed_forward_length,
                self.config
                    .rope_dimension_count
                    .map(|v| v as usize)
                    .unwrap_or(dims.head_dim),
                kv_cap,
                dims.vocab_size,
                rms_eps,
                tables.split_half_pairing,
                false,
                None,
                1,
                self.config.kv_quant,
                false,
            )
            .ok_or_else(|| invalid("could not build the GPU suffix; increase CPU prefix layers"))?;
            if engine.is_offloaded() {
                drop(engine);
                crate::cuda::release_async_pool();
                return Err(invalid(
                    "GPU suffix would still stream weights; increase CPU prefix layers",
                ));
            }
            let status = crate::offload::OffloadRunStatus {
                total_layers: dims.block_count,
                layers_resident: suffix_layers,
                layers_offloaded: prefix_layers,
                per_layer_bytes: 0,
                free_vram_bytes,
                pcie_gbps: None,
                source: "cpu-prefix",
            };
            eprintln!("{}", status.describe());
            crate::offload::set_offload_run_status(Some(status));
            self.cuda_cpu_prefix = Some(Box::new(std::sync::Mutex::new(State {
                prefix_layers,
                weights_identity,
                config: self.config.clone(),
                engine,
                poisoned: false,
            })));
        }

        // Keep the engine outside self during the operation so the CPU layer
        // walk can borrow the ordinary KV cache. Restore it on both outcomes;
        // a failed partial forward remains poisoned and can never be retried.
        let mut owned_state = self.cuda_cpu_prefix.take().expect("constructed above");
        // Sessions may be shared immutably in cached prompt prefixes. The
        // engine is Send but not Sync; exclusive session access lets us borrow
        // through the mutex directly without locking or adding an unsafe Sync.
        let state = match owned_state.get_mut() {
            Ok(state) => state,
            Err(_) => {
                self.cuda_cpu_prefix = Some(owned_state);
                return Err(invalid(
                    "GPU suffix state mutex was poisoned; start a new session",
                ));
            }
        };
        let restore_watermark = self.kv_cache.materialized_through;
        let result = (|| {
            if state.poisoned {
                return Err(invalid(
                    "a previous partial forward failed; start a new session",
                ));
            }
            if state.prefix_layers != prefix_layers
                || state.weights_identity != weights_identity
                || state.config != self.config
            {
                return Err(invalid("model or CPU/GPU split changed within the session"));
            }
            if position >= state.engine.max_pos() {
                return Err(invalid("GPU suffix context capacity was exceeded"));
            }
            // Initial CPU prefill, a clone, a rollback, or an explicit CPU
            // diagnostic step may require a reseed. Mirrored F16 values make
            // the round trip exact for this deliberately narrow KV contract.
            if state.engine.filled() != position {
                for layer in 0..suffix_layers {
                    let mut keys =
                        vec![0.0; dims.attention_head_count_kv * position * dims.head_dim];
                    let mut values = vec![0.0; keys.len()];
                    for head in 0..dims.attention_head_count_kv {
                        for pos in 0..position {
                            let offset = (head * position + pos) * dims.head_dim;
                            self.kv_cache.copy_key_row_into(
                                prefix_layers + layer,
                                pos,
                                head,
                                &mut keys[offset..offset + dims.head_dim],
                            );
                            self.kv_cache.copy_value_row_into(
                                prefix_layers + layer,
                                pos,
                                head,
                                &mut values[offset..offset + dims.head_dim],
                            );
                        }
                    }
                    state
                        .engine
                        .seed_layer(layer, &keys, &values, position)
                        .map_err(invalid)?;
                }
                state.engine.set_filled(position);
            }
            self.kv_cache.ensure_position_capacity(position + 1)?;
            let mut plan = ResolvedRuntimePlan::from_env()?;
            // Keep plan-aware projections on CPU. Admission also rejects the
            // independent CUDA Q8 switch because shared QKV reloads its flags.
            plan.q8.cuda = false;
            let mut hidden = embedding.clone();
            for (layer_idx, layer) in self.weights.layers[..prefix_layers].iter().enumerate() {
                hidden = forward_layer_timed(
                    &hidden,
                    layer,
                    ForwardLayerParams {
                        config: &self.config,
                        rope_freqs: self.weights.rope_freqs.as_ref(),
                        rms_norm_epsilon: rms_eps,
                        layer_idx,
                        collect_diagnostics: false,
                        runtime_plan: &plan,
                    },
                    &mut self.kv_cache,
                )?
                .output;
            }
            let output = if let Some((inv_temp, seed)) = sample {
                ResidentForward::Sampled(
                    state
                        .engine
                        .forward_token_sample(
                            &hidden.data,
                            &tables.cos,
                            &tables.sin,
                            position,
                            scale,
                            inv_temp,
                            seed,
                        )
                        .map_err(invalid)?,
                )
            } else if gpu_sample_token.is_some() {
                ResidentForward::Sampled(
                    state
                        .engine
                        .forward_token(
                            &hidden.data,
                            &tables.cos,
                            &tables.sin,
                            position,
                            scale,
                            true,
                        )
                        .map_err(invalid)?
                        .ok_or_else(|| invalid("GPU suffix declined after the CPU prefix ran"))?,
                )
            } else {
                let logits = state
                    .engine
                    .forward_token_logits(&hidden.data, &tables.cos, &tables.sin, position, scale)
                    .map_err(invalid)?;
                ResidentForward::Logits(CpuTensor::from_f32(
                    "cpu_prefix_cuda_logits",
                    vec![1, dims.vocab_size],
                    logits,
                )?)
            };
            let (keys, values) = state
                .engine
                .read_kv_row_all_layers(position)
                .map_err(invalid)?;
            for layer in 0..suffix_layers {
                for head in 0..dims.attention_head_count_kv {
                    let offset = (layer * dims.attention_head_count_kv + head) * dims.head_dim;
                    self.kv_cache.store_kv_head_row(
                        prefix_layers + layer,
                        position,
                        head,
                        &keys[offset..offset + dims.head_dim],
                        &values[offset..offset + dims.head_dim],
                    );
                }
            }
            state.engine.set_filled(position + 1);
            Ok(output)
        })();
        if result.is_err() {
            state.poisoned = true;
            self.kv_cache.materialized_through = restore_watermark;
        }
        self.cuda_cpu_prefix = Some(owned_state);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_prefix_preserves_session_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlamaInferenceSession>();
    }

    #[test]
    fn cpu_prefix_requires_an_explicit_split_with_both_devices() {
        assert_eq!(parse_prefix(None, 32).unwrap(), None);
        assert_eq!(parse_prefix(Some("0"), 32).unwrap(), None);
        assert_eq!(parse_prefix(Some("16"), 32).unwrap(), Some(16));
        for value in ["32", "33", "-1", "auto", ""] {
            assert!(parse_prefix(Some(value), 32).is_err(), "{value}");
        }
    }
}

#[cfg(test)]
#[path = "cuda_cpu_prefix/tests.rs"]
mod integration_tests;
