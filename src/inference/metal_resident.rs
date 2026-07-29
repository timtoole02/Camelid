//! Metal GPU resident-decode engine usage, relocated out of inference.rs so the
//! shared inference path carries no `metal::` references. metal.rs provides
//! non-macOS stubs for these types/fns, so this compiles on every target (dead
//! off macOS, where ResidentDecodeState::new returns None). Verbatim relocation —
//! reduction order and behaviour are byte-identical.

use super::*;
use crate::metal;

pub(super) type ResidentDecodeState = metal::ResidentDecodeState;

#[derive(Clone, Copy)]
enum MetalSampleRequest {
    Greedy {
        input_token: u32,
    },
    Temperature {
        input_token: u32,
        inv_temperature: f32,
        base_seed: u64,
    },
}

impl MetalSampleRequest {
    fn input_token(self) -> u32 {
        match self {
            Self::Greedy { input_token } | Self::Temperature { input_token, .. } => input_token,
        }
    }

    fn mode(self) -> metal::SampleMode {
        match self {
            Self::Greedy { .. } => metal::SampleMode::Greedy,
            Self::Temperature {
                inv_temperature,
                base_seed,
                ..
            } => metal::SampleMode::Temperature {
                inv_temperature,
                base_seed,
            },
        }
    }
}

/// Maximum speculative-verify window (`[last_token, drafts...]`), mirroring the CUDA host's
/// `MAX_VERIFY_K`. `k = drafts.len() + 1 <= MAX_VERIFY_K`.
// Used only by the non-cuda Metal verify seam (verify_drafts_metal / verify_tree_metal), whose
// callers are `#[cfg(not(feature = "cuda"))]` — so on a cuda build (Windows default / Linux
// --all-features) this is genuinely unused; allow it rather than trip clippy `-D dead_code`.
#[allow(dead_code)]
pub(super) const MAX_VERIFY_K: usize = 8;

/// The resident stack's view of one weight's bytes: page-aligned wire pages when
/// the fast-load path attached them (the GPU wraps them in place), else the
/// materialized 36-byte CPU blocks.
pub(super) fn resident_weight_bytes(tensor: &CpuTensor) -> metal::ResidentWeightBytes<'_> {
    match tensor.q8_0_wire_pages.as_ref() {
        Some(pages) => metal::ResidentWeightBytes::WirePages(pages),
        None => metal::ResidentWeightBytes::Blocks36(q8_0_blocks_as_bytes(
            tensor.q8_0_blocks.as_ref().unwrap(),
        )),
    }
}

impl super::LlamaInferenceSession {
    pub(super) fn try_metal_resident_prefill(&mut self, token_ids: &[u32]) -> Result<bool> {
        if std::env::var("CAMELID_METAL_RESIDENT_PREFILL")
            .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
            || token_ids.len() < 2
            || token_ids.len() > 16384
            || self.kv_cache.position != 0
            || self.weights.layer_range.is_some()
            || !self.resident_decode_eligible(false)?
        {
            return Ok(false);
        }
        let weights = Arc::clone(&self.weights);
        let dims = DenseLlamaDims::from_config(&self.config)?;
        let n_layers = dims.block_count;
        let n_heads = self.config.attention_head_count as usize;
        let n_kv = dims.attention_head_count_kv;
        let head_dim = dims.head_dim;
        let kv_cap = self.config.context_length as usize;
        let n = token_ids.len();
        if n >= kv_cap {
            return Ok(false);
        }
        let rms_eps = diagnostic_rms_norm_epsilon(self.config.rms_norm_epsilon)?;
        let scale = attention_score_scale_value(head_dim, diagnostic_attention_score_scale()?);
        // CAMELID_PREFILL_TIME=1: report the CPU-side edges around the GPU command buffer.
        let time_edges = std::env::var_os("CAMELID_PREFILL_TIME").is_some();
        let edge_started = Instant::now();

        // Rope tables for every prefill position, flattened.
        let tables = match rope::resident_prefill_rope_tables(
            n,
            head_dim,
            &self.config,
            weights.rope_freqs.as_ref(),
        )? {
            Some(t) => t,
            None => return Ok(false),
        };
        let (cos_all, sin_all, split_half_pairing) =
            (tables.cos, tables.sin, tables.split_half_pairing);

        let rope_us = edge_started.elapsed().as_micros();
        let session_started = Instant::now();
        let initial_positions = (n + 1).next_multiple_of(512).min(kv_cap);
        let mut session = match metal::ResidentDecodeState::new(
            n_layers,
            n_heads,
            n_kv,
            head_dim,
            dims.embedding_length,
            dims.feed_forward_length,
            initial_positions,
            kv_cap,
            rms_eps,
            split_half_pairing,
        ) {
            Some(s) => s,
            None => return Ok(false),
        };

        let session_us = session_started.elapsed().as_micros();
        let embed_started = Instant::now();
        let embeddings = self
            .weights
            .token_embedding
            .embedding_lookup(token_ids, "token_embedding_resident_prefill")?;
        let layer_views: Vec<metal::ResidentLayerWeights> = weights
            .layers
            .iter()
            .map(|l| metal::ResidentLayerWeights {
                attn_norm: &l.attention_norm.data,
                ffn_norm: &l.ffn_norm.data,
                q_norm: l.attention_q_norm.as_ref().map(|t| t.data.as_slice()),
                k_norm: l.attention_k_norm.as_ref().map(|t| t.data.as_slice()),
                q_weight_blocks: resident_weight_bytes(&l.attention_q),
                k_weight_blocks: resident_weight_bytes(&l.attention_k),
                v_weight_blocks: resident_weight_bytes(&l.attention_v),
                o_weight_blocks: resident_weight_bytes(&l.attention_output),
                gate_weight_blocks: resident_weight_bytes(&l.ffn_gate),
                up_weight_blocks: resident_weight_bytes(&l.ffn_up),
                down_weight_blocks: resident_weight_bytes(&l.ffn_down),
            })
            .collect();

        let embed_us = embed_started.elapsed().as_micros();
        let gpu_started = Instant::now();
        if session
            .prefill_tokens(&embeddings.data, n, &layer_views, &cos_all, &sin_all, scale)
            .is_none()
        {
            return Ok(false);
        }
        if time_edges {
            eprintln!(
                "[prefill-time] rope {:.1}ms | session {:.1}ms | embed+views {:.1}ms | prefill_tokens {:.1}ms | total {:.1}ms",
                rope_us as f64 / 1000.0,
                session_us as f64 / 1000.0,
                embed_us as f64 / 1000.0,
                gpu_started.elapsed().as_micros() as f64 / 1000.0,
                edge_started.elapsed().as_micros() as f64 / 1000.0,
            );
        }
        // GPU cache now holds positions 0..n; the resident decode continues this sequence.
        self.kv_cache.position = n;
        self.resident_decode = Some(session);
        Ok(true)
    }

    /// Make the CPU KV cache hold this sequence's real history before a CPU forward reads it.
    ///
    /// The GPU-resident lanes advance `kv_cache.position` while writing K/V only into the GPU
    /// cache (`try_metal_resident_prefill` sets `position = n` outright; each resident decode
    /// step appends on the GPU). The CPU buffers stay empty. That is fine for as long as the
    /// sequence stays resident — but a CPU forward reads `kv_cache.keys` over the whole
    /// `[0, position]` range, so the first step that falls back attends over a zeroed prompt,
    /// and its `ensure_position_capacity` call then makes the zeros look addressable to every
    /// later reseed. Silently wrong output for the rest of the generation.
    ///
    /// So: mirror the resident engine's KV back into the CPU cache — lazily, on the fallback
    /// that needs it, which is the same thing CUDA does eagerly after its prefill
    /// (`copy_resident_cuda_kv_to_host`) but at a price the decode path can afford. Metal
    /// buffers are shared storage, so recovery is a strided memcpy over unified memory; the
    /// CUDA half pays a device→host copy, and both pay it at most once per fallback rather than
    /// once per token. Writes go through `store_kv_head_row`, which rounds through f16 exactly
    /// as every other CPU write does and advances the watermark.
    ///
    /// No-op when the CPU history is already materialized — the common case, covering every
    /// pure-CPU run and the CUDA prefill (which mirrors eagerly).
    ///
    /// BACKENDS. Both GPU-resident lanes are asked, in the order they can be trusted. The Metal
    /// engine hangs off the session, so it is unambiguously this sequence's. The CUDA engine
    /// lives in a process-global cache, so its recovery has to establish identity first — see
    /// `recover_cpu_kv_from_cuda_resident`. When neither can supply the gap the history is
    /// genuinely lost (the engine was evicted or rebuilt for another model); that is not
    /// recoverable here, so it warns rather than pretending, and the CPU forward proceeds over
    /// whatever prefix it has.
    ///
    /// NEVER RETURNS Err FOR A FAILED RECOVERY, and never leaves the watermark vouching for a
    /// half-done one — see the two comments inside. Both rules exist because this sits on the
    /// ordinary CPU forward path, where the alternatives are worse than a warning.
    pub(super) fn ensure_cpu_kv_materialized(&mut self) -> Result<()> {
        let position = self.kv_cache.position;
        if position == 0 || self.kv_cache.materialized_through >= position {
            return Ok(());
        }
        // The watermark advances on the FIRST row a recovery writes, so a recovery that dies
        // part way through (a device readback error on layer 12 of 16) would leave it claiming
        // a history that is still zero for the layers never reached — every later
        // `history_materialized` / `cpu_kv_authoritative` check would then pass over exactly
        // the hollow prefix this function exists to prevent, and the next GPU reseed would
        // launder it. Strictly worse than not trying. So on any failure the watermark goes
        // back to where it was: the rows already written stay (they are real K/V, not damage),
        // they simply stop being vouched for.
        let restore = self.kv_cache.materialized_through;
        let attempt = match self.recover_cpu_kv_from_metal_resident(position) {
            Ok(true) => Ok(true),
            Ok(false) => self.recover_cpu_kv_from_cuda_resident(position),
            Err(e) => Err(e),
        };
        let recovered = match attempt {
            Ok(recovered) => recovered,
            // A readback failure must not abort the caller's forward. This is called from the
            // ordinary CPU path, so propagating would turn a recoverable degradation into a
            // failed request; the CUDA prefill lane already treats the identical
            // `copy_resident_cuda_kv_to_host` failure as `Ok(false)` + a trace line. Report it
            // and fall through to the warning, which says what the consequence actually is.
            Err(e) => {
                self.kv_cache.materialized_through = restore;
                static READBACK_WARNED: std::sync::Once = std::sync::Once::new();
                READBACK_WARNED.call_once(|| {
                    eprintln!("[resident-kv] WARNING: GPU KV readback failed: {e}");
                });
                false
            }
        };
        if recovered {
            return Ok(());
        }
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "[resident-kv] WARNING: the CPU KV history is materialized only through {} but \
                 the sequence is at position {position}, and no GPU-resident engine holds the \
                 gap — this CPU forward attends over a zero-filled prefix. (See \
                 `ensure_cpu_kv_materialized`.)",
                self.kv_cache.materialized_through
            );
        });
        Ok(())
    }

    /// Recover `[0, position)` from the SESSION-resident Metal engine. `Ok(false)` when this
    /// session has no Metal engine, or it does not hold the range (so the caller tries the next
    /// backend); `Ok(true)` when the CPU history is materialized on return.
    fn recover_cpu_kv_from_metal_resident(&mut self, position: usize) -> Result<bool> {
        // The engine must still hold the history we are missing.
        if self
            .resident_decode
            .as_ref()
            .is_none_or(|s| s.filled() < position)
        {
            return Ok(false);
        }
        let dims = DenseLlamaDims::from_config(&self.config)?;
        let range = self
            .weights
            .layer_range
            .clone()
            .unwrap_or(0..dims.block_count);

        // Read each owned layer out of the GPU cache and store it at its ABSOLUTE layer id
        // (the resident session is built over the owned subrange, so its slots are relative).
        for (slot, layer_idx) in range.clone().enumerate() {
            let session = self
                .resident_decode
                .as_ref()
                .expect("resident session present (checked above)");
            let (keys, values) = session.read_kv_layer(slot, position).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(format!(
                    "resident KV readback failed for layer {layer_idx} at {position} positions"
                ))
            })?;
            self.kv_cache
                .store_mirrored_layer_kv(layer_idx, position, &keys, &values)?;
        }
        if std::env::var_os("CAMELID_RESIDENT_TRACE").is_some() {
            eprintln!(
                "[resident-kv-mirror] recovered {position} positions x {} layers from the Metal \
                 resident cache into the CPU KV history",
                range.len()
            );
        }
        Ok(true)
    }

    pub(super) fn try_resident_decode_forward_metal(
        &mut self,
        embedding: &CpuTensor,
        compute_logits: bool,
        gpu_sample_token: Option<u32>,
    ) -> Result<Option<ResidentForward>> {
        self.try_resident_decode_forward_metal_inner(
            embedding,
            compute_logits,
            gpu_sample_token.map(|input_token| MetalSampleRequest::Greedy { input_token }),
        )
    }

    pub(super) fn try_resident_decode_forward_metal_sample(
        &mut self,
        embedding: &CpuTensor,
        input_token: u32,
        inv_temperature: f32,
        base_seed: u64,
    ) -> Result<Option<ResidentForward>> {
        self.try_resident_decode_forward_metal_inner(
            embedding,
            true,
            Some(MetalSampleRequest::Temperature {
                input_token,
                inv_temperature,
                base_seed,
            }),
        )
    }

    fn try_resident_decode_forward_metal_inner(
        &mut self,
        embedding: &CpuTensor,
        compute_logits: bool,
        gpu_sample: Option<MetalSampleRequest>,
    ) -> Result<Option<ResidentForward>> {
        if !self.resident_decode_eligible(compute_logits)? {
            return Ok(None);
        }
        let weights = Arc::clone(&self.weights);
        let dims = DenseLlamaDims::from_config(&self.config)?;
        let n_heads = self.config.attention_head_count as usize;
        let n_kv = dims.attention_head_count_kv;
        let head_dim = dims.head_dim;
        let hidden = dims.embedding_length;
        let ffn_dim = dims.feed_forward_length;
        // Pipeline-sharded nodes run only their owned layer range; the resident session is
        // built over that subset (relative slots) while KV seeding uses absolute layer ids.
        let range = weights.layer_range.clone().unwrap_or(0..dims.block_count);
        let n_layers = range.len();
        let vocab = dims.vocab_size;
        // The on-GPU KV cache grows on demand up to `kv_cap` (the model context length); sizing
        // it to the full (often 128K) context up front would allocate tens of GB and thrash
        // unified memory. Start sized to the current need plus a chunk and let the session grow.
        let kv_cap = self.config.context_length as usize;
        let position = self.kv_cache.position;
        let initial_positions = ((position + 1).max(512)).next_multiple_of(512).min(kv_cap);
        if position >= kv_cap
            || embedding.data.len() != hidden
            || weights.layers.len() != dims.block_count
            || range.end > weights.layers.len()
        {
            return Ok(None);
        }
        let rms_eps = diagnostic_rms_norm_epsilon(self.config.rms_norm_epsilon)?;
        let tables = match rope::resident_decode_rope_tables(
            position,
            head_dim,
            &self.config,
            weights.rope_freqs.as_ref(),
        )? {
            Some(t) => t,
            None => return Ok(None),
        };
        let scale = attention_score_scale_value(head_dim, diagnostic_attention_score_scale()?);

        // (Re)build + seed the session when starting a sequence (or resuming at a position the
        // session has not materialized): copy the CPU KV history [0, position) into the GPU
        // cache so resident decode can take over after the batched CPU prefill.
        let rebuild = match &self.resident_decode {
            Some(s) => s.filled() != position,
            None => true,
        };
        if rebuild {
            let mut session = match metal::ResidentDecodeState::new(
                n_layers,
                n_heads,
                n_kv,
                head_dim,
                hidden,
                ffn_dim,
                initial_positions,
                kv_cap,
                rms_eps,
                tables.split_half_pairing,
            ) {
                Some(s) => s,
                None => return Ok(None),
            };
            if position > 0 {
                // Seeding reads the CPU KV history [0, position) through the
                // dtype-neutral row accessors, so that range must actually be written.
                //
                // Capacity alone is not enough: one CPU fallback can grow buffers for its own
                // position while earlier GPU-produced positions remain zero-filled. Seeding
                // from that hollow history would silently erase the prompt context.
                //
                // The materialized-through watermark distinguishes the two. Reaching this
                // point with a hollow history should now be impossible —
                // `ensure_cpu_kv_materialized` mirrors the GPU history back before
                // any CPU fallback runs — so this is the backstop, not the fix; declining is
                // lossless and the caller takes the CPU path.
                if !self.kv_cache.history_materialized(position) {
                    return Ok(None);
                }
                let kv_dim = n_kv * head_dim;
                for layer in 0..n_layers {
                    let mut ck = vec![0.0f32; kv_dim * position];
                    let mut cv = vec![0.0f32; kv_dim * position];
                    for p in 0..position {
                        for h in 0..n_kv {
                            let dst = (h * position + p) * head_dim;
                            self.kv_cache.copy_key_row_into(
                                range.start + layer,
                                p,
                                h,
                                &mut ck[dst..dst + head_dim],
                            );
                            self.kv_cache.copy_value_row_into(
                                range.start + layer,
                                p,
                                h,
                                &mut cv[dst..dst + head_dim],
                            );
                        }
                    }
                    if !session.seed_layer(layer, &ck, &cv, position) {
                        return Ok(None);
                    }
                }
            }
            session.set_filled(position);
            self.resident_decode = Some(session);
        }

        let layer_views: Vec<metal::ResidentLayerWeights> = weights.layers[range.clone()]
            .iter()
            .map(|l| metal::ResidentLayerWeights {
                attn_norm: &l.attention_norm.data,
                ffn_norm: &l.ffn_norm.data,
                q_norm: l.attention_q_norm.as_ref().map(|t| t.data.as_slice()),
                k_norm: l.attention_k_norm.as_ref().map(|t| t.data.as_slice()),
                q_weight_blocks: resident_weight_bytes(&l.attention_q),
                k_weight_blocks: resident_weight_bytes(&l.attention_k),
                v_weight_blocks: resident_weight_bytes(&l.attention_v),
                o_weight_blocks: resident_weight_bytes(&l.attention_output),
                gate_weight_blocks: resident_weight_bytes(&l.ffn_gate),
                up_weight_blocks: resident_weight_bytes(&l.ffn_up),
                down_weight_blocks: resident_weight_bytes(&l.ffn_down),
            })
            .collect();

        // When logits are wanted, run the final RMSNorm + output projection on the GPU too
        // (in the same command buffer) so the large vocab matmul stays off the CPU.
        let logits_stage = if compute_logits {
            Some(metal::LogitsStage {
                final_norm: &weights.output_norm.data,
                output_weight_blocks: resident_weight_bytes(weights.output_projection()),
                vocab_size: vocab,
            })
        } else {
            None
        };

        // GPU-side greedy sampling stage: only when the caller asked for it, logits run on
        // the GPU, and the token embedding table is plain Q8_0 (the gather reads its rows).
        let sample_stage = match gpu_sample {
            Some(_)
                if compute_logits
                    && weights.token_embedding.source_type == Some(GgufTensorType::Q8_0)
                    && (weights.token_embedding.q8_0_blocks.is_some()
                        || weights.token_embedding.q8_0_wire_pages.is_some()) =>
            {
                let embedding_blocks = resident_weight_bytes(&weights.token_embedding);
                (embedding_blocks.block_count() == vocab * (hidden / 32)).then_some(
                    metal::SampleStage {
                        embedding_blocks,
                        mode: gpu_sample.expect("sample request matched above").mode(),
                    },
                )
            }
            _ => None,
        };
        // Temperature sampling must fail before advancing resident KV when the
        // device-side tail cannot be built. Returning logits here would already
        // have executed the position and make a caller fallback run it twice.
        if matches!(gpu_sample, Some(MetalSampleRequest::Temperature { .. }))
            && sample_stage.is_none()
        {
            return Ok(None);
        }

        // Rope tables for position+1 feed the encode-ahead pipeline: the session encodes
        // the NEXT token's command buffer while this token executes on the GPU.
        let next_tables = rope::resident_decode_rope_tables(
            position + 1,
            head_dim,
            &self.config,
            weights.rope_freqs.as_ref(),
        )?;
        let session = self
            .resident_decode
            .as_mut()
            .expect("resident session built above");
        let out = match session.forward_token(
            &embedding.data,
            &layer_views,
            &tables.cos,
            &tables.sin,
            position,
            scale,
            logits_stage,
            sample_stage,
            gpu_sample
                .map(MetalSampleRequest::input_token)
                .unwrap_or(u32::MAX),
            next_tables
                .as_ref()
                .map(|t| (t.cos.as_slice(), t.sin.as_slice())),
        ) {
            Some(o) => o,
            None => return Ok(None),
        };
        match out {
            metal::ResidentTokenOut::Sampled(id) => Ok(Some(ResidentForward::Sampled(id))),
            metal::ResidentTokenOut::Data(out) if compute_logits => {
                Ok(Some(ResidentForward::Logits(CpuTensor::from_f32(
                    "resident_logits",
                    vec![1, vocab],
                    out,
                )?)))
            }
            metal::ResidentTokenOut::Data(out) => Ok(Some(ResidentForward::Hidden(
                CpuTensor::from_f32("resident_hidden", vec![1, hidden], out)?,
            ))),
        }
    }

    /// macOS speculative-verify seam: verify a batch of draft tokens against the resident
    /// Metal engine in ONE batched forward (`metal::ResidentDecodeState::verify_batch`,
    /// bit-identical to `k` single-token decodes) and return the accepted prefix (the longest
    /// run the model confirms plus the bonus token at the first mismatch). Mirrors the CUDA
    /// `verify_drafts_gpu` host orchestration over `self.resident_decode`. Returns `Ok(None)`
    /// (caller takes a normal step / CPU chunk-verify) whenever the engine isn't ready exactly
    /// at the current KV position or the config is unsupported — lossless either way, since the
    /// target verify is authoritative and `accepted` is exactly what greedy decode would emit.
    #[cfg(target_os = "macos")]
    pub(super) fn verify_drafts_metal(
        &mut self,
        last_token: u32,
        drafts: &[u32],
    ) -> Result<Option<Vec<u32>>> {
        if drafts.is_empty() || self.resident_paths_disabled || !resident_decode_metal_enabled() {
            return Ok(None);
        }
        let position = self.kv_cache.position;
        let k = drafts.len() + 1;
        if k > MAX_VERIFY_K
            || position + k > self.kv_cache.plan.max_sequence_length
            || !self.resident_decode_eligible(true)?
        {
            return Ok(None);
        }
        // The engine must already hold this sequence with KV materialized exactly to `position`
        // (mid-decode). Otherwise route the caller to its lossless CPU fallback, which seeds /
        // rebuilds the engine on a normal step.
        if self
            .resident_decode
            .as_ref()
            .is_none_or(|s| s.filled() != position)
        {
            return Ok(None);
        }

        let weights = Arc::clone(&self.weights);
        let dims = DenseLlamaDims::from_config(&self.config)?;
        let head_dim = dims.head_dim;
        let vocab = dims.vocab_size;
        // `verify_batch` runs the whole decode stack + logits; a pipeline-sharded node owns only
        // a layer subrange (no logits stage), so it falls back to the CPU verify.
        if weights.layer_range.is_some() {
            return Ok(None);
        }
        let scale = attention_score_scale_value(head_dim, diagnostic_attention_score_scale()?);

        // Inputs `[last_token, drafts...]` land at positions `[position, position+k)`.
        let mut inputs = Vec::with_capacity(k);
        inputs.push(last_token);
        inputs.extend_from_slice(drafts);
        let embeddings = self
            .weights
            .token_embedding
            .embedding_lookup(&inputs, "token_embedding_spec_verify")?;

        // Per-position RoPE tables (position `base+i`), flattened position-major.
        let mut cos_all = Vec::with_capacity(k * head_dim);
        let mut sin_all = Vec::with_capacity(k * head_dim);
        for i in 0..k {
            match rope::resident_decode_rope_tables(
                position + i,
                head_dim,
                &self.config,
                weights.rope_freqs.as_ref(),
            )? {
                Some(t) => {
                    cos_all.extend_from_slice(&t.cos);
                    sin_all.extend_from_slice(&t.sin);
                }
                _ => return Ok(None),
            }
        }

        let layer_views: Vec<metal::ResidentLayerWeights> = weights
            .layers
            .iter()
            .map(|l| metal::ResidentLayerWeights {
                attn_norm: &l.attention_norm.data,
                ffn_norm: &l.ffn_norm.data,
                q_norm: l.attention_q_norm.as_ref().map(|t| t.data.as_slice()),
                k_norm: l.attention_k_norm.as_ref().map(|t| t.data.as_slice()),
                q_weight_blocks: resident_weight_bytes(&l.attention_q),
                k_weight_blocks: resident_weight_bytes(&l.attention_k),
                v_weight_blocks: resident_weight_bytes(&l.attention_v),
                o_weight_blocks: resident_weight_bytes(&l.attention_output),
                gate_weight_blocks: resident_weight_bytes(&l.ffn_gate),
                up_weight_blocks: resident_weight_bytes(&l.ffn_up),
                down_weight_blocks: resident_weight_bytes(&l.ffn_down),
            })
            .collect();
        let logits_stage = metal::LogitsStage {
            final_norm: &weights.output_norm.data,
            output_weight_blocks: resident_weight_bytes(weights.output_projection()),
            vocab_size: vocab,
        };

        let session = self
            .resident_decode
            .as_mut()
            .expect("resident session present (readiness checked above)");
        let predicted = match session.verify_batch(
            &embeddings.data,
            &cos_all,
            &sin_all,
            &layer_views,
            &logits_stage,
            position,
            k,
            scale,
        ) {
            Some(p) => p,
            None => return Ok(None),
        };

        // Accept the longest prefix of drafts the model confirms, plus the bonus token at the
        // first mismatch (`predicted[0]` is always taken). Identical accept rule to the CUDA arm.
        let acc = crate::inference::speculative::accepted_draft_prefix(
            drafts,
            &predicted[..drafts.len()],
        );
        let emitted = predicted[..=acc].to_vec();
        let new_position = position + emitted.len();
        session.set_filled(new_position);
        self.kv_cache.position = new_position;
        if std::env::var_os("CAMELID_SPEC_VERIFY_TRACE").is_some() {
            eprintln!(
                "[metal-spec-verify] base={position} k={k} accepted={acc} emitted_len={}",
                emitted.len()
            );
        }
        Ok(Some(emitted))
    }

    /// macOS speculative-verify seam (TREE variant): verify a draft TOKEN TREE against the
    /// resident Metal engine in ONE batched forward (`metal::ResidentDecodeState::verify_batch_tree`,
    /// bit-identical to `verify_batch` on a single-branch tree) and return the accepted longest
    /// path — every emitted token is the target's own greedy argmax along that path
    /// (`accept_longest_path`). Mirrors the CUDA `verify_tree_gpu` host orchestration over
    /// `self.resident_decode`. Returns `Ok(None)` (caller takes a normal step) whenever the engine
    /// isn't ready exactly at the current KV position or the config is unsupported — lossless
    /// either way, since the target verify is authoritative.
    #[cfg(target_os = "macos")]
    pub(super) fn verify_tree_metal(
        &mut self,
        tree: &spec_tree::TokenTree,
    ) -> Result<Option<Vec<u32>>> {
        use spec_tree::TREE_MAX_NODES;
        if self.resident_paths_disabled || !resident_decode_metal_enabled() {
            return Ok(None);
        }
        let n = tree.nodes();
        if n == 0 {
            return Ok(None);
        }
        let position = self.kv_cache.position;
        // Each node lands at slot base+BFS-idx; the committed path is at most `n` tokens.
        // Bound by the cache and the node cap (mirrors the cuda host).
        if n > TREE_MAX_NODES
            || position + n > self.kv_cache.plan.max_sequence_length
            || !self.resident_decode_eligible(true)?
        {
            return Ok(None);
        }
        // The engine must already hold this sequence with KV materialized exactly to `position`
        // (mid-decode). Otherwise route the caller to its lossless fallback / normal step.
        if self
            .resident_decode
            .as_ref()
            .is_none_or(|s| s.filled() != position)
        {
            return Ok(None);
        }

        let weights = Arc::clone(&self.weights);
        let dims = DenseLlamaDims::from_config(&self.config)?;
        let head_dim = dims.head_dim;
        let vocab = dims.vocab_size;
        // `verify_batch_tree` runs the whole decode stack + logits; a pipeline-sharded node owns
        // only a layer subrange (no logits stage), so it falls back to a normal step.
        if weights.layer_range.is_some() {
            return Ok(None);
        }
        let scale = attention_score_scale_value(head_dim, diagnostic_attention_score_scale()?);

        // Embeddings in BFS (node) order: node 0 is the anchor, nodes 1.. the drafts.
        let embeddings = self
            .weights
            .token_embedding
            .embedding_lookup(&tree.tokens, "token_embedding_tree_verify")?;

        // Per-node RoPE tables at position `base + node_depth[i]` (flattened node-major).
        let node_depth = tree.node_depth();
        let mut cos_all = Vec::with_capacity(n * head_dim);
        let mut sin_all = Vec::with_capacity(n * head_dim);
        for &d in &node_depth {
            match rope::resident_decode_rope_tables(
                position + d as usize,
                head_dim,
                &self.config,
                weights.rope_freqs.as_ref(),
            )? {
                Some(t) => {
                    cos_all.extend_from_slice(&t.cos);
                    sin_all.extend_from_slice(&t.sin);
                }
                _ => return Ok(None),
            }
        }
        let node_kvslot = tree.node_kvslot(position);
        let (ancestor_bits, words) = tree.ancestor_bitset();

        let layer_views: Vec<metal::ResidentLayerWeights> = weights
            .layers
            .iter()
            .map(|l| metal::ResidentLayerWeights {
                attn_norm: &l.attention_norm.data,
                ffn_norm: &l.ffn_norm.data,
                q_norm: l.attention_q_norm.as_ref().map(|t| t.data.as_slice()),
                k_norm: l.attention_k_norm.as_ref().map(|t| t.data.as_slice()),
                q_weight_blocks: resident_weight_bytes(&l.attention_q),
                k_weight_blocks: resident_weight_bytes(&l.attention_k),
                v_weight_blocks: resident_weight_bytes(&l.attention_v),
                o_weight_blocks: resident_weight_bytes(&l.attention_output),
                gate_weight_blocks: resident_weight_bytes(&l.ffn_gate),
                up_weight_blocks: resident_weight_bytes(&l.ffn_up),
                down_weight_blocks: resident_weight_bytes(&l.ffn_down),
            })
            .collect();
        let logits_stage = metal::LogitsStage {
            final_norm: &weights.output_norm.data,
            output_weight_blocks: resident_weight_bytes(weights.output_projection()),
            vocab_size: vocab,
        };

        let session = self
            .resident_decode
            .as_mut()
            .expect("resident session present (readiness checked above)");
        let predicted = match session.verify_batch_tree(
            &embeddings.data,
            &cos_all,
            &sin_all,
            &layer_views,
            &logits_stage,
            &node_kvslot,
            &ancestor_bits,
            words,
            position,
            n,
            scale,
        ) {
            Some(p) => p,
            None => return Ok(None),
        };

        // Host accept: longest greedy-exact path through the tree, then COMPACT the accepted
        // path's KV into contiguous slots base..base+L-1 so the cache matches a linear decode of
        // that path (no-op for a single-branch tree). Identical accept rule to the CUDA arm.
        let (emitted, leaf) = tree.accept_longest_path(&predicted);
        let path = tree.path_to(leaf); // includes the anchor (node 0); root first
        session.compact_tree_kv_path(&path, position).map_err(|e| {
            BackendError::RuntimeShapeMismatch(format!("tree KV compaction failed: {e}"))
        })?;
        let new_position = position + emitted.len();
        session.set_filled(new_position);
        self.kv_cache.position = new_position;
        if std::env::var_os("CAMELID_SPEC_VERIFY_TRACE").is_some() {
            // Max fan-out = the most children any node has (1 == single-branch / linear).
            let mut child_count = vec![0u32; n];
            for i in 1..n {
                let p = tree.parent[i];
                if p >= 0 {
                    child_count[p as usize] += 1;
                }
            }
            let max_fanout = child_count.iter().copied().max().unwrap_or(0);
            eprintln!(
                "[metal-tree-verify] base={position} n={n} emitted_len={} max_fanout={max_fanout}",
                emitted.len()
            );
        }
        Ok(Some(emitted))
    }

    /// Non-macOS build: the Metal resident speculative-verify path is unavailable, so return
    /// `Ok(None)` and let the caller fall back to the CPU chunk verify (lossless either way).
    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)] // unused on cuda builds: the caller is #[cfg(not(feature = "cuda"))]
    pub(super) fn verify_drafts_metal(
        &mut self,
        _last_token: u32,
        _drafts: &[u32],
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Non-macOS build: the Metal resident tree-verify path is unavailable — return `Ok(None)`
    /// so the caller takes a normal step (lossless either way).
    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)] // unused on cuda builds: the caller is #[cfg(not(feature = "cuda"))]
    pub(super) fn verify_tree_metal(
        &mut self,
        _tree: &spec_tree::TokenTree,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
}
