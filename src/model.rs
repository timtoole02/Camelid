use serde::Serialize;

pub use crate::tensor::kv_quant::KvCacheQuantization;

use crate::{
    gguf::{GgufFile, GgufTensorDescriptor},
    BackendError, Result,
};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaModelConfig {
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub kv_quant: KvCacheQuantization,
    pub rope_dimension_count: Option<u32>,
    pub rope_freq_base: Option<f32>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context_length: Option<u32>,
    pub rope_scaling_low_freq_factor: Option<f32>,
    pub rope_scaling_high_freq_factor: Option<f32>,
    pub rms_norm_epsilon: f32,
    pub vocab_size: Option<u32>,
    pub file_type: Option<u32>,
    /// Explicit per-head dimension from `<arch>.attention.key_length`, when the
    /// GGUF sets it. Qwen3 sizes where `head_dim != embedding_length/head_count`
    /// (0.6B/4B/32B) rely on this; `None` falls back to `embedding/head_count`
    /// (llama/mistral, and the Qwen3 sizes where they happen to be equal). The
    /// Llama dense path reads this in [`DenseLlamaDims`]; gemma4 keeps its own
    /// per-layer head-dim handling in [`Gemma4Metadata`].
    pub attention_key_length: Option<u32>,
    /// RoPE uses NEOX "split-half" pairing (dim `d` rotated with `d + rope_dim/2`)
    /// rather than the default "adjacent even/odd" pairing. llama.cpp permutes the
    /// Q/K projection weights for LLaMA-family conversions so that the adjacent
    /// pairing reproduces rotate-half; Qwen GGUFs are NOT permuted, so they must
    /// be roped with split-half to match the reference. `true` for qwen3 (verified
    /// against llama.cpp); `false` for llama/mistral/etc. The env override
    /// `CAMELID_ROPE_PAIRING` still takes precedence for diagnostics.
    pub rope_neox_pairing: bool,
    /// Logit scale — applied before softmax, commonly in Command R models.
    pub logit_scale: Option<f32>,
    pub moe: Option<MixtralMoeMetadata>,
    /// Gemma 4 (`general.architecture = "gemma4"`) specific metadata. `None` for
    /// every other architecture. Holds the per-layer-type attention dims, dual
    /// RoPE bases, sliding-window pattern, KV-sharing depth, Per-Layer-Embedding
    /// width, and final logit soft-cap that a Llama-shaped config cannot express.
    pub gemma4: Option<Gemma4Metadata>,
    /// Qwen3.5 (`general.architecture = "qwen35"`) hybrid linear-attention metadata.
    /// `None` for every other architecture. Holds the gated-delta-net (SSM) dims and
    /// the per-layer recurrent/full-attention schedule that a dense Llama config
    /// cannot express. See [`Qwen35Metadata`].
    pub qwen35: Option<Qwen35Metadata>,
    /// Multi-Head Latent Attention (MLA) metadata for DeepSeek models.
    pub mla: Option<MlaMetadata>,
}

/// Whether `architecture` is one of the dense-decoder families Camelid actually
/// implements. This MUST mirror the accepted set in [`LlamaModelConfig::from_gguf`]
/// below (the `unit test implemented_set_matches_from_gguf_accept_arm` guards the
/// two against drift). It is a pure architecture-string check for classification
/// (e.g. labeling a loaded model's lane); it makes NO support/parity claim — an
/// implemented architecture is only *attemptable*, never automatically supported.
pub fn is_implemented_architecture(architecture: &str) -> bool {
    matches!(
        architecture,
        "llama"
            | "mistral"
            | "qwen2"
            | "qwen3"
            | "qwen35"
            | "smollm3"
            | "gemma3"
            | "gemma4"
            | "phi3"
            | "lfm2"
    )
}

impl LlamaModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = match gguf.architecture() {
            Some(
                architecture @ ("llama" | "mistral" | "qwen2" | "qwen3" | "qwen35" | "smollm3"
                | "gemma3" | "gemma4" | "phi3" | "lfm2"),
            ) => architecture,
            // Gemma 4 MTP/assistant drafter heads ship as a distinct architecture.
            // The tensor map parses (q-only attention layers, per-layer
            // `layer_output_scale`, `nextn.pre/post_projection`), but the file
            // carries no K/V projections — all layers declare shared KV sourced
            // from the HOST model, and the host-hidden handoff plus the
            // speculative acceptance contract are undocumented. Fail closed with
            // the exact blocker rather than mis-binding it as a standalone model.
            Some("gemma4-assistant") => {
                return Err(BackendError::UnsupportedModelArchitecture(
                    "gemma4-assistant (Gemma 4 MTP/drafter head): blocked — the GGUF \
                     has no attn_k/attn_v tensors (KV is sourced from the host gemma4 \
                     model under an undocumented contract) and the nextn pre/post \
                     projection + acceptance semantics have no reference oracle yet; \
                     Camelid fails closed until lossless speculative decode can be \
                     proven token-identical to vanilla greedy"
                        .into(),
                ))
            }
            // DiffusionGemma (general.architecture spellings: diffusion_gemma /
            // diffusiongemma / gemma-diffusion). Despite the Gemma 4 26B-A4B MoE
            // foundation, this is NOT an autoregressive model: it is a discrete
            // block-diffusion encoder-decoder that generates by iteratively
            // denoising a token "canvas" with bidirectional decoder attention and
            // cross-attention to an encoder KV cache (multi-canvas sampling +
            // Entropy-Bound diffusion sampler), and it is multimodal (image/video
            // inputs). Camelid is a decoder-only autoregressive engine (causal
            // attention, KV cache, greedy next-token decode) and cannot run the
            // diffusion decode loop through THIS runtime. DiffusionGemma is instead
            // supported through its own dedicated lane (`DgEncoderRuntime`, the
            // `diffusion-gemma-chat` subcommand), which is bit-exact-validated
            // against the pinned reference. Redirect rather than mis-bind the
            // shared gemma4 tensors onto the autoregressive path.
            Some(other) if other.to_ascii_lowercase().contains("diffusion") => {
                return Err(BackendError::UnsupportedModelArchitecture(format!(
                    "{other} (DiffusionGemma): blocked — not an autoregressive model. \
                     It is a discrete block-diffusion encoder-decoder (bidirectional attention \
                     over a denoising token canvas, multi-canvas iterative sampling, \
                     Entropy-Bound diffusion sampler). The autoregressive engine cannot \
                     run the diffusion decode loop; use the dedicated DiffusionGemma lane \
                     instead: `camelid diffusion-gemma-chat <model.gguf>` (bit-exact \
                     CPU-pure runtime, experimental)"
                )))
            }
            Some(other) => return Err(BackendError::UnsupportedModelArchitecture(other.into())),
            None => {
                return Err(BackendError::InvalidModelMetadata(
                    "required metadata general.architecture is missing".into(),
                ))
            }
        };

        let moe = MixtralMoeMetadata::from_gguf(gguf, architecture);
        let gemma4 = Gemma4Metadata::from_gguf(gguf, architecture);
        let qwen35 = Qwen35Metadata::from_gguf(gguf, architecture);

        let attention_head_count = required_u32(
            gguf,
            &architecture_key(architecture, "attention.head_count"),
        )?;
        // Gemma 4 rows carry per-layer arrays for `feed_forward_length` (E2B) and
        // `attention.head_count_kv` (12B). The per-layer truth lives in
        // `Gemma4Metadata` (`ffn_length_at`/`kv_heads_at`); these config scalars
        // hold the per-layer MAX so generic sizing stays safe. Gemma 4 forward
        // paths must use the per-layer accessors, never these scalars.
        let attention_head_count_kv = match gemma4.as_ref() {
            Some(g) => g.max_kv_heads(),
            None => llama_attention_head_count_kv(gguf, architecture, attention_head_count),
        };
        let feed_forward_length = match gemma4.as_ref() {
            Some(g) if g.max_ffn_length() > 0 => g.max_ffn_length(),
            _ => required_u32(gguf, &architecture_key(architecture, "feed_forward_length"))?,
        };
        Ok(Self {
            context_length: required_u32(gguf, &architecture_key(architecture, "context_length"))?,
            embedding_length: required_u32(
                gguf,
                &architecture_key(architecture, "embedding_length"),
            )?,
            block_count: required_u32(gguf, &architecture_key(architecture, "block_count"))?,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            kv_quant: std::env::var("CAMELID_KV_QUANT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            rope_dimension_count: gguf
                .metadata_u32(&architecture_key(architecture, "rope.dimension_count")),
            rope_freq_base: gguf.metadata_f32(&architecture_key(architecture, "rope.freq_base")),
            rope_scaling_type: gguf
                .metadata_string(&architecture_key(architecture, "rope.scaling.type"))
                .map(str::to_string),
            rope_scaling_factor: gguf
                .metadata_f32(&architecture_key(architecture, "rope.scaling.factor")),
            rope_scaling_original_context_length: gguf.metadata_u32(&architecture_key(
                architecture,
                "rope.scaling.original_context_length",
            )),
            rope_scaling_low_freq_factor: gguf.metadata_f32(&architecture_key(
                architecture,
                "rope.scaling.low_freq_factor",
            )),
            rope_scaling_high_freq_factor: gguf.metadata_f32(&architecture_key(
                architecture,
                "rope.scaling.high_freq_factor",
            )),
            rms_norm_epsilon: gguf
                .metadata_f32(&architecture_key(
                    architecture,
                    "attention.layer_norm_rms_epsilon",
                ))
                .unwrap_or(1e-5),
            vocab_size: gguf
                .metadata_u32(&architecture_key(architecture, "vocab_size"))
                .or_else(|| {
                    infer_vocab_size_from_token_embedding(
                        gguf,
                        "token_embd.weight",
                        required_u32(gguf, &architecture_key(architecture, "embedding_length"))
                            .ok()?,
                    )
                }),
            file_type: gguf.metadata_u32("general.file_type"),
            // Explicit head_dim for the dense path (gemma4 has its own per-layer
            // head-dim handling, so don't surface it here for that arch).
            attention_key_length: if gemma4.is_some() {
                None
            } else {
                gguf.metadata_u32(&architecture_key(architecture, "attention.key_length"))
            },
            // Qwen3 GGUFs are not weight-permuted (unlike LLaMA conversions), so
            // their RoPE must use NEOX split-half pairing to match llama.cpp.
            // Verified token-identical against the pinned reference for
            // Qwen3-1.7B Q8_0. phi3 is likewise unpermuted and was PROVEN to need
            // NEOX during MUSTER M-A2: with adjacent even/odd pairing, long
            // open-ended generation degenerates within a handful of tokens (the
            // known 92029b7e limitation), while a CAMELID_ROPE_PAIRING=split_half
            // probe on the exact Phi-3-mini-4k Q8_0 row produces coherent
            // long-form output — the runnable lane independently asserts NEOX for
            // phi3. Qwen2/Qwen2.5 uses the same unpermuted split-half layout;
            // this is exercised by the real Qwen2.5 Q3_K_M mini2 smoke lane.
            // Other unpermuted archs (gemma3/…) stay out of this path until their
            // own rows prove it (gemma3 serves via the runnable lane).
            // qwen35 full-attention layers are also unpermuted (NEOX split-half),
            // with partial RoPE over the first `rope.dimension_count` (64) of the
            // 256-wide head — handled in the runnable qwen35 path.
            rope_neox_pairing: arch_uses_neox_rope_pairing(architecture),
            logit_scale: gguf.metadata_f32(&architecture_key(architecture, "logit_scale")),
            moe,
            gemma4,
            qwen35,
            mla: MlaMetadata::from_gguf(gguf, architecture),
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MlaMetadata {
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub nope_head_dim: u32,
    pub rope_head_dim: u32,
}

impl MlaMetadata {
    pub fn from_gguf(gguf: &GgufFile, architecture: &str) -> Option<Self> {
        if architecture != "deepseek2" && architecture != "deepseek3" {
            return None;
        }
        Some(Self {
            q_lora_rank: gguf
                .metadata_u32(&architecture_key(architecture, "attention.q_lora_rank"))
                .unwrap_or(1536),
            kv_lora_rank: gguf
                .metadata_u32(&architecture_key(architecture, "attention.kv_lora_rank"))
                .unwrap_or(512),
            nope_head_dim: gguf
                .metadata_u32(&architecture_key(architecture, "attention.key_length"))
                .unwrap_or(128),
            rope_head_dim: gguf
                .metadata_u32(&architecture_key(
                    architecture,
                    "attention.rope_dimension_count",
                ))
                .unwrap_or(64),
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MixtralMoeMetadata {
    pub family_label: &'static str,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32,
}

impl MixtralMoeMetadata {
    pub fn from_gguf(gguf: &GgufFile, architecture: &str) -> Option<Self> {
        let expert_count = gguf.metadata_u32(&architecture_key(architecture, "expert_count"))?;
        let expert_used_count =
            gguf.metadata_u32(&architecture_key(architecture, "expert_used_count"))?;
        let model_name = gguf.model_name().unwrap_or_default().to_ascii_lowercase();
        let basename = gguf
            .metadata_string("general.basename")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let family_label = if model_name.contains("mixtral") || basename.contains("mixtral") {
            "Mixtral"
        } else {
            "MoE"
        };

        let expert_weights_scale = gguf
            .metadata_f32(&architecture_key(architecture, "expert_weights_scale"))
            .unwrap_or(1.0);
        let expert_weights_norm = gguf
            .metadata_bool(&architecture_key(architecture, "expert_weights_norm"))
            .unwrap_or(false);
        let expert_gating_func = gguf
            .metadata_u32(&architecture_key(architecture, "expert_gating_func"))
            .unwrap_or(0);

        Some(Self {
            family_label,
            expert_count,
            expert_used_count,
            expert_weights_scale,
            expert_weights_norm,
            expert_gating_func,
        })
    }
}

/// Gemma 4 (`general.architecture = "gemma4"`) attention/embedding metadata that
/// the shared Llama config cannot represent. Parsed from the `gemma4.*` GGUF keys.
///
/// Gemma 4 alternates sliding (local) and full (global) attention on a 5:1
/// schedule, and — unlike Llama — the two layer types use *different* per-head
/// dimensions and RoPE bases. The elastic "E" variants additionally feed a
/// Per-Layer-Embedding stream into every block. None of this drives the forward
/// pass yet; this struct only captures the parsed values for the Gemma 4 runtime.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Gemma4Metadata {
    /// Per-head dim for sliding (local) layers — GGUF `attention.key_length_swa`.
    pub head_dim_sliding: u32,
    /// Per-head dim for full (global) layers — GGUF `attention.key_length`.
    pub head_dim_global: u32,
    /// RoPE base for full (global) layers — GGUF `rope.freq_base`.
    pub rope_freq_base_global: f32,
    /// RoPE base for sliding (local) layers — GGUF `rope.freq_base_swa`.
    pub rope_freq_base_sliding: f32,
    /// Rotary dim applied on full (global) layers — GGUF `rope.dimension_count`.
    pub rope_dim_global: u32,
    /// Rotary dim applied on sliding (local) layers — GGUF `rope.dimension_count_swa`.
    pub rope_dim_sliding: u32,
    /// Local attention window — GGUF `attention.sliding_window`.
    pub sliding_window: u32,
    /// Count of trailing layers that share KV projections — GGUF
    /// `attention.shared_kv_layers` (0 = no cross-layer KV sharing).
    pub num_kv_shared_layers: u32,
    /// Per-Layer-Embedding width — GGUF `embedding_length_per_layer_input`
    /// (0 for the dense variants, which carry no PLE stream).
    pub per_layer_input_dim: u32,
    /// Final logit soft-cap — GGUF `final_logit_softcapping` (None if absent).
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer attention type: `true` = sliding (local), `false` = full (global).
    /// Derived from the 5:1 schedule with a forced full final layer, matching the
    /// Gemma 4 reference and the observed `attention.sliding_window_pattern`.
    pub layer_is_sliding: Vec<bool>,
    /// Per-layer FFN width. Gemma 4 rows are NOT uniform here: E2B carries a
    /// per-layer `feed_forward_length` array (6144 for the first 15 layers,
    /// 12288 for the rest), while E4B/12B carry a scalar (broadcast).
    pub ffn_lengths: Vec<u32>,
    /// Per-layer KV head count. The 12B row carries a per-layer
    /// `attention.head_count_kv` array (8 on sliding layers, 1 on global
    /// layers); E2B/E4B carry a scalar (broadcast).
    pub kv_heads_per_layer: Vec<u32>,
}

impl Gemma4Metadata {
    /// Returns `Some` only for the `gemma4` and `diffusion-gemma`
    /// architectures; `None` otherwise. `diffusion-gemma` shares the Gemma 4
    /// backbone and key suffixes (different prefix). Parsing here is
    /// metadata-layer only: the AR runtime still fails closed on any
    /// diffusion architecture in `LlamaModelConfig::from_gguf`; only the
    /// experimental DiffusionGemma lane consumes this struct for that arch.
    pub fn from_gguf(gguf: &GgufFile, architecture: &str) -> Option<Self> {
        if architecture != "gemma4" && architecture != "diffusion-gemma" {
            return None;
        }
        let key = |suffix: &str| architecture_key(architecture, suffix);
        let head_dim_sliding = gguf
            .metadata_u32(&key("attention.key_length_swa"))
            .or_else(|| gguf.metadata_u32(&key("attention.key_length")))
            .unwrap_or(256);
        let head_dim_global = gguf
            .metadata_u32(&key("attention.key_length"))
            .unwrap_or(head_dim_sliding);
        let block_count = gguf.metadata_u32(&key("block_count")).unwrap_or(0);
        // The GGUF's own `attention.sliding_window_pattern` bool array is the
        // authoritative per-layer schedule when it covers every layer; the 5:1
        // formula is the fallback for files that omit it. A row whose pattern
        // diverges from the formula (anything other than E4B's 42-layer layout
        // has never been proven) must not be silently mis-scheduled.
        let layer_is_sliding =
            match gguf.metadata_array_bools_optional(&key("attention.sliding_window_pattern")) {
                Ok(Some(pattern)) if pattern.len() == block_count as usize => pattern,
                _ => gemma4_sliding_schedule(block_count),
            };
        // Per-layer-or-scalar keys: a scalar broadcasts to every layer; an array
        // must cover every layer to be honored (anything else falls back to the
        // scalar default so the shape validation in Gemma4Binding fails loudly
        // instead of silently mis-binding).
        let per_layer_or_scalar = |suffix: &str, default: u32| -> Vec<u32> {
            if let Some(scalar) = gguf.metadata_u32(&key(suffix)) {
                return vec![scalar; block_count as usize];
            }
            match gguf.metadata_array_u32_optional(&key(suffix)) {
                Ok(Some(values)) if values.len() == block_count as usize => values,
                _ => vec![default; block_count as usize],
            }
        };
        let ffn_lengths = per_layer_or_scalar("feed_forward_length", 0);
        let head_count = gguf.metadata_u32(&key("attention.head_count")).unwrap_or(0);
        let kv_heads_per_layer = per_layer_or_scalar("attention.head_count_kv", head_count);
        Some(Self {
            head_dim_sliding,
            head_dim_global,
            rope_freq_base_global: gguf
                .metadata_f32(&key("rope.freq_base"))
                .unwrap_or(1_000_000.0),
            rope_freq_base_sliding: gguf
                .metadata_f32(&key("rope.freq_base_swa"))
                .unwrap_or(10_000.0),
            rope_dim_global: gguf
                .metadata_u32(&key("rope.dimension_count"))
                .unwrap_or(head_dim_global),
            rope_dim_sliding: gguf
                .metadata_u32(&key("rope.dimension_count_swa"))
                .unwrap_or(head_dim_sliding),
            sliding_window: gguf
                .metadata_u32(&key("attention.sliding_window"))
                .unwrap_or(512),
            num_kv_shared_layers: gguf
                .metadata_u32(&key("attention.shared_kv_layers"))
                .unwrap_or(0),
            per_layer_input_dim: gguf
                .metadata_u32(&key("embedding_length_per_layer_input"))
                .unwrap_or(0),
            final_logit_softcapping: gguf.metadata_f32(&key("final_logit_softcapping")),
            layer_is_sliding,
            ffn_lengths,
            kv_heads_per_layer,
        })
    }

    /// Per-layer FFN width (E2B varies this across layers).
    pub fn ffn_length_at(&self, idx: usize) -> u32 {
        self.ffn_lengths.get(idx).copied().unwrap_or(0)
    }

    /// Per-layer KV head count (12B varies this across layers).
    pub fn kv_heads_at(&self, idx: usize) -> u32 {
        self.kv_heads_per_layer.get(idx).copied().unwrap_or(0)
    }

    /// Largest per-layer FFN width — for code that needs a single bound.
    pub fn max_ffn_length(&self) -> u32 {
        self.ffn_lengths.iter().copied().max().unwrap_or(0)
    }

    /// Largest per-layer KV head count — for code that needs a single bound.
    pub fn max_kv_heads(&self) -> u32 {
        self.kv_heads_per_layer.iter().copied().max().unwrap_or(0)
    }

    /// True if decoder layer `idx` uses sliding (local) attention.
    pub fn is_sliding_layer(&self, idx: usize) -> bool {
        self.layer_is_sliding.get(idx).copied().unwrap_or(false)
    }

    /// Per-head attention dim for layer `idx`. Gemma 4 uses a smaller head dim on
    /// sliding (local) layers than on full (global) layers.
    pub fn head_dim_at(&self, idx: usize) -> u32 {
        if self.is_sliding_layer(idx) {
            self.head_dim_sliding
        } else {
            self.head_dim_global
        }
    }

    /// Per-head rotary dim for layer `idx` (sliding vs global).
    pub fn rope_dim_at(&self, idx: usize) -> u32 {
        if self.is_sliding_layer(idx) {
            self.rope_dim_sliding
        } else {
            self.rope_dim_global
        }
    }

    /// RoPE base (theta) for layer `idx` (sliding θ vs global θ).
    pub fn rope_freq_base_at(&self, idx: usize) -> f32 {
        if self.is_sliding_layer(idx) {
            self.rope_freq_base_sliding
        } else {
            self.rope_freq_base_global
        }
    }

    /// Per-layer decode plan for the GPU-resident runtime: resolves each layer's
    /// per-type dims, RoPE θ, sliding window, and — for the trailing
    /// `num_kv_shared_layers` layers that don't project their own K/V — which
    /// earlier same-type layer's KV cache it reads. This is the single source of
    /// truth for gemma's per-layer-type attention + cross-layer KV sharing, mirrored
    /// from the CPU `Gemma4Runtime` (`first_kv_shared`, `last_sliding/full_layer`).
    pub fn layer_plan(&self, block_count: usize, heads: usize) -> Vec<Gemma4LayerPlan> {
        let first_kv_shared = block_count.saturating_sub(self.num_kv_shared_layers as usize);
        // The last owning (non-shared) layer of each attention type — the cache a
        // trailing shared layer of that type reads.
        let last_sliding = (0..first_kv_shared)
            .rev()
            .find(|&l| self.is_sliding_layer(l))
            .unwrap_or(0);
        let last_global = (0..first_kv_shared)
            .rev()
            .find(|&l| !self.is_sliding_layer(l))
            .unwrap_or(0);
        (0..block_count)
            .map(|l| {
                let sliding = self.is_sliding_layer(l);
                let head_dim = self.head_dim_at(l) as usize;
                let owns_kv = l < first_kv_shared;
                let kv_source_layer = if owns_kv {
                    l
                } else if sliding {
                    last_sliding
                } else {
                    last_global
                };
                // A shared layer reads its SOURCE layer's cache, so its KV
                // geometry must be the source's (same-type layers share head_dim,
                // but per-layer kv head counts make this explicit).
                let kv_heads = self.kv_heads_at(kv_source_layer) as usize;
                Gemma4LayerPlan {
                    sliding,
                    head_dim,
                    q_dim: heads * head_dim,
                    kv_heads,
                    kv_dim: kv_heads * head_dim,
                    theta: self.rope_freq_base_at(l),
                    window: if sliding {
                        Some(self.sliding_window as usize)
                    } else {
                        None
                    },
                    owns_kv,
                    kv_source_layer,
                }
            })
            .collect()
    }
}

/// Resolved per-layer attention geometry for the gemma4 GPU-resident decode graph
/// (see [`Gemma4Metadata::layer_plan`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4LayerPlan {
    /// Sliding (local) vs full (global) attention.
    pub sliding: bool,
    /// Per-head dim for this layer (256 sliding / 512 global on E4B).
    pub head_dim: usize,
    /// Query projection width = `heads * head_dim`.
    pub q_dim: usize,
    /// KV head count for the cache this layer READS (the source layer's when
    /// KV is shared; 12B varies kv heads per layer).
    pub kv_heads: usize,
    /// K/V projection width = `kv_heads * head_dim`.
    pub kv_dim: usize,
    /// RoPE base (θ) for this layer's type.
    pub theta: f32,
    /// `Some(window)` for sliding layers (attend `[pos+1-window ..= pos]`), `None`
    /// for global layers (attend `[0 ..= pos]`).
    pub window: Option<usize>,
    /// True if this layer projects + caches its own K/V; false for the trailing
    /// `num_kv_shared_layers` layers, which read `kv_source_layer`'s cache.
    pub owns_kv: bool,
    /// Layer whose KV cache this layer reads (itself when `owns_kv`).
    pub kv_source_layer: usize,
}

/// Gemma 4's per-layer attention schedule: a 5:1 sliding:full repeat (every 6th
/// layer is full/global) with the final layer forced to full attention. This
/// mirrors `Gemma4TextConfig.__post_init__` and the `attention.sliding_window_pattern`
/// array carried in the GGUF. `true` = sliding (local), `false` = full (global).
fn gemma4_sliding_schedule(block_count: u32) -> Vec<bool> {
    const SLIDING_PERIOD: u32 = 6;
    let mut schedule: Vec<bool> = (0..block_count)
        .map(|i| (i + 1) % SLIDING_PERIOD != 0)
        .collect();
    if let Some(last) = schedule.last_mut() {
        *last = false;
    }
    schedule
}

#[cfg(test)]
mod gemma4_tests {
    use super::{gemma4_sliding_schedule, Gemma4Metadata};

    fn e4b_meta() -> Gemma4Metadata {
        Gemma4Metadata {
            head_dim_sliding: 256,
            head_dim_global: 512,
            rope_freq_base_global: 1_000_000.0,
            rope_freq_base_sliding: 10_000.0,
            rope_dim_global: 512,
            rope_dim_sliding: 256,
            sliding_window: 512,
            num_kv_shared_layers: 18,
            per_layer_input_dim: 256,
            final_logit_softcapping: Some(30.0),
            layer_is_sliding: gemma4_sliding_schedule(42),
            ffn_lengths: vec![10240; 42],
            kv_heads_per_layer: vec![2; 42],
        }
    }

    #[test]
    fn layer_plan_resolves_dims_window_and_kv_sharing() {
        let meta = e4b_meta();
        let plan = meta.layer_plan(42, 8);
        assert_eq!(plan.len(), 42);
        // first_kv_shared = 42 - 18 = 24.
        for (l, p) in plan.iter().enumerate() {
            assert_eq!(p.owns_kv, l < 24, "owns_kv layer {l}");
            assert_eq!(p.q_dim, 8 * p.head_dim);
            assert_eq!(p.kv_dim, 2 * p.head_dim);
            if p.sliding {
                assert_eq!(p.head_dim, 256);
                assert_eq!(p.window, Some(512));
                assert_eq!(p.theta, 10_000.0);
            } else {
                assert_eq!(p.head_dim, 512);
                assert_eq!(p.window, None);
                assert_eq!(p.theta, 1_000_000.0);
            }
            // Owning layers read their own cache; the trailing shared layers read an
            // earlier OWNING layer of the SAME attention type.
            if p.owns_kv {
                assert_eq!(p.kv_source_layer, l);
            } else {
                let src = &plan[p.kv_source_layer];
                assert!(
                    src.owns_kv,
                    "layer {l} source {} must own KV",
                    p.kv_source_layer
                );
                assert_eq!(src.sliding, p.sliding, "layer {l} source must match type");
                assert!(p.kv_source_layer < 24);
            }
        }
        // Spot checks: last sliding/global owning layer before the shared block is 22/23.
        assert_eq!(plan[24].kv_source_layer, 22); // layer 24 sliding -> last owning sliding
        assert_eq!(plan[41].kv_source_layer, 23); // layer 41 (forced global) -> last owning global
        assert!(!plan[41].sliding);
    }

    #[test]
    fn sliding_schedule_is_5to1_with_full_final_layer() {
        // E4B has 42 layers; the reference forces full attention every 6th layer.
        let schedule = gemma4_sliding_schedule(42);
        assert_eq!(schedule.len(), 42);
        let full_layers: Vec<usize> = schedule
            .iter()
            .enumerate()
            .filter(|(_, sliding)| !**sliding)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_layers, vec![5, 11, 17, 23, 29, 35, 41]);
        // The first five are sliding, the sixth (index 5) is full — matches the
        // observed GGUF pattern [1,1,1,1,1,0,...].
        assert_eq!(&schedule[..6], &[true, true, true, true, true, false]);
        // Final layer must always be full attention even when the count is not a
        // multiple of six.
        let odd = gemma4_sliding_schedule(40);
        assert_eq!(odd.last(), Some(&false));
    }
}

/// Qwen3.5 (`general.architecture = "qwen35"`) hybrid linear-attention metadata.
///
/// Qwen3.5 alternates **gated-delta-net (SSM / linear-attention)** layers with
/// standard **full-attention** layers on a `full_attention_interval` schedule
/// (layer `i` is recurrent iff `(i+1) % interval != 0`). The SSM layers carry a
/// distinct tensor set (`attn_qkv`/`attn_gate`/`ssm_*`) and run a per-head
/// recurrent state instead of K/V attention; this struct captures the SSM dims and
/// the per-layer schedule that a dense Llama config cannot represent. Parsed from
/// the `qwen35.*` GGUF keys. None of this drives the optimized lane; only the
/// runnable lane's `qwen35` path consumes it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Qwen35Metadata {
    /// Causal conv1d kernel width — GGUF `ssm.conv_kernel` (4).
    pub ssm_d_conv: u32,
    /// SSM inner size = `num_v_heads * head_v_dim` — GGUF `ssm.inner_size` (4096).
    pub ssm_d_inner: u32,
    /// Per-head state dim (= head_k_dim = head_v_dim) — GGUF `ssm.state_size` (128).
    pub ssm_d_state: u32,
    /// Number of value/delta heads — GGUF `ssm.time_step_rank` (32).
    pub ssm_dt_rank: u32,
    /// Number of key/query heads (groups) — GGUF `ssm.group_count` (16).
    pub ssm_n_group: u32,
    /// Full-attention cadence — GGUF `full_attention_interval` (4).
    pub full_attention_interval: u32,
    /// Per-layer schedule: `true` = recurrent (SSM/linear-attn), `false` = full
    /// attention. The explicit `attention.recurrent_layers` bool array (when it
    /// covers every layer) overrides the interval rule; otherwise derived from it.
    pub layer_is_recurrent: Vec<bool>,
}

impl Qwen35Metadata {
    pub fn from_gguf(gguf: &GgufFile, architecture: &str) -> Option<Self> {
        if architecture != "qwen35" {
            return None;
        }
        let key = |suffix: &str| architecture_key(architecture, suffix);
        let block_count = gguf.metadata_u32(&key("block_count")).unwrap_or(0);
        let full_attention_interval = gguf
            .metadata_u32(&key("full_attention_interval"))
            .unwrap_or(4)
            .max(1);
        let layer_is_recurrent =
            match gguf.metadata_array_bools_optional(&key("attention.recurrent_layers")) {
                Ok(Some(arr)) if arr.len() == block_count as usize => arr,
                _ => (0..block_count)
                    .map(|i| (i + 1) % full_attention_interval != 0)
                    .collect(),
            };
        Some(Self {
            ssm_d_conv: gguf.metadata_u32(&key("ssm.conv_kernel")).unwrap_or(4),
            ssm_d_inner: gguf.metadata_u32(&key("ssm.inner_size")).unwrap_or(0),
            ssm_d_state: gguf.metadata_u32(&key("ssm.state_size")).unwrap_or(0),
            ssm_dt_rank: gguf.metadata_u32(&key("ssm.time_step_rank")).unwrap_or(0),
            ssm_n_group: gguf.metadata_u32(&key("ssm.group_count")).unwrap_or(0),
            full_attention_interval,
            layer_is_recurrent,
        })
    }

    /// True if decoder layer `idx` is a recurrent (SSM / linear-attention) layer.
    pub fn is_recurrent_layer(&self, idx: usize) -> bool {
        self.layer_is_recurrent.get(idx).copied().unwrap_or(false)
    }
}

/// NEOX split-half RoPE pairing per architecture: qwen2/qwen3/qwen35 and phi3
/// (unpermuted weights;
/// proven during MUSTER M-A2 — adjacent even/odd degenerates long generation,
/// split-half restores coherence; the runnable lane independently asserts NEOX
/// for phi3). Everything else keeps adjacent even/odd (LLaMA-style permuted
/// conversions). Pure so the gate is unit-testable.
fn arch_uses_neox_rope_pairing(architecture: &str) -> bool {
    matches!(architecture, "qwen2" | "qwen3" | "qwen35" | "phi3")
}

fn architecture_key(architecture: &str, suffix: &str) -> String {
    format!("{architecture}.{suffix}")
}

fn llama_attention_head_count_kv(
    gguf: &GgufFile,
    architecture: &str,
    attention_head_count: u32,
) -> u32 {
    gguf.metadata_u32(&architecture_key(architecture, "attention.head_count_kv"))
        .unwrap_or(attention_head_count)
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaLayerTensors {
    pub attention_norm: GgufTensorDescriptor,
    pub attention: LlamaAttentionTensors,
    pub attention_output: GgufTensorDescriptor,
    pub ffn_norm: GgufTensorDescriptor,
    pub ffn: LlamaFfnTensors,
}

impl LlamaLayerTensors {
    pub fn attention_q(&self) -> Option<&GgufTensorDescriptor> {
        match &self.attention {
            LlamaAttentionTensors::Standard { q, .. } => Some(q),
            LlamaAttentionTensors::Mla { .. } => None,
        }
    }

    pub fn attention_k(&self) -> Option<&GgufTensorDescriptor> {
        match &self.attention {
            LlamaAttentionTensors::Standard { k, .. } => Some(k),
            LlamaAttentionTensors::Mla { .. } => None,
        }
    }

    pub fn attention_v(&self) -> Option<&GgufTensorDescriptor> {
        match &self.attention {
            LlamaAttentionTensors::Standard { v, .. } => Some(v),
            LlamaAttentionTensors::Mla { .. } => None,
        }
    }

    pub fn attention_q_norm(&self) -> Option<&GgufTensorDescriptor> {
        match &self.attention {
            LlamaAttentionTensors::Standard { q_norm, .. } => q_norm.as_ref(),
            LlamaAttentionTensors::Mla { .. } => None,
        }
    }

    pub fn attention_k_norm(&self) -> Option<&GgufTensorDescriptor> {
        match &self.attention {
            LlamaAttentionTensors::Standard { k_norm, .. } => k_norm.as_ref(),
            LlamaAttentionTensors::Mla { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaAttentionBiasTensors {
    pub q: GgufTensorDescriptor,
    pub k: GgufTensorDescriptor,
    pub v: GgufTensorDescriptor,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum LlamaAttentionTensors {
    Standard {
        q: GgufTensorDescriptor,
        k: GgufTensorDescriptor,
        v: GgufTensorDescriptor,
        /// Per-head RMSNorm applied to the Q projection *after* reshape-to-heads and
        /// *before* RoPE. `Some` only for architectures that use QK-norm (Qwen3);
        /// `None` for plain Llama-family rows (llama/mistral/qwen2/…). When `Some`
        /// the descriptor shape is `[head_dim]`. See [`LlamaTensorBinding::bind`] for
        /// the per-architecture presence invariant.
        q_norm: Option<GgufTensorDescriptor>,
        /// Per-head RMSNorm applied to the K projection, mirroring
        /// [`Self::attention_q_norm`]. Bound in lockstep with it (both `Some` or both
        /// `None`).
        k_norm: Option<GgufTensorDescriptor>,
        /// Optional Q/K/V projection biases. Qwen2/Qwen2.5 requires all three;
        /// most Llama-shaped architectures carry none.
        biases: Option<LlamaAttentionBiasTensors>,
    },
    Mla {
        q_a_proj: GgufTensorDescriptor,
        q_a_layernorm: GgufTensorDescriptor,
        q_b_proj: GgufTensorDescriptor,
        kv_a_proj_with_mqa: GgufTensorDescriptor,
        kv_a_layernorm: GgufTensorDescriptor,
        kv_b_proj: GgufTensorDescriptor,
    },
}

// The descriptor tree is built once at model-bind time and its direct, named fields keep
// architecture validation auditable. Boxing a single DeepSeek-only field merely to shave the
// enum discriminant size would complicate every binding consumer without affecting inference
// memory, which is dominated by the model tensors themselves.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum LlamaFfnTensors {
    Dense {
        gate: GgufTensorDescriptor,
        up: GgufTensorDescriptor,
        down: GgufTensorDescriptor,
    },
    MoE {
        router: GgufTensorDescriptor,
        gate_experts: LlamaMoeExpertTensors,
        up_experts: LlamaMoeExpertTensors,
        down_experts: LlamaMoeExpertTensors,
    },
    DeepSeekMoE {
        shared_gate: GgufTensorDescriptor,
        shared_up: GgufTensorDescriptor,
        shared_down: GgufTensorDescriptor,
        router: GgufTensorDescriptor,
        gate_experts: LlamaMoeExpertTensors,
        up_experts: LlamaMoeExpertTensors,
        down_experts: LlamaMoeExpertTensors,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum LlamaMoeExpertTensors {
    Merged(GgufTensorDescriptor),
    Split(Vec<GgufTensorDescriptor>),
}

impl LlamaMoeExpertTensors {
    pub fn descriptors(&self) -> &[GgufTensorDescriptor] {
        match self {
            Self::Merged(desc) => std::slice::from_ref(desc),
            Self::Split(descs) => descs,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LlamaTensorBinding {
    pub token_embedding: GgufTensorDescriptor,
    pub output_norm: GgufTensorDescriptor,
    pub output: GgufTensorDescriptor,
    pub output_is_tied_embedding: bool,
    pub rope_freqs: Option<GgufTensorDescriptor>,
    pub mla_metadata: Option<MlaMetadata>,
    pub attention_head_count: usize,
    pub hidden_size: usize,
    pub layers: Vec<LlamaLayerTensors>,
}

impl LlamaTensorBinding {
    pub fn bind(gguf: &GgufFile, config: &LlamaModelConfig) -> Result<Self> {
        let token_embedding = required_tensor(gguf, "token_embd.weight")?;
        let output_norm = required_tensor(gguf, "output_norm.weight")?;
        let (output, output_is_tied_embedding) = match find_tensor(gguf, "output.weight") {
            Some(desc) => (desc.clone(), false),
            None => (token_embedding.clone(), true),
        };
        let rope_freqs = find_tensor(gguf, "rope_freqs.weight").cloned();

        // Per-architecture QK-norm classification. Qwen3 applies a per-head
        // RMSNorm to Q and K after the projections and before RoPE
        // (`attn_q_norm`/`attn_k_norm`, shape `[head_dim]`); the plain
        // Llama-family rows do not. We classify every architecture that reaches
        // this dense binder so a model can never be silently mis-bound in either
        // direction (carrying QK-norm weights that the forward path would drop,
        // or fabricating them where none exist).
        let architecture = gguf.architecture().unwrap_or_default();
        let expects_qk_norm = architecture == "qwen3" || architecture == "command-r";
        let forbids_qk_norm = matches!(architecture, "llama" | "mistral" | "qwen2");

        // Qwen3 sets the per-head dim explicitly via `attention.key_length` /
        // `attention.value_length` and it is NOT guaranteed to equal
        // `embedding_length / head_count` (0.6B/4B/32B differ). The dense path
        // carries the explicit head_dim through `LlamaModelConfig.attention_key_length`
        // / `DenseLlamaDims`. The engine assumes a single head_dim for K and V, so
        // require key_length == value_length and fail closed otherwise.
        if expects_qk_norm {
            let key_length =
                gguf.metadata_u32(&architecture_key(architecture, "attention.key_length"));
            let value_length =
                gguf.metadata_u32(&architecture_key(architecture, "attention.value_length"));
            if let (Some(k), Some(v)) = (key_length, value_length) {
                if k != v {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "qwen3 attention.key_length={k} != value_length={v}; the engine assumes a \
                         single per-head dimension for K and V, so this row fails closed"
                    )));
                }
            }
        }

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for layer_idx in 0..config.block_count {
            let q_norm_name = format!("blk.{layer_idx}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{layer_idx}.attn_k_norm.weight");
            let (attention_q_norm, attention_k_norm) = if expects_qk_norm {
                // Required for Qwen3: both must be present, or fail closed.
                let q = find_tensor(gguf, &q_norm_name).cloned();
                let k = find_tensor(gguf, &k_norm_name).cloned();
                if q.is_none() || k.is_none() {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "qwen3/command-r layer {layer_idx} is missing QK-norm tensors \
                         (attn_q_norm present: {}, attn_k_norm present: {}); this architecture applies \
                         per-head norm to Q and K and cannot be run correctly without them",
                        q.is_some(),
                        k.is_some()
                    )));
                }
                (q, k)
            } else {
                // Forbidden for the plain Llama-family rows: if a GGUF unexpectedly
                // carries QK-norm tensors under one of these architectures, the
                // forward path would silently drop them — fail closed instead.
                if forbids_qk_norm
                    && (find_tensor(gguf, &q_norm_name).is_some()
                        || find_tensor(gguf, &k_norm_name).is_some())
                {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "architecture {architecture:?} unexpectedly carries QK-norm tensors at \
                         layer {layer_idx} (attn_q_norm/attn_k_norm); the Llama-family forward \
                         path does not apply them, so Camelid fails closed rather than running \
                         a model whose weights it would silently ignore"
                    )));
                }
                (None, None)
            };
            let q_bias = find_tensor(gguf, &format!("blk.{layer_idx}.attn_q.bias")).cloned();
            let k_bias = find_tensor(gguf, &format!("blk.{layer_idx}.attn_k.bias")).cloned();
            let v_bias = find_tensor(gguf, &format!("blk.{layer_idx}.attn_v.bias")).cloned();
            let attention_biases = match (q_bias, k_bias, v_bias) {
                (Some(q), Some(k), Some(v)) => Some(LlamaAttentionBiasTensors { q, k, v }),
                (None, None, None) if architecture == "qwen2" => {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "qwen2 layer {layer_idx} is missing required attn_q/attn_k/attn_v bias tensors"
                    )))
                }
                (None, None, None) => None,
                (q, k, v) => {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "layer {layer_idx} has an incomplete attention bias set \
                         (q={}, k={}, v={}); Q/K/V projection biases must be present together",
                        q.is_some(),
                        k.is_some(),
                        v.is_some()
                    )))
                }
            };
            let attention = if architecture == "deepseek2" || architecture == "deepseek3" {
                LlamaAttentionTensors::Mla {
                    q_a_proj: required_tensor(
                        gguf,
                        &format!("blk.{layer_idx}.attn_q_a_proj.weight"),
                    )?,
                    q_a_layernorm: required_tensor(
                        gguf,
                        &format!("blk.{layer_idx}.attn_q_a_layernorm.weight"),
                    )?,
                    q_b_proj: required_tensor(
                        gguf,
                        &format!("blk.{layer_idx}.attn_q_b_proj.weight"),
                    )?,
                    kv_a_proj_with_mqa: required_tensor(
                        gguf,
                        &format!("blk.{layer_idx}.attn_kv_a_proj_with_mqa.weight"),
                    )?,
                    kv_a_layernorm: required_tensor(
                        gguf,
                        &format!("blk.{layer_idx}.attn_kv_a_layernorm.weight"),
                    )?,
                    kv_b_proj: required_tensor(
                        gguf,
                        &format!("blk.{layer_idx}.attn_kv_b_proj.weight"),
                    )?,
                }
            } else {
                LlamaAttentionTensors::Standard {
                    q: required_tensor(gguf, &format!("blk.{layer_idx}.attn_q.weight"))?,
                    k: required_tensor(gguf, &format!("blk.{layer_idx}.attn_k.weight"))?,
                    v: required_tensor(gguf, &format!("blk.{layer_idx}.attn_v.weight"))?,
                    q_norm: attention_q_norm,
                    k_norm: attention_k_norm,
                    biases: attention_biases,
                }
            };

            layers.push(LlamaLayerTensors {
                attention_norm: required_tensor(
                    gguf,
                    &format!("blk.{layer_idx}.attn_norm.weight"),
                )?,
                attention,
                attention_output: required_tensor(
                    gguf,
                    &format!("blk.{layer_idx}.attn_output.weight"),
                )?,
                ffn_norm: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_norm.weight"))?,
                ffn: if let Some(moe) = config.moe.as_ref() {
                    LlamaFfnTensors::MoE {
                        router: required_tensor(
                            gguf,
                            &format!("blk.{layer_idx}.ffn_gate_inp.weight"),
                        )?,
                        gate_experts: bind_moe_expert_tensors(
                            gguf,
                            layer_idx,
                            "gate",
                            moe.expert_count,
                        )?,
                        up_experts: bind_moe_expert_tensors(
                            gguf,
                            layer_idx,
                            "up",
                            moe.expert_count,
                        )?,
                        down_experts: bind_moe_expert_tensors(
                            gguf,
                            layer_idx,
                            "down",
                            moe.expert_count,
                        )?,
                    }
                } else {
                    LlamaFfnTensors::Dense {
                        gate: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_gate.weight"))?,
                        up: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_up.weight"))?,
                        down: required_tensor(gguf, &format!("blk.{layer_idx}.ffn_down.weight"))?,
                    }
                },
            });
        }

        let binding = Self {
            token_embedding,
            output_norm,
            output,
            output_is_tied_embedding,
            rope_freqs,
            mla_metadata: config.mla.clone(),
            attention_head_count: config.attention_head_count as usize,
            hidden_size: config.embedding_length as usize,
            layers,
        };
        binding.validate_dense_shapes(config)?;
        Ok(binding)
    }

    pub fn validate_dense_shapes(&self, config: &LlamaModelConfig) -> Result<()> {
        let dims = DenseLlamaDims::from_config(config)?;
        require_descriptor_matrix_shape(
            &self.token_embedding,
            dims.embedding_length,
            dims.vocab_size,
            "token embedding",
        )?;
        require_descriptor_shape(&self.output_norm, &[dims.embedding_length], "output norm")?;
        require_descriptor_matrix_shape(
            &self.output,
            dims.embedding_length,
            dims.vocab_size,
            "output projection",
        )?;
        validate_output_projection_storage_layout(
            &self.output,
            dims.embedding_length,
            dims.vocab_size,
        )?;
        if let Some(rope_freqs) = &self.rope_freqs {
            // Gemma 4 carries a single rope_freqs table sized for the global
            // (full-attention) layers; sliding layers derive their own shorter
            // rotary from rope.freq_base_swa at runtime. Validate against the
            // global rope dim there, and against the uniform head dim otherwise.
            let (rope_dim, head_dim_bound) = match config.gemma4.as_ref() {
                Some(g) => (g.rope_dim_global as usize, g.head_dim_global as usize),
                None => (
                    config.rope_dimension_count.unwrap_or(dims.head_dim as u32) as usize,
                    dims.head_dim,
                ),
            };
            if rope_dim == 0 || rope_dim > head_dim_bound || !rope_dim.is_multiple_of(2) {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "RoPE dimension count {rope_dim} must be even and within head dimension {head_dim_bound}"
                )));
            }
            require_descriptor_shape(rope_freqs, &[rope_dim / 2], "rope frequencies")?;
        }

        if self.layers.len() != dims.block_count {
            return Err(BackendError::InvalidModelMetadata(format!(
                "config block count {} does not match bound layer count {}",
                dims.block_count,
                self.layers.len()
            )));
        }

        for (idx, layer) in self.layers.iter().enumerate() {
            require_descriptor_shape(
                &layer.attention_norm,
                &[dims.embedding_length],
                &format!("layer {idx} attention norm"),
            )?;
            // Per-layer-type attention widths. For Llama these collapse to the
            // uniform case (head_dim = embedding/heads, so q_width = embedding);
            // for Gemma 4 the sliding and full layers use different head dims, so
            // the projection widths vary per layer.
            let head_dim = match config.gemma4.as_ref() {
                Some(g) => g.head_dim_at(idx) as usize,
                None => dims.head_dim,
            };
            let q_width = config.attention_head_count as usize * head_dim;
            let kv_width = config.attention_head_count_kv as usize * head_dim;
            match &layer.attention {
                LlamaAttentionTensors::Standard {
                    q,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    biases,
                } => {
                    require_descriptor_matrix_shape(
                        q,
                        dims.embedding_length,
                        q_width,
                        &format!("layer {idx} attention q"),
                    )?;
                    require_descriptor_matrix_shape(
                        k,
                        dims.embedding_length,
                        kv_width,
                        &format!("layer {idx} attention k"),
                    )?;
                    require_descriptor_matrix_shape(
                        v,
                        dims.embedding_length,
                        kv_width,
                        &format!("layer {idx} attention v"),
                    )?;
                    match (q_norm, k_norm) {
                        (Some(qn), Some(kn)) => {
                            require_descriptor_shape(
                                qn,
                                &[head_dim],
                                &format!("layer {idx} attention q_norm"),
                            )?;
                            require_descriptor_shape(
                                kn,
                                &[head_dim],
                                &format!("layer {idx} attention k_norm"),
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(BackendError::InvalidModelMetadata(format!(
                                "layer {idx} has exactly one of attn_q_norm/attn_k_norm bound; QK-norm \
                                 weights must be present as a pair"
                            )));
                        }
                    }
                    if let Some(biases) = biases {
                        require_descriptor_shape(
                            &biases.q,
                            &[q_width],
                            &format!("layer {idx} attention q bias"),
                        )?;
                        require_descriptor_shape(
                            &biases.k,
                            &[kv_width],
                            &format!("layer {idx} attention k bias"),
                        )?;
                        require_descriptor_shape(
                            &biases.v,
                            &[kv_width],
                            &format!("layer {idx} attention v bias"),
                        )?;
                    }
                }
                LlamaAttentionTensors::Mla { .. } => {
                    // DeepSeek MLA shape validation not yet implemented
                }
            }
            require_descriptor_matrix_shape(
                &layer.attention_output,
                q_width,
                dims.embedding_length,
                &format!("layer {idx} attention output"),
            )?;
            require_descriptor_shape(
                &layer.ffn_norm,
                &[dims.embedding_length],
                &format!("layer {idx} ffn norm"),
            )?;
            match &layer.ffn {
                LlamaFfnTensors::Dense { gate, up, down } => {
                    require_descriptor_matrix_shape(
                        gate,
                        dims.embedding_length,
                        dims.feed_forward_length,
                        &format!("layer {idx} ffn gate"),
                    )?;
                    require_descriptor_matrix_shape(
                        up,
                        dims.embedding_length,
                        dims.feed_forward_length,
                        &format!("layer {idx} ffn up"),
                    )?;
                    require_descriptor_matrix_shape(
                        down,
                        dims.feed_forward_length,
                        dims.embedding_length,
                        &format!("layer {idx} ffn down"),
                    )?;
                }
                LlamaFfnTensors::DeepSeekMoE { .. } => {
                    // DeepSeekMoE shape validation not yet implemented
                }
                LlamaFfnTensors::MoE {
                    router,
                    gate_experts,
                    up_experts,
                    down_experts,
                } => {
                    let moe = config.moe.as_ref().ok_or_else(|| {
                        BackendError::InvalidModelMetadata(
                            "MoE tensors were bound for a dense config".to_string(),
                        )
                    })?;
                    require_descriptor_matrix_shape(
                        router,
                        dims.embedding_length,
                        moe.expert_count as usize,
                        &format!("layer {idx} ffn router"),
                    )?;
                    validate_moe_expert_tensor_shape(
                        gate_experts,
                        dims.embedding_length,
                        dims.feed_forward_length,
                        moe.expert_count as usize,
                        &format!("layer {idx} ffn gate experts"),
                    )?;
                    validate_moe_expert_tensor_shape(
                        up_experts,
                        dims.embedding_length,
                        dims.feed_forward_length,
                        moe.expert_count as usize,
                        &format!("layer {idx} ffn up experts"),
                    )?;
                    validate_moe_expert_tensor_shape(
                        down_experts,
                        dims.feed_forward_length,
                        dims.embedding_length,
                        moe.expert_count as usize,
                        &format!("layer {idx} ffn down experts"),
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// Per-layer tensor descriptors for a Gemma 4 decoder block.
///
/// Captures everything the Gemma 4 forward pass needs beyond the Llama set: the
/// per-layer-type attention projections (their widths vary with the sliding/full
/// schedule), QK-norm, the extra Gemma norms (post-attention, post-FFN, and the
/// Gemma 4 per-layer `post_norm`), and — for the elastic "E" variants — the
/// Per-Layer-Embedding injection (`inp_gate`, `proj`, `layer_output_scale`).
/// Dense variants (12B/31B) carry no PLE tensors, so those are `None`.
#[derive(Debug, Clone)]
pub struct Gemma4LayerTensors {
    pub attn_norm: GgufTensorDescriptor,
    pub attn_q: GgufTensorDescriptor,
    /// `None` on shared-KV layers in exports that trim unused projections:
    /// the QAT GGUFs omit `attn_k`/`attn_v`/`attn_k_norm` on layers that source
    /// their cache from an earlier layer (the Q8_0 exports carry them unused).
    /// Owning layers (`idx < first_kv_shared`) must always bind them.
    pub attn_k: Option<GgufTensorDescriptor>,
    /// `None` on V-less layers: the 12B row's full-attention layers carry no
    /// `attn_v` tensor — the reference (llama.cpp `gemma4-iswa`) uses the K
    /// projection output as V (`if v_proj is not present, use Kcur as Vcur`),
    /// then applies the usual weightless V norm and no RoPE.
    pub attn_v: Option<GgufTensorDescriptor>,
    pub attn_output: GgufTensorDescriptor,
    pub attn_q_norm: GgufTensorDescriptor,
    pub attn_k_norm: Option<GgufTensorDescriptor>,
    pub post_attention_norm: GgufTensorDescriptor,
    pub ffn_norm: GgufTensorDescriptor,
    pub post_ffw_norm: GgufTensorDescriptor,
    pub post_norm: Option<GgufTensorDescriptor>,
    pub ffn_gate: GgufTensorDescriptor,
    pub ffn_up: GgufTensorDescriptor,
    pub ffn_down: GgufTensorDescriptor,
    pub ple_inp_gate: Option<GgufTensorDescriptor>,
    pub ple_proj: Option<GgufTensorDescriptor>,
    pub ple_output_scale: Option<GgufTensorDescriptor>,
    /// Gemma 4 A4B (26B) MoE tensors. Present together on every MoE layer
    /// (`ffn_gate_inp` is the presence marker); `None` on dense rows (E2B/E4B/12B).
    /// The dense `ffn_gate/up/down` above are the "shared expert" MLP branch;
    /// these are the sparse 128-expert branch that runs in parallel.
    pub moe: Option<Gemma4MoeLayerTensors>,
}

/// The sparse-expert tensors of one Gemma 4 A4B MoE layer.
#[derive(Debug, Clone)]
pub struct Gemma4MoeLayerTensors {
    /// Router projection `ffn_gate_inp` [n_embd, n_expert] (F32).
    pub gate_inp: GgufTensorDescriptor,
    /// Router input scale `ffn_gate_inp.scale` [n_embd] (F32, elementwise).
    pub gate_inp_scale: GgufTensorDescriptor,
    /// Fused per-expert gate‖up `ffn_gate_up_exps` [n_embd, 2*n_ff_exp, n_expert].
    pub gate_up_exps: GgufTensorDescriptor,
    /// Per-expert down `ffn_down_exps` [n_ff_exp, n_embd, n_expert].
    pub down_exps: GgufTensorDescriptor,
    /// Per-expert down scale `ffn_down_exps.scale` [n_expert] (F32, scalar/expert).
    pub down_exps_scale: GgufTensorDescriptor,
    /// `ffn_pre_norm_2` [n_embd]: pre-norm for the expert branch.
    pub pre_norm_2: GgufTensorDescriptor,
    /// `ffn_post_norm_1` [n_embd]: post-norm for the dense (shared-expert) branch.
    pub post_norm_1: GgufTensorDescriptor,
    /// `ffn_post_norm_2` [n_embd]: post-norm for the expert branch.
    pub post_norm_2: GgufTensorDescriptor,
}

/// Full Gemma 4 weight binding (the gemma4 counterpart to [`LlamaTensorBinding`]).
#[derive(Debug, Clone)]
pub struct Gemma4Binding {
    pub token_embedding: GgufTensorDescriptor,
    pub output_norm: GgufTensorDescriptor,
    pub output: GgufTensorDescriptor,
    pub output_is_tied_embedding: bool,
    pub rope_freqs: Option<GgufTensorDescriptor>,
    /// Per-Layer-Embedding tables (E-series only; `None` for dense variants).
    pub per_layer_token_embd: Option<GgufTensorDescriptor>,
    pub per_layer_model_proj: Option<GgufTensorDescriptor>,
    pub per_layer_proj_norm: Option<GgufTensorDescriptor>,
    pub layers: Vec<Gemma4LayerTensors>,
}

impl Gemma4Binding {
    /// Bind every Gemma 4 tensor by name and validate the per-layer-type shapes.
    pub fn bind(gguf: &GgufFile, config: &LlamaModelConfig) -> Result<Self> {
        let gemma4 = config.gemma4.as_ref().ok_or_else(|| {
            BackendError::InvalidModelMetadata(
                "Gemma4Binding requires the gemma4 architecture".into(),
            )
        })?;

        // Gemma 4 A4B (26B) MoE rows carry a router (`ffn_gate_inp`) and fused
        // expert tensors (`ffn_gate_up_exps`/`ffn_down_exps`) ALONGSIDE the dense
        // `ffn_gate/up/down` shared-expert MLP. We bind both branches below. The
        // legacy split-expert layout (`ffn_gate_exps`/`ffn_up_exps`) is a
        // different (Mixtral-style) packing we do not model — fail closed on it.
        if find_tensor(gguf, "blk.0.ffn_gate_exps.weight").is_some()
            && find_tensor(gguf, "blk.0.ffn_gate_up_exps.weight").is_none()
        {
            return Err(BackendError::UnsupportedModelArchitecture(
                "gemma4 MoE row: blocked — split-expert (ffn_gate_exps/ffn_up_exps) \
                 packing is not modeled; only the fused ffn_gate_up_exps layout is"
                    .into(),
            ));
        }
        if let Some(moe) = config.moe.as_ref() {
            if find_tensor(gguf, "blk.0.ffn_gate_up_exps.weight").is_none() {
                return Err(BackendError::UnsupportedModelArchitecture(format!(
                    "gemma4 MoE row (expert_count={}, expert_used_count={}): blocked — \
                     gemma4.expert_count is set but no fused ffn_gate_up_exps tensor is present",
                    moe.expert_count, moe.expert_used_count
                )));
            }
        }
        // A router (`ffn_gate_inp`) with no fused expert tensors is a malformed /
        // unmodeled MoE row — fail closed by name rather than surfacing a generic
        // missing-tensor error from the per-layer binding below.
        if find_tensor(gguf, "blk.0.ffn_gate_inp.weight").is_some()
            && find_tensor(gguf, "blk.0.ffn_gate_up_exps.weight").is_none()
        {
            return Err(BackendError::UnsupportedModelArchitecture(
                "gemma4 MoE row: blocked — blk.0.ffn_gate_inp router is present but no \
                 fused ffn_gate_up_exps experts; only the fused MoE layout is modeled"
                    .into(),
            ));
        }

        let token_embedding = required_tensor(gguf, "token_embd.weight")?;
        let output_norm = required_tensor(gguf, "output_norm.weight")?;
        let (output, output_is_tied_embedding) = match find_tensor(gguf, "output.weight") {
            Some(desc) => (desc.clone(), false),
            None => (token_embedding.clone(), true),
        };

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for layer_idx in 0..config.block_count {
            let req =
                |suffix: &str| required_tensor(gguf, &format!("blk.{layer_idx}.{suffix}.weight"));
            let opt = |suffix: &str| {
                find_tensor(gguf, &format!("blk.{layer_idx}.{suffix}.weight")).cloned()
            };
            layers.push(Gemma4LayerTensors {
                attn_norm: req("attn_norm")?,
                attn_q: req("attn_q")?,
                attn_k: opt("attn_k"),
                attn_v: opt("attn_v"),
                attn_output: req("attn_output")?,
                attn_q_norm: req("attn_q_norm")?,
                attn_k_norm: opt("attn_k_norm"),
                post_attention_norm: req("post_attention_norm")?,
                ffn_norm: req("ffn_norm")?,
                post_ffw_norm: req("post_ffw_norm")?,
                post_norm: opt("post_norm"),
                ffn_gate: req("ffn_gate")?,
                ffn_up: req("ffn_up")?,
                ffn_down: req("ffn_down")?,
                ple_inp_gate: opt("inp_gate"),
                ple_proj: opt("proj"),
                ple_output_scale: opt("layer_output_scale"),
                moe: match find_tensor(gguf, &format!("blk.{layer_idx}.ffn_gate_inp.weight")) {
                    Some(_) => Some(Gemma4MoeLayerTensors {
                        gate_inp: req("ffn_gate_inp")?,
                        gate_inp_scale: required_tensor(
                            gguf,
                            &format!("blk.{layer_idx}.ffn_gate_inp.scale"),
                        )?,
                        gate_up_exps: req("ffn_gate_up_exps")?,
                        down_exps: req("ffn_down_exps")?,
                        down_exps_scale: required_tensor(
                            gguf,
                            &format!("blk.{layer_idx}.ffn_down_exps.scale"),
                        )?,
                        // GGUF tensor names (not llama.cpp's logical names):
                        // pre_ffw_norm_2 / post_ffw_norm_1 / post_ffw_norm_2.
                        pre_norm_2: req("pre_ffw_norm_2")?,
                        post_norm_1: req("post_ffw_norm_1")?,
                        post_norm_2: req("post_ffw_norm_2")?,
                    }),
                    None => None,
                },
            });
        }

        let binding = Self {
            token_embedding,
            output_norm,
            output,
            output_is_tied_embedding,
            rope_freqs: find_tensor(gguf, "rope_freqs.weight").cloned(),
            per_layer_token_embd: find_tensor(gguf, "per_layer_token_embd.weight").cloned(),
            per_layer_model_proj: find_tensor(gguf, "per_layer_model_proj.weight").cloned(),
            per_layer_proj_norm: find_tensor(gguf, "per_layer_proj_norm.weight").cloned(),
            layers,
        };
        binding.validate(config, gemma4)?;
        Ok(binding)
    }

    /// `true` when this is an elastic "E" variant carrying a Per-Layer-Embedding
    /// stream (E2B/E4B); `false` for the dense 12B/31B.
    pub fn has_per_layer_embeddings(&self) -> bool {
        self.per_layer_token_embd.is_some() && self.layers.iter().all(|l| l.ple_proj.is_some())
    }

    fn validate(&self, config: &LlamaModelConfig, gemma4: &Gemma4Metadata) -> Result<()> {
        if self.layers.len() != config.block_count as usize {
            return Err(BackendError::InvalidModelMetadata(format!(
                "gemma4 block count {} does not match bound layer count {}",
                config.block_count,
                self.layers.len()
            )));
        }
        let emb = config.embedding_length as usize;
        let heads = config.attention_head_count as usize;
        for (idx, layer) in self.layers.iter().enumerate() {
            let head_dim = gemma4.head_dim_at(idx) as usize;
            let kv_heads = gemma4.kv_heads_at(idx) as usize;
            let ffn_len = gemma4.ffn_length_at(idx) as usize;
            require_descriptor_matrix_shape(
                &layer.attn_q,
                emb,
                heads * head_dim,
                &format!("gemma4 layer {idx} attention q"),
            )?;
            let first_kv_shared =
                config.block_count as usize - gemma4.num_kv_shared_layers as usize;
            match layer.attn_k.as_ref() {
                Some(attn_k) => require_descriptor_matrix_shape(
                    attn_k,
                    emb,
                    kv_heads * head_dim,
                    &format!("gemma4 layer {idx} attention k"),
                )?,
                None if idx < first_kv_shared => {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "gemma4 layer {idx} owns its KV cache but binds no attn_k tensor                          (only shared-KV layers may omit K/V projections)"
                    )))
                }
                // Shared-KV layers source K/V from an earlier layer; trimmed
                // exports (QAT) legitimately omit the unused projections.
                None => {}
            }
            if layer.attn_k_norm.is_none() && idx < first_kv_shared {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 layer {idx} owns its KV cache but binds no attn_k_norm tensor"
                )));
            }
            // V-less layers (12B full-attention) reuse the K projection as V;
            // when the tensor exists it must match the K geometry.
            if let Some(attn_v) = layer.attn_v.as_ref() {
                require_descriptor_matrix_shape(
                    attn_v,
                    emb,
                    kv_heads * head_dim,
                    &format!("gemma4 layer {idx} attention v"),
                )?;
            }
            require_descriptor_matrix_shape(
                &layer.attn_output,
                heads * head_dim,
                emb,
                &format!("gemma4 layer {idx} attention output"),
            )?;
            require_descriptor_shape(
                &layer.attn_q_norm,
                &[head_dim],
                &format!("gemma4 layer {idx} q_norm"),
            )?;
            if let Some(attn_k_norm) = layer.attn_k_norm.as_ref() {
                require_descriptor_shape(
                    attn_k_norm,
                    &[head_dim],
                    &format!("gemma4 layer {idx} k_norm"),
                )?;
            }
            require_descriptor_shape(
                &layer.attn_norm,
                &[emb],
                &format!("gemma4 layer {idx} attn_norm"),
            )?;
            require_descriptor_matrix_shape(
                &layer.ffn_gate,
                emb,
                ffn_len,
                &format!("gemma4 layer {idx} ffn gate"),
            )?;
            require_descriptor_matrix_shape(
                &layer.ffn_up,
                emb,
                ffn_len,
                &format!("gemma4 layer {idx} ffn up"),
            )?;
            require_descriptor_matrix_shape(
                &layer.ffn_down,
                ffn_len,
                emb,
                &format!("gemma4 layer {idx} ffn down"),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseLlamaDims {
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub attention_head_count_kv: usize,
    pub head_dim: usize,
    /// Query projection width = `attention_head_count * head_dim`. Equals
    /// `embedding_length` only when `head_dim == embedding_length/head_count`;
    /// Qwen3 0.6B/4B/32B set an explicit larger head_dim so `q_width > embedding`.
    pub q_width: usize,
    pub kv_width: usize,
    pub vocab_size: usize,
}

impl DenseLlamaDims {
    pub(crate) fn from_config(config: &LlamaModelConfig) -> Result<Self> {
        let embedding_length = config.embedding_length as usize;
        let attention_head_count = config.attention_head_count as usize;
        if attention_head_count == 0 || !embedding_length.is_multiple_of(attention_head_count) {
            return Err(BackendError::InvalidModelMetadata(format!(
                "embedding length {embedding_length} is not divisible by attention head count {attention_head_count}"
            )));
        }

        let attention_head_count_kv = config.attention_head_count_kv as usize;
        if attention_head_count_kv == 0 {
            return Err(BackendError::InvalidModelMetadata(
                "attention kv head count must be greater than zero".to_string(),
            ));
        }
        if !attention_head_count.is_multiple_of(attention_head_count_kv) {
            return Err(BackendError::InvalidModelMetadata(format!(
                "attention head count {attention_head_count} must be a multiple of kv head count {attention_head_count_kv}"
            )));
        }

        let vocab_size = config.vocab_size.ok_or_else(|| {
            BackendError::InvalidModelMetadata(
                "required metadata llama.vocab_size is missing for dense tensor validation"
                    .to_string(),
            )
        })? as usize;
        if vocab_size == 0 {
            return Err(BackendError::InvalidModelMetadata(
                "llama.vocab_size must be greater than zero".to_string(),
            ));
        }

        // Prefer the GGUF's explicit per-head dim (`attention.key_length`) when
        // present — Qwen3 0.6B/4B/32B set head_dim != embedding/head_count. Fall
        // back to embedding/head_count for rows that don't carry it.
        let head_dim = match config.attention_key_length {
            Some(key_length) if key_length > 0 => key_length as usize,
            _ => embedding_length / attention_head_count,
        };
        Ok(Self {
            embedding_length,
            block_count: config.block_count as usize,
            feed_forward_length: config.feed_forward_length as usize,
            attention_head_count_kv,
            head_dim,
            q_width: attention_head_count * head_dim,
            kv_width: attention_head_count_kv * head_dim,
            vocab_size,
        })
    }
}

fn bind_moe_expert_tensors(
    gguf: &GgufFile,
    layer_idx: u32,
    role: &str,
    expert_count: u32,
) -> Result<LlamaMoeExpertTensors> {
    let merged_name = format!("blk.{layer_idx}.ffn_{role}_exps.weight");
    if let Some(desc) = find_tensor(gguf, &merged_name) {
        return Ok(LlamaMoeExpertTensors::Merged(desc.clone()));
    }

    let mut split = Vec::with_capacity(expert_count as usize);
    for expert_idx in 0..expert_count {
        split.push(required_tensor(
            gguf,
            &format!("blk.{layer_idx}.ffn_{role}.{expert_idx}.weight"),
        )?);
    }
    Ok(LlamaMoeExpertTensors::Split(split))
}

fn validate_moe_expert_tensor_shape(
    experts: &LlamaMoeExpertTensors,
    input_width: usize,
    output_width: usize,
    expert_count: usize,
    role: &str,
) -> Result<()> {
    match experts {
        LlamaMoeExpertTensors::Merged(desc) => {
            require_descriptor_shape(desc, &[input_width, output_width, expert_count], role)
        }
        LlamaMoeExpertTensors::Split(descs) => {
            if descs.len() != expert_count {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "{role} expected {expert_count} split expert tensors, got {}",
                    descs.len()
                )));
            }
            for (expert_idx, desc) in descs.iter().enumerate() {
                require_descriptor_shape(
                    desc,
                    &[input_width, output_width],
                    &format!("{role} split expert {expert_idx}"),
                )?;
            }
            Ok(())
        }
    }
}

fn require_descriptor_shape(
    tensor: &GgufTensorDescriptor,
    expected: &[usize],
    role: &str,
) -> Result<()> {
    let actual = descriptor_dims(tensor)?;
    if actual != expected {
        return Err(BackendError::InvalidModelMetadata(format!(
            "{role} tensor {} expected descriptor shape {:?}, got {:?}",
            tensor.name, expected, actual
        )));
    }
    Ok(())
}

fn require_descriptor_matrix_shape(
    tensor: &GgufTensorDescriptor,
    input_width: usize,
    output_width: usize,
    role: &str,
) -> Result<()> {
    let actual = descriptor_dims(tensor)?;
    let direct = [input_width, output_width];
    let transposed = [output_width, input_width];
    if actual.as_slice() != direct && actual.as_slice() != transposed {
        return Err(BackendError::InvalidModelMetadata(format!(
            "{role} tensor {} expected descriptor shape {:?} or {:?}, got {:?}",
            tensor.name, direct, transposed, actual
        )));
    }
    Ok(())
}

fn validate_output_projection_storage_layout(
    tensor: &GgufTensorDescriptor,
    hidden_width: usize,
    vocab_size: usize,
) -> Result<()> {
    let actual = descriptor_dims(tensor)?;
    let (row_values, row_count, layout) = match actual.as_slice() {
        [hidden, vocab] if *hidden == hidden_width && *vocab == vocab_size => {
            (*hidden, *vocab, "gguf_hidden_vocab_token_rows")
        }
        [vocab, hidden] if *hidden == hidden_width && *vocab == vocab_size => {
            (*hidden, *vocab, "output_input_token_rows")
        }
        _ => return Ok(()),
    };

    let (block_size, type_size_bytes) = tensor.tensor_type.layout().ok_or_else(|| {
        BackendError::InvalidModelMetadata(format!(
            "output projection tensor {} has unsupported storage type {:?} for token-row validation",
            tensor.name, tensor.tensor_type
        ))
    })?;
    let row_values = u64::try_from(row_values).map_err(|_| {
        BackendError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row width {row_values} does not fit u64",
            tensor.name
        ))
    })?;
    let row_count = u64::try_from(row_count).map_err(|_| {
        BackendError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row count {row_count} does not fit u64",
            tensor.name
        ))
    })?;
    if !row_values.is_multiple_of(block_size) {
        return Err(BackendError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row width {row_values} is not divisible by {:?} block size {block_size}",
            tensor.name, tensor.tensor_type
        )));
    }

    let row_size_bytes = row_values
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size_bytes))
        .ok_or_else(|| {
            BackendError::InvalidModelMetadata(format!(
                "output projection tensor {} token-row byte size overflow",
                tensor.name
            ))
        })?;
    let row_stride_bytes = row_size_bytes;
    let expected_bytes = row_stride_bytes.checked_mul(row_count).ok_or_else(|| {
        BackendError::InvalidModelMetadata(format!(
            "output projection tensor {} token-row byte count overflow",
            tensor.name
        ))
    })?;

    if tensor.n_bytes != expected_bytes {
        return Err(BackendError::InvalidModelMetadata(format!(
            "output projection tensor {} token-major storage validation failed for {layout}: row_values={row_values}, row_count={row_count}, row_size_bytes={row_size_bytes}, row_stride_bytes={row_stride_bytes}, expected_n_bytes={expected_bytes}, actual_n_bytes={}",
            tensor.name, tensor.n_bytes
        )));
    }

    Ok(())
}

fn descriptor_dims(tensor: &GgufTensorDescriptor) -> Result<Vec<usize>> {
    tensor
        .dimensions
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                BackendError::InvalidModelMetadata(format!(
                    "tensor {} dimension {dim} does not fit usize",
                    tensor.name
                ))
            })
        })
        .collect()
}

fn required_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.metadata_u32(key).ok_or_else(|| {
        BackendError::InvalidModelMetadata(format!("required metadata {key} is missing or not u32"))
    })
}

fn infer_vocab_size_from_token_embedding(
    gguf: &GgufFile,
    tensor_name: &str,
    embedding_length: u32,
) -> Option<u32> {
    let embedding_length = u64::from(embedding_length);
    let tensor = find_tensor(gguf, tensor_name)?;
    if tensor.dimensions.len() != 2 {
        return None;
    }
    let dims = tensor.dimensions.as_slice();
    let inferred = if dims[0] == embedding_length {
        dims[1]
    } else if dims[1] == embedding_length {
        dims[0]
    } else {
        return None;
    };
    inferred.try_into().ok()
}

fn required_tensor(gguf: &GgufFile, name: &str) -> Result<GgufTensorDescriptor> {
    find_tensor(gguf, name)
        .cloned()
        .ok_or_else(|| BackendError::TensorNotFound(name.to_string()))
}

fn find_tensor<'a>(gguf: &'a GgufFile, name: &str) -> Option<&'a GgufTensorDescriptor> {
    gguf.tensors.iter().find(|tensor| tensor.name == name)
}

/// A descriptor for a contiguous slice of `parent`'s **output rows** (`dimensions[1]`),
/// e.g. the Q slice of a fused `attn_qkv` weight. Quantized weights store each output
/// row as an integer number of blocks along the input dim, so a row range maps to an
/// exact byte range — no re-encoding, just a sub-offset into the same file bytes.
fn sub_row_descriptor(
    parent: &GgufTensorDescriptor,
    name: String,
    row_start: u64,
    row_count: u64,
) -> Result<GgufTensorDescriptor> {
    if parent.dimensions.len() != 2 {
        return Err(BackendError::InvalidModelMetadata(format!(
            "cannot slice non-2D tensor {} (dims {:?})",
            parent.name, parent.dimensions
        )));
    }
    let in_dim = parent.dimensions[0];
    let total_rows = parent.dimensions[1];
    let (block, type_size) = parent.tensor_type.layout().ok_or_else(|| {
        BackendError::UnsupportedTensorType(format!(
            "tensor {} type {:?} has no known block layout; cannot slice a fused tensor",
            parent.name, parent.tensor_type
        ))
    })?;
    if block == 0 || !in_dim.is_multiple_of(block) {
        return Err(BackendError::InvalidModelMetadata(format!(
            "tensor {} input dim {in_dim} is not a multiple of block size {block}; cannot slice",
            parent.name
        )));
    }
    if row_start + row_count > total_rows {
        return Err(BackendError::InvalidModelMetadata(format!(
            "slice [{row_start}..{}] of {} exceeds its {total_rows} rows",
            row_start + row_count,
            parent.name
        )));
    }
    let row_bytes = (in_dim / block) * type_size;
    let byte_start = row_start * row_bytes;
    let n_bytes = row_count * row_bytes;
    Ok(GgufTensorDescriptor {
        name,
        dimensions: vec![in_dim, row_count],
        tensor_type: parent.tensor_type,
        relative_offset: parent.relative_offset + byte_start,
        absolute_offset: parent.absolute_offset + byte_start,
        n_bytes,
    })
}

/// Expand a dense decoder's **fused** projections into the split tensors the binder
/// and forward path expect. Some conversions (notably `phi3`) ship a single
/// `attn_qkv` (Q‖K‖V stacked by output row) and a single `ffn_up` carrying the
/// gate‖up halves, instead of separate `attn_q/attn_k/attn_v` and `ffn_gate/ffn_up`.
///
/// Rather than special-case the engine, we synthesize name-addressable
/// `GgufTensorDescriptor`s that point at the exact byte sub-ranges of the fused
/// tensors (legal because quantized output rows are block-aligned), then append them
/// to `gguf.tensors`. Everything downstream (`bind`, `TensorStore`, the forward
/// path) resolves tensors by name from this list, so the split rows flow through the
/// **unchanged** code path — a model with genuinely-separate tensors is byte-for-byte
/// unaffected (this is a no-op unless a fused tensor is present and a split one is
/// absent). It makes no parity claim; it only lets the fused layout be attempted.
pub fn expand_fused_dense_tensors(gguf: &mut GgufFile, config: &LlamaModelConfig) -> Result<()> {
    // MoE rows carry their own (already-split) expert tensors; leave them alone.
    if config.moe.is_some() {
        return Ok(());
    }
    let head_count = config.attention_head_count.max(1);
    let head_dim = config
        .attention_key_length
        .unwrap_or(config.embedding_length / head_count);
    let q_rows = u64::from(head_dim) * u64::from(config.attention_head_count);
    let kv_rows = u64::from(head_dim) * u64::from(config.attention_head_count_kv);
    let ffn = u64::from(config.feed_forward_length);

    let mut additions: Vec<GgufTensorDescriptor> = Vec::new();
    let mut renames: Vec<(usize, String)> = Vec::new();

    for layer in 0..config.block_count {
        // Fused attention QKV → attn_q / attn_k / attn_v (no rename: distinct names).
        let q_name = format!("blk.{layer}.attn_q.weight");
        if find_tensor(gguf, &q_name).is_none() {
            if let Some(qkv) = find_tensor(gguf, &format!("blk.{layer}.attn_qkv.weight")) {
                if qkv.dimensions.len() == 2 && qkv.dimensions[1] == q_rows + 2 * kv_rows {
                    let qkv = qkv.clone();
                    additions.push(sub_row_descriptor(&qkv, q_name, 0, q_rows)?);
                    additions.push(sub_row_descriptor(
                        &qkv,
                        format!("blk.{layer}.attn_k.weight"),
                        q_rows,
                        kv_rows,
                    )?);
                    additions.push(sub_row_descriptor(
                        &qkv,
                        format!("blk.{layer}.attn_v.weight"),
                        q_rows + kv_rows,
                        kv_rows,
                    )?);
                }
            }
        }

        // Fused gate+up → ffn_gate (first half) + ffn_up (second half). The fused
        // tensor reuses the `ffn_up` name, so rename it before re-adding a half-sized
        // `ffn_up` virtual to avoid an ambiguous duplicate name.
        let gate_name = format!("blk.{layer}.ffn_gate.weight");
        let up_name = format!("blk.{layer}.ffn_up.weight");
        if find_tensor(gguf, &gate_name).is_none() {
            if let Some((idx, up)) = gguf
                .tensors
                .iter()
                .enumerate()
                .find(|(_, t)| t.name == up_name)
            {
                if up.dimensions.len() == 2 && up.dimensions[1] == 2 * ffn {
                    let up = up.clone();
                    renames.push((idx, format!("blk.{layer}.ffn_up_fused.weight")));
                    additions.push(sub_row_descriptor(&up, gate_name, 0, ffn)?);
                    additions.push(sub_row_descriptor(&up, up_name, ffn, ffn)?);
                }
            }
        }
    }

    for (idx, new_name) in renames {
        gguf.tensors[idx].name = new_name;
    }
    gguf.tensors.extend(additions);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_output_projection_storage_layout;
    use crate::gguf::{GgufTensorDescriptor, GgufTensorType};

    #[test]
    fn neox_rope_pairing_covers_exactly_the_proven_archs() {
        // qwen2/qwen3/qwen35 verified vs real rows; phi3 proven during
        // MUSTER M-A2 (adjacent even/odd degenerates long generation on the
        // exact Phi-3-mini-4k row, split-half restores coherence). Everything
        // else — including the other unpermuted-but-unproven archs — must stay
        // on adjacent even/odd until its own row proves the flip.
        assert!(super::arch_uses_neox_rope_pairing("qwen3"));
        assert!(super::arch_uses_neox_rope_pairing("qwen35"));
        assert!(super::arch_uses_neox_rope_pairing("phi3"));
        assert!(super::arch_uses_neox_rope_pairing("qwen2"));
        assert!(!super::arch_uses_neox_rope_pairing("llama"));
        assert!(!super::arch_uses_neox_rope_pairing("mistral"));
        assert!(!super::arch_uses_neox_rope_pairing("gemma3"));
        assert!(!super::arch_uses_neox_rope_pairing("gemma4"));
    }

    #[test]
    fn implemented_architecture_set_is_exactly_the_from_gguf_accept_arm() {
        // These nine must stay byte-for-byte in sync with the match arm in
        // LlamaModelConfig::from_gguf. If you change one, change both.
        for arch in [
            "llama", "mistral", "qwen2", "qwen3", "smollm3", "gemma3", "gemma4", "phi3", "lfm2",
        ] {
            assert!(
                super::is_implemented_architecture(arch),
                "{arch} should be implemented"
            );
        }
        for arch in [
            "falcon",
            "gpt2",
            "mamba",
            "bert",
            "rwkv",
            "diffusion-gemma",
            "",
        ] {
            assert!(
                !super::is_implemented_architecture(arch),
                "{arch} must not be implemented"
            );
        }
    }

    #[test]
    fn sub_row_descriptor_slices_q8_0_by_output_row() {
        // in=64 → Q8_0 row = (64/32)*34 = 68 bytes; out=6 rows.
        let parent = GgufTensorDescriptor {
            name: "blk.0.attn_qkv.weight".into(),
            dimensions: vec![64, 6],
            tensor_type: GgufTensorType::Q8_0,
            relative_offset: 0,
            absolute_offset: 1000,
            n_bytes: 6 * 68,
        };
        let q = super::sub_row_descriptor(&parent, "q".into(), 0, 2).unwrap();
        assert_eq!(q.dimensions, vec![64, 2]);
        assert_eq!(q.absolute_offset, 1000);
        assert_eq!(q.n_bytes, 2 * 68);
        let k = super::sub_row_descriptor(&parent, "k".into(), 2, 4).unwrap();
        assert_eq!(k.absolute_offset, 1000 + 2 * 68);
        assert_eq!(k.n_bytes, 4 * 68);
        // The slices tile the parent exactly (no gap, no overlap).
        assert_eq!(q.n_bytes + k.n_bytes, parent.n_bytes);
        // Out-of-range fails closed rather than reading past the tensor.
        assert!(super::sub_row_descriptor(&parent, "x".into(), 4, 4).is_err());
    }

    #[test]
    fn validates_q8_output_projection_token_row_storage_math() {
        let desc = output_desc(vec![2048, 32_000], 69_632_000);

        validate_output_projection_storage_layout(&desc, 2048, 32_000).unwrap();
    }

    #[test]
    fn validates_q8_output_input_token_row_storage_math() {
        let desc = output_desc(vec![32_000, 2048], 69_632_000);

        validate_output_projection_storage_layout(&desc, 2048, 32_000).unwrap();
    }

    #[test]
    fn validates_f16_output_projection_token_row_storage_math() {
        let desc = GgufTensorDescriptor {
            tensor_type: GgufTensorType::F16,
            ..output_desc(vec![2048, 32_000], 131_072_000)
        };

        validate_output_projection_storage_layout(&desc, 2048, 32_000).unwrap();
    }

    #[test]
    fn rejects_q8_output_projection_token_row_nbytes_mismatch() {
        let desc = output_desc(vec![2048, 32_000], 69_632_034);

        let err = validate_output_projection_storage_layout(&desc, 2048, 32_000)
            .unwrap_err()
            .to_string();

        assert!(err.contains("output.weight"));
        assert!(err.contains("row_values=2048"));
        assert!(err.contains("row_count=32000"));
        assert!(err.contains("row_size_bytes=2176"));
        assert!(err.contains("row_stride_bytes=2176"));
        assert!(err.contains("expected_n_bytes=69632000"));
        assert!(err.contains("actual_n_bytes=69632034"));
    }

    #[test]
    fn rejects_q8_output_projection_token_rows_that_do_not_fill_blocks() {
        let desc = output_desc(vec![2032, 32_000], 69_088_000);

        let err = validate_output_projection_storage_layout(&desc, 2032, 32_000)
            .unwrap_err()
            .to_string();

        assert!(err.contains("token-row width 2032"));
        assert!(err.contains("block size 32"));
    }

    fn output_desc(dimensions: Vec<u64>, n_bytes: u64) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: "output.weight".to_string(),
            dimensions,
            tensor_type: GgufTensorType::Q8_0,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }
}
