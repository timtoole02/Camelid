use serde::Serialize;

pub use crate::tensor::kv_quant::KvCacheQuantization;

use crate::{
    gguf::{GgufFile, GgufTensorDescriptor},
    BackendError, Result,
};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LlamaModelConfig {
    /// The GGUF `general.architecture` this config was built from (e.g. "llama",
    /// "qwen3", "gemma3"). Carried so downstream engine gates can key on the
    /// architecture itself instead of inferring it from config shape — the
    /// resident-decode eligibility check uses it to fail closed for architectures
    /// whose only correct forward lives in the runnable lane (see
    /// [`is_runnable_only_arch`] and `resident_decode_eligible`). Synthetic
    /// configs (benches, tests, the SafeTensors summary) set the family they
    /// emulate ("llama").
    pub architecture: String,
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
    /// NoPE (no-positional-encoding) layer step. When `Some(step)` with `step > 0`,
    /// decoder layer `il` (0-based) SKIPS RoPE on BOTH Q and K whenever
    /// `(il + 1) % step == 0`; every other layer ropes normally. `None` means every
    /// layer is roped, which is the case for every admitted architecture except
    /// `smollm3` (each verified against its llama.cpp graph builder).
    ///
    /// Hardcoded per architecture because llama.cpp hardcodes it too:
    /// `hparams.n_no_rope_layer_step` has NO GGUF key backing it — llama.cpp
    /// `src/models/smollm3.cpp:5` assigns the literal `4` in `load_arch_hparams`,
    /// `src/llama-hparams.h:203` is only the struct default, and
    /// `tests/test-llama-archs.cpp:93` records it as "hard-coded to 4". There is no
    /// `LLM_KV_*` enum and no writer, so it cannot be inferred from the file.
    ///
    /// (HuggingFace's `SmolLM3Config` does carry a `no_rope_layers` array, but
    /// llama.cpp ignores it. The parity target here is llama.cpp, so hardcoding 4
    /// is the only parity-faithful choice — disclosed rather than hidden.)
    ///
    /// See [`LlamaModelConfig::layer_uses_rope`].
    pub no_rope_layer_step: Option<u32>,
    /// Logit scale — applied before softmax, commonly in Command R models.
    pub logit_scale: Option<f32>,
    pub moe: Option<MixtralMoeMetadata>,
    /// Gemma 3 (`general.architecture = "gemma3"`) specific metadata. `None` for
    /// every other architecture. Holds the sliding-window size, the local:global
    /// layer cadence, the dual RoPE bases, and the structural forward-pass facts
    /// (GeGLU, sqrt(d_model) embed scale, forced split-half RoPE pairing) that a
    /// Llama-shaped config cannot express. Parsed fail-closed — see
    /// [`Gemma3Metadata::from_gguf`]. Consumed by the Metal-resident lane
    /// (Phase 2/3b of the gemma3 Metal campaign) and by
    /// [`arch_has_windowed_attention`], which keys every windowed-arch guard
    /// (prefix-cache bypass, single-token prefill, the CPU dense fail-closed).
    pub gemma3: Option<Gemma3Metadata>,
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
    /// LFM2 / LFM2.5 (`general.architecture = "lfm2"`) hybrid short-convolution
    /// metadata. `None` for every other architecture. Holds the short-conv
    /// kernel width and the per-layer conv/attention schedule that a dense
    /// Llama config cannot express. See [`Lfm2Metadata`].
    pub lfm2: Option<Lfm2Metadata>,
    /// Multi-Head Latent Attention (MLA) metadata for DeepSeek models.
    pub mla: Option<MlaMetadata>,
}

/// Whether `architecture` is one of the dense-decoder families Camelid actually
/// implements. This MUST mirror the accepted set in [`LlamaModelConfig::from_gguf`]
/// below (the `unit test implemented_set_matches_from_gguf_accept_arm` guards the
/// two against drift). It is a pure architecture-string check for classification
/// (e.g. labeling a loaded model's lane); it makes NO support/parity claim — an
/// implemented architecture is only *attemptable*, never automatically supported.
///
/// NOTE on `smollm3`: it is a **NoPE** architecture — every 4th layer (0-based
/// `il` with `(il + 1) % 4 == 0`) skips RoPE on Q and K, per llama.cpp
/// `src/models/smollm3.cpp:5,69`. Before that schedule was implemented this
/// function claimed `smollm3` while the engine roped every layer, which is
/// silently wrong output rather than a clean refusal. The skip now matches the
/// reference graph on the **CPU path only**; the resident GPU engines fail closed
/// to CPU for NoPE models (see `resident_decode_eligible`).
///
/// EVIDENCE OWED: no SmolLM3 GGUF has been run against the pinned llama.cpp
/// reference, so there is no greedy-parity receipt. Unproven specifically:
/// end-to-end token identity on real weights, tokenizer/chat-template fidelity,
/// the interaction of NoPE layers with SmolLM3's long-context rope scaling, and
/// every GPU lane (deliberately refused). The correct claim today is "smollm3 is
/// attemptable and its NoPE schedule matches the reference graph" — NOT that
/// smollm3 is supported. No ledger row claims it, and none should be added until
/// a receipt exists.
///
/// NOTE on `lfm2` (and `qwen35`): these are NOT dense decoders. LFM2 interleaves
/// double-gated short-convolution blocks with GQA attention, so its only correct
/// forward lives in the runnable lane ([`is_runnable_only_arch`]); it is listed
/// here because `from_gguf` must still parse it to build the config that lane
/// consumes. Unlike smollm3, lfm2 DOES have a greedy-parity receipt — token
/// identity vs llama.cpp b9632 on the LFM2.5-2.6B Q8_0 row
/// (`tests/lfm2_parity.rs`, `qa/runnable/lfm2-parity.json`). That receipt proves
/// the forward GRAPH only; tokenizer parity and chat-template fidelity are
/// separate gates and no row should claim them off it.
pub fn is_implemented_architecture(architecture: &str) -> bool {
    matches!(
        architecture,
        "llama"
            | "mistral"
            | "qwen2"
            | "qwen3"
            | "qwen3moe"
            | "qwen35"
            | "smollm3"
            | "gemma2"
            | "gemma3"
            | "gemma4"
            | "phi3"
            | "command-r"
            | "lfm2"
            | "bitnet-b1.58"
            | "mobilemoe"
    )
}

/// Exact official Microsoft BitNet embedding GGUFs. They intentionally reuse
/// qwen3/gemma3 architecture identifiers while carrying an embedding-only
/// projection-norm graph and no language-model output tensor.
pub fn is_bitnet_embedding_model(gguf: &GgufFile) -> bool {
    matches!(
        (gguf.architecture(), gguf.model_name()),
        (Some("qwen3"), Some("bitnet-embeddings-0.6b"))
            | (Some("gemma3"), Some("bitnet-embeddings-270m"))
    )
}

/// Architectures whose ONLY correct forward pass lives in the runnable lane
/// (`crate::runnable`), on EVERY host, because no optimized lane can run them
/// correctly. gemma2: the binder still silently drops its sandwich norms, and
/// the dense forward lacks its soft-caps and alternating-attention schedule.
/// qwen35: the hybrid gated-delta-net layers do not fit the dense tensor map
/// at all. lfm2: same shape — its short-conv layers carry
/// `shortconv.{conv,in_proj,out_proj}` and no `attn_q/k/v`, so the dense
/// binding cannot bind them (BACKEND_ASKS RA-6b). Every lane that would
/// construct a direct dense session for these archs must fail closed instead
/// of decoding fluent-looking garbage.
/// bitnet-b1.58: its I2_S projections and attention/FFN SubLN tensors exist
/// only in the runnable graph; the optimized binder would drop both SubLNs.
///
/// gemma3 LEFT this set in Phase 3b of the Metal campaign: the Metal-resident
/// forward carries its full structure (QK + sandwich norms, GeGLU, dual-theta
/// RoPE, sliding-window mask — Phase 2, real-row parity §9b/§10b), so on a
/// host where that lane can serve, gemma3 chat runs dense/resident. Routing
/// for gemma3 is therefore CAPABILITY-AWARE — see
/// [`arch_requires_runnable_bridge`], the predicate serve and the CLI direct
/// lanes now key on. The CPU dense forward remains WRONG for gemma3 (no
/// window mask; hazard H4) and fails closed at forward dispatch.
pub fn is_runnable_only_arch(architecture: &str) -> bool {
    matches!(
        architecture,
        "qwen35" | "gemma2" | "command-r" | "lfm2" | "bitnet-b1.58"
    )
}

/// Capability-aware serve/direct-session routing predicate (gemma3→Metal
/// Phase 3b, quant-aware since Phase 3c): true when this MODEL FILE, ON THIS
/// HOST, must be served through the runnable bridge because no optimized lane
/// can run it correctly here.
///
/// The fallback is the runnable bridge, NEVER the CPU dense forward: the CPU
/// dense forward has no sliding-window mask and fails closed for windowed
/// archs at every per-layer dispatch (hazard H4), so a routing mistake
/// surfaces as a typed error instead of fluent-looking full-causal garbage.
///
/// - qwen35 / gemma2 / lfm2: always true ([`is_runnable_only_arch`]).
/// - a Q8_0 gemma3: true only where the Metal-resident lane cannot serve it
///   (non-macOS hosts, `CAMELID_METAL_RESIDENT_DECODE=0` / deterministic
///   mode, no Metal device, or a CUDA-resident process — the CUDA engine has
///   no windowed forward). On a resident-capable host it routes to the
///   dense/resident path and this returns false.
/// - a non-Q8_0 gemma3 (Q4_K_M, Q5_K, …): true on EVERY host. Resident
///   admission is pinned to the Q8_0 exact row in the mechanism (hazard H5 —
///   the K-quant resident gather drops gemma3's embed scale and no windowed
///   K-quant lane has a parity receipt), so a K-quant gemma3 has no resident
///   lane to fall onto anywhere and must take the bridge that served it
///   before the flip.
///
/// This is keyed on the FILE, not the arch string, precisely because the
/// quantization is half the decision (Phase 3c finding F3: an arch-only
/// predicate stranded every non-Q8_0 gemma3 — routed to the dense lane,
/// declined by H5, then H4-errored on every request).
pub fn file_requires_runnable_bridge(gguf: &GgufFile) -> bool {
    let architecture = gguf.architecture().unwrap_or_default();
    arch_requires_runnable_bridge_given(
        architecture,
        crate::inference::windowed_arch_resident_host_available(),
        windowed_arch_resident_quant_admissible(gguf),
    )
}

/// Pure decision core of [`file_requires_runnable_bridge`], split so the
/// routing split is unit-testable without touching process env, a Metal
/// device, or a GGUF on disk.
///
/// - `windowed_resident_host_available`: the host-capability probe result
///   ([`crate::inference::windowed_arch_resident_host_available`]).
/// - `windowed_resident_quant_admissible`: whether the FILE's weights satisfy
///   the H5 Q8_0 pin ([`windowed_arch_resident_quant_admissible`]).
///
/// A windowed arch needs BOTH to take the resident lane; failing either sends
/// it to the runnable bridge, which is correct (slow) for every gemma3 quant.
pub fn arch_requires_runnable_bridge_given(
    architecture: &str,
    windowed_resident_host_available: bool,
    windowed_resident_quant_admissible: bool,
) -> bool {
    is_runnable_only_arch(architecture)
        || (arch_string_has_windowed_attention(architecture)
            && !(windowed_resident_host_available && windowed_resident_quant_admissible))
}

/// Arch-string mirror of [`arch_has_windowed_attention`] for the surfaces that
/// run BEFORE the config is parsed (routing, the planner, the CLI admission
/// guards). gemma3 is the only windowed arch today; a future one must be added
/// here in lockstep with the parsed-metadata predicate.
pub fn arch_string_has_windowed_attention(architecture: &str) -> bool {
    architecture == "gemma3"
}

/// Whether this file's weights satisfy the windowed-arch resident admission
/// pin (hazard H5), decided from GGUF metadata BEFORE any weights load.
///
/// Mirrors the engine-level pin in `inference::resident_decode_eligible`
/// exactly: every per-layer linear (Q/K/V/O + FFN gate/up/down) must be Q8_0.
/// The engine remains authoritative — this is the routing-time predicate that
/// keeps a file the engine will decline from being routed onto the dense lane
/// in the first place. Non-windowed archs are unaffected (always `true`); the
/// caller pairs this with an arch check.
pub fn windowed_arch_resident_quant_admissible(gguf: &GgufFile) -> bool {
    use crate::gguf::GgufTensorType;
    const LAYER_LINEAR_SUFFIXES: [&str; 7] = [
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
    ];
    let mut saw_layer_linear = false;
    for tensor in &gguf.tensors {
        if !tensor.name.starts_with("blk.") {
            continue;
        }
        if !LAYER_LINEAR_SUFFIXES
            .iter()
            .any(|suffix| tensor.name.ends_with(suffix))
        {
            continue;
        }
        saw_layer_linear = true;
        if tensor.tensor_type != GgufTensorType::Q8_0 {
            return false;
        }
    }
    // A file with no recognizable per-layer linears is not something the
    // resident lane can admit either; fail closed to the bridge.
    saw_layer_linear
}

/// Whether this model's attention carries a per-layer sliding-window schedule
/// (gemma3 today). Keyed on the PARSED metadata (`config.gemma3`), not the
/// arch string, so a future windowed arch that binds the same schedule
/// inherits every guard that consults this.
///
/// Why the guards exist (GEMMA3_METAL_CONDUCTOR.md §9e-2, hazard H1/H2): the
/// CPU dense prefill lanes and the prompt-prefix-cache resume path have no
/// sliding-window mask at all. A windowed arch must therefore never store a
/// prompt-prefix-cache entry, never take a partial-hit resume (the divergent
/// suffix would be re-prefilled at `kv_position > 0`, which the resident
/// prefill hook refuses, landing it on the window-less CPU dense forward),
/// and must prefill token-by-token through the single-token decode path — the
/// only lane whose resident forward carries the schedule.
pub fn arch_has_windowed_attention(config: &LlamaModelConfig) -> bool {
    config.gemma3.is_some()
}

impl LlamaModelConfig {
    /// True when decoder layer `layer_idx` (0-based) applies RoPE to Q and K.
    ///
    /// Verbatim llama.cpp `src/models/smollm3.cpp:69`:
    /// `use_rope = (il + 1) % n_no_rope_layer_step != 0`.
    ///
    /// The `(il + 1)` is load-bearing and NOT interchangeable with `il % step`:
    /// llama.cpp uses the `il % step` convention elsewhere
    /// (`src/models/smallthinker.cpp:109`), and the two disagree on which layers
    /// are skipped. For a 36-layer SmolLM3-3B this formula skips layers
    /// 3, 7, 11, 15, 19, 23, 27, 31, 35 — including the final layer.
    ///
    /// A step of 0 is treated as "rope every layer", mirroring the `step > 0 &&`
    /// guard in llama.cpp `src/models/afmoe.cpp:137` and keeping the modulo total.
    pub fn layer_uses_rope(&self, layer_idx: usize) -> bool {
        match self.no_rope_layer_step {
            // `!x.is_multiple_of(step)` is exactly `x % step != 0`; spelled this
            // way because clippy rejects the manual modulo under `-D warnings`.
            Some(step) if step > 0 => !(layer_idx + 1).is_multiple_of(step as usize),
            _ => true,
        }
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = match gguf.architecture() {
            Some(
                architecture @ ("llama" | "mistral" | "qwen2" | "qwen3" | "qwen3moe" | "qwen35"
                | "smollm3" | "gemma2" | "gemma3" | "gemma4" | "phi3" | "command-r"
                | "lfm2" | "bitnet-b1.58" | "mobilemoe"),
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
        let gemma3 = Gemma3Metadata::from_gguf(gguf, architecture)?;
        let gemma4 = Gemma4Metadata::from_gguf(gguf, architecture);
        let qwen35 = Qwen35Metadata::from_gguf(gguf, architecture);
        let lfm2 = Lfm2Metadata::from_gguf(gguf, architecture);
        if let Some(meta) = lfm2.as_ref() {
            lfm2_reject_unrunnable_shapes(gguf, architecture, meta)?;
        }

        let attention_head_count = required_u32(
            gguf,
            &architecture_key(architecture, "attention.head_count"),
        )?;
        // Gemma 4 rows carry per-layer arrays for `feed_forward_length` (E2B) and
        // `attention.head_count_kv` (12B). The per-layer truth lives in
        // `Gemma4Metadata` (`ffn_length_at`/`kv_heads_at`); these config scalars
        // hold the per-layer MAX so generic sizing stays safe. Gemma 4 forward
        // paths must use the per-layer accessors, never these scalars.
        // LFM2 carries the same per-layer `attention.head_count_kv` array shape,
        // but its zeros are STRUCTURAL (they mark conv layers, not a KV width).
        // Taking the max skips them so the scalar describes the attention
        // layers; `Lfm2Metadata` holds the per-layer truth. A plain
        // `metadata_u32` read here would miss the array entirely and fall back
        // to `attention_head_count` (32 instead of 8 for LFM2.5-2.6B).
        let attention_head_count_kv = match (gemma4.as_ref(), lfm2.as_ref()) {
            (Some(g), _) => g.max_kv_heads(),
            (None, Some(l)) => l.max_kv_heads(),
            (None, None) => llama_attention_head_count_kv(gguf, architecture, attention_head_count),
        };
        let feed_forward_length = match gemma4.as_ref() {
            Some(g) if g.max_ffn_length() > 0 => g.max_ffn_length(),
            _ => required_u32(gguf, &architecture_key(architecture, "feed_forward_length"))?,
        };
        Ok(Self {
            architecture: architecture.to_string(),
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
                // Command R uses ordinary LayerNorm and publishes the epsilon
                // under `layer_norm_epsilon` (without the RMS qualifier). Keep
                // the shared config field as the numeric epsilon consumed by
                // the runnable lane; `RunnableModel::apply_norm` selects the
                // correct normalization operation from the architecture.
                .or_else(|| {
                    (architecture == "command-r")
                        .then(|| {
                            gguf.metadata_f32(&architecture_key(
                                architecture,
                                "attention.layer_norm_epsilon",
                            ))
                        })
                        .flatten()
                })
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
            // own rows prove it. gemma3's resident-lane pairing fact lives on
            // `Gemma3Metadata.rope_neox_pairing`, so this dense-path flag stays
            // untouched even though the Metal resident lane is now gemma3's
            // DEFAULT serve lane (Phase 3b) — the CPU dense path it guards is
            // exactly the path gemma3 fails closed on (D20.2).
            // qwen35 full-attention layers are also unpermuted (NEOX split-half),
            // with partial RoPE over the first `rope.dimension_count` (64) of the
            // 256-wide head — handled in the runnable qwen35 path.
            rope_neox_pairing: arch_uses_neox_rope_pairing(architecture),
            no_rope_layer_step: arch_no_rope_layer_step(architecture),
            logit_scale: gguf.metadata_f32(&architecture_key(architecture, "logit_scale")),
            moe,
            gemma3,
            gemma4,
            qwen35,
            lfm2,
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
    /// Per-expert FFN width when it differs from the dense `feed_forward_length`
    /// (fine-grained MoE like qwen3moe: `{arch}.expert_feed_forward_length`).
    /// `None` keeps the Mixtral convention of expert width == dense FFN width.
    pub expert_feed_forward_length: Option<u32>,
    /// Always-on shared-expert FFN width (`{arch}.expert_shared_feed_forward_length`).
    /// MobileMoE sizes this independently of the routed experts (1536 vs 384), so it
    /// cannot be inferred from `expert_feed_forward_length`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_shared_feed_forward_length: Option<u32>,
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
        let expert_feed_forward_length = gguf.metadata_u32(&architecture_key(
            architecture,
            "expert_feed_forward_length",
        ));
        let expert_shared_feed_forward_length = gguf.metadata_u32(&architecture_key(
            architecture,
            "expert_shared_feed_forward_length",
        ));

        Some(Self {
            family_label,
            expert_count,
            expert_used_count,
            expert_weights_scale,
            expert_weights_norm,
            expert_gating_func,
            expert_feed_forward_length,
            expert_shared_feed_forward_length,
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

/// Gemma 3 (`general.architecture = "gemma3"`) window/RoPE metadata plus the
/// structural forward-pass facts a Llama-shaped config cannot express.
///
/// Gemma 3 alternates local (sliding-window) and global (full) attention on a
/// local:global cadence — every `sliding_window_pattern`-th layer is global
/// (layer `i` is global iff `(i + 1) % pattern == 0`; for the 1B's 26 layers and
/// pattern 6 the global layers are 5/11/17/23) — and the two layer types use
/// different RoPE bases. Unlike Gemma 4, the FINAL layer is NOT forced global
/// (layer 25 of the 1B is local), the head geometry is uniform across layers,
/// and there is no cross-layer KV sharing, PLE stream, or logit soft-cap.
///
/// Key sourcing (verified against the supported gemma-3-1b-it-Q8_0 row, whose
/// ONLY window/rope keys are `gemma3.attention.sliding_window=512` and
/// `gemma3.rope.freq_base=1e6`):
/// - `sliding_window` and the global RoPE base are REQUIRED GGUF keys — absent
///   keys fail closed with a typed error instead of assuming a value.
/// - The cadence (6) and the local-layer RoPE base (10000.0) have NO GGUF key in
///   any known gemma3 conversion — the converter never writes one and the
///   pinned reference hardcodes both (the same disclosed-constant situation as
///   smollm3's `no_rope_layer_step`, see [`arch_no_rope_layer_step`]). They are
///   reference-pinned constants here; an explicit
///   `gemma3.attention.sliding_window_pattern` (scalar period) or
///   `gemma3.rope.freq_base_swa` key is honored if a future conversion writes
///   one, and a malformed value for either is a hard error, never a silent
///   fallback.
///
/// This struct records the parsed values the Metal-resident lane consumes.
/// Since Phase 3b of the gemma3 Metal campaign that lane is REACHABLE on a
/// resident-capable host (Q8_0 exact row only, hazard H5); everywhere else
/// gemma3 serves via the runnable bridge ([`arch_requires_runnable_bridge`])
/// and the CPU dense forward stays fail-closed (hazard H4).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Gemma3Metadata {
    /// Local attention window in positions — GGUF `attention.sliding_window`
    /// (REQUIRED). The window INCLUDES the current position: a local layer at
    /// position `pos` attends `[pos + 1 - window ..= pos]` (identical
    /// convention to [`Gemma4LayerPlan::window`] and the runnable reference).
    pub sliding_window: u32,
    /// Local:global cadence — every `pattern`-th layer is global. GGUF
    /// `attention.sliding_window_pattern` (scalar) when present; otherwise the
    /// reference-pinned 6.
    pub sliding_window_pattern: u32,
    /// RoPE base for global (full-attention) layers — GGUF `rope.freq_base`
    /// (REQUIRED; 1e6 on the 1B row).
    pub rope_freq_base_global: f32,
    /// RoPE base for local (sliding-window) layers — GGUF `rope.freq_base_swa`
    /// when present; otherwise the reference-pinned 10000.0 (no gemma3
    /// conversion writes this key).
    pub rope_freq_base_local: f32,
    /// Per-layer attention type derived from the cadence: `true` = local
    /// (sliding), `false` = global (full). Length = `block_count`.
    pub layer_is_sliding: Vec<bool>,
    /// Token-embedding scale sqrt(d_model) (sqrt(1152) for the 1B). The
    /// resident embed gather must multiply by this before layer 0.
    pub embed_scale: f32,
    /// Gemma's FFN activation is GeGLU — `gelu_tanh(gate) * up` — not the
    /// Llama-family SiLU. The resident FFN encode must select the GeGLU kernel.
    pub ffn_geglu: bool,
    /// gemma3 Q/K projection weights are NOT permuted for adjacent even/odd
    /// RoPE, so the resident lane must force NEOX split-half pairing host-side
    /// (gemma4-encode precedent). Deliberately NOT surfaced through
    /// [`LlamaModelConfig::rope_neox_pairing`] / `arch_uses_neox_rope_pairing`:
    /// that flag drives the dense CPU path, whose gemma3 forward stays
    /// fail-closed — flipping it there would perturb a guarded-off path for no
    /// benefit. The runnable lane independently asserts NEOX for gemma
    /// (`is_gemma` in `runnable::model`).
    pub rope_neox_pairing: bool,
}

impl Gemma3Metadata {
    /// Reference-pinned local:global cadence (every 6th layer global). No gemma3
    /// GGUF key carries this; see the struct docs for the disclosure.
    pub const REFERENCE_SLIDING_WINDOW_PATTERN: u32 = 6;
    /// Reference-pinned RoPE base for local (sliding-window) layers. No gemma3
    /// GGUF key carries this; see the struct docs for the disclosure.
    pub const REFERENCE_LOCAL_ROPE_FREQ_BASE: f32 = 10_000.0;

    /// Returns `Ok(Some)` for `gemma3`, `Ok(None)` for every other
    /// architecture, and `Err` for a gemma3 file whose required window/rope
    /// keys are missing or malformed — fail closed, no silent defaults.
    pub fn from_gguf(gguf: &GgufFile, architecture: &str) -> Result<Option<Self>> {
        if architecture != "gemma3" {
            return Ok(None);
        }
        // Microsoft's 270M BitNet embedding checkpoint uses the Gemma 3 dense
        // graph but deliberately omits sliding-window metadata. The pinned
        // BitNet llama.cpp loader treats that omission as full attention.
        if is_bitnet_embedding_model(gguf) {
            return Ok(None);
        }
        let key = |suffix: &str| architecture_key(architecture, suffix);
        let block_count = required_u32(gguf, &key("block_count"))?;
        let embedding_length = required_u32(gguf, &key("embedding_length"))?;

        let window_key = key("attention.sliding_window");
        let sliding_window = match gguf.metadata_u32(&window_key) {
            Some(window) if window > 0 => window,
            Some(_) => {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "metadata {window_key} must be greater than zero; a zero-width sliding \
                     window cannot mask anything, so Camelid fails closed"
                )))
            }
            None => {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "required metadata {window_key} is missing or not an integer; the gemma3 \
                     sliding-window mask cannot be sized without it, so Camelid fails closed \
                     instead of assuming a window"
                )))
            }
        };

        let global_base_key = key("rope.freq_base");
        let rope_freq_base_global = match gguf.metadata_f32(&global_base_key) {
            Some(base) if base > 0.0 => base,
            Some(_) => {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "metadata {global_base_key} must be greater than zero"
                )))
            }
            None => {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "required metadata {global_base_key} is missing or not a float; the gemma3 \
                     global-layer RoPE base cannot be derived, so Camelid fails closed instead \
                     of assuming one"
                )))
            }
        };

        // Optional override keys: honored when present, hard error when present
        // but malformed, reference-pinned constant when absent. Presence is
        // checked on the raw metadata map so a wrong-typed value cannot be
        // confused with an absent key (the typed accessors return None for both).
        let pattern_key = key("attention.sliding_window_pattern");
        let sliding_window_pattern = if gguf.metadata.contains_key(&pattern_key) {
            match gguf.metadata_u32(&pattern_key) {
                Some(period) if period > 0 => period,
                _ => {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "metadata {pattern_key} must be a positive integer period (gemma3 \
                         declares a scalar local:global cadence); Camelid fails closed rather \
                         than falling back to the reference cadence over an explicit key"
                    )))
                }
            }
        } else {
            Self::REFERENCE_SLIDING_WINDOW_PATTERN
        };

        let local_base_key = key("rope.freq_base_swa");
        let rope_freq_base_local = if gguf.metadata.contains_key(&local_base_key) {
            match gguf.metadata_f32(&local_base_key) {
                Some(base) if base > 0.0 => base,
                _ => {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "metadata {local_base_key} must be a positive float; Camelid fails \
                         closed rather than falling back to the reference local base over an \
                         explicit key"
                    )))
                }
            }
        } else {
            Self::REFERENCE_LOCAL_ROPE_FREQ_BASE
        };

        // Layer `i` is global iff `(i + 1) % pattern == 0`; everything else is
        // local. NO forced-global final layer (that is a Gemma 4 rule): for 26
        // layers at pattern 6 the globals are 5/11/17/23 and layer 25 is local,
        // matching the runnable reference schedule.
        let layer_is_sliding = (0..block_count)
            .map(|i| !(i + 1).is_multiple_of(sliding_window_pattern))
            .collect();

        Ok(Some(Self {
            sliding_window,
            sliding_window_pattern,
            rope_freq_base_global,
            rope_freq_base_local,
            layer_is_sliding,
            embed_scale: (embedding_length as f32).sqrt(),
            ffn_geglu: true,
            rope_neox_pairing: true,
        }))
    }

    /// True if decoder layer `idx` uses local (sliding-window) attention.
    pub fn is_sliding_layer(&self, idx: usize) -> bool {
        self.layer_is_sliding.get(idx).copied().unwrap_or(false)
    }

    /// RoPE base (θ) for layer `idx` (local θ vs global θ).
    pub fn rope_freq_base_at(&self, idx: usize) -> f32 {
        if self.is_sliding_layer(idx) {
            self.rope_freq_base_local
        } else {
            self.rope_freq_base_global
        }
    }

    /// `Some(window)` for local layers, `None` for global layers. The window
    /// INCLUDES the current position: attend `[pos + 1 - window ..= pos]`
    /// (identical convention to [`Gemma4LayerPlan::window`]).
    pub fn layer_window(&self, idx: usize) -> Option<u32> {
        if self.is_sliding_layer(idx) {
            Some(self.sliding_window)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod gemma3_tests {
    use super::Gemma3Metadata;

    fn one_b_meta() -> Gemma3Metadata {
        // The schedule is a LITERAL list (false = global at 5/11/17/23), not
        // the production `(i + 1) % pattern` expression — duplicating the
        // formula here would make the test a tautology that passes even if the
        // derivation regressed. The `from_gguf`-driven fixture tests
        // (tests/model_binding.rs) cover the derivation itself; this unit test
        // covers the accessors over a known schedule.
        #[rustfmt::skip]
        let layer_is_sliding = vec![
            true, true, true, true, true, false, // layers 0-5
            true, true, true, true, true, false, // layers 6-11
            true, true, true, true, true, false, // layers 12-17
            true, true, true, true, true, false, // layers 18-23
            true, true, // layers 24-25 (no forced-global final layer)
        ];
        Gemma3Metadata {
            sliding_window: 512,
            sliding_window_pattern: 6,
            rope_freq_base_global: 1_000_000.0,
            rope_freq_base_local: 10_000.0,
            layer_is_sliding,
            embed_scale: (1152.0f32).sqrt(),
            ffn_geglu: true,
            rope_neox_pairing: true,
        }
    }

    #[test]
    fn one_b_schedule_globals_at_5_11_17_23_and_no_forced_global_final_layer() {
        let meta = one_b_meta();
        let globals: Vec<usize> = meta
            .layer_is_sliding
            .iter()
            .enumerate()
            .filter(|(_, sliding)| !**sliding)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(globals, vec![5, 11, 17, 23]);
        // Unlike gemma4, the final layer is NOT forced global: layer 25 is local.
        assert!(meta.is_sliding_layer(25));
        for idx in 0..26 {
            if globals.contains(&idx) {
                assert_eq!(meta.rope_freq_base_at(idx), 1_000_000.0, "layer {idx}");
                assert_eq!(meta.layer_window(idx), None, "layer {idx}");
            } else {
                assert_eq!(meta.rope_freq_base_at(idx), 10_000.0, "layer {idx}");
                assert_eq!(meta.layer_window(idx), Some(512), "layer {idx}");
            }
        }
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

/// LFM2 / LFM2.5 (`general.architecture = "lfm2"`) hybrid short-convolution
/// metadata. `None` for every other architecture.
///
/// LFM2 interleaves **double-gated short convolution** blocks with GQA
/// attention blocks. The conv layers carry `shortconv.{conv,in_proj,out_proj}`
/// and **no `attn_q/k/v` at all**, so a dense Llama tensor map cannot express
/// the model — the same shape as qwen35, and the reason `lfm2` is classified
/// [`is_runnable_only_arch`].
///
/// The per-layer schedule is NOT a separate key: llama.cpp derives it from the
/// per-layer `attention.head_count_kv` array, where a **0 marks a conv layer**
/// (`src/models/lfm2.cpp:10` — `is_recr_impl[il] = n_head_kv(il) == 0`), which
/// the converter writes from `layer_types` (`conversion/lfm2.py:36-39`). That
/// zero is also why the scalar `attention_head_count_kv` on
/// [`LlamaModelConfig`] must come from [`Self::max_kv_heads`] and never from
/// the raw array: a 0 would size the attention layers' KV to nothing.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Lfm2Metadata {
    /// Short-conv kernel width — GGUF `lfm2.shortconv.l_cache` (3 for every
    /// published LFM2 row). The rolling conv state is `l_cache - 1` wide;
    /// llama.cpp asserts `l_cache > 1` (`lfm2.cpp:167`).
    pub shortconv_l_cache: u32,
    /// Per-layer KV head count, verbatim from the GGUF array. `0` entries are
    /// conv layers; non-zero entries are GQA attention layers.
    pub kv_heads_per_layer: Vec<u32>,
    /// Per-layer schedule: `true` = short-conv (recurrent), `false` = attention.
    pub layer_is_conv: Vec<bool>,
}

impl Lfm2Metadata {
    pub fn from_gguf(gguf: &GgufFile, architecture: &str) -> Option<Self> {
        if architecture != "lfm2" {
            return None;
        }
        let key = |suffix: &str| architecture_key(architecture, suffix);
        let block_count = gguf.metadata_u32(&key("block_count")).unwrap_or(0);
        let head_count = gguf.metadata_u32(&key("attention.head_count")).unwrap_or(0);
        // Per-layer-or-scalar, mirroring `Gemma4Metadata`: a scalar broadcasts,
        // an array must cover every layer to be honored. A file that honors
        // neither shape falls back to the scalar head count, which makes every
        // layer look like attention and fails loudly at conv-tensor binding
        // rather than silently skipping the conv schedule.
        let kv_heads_per_layer =
            if let Some(scalar) = gguf.metadata_u32(&key("attention.head_count_kv")) {
                vec![scalar; block_count as usize]
            } else {
                match gguf.metadata_array_u32_optional(&key("attention.head_count_kv")) {
                    Ok(Some(values)) if values.len() == block_count as usize => values,
                    _ => vec![head_count; block_count as usize],
                }
            };
        let layer_is_conv = kv_heads_per_layer.iter().map(|&kv| kv == 0).collect();
        Some(Self {
            shortconv_l_cache: gguf.metadata_u32(&key("shortconv.l_cache")).unwrap_or(0),
            kv_heads_per_layer,
            layer_is_conv,
        })
    }

    /// True if decoder layer `idx` is a short-conv (recurrent) layer.
    pub fn is_conv_layer(&self, idx: usize) -> bool {
        self.layer_is_conv.get(idx).copied().unwrap_or(false)
    }

    /// KV heads for the ATTENTION layers — the largest per-layer value, which
    /// skips the conv layers' structural zeros. Used for the config scalar and
    /// for generic KV sizing.
    pub fn max_kv_heads(&self) -> u32 {
        self.kv_heads_per_layer.iter().copied().max().unwrap_or(0)
    }
}

/// Fail closed on the two LFM2 shapes the runnable lane's forward cannot execute.
///
/// Both are absent from LFM2.5-2.6B, so this changes nothing for the shipped row —
/// it stops a future/variant LFM2 file from decoding fluent-looking but wrong
/// output under an architecture Camelid claims to implement (the failure mode the
/// smollm3 NoPE audit was opened for).
///
/// 1. **Sliding-window attention.** llama.cpp `src/models/lfm2.cpp:23-28`
///    (`load_arch_hparams`) reads `<arch>.attention.sliding_window` and, when it is
///    present and non-zero, marks EVERY non-recurrent layer SWA:
///    `hparams.swa_type = LLAMA_SWA_TYPE_STANDARD; ... is_swa_impl[il] =
///    !is_recr_impl[il];`. The runnable lane's attention is full-causal with no
///    window mask, so it would silently attend outside the model's trained span.
/// 2. **A heterogeneous attention schedule.** K/V are sized from one global head
///    count ([`Lfm2Metadata::max_kv_heads`]), so attention layers with DIFFERENT
///    non-zero KV head counts would be executed with the wrong cache stride.
fn lfm2_reject_unrunnable_shapes(
    gguf: &GgufFile,
    architecture: &str,
    meta: &Lfm2Metadata,
) -> Result<()> {
    let swa_key = architecture_key(architecture, "attention.sliding_window");
    if let Some(window) = gguf.metadata_u32(&swa_key) {
        if window > 0 {
            return Err(BackendError::UnsupportedGguf(format!(
                "lfm2: {swa_key} = {window} makes every attention layer sliding-window \
                 (llama.cpp src/models/lfm2.cpp load_arch_hparams), and the runnable lane's \
                 attention is full-causal with no window mask; refusing rather than decoding \
                 with the wrong attention span"
            )));
        }
    }

    let widths: std::collections::BTreeSet<u32> = meta
        .kv_heads_per_layer
        .iter()
        .copied()
        .filter(|&n| n > 0)
        .collect();
    if widths.len() > 1 {
        return Err(BackendError::UnsupportedGguf(format!(
            "lfm2: {} carries a heterogeneous attention schedule (distinct non-zero KV head \
             counts {widths:?}); the runnable lane sizes K/V from one global count and would \
             mis-stride the cache",
            architecture_key(architecture, "attention.head_count_kv")
        )));
    }
    Ok(())
}

/// NEOX split-half RoPE pairing per architecture: qwen2/qwen3/qwen3moe/qwen35,
/// phi3, lfm2, and bitnet-b1.58 (unpermuted weights;
/// proven during MUSTER M-A2 — adjacent even/odd degenerates long generation,
/// split-half restores coherence; the runnable lane independently asserts NEOX
/// for phi3). Everything else keeps adjacent even/odd (LLaMA-style permuted
/// conversions). Pure so the gate is unit-testable.
///
/// `lfm2` is NEOX per llama.cpp `llama-model.cpp:2477` (LLM_ARCH_LFM2 falls
/// into the `LLAMA_ROPE_TYPE_NEOX` group at `:2492`), and its converter
/// (`conversion/lfm2.py`) does not permute Q/K, so the split-half pairing is
/// the one that matches the weights on disk.
fn arch_uses_neox_rope_pairing(architecture: &str) -> bool {
    matches!(
        architecture,
        "qwen2" | "qwen3" | "qwen3moe" | "qwen35" | "phi3" | "lfm2" | "bitnet-b1.58"
    )
}

/// NoPE layer step per architecture.
///
/// `smollm3` is the only NoPE architecture in the admitted set: llama.cpp
/// `src/models/smollm3.cpp:5` sets `n_no_rope_layer_step = 4` and its graph at
/// `:69` gates both `ggml_rope_ext` calls on `(il + 1) % step != 0`.
///
/// Every other admitted architecture ropes unconditionally — verified against
/// llama.cpp `models/llama.cpp:146,152`, `qwen2.cpp:86,92`, `qwen3.cpp:91,100`,
/// `phi3.cpp:107,113` and `mistral3.cpp:137,143`. `gemma3`/`gemma4`/`qwen35`
/// carry per-layer rope *bases* or schedules rather than skips, and those are
/// modelled elsewhere (`Gemma4Metadata`, `Qwen35Metadata`,
/// `runnable::model::layer_rope_base`), not here.
///
/// Pure so the gate is unit-testable without a model file.
fn arch_no_rope_layer_step(architecture: &str) -> Option<u32> {
    match architecture {
        "smollm3" => Some(4),
        _ => None,
    }
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
    /// gemma3 sandwich norm: RMSNorm applied to the attention block's output
    /// BEFORE its residual add (`post_attention_norm`, shape
    /// `[embedding_length]`). `Some` only for architectures with the 4-norm
    /// sandwich structure (gemma3); `None` for the Llama 2-norm structure.
    /// Bound in lockstep with [`Self::post_ffw_norm`] (both `Some` or both
    /// `None`). See [`LlamaTensorBinding::bind`] for the per-architecture
    /// presence invariant.
    pub post_attention_norm: Option<GgufTensorDescriptor>,
    pub ffn_norm: GgufTensorDescriptor,
    /// gemma3 sandwich norm: RMSNorm applied to the FFN block's output BEFORE
    /// its residual add (`post_ffw_norm`, shape `[embedding_length]`),
    /// mirroring [`Self::post_attention_norm`].
    pub post_ffw_norm: Option<GgufTensorDescriptor>,
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
        /// Frozen per-expert SELECTION bias (`blk.N.exp_probs_b.bias`). Added to the
        /// sigmoid scores before top-k; the committed weights are gathered from the
        /// UNBIASED scores. Optional: DeepSeek-V2 rows do not ship it.
        expert_bias: Option<GgufTensorDescriptor>,
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

        // Per-architecture QK-norm classification. Qwen3 and gemma3 apply a
        // per-head RMSNorm to Q and K after the projections and before RoPE
        // (`attn_q_norm`/`attn_k_norm`, shape `[head_dim]`); the plain
        // Llama-family rows do not. We classify every architecture that reaches
        // this dense binder so a model can never be silently mis-bound in either
        // direction (carrying QK-norm weights that the forward path would drop,
        // or fabricating them where none exist). gemma3 moved from unclassified
        // (which silently bound `(None, None)` and dropped all 104 of the 1B's
        // norm tensors — the mis-binding disclosed by the serve router's
        // fail-closed divert) to EXPECTED in Phase 1 of the gemma3 Metal
        // campaign; since Phase 3b the Metal-resident forward APPLIES them on
        // resident-capable hosts, while the CPU dense forward stays fail-closed
        // for windowed archs (hazard H4).
        let architecture = gguf.architecture().unwrap_or_default();
        let requires_qk_norm = matches!(architecture, "qwen3" | "qwen3moe" | "gemma3");
        // Command R's tensor contract is depth-dependent in the pinned
        // reference graph: 32-layer Aya Expanse has no Q/K norms, while the
        // >=64-layer variants carry them. The runnable attemptability slice is
        // deliberately anchored to the former; the loader rejects the latter's
        // per-head-distinct layout until that graph has parity evidence.
        let allows_optional_qk_norm = architecture == "command-r";
        let forbids_qk_norm = matches!(architecture, "llama" | "mistral" | "qwen2");
        // gemma3's 4-norm "sandwich" structure: an extra RMSNorm on the attention
        // output and on the FFN output, each applied BEFORE its residual add
        // (`post_attention_norm`/`post_ffw_norm`, shape `[embedding_length]`).
        // Required in lockstep with the QK-norm pair — a gemma3 row missing any
        // of the four cannot be run correctly and fails closed.
        let expects_post_norms = architecture == "gemma3";

        // Qwen3 (and gemma3) set the per-head dim explicitly via
        // `attention.key_length` / `attention.value_length` and it is NOT
        // guaranteed to equal `embedding_length / head_count` (Qwen3
        // 0.6B/4B/32B and gemma3-1B — 1152/4 = 288 vs head_dim 256 — differ).
        // The dense path carries the explicit head_dim through
        // `LlamaModelConfig.attention_key_length` / `DenseLlamaDims`. The engine
        // assumes a single head_dim for K and V, so require
        // key_length == value_length and fail closed otherwise.
        if requires_qk_norm || allows_optional_qk_norm {
            let key_length =
                gguf.metadata_u32(&architecture_key(architecture, "attention.key_length"));
            let value_length =
                gguf.metadata_u32(&architecture_key(architecture, "attention.value_length"));
            if let (Some(k), Some(v)) = (key_length, value_length) {
                if k != v {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "{architecture} attention.key_length={k} != value_length={v}; the engine \
                         assumes a single per-head dimension for K and V, so this row fails closed"
                    )));
                }
            }
        }

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for layer_idx in 0..config.block_count {
            let q_norm_name = format!("blk.{layer_idx}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{layer_idx}.attn_k_norm.weight");
            let (attention_q_norm, attention_k_norm) = if requires_qk_norm {
                // Required for Qwen3: both must be present, or fail closed.
                let q = find_tensor(gguf, &q_norm_name).cloned();
                let k = find_tensor(gguf, &k_norm_name).cloned();
                if q.is_none() || k.is_none() {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "{architecture} layer {layer_idx} is missing QK-norm tensors \
                         (attn_q_norm present: {}, attn_k_norm present: {}); this architecture applies \
                         per-head norm to Q and K and cannot be run correctly without them",
                        q.is_some(),
                        k.is_some()
                    )));
                }
                (q, k)
            } else if allows_optional_qk_norm {
                // Aya Expanse 8B (32 layers) legitimately omits this pair.
                // Still reject an incomplete pair: accepting one weight would
                // silently change only half of the attention graph.
                let q = find_tensor(gguf, &q_norm_name).cloned();
                let k = find_tensor(gguf, &k_norm_name).cloned();
                match (q, k) {
                    (Some(q), Some(k)) => (Some(q), Some(k)),
                    (None, None) => (None, None),
                    (q, k) => {
                        return Err(BackendError::UnsupportedModelArchitecture(format!(
                            "command-r layer {layer_idx} has an incomplete optional QK-norm pair \
                             (attn_q_norm present: {}, attn_k_norm present: {}); refusing rather \
                             than applying a one-sided norm",
                            q.is_some(),
                            k.is_some()
                        )))
                    }
                }
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
            let post_attention_norm_name = format!("blk.{layer_idx}.post_attention_norm.weight");
            let post_ffw_norm_name = format!("blk.{layer_idx}.post_ffw_norm.weight");
            let (post_attention_norm, post_ffw_norm) = if expects_post_norms {
                // Required for gemma3: both must be present, or fail closed.
                let post_attn = find_tensor(gguf, &post_attention_norm_name).cloned();
                let post_ffw = find_tensor(gguf, &post_ffw_norm_name).cloned();
                if post_attn.is_none() || post_ffw.is_none() {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "{architecture} layer {layer_idx} is missing sandwich norm tensors \
                         (post_attention_norm present: {}, post_ffw_norm present: {}); this \
                         architecture norms the attention and FFN outputs before each residual \
                         add and cannot be run correctly without them",
                        post_attn.is_some(),
                        post_ffw.is_some()
                    )));
                }
                (post_attn, post_ffw)
            } else {
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

            let attention_norm =
                required_tensor(gguf, &format!("blk.{layer_idx}.attn_norm.weight"))?;
            // Command R is a parallel-residual block: attention and FFN consume
            // the same LayerNorm output, and there is no `ffn_norm` tensor. The
            // dense binding remains useful for shape auditing even though all
            // execution is routed to the runnable lane.
            let ffn_norm = if architecture == "command-r" {
                attention_norm.clone()
            } else {
                required_tensor(gguf, &format!("blk.{layer_idx}.ffn_norm.weight"))?
            };

            layers.push(LlamaLayerTensors {
                attention_norm,
                attention,
                attention_output: required_tensor(
                    gguf,
                    &format!("blk.{layer_idx}.attn_output.weight"),
                )?,
                post_attention_norm,
                ffn_norm,
                post_ffw_norm,
                ffn: if let Some(moe) = config.moe.as_ref() {
                    // An always-on shared expert (DeepSeek-V3 / MobileMoE shape) ships as
                    // `ffn_{gate,up,down}_shexp`. When all three are present, bind the
                    // DeepSeekMoE variant so the sigmoid/normalised/scaled routing path in
                    // `deepseek_moe_ffn` is reachable at all; without this the binder only
                    // ever emitted `MoE`, `moe_shared_*` stayed None, and every model fell
                    // through to `mixtral_moe_ffn`'s hardcoded softmax.
                    // Only `mobilemoe` is admitted to the shared-expert (DeepSeek-style)
                    // binding: its routing is what `deepseek_moe_ffn` and the Metal kernels
                    // implement. Any other row keeps the plain `MoE` binding even if a future
                    // GGUF ships `shexp` tensors.
                    let shexp = if config.architecture == "mobilemoe" {
                        (
                            find_tensor(gguf, &format!("blk.{layer_idx}.ffn_gate_shexp.weight")),
                            find_tensor(gguf, &format!("blk.{layer_idx}.ffn_up_shexp.weight")),
                            find_tensor(gguf, &format!("blk.{layer_idx}.ffn_down_shexp.weight")),
                        )
                    } else {
                        (None, None, None)
                    };
                    if let (Some(shared_gate), Some(shared_up), Some(shared_down)) = shexp {
                        LlamaFfnTensors::DeepSeekMoE {
                            expert_bias: find_tensor(
                                gguf,
                                &format!("blk.{layer_idx}.exp_probs_b.bias"),
                            )
                            .cloned(),
                            shared_gate: shared_gate.clone(),
                            shared_up: shared_up.clone(),
                            shared_down: shared_down.clone(),
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
            match (&layer.post_attention_norm, &layer.post_ffw_norm) {
                (Some(post_attn), Some(post_ffw)) => {
                    require_descriptor_shape(
                        post_attn,
                        &[dims.embedding_length],
                        &format!("layer {idx} post attention norm"),
                    )?;
                    require_descriptor_shape(
                        post_ffw,
                        &[dims.embedding_length],
                        &format!("layer {idx} post ffn norm"),
                    )?;
                }
                (None, None) => {}
                _ => {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "layer {idx} has exactly one of post_attention_norm/post_ffw_norm bound; \
                         the sandwich norms must be present as a pair"
                    )));
                }
            }
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
                    let expert_ff = moe
                        .expert_feed_forward_length
                        .map(|v| v as usize)
                        .unwrap_or(dims.feed_forward_length);
                    validate_moe_expert_tensor_shape(
                        gate_experts,
                        dims.embedding_length,
                        expert_ff,
                        moe.expert_count as usize,
                        &format!("layer {idx} ffn gate experts"),
                    )?;
                    validate_moe_expert_tensor_shape(
                        up_experts,
                        dims.embedding_length,
                        expert_ff,
                        moe.expert_count as usize,
                        &format!("layer {idx} ffn up experts"),
                    )?;
                    validate_moe_expert_tensor_shape(
                        down_experts,
                        expert_ff,
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
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};

    /// Builds a bare `GgufFile` carrying only metadata — enough to exercise the
    /// pure `*Metadata::from_gguf` parsers without touching a real file.
    fn meta_only_gguf(pairs: Vec<(&str, GgufMetadataValue)>) -> GgufFile {
        let mut gguf = GgufFile {
            path: std::path::PathBuf::new(),
            version: 3,
            tensor_count: 0,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: Default::default(),
            tensors: Vec::new(),
        };
        for (k, v) in pairs {
            gguf.metadata.insert(k.to_string(), v);
        }
        gguf
    }

    fn f32_tensor(name: &str, dimensions: Vec<u64>) -> GgufTensorDescriptor {
        let n_bytes = dimensions.iter().product::<u64>() * 4;
        GgufTensorDescriptor {
            name: name.to_string(),
            dimensions,
            tensor_type: GgufTensorType::F32,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }

    fn tiny_command_r_gguf() -> GgufFile {
        let mut gguf = meta_only_gguf(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("command-r".into()),
            ),
            ("command-r.context_length", GgufMetadataValue::U32(32)),
            ("command-r.embedding_length", GgufMetadataValue::U32(4)),
            ("command-r.block_count", GgufMetadataValue::U32(1)),
            ("command-r.feed_forward_length", GgufMetadataValue::U32(8)),
            ("command-r.attention.head_count", GgufMetadataValue::U32(2)),
            (
                "command-r.attention.head_count_kv",
                GgufMetadataValue::U32(1),
            ),
            (
                "command-r.attention.layer_norm_epsilon",
                GgufMetadataValue::F32(1e-5),
            ),
            ("command-r.logit_scale", GgufMetadataValue::F32(0.125)),
        ]);
        gguf.tensors = vec![
            f32_tensor("token_embd.weight", vec![4, 8]),
            f32_tensor("output_norm.weight", vec![4]),
            f32_tensor("blk.0.attn_norm.weight", vec![4]),
            f32_tensor("blk.0.attn_q.weight", vec![4, 4]),
            f32_tensor("blk.0.attn_k.weight", vec![4, 2]),
            f32_tensor("blk.0.attn_v.weight", vec![4, 2]),
            f32_tensor("blk.0.attn_output.weight", vec![4, 4]),
            f32_tensor("blk.0.ffn_gate.weight", vec![4, 8]),
            f32_tensor("blk.0.ffn_up.weight", vec![4, 8]),
            f32_tensor("blk.0.ffn_down.weight", vec![8, 4]),
        ];
        gguf.tensor_count = gguf.tensors.len() as i64;
        gguf
    }

    #[test]
    fn command_r_aya_expanse_metadata_is_attemptable_but_runnable_only() {
        // Exact architecture/config values observed in the immutable
        // bartowski/aya-expanse-8b-GGUF Q4_K_M header. No weight bytes are
        // needed to prove config acceptance and the fail-closed routing facts.
        let mut gguf = meta_only_gguf(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("command-r".into()),
            ),
            ("command-r.context_length", GgufMetadataValue::U32(8_192)),
            ("command-r.embedding_length", GgufMetadataValue::U32(4_096)),
            ("command-r.block_count", GgufMetadataValue::U32(32)),
            (
                "command-r.feed_forward_length",
                GgufMetadataValue::U32(14_336),
            ),
            ("command-r.attention.head_count", GgufMetadataValue::U32(32)),
            (
                "command-r.attention.head_count_kv",
                GgufMetadataValue::U32(8),
            ),
            (
                "command-r.attention.layer_norm_epsilon",
                GgufMetadataValue::F32(1e-5),
            ),
            ("command-r.rope.freq_base", GgufMetadataValue::F32(10_000.0)),
            (
                "command-r.rope.scaling.type",
                GgufMetadataValue::String("none".into()),
            ),
            ("command-r.logit_scale", GgufMetadataValue::F32(0.125)),
        ]);
        gguf.tensors = vec![f32_tensor("token_embd.weight", vec![4_096, 256_000])];
        gguf.tensor_count = 1;

        let config = super::LlamaModelConfig::from_gguf(&gguf).expect("command-r config");
        assert_eq!(config.architecture, "command-r");
        assert_eq!(config.context_length, 8_192);
        assert_eq!(config.embedding_length, 4_096);
        assert_eq!(config.block_count, 32);
        assert_eq!(config.feed_forward_length, 14_336);
        assert_eq!(config.attention_head_count, 32);
        assert_eq!(config.attention_head_count_kv, 8);
        assert_eq!(config.vocab_size, Some(256_000));
        assert_eq!(config.rms_norm_epsilon, 1e-5);
        assert_eq!(config.logit_scale, Some(0.125));
        assert!(!config.rope_neox_pairing);
        assert!(super::is_implemented_architecture("command-r"));
        assert!(super::is_runnable_only_arch("command-r"));
    }

    #[test]
    fn command_r_binding_accepts_shared_norm_and_tied_output() {
        let gguf = tiny_command_r_gguf();
        let config = super::LlamaModelConfig::from_gguf(&gguf).expect("command-r config");
        let binding = super::LlamaTensorBinding::bind(&gguf, &config)
            .expect("32-layer-style command-r tensor contract must bind");

        assert!(binding.output_is_tied_embedding);
        assert_eq!(
            binding.layers[0].attention_norm.name,
            "blk.0.attn_norm.weight"
        );
        assert_eq!(binding.layers[0].ffn_norm.name, "blk.0.attn_norm.weight");
        match &binding.layers[0].attention {
            super::LlamaAttentionTensors::Standard { q_norm, k_norm, .. } => {
                assert!(q_norm.is_none());
                assert!(k_norm.is_none());
            }
            super::LlamaAttentionTensors::Mla { .. } => panic!("command-r must bind standard GQA"),
        }
    }

    #[test]
    fn command_r_binding_rejects_an_incomplete_optional_qk_norm_pair() {
        let mut gguf = tiny_command_r_gguf();
        gguf.tensors
            .push(f32_tensor("blk.0.attn_q_norm.weight", vec![2]));
        gguf.tensor_count = gguf.tensors.len() as i64;
        let config = super::LlamaModelConfig::from_gguf(&gguf).expect("command-r config");
        let err = super::LlamaTensorBinding::bind(&gguf, &config)
            .expect_err("one-sided command-r QK norm must fail closed");
        let msg = err.to_string();
        assert!(msg.contains("incomplete optional QK-norm pair"), "{msg}");
    }

    #[test]
    fn exact_bitnet_embedding_rows_are_detected_and_gemma_uses_full_attention() {
        let qwen = meta_only_gguf(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("qwen3".into()),
            ),
            (
                "general.name",
                GgufMetadataValue::String("bitnet-embeddings-0.6b".into()),
            ),
        ]);
        assert!(super::is_bitnet_embedding_model(&qwen));

        let gemma = meta_only_gguf(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("gemma3".into()),
            ),
            (
                "general.name",
                GgufMetadataValue::String("bitnet-embeddings-270m".into()),
            ),
            ("gemma3.context_length", GgufMetadataValue::U32(32_768)),
            ("gemma3.embedding_length", GgufMetadataValue::U32(640)),
            ("gemma3.block_count", GgufMetadataValue::U32(18)),
            ("gemma3.feed_forward_length", GgufMetadataValue::U32(2_048)),
            ("gemma3.attention.head_count", GgufMetadataValue::U32(4)),
            ("gemma3.attention.head_count_kv", GgufMetadataValue::U32(1)),
            ("gemma3.attention.key_length", GgufMetadataValue::U32(256)),
            ("gemma3.vocab_size", GgufMetadataValue::U32(262_144)),
            ("gemma3.rope.freq_base", GgufMetadataValue::F32(1_000_000.0)),
        ]);
        assert!(super::is_bitnet_embedding_model(&gemma));
        let config = super::LlamaModelConfig::from_gguf(&gemma).expect("BitNet Gemma config");
        assert!(
            config.gemma3.is_none(),
            "missing SWA metadata means full attention"
        );
        assert_eq!(config.attention_key_length, Some(256));
    }

    #[test]
    fn gemma2_header_parses_for_the_runnable_bridge() {
        // The runnable runtime already implements Gemma 2's sandwich norms and
        // logit soft-cap. The shared config is still required by the load/inspect
        // pipeline before routing reaches that runtime, so this header parse is
        // attemptability plumbing rather than a broad-family support claim.
        let gguf = meta_only_gguf(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("gemma2".into()),
            ),
            ("gemma2.context_length", GgufMetadataValue::U32(8_192)),
            ("gemma2.embedding_length", GgufMetadataValue::U32(3_584)),
            ("gemma2.block_count", GgufMetadataValue::U32(42)),
            ("gemma2.feed_forward_length", GgufMetadataValue::U32(14_336)),
            ("gemma2.attention.head_count", GgufMetadataValue::U32(16)),
            ("gemma2.attention.head_count_kv", GgufMetadataValue::U32(8)),
            (
                "gemma2.attention.layer_norm_rms_epsilon",
                GgufMetadataValue::F32(1e-6),
            ),
        ]);

        let config = super::LlamaModelConfig::from_gguf(&gguf)
            .expect("gemma2 header must reach its runnable-only route");
        assert_eq!(config.architecture, "gemma2");
        assert!(super::is_implemented_architecture("gemma2"));
        assert!(super::is_runnable_only_arch("gemma2"));
    }

    /// The real LFM2.5-2.6B row: 30 layers, `head_count_kv` an **i32 array**
    /// whose zeros mark the 22 short-conv layers and whose 8s mark the 8 GQA
    /// layers (verbatim from the published GGUF).
    fn lfm2_2_6b_kv_heads() -> Vec<i32> {
        vec![
            0, 0, 8, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0,
            0,
        ]
    }

    /// The runnable lane's attention is full-causal. llama.cpp turns a non-zero
    /// `attention.sliding_window` into SWA on EVERY attention layer, so such a file
    /// must be refused rather than decoded with the wrong attention span.
    #[test]
    fn lfm2_sliding_window_row_is_refused() {
        let base = |extra: Vec<(&'static str, GgufMetadataValue)>| {
            let mut kv: Vec<(&'static str, GgufMetadataValue)> = vec![
                (
                    "general.architecture",
                    GgufMetadataValue::String("lfm2".into()),
                ),
                ("lfm2.block_count", GgufMetadataValue::U32(30)),
                ("lfm2.attention.head_count", GgufMetadataValue::U32(32)),
                ("lfm2.context_length", GgufMetadataValue::U32(4096)),
                ("lfm2.embedding_length", GgufMetadataValue::U32(2048)),
                ("lfm2.feed_forward_length", GgufMetadataValue::U32(10752)),
                (
                    "lfm2.attention.head_count_kv",
                    GgufMetadataValue::Array(
                        lfm2_2_6b_kv_heads()
                            .into_iter()
                            .map(GgufMetadataValue::I32)
                            .collect(),
                    ),
                ),
                ("lfm2.shortconv.l_cache", GgufMetadataValue::U32(3)),
            ];
            kv.extend(extra);
            meta_only_gguf(kv)
        };

        // The real row carries no sliding_window key and must still parse.
        assert!(super::LlamaModelConfig::from_gguf(&base(vec![])).is_ok());

        // A windowed row is refused, and the message names the blocker.
        let err = super::LlamaModelConfig::from_gguf(&base(vec![(
            "lfm2.attention.sliding_window",
            GgufMetadataValue::U32(512),
        )]))
        .expect_err("a windowed lfm2 row must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("sliding_window") && msg.contains("full-causal"),
            "refusal must name the window blocker, got: {msg}"
        );

        // A heterogeneous attention schedule is refused too: K/V are sized from
        // one global head count and would otherwise be mis-strided.
        let mut mixed = lfm2_2_6b_kv_heads();
        mixed[5] = 4; // an attention layer with a different width
        let err = super::LlamaModelConfig::from_gguf(&meta_only_gguf(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("lfm2".into()),
            ),
            ("lfm2.block_count", GgufMetadataValue::U32(30)),
            ("lfm2.attention.head_count", GgufMetadataValue::U32(32)),
            ("lfm2.context_length", GgufMetadataValue::U32(4096)),
            ("lfm2.embedding_length", GgufMetadataValue::U32(2048)),
            ("lfm2.feed_forward_length", GgufMetadataValue::U32(10752)),
            (
                "lfm2.attention.head_count_kv",
                GgufMetadataValue::Array(mixed.into_iter().map(GgufMetadataValue::I32).collect()),
            ),
            ("lfm2.shortconv.l_cache", GgufMetadataValue::U32(3)),
        ]))
        .expect_err("a heterogeneous lfm2 schedule must be refused");
        assert!(
            err.to_string().contains("heterogeneous"),
            "refusal must name the schedule blocker, got: {err}"
        );
    }

    #[test]
    fn lfm2_metadata_reads_the_real_2_6b_row() {
        let gguf = meta_only_gguf(vec![
            ("lfm2.block_count", GgufMetadataValue::U32(30)),
            ("lfm2.attention.head_count", GgufMetadataValue::U32(32)),
            (
                "lfm2.attention.head_count_kv",
                GgufMetadataValue::Array(
                    lfm2_2_6b_kv_heads()
                        .into_iter()
                        .map(GgufMetadataValue::I32)
                        .collect(),
                ),
            ),
            ("lfm2.shortconv.l_cache", GgufMetadataValue::U32(3)),
        ]);
        let meta = super::Lfm2Metadata::from_gguf(&gguf, "lfm2").expect("lfm2 metadata");

        assert_eq!(meta.shortconv_l_cache, 3);
        // 22 conv + 8 attention, matching the published "22 short-conv + 8 GQA".
        assert_eq!(meta.layer_is_conv.iter().filter(|c| **c).count(), 22);
        assert_eq!(meta.layer_is_conv.iter().filter(|c| !**c).count(), 8);
        // Spot-check the schedule against the array: 0 => conv, 8 => attention.
        assert!(meta.is_conv_layer(0));
        assert!(meta.is_conv_layer(1));
        assert!(!meta.is_conv_layer(2));
        assert!(!meta.is_conv_layer(29 - 2)); // layer 27 carries 8 kv heads
        assert!(meta.is_conv_layer(29));
        // Out-of-range must not panic and must not claim a conv layer.
        assert!(!meta.is_conv_layer(30));

        // The scalar must skip the structural zeros — 8, never 0 and never the
        // 32-head fallback. This is the value the KV cache is sized from.
        assert_eq!(meta.max_kv_heads(), 8);
    }

    #[test]
    fn lfm2_metadata_is_none_for_other_architectures() {
        let gguf = meta_only_gguf(vec![("lfm2.shortconv.l_cache", GgufMetadataValue::U32(3))]);
        assert!(super::Lfm2Metadata::from_gguf(&gguf, "llama").is_none());
        assert!(super::Lfm2Metadata::from_gguf(&gguf, "qwen35").is_none());
        assert!(super::Lfm2Metadata::from_gguf(&gguf, "lfm2").is_some());
    }

    #[test]
    fn lfm2_scalar_head_count_kv_broadcasts() {
        // A hypothetical all-attention LFM2 row carrying a SCALAR head_count_kv
        // must broadcast to every layer and report zero conv layers, rather
        // than falling through the array arm to the head-count default.
        let gguf = meta_only_gguf(vec![
            ("lfm2.block_count", GgufMetadataValue::U32(4)),
            ("lfm2.attention.head_count", GgufMetadataValue::U32(32)),
            ("lfm2.attention.head_count_kv", GgufMetadataValue::U32(8)),
            ("lfm2.shortconv.l_cache", GgufMetadataValue::U32(3)),
        ]);
        let meta = super::Lfm2Metadata::from_gguf(&gguf, "lfm2").expect("lfm2 metadata");
        assert_eq!(meta.kv_heads_per_layer, vec![8, 8, 8, 8]);
        assert!(meta.layer_is_conv.iter().all(|c| !*c));
        assert_eq!(meta.max_kv_heads(), 8);
    }

    #[test]
    fn lfm2_short_array_falls_back_instead_of_mis_scheduling() {
        // An array that does not cover every layer must NOT be honored — a
        // partial schedule would silently mark real attention layers as conv
        // and bind the wrong tensors. Falling back to the head-count default
        // yields an all-attention schedule, which fails loudly at conv-tensor
        // binding instead.
        let gguf = meta_only_gguf(vec![
            ("lfm2.block_count", GgufMetadataValue::U32(30)),
            ("lfm2.attention.head_count", GgufMetadataValue::U32(32)),
            (
                "lfm2.attention.head_count_kv",
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::I32(0),
                    GgufMetadataValue::I32(8),
                ]),
            ),
            ("lfm2.shortconv.l_cache", GgufMetadataValue::U32(3)),
        ]);
        let meta = super::Lfm2Metadata::from_gguf(&gguf, "lfm2").expect("lfm2 metadata");
        assert_eq!(meta.kv_heads_per_layer.len(), 30);
        assert!(meta.layer_is_conv.iter().all(|c| !*c));
    }

    #[test]
    fn neox_rope_pairing_covers_exactly_the_proven_archs() {
        // qwen2/qwen3/qwen35 verified vs real rows; phi3 proven during
        // MUSTER M-A2 (adjacent even/odd degenerates long generation on the
        // exact Phi-3-mini-4k row, split-half restores coherence). Everything
        // else — including the other unpermuted-but-unproven archs — must stay
        // on adjacent even/odd until its own row proves the flip.
        assert!(super::arch_uses_neox_rope_pairing("qwen3"));
        assert!(super::arch_uses_neox_rope_pairing("qwen3moe"));
        assert!(super::arch_uses_neox_rope_pairing("qwen35"));
        assert!(super::arch_uses_neox_rope_pairing("phi3"));
        assert!(super::arch_uses_neox_rope_pairing("qwen2"));
        // lfm2: llama.cpp classifies LLM_ARCH_LFM2 as LLAMA_ROPE_TYPE_NEOX
        // (`llama-model.cpp:2477` → `:2492`) and its converter leaves Q/K
        // unpermuted, so split-half is what the on-disk weights expect.
        assert!(super::arch_uses_neox_rope_pairing("lfm2"));
        assert!(super::arch_uses_neox_rope_pairing("bitnet-b1.58"));
        assert!(!super::arch_uses_neox_rope_pairing("llama"));
        assert!(!super::arch_uses_neox_rope_pairing("mistral"));
        assert!(!super::arch_uses_neox_rope_pairing("gemma3"));
        assert!(!super::arch_uses_neox_rope_pairing("gemma4"));
        // llama.cpp classifies LLM_ARCH_COMMAND_R in its ordinary RoPE group:
        // adjacent pairs, not NEOX split-half.
        assert!(!super::arch_uses_neox_rope_pairing("command-r"));
    }

    #[test]
    fn no_rope_layer_step_covers_exactly_smollm3() {
        // smollm3 is the ONLY NoPE architecture in the admitted set. llama.cpp
        // `src/models/smollm3.cpp:5` hardcodes n_no_rope_layer_step = 4 (there is
        // no GGUF key for it) and `:69` skips rotary on both Q and K when
        // (il + 1) % 4 == 0.
        assert_eq!(super::arch_no_rope_layer_step("smollm3"), Some(4));

        // Every other admitted architecture ropes unconditionally, verified
        // against its llama.cpp graph builder: models/llama.cpp:146,152 ·
        // qwen2.cpp:86,92 · qwen3.cpp:91,100 · phi3.cpp:107,113 ·
        // mistral3.cpp:137,143. gemma3/gemma4/qwen35 carry per-layer rope BASES
        // or schedules — not skips — and those live in their own metadata.
        for arch in [
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "qwen3moe",
            "qwen35",
            "gemma3",
            "gemma4",
            "phi3",
            "command-r",
            "lfm2",
            "bitnet-b1.58",
        ] {
            assert!(
                super::arch_no_rope_layer_step(arch).is_none(),
                "{arch} must not be treated as a NoPE architecture"
            );
        }
    }

    #[test]
    fn runnable_only_arch_set_is_exactly_the_unconditional_bridge_set() {
        // These archs have no correct optimized forward on ANY host: gemma2's
        // sandwich norms are still dropped at bind, and qwen35's + lfm2's
        // hybrid layers do not fit the dense tensor map — so every direct
        // dense-session lane fails closed for them instead of decoding
        // fluent-looking garbage. gemma3 left this set in Phase 3b of the Metal
        // campaign (its routing is capability-aware — pinned by the split test
        // below).
        for arch in ["qwen35", "gemma2", "command-r", "lfm2", "bitnet-b1.58"] {
            assert!(
                super::is_runnable_only_arch(arch),
                "{arch} must be classified runnable-lane-only"
            );
        }
        for arch in [
            "llama", "mistral", "qwen2", "qwen3", "qwen3moe", "smollm3", "gemma3", "gemma4",
            "phi3", "",
        ] {
            assert!(
                !super::is_runnable_only_arch(arch),
                "{arch:?} must not be classified runnable-lane-only"
            );
        }
    }

    #[test]
    fn runnable_bridge_predicate_splits_gemma3_by_host_capability() {
        // gemma3→Metal Phase 3b: the serve router and the CLI direct-session
        // guard key on `arch_requires_runnable_bridge`. The split under test:
        // gemma3 rides the dense/resident lane ONLY where the Metal-resident
        // host capability holds, and falls back to the runnable bridge —
        // never the CPU dense forward — everywhere else. qwen35/gemma2 are
        // bridge-only regardless of capability; dense archs never bridge.
        for capable in [false, true] {
            for q8 in [false, true] {
                for arch in ["qwen35", "gemma2", "command-r", "bitnet-b1.58"] {
                    assert!(
                        super::arch_requires_runnable_bridge_given(arch, capable, q8),
                        "{arch} must require the runnable bridge on every host"
                    );
                }
                for arch in ["llama", "mistral", "qwen2", "qwen3", "gemma4", "phi3", ""] {
                    assert!(
                        !super::arch_requires_runnable_bridge_given(arch, capable, q8),
                        "{arch:?} must never require the runnable bridge"
                    );
                }
            }
        }
        assert!(
            super::arch_requires_runnable_bridge_given("gemma3", false, true),
            "gemma3 must fall back to the runnable bridge where the resident lane cannot serve"
        );
        assert!(
            !super::arch_requires_runnable_bridge_given("gemma3", true, true),
            "gemma3 must route to the dense/resident lane on a resident-capable host"
        );
    }

    /// Phase 3c finding F3: routing was quant-blind, so a gemma3 Q4_K_M on a
    /// resident-capable Mac was routed onto the dense lane, declined by the
    /// H5 Q8_0 pin, and then H4-errored on every request — a hard regression
    /// against the pre-flip bridge, which served every gemma3 quant. The
    /// quantization is now half the decision.
    #[test]
    fn a_non_q8_windowed_row_requires_the_runnable_bridge_on_every_host() {
        for capable in [false, true] {
            assert!(
                super::arch_requires_runnable_bridge_given("gemma3", capable, false),
                "a non-Q8_0 gemma3 has no resident lane on any host \
                 (windowed_resident_host_available={capable}); it must take the bridge"
            );
        }
        // The Q8_0 row on a capable host is the ONLY combination that routes
        // to the dense/resident lane — the causality control for the above.
        assert!(!super::arch_requires_runnable_bridge_given(
            "gemma3", true, true
        ));
    }

    /// The routing-time Q8_0 pin must mirror the engine-level H5 pin in
    /// `inference::resident_decode_eligible`: EVERY per-layer linear Q8_0.
    /// One non-Q8_0 layer linear anywhere is enough to send the file to the
    /// bridge, and a file with no recognizable layer linears fails closed too.
    #[test]
    fn windowed_quant_admission_requires_every_layer_linear_to_be_q8_0() {
        use crate::gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType};
        let tensor = |name: &str, tensor_type: GgufTensorType| GgufTensorDescriptor {
            name: name.to_string(),
            dimensions: vec![32, 32],
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 0,
        };
        let names = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let empty = || GgufFile {
            path: std::path::PathBuf::new(),
            version: 3,
            tensor_count: 0,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: Default::default(),
            tensors: Vec::new(),
        };
        let file = |types: &[GgufTensorType]| {
            let mut gguf = empty();
            gguf.tensors = names
                .iter()
                .zip(types)
                .map(|(name, t)| tensor(name, *t))
                .collect();
            // A non-layer tensor at a different quant must not affect the
            // decision (token_embd is Q8_0 on this row, output is tied).
            gguf.tensors
                .push(tensor("token_embd.weight", GgufTensorType::Q8_0));
            gguf
        };
        let all_q8 = [GgufTensorType::Q8_0; 7];
        assert!(super::windowed_arch_resident_quant_admissible(&file(
            &all_q8
        )));
        // Each single non-Q8_0 layer linear must disqualify on its own.
        for idx in 0..names.len() {
            let mut types = all_q8;
            types[idx] = GgufTensorType::Q4K;
            assert!(
                !super::windowed_arch_resident_quant_admissible(&file(&types)),
                "a Q4_K {} must disqualify the resident lane",
                names[idx]
            );
        }
        assert!(
            !super::windowed_arch_resident_quant_admissible(&empty()),
            "a file with no per-layer linears must fail closed to the bridge"
        );
    }

    #[test]
    fn implemented_architecture_set_is_exactly_the_from_gguf_accept_arm() {
        // This list must stay byte-for-byte in sync with the match arm in
        // LlamaModelConfig::from_gguf. If you change one, change both.
        for arch in [
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "qwen3moe",
            "smollm3",
            "gemma2",
            "gemma3",
            "gemma4",
            "phi3",
            "command-r",
            "lfm2",
            "bitnet-b1.58",
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
