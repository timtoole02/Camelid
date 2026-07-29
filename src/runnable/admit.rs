//! Admission gate for the runnable lane.
//!
//! A GGUF runs iff **all three** coverage axes are covered:
//!
//! ```text
//! architecture graph  ×  quant dequant  ×  tokenizer
//! ```
//!
//! `admit` evaluates a parsed [`GgufFile`] against the v1 covered-set and either
//! returns an [`AdmissionOk`] summary or an [`AdmissionReject`] naming the offending
//! axis, the offending value, and (for the quant axis) the offending tensor. The
//! reject is `Serialize` so the refusal reason is machine-readable, per
//! `RUNNABLE_LANE_SPEC.md` principle #2.
//!
//! The covered-set here is **authoritative for the runnable lane** and is taken
//! verbatim from the spec — it intentionally differs from `model.rs`'s
//! optimized-lane architecture allowlist (see `BACKEND_ASKS.md` RA-4). In
//! particular the runnable set includes `gemma2` and excludes
//! `mistral`/`smollm3`/`gemma4`/`lfm2`. That is not an admit-then-fail gap:
//! admitted files bind through the runnable lane's own generic runtime
//! (`runnable::model`, which implements the gemma2 attention/final logit
//! soft-caps), never through `model.rs`'s `LlamaModelConfig`.
//!
//! **BASALT D-B3 pilot carve-out (until Gate G3):** NVFP4 is a covered quant for
//! the `gemma4` pilot ONLY — `(arch == "gemma4", quant == NVFP4)` admits, any other
//! architecture carrying NVFP4 tensors is refused with a machine-readable reject
//! naming the D-B3 scope. Because `gemma4` is otherwise outside
//! `COVERED_ARCHITECTURES`, the architecture axis carries the mirror-image
//! carve-out: a gemma4 GGUF passes that axis iff it carries NVFP4 tensors (the
//! pilot shape); gemma4 files without NVFP4 keep today's architecture refusal.
//! NOTE: the REAL produced pilot row additionally carries one BF16 tensor
//! (`per_layer_model_proj.weight`). As of BASALT D-B6 (2026-07-17) BF16 is a
//! covered exact-decode quant (see `is_covered_quant`), so the real artifact now
//! admits fully: it passes the architecture carve-out AND the quant axis — the
//! single BF16 tensor decodes losslessly via `crate::tensor::decode_bf16_tensor`
//! (bf16 = the high 16 bits of f32; widening is exact). Full-file admission of an
//! NVFP4 gemma4 GGUF no longer requires a BF16-free file (the prior blocker is gone).
//! Admitting that shape here is a metadata-level classification for the BASALT
//! interop legs (inspect / dequant spot-checks); it is NOT a claim the generic
//! runnable runtime executes the gemma4 graph — smoke stays refused via the
//! oracle-qualification guardrail (`smoke::is_oracle_qualified`, anchored at G3),
//! and the serve bridge does not route gemma4.

use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;

use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufTensorType};

/// v1 covered architectures (`general.architecture`).
pub const COVERED_ARCHITECTURES: &[&str] = &[
    "llama", "qwen2", "qwen3", "qwen35", "gemma2", "gemma3", "phi3",
];

/// v1 covered tokenizer models (`tokenizer.ggml.model`), grouped by family below.
/// SPM (sentencepiece/llama-style) + BPE (gpt2-style) are the two covered families.
const SPM_TOKENIZERS: &[&str] = &["llama", "gemma", "gemma4"];
const BPE_TOKENIZERS: &[&str] = &["gpt2"];

/// Sentinel used in a reject when the offending axis value is absent from metadata.
const ABSENT: &str = "<absent>";

/// The coverage axis a GGUF failed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionAxis {
    Architecture,
    Quant,
    Tokenizer,
}

impl AdmissionAxis {
    pub fn as_str(self) -> &'static str {
        match self {
            AdmissionAxis::Architecture => "architecture",
            AdmissionAxis::Quant => "quant",
            AdmissionAxis::Tokenizer => "tokenizer",
        }
    }
}

/// Which covered tokenizer family a model resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerFamily {
    /// SentencePiece-style (llama/gemma).
    Spm,
    /// GPT-2-style byte-level BPE (qwen/phi/gpt2).
    Bpe,
}

/// Structured admission rejection. Names the offending axis + value (+ tensor for
/// the quant axis) so the refusal is machine-readable, not just a string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionReject {
    pub axis: AdmissionAxis,
    /// The offending value (architecture name, quant type, or tokenizer model);
    /// `"<absent>"` when the value was missing from metadata entirely.
    pub offending_value: String,
    /// For the quant axis, the first tensor carrying the unsupported quant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
    /// Human-readable single-line reason.
    pub message: String,
}

impl fmt::Display for AdmissionReject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AdmissionReject {}

/// An admission rejection maps onto the existing `UnsupportedGguf` backend error so
/// callers that thread `BackendError` keep working; the structured form is preserved
/// in the message.
impl From<AdmissionReject> for BackendError {
    fn from(reject: AdmissionReject) -> Self {
        BackendError::UnsupportedGguf(reject.message)
    }
}

/// Summary of an admitted GGUF: the resolved architecture, tokenizer family, and the
/// distinct set of quant types present (handy for downstream dequant wiring).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionOk {
    pub architecture: String,
    pub tokenizer: TokenizerFamily,
    pub quants: BTreeSet<GgufTensorType>,
}

/// Evaluate the three coverage axes against the v1 covered-set.
///
/// Axes are checked in a fixed order — architecture, tokenizer, then every tensor's
/// quant — so the reported failure is deterministic. The quant scan reports the
/// **first** tensor (in file order) carrying an unsupported type.
pub fn admit(file: &GgufFile) -> Result<AdmissionOk, AdmissionReject> {
    let architecture = check_architecture(file)?;
    let tokenizer = check_tokenizer(file)?;
    let quants = check_quants(file, &architecture)?;
    Ok(AdmissionOk {
        architecture,
        tokenizer,
        quants,
    })
}

/// True iff the file carries at least one NVFP4 tensor (the BASALT pilot shape).
fn has_nvfp4_tensors(file: &GgufFile) -> bool {
    file.tensors
        .iter()
        .any(|t| t.tensor_type == GgufTensorType::NVFP4)
}

fn check_architecture(file: &GgufFile) -> Result<String, AdmissionReject> {
    match file.architecture() {
        Some(arch) if COVERED_ARCHITECTURES.contains(&arch) => Ok(arch.to_string()),
        // BASALT D-B3 pilot carve-out: gemma4 is otherwise outside the runnable
        // covered set (its production forward lives in `gemma4_runtime`), but the
        // NVFP4 pilot rows are gemma4 files, so a gemma4 GGUF that carries NVFP4
        // tensors passes this axis. gemma4 files WITHOUT NVFP4 keep the refusal
        // below unchanged. Lane-wide NVFP4 admission is gated on G3.
        Some(arch @ "gemma4") if has_nvfp4_tensors(file) => Ok(arch.to_string()),
        Some(arch) => Err(AdmissionReject {
            axis: AdmissionAxis::Architecture,
            offending_value: arch.to_string(),
            tensor: None,
            message: format!(
                "unsupported architecture {arch:?}; runnable v1 covers {}",
                joined(COVERED_ARCHITECTURES)
            ),
        }),
        None => Err(AdmissionReject {
            axis: AdmissionAxis::Architecture,
            offending_value: ABSENT.to_string(),
            tensor: None,
            message: "missing general.architecture; runnable admission requires it".to_string(),
        }),
    }
}

fn check_tokenizer(file: &GgufFile) -> Result<TokenizerFamily, AdmissionReject> {
    match file.metadata_string("tokenizer.ggml.model") {
        Some(model) if SPM_TOKENIZERS.contains(&model) => Ok(TokenizerFamily::Spm),
        Some(model) if BPE_TOKENIZERS.contains(&model) => Ok(TokenizerFamily::Bpe),
        Some(model) => Err(AdmissionReject {
            axis: AdmissionAxis::Tokenizer,
            offending_value: model.to_string(),
            tensor: None,
            message: format!(
                "unsupported tokenizer.ggml.model {model:?}; runnable v1 covers SPM ({}) and BPE ({})",
                joined(SPM_TOKENIZERS),
                joined(BPE_TOKENIZERS)
            ),
        }),
        None => Err(AdmissionReject {
            axis: AdmissionAxis::Tokenizer,
            offending_value: ABSENT.to_string(),
            tensor: None,
            message: "missing tokenizer.ggml.model; runnable admission requires it".to_string(),
        }),
    }
}

/// A GGUF tensor quant type is covered iff the runnable lane has a dequant-to-f32
/// routine for it. K-quant *mix* recipes (Q4_K_M, Q5_K_M) are not distinct ggml
/// types — they appear on the wire as Q4K/Q5K/Q6K/Q8_0 tensors, all covered below.
///
/// NVFP4 is deliberately NOT in this blanket set: a dequant routine exists
/// (`crate::tensor::decode_nvfp4_tensor`), but admission is pilot-scoped to the
/// gemma4 architecture until Gate G3 (BASALT D-B3) — see `check_quants`.
///
/// BASALT D-B6 (2026-07-17): BF16 joined the covered set as an exact-decode type.
/// bf16 stores the high 16 bits of the f32 encoding, so decode is the lossless,
/// bit-deterministic widening `crate::tensor::decode_bf16_tensor` — no new numeric
/// code, and definitionally identical to the pin's `ggml_bf16_to_fp32`. This admits
/// legitimate mixed-type files (the gemma4-E4B pilot's single `per_layer_model_proj`
/// BF16 tensor) under the existing whole-file coverage model.
/// P1 loadability (2026-07-29): six formats the CPU tensor path already decoded
/// join the covered set — Q4_1, Q5_0, Q5_1, IQ4_NL, TQ1_0 and TQ2_0.
///
/// Every one had a working decoder in `crate::tensor` (`decode_q4_1_tensor`,
/// `decode_q5_0_tensor`, `decode_q5_1_tensor`, `decode_iq4_nl_tensor`,
/// `decode_tq1_0_tensor`, `decode_tq2_0_tensor`), and TQ2_0 additionally has a
/// wire-streaming block-dot lane. The gate refused them anyway, so files whose
/// only uncovered axis was a format this engine already reads were rejected.
///
/// Each is admitted behind a per-format dequant-parity receipt pinning the
/// decoder against llama.cpp `ggml/src/ggml-quants.c`:
/// `tensor::tests::legacy_quant_dequant_matches_the_reference_layout` and
/// `tensor::tests::tq{1,2}_0_dequant_matches_the_reference_layout`. Both cover the
/// element ORDER, which is the silent-corruption surface here — the legacy formats
/// interleave split-half (low nibble of `qs[j]` is element `j`, high nibble is
/// element `j + 16`), and TQ2_0's four 2-bit codes decode 32 apart.
///
/// **Q5_1 additionally required a reader fix**: `GgufTensorType::layout` folded it
/// in with Q5_0 at 22 bytes, but Q5_1 carries an extra f16 `m` and its block is 24.
/// Until that was corrected, admitting Q5_1 would have turned an honest parse-time
/// refusal into a broken load. See `gguf::reader` and its layout table test.
///
/// **Q8_K is deliberately NOT admitted**, despite `decode_q8_k_tensor` existing.
/// It is not a weight storage type: ggml documents it as "only used for
/// intermediate quantization and dot products" (`ggml/src/ggml-common.h`) and it
/// appears zero times in `tools/quantize/quantize.cpp`'s target list, so no real
/// GGUF stores weights as Q8_K. Admitting it would be a claim with no artifact
/// that could ever validate it. Having a decoder is not grounds for admission.
fn is_covered_quant(tt: GgufTensorType) -> bool {
    matches!(
        tt,
        GgufTensorType::F32
            | GgufTensorType::F16
            | GgufTensorType::Q8_0
            | GgufTensorType::Q6K
            | GgufTensorType::Q5K
            | GgufTensorType::Q4K
            | GgufTensorType::Q2K
            | GgufTensorType::Q3K
            | GgufTensorType::Q4_0
            | GgufTensorType::IQ4XS
            | GgufTensorType::BF16
            | GgufTensorType::Q4_1
            | GgufTensorType::Q5_0
            | GgufTensorType::Q5_1
            | GgufTensorType::IQ4NL
            | GgufTensorType::Tq1_0
            | GgufTensorType::Tq2_0
    )
}

/// The BASALT pilot architecture: the only arch for which NVFP4 tensors admit
/// until Gate G3 anchors the lane-wide decision (DECISIONS.md D17 / D-B3).
const NVFP4_PILOT_ARCH: &str = "gemma4";

fn check_quants(
    file: &GgufFile,
    architecture: &str,
) -> Result<BTreeSet<GgufTensorType>, AdmissionReject> {
    let mut seen = BTreeSet::new();
    for tensor in &file.tensors {
        // BASALT D-B3: NVFP4 is arch-conditional, not a blanket covered quant.
        let nvfp4_pilot =
            tensor.tensor_type == GgufTensorType::NVFP4 && architecture == NVFP4_PILOT_ARCH;
        if !is_covered_quant(tensor.tensor_type) && !nvfp4_pilot {
            if tensor.tensor_type == GgufTensorType::NVFP4 {
                return Err(AdmissionReject {
                    axis: AdmissionAxis::Quant,
                    offending_value: format!("{:?}", tensor.tensor_type),
                    tensor: Some(tensor.name.clone()),
                    message: format!(
                        "unsupported quant NVFP4 in tensor {} for architecture \
                         {architecture:?}: NVFP4 is pilot-scoped to gemma4 until Gate G3 \
                         (BASALT D-B3)",
                        tensor.name
                    ),
                });
            }
            return Err(AdmissionReject {
                axis: AdmissionAxis::Quant,
                offending_value: format!("{:?}", tensor.tensor_type),
                tensor: Some(tensor.name.clone()),
                message: format!(
                    "unsupported quant {:?} in tensor {}; runnable v1 covers \
                     F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, \
                     Q5_K, Q6_K, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0",
                    tensor.tensor_type, tensor.name
                ),
            });
        }
        seen.insert(tensor.tensor_type);
    }

    // BASALT D-B2: sidecar per-tensor scales fail closed at admission.
    //
    // ModelOpt-converted NVFP4 GGUFs carry optional F32 sidecar tensors
    // (`<name>.scale` = weight_scale_2, `<name>.input_scale`) that the reference
    // implementation applies POST-matmul via a ggml_mul node. Camelid v1 implements
    // only the in-block UE4M3 sub-scales; silently ignoring sidecar scales would
    // compute wrong logits — quiet corruption, so we refuse the whole file.
    //
    // Seam split (deliberate): admission is METADATA-only — it sees tensor names,
    // types, and shapes, never wire bytes, so the sidecar check (name-visible)
    // lives here, while the D17/T5 NaN-sentinel refusal (0x7F/0xFF UE4M3 scale
    // bytes, byte-visible only) fires at decode time in
    // `crate::tensor::decode_nvfp4_tensor` via `runnable::dequant`.
    if seen.contains(&GgufTensorType::NVFP4) {
        if let Some(sidecar) = file
            .tensors
            .iter()
            .find(|t| t.name.ends_with(".scale") || t.name.ends_with(".input_scale"))
        {
            return Err(AdmissionReject {
                axis: AdmissionAxis::Quant,
                offending_value: "NVFP4".to_string(),
                tensor: Some(sidecar.name.clone()),
                message: format!(
                    "NVFP4 GGUF carries per-tensor scale sidecar tensor {}; \
                     sidecar-bearing (ModelOpt-converted) NVFP4 files are not yet \
                     supported — ignoring their scales would silently corrupt logits, \
                     so admission fails closed (BASALT D-B2)",
                    sidecar.name
                ),
            });
        }
    }

    // BASALT Amendment 3 §9 platform gate + GABBRO M2 narrowing: NVFP4 admission
    // is allowed on Windows AND macOS in this release, and refused on every other
    // target (macOS joined once its Apple-Silicon CPU decode was proven bit-exact,
    // GABBRO Gate G-M1). A RUNTIME check (`cfg!` inside ordinary code),
    // deliberately not a `#[cfg]` wall — the crate compiles identically on every
    // target and refused hosts get this named refusal (DECISIONS.md D17
    // micro-decisions, §9.1). It fires AFTER the D-B3 scope and D-B2 sidecar
    // checks so those signed postures stay platform-independent: only a file
    // that would otherwise fully admit reaches this refusal. Mirrored in the
    // gemma4 wire-lane load path (`gemma4_runtime::nvfp4_windows_only_check`).
    // NOTE (GABBRO M2): the refusal message reads "Windows/macOS-only" and the
    // support matrices are truthed-up in this same ratchet PR (Tim's ruling).
    // macOS runs NVFP4 on the CPU wire lane (used by `serve`) and, opt-in via the
    // gemma4-generate-gpu subcommand, on the Metal GPU resident lane (kernel
    // nvfp4_block_linear_row_ksplit_f32y_wire; GABBRO M3 + M3-followup, self-parity-
    // proven vs the CPU oracle, T5 sentinel guard).
    if !cfg!(target_os = "windows")
        && !cfg!(target_os = "macos")
        && seen.contains(&GgufTensorType::NVFP4)
    {
        let tensor = file
            .tensors
            .iter()
            .find(|t| t.tensor_type == GgufTensorType::NVFP4)
            .map(|t| t.name.clone());
        return Err(AdmissionReject {
            axis: AdmissionAxis::Quant,
            offending_value: "NVFP4".to_string(),
            tensor,
            message: "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX".to_string(),
        });
    }

    Ok(seen)
}

fn joined(items: &[&str]) -> String {
    items.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufMetadataValue, GgufTensorDescriptor};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// Build a minimal in-set GGUF (llama / SPM / Q8_0) we can mutate per test.
    fn base_fixture() -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("llama".into()),
        );
        metadata.insert(
            "tokenizer.ggml.model".into(),
            GgufMetadataValue::String("llama".into()),
        );
        GgufFile {
            path: PathBuf::from("/tmp/model.gguf"),
            version: 3,
            tensor_count: 2,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![
                tensor("token_embd.weight", GgufTensorType::F32),
                tensor("blk.0.attn_q.weight", GgufTensorType::Q8_0),
            ],
        }
    }

    fn tensor(name: &str, tt: GgufTensorType) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.into(),
            dimensions: vec![32, 32],
            tensor_type: tt,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 34,
        }
    }

    fn set_meta(file: &mut GgufFile, key: &str, value: &str) {
        file.metadata
            .insert(key.into(), GgufMetadataValue::String(value.into()));
    }

    #[test]
    fn accepts_in_set_llama_spm_q8() {
        let ok = admit(&base_fixture()).expect("in-set GGUF must admit");
        assert_eq!(ok.architecture, "llama");
        assert_eq!(ok.tokenizer, TokenizerFamily::Spm);
        assert!(ok.quants.contains(&GgufTensorType::Q8_0));
        assert!(ok.quants.contains(&GgufTensorType::F32));
    }

    #[test]
    fn accepts_each_covered_architecture() {
        for arch in COVERED_ARCHITECTURES {
            let mut file = base_fixture();
            set_meta(&mut file, "general.architecture", arch);
            // qwen/phi ship a BPE tokenizer; keep SPM here — admission is per-axis
            // independent, and the architecture axis is what we're exercising.
            assert!(
                admit(&file).is_ok(),
                "covered architecture {arch} must admit"
            );
        }
    }

    #[test]
    fn accepts_bpe_tokenizer() {
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "qwen3");
        set_meta(&mut file, "tokenizer.ggml.model", "gpt2");
        let ok = admit(&file).expect("qwen3 + gpt2-BPE must admit");
        assert_eq!(ok.tokenizer, TokenizerFamily::Bpe);
    }

    #[test]
    fn accepts_covered_kquants() {
        for tt in [
            GgufTensorType::F16,
            GgufTensorType::Q4_0,
            GgufTensorType::Q3K,
            GgufTensorType::Q4K,
            GgufTensorType::Q5K,
            GgufTensorType::Q6K,
        ] {
            let mut file = base_fixture();
            file.tensors.push(tensor("blk.0.ffn_down.weight", tt));
            assert!(admit(&file).is_ok(), "covered quant {tt:?} must admit");
        }
    }

    #[test]
    fn accepts_iq4_xs_quant() {
        // IQ4_XS gained a runnable-lane dequant (decode_iq4_xs_tensor); an IQ4_XS model
        // whose other tensors are covered must now admit. (IQ4_NL remains an explicit gap.)
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.0.ffn_down.weight", GgufTensorType::IQ4XS));
        let ok = admit(&file).expect("IQ4_XS must admit");
        assert!(ok.quants.contains(&GgufTensorType::IQ4XS));
    }

    #[test]
    fn accepts_legacy_block_quants() {
        // Q4_1/Q5_0/Q5_1/IQ4_NL all had working decoders while the gate refused
        // them. Pinned by tensor::tests::legacy_quant_dequant_matches_the_reference_layout.
        for tt in [
            GgufTensorType::Q4_1,
            GgufTensorType::Q5_0,
            GgufTensorType::Q5_1,
            GgufTensorType::IQ4NL,
        ] {
            let mut file = base_fixture();
            file.tensors.push(tensor("blk.0.ffn_down.weight", tt));
            let ok = admit(&file).unwrap_or_else(|e| panic!("{tt:?} must admit: {e}"));
            assert!(ok.quants.contains(&tt));
        }
    }

    #[test]
    fn accepts_ternary_quants() {
        // TQ1_0/TQ2_0 have had CPU decoders (decode_tq{1,2}_0_tensor) and, for
        // TQ2_0, a wire-streaming block-dot lane, while the gate still refused
        // them. Pinned by the dequant-parity receipts in tensor::tests.
        for tt in [GgufTensorType::Tq1_0, GgufTensorType::Tq2_0] {
            let mut file = base_fixture();
            file.tensors.push(tensor("blk.0.ffn_down.weight", tt));
            let ok = admit(&file).unwrap_or_else(|e| panic!("{tt:?} must admit: {e}"));
            assert!(ok.quants.contains(&tt));
        }
    }

    #[test]
    fn rejects_q8_k_because_it_is_not_a_storage_type() {
        // decode_q8_k_tensor exists, but Q8_K is an intermediate dot-product type
        // in ggml ("only used for intermediate quantization and dot products",
        // ggml-common.h) and is not a quantize target, so no real GGUF stores
        // weights as Q8_K. Having a decoder is NOT sufficient grounds to admit.
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.0.ffn_down.weight", GgufTensorType::Q8K));
        assert!(
            admit(&file).is_err(),
            "Q8_K must stay refused: it is not a weight storage type"
        );
    }

    #[test]
    fn ternary_bonsai_tensor_type_mix_admits() {
        // Regression for the real artifact that motivated this change:
        // Ternary-Bonsai-4B-TQ2_0.gguf is architecture `qwen3` with tokenizer
        // `gpt2`, and carries 252 Tq2_0 + 145 F32 + 1 Q6K tensors. Every axis but
        // the quant axis already passed, so the file was refused purely on a
        // format the engine already decodes.
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "qwen3");
        set_meta(&mut file, "tokenizer.ggml.model", "gpt2");
        file.tensors
            .push(tensor("blk.0.ffn_down.weight", GgufTensorType::Tq2_0));
        file.tensors
            .push(tensor("blk.0.attn_output.weight", GgufTensorType::Q6K));

        let ok = admit(&file).expect("the Ternary-Bonsai tensor mix must admit");
        assert!(ok.quants.contains(&GgufTensorType::Tq2_0));
        assert!(ok.quants.contains(&GgufTensorType::Q6K));
        assert!(ok.quants.contains(&GgufTensorType::F32));
    }

    // --- BASALT D-B3 pilot scoping + D-B2 sidecar fail-closed ---

    /// A BF16-free pilot-like base shape: gemma4 with NVFP4 matmuls (embeddings
    /// Q8_0, norms F32). NOTE: the real produced `gemma-4-E4B-it-NVFP4-mm` row ALSO
    /// carries one BF16 tensor (`per_layer_model_proj.weight`); as of BASALT D-B6
    /// that tensor is a covered exact-decode quant, so the real shape now ADMITS
    /// FULLY — pinned by `gemma4_nvfp4_with_bf16_admits_fully_after_d_b6` below (the
    /// off-Windows twin keeps the §9 platform gate).
    fn gemma4_nvfp4_fixture() -> GgufFile {
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "gemma4");
        set_meta(&mut file, "tokenizer.ggml.model", "gemma4");
        file.tensors
            .push(tensor("blk.0.ffn_down.weight", GgufTensorType::NVFP4));
        file
    }

    // Amendment 3 §9 + GABBRO M2: NVFP4 admission is allowed on Windows and
    // macOS, so the tests that expect an ADMIT run on those legs, and the
    // remaining-platform twin (below) pins the named platform refusal on the
    // ubuntu/linux CI leg.
    #[test]
    #[cfg(target_os = "windows")]
    fn admits_gemma4_nvfp4_pilot() {
        let ok = admit(&gemma4_nvfp4_fixture()).expect("gemma4+NVFP4 pilot must admit (D-B3)");
        assert_eq!(ok.architecture, "gemma4");
        assert_eq!(ok.tokenizer, TokenizerFamily::Spm);
        assert!(ok.quants.contains(&GgufTensorType::NVFP4));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn admits_gemma4_nvfp4_pilot_on_macos() {
        // GABBRO M2 twin: the pilot shape now admits on macOS too (bit-exact
        // Apple-Silicon CPU decode, Gate G-M1), mirroring the Windows admit.
        let ok =
            admit(&gemma4_nvfp4_fixture()).expect("gemma4+NVFP4 pilot must admit on macOS (M2)");
        assert_eq!(ok.architecture, "gemma4");
        assert_eq!(ok.tokenizer, TokenizerFamily::Spm);
        assert!(ok.quants.contains(&GgufTensorType::NVFP4));
    }

    #[test]
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    fn gemma4_nvfp4_pilot_refuses_off_windows_with_platform_gate() {
        // Amendment 3 §9 twin: the otherwise-admitting pilot shape gets the
        // named TK2 refusal on unvalidated targets — a runtime gate, not a
        // compile wall, so this exact path runs on the ubuntu/linux CI leg.
        let reject =
            admit(&gemma4_nvfp4_fixture()).expect_err("NVFP4 must refuse on unvalidated platforms");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "NVFP4");
        assert_eq!(reject.tensor.as_deref(), Some("blk.0.ffn_down.weight"));
        assert_eq!(
            reject.message,
            "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"
        );
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    fn gemma4_nvfp4_with_bf16_admits_fully_after_d_b6() {
        // The REAL produced pilot row's shape (G2 receipt): NVFP4 matmuls PLUS one
        // BF16 tensor (per_layer_model_proj.weight). As of BASALT D-B6 (2026-07-17)
        // BF16 is a covered exact-decode quant, so the whole file now ADMITS FULLY:
        // the architecture carve-out passes AND the quant axis passes (the BF16
        // tensor decodes losslessly via crate::tensor::decode_bf16_tensor). This
        // inverts the pre-D-B6 refusal pin so the admission flip can't drift
        // silently. (Amendment 3 §9 + GABBRO M2: full NVFP4 admission is Windows AND
        // macOS, so the ADMIT expectation runs on the Windows and macOS legs; the
        // off-Windows/macOS twin below pins the platform gate.)
        let mut file = gemma4_nvfp4_fixture();
        file.tensors
            .push(tensor("per_layer_model_proj.weight", GgufTensorType::BF16));
        let ok = admit(&file).expect("real pilot shape must admit fully after D-B6");
        assert_eq!(ok.architecture, "gemma4");
        assert_eq!(ok.tokenizer, TokenizerFamily::Spm);
        assert!(ok.quants.contains(&GgufTensorType::NVFP4));
        assert!(
            ok.quants.contains(&GgufTensorType::BF16),
            "the pilot's BF16 tensor is now a covered quant (D-B6)"
        );
    }

    #[test]
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    fn gemma4_nvfp4_with_bf16_refuses_off_windows_platform_gate() {
        // Unvalidated-platform twin of the D-B6 admission pin (Linux leg — macOS now
        // admits, GABBRO M2): with BF16 covered, the real pilot shape (NVFP4 + BF16)
        // no longer refuses on the BF16 quant axis, so it reaches the Amendment 3 §9
        // platform gate — which refuses NVFP4 on targets other than Windows/macOS with
        // the named TK2 message. The refusal is the PLATFORM gate, never a BF16 quant
        // refusal (that would mean the covered-set widening regressed).
        let mut file = gemma4_nvfp4_fixture();
        file.tensors
            .push(tensor("per_layer_model_proj.weight", GgufTensorType::BF16));
        let reject = admit(&file).expect_err("NVFP4 must refuse on unvalidated platforms");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "NVFP4");
        assert_eq!(reject.tensor.as_deref(), Some("blk.0.ffn_down.weight"));
        assert_eq!(
            reject.message,
            "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"
        );
    }

    #[test]
    fn rejects_nvfp4_outside_pilot_arch() {
        // qwen3 + NVFP4 (the Phase 0/2 refusal artifact's exact shape) must refuse
        // with the D-B3 scope message on the quant axis — not a generic
        // quant-not-covered message and not a parse failure.
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "qwen3");
        set_meta(&mut file, "tokenizer.ggml.model", "gpt2");
        file.tensors
            .push(tensor("blk.0.ffn_down.weight", GgufTensorType::NVFP4));
        let reject = admit(&file).expect_err("qwen3+NVFP4 must reject until G3");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "NVFP4");
        assert_eq!(reject.tensor.as_deref(), Some("blk.0.ffn_down.weight"));
        assert!(
            reject
                .message
                .contains("pilot-scoped to gemma4 until Gate G3"),
            "message must name the D-B3 scope: {}",
            reject.message
        );
        // Machine-readable, like every other reject.
        let json = serde_json::to_value(&reject).expect("reject serializes");
        assert_eq!(json["axis"], "quant");
        assert_eq!(json["offending_value"], "NVFP4");
    }

    #[test]
    fn rejects_gemma4_without_nvfp4() {
        // The carve-out is pilot-shaped, not a blanket gemma4 admission: a gemma4
        // file with no NVFP4 tensors (e.g. the E4B Q8_0 row) keeps today's
        // architecture-axis refusal.
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "gemma4");
        let reject = admit(&file).expect_err("gemma4 without NVFP4 must still reject");
        assert_eq!(reject.axis, AdmissionAxis::Architecture);
        assert_eq!(reject.offending_value, "gemma4");
    }

    #[test]
    fn rejects_nvfp4_sidecar_scale_tensor() {
        // D-B2: a ModelOpt-style `<name>.scale` sidecar in an NVFP4 file fails
        // closed at admission (ignoring it would silently corrupt logits).
        let mut file = gemma4_nvfp4_fixture();
        file.tensors
            .push(tensor("blk.0.ffn_down.weight.scale", GgufTensorType::F32));
        let reject = admit(&file).expect_err("sidecar-bearing NVFP4 must reject");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "NVFP4");
        assert_eq!(
            reject.tensor.as_deref(),
            Some("blk.0.ffn_down.weight.scale")
        );
        assert!(
            reject.message.contains("sidecar") && reject.message.contains("D-B2"),
            "message must name the sidecar refusal: {}",
            reject.message
        );
    }

    #[test]
    fn rejects_nvfp4_sidecar_input_scale_tensor() {
        let mut file = gemma4_nvfp4_fixture();
        file.tensors.push(tensor(
            "blk.0.ffn_down.weight.input_scale",
            GgufTensorType::F32,
        ));
        let reject = admit(&file).expect_err("input_scale sidecar must reject");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(
            reject.tensor.as_deref(),
            Some("blk.0.ffn_down.weight.input_scale")
        );
    }

    #[test]
    fn sidecar_names_without_nvfp4_admit() {
        // The D-B2 refusal is scoped to NVFP4-bearing files: a covered-quant model
        // that happens to carry a `.scale`-suffixed tensor name is untouched.
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.0.some.scale", GgufTensorType::F32));
        assert!(
            admit(&file).is_ok(),
            ".scale names without NVFP4 must keep admitting"
        );
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    fn pilot_layer_output_scale_weight_is_not_a_sidecar() {
        // The real gemma4 pilot carries 42 F32 `blk.N.layer_output_scale.weight`
        // tensors. They end in `.weight`, not `.scale` — the sidecar check must not
        // false-positive on them or the pilot row itself would be refused. Runs on
        // the admit legs (Windows + macOS since GABBRO M2).
        let mut file = gemma4_nvfp4_fixture();
        file.tensors.push(tensor(
            "blk.0.layer_output_scale.weight",
            GgufTensorType::F32,
        ));
        let ok = admit(&file).expect("pilot layer_output_scale.weight must admit");
        assert!(ok.quants.contains(&GgufTensorType::NVFP4));
    }

    #[test]
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    fn pilot_layer_output_scale_weight_is_not_a_sidecar_off_windows() {
        // Unvalidated-platform twin of the false-positive pin (Linux leg — macOS
        // now admits, GABBRO M2): the refusal must be the §9 PLATFORM gate (which
        // fires after the sidecar check), never a sidecar misclassification of
        // `layer_output_scale.weight`.
        let mut file = gemma4_nvfp4_fixture();
        file.tensors.push(tensor(
            "blk.0.layer_output_scale.weight",
            GgufTensorType::F32,
        ));
        let reject = admit(&file).expect_err("NVFP4 refuses off Windows");
        assert_eq!(
            reject.message,
            "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"
        );
        assert!(
            !reject.message.contains("sidecar"),
            "must not misread layer_output_scale.weight as a sidecar: {}",
            reject.message
        );
    }

    #[test]
    fn rejects_unknown_architecture() {
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "mixtral");
        let reject = admit(&file).expect_err("unknown arch must reject");
        assert_eq!(reject.axis, AdmissionAxis::Architecture);
        assert_eq!(reject.offending_value, "mixtral");
        assert!(reject.tensor.is_none());
        assert!(reject.message.contains("mixtral"));
    }

    #[test]
    fn rejects_missing_architecture() {
        let mut file = base_fixture();
        file.metadata.remove("general.architecture");
        let reject = admit(&file).expect_err("missing arch must reject");
        assert_eq!(reject.axis, AdmissionAxis::Architecture);
        assert_eq!(reject.offending_value, "<absent>");
    }

    #[test]
    fn accepts_q2_k_quant() {
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.12.ffn_down.weight", GgufTensorType::Q2K));
        let admitted = admit(&file).expect("Q2_K has runnable dequant and wire-dot coverage");
        assert!(admitted.quants.contains(&GgufTensorType::Q2K));
    }

    #[test]
    fn rejects_unknown_quant_naming_tensor() {
        // Q8_K stands in for "decodable but not a storage type" — see
        // rejects_q8_k_because_it_is_not_a_storage_type. Q4_1 used to be the
        // example here; it is now covered.
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.12.ffn_down.weight", GgufTensorType::Q8K));
        let reject = admit(&file).expect_err("Q8_K must reject");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "Q8K");
        assert_eq!(reject.tensor.as_deref(), Some("blk.12.ffn_down.weight"));
        assert!(
            reject.message.ends_with("TQ1_0, TQ2_0"),
            "generic covered-set message must name its complete supported tail: {}",
            reject.message
        );
    }

    #[test]
    fn rejects_iquant_naming_tensor() {
        // IQ4_NL is now covered; the remaining i-quant family (IQ2_*, IQ3_*,
        // IQ1_*) has no decoder at all and parses as an Unknown type id.
        // ggml type ids: IQ2_XXS = 16, per ggml.h.
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.3.attn_k.weight", GgufTensorType::from_id(16)));
        let reject = admit(&file).expect_err("an undecodable i-quant must reject");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.tensor.as_deref(), Some("blk.3.attn_k.weight"));
    }

    #[test]
    fn rejects_unknown_tokenizer() {
        let mut file = base_fixture();
        set_meta(&mut file, "tokenizer.ggml.model", "rwkv");
        let reject = admit(&file).expect_err("unknown tokenizer must reject");
        assert_eq!(reject.axis, AdmissionAxis::Tokenizer);
        assert_eq!(reject.offending_value, "rwkv");
        assert!(reject.tensor.is_none());
    }

    #[test]
    fn rejects_missing_tokenizer() {
        let mut file = base_fixture();
        file.metadata.remove("tokenizer.ggml.model");
        let reject = admit(&file).expect_err("missing tokenizer must reject");
        assert_eq!(reject.axis, AdmissionAxis::Tokenizer);
        assert_eq!(reject.offending_value, "<absent>");
    }

    #[test]
    fn architecture_axis_checked_before_quant() {
        // A file failing on multiple axes reports architecture first (fixed order).
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "mixtral");
        file.tensors
            .push(tensor("blk.0.ffn_down.weight", GgufTensorType::Q2K));
        let reject = admit(&file).expect_err("must reject");
        assert_eq!(reject.axis, AdmissionAxis::Architecture);
    }

    #[test]
    fn reject_serializes_to_machine_readable_json() {
        let mut file = base_fixture();
        file.tensors
            .push(tensor("blk.12.ffn_down.weight", GgufTensorType::Q8K));
        let reject = admit(&file).expect_err("Q8_K must reject");
        let json = serde_json::to_value(&reject).expect("reject serializes");
        assert_eq!(json["axis"], "quant");
        assert_eq!(json["offending_value"], "Q8K");
        assert_eq!(json["tensor"], "blk.12.ffn_down.weight");
    }

    #[test]
    fn reject_converts_to_backend_error() {
        let mut file = base_fixture();
        set_meta(&mut file, "general.architecture", "mixtral");
        let reject = admit(&file).expect_err("must reject");
        let err: BackendError = reject.into();
        assert!(matches!(err, BackendError::UnsupportedGguf(_)));
    }
}
