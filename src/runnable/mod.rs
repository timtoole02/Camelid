//! Runnable lane — generic, breadth-first GGUF execution path (the CPU reference
//! forward is pure f32; qwen35 and lfm2 additionally carry resident GPU graphs
//! that consume packed quantized weights directly).
//!
//! The runnable lane is the promotion oracle for the supported lane: any GGUF in
//! the covered-set must either run deterministically or be **refused at admission**
//! with a precise, machine-readable reason. Refusal logic is as load-bearing as
//! execution logic — it is the evidence gate applied at the door
//! (`RUNNABLE_LANE_SPEC.md`, principle #2).
//!
//! Phase 1 delivers the admission gate (`admit`). Execution (dequant → parametric
//! decoder block → logits) lands in later phases.

pub mod admit;
pub mod dequant;
pub mod model;
pub mod smoke;
mod vision;
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
mod vision_cuda;

pub use admit::{admit, AdmissionAxis, AdmissionOk, AdmissionReject, TokenizerFamily};
pub use dequant::dequantize;
#[cfg(target_os = "macos")]
pub(crate) use model::lfm2_prefill_mm_enabled;
pub(crate) use model::Qwen35PromptCacheStats;
pub use model::RunnableModel;
pub use smoke::{headline_quant_of, oracle_qualified, smoke_admit, SmokeReport};
pub use vision::{PrismVisionEmbedding, PrismVisionProjector};
