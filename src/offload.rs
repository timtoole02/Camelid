//! Layer offloading — Phase 1: VRAM accounting and the layer→location split.
//!
//! A model too large for VRAM still runs with the GPU computing every layer: as
//! many transformer layers as fit live in VRAM (resident); the rest live in system
//! RAM and stream their weights to a GPU scratch buffer per forward pass. The split
//! is computed from detected free VRAM — no manual flag. This is a *capacity*
//! feature, not a speed one: offloaded layers pay a per-forward PCIe transfer.
//!
//! This module is the single source of truth for the split. It computes the byte
//! budget and the `layer_resident` map; it performs NO compute and NO streaming
//! (that is Phase 2+). Where a weight physically lives never changes the math, so
//! parity is unaffected by the split ratio.

use crate::gguf::GgufFile;
use crate::model::LlamaModelConfig;
use serde::Serialize;
use std::sync::{Mutex, OnceLock};

const MIB: u64 = 1024 * 1024;

/// Honest run labeling for layer offloading (Phase 4). A model run that streams some
/// layers from host RAM each forward is a *capacity* mode with materially lower
/// throughput than an all-resident run; this status makes that visible everywhere a
/// run is reported (load banner, bench record) so an offloaded run never reads like a
/// native one. `None`/`source == "none"` means the model is fully resident.
#[derive(Debug, Clone, Serialize)]
pub struct OffloadRunStatus {
    pub total_layers: usize,
    pub layers_resident: usize,
    pub layers_offloaded: usize,
    /// Bytes streamed host->device per offloaded layer, per forward.
    pub per_layer_bytes: u64,
    /// Detected free VRAM at engine-build time.
    pub free_vram_bytes: u64,
    /// Measured copy-stream peak H2D throughput (GB/s), if probed at build.
    pub pcie_gbps: Option<f64>,
    /// How the split was decided: "forced" (CAMELID_OFFLOAD_FORCE_LAYERS), "auto"
    /// (VRAM-driven), or "none" (fully resident).
    pub source: &'static str,
}

impl OffloadRunStatus {
    /// Fully-resident marker — nothing offloaded.
    pub fn resident(total_layers: usize, free_vram_bytes: u64) -> Self {
        Self {
            total_layers,
            layers_resident: total_layers,
            layers_offloaded: 0,
            per_layer_bytes: 0,
            free_vram_bytes,
            pcie_gbps: None,
            source: "none",
        }
    }

    /// One-line, human-facing summary for the load banner. Offloaded runs lead with a
    /// clear capacity-mode warning and the measured PCIe rate; resident runs are terse.
    pub fn describe(&self) -> String {
        if self.layers_offloaded == 0 {
            return format!(
                "[gpu] all {} layers resident in VRAM ({} MiB free) — full GPU speed",
                self.total_layers,
                self.free_vram_bytes / MIB,
            );
        }
        let pcie = match self.pcie_gbps {
            Some(g) => format!("{g:.1} GB/s H2D"),
            None => "unmeasured".to_string(),
        };
        format!(
            "[gpu] CAPACITY MODE: {}/{} layers offloaded to host RAM ({} resident), \
             streaming {} MiB/layer per forward over PCIe ({pcie}); expect REDUCED tok/s \
             vs fully-resident. split={}",
            self.layers_offloaded,
            self.total_layers,
            self.layers_resident,
            self.per_layer_bytes / MIB,
            self.source,
        )
    }
}

static OFFLOAD_RUN_STATUS: OnceLock<Mutex<Option<OffloadRunStatus>>> = OnceLock::new();

fn status_cell() -> &'static Mutex<Option<OffloadRunStatus>> {
    OFFLOAD_RUN_STATUS.get_or_init(|| Mutex::new(None))
}

/// Record the active engine's offload split (called once per resident-engine build).
pub fn set_offload_run_status(status: Option<OffloadRunStatus>) {
    *status_cell().lock().unwrap_or_else(|p| p.into_inner()) = status;
}

/// The active engine's offload split, if a resident engine has been built this process.
pub fn offload_run_status() -> Option<OffloadRunStatus> {
    status_cell()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

/// GPU-resident byte cost of one GGUF tensor. Q8_0 keeps its 34-byte footprint
/// (32 i8 quants plus f16 scale bits) in the resident SoA layout.
fn tensor_gpu_bytes(desc: &crate::gguf::GgufTensorDescriptor) -> u64 {
    use crate::gguf::GgufTensorType;
    let elements: u64 = desc.dimensions.iter().product();
    match desc.tensor_type {
        GgufTensorType::Q8_0 => elements.div_ceil(32) * 34,
        _ => desc.n_bytes,
    }
}

/// f32 KV-cache bytes for ALL layers at `context` positions. Every layer computes
/// on the GPU, so its K/V live in VRAM regardless of where its weights live.
fn kv_cache_bytes(config: &LlamaModelConfig, context: u64) -> u64 {
    let head_dim = config
        .attention_key_length
        .unwrap_or(config.embedding_length / config.attention_head_count.max(1))
        as u64;
    let n_kv = config.attention_head_count_kv as u64;
    let n_layers = config.block_count as u64;
    n_layers * n_kv * head_dim * 2 /*K,V*/ * 4 /*f32*/ * context
}

/// The planned split. `layer_resident[i] == true` means layer `i` lives in VRAM.
#[derive(Debug, Clone)]
pub struct OffloadPlan {
    pub total_layers: usize,
    pub layers_resident: usize,
    pub layers_offloaded: usize,
    pub layer_resident: Vec<bool>,
    // Byte budget the decision came from.
    pub free_vram: u64,
    pub kv_cache: u64,
    pub ends: u64,
    pub safety: u64,
    pub scratch: u64,
    pub per_layer_max: u64,
    pub total_layer_bytes: u64,
    pub resident_budget: u64,
    pub fits_fully: bool,
    /// Set when even the fixed overhead + one streaming buffer will not fit; the
    /// caller should refuse to run rather than thrash.
    pub error: Option<String>,
}

impl OffloadPlan {
    /// Build the split from a loaded GGUF (accurate: sums each layer's real tensor
    /// bytes and the resident "ends" — token embedding, output projection, norms).
    pub fn from_gguf(
        gguf: &GgufFile,
        config: &LlamaModelConfig,
        free_vram: u64,
        context: u64,
        safety_mb: u64,
    ) -> Self {
        let n_layers = config.block_count as usize;
        let mut layer_bytes = vec![0u64; n_layers];
        let mut ends = 0u64;
        for desc in &gguf.tensors {
            let bytes = tensor_gpu_bytes(desc);
            match parse_layer_index(&desc.name) {
                Some(i) if i < n_layers => layer_bytes[i] += bytes,
                _ => ends += bytes, // token_embd, output, output_norm, rope_freqs, ...
            }
        }
        Self::compute(layer_bytes, ends, config, free_vram, context, safety_mb)
    }

    /// Build the split from architecture dims alone (when the GGUF file isn't
    /// present). Estimates a uniform per-layer cost and assumes a separate (untied)
    /// output projection — the conservative case.
    pub fn from_dims(
        config: &LlamaModelConfig,
        free_vram: u64,
        context: u64,
        safety_mb: u64,
    ) -> Self {
        let hidden = config.embedding_length as u64;
        let ffn = config.feed_forward_length as u64;
        let n_heads = config.attention_head_count as u64;
        let n_kv = config.attention_head_count_kv as u64;
        let head_dim = config
            .attention_key_length
            .unwrap_or(config.embedding_length / config.attention_head_count.max(1))
            as u64;
        let vocab = config.vocab_size.unwrap_or(0) as u64;
        let q_width = n_heads * head_dim;
        let kv_width = n_kv * head_dim;
        // Q8_0: 34 bytes per 32-element block = 1.0625 B/elem. Norms are f32.
        let q8 = |params: u64| params.div_ceil(32) * 34;
        let attn = 2 * hidden * q_width + 2 * hidden * kv_width; // q,o + k,v
        let ffn_p = 3 * hidden * ffn; // gate,up,down
        let per_layer = q8(attn + ffn_p) + 2 * hidden * 4 /*attn_norm + ffn_norm, f32*/;
        let layer_bytes = vec![per_layer; config.block_count as usize];
        let ends = q8(vocab * hidden) /*token_embd*/ + q8(vocab * hidden) /*output*/ + hidden * 4;
        Self::compute(layer_bytes, ends, config, free_vram, context, safety_mb)
    }

    fn compute(
        layer_bytes: Vec<u64>,
        ends: u64,
        config: &LlamaModelConfig,
        free_vram: u64,
        context: u64,
        safety_mb: u64,
    ) -> Self {
        let total_layers = layer_bytes.len();
        let kv_cache = kv_cache_bytes(config, context);
        let safety = safety_mb * MIB;
        let per_layer_max = layer_bytes.iter().copied().max().unwrap_or(0);
        let total_layer_bytes: u64 = layer_bytes.iter().sum();
        let fixed = kv_cache + ends + safety; // always resident: KV + ends + safety

        // Fits fully? Then nothing offloads and no streaming scratch is needed.
        if free_vram >= fixed + total_layer_bytes {
            return OffloadPlan {
                total_layers,
                layers_resident: total_layers,
                layers_offloaded: 0,
                layer_resident: vec![true; total_layers],
                free_vram,
                kv_cache,
                ends,
                safety,
                scratch: 0,
                per_layer_max,
                total_layer_bytes,
                resident_budget: free_vram.saturating_sub(fixed),
                fits_fully: true,
                error: None,
            };
        }

        // Offloading: reserve one scratch buffer (the largest layer) to stream into.
        let scratch = per_layer_max;
        // Even fully offloaded (0 resident) needs fixed + scratch to fit.
        if free_vram < fixed + scratch {
            let need = fixed + scratch;
            let short = need - free_vram;
            return OffloadPlan {
                total_layers,
                layers_resident: 0,
                layers_offloaded: total_layers,
                layer_resident: vec![false; total_layers],
                free_vram,
                kv_cache,
                ends,
                safety,
                scratch,
                per_layer_max,
                total_layer_bytes,
                resident_budget: 0,
                fits_fully: false,
                error: Some(format!(
                    "model cannot run even fully offloaded: need {} MiB (KV {} + ends {} + safety {} + 1 layer scratch {}) but only {} MiB free — short by {} MiB",
                    need / MIB,
                    kv_cache / MIB,
                    ends / MIB,
                    safety / MIB,
                    scratch / MIB,
                    free_vram / MIB,
                    short / MIB,
                )),
            };
        }

        let resident_budget = free_vram - fixed - scratch;
        // Greedily make the first layers resident until the budget is exhausted.
        let mut used = 0u64;
        let mut resident = 0usize;
        for &b in &layer_bytes {
            if used + b <= resident_budget {
                used += b;
                resident += 1;
            } else {
                break;
            }
        }
        let layer_resident = (0..total_layers).map(|i| i < resident).collect();
        OffloadPlan {
            total_layers,
            layers_resident: resident,
            layers_offloaded: total_layers - resident,
            layer_resident,
            free_vram,
            kv_cache,
            ends,
            safety,
            scratch,
            per_layer_max,
            total_layer_bytes,
            resident_budget,
            fits_fully: false,
            error: None,
        }
    }

    /// Human-readable plan, for the load-time checkpoint and benchmark labels.
    pub fn describe(&self) -> String {
        if let Some(err) = &self.error {
            return format!("[offload] PLAN FAILED: {err}");
        }
        let avg_layer = if self.total_layers > 0 {
            self.total_layer_bytes / self.total_layers as u64
        } else {
            0
        };
        let mut s = String::new();
        s.push_str(&format!(
            "[offload] {} layers: {} resident in VRAM, {} offloaded to host{}\n",
            self.total_layers,
            self.layers_resident,
            self.layers_offloaded,
            if self.fits_fully {
                " (fits fully — no offload)"
            } else {
                ""
            }
        ));
        s.push_str("[offload] budget (MiB): ");
        s.push_str(&format!(
            "free {} = KV {} + ends {} + safety {} + scratch {} + resident-layers {} (of total-layers {})\n",
            self.free_vram / MIB,
            self.kv_cache / MIB,
            self.ends / MIB,
            self.safety / MIB,
            self.scratch / MIB,
            (self.layers_resident as u64 * avg_layer) / MIB,
            self.total_layer_bytes / MIB,
        ));
        s.push_str(&format!(
            "[offload] per-layer ~{} MiB | resident-weights budget {} MiB | {}",
            avg_layer / MIB,
            self.resident_budget / MIB,
            if self.layers_offloaded > 0 {
                format!(
                    "expect REDUCED tok/s: {} of {} layers stream from host each forward",
                    self.layers_offloaded, self.total_layers
                )
            } else {
                "fully resident".to_string()
            }
        ));
        s
    }
}

/// Parse the layer index from a GGUF tensor name like `blk.12.attn_q.weight`.
/// Returns `None` for non-layer tensors (token_embd, output, *_norm at the ends).
fn parse_layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let idx_str = rest.split('.').next()?;
    idx_str.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layer_index_handles_blk_and_ends() {
        assert_eq!(parse_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_index("blk.27.ffn_down.weight"), Some(27));
        assert_eq!(parse_layer_index("token_embd.weight"), None);
        assert_eq!(parse_layer_index("output.weight"), None);
        assert_eq!(parse_layer_index("output_norm.weight"), None);
    }

    #[test]
    fn resident_status_reads_as_full_speed() {
        let s = OffloadRunStatus::resident(28, 5 * 1024 * MIB);
        assert_eq!(s.layers_offloaded, 0);
        assert_eq!(s.source, "none");
        let line = s.describe();
        assert!(line.contains("all 28 layers resident"));
        assert!(line.contains("full GPU speed"));
        // A fully-resident run must never carry a capacity-mode warning.
        assert!(!line.to_lowercase().contains("capacity mode"));
        assert!(!line.contains("REDUCED"));
    }

    #[test]
    fn offloaded_status_labels_capacity_mode_with_pcie() {
        let s = OffloadRunStatus {
            total_layers: 28,
            layers_resident: 14,
            layers_offloaded: 14,
            per_layer_bytes: 108 * MIB,
            free_vram_bytes: 2 * 1024 * MIB,
            pcie_gbps: Some(11.9),
            source: "forced",
        };
        let line = s.describe();
        // The tradeoff must be visible: split, per-layer streaming, PCIe, and a warning.
        assert!(line.contains("CAPACITY MODE"));
        assert!(line.contains("14/28 layers offloaded"));
        assert!(line.contains("108 MiB/layer"));
        assert!(line.contains("11.9 GB/s"));
        assert!(line.contains("REDUCED tok/s"));
        assert!(line.contains("split=forced"));
    }
}
