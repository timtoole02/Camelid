//! Prism/Bonsai Qwen3-VL vision-projector loader and image preprocessing.
//!
//! The published 27B rows pair the Qwen3.5 language model with a separate
//! `qwen3vl_merger` GGUF. Large matrices remain page-backed so Metal can wrap
//! them with `newBufferWithBytesNoCopy` or CUDA can stream them one matrix at a
//! time; only biases, norms and the learned position table are decoded to f32
//! on the host.

use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(target_os = "macos", feature = "cuda"))]
use std::sync::Mutex;

use image::{imageops::FilterType, RgbImage};
#[cfg(not(target_os = "macos"))]
use rayon::prelude::*;

use crate::error::{BackendError, Result};
use crate::gguf::{self, GgufFile, GgufTensorDescriptor, GgufTensorType};
use crate::wire_mmap::WirePages;

use super::dequant;

#[derive(Clone)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct VisionMat {
    pub pages: Arc<WirePages>,
    pub tensor_type: GgufTensorType,
    pub input: usize,
    pub output: usize,
}

impl VisionMat {
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_weight(&self) -> Result<crate::metal::ResidentWeightBytes<'_>> {
        let format = match self.tensor_type {
            GgufTensorType::F32 => crate::metal::ResidentWeightFormat::DenseF32,
            GgufTensorType::F16 => crate::metal::ResidentWeightFormat::DenseF16,
            GgufTensorType::Q8_0 => crate::metal::ResidentWeightFormat::Q8_0,
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "Qwen3-VL Metal weight uses unsupported type {other:?}"
                )))
            }
        };
        Ok(crate::metal::ResidentWeightBytes::WirePages {
            format,
            pages: &self.pages,
        })
    }

    /// Apply this GGUF matrix to token-major f32 activations. The projector's
    /// dense F32/F16 and Q8_0 rows are decoded one matrix at a time, keeping the
    /// full 630 MiB projector page-backed while Rayon spreads the contractions
    /// across host cores. This is the portable fallback when neither the Metal
    /// graph nor the streaming CUDA graph can execute.
    #[cfg(not(target_os = "macos"))]
    fn linear(&self, input: &[f32], tokens: usize, label: &str) -> Result<Vec<f32>> {
        if input.len() != tokens * self.input {
            return Err(BackendError::InvalidTensorData(format!(
                "{label}: input has {} values, expected {}x{}",
                input.len(),
                tokens,
                self.input
            )));
        }
        let elements = self.input.checked_mul(self.output).ok_or_else(|| {
            BackendError::InvalidTensorData(format!("{label}: matrix geometry overflow"))
        })?;
        let weights = dequant::dequantize(self.tensor_type, self.pages.bytes(), elements, label)?;
        if weights.len() != elements {
            return Err(BackendError::InvalidTensorData(format!(
                "{label}: decoded {} weights, expected {elements}",
                weights.len()
            )));
        }
        // Row-major scheduling reuses one decoded weight row across every image
        // token before moving on. Image activations fit in cache while the full
        // projector does not, so this avoids streaming each matrix from RAM once
        // per token. Each individual dot retains the same element order.
        let row_major = (0..self.output)
            .into_par_iter()
            .flat_map_iter(|row| {
                let weight = &weights[row * self.input..(row + 1) * self.input];
                (0..tokens).map(move |token| {
                    let activation = &input[token * self.input..(token + 1) * self.input];
                    activation
                        .iter()
                        .zip(weight.iter())
                        .map(|(&x, &w)| x * w)
                        .sum::<f32>()
                })
            })
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; tokens * self.output];
        for row in 0..self.output {
            for token in 0..tokens {
                output[token * self.output + row] = row_major[row * tokens + token];
            }
        }
        Ok(output)
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct PrismVisionLayer {
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub qkv: VisionMat,
    pub qkv_bias: Vec<f32>,
    pub attn_output: VisionMat,
    pub attn_output_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
    pub ffn_up: VisionMat,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down: VisionMat,
    pub ffn_down_bias: Vec<f32>,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct PrismVisionModel {
    pub image_size: usize,
    pub patch_size: usize,
    pub merge: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub eps: f32,
    pub projection: usize,
    pub patch_0: VisionMat,
    pub patch_1: VisionMat,
    pub patch_bias: Vec<f32>,
    /// Learned 48x48 table, position-major `[2304, hidden]`.
    pub position: Vec<f32>,
    pub layers: Vec<PrismVisionLayer>,
    pub post_weight: Vec<f32>,
    pub post_bias: Vec<f32>,
    pub merger_0: VisionMat,
    pub merger_0_bias: Vec<f32>,
    pub merger_2: VisionMat,
    pub merger_2_bias: Vec<f32>,
}

#[derive(Clone)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct PrismVisionInput {
    /// Tile-major patches `[patch_count, 3 * patch_size * patch_size]`.
    pub patches: Vec<f32>,
    /// Learned position rows in the same tile-major order `[patch_count, hidden]`.
    pub position: Vec<f32>,
    pub patch_width: usize,
    pub patch_height: usize,
}

/// Image embeddings plus their merged 2-D token grid. The language decoder
/// needs both: embeddings feed the model while the grid drives Qwen3.5's
/// interleaved multimodal RoPE positions.
pub struct PrismVisionEmbedding {
    pub embeddings: Vec<Vec<f32>>,
    pub grid_width: usize,
    pub grid_height: usize,
}

/// Loaded Prism Qwen3-VL projector. macOS retains its native Metal graph;
/// CUDA hosts stream page-backed matrices through a bounded-VRAM graph, with a
/// portable CPU implementation as the final fallback.
pub struct PrismVisionProjector {
    model: PrismVisionModel,
    #[cfg(target_os = "macos")]
    encoder: Mutex<crate::metal::PrismVisionMetalEncoder>,
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    cuda: PrismVisionCudaRuntime,
}

/// Whether this build has any lane that can *decode* with image embeddings.
///
/// Encoding an image is always possible — there is a CPU reference path — but
/// generation with one is implemented only on Metal and CUDA. Health readiness
/// has to answer for the whole request, so it consults this rather than asking
/// the projector whether it can encode. Kept as a plain `const` rather than a
/// `cfg!` at the use site so one cfg-independent test can pin it on every CI
/// leg, including the legs that do not compile the arm it guards.
pub(crate) const VISION_DECODE_LANE_EXISTS: bool = cfg!(any(target_os = "macos", feature = "cuda"));

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
// The encoder is a single long-lived execution object behind a mutex. Keeping
// it inline avoids an extra heap indirection on every image request; the small
// control variants never move through a hot collection.
#[allow(clippy::large_enum_variant)]
enum PrismVisionCudaLane {
    Pending,
    Ready(super::vision_cuda::CudaVisionEncoder),
    Disabled,
    Failed(String),
}

/// CUDA execution is serialized, but liveness must remain observable while a
/// long image projection owns the lane mutex.
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
struct PrismVisionCudaRuntime {
    lane: Mutex<PrismVisionCudaLane>,
    ready: AtomicBool,
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
impl PrismVisionCudaRuntime {
    fn new() -> Self {
        Self {
            lane: Mutex::new(PrismVisionCudaLane::Pending),
            ready: AtomicBool::new(false),
        }
    }

    fn mark_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn prism_27b_windows_cuda_lane_selected_with(
    projection: usize,
    cuda_available: bool,
    qwen35_cuda: Option<&str>,
) -> bool {
    cfg!(target_os = "windows")
        && projection == 5_120
        && cuda_available
        && qwen35_cuda
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(true)
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn prism_27b_windows_cuda_lane_selected(projection: usize) -> bool {
    let qwen35_cuda = std::env::var("CAMELID_QWEN35_CUDA").ok();
    prism_27b_windows_cuda_lane_selected_with(
        projection,
        crate::cuda::is_available(),
        qwen35_cuda.as_deref(),
    )
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn prism_27b_cuda_failure_message(stage: &str, cause: &str, ordinal: usize) -> String {
    format!(
        "Prism 27B image support requires the Windows CUDA projector, but CUDA {stage} failed on device {ordinal}: {cause}. CPU fallback is disabled for this supported GPU lane. Verify CAMELID_CUDA_DEVICE selects the intended NVIDIA GPU, update the NVIDIA driver and CUDA toolkit/NVRTC, free enough VRAM, then reload the model or restart Camelid"
    )
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn prism_27b_cuda_failure_diagnostic(
    projection: usize,
    stage: &str,
    cause: &str,
) -> Option<String> {
    if !prism_27b_windows_cuda_lane_selected(projection) {
        return None;
    }
    let ordinal = crate::cuda::selected_device_ordinal();
    Some(prism_27b_cuda_failure_message(stage, cause, ordinal))
}

impl PrismVisionProjector {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model = PrismVisionModel::load(path)?;
        #[cfg(target_os = "macos")]
        {
            let encoder = model.metal_encoder()?;
            Ok(Self {
                model,
                encoder: Mutex::new(encoder),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                model,
                #[cfg(feature = "cuda")]
                cuda: PrismVisionCudaRuntime::new(),
            })
        }
    }

    pub fn projection_dim(&self) -> usize {
        self.model.projection
    }

    /// Initialize the execution backend without encoding an image. This lets a
    /// serving layer distinguish a loaded projector file from a projector that
    /// can actually execute. CPU-only builds remain ready by construction.
    pub fn ensure_backend_ready(&self) -> Result<()> {
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            let mut lane = self.cuda.lane.lock().map_err(|_| {
                self.cuda.mark_ready(false);
                BackendError::InvalidTensorData("Prism vision CUDA mutex poisoned".into())
            })?;
            self.initialize_cuda_lane(&mut lane)?;
        }
        // Must stay `Ok` here: this runs inside the serve-runtime load, and
        // failing would make the row unloadable for text too. Say why the image
        // affordance will be missing instead of leaving the operator guessing.
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            eprintln!(
                "[qwen3vl] projector loaded, but this build has no Metal or CUDA lane to decode \
                 with image embeddings; image input stays unavailable and text is unaffected"
            );
        }
        Ok(())
    }

    /// Non-blocking readiness query for health/liveness endpoints. Image
    /// projection owns `cuda` for the duration of GPU execution; consulting
    /// that mutex here would let an ordinary image request stall health polls.
    ///
    /// This answers for the whole image request, not just the projector: a
    /// build can encode an image on CPU and still have no lane that can decode
    /// with it, so readiness has to account for [`VISION_DECODE_LANE_EXISTS`].
    pub fn backend_ready(&self) -> bool {
        // A projector that can only encode is not readiness. CPU encode works
        // everywhere, but generation with the result errors out unconditionally
        // off Metal and CUDA, so a build with no decode lane must report false
        // however healthy its projector is — reporting true would advertise a
        // capability `src/api/contract.rs` scopes to
        // `prism_single_image_data_url_metal_or_windows_cuda`.
        if !VISION_DECODE_LANE_EXISTS {
            return false;
        }
        // Truthful because both `mark_ready` call sites in `initialize_cuda_lane`
        // now report the lane's actual final state rather than assuming success.
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            self.cuda.is_ready()
        }
        // macOS. Load already proved the Metal graph: `PrismVisionProjector::load`
        // fails unless `model.metal_encoder()` succeeded, and constructing that
        // encoder forces the process-global pipeline cache, so a projector we
        // hold has its weights uploaded and its pipelines compiled. This is a
        // claim about the lane existing, not a promise that every image encodes
        // — per-request geometry can still be rejected.
        #[cfg(not(all(not(target_os = "macos"), feature = "cuda")))]
        {
            true
        }
    }

    pub fn encode_image(
        &self,
        path: impl AsRef<Path>,
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionEmbedding> {
        let input = self.model.preprocess(path, min_tokens, max_tokens)?;
        self.encode_preprocessed(input)
    }

    /// Decode an uploaded PNG/JPEG directly from memory and project it without
    /// writing user content to a temporary file.
    pub fn encode_image_bytes(
        &self,
        bytes: &[u8],
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionEmbedding> {
        let mut reader = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| BackendError::InvalidTensorData(format!("detect image: {error}")))?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(16_384);
        limits.max_image_height = Some(16_384);
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        let image = reader
            .decode()
            .map_err(|error| BackendError::InvalidTensorData(format!("decode image: {error}")))?
            .to_rgb8();
        let input = self.model.preprocess_rgb(image, min_tokens, max_tokens)?;
        self.encode_preprocessed(input)
    }

    fn encode_preprocessed(&self, input: PrismVisionInput) -> Result<PrismVisionEmbedding> {
        #[cfg(target_os = "macos")]
        {
            let mut encoder = self.encoder.lock().map_err(|_| {
                BackendError::InvalidTensorData("Prism vision Metal mutex poisoned".into())
            })?;
            let embeddings = encoder
                .encode(
                    &input.patches,
                    &input.position,
                    input.patch_width,
                    input.patch_height,
                )
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(
                        "Prism vision Metal graph refused the preprocessed image".into(),
                    )
                })?;
            let grid_width = input.patch_width / self.model.merge;
            let grid_height = input.patch_height / self.model.merge;
            if embeddings.len() != grid_width * grid_height {
                return Err(BackendError::InvalidTensorData(
                    "Prism vision encoder returned the wrong image-token count".into(),
                ));
            }
            Ok(PrismVisionEmbedding {
                embeddings,
                grid_width,
                grid_height,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            #[cfg(feature = "cuda")]
            {
                let mut lane = self.cuda.lane.lock().map_err(|_| {
                    self.cuda.mark_ready(false);
                    BackendError::InvalidTensorData("Prism vision CUDA mutex poisoned".into())
                })?;
                self.initialize_cuda_lane(&mut lane)?;
                if let PrismVisionCudaLane::Ready(encoder) = &mut *lane {
                    match encoder.encode(&self.model, &input) {
                        Ok(embeddings) => {
                            let grid_width = input.patch_width / self.model.merge;
                            let grid_height = input.patch_height / self.model.merge;
                            if embeddings.len() != grid_width * grid_height {
                                return Err(BackendError::InvalidTensorData(
                                    "Prism vision CUDA encoder returned the wrong image-token count"
                                        .into(),
                                ));
                            }
                            return Ok(PrismVisionEmbedding {
                                embeddings,
                                grid_width,
                                grid_height,
                            });
                        }
                        Err(error) => {
                            if let Some(diagnostic) = prism_27b_cuda_failure_diagnostic(
                                self.model.projection,
                                "execution",
                                &error,
                            ) {
                                self.cuda.mark_ready(false);
                                *lane = PrismVisionCudaLane::Failed(diagnostic.clone());
                                return Err(BackendError::UnsupportedGguf(diagnostic));
                            }
                            eprintln!(
                                "[qwen3vl] CUDA projector failed ({error}); using CPU fallback"
                            );
                            *lane = PrismVisionCudaLane::Disabled;
                        }
                    }
                }
            }
            self.model.encode_cpu(input)
        }
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    fn initialize_cuda_lane(&self, lane: &mut PrismVisionCudaLane) -> Result<()> {
        if matches!(*lane, PrismVisionCudaLane::Pending) {
            if cfg!(target_os = "windows")
                && self.model.projection == 5_120
                && !prism_27b_windows_cuda_lane_selected(self.model.projection)
            {
                eprintln!(
                    "[qwen3vl] Windows CUDA projector not selected or unavailable; using CPU"
                );
                *lane = PrismVisionCudaLane::Disabled;
                // CPU encode still works, but nothing on this build can decode
                // with the result, so readiness must reflect the lane, not the
                // fact that we returned without an error.
                self.cuda.mark_ready(false);
                return Ok(());
            }
            *lane = match super::vision_cuda::CudaVisionEncoder::new() {
                Ok(encoder) => {
                    eprintln!("[qwen3vl] streaming CUDA projector active (bounded VRAM)");
                    PrismVisionCudaLane::Ready(encoder)
                }
                Err(error) => {
                    if let Some(diagnostic) = prism_27b_cuda_failure_diagnostic(
                        self.model.projection,
                        "initialization",
                        &error,
                    ) {
                        self.cuda.mark_ready(false);
                        *lane = PrismVisionCudaLane::Failed(diagnostic.clone());
                        return Err(BackendError::UnsupportedGguf(diagnostic));
                    }
                    eprintln!("[qwen3vl] CUDA projector unavailable ({error}); using CPU");
                    PrismVisionCudaLane::Disabled
                }
            };
        }
        if let PrismVisionCudaLane::Failed(diagnostic) = lane {
            self.cuda.mark_ready(false);
            return Err(BackendError::UnsupportedGguf(diagnostic.clone()));
        }
        // Report the lane's actual state. Reaching here is not proof of success:
        // a lane disabled during a previous encode is no longer `Pending`, so it
        // skips the initialization block above and arrives here unchanged.
        self.cuda
            .mark_ready(matches!(*lane, PrismVisionCudaLane::Ready(_)));
        Ok(())
    }
}

#[cfg(all(test, target_os = "macos"))]
impl PrismVisionInput {
    pub(crate) fn patch_count(&self) -> usize {
        self.patch_width * self.patch_height
    }

    pub(crate) fn output_tokens(&self, merge: usize) -> usize {
        self.patch_count() / (merge * merge)
    }
}

impl PrismVisionModel {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let gguf = gguf::read_metadata(path)?;
        if gguf.architecture() != Some("clip")
            || gguf.metadata_string("clip.projector_type") != Some("qwen3vl_merger")
            || gguf.metadata_bool("clip.has_vision_encoder") != Some(true)
        {
            return Err(BackendError::InvalidModelMetadata(
                "vision GGUF must be clip/qwen3vl_merger with a vision encoder".into(),
            ));
        }
        let get_u32 = |key: &str| {
            gguf.metadata_u32(key).ok_or_else(|| {
                BackendError::InvalidModelMetadata(format!("vision GGUF missing {key}"))
            })
        };
        let image_size = get_u32("clip.vision.image_size")? as usize;
        let patch_size = get_u32("clip.vision.patch_size")? as usize;
        let merge = get_u32("clip.vision.spatial_merge_size")? as usize;
        let hidden = get_u32("clip.vision.embedding_length")? as usize;
        let ffn = get_u32("clip.vision.feed_forward_length")? as usize;
        let heads = get_u32("clip.vision.attention.head_count")? as usize;
        let blocks = get_u32("clip.vision.block_count")? as usize;
        let projection = get_u32("clip.vision.projection_dim")? as usize;
        let eps = gguf
            .metadata_f32("clip.vision.attention.layer_norm_epsilon")
            .ok_or_else(|| {
                BackendError::InvalidModelMetadata(
                    "vision GGUF missing attention LayerNorm epsilon".into(),
                )
            })?;
        if image_size == 0
            || patch_size == 0
            || merge != 2
            || hidden == 0
            || ffn == 0
            || heads == 0
            || !hidden.is_multiple_of(heads)
            || blocks == 0
            || projection == 0
        {
            return Err(BackendError::InvalidModelMetadata(
                "unsupported or degenerate qwen3vl_merger geometry".into(),
            ));
        }

        let mut file = File::open(path).map_err(|source| BackendError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let patch_0 = load_mat(&mut file, &gguf, "v.patch_embd.weight")?;
        let patch_1 = load_mat(&mut file, &gguf, "v.patch_embd.weight.1")?;
        let patch_bias = load_vec(&mut file, &gguf, "v.patch_embd.bias")?;
        let position = load_vec(&mut file, &gguf, "v.position_embd.weight")?;
        if patch_0.input != 3 * patch_size * patch_size
            || patch_0.output != hidden
            || patch_1.input != patch_0.input
            || patch_1.output != hidden
            || patch_bias.len() != hidden
            || position.len() != hidden * (image_size / patch_size).pow(2)
        {
            return Err(BackendError::InvalidTensorData(
                "qwen3vl patch/position tensor geometry mismatch".into(),
            ));
        }

        let mut layers = Vec::with_capacity(blocks);
        for layer in 0..blocks {
            let n = |suffix: &str| format!("v.blk.{layer}.{suffix}");
            layers.push(PrismVisionLayer {
                ln1_weight: load_vec(&mut file, &gguf, &n("ln1.weight"))?,
                ln1_bias: load_vec(&mut file, &gguf, &n("ln1.bias"))?,
                qkv: load_mat(&mut file, &gguf, &n("attn_qkv.weight"))?,
                qkv_bias: load_vec(&mut file, &gguf, &n("attn_qkv.bias"))?,
                attn_output: load_mat(&mut file, &gguf, &n("attn_out.weight"))?,
                attn_output_bias: load_vec(&mut file, &gguf, &n("attn_out.bias"))?,
                ln2_weight: load_vec(&mut file, &gguf, &n("ln2.weight"))?,
                ln2_bias: load_vec(&mut file, &gguf, &n("ln2.bias"))?,
                ffn_up: load_mat(&mut file, &gguf, &n("ffn_up.weight"))?,
                ffn_up_bias: load_vec(&mut file, &gguf, &n("ffn_up.bias"))?,
                ffn_down: load_mat(&mut file, &gguf, &n("ffn_down.weight"))?,
                ffn_down_bias: load_vec(&mut file, &gguf, &n("ffn_down.bias"))?,
            });
        }
        for (index, layer) in layers.iter().enumerate() {
            let vector_lengths_ok = layer.ln1_weight.len() == hidden
                && layer.ln1_bias.len() == hidden
                && layer.qkv_bias.len() == 3 * hidden
                && layer.attn_output_bias.len() == hidden
                && layer.ln2_weight.len() == hidden
                && layer.ln2_bias.len() == hidden
                && layer.ffn_up_bias.len() == ffn
                && layer.ffn_down_bias.len() == hidden;
            let matrix_shapes_ok = layer.qkv.input == hidden
                && layer.qkv.output == 3 * hidden
                && layer.attn_output.input == hidden
                && layer.attn_output.output == hidden
                && layer.ffn_up.input == hidden
                && layer.ffn_up.output == ffn
                && layer.ffn_down.input == ffn
                && layer.ffn_down.output == hidden;
            if !vector_lengths_ok || !matrix_shapes_ok {
                return Err(BackendError::InvalidTensorData(format!(
                    "qwen3vl layer {index} tensor geometry mismatch"
                )));
            }
        }

        let post_weight = load_vec(&mut file, &gguf, "v.post_ln.weight")?;
        let post_bias = load_vec(&mut file, &gguf, "v.post_ln.bias")?;
        let merger_0 = load_mat(&mut file, &gguf, "mm.0.weight")?;
        let merger_0_bias = load_vec(&mut file, &gguf, "mm.0.bias")?;
        let merger_2 = load_mat(&mut file, &gguf, "mm.2.weight")?;
        let merger_2_bias = load_vec(&mut file, &gguf, "mm.2.bias")?;
        let merged = hidden * merge * merge;
        if post_weight.len() != hidden
            || post_bias.len() != hidden
            || merger_0.input != merged
            || merger_0.output != merged
            || merger_0_bias.len() != merged
            || merger_2.input != merged
            || merger_2.output != projection
            || merger_2_bias.len() != projection
        {
            return Err(BackendError::InvalidTensorData(
                "qwen3vl post-norm/merger tensor geometry mismatch".into(),
            ));
        }

        Ok(Self {
            image_size,
            patch_size,
            merge,
            hidden,
            ffn,
            heads,
            eps,
            projection,
            patch_0,
            patch_1,
            patch_bias,
            position,
            layers,
            post_weight,
            post_bias,
            merger_0,
            merger_0_bias,
            merger_2,
            merger_2_bias,
        })
    }

    /// Decode, smart-resize, normalize and patchify one image. `max_tokens`
    /// controls the Qwen3-VL 32x32-pixel output-token cap used by Bonsai-demo.
    pub(crate) fn preprocess(
        &self,
        path: impl AsRef<Path>,
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionInput> {
        let image = image::ImageReader::open(path.as_ref())
            .map_err(|source| BackendError::Io {
                path: path.as_ref().to_path_buf(),
                source,
            })?
            .decode()
            .map_err(|error| BackendError::InvalidTensorData(format!("decode image: {error}")))?
            .to_rgb8();
        self.preprocess_rgb(image, min_tokens, max_tokens)
    }

    pub(crate) fn preprocess_rgb(
        &self,
        image: RgbImage,
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<PrismVisionInput> {
        let align = self.patch_size * self.merge;
        let (width, height) = smart_resize(
            image.width() as usize,
            image.height() as usize,
            align,
            min_tokens.max(1) * align * align,
            max_tokens.max(min_tokens.max(1)) * align * align,
        );
        let resized =
            image::imageops::resize(&image, width as u32, height as u32, FilterType::Triangle);
        let patch_width = width / self.patch_size;
        let patch_height = height / self.patch_size;
        if !patch_width.is_multiple_of(self.merge) || !patch_height.is_multiple_of(self.merge) {
            return Err(BackendError::InvalidTensorData(
                "smart-resized image is not aligned to the spatial merge".into(),
            ));
        }
        let patch_elements = 3 * self.patch_size * self.patch_size;
        let mut patches = Vec::with_capacity(patch_width * patch_height * patch_elements);
        for tile_y in (0..patch_height).step_by(self.merge) {
            for tile_x in (0..patch_width).step_by(self.merge) {
                for dy in 0..self.merge {
                    for dx in 0..self.merge {
                        let patch_y = tile_y + dy;
                        let patch_x = tile_x + dx;
                        for channel in 0..3 {
                            for y in 0..self.patch_size {
                                for x in 0..self.patch_size {
                                    let pixel = resized.get_pixel(
                                        (patch_x * self.patch_size + x) as u32,
                                        (patch_y * self.patch_size + y) as u32,
                                    );
                                    patches.push(pixel[channel] as f32 * (2.0 / 255.0) - 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
        let position = interpolate_positions_tile_major(
            &self.position,
            self.hidden,
            self.image_size / self.patch_size,
            patch_width,
            patch_height,
            self.merge,
        );
        Ok(PrismVisionInput {
            patches,
            position,
            patch_width,
            patch_height,
        })
    }

    /// Portable reference implementation of the Qwen3-VL projector graph used
    /// by the Prism 27B image row. Its operation order mirrors the Metal graph:
    /// dual patch projection, 27 bidirectional transformer blocks, post norm,
    /// spatial merge and the two-layer language projection.
    #[cfg(not(target_os = "macos"))]
    fn encode_cpu(&self, input: PrismVisionInput) -> Result<PrismVisionEmbedding> {
        let tokens = input
            .patch_width
            .checked_mul(input.patch_height)
            .ok_or_else(|| {
                BackendError::InvalidTensorData("vision patch geometry overflow".into())
            })?;
        if tokens == 0
            || tokens > 4096
            || input.patches.len() != tokens * self.patch_0.input
            || input.position.len() != tokens * self.hidden
            || !input.patch_width.is_multiple_of(self.merge)
            || !input.patch_height.is_multiple_of(self.merge)
        {
            return Err(BackendError::InvalidTensorData(
                "Qwen3-VL CPU projector refused the preprocessed image geometry".into(),
            ));
        }
        let head_dim = self.hidden / self.heads;
        if !head_dim.is_multiple_of(2) {
            return Err(BackendError::InvalidModelMetadata(
                "Qwen3-VL vision head width must be even".into(),
            ));
        }

        let patch_first = self
            .patch_0
            .linear(&input.patches, tokens, "vision patch 0")?;
        let patch_second = self
            .patch_1
            .linear(&input.patches, tokens, "vision patch 1")?;
        let mut hidden = patch_first;
        hidden
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                *value += patch_second[index]
                    + self.patch_bias[index % self.hidden]
                    + input.position[index];
            });

        let (cosine, sine) =
            vision_rope_tables(input.patch_width, input.patch_height, self.merge, head_dim);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let normalized = vision_layer_norm(
                &hidden,
                tokens,
                self.hidden,
                &layer.ln1_weight,
                &layer.ln1_bias,
                self.eps,
            );
            let mut qkv = layer.qkv.linear(
                &normalized,
                tokens,
                &format!("vision layer {layer_index} qkv"),
            )?;
            add_bias(&mut qkv, &layer.qkv_bias);
            apply_vision_rope(
                &mut qkv,
                &cosine,
                &sine,
                tokens,
                self.hidden,
                self.heads,
                head_dim,
            );
            let attention = vision_attention(&qkv, tokens, self.hidden, self.heads, head_dim);
            let projected = layer.attn_output.linear(
                &attention,
                tokens,
                &format!("vision layer {layer_index} attention output"),
            )?;
            let mut after_attention = hidden;
            add_bias_residual(&mut after_attention, &projected, &layer.attn_output_bias);

            let normalized = vision_layer_norm(
                &after_attention,
                tokens,
                self.hidden,
                &layer.ln2_weight,
                &layer.ln2_bias,
                self.eps,
            );
            let mut ffn = layer.ffn_up.linear(
                &normalized,
                tokens,
                &format!("vision layer {layer_index} ffn up"),
            )?;
            add_bias_gelu(&mut ffn, &layer.ffn_up_bias);
            let projected = layer.ffn_down.linear(
                &ffn,
                tokens,
                &format!("vision layer {layer_index} ffn down"),
            )?;
            hidden = after_attention;
            add_bias_residual(&mut hidden, &projected, &layer.ffn_down_bias);
        }

        let normalized = vision_layer_norm(
            &hidden,
            tokens,
            self.hidden,
            &self.post_weight,
            &self.post_bias,
            self.eps,
        );
        let output_tokens = tokens / (self.merge * self.merge);
        let merged = self.hidden * self.merge * self.merge;
        // Preprocessing is tile-major, so each consecutive group of four patch
        // rows is already one flattened 2x2 merger row.
        debug_assert_eq!(normalized.len(), output_tokens * merged);
        let mut merger_hidden =
            self.merger_0
                .linear(&normalized, output_tokens, "vision merger 0")?;
        add_bias_gelu(&mut merger_hidden, &self.merger_0_bias);
        let mut projected =
            self.merger_2
                .linear(&merger_hidden, output_tokens, "vision merger 2")?;
        add_bias(&mut projected, &self.merger_2_bias);
        if projected.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::InvalidTensorData(
                "Qwen3-VL CPU projector produced non-finite embeddings".into(),
            ));
        }
        Ok(PrismVisionEmbedding {
            embeddings: projected
                .chunks_exact(self.projection)
                .map(<[f32]>::to_vec)
                .collect(),
            grid_width: input.patch_width / self.merge,
            grid_height: input.patch_height / self.merge,
        })
    }

    /// Resolve the page-backed projector into a resident Metal encoder. The
    /// returned object holds all small bias/norm buffers and reuses the GGUF
    /// pages for every large matrix without copying them into a second host
    /// allocation.
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_encoder(&self) -> Result<crate::metal::PrismVisionMetalEncoder> {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(crate::metal::PrismVisionMetalLayerInput {
                    ln1_weight: &layer.ln1_weight,
                    ln1_bias: &layer.ln1_bias,
                    qkv: layer.qkv.metal_weight()?,
                    qkv_bias: &layer.qkv_bias,
                    attn_output: layer.attn_output.metal_weight()?,
                    attn_output_bias: &layer.attn_output_bias,
                    ln2_weight: &layer.ln2_weight,
                    ln2_bias: &layer.ln2_bias,
                    ffn_up: layer.ffn_up.metal_weight()?,
                    ffn_up_bias: &layer.ffn_up_bias,
                    ffn_down: layer.ffn_down.metal_weight()?,
                    ffn_down_bias: &layer.ffn_down_bias,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        crate::metal::PrismVisionMetalEncoder::new(
            crate::metal::PrismVisionMetalConfig {
                patch_input: 3 * self.patch_size * self.patch_size,
                hidden: self.hidden,
                ffn: self.ffn,
                heads: self.heads,
                merge: self.merge,
                projection: self.projection,
                eps: self.eps,
            },
            self.patch_0.metal_weight()?,
            self.patch_1.metal_weight()?,
            &self.patch_bias,
            &layers,
            &self.post_weight,
            &self.post_bias,
            self.merger_0.metal_weight()?,
            &self.merger_0_bias,
            self.merger_2.metal_weight()?,
            &self.merger_2_bias,
        )
        .ok_or_else(|| {
            BackendError::InvalidTensorData(
                "Qwen3-VL projector could not initialize its resident Metal graph".into(),
            )
        })
    }
}

#[cfg(not(target_os = "macos"))]
fn vision_layer_norm(
    input: &[f32],
    tokens: usize,
    width: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), tokens * width);
    let mut output = vec![0.0f32; input.len()];
    output
        .par_chunks_mut(width)
        .zip(input.par_chunks(width))
        .for_each(|(out, row)| {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row
                .iter()
                .map(|&value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / width as f32;
            let inverse = 1.0 / (variance + eps).sqrt();
            for index in 0..width {
                out[index] = (row[index] - mean) * inverse * weight[index] + bias[index];
            }
        });
    output
}

#[cfg(not(target_os = "macos"))]
fn add_bias(values: &mut [f32], bias: &[f32]) {
    values
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| *value += bias[index % bias.len()]);
}

#[cfg(not(target_os = "macos"))]
fn add_bias_residual(residual: &mut [f32], projected: &[f32], bias: &[f32]) {
    residual
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| *value += projected[index] + bias[index % bias.len()]);
}

#[cfg(not(target_os = "macos"))]
fn add_bias_gelu(values: &mut [f32], bias: &[f32]) {
    values
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let x = *value + bias[index % bias.len()];
            let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            *value = 0.5 * x * (1.0 + inner.clamp(-15.0, 15.0).tanh());
        });
}

#[cfg(not(target_os = "macos"))]
fn vision_rope_tables(
    patch_width: usize,
    patch_height: usize,
    merge: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let section = half / 2;
    let mut cosine = Vec::with_capacity(patch_width * patch_height * half);
    let mut sine = Vec::with_capacity(cosine.capacity());
    for tile_y in (0..patch_height).step_by(merge) {
        for tile_x in (0..patch_width).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    for pair in 0..half {
                        let (position, dimension) = if pair < section {
                            ((tile_y + dy) as f32, pair)
                        } else {
                            ((tile_x + dx) as f32, pair - section)
                        };
                        let angle =
                            position * 10_000.0f32.powf(-2.0 * dimension as f32 / half as f32);
                        cosine.push(angle.cos());
                        sine.push(angle.sin());
                    }
                }
            }
        }
    }
    (cosine, sine)
}

#[cfg(not(target_os = "macos"))]
fn apply_vision_rope(
    qkv: &mut [f32],
    cosine: &[f32],
    sine: &[f32],
    tokens: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    qkv.par_chunks_mut(3 * hidden)
        .enumerate()
        .for_each(|(token, row)| {
            debug_assert!(token < tokens);
            for qk in 0..2 {
                for head in 0..heads {
                    let base = qk * hidden + head * head_dim;
                    for pair in 0..half {
                        let x0 = row[base + pair];
                        let x1 = row[base + pair + half];
                        let c = cosine[token * half + pair];
                        let s = sine[token * half + pair];
                        row[base + pair] = x0 * c - x1 * s;
                        row[base + pair + half] = x0 * s + x1 * c;
                    }
                }
            }
        });
}

#[cfg(not(target_os = "macos"))]
fn vision_attention(
    qkv: &[f32],
    tokens: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; tokens * hidden];
    output
        .par_chunks_mut(hidden)
        .enumerate()
        .for_each(|(query_token, out)| {
            for head in 0..heads {
                let qbase = query_token * 3 * hidden + head * head_dim;
                let mut scores = Vec::with_capacity(tokens);
                let mut maximum = f32::NEG_INFINITY;
                for key_token in 0..tokens {
                    let kbase = key_token * 3 * hidden + hidden + head * head_dim;
                    let mut score = 0.0f32;
                    for dimension in 0..head_dim {
                        score += qkv[qbase + dimension] * qkv[kbase + dimension];
                    }
                    score *= scale;
                    maximum = maximum.max(score);
                    scores.push(score);
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                    sum += *score;
                }
                let inverse = 1.0 / sum;
                for dimension in 0..head_dim {
                    let mut value = 0.0f32;
                    for (key_token, &score) in scores.iter().enumerate() {
                        let vbase = key_token * 3 * hidden + 2 * hidden + head * head_dim;
                        value += score * inverse * qkv[vbase + dimension];
                    }
                    out[head * head_dim + dimension] = value;
                }
            }
        });
    output
}

fn find<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a GgufTensorDescriptor> {
    gguf.tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| BackendError::TensorNotFound(name.into()))
}

fn load_mat(file: &mut File, gguf: &GgufFile, name: &str) -> Result<VisionMat> {
    let descriptor = find(gguf, name)?;
    if descriptor.dimensions.len() < 2 {
        return Err(BackendError::InvalidTensorData(format!(
            "vision matrix {name} is not rank >= 2"
        )));
    }
    if !matches!(
        descriptor.tensor_type,
        GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::Q8_0
    ) {
        return Err(BackendError::UnsupportedTensorType(format!(
            "vision matrix {name} uses {:?}",
            descriptor.tensor_type
        )));
    }
    let output = *descriptor.dimensions.last().unwrap() as usize;
    let input = descriptor.dimensions[..descriptor.dimensions.len() - 1]
        .iter()
        .product::<u64>() as usize;
    let byte_len = usize::try_from(descriptor.n_bytes).map_err(|_| {
        BackendError::InvalidTensorData(format!("vision matrix {name} is too large"))
    })?;
    Ok(VisionMat {
        pages: WirePages::read_from_file(file, descriptor.absolute_offset, byte_len)?,
        tensor_type: descriptor.tensor_type,
        input,
        output,
    })
}

fn load_vec(file: &mut File, gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let descriptor = find(gguf, name)?;
    let mut bytes = vec![0u8; descriptor.n_bytes as usize];
    file.seek(SeekFrom::Start(descriptor.absolute_offset))
        .map_err(|source| BackendError::Io {
            path: gguf.path.clone(),
            source,
        })?;
    file.read_exact(&mut bytes)
        .map_err(|source| BackendError::Io {
            path: gguf.path.clone(),
            source,
        })?;
    let elements = descriptor.dimensions.iter().product::<u64>() as usize;
    dequant::dequantize(descriptor.tensor_type, &bytes, elements, name)
}

fn round_to(value: f64, factor: usize) -> usize {
    ((value / factor as f64).round() as usize).max(1) * factor
}

fn smart_resize(
    width: usize,
    height: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let mut out_w = round_to(width as f64, align);
    let mut out_h = round_to(height as f64, align);
    if out_w * out_h > max_pixels {
        let beta = ((width * height) as f64 / max_pixels as f64).sqrt();
        out_w = (((width as f64 / beta) / align as f64).floor() as usize).max(1) * align;
        out_h = (((height as f64 / beta) / align as f64).floor() as usize).max(1) * align;
    } else if out_w * out_h < min_pixels {
        let beta = (min_pixels as f64 / (width * height).max(1) as f64).sqrt();
        out_w = (((width as f64 * beta) / align as f64).ceil() as usize).max(1) * align;
        out_h = (((height as f64 * beta) / align as f64).ceil() as usize).max(1) * align;
    }
    (out_w, out_h)
}

fn interpolate_positions_tile_major(
    source: &[f32],
    hidden: usize,
    source_side: usize,
    width: usize,
    height: usize,
    merge: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(width * height * hidden);
    for tile_y in (0..height).step_by(merge) {
        for tile_x in (0..width).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    let y = tile_y + dy;
                    let x = tile_x + dx;
                    let source_y = ((y as f32 + 0.5) * source_side as f32 / height as f32 - 0.5)
                        .clamp(0.0, (source_side - 1) as f32);
                    let source_x = ((x as f32 + 0.5) * source_side as f32 / width as f32 - 0.5)
                        .clamp(0.0, (source_side - 1) as f32);
                    let y0 = source_y.floor() as usize;
                    let x0 = source_x.floor() as usize;
                    let y1 = (y0 + 1).min(source_side - 1);
                    let x1 = (x0 + 1).min(source_side - 1);
                    let fy = source_y - y0 as f32;
                    let fx = source_x - x0 as f32;
                    for channel in 0..hidden {
                        let at = |py: usize, px: usize| {
                            source[(py * source_side + px) * hidden + channel]
                        };
                        let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
                        let bottom = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
                        output.push(top + (bottom - top) * fy);
                    }
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deliberately cfg-independent so it runs on every CI leg, including the
    /// ones that never compile the CPU-only arm of `backend_ready`. The defect
    /// this guards against — advertising vision readiness on a build with no
    /// decode lane — lived only in a configuration no CI leg builds, so a
    /// cfg-gated test would not have caught it.
    #[test]
    fn vision_decode_lane_constant_agrees_with_the_lanes_that_exist() {
        assert_eq!(
            VISION_DECODE_LANE_EXISTS,
            cfg!(any(target_os = "macos", feature = "cuda")),
            "VISION_DECODE_LANE_EXISTS must track the cfgs that actually \
             implement image decode; readiness is derived from it"
        );
    }

    #[test]
    fn qwen3vl_smart_resize_is_merge_aligned_and_bounded() {
        assert_eq!(
            smart_resize(1442, 720, 32, 8 * 1024, 1024 * 1024),
            (1440, 704)
        );
        assert_eq!(smart_resize(32, 32, 32, 1024, 1024), (32, 32));
        let (width, height) = smart_resize(12_000, 9_000, 32, 8 * 1024, 1024 * 1024);
        assert_eq!(width % 32, 0);
        assert_eq!(height % 32, 0);
        assert!(width * height <= 1024 * 1024);
    }

    #[cfg(all(target_os = "windows", feature = "cuda"))]
    #[test]
    fn cuda_readiness_does_not_wait_for_the_projection_lane_mutex() {
        let runtime = std::sync::Arc::new(PrismVisionCudaRuntime::new());
        runtime.mark_ready(true);
        let projection_in_flight = runtime.lane.lock().expect("lane lock");
        let (send, receive) = std::sync::mpsc::channel();
        let health_runtime = std::sync::Arc::clone(&runtime);
        let health = std::thread::spawn(move || send.send(health_runtime.is_ready()));

        let result = receive.recv_timeout(std::time::Duration::from_millis(500));
        drop(projection_in_flight);
        health
            .join()
            .expect("health thread")
            .expect("health result");
        assert_eq!(
            result,
            Ok(true),
            "health readiness must remain lock-free during image projection"
        );
    }

    #[cfg(all(target_os = "windows", feature = "cuda"))]
    #[test]
    fn prism_27b_windows_cuda_failures_are_actionable_and_fail_closed() {
        assert!(prism_27b_windows_cuda_lane_selected_with(5_120, true, None));
        assert!(prism_27b_windows_cuda_lane_selected_with(
            5_120,
            true,
            Some("on")
        ));
        assert!(!prism_27b_windows_cuda_lane_selected_with(
            5_120, false, None
        ));
        assert!(!prism_27b_windows_cuda_lane_selected_with(
            5_120,
            true,
            Some("0")
        ));
        assert!(!prism_27b_windows_cuda_lane_selected_with(
            4_096, true, None
        ));

        let diagnostic = prism_27b_cuda_failure_message("execution", "CUDA_ERROR_OUT_OF_MEMORY", 0);
        assert!(diagnostic.contains("Prism 27B image support"));
        assert!(diagnostic.contains("CUDA execution failed"));
        assert!(diagnostic.contains("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(diagnostic.contains("CPU fallback is disabled"));
        assert!(diagnostic.contains("CAMELID_CUDA_DEVICE"));
        assert!(diagnostic.contains("free enough VRAM"));
        assert!(diagnostic.contains("reload the model or restart Camelid"));
    }

    #[test]
    fn real_prism_mmproj_loads_as_page_backed_qwen3vl() {
        let Ok(path) = std::env::var("CAMELID_PRISM_MMPROJ") else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ");
            return;
        };
        let model = PrismVisionModel::load(path).expect("load Prism mmproj");
        assert_eq!(model.hidden, 1152);
        assert_eq!(model.layers.len(), 27);
        assert_eq!(model.projection, 5120);
        assert_eq!(model.patch_0.pages.byte_len(), 16 * 16 * 3 * 1152 * 4);
        assert_eq!(model.layers[0].qkv.tensor_type, GgufTensorType::Q8_0);
        assert_eq!(model.layers[0].ffn_down.tensor_type, GgufTensorType::F16);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn real_prism_mmproj_encodes_a_tiny_image_on_cpu() {
        let Ok(model_path) = std::env::var("CAMELID_PRISM_MMPROJ") else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ");
            return;
        };
        let model = PrismVisionModel::load(model_path).expect("load Prism projector");
        let image = RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 7) as u8, ((x + y) * 3) as u8])
        });
        let input = model
            .preprocess_rgb(image, 1, 1)
            .expect("preprocess tiny image");
        let output = model.encode_cpu(input).expect("encode tiny image on CPU");
        assert_eq!(output.grid_width, 1);
        assert_eq!(output.grid_height, 1);
        assert_eq!(output.embeddings.len(), 1);
        assert_eq!(output.embeddings[0].len(), model.projection);
        assert!(output.embeddings[0].iter().all(|value| value.is_finite()));
        let l2 = output.embeddings[0]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        eprintln!("Qwen3-VL tiny image CPU embedding: l2={l2:.6}");
        assert!(l2 > 1.0);
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    #[test]
    fn real_prism_mmproj_cuda_matches_cpu_for_a_tiny_image() {
        let Ok(model_path) = std::env::var("CAMELID_PRISM_MMPROJ") else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ");
            return;
        };
        let model = PrismVisionModel::load(model_path).expect("load Prism projector");
        let image = RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 7) as u8, ((x + y) * 3) as u8])
        });
        let input = model
            .preprocess_rgb(image, 1, 1)
            .expect("preprocess tiny image");
        let cpu = model
            .encode_cpu(input.clone())
            .expect("encode CPU reference");
        let mut encoder =
            super::super::vision_cuda::CudaVisionEncoder::new().expect("initialize CUDA projector");
        let started = std::time::Instant::now();
        let cuda = encoder.encode(&model, &input).expect("encode on CUDA");
        let elapsed = started.elapsed().as_secs_f64();
        assert_eq!(cuda.len(), 1);
        assert_eq!(cuda[0].len(), model.projection);
        let (mut dot, mut cpu_l2, mut cuda_l2) = (0.0f64, 0.0f64, 0.0f64);
        let mut max_abs = 0.0f32;
        for (&expected, &actual) in cpu.embeddings[0].iter().zip(cuda[0].iter()) {
            dot += expected as f64 * actual as f64;
            cpu_l2 += expected as f64 * expected as f64;
            cuda_l2 += actual as f64 * actual as f64;
            max_abs = max_abs.max((expected - actual).abs());
        }
        let cosine = dot / (cpu_l2.sqrt() * cuda_l2.sqrt());
        eprintln!(
            "Qwen3-VL CUDA projector: elapsed={elapsed:.3}s cosine={cosine:.9} max_abs={max_abs:.6}"
        );
        assert!(cosine > 0.999, "CUDA projector cosine similarity {cosine}");
        assert!(max_abs < 0.1, "CUDA projector max abs error {max_abs}");
    }

    /// Full-resolution projector timing harness. Unlike the end-to-end hidden
    /// CLI benchmark this does not load the 27B language graph, so it isolates
    /// the image encoder and can run beside language-kernel development.
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    #[test]
    fn real_prism_mmproj_cuda_encodes_configured_image() {
        let (Ok(model_path), Ok(image_path)) = (
            std::env::var("CAMELID_PRISM_MMPROJ"),
            std::env::var("CAMELID_PRISM_IMAGE"),
        ) else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ and CAMELID_PRISM_IMAGE");
            return;
        };
        let model = PrismVisionModel::load(model_path).expect("load Prism projector");
        let input = model
            .preprocess(image_path, 1, 128)
            .expect("preprocess configured image");
        let grid_width = input.patch_width / model.merge;
        let grid_height = input.patch_height / model.merge;
        let mut encoder =
            super::super::vision_cuda::CudaVisionEncoder::new().expect("initialize CUDA projector");
        let started = std::time::Instant::now();
        let cuda = encoder.encode(&model, &input).expect("encode on CUDA");
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "Qwen3-VL CUDA full image: grid={grid_width}x{grid_height} tokens={} elapsed={elapsed:.6}s",
            cuda.len()
        );
        assert_eq!(cuda.len(), grid_width * grid_height);
        assert!(cuda
            .iter()
            .all(|row| row.len() == model.projection && row.iter().all(|value| value.is_finite())));
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    #[test]
    fn real_prism_mmproj_cuda_tensor_cores_match_scalar_for_configured_image() {
        let (Ok(model_path), Ok(image_path)) = (
            std::env::var("CAMELID_PRISM_MMPROJ"),
            std::env::var("CAMELID_PRISM_IMAGE"),
        ) else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ and CAMELID_PRISM_IMAGE");
            return;
        };
        let model = PrismVisionModel::load(model_path).expect("load Prism projector");
        let input = model
            .preprocess(image_path, 1, 128)
            .expect("preprocess configured image");

        let mut scalar = super::super::vision_cuda::CudaVisionEncoder::new()
            .expect("initialize scalar CUDA projector");
        scalar.disable_tensor_cores_for_test();
        let scalar_started = std::time::Instant::now();
        let expected = scalar
            .encode(&model, &input)
            .expect("encode with scalar CUDA kernels");
        let scalar_seconds = scalar_started.elapsed().as_secs_f64();
        drop(scalar);

        let mut tensor = super::super::vision_cuda::CudaVisionEncoder::new()
            .expect("initialize tensor-core CUDA projector");
        let tensor_started = std::time::Instant::now();
        let actual = tensor
            .encode(&model, &input)
            .expect("encode with tensor-core CUDA kernels");
        let tensor_seconds = tensor_started.elapsed().as_secs_f64();
        assert_eq!(actual.len(), expected.len());
        let (mut dot, mut expected_l2, mut actual_l2, mut error_l2) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut max_abs = 0.0f32;
        let mut max_reference_abs = 0.0f32;
        for (&reference, &candidate) in expected.iter().flatten().zip(actual.iter().flatten()) {
            dot += reference as f64 * candidate as f64;
            expected_l2 += reference as f64 * reference as f64;
            actual_l2 += candidate as f64 * candidate as f64;
            error_l2 += (reference - candidate) as f64 * (reference - candidate) as f64;
            max_abs = max_abs.max((reference - candidate).abs());
            max_reference_abs = max_reference_abs.max(reference.abs());
        }
        let cosine = dot / (expected_l2.sqrt() * actual_l2.sqrt());
        let relative_l2 = (error_l2 / expected_l2).sqrt();
        eprintln!(
            "Qwen3-VL CUDA tensor-core parity: scalar={scalar_seconds:.6}s tensor={tensor_seconds:.6}s cosine={cosine:.9} relative_l2={relative_l2:.6} max_abs={max_abs:.6} max_reference_abs={max_reference_abs:.6}"
        );
        assert!(cosine > 0.999, "tensor-core projector cosine {cosine}");
        assert!(
            relative_l2 < 0.04,
            "tensor-core projector relative L2 {relative_l2}"
        );
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    #[test]
    fn real_prism_27b_generates_from_a_real_cuda_projector_embedding() {
        let (Ok(model_path), Ok(projector_path)) = (
            std::env::var("CAMELID_PRISM_27B_GGUF"),
            std::env::var("CAMELID_PRISM_MMPROJ"),
        ) else {
            eprintln!("SKIP: set CAMELID_PRISM_27B_GGUF and CAMELID_PRISM_MMPROJ");
            return;
        };
        let projector = PrismVisionProjector::load(projector_path).expect("load Prism projector");
        let image = RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 7) as u8, ((x + y) * 3) as u8])
        });
        let input = projector
            .model
            .preprocess_rgb(image, 1, 1)
            .expect("preprocess tiny image");
        let embedding = projector
            .encode_preprocessed(input)
            .expect("encode tiny image on CUDA");
        let model =
            crate::runnable::RunnableModel::load(&model_path).expect("load Prism 27B language row");
        let generated = model
            .generate_vision(&[0], &embedding, &[0], 1, &[])
            .expect("generate from real CUDA projector embedding");
        assert_eq!(generated.len(), 1);
        assert!((generated[0] as usize) < model.vocab);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn real_prism_mmproj_encodes_a_tiny_image_on_metal() {
        let (Ok(model_path), Ok(image_path)) = (
            std::env::var("CAMELID_PRISM_MMPROJ"),
            std::env::var("CAMELID_PRISM_IMAGE"),
        ) else {
            eprintln!("SKIP: set CAMELID_PRISM_MMPROJ and CAMELID_PRISM_IMAGE");
            return;
        };
        let model = PrismVisionModel::load(model_path).expect("load Prism mmproj");
        let input = model
            .preprocess(image_path, 1, 1)
            .expect("preprocess tiny image");
        assert_eq!(input.patch_count(), 4);
        assert_eq!(input.output_tokens(model.merge), 1);
        let mut encoder = model.metal_encoder().expect("build Metal encoder");
        let output = encoder
            .encode(
                &input.patches,
                &input.position,
                input.patch_width,
                input.patch_height,
            )
            .expect("encode tiny image");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), model.projection);
        assert!(output[0].iter().all(|value| value.is_finite()));
        let l2 = output[0]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        eprintln!("Qwen3-VL tiny image embedding: l2={l2:.6}");
        assert!(l2 > 1.0);
    }
}
