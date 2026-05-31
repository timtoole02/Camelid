use serde::Serialize;

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
    pub moe: Option<MixtralMoeMetadata>,
}

impl LlamaModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = match gguf.architecture() {
            Some(architecture @ ("llama" | "mistral" | "qwen2" | "qwen3" | "smollm3" | "gemma3" | "phi3" | "lfm2")) => architecture,
            Some(other) => return Err(BackendError::UnsupportedModelArchitecture(other.into())),
            None => {
                return Err(BackendError::InvalidModelMetadata(
                    "required metadata general.architecture is missing".into(),
                ))
            }
        };

        let moe = MixtralMoeMetadata::from_gguf(gguf, architecture);

        let attention_head_count = required_u32(
            gguf,
            &architecture_key(architecture, "attention.head_count"),
        )?;
        let attention_head_count_kv =
            llama_attention_head_count_kv(gguf, architecture, attention_head_count);
        Ok(Self {
            context_length: required_u32(gguf, &architecture_key(architecture, "context_length"))?,
            embedding_length: required_u32(
                gguf,
                &architecture_key(architecture, "embedding_length"),
            )?,
            block_count: required_u32(gguf, &architecture_key(architecture, "block_count"))?,
            feed_forward_length: required_u32(
                gguf,
                &architecture_key(architecture, "feed_forward_length"),
            )?,
            attention_head_count,
            attention_head_count_kv,
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
            moe,
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MixtralMoeMetadata {
    pub family_label: &'static str,
    pub expert_count: u32,
    pub expert_used_count: u32,
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

        Some(Self {
            family_label,
            expert_count,
            expert_used_count,
        })
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
    pub attention_q: GgufTensorDescriptor,
    pub attention_k: GgufTensorDescriptor,
    pub attention_v: GgufTensorDescriptor,
    pub attention_output: GgufTensorDescriptor,
    pub ffn_norm: GgufTensorDescriptor,
    pub ffn: LlamaFfnTensors,
}

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

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for layer_idx in 0..config.block_count {
            layers.push(LlamaLayerTensors {
                attention_norm: required_tensor(
                    gguf,
                    &format!("blk.{layer_idx}.attn_norm.weight"),
                )?,
                attention_q: required_tensor(gguf, &format!("blk.{layer_idx}.attn_q.weight"))?,
                attention_k: required_tensor(gguf, &format!("blk.{layer_idx}.attn_k.weight"))?,
                attention_v: required_tensor(gguf, &format!("blk.{layer_idx}.attn_v.weight"))?,
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
            let rope_dim = config.rope_dimension_count.unwrap_or(dims.head_dim as u32) as usize;
            if rope_dim == 0 || rope_dim > dims.head_dim || !rope_dim.is_multiple_of(2) {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "RoPE dimension count {rope_dim} must be even and within head dimension {}",
                    dims.head_dim
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
            require_descriptor_matrix_shape(
                &layer.attention_q,
                dims.embedding_length,
                dims.embedding_length,
                &format!("layer {idx} attention q"),
            )?;
            require_descriptor_matrix_shape(
                &layer.attention_k,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention k"),
            )?;
            require_descriptor_matrix_shape(
                &layer.attention_v,
                dims.embedding_length,
                dims.kv_width,
                &format!("layer {idx} attention v"),
            )?;
            require_descriptor_matrix_shape(
                &layer.attention_output,
                dims.embedding_length,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseLlamaDims {
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub attention_head_count_kv: usize,
    pub head_dim: usize,
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

        let head_dim = embedding_length / attention_head_count;
        Ok(Self {
            embedding_length,
            block_count: config.block_count as usize,
            feed_forward_length: config.feed_forward_length as usize,
            attention_head_count_kv,
            head_dim,
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

#[cfg(test)]
mod tests {
    use super::validate_output_projection_storage_layout;
    use crate::gguf::{GgufTensorDescriptor, GgufTensorType};

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
