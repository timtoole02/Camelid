use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("I/O error while reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid GGUF file: {0}")]
    InvalidGguf(String),

    #[error("unsupported GGUF feature: {0}")]
    UnsupportedGguf(String),

    #[error("unsupported tokenizer: {0}")]
    UnsupportedTokenizer(String),

    #[error("invalid tokenizer metadata: {0}")]
    InvalidTokenizerMetadata(String),

    #[error("tokenizer metadata is not available in the loaded model")]
    TokenizerNotAvailable,

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("unsupported tensor type: {0}")]
    UnsupportedTensorType(String),

    #[error("invalid tensor data: {0}")]
    InvalidTensorData(String),

    #[error("runtime shape mismatch: {0}")]
    RuntimeShapeMismatch(String),

    // The env name is mirrored as a literal here (kept in sync with
    // `KV_CACHE_BUDGET_LIMIT_ENV` in `src/inference/kv_cache.rs`) so the
    // user-facing message names the override knob without threading the const
    // through the variant. Message wording is load-bearing: the serve path
    // surfaces it in the generation-error body.
    #[error(
        "KV cache growth to {positions} positions needs {needed_bytes} \
         bytes of f32 K+V, above the {budget_bytes} byte budget for this host; reduce the prompt/context \
         length or set CAMELID_MAX_KV_CACHE_BYTES deliberately for a controlled run"
    )]
    KvCacheBudgetExceeded {
        positions: usize,
        needed_bytes: u64,
        budget_bytes: u64,
    },

    /// A resident batch row asked for a position past the engine's capacity. Raised by a
    /// bounds check before any KV page is written, so the caller may retry the row.
    #[error("resident position {position} exceeds the engine capacity of {capacity}")]
    ResidentPositionOutOfRange { position: usize, capacity: usize },

    #[error("invalid model metadata: {0}")]
    InvalidModelMetadata(String),

    #[error("unsupported model architecture: {0}")]
    UnsupportedModelArchitecture(String),

    #[error("model is not loaded")]
    ModelNotLoaded,

    #[error("a Workspace session is using the active model; finish or cancel it before loading or unloading a model")]
    ModelOperationInProgress,

    #[error("inference is not implemented yet; current build supports health checks and GGUF metadata inspection only")]
    InferenceNotImplemented,

    /// The caller's cancel check fired part-way through a generation (for
    /// example while reading the prompt). The partial state is unusable and
    /// must be dropped, not resumed.
    #[error("generation cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, BackendError>;
