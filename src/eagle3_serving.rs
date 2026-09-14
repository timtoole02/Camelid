//! Serving orchestration for target-verified Llama-3.2-3B and Qwen3-4B EAGLE-3 speculation.
//!
//! The target model remains authoritative for every emitted token. A cheap
//! suffix chain gets first refusal (and can use all 16 verifier rows); misses
//! fall through to the bounded N8/K4/X5 learned tree. Target activation rows
//! from successful suffix rounds are buffered and applied to the learned head
//! only when that fallback is actually needed.
//!
//! The Metal-resident draft head (the uploaded draft wire plus its private
//! one-layer cache) is request-independent and expensive to upload, so serving
//! keeps one in a single-slot checkout pool keyed
//! by [`Eagle3ServeHeadKey`]. Everything a generation mutates -- the head's
//! cache watermark and root seed, the suffix statistics, the pending catch-up
//! rows -- is either reset when the head is returned or lives in the
//! per-request [`Eagle3ServingState`]. A head is never shared: a checkout
//! empties the slot, so a concurrent second generation uploads its own.

use std::{
    ops::RangeInclusive,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use crate::{
    eagle3::Eagle3DraftModel,
    eagle3_runtime::{Eagle3AuthoritativeCatchup, Eagle3Drafter, Eagle3DynamicFrontierConfig},
    error::{BackendError, Result},
    inference::{
        spec_tree::TREE_MAX_NODES,
        speculative::accepted_draft_prefix,
        suffix_decoding::{SuffixAdmissionEvidence, SuffixDecodingDrafter},
        LlamaForwardTimings, LlamaInferenceSession, LlamaLoadedWeights,
    },
};

/// The packed Metal verifier has a hard 16-row ceiling, root included.
pub const MAX_DRAFT_TOKENS: usize = TREE_MAX_NODES - 1;
/// Default serving envelope. Larger rungs are explicit because each one owes
/// its own exact-token, memory, and throughput receipt on the deployment host.
pub const DEFAULT_MAX_LOGICAL_TOKENS: usize = 2_048;
pub const LOGICAL_TOKEN_LIMIT_ENV: &str = "CAMELID_EAGLE3_LOGICAL_TOKENS";
pub const SUPPORTED_LOGICAL_TOKEN_LIMITS: [usize; 2] = [2_048, 4_096];

/// Widths 9..=16 deliberately leave Metal's row-dimensional F16 attention
/// path once any verifier row crosses position 2,048. N8 retains the checked
/// deep-context batch path, so suffix verification narrows before crossing.
const WIDE_VERIFY_POSITION_LIMIT: usize = 2_048;
const DEEP_CONTEXT_VERIFY_NODES: usize = 8;

const DEFAULT_DYNAMIC_VERIFY_NODES: usize = 8;
const DEFAULT_DYNAMIC_TOP_K: usize = 4;
/// Bounded default: at most five frontier expansions per verification round.
const DEFAULT_DYNAMIC_EXPANSIONS: usize = 5;

/// Operator overrides for the learned-tree shape. Each is optional; a set value
/// must parse and sit inside its range, and any bad value rejects the whole
/// override (see [`dynamic_tree_from_values`]).
pub const SERVE_TREE_NODES_ENV: &str = "CAMELID_EAGLE3_SERVE_TREE_NODES";
pub const SERVE_TREE_TOP_K_ENV: &str = "CAMELID_EAGLE3_SERVE_TREE_TOP_K";
pub const SERVE_TREE_EXPANSIONS_ENV: &str = "CAMELID_EAGLE3_SERVE_TREE_EXPANSIONS";
/// A tree verify needs the root plus at least one drafted row.
const MIN_DYNAMIC_VERIFY_NODES: usize = 2;
/// The head retains this many ranked candidates per expansion; a larger top-k
/// would be silently truncated to it.
const MAX_DYNAMIC_TOP_K: usize = crate::metal::EAGLE3_TOP_K_CANDIDATES;
/// Sanity ceiling on head forwards per round; every certified point is <= 8.
const MAX_DYNAMIC_EXPANSIONS: usize = 32;

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidModelMetadata(message.into())
}

/// The learned-tree shape the serving lane drafts when the suffix lane declines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eagle3DynamicTree {
    pub verify_nodes: usize,
    pub top_k: usize,
    pub expansions: usize,
}

impl Eagle3DynamicTree {
    pub const DEFAULT: Self = Self {
        verify_nodes: DEFAULT_DYNAMIC_VERIFY_NODES,
        top_k: DEFAULT_DYNAMIC_TOP_K,
        expansions: DEFAULT_DYNAMIC_EXPANSIONS,
    };

    fn label(self) -> String {
        format!(
            "N{}/K{}/X{}",
            self.verify_nodes, self.top_k, self.expansions
        )
    }
}

fn parse_tree_value(
    name: &str,
    value: Option<&str>,
    default: usize,
    range: RangeInclusive<usize>,
) -> std::result::Result<usize, String> {
    let Some(raw) = value.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(default);
    };
    let parsed = raw.parse::<usize>().map_err(|error| {
        format!(
            "{name} must be an integer in {}..={}, got {raw:?}: {error}",
            range.start(),
            range.end()
        )
    })?;
    if !range.contains(&parsed) {
        return Err(format!(
            "{name} must be in {}..={}, got {parsed}",
            range.start(),
            range.end()
        ));
    }
    Ok(parsed)
}

/// Parse the operator's serve-tree override. Unset or empty fields keep their
/// defaults; a malformed or out-of-range field rejects the WHOLE override, so a
/// half-applied shape can never run.
pub fn dynamic_tree_from_values(
    nodes: Option<&str>,
    top_k: Option<&str>,
    expansions: Option<&str>,
) -> std::result::Result<Eagle3DynamicTree, String> {
    Ok(Eagle3DynamicTree {
        verify_nodes: parse_tree_value(
            SERVE_TREE_NODES_ENV,
            nodes,
            DEFAULT_DYNAMIC_VERIFY_NODES,
            MIN_DYNAMIC_VERIFY_NODES..=TREE_MAX_NODES,
        )?,
        top_k: parse_tree_value(
            SERVE_TREE_TOP_K_ENV,
            top_k,
            DEFAULT_DYNAMIC_TOP_K,
            1..=MAX_DYNAMIC_TOP_K,
        )?,
        expansions: parse_tree_value(
            SERVE_TREE_EXPANSIONS_ENV,
            expansions,
            DEFAULT_DYNAMIC_EXPANSIONS,
            1..=MAX_DYNAMIC_EXPANSIONS,
        )?,
    })
}

/// Process-wide serve tree shape, read once. A rejected override prints one
/// stderr line and serving keeps the default tree; an accepted non-default
/// override prints one line naming the shape it will run.
pub fn configured_dynamic_tree() -> Eagle3DynamicTree {
    static TREE: OnceLock<Eagle3DynamicTree> = OnceLock::new();
    *TREE.get_or_init(|| {
        let nodes = std::env::var(SERVE_TREE_NODES_ENV).ok();
        let top_k = std::env::var(SERVE_TREE_TOP_K_ENV).ok();
        let expansions = std::env::var(SERVE_TREE_EXPANSIONS_ENV).ok();
        match dynamic_tree_from_values(nodes.as_deref(), top_k.as_deref(), expansions.as_deref()) {
            Ok(tree) => {
                if tree != Eagle3DynamicTree::DEFAULT {
                    eprintln!(
                        "[eagle3-serve-tree] learned tree {} (operator override of the certified {})",
                        tree.label(),
                        Eagle3DynamicTree::DEFAULT.label()
                    );
                }
                tree
            }
            Err(message) => {
                eprintln!(
                    "[eagle3-serve-tree] {message}; serving keeps the certified {} tree",
                    Eagle3DynamicTree::DEFAULT.label()
                );
                Eagle3DynamicTree::DEFAULT
            }
        }
    })
}

/// Identity of the request-independent part of the serving drafter: the
/// Metal-resident draft wire plus its private cache allocation. Two requests
/// may share one uploaded head only when every field agrees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3ServeHeadKey {
    pub checkpoint_path: PathBuf,
    pub checkpoint_sha256: String,
    pub target_sha256: String,
    /// `crate::metal::eagle3_draft_wire_identity()`: the body/lm_head wire
    /// formats and lm_head rows the upload was planned with.
    pub draft_wire: String,
    pub max_positions: usize,
}

impl Eagle3ServeHeadKey {
    /// Head cache capacity that admits every request the serving envelope can
    /// accept: `prompt + max_tokens` never exceeds the logical limit, and one
    /// round holds at most `MAX_DRAFT_TOKENS` drafted rows plus the anchor.
    pub fn pooled_max_positions(logical_token_limit: usize) -> Result<usize> {
        logical_token_limit
            .checked_add(MAX_DRAFT_TOKENS + 1)
            .ok_or_else(|| invalid("EAGLE-3 pooled head capacity overflow"))
    }

    /// The identity a request would need right now, under the current wire
    /// gates and logical envelope.
    pub fn current(
        checkpoint_path: &Path,
        checkpoint_sha256: &str,
        target_sha256: &str,
    ) -> Result<Self> {
        Ok(Self {
            checkpoint_path: checkpoint_path.to_path_buf(),
            checkpoint_sha256: checkpoint_sha256.to_string(),
            target_sha256: target_sha256.to_string(),
            draft_wire: if cfg!(all(feature = "cuda", not(target_os = "macos"))) {
                "cuda-q8_128-f32-linear1-v2".to_string()
            } else {
                crate::metal::eagle3_draft_wire_identity().map_err(invalid)?
            },
            max_positions: Self::pooled_max_positions(configured_logical_token_limit()?)?,
        })
    }

    fn summary(&self) -> String {
        format!(
            "head={} sha={} target={} wire=[{}] positions={}",
            self.checkpoint_path.display(),
            &self.checkpoint_sha256[..self.checkpoint_sha256.len().min(12)],
            &self.target_sha256[..self.target_sha256.len().min(12)],
            self.draft_wire,
            self.max_positions
        )
    }
}

/// Result of asking the pool for a head.
pub enum Eagle3HeadCheckout<T> {
    /// The pooled head matched and is now owned by the caller.
    Hit(T),
    /// Nothing pooled (never populated, or checked out by another generation).
    Empty,
    /// The pooled head belonged to a different identity and has been dropped.
    Invalidated(Eagle3ServeHeadKey),
}

/// Single-slot checkout pool for one expensive, request-independent value.
/// A checkout removes the value, so it can never be observed by two owners;
/// a restore into an occupied slot drops the newcomer, so the pool holds at
/// most one value (the memory cost stays one head, even under overlap).
pub struct Eagle3HeadPool<T> {
    slot: Mutex<Option<(Eagle3ServeHeadKey, T)>>,
}

impl<T> Eagle3HeadPool<T> {
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// Take the pooled value when its identity matches `key`. A pooled value with
    /// a different identity (model, head, wire gates or envelope changed under
    /// the server) is discarded here so a stale head is never handed out.
    pub fn checkout(&self, key: &Eagle3ServeHeadKey) -> Eagle3HeadCheckout<T> {
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        match slot.take() {
            Some((pooled_key, value)) if pooled_key == *key => Eagle3HeadCheckout::Hit(value),
            Some((pooled_key, value)) => {
                drop(value);
                Eagle3HeadCheckout::Invalidated(pooled_key)
            }
            None => Eagle3HeadCheckout::Empty,
        }
    }

    /// Return a value to the pool. `false` means the slot was already occupied
    /// and the value was dropped instead.
    pub fn restore(&self, key: Eagle3ServeHeadKey, value: T) -> bool {
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_some() {
            return false;
        }
        *slot = Some((key, value));
        true
    }

    /// Whether the slot currently holds a value (diagnostics/tests only).
    pub fn is_populated(&self) -> bool {
        self.slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }
}

impl<T> Default for Eagle3HeadPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The serve-wide pool of the one uploaded draft head.
static SERVE_HEAD_POOL: Eagle3HeadPool<Eagle3Drafter> = Eagle3HeadPool::new();
#[cfg(all(feature = "cuda", not(target_os = "macos")))]
static CUDA_SERVE_HEAD_POOL: Eagle3HeadPool<crate::eagle3_cuda::CudaEagle3Head> =
    Eagle3HeadPool::new();

pub fn logical_token_limit_from_value(value: Option<&str>) -> Result<usize> {
    let Some(raw) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(DEFAULT_MAX_LOGICAL_TOKENS);
    };
    let parsed = raw.parse::<usize>().map_err(|error| {
        invalid(format!(
            "{LOGICAL_TOKEN_LIMIT_ENV} must be 2048 or 4096, got {raw:?}: {error}"
        ))
    })?;
    if !SUPPORTED_LOGICAL_TOKEN_LIMITS.contains(&parsed) {
        return Err(invalid(format!(
            "{LOGICAL_TOKEN_LIMIT_ENV} must be 2048 or 4096, got {parsed}"
        )));
    }
    Ok(parsed)
}

pub fn configured_logical_token_limit() -> Result<usize> {
    let value = std::env::var(LOGICAL_TOKEN_LIMIT_ENV).ok();
    logical_token_limit_from_value(value.as_deref())
}

fn validate_logical_budget_at_limit(
    prompt_tokens: usize,
    max_tokens: usize,
    logical_token_limit: usize,
) -> Result<usize> {
    let logical_tokens = prompt_tokens
        .checked_add(max_tokens)
        .ok_or_else(|| invalid("EAGLE-3 logical token budget overflow"))?;
    if logical_tokens > logical_token_limit {
        return Err(invalid(format!(
            "EAGLE-3 serving is fail-closed above {logical_token_limit} logical tokens; prompt {prompt_tokens} + max_tokens {max_tokens} = {logical_tokens}"
        )));
    }
    Ok(logical_tokens)
}

pub fn validate_logical_budget(prompt_tokens: usize, max_tokens: usize) -> Result<usize> {
    validate_logical_budget_at_limit(prompt_tokens, max_tokens, configured_logical_token_limit()?)
}

/// Treat an API `max_tokens` value as the upper bound it is: shorten it to the
/// exact room left in EAGLE-3's verified logical envelope. A prompt that has
/// already filled that envelope still fails closed because there is no valid
/// token budget to admit.
pub fn clamp_max_tokens_to_logical_budget(
    prompt_tokens: usize,
    max_tokens: usize,
) -> Result<usize> {
    clamp_max_tokens_to_logical_budget_at_limit(
        prompt_tokens,
        max_tokens,
        configured_logical_token_limit()?,
    )
}

fn clamp_max_tokens_to_logical_budget_at_limit(
    prompt_tokens: usize,
    max_tokens: usize,
    logical_token_limit: usize,
) -> Result<usize> {
    if prompt_tokens >= logical_token_limit {
        return Err(invalid(format!(
            "EAGLE-3 serving is fail-closed above {logical_token_limit} logical tokens; prompt {prompt_tokens} leaves no room for generation"
        )));
    }
    Ok(max_tokens.min(logical_token_limit - prompt_tokens))
}

/// Keep verifier requests on Metal's checked row-count/position contract.
/// N8 is valid at every supported 4K position; widths 9..=16 are retained only
/// when their final row remains inside the receipted 2K wide-attention lane.
pub fn cap_verify_nodes_for_position(target_position: usize, requested_nodes: usize) -> usize {
    if requested_nodes > DEEP_CONTEXT_VERIFY_NODES
        && target_position.saturating_add(requested_nodes) > WIDE_VERIFY_POSITION_LIMIT
    {
        DEEP_CONTEXT_VERIFY_NODES
    } else {
        requested_nodes
    }
}

fn suffix_verify_node_limit(target_position: usize) -> usize {
    cap_verify_nodes_for_position(target_position, TREE_MAX_NODES)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eagle3ServingConfig {
    pub draft_tokens: usize,
    pub dynamic_verify_nodes: usize,
    pub dynamic_top_k: usize,
    pub dynamic_expansions: usize,
}

impl Eagle3ServingConfig {
    /// The serving width plus the process-wide learned tree
    /// ([`configured_dynamic_tree`]: certified N8/K4/X5 unless overridden).
    pub fn new(draft_tokens: usize) -> Result<Self> {
        Self::with_dynamic_tree(draft_tokens, configured_dynamic_tree())
    }

    pub fn with_dynamic_tree(draft_tokens: usize, tree: Eagle3DynamicTree) -> Result<Self> {
        if !(1..=MAX_DRAFT_TOKENS).contains(&draft_tokens) {
            return Err(invalid(format!(
                "EAGLE-3 serving draft width must be in 1..={MAX_DRAFT_TOKENS}, got {draft_tokens}"
            )));
        }
        Ok(Self {
            draft_tokens,
            // The learned tree is the fallback shape; the suffix lane can still
            // consume the full physical width 16.
            dynamic_verify_nodes: tree.verify_nodes,
            dynamic_top_k: tree.top_k,
            dynamic_expansions: tree.expansions,
        })
    }
}

pub struct Eagle3ServingBootstrap {
    pub first_token: u32,
    pub timings: LlamaForwardTimings,
}

pub struct Eagle3ServingRound {
    pub emitted: Vec<u32>,
    pub offered: usize,
    pub verify_nodes: usize,
    pub suffix: bool,
    pub suffix_evidence: SuffixAdmissionEvidence,
    pub timings: LlamaForwardTimings,
}

/// Per-request mutable draft state. The host checkpoint is shared across
/// requests and the uploaded head is pooled (see [`Eagle3ServeHeadKey`]);
/// the suffix statistics and pending catch-up rows are private to a generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Eagle3ServingTimings {
    pub head_bootstrap_us: u128,
    pub draft_us: u128,
    pub verify_us: u128,
    pub head_update_us: u128,
}

pub struct Eagle3ServingState {
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    cuda_head: Option<crate::eagle3_cuda::CudaEagle3Head>,
    phases: Eagle3ServingTimings,
    capture_layer_ids: [usize; 3],
    checkpoint: Option<Arc<Eagle3DraftModel>>,
    drafter: Option<Eagle3Drafter>,
    /// When set, `bootstrap` checks the head out of the serve-wide pool under
    /// this identity and `Drop` returns it. `None` keeps the original
    /// per-request upload.
    pooled_head: Option<Eagle3ServeHeadKey>,
    head_reused: bool,
    early_exit_rounds: u64,
    suffix: SuffixDecodingDrafter,
    pending_suffix_head: Eagle3AuthoritativeCatchup,
    config: Eagle3ServingConfig,
}

impl Eagle3ServingState {
    pub fn new(checkpoint: Arc<Eagle3DraftModel>, config: Eagle3ServingConfig) -> Self {
        Self {
            #[cfg(all(feature = "cuda", not(target_os = "macos")))]
            cuda_head: None,
            capture_layer_ids: checkpoint.config.geometry().target_layer_input_ids(),
            phases: Eagle3ServingTimings::default(),
            checkpoint: Some(checkpoint),
            drafter: None,
            pooled_head: None,
            head_reused: false,
            early_exit_rounds: 0,
            suffix: SuffixDecodingDrafter::default(),
            pending_suffix_head: Eagle3AuthoritativeCatchup::default(),
            config,
        }
    }

    /// Serve the uploaded head from the process-wide single-slot pool under
    /// `key` instead of uploading it for this request alone.
    pub fn with_pooled_head(mut self, key: Eagle3ServeHeadKey) -> Self {
        self.pooled_head = Some(key);
        self
    }

    pub fn phase_timings(&self) -> Eagle3ServingTimings {
        self.phases
    }

    pub fn is_initialized(&self) -> bool {
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        if self.cuda_head.is_some() {
            return true;
        }
        self.drafter.is_some()
    }

    /// Whether `bootstrap` reused an already-uploaded head.
    pub fn head_reused(&self) -> bool {
        self.head_reused
    }

    /// Learned-tree rounds in which the confidence-gated early exit
    /// (`CAMELID_EAGLE3_DRAFT_EARLY_EXIT`) ended drafting before the
    /// expansion budget was spent.
    pub fn early_exit_rounds(&self) -> u64 {
        self.early_exit_rounds
    }

    pub fn dynamic_tree(&self) -> Eagle3DynamicTree {
        if cfg!(all(feature = "cuda", not(target_os = "macos"))) {
            return Eagle3DynamicTree {
                verify_nodes: 2,
                top_k: 1,
                expansions: 1,
            };
        }
        Eagle3DynamicTree {
            verify_nodes: self.config.dynamic_verify_nodes,
            top_k: self.config.dynamic_top_k,
            expansions: self.config.dynamic_expansions,
        }
    }

    /// Obtain the uploaded head for this generation: a pooled one when its
    /// identity matches and its allocation covers this request, otherwise a
    /// fresh upload (which `Drop` will pool when it carries an identity).
    fn acquire_head(
        &mut self,
        checkpoint: &Eagle3DraftModel,
        required_capacity: usize,
    ) -> Result<Eagle3Drafter> {
        let Some(key) = self
            .pooled_head
            .clone()
            .filter(|key| key.max_positions >= required_capacity)
        else {
            // Uncached by construction, or a request the pooled allocation
            // cannot hold: a private head with the exact capacity, as before.
            self.pooled_head = None;
            return Eagle3Drafter::new(checkpoint, required_capacity);
        };
        match SERVE_HEAD_POOL.checkout(&key) {
            Eagle3HeadCheckout::Hit(mut drafter) => {
                if drafter.max_positions() != key.max_positions {
                    return Err(invalid(format!(
                        "EAGLE-3 pooled head holds {} positions but its key says {}",
                        drafter.max_positions(),
                        key.max_positions
                    )));
                }
                // `restore` already reset it; repeating is free and keeps the
                // fresh-drafter contract of `seed_prompt` local to this seam.
                drafter.reset_for_reuse();
                self.head_reused = true;
                Ok(drafter)
            }
            Eagle3HeadCheckout::Invalidated(stale) => {
                eprintln!(
                    "[eagle3-serve] cached draft head no longer matches ({}); uploading a fresh head for {}",
                    stale.summary(),
                    key.summary()
                );
                Eagle3Drafter::new(checkpoint, key.max_positions)
            }
            Eagle3HeadCheckout::Empty => Eagle3Drafter::new(checkpoint, key.max_positions),
        }
    }

    /// Capture the real target prompt activations, obtain the first target
    /// greedy token, upload the head, and seed its authoritative cache.
    pub fn bootstrap(
        &mut self,
        session: &mut LlamaInferenceSession,
        target_weights: &Arc<LlamaLoadedWeights>,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<Eagle3ServingBootstrap> {
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        {
            self.bootstrap_cuda(session, target_weights, prompt_tokens, max_tokens)
        }
        #[cfg(not(all(feature = "cuda", not(target_os = "macos"))))]
        {
            self.bootstrap_metal(session, target_weights, prompt_tokens, max_tokens)
        }
    }

    #[cfg_attr(all(feature = "cuda", not(target_os = "macos")), allow(dead_code))]
    fn bootstrap_metal(
        &mut self,
        session: &mut LlamaInferenceSession,
        target_weights: &Arc<LlamaLoadedWeights>,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<Eagle3ServingBootstrap> {
        if self.is_initialized() {
            return Err(invalid("EAGLE-3 serving bootstrap may only run once"));
        }
        if prompt_tokens.len() < 3 {
            return Err(invalid(format!(
                "EAGLE-3 resident activation capture needs at least 3 prompt tokens, got {}",
                prompt_tokens.len()
            )));
        }
        if !session.prewarm_resident_weights() {
            return Err(invalid(
                "EAGLE-3 serving requires the resident Metal target lane",
            ));
        }
        // Target and head share Metal's serial queue. A precommitted target
        // decode graph must never sit ahead of a head update.
        session.set_resident_encode_ahead_enabled(false);
        let prompt = session
            .forward_greedy_resident_prefill_with_layer_inputs(
                prompt_tokens,
                &self.capture_layer_ids,
            )?
            .ok_or_else(|| {
                invalid("resident Metal prompt prefill with EAGLE-3 capture is unavailable")
            })?;
        let first_token = *prompt
            .predictions
            .last()
            .ok_or_else(|| invalid("EAGLE-3 target prompt produced no greedy prediction"))?;
        let head_capacity = prompt_tokens
            .len()
            .checked_add(max_tokens)
            .and_then(|positions| positions.checked_add(self.config.draft_tokens + 1))
            .ok_or_else(|| invalid("EAGLE-3 head cache capacity overflow"))?;
        let checkpoint = self
            .checkpoint
            .take()
            .ok_or_else(|| invalid("EAGLE-3 checkpoint was consumed before bootstrap"))?;
        let head_started = std::time::Instant::now();
        let mut drafter = self.acquire_head(checkpoint.as_ref(), head_capacity)?;
        drafter.seed_prompt(
            target_weights,
            prompt_tokens,
            first_token,
            &prompt.layer_inputs,
        )?;
        self.phases.head_bootstrap_us += head_started.elapsed().as_micros();
        self.drafter = Some(drafter);
        Ok(Eagle3ServingBootstrap {
            first_token,
            timings: prompt.timings,
        })
    }

    /// Run one target-authoritative serving round. A `None` result means there
    /// is no context room left; callers should finish rather than changing
    /// execution lanes after EAGLE has made target KV GPU-authoritative.
    pub fn run_round(
        &mut self,
        session: &mut LlamaInferenceSession,
        target_weights: &Arc<LlamaLoadedWeights>,
        history: &[u32],
        remaining_output: usize,
    ) -> Result<Option<Eagle3ServingRound>> {
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        {
            self.run_round_cuda(session, target_weights, history, remaining_output)
        }
        #[cfg(not(all(feature = "cuda", not(target_os = "macos"))))]
        {
            self.run_round_metal(session, target_weights, history, remaining_output)
        }
    }

    #[cfg_attr(all(feature = "cuda", not(target_os = "macos")), allow(dead_code))]
    fn run_round_metal(
        &mut self,
        session: &mut LlamaInferenceSession,
        target_weights: &Arc<LlamaLoadedWeights>,
        history: &[u32],
        remaining_output: usize,
    ) -> Result<Option<Eagle3ServingRound>> {
        let anchor = *history
            .last()
            .ok_or_else(|| invalid("EAGLE-3 serving history is empty"))?;
        let drafter = self
            .drafter
            .as_mut()
            .ok_or_else(|| invalid("EAGLE-3 serving round ran before bootstrap"))?;
        if remaining_output == 0 {
            return Ok(None);
        }
        let context_room = session.remaining_context();
        if context_room == 0 {
            return Ok(None);
        }
        let budget = self
            .config
            .draft_tokens
            .min(remaining_output.saturating_sub(1))
            .min(context_room.saturating_sub(1));

        // The last requested token has no useful successor to draft. Keep it
        // on the resident target; no head update is observable after it.
        if budget == 0 {
            let started = std::time::Instant::now();
            let (token, _sample_us) = session
                .generate_next_token_greedy_resident(anchor)?
                .ok_or_else(|| invalid("resident Metal target became unavailable"))?;
            self.phases.verify_us += started.elapsed().as_micros();
            return Ok(Some(Eagle3ServingRound {
                emitted: vec![token],
                offered: 0,
                verify_nodes: 1,
                suffix: false,
                suffix_evidence: SuffixAdmissionEvidence::default(),
                timings: LlamaForwardTimings::default(),
            }));
        }

        let target_before = session.kv_position();
        let suffix_node_budget = (budget + 1)
            .min(context_room)
            .min(suffix_verify_node_limit(target_before));
        let suffix_proposal =
            self.suffix
                .draft_confident_chain(history, anchor, suffix_node_budget, budget);
        let suffix_evidence = suffix_proposal.evidence;
        let suffix_drafts = suffix_proposal.tokens;
        tracing::debug!(
            raw_depth = suffix_evidence.raw_depth,
            confident_depth = suffix_evidence.confident_depth,
            root_match_len = suffix_evidence.root_match_len,
            root_support = suffix_evidence.root_support,
            root_branch_count = suffix_evidence.root_branch_count,
            expected_accepted_q16 = suffix_evidence.expected_accepted_q16,
            terminal_survival_q16 = suffix_evidence.terminal_survival_q16,
            admitted = suffix_evidence.admitted,
            "EAGLE-3 suffix confidence admission"
        );

        let round = if !suffix_drafts.is_empty() {
            let verify_started = std::time::Instant::now();
            let verified = session
                .verify_drafts_metal_with_layer_inputs(
                    anchor,
                    &suffix_drafts,
                    &self.capture_layer_ids,
                )?
                .ok_or_else(|| {
                    invalid(format!(
                        "resident Metal suffix verify became unavailable at target position {target_before}"
                    ))
                })?;
            self.phases.verify_us += verify_started.elapsed().as_micros();
            if verified.predictions.len() != suffix_drafts.len() + 1 {
                return Err(invalid(format!(
                    "EAGLE-3 suffix target returned {} predictions for {} drafts",
                    verified.predictions.len(),
                    suffix_drafts.len()
                )));
            }
            let accepted = accepted_draft_prefix(&suffix_drafts, &verified.predictions);
            let emitted = verified.predictions[..=accepted].to_vec();
            self.pending_suffix_head
                .push(&verified.layer_inputs, &emitted)?;
            Eagle3ServingRound {
                emitted,
                offered: suffix_drafts.len(),
                verify_nodes: suffix_drafts.len() + 1,
                suffix: true,
                suffix_evidence,
                timings: verified.timings,
            }
        } else {
            if !self.pending_suffix_head.is_empty() {
                let update_started = std::time::Instant::now();
                drafter
                    .accept_authoritative_catchup(target_weights, &mut self.pending_suffix_head)?;
                self.phases.head_update_us += update_started.elapsed().as_micros();
            }
            if drafter.filled() != session.kv_position() {
                return Err(invalid(format!(
                    "EAGLE-3 catch-up watermark diverged: head={} target={}",
                    drafter.filled(),
                    session.kv_position()
                )));
            }

            // N8 is valid at every supported position; an operator-widened
            // tree narrows to the checked deep-context lane like the suffix
            // lane does (a no-op at the certified N8).
            let node_budget =
                cap_verify_nodes_for_position(target_before, self.config.dynamic_verify_nodes)
                    .min(context_room)
                    .min(budget + 1);
            if node_budget < 2 {
                return Err(invalid(format!(
                    "EAGLE-3 dynamic verifier has only {node_budget} rows"
                )));
            }
            let lattice_nodes = self
                .config
                .dynamic_top_k
                .checked_mul(self.config.dynamic_expansions)
                .and_then(|nodes| nodes.checked_add(1))
                .map(|nodes| nodes.max(node_budget))
                .ok_or_else(|| invalid("EAGLE-3 dynamic lattice budget overflow"))?;
            let draft_started = std::time::Instant::now();
            let frontier = drafter.draft_dynamic_frontier(
                target_weights,
                anchor,
                Eagle3DynamicFrontierConfig {
                    max_verify_nodes: node_budget,
                    max_lattice_nodes: lattice_nodes,
                    max_depth: budget,
                    candidates_per_parent: self.config.dynamic_top_k,
                    max_head_expansions: self.config.dynamic_expansions,
                    adaptive_branching: false,
                    certified_argmax_shadow: false,
                },
            )?;
            // The confidence-gated early exit lives inside the shared
            // `draft_dynamic_frontier` scheduler; count the rounds it fired.
            self.phases.draft_us += draft_started.elapsed().as_micros();
            let early_exit = frontier.draft_early_exit().is_some();
            let forest = frontier.finish()?;
            let actual_nodes = forest.scored.tree.nodes();
            if !(2..=node_budget).contains(&actual_nodes) {
                return Err(invalid(format!(
                    "dynamic EAGLE-3 forest produced {actual_nodes} rows for budget {node_budget}"
                )));
            }
            let verify_started = std::time::Instant::now();
            let verified = session
                .verify_tree_metal_with_layer_inputs_and_e1_shadow(
                    &forest.scored.tree,
                    &self.capture_layer_ids,
                    drafter.authoritative_e1_shadow_head_mut(),
                )?
                .ok_or_else(|| {
                    invalid(format!(
                        "resident Metal EAGLE-3 tree verify became unavailable at target position {target_before}"
                    ))
                })?;
            self.phases.verify_us += verify_started.elapsed().as_micros();
            if verified.predictions.len() != actual_nodes {
                return Err(invalid(format!(
                    "EAGLE-3 tree target returned {} predictions for {actual_nodes} rows",
                    verified.predictions.len()
                )));
            }
            let acceptance = forest.accept_target_predictions(&verified.predictions)?;
            if acceptance.capture_rows.len() != acceptance.emitted_tokens.len() {
                return Err(invalid(format!(
                    "EAGLE-3 accepted tree capture/token lengths diverged: {}/{}",
                    acceptance.capture_rows.len(),
                    acceptance.emitted_tokens.len()
                )));
            }
            let update_started = std::time::Instant::now();
            drafter.accept_authoritative_forest(
                target_weights,
                &verified.layer_inputs,
                &acceptance,
            )?;
            self.phases.head_update_us += update_started.elapsed().as_micros();
            if early_exit {
                self.early_exit_rounds += 1;
            }
            Eagle3ServingRound {
                emitted: acceptance.emitted_tokens,
                offered: actual_nodes - 1,
                verify_nodes: actual_nodes,
                suffix: false,
                suffix_evidence,
                timings: verified.timings,
            }
        };

        if session.kv_position() != target_before + round.emitted.len() {
            return Err(invalid(format!(
                "EAGLE-3 target watermark advanced {} rows for {} emitted tokens",
                session.kv_position().saturating_sub(target_before),
                round.emitted.len()
            )));
        }
        let effective_head = self
            .pending_suffix_head
            .effective_filled(drafter.filled())?;
        if effective_head != session.kv_position() {
            return Err(invalid(format!(
                "EAGLE-3 cache watermarks diverged: materialized_head={} pending_head={} target={}",
                drafter.filled(),
                self.pending_suffix_head.pending_rows(),
                session.kv_position()
            )));
        }
        Ok(Some(round))
    }
}

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
impl Eagle3ServingState {
    fn bootstrap_cuda(
        &mut self,
        session: &mut LlamaInferenceSession,
        weights: &Arc<LlamaLoadedWeights>,
        prompt: &[u32],
        max_tokens: usize,
    ) -> Result<Eagle3ServingBootstrap> {
        if self.is_initialized() || prompt.len() < 3 || max_tokens == 0 {
            return Err(invalid("CUDA EAGLE bootstrap requires a fresh state, at least three prompt tokens and nonzero output budget"));
        }
        validate_logical_budget(prompt.len(), max_tokens)?;
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| invalid("missing EAGLE checkpoint"))?;
        if checkpoint.config.geometry() != crate::eagle3::Eagle3Geometry::QWEN {
            return Err(invalid(
                "CUDA EAGLE serving currently requires the pinned Qwen3-4B head",
            ));
        }
        session.set_resident_encode_ahead_enabled(false);
        let started = std::time::Instant::now();
        session.limit_cuda_eagle_context(configured_logical_token_limit()?)?;
        let capture = session
            .forward_greedy_cuda_prefill_with_layer_inputs(prompt, &self.capture_layer_ids)?;
        let anchor = *capture
            .predictions
            .last()
            .ok_or_else(|| invalid("empty CUDA prompt prediction"))?;
        let capacity = self
            .pooled_head
            .as_ref()
            .map_or(prompt.len() + max_tokens, |key| key.max_positions);
        let mut head = match self
            .pooled_head
            .as_ref()
            .map(|key| CUDA_SERVE_HEAD_POOL.checkout(key))
        {
            Some(Eagle3HeadCheckout::Hit(mut head)) => {
                head.reset();
                self.head_reused = true;
                head
            }
            _ => crate::eagle3_cuda::CudaEagle3Head::new(checkpoint, capacity)?,
        };
        if head.capacity() < prompt.len() + max_tokens {
            return Err(invalid("pooled CUDA head capacity is insufficient"));
        }
        let features =
            crate::eagle3_runtime::interleave_target_layer_inputs(&capture.layer_inputs)?;
        let mut paired = prompt[1..].to_vec();
        paired.push(anchor);
        let embeddings = weights
            .token_embedding
            .embedding_lookup(&paired, "eagle3_cuda_prompt_next_embeddings")?;
        for row in 0..prompt.len() {
            head.append(
                &features[row * 7680..(row + 1) * 7680],
                &embeddings.data[row * 2560..(row + 1) * 2560],
                row + 1 == prompt.len(),
            )?;
        }
        if head.filled() != session.kv_position() {
            return Err(invalid("CUDA EAGLE prompt watermarks disagree"));
        }
        self.phases.head_bootstrap_us += started.elapsed().as_micros();
        self.cuda_head = Some(head);
        self.checkpoint = None;
        tracing::info!(
            head_reused = self.head_reused,
            prompt_tokens = prompt.len(),
            "EAGLE-3 CUDA learned head initialized (Q8/128 weights, FP32 KV, confidence-admitted draft)"
        );
        Ok(Eagle3ServingBootstrap {
            first_token: anchor,
            timings: capture.timings,
        })
    }

    fn run_round_cuda(
        &mut self,
        session: &mut LlamaInferenceSession,
        weights: &Arc<LlamaLoadedWeights>,
        history: &[u32],
        remaining: usize,
    ) -> Result<Option<Eagle3ServingRound>> {
        if remaining == 0 || session.remaining_context() == 0 {
            return Ok(None);
        }
        let anchor = *history
            .last()
            .ok_or_else(|| invalid("empty CUDA EAGLE history"))?;
        let head = self
            .cuda_head
            .as_mut()
            .ok_or_else(|| invalid("CUDA EAGLE round before bootstrap"))?;
        if head.filled() != session.kv_position() {
            return Err(invalid("CUDA EAGLE authoritative watermarks disagree"));
        }
        if remaining == 1 || session.remaining_context() == 1 {
            let started = std::time::Instant::now();
            let token = session
                .generate_next_token_greedy_resident(anchor)?
                .ok_or_else(|| invalid("CUDA EAGLE target unavailable"))?
                .0;
            self.phases.verify_us += started.elapsed().as_micros();
            return Ok(Some(Eagle3ServingRound {
                emitted: vec![token],
                offered: 0,
                verify_nodes: 1,
                suffix: false,
                suffix_evidence: SuffixAdmissionEvidence::default(),
                timings: LlamaForwardTimings::default(),
            }));
        }
        // The Q4_K_M target admits two verify rows. The stable learned head
        // prediction supplies one draft; the target supplies the bonus/correction.
        let draft = head.next_token()?;
        // A two-row target pass costs more than one ordinary step. Admit a
        // learned proposal only when its probability gives useful headroom
        // over that cost; otherwise maintain the head with one authoritative row.
        let offered = usize::from(head.confidence().unwrap_or(0.0) >= 0.6);
        let started = std::time::Instant::now();
        let capture = if offered == 1 {
            session.verify_drafts_cuda_with_layer_inputs(
                anchor,
                &[draft],
                &self.capture_layer_ids,
            )?
        } else {
            session.forward_greedy_cuda_with_layer_inputs(anchor, &self.capture_layer_ids)?
        }
        .ok_or_else(|| {
            invalid("CUDA EAGLE verification unavailable; cannot change KV authority mid-request")
        })?;
        self.phases.verify_us += started.elapsed().as_micros();
        let accepted = if offered == 1 {
            accepted_draft_prefix(&[draft], &capture.predictions)
        } else {
            0
        };
        let emitted = capture.predictions[..=accepted].to_vec();
        let started = std::time::Instant::now();
        let features =
            crate::eagle3_runtime::interleave_target_layer_inputs(&capture.layer_inputs)?;
        let embeddings = weights
            .token_embedding
            .embedding_lookup(&emitted, "eagle3_cuda_authoritative_next_embeddings")?;
        for row in 0..emitted.len() {
            head.append(
                &features[row * 7680..(row + 1) * 7680],
                &embeddings.data[row * 2560..(row + 1) * 2560],
                row + 1 == emitted.len() && remaining > emitted.len() + 1,
            )?;
        }
        self.phases.head_update_us += started.elapsed().as_micros();
        if head.filled() != session.kv_position() {
            return Err(invalid("CUDA EAGLE commit watermarks disagree"));
        }
        tracing::debug!(
            draft,
            accepted,
            target_position = session.kv_position(),
            "EAGLE-3 CUDA learned verification"
        );
        Ok(Some(Eagle3ServingRound {
            emitted,
            offered,
            verify_nodes: offered + 1,
            suffix: false,
            suffix_evidence: SuffixAdmissionEvidence::default(),
            timings: capture.timings,
        }))
    }
}

impl Drop for Eagle3ServingState {
    fn drop(&mut self) {
        // A panicking generation may have left a Metal command mid-flight;
        // never pool that head.
        if std::thread::panicking() {
            return;
        }
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        if let Some(mut head) = self.cuda_head.take() {
            if let Some(key) = self.pooled_head.take() {
                head.reset();
                CUDA_SERVE_HEAD_POOL.restore(key, head);
            }
            return;
        }
        let (Some(key), Some(mut drafter)) = (self.pooled_head.take(), self.drafter.take()) else {
            return;
        };
        drafter.reset_for_reuse();
        SERVE_HEAD_POOL.restore(key, drafter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_width_is_strict_and_keeps_dynamic_tree_at_certified_point() {
        assert!(Eagle3ServingConfig::new(0).is_err());
        assert!(Eagle3ServingConfig::new(MAX_DRAFT_TOKENS + 1).is_err());
        let config =
            Eagle3ServingConfig::with_dynamic_tree(MAX_DRAFT_TOKENS, Eagle3DynamicTree::DEFAULT)
                .unwrap();
        assert_eq!(config.draft_tokens + 1, TREE_MAX_NODES);
        assert_eq!(config.dynamic_verify_nodes, 8);
        assert_eq!(config.dynamic_top_k, 4);
        assert_eq!(config.dynamic_expansions, 5);
        assert_eq!(Eagle3DynamicTree::DEFAULT.label(), "N8/K4/X5");
    }

    #[test]
    fn serve_tree_env_keeps_defaults_and_accepts_in_range_overrides() {
        let default = dynamic_tree_from_values(None, None, None).unwrap();
        assert_eq!(default, Eagle3DynamicTree::DEFAULT);
        assert_eq!(
            dynamic_tree_from_values(Some(""), Some("  "), Some("")).unwrap(),
            Eagle3DynamicTree::DEFAULT
        );
        assert_eq!(
            dynamic_tree_from_values(Some("8"), Some("4"), Some(" 5 ")).unwrap(),
            Eagle3DynamicTree::DEFAULT
        );
        assert_eq!(
            dynamic_tree_from_values(None, None, Some("4")).unwrap(),
            Eagle3DynamicTree {
                expansions: 4,
                ..Eagle3DynamicTree::DEFAULT
            }
        );
        assert_eq!(
            dynamic_tree_from_values(Some("16"), Some("8"), Some("32")).unwrap(),
            Eagle3DynamicTree {
                verify_nodes: 16,
                top_k: 8,
                expansions: 32,
            }
        );
        assert_eq!(
            dynamic_tree_from_values(Some("2"), Some("1"), Some("1")).unwrap(),
            Eagle3DynamicTree {
                verify_nodes: 2,
                top_k: 1,
                expansions: 1,
            }
        );
    }

    #[test]
    fn serve_tree_env_rejects_the_whole_override_on_any_bad_field() {
        for (nodes, top_k, expansions) in [
            (Some("1"), None, None),
            (Some("17"), None, None),
            (Some("0"), None, None),
            (None, Some("0"), None),
            (None, Some("9"), None),
            (None, None, Some("0")),
            (None, None, Some("33")),
            (Some("eight"), None, None),
            (None, Some("4.0"), None),
            (None, None, Some("-5")),
            // One bad field poisons an otherwise valid override.
            (Some("12"), Some("4"), Some("zero")),
        ] {
            let error = dynamic_tree_from_values(nodes, top_k, expansions).unwrap_err();
            assert!(
                error.contains("CAMELID_EAGLE3_SERVE_TREE_"),
                "{nodes:?}/{top_k:?}/{expansions:?}: {error}"
            );
        }
        let error = dynamic_tree_from_values(Some("12"), Some("4"), Some("zero")).unwrap_err();
        assert!(error.contains(SERVE_TREE_EXPANSIONS_ENV), "{error}");
    }

    fn head_key(tag: &str) -> Eagle3ServeHeadKey {
        Eagle3ServeHeadKey {
            checkpoint_path: PathBuf::from(format!("/heads/{tag}")),
            checkpoint_sha256: format!("{tag}-head-sha"),
            target_sha256: format!("{tag}-target-sha"),
            draft_wire: "body=q4_k lm_head=q4_k rows=32000".to_string(),
            max_positions: 2_064,
        }
    }

    #[test]
    fn serve_head_pool_hands_out_one_head_and_never_shares_it() {
        let pool = Eagle3HeadPool::<Arc<()>>::new();
        let key = head_key("a");
        assert!(matches!(pool.checkout(&key), Eagle3HeadCheckout::Empty));
        assert!(!pool.is_populated());

        let head = Arc::new(());
        assert!(pool.restore(key.clone(), Arc::clone(&head)));
        assert!(pool.is_populated());
        // A second restore into the occupied slot is dropped, not queued.
        let extra = Arc::new(());
        assert!(!pool.restore(key.clone(), Arc::clone(&extra)));
        assert_eq!(Arc::strong_count(&extra), 1);

        let Eagle3HeadCheckout::Hit(checked_out) = pool.checkout(&key) else {
            panic!("matching key must hit")
        };
        assert!(Arc::ptr_eq(&checked_out, &head));
        // While one generation owns it, another cannot get the same head.
        assert!(matches!(pool.checkout(&key), Eagle3HeadCheckout::Empty));
        assert!(pool.restore(key.clone(), checked_out));
        assert!(matches!(pool.checkout(&key), Eagle3HeadCheckout::Hit(_)));
    }

    #[test]
    fn serve_head_pool_invalidates_on_any_identity_change() {
        let base = head_key("a");
        let variants = [
            Eagle3ServeHeadKey {
                checkpoint_path: PathBuf::from("/heads/other"),
                ..base.clone()
            },
            Eagle3ServeHeadKey {
                checkpoint_sha256: "other-head-sha".to_string(),
                ..base.clone()
            },
            Eagle3ServeHeadKey {
                target_sha256: "other-target-sha".to_string(),
                ..base.clone()
            },
            Eagle3ServeHeadKey {
                draft_wire: "body=q8_0 lm_head=q4_k rows=32000".to_string(),
                ..base.clone()
            },
            Eagle3ServeHeadKey {
                max_positions: 4_112,
                ..base.clone()
            },
        ];
        for changed in variants {
            let pool = Eagle3HeadPool::<Arc<()>>::new();
            let head = Arc::new(());
            assert!(pool.restore(base.clone(), Arc::clone(&head)));
            let Eagle3HeadCheckout::Invalidated(stale) = pool.checkout(&changed) else {
                panic!("{changed:?} must invalidate the pooled head")
            };
            assert_eq!(stale, base);
            // The stale head was dropped inside the pool, not leaked or kept.
            assert_eq!(Arc::strong_count(&head), 1);
            assert!(!pool.is_populated());
            assert!(matches!(pool.checkout(&base), Eagle3HeadCheckout::Empty));
        }
    }

    #[test]
    fn pooled_head_capacity_covers_the_whole_serving_envelope() {
        for limit in SUPPORTED_LOGICAL_TOKEN_LIMITS {
            let pooled = Eagle3ServeHeadKey::pooled_max_positions(limit).unwrap();
            assert_eq!(pooled, limit + TREE_MAX_NODES);
            // Every admissible (prompt, max_tokens) pair at the widest draft
            // fits the pooled allocation, so a hit never needs a re-upload.
            for prompt_tokens in [3, limit / 2, limit - 1] {
                let max_tokens =
                    clamp_max_tokens_to_logical_budget_at_limit(prompt_tokens, limit, limit)
                        .unwrap();
                assert!(prompt_tokens + max_tokens + MAX_DRAFT_TOKENS < pooled);
            }
        }
        assert!(Eagle3ServeHeadKey::pooled_max_positions(usize::MAX).is_err());
    }

    #[test]
    fn logical_budget_ladder_is_explicit_and_fail_closed() {
        assert_eq!(logical_token_limit_from_value(None).unwrap(), 2_048);
        for limit in SUPPORTED_LOGICAL_TOKEN_LIMITS {
            assert_eq!(
                logical_token_limit_from_value(Some(&limit.to_string())).unwrap(),
                limit
            );
            assert_eq!(
                validate_logical_budget_at_limit(limit / 2, limit / 2, limit).unwrap(),
                limit
            );
            assert!(validate_logical_budget_at_limit(limit / 2, limit / 2 + 1, limit).is_err());
        }
        assert!(logical_token_limit_from_value(Some("4097")).is_err());
        assert!(validate_logical_budget_at_limit(usize::MAX, 1, 8_192).is_err());
    }

    #[test]
    fn api_max_tokens_is_clamped_to_the_verified_logical_room() {
        for limit in SUPPORTED_LOGICAL_TOKEN_LIMITS {
            assert_eq!(
                clamp_max_tokens_to_logical_budget_at_limit(28, 8_192, limit).unwrap(),
                (limit - 28).min(8_192)
            );
            assert_eq!(
                clamp_max_tokens_to_logical_budget_at_limit(limit - 1, 8_192, limit).unwrap(),
                1
            );
            assert!(clamp_max_tokens_to_logical_budget_at_limit(limit, 1, limit).is_err());
        }
    }

    #[test]
    fn deep_context_suffixes_stay_on_the_eight_row_fast_path() {
        assert_eq!(suffix_verify_node_limit(2_032), TREE_MAX_NODES);
        assert_eq!(suffix_verify_node_limit(2_033), DEEP_CONTEXT_VERIFY_NODES);
        assert_eq!(suffix_verify_node_limit(4_000), DEEP_CONTEXT_VERIFY_NODES);
        assert_eq!(cap_verify_nodes_for_position(4_000, 8), 8);
        // Nine rows from base 2,039 have position-counts 2,040..=2,048,
        // exactly the inclusive wide-batch boundary proved by Metal.
        assert_eq!(cap_verify_nodes_for_position(2_039, 9), 9);
        // Advancing the base once makes the final position-count 2,049, so
        // the same request must narrow to the deep-safe eight-row lane.
        assert_eq!(cap_verify_nodes_for_position(2_040, 9), 8);
        assert_eq!(cap_verify_nodes_for_position(2_039, 6), 6);
    }
}
