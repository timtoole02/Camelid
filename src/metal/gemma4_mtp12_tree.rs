//! Opt-in draft-only W8 tree. The ordinary resident-chain entry points do not
//! call this module, allocate its scratch, or change their command sequence.
use super::*;
use crate::gemma4_mtp12_tree_menu as menu;
#[cfg(test)]
#[path = "gemma4_mtp12_tree_oracle.rs"]
mod oracle;

const PRIMARY: usize = 4;
const DEFAULT_MAX_MARGIN: f32 = 2.0;
const MAX_MARGIN_ENV: &str = "CAMELID_GEMMA4_MTP12_TREE_MAX_MARGIN";

fn parse_max_margin(value: Option<&str>) -> Result<f32> {
    let Some(text) = value else {
        return Ok(DEFAULT_MAX_MARGIN);
    };
    let margin = text.parse::<f32>().map_err(|_| {
        invalid(format!(
            "{MAX_MARGIN_ENV} must be a finite nonnegative number; got {text:?}"
        ))
    })?;
    if !margin.is_finite() || margin.is_sign_negative() {
        return Err(invalid(format!(
            "{MAX_MARGIN_ENV} must be a finite nonnegative number; got {text:?}"
        )));
    }
    Ok(margin)
}

/// Read one selector once and parse it strictly, exactly like the margin
/// selector: absent is the documented default, anything unparseable is an
/// error rather than a silent fallback.
fn env_once<T>(name: &str, parse: fn(Option<&str>) -> std::result::Result<T, String>) -> Result<T> {
    match std::env::var(name) {
        Ok(value) => parse(Some(&value)).map_err(invalid),
        Err(std::env::VarError::NotPresent) => parse(None).map_err(invalid),
        Err(error) => Err(invalid(format!("{name}: {error}"))),
    }
}

fn max_margin_from_env() -> Result<f32> {
    match std::env::var(MAX_MARGIN_ENV) {
        Ok(value) => parse_max_margin(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_max_margin(None),
        Err(error) => Err(invalid(format!("{MAX_MARGIN_ENV}: {error}"))),
    }
}

#[derive(Clone, Debug)]
pub struct Gemma4Mtp12TreeProposal {
    /// Eight physical verifier rows, including the already selected anchor.
    pub tokens: Vec<u32>,
    /// Root=-1; every other parent precedes its child in physical row order.
    pub parents: Vec<i32>,
    /// Logical token position relative to the anchor, independent of row order.
    pub depths: Vec<u32>,
    /// Physical rows of the ordinary primary chain, including the anchor.
    pub primary_rows: Vec<usize>,
    /// Zero-based primary query where the rank-two fork was selected. Under
    /// every policy this stays the EARLIEST kept fork among the first four
    /// forwards, or `None` on a linear tree, so existing receipt readers keep
    /// their meaning; `fork_forwards` carries the rest.
    pub branch_primary_step: Option<usize>,
    /// Every primary forward whose rank-two child this tree kept, ascending.
    pub fork_forwards: Vec<usize>,
    pub primary_margins: [f32; PRIMARY],
    /// Top-1 minus top-2 logit of EVERY forward this round ran, in forward
    /// order. Legacy records only its four primaries; the menu policies record
    /// all of them, which is the out-of-sample data a recalibration needs.
    pub forward_margins: Vec<f32>,
    /// Rank-two id of every forward in `forward_margins`, same order.
    pub runner_up_ids: Vec<u32>,
    /// Modeled probability that each physical row is committed, row order.
    /// Empty on the legacy path, whose continuations record no top-2.
    pub node_p: Vec<f32>,
    /// Selector value that produced this round: `legacy`, `dyn`, `fixed:<shape>`.
    pub policy: String,
    /// Named topology this round emitted, e.g. `4+1+2`, `5+1+1`, `lin7`.
    pub shape: String,
    /// Four to seven forwards, decided by the chosen shape.
    pub assistant_steps: usize,
    pub timing: Gemma4Mtp12ChainTiming,
    pub ledger: Gemma4Mtp12ChainLedger,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TopTwo {
    values: [f32; 2],
    ids: [u32; 2],
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct TopTwo { float2 values; uint2 ids; };
inline bool better(float value, uint id, float old, uint old_id) {
    return value > old || (value == old && id < old_id);
}
inline void insert_pair(float value, uint id, thread float2& values, thread uint2& ids) {
    if (isnan(value) || id == 0xffffffffu || id == ids.x || id == ids.y) return;
    if (better(value, id, values.x, ids.x)) {
        values.y = values.x; ids.y = ids.x; values.x = value; ids.x = id;
    } else if (better(value, id, values.y, ids.y)) {
        values.y = value; ids.y = id;
    }
}
kernel void mtp12_tree_top2_partial(
    device const float* logits [[buffer(0)]], device TopTwo* partials [[buffer(1)]],
    constant uint& count [[buffer(2)]], constant uint& chunk [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 values[256]; threadgroup uint2 ids[256];
    float2 mine = float2(-INFINITY); uint2 mine_ids = uint2(0xffffffffu);
    uint end = min(count, (group + 1u) * chunk);
    for (uint i = group * chunk + tid; i < end; i += 256u)
        insert_pair(logits[i], i, mine, mine_ids);
    values[tid] = mine; ids[tid] = mine_ids;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride; stride >>= 1u) {
        if (tid < stride) {
            mine = values[tid]; mine_ids = ids[tid];
            insert_pair(values[tid+stride].x, ids[tid+stride].x, mine, mine_ids);
            insert_pair(values[tid+stride].y, ids[tid+stride].y, mine, mine_ids);
            values[tid] = mine; ids[tid] = mine_ids;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { partials[group].values = values[0]; partials[group].ids = ids[0]; }
}
kernel void mtp12_tree_top2_merge(
    device const TopTwo* partials [[buffer(0)]], constant uint& count [[buffer(1)]],
    device TopTwo* result [[buffer(2)]], device uint* output_token [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 values[256]; threadgroup uint2 ids[256];
    float2 mine = float2(-INFINITY); uint2 mine_ids = uint2(0xffffffffu);
    for (uint i = tid; i < count; i += 256u) {
        insert_pair(partials[i].values.x, partials[i].ids.x, mine, mine_ids);
        insert_pair(partials[i].values.y, partials[i].ids.y, mine, mine_ids);
    }
    values[tid] = mine; ids[tid] = mine_ids;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride; stride >>= 1u) {
        if (tid < stride) {
            mine = values[tid]; mine_ids = ids[tid];
            insert_pair(values[tid+stride].x, ids[tid+stride].x, mine, mine_ids);
            insert_pair(values[tid+stride].y, ids[tid+stride].y, mine, mine_ids);
            values[tid] = mine; ids[tid] = mine_ids;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        uint2 answer = ids[0];
        // Existing argmax starts at (-inf, id=0), so its degenerate
        // all-NaN/-inf answer remains zero even when logit[0] is NaN.
        if (values[0].x == -INFINITY) {
            if (answer.x != 0u) answer.y = answer.x;
            answer.x = 0u;
        }
        result[0].values = values[0]; result[0].ids = answer;
        output_token[0] = answer.x;
    }
}
"#;

pub(super) struct TreeState {
    // Read once on this assistant's first tree proposal. Ordinary linear
    // drafting never reads the experimental selectors or allocates this state.
    max_margin: f32,
    policy: menu::Policy,
    lambda: f32,
    calib: menu::Calibration,
    partial: ComputePipelineState,
    merge: ComputePipelineState,
    partials: Buffer,
    results: Buffer,
}

impl TreeState {
    fn new(device: &Device) -> Result<Self> {
        let max_margin = max_margin_from_env()?;
        let policy = env_once(menu::POLICY_ENV, menu::parse_policy)?;
        let lambda = env_once(menu::LAMBDA_ENV, menu::parse_lambda)?;
        let calib = env_once(menu::CALIB_ENV, menu::parse_calibration)?;
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(SHADER, &options)
            .map_err(|e| invalid(format!("tree top2 shader: {e}")))?;
        let pipeline = |name| -> Result<ComputePipelineState> {
            let function = library.get_function(name, None).map_err(invalid)?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(invalid)
        };
        Ok(Self {
            max_margin,
            policy,
            lambda,
            calib,
            partial: pipeline("mtp12_tree_top2_partial")?,
            merge: pipeline("mtp12_tree_top2_merge")?,
            partials: shared_buffer(device, MTP12_ARGMAX_MAX_PARTIALS * 16),
            // One TopTwo per forward, not per primary: the menu policies read
            // the top-2 of the continuation forwards as well.
            results: shared_buffer(device, MTP12_CHAIN_MAX_DRAFTS * 16),
        })
    }

    fn byte_len(&self) -> u64 {
        self.partials.length() + self.results.length()
    }

    fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        logits: &Buffer,
        output: &Buffer,
        primary_step: usize,
        count: usize,
    ) {
        let count = count as u32;
        let chunk = MTP12_ARGMAX_CHUNK as u32;
        let partials = count.div_ceil(chunk);
        encoder.set_compute_pipeline_state(&self.partial);
        encoder.set_buffer(0, Some(logits), 0);
        encoder.set_buffer(1, Some(&self.partials), 0);
        encoder.set_bytes(2, 4, &count as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &chunk as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: partials as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&self.merge);
        encoder.set_buffer(0, Some(&self.partials), 0);
        encoder.set_bytes(1, 4, &partials as *const u32 as *const c_void);
        encoder.set_buffer(2, Some(&self.results), (primary_step * 16) as u64);
        encoder.set_buffer(3, Some(output), ((primary_step + 1) * 4) as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
}

fn select_branch(top: &[TopTwo; PRIMARY], max_margin: f32) -> Option<usize> {
    top.iter().position(|pair| {
        let gap = pair.values[0] - pair.values[1];
        pair.ids[0] < VOCAB as u32
            && pair.ids[1] < VOCAB as u32
            && pair.ids[0] != pair.ids[1]
            && gap.is_finite()
            && (0.0..=max_margin).contains(&gap)
    })
}

fn topology(branch: Option<usize>) -> (Vec<i32>, Vec<u32>, Vec<usize>) {
    let mut parents = vec![-1, 0, 1, 2, 3, 4, 5, 6];
    let mut depths: Vec<u32> = (0..8).collect();
    if let Some(step) = branch {
        assert!(step < PRIMARY);
        parents[5] = step as i32;
        depths[5] = step as u32 + 1;
        depths[6] = step as u32 + 2;
        depths[7] = step as u32 + 3;
        (parents, depths, (0..=4).collect())
    } else {
        (parents, depths, (0..=7).collect())
    }
}

impl Gemma4Mtp12AssistantMetal {
    #[allow(clippy::too_many_arguments)]
    fn encode_tree_step(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        table: Gemma4Mtp12Q6KEmbeddingTable<'_>,
        sliding: Gemma4Mtp12DeviceKv<'_>,
        full: Gemma4Mtp12DeviceKv<'_>,
        prefix: usize,
        position: usize,
        query_step: usize,
        input_row: usize,
        output_row: usize,
        recurrent: &BufferRef,
        recurrent_offset: u64,
        history_step: usize,
        scores: &Buffer,
        top2: bool,
    ) {
        encode_q6k_embedding_and_recurrent_gather(
            encoder,
            &self.pipelines.gather_q6k_embedding_and_recurrent,
            &self.scratch.output_token,
            input_row,
            table,
            recurrent,
            recurrent_offset,
            &self.scratch.pre_input,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.pre_input,
            &self.scratch.hidden,
            self.layout.pre_projection,
            true,
        );
        // Under FUSE_NORM each layer's tail computes the NEXT layer's input
        // norm (or the final norm), so only layer 0's input norm stands alone
        // and no separate final rms_norm may follow (rms_norm is not
        // idempotent).  Unset, encode_layer_k1_fused encodes exactly the
        // established per-layer list and the final norm is encoded here.
        let fuse_norm = self.fuse.norm;
        if fuse_norm {
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.hidden,
                &self.layers[0].input_norm,
                &self.scratch.normed,
                ASSISTANT_HIDDEN,
                1,
            );
        }
        for layer in 0..N_LAYERS {
            let local = layer < 3;
            let (kv, heads, dim, cos, sin) = if local {
                (
                    sliding,
                    LOCAL_KV_HEADS,
                    LOCAL_HEAD_DIM,
                    &self.scratch.local_cos,
                    &self.scratch.local_sin,
                )
            } else {
                (
                    full,
                    FULL_KV_HEADS,
                    FULL_HEAD_DIM,
                    &self.scratch.full_cos,
                    &self.scratch.full_sin,
                )
            };
            let compact = if local {
                chain_query_position(position, query_step, self.single_position)
                    .saturating_sub(LOCAL_WINDOW + 1)
                    .min(prefix)
            } else {
                0
            };
            let (next_norm, next_normed) = if layer + 1 < N_LAYERS {
                (&self.layers[layer + 1].input_norm, &self.scratch.normed)
            } else {
                (&self.final_norm, &self.scratch.final_normalized)
            };
            self.encode_layer_k1_fused(
                encoder,
                layer,
                kv.key.buffer,
                kv.value.buffer,
                kv.key.byte_offset,
                kv.value.byte_offset,
                kv.max_positions,
                heads,
                dim,
                prefix,
                compact,
                prefix - compact,
                cos,
                sin,
                (query_step * dim / 2 * 4) as u64,
                scores,
                next_norm,
                next_normed,
            );
        }
        if !fuse_norm {
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.hidden,
                &self.final_norm,
                &self.scratch.final_normalized,
                ASSISTANT_HIDDEN,
                1,
            );
        }
        #[cfg(test)]
        encode_copy_f32_to_offset(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.final_normalized,
            &self.scratch.chain_final_normalized,
            (history_step * ASSISTANT_HIDDEN * 4) as u64,
            ASSISTANT_HIDDEN,
        );
        // The post projection lands directly in this step's history slot;
        // the next step's gather reads it back from there (no copy dispatch).
        let recurrent_byte_offset = (history_step * TARGET_HIDDEN * 4) as u64;
        debug_assert!(
            recurrent_byte_offset + (TARGET_HIDDEN * 4) as u64
                <= self.scratch.chain_recurrent_hidden.length(),
            "tree history slot {history_step} exceeds chain_recurrent_hidden"
        );
        self.encode_dense_gemv_at_offset(
            encoder,
            &self.scratch.final_normalized,
            &self.scratch.chain_recurrent_hidden,
            recurrent_byte_offset,
            self.layout.post_projection,
            true,
        );
        self.encode_draft_head(encoder);
        if top2 {
            self.tree_state.as_ref().unwrap().encode(
                encoder,
                &self.scratch.logits,
                &self.scratch.output_token,
                history_step,
                VOCAB,
            );
        } else {
            self.encode_vocab_argmax(encoder, (output_row * 4) as u64);
        }
    }

    /// Explicit W8 tree entry point. Caller must use ordinary linear drafting
    /// for shorter output-budget tails and must keep target verification authoritative.
    /// `CAMELID_GEMMA4_MTP12_TREE_MAX_MARGIN` is read once on this assistant's
    /// first tree call (default 2); a new assistant is required to change it.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_tree_w8_from_cpu_hidden(
        &mut self,
        anchor_token: u32,
        initial_hidden: &[f32],
        table: Gemma4Mtp12Q6KEmbeddingTable<'_>,
        sliding: Gemma4Mtp12DeviceKv<'_>,
        full: Gemma4Mtp12DeviceKv<'_>,
        target_kv_len: usize,
        proposal_position: usize,
    ) -> Result<Gemma4Mtp12TreeProposal> {
        let started = Instant::now();
        if initial_hidden.len() != TARGET_HIDDEN
            || initial_hidden.iter().any(|v| !v.is_finite())
            || anchor_token as usize >= VOCAB
        {
            return Err(invalid("invalid tree initial hidden or anchor"));
        }
        let device = self.packed_q4.device().to_owned();
        let registry = device.registry_id();
        table.validate(registry)?;
        sliding.validate(
            registry,
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            "tree sliding",
        )?;
        full.validate(
            registry,
            GEMMA4_12B_MTP_FULL_HOST_LAYER,
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            "tree full",
        )?;
        validate_device_chain_positions(
            sliding.max_positions,
            full.max_positions,
            target_kv_len,
            proposal_position,
            7,
        )?;
        if self.tree_state.is_none() {
            let tree = TreeState::new(&device)?;
            self.resident_ledger.fixed_scratch_bytes += tree.byte_len();
            self.tree_state = Some(tree);
        }
        let (policy, lambda, calib, max_margin) = {
            let tree = self.tree_state.as_ref().expect("tree state just installed");
            (tree.policy, tree.lambda, tree.calib, tree.max_margin)
        };
        write_buffer_f32(&self.scratch.chain_initial_recurrent_hidden, initial_hidden)?;
        write_chain_rope_tables(proposal_position, 7, self.single_position, &self.scratch)?;
        unsafe {
            *self.scratch.output_token.contents().cast::<u32>() = anchor_token;
        }
        let scores = shared_buffer(
            &device,
            N_HEADS
                .checked_mul(target_kv_len)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| invalid("tree attention scratch overflow"))?,
        );
        let mut timing = Gemma4Mtp12ChainTiming {
            cpu_prepare_us: started.elapsed().as_micros(),
            ..Default::default()
        };
        let first = self.queue.new_command_buffer();
        let encoder = first.new_compute_command_encoder();
        let encoded = Instant::now();
        for step in 0..PRIMARY {
            let (recurrent, offset): (&BufferRef, u64) = if step == 0 {
                (&self.scratch.chain_initial_recurrent_hidden, 0)
            } else {
                (
                    &self.scratch.chain_recurrent_hidden,
                    ((step - 1) * TARGET_HIDDEN * 4) as u64,
                )
            };
            self.encode_tree_step(
                encoder,
                table,
                sliding,
                full,
                target_kv_len,
                proposal_position,
                step,
                step,
                step + 1,
                recurrent,
                offset,
                step,
                &scores,
                true,
            );
        }
        encoder.end_encoding();
        timing.encode_us += encoded.elapsed().as_micros();
        first.commit();
        let waited = Instant::now();
        first.wait_until_completed();
        timing.wait_us += waited.elapsed().as_micros();
        if first.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid("tree primary command failed"));
        }
        let (gpu, kernel) = super::super::command_buffer_gpu_times_us(&first.to_owned());
        timing.gpu_us += gpu;
        timing.kernel_us += kernel;
        let top: [TopTwo; PRIMARY] = unsafe {
            std::slice::from_raw_parts(
                self.tree_state
                    .as_ref()
                    .unwrap()
                    .results
                    .contents()
                    .cast::<TopTwo>(),
                PRIMARY,
            )
        }
        .try_into()
        .unwrap();
        let margins: [f32; PRIMARY] = std::array::from_fn(|i| top[i].values[0] - top[i].values[1]);
        if policy == menu::Policy::Legacy {
            // Byte-for-byte the qualified V3 proposal: the same earliest
            // eligible fork rule, the same two argmax continuations off the
            // rank-two slot, the same linear fallback and the same ledger.
            let branch = select_branch(&top, max_margin);
            let prepare = Instant::now();
            if let Some(step) = branch {
                unsafe {
                    *self.scratch.output_token.contents().cast::<u32>().add(5) = top[step].ids[1];
                }
            }
            timing.cpu_prepare_us += prepare.elapsed().as_micros();
            let second = self.queue.new_command_buffer();
            let encoder = second.new_compute_command_encoder();
            let encoded = Instant::now();
            let mut query_steps: Vec<usize> = (0..PRIMARY).collect();
            if let Some(step) = branch {
                for continuation in 0..2 {
                    // Continuation A resumes from the forked primary's slot;
                    // continuation B from A's own slot (history step 4).
                    let history_slot = if continuation == 0 { step } else { 4 };
                    let (recurrent, offset): (&BufferRef, u64) = (
                        &self.scratch.chain_recurrent_hidden,
                        (history_slot * TARGET_HIDDEN * 4) as u64,
                    );
                    let query = step + 1 + continuation;
                    query_steps.push(query);
                    self.encode_tree_step(
                        encoder,
                        table,
                        sliding,
                        full,
                        target_kv_len,
                        proposal_position,
                        query,
                        5 + continuation,
                        6 + continuation,
                        recurrent,
                        offset,
                        4 + continuation,
                        &scores,
                        false,
                    );
                }
            } else {
                for step in 4..7 {
                    query_steps.push(step);
                    self.encode_tree_step(
                        encoder,
                        table,
                        sliding,
                        full,
                        target_kv_len,
                        proposal_position,
                        step,
                        step,
                        step + 1,
                        &self.scratch.chain_recurrent_hidden,
                        ((step - 1) * TARGET_HIDDEN * 4) as u64,
                        step,
                        &scores,
                        false,
                    );
                }
            }
            encoder.end_encoding();
            timing.encode_us += encoded.elapsed().as_micros();
            second.commit();
            let waited = Instant::now();
            second.wait_until_completed();
            timing.wait_us += waited.elapsed().as_micros();
            if second.status() != MTLCommandBufferStatus::Completed {
                return Err(invalid("tree continuation command failed"));
            }
            let (gpu, kernel) = super::super::command_buffer_gpu_times_us(&second.to_owned());
            timing.gpu_us += gpu;
            timing.kernel_us += kernel;
            let tokens = unsafe {
                std::slice::from_raw_parts(self.scratch.output_token.contents().cast::<u32>(), 8)
            }
            .to_vec();
            if tokens.iter().any(|t| *t as usize >= VOCAB) {
                return Err(invalid("tree returned invalid token"));
            }
            let (parents, depths, primary_rows) = topology(branch);
            let mut kv_reads = 0u64;
            for &step in &query_steps {
                let compact = chain_query_position(proposal_position, step, self.single_position)
                    .saturating_sub(LOCAL_WINDOW + 1)
                    .min(target_kv_len);
                kv_reads = kv_reads
                    .checked_add(target_kv_read_bytes(
                        target_kv_len - compact,
                        target_kv_len,
                    )?)
                    .ok_or_else(|| invalid("tree KV read ledger overflow"))?;
            }
            let steps = query_steps.len();
            let ledger = Gemma4Mtp12ChainLedger {
                draft_k: 7,
                command_buffers: 2,
                command_buffer_waits: 2,
                target_q6k_table_alias_bytes: table.wire.byte_len,
                target_kv_alias_bytes: sliding.key.byte_len
                    + sliding.value.byte_len
                    + full.key.byte_len
                    + full.value.byte_len,
                assistant_matrix_read_bytes: self
                    .dense_bf16
                    .as_ref()
                    .map_or(FULL_Q4_MATRIX_BYTES, |dense| {
                        self.layout.embedding.byte_len + dense.byte_len()
                    })
                    * steps as u64,
                target_kv_read_bytes: kv_reads,
                dynamic_attention_scratch_bytes: scores.length(),
                resident_chain_state_bytes: self.scratch.chain_initial_recurrent_hidden.length()
                    + self.scratch.chain_recurrent_hidden.length()
                    + self.scratch.output_token.length()
                    + self.tree_state.as_ref().unwrap().byte_len(),
                initial_hidden_upload_bytes: (TARGET_HIDDEN * 4) as u64,
                readback_bytes: (PRIMARY * 16 + 8 * 4) as u64,
            };
            timing.wall_us = started.elapsed().as_micros();
            return Ok(Gemma4Mtp12TreeProposal {
                tokens,
                parents,
                depths,
                primary_rows,
                branch_primary_step: branch,
                fork_forwards: branch.into_iter().collect(),
                primary_margins: margins,
                forward_margins: margins.to_vec(),
                runner_up_ids: top.iter().map(|pair| pair.ids[1]).collect(),
                // The legacy continuations use the argmax kernel, so they have
                // no recorded rank-two answer and no modeled node probability.
                node_p: Vec::new(),
                policy: policy.name(),
                shape: if branch.is_some() { "4+1+2" } else { "lin7" }.to_string(),
                assistant_steps: steps,
                timing,
                ledger,
            });
        }
        // Menu policies choose the shape from the four primary margins, run
        // only the forwards that shape needs, and assemble the eight rows from
        // the recorded top-2 of every forward.  The node set is fixed by the
        // shape, never re-ranked, so the runtime can only emit topologies that
        // `menu::gate_topologies` enumerates and the model gate covers.
        let primary_top: [menu::ForwardTop; menu::PRIMARY] =
            std::array::from_fn(|i| menu::ForwardTop::from_pair(top[i].values, top[i].ids, VOCAB));
        let choices = menu::Menu::new(&primary_top, calib);
        let shape = menu::choose(policy, &choices, lambda);
        let alt_steps = choices
            .alt_steps(shape)
            .ok_or_else(|| invalid("tree policy chose a shape this round cannot build"))?;
        let plan = menu::layout(shape, &alt_steps)
            .ok_or_else(|| invalid("tree policy produced an unbuildable layout"))?;
        let prepare = Instant::now();
        if let Some((slot, step)) = plan.runner_up_write {
            // The expanded rank-two token is the only forward input the GPU
            // does not write; slots 8.. never collide with the GPU's 1..=7.
            unsafe {
                *self.scratch.output_token.contents().cast::<u32>().add(slot) = top[step].ids[1];
            }
        }
        timing.cpu_prepare_us += prepare.elapsed().as_micros();
        let mut query_steps: Vec<usize> = (0..PRIMARY).collect();
        let mut command_buffers = 1u32;
        if !plan.cb2.is_empty() {
            command_buffers = 2;
            let second = self.queue.new_command_buffer();
            let encoder = second.new_compute_command_encoder();
            let encoded = Instant::now();
            for spec in &plan.cb2 {
                query_steps.push(spec.query_step);
                self.encode_tree_step(
                    encoder,
                    table,
                    sliding,
                    full,
                    target_kv_len,
                    proposal_position,
                    spec.query_step,
                    spec.input_slot,
                    spec.history_step + 1,
                    // Always the explicit history slot of the forward that
                    // produced this node's parent.  The live recurrent buffer
                    // is never reused: CB2 interleaves parents (5+1+1 runs
                    // chain node 5 and then a sibling from an earlier step).
                    &self.scratch.chain_recurrent_hidden,
                    (spec.recurrent_slot * TARGET_HIDDEN * 4) as u64,
                    spec.history_step,
                    &scores,
                    true,
                );
            }
            encoder.end_encoding();
            timing.encode_us += encoded.elapsed().as_micros();
            second.commit();
            let waited = Instant::now();
            second.wait_until_completed();
            timing.wait_us += waited.elapsed().as_micros();
            if second.status() != MTLCommandBufferStatus::Completed {
                return Err(invalid("tree continuation command failed"));
            }
            let (gpu, kernel) = super::super::command_buffer_gpu_times_us(&second.to_owned());
            timing.gpu_us += gpu;
            timing.kernel_us += kernel;
        }
        let steps = shape.forwards();
        let gpu_tokens = unsafe {
            std::slice::from_raw_parts(self.scratch.output_token.contents().cast::<u32>(), 8)
        }
        .to_vec();
        // Every forward records its own top-2 now, so the receipt carries the
        // out-of-sample margins a later calibration needs.
        let forward_top: Vec<menu::ForwardTop> = unsafe {
            std::slice::from_raw_parts(
                self.tree_state
                    .as_ref()
                    .unwrap()
                    .results
                    .contents()
                    .cast::<TopTwo>(),
                steps,
            )
        }
        .iter()
        .map(|pair| menu::ForwardTop::from_pair(pair.values, pair.ids, VOCAB))
        .collect();
        let finalized = menu::finalize(&plan, &forward_top, &gpu_tokens, anchor_token, calib)
            .map_err(invalid)?;
        if finalized.tokens.iter().any(|t| *t as usize >= VOCAB) {
            return Err(invalid("tree returned invalid token"));
        }
        let mut kv_reads = 0u64;
        for &step in &query_steps {
            let compact = chain_query_position(proposal_position, step, self.single_position)
                .saturating_sub(LOCAL_WINDOW + 1)
                .min(target_kv_len);
            kv_reads = kv_reads
                .checked_add(target_kv_read_bytes(
                    target_kv_len - compact,
                    target_kv_len,
                )?)
                .ok_or_else(|| invalid("tree KV read ledger overflow"))?;
        }
        debug_assert_eq!(query_steps.len(), steps);
        let ledger = Gemma4Mtp12ChainLedger {
            draft_k: 7,
            command_buffers,
            command_buffer_waits: command_buffers,
            target_q6k_table_alias_bytes: table.wire.byte_len,
            target_kv_alias_bytes: sliding.key.byte_len
                + sliding.value.byte_len
                + full.key.byte_len
                + full.value.byte_len,
            assistant_matrix_read_bytes: self
                .dense_bf16
                .as_ref()
                .map_or(FULL_Q4_MATRIX_BYTES, |dense| {
                    self.layout.embedding.byte_len + dense.byte_len()
                })
                * steps as u64,
            target_kv_read_bytes: kv_reads,
            dynamic_attention_scratch_bytes: scores.length(),
            resident_chain_state_bytes: self.scratch.chain_initial_recurrent_hidden.length()
                + self.scratch.chain_recurrent_hidden.length()
                + self.scratch.output_token.length()
                + self.tree_state.as_ref().unwrap().byte_len(),
            initial_hidden_upload_bytes: (TARGET_HIDDEN * 4) as u64,
            readback_bytes: (steps * 16 + (steps + 1) * 4) as u64,
        };
        timing.wall_us = started.elapsed().as_micros();
        Ok(Gemma4Mtp12TreeProposal {
            tokens: finalized.tokens,
            parents: finalized.parents,
            depths: finalized.depths,
            primary_rows: finalized.primary_rows,
            branch_primary_step: finalized.branch_primary_step,
            fork_forwards: finalized.fork_forwards,
            primary_margins: margins,
            forward_margins: forward_top.iter().map(|pair| pair.margin).collect(),
            runner_up_ids: forward_top.iter().map(|pair| pair.runner_up_id).collect(),
            node_p: finalized.node_p,
            policy: policy.name(),
            shape: shape.name().to_string(),
            assistant_steps: steps,
            timing,
            ledger,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_max_margin_parse_is_explicit_and_defaults_to_two() {
        assert_eq!(parse_max_margin(None).unwrap().to_bits(), 2.0f32.to_bits());
        for (text, expected) in [
            ("0", 0.0f32),
            ("0.5", 0.5),
            ("2", 2.0),
            ("2.5", 2.5),
            ("3e0", 3.0),
        ] {
            assert_eq!(
                parse_max_margin(Some(text)).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        for text in [
            "", " ", " 2", "2 ", "two", "2,5", "-1", "-0.5", "-0", "-1e-100", "NaN", "inf", "+inf",
            "-inf", "1e100",
        ] {
            assert!(
                parse_max_margin(Some(text)).is_err(),
                "must reject {text:?}"
            );
        }
    }

    #[test]
    fn tree_max_margin_boundaries_preserve_first_eligible_and_linear_fallback() {
        let wide = TopTwo {
            values: [4.0, 0.0],
            ids: [8, 9],
        };
        let mut pairs = [wide; PRIMARY];
        pairs[0].values[0] = f32::from_bits(2.5f32.to_bits() + 1);
        pairs[1].values[0] = 2.5;
        assert_eq!(select_branch(&pairs, 2.5), Some(1));
        assert_eq!(select_branch(&pairs, 3.0), Some(0));
        assert_eq!(select_branch(&pairs, DEFAULT_MAX_MARGIN), None);
        assert_eq!(select_branch(&pairs, parse_max_margin(None).unwrap()), None);
        let (parents, depths, primary) = topology(select_branch(&pairs, DEFAULT_MAX_MARGIN));
        assert_eq!(parents, [-1, 0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(depths, (0..8).collect::<Vec<_>>());
        assert_eq!(primary, (0..8).collect::<Vec<_>>());
        pairs[0].values = [1.0, 1.0];
        assert_eq!(select_branch(&pairs, 0.0), Some(0));
        pairs[0].ids = [8, 8];
        assert_eq!(select_branch(&pairs, 0.0), None);
    }

    #[test]
    fn tree_policy_keeps_first_threshold_crossing_and_valid_topology() {
        let wide = TopTwo {
            values: [3.0, 0.0],
            ids: [8, 9],
        };
        let mut pairs = [wide; PRIMARY];
        assert_eq!(select_branch(&pairs, DEFAULT_MAX_MARGIN), None);
        pairs[2].values = [2.0, 0.0];
        assert_eq!(select_branch(&pairs, DEFAULT_MAX_MARGIN), Some(2));
        pairs[1].values = [1.0, 0.0];
        assert_eq!(select_branch(&pairs, DEFAULT_MAX_MARGIN), Some(1));
        pairs[0].values = [f32::INFINITY, f32::INFINITY];
        assert_eq!(select_branch(&pairs, DEFAULT_MAX_MARGIN), Some(1));
        for branch in [None, Some(0), Some(1), Some(2), Some(3)] {
            let (parents, depths, primary) = topology(branch);
            assert_eq!((parents[0], depths[0]), (-1, 0));
            for row in 1..8 {
                assert!((parents[row] as usize) < row);
                assert_eq!(depths[row], depths[parents[row] as usize] + 1);
            }
            assert_eq!(
                primary,
                (0..if branch.is_some() { 5 } else { 8 }).collect::<Vec<_>>()
            );
            if let Some(step) = branch {
                assert_eq!(parents[5], step as i32);
                assert_eq!(
                    &depths[5..],
                    &[step as u32 + 1, step as u32 + 2, step as u32 + 3]
                );
            }
        }
    }

    #[test]
    fn tree_menu_reproduces_the_legacy_row_order_and_fork_topology() {
        // The menu's 4+1+2 must be the SAME physical tree the qualified V3
        // proposal emits, node for node, or the two lanes are not comparable.
        let gated = menu::gate_topologies();
        for step in 0..PRIMARY {
            let (parents, depths, primary_rows) = topology(Some(step));
            if step < menu::ALT_STEPS {
                let plan = menu::layout(menu::Shape::P4A1C2, &[step]).unwrap();
                assert_eq!(plan.parents, parents, "fork {step} parents");
                assert_eq!(plan.depths, depths, "fork {step} depths");
                assert_eq!(plan.primary_rows, primary_rows, "fork {step} primary rows");
                assert_eq!(plan.fork_forwards, vec![step]);
                assert_eq!(
                    plan.runner_up_write,
                    Some((menu::runner_up_slot(step), step))
                );
            }
            // Legacy may still fork at the fourth primary, which the menu
            // itself never does; the gate covers that topology all the same.
            assert!(
                gated.iter().any(|(p, d)| *p == parents && *d == depths),
                "legacy fork {step} is emittable but not gated"
            );
        }
        let (parents, depths, primary_rows) = topology(None);
        let plan = menu::layout(menu::Shape::Lin7, &[]).unwrap();
        assert_eq!(plan.parents, parents);
        assert_eq!(plan.depths, depths);
        assert_eq!(plan.primary_rows, primary_rows);
        assert!(plan.runner_up_write.is_none());
        // The legacy linear fallback encodes exactly these three forwards.
        assert_eq!(plan.cb2.len(), 3);
        for (index, spec) in plan.cb2.iter().enumerate() {
            let forward = PRIMARY + index;
            assert_eq!(spec.history_step, forward);
            assert_eq!(spec.input_slot, forward);
            assert_eq!(spec.query_step, forward);
            assert_eq!(spec.recurrent_slot, forward - 1);
        }
    }

    #[test]
    fn tree_menu_forward_slots_fit_the_resident_chain_scratch() {
        for shape in menu::Shape::ALL {
            for steps in menu::alt_step_sets(shape.alts()) {
                let plan = menu::layout(shape, &steps).unwrap();
                let forwards = shape.forwards();
                // The top-2 merge writes output_token[history_step + 1] and
                // results[history_step]; the CPU owns slots 8 and up only.
                let gpu_slots: Vec<usize> = (0..forwards).map(|f| f + 1).collect();
                if let Some((slot, step)) = plan.runner_up_write {
                    assert_eq!(slot, menu::runner_up_slot(step));
                    assert!(!gpu_slots.contains(&slot), "{shape} clobbers slot {slot}");
                    assert!(slot < MTP12_CHAIN_TOKEN_SLOTS);
                }
                for spec in &plan.cb2 {
                    assert!(spec.history_step + 1 < MTP12_CHAIN_TOKEN_SLOTS);
                    assert!(spec.history_step < MTP12_CHAIN_MAX_DRAFTS);
                    assert!(
                        gpu_slots.contains(&spec.input_slot) || spec.input_slot >= 8,
                        "{shape} reads an unwritten slot {}",
                        spec.input_slot
                    );
                    // The parent's hidden must already exist when this runs.
                    assert!(spec.recurrent_slot < spec.history_step);
                    // Chain RoPE tables are written for seven query steps.
                    assert!(spec.query_step < 7);
                }
                let state_bytes = (forwards * 16) as u64;
                assert!(state_bytes <= (MTP12_CHAIN_MAX_DRAFTS * 16) as u64);
            }
        }
    }

    #[test]
    fn tree_top2_gpu_matches_stable_vocabulary_order() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let tree = TreeState::new(&device).unwrap();
        let queue = device.new_command_queue();
        for count in [1, 31, 1023, 1024, 1025, VOCAB] {
            for variant in 0..5 {
                let mut logits: Vec<f32> = (0..count)
                    .map(|i| ((i * 17 % 31) as f32 - 15.0) / 8.0)
                    .collect();
                if variant == 1 {
                    logits.fill(f32::NEG_INFINITY);
                }
                if variant == 2 {
                    logits.fill(f32::NAN);
                }
                if variant == 3 {
                    for i in (0..count).step_by(29) {
                        logits[i] = f32::NAN;
                    }
                    logits[count - 1] = f32::INFINITY;
                    if count > 1 {
                        logits[count - 2] = f32::INFINITY;
                    }
                }
                if variant == 4 {
                    logits.fill(f32::NEG_INFINITY);
                    logits[0] = f32::NAN;
                }
                let mut expected: Vec<usize> =
                    (0..count).filter(|i| !logits[*i].is_nan()).collect();
                expected
                    .sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap().then(a.cmp(b)));
                if expected
                    .first()
                    .is_none_or(|i| logits[*i] == f32::NEG_INFINITY)
                {
                    expected.retain(|i| *i != 0);
                    expected.insert(0, 0);
                }
                let wanted = [
                    expected.first().copied().unwrap_or(0) as u32,
                    expected
                        .get(1)
                        .copied()
                        .map(|i| i as u32)
                        .unwrap_or(u32::MAX),
                ];
                let input = f32_buffer(&device, &logits).unwrap();
                let output = shared_buffer(&device, 32);
                let command = queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                tree.encode(encoder, &input, &output, 0, count);
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let actual = unsafe { *tree.results.contents().cast::<TopTwo>() };
                assert_eq!(actual.ids, wanted, "count={count} variant={variant}");
                assert_eq!(
                    unsafe { *output.contents().cast::<u32>().add(1) },
                    wanted[0]
                );
                for (j, id) in expected.iter().take(2).enumerate() {
                    let value = if logits[*id].is_nan() {
                        f32::NEG_INFINITY
                    } else {
                        logits[*id]
                    };
                    assert_eq!(actual.values[j].to_bits(), value.to_bits());
                }
            }
        }
    }

    // ---- Ignored per-step benches for the W8 tree assistant -----------------
    //
    // Invoke (release, on the rig):
    //   cargo test --release --lib gemma4_mtp12::tree::tests::assistant_tree_step_bench \
    //     -- --ignored --nocapture
    // Optional environment:
    //   CAMELID_MTP12_BENCH_PREFIX=620,1500        prefix lengths (default)
    //   CAMELID_GEMMA4_MTP12_ASSISTANT=1|<path>    real weights (default synthetic)
    //   CAMELID_GEMMA4_MTP12_SHORTLIST=<sidecar>   real shortlist (real weights only)
    //   CAMELID_GEMMA4_MTP12_SHORTLIST_COMPACT=1   the production compact head
    // The fuse selectors are NOT read from the environment here: every
    // configuration is written straight onto the assistant's `fuse` field, so
    // one process measures them all.

    const BENCH_REPS: usize = 20;
    const BENCH_BUFFERS: usize = 5;

    /// Prefix lengths from `CAMELID_MTP12_BENCH_PREFIX` (comma separated).
    fn bench_prefixes() -> Vec<usize> {
        let raw =
            std::env::var("CAMELID_MTP12_BENCH_PREFIX").unwrap_or_else(|_| "620,1500".to_string());
        let prefixes: Vec<usize> = raw
            .split(',')
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(|text| {
                text.parse::<usize>().unwrap_or_else(|_| {
                    panic!("CAMELID_MTP12_BENCH_PREFIX entry {text:?} is not an integer")
                })
            })
            .collect();
        assert!(
            !prefixes.is_empty(),
            "CAMELID_MTP12_BENCH_PREFIX listed no prefixes"
        );
        assert!(prefixes.iter().all(|p| *p > 0), "prefixes must be positive");
        prefixes
    }

    /// Median over `BENCH_BUFFERS` command buffers of (hardware GPU time of one
    /// buffer holding `BENCH_REPS` encodes) / `BENCH_REPS`, after one discarded
    /// warm-up buffer.
    fn bench_median_us(
        queue: &metal::CommandQueue,
        encode: &dyn Fn(&metal::ComputeCommandEncoderRef),
    ) -> f64 {
        let mut samples = Vec::with_capacity(BENCH_BUFFERS);
        for pass in 0..=BENCH_BUFFERS {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            for _ in 0..BENCH_REPS {
                encode(encoder);
            }
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            if pass > 0 {
                let (us, _) = crate::metal::command_buffer_gpu_times_us(&cb.to_owned());
                samples.push(us as f64 / BENCH_REPS as f64);
            }
        }
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    /// The four-layer body of [`Gemma4Mtp12AssistantMetal::encode_tree_step`]
    /// at `query_step` 0, mirrored here so the bench can time it on its own.
    /// Any change to the step's layer loop must be mirrored back.
    fn bench_encode_layers(
        assistant: &Gemma4Mtp12AssistantMetal,
        encoder: &metal::ComputeCommandEncoderRef,
        sliding: Gemma4Mtp12DeviceKv<'_>,
        full: Gemma4Mtp12DeviceKv<'_>,
        prefix: usize,
        position: usize,
        scores: &Buffer,
    ) {
        for layer in 0..N_LAYERS {
            let local = layer < 3;
            let (kv, heads, dim, cos, sin) = if local {
                (
                    sliding,
                    LOCAL_KV_HEADS,
                    LOCAL_HEAD_DIM,
                    &assistant.scratch.local_cos,
                    &assistant.scratch.local_sin,
                )
            } else {
                (
                    full,
                    FULL_KV_HEADS,
                    FULL_HEAD_DIM,
                    &assistant.scratch.full_cos,
                    &assistant.scratch.full_sin,
                )
            };
            let compact = if local {
                chain_query_position(position, 0, assistant.single_position)
                    .saturating_sub(LOCAL_WINDOW + 1)
                    .min(prefix)
            } else {
                0
            };
            let (next_norm, next_normed) = if layer + 1 < N_LAYERS {
                (
                    &assistant.layers[layer + 1].input_norm,
                    &assistant.scratch.normed,
                )
            } else {
                (&assistant.final_norm, &assistant.scratch.final_normalized)
            };
            assistant.encode_layer_k1_fused(
                encoder,
                layer,
                kv.key.buffer,
                kv.value.buffer,
                kv.key.byte_offset,
                kv.value.byte_offset,
                kv.max_positions,
                heads,
                dim,
                prefix,
                compact,
                prefix - compact,
                cos,
                sin,
                0,
                scores,
                next_norm,
                next_normed,
            );
        }
    }

    /// One production assistant tree step (gather, pre GEMV, four layers, final
    /// norm, post GEMV, draft head, top-2) at every prefix in
    /// `CAMELID_MTP12_BENCH_PREFIX`, for the all-off baseline, each fusion
    /// selector on its own, the leveled staged-GEMV variants, two hand-picked
    /// combinations and the per-selector best combination; plus a single-stage
    /// breakdown of the all-off step so the round cost can be attributed.
    #[test]
    #[ignore]
    fn assistant_tree_step_bench() {
        let Some(default_device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping assistant tree-step bench");
            return;
        };
        let requested = std::env::var("CAMELID_GEMMA4_MTP12_ASSISTANT").ok();
        let mut assistant = match requested.as_deref() {
            None => {
                eprintln!(
                    "[mtp12-tree-bench] weights SYNTHETIC (sparse unit weights in a full-size \
                           Q4_0 pack: bandwidth-representative, numerically trivial). Set \
                           CAMELID_GEMMA4_MTP12_ASSISTANT=1 for the staged official assistant."
                );
                super::super::tests::synthetic_assistant(&default_device)
            }
            Some("1") | Some("true") => {
                eprintln!("[mtp12-tree-bench] weights REAL (staged official assistant)");
                Gemma4Mtp12AssistantMetal::load_staged_official()
                    .expect("staged official assistant")
            }
            Some(path) => {
                eprintln!("[mtp12-tree-bench] weights REAL ({path})");
                Gemma4Mtp12AssistantMetal::load(std::path::Path::new(path)).expect("assistant")
            }
        };
        let device = assistant.packed_q4.device().to_owned();
        let queue = device.new_command_queue();

        if assistant.shortlist.is_none() {
            // Zero centroids score every cluster 0, so the stable top-T keeps
            // clusters 0..top-1; against a uniformly random three-cluster
            // assignment that retains 1 - (1 - 384/2048)^3 = 46.4% of the
            // vocabulary, the production shortlist's retained fraction.
            let (token_clusters, selected) =
                super::super::tests::synthetic_head_selection(&device, VOCAB, 384, 0xb0a7);
            assistant.shortlist = Some(Mtp12Shortlist {
                centroids: f32_buffer(
                    &device,
                    &vec![0.0f32; MTP12_SHORTLIST_CLUSTERS * ASSISTANT_HIDDEN],
                )
                .expect("bench centroids"),
                token_clusters,
                scores: shared_buffer(&device, MTP12_SHORTLIST_CLUSTERS * 4),
                selected,
                top: 384,
            });
            eprintln!(
                "[mtp12-tree-bench] shortlist SYNTHETIC top=384/2048 (~46.4% of vocab retained)"
            );
        } else {
            eprintln!("[mtp12-tree-bench] shortlist REAL (CAMELID_GEMMA4_MTP12_SHORTLIST)");
        }
        eprintln!(
            "[mtp12-tree-bench] compact head {} (CAMELID_GEMMA4_MTP12_SHORTLIST_COMPACT), attn_v2 {}, \
             {BENCH_REPS} encodes/buffer, median of {BENCH_BUFFERS} buffers",
            if mtp12_shortlist_compact_enabled() { "ON" } else { "OFF - masked full head" },
            if mtp12_attention_v2_enabled() { "ON" } else { "OFF" },
        );
        assistant.tree_state = Some(TreeState::new(&device).expect("tree state"));

        let prefixes = bench_prefixes();
        let capacity = prefixes.iter().copied().max().expect("prefixes") + 32;
        let table_offset = 32u64;
        let table_buffer = shared_buffer(
            &device,
            (table_offset + GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES + 32) as usize,
        );
        let q6_row = super::super::tests::synthetic_q6k_embedding_row();
        unsafe {
            std::ptr::copy_nonoverlapping(
                q6_row.as_ptr(),
                table_buffer
                    .contents()
                    .cast::<u8>()
                    .add(table_offset as usize),
                q6_row.len(),
            );
        }
        let table = Gemma4Mtp12Q6KEmbeddingTable {
            wire: Gemma4Mtp12MetalBufferView {
                buffer: &table_buffer,
                byte_offset: table_offset,
                byte_len: GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES,
            },
            hidden: TARGET_HIDDEN,
            vocab: VOCAB,
            target_model_sha256: GEMMA4_12B_QAT_Q4_0_TARGET_SHA256,
        };

        let (_, _, sliding_key, sliding_value) = super::super::tests::synthetic_kv(
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            capacity - 32,
            capacity,
            3,
        );
        let (_, _, full_key, full_value) = super::super::tests::synthetic_kv(
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            capacity - 32,
            capacity,
            19,
        );
        let kv_buffer = |values: &[f32]| f32_buffer(&device, values).expect("bench KV buffer");
        let sliding_key_buffer = kv_buffer(&sliding_key);
        let sliding_value_buffer = kv_buffer(&sliding_value);
        let full_key_buffer = kv_buffer(&full_key);
        let full_value_buffer = kv_buffer(&full_value);
        let sliding = Gemma4Mtp12DeviceKv {
            key: super::super::tests::f32_view(&sliding_key_buffer, 0, sliding_key.len()),
            value: super::super::tests::f32_view(&sliding_value_buffer, 0, sliding_value.len()),
            source_layer: GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            kv_heads: LOCAL_KV_HEADS,
            head_dim: LOCAL_HEAD_DIM,
            max_positions: capacity,
        };
        let full = Gemma4Mtp12DeviceKv {
            key: super::super::tests::f32_view(&full_key_buffer, 0, full_key.len()),
            value: super::super::tests::f32_view(&full_value_buffer, 0, full_value.len()),
            source_layer: GEMMA4_12B_MTP_FULL_HOST_LAYER,
            kv_heads: FULL_KV_HEADS,
            head_dim: FULL_HEAD_DIM,
            max_positions: capacity,
        };

        let staged = |level: u8, rows: u32| Mtp12FuseFlags {
            gemv_x4: level,
            gemv_staged_rows: rows,
            ..Mtp12FuseFlags::default()
        };
        let mut singles: Vec<(String, Mtp12FuseFlags)> = vec![(
            "GEMV_X4=1".to_string(),
            Mtp12FuseFlags {
                gemv_x4: 1,
                ..Default::default()
            },
        )];
        for level in [2u8, 3] {
            for rows in [2u32, 4, 8] {
                singles.push((
                    format!("GEMV_X4={level} STAGED_ROWS={rows}"),
                    staged(level, rows),
                ));
            }
        }
        // Levels 4/5 fold each row in a different order: they change the
        // drafts, so they are measured but must never be reported as a
        // shippable lane without an acceptance re-measurement.
        for level in [4u8, 5] {
            singles.push((
                format!(
                    "GEMV_X4={level} (INEXACT split x{})",
                    if level == 4 { 2 } else { 4 }
                ),
                Mtp12FuseFlags {
                    gemv_x4: level,
                    ..Default::default()
                },
            ));
        }
        singles.push((
            "GATEUP=1".to_string(),
            Mtp12FuseFlags {
                gate_up: true,
                ..Default::default()
            },
        ));
        singles.push((
            "NORM=1".to_string(),
            Mtp12FuseFlags {
                norm: true,
                ..Default::default()
            },
        ));
        singles.push((
            "QROPE=1".to_string(),
            Mtp12FuseFlags {
                qrope: true,
                ..Default::default()
            },
        ));
        for level in 1u8..=3 {
            singles.push((
                format!("SOFTMAX_CTX={level}"),
                Mtp12FuseFlags {
                    softmax_ctx: level,
                    ..Default::default()
                },
            ));
        }
        for level in 1u8..=2 {
            singles.push((
                format!("HEAD_PREFETCH={level}"),
                Mtp12FuseFlags {
                    head_prefetch: level,
                    ..Default::default()
                },
            ));
        }

        for &prefix in &prefixes {
            let position = prefix;
            write_chain_rope_tables(position, 7, assistant.single_position, &assistant.scratch)
                .expect("bench rope tables");
            let scores = shared_buffer(&device, N_HEADS * prefix * 4);
            let local_window = chain_query_position(position, 0, assistant.single_position)
                .saturating_sub(LOCAL_WINDOW + 1)
                .min(prefix);
            eprintln!(
                "\n[mtp12-tree-bench] ==== prefix {prefix} (query position {position}; sliding \
                 attends {} of {prefix}, full attends {prefix}) ====",
                prefix - local_window
            );

            let step = |assistant: &Gemma4Mtp12AssistantMetal,
                        encoder: &metal::ComputeCommandEncoderRef| {
                assistant.encode_tree_step(
                    encoder,
                    table,
                    sliding,
                    full,
                    prefix,
                    position,
                    0,
                    0,
                    1,
                    &assistant.scratch.chain_initial_recurrent_hidden,
                    0,
                    0,
                    &scores,
                    true,
                );
            };

            assistant.fuse = Mtp12FuseFlags::default();
            let baseline = bench_median_us(&queue, &|encoder| step(&assistant, encoder));
            eprintln!(
                "[mtp12-tree-bench] {:<34} {baseline:9.1} us/step",
                "all-off (baseline)"
            );

            let mut measured: Vec<(String, Mtp12FuseFlags, f64)> = Vec::new();
            for (label, flags) in &singles {
                assistant.fuse = *flags;
                let us = bench_median_us(&queue, &|encoder| step(&assistant, encoder));
                eprintln!(
                    "[mtp12-tree-bench] {label:<34} {us:9.1} us/step  {:+6.1}%",
                    (us - baseline) * 100.0 / baseline
                );
                measured.push((label.clone(), *flags, us));
            }

            // Per-selector best (never worse than off), then the combinations.
            let best =
                |pick: &dyn Fn(&Mtp12FuseFlags) -> bool| -> Option<(String, Mtp12FuseFlags, f64)> {
                    measured
                        .iter()
                        .filter(|(_, flags, us)| pick(flags) && *us < baseline)
                        .min_by(|a, b| a.2.total_cmp(&b.2))
                        .cloned()
                };
            // The all-on-best combination stays bit-identical: the split
            // levels are measured above but never folded into it.
            let best_gemv =
                best(&|f| f.gemv_x4 != 0 && f.gemv_x4 < MTP12_FUSE_GEMV_X4_FIRST_INEXACT);
            let best_softmax = best(&|f| f.softmax_ctx != 0);
            let best_head = best(&|f| f.head_prefetch != 0);
            let best_gateup = best(&|f| f.gate_up);
            let best_norm = best(&|f| f.norm);
            let best_qrope = best(&|f| f.qrope);
            let mut all_on = Mtp12FuseFlags::default();
            if let Some((_, f, _)) = &best_gemv {
                all_on.gemv_x4 = f.gemv_x4;
                all_on.gemv_staged_rows = f.gemv_staged_rows;
            }
            if let Some((_, f, _)) = &best_softmax {
                all_on.softmax_ctx = f.softmax_ctx;
            }
            if let Some((_, f, _)) = &best_head {
                all_on.head_prefetch = f.head_prefetch;
            }
            all_on.gate_up = best_gateup.is_some();
            all_on.norm = best_norm.is_some();
            all_on.qrope = best_qrope.is_some();

            let mut norm_qrope = Mtp12FuseFlags {
                norm: true,
                qrope: true,
                ..Default::default()
            };
            let mut combos = vec![("NORM=1 QROPE=1".to_string(), norm_qrope)];
            norm_qrope.head_prefetch = best_head.as_ref().map_or(2, |(_, f, _)| f.head_prefetch);
            combos.push((
                format!("NORM=1 QROPE=1 HEAD_PREFETCH={}", norm_qrope.head_prefetch),
                norm_qrope,
            ));
            combos.push((format!("all-on-best {all_on:?}"), all_on));
            for (label, flags) in combos {
                assistant.fuse = flags;
                let us = bench_median_us(&queue, &|encoder| step(&assistant, encoder));
                eprintln!(
                    "[mtp12-tree-bench] {label:<34} {us:9.1} us/step  {:+6.1}%",
                    (us - baseline) * 100.0 / baseline
                );
            }

            // Single-stage breakdown of the all-off step.  The stages are
            // encoded independently, so they do not overlap and their sum
            // exceeds the fused step by the per-dispatch launch cost.
            assistant.fuse = Mtp12FuseFlags::default();
            let gather_pre = bench_median_us(&queue, &|encoder| {
                encode_q6k_embedding_and_recurrent_gather(
                    encoder,
                    &assistant.pipelines.gather_q6k_embedding_and_recurrent,
                    &assistant.scratch.output_token,
                    0,
                    table,
                    &assistant.scratch.chain_initial_recurrent_hidden,
                    0,
                    &assistant.scratch.pre_input,
                );
                assistant.encode_dense_gemv(
                    encoder,
                    &assistant.scratch.pre_input,
                    &assistant.scratch.hidden,
                    assistant.layout.pre_projection,
                    true,
                );
            });
            let layers = bench_median_us(&queue, &|encoder| {
                bench_encode_layers(
                    &assistant, encoder, sliding, full, prefix, position, &scores,
                );
            });
            let post_final = bench_median_us(&queue, &|encoder| {
                encode_rms_norm(
                    encoder,
                    &assistant.pipelines.rms_norm,
                    &assistant.scratch.hidden,
                    &assistant.final_norm,
                    &assistant.scratch.final_normalized,
                    ASSISTANT_HIDDEN,
                    1,
                );
                // The step's test-only draft-query dump, kept here so the
                // stages account for every dispatch the measured step issues.
                encode_copy_f32_to_offset(
                    encoder,
                    &assistant.pipelines.copy_f32,
                    &assistant.scratch.final_normalized,
                    &assistant.scratch.chain_final_normalized,
                    0,
                    ASSISTANT_HIDDEN,
                );
                assistant.encode_dense_gemv_at_offset(
                    encoder,
                    &assistant.scratch.final_normalized,
                    &assistant.scratch.chain_recurrent_hidden,
                    0,
                    assistant.layout.post_projection,
                    true,
                );
            });
            let head = bench_median_us(&queue, &|encoder| assistant.encode_draft_head(encoder));
            let top2 = bench_median_us(&queue, &|encoder| {
                assistant.tree_state.as_ref().unwrap().encode(
                    encoder,
                    &assistant.scratch.logits,
                    &assistant.scratch.output_token,
                    0,
                    VOCAB,
                );
            });
            let sum = gather_pre + layers + post_final + head + top2;
            for (label, us) in [
                ("gather + pre GEMV", gather_pre),
                ("4 layers", layers),
                ("final norm + post GEMV", post_final),
                ("compact draft head", head),
                ("tree top2", top2),
            ] {
                eprintln!(
                    "[mtp12-tree-bench]   stage {label:<26} {us:9.1} us  {:5.1}% of stage sum",
                    us * 100.0 / sum
                );
            }
            eprintln!(
                "[mtp12-tree-bench]   stage {:<26} {sum:9.1} us  (fused step {baseline:.1} us)",
                "sum"
            );
        }
    }
}
