//! Opt-in draft-only W8 tree. The ordinary resident-chain entry points do not
//! call this module, allocate its scratch, or change their command sequence.
use super::*;
#[cfg(test)]
#[path = "gemma4_mtp12_tree_oracle.rs"]
mod oracle;

const PRIMARY: usize = 4;
const DEFAULT_MAX_MARGIN: f32 = 2.0;
const MAX_MARGIN_ENV: &str = "CAMELID_GEMMA4_MTP12_TREE_MAX_MARGIN";

fn parse_max_margin(value: Option<&str>) -> Result<f32> {
    let Some(text) = value else { return Ok(DEFAULT_MAX_MARGIN); };
    let margin = text.parse::<f32>().map_err(|_| {
        invalid(format!("{MAX_MARGIN_ENV} must be a finite nonnegative number; got {text:?}"))
    })?;
    if !margin.is_finite() || margin.is_sign_negative() {
        return Err(invalid(format!("{MAX_MARGIN_ENV} must be a finite nonnegative number; got {text:?}")));
    }
    Ok(margin)
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
    /// Zero-based primary query where the rank-two fork was selected.
    pub branch_primary_step: Option<usize>,
    pub primary_margins: [f32; PRIMARY],
    /// Six forwards for a tree; seven for the no-eligible-branch linear fallback.
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
    // drafting never reads the experimental selector or allocates this state.
    max_margin: f32,
    partial: ComputePipelineState,
    merge: ComputePipelineState,
    partials: Buffer,
    results: Buffer,
}

impl TreeState {
    fn new(device: &Device) -> Result<Self> {
        let max_margin = max_margin_from_env()?;
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
            partial: pipeline("mtp12_tree_top2_partial")?,
            merge: pipeline("mtp12_tree_top2_merge")?,
            partials: shared_buffer(device, MTP12_ARGMAX_MAX_PARTIALS * 16),
            results: shared_buffer(device, PRIMARY * 16),
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
            self.encode_layer_k1_from_views(
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
            );
        }
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.hidden,
            &self.final_norm,
            &self.scratch.final_normalized,
            ASSISTANT_HIDDEN,
            1,
        );
        #[cfg(test)]
        encode_copy_f32_to_offset(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.final_normalized,
            &self.scratch.chain_final_normalized,
            (history_step * ASSISTANT_HIDDEN * 4) as u64,
            ASSISTANT_HIDDEN,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.final_normalized,
            &self.scratch.recurrent_hidden,
            self.layout.post_projection,
            true,
        );
        encode_copy_f32_to_offset(
            encoder,
            &self.pipelines.copy_f32,
            &self.scratch.recurrent_hidden,
            &self.scratch.chain_recurrent_hidden,
            (history_step * TARGET_HIDDEN * 4) as u64,
            TARGET_HIDDEN,
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
            let recurrent = if step == 0 {
                &self.scratch.chain_initial_recurrent_hidden
            } else {
                &self.scratch.recurrent_hidden
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
                0,
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
        let branch = select_branch(&top, self.tree_state.as_ref().unwrap().max_margin);
        let margins = std::array::from_fn(|i| top[i].values[0] - top[i].values[1]);
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
                let (recurrent, offset): (&BufferRef, u64) = if continuation == 0 {
                    (
                        &self.scratch.chain_recurrent_hidden,
                        (step * TARGET_HIDDEN * 4) as u64,
                    )
                } else {
                    (&self.scratch.recurrent_hidden, 0)
                };
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
                    &self.scratch.recurrent_hidden,
                    0,
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
        Ok(Gemma4Mtp12TreeProposal {
            tokens,
            parents,
            depths,
            primary_rows,
            branch_primary_step: branch,
            primary_margins: margins,
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
        for (text, expected) in [("0", 0.0f32), ("0.5", 0.5), ("2", 2.0), ("2.5", 2.5), ("3e0", 3.0)] {
            assert_eq!(parse_max_margin(Some(text)).unwrap().to_bits(), expected.to_bits());
        }
        for text in ["", " ", " 2", "2 ", "two", "2,5", "-1", "-0.5", "-0", "-1e-100", "NaN", "inf", "+inf", "-inf", "1e100"] {
            assert!(parse_max_margin(Some(text)).is_err(), "must reject {text:?}");
        }
    }

    #[test]
    fn tree_max_margin_boundaries_preserve_first_eligible_and_linear_fallback() {
        let wide = TopTwo { values: [4.0, 0.0], ids: [8, 9] };
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
                if expected.first().is_none_or(|i| logits[*i] == f32::NEG_INFINITY) {
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
                    let value = if logits[*id].is_nan() { f32::NEG_INFINITY } else { logits[*id] };
                    assert_eq!(actual.values[j].to_bits(), value.to_bits());
                }
            }
        }
    }
}
