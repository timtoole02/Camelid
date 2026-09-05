//! Bounded W8 target trees. Logical ancestor order is separate from physical KV
//! storage. This module and its shader library are only used by explicit tree calls.
use super::*;

const ROWS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Gemma4DenseTreePlan {
    parents: [i32; ROWS],
    depths: [u32; ROWS],
    ancestors: [[u32; ROWS]; ROWS],
}

impl Gemma4DenseTreePlan {
    pub(crate) fn new(parents: &[i32], depths: &[u32]) -> Option<Self> {
        let parents: [i32; ROWS] = parents.try_into().ok()?;
        let depths: [u32; ROWS] = depths.try_into().ok()?;
        if parents[0] != -1 || depths[0] != 0 {
            return None;
        }
        let mut ancestors = [[0u32; ROWS]; ROWS];
        for row in 1..ROWS {
            let parent = usize::try_from(parents[row]).ok()?;
            if parent >= row
                || depths[row] != depths[parent].checked_add(1)?
                || depths[row] >= ROWS as u32
            {
                return None;
            }
            ancestors[row] = ancestors[parent];
            ancestors[row][depths[row] as usize] = row as u32;
        }
        Some(Self {
            parents,
            depths,
            ancestors,
        })
    }

    pub(crate) fn parents(&self) -> &[i32] {
        &self.parents
    }
    pub(crate) fn depths(&self) -> &[u32] {
        &self.depths
    }

    pub(crate) fn path_valid(&self, path: &[usize]) -> bool {
        !path.is_empty()
            && path.len() <= ROWS
            && path[0] == 0
            && path.iter().all(|&row| row < ROWS)
            && path
                .windows(2)
                .all(|pair| self.parents[pair[1]] == pair[0] as i32)
    }

    fn row_plan(
        &self,
        base: usize,
        window: Option<usize>,
        capacity: usize,
        heads: usize,
    ) -> Option<(Vec<Gemma4DenseAttentionRowMeta>, usize)> {
        if base.checked_add(ROWS)? > capacity || heads == 0 || window == Some(0) {
            return None;
        }
        let mut rows = Vec::with_capacity(ROWS);
        let mut score_elements = 0usize;
        for &depth in &self.depths {
            let end = base.checked_add(depth as usize)?.checked_add(1)?;
            let start = window.map_or(0, |w| end.saturating_sub(w));
            let count = end.checked_sub(start)?;
            rows.push(Gemma4DenseAttentionRowMeta {
                window_start: u32::try_from(start).ok()?,
                position_count: u32::try_from(count).ok()?,
                score_offset: u32::try_from(score_elements).ok()?,
                visible_end: u32::try_from(end).ok()?,
            });
            score_elements = score_elements.checked_add(heads.checked_mul(count)?)?;
        }
        u32::try_from(score_elements).ok()?;
        Some((rows, score_elements.checked_mul(4)?))
    }
}

struct TreePipelines {
    device_id: u64,
    suffix_scores: ComputePipelineState,
    context: ComputePipelineState,
    context_p2: ComputePipelineState,
    compact: ComputePipelineState,
}

fn pipelines(kernel: &MetalLinearKernel) -> Option<&'static TreePipelines> {
    static PIPELINES: OnceLock<Option<TreePipelines>> = OnceLock::new();
    let result = PIPELINES
        .get_or_init(|| {
            let options = CompileOptions::new();
            let library = kernel
                .device
                .new_library_with_source(include_str!("gemma4_tree.metal"), &options)
                .map_err(|e| eprintln!("[gemma4-tree] shader compile failed: {e}"))
                .ok()?;
            let make = |name| {
                let function = library.get_function(name, None).ok()?;
                kernel
                    .device
                    .new_compute_pipeline_state_with_function(&function)
                    .ok()
            };
            Some(TreePipelines {
                device_id: kernel.device.registry_id(),
                suffix_scores: make("gemma4_tree_scores_suffix")?,
                context: make("gemma4_tree_context_nest")?,
                context_p2: make("gemma4_tree_context_hd256_p2")?,
                compact: make("gemma4_tree_compact_kv")?,
            })
        })
        .as_ref()?;
    (result.device_id == kernel.device.registry_id()).then_some(result)
}

impl Gemma4ResidentModel {
    /// Physically writes eight node rows at base+i; the caller supplies each
    /// node's RoPE/window inputs at semantic position base+depth[i].
    pub(crate) fn verify_tree_hidden_ordered_q4(
        &self,
        h0_rows: &[f32],
        inputs_by_row: &[Vec<Gemma4TokenLayerInput>],
        base_position: usize,
        plan: &Gemma4DenseTreePlan,
    ) -> Option<Vec<f32>> {
        self.verify_hidden_ordered_q4_plan(h0_rows, inputs_by_row, base_position, Some(plan))
    }

    /// Finish before advancing the runtime ticket/cursor. Prefix <base is never
    /// touched, and each head stages the complete selected path before writes.
    pub(crate) fn compact_tree_kv_path(
        &self,
        base_position: usize,
        plan: &Gemma4DenseTreePlan,
        path: &[usize],
    ) -> Option<()> {
        if !plan.path_valid(path)
            || base_position.checked_add(ROWS)? > self.max_positions
            || self.layers.len() != 48
            || self.caches.len() != 48
            || self.owns_kv.iter().any(|&owns| !owns)
            || self
                .kv_source
                .iter()
                .enumerate()
                .any(|(layer, &source)| layer != source)
        {
            return None;
        }
        if path
            .iter()
            .enumerate()
            .all(|(destination, &source)| destination == source)
        {
            return Some(());
        }
        u32::try_from(self.max_positions).ok()?;
        let kernel = metal_linear_kernel()?;
        let tree = pipelines(kernel)?;
        // Validate every layer before encoding any mutation.
        for (layer, cache) in self.layers.iter().zip(&self.caches) {
            let (keys, values) = cache.as_ref()?;
            if !matches!(layer.head_dim, 256 | 512) || layer.n_kv_heads == 0 {
                return None;
            }
            let bytes = layer
                .n_kv_heads
                .checked_mul(self.max_positions)?
                .checked_mul(layer.head_dim)?
                .checked_mul(4)? as u64;
            if keys.length() < bytes || values.length() < bytes {
                return None;
            }
        }
        let command = kernel.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let path_u32: Vec<u32> = path.iter().map(|&row| row as u32).collect();
        for (layer, cache) in self.layers.iter().zip(&self.caches) {
            let (keys, values) = cache.as_ref()?;
            encode_compact(
                encoder,
                &tree.compact,
                keys,
                values,
                base_position,
                self.max_positions,
                layer.n_kv_heads,
                layer.head_dim,
                &path_u32,
            );
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        (command.status() == metal::MTLCommandBufferStatus::Completed).then_some(())
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_compact(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    keys: &Buffer,
    values: &Buffer,
    base: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    path: &[u32],
) {
    let args = [
        base as u32,
        capacity as u32,
        head_dim as u32,
        path.len() as u32,
    ];
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(keys), 0);
    encoder.set_buffer(1, Some(values), 0);
    encoder.set_bytes(2, std::mem::size_of_val(&args) as u64, args.as_ptr().cast());
    encoder.set_bytes(3, std::mem::size_of_val(path) as u64, path.as_ptr().cast());
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: kv_heads as u64,
            height: 2,
            depth: 1,
        },
        metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_tree_attention(
    encoder: &metal::ComputeCommandEncoderRef,
    kernel: &MetalLinearKernel,
    query: &Buffer,
    keys: &Buffer,
    values: &Buffer,
    scores: &Buffer,
    denom: &Buffer,
    output: &Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_positions: usize,
    base_position: usize,
    sliding_window: Option<usize>,
    scale: f32,
    plan: &Gemma4DenseTreePlan,
    variant: Gemma4DenseAttentionRowsV2Variant,
) -> bool {
    encode_tree_attention_inner(
        encoder,
        kernel,
        query,
        keys,
        values,
        scores,
        denom,
        output,
        n_heads,
        n_kv_heads,
        head_dim,
        max_positions,
        base_position,
        sliding_window,
        scale,
        plan,
        variant,
    )
    .is_some()
}

#[allow(clippy::too_many_arguments)]
fn encode_tree_attention_inner(
    encoder: &metal::ComputeCommandEncoderRef,
    kernel: &MetalLinearKernel,
    query: &Buffer,
    keys: &Buffer,
    values: &Buffer,
    scores: &Buffer,
    denom: &Buffer,
    output: &Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_positions: usize,
    base_position: usize,
    sliding_window: Option<usize>,
    scale: f32,
    plan: &Gemma4DenseTreePlan,
    variant: Gemma4DenseAttentionRowsV2Variant,
) -> Option<()> {
    let tree = pipelines(kernel)?;
    let (score_name, score_pairs) = variant.scores_kernel(head_dim)?;
    let prefix_pipeline = kernel.gemma4_attn_rows_v2_pipelines.get(score_name)?;
    let softmax_pipeline = kernel
        .gemma4_attn_rows_v2_pipelines
        .get(GEMMA4_ATTN_V2_SOFTMAX)?;
    let group = n_heads.checked_div(n_kv_heads)?;
    let (row_plan, score_bytes) =
        plan.row_plan(base_position, sliding_window, max_positions, n_heads)?;
    let q_elements = ROWS.checked_mul(n_heads)?.checked_mul(head_dim)?;
    let kv_stride = max_positions.checked_mul(head_dim)?;
    let kv_elements = n_kv_heads.checked_mul(kv_stride)?;
    let denom_elements = ROWS.checked_mul(n_heads)?;
    u32::try_from(kv_elements).ok()?;
    u32::try_from(q_elements).ok()?;
    if group == 0
        || n_heads % n_kv_heads != 0
        || !matches!(head_dim, 256 | 512)
        || !scale.is_finite()
        || query.length() < q_elements.checked_mul(4)? as u64
        || output.length() < q_elements.checked_mul(4)? as u64
        || keys.length() < kv_elements.checked_mul(4)? as u64
        || values.length() < kv_elements.checked_mul(4)? as u64
        || scores.length() < score_bytes as u64
        || denom.length() < denom_elements.checked_mul(4)? as u64
    {
        return None;
    }
    let union_start = row_plan.iter().map(|r| r.window_start).min()?;
    let union_end = row_plan.iter().map(|r| r.visible_end).max()?;
    let max_count = row_plan.iter().map(|r| r.position_count).max()? as usize;
    let score_blocks = max_count.div_ceil(32).max(1);
    let dim_blocks = head_dim.div_ceil(32).max(1);
    let args = Gemma4AttnV2Args {
        n_heads: u32::try_from(n_heads).ok()?,
        head_dim: head_dim as u32,
        rows: ROWS as u32,
        group: u32::try_from(group).ok()?,
        scale,
        position_stride: head_dim as u32,
        kv_head_stride: u32::try_from(kv_stride).ok()?,
        kv_base_offset: 0,
        union_start,
        union_end,
        score_blocks: u32::try_from(score_blocks).ok()?,
        dim_blocks: dim_blocks as u32,
    };
    let base_u32 = u32::try_from(base_position).ok()?;
    let bind = |e: &metal::ComputeCommandEncoderRef, args: &Gemma4AttnV2Args| {
        e.set_bytes(
            5,
            std::mem::size_of_val(args) as u64,
            (args as *const Gemma4AttnV2Args).cast(),
        );
        e.set_bytes(
            6,
            std::mem::size_of_val(row_plan.as_slice()) as u64,
            row_plan.as_ptr().cast(),
        );
        e.set_bytes(
            8,
            std::mem::size_of_val(&plan.ancestors) as u64,
            plan.ancestors.as_ptr().cast(),
        );
        e.set_bytes(9, 4, (&base_u32 as *const u32).cast());
    };
    let tg32 = metal::MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    // Unchanged fused arithmetic over already committed keys. The nest control
    // may also fill the suffix; mapped suffix scores overwrite it before softmax.
    if union_start < base_u32 {
        let mut prefix_args = args;
        prefix_args.union_end = base_u32;
        encoder.set_compute_pipeline_state(prefix_pipeline);
        encoder.set_buffer(0, Some(query), 0);
        encoder.set_buffer(1, Some(keys), 0);
        encoder.set_buffer(3, Some(scores), 0);
        bind(encoder, &prefix_args);
        let threads = 32 * variant.scores_simdgroups();
        let grid = if score_pairs == 0 {
            metal::MTLSize {
                width: n_heads as u64,
                height: ROWS as u64,
                depth: score_blocks as u64,
            }
        } else {
            metal::MTLSize {
                width: n_kv_heads as u64,
                height: (base_position - union_start as usize).div_ceil(threads) as u64,
                depth: (group * ROWS).div_ceil(score_pairs) as u64,
            }
        };
        encoder.dispatch_thread_groups(
            grid,
            metal::MTLSize {
                width: threads as u64,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.set_compute_pipeline_state(&tree.suffix_scores);
    encoder.set_buffer(0, Some(query), 0);
    encoder.set_buffer(1, Some(keys), 0);
    encoder.set_buffer(3, Some(scores), 0);
    bind(encoder, &args);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: n_heads as u64,
            height: ROWS as u64,
            depth: 1,
        },
        tg32,
    );

    encoder.set_compute_pipeline_state(softmax_pipeline);
    encoder.set_buffer(3, Some(scores), 0);
    encoder.set_buffer(7, Some(denom), 0);
    bind(encoder, &args);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: n_heads as u64,
            height: ROWS as u64,
            depth: 1,
        },
        tg32,
    );

    let paired_context = head_dim == 256 && variant.context == 3;
    encoder.set_compute_pipeline_state(if paired_context {
        &tree.context_p2
    } else {
        &tree.context
    });
    encoder.set_buffer(2, Some(values), 0);
    encoder.set_buffer(3, Some(scores), 0);
    encoder.set_buffer(4, Some(output), 0);
    encoder.set_buffer(7, Some(denom), 0);
    bind(encoder, &args);
    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: if paired_context { n_kv_heads } else { n_heads } as u64,
            height: if paired_context {
                (group * ROWS).div_ceil(2)
            } else {
                ROWS
            } as u64,
            depth: dim_blocks as u64,
        },
        tg32,
    );
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fork(step: usize) -> Gemma4DenseTreePlan {
        Gemma4DenseTreePlan::new(
            &[-1, 0, 1, 2, 3, step as i32, 5, 6],
            &[
                0,
                1,
                2,
                3,
                4,
                step as u32 + 1,
                step as u32 + 2,
                step as u32 + 3,
            ],
        )
        .unwrap()
    }

    #[test]
    fn gemma4_tree_plan_rejects_bad_topology_and_preserves_logical_slots() {
        for step in 0..4 {
            let plan = fork(step);
            for row in 0..ROWS {
                let path: Vec<usize> = plan.ancestors[row][..=plan.depths[row] as usize]
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                assert!(plan.path_valid(&path));
                assert_eq!(*path.last().unwrap(), row);
            }
            assert!(!plan.path_valid(&[]));
            assert!(!plan.path_valid(&[1]));
            assert!(!plan.path_valid(&[0, 8]));
            assert!(!plan.path_valid(&[0, 0]));
            for base in [0, 529, 1023, 1024, 1025] {
                let (rows, bytes) = plan.row_plan(base, Some(1024), 2048, 16).unwrap();
                let mut offset = 0;
                for (row, meta) in rows.iter().enumerate() {
                    let end = base + plan.depths[row] as usize + 1;
                    assert_eq!(meta.visible_end as usize, end);
                    assert_eq!(meta.window_start as usize, end.saturating_sub(1024));
                    assert_eq!(meta.position_count as usize, end.min(1024));
                    assert_eq!(meta.score_offset as usize, offset);
                    offset += 16 * end.min(1024);
                }
                assert_eq!(bytes, offset * 4);
            }
            assert!(plan.row_plan(2041, None, 2048, 16).is_none());
            let mut parents = plan.parents;
            parents[5] = 5;
            assert!(Gemma4DenseTreePlan::new(&parents, &plan.depths).is_none());
            let mut depths = plan.depths;
            depths[5] += 1;
            assert!(Gemma4DenseTreePlan::new(&plan.parents, &depths).is_none());
        }
        assert!(Gemma4DenseTreePlan::new(&[-1], &[0]).is_none());
    }

    fn bits_equal(label: &str, expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len());
        if let Some((index, (a, b))) = expected
            .iter()
            .zip(actual)
            .enumerate()
            .find(|(_, (a, b))| a.to_bits() != b.to_bits())
        {
            panic!(
                "{label}: index={index} expected={a:?}/{:08x} actual={b:?}/{:08x}",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    #[test]
    fn metal_gemma4_tree_attention_matches_independent_linear_paths() {
        let kernel = metal_linear_kernel().expect("Metal device required for tree gate");
        let buffer = |elements: usize| {
            kernel.device.new_buffer(
                (elements.max(1) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let read = |buffer: &Buffer, elements: usize| {
            let mut result = vec![0.0; elements];
            read_buffer_f32(buffer, &mut result);
            result
        };
        let mut cases = 0;
        for (heads, kv_heads, hd, window) in [(16, 8, 256, Some(1024)), (16, 1, 512, None)] {
            for base in [0, 529, 1023, 1024, 1025] {
                let capacity = base + ROWS + 3;
                let (query, mut keys, mut values) =
                    gemma4_dense_attention_v2_fixture(heads, kv_heads, hd, capacity, base, ROWS);
                for head in 0..kv_heads {
                    let end = (head + 1) * capacity * hd;
                    keys[(head * capacity + base + ROWS) * hd..end].fill(f32::NAN);
                    values[(head * capacity + base + ROWS) * hd..end].fill(f32::NAN);
                }
                let qdim = heads * hd;
                let tq = buffer(query.len());
                let tk = buffer(keys.len());
                let tv = buffer(values.len());
                let rq = buffer(query.len());
                let rk = buffer(keys.len());
                let rv = buffer(values.len());
                write_buffer_f32(&tq, &query);
                write_buffer_f32(&tk, &keys);
                write_buffer_f32(&tv, &values);
                write_buffer_f32(&rq, &query);
                write_buffer_f32(&rk, &keys);
                write_buffer_f32(&rv, &values);
                let (reference_meta, reference_bytes) =
                    gemma4_dense_attention_row_plan(base, ROWS, window, capacity, heads).unwrap();
                let rs = buffer(reference_bytes / 4);
                let rd = buffer(ROWS * heads);
                let ro = buffer(query.len());
                let mut plans: Vec<_> = (0..4).map(fork).collect();
                // Topological physical order interleaves both branches.
                plans.push(
                    Gemma4DenseTreePlan::new(&[-1, 0, 0, 1, 2, 3, 4, 5], &[0, 1, 1, 2, 2, 3, 3, 4])
                        .unwrap(),
                );
                for plan in plans {
                    let (tree_meta, tree_bytes) =
                        plan.row_plan(base, window, capacity, heads).unwrap();
                    let ts = buffer(tree_bytes / 4);
                    let td = buffer(ROWS * heads);
                    let to = buffer(query.len());
                    for variant in [
                        Gemma4DenseAttentionRowsV2Variant::DEFAULT,
                        Gemma4DenseAttentionRowsV2Variant::NEST,
                        Gemma4DenseAttentionRowsV2Variant {
                            scores: 2,
                            context: 3,
                        },
                    ] {
                        let command = kernel.queue.new_command_buffer();
                        let encoder = command.new_compute_command_encoder();
                        assert!(encode_tree_attention(
                            encoder, kernel, &tq, &tk, &tv, &ts, &td, &to, heads, kv_heads, hd,
                            capacity, base, window, 1.0, &plan, variant
                        ));
                        encoder.end_encoding();
                        command.commit();
                        command.wait_until_completed();
                        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                        let tree_scores = read(&ts, tree_bytes / 4);
                        let tree_denom = read(&td, ROWS * heads);
                        let tree_output = read(&to, query.len());
                        for node in 0..ROWS {
                            let depth = plan.depths[node] as usize;
                            // Each reference is a separate contiguous root-to-node sequence,
                            // padded to W8. All non-ancestors/future positions are NaN.
                            unsafe {
                                let rkeys = std::slice::from_raw_parts_mut(
                                    rk.contents().cast::<f32>(),
                                    keys.len(),
                                );
                                let rvalues = std::slice::from_raw_parts_mut(
                                    rv.contents().cast::<f32>(),
                                    values.len(),
                                );
                                let rquery = std::slice::from_raw_parts_mut(
                                    rq.contents().cast::<f32>(),
                                    query.len(),
                                );
                                rquery[depth * qdim..(depth + 1) * qdim]
                                    .copy_from_slice(&query[node * qdim..(node + 1) * qdim]);
                                for head in 0..kv_heads {
                                    rkeys
                                        [(head * capacity + base) * hd..(head + 1) * capacity * hd]
                                        .fill(f32::NAN);
                                    rvalues
                                        [(head * capacity + base) * hd..(head + 1) * capacity * hd]
                                        .fill(f32::NAN);
                                    for logical in 0..=depth {
                                        let physical = plan.ancestors[node][logical] as usize;
                                        let src = (head * capacity + base + physical) * hd;
                                        let dst = (head * capacity + base + logical) * hd;
                                        rkeys[dst..dst + hd].copy_from_slice(&keys[src..src + hd]);
                                        rvalues[dst..dst + hd]
                                            .copy_from_slice(&values[src..src + hd]);
                                    }
                                }
                            }
                            let command = kernel.queue.new_command_buffer();
                            let encoder = command.new_compute_command_encoder();
                            assert!(encode_gemma4_dense_attention_rows_v2_f32(
                                encoder, kernel, &rq, &rk, &rv, &rs, &rd, &ro, heads, kv_heads, hd,
                                capacity, base, ROWS, window, 1.0, variant
                            ));
                            encoder.end_encoding();
                            command.commit();
                            command.wait_until_completed();
                            assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                            let scores = read(&rs, reference_bytes / 4);
                            let denom = read(&rd, ROWS * heads);
                            let output = read(&ro, query.len());
                            let label = format!(
                                "hd={hd} base={base} parents={:?} node={node} variant={variant:?}",
                                plan.parents
                            );
                            let tm = tree_meta[node];
                            let rm = reference_meta[depth];
                            assert_eq!(tm.position_count, rm.position_count);
                            let count = tm.position_count as usize * heads;
                            bits_equal(
                                &format!("exp {label}"),
                                &scores[rm.score_offset as usize..rm.score_offset as usize + count],
                                &tree_scores
                                    [tm.score_offset as usize..tm.score_offset as usize + count],
                            );
                            bits_equal(
                                &format!("denom {label}"),
                                &denom[depth * heads..(depth + 1) * heads],
                                &tree_denom[node * heads..(node + 1) * heads],
                            );
                            bits_equal(
                                &format!("context {label}"),
                                &output[depth * qdim..(depth + 1) * qdim],
                                &tree_output[node * qdim..(node + 1) * qdim],
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
        eprintln!("[gemma4-tree] exact attention: {cases} tree configurations, {} independent W8-padded paths", cases * ROWS);
    }

    #[test]
    fn metal_gemma4_tree_compaction_stages_overlapping_sources_and_preserves_bits() {
        let kernel = metal_linear_kernel().expect("Metal device required for tree gate");
        let pipelines = pipelines(kernel).unwrap();
        let capacity = 19;
        let base = 7;
        for (heads, hd) in [(8, 256), (1, 512)] {
            let elements = heads * capacity * hd;
            // Arbitrary bits include NaNs; compaction is a byte operation.
            let source: Vec<u32> = (0..elements)
                .map(|i| (i as u32).wrapping_mul(0x9e3779b9))
                .collect();
            for path in [
                vec![0, 2, 3],
                vec![0, 1, 5, 6, 7],
                vec![0, 2, 4, 6],
                (0..8).collect(),
            ] {
                let keys = kernel.device.new_buffer_with_data(
                    source.as_ptr().cast(),
                    (elements * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let values = kernel.device.new_buffer_with_data(
                    source.as_ptr().cast(),
                    (elements * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let command = kernel.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                encode_compact(
                    encoder,
                    &pipelines.compact,
                    &keys,
                    &values,
                    base,
                    capacity,
                    heads,
                    hd,
                    &path,
                );
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                let mut expected = source.clone();
                for head in 0..heads {
                    for (destination, &physical) in path.iter().enumerate() {
                        let src = (head * capacity + base + physical as usize) * hd;
                        let dst = (head * capacity + base + destination) * hd;
                        expected[dst..dst + hd].copy_from_slice(&source[src..src + hd]);
                    }
                }
                for buffer in [&keys, &values] {
                    let actual = unsafe {
                        std::slice::from_raw_parts(buffer.contents().cast::<u32>(), elements)
                    };
                    assert_eq!(actual, expected, "heads={heads} hd={hd} path={path:?}");
                }
            }
        }
    }
}

// Append to src/metal/gemma4_tree.rs. Test-only readback of already completed
// SPEC50 output; no additional projection or production behavior.
#[cfg(test)]
impl Gemma4Q6KHead {
    pub(crate) fn tree_test_last_spec50_logits(&self, columns: usize) -> Option<Vec<f32>> {
        if !matches!(columns, 1 | 2 | 4 | 8) {
            return None;
        }
        let state = self.inner.lock().ok()?;
        if state.last_spec50_timing.columns != columns as u32 {
            return None;
        }
        let batch = state.batch.as_ref()?;
        let elements = columns.checked_mul(state.vocab)?;
        if batch.logits.length() < elements.checked_mul(4)? as u64 {
            return None;
        }
        let mut logits = vec![0.0; elements];
        read_buffer_f32(&batch.logits, &mut logits);
        Some(logits)
    }
}
