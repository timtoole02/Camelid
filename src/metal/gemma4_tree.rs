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

    /// The linear W8 chain: parents `[-1, 0, 1, .., 6]`, depths `0..=7`, so
    /// `ancestors[row][d] == d` (every mapped address equals its logical position)
    /// and [`Self::row_plan`] equals `gemma4_dense_attention_row_plan` for 8 rows.
    /// Used by the opt-in `CAMELID_GEMMA4_LINEAR_K8_VIA_TREE=1` route.
    pub(crate) fn chain() -> &'static Self {
        static CHAIN: OnceLock<Gemma4DenseTreePlan> = OnceLock::new();
        CHAIN.get_or_init(|| {
            Self::new(&[-1, 0, 1, 2, 3, 4, 5, 6], &[0, 1, 2, 3, 4, 5, 6, 7])
                .expect("the static W8 chain plan is a valid tree")
        })
    }

    /// True when every row's parent is the previous row at depth `row`: the whole
    /// union is then linear-addressed and the encoder needs no suffix dispatch.
    pub(crate) fn is_linear_chain(&self) -> bool {
        self.parents
            .iter()
            .enumerate()
            .all(|(row, &parent)| parent == row as i32 - 1)
            && self
                .depths
                .iter()
                .enumerate()
                .all(|(row, &depth)| depth == row as u32)
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
    context_hd256_p2x: ComputePipelineState,
    context_hd256_p4x: ComputePipelineState,
    context_hd256_p8x: ComputePipelineState,
    context_hd512_p2x: ComputePipelineState,
    context_hd512_p4x: ComputePipelineState,
    context_hd512_p8x: ComputePipelineState,
    context_hd512_p16x: ComputePipelineState,
    compact: ComputePipelineState,
}

const TREE_CONTEXT_NEST: &str = "gemma4_tree_context_nest";
const TREE_CONTEXT_HD256_P2: &str = "gemma4_tree_context_hd256_p2";
const TREE_CONTEXT_HD256_P2X: &str = "gemma4_tree_context_hd256_p2x";
const TREE_CONTEXT_HD256_P4X: &str = "gemma4_tree_context_hd256_p4x";
const TREE_CONTEXT_HD256_P8X: &str = "gemma4_tree_context_hd256_p8x";
const TREE_CONTEXT_HD512_P2X: &str = "gemma4_tree_context_hd512_p2x";
const TREE_CONTEXT_HD512_P4X: &str = "gemma4_tree_context_hd512_p4x";
const TREE_CONTEXT_HD512_P8X: &str = "gemma4_tree_context_hd512_p8x";
const TREE_CONTEXT_HD512_P16X: &str = "gemma4_tree_context_hd512_p16x";

impl TreePipelines {
    fn context_pipeline(&self, name: &str) -> Option<&ComputePipelineState> {
        Some(match name {
            TREE_CONTEXT_NEST => &self.context,
            TREE_CONTEXT_HD256_P2 => &self.context_p2,
            TREE_CONTEXT_HD256_P2X => &self.context_hd256_p2x,
            TREE_CONTEXT_HD256_P4X => &self.context_hd256_p4x,
            TREE_CONTEXT_HD256_P8X => &self.context_hd256_p8x,
            TREE_CONTEXT_HD512_P2X => &self.context_hd512_p2x,
            TREE_CONTEXT_HD512_P4X => &self.context_hd512_p4x,
            TREE_CONTEXT_HD512_P8X => &self.context_hd512_p8x,
            TREE_CONTEXT_HD512_P16X => &self.context_hd512_p16x,
            _ => return None,
        })
    }
}

/// One tree-library context kernel form. `P2` is today's dispatch (the qualified
/// HD256 PAIRS2 kernel under V2 context form 3, else the nest / HD512 split); the
/// others are the pipelined `gemma4_tree_context_pipelined<HD, PAIRS>` instantiations
/// (`P2X` = pipelined PAIRS 2; P2X/P4/P8 exist at both head dims; P16 at HD512 only,
/// measurement-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TreeContextForm {
    P2,
    P2X,
    P4,
    P8,
    P16,
}

impl TreeContextForm {
    fn parse(value: &str) -> Option<Self> {
        Some(match value.trim().to_ascii_lowercase().as_str() {
            "p2" => Self::P2,
            "p2x" => Self::P2X,
            "p4" | "p4x" => Self::P4,
            "p8" | "p8x" => Self::P8,
            "p16" | "p16x" => Self::P16,
            _ => return None,
        })
    }

    fn label(self) -> &'static str {
        match self {
            Self::P2 => "p2",
            Self::P2X => "p2x",
            Self::P4 => "p4",
            Self::P8 => "p8",
            Self::P16 => "p16",
        }
    }

    /// Which instantiations exist per head dimension.
    fn available(self, head_dim: usize) -> bool {
        matches!(
            (head_dim, self),
            (256, Self::P2 | Self::P2X | Self::P4 | Self::P8)
                | (512, Self::P2 | Self::P2X | Self::P4 | Self::P8 | Self::P16)
        )
    }
}

/// The tree-only context-kernel selection: one form for the sliding HD256 layers and
/// one for the global HD512 layers (`CAMELID_GEMMA4_TREE_CONTEXT_FORM`, see
/// [`context_selection`]). `TODAY` is byte-for-byte today's dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TreeContextSelection {
    pub(super) sliding: TreeContextForm,
    pub(super) global: TreeContextForm,
}

impl TreeContextSelection {
    pub(super) const TODAY: Self = Self {
        sliding: TreeContextForm::P2,
        global: TreeContextForm::P2,
    };

    /// `p2|p2x|p4|p8|p16` applies one form to both head dims where it is instantiated
    /// (P16 exists only at HD512; the other dim keeps `p2`);
    /// `<sliding>,<global>` selects each explicitly and refuses a missing instantiation.
    pub(super) fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        if let Some((sliding, global)) = value.split_once(',') {
            let sliding = TreeContextForm::parse(sliding)?;
            let global = TreeContextForm::parse(global)?;
            return (sliding.available(256) && global.available(512))
                .then_some(Self { sliding, global });
        }
        let form = TreeContextForm::parse(value)?;
        Some(Self {
            sliding: if form.available(256) { form } else { TreeContextForm::P2 },
            global: if form.available(512) { form } else { TreeContextForm::P2 },
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn label(self) -> String {
        format!("{},{}", self.sliding.label(), self.global.label())
    }

    /// The tree-library kernel this selection binds for `head_dim` under the V2
    /// `variant`: `(function name, pairs per threadgroup)`; pairs = 0 is the nest
    /// grid `(head, row, dim_block)`, otherwise `(kv_head, pair_chunk, dim_block)`.
    pub(super) fn kernel(
        self,
        head_dim: usize,
        variant: Gemma4DenseAttentionRowsV2Variant,
    ) -> Option<(&'static str, usize)> {
        let form = match head_dim {
            256 => self.sliding,
            512 => self.global,
            _ => return None,
        };
        Some(match (head_dim, form) {
            (256, TreeContextForm::P2) if variant.context == 3 => (TREE_CONTEXT_HD256_P2, 2),
            (_, TreeContextForm::P2) => (TREE_CONTEXT_NEST, 0),
            (256, TreeContextForm::P2X) => (TREE_CONTEXT_HD256_P2X, 2),
            (256, TreeContextForm::P4) => (TREE_CONTEXT_HD256_P4X, 4),
            (256, TreeContextForm::P8) => (TREE_CONTEXT_HD256_P8X, 8),
            (512, TreeContextForm::P2X) => (TREE_CONTEXT_HD512_P2X, 2),
            (512, TreeContextForm::P4) => (TREE_CONTEXT_HD512_P4X, 4),
            (512, TreeContextForm::P8) => (TREE_CONTEXT_HD512_P8X, 8),
            (512, TreeContextForm::P16) => (TREE_CONTEXT_HD512_P16X, 16),
            _ => return None,
        })
    }

    /// Human-readable RESOLVED kernel name for receipts (the HD512 nest entry point
    /// runs `gemma4_tree_context_hd512_split` internally).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn resolved_name(
        self,
        head_dim: usize,
        variant: Gemma4DenseAttentionRowsV2Variant,
    ) -> &'static str {
        match self.kernel(head_dim, variant) {
            Some((TREE_CONTEXT_NEST, _)) if head_dim == 512 => {
                "gemma4_tree_context_nest(hd512_split)"
            }
            Some((name, _)) => name,
            None => "<unavailable>",
        }
    }

    /// Every selection the raw gate proves for one head dimension (the other dim's
    /// form is irrelevant to that geometry and stays `p2`).
    #[cfg(test)]
    pub(super) fn gate_forms(head_dim: usize) -> Vec<Self> {
        let forms = [
            TreeContextForm::P2,
            TreeContextForm::P2X,
            TreeContextForm::P4,
            TreeContextForm::P8,
            TreeContextForm::P16,
        ];
        forms
            .into_iter()
            .filter(|form| form.available(head_dim))
            .map(|form| match head_dim {
                256 => Self {
                    sliding: form,
                    global: TreeContextForm::P2,
                },
                _ => Self {
                    sliding: TreeContextForm::P2,
                    global: form,
                },
            })
            .collect()
    }

    /// Default measurement list for the tree receipt: today's forms, each new form in
    /// isolation per head dimension, then the candidate combinations.
    #[cfg(test)]
    pub(super) fn receipt_forms() -> Vec<Self> {
        [
            "p2", "p2x,p2", "p4,p2", "p8,p2", "p2,p2x", "p2,p4", "p2,p8", "p2,p16", "p2x,p2x",
            "p2x,p4", "p4,p4", "p4,p8", "p8,p8",
        ]
            .into_iter()
            .map(|spec| Self::parse(spec).expect("static receipt form"))
            .collect()
    }
}

/// `CAMELID_GEMMA4_TREE_CONTEXT_FORM`, read once per process: tree-only context
/// kernel selector (`p2` = today's kernels, `p2x`, `p4`, `p8`, `p16`, or
/// `<sliding>,<global>` such as `p2x,p4`). Unset = today's dispatch byte-for-byte; an unparsable value
/// keeps today's dispatch and says so once. The V2 variant matrix is not widened.
pub(super) fn context_selection() -> TreeContextSelection {
    static SELECTION: OnceLock<TreeContextSelection> = OnceLock::new();
    *SELECTION.get_or_init(|| {
        let Ok(value) = std::env::var("CAMELID_GEMMA4_TREE_CONTEXT_FORM") else {
            return TreeContextSelection::TODAY;
        };
        match TreeContextSelection::parse(&value) {
            Some(selection) => {
                eprintln!(
                    "[gemma4-tree] CAMELID_GEMMA4_TREE_CONTEXT_FORM={value:?} -> sliding HD256 {} / \
                     global HD512 {}",
                    selection.sliding.label(),
                    selection.global.label()
                );
                selection
            }
            None => {
                eprintln!(
                    "[gemma4-tree] CAMELID_GEMMA4_TREE_CONTEXT_FORM={value:?} is not p2|p2x|p4|p8|p16 \
                     or <sliding p2|p2x|p4|p8>,<global p2|p2x|p4|p8|p16>; keeping p2 (today's kernels)"
                );
                TreeContextSelection::TODAY
            }
        }
    })
}

/// `CAMELID_GEMMA4_LINEAR_K8_VIA_TREE=1`, read once per process: route linear
/// (non-branched) K=8 V2 verifies through [`encode_tree_attention`] with the static
/// chain plan so they bind the tree-library context kernels. Unset = today's linear
/// attention. Leave it UNSET for the full-model tree gate so its linear reference arm
/// stays independent of the tree library.
pub(super) fn linear_k8_via_tree_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_GEMMA4_LINEAR_K8_VIA_TREE")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
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
                context: make(TREE_CONTEXT_NEST)?,
                context_p2: make(TREE_CONTEXT_HD256_P2)?,
                context_hd256_p2x: make(TREE_CONTEXT_HD256_P2X)?,
                context_hd256_p4x: make(TREE_CONTEXT_HD256_P4X)?,
                context_hd256_p8x: make(TREE_CONTEXT_HD256_P8X)?,
                context_hd512_p2x: make(TREE_CONTEXT_HD512_P2X)?,
                context_hd512_p4x: make(TREE_CONTEXT_HD512_P4X)?,
                context_hd512_p8x: make(TREE_CONTEXT_HD512_P8X)?,
                context_hd512_p16x: make(TREE_CONTEXT_HD512_P16X)?,
                compact: make("gemma4_tree_compact_kv")?,
            })
        })
        .as_ref()?;
    (result.device_id == kernel.device.registry_id()).then_some(result)
}

impl Gemma4ResidentModel {
    /// Physically writes eight node rows at base+i; the caller supplies each
    /// node's RoPE/window inputs at semantic position base+depth[i].
    #[allow(dead_code)] // the runtime threads its fused-glue mask through `_with_glue`
    pub(crate) fn verify_tree_hidden_ordered_q4(
        &self,
        h0_rows: &[f32],
        inputs_by_row: &[Vec<Gemma4TokenLayerInput>],
        base_position: usize,
        plan: &Gemma4DenseTreePlan,
    ) -> Option<Vec<f32>> {
        self.verify_tree_hidden_ordered_q4_with_glue(h0_rows, inputs_by_row, base_position, plan, None)
    }

    /// [`Self::verify_tree_hidden_ordered_q4`] with an explicit fused-glue
    /// mask (`None` = `CAMELID_GEMMA4_VERIFY_FUSED_GLUE`, read once and
    /// refusing on garbage; `Some(mask)` pins this call for in-process A/B).
    pub(crate) fn verify_tree_hidden_ordered_q4_with_glue(
        &self,
        h0_rows: &[f32],
        inputs_by_row: &[Vec<Gemma4TokenLayerInput>],
        base_position: usize,
        plan: &Gemma4DenseTreePlan,
        fused_glue: Option<u32>,
    ) -> Option<Vec<f32>> {
        let fused_glue_mask = match fused_glue {
            Some(mask) => mask,
            None => super::gemma4_verify_fused_glue_mask()?,
        };
        self.verify_hidden_ordered_q4_plan(
            h0_rows,
            inputs_by_row,
            base_position,
            Some(plan),
            fused_glue_mask,
        )
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
    encode_tree_attention_with_form(
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
        context_selection(),
    )
}

/// [`encode_tree_attention`] with an explicit context-kernel selection (tests and
/// receipts); production reads the selection once from the environment.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_tree_attention_with_form(
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
    form: TreeContextSelection,
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
        form,
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
    form: TreeContextSelection,
) -> Option<()> {
    let tree = pipelines(kernel)?;
    let (score_name, score_pairs) = variant.scores_kernel(head_dim)?;
    let prefix_pipeline = kernel.gemma4_attn_rows_v2_pipelines.get(score_name)?;
    let (context_name, context_pairs) = form.kernel(head_dim, variant)?;
    let context_pipeline = tree.context_pipeline(context_name)?;
    // A linear chain (ancestors[row][d] == d) is entirely linear-addressed: the
    // prefix scores dispatch then covers the WHOLE union with the linear encoder's
    // grid (height from union_end - union_start, dispatched even when union_start ==
    // base) and the mapped suffix dispatch is skipped: 3 dispatches per layer.
    let chain = plan.is_linear_chain();
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
    let prefix_end = if chain { union_end } else { base_u32 };
    if union_start < prefix_end {
        let mut prefix_args = args;
        prefix_args.union_end = prefix_end;
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
                height: ((prefix_end - union_start) as usize).div_ceil(threads) as u64,
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
    if !chain {
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
    }

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

    encoder.set_compute_pipeline_state(context_pipeline);
    encoder.set_buffer(2, Some(values), 0);
    encoder.set_buffer(3, Some(scores), 0);
    encoder.set_buffer(4, Some(output), 0);
    encoder.set_buffer(7, Some(denom), 0);
    bind(encoder, &args);
    let context_grid = if context_pairs == 0 {
        metal::MTLSize {
            width: n_heads as u64,
            height: ROWS as u64,
            depth: dim_blocks as u64,
        }
    } else {
        metal::MTLSize {
            width: n_kv_heads as u64,
            height: (group * ROWS).div_ceil(context_pairs) as u64,
            depth: dim_blocks as u64,
        }
    };
    encoder.dispatch_thread_groups(context_grid, tg32);
    Some(())
}

/// The 4+1+2 production topology forked at primary `step` (0..4): rows 0..4 are the
/// primary chain, rows 5..7 continue from row `step`. Test/receipt shape only.
#[cfg(test)]
pub(super) fn fork_plan(step: usize) -> Gemma4DenseTreePlan {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fork(step: usize) -> Gemma4DenseTreePlan {
        fork_plan(step)
    }

    #[test]
    fn gemma4_tree_chain_plan_matches_linear_row_plan_and_identity_ancestors() {
        let chain = Gemma4DenseTreePlan::chain();
        assert!(chain.is_linear_chain());
        assert_eq!(chain.parents(), &[-1, 0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(chain.depths(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        for row in 0..ROWS {
            for d in 0..=row {
                assert_eq!(chain.ancestors[row][d] as usize, d, "row={row} d={d}");
            }
            let path: Vec<usize> = (0..=row).collect();
            assert!(chain.path_valid(&path));
        }
        for step in 0..4 {
            assert!(!fork(step).is_linear_chain());
        }
        assert!(
            !Gemma4DenseTreePlan::new(&[-1, 0, 0, 1, 2, 3, 4, 5], &[0, 1, 1, 2, 2, 3, 3, 4])
                .unwrap()
                .is_linear_chain()
        );
        // The chain's row plan is the linear verifier's row plan for 8 rows: same
        // visible_end / window_start / position_count / score_offset and scratch bytes,
        // across the sliding-window edges and the capacity edge.
        for window in [Some(1024), None] {
            for base in [0, 1, 529, 640, 647, 1016, 1017, 1023, 1024, 1025, 1500, 2040] {
                let (tree_rows, tree_bytes) = chain.row_plan(base, window, 2048, 16).unwrap();
                let (linear_rows, linear_bytes) =
                    gemma4_dense_attention_row_plan(base, ROWS, window, 2048, 16).unwrap();
                assert_eq!(tree_rows, linear_rows, "window={window:?} base={base}");
                assert_eq!(tree_bytes, linear_bytes, "window={window:?} base={base}");
                for (row, meta) in tree_rows.iter().enumerate() {
                    assert_eq!(meta.visible_end as usize, base + row + 1);
                    assert_eq!(
                        meta.window_start as usize,
                        window.map_or(0, |w| (base + row + 1).saturating_sub(w))
                    );
                }
            }
            assert!(chain.row_plan(2041, window, 2048, 16).is_none());
            assert!(gemma4_dense_attention_row_plan(2041, ROWS, window, 2048, 16).is_none());
        }
    }

    #[test]
    fn gemma4_tree_context_selection_parses_and_resolves_kernels() {
        use TreeContextForm::{P16, P2, P2X, P4, P8};
        let v23 = Gemma4DenseAttentionRowsV2Variant {
            scores: 2,
            context: 3,
        };
        let v00 = Gemma4DenseAttentionRowsV2Variant::DEFAULT;
        let today = TreeContextSelection::TODAY;
        assert_eq!(TreeContextSelection::parse("p2"), Some(today));
        assert_eq!(today.kernel(256, v23), Some((TREE_CONTEXT_HD256_P2, 2)));
        assert_eq!(today.kernel(256, v00), Some((TREE_CONTEXT_NEST, 0)));
        assert_eq!(today.kernel(512, v23), Some((TREE_CONTEXT_NEST, 0)));
        assert_eq!(today.kernel(128, v23), None);
        assert_eq!(
            today.resolved_name(512, v23),
            "gemma4_tree_context_nest(hd512_split)"
        );
        assert_eq!(today.resolved_name(256, v23), TREE_CONTEXT_HD256_P2);
        let p8 = TreeContextSelection::parse(" P8 ").unwrap();
        assert_eq!(
            p8,
            TreeContextSelection {
                sliding: P8,
                global: P8
            }
        );
        assert_eq!(p8.kernel(256, v00), Some((TREE_CONTEXT_HD256_P8X, 8)));
        assert_eq!(p8.kernel(512, v23), Some((TREE_CONTEXT_HD512_P8X, 8)));
        assert_eq!(p8.label(), "p8,p8");
        // A single form keeps p2 where it is not instantiated.
        assert_eq!(
            TreeContextSelection::parse("p2x"),
            Some(TreeContextSelection {
                sliding: P2X,
                global: P2X
            })
        );
        assert_eq!(
            TreeContextSelection::parse("p2x").unwrap().kernel(512, v00),
            Some((TREE_CONTEXT_HD512_P2X, 2))
        );
        assert_eq!(
            TreeContextSelection::parse("p4"),
            Some(TreeContextSelection {
                sliding: P4,
                global: P4
            })
        );
        let p2x4 = TreeContextSelection::parse("p2x,p4").unwrap();
        assert_eq!(p2x4.kernel(256, v23), Some((TREE_CONTEXT_HD256_P2X, 2)));
        assert_eq!(p2x4.kernel(256, v00), Some((TREE_CONTEXT_HD256_P2X, 2)));
        assert_eq!(p2x4.kernel(512, v23), Some((TREE_CONTEXT_HD512_P4X, 4)));
        assert_eq!(p2x4.label(), "p2x,p4");
        assert_eq!(
            TreeContextSelection::parse("p16x"),
            Some(TreeContextSelection {
                sliding: P2,
                global: P16
            })
        );
        let pair = TreeContextSelection::parse("p4,p16").unwrap();
        assert_eq!(pair.kernel(256, v23), Some((TREE_CONTEXT_HD256_P4X, 4)));
        assert_eq!(pair.kernel(512, v23), Some((TREE_CONTEXT_HD512_P16X, 16)));
        assert_eq!(pair.label(), "p4,p16");
        // Explicit pairs refuse a missing instantiation; garbage is refused.
        assert_eq!(TreeContextSelection::parse("p16,p8"), None);
        assert_eq!(TreeContextSelection::parse("p16,p2x"), None);
        assert_eq!(TreeContextSelection::parse("p2x,p16x"), Some(TreeContextSelection {
            sliding: P2X,
            global: P16
        }));
        assert_eq!(TreeContextSelection::parse("p3"), None);
        assert_eq!(TreeContextSelection::parse(""), None);
        assert_eq!(TreeContextSelection::parse("p8,p8,p8"), None);
        assert_eq!(TreeContextSelection::gate_forms(256).len(), 4);
        assert_eq!(TreeContextSelection::gate_forms(512).len(), 5);
        assert_eq!(TreeContextSelection::receipt_forms().len(), 13);
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
                // The linear chain: whole-union prefix scores, no suffix dispatch.
                plans.push(Gemma4DenseTreePlan::chain().clone());
                for plan in plans {
                    let (tree_meta, tree_bytes) =
                        plan.row_plan(base, window, capacity, heads).unwrap();
                    let ts = buffer(tree_bytes / 4);
                    let td = buffer(ROWS * heads);
                    let to = buffer(query.len());
                    let variants = [
                        Gemma4DenseAttentionRowsV2Variant::DEFAULT,
                        Gemma4DenseAttentionRowsV2Variant::NEST,
                        Gemma4DenseAttentionRowsV2Variant {
                            scores: 2,
                            context: 3,
                        },
                    ];
                    // Every tree-only context form for this head dimension (the
                    // tree selector, not the V2 variant list) under every scores form.
                    for (variant, form) in variants.into_iter().flat_map(|variant| {
                        TreeContextSelection::gate_forms(hd)
                            .into_iter()
                            .map(move |form| (variant, form))
                    }) {
                        // Poison the packed score scratch so an element in [base, base+8)
                        // left unwritten by a short prefix grid can never pass by
                        // accident: the exp-score comparison would see the NaN bits.
                        write_buffer_f32(&ts, &vec![f32::NAN; tree_bytes / 4]);
                        write_buffer_f32(&to, &vec![f32::NAN; query.len()]);
                        let command = kernel.queue.new_command_buffer();
                        let encoder = command.new_compute_command_encoder();
                        assert!(
                            encode_tree_attention_with_form(
                                encoder, kernel, &tq, &tk, &tv, &ts, &td, &to, heads, kv_heads,
                                hd, capacity, base, window, 1.0, &plan, variant, form
                            ),
                            "hd={hd} base={base} parents={:?} variant={variant:?} form={} declined",
                            plan.parents,
                            form.label()
                        );
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
                                "hd={hd} base={base} parents={:?} node={node} variant={variant:?} form={}",
                                plan.parents,
                                form.label()
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
                            if std::env::var_os("CAMELID_TREE_DEBUG_ROUNDING").is_some() {
                                let expected = &output[depth * qdim..(depth + 1) * qdim];
                                let actual = &tree_output[node * qdim..(node + 1) * qdim];
                                if let Some(index) = expected.iter().zip(actual)
                                    .position(|(a, b)| a.to_bits() != b.to_bits()) {
                                    let head = index / hd;
                                    let dim = index % hd;
                                    let kv_head = head / (heads / kv_heads);
                                    let denominator = tree_denom[node * heads + head];
                                    let mut terms = Vec::new();
                                    for p in 0..tm.position_count as usize {
                                        let logical = tm.window_start as usize + p;
                                        let physical = if logical < base { logical } else {
                                            base + plan.ancestors[node][logical - base] as usize
                                        };
                                        let score = tree_scores[tm.score_offset as usize + head * tm.position_count as usize + p];
                                        let value = values[(kv_head * capacity + physical) * hd + dim];
                                        terms.push((score, value));
                                    }
                                    eprintln!("[tree-rounding] {label} index={index} head={head} dim={dim} denom={:08x} expected={:08x} actual={:08x}",
                                        denominator.to_bits(), expected[index].to_bits(), actual[index].to_bits());
                                    if terms.len() <= 8 {
                                        for (p, &(score, value)) in terms.iter().enumerate() {
                                            eprintln!("[tree-rounding] p={p} exp={:08x} value={:08x}", score.to_bits(), value.to_bits());
                                        }
                                    }
                                    // Rust f32 mul_add provides an explicit one-rounding FMA.
                                    // Screen rounded divide and its adjacent representations;
                                    // GPU inverse/contraction still needs direct evidence.
                                    let inverse = 1.0f32 / denominator;
                                    for inverse_bits in [inverse.to_bits() - 1, inverse.to_bits(), inverse.to_bits() + 1] {
                                        let inv = f32::from_bits(inverse_bits);
                                        let mut svi_fma = 0.0f32;
                                        let mut siv_fma = 0.0f32;
                                        let mut vis_fma = 0.0f32;
                                        let mut unfused = 0.0f32;
                                        let mut dot_fma = 0.0f32;
                                        let mut dot_unfused = 0.0f32;
                                        for &(score, value) in &terms {
                                            svi_fma = (score * value).mul_add(inv, svi_fma);
                                            siv_fma = (score * inv).mul_add(value, siv_fma);
                                            vis_fma = (value * inv).mul_add(score, vis_fma);
                                            unfused = unfused + ((score * inv) * value);
                                            dot_fma = score.mul_add(value, dot_fma);
                                            dot_unfused = dot_unfused + score * value;
                                        }
                                        for (name, value) in [("fma(score*value,inv,acc)", svi_fma),
                                            ("fma(score*inv,value,acc)", siv_fma), ("fma(value*inv,score,acc)", vis_fma),
                                            ("add(mul(mul(score,inv),value))", unfused), ("fma_dot_then_inv", dot_fma * inv),
                                            ("unfused_dot_then_inv", dot_unfused * inv)] {
                                            eprintln!("[tree-rounding] inv={inverse_bits:08x} form={name} value={:08x} expected_match={} actual_match={}",
                                                value.to_bits(), value.to_bits() == expected[index].to_bits(), value.to_bits() == actual[index].to_bits());
                                        }
                                    }
                                }
                            }
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
        eprintln!(
            "[gemma4-tree] exact attention: {cases} tree configurations, {} independent W8-padded paths \
             (plans: 4 forks + interleaved + linear chain; context forms p2/p2x/p4/p8 at HD256, \
             p2/p2x/p4/p8/p16 at HD512; scores forms default/nest/sg4)",
            cases * ROWS
        );
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
