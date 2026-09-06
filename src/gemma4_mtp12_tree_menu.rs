//! Metal-free decision logic for the Gemma 4 12B MTP12 W8 draft tree.
//!
//! The assistant runs four "primary" forwards in the first command buffer.
//! From those four top-1/top-2 margins this module chooses which eight-node
//! tree to finish in the second command buffer, which forwards that costs, and
//! how the resulting nodes map onto the eight physical verifier rows.
//!
//! Two properties are deliberate and load bearing:
//!
//! * **Shape preserving.** `layout` fixes the node set from the chosen shape
//!   alone; `finalize` never re-ranks nodes and never drops a top-1 chain node.
//!   The consequence is that the set of topologies the runtime can emit is
//!   finite and enumerable (`gate_topologies`), so the model-level bit-exact
//!   gate can cover exactly what the runtime can produce.
//! * **GPU free.** Everything here is pure CPU arithmetic over the recorded
//!   top-2 pairs, so the whole decision path is unit-testable without a device.
//!
//! Rank-two siblings are only taken from the first `ALT_STEPS` primary
//! forwards; that bound is what keeps `gate_topologies()` at seventeen entries
//! instead of the thirty-five a free fork step would produce, which is what
//! keeps the model-level gate affordable per build cycle.

use std::fmt;

pub(crate) const POLICY_ENV: &str = "CAMELID_GEMMA4_MTP12_TREE_POLICY";
pub(crate) const LAMBDA_ENV: &str = "CAMELID_GEMMA4_MTP12_TREE_LAMBDA";
pub(crate) const CALIB_ENV: &str = "CAMELID_GEMMA4_MTP12_TREE_CALIB";

/// Forwards in the first command buffer; their margins drive the choice.
pub(crate) const PRIMARY: usize = 4;
/// Physical rows in a W8 tree, including the already proved anchor.
pub(crate) const NODES: usize = 8;
/// Rank-two siblings may only be taken from these first CB1 forwards.
pub(crate) const ALT_STEPS: usize = 3;
/// Modeled margins saturate long before this; `+inf` clamps here.
pub(crate) const MARGIN_CLAMP: f32 = 25.0;
/// Committed rows a single assistant forward has to be worth to pay for
/// itself: 2.43 ms measured marginal GPU cost x 3.75 rows / 107.1 ms round.
pub(crate) const DEFAULT_LAMBDA: f32 = 0.08;

/// Named eight-node trees. `a+b+c` reads as "a top-1 chain nodes, b rank-two
/// siblings of chain nodes, c descendants of a rank-two sibling".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Shape {
    /// Seven chained forwards, no fork. Verified by the linear lane.
    Lin7,
    /// Today's default tree: fork at one step, two chained continuations.
    P4A1C2,
    /// Six chain nodes plus one sibling.
    P6A1,
    /// Five chain nodes, one sibling and one continuation behind it.
    P5A1C1,
    /// Five chain nodes plus two siblings (no continuation).
    P5A2,
    /// Four chain nodes, two siblings, one continuation behind the earlier.
    P4A2C1,
    /// Four chain nodes plus all three eligible siblings; no second buffer.
    P4A3,
}

impl Shape {
    pub(crate) const ALL: [Shape; 7] = [
        Shape::Lin7,
        Shape::P4A1C2,
        Shape::P6A1,
        Shape::P5A1C1,
        Shape::P5A2,
        Shape::P4A2C1,
        Shape::P4A3,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Shape::Lin7 => "lin7",
            Shape::P4A1C2 => "4+1+2",
            Shape::P6A1 => "6+1",
            Shape::P5A1C1 => "5+1+1",
            Shape::P5A2 => "5+2",
            Shape::P4A2C1 => "4+2+1",
            Shape::P4A3 => "4+3",
        }
    }

    /// Length of the top-1 chain rooted at the anchor, excluding the anchor.
    pub(crate) fn chain(self) -> usize {
        match self {
            Shape::Lin7 => 7,
            Shape::P6A1 => 6,
            Shape::P5A1C1 | Shape::P5A2 => 5,
            Shape::P4A1C2 | Shape::P4A2C1 | Shape::P4A3 => 4,
        }
    }

    /// Rank-two siblings kept, all of them children of chain nodes 0..3.
    pub(crate) fn alts(self) -> usize {
        match self {
            Shape::Lin7 => 0,
            Shape::P6A1 | Shape::P5A1C1 | Shape::P4A1C2 => 1,
            Shape::P5A2 | Shape::P4A2C1 => 2,
            Shape::P4A3 => 3,
        }
    }

    /// Nodes chained behind the expanded sibling.
    pub(crate) fn continuations(self) -> usize {
        match self {
            Shape::P4A1C2 => 2,
            Shape::P5A1C1 | Shape::P4A2C1 => 1,
            _ => 0,
        }
    }

    /// Assistant forwards the whole round costs, both command buffers.
    pub(crate) fn forwards(self) -> usize {
        self.chain() + self.continuations()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Policy {
    /// Byte-for-byte the previously qualified `select_branch` proposal.
    Legacy,
    /// Per-round menu over the named shapes.
    Dyn,
    /// One named shape every round (falls back to `Lin7` when the round has
    /// too few valid rank-two siblings to build that shape).
    Fixed(Shape),
}

impl Policy {
    pub(crate) fn name(self) -> String {
        match self {
            Policy::Legacy => "legacy".to_string(),
            Policy::Dyn => "dyn".to_string(),
            Policy::Fixed(shape) => format!("fixed:{}", shape.name()),
        }
    }
}

/// Strict, single-read selector parsing in the style of `parse_max_margin`.
/// Unset is `legacy`, which keeps the qualified V3 proposal unchanged.
pub(crate) fn parse_policy(value: Option<&str>) -> Result<Policy, String> {
    let Some(text) = value else {
        return Ok(Policy::Legacy);
    };
    if text == "legacy" {
        return Ok(Policy::Legacy);
    }
    if text == "dyn" {
        return Ok(Policy::Dyn);
    }
    if let Some(name) = text.strip_prefix("fixed:") {
        if let Some(shape) = Shape::ALL.iter().find(|shape| shape.name() == name) {
            return Ok(Policy::Fixed(*shape));
        }
    }
    Err(policy_error(text))
}

fn policy_error(text: &str) -> String {
    let names: Vec<String> = Shape::ALL
        .iter()
        .map(|shape| format!("fixed:{}", shape.name()))
        .collect();
    format!(
        "{POLICY_ENV} must be exactly one of legacy, dyn, {}; got {text:?}",
        names.join(", ")
    )
}

/// Committed rows one forward must be worth. Strict like the margin selector.
pub(crate) fn parse_lambda(value: Option<&str>) -> Result<f32, String> {
    let Some(text) = value else {
        return Ok(DEFAULT_LAMBDA);
    };
    let lambda = text.parse::<f32>().map_err(|_| lambda_error(text))?;
    if !lambda.is_finite() || lambda.is_sign_negative() {
        return Err(lambda_error(text));
    }
    Ok(lambda)
}

fn lambda_error(text: &str) -> String {
    format!("{LAMBDA_ENV} must be a finite nonnegative number; got {text:?}")
}

/// The trace-fitted acceptance model. Every field is overridable through
/// `CAMELID_GEMMA4_MTP12_TREE_CALIB` so a pessimistic set can be screened
/// without a rebuild.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Calibration {
    /// `p(m) = sigmoid(a + b*ln(1+m))`, Newton MLE over 144 primary rows.
    pub a: f32,
    pub b: f32,
    /// `P(runner-up == target | top-1 rejected)` for m<=2, 2<m<=4, m>4.
    pub q: [f32; 3],
    /// Acceptance of the first and second continuation behind a correct alt.
    pub c: [f32; 2],
    /// Acceptance of chain nodes 5, 6 and 7, conditioned on their parent.
    pub pt: [f32; 3],
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            a: -1.079,
            b: 1.346,
            q: [0.5, 0.35, 0.2],
            c: [0.60, 0.65],
            pt: [0.62, 0.80, 0.80],
        }
    }
}

impl Calibration {
    /// `p(m)`: probability the top-1 child is the target's greedy id.
    pub(crate) fn p_accept(&self, margin: f32) -> f32 {
        let m = model_margin(margin);
        let logit = self.a + self.b * m.ln_1p();
        let p = 1.0 / (1.0 + (-logit).exp());
        debug_assert!(p.is_finite());
        p.clamp(0.0, 1.0)
    }

    /// `q(m)`: probability the rank-two child is the target's greedy id,
    /// given that the top-1 child is not.
    pub(crate) fn q_runner_up(&self, margin: f32) -> f32 {
        let m = model_margin(margin);
        if m <= 2.0 {
            self.q[0]
        } else if m <= 4.0 {
            self.q[1]
        } else {
            self.q[2]
        }
    }
}

/// Total margin -> model input. A NaN gap (two `+inf` logits, or the all
/// `-inf` degenerate answer) is not a confidence signal, so it models as zero;
/// `+inf` saturates at the clamp. Never returns NaN, so no node probability
/// can be NaN and no comparison can trap.
pub(crate) fn model_margin(margin: f32) -> f32 {
    if margin.is_nan() {
        0.0
    } else {
        margin.clamp(0.0, MARGIN_CLAMP)
    }
}

/// Ten comma-separated finite numbers: `a,b,q1,q2,q3,c0,c1,pt1,pt2,pt3`.
pub(crate) fn parse_calibration(value: Option<&str>) -> Result<Calibration, String> {
    let Some(text) = value else {
        return Ok(Calibration::default());
    };
    let fields: Vec<&str> = text.split(',').collect();
    if fields.len() != 10 {
        return Err(calib_error(text));
    }
    let mut values = [0.0f32; 10];
    for (slot, field) in values.iter_mut().zip(&fields) {
        *slot = field.parse::<f32>().map_err(|_| calib_error(text))?;
        if !slot.is_finite() {
            return Err(calib_error(text));
        }
    }
    // Probabilities stay probabilities; the logistic coefficients are free.
    if values[2..].iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(calib_error(text));
    }
    Ok(Calibration {
        a: values[0],
        b: values[1],
        q: [values[2], values[3], values[4]],
        c: [values[5], values[6]],
        pt: [values[7], values[8], values[9]],
    })
}

fn calib_error(text: &str) -> String {
    format!(
        "{CALIB_ENV} must be a,b,q1,q2,q3,c0,c1,pt1,pt2,pt3 with finite numbers \
         and probabilities in [0,1]; got {text:?}"
    )
}

/// One forward's recorded top-2 answer, already reduced to what the policy
/// uses. `runner_up_valid` is the legacy `select_branch` predicate verbatim:
/// both ids in vocabulary, distinct, and a finite gap.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ForwardTop {
    pub margin: f32,
    pub runner_up_id: u32,
    pub runner_up_valid: bool,
}

impl ForwardTop {
    pub(crate) fn from_pair(values: [f32; 2], ids: [u32; 2], vocab: usize) -> Self {
        let margin = values[0] - values[1];
        let runner_up_valid = (ids[0] as usize) < vocab
            && (ids[1] as usize) < vocab
            && ids[0] != ids[1]
            && margin.is_finite();
        Self {
            margin,
            runner_up_id: ids[1],
            runner_up_valid,
        }
    }
}

/// Per-round modeled quantities derived from the four CB1 margins only. The
/// common `P_1 + .. + P_4` term is dropped from every shape value because it
/// is identical across the menu.
#[derive(Clone, Debug)]
pub(crate) struct Menu {
    /// `reach[k] = prod_{j<k} p(m_j)`: probability chain node k is committed.
    pub reach: [f32; PRIMARY + 1],
    /// Rank-two sibling values `a_s`, descending, with their CB1 step.
    pub alts: Vec<(usize, f32)>,
    calib: Calibration,
}

impl Menu {
    pub(crate) fn new(top: &[ForwardTop; PRIMARY], calib: Calibration) -> Self {
        let mut reach = [1.0f32; PRIMARY + 1];
        for step in 0..PRIMARY {
            reach[step + 1] = reach[step] * calib.p_accept(top[step].margin);
        }
        let mut alts: Vec<(usize, f32)> = (0..ALT_STEPS)
            .filter(|&step| top[step].runner_up_valid)
            .map(|step| {
                let p = calib.p_accept(top[step].margin);
                let q = calib.q_runner_up(top[step].margin);
                (step, reach[step] * (1.0 - p) * q)
            })
            .collect();
        // Descending value, earlier step first on a tie: fully deterministic.
        alts.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        Self { reach, alts, calib }
    }

    /// Modeled committed rows beyond the anchor and the four chain nodes every
    /// shape shares, or `None` when the round has too few valid siblings.
    pub(crate) fn value(&self, shape: Shape) -> Option<f32> {
        if self.alts.len() < shape.alts() {
            return None;
        }
        let tail = |i: usize| -> f32 {
            // Chain nodes 5..7 are proposed before their margins exist, so
            // they are worth the calibrated primary-tail acceptance only.
            (0..=i).fold(self.reach[PRIMARY], |acc, j| acc * self.calib.pt[j])
        };
        let alt = |i: usize| self.alts[i].1;
        Some(match shape {
            Shape::Lin7 => tail(0) + tail(1) + tail(2),
            Shape::P6A1 => tail(0) + tail(1) + alt(0),
            Shape::P5A1C1 => tail(0) + alt(0) + alt(0) * self.calib.c[0],
            Shape::P4A1C2 => {
                alt(0) + alt(0) * self.calib.c[0] + alt(0) * self.calib.c[0] * self.calib.c[1]
            }
            Shape::P5A2 => tail(0) + alt(0) + alt(1),
            Shape::P4A2C1 => {
                // The continuation always hangs off the earlier of the two
                // kept siblings, which bounds the emittable topology set.
                let earlier = if self.alts[0].0 <= self.alts[1].0 {
                    alt(0)
                } else {
                    alt(1)
                };
                alt(0) + alt(1) + earlier * self.calib.c[0]
            }
            Shape::P4A3 => alt(0) + alt(1) + alt(2),
        })
    }

    /// CB1 steps whose rank-two child this shape keeps, ascending.
    pub(crate) fn alt_steps(&self, shape: Shape) -> Option<Vec<usize>> {
        if self.alts.len() < shape.alts() {
            return None;
        }
        let mut steps: Vec<usize> = self.alts[..shape.alts()]
            .iter()
            .map(|(step, _)| *step)
            .collect();
        steps.sort_unstable();
        Some(steps)
    }
}

/// Pick the shape. `Legacy` never reaches here; `Fixed` degrades to `Lin7`
/// only when the round cannot build the requested shape at all.
pub(crate) fn choose(policy: Policy, menu: &Menu, lambda: f32) -> Shape {
    match policy {
        Policy::Legacy => Shape::P4A1C2,
        Policy::Fixed(shape) => {
            if menu.value(shape).is_some() {
                shape
            } else {
                Shape::Lin7
            }
        }
        Policy::Dyn => {
            let mut best = Shape::Lin7;
            let mut best_score = f32::NEG_INFINITY;
            for shape in Shape::ALL {
                let Some(value) = menu.value(shape) else {
                    continue;
                };
                let score = value - lambda * shape.forwards() as f32;
                let better = score > best_score
                    // Deterministic ties: fewer forwards, then declaration order.
                    || (score == best_score && shape.forwards() < best.forwards());
                if better {
                    best = shape;
                    best_score = score;
                }
            }
            best
        }
    }
}

/// Where a physical row's token comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NodeSource {
    /// Row zero: the already proved anchor.
    Anchor,
    /// Top-1 answer of this forward (GPU writes `output_token[f + 1]`).
    Top1(usize),
    /// Rank-two answer of this CB1 forward (the CPU writes its slot).
    RunnerUp(usize),
}

/// One second-command-buffer forward. Recurrent state is always addressed
/// explicitly by the slot of the forward that produced the parent's hidden;
/// the live `recurrent_hidden` is never reused, because CB2 interleaves
/// parents (5+1+1 runs chain node 5 and then a sibling from step s).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ForwardSpec {
    /// `output_token` slot holding the input token of this forward.
    pub input_slot: usize,
    /// `chain_recurrent_hidden` slot holding the input node's recurrent state.
    pub recurrent_slot: usize,
    /// Depth of the input node; selects the chain RoPE table.
    pub query_step: usize,
    /// Forward index; also the results/hidden slot this forward writes.
    pub history_step: usize,
}

/// Where the CPU stores a rank-two token so a CB2 forward can read it. GPU
/// forwards own slots 1..=7, so the eight-and-up slots never collide.
pub(crate) fn runner_up_slot(step: usize) -> usize {
    NODES + step
}

/// The complete physical shape of one round's tree: rows, edges, the forwards
/// the second command buffer must run, and the slot the CPU has to fill.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Layout {
    pub shape: Shape,
    pub parents: Vec<i32>,
    pub depths: Vec<u32>,
    pub sources: Vec<NodeSource>,
    /// Rows of the ordinary top-1 chain, including the anchor.
    pub primary_rows: Vec<usize>,
    /// CB1 steps whose rank-two child is kept, ascending.
    pub fork_forwards: Vec<usize>,
    /// Rank-two token the CPU must publish before CB2, if any.
    pub runner_up_write: Option<(usize, usize)>,
    pub cb2: Vec<ForwardSpec>,
}

/// Build the physical layout of `shape` with rank-two siblings at `alt_steps`
/// (ascending, distinct, each `< ALT_STEPS`). Rows are the top-1 chain, then
/// the siblings in step order, then the continuations behind the earliest
/// sibling. Row order is exactly the legacy one for `P4A1C2`.
pub(crate) fn layout(shape: Shape, alt_steps: &[usize]) -> Option<Layout> {
    if alt_steps.len() != shape.alts()
        || alt_steps.iter().any(|step| *step >= ALT_STEPS)
        || alt_steps.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let chain = shape.chain();
    let mut parents = vec![-1i32; NODES];
    let mut depths = vec![0u32; NODES];
    let mut sources = vec![NodeSource::Anchor; NODES];
    for row in 1..=chain {
        parents[row] = row as i32 - 1;
        depths[row] = row as u32;
        sources[row] = NodeSource::Top1(row - 1);
    }
    for (index, &step) in alt_steps.iter().enumerate() {
        let row = chain + 1 + index;
        parents[row] = step as i32;
        depths[row] = step as u32 + 1;
        sources[row] = NodeSource::RunnerUp(step);
    }
    // Continuations hang off the earliest kept sibling, which is its own row.
    let expanded_row = chain + 1;
    let mut cb2 = Vec::new();
    for forward in PRIMARY..chain {
        cb2.push(ForwardSpec {
            input_slot: forward,
            recurrent_slot: forward - 1,
            query_step: forward,
            history_step: forward,
        });
    }
    let mut runner_up_write = None;
    for index in 0..shape.continuations() {
        let forward = chain + index;
        let row = chain + shape.alts() + 1 + index;
        let parent_row = if index == 0 {
            expanded_row
        } else {
            chain + shape.alts() + index
        };
        parents[row] = parent_row as i32;
        depths[row] = depths[parent_row] + 1;
        sources[row] = NodeSource::Top1(forward);
        let (input_slot, recurrent_slot) = if index == 0 {
            let step = alt_steps[0];
            runner_up_write = Some((runner_up_slot(step), step));
            (runner_up_slot(step), step)
        } else {
            (forward, forward - 1)
        };
        cb2.push(ForwardSpec {
            input_slot,
            recurrent_slot,
            query_step: depths[parent_row] as usize,
            history_step: forward,
        });
    }
    if parents.iter().skip(1).any(|parent| *parent < 0) {
        return None;
    }
    Some(Layout {
        shape,
        parents,
        depths,
        sources,
        primary_rows: (0..=chain).collect(),
        fork_forwards: alt_steps.to_vec(),
        runner_up_write,
        cb2,
    })
}

/// Everything the caller needs after the second command buffer completes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Finalized {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
    pub depths: Vec<u32>,
    pub primary_rows: Vec<usize>,
    /// Earliest kept fork among the first four forwards; `None` iff linear.
    pub branch_primary_step: Option<usize>,
    pub fork_forwards: Vec<usize>,
    /// Modeled probability that each physical row is committed, from the
    /// observed margin of the forward that produced it. Row zero is the
    /// anchor and is always 1.0.
    pub node_p: Vec<f32>,
    pub linear: bool,
}

/// Assemble the proposal. `tokens` are read from the layout's node sources;
/// `top` holds every forward's recorded top-2. The linearity invariant the
/// round loop checks is computed here, not argued, and the fork field is
/// derived from it so `linear_topology != branch_primary_step.is_some()` can
/// never fire.
pub(crate) fn finalize(
    layout: &Layout,
    top: &[ForwardTop],
    gpu_tokens: &[u32],
    anchor: u32,
    calib: Calibration,
) -> Result<Finalized, String> {
    let forwards = layout.shape.forwards();
    if top.len() < forwards || gpu_tokens.len() < forwards + 1 {
        return Err(format!(
            "tree finalize needs {forwards} forward results; got {} tops and {} tokens",
            top.len(),
            gpu_tokens.len()
        ));
    }
    let mut tokens = vec![0u32; NODES];
    let mut node_p = vec![0.0f32; NODES];
    for row in 0..NODES {
        match layout.sources[row] {
            NodeSource::Anchor => {
                tokens[row] = anchor;
                node_p[row] = 1.0;
            }
            NodeSource::Top1(forward) => {
                tokens[row] = gpu_tokens[forward + 1];
                let parent = layout.parents[row] as usize;
                node_p[row] = node_p[parent] * calib.p_accept(top[forward].margin);
            }
            NodeSource::RunnerUp(forward) => {
                if !top[forward].runner_up_valid {
                    return Err(format!(
                        "tree finalize kept an invalid rank-two at {forward}"
                    ));
                }
                tokens[row] = top[forward].runner_up_id;
                let parent = layout.parents[row] as usize;
                let margin = top[forward].margin;
                node_p[row] =
                    node_p[parent] * (1.0 - calib.p_accept(margin)) * calib.q_runner_up(margin);
            }
        }
    }
    if node_p.iter().any(|p| !p.is_finite()) {
        return Err("tree finalize produced a non-finite node probability".to_string());
    }
    // Computed, never assumed: the round loop turns a disagreement between
    // these two into a hard mid-generation RuntimeShapeMismatch.
    let linear = layout
        .parents
        .iter()
        .enumerate()
        .all(|(row, parent)| *parent == row as i32 - 1);
    let branch_primary_step = if linear {
        None
    } else {
        layout.fork_forwards.first().copied()
    };
    if linear != branch_primary_step.is_none() || linear != layout.fork_forwards.is_empty() {
        return Err("tree finalize could not agree on the linearity invariant".to_string());
    }
    if branch_primary_step.is_some_and(|step| step >= PRIMARY) {
        return Err("tree finalize forked outside the primary forwards".to_string());
    }
    Ok(Finalized {
        tokens,
        parents: layout.parents.clone(),
        depths: layout.depths.clone(),
        primary_rows: layout.primary_rows.clone(),
        branch_primary_step,
        fork_forwards: layout.fork_forwards.clone(),
        node_p,
        linear,
    })
}

/// Every distinct non-linear topology the runtime can emit under any policy
/// value, including today's legacy fork steps. The model-level bit-exactness
/// gate iterates this so the gated set equals the emittable set.
pub(crate) fn gate_topologies() -> Vec<(Vec<i32>, Vec<u32>)> {
    let mut out: Vec<(Vec<i32>, Vec<u32>)> = Vec::new();
    let mut push = |parents: Vec<i32>, depths: Vec<u32>| {
        if !out.iter().any(|(p, d)| *p == parents && *d == depths) {
            out.push((parents, depths));
        }
    };
    // Legacy `select_branch` may fork at any of the four primary steps; the
    // menu itself only forks at the first three.
    for step in 0..PRIMARY as i32 {
        push(
            vec![-1, 0, 1, 2, 3, step, 5, 6],
            vec![
                0,
                1,
                2,
                3,
                4,
                step as u32 + 1,
                step as u32 + 2,
                step as u32 + 3,
            ],
        );
    }
    for shape in Shape::ALL {
        for steps in alt_step_sets(shape.alts()) {
            if let Some(layout) = layout(shape, &steps) {
                if layout
                    .parents
                    .iter()
                    .enumerate()
                    .any(|(row, parent)| *parent != row as i32 - 1)
                {
                    push(layout.parents, layout.depths);
                }
            }
        }
    }
    out
}

/// Ascending `k`-subsets of the eligible fork steps.
pub(crate) fn alt_step_sets(k: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for _ in 0..k {
        let mut next = Vec::new();
        for prefix in &out {
            let start = prefix.last().map_or(0, |last| last + 1);
            for step in start..ALT_STEPS {
                let mut extended = prefix.clone();
                extended.push(step);
                next.push(extended);
            }
        }
        out = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 262_144;

    fn pair(margin: f32) -> ForwardTop {
        ForwardTop::from_pair([margin, 0.0], [7, 9], VOCAB)
    }

    fn tops(margins: [f32; PRIMARY]) -> [ForwardTop; PRIMARY] {
        std::array::from_fn(|i| pair(margins[i]))
    }

    fn all_tops(margins: [f32; PRIMARY]) -> Vec<ForwardTop> {
        let mut out = tops(margins).to_vec();
        // CB2 forwards report their own top-2 as well.
        out.extend((0..3).map(|i| pair(1.0 + i as f32)));
        out
    }

    #[test]
    fn tree_policy_selector_is_strict_and_defaults_to_legacy() {
        assert_eq!(parse_policy(None), Ok(Policy::Legacy));
        assert_eq!(parse_policy(Some("legacy")), Ok(Policy::Legacy));
        assert_eq!(parse_policy(Some("dyn")), Ok(Policy::Dyn));
        for shape in Shape::ALL {
            let text = format!("fixed:{}", shape.name());
            assert_eq!(parse_policy(Some(&text)), Ok(Policy::Fixed(shape)));
            assert_eq!(parse_policy(Some(&text)).unwrap().name(), text);
        }
        for text in [
            "",
            " ",
            "Legacy",
            "legacy ",
            " legacy",
            "DYN",
            "dynamic",
            "fixed",
            "fixed:",
            "fixed:lin",
            "fixed:4+1+3",
            "fixed: 6+1",
            "fixed:6+1 ",
            "1",
            "0",
            "4+1+2",
        ] {
            assert!(parse_policy(Some(text)).is_err(), "must reject {text:?}");
        }
    }

    #[test]
    fn tree_menu_lambda_and_calibration_parse_strictly() {
        assert_eq!(
            parse_lambda(None).unwrap().to_bits(),
            DEFAULT_LAMBDA.to_bits()
        );
        for (text, expected) in [("0", 0.0f32), ("0.08", 0.08), ("1", 1.0), ("2e-2", 0.02)] {
            assert_eq!(
                parse_lambda(Some(text)).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        for text in [
            "", " ", "-1", "-0.0", "NaN", "inf", "-inf", "0,08", "0.08 ", "x",
        ] {
            assert!(parse_lambda(Some(text)).is_err(), "must reject {text:?}");
        }
        assert_eq!(parse_calibration(None).unwrap(), Calibration::default());
        let pessimistic = "-1.079,1.346,0.4,0.3,0.15,0.5,0.6,0.55,0.6,0.6";
        let parsed = parse_calibration(Some(pessimistic)).unwrap();
        assert_eq!(parsed.q, [0.4, 0.3, 0.15]);
        assert_eq!(parsed.c, [0.5, 0.6]);
        assert_eq!(parsed.pt, [0.55, 0.6, 0.6]);
        for text in [
            "",
            "-1.079,1.346",
            "-1.079,1.346,0.4,0.3,0.15,0.5,0.6,0.55,0.6",
            "-1.079,1.346,0.4,0.3,0.15,0.5,0.6,0.55,0.6,0.6,0.6",
            "-1.079,1.346,1.4,0.3,0.15,0.5,0.6,0.55,0.6,0.6",
            "-1.079,1.346,-0.1,0.3,0.15,0.5,0.6,0.55,0.6,0.6",
            "NaN,1.346,0.4,0.3,0.15,0.5,0.6,0.55,0.6,0.6",
        ] {
            assert!(
                parse_calibration(Some(text)).is_err(),
                "must reject {text:?}"
            );
        }
    }

    #[test]
    fn tree_menu_acceptance_model_matches_the_fitted_curve() {
        let calib = Calibration::default();
        for (margin, expected) in [
            (0.0f32, 0.25f32),
            (0.5, 0.37),
            (1.0, 0.46),
            (2.0, 0.60),
            (3.0, 0.69),
            (5.0, 0.79),
            (10.0, 0.90),
            (20.0, 0.95),
        ] {
            let p = calib.p_accept(margin);
            assert!(
                (p - expected).abs() < 0.02,
                "p({margin}) = {p}, design says {expected}"
            );
        }
        // Saturation and the total-order guarantees the ranking relies on.
        assert_eq!(
            calib.p_accept(f32::INFINITY).to_bits(),
            calib.p_accept(MARGIN_CLAMP).to_bits()
        );
        assert_eq!(
            calib.p_accept(f32::NAN).to_bits(),
            calib.p_accept(0.0).to_bits()
        );
        assert!(calib.p_accept(f32::NAN).is_finite());
        assert_eq!(calib.q_runner_up(2.0), 0.5);
        assert_eq!(calib.q_runner_up(2.0001), 0.35);
        assert_eq!(calib.q_runner_up(4.0), 0.35);
        assert_eq!(calib.q_runner_up(4.0001), 0.2);
        assert_eq!(calib.q_runner_up(f32::INFINITY), 0.2);
        assert_eq!(calib.q_runner_up(f32::NAN), 0.5);
    }

    #[test]
    fn tree_menu_runner_up_validity_matches_the_legacy_predicate() {
        // Same predicate as `select_branch`: in vocabulary, distinct, finite.
        assert!(ForwardTop::from_pair([3.0, 1.0], [7, 9], VOCAB).runner_up_valid);
        assert!(!ForwardTop::from_pair([3.0, 1.0], [7, 7], VOCAB).runner_up_valid);
        assert!(!ForwardTop::from_pair([3.0, 1.0], [7, u32::MAX], VOCAB).runner_up_valid);
        assert!(!ForwardTop::from_pair([3.0, 1.0], [u32::MAX, 9], VOCAB).runner_up_valid);
        let both_inf = ForwardTop::from_pair([f32::INFINITY, f32::INFINITY], [7, 9], VOCAB);
        assert!(!both_inf.runner_up_valid);
        assert!(both_inf.margin.is_nan());
        assert_eq!(model_margin(both_inf.margin), 0.0);
        let one_inf = ForwardTop::from_pair([f32::INFINITY, 1.0], [7, 9], VOCAB);
        assert!(!one_inf.runner_up_valid);
        assert_eq!(model_margin(one_inf.margin), MARGIN_CLAMP);
        let all_neg = ForwardTop::from_pair([f32::NEG_INFINITY, f32::NEG_INFINITY], [0, 0], VOCAB);
        assert!(!all_neg.runner_up_valid);
        assert!(model_margin(all_neg.margin).is_finite());
    }

    #[test]
    fn tree_menu_shapes_cost_the_forwards_they_claim() {
        for shape in Shape::ALL {
            assert_eq!(
                shape.chain() + shape.alts() + shape.continuations(),
                NODES - 1,
                "{shape} must fill seven non-anchor rows"
            );
            assert_eq!(
                shape.forwards(),
                shape.chain() + shape.continuations(),
                "{shape} forward count"
            );
            assert!((4..=7).contains(&shape.forwards()));
        }
        assert_eq!(Shape::Lin7.forwards(), 7);
        assert_eq!(Shape::P4A1C2.forwards(), 6);
        assert_eq!(Shape::P6A1.forwards(), 6);
        assert_eq!(Shape::P5A1C1.forwards(), 6);
        assert_eq!(Shape::P5A2.forwards(), 5);
        assert_eq!(Shape::P4A2C1.forwards(), 5);
        assert_eq!(Shape::P4A3.forwards(), 4);
    }

    #[test]
    fn tree_menu_layouts_match_the_named_topologies() {
        // The legacy row order, node for node.
        for step in 0..ALT_STEPS {
            let l = layout(Shape::P4A1C2, &[step]).unwrap();
            assert_eq!(l.parents, vec![-1, 0, 1, 2, 3, step as i32, 5, 6]);
            assert_eq!(
                l.depths,
                vec![
                    0,
                    1,
                    2,
                    3,
                    4,
                    step as u32 + 1,
                    step as u32 + 2,
                    step as u32 + 3
                ]
            );
            assert_eq!(l.primary_rows, vec![0, 1, 2, 3, 4]);
        }
        let l = layout(Shape::P6A1, &[1]).unwrap();
        assert_eq!(l.parents, vec![-1, 0, 1, 2, 3, 4, 5, 1]);
        assert_eq!(l.depths, vec![0, 1, 2, 3, 4, 5, 6, 2]);
        let l = layout(Shape::P5A1C1, &[1]).unwrap();
        assert_eq!(l.parents, vec![-1, 0, 1, 2, 3, 4, 1, 6]);
        assert_eq!(l.depths, vec![0, 1, 2, 3, 4, 5, 2, 3]);
        let l = layout(Shape::P5A2, &[0, 2]).unwrap();
        assert_eq!(l.parents, vec![-1, 0, 1, 2, 3, 4, 0, 2]);
        assert_eq!(l.depths, vec![0, 1, 2, 3, 4, 5, 1, 3]);
        let l = layout(Shape::P4A2C1, &[0, 2]).unwrap();
        assert_eq!(l.parents, vec![-1, 0, 1, 2, 3, 0, 2, 5]);
        assert_eq!(l.depths, vec![0, 1, 2, 3, 4, 1, 3, 2]);
        let l = layout(Shape::P4A3, &[0, 1, 2]).unwrap();
        assert_eq!(l.parents, vec![-1, 0, 1, 2, 3, 0, 1, 2]);
        assert_eq!(l.depths, vec![0, 1, 2, 3, 4, 1, 2, 3]);
        let l = layout(Shape::Lin7, &[]).unwrap();
        assert_eq!(l.parents, vec![-1, 0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(l.depths, (0..8).collect::<Vec<u32>>());
        assert!(l.runner_up_write.is_none());
        // Malformed sibling sets are refused, not silently repaired.
        assert!(layout(Shape::P4A3, &[0, 1]).is_none());
        assert!(layout(Shape::P5A2, &[1, 1]).is_none());
        assert!(layout(Shape::P5A2, &[2, 0]).is_none());
        assert!(layout(Shape::P6A1, &[ALT_STEPS]).is_none());
        assert!(layout(Shape::Lin7, &[0]).is_none());
    }

    #[test]
    fn tree_menu_forward_specs_bind_the_parent_recurrent_slot() {
        // A continuation must read the hidden of the forward that produced its
        // parent, never the live recurrent buffer: CB2 interleaves parents.
        let l = layout(Shape::P5A1C1, &[2]).unwrap();
        assert_eq!(
            l.cb2,
            vec![
                // Chain node 5 continues the primary chain.
                ForwardSpec {
                    input_slot: 4,
                    recurrent_slot: 3,
                    query_step: 4,
                    history_step: 4
                },
                // Then the sibling of chain node 3, whose hidden is slot 2.
                ForwardSpec {
                    input_slot: runner_up_slot(2),
                    recurrent_slot: 2,
                    query_step: 3,
                    history_step: 5
                },
            ]
        );
        assert_eq!(l.runner_up_write, Some((runner_up_slot(2), 2)));
        let l = layout(Shape::P6A1, &[0]).unwrap();
        assert_eq!(
            l.cb2,
            vec![
                ForwardSpec {
                    input_slot: 4,
                    recurrent_slot: 3,
                    query_step: 4,
                    history_step: 4
                },
                ForwardSpec {
                    input_slot: 5,
                    recurrent_slot: 4,
                    query_step: 5,
                    history_step: 5
                },
            ]
        );
        assert!(l.runner_up_write.is_none());
        // Today's shape: the fork, then its own child.
        let l = layout(Shape::P4A1C2, &[1]).unwrap();
        assert_eq!(
            l.cb2,
            vec![
                ForwardSpec {
                    input_slot: runner_up_slot(1),
                    recurrent_slot: 1,
                    query_step: 2,
                    history_step: 4
                },
                ForwardSpec {
                    input_slot: 5,
                    recurrent_slot: 4,
                    query_step: 3,
                    history_step: 5
                },
            ]
        );
        let l = layout(Shape::P4A2C1, &[1, 2]).unwrap();
        assert_eq!(
            l.cb2,
            vec![ForwardSpec {
                input_slot: runner_up_slot(1),
                recurrent_slot: 1,
                query_step: 2,
                history_step: 4
            }]
        );
        assert!(layout(Shape::P4A3, &[0, 1, 2]).unwrap().cb2.is_empty());
        // Every emittable forward stays inside the seven chain RoPE tables and
        // inside the sixteen token slots.
        for shape in Shape::ALL {
            for steps in alt_step_sets(shape.alts()) {
                let l = layout(shape, &steps).unwrap();
                assert_eq!(l.cb2.len(), shape.forwards() - PRIMARY);
                for spec in &l.cb2 {
                    assert!(spec.query_step <= 6, "{shape} query step");
                    assert!(spec.input_slot < 16, "{shape} input slot");
                    assert!(
                        spec.recurrent_slot < spec.history_step,
                        "{shape} recurrence"
                    );
                    assert!(spec.history_step < 7, "{shape} history slot");
                }
                assert!(l.depths.iter().all(|depth| (*depth as usize) < NODES));
            }
        }
    }

    #[test]
    fn tree_menu_gate_covers_every_emittable_topology() {
        let gated = gate_topologies();
        for (parents, depths) in &gated {
            assert_eq!(parents.len(), NODES);
            assert_eq!(depths.len(), NODES);
            assert_eq!((parents[0], depths[0]), (-1, 0));
            for row in 1..NODES {
                assert!((parents[row] as usize) < row, "parent order {parents:?}");
                assert_eq!(depths[row], depths[parents[row] as usize] + 1);
            }
            // The gate is for the tree verifier; linear rounds do not use it.
            assert!(parents
                .iter()
                .enumerate()
                .any(|(row, parent)| *parent != row as i32 - 1));
        }
        for shape in Shape::ALL {
            for steps in alt_step_sets(shape.alts()) {
                let l = layout(shape, &steps).unwrap();
                let linear = l
                    .parents
                    .iter()
                    .enumerate()
                    .all(|(row, parent)| *parent == row as i32 - 1);
                assert_eq!(
                    linear,
                    shape == Shape::Lin7,
                    "{shape} linearity with {steps:?}"
                );
                assert!(
                    linear || gated.iter().any(|(p, d)| *p == l.parents && *d == l.depths),
                    "{shape} {steps:?} is emittable but not gated"
                );
            }
        }
        // Legacy may fork at step three, which the menu itself never does.
        assert!(gated
            .iter()
            .any(|(p, _)| *p == vec![-1, 0, 1, 2, 3, 3, 5, 6]));
        // Enumerated, so the gate cost is a known constant.
        assert_eq!(gated.len(), 17, "emittable topologies: {gated:?}");
    }

    #[test]
    fn tree_menu_finalize_holds_the_round_loop_invariants_over_a_margin_grid() {
        let calib = Calibration::default();
        let grid = [
            0.0f32,
            0.1,
            0.5,
            1.0,
            2.0,
            4.0,
            10.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let anchor = 5u32;
        let gated = gate_topologies();
        let mut seen_shapes = 0;
        for shape in Shape::ALL {
            seen_shapes += 1;
            for steps in alt_step_sets(shape.alts()) {
                let l = layout(shape, &steps).unwrap();
                for (i, &m0) in grid.iter().enumerate() {
                    for (j, &m1) in grid.iter().enumerate() {
                        let margins = [m0, m1, grid[(i + j) % grid.len()], grid[j]];
                        let top = all_tops(margins);
                        let gpu: Vec<u32> = (0..8).map(|f| 100 + f as u32).collect();
                        let out = match finalize(&l, &top, &gpu, anchor, calib) {
                            Ok(out) => out,
                            Err(_) => {
                                // The only refusal is a shape that needs a
                                // rank-two child this round cannot supply.
                                assert!(
                                    steps.iter().any(|s| !top[*s].runner_up_valid),
                                    "{shape} {steps:?} refused with margins {margins:?}"
                                );
                                continue;
                            }
                        };
                        // The three checks the MTP12 round loop performs.
                        assert_eq!(out.tokens[0], anchor);
                        assert!(crate::gemma4_runtime::mtp12_tree_policy::validate(
                            &out.tokens,
                            &out.parents,
                            &out.depths,
                            VOCAB,
                        ));
                        assert_eq!(
                            out.linear,
                            out.branch_primary_step.is_none(),
                            "{shape} {steps:?} margins {margins:?}"
                        );
                        assert!(out.node_p.iter().all(|p| p.is_finite() && *p <= 1.0));
                        assert_eq!(out.node_p[0], 1.0);
                        assert_eq!(out.fork_forwards, steps);
                        assert!(out.branch_primary_step.is_none_or(|s| s < PRIMARY));
                        assert!(
                            out.linear
                                || gated
                                    .iter()
                                    .any(|(p, d)| *p == out.parents && *d == out.depths)
                        );
                    }
                }
            }
        }
        assert_eq!(seen_shapes, Shape::ALL.len());
    }

    #[test]
    fn tree_menu_finalize_survives_adversarial_top_two_answers() {
        let calib = Calibration::default();
        // The exact degenerate answers the top-2 GPU test enumerates.
        let adversarial = [
            ForwardTop::from_pair([f32::NEG_INFINITY, f32::NEG_INFINITY], [0, u32::MAX], VOCAB),
            ForwardTop::from_pair([f32::NAN, f32::NAN], [0, u32::MAX], VOCAB),
            ForwardTop::from_pair([f32::INFINITY, f32::INFINITY], [3, 4], VOCAB),
            ForwardTop::from_pair([f32::NEG_INFINITY, f32::NEG_INFINITY], [0, 7], VOCAB),
            ForwardTop::from_pair([1.0, 1.0], [11, 11], VOCAB),
            ForwardTop::from_pair([1.0, 0.0], [VOCAB as u32, 3], VOCAB),
            ForwardTop::from_pair([1.0, 0.0], [3, VOCAB as u32], VOCAB),
        ];
        for bad in adversarial {
            let top = all_tops([1.0, 1.0, 1.0, 1.0]);
            for slot in 0..PRIMARY {
                let mut round = top.clone();
                round[slot] = bad;
                let primary: [ForwardTop; PRIMARY] = std::array::from_fn(|i| round[i]);
                let menu = Menu::new(&primary, calib);
                assert!(menu.reach.iter().all(|p| p.is_finite()));
                assert!(menu.alts.iter().all(|(_, value)| value.is_finite()));
                assert!(menu
                    .alts
                    .iter()
                    .all(|(step, _)| *step != slot || slot >= ALT_STEPS));
                let shape = choose(Policy::Dyn, &menu, DEFAULT_LAMBDA);
                let steps = menu.alt_steps(shape).unwrap();
                let l = layout(shape, &steps).unwrap();
                let gpu: Vec<u32> = (0..8).map(|f| 100 + f as u32).collect();
                let out = finalize(&l, &round, &gpu, 5, calib).unwrap();
                assert!(out.node_p.iter().all(|p| p.is_finite()));
                assert!(crate::gemma4_runtime::mtp12_tree_policy::validate(
                    &out.tokens,
                    &out.parents,
                    &out.depths,
                    VOCAB,
                ));
                assert_eq!(out.linear, out.branch_primary_step.is_none());
                // A shape that needs a sibling never keeps an invalid one.
                for row in 0..NODES {
                    if let NodeSource::RunnerUp(f) = l.sources[row] {
                        assert!(round[f].runner_up_valid);
                    }
                }
            }
        }
        // No eligible sibling at all: every policy degrades to the linear lane.
        let none = ForwardTop::from_pair([1.0, 1.0], [11, 11], VOCAB);
        let primary = [none; PRIMARY];
        let menu = Menu::new(&primary, calib);
        assert!(menu.alts.is_empty());
        assert_eq!(choose(Policy::Dyn, &menu, DEFAULT_LAMBDA), Shape::Lin7);
        for shape in Shape::ALL {
            let chosen = choose(Policy::Fixed(shape), &menu, DEFAULT_LAMBDA);
            assert_eq!(
                chosen,
                if shape == Shape::Lin7 {
                    shape
                } else {
                    Shape::Lin7
                }
            );
        }
        // A finalize asked for an invalid sibling refuses instead of emitting
        // an out-of-vocabulary or duplicate-sibling tree.
        let l = layout(Shape::P6A1, &[0]).unwrap();
        let mut round = all_tops([1.0, 1.0, 1.0, 1.0]);
        round[0] = none;
        let gpu: Vec<u32> = (0..8).map(|f| 100 + f as u32).collect();
        assert!(finalize(&l, &round, &gpu, 5, calib).is_err());
        assert!(finalize(&l, &round[..2], &gpu, 5, calib).is_err());
    }

    #[test]
    fn tree_menu_choice_follows_the_modeled_value_minus_forward_cost() {
        let calib = Calibration::default();
        // Confident round from the trace: the fourth margin is the only small
        // one, so the sibling is cheap and the chain is worth extending.
        let confident = Menu::new(&tops([12.7, 27.8, 21.4, 1.19]), calib);
        assert_eq!(confident.alts.len(), ALT_STEPS);
        let chosen = choose(Policy::Dyn, &confident, DEFAULT_LAMBDA);
        assert!(
            matches!(chosen, Shape::P6A1 | Shape::P5A1C1 | Shape::Lin7),
            "confident round chose {chosen}"
        );
        assert!(confident.value(Shape::P6A1).unwrap() > confident.value(Shape::P4A3).unwrap());
        // Unconfident early steps: siblings carry most of the mass, and the
        // shapes that buy them with fewer forwards win.
        let shaky = Menu::new(&tops([1.11, 0.52, 5.04, 8.86]), calib);
        let chosen = choose(Policy::Dyn, &shaky, DEFAULT_LAMBDA);
        assert!(
            matches!(chosen, Shape::P4A3 | Shape::P4A2C1 | Shape::P5A2),
            "shaky round chose {chosen}"
        );
        assert!(shaky.value(Shape::P4A3).unwrap() > shaky.value(Shape::Lin7).unwrap());
        // lambda is a real knob in the direction the round cost says it is.
        let greedy = choose(Policy::Dyn, &shaky, 0.0);
        let thrifty = choose(Policy::Dyn, &shaky, 1.0);
        assert!(greedy.forwards() >= thrifty.forwards());
        assert_eq!(thrifty, Shape::P4A3);
        // Fixed always gets its shape when the round can build it.
        for shape in Shape::ALL {
            assert_eq!(choose(Policy::Fixed(shape), &shaky, DEFAULT_LAMBDA), shape);
        }
        assert_eq!(
            choose(Policy::Legacy, &shaky, DEFAULT_LAMBDA),
            Shape::P4A1C2
        );
        // The value of a shape never depends on the common four-node prefix.
        for shape in Shape::ALL {
            assert!(shaky.value(shape).unwrap().is_finite());
        }
    }

    #[test]
    fn tree_menu_alt_steps_are_the_best_valued_and_ascending() {
        let calib = Calibration::default();
        // Step 1 has the smallest margin, so its sibling is the most valuable
        // even though step 0 is reached with probability one.
        let menu = Menu::new(&tops([6.0, 0.2, 3.0, 1.0]), calib);
        assert_eq!(menu.alts[0].0, 1);
        assert_eq!(menu.alt_steps(Shape::P6A1).unwrap(), vec![1]);
        let two = menu.alt_steps(Shape::P5A2).unwrap();
        assert_eq!(two.len(), 2);
        assert!(two[0] < two[1]);
        assert_eq!(menu.alt_steps(Shape::P4A3).unwrap(), vec![0, 1, 2]);
        assert_eq!(menu.alt_steps(Shape::Lin7).unwrap(), Vec::<usize>::new());
        // Only the first three forwards may fork, by construction.
        let mut pairs = tops([9.0, 9.0, 9.0, 0.0]);
        pairs[3] = ForwardTop::from_pair([0.0, 0.0], [7, 9], VOCAB);
        let menu = Menu::new(&pairs, calib);
        assert!(menu.alts.iter().all(|(step, _)| *step < ALT_STEPS));
        assert!(menu.value(Shape::P4A3).is_some());
        // Two eligible siblings cannot build the three-sibling shape.
        let mut pairs = tops([1.0, 1.0, 1.0, 1.0]);
        pairs[2] = ForwardTop::from_pair([1.0, 1.0], [7, 7], VOCAB);
        let menu = Menu::new(&pairs, calib);
        assert_eq!(menu.alts.len(), 2);
        assert!(menu.value(Shape::P4A3).is_none());
        assert!(menu.value(Shape::P5A2).is_some());
        assert_eq!(choose(Policy::Fixed(Shape::P4A3), &menu, 0.0), Shape::Lin7);
    }
}
