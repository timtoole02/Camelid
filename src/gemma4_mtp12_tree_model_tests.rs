//! Ignored full-model qualification for target trees. This is teacher-forced QA,
//! not a generation route or an acceptance/performance claim. Run alone on the
//! model rig with its established exact release profile and V2 selectors.
use super::*;

fn exact(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    assert!(
        expected.iter().all(|x| x.is_finite()),
        "{label}: non-finite reference"
    );
    assert!(
        actual.iter().all(|x| x.is_finite()),
        "{label}: non-finite tree output"
    );
    if let Some((i, (a, b))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
    {
        panic!(
            "{label}: index={i} expected={a:?}/{:08x} actual={b:?}/{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
}

fn path_to(parents: &[i32], node: usize) -> Vec<usize> {
    let mut path = vec![node];
    while *path.last().unwrap() != 0 {
        path.push(parents[*path.last().unwrap()] as usize);
    }
    path.reverse();
    path
}

/// Capture only requested physical rows, in position/layer/K-or-V/head/dim
/// order. No future row is ever used to construct a reference prefix.
fn kv_rows(runtime: &Gemma4GpuRuntime, positions: &[usize]) -> Vec<Vec<u32>> {
    runtime
        .model
        .with_kv_device_views(&(0..48).collect::<Vec<_>>(), |views| {
            let mut rows = vec![Vec::new(); positions.len()];
            for view in views {
                for buffer in [view.key, view.value] {
                    let data = unsafe {
                        std::slice::from_raw_parts(
                            buffer
                                .contents()
                                .cast::<u8>()
                                .add(view.byte_offset as usize)
                                .cast::<u32>(),
                            view.byte_len as usize / 4,
                        )
                    };
                    for (index, &position) in positions.iter().enumerate() {
                        assert!(position < view.max_positions);
                        for head in 0..view.kv_heads {
                            let start = (head * view.max_positions + position) * view.head_dim;
                            rows[index].extend_from_slice(&data[start..start + view.head_dim]);
                        }
                    }
                }
            }
            rows
        })
        .expect("all 48 target layers own KV")
}

fn poison_tentative_rows(runtime: &Gemma4GpuRuntime, base: usize) {
    runtime
        .model
        .with_kv_device_views(&(0..48).collect::<Vec<_>>(), |views| {
            for view in views {
                for buffer in [view.key, view.value] {
                    let data = unsafe {
                        std::slice::from_raw_parts_mut(
                            buffer
                                .contents()
                                .cast::<u8>()
                                .add(view.byte_offset as usize)
                                .cast::<u32>(),
                            view.byte_len as usize / 4,
                        )
                    };
                    for head in 0..view.kv_heads {
                        for position in base..base + 11 {
                            let start = (head * view.max_positions + position) * view.head_dim;
                            data[start..start + view.head_dim].fill(0x7fc12345);
                        }
                    }
                }
            }
        })
        .unwrap();
}

/// Rebuild through the public reset/prefill protocol. Boundary contexts append
/// repeated *prompt inputs*, never generated/golden outputs or future KV.
fn prepare_prefix(runtime: &Gemma4GpuRuntime, prompt: &str, prompt_ids: &[u32], base: usize) {
    let prefill = runtime.prefill_ordered_q4(prompt).unwrap();
    assert_eq!(prefill.prompt_token_count, 529);
    let mut position = prompt_ids.len();
    while position < base {
        let width = gemma4_ordered_q4_prefill_chunk_width(base - position).unwrap();
        let tokens: Vec<u32> = (position..position + width)
            .map(|p| prompt_ids[(p - prompt_ids.len()) % prompt_ids.len()])
            .collect();
        let (ticket, ()) = runtime
            .with_consecutive_hidden_ordered_q4(&tokens, position, |_, _| Ok(()))
            .unwrap();
        position = runtime.commit_verifier_prefix(ticket, width).unwrap();
    }
    assert_eq!(runtime.dense_verifier_logical_len().unwrap(), base);
}

fn logits(runtime: &Gemma4GpuRuntime, columns: usize) -> Vec<f32> {
    runtime
        .q6k_gpu_head
        .as_ref()
        .unwrap()
        .tree_test_last_spec50_logits(columns)
        .unwrap()
}

#[test]
#[ignore = "requires official 12B QAT target, native sidecar and frozen 529-token trace; run alone"]
fn target_tree_w8_model_paths_logits_kv_and_compaction_are_bit_exact() {
    let model_path =
        std::env::var("CAMELID_MTP12_TEST_MODEL").expect("explicit official target GGUF path");
    let trace_path = std::env::var("CAMELID_TREE_TEST_TRACE")
        .expect("explicit full-trace JSON with rendered 529-token prompt");
    let trace: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace_path).unwrap()).unwrap();
    assert_eq!(trace["prompt_tokens"], 529);
    for flag in [
        "token_ids_exact",
        "raw_text_exact",
        "visible_text_exact",
        "prompt_tokens_exact",
    ] {
        assert_eq!(trace["validation"][flag], true, "trace provenance {flag}");
    }
    let prompt = trace["rendered_prompt"].as_str().unwrap();
    let prompt_ids: Vec<u32> =
        serde_json::from_value(trace["rendered_prompt_token_ids"].clone()).unwrap();
    assert_eq!(prompt_ids.len(), 529);
    let runtime = Gemma4GpuRuntime::load(Path::new(&model_path), 2048).unwrap();
    runtime.admit_mtp12_target_identity().unwrap();
    assert_eq!(
        runtime.tokenizer.encode(prompt, true, true).unwrap(),
        prompt_ids
    );
    assert!(
        runtime.q6k_gpu_head.is_some(),
        "SPEC50 head must be enabled"
    );
    // Eight distinct ordinary in-vocabulary IDs chosen from prompt inputs.
    let mut tokens = Vec::new();
    for &token in &prompt_ids {
        if !tokens.contains(&token) {
            tokens.push(token);
        }
        if tokens.len() == 8 {
            break;
        }
    }
    assert_eq!(tokens.len(), 8);
    let started = std::time::Instant::now();
    let mut node_checks = 0;
    // Exactly the topologies the draft policy can emit, in the same order the
    // menu enumerates them: the four legacy fork steps first, then every
    // named shape the dyn/fixed selectors can produce. The runtime's finalize
    // is shape preserving, so this enumeration IS the emittable set.
    let mut shapes: Vec<(Vec<i32>, Vec<u32>)> = crate::gemma4_mtp12_tree_menu::gate_topologies();
    // Interleave independent branches in physical storage. No policy emits
    // this one; it stays as an extra generality proof for the verifier.
    shapes.push((vec![-1, 0, 0, 1, 2, 3, 4, 5], vec![0, 1, 1, 2, 2, 3, 3, 4]));
    // Two bases per build cycle. Run all four, which adds the sliding-window
    // boundary, once for the finally adopted policy before qualification:
    // CAMELID_GEMMA4_TREE_GATE_ALL_BASES=1.
    let bases: Vec<usize> =
        if std::env::var("CAMELID_GEMMA4_TREE_GATE_ALL_BASES").as_deref() == Ok("1") {
            vec![529, 1023, 1024, 1025]
        } else {
            vec![529, 1024]
        };
    eprintln!(
        "[tree-model] {} topologies x {} bases = {} node comparisons",
        shapes.len(),
        bases.len(),
        shapes.len() * bases.len() * 8
    );
    for base in bases {
        prepare_prefix(&runtime, prompt, &prompt_ids, base);
        // Also check prefix guard rows before/after every tentative comparison.
        let guard_positions = [0, base.saturating_sub(1024), base - 1];
        let guard = kv_rows(&runtime, &guard_positions);
        for (shape, (parents, depths)) in shapes.iter().enumerate() {
            poison_tentative_rows(&runtime, base);
            let tree = runtime
                .verify_tree_greedy(&tokens, parents, depths, base)
                .unwrap();
            let tree_logits = logits(&runtime, 8);
            let tree_kv = kv_rows(&runtime, &(base..base + 8).collect::<Vec<_>>());
            assert_eq!(runtime.rollback_verifier_batch(tree.ticket).unwrap(), base);
            for node in 0..8 {
                let path = path_to(parents, node);
                let mut linear_tokens: Vec<u32> = path.iter().map(|&row| tokens[row]).collect();
                let logical = linear_tokens.len();
                linear_tokens.resize(8, tokens[0]);
                // Rollback means every independent reference starts at the same
                // committed prefix. Causal padding is physically W8 in both lanes.
                poison_tentative_rows(&runtime, base);
                let linear = runtime
                    .verify_consecutive_greedy(&linear_tokens, base)
                    .unwrap();
                let linear_logits = logits(&runtime, 8);
                let label = format!("base={base} shape={shape} node={node} path={path:?}");
                exact(
                    &format!("raw hidden {label}"),
                    &linear.final_hidden[logical - 1],
                    &tree.final_hidden[node],
                );
                assert_eq!(
                    linear.greedy_ids[logical - 1],
                    tree.greedy_ids[node],
                    "argmax {label}"
                );
                exact(
                    &format!("all SPEC50 logits {label}"),
                    &linear_logits[(logical - 1) * runtime.vocab..logical * runtime.vocab],
                    &tree_logits[node * runtime.vocab..(node + 1) * runtime.vocab],
                );
                assert_eq!(
                    kv_rows(&runtime, &[base + logical - 1])[0],
                    tree_kv[node],
                    "all-layer KV {label}"
                );
                assert_eq!(
                    runtime.rollback_verifier_batch(linear.ticket).unwrap(),
                    base
                );
                node_checks += 1;
            }
            // Poison the sibling branch's token identity, not any saved oracle.
            // The primary root-through-depth4 path must remain bit-identical.
            if shape < 4 {
                let mut sibling_tokens = tokens.clone();
                for token in &mut sibling_tokens[5..8] {
                    *token = (*token + 7919) % runtime.vocab as u32;
                }
                let sibling = runtime
                    .verify_tree_greedy(&sibling_tokens, parents, depths, base)
                    .unwrap();
                let sibling_logits = logits(&runtime, 8);
                for node in 0..5 {
                    exact(
                        "sibling-independent hidden",
                        &tree.final_hidden[node],
                        &sibling.final_hidden[node],
                    );
                    exact(
                        "sibling-independent all logits",
                        &tree_logits[node * runtime.vocab..(node + 1) * runtime.vocab],
                        &sibling_logits[node * runtime.vocab..(node + 1) * runtime.vocab],
                    );
                }
                runtime.rollback_verifier_batch(sibling.ticket).unwrap();
            }
            assert_eq!(
                kv_rows(&runtime, &guard_positions),
                guard,
                "committed prefix guards base={base} shape={shape}"
            );
        }

        // Exercise actual public commits and a subsequent K1 target step on an
        // overlapping non-prefix path; rebuild the original prefix in between.
        let parents = [-1, 0, 1, 2, 3, 1, 5, 6];
        let depths = [0, 1, 2, 3, 4, 2, 3, 4];
        let path = [0usize, 1, 5, 6, 7];
        let mut linear_tokens: Vec<u32> = path.iter().map(|&row| tokens[row]).collect();
        linear_tokens.resize(8, tokens[0]);
        let linear = runtime
            .verify_consecutive_greedy(&linear_tokens, base)
            .unwrap();
        let linear_path_kv = kv_rows(&runtime, &(base..base + path.len()).collect::<Vec<_>>());
        let next_anchor = linear.greedy_ids[path.len() - 1];
        assert_eq!(
            runtime
                .commit_verifier_prefix(linear.ticket, path.len())
                .unwrap(),
            base + path.len()
        );
        let next_linear = runtime
            .verify_consecutive_greedy(&[next_anchor], base + path.len())
            .unwrap();
        let next_linear_logits = logits(&runtime, 1);
        let next_linear_kv = kv_rows(&runtime, &[base + path.len()]);
        runtime.rollback_verifier_batch(next_linear.ticket).unwrap();

        prepare_prefix(&runtime, prompt, &prompt_ids, base);
        let tree = runtime
            .verify_tree_greedy(&tokens, &parents, &depths, base)
            .unwrap();
        // A nonzero linear commit must fail without resolving the tree ticket.
        assert!(runtime
            .commit_verifier_prefix(tree.ticket, path.len())
            .is_err());
        assert_eq!(runtime.dense_verifier_logical_len().unwrap(), base);
        assert_eq!(
            runtime
                .commit_verifier_tree_path(tree.ticket, &path)
                .unwrap(),
            base + path.len()
        );
        assert_eq!(
            kv_rows(&runtime, &(base..base + path.len()).collect::<Vec<_>>()),
            linear_path_kv,
            "compacted all-layer KV base={base}"
        );
        assert_eq!(tree.greedy_ids[7], next_anchor);
        let next_tree = runtime
            .verify_consecutive_greedy(&[next_anchor], base + path.len())
            .unwrap();
        exact(
            "post-compaction K1 hidden",
            &next_linear.final_hidden[0],
            &next_tree.final_hidden[0],
        );
        assert_eq!(next_linear.greedy_ids, next_tree.greedy_ids);
        exact(
            "post-compaction K1 all logits",
            &next_linear_logits,
            &logits(&runtime, 1),
        );
        assert_eq!(
            kv_rows(&runtime, &[base + path.len()]),
            next_linear_kv,
            "post-compaction K1 all-layer KV"
        );
        runtime.rollback_verifier_batch(next_tree.ticket).unwrap();
        eprintln!(
            "[tree-model] base={base}: all {} topologies, sibling isolation and compacted continuation EXACT ({:?} elapsed)",
            shapes.len(),
            started.elapsed()
        );
    }
    eprintln!(
        "[tree-model] qualified {node_checks} node comparisons and one compacted K1 continuation per base in {:?}",
        started.elapsed()
    );
}

/// One verify call shape for the fused-glue A/B: a linear K-row batch or one
/// of the W8 tree topologies the gate above exercises.
enum GlueCase {
    Linear(usize),
    Tree { parents: Vec<i32>, depths: Vec<u32> },
}

/// Everything the target side is contracted on: final hidden bits, SPEC50
/// argmax ids, every SPEC50 logit and the 48-layer K/V bits of the rows the
/// call wrote. The batch is rolled back so every run starts from the same
/// committed prefix.
fn glue_observe(
    runtime: &Gemma4GpuRuntime,
    tokens: &[u32],
    base: usize,
    case: &GlueCase,
    mask: u32,
) -> (Vec<Vec<f32>>, Vec<u32>, Vec<f32>, Vec<Vec<u32>>) {
    poison_tentative_rows(runtime, base);
    let (batch, width) = match case {
        GlueCase::Linear(width) => (
            runtime
                .verify_consecutive_greedy_with_glue(&tokens[..*width], base, Some(mask))
                .unwrap(),
            *width,
        ),
        GlueCase::Tree { parents, depths } => (
            runtime
                .verify_tree_greedy_with_glue(tokens, parents, depths, base, Some(mask))
                .unwrap(),
            8,
        ),
    };
    let logits = logits(runtime, width);
    let kv = kv_rows(runtime, &(base..base + width).collect::<Vec<_>>());
    assert_eq!(runtime.rollback_verifier_batch(batch.ticket).unwrap(), base);
    (batch.final_hidden, batch.greedy_ids, logits, kv)
}

/// Fused-glue A/B against the legacy encode. The tree/linear gate above runs
/// both of its lanes through the same `verify_hidden_ordered_q4_plan`, so a
/// consistent last-ulp drift of a fused kernel would pass it; this is the
/// old-vs-new oracle. For bases 529/1023/1024/1025, linear K=1 and K=8 and
/// the five tree shapes, mask 0 is compared with every shipping combination
/// (127 = all seven fusions, 95 = cycle-1 + C4, 119 = cycle-1 + C6, 87 =
/// cycle-1, the C5-lite variant) and every single bit including C4 (8) and
/// C6 (32): identical final hidden bits, identical SPEC50 argmax ids,
/// identical SPEC50 logits and identical 48-layer K/V bits of every written
/// row. Runtime is a few minutes.
#[test]
#[ignore = "requires official 12B QAT target, native sidecar and frozen 529-token trace; run alone"]
fn target_verify_fused_glue_masks_are_bit_exact_against_legacy() {
    use crate::metal::{
        GEMMA4_FUSED_GLUE_ALL, GEMMA4_FUSED_GLUE_C1_STAGE_ONCE, GEMMA4_FUSED_GLUE_C2_NORM_QUANTIZE,
        GEMMA4_FUSED_GLUE_C3_GELU_QUANTIZE, GEMMA4_FUSED_GLUE_C4_RESIDUAL_NORM,
        GEMMA4_FUSED_GLUE_C5_HEAD_ROPE_SCATTER, GEMMA4_FUSED_GLUE_C5_LITE_HEAD_NORMS,
        GEMMA4_FUSED_GLUE_C6_QUANTIZE_STAGE, GEMMA4_FUSED_GLUE_C7_SEGMENT_MMA,
        GEMMA4_FUSED_GLUE_CYCLE1,
    };
    let model_path =
        std::env::var("CAMELID_MTP12_TEST_MODEL").expect("explicit official target GGUF path");
    let trace_path = std::env::var("CAMELID_TREE_TEST_TRACE")
        .expect("explicit full-trace JSON with rendered 529-token prompt");
    let trace: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace_path).unwrap()).unwrap();
    assert_eq!(trace["prompt_tokens"], 529);
    let prompt = trace["rendered_prompt"].as_str().unwrap();
    let prompt_ids: Vec<u32> =
        serde_json::from_value(trace["rendered_prompt_token_ids"].clone()).unwrap();
    assert_eq!(prompt_ids.len(), 529);
    let runtime = Gemma4GpuRuntime::load(Path::new(&model_path), 2048).unwrap();
    runtime.admit_mtp12_target_identity().unwrap();
    assert!(
        runtime.q6k_gpu_head.is_some(),
        "SPEC50 head must be enabled"
    );
    let mut tokens = Vec::new();
    for &token in &prompt_ids {
        if !tokens.contains(&token) {
            tokens.push(token);
        }
        if tokens.len() == 8 {
            break;
        }
    }
    assert_eq!(tokens.len(), 8);

    let masks: Vec<(&str, u32)> = vec![
        ("all", GEMMA4_FUSED_GLUE_ALL),
        ("cycle1", GEMMA4_FUSED_GLUE_CYCLE1),
        (
            "cycle1+c4",
            GEMMA4_FUSED_GLUE_CYCLE1 | GEMMA4_FUSED_GLUE_C4_RESIDUAL_NORM,
        ),
        (
            "cycle1+c6",
            GEMMA4_FUSED_GLUE_CYCLE1 | GEMMA4_FUSED_GLUE_C6_QUANTIZE_STAGE,
        ),
        (
            "all-c5-lite",
            GEMMA4_FUSED_GLUE_C1_STAGE_ONCE
                | GEMMA4_FUSED_GLUE_C2_NORM_QUANTIZE
                | GEMMA4_FUSED_GLUE_C3_GELU_QUANTIZE
                | GEMMA4_FUSED_GLUE_C4_RESIDUAL_NORM
                | GEMMA4_FUSED_GLUE_C6_QUANTIZE_STAGE
                | GEMMA4_FUSED_GLUE_C7_SEGMENT_MMA
                | GEMMA4_FUSED_GLUE_C5_LITE_HEAD_NORMS,
        ),
        ("c1", GEMMA4_FUSED_GLUE_C1_STAGE_ONCE),
        ("c2", GEMMA4_FUSED_GLUE_C2_NORM_QUANTIZE),
        ("c3", GEMMA4_FUSED_GLUE_C3_GELU_QUANTIZE),
        ("c4", GEMMA4_FUSED_GLUE_C4_RESIDUAL_NORM),
        ("c7", GEMMA4_FUSED_GLUE_C7_SEGMENT_MMA),
        ("c5", GEMMA4_FUSED_GLUE_C5_HEAD_ROPE_SCATTER),
        ("c6", GEMMA4_FUSED_GLUE_C6_QUANTIZE_STAGE),
        ("c5-lite", GEMMA4_FUSED_GLUE_C5_LITE_HEAD_NORMS),
    ];
    // The decimal masks the selector is documented with must be exactly the
    // ones this gate compares, so a receipt naming a mask names a tested one.
    assert_eq!(GEMMA4_FUSED_GLUE_CYCLE1, 87);
    assert_eq!(GEMMA4_FUSED_GLUE_ALL, 127);
    for &(label, expected) in &[
        ("all", 127u32),
        ("cycle1", 87),
        ("cycle1+c4", 95),
        ("cycle1+c6", 119),
        ("c4", 8),
        ("c6", 32),
    ] {
        let found = masks
            .iter()
            .find(|(name, _)| *name == label)
            .map(|(_, mask)| *mask);
        assert_eq!(found, Some(expected), "mask {label} must be {expected}");
    }
    let mut cases: Vec<(String, GlueCase)> = vec![
        ("linear K=1".to_string(), GlueCase::Linear(1)),
        ("linear K=8".to_string(), GlueCase::Linear(8)),
    ];
    for step in 0..4u32 {
        cases.push((
            format!("tree shape={step}"),
            GlueCase::Tree {
                parents: vec![-1, 0, 1, 2, 3, step as i32, 5, 6],
                depths: vec![0, 1, 2, 3, 4, step + 1, step + 2, step + 3],
            },
        ));
    }
    cases.push((
        "tree shape=4".to_string(),
        GlueCase::Tree {
            parents: vec![-1, 0, 0, 1, 2, 3, 4, 5],
            depths: vec![0, 1, 1, 2, 2, 3, 3, 4],
        },
    ));

    let started = std::time::Instant::now();
    let mut comparisons = 0;
    for base in [529, 1023, 1024, 1025] {
        prepare_prefix(&runtime, prompt, &prompt_ids, base);
        let guard_positions = [0, base.saturating_sub(1024), base - 1];
        let guard = kv_rows(&runtime, &guard_positions);
        for (case_label, case) in &cases {
            let (hidden, ids, logits_ref, kv) = glue_observe(&runtime, &tokens, base, case, 0);
            for &(mask_label, mask) in &masks {
                let (hidden_m, ids_m, logits_m, kv_m) =
                    glue_observe(&runtime, &tokens, base, case, mask);
                let label = format!("base={base} {case_label} mask={mask_label}({mask})");
                assert_eq!(hidden.len(), hidden_m.len(), "{label}: row count");
                for (row, (expected, actual)) in hidden.iter().zip(&hidden_m).enumerate() {
                    exact(&format!("final hidden {label} row={row}"), expected, actual);
                }
                assert_eq!(ids, ids_m, "SPEC50 argmax ids {label}");
                exact(&format!("all SPEC50 logits {label}"), &logits_ref, &logits_m);
                assert_eq!(kv, kv_m, "48-layer K/V bits {label}");
                comparisons += 1;
            }
            assert_eq!(
                kv_rows(&runtime, &guard_positions),
                guard,
                "committed prefix guards base={base} {case_label}"
            );
        }
        eprintln!(
            "[fused-glue-ab] base={base}: {} cases x {} masks EXACT against mask=0 ({:?} elapsed)",
            cases.len(),
            masks.len(),
            started.elapsed()
        );
    }
    eprintln!(
        "[fused-glue-ab] {comparisons} mask comparisons bit-exact against the legacy encode in {:?}",
        started.elapsed()
    );
}
