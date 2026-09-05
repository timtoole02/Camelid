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
    for base in [529, 1023, 1024, 1025] {
        prepare_prefix(&runtime, prompt, &prompt_ids, base);
        // Also check prefix guard rows before/after every tentative comparison.
        let guard_positions = [0, base.saturating_sub(1024), base - 1];
        let guard = kv_rows(&runtime, &guard_positions);
        let mut shapes: Vec<(Vec<i32>, Vec<u32>)> = (0..4)
            .map(|step| {
                (
                    vec![-1, 0, 1, 2, 3, step as i32, 5, 6],
                    vec![0, 1, 2, 3, 4, step + 1, step + 2, step + 3],
                )
            })
            .collect();
        // Interleave independent branches in physical storage.
        shapes.push((vec![-1, 0, 0, 1, 2, 3, 4, 5], vec![0, 1, 1, 2, 2, 3, 3, 4]));
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
        eprintln!("[tree-model] base={base}: all five shapes, sibling isolation and compacted continuation EXACT");
    }
    eprintln!("[tree-model] qualified {node_checks} node comparisons, 4 compacted K1 continuations in {:?}", started.elapsed());
}
