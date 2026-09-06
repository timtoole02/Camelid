//! Direct-runtime TREE=0 versus TREE=1 qualification with padded-tail=1 on both.
//! Public generation performs a fresh reset/prefill on every call; this helper
//! also checks the public logical cursor before and after every generation.
//! At a budget boundary, one mode can commit the last output while the other
//! leaves it as the free bonus. Their raw cursors may then differ by one; each
//! cursor must still equal prompt length plus its own committed trace. Public
//! termination is length at the cap, otherwise stop; extra EOS lookahead at the
//! cap is diagnostic metadata, not a different user-visible termination.
//!
//! Invoke separate processes with the chosen assistant position/shortlist env:
//! - MODEL ASSISTANT PROMPT 32 4,5,6,7,8,9,16,24 0 1
//! - MODEL ASSISTANT PROMPT 8 5,6,7,8 0 0
//! - MODEL ASSISTANT PROMPT 7 5,6,7 0 0
//! - MODEL ASSISTANT STOP_PROMPT 32 24 1 0
//!
//! TREE and padded-tail selectors are read per generation. Assistant position,
//! shortlist and several Metal selectors are load-time or OnceLock values: do
//! not toggle those within one process. Only TREE changes between each pair.
//! This is a correctness helper; its repeated-prefill timings are not benchmarks.
#[cfg(target_os = "macos")]
use camelid::{
    gemma4_runtime::{Gemma4GpuRuntime, Gemma4Mtp12MetalGeneration},
    metal::Gemma4Mtp12AssistantMetal,
};
#[cfg(target_os = "macos")]
use std::{collections::BTreeMap, error::Error, fs, path::Path};

#[cfg(target_os = "macos")]
fn bit(value: &str) -> Result<bool, Box<dyn Error>> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err("REQUIRE_STOP and REQUIRE_BRANCH must be exactly 0 or 1".into()),
    }
}

#[cfg(target_os = "macos")]
fn check_generation(
    generation: &Gemma4Mtp12MetalGeneration,
    prompt_tokens: usize,
    available: usize,
    budget: usize,
    cursor: usize,
    tree_enabled: bool,
) {
    let stats = &generation.stats;
    let trace = &stats.trace;
    assert_eq!(generation.prompt_token_count, prompt_tokens);
    assert!(generation.token_ids.len() <= budget);
    assert_eq!(stats.configured_verify_width, 8);
    assert!(stats.width_schedule.w8_padded_tail_active);
    assert_eq!(
        stats
            .width_schedule
            .w8_padded_tail_selector_value
            .as_deref(),
        Some("1")
    );
    assert_eq!(stats.rounds, trace.len() as u64);
    assert_eq!(
        stats.target_verify_rows,
        trace
            .iter()
            .map(|r| r.physical_verify_width as u64)
            .sum::<u64>()
    );
    assert_eq!(
        stats.committed_input_rows,
        trace
            .iter()
            .map(|r| r.committed_input_rows as u64)
            .sum::<u64>()
    );
    assert_eq!(
        stats.accepted_drafts,
        trace.iter().map(|r| r.accepted_drafts as u64).sum::<u64>()
    );
    assert_eq!(
        stats.drafted,
        trace.iter().map(|r| r.drafts.len() as u64).sum::<u64>()
    );
    assert_eq!(stats.emitted_tokens, generation.token_ids.len() as u64);
    assert_eq!(
        stats.budget_tail_rounds,
        trace.iter().filter(|r| r.budget_truncated).count() as u64
    );
    assert_eq!(
        stats.tree_proposal_rounds,
        trace.iter().filter(|r| r.tree.is_some()).count() as u64
    );
    assert_eq!(
        stats.tree_branch_rounds,
        trace
            .iter()
            .filter(|r| r
                .tree
                .as_ref()
                .is_some_and(|t| t.branch_primary_step.is_some()))
            .count() as u64
    );
    assert_eq!(
        stats.tree_compaction_us,
        trace
            .iter()
            .filter_map(|r| r.tree.as_ref())
            .map(|t| t.compaction_us)
            .sum::<u128>()
    );
    if !tree_enabled {
        assert_eq!(stats.tree_proposal_rounds, 0);
        assert_eq!(stats.tree_branch_rounds, 0);
        assert_eq!(stats.tree_compaction_us, 0);
    }
    let mut expected_cursor = prompt_tokens;
    let mut traced_output = Vec::new();
    for (index, round) in trace.iter().enumerate() {
        assert_eq!(
            round.position, expected_cursor,
            "cursor discontinuity in round {index}"
        );
        assert_eq!(round.configured_verify_width, 8);
        assert_eq!(round.verifier_width, round.physical_verify_width);
        assert!(matches!(round.physical_verify_width, 2 | 4 | 8));
        assert!(round.position + round.physical_verify_width <= prompt_tokens + available);
        assert!(
            round.committed_input_rows > 0
                && round.committed_input_rows <= round.logical_verify_width
        );
        assert_eq!(round.accepted_drafts + 1, round.committed_input_rows);
        assert_eq!(round.drafts.len() + 1, round.logical_verify_width);
        assert_eq!(round.target_greedy_ids.len(), round.physical_verify_width);
        assert_eq!(
            round.padding_candidate_ids.len(),
            round.physical_verify_width - round.logical_verify_width
        );
        assert!(round
            .padding_candidate_ids
            .iter()
            .all(|&id| id == round.anchor_token));
        assert_eq!(round.committed_input_rows, round.emitted_token_ids.len());
        assert_eq!(round.assistant_ledger.draft_k as usize, round.drafts.len());
        let mut candidates = vec![round.anchor_token];
        candidates.extend_from_slice(&round.drafts);
        let path = if let Some(tree) = &round.tree {
            assert!(tree_enabled);
            assert_eq!(
                (round.logical_verify_width, round.physical_verify_width),
                (8, 8)
            );
            assert!(round.padding_candidate_ids.is_empty());
            assert_eq!((tree.parents.len(), tree.depths.len()), (8, 8));
            assert_eq!((tree.parents[0], tree.depths[0]), (-1, 0));
            for row in 1..8 {
                let parent =
                    usize::try_from(tree.parents[row]).expect("non-root parent is nonnegative");
                assert!(parent < row);
                assert_eq!(tree.depths[row], tree.depths[parent] + 1);
            }
            // Legacy is fixed at six forwards on a fork and seven on the
            // linear fallback; the menu policies pick four to seven, and only
            // the linear shape ever costs seven.
            if tree.policy == "legacy" {
                assert_eq!(
                    tree.assistant_steps,
                    if tree.branch_primary_step.is_some() {
                        6
                    } else {
                        7
                    }
                );
                assert_eq!(
                    tree.primary_rows,
                    (0..if tree.branch_primary_step.is_some() {
                        5
                    } else {
                        8
                    })
                        .collect::<Vec<_>>()
                );
            } else {
                assert!((4..=7).contains(&tree.assistant_steps));
                assert_eq!(
                    tree.assistant_steps == 7,
                    tree.branch_primary_step.is_none()
                );
                assert_eq!(tree.forward_margins.len(), tree.assistant_steps);
                assert_eq!(tree.runner_up_ids.len(), tree.assistant_steps);
                assert_eq!(tree.node_p.len(), 8);
            }
            // The ordinary chain is always a contiguous physical prefix.
            assert_eq!(
                tree.primary_rows,
                (0..tree.primary_rows.len()).collect::<Vec<_>>()
            );
            assert!(tree.branch_primary_step.is_none_or(|step| step < 4));
            assert_eq!(
                tree.fork_forwards.first().copied(),
                tree.branch_primary_step
            );
            assert_eq!(tree.committed_path.first(), Some(&0));
            for edge in tree.committed_path.windows(2) {
                assert_eq!(tree.parents[edge[1]], edge[0] as i32);
            }
            tree.committed_path.clone()
        } else {
            // Every full W8 candidate must exercise the opt-in proposer; its
            // no-margin fallback still carries a linear tree receipt.
            assert!(
                !tree_enabled
                    || round.logical_verify_width != 8
                    || round.physical_verify_width != 8
            );
            (0..round.committed_input_rows).collect::<Vec<_>>()
        };
        assert_eq!(path.len(), round.committed_input_rows);
        assert!(path.iter().all(|&row| row < round.logical_verify_width));
        let emitted = path.iter().map(|&row| candidates[row]).collect::<Vec<_>>();
        assert_eq!(round.emitted_token_ids, emitted);
        for edge in path.windows(2) {
            assert_eq!(
                round.target_greedy_ids[edge[0]], candidates[edge[1]],
                "accepted edge lacks target agreement"
            );
        }
        let bonus = round.target_greedy_ids[*path.last().unwrap()];
        if let Some(stop) = round.stop_token {
            assert_eq!(Some(stop), stats.terminal_stop_token);
            assert_eq!(stop, bonus);
            assert_eq!(round.next_anchor_token, None);
            assert_eq!(index + 1, trace.len());
            assert!(!generation.token_ids.contains(&stop));
        } else {
            assert_eq!(round.next_anchor_token, Some(bonus));
        }
        traced_output.extend_from_slice(&round.emitted_token_ids);
        expected_cursor += round.committed_input_rows;
    }
    assert_eq!(
        cursor, expected_cursor,
        "actual cursor disagrees with committed trace"
    );
    assert_eq!(cursor, prompt_tokens + stats.committed_input_rows as usize);
    assert!(cursor <= prompt_tokens + available);
    assert!(generation.token_ids.starts_with(&traced_output));
    let free_bonus = generation.token_ids.len() - traced_output.len();
    assert!(free_bonus <= 1);
    if free_bonus == 1 {
        assert!(stats.terminal_stop_token.is_none());
        if let Some(last) = trace.last() {
            assert_eq!(last.next_anchor_token, generation.token_ids.last().copied());
        } else {
            // Budget one returns the already-proved prefill token without a
            // decode round or any generated-token KV commit.
            assert_eq!(budget, 1);
            assert_eq!(generation.token_ids.len(), 1);
            assert_eq!(stats.rounds, 0);
            assert_eq!(stats.committed_input_rows, 0);
            assert_eq!(cursor, prompt_tokens);
        }
    }
    if stats.terminal_stop_token.is_none() {
        assert_eq!(
            generation.token_ids.len(),
            budget,
            "nonterminal run ended before its output budget"
        );
    }
    if let Some(first) = trace.first() {
        if available >= 8 && (5..=8).contains(&budget) {
            assert_eq!(
                (first.logical_verify_width, first.physical_verify_width),
                (budget - 1, 8)
            );
        }
        if available < 8 || budget <= 4 {
            assert!(trace.iter().all(|r| r.padding_candidate_ids.is_empty()));
        }
    }
    if budget <= 8 || available < 8 {
        assert_eq!(
            stats.tree_proposal_rounds, 0,
            "short/capacity tail entered tree drafting"
        );
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 8 {
        return Err("usage: gemma4_mtp12_validate_tree_tail MODEL ASSISTANT PROMPT_FILE AVAILABLE_POSITIONS BUDGETS_CSV REQUIRE_STOP_0_OR_1 REQUIRE_BRANCH_0_OR_1".into());
    }
    let prompt = fs::read_to_string(&args[3])?;
    let available: usize = args[4].parse()?;
    let budgets = args[5]
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    let require_stop = bit(&args[6])?;
    let require_branch = bit(&args[7])?;
    assert!(!budgets.is_empty() && budgets.iter().all(|&b| b > 0 && b <= available));
    if require_branch {
        assert!(
            available >= 8 && budgets.iter().any(|&b| b >= 9),
            "branch coverage requires at least one full-W8 budget>=9"
        );
    }
    for key in [
        "CAMELID_MTP12_DUMP_DRAFT_QUERIES",
        "CAMELID_MTP12_DUMP_FINAL_KV",
    ] {
        if std::env::var_os(key).is_some_and(|v| !v.is_empty()) {
            return Err(format!("disable diagnostic {key} for this paired qualification").into());
        }
    }
    std::env::set_var("CAMELID_GEMMA4_MTP12_W8_PADDED_TAIL", "1");
    std::env::set_var("CAMELID_GEMMA4_MTP12_TREE_W8", "0");
    let selectors = std::env::vars()
        .filter(|(k, _)| k.starts_with("CAMELID_GEMMA4_") || k.starts_with("CAMELID_MTP12_"))
        .collect::<BTreeMap<_, _>>();
    let metadata = camelid::gguf::read_metadata(&args[1])?;
    let tokenizer = camelid::tokenizer::Tokenizer::from_gguf(&metadata)?;
    let prompt_tokens = tokenizer.encode(&prompt, true, true)?.len();
    let capacity = prompt_tokens
        .checked_add(available)
        .ok_or("capacity overflow")?;
    let runtime = Gemma4GpuRuntime::load(Path::new(&args[1]), capacity)?;
    let mut assistant = Gemma4Mtp12AssistantMetal::load(Path::new(&args[2]))?;
    let mut receipts = Vec::new();
    let mut saw_stop = false;
    let mut branch_budgets = Vec::new();
    for budget in budgets {
        runtime.reset_dense_verifier_sequence()?;
        assert_eq!(runtime.dense_verifier_logical_len()?, 0);
        std::env::set_var("CAMELID_GEMMA4_MTP12_TREE_W8", "0");
        let control =
            runtime.generate_greedy_mtp12_ordered_q4(&mut assistant, &prompt, budget, 8)?;
        let control_cursor = runtime.dense_verifier_logical_len()?;
        check_generation(
            &control,
            prompt_tokens,
            available,
            budget,
            control_cursor,
            false,
        );
        runtime.reset_dense_verifier_sequence()?;
        assert_eq!(runtime.dense_verifier_logical_len()?, 0);
        std::env::set_var("CAMELID_GEMMA4_MTP12_TREE_W8", "1");
        let candidate =
            runtime.generate_greedy_mtp12_ordered_q4(&mut assistant, &prompt, budget, 8)?;
        let candidate_cursor = runtime.dense_verifier_logical_len()?;
        check_generation(
            &candidate,
            prompt_tokens,
            available,
            budget,
            candidate_cursor,
            true,
        );
        assert_eq!(
            candidate.token_ids, control.token_ids,
            "TREE0/1 token parity at budget {budget}"
        );
        assert_eq!(
            candidate.text, control.text,
            "TREE0/1 text parity at budget {budget}"
        );
        let control_termination = if control.token_ids.len() == budget {
            "length"
        } else {
            "stop"
        };
        let candidate_termination = if candidate.token_ids.len() == budget {
            "length"
        } else {
            "stop"
        };
        assert_eq!(
            candidate_termination, control_termination,
            "TREE0/1 public termination at budget {budget}"
        );
        if candidate.token_ids.len() < budget {
            assert!(candidate.stats.terminal_stop_token.is_some());
            assert!(control.stats.terminal_stop_token.is_some());
            assert_eq!(
                candidate.stats.terminal_stop_token, control.stats.terminal_stop_token,
                "TREE0/1 natural stop token at budget {budget}"
            );
        }
        let control_free_bonus =
            control.token_ids.len() - control.stats.committed_input_rows as usize;
        let candidate_free_bonus =
            candidate.token_ids.len() - candidate.stats.committed_input_rows as usize;
        assert!(control_free_bonus <= 1 && candidate_free_bonus <= 1);
        assert!(candidate_cursor.abs_diff(control_cursor) <= 1);
        assert_eq!(
            candidate_cursor + candidate_free_bonus,
            control_cursor + control_free_bonus,
            "TREE0/1 cursor difference must be exactly the unforwarded final bonus"
        );
        if candidate_cursor != control_cursor {
            assert_ne!(candidate_free_bonus, control_free_bonus);
            assert_eq!(candidate_termination, "length");
        }
        let raw_cursor_equal = candidate_cursor == control_cursor;
        let stop_metadata_equal =
            candidate.stats.terminal_stop_token == control.stats.terminal_stop_token;
        saw_stop |= candidate_termination == "stop";
        if candidate.stats.tree_branch_rounds > 0 {
            assert!(budget >= 9);
            branch_budgets.push(budget);
        }
        eprintln!("TREE tail pair passed: budget={budget} available={available} tokens={} stop={:?} branches={} cursor={candidate_cursor}",
            candidate.token_ids.len(), candidate.stats.terminal_stop_token, candidate.stats.tree_branch_rounds);
        receipts.push(
            serde_json::json!({"budget":budget,"tree_off_on_ids_exact":true,"text_exact":true,
            "termination_semantics_exact":true,"cursor_accounting_valid":true,
            "control_termination":control_termination,"candidate_termination":candidate_termination,
            "raw_cursor_equal":raw_cursor_equal,"stop_metadata_equal":stop_metadata_equal,
            "control_cursor":control_cursor,"candidate_cursor":candidate_cursor,
            "control_free_bonus":control_free_bonus,"candidate_free_bonus":candidate_free_bonus,
            "control":control,"candidate":candidate}),
        );
    }
    let coverage_ok =
        (!require_stop || saw_stop) && (!require_branch || !branch_budgets.is_empty());
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"camelid.w8_tree_tail_direct_validation.v2","prompt":prompt,"prompt_tokens":prompt_tokens,
            "available_positions":available,"max_positions":capacity,"selectors_before_pairs":selectors,
            "controls":{"tree_off":0,"tree_on":1,"padded_tail_both":1,"fresh_reset_and_prefill_each":true},
            "require_stop":require_stop,"require_branch":require_branch,"saw_stop":saw_stop,"branch_budgets":branch_budgets,
            "coverage_ok":coverage_ok,"receipts":receipts,
        }))?
    );
    if !coverage_ok {
        return Err("missing required coverage: choose a prompt that reaches a natural stop and/or a real tree branch, as requested by the CLI flags".into());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This qualification example requires macOS and Metal.");
    std::process::exit(1);
}
