//! Direct-runtime qualification, because the CLI refuses budgets < W and serve
//! reserves 16 KV headroom rows before entering the generation loop.
//! Run only after an exclusive mini2 GPU slot is granted.
#[cfg(target_os = "macos")]
use camelid::{gemma4_runtime::Gemma4GpuRuntime, metal::Gemma4Mtp12AssistantMetal};
#[cfg(target_os = "macos")]
use std::{error::Error, fs, path::Path};

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 7 {
        return Err("usage: validate_tail MODEL ASSISTANT PROMPT_FILE AVAILABLE_POSITIONS BUDGETS_CSV REQUIRE_STOP_0_OR_1".into());
    }
    let prompt = fs::read_to_string(&args[3])?;
    let metadata = camelid::gguf::read_metadata(&args[1])?;
    let tokenizer = camelid::tokenizer::Tokenizer::from_gguf(&metadata)?;
    let prompt_tokens = tokenizer.encode(&prompt, true, true)?.len();
    let available: usize = args[4].parse()?;
    let budgets = args[5]
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    let require_stop = args[6] == "1";
    assert!(!budgets.is_empty());
    assert!(budgets.iter().all(|&b| b > 0 && b <= available));
    let runtime = Gemma4GpuRuntime::load(Path::new(&args[1]), prompt_tokens + available)?;
    let mut assistant = Gemma4Mtp12AssistantMetal::load(Path::new(&args[2]))?;
    let mut receipts = Vec::new();
    let mut saw_stop = false;
    for budget in budgets {
        std::env::set_var("CAMELID_GEMMA4_MTP12_W8_PADDED_TAIL", "0");
        let control =
            runtime.generate_greedy_mtp12_ordered_q4(&mut assistant, &prompt, budget, 8)?;
        std::env::set_var("CAMELID_GEMMA4_MTP12_W8_PADDED_TAIL", "1");
        let candidate =
            runtime.generate_greedy_mtp12_ordered_q4(&mut assistant, &prompt, budget, 8)?;
        assert_eq!(
            candidate.token_ids, control.token_ids,
            "token parity at budget {budget}"
        );
        assert_eq!(
            candidate.text, control.text,
            "text parity at budget {budget}"
        );
        assert_eq!(
            candidate.stats.terminal_stop_token,
            control.stats.terminal_stop_token
        );
        assert!(candidate.token_ids.len() <= budget);
        let trace = &candidate.stats.trace;
        assert_eq!(candidate.stats.rounds, trace.len() as u64);
        assert_eq!(
            candidate.stats.target_verify_rows,
            trace
                .iter()
                .map(|r| r.physical_verify_width as u64)
                .sum::<u64>()
        );
        assert_eq!(
            candidate.stats.committed_input_rows,
            trace
                .iter()
                .map(|r| r.committed_input_rows as u64)
                .sum::<u64>()
        );
        assert_eq!(
            candidate.stats.accepted_drafts,
            trace.iter().map(|r| r.accepted_drafts as u64).sum::<u64>()
        );
        assert_eq!(
            candidate.stats.drafted,
            trace.iter().map(|r| r.drafts.len() as u64).sum::<u64>()
        );
        assert_eq!(
            candidate.stats.emitted_tokens,
            candidate.token_ids.len() as u64
        );
        let traced_output = trace
            .iter()
            .flat_map(|r| r.emitted_token_ids.iter().copied())
            .collect::<Vec<_>>();
        assert!(candidate.token_ids.starts_with(&traced_output));
        assert!(candidate.token_ids.len() - traced_output.len() <= 1);
        if candidate.token_ids.len() != traced_output.len() {
            assert_eq!(
                trace.last().unwrap().next_anchor_token,
                candidate.token_ids.last().copied()
            );
        }
        for round in trace {
            assert!(round.position + round.physical_verify_width <= prompt_tokens + available);
            assert!(round.committed_input_rows <= round.logical_verify_width);
            assert_eq!(round.drafts.len() + 1, round.logical_verify_width);
            assert_eq!(
                round.padding_candidate_ids.len(),
                round.physical_verify_width - round.logical_verify_width
            );
            assert!(round
                .padding_candidate_ids
                .iter()
                .all(|&id| id == round.anchor_token));
            assert_eq!(round.committed_input_rows, round.emitted_token_ids.len());
        }
        if let Some(first) = trace.first() {
            if available >= 8 && (5..=8).contains(&budget) {
                assert_eq!(
                    (first.logical_verify_width, first.physical_verify_width),
                    (budget - 1, 8)
                );
            } else if available < 8 || budget <= 4 {
                assert!(trace.iter().all(|r| r.padding_candidate_ids.is_empty()));
                assert_eq!(
                    candidate.stats.target_verify_rows,
                    control.stats.target_verify_rows
                );
                assert_eq!(candidate.stats.rounds, control.stats.rounds);
            }
        }
        saw_stop |= candidate.stats.terminal_stop_token.is_some();
        receipts.push(serde_json::json!({
            "budget": budget, "token_ids_exact": true, "text_exact": true,
            "control": control, "candidate": candidate,
        }));
    }
    assert!(
        !require_stop || saw_stop,
        "stop case did not reach a natural terminal token; choose a shorter reply prompt"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "camelid.w8_tail_direct_validation.v1", "prompt": prompt,
            "prompt_tokens": prompt_tokens, "available_positions": available,
            "max_positions": prompt_tokens + available, "saw_stop": saw_stop,
            "receipts": receipts,
        }))?
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This qualification example requires macOS and Metal.");
    std::process::exit(1);
}
