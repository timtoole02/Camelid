//! Capture one ordinary generation's complete verifier trace for offline QA.
//! Golden output is read only after generation; it is never supplied as a draft.
//! Run under cam-lock after obtaining the exclusive mini2 slot.
#[cfg(target_os = "macos")]
use std::{collections::BTreeMap, error::Error, fs, path::Path};

#[cfg(target_os = "macos")]
fn visible_chat_text(text: &str) -> String {
    // Mirrors the serve lane's complete-response channel suppression. The public
    // chat renderer below is canonical; this helper only checks visible output.
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<|channel>") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "<|channel>".len()..];
        match after.find("<channel|>") {
            Some(end) => rest = &after[end + "<channel|>".len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn Error>> {
    use camelid::{
        api::{render_gemma4_chat_prompt, ChatMessage},
        gemma4_runtime::Gemma4GpuRuntime,
        metal::Gemma4Mtp12AssistantMetal,
    };
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 7 {
        return Err("usage: gemma4_mtp12_full_trace MODEL ASSISTANT REQUEST_JSON GOLDEN_RESPONSE_JSON OUTPUT_JSON EXPECTED_PROMPT_TOKENS".into());
    }
    let request: serde_json::Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let messages: Vec<ChatMessage> = serde_json::from_value(request["messages"].clone())?;
    if request
        .get("camelid_enable_thinking")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return Err("this QA capture requires the non-thinking template".into());
    }
    if request
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        != 0.0
    {
        return Err("this QA capture requires greedy temperature=0".into());
    }
    if messages
        .iter()
        .any(|m| !m.image_urls.is_empty() || !m.unsupported_content_parts.is_empty())
    {
        return Err("this QA capture accepts text-only messages".into());
    }
    let budget = request["max_tokens"]
        .as_u64()
        .ok_or("max_tokens is required")? as usize;
    let expected_prompt_tokens: usize = args[6].parse()?;
    let prompt = render_gemma4_chat_prompt(&messages, false);
    let metadata = camelid::gguf::read_metadata(&args[1])?;
    let tokenizer = camelid::tokenizer::Tokenizer::from_gguf(&metadata)?;
    let rendered_prompt_token_ids = tokenizer.encode(&prompt, true, true)?;
    let prompt_tokens = rendered_prompt_token_ids.len();
    assert_eq!(
        prompt_tokens, expected_prompt_tokens,
        "exact rendered prompt token count"
    );
    assert!(budget > 0 && prompt_tokens + budget <= 2_048);

    // Ordinary runtime entry point: no golden ids, output cache, or render-draft
    // API is involved. Existing snapshot/query-dump environment flags propagate.
    let runtime = Gemma4GpuRuntime::load(Path::new(&args[1]), 2_048)?;
    let mut assistant = Gemma4Mtp12AssistantMetal::load(Path::new(&args[2]))?;
    let generation =
        runtime.generate_greedy_mtp12_ordered_q4(&mut assistant, &prompt, budget, 8)?;

    let golden: serde_json::Value = serde_json::from_slice(&fs::read(&args[4])?)?;
    let golden_ids: Vec<u32> =
        serde_json::from_value(golden["camelid"]["generated_token_ids"].clone())?;
    let golden_text = golden["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("golden visible text missing")?;
    let raw_expected_text = tokenizer.decode(&golden_ids, true)?;
    let visible_text = visible_chat_text(&generation.text);
    let ids_exact = generation.token_ids == golden_ids;
    let raw_text_exact = generation.text == raw_expected_text;
    let visible_text_exact = visible_text == golden_text;
    let prompt_exact = generation.prompt_token_count == prompt_tokens
        && golden["usage"]["prompt_tokens"].as_u64() == Some(prompt_tokens as u64);
    let trace = &generation.stats.trace;
    assert_eq!(trace.len() as u64, generation.stats.rounds);
    assert_eq!(
        trace.iter().map(|r| r.accepted_drafts as u64).sum::<u64>(),
        generation.stats.accepted_drafts
    );
    assert_eq!(
        trace
            .iter()
            .map(|r| r.physical_verify_width as u64)
            .sum::<u64>(),
        generation.stats.target_verify_rows
    );
    for r in trace {
        assert_eq!(r.target_greedy_ids.len(), r.physical_verify_width);
        assert_eq!(r.drafts.len() + 1, r.logical_verify_width);
        assert_eq!(r.committed_input_rows, r.accepted_drafts + 1);
        assert_eq!(r.emitted_token_ids.len(), r.committed_input_rows);
    }
    let selectors = std::env::vars()
        .filter(|(k, _)| k.starts_with("CAMELID_GEMMA4_"))
        .collect::<BTreeMap<_, _>>();
    let receipt = serde_json::json!({
        "schema": "camelid.mtp12_full_target_trace.v1",
        "capture_method": "one ordinary greedy MTP generation; golden loaded afterward only for assertions",
        "max_positions": 2_048, "configured_verify_width": 8, "max_new_tokens": budget,
        "request_file": args[3], "golden_response_file": args[4],
        "rendered_prompt": prompt, "prompt_tokens": prompt_tokens,
        "rendered_prompt_token_ids": rendered_prompt_token_ids,
        "selectors": selectors,
        "validation": {"token_ids_exact": ids_exact, "raw_text_exact": raw_text_exact,
            "visible_text_exact": visible_text_exact, "prompt_tokens_exact": prompt_exact},
        "visible_text": visible_text,
        "generation": generation,
    });
    let json = serde_json::to_string_pretty(&receipt)?;
    fs::write(&args[5], format!("{json}\n"))?;
    println!("{json}");
    assert!(
        ids_exact && raw_text_exact && visible_text_exact && prompt_exact,
        "trace capture differs from the old receipt; full diagnostic trace was written"
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This trace capture requires macOS and Metal.");
    std::process::exit(1);
}
