use std::{fs, path::Path};

use camelid::{gguf::read_metadata, tokenizer::Tokenizer, BackendError};

#[test]
fn loads_llama_spm_tokenizer_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let gguf = read_metadata(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    assert_eq!(tokenizer.special.bos, Some(1));
    assert_eq!(tokenizer.special.eos, Some(2));
    assert!(tokenizer.config.add_bos);
    assert!(!tokenizer.config.add_eos);
    assert_eq!(tokenizer.tokens.len(), 14);
}

#[test]
fn preserves_chat_template_metadata_when_present() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-chat-template.gguf");
    let template =
        "{% for message in messages %}<|user|><|assistant|><|system|>{{ eos_token }}{% endfor %}";
    write_tokenizer_gguf_with_overrides(
        &path,
        TokenizerFixtureOverrides {
            model: "llama",
            add_bos: true,
            add_space_prefix: true,
            chat_template: Some(template),
            ..TokenizerFixtureOverrides::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    assert_eq!(tokenizer.chat_template.as_deref(), Some(template));
}

#[test]
fn loads_gpt2_bpe_tokenizer_metadata_and_specials() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-gpt2.gguf");
    write_gpt2_bpe_tokenizer_gguf(&path);

    let gguf = read_metadata(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    assert_eq!(tokenizer.model.as_summary_model(), "gpt2_bpe");
    assert_eq!(tokenizer.special.bos, Some(12));
    assert_eq!(tokenizer.special.eos, Some(13));
    assert_eq!(tokenizer.special.eot, Some(16));
    assert!(tokenizer.special.eog.contains(&13));
    assert!(tokenizer.special.eog.contains(&16));
    assert_eq!(tokenizer.tokens.len(), 35);
    assert_eq!(tokenizer.bpe_ranks.len(), 17);
    assert_eq!(
        tokenizer.chat_template.as_deref(),
        Some(LLAMA3_CHAT_TEMPLATE)
    );
}

#[test]
fn encodes_and_decodes_minimal_gpt2_bpe_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-gpt2.gguf");
    write_gpt2_bpe_tokenizer_gguf(&path);

    let gguf = read_metadata(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    assert_eq!(tokenizer.encode("hello", true, false).unwrap(), vec![12, 9]);
    assert_eq!(tokenizer.encode(" hello", false, false).unwrap(), vec![10]);
    assert_eq!(tokenizer.encode("\n\n", false, false).unwrap(), vec![11]);
    assert_eq!(
        tokenizer.encode("<|eot_id|>hello", false, true).unwrap(),
        vec![16, 9]
    );
    assert_eq!(
        tokenizer
            .encode(
                "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                true,
                true,
            )
            .unwrap(),
        vec![12, 14, 26, 15, 11, 9, 16, 14, 34, 15, 11]
    );
    assert_eq!(tokenizer.decode(&[12, 9, 16], true).unwrap(), "hello");
    assert_eq!(tokenizer.decode(&[10, 11], false).unwrap(), " hello\n\n");
}

#[test]
fn verifies_llama3_real_vocab_ids_match_gguf_metadata_when_available() {
    let Some((tokenizer, token_texts)) = load_real_llama3_tokenizer() else {
        return;
    };

    assert_eq!(token_texts.len(), 128_256);
    assert_eq!(tokenizer.tokens.len(), 128_256);
    assert_eq!(tokenizer.model.as_summary_model(), "gpt2_bpe");
    assert_eq!(tokenizer.special.bos, Some(128_000));
    assert_eq!(tokenizer.special.eos, Some(128_001));
    assert_eq!(tokenizer.special.eot, Some(128_009));

    for (idx, expected_text) in token_texts.iter().enumerate() {
        let token = &tokenizer.tokens[idx];
        assert_eq!(token.id as usize, idx);
        assert_eq!(
            &token.text, expected_text,
            "token text mismatch at id {idx}"
        );
    }
}

#[test]
fn llama3_reference_fixture_records_llama_cpp_token_ids() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/tokenizer/llama3-reference-tokenizer.json"
    ))
    .unwrap();

    assert_eq!(
        fixture["reference"]["tool"],
        "llama.cpp llama-tokenize --ids"
    );
    assert_eq!(
        fixture["cases"]["quick_brown_fox"]["tokens"],
        serde_json::json!([128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13])
    );
    assert_eq!(
        fixture["cases"]["begin_text_hows_it_going"]["text"],
        "<|begin_of_text|>hello how's it going?"
    );
    assert_eq!(
        fixture["cases"]["begin_text_hows_it_going"]["tokens"],
        serde_json::json!([128000, 15339, 1268, 596, 433, 2133, 30])
    );
}

#[test]
fn encodes_llama3_real_prompts_like_llama_cpp_when_available() {
    let Some((tokenizer, _)) = load_real_llama3_tokenizer() else {
        return;
    };

    let cases = [
        (
            "quick_brown_fox",
            "The quick brown fox jumps over the lazy dog.",
            true,
            false,
            vec![
                128_000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13,
            ],
        ),
        (
            "begin_text_hows_it_going",
            "<|begin_of_text|>hello how's it going?",
            false,
            true,
            vec![128_000, 15339, 1268, 596, 433, 2133, 30],
        ),
    ];

    for (name, text, add_special, parse_special, expected) in cases {
        assert_eq!(
            tokenizer.encode(text, add_special, parse_special).unwrap(),
            expected,
            "Llama 3 tokenizer parity failed for {name}; expected IDs are from fixtures/tokenizer/llama3-reference-tokenizer.json generated with scripts/llama3-tokenizer-reference.mjs against llama.cpp llama-tokenize"
        );
    }
}

#[test]
fn tinyllama_edge_reference_pack_records_required_prompt_shapes_and_tokens() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/tokenizer/tinyllama-chat-template-edge-cases.json"
    ))
    .unwrap();

    assert_eq!(fixture["schema"], "camelid.tokenizer_reference_pack.v1");
    assert_eq!(fixture["row_id"], "tinyllama_1_1b_chat_q8_0");
    assert_eq!(
        fixture["tokenizer_fixture_id"],
        "tinyllama-chat-template-edge-cases-v1"
    );
    assert_eq!(
        fixture["reference"]["evidence_bundle"],
        "qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/manifest.json"
    );

    let cases = fixture["cases"].as_object().unwrap();
    for name in [
        "trailing_whitespace",
        "special_control_chars",
        "multiline_chat",
        "assistant_final_eos",
        "multiline_whitespace",
    ] {
        let case = &cases[name];
        assert_eq!(case["prompt_tokens_match"], true, "case {name}");
        assert_eq!(case["add_special"], false, "case {name}");
        assert_eq!(case["parse_special"], true, "case {name}");
        assert_eq!(
            case["tokens"].as_array().unwrap().len(),
            case["reference_prompt_token_count"].as_u64().unwrap() as usize,
            "case {name}"
        );
    }

    assert!(cases["multiline_whitespace"]["expected_prompt"]
        .as_str()
        .unwrap()
        .contains("\n\n  Indented line two.  </s>"));
    assert!(cases["assistant_final_eos"]["expected_prompt"]
        .as_str()
        .unwrap()
        .ends_with("elid</s>\n"));
}

#[test]
fn encodes_tinyllama_edge_reference_pack_like_llama_cpp_when_available() {
    let Some(tokenizer) = load_real_tinyllama_tokenizer() else {
        return;
    };

    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/tokenizer/tinyllama-chat-template-edge-cases.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_object().unwrap();

    for (name, case) in cases {
        let text = case["expected_prompt"].as_str().unwrap();
        let add_special = case["add_special"].as_bool().unwrap();
        let parse_special = case["parse_special"].as_bool().unwrap();
        let expected: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();

        assert_eq!(
            tokenizer.encode(text, add_special, parse_special).unwrap(),
            expected,
            "TinyLlama tokenizer/chat-template edge parity failed for {name}; expected IDs are from fixtures/tokenizer/tinyllama-chat-template-edge-cases.json backed by the committed llama.cpp parity reports"
        );
    }
}

#[test]
fn mistral_reference_pack_records_required_prompt_shapes_and_tokens() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/tokenizer/mistral-7b-instruct-v0.3-reference-pack.json"
    ))
    .unwrap();

    assert_eq!(fixture["status"], "reference_capture");
    assert_eq!(
        fixture["expected_artifacts"]["gguf_sha256"],
        "404857e776114baada71a08ebd3bba79d721ec7fca99705e7e7b892ae8bc583f"
    );
    assert_eq!(
        fixture["expected_artifacts"]["tokenizer_fixture_id"],
        "mistral-instruct-v0.3-tokenizer-v1"
    );
    assert_eq!(
        fixture["expected_artifacts"]["chat_template_fixture_id"],
        "mistral-instruct-v0.3-chat-template-pack-v1"
    );
    assert_eq!(
        fixture["expected_artifacts"]["prompt_cases"][1]["rendered_prompt"],
        "<s>[INST] Hello [/INST]"
    );
    assert_eq!(
        fixture["expected_artifacts"]["prompt_cases"][1]["expected_tokens"],
        serde_json::json!([1, 3, 29473, 23325, 29473, 4])
    );
    assert_eq!(
        fixture["expected_artifacts"]["prompt_cases"][2]["rendered_prompt"],
        "<s>[INST] Be brief.\n\nHello there. [/INST]"
    );
    assert_eq!(
        fixture["expected_artifacts"]["prompt_cases"][3]["rendered_prompt"],
        "<s>[INST] Complete cam [/INST] elid</s><s>[INST] Now say hi [/INST]"
    );
}

#[test]
fn mixtral_reference_pack_records_required_prompt_shapes_and_tokens() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/tokenizer/mixtral-8x7b-instruct-v0.1-reference-pack.json"
    ))
    .unwrap();

    assert_eq!(fixture["status"], "reference_capture");
    assert_eq!(
        fixture["expected_artifacts"]["hf_commit_sha"],
        "93c0492d1891b5147f42b2648d9fccc140910a2f"
    );
    assert_eq!(fixture["expected_artifacts"]["license"], "apache-2.0");
    assert_eq!(
        fixture["expected_artifacts"]["tokenizer_fixture_id"],
        "mixtral-instruct-v0.1-tokenizer-v1"
    );
    assert_eq!(
        fixture["expected_artifacts"]["chat_template_fixture_id"],
        "mixtral-instruct-v0.1-chat-template-pack-v1"
    );
    assert_eq!(
        fixture["expected_artifacts"]["prompt_cases"][1]["rendered_prompt"],
        "<s> [INST] Hello [/INST]"
    );
    assert_eq!(
        fixture["expected_artifacts"]["prompt_cases"][1]["expected_tokens"],
        serde_json::json!([1, 733, 16289, 28793, 22557, 733, 28748, 16289, 28793])
    );
    assert_eq!(
        fixture["expected_artifacts"]["prompt_cases"][2]["rendered_prompt"],
        "<s> [INST] Be brief.\n\nHello there. [/INST]"
    );
}

#[test]
fn encodes_mixtral_real_prompts_like_llama_cpp_when_available() {
    let Some(tokenizer) = load_real_mixtral_tokenizer() else {
        return;
    };

    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/tokenizer/mixtral-8x7b-instruct-v0.1-reference-pack.json"
    ))
    .unwrap();
    let cases = fixture["expected_artifacts"]["prompt_cases"]
        .as_array()
        .unwrap();

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let text = case["rendered_prompt"].as_str().unwrap();
        let add_special = case["add_bos"].as_bool().unwrap();
        let parse_special = case["parse_special"].as_bool().unwrap();
        let expected: Vec<u32> = case["expected_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();

        assert_eq!(
            tokenizer.encode(text, add_special, parse_special).unwrap(),
            expected,
            "Mixtral tokenizer parity failed for {name}; expected IDs are from llama.cpp llama-tokenize against the exact Mixtral-8x7B-Instruct-v0.1.Q8_0 GGUF metadata"
        );
    }
}

#[test]
fn encodes_mistral_real_prompts_like_llama_cpp_when_available() {
    let Some(tokenizer) = load_real_mistral_tokenizer() else {
        return;
    };

    let cases = [
        ("hello", "hello", true, false, vec![1, 7080, 29477]),
        (
            "single_user_turn",
            "<s>[INST] Hello [/INST]",
            false,
            true,
            vec![1, 3, 29473, 23325, 29473, 4],
        ),
        (
            "broader_prompt_give_split",
            "<s>[INST] You write compact engineering checklists.\n\ngive three short steps for checking a rust change before a public push. [/INST]",
            false,
            true,
            vec![
                1, 3, 29473, 1763, 4092, 13192, 14088, 2645, 24707, 29491, 781, 781,
                29489, 1263, 2480, 3253, 6712, 1122, 13547, 1032, 15680, 3036, 1927,
                1032, 1566, 6464, 29491, 29473, 4,
            ],
        ),
        (
            "broader_prompt_tokenizer_stress_words",
            "<s>[INST] You are checking tokenizer coverage. Reply with one simple sentence.\n\nUse these exact words once: Fact alpaca MSTR mstr CMLD checksum gamma llama. Then say the check is done. [/INST]",
            false,
            true,
            vec![
                1, 3, 29473, 1763, 1228, 13547, 6797, 4792, 11634, 29491, 4125,
                1114, 1163, 1392, 4356, 13039, 29491, 781, 781, 9311, 1935,
                4227, 3853, 3095, 29515, 1169, 1340, 1157, 29488, 14458, 1119,
                5340, 1058, 1810, 1102, 4595, 29525, 2645, 2569, 1087, 3421,
                9582, 3554, 29491, 3247, 2083, 1040, 2645, 1117, 2971, 29491,
                29473, 4,
            ],
        ),
        (
            "broader_prompt_later_drift_question",
            "<s>[INST] You are helpful, direct, and practical.\n\nwhat should a developer look at first if a local chat model gives the right first token but drifts later? [/INST]",
            false,
            true,
            vec![
                1, 3, 29473, 1763, 1228, 11633, 29493, 2631, 29493, 1072, 11886,
                29491, 781, 781, 7570, 1791, 1032, 22550, 1681, 1206, 1675, 1281,
                1032, 2630, 11474, 2997, 5980, 1040, 1871, 1675, 6797, 1330,
                2373, 9805, 2830, 29572, 29473, 4,
            ],
        ),
    ];

    for (name, text, add_special, parse_special, expected) in cases {
        assert_eq!(
            tokenizer.encode(text, add_special, parse_special).unwrap(),
            expected,
            "Mistral tokenizer parity failed for {name}; expected IDs are from llama.cpp llama-tokenize against the exact Mistral-7B-Instruct-v0.3-Q8_0.gguf row"
        );
    }
}

#[test]
fn rejects_gpt2_bpe_without_supported_pre_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-gpt2-wrong-pre.gguf");
    // A GPT-2/BPE pre-tokenizer we do not support (not llama-bpe or qwen2) must
    // still fail closed.
    write_gpt2_bpe_tokenizer_gguf_with_pre(&path, "deepseek-llm");

    let gguf = read_metadata(&path).unwrap();
    let err = Tokenizer::from_gguf(&gguf).unwrap_err().to_string();
    assert!(err.contains("unsupported GPT-2/BPE pre-tokenizer"));
    assert!(err.contains("llama-bpe"));
    assert!(err.contains("qwen2"));
}

#[test]
fn accepts_qwen2_gpt2_bpe_pre_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-gpt2-qwen2-pre.gguf");
    write_gpt2_bpe_tokenizer_gguf_with_pre(&path, "qwen2");

    let gguf = read_metadata(&path).unwrap();
    // Qwen2/Qwen3 GPT-2/BPE pre-tokenizer is supported (single-digit grouping).
    Tokenizer::from_gguf(&gguf).expect("qwen2 pre-tokenizer must be accepted");
}

#[test]
fn rejects_unsupported_tokenizer_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "wordpiece", true, false, true);

    let gguf = read_metadata(&path).unwrap();
    let err = Tokenizer::from_gguf(&gguf).unwrap_err();
    assert!(matches!(err, BackendError::UnsupportedTokenizer(_)));
}

#[test]
fn encodes_empty_with_bos_when_special_enabled() {
    let tokenizer = load_fixture(true, false, true);
    assert_eq!(tokenizer.encode("", true, false).unwrap(), vec![1]);
    assert!(tokenizer.encode("", false, false).unwrap().is_empty());
}

#[test]
fn encodes_known_piece_with_space_prefix() {
    let tokenizer = load_fixture(true, false, true);
    assert_eq!(tokenizer.encode("hello", true, false).unwrap(), vec![1, 3]);
}

#[test]
fn parse_special_spm_merges_dummy_prefix_into_the_following_piece() {
    let tokenizer = load_fixture(false, false, true);

    // The dummy prefix after a control token belongs to the NEXT fragment, not to
    // a token of its own: the reference prepends a space to the fragment and lets
    // SPM merge it, yielding `▁hello` (3). Emitting the bare `▁` (6) followed by
    // `hello` (4) is the divergence this asserts against — it split `▁What` into
    // `▁` + `What` on every real prompt that followed a special.
    assert_eq!(
        tokenizer.encode("<s>hello", false, true).unwrap(),
        vec![1, 3]
    );
}

#[test]
fn falls_back_to_byte_tokens() {
    let tokenizer = load_fixture(false, false, false);
    assert_eq!(tokenizer.encode("!", false, false).unwrap(), vec![5]);
}

#[test]
fn applies_tokenizer_merges_when_present() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-merges.gguf");
    // A merge-driven tokenizer model. `llama` is SPM proper: the reference
    // segments it by token score and ignores any merge list the converter
    // embedded, so asserting merge behaviour under `llama` asserted the bug that
    // switched TinyLlama (which ships 61k merges) onto rank-based BPE.
    write_tokenizer_gguf_with_overrides(
        &path,
        TokenizerFixtureOverrides {
            model: "gemma4",
            merges: &["h e", "he l", "hel l", "hell o"],
            ..TokenizerFixtureOverrides::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    assert_eq!(tokenizer.encode("hello", false, false).unwrap(), vec![4]);
}

#[test]
fn adds_dummy_prefix_after_control_token_before_newline() {
    let tokenizer = load_fixture(false, false, true);
    // parse_special=true: a control token is only recognised as a token when
    // specials are being parsed. This previously passed with parse_special=false
    // because longest-match happened to swallow the literal text `</s>` as its own
    // id — something the reference never does with specials off.
    assert_eq!(
        tokenizer.encode("hello</s>\nhello", false, true).unwrap(),
        vec![3, 2, 6, 0, 4]
    );
}

#[test]
fn does_not_duplicate_dummy_prefix_after_control_token_before_space() {
    let tokenizer = load_fixture(false, false, true);
    // parse_special=true for the same reason as the newline case above.
    assert_eq!(
        tokenizer.encode("hello</s> hello", false, true).unwrap(),
        vec![3, 2, 3]
    );
}

#[test]
fn decodes_byte_tokens_and_sentencepiece_space() {
    let tokenizer = load_fixture(false, false, false);
    assert_eq!(tokenizer.decode(&[3, 5], true).unwrap(), " hello!");
}

#[test]
fn rejects_special_token_id_outside_vocab() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-bad-special.gguf");
    write_tokenizer_gguf_with_overrides(
        &path,
        TokenizerFixtureOverrides {
            model: "llama",
            add_bos: true,
            add_space_prefix: true,
            bos: Some(99),
            ..TokenizerFixtureOverrides::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = Tokenizer::from_gguf(&gguf).unwrap_err().to_string();

    assert!(err.contains("bos token id 99"));
    assert!(err.contains("out of range for vocab size 14"));
}

#[test]
fn rejects_score_array_shorter_than_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-short-scores.gguf");
    write_tokenizer_gguf_with_overrides(
        &path,
        TokenizerFixtureOverrides {
            model: "llama",
            add_bos: true,
            add_space_prefix: true,
            score_len: Some(3),
            ..TokenizerFixtureOverrides::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = Tokenizer::from_gguf(&gguf).unwrap_err().to_string();

    assert!(err.contains("tokenizer.ggml.scores length 3"));
    assert!(err.contains("shorter than token count 14"));
}

#[test]
fn rejects_unknown_token_type_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer-bad-type.gguf");
    write_tokenizer_gguf_with_overrides(
        &path,
        TokenizerFixtureOverrides {
            model: "llama",
            add_bos: true,
            add_space_prefix: true,
            token_type: Some(42),
            ..TokenizerFixtureOverrides::default()
        },
    );

    let gguf = read_metadata(&path).unwrap();
    let err = Tokenizer::from_gguf(&gguf).unwrap_err().to_string();

    assert!(err.contains("unknown tokenizer token type 42"));
}

fn load_real_llama3_tokenizer() -> Option<(Tokenizer, Vec<String>)> {
    let path = std::env::var("LLAMA3_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("models/Meta-Llama-3-8B-Instruct-Q8_0.gguf"));
    if !path.exists() {
        eprintln!(
            "skipping real Llama 3 tokenizer parity; set LLAMA3_GGUF or place the artifact at {}",
            path.display()
        );
        return None;
    }

    let gguf = read_metadata(&path).unwrap();
    let token_texts = gguf
        .metadata_array_strings("tokenizer.ggml.tokens")
        .unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    Some((tokenizer, token_texts))
}

fn load_real_tinyllama_tokenizer() -> Option<Tokenizer> {
    let path = std::env::var("TINYLLAMA_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"));
    if !path.exists() {
        eprintln!(
            "skipping real TinyLlama tokenizer parity; set TINYLLAMA_GGUF or place the artifact at {}",
            path.display()
        );
        return None;
    }

    let gguf = read_metadata(&path).unwrap();
    Some(Tokenizer::from_gguf(&gguf).unwrap())
}

fn load_real_mistral_tokenizer() -> Option<Tokenizer> {
    let path = std::env::var("MISTRAL_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("models/Mistral-7B-Instruct-v0.3-Q8_0.gguf"));
    if !path.exists() {
        eprintln!(
            "skipping real Mistral tokenizer parity; set MISTRAL_GGUF or place the artifact at {}",
            path.display()
        );
        return None;
    }

    let gguf = read_metadata(&path).unwrap();
    Some(Tokenizer::from_gguf(&gguf).unwrap())
}

fn load_real_mixtral_tokenizer() -> Option<Tokenizer> {
    let path = std::env::var("MIXTRAL_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("models/mixtral-8x7b-instruct-v0.1-q8_0.gguf")
        });
    if !path.exists() {
        eprintln!(
            "skipping real Mixtral tokenizer parity; set MIXTRAL_GGUF or place the artifact at {}",
            path.display()
        );
        return None;
    }

    let gguf = read_metadata(&path).unwrap();
    Some(Tokenizer::from_gguf(&gguf).unwrap())
}

const LLAMA3_CHAT_TEMPLATE: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

fn write_gpt2_bpe_tokenizer_gguf(path: &Path) {
    write_gpt2_bpe_tokenizer_gguf_with_pre(path, "llama-bpe");
}

fn write_gpt2_bpe_tokenizer_gguf_with_pre(path: &Path, pre_tokenizer: &str) {
    let tokens = [
        "h",
        "e",
        "l",
        "o",
        "Ġ",
        "Ċ",
        "he",
        "hel",
        "hell",
        "hello",
        "Ġhello",
        "ĊĊ",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "u",
        "s",
        "r",
        "a",
        "i",
        "t",
        "n",
        "us",
        "use",
        "user",
        "as",
        "ass",
        "assi",
        "assis",
        "assist",
        "assista",
        "assistan",
        "assistant",
    ];
    let token_types = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    ];
    let merges = [
        "h e",
        "he l",
        "hel l",
        "hell o",
        "Ġ hello",
        "Ċ Ċ",
        "u s",
        "us e",
        "use r",
        "a s",
        "as s",
        "ass i",
        "assi s",
        "assis t",
        "assist a",
        "assista n",
        "assistan t",
    ];

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, 0); // tensor count
    push_i64(&mut b, 11); // metadata count

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_string(&mut b, "tokenizer.ggml.model", "gpt2");
    push_kv_string(&mut b, "tokenizer.ggml.pre", pre_tokenizer);
    push_kv_array_strings(&mut b, "tokenizer.ggml.tokens", &tokens);
    push_kv_array_i32(&mut b, "tokenizer.ggml.token_type", &token_types);
    push_kv_array_strings(&mut b, "tokenizer.ggml.merges", &merges);
    push_kv_u32(&mut b, "tokenizer.ggml.bos_token_id", 12);
    push_kv_u32(&mut b, "tokenizer.ggml.eos_token_id", 13);
    push_kv_bool(&mut b, "tokenizer.ggml.add_bos_token", true);
    push_kv_bool(&mut b, "tokenizer.ggml.add_eos_token", false);
    push_kv_string(&mut b, "tokenizer.chat_template", LLAMA3_CHAT_TEMPLATE);

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    fs::write(path, b).unwrap();
}

fn load_fixture(add_bos: bool, add_eos: bool, add_space_prefix: bool) -> Tokenizer {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", add_bos, add_eos, add_space_prefix);
    let gguf = read_metadata(&path).unwrap();
    Tokenizer::from_gguf(&gguf).unwrap()
}

fn write_tokenizer_gguf(
    path: &Path,
    model: &str,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
) {
    write_tokenizer_gguf_with_overrides(
        path,
        TokenizerFixtureOverrides {
            model,
            add_bos,
            add_eos,
            add_space_prefix,
            ..TokenizerFixtureOverrides::default()
        },
    );
}

#[derive(Default)]
struct TokenizerFixtureOverrides<'a> {
    model: &'a str,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    bos: Option<u32>,
    score_len: Option<usize>,
    token_type: Option<i32>,
    chat_template: Option<&'a str>,
    merges: &'a [&'a str],
}

fn write_tokenizer_gguf_with_overrides(path: &Path, overrides: TokenizerFixtureOverrides<'_>) {
    // SPM reaches a token only through successive pairwise merges in which every
    // intermediate is ITSELF in the vocab. Without the single-step pieces below,
    // `▁hello` and `hello` are unreachable and every character falls back to
    // <unk> — the original 7-entry vocab was only tokenizable by a longest-match
    // encoder, not by the reference algorithm. Appended (never reordered) so the
    // ids every other test asserts stay put.
    let tokens = [
        "<unk>", "<s>", "</s>", "▁hello", "hello", "<0x21>", "▁", "▁h", "▁he", "▁hel", "▁hell",
        "he", "hel", "hell",
    ];
    // The `▁`-prefixed chain outscores the bare chain so each step has a single
    // best candidate and the merge order is deterministic.
    let all_scores = [
        0.0, 0.0, 0.0, 10.0, 2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 2.5, 2.6, 2.7,
    ];
    let mut token_types = [2, 3, 3, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1];
    if let Some(token_type) = overrides.token_type {
        token_types[3] = token_type;
    }
    let score_len = overrides.score_len.unwrap_or(all_scores.len());
    let scores = &all_scores[..score_len];

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, 0); // tensor count
    let metadata_count =
        10 + i64::from(overrides.chat_template.is_some()) + i64::from(!overrides.merges.is_empty());
    push_i64(&mut b, metadata_count); // metadata count

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_string(&mut b, "tokenizer.ggml.model", overrides.model);
    push_kv_array_strings(&mut b, "tokenizer.ggml.tokens", &tokens);
    push_kv_array_f32(&mut b, "tokenizer.ggml.scores", scores);
    push_kv_array_i32(&mut b, "tokenizer.ggml.token_type", &token_types);
    push_kv_u32(
        &mut b,
        "tokenizer.ggml.bos_token_id",
        overrides.bos.unwrap_or(1),
    );
    push_kv_u32(&mut b, "tokenizer.ggml.eos_token_id", 2);
    push_kv_bool(&mut b, "tokenizer.ggml.add_bos_token", overrides.add_bos);
    push_kv_bool(&mut b, "tokenizer.ggml.add_eos_token", overrides.add_eos);
    push_kv_bool(
        &mut b,
        "tokenizer.ggml.add_space_prefix",
        overrides.add_space_prefix,
    );
    if let Some(chat_template) = overrides.chat_template {
        push_kv_string(&mut b, "tokenizer.chat_template", chat_template);
    }
    if !overrides.merges.is_empty() {
        push_kv_array_strings(&mut b, "tokenizer.ggml.merges", overrides.merges);
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    fs::write(path, b).unwrap();
}

fn push_kv_string(b: &mut Vec<u8>, key: &str, value: &str) {
    push_string(b, key);
    push_i32(b, 8);
    push_string(b, value);
}

fn push_kv_u32(b: &mut Vec<u8>, key: &str, value: u32) {
    push_string(b, key);
    push_i32(b, 4);
    push_u32(b, value);
}

fn push_kv_bool(b: &mut Vec<u8>, key: &str, value: bool) {
    push_string(b, key);
    push_i32(b, 7);
    b.push(u8::from(value));
}

fn push_kv_array_strings(b: &mut Vec<u8>, key: &str, values: &[&str]) {
    push_string(b, key);
    push_i32(b, 9);
    push_i32(b, 8);
    push_u64(b, values.len() as u64);
    for value in values {
        push_string(b, value);
    }
}

fn push_kv_array_f32(b: &mut Vec<u8>, key: &str, values: &[f32]) {
    push_string(b, key);
    push_i32(b, 9);
    push_i32(b, 6);
    push_u64(b, values.len() as u64);
    for value in values {
        b.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_kv_array_i32(b: &mut Vec<u8>, key: &str, values: &[i32]) {
    push_string(b, key);
    push_i32(b, 9);
    push_i32(b, 5);
    push_u64(b, values.len() as u64);
    for value in values {
        push_i32(b, *value);
    }
}

fn push_string(b: &mut Vec<u8>, value: &str) {
    push_u64(b, value.len() as u64);
    b.extend_from_slice(value.as_bytes());
}

fn push_u32(b: &mut Vec<u8>, value: u32) {
    b.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(b: &mut Vec<u8>, value: i32) {
    b.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(b: &mut Vec<u8>, value: u64) {
    b.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(b: &mut Vec<u8>, value: i64) {
    b.extend_from_slice(&value.to_le_bytes());
}
