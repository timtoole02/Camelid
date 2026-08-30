use std::{
    collections::BTreeMap,
    io::Write,
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(
        qos_class: u32,
        relative_priority: std::os::raw::c_int,
    ) -> std::os::raw::c_int;
}

/// Commit prefixes used by the rejected-tail overwrite receipt for one physical
/// verifier width. The middle case straddles the boundary between the two K8
/// fragments at W16, while zero/full pin rollback and complete-commit behavior.
#[cfg(any(target_os = "macos", test))]
fn gemma4_rejected_tail_commit_prefixes(width: usize) -> Option<Vec<usize>> {
    if !matches!(width, 1 | 2 | 4 | 8 | 16) {
        return None;
    }
    let mut prefixes = vec![
        0,
        1,
        (width / 2).saturating_sub(1),
        width.saturating_sub(1),
        width,
    ];
    prefixes.sort_unstable();
    prefixes.dedup();
    Some(prefixes)
}

#[cfg(test)]
mod gemma4_verifier_receipt_tests {
    use super::gemma4_rejected_tail_commit_prefixes;

    #[test]
    fn rejected_tail_prefixes_scale_through_physical_w16() {
        let cases: [(usize, &[usize]); 5] = [
            (1, &[0, 1]),
            (2, &[0, 1, 2]),
            (4, &[0, 1, 3, 4]),
            (8, &[0, 1, 3, 7, 8]),
            (16, &[0, 1, 7, 15, 16]),
        ];
        for (width, expected) in cases {
            assert_eq!(
                gemma4_rejected_tail_commit_prefixes(width).as_deref(),
                Some(expected),
                "physical W{width}",
            );
        }
        for refused in [0usize, 3, 6, 12, 15, 17, usize::MAX] {
            assert!(gemma4_rejected_tail_commit_prefixes(refused).is_none());
        }
    }
}

#[cfg(test)]
mod ghost_moe_cli_tests {
    use super::*;

    fn on_cli_test_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("ghost-moe-cli-parse-test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(test)
            .expect("spawn CLI parse test")
            .join()
            .expect("CLI parse test panicked");
    }

    #[test]
    fn gpu_off_and_deterministic_resolve_before_model_load() {
        assert_eq!(resolved_gpu_switch(GpuMode::Auto, false), None);
        assert_eq!(resolved_gpu_switch(GpuMode::On, false), Some(true));
        assert_eq!(resolved_gpu_switch(GpuMode::Off, false), Some(false));
        assert_eq!(resolved_gpu_switch(GpuMode::Auto, true), Some(false));
        assert_eq!(resolved_gpu_switch(GpuMode::On, true), Some(false));
        assert_eq!(resolved_gpu_switch(GpuMode::Off, true), Some(false));
    }

    #[test]
    fn ghost_run_parses_global_expert_cache_budget() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "ghost-run",
                "gemma4.gguf",
                "--cghost",
                "gemma4.cghost",
                "--expert-cache-mib",
                "64",
            ])
            .expect("parse Ghost-MoE flags");
            match cli.command {
                Some(Command::GhostRun {
                    model,
                    cghost,
                    expert_cache_mib,
                    ..
                }) => {
                    assert_eq!(model, PathBuf::from("gemma4.gguf"));
                    assert_eq!(cghost, PathBuf::from("gemma4.cghost"));
                    assert_eq!(expert_cache_mib, 64);
                }
                other => panic!("expected GhostRun, got {other:?}"),
            }
        });
    }

    #[test]
    fn ghost_run_defaults_to_1024_mib_global_expert_cache() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "ghost-run",
                "model.gguf",
                "--cghost",
                "model.cghost",
            ])
            .expect("parse GhostRun defaults");
            match cli.command {
                Some(Command::GhostRun {
                    expert_cache_mib, ..
                }) => assert_eq!(expert_cache_mib, 1024),
                other => panic!("expected GhostRun, got {other:?}"),
            }
        });
    }

    #[test]
    fn bench_eagle3_dynamic_tree_flags_are_optional_and_deterministic() {
        on_cli_test_stack(|| {
            let defaults =
                Cli::try_parse_from(["camelid", "bench-eagle3", "target.gguf", "--eagle3", "head"])
                    .expect("parse EAGLE-3 linear defaults");
            match defaults.command {
                Some(Command::BenchEagle3 {
                    draft_tokens,
                    tree_nodes,
                    tree_topk,
                    tree_expansions,
                    suffix_first,
                    ..
                }) => {
                    assert_eq!(draft_tokens, 4);
                    assert_eq!(tree_nodes, None);
                    assert_eq!(tree_topk, 4);
                    assert_eq!(tree_expansions, 4);
                    assert!(!suffix_first);
                }
                other => panic!("expected BenchEagle3, got {other:?}"),
            }

            let tree = Cli::try_parse_from([
                "camelid",
                "bench-eagle3",
                "target.gguf",
                "--eagle3",
                "head",
                "--draft-tokens",
                "7",
                "--tree-nodes",
                "12",
                "--tree-topk",
                "8",
                "--tree-expansions",
                "5",
                "--suffix-first",
            ])
            .expect("parse EAGLE-3 dynamic tree flags");
            match tree.command {
                Some(Command::BenchEagle3 {
                    draft_tokens,
                    tree_nodes,
                    tree_topk,
                    tree_expansions,
                    suffix_first,
                    ..
                }) => {
                    assert_eq!(draft_tokens, 7);
                    assert_eq!(tree_nodes, Some(12));
                    assert_eq!(tree_topk, 8);
                    assert_eq!(tree_expansions, 5);
                    assert!(suffix_first);
                }
                other => panic!("expected BenchEagle3, got {other:?}"),
            }

            let suffix_without_tree = Cli::try_parse_from([
                "camelid",
                "bench-eagle3",
                "target.gguf",
                "--eagle3",
                "head",
                "--suffix-first",
            ])
            .expect_err("--suffix-first must require --tree-nodes");
            assert_eq!(
                suffix_without_tree.kind(),
                clap::error::ErrorKind::MissingRequiredArgument
            );
        });
    }

    #[test]
    fn deepest_suffix_chain_prefers_the_first_deepest_branch_within_budget() {
        let tree = camelid::inference::spec_tree::TokenTree {
            tokens: vec![10, 20, 30, 21, 31, 22],
            parent: vec![-1, 0, 0, 1, 2, 3],
            depth: vec![0, 1, 1, 2, 2, 3],
        };
        assert_eq!(deepest_suffix_chain(&tree, 3), vec![20, 21, 22]);
        assert_eq!(deepest_suffix_chain(&tree, 2), vec![20, 21]);
        assert!(deepest_suffix_chain(&tree, 0).is_empty());
        assert!(deepest_suffix_chain(
            &camelid::inference::spec_tree::TokenTree::linear(10, &[]),
            4
        )
        .is_empty());
    }

    #[test]
    fn inspect_source_accepts_a_hugging_face_directory() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from(["camelid", "inspect-source", "hf-model"])
                .expect("parse source inspection command");
            match cli.command {
                Some(Command::InspectSource { path }) => {
                    assert_eq!(path, PathBuf::from("hf-model"));
                }
                other => panic!("expected InspectSource, got {other:?}"),
            }
        });
    }

    #[test]
    fn gemma4_mtp12_gpu_parses_exact_assistant_and_target_widths() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "gemma4-mtp12-gpu",
                "target.gguf",
                "--assistant",
                "assistant/model.safetensors",
                "--max-tokens",
                "96",
                "--widths",
                "2,4,8,16",
            ])
            .expect("parse Gemma 4 MTP12 Metal harness");
            match cli.command {
                Some(Command::Gemma4Mtp12Gpu {
                    path,
                    assistant,
                    max_tokens,
                    widths,
                    ..
                }) => {
                    assert_eq!(path, PathBuf::from("target.gguf"));
                    assert_eq!(
                        assistant,
                        PathBuf::from("assistant/model.safetensors")
                    );
                    assert_eq!(max_tokens, 96);
                    assert_eq!(widths, vec![2, 4, 8, 16]);
                }
                other => panic!("expected Gemma4Mtp12Gpu, got {other:?}"),
            }
        });
    }

    #[test]
    fn gemma4_q4_repack_requires_an_explicit_sidecar_destination() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "gemma4-q4-repack",
                "target.gguf",
                "--output",
                "target.q4-native",
            ])
            .expect("parse Gemma 4 native Q4 repack command");
            match cli.command {
                Some(Command::Gemma4Q4Repack { path, output }) => {
                    assert_eq!(path, PathBuf::from("target.gguf"));
                    assert_eq!(output, PathBuf::from("target.q4-native"));
                }
                other => panic!("expected Gemma4Q4Repack, got {other:?}"),
            }

            let missing = Cli::try_parse_from(["camelid", "gemma4-q4-repack", "target.gguf"])
                .expect_err("native Q4 repack must not invent an output path");
            assert_eq!(
                missing.kind(),
                clap::error::ErrorKind::MissingRequiredArgument
            );
        });
    }

    #[test]
    fn inspect_prefix_requires_the_declared_artifact_length() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "inspect-prefix",
                "header.gguf",
                "--declared-len",
                "32483931648",
            ])
            .expect("parse ranged GGUF inspection command");
            match cli.command {
                Some(Command::InspectPrefix { path, declared_len }) => {
                    assert_eq!(path, PathBuf::from("header.gguf"));
                    assert_eq!(declared_len, 32_483_931_648);
                }
                other => panic!("expected InspectPrefix, got {other:?}"),
            }
        });
    }

    #[test]
    fn tokenize_prefix_requires_the_declared_artifact_length() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "tokenize",
                "--model",
                "header.gguf",
                "--declared-len",
                "9827149312",
                "--prompt",
                "Hello",
            ])
            .expect("parse ranged GGUF tokenizer command");
            match cli.command {
                Some(Command::Tokenize {
                    model,
                    declared_len,
                    prompt,
                    ..
                }) => {
                    assert_eq!(model, PathBuf::from("header.gguf"));
                    assert_eq!(declared_len, Some(9_827_149_312));
                    assert_eq!(prompt.as_deref(), Some("Hello"));
                }
                other => panic!("expected Tokenize, got {other:?}"),
            }
        });
    }

    #[test]
    fn tokenizer_prefix_matches_ordinary_file_and_fails_closed() {
        fn push_u32(bytes: &mut Vec<u8>, value: u32) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fn push_i32(bytes: &mut Vec<u8>, value: i32) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fn push_u64(bytes: &mut Vec<u8>, value: u64) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fn push_i64(bytes: &mut Vec<u8>, value: i64) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fn push_string(bytes: &mut Vec<u8>, value: &str) {
            push_u64(bytes, value.len() as u64);
            bytes.extend_from_slice(value.as_bytes());
        }
        fn kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
            push_string(bytes, key);
            push_u32(bytes, 8); // GGUF_TYPE_STRING
            push_string(bytes, value);
        }
        fn kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
            push_string(bytes, key);
            push_u32(bytes, 4); // GGUF_TYPE_UINT32
            push_u32(bytes, value);
        }
        fn kv_bool(bytes: &mut Vec<u8>, key: &str, value: bool) {
            push_string(bytes, key);
            push_u32(bytes, 7); // GGUF_TYPE_BOOL
            bytes.push(u8::from(value));
        }
        fn kv_strings(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
            push_string(bytes, key);
            push_u32(bytes, 9); // GGUF_TYPE_ARRAY
            push_u32(bytes, 8); // element GGUF_TYPE_STRING
            push_u64(bytes, values.len() as u64);
            for value in values {
                push_string(bytes, value);
            }
        }

        // A tiny Gemma-shaped tokenizer plus one F32 tensor. The prefix contains
        // every metadata value and descriptor but deliberately omits the tensor
        // payload, matching the immutable HTTP Range qualification path.
        let mut prefix = Vec::new();
        prefix.extend_from_slice(b"GGUF");
        push_u32(&mut prefix, 3);
        push_i64(&mut prefix, 1); // tensor_count
        push_i64(&mut prefix, 9); // metadata_count
        kv_string(&mut prefix, "general.architecture", "gemma2");
        kv_string(&mut prefix, "tokenizer.ggml.model", "llama");
        kv_strings(
            &mut prefix,
            "tokenizer.ggml.tokens",
            &["<unk>", "</s>", "<s>", "h"],
        );
        kv_u32(&mut prefix, "tokenizer.ggml.bos_token_id", 2);
        kv_u32(&mut prefix, "tokenizer.ggml.eos_token_id", 1);
        kv_u32(&mut prefix, "tokenizer.ggml.unknown_token_id", 0);
        kv_bool(&mut prefix, "tokenizer.ggml.add_bos_token", true);
        kv_bool(&mut prefix, "tokenizer.ggml.add_eos_token", false);
        kv_bool(&mut prefix, "tokenizer.ggml.add_space_prefix", false);
        push_string(&mut prefix, "token_embd.weight");
        push_u32(&mut prefix, 2); // dimensions
        push_i64(&mut prefix, 1);
        push_i64(&mut prefix, 1);
        push_i32(&mut prefix, 0); // F32
        push_u64(&mut prefix, 0); // relative data offset
        while !prefix.len().is_multiple_of(32) {
            prefix.push(0);
        }
        let full_len = prefix.len() as u64 + 4;
        let mut full = prefix.clone();
        full.extend_from_slice(&0_f32.to_le_bytes());

        let dir = std::env::temp_dir().join(format!(
            "camelid-tokenize-prefix-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let full_path = dir.join("full.gguf");
        let prefix_path = dir.join("prefix.gguf");
        std::fs::write(&full_path, &full).unwrap();
        std::fs::write(&prefix_path, &prefix).unwrap();

        let ordinary = read_tokenizer_gguf(&full_path, None).expect("ordinary full GGUF");
        let ranged = read_tokenizer_gguf(&prefix_path, Some(full_len))
            .expect("complete tokenizer header prefix");
        let ordinary_ids = Tokenizer::from_gguf(&ordinary)
            .unwrap()
            .encode("h", true, false)
            .unwrap();
        let ranged_ids = Tokenizer::from_gguf(&ranged)
            .unwrap()
            .encode("h", true, false)
            .unwrap();
        assert_eq!(ordinary_ids, vec![2, 3]);
        assert_eq!(ranged_ids, ordinary_ids);

        // The legacy path remains stat-based and must reject a truncated body.
        assert!(read_tokenizer_gguf(&prefix_path, None).is_err());

        let short_path = dir.join("short.gguf");
        std::fs::write(&short_path, &prefix[..prefix.len() - 24]).unwrap();
        let short_err = read_tokenizer_gguf(&short_path, Some(full_len))
            .expect_err("an incomplete descriptor/header must fail closed")
            .to_string();
        assert!(
            short_err.contains("unexpected EOF")
                || short_err.contains("unexpected end of file")
                || short_err.contains("prefix ends before aligned tensor data start"),
            "unexpected truncated-prefix error: {short_err}"
        );

        let malformed_path = dir.join("malformed.gguf");
        let mut malformed = prefix.clone();
        malformed[0] = b'X';
        std::fs::write(&malformed_path, malformed).unwrap();
        assert!(read_tokenizer_gguf(&malformed_path, Some(full_len))
            .expect_err("bad magic must fail closed")
            .to_string()
            .contains("bad magic"));

        assert!(
            read_tokenizer_gguf(&prefix_path, Some(prefix.len() as u64 - 1))
                .expect_err("declared length below physical prefix must fail closed")
                .to_string()
                .contains("larger than declared artifact length")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn serve_parses_ghost_moe_artifact_and_cache_budget() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "serve",
                "--model",
                "gemma4.gguf",
                "--cghost",
                "gemma4.cghost",
                "--expert-cache-mib",
                "2048",
                "--ghost-strict-cache",
                "--no-open",
            ])
            .expect("parse Ghost-MoE serve flags");
            match cli.command {
                Some(Command::Serve {
                    model,
                    cghost,
                    expert_cache_mib,
                    ghost_strict_cache,
                    no_open,
                    ..
                }) => {
                    assert_eq!(model, Some(PathBuf::from("gemma4.gguf")));
                    assert_eq!(cghost, Some(PathBuf::from("gemma4.cghost")));
                    assert_eq!(expert_cache_mib, 2048);
                    assert!(ghost_strict_cache);
                    assert!(no_open);
                }
                other => panic!("expected Serve, got {other:?}"),
            }
        });
    }

    #[test]
    fn serve_uses_buffered_ghost_reads_by_default() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "serve",
                "--model",
                "gemma4.gguf",
                "--cghost",
                "gemma4.cghost",
            ])
            .expect("parse Ghost-MoE serve defaults");
            match cli.command {
                Some(Command::Serve {
                    ghost_strict_cache, ..
                }) => assert!(!ghost_strict_cache),
                other => panic!("expected Serve, got {other:?}"),
            }
        });
    }

    #[test]
    fn serve_parses_lan_chat_only_and_refuses_the_anonymous_override() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "serve",
                "--lan-chat-only",
                "--api-key-file",
                "camelid.key",
                "--allow-cleartext-remote",
                "--no-open",
            ])
            .expect("parse authenticated LAN Chat flags");
            match cli.command {
                Some(Command::Serve {
                    server, no_open, ..
                }) => {
                    assert!(server.lan_chat_only);
                    assert!(server.allow_cleartext_remote);
                    assert_eq!(server.api_key_file, Some(PathBuf::from("camelid.key")));
                    assert!(no_open);
                }
                other => panic!("expected Serve, got {other:?}"),
            }

            Cli::try_parse_from([
                "camelid",
                "serve",
                "--lan-chat-only",
                "--allow-unauthenticated-remote",
            ])
            .expect_err("LAN Chat accepted the anonymous remote override");
        });
    }

    #[test]
    fn lan_key_parses_with_rotation_explicit_and_off_by_default() {
        on_cli_test_stack(|| {
            for (args, expected) in [
                (vec!["camelid", "lan-key"], false),
                (vec!["camelid", "lan-key", "--rotate"], true),
            ] {
                let cli = Cli::try_parse_from(args).unwrap();
                match cli.command {
                    Some(Command::LanKey { rotate }) => assert_eq!(rotate, expected),
                    other => panic!("expected LanKey, got {other:?}"),
                }
            }
        });
    }

    #[test]
    fn lan_key_prints_a_server_command_that_passes_both_remote_guards() {
        assert_eq!(
            lan_chat_serve_command(std::path::Path::new("camelid.key")),
            "camelid serve --lan-chat-only --api-key-file \"camelid.key\" \
             --allow-cleartext-remote --addr <LAPTOP-LAN-IP>:8181 --model <MODEL.gguf>"
        );
    }

    #[test]
    fn remote_chat_actions_parse_a_loopback_backend_and_explicit_tailscale_binary() {
        on_cli_test_stack(|| {
            let start = Cli::try_parse_from(["camelid", "remote-chat", "start"])
                .expect("parse remote Chat start");
            match start.command {
                Some(Command::RemoteChat {
                    action: RemoteChatAction::Start { remote },
                }) => {
                    assert_eq!(remote.backend, "127.0.0.1:8181".parse().unwrap());
                    assert_eq!(remote.tailscale_bin, None);
                }
                other => panic!("expected remote Chat start, got {other:?}"),
            }

            let status = Cli::try_parse_from([
                "camelid",
                "remote-chat",
                "status",
                "--backend",
                "127.0.0.1:9191",
                "--tailscale-bin",
                "tailscale-test-bin",
                "--json",
            ])
            .expect("parse remote Chat status");
            match status.command {
                Some(Command::RemoteChat {
                    action: RemoteChatAction::Status { remote, json },
                }) => {
                    assert_eq!(remote.backend, "127.0.0.1:9191".parse().unwrap());
                    assert_eq!(
                        remote.tailscale_bin,
                        Some(PathBuf::from("tailscale-test-bin"))
                    );
                    assert!(json);
                }
                other => panic!("expected remote Chat status, got {other:?}"),
            }

            Cli::try_parse_from(["camelid", "remote-chat"])
                .expect_err("remote Chat accepted no lifecycle action");
        });
    }
}

use camelid::{
    api, chat,
    cluster::{
        recv_activation_packet, recv_token_feedback, send_activation_packet, send_token_feedback,
    },
    gguf::{read_metadata, read_metadata_with_len, GgufTensorType},
    ghost::{GhostFile, GhostPipelinePrefetcher, GhostPrefetcher},
    inference::{
        speculative::{
            accepted_draft_prefix, ModelDrafter, NGramDrafter, SpeculativeDrafter,
            DEFAULT_MODEL_DRAFT_TOKENS, DEFAULT_NGRAM_DRAFT_TOKENS,
        },
        LlamaForwardTimings, LlamaInferenceSession, LlamaLayerWeights, LlamaLoadedWeights,
        LlamaSampler, Q8ResidencyReport, SamplingConfig,
    },
    metal::detect_metal_device,
    model::{KvCacheQuantization, LlamaModelConfig, LlamaTensorBinding},
    model_source::inspect_model_source,
    tensor::{CpuTensor, Q8_0TensorBlocks, TensorStore},
    tokenizer::Tokenizer,
};

fn lan_chat_serve_command(key_path: &std::path::Path) -> String {
    format!(
        "camelid serve --lan-chat-only --api-key-file \"{}\" \
         --allow-cleartext-remote --addr <LAPTOP-LAN-IP>:8181 --model <MODEL.gguf>",
        key_path.display()
    )
}
use clap::{Args, Parser, Subcommand};
use rayon::ThreadPoolBuilder;
use serde::Serialize;

// Prefer the git describe stamped in by build.rs (e.g. "v0.1.1" or
// "v0.1.1-3-gabcdef-dirty"); fall back to the crate version for builds without
// a git checkout.
const VERSION: &str = match option_env!("CAMELID_GIT_DESCRIBE") {
    Some(describe) => describe,
    None => env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Parser)]
#[command(
    name = "camelid",
    version = VERSION,
    about = "Rust-native local GGUF inference backend"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

/// The action taken when the binary is launched with no subcommand (e.g. a
/// double-click of `camelid.exe`): start the local server and open the chat UI,
/// with the GPU resident-decode path armed automatically. This is what makes the
/// shipped Windows build a single open-and-use app — no terminal, flags, or
/// toggles required.
fn default_launch_command() -> Command {
    Command::Serve {
        addr: "127.0.0.1:8181".parse().expect("valid default serve addr"),
        model: std::env::var_os("CAMELID_MODEL").map(PathBuf::from),
        threads: None,
        parallel_linear_min_outputs: None,
        apple_accelerate_min_elements: None,
        metal_linear: false,
        metal_q8: false,
        log_acceleration: true,
        spec_decode: None,
        spec_draft_model: None,
        spec_draft_tokens: None,
        cghost: std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(PathBuf::from),
        expert_cache_mib: std::env::var("CAMELID_GEMMA4_GHOST_CACHE_MIB")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1024),
        ghost_strict_cache: std::env::var("CAMELID_GEMMA4_GHOST_STRICT_CACHE")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes" | "enabled"
                )
            })
            .unwrap_or(false),
        no_open: false,
        deterministic: false,
        enable_thinking: false,
        models_dir: std::env::var_os("CAMELID_MODELS_DIR").map(PathBuf::from),
        // The double-click launch bypasses clap parsing, so the `env` on the `--gpu`
        // arg does not apply here; read CAMELID_GPU explicitly, like the sibling
        // env-backed fields above. Default (and any unrecognised value) is Auto.
        gpu: GpuMode::from_env(),
        kv_quant: std::env::var("CAMELID_KV_QUANT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_default(),
        server: ServerPolicyArgs::from_env(),
    }
}

#[derive(Clone, Debug, Args)]
struct ServerPolicyArgs {
    /// Require this bearer/API key on network API routes. Prefer
    /// --api-key-file on shared machines so the secret is not present in the
    /// process command line.
    #[arg(long, env = "CAMELID_API_KEY", conflicts_with = "api_key_file")]
    api_key: Option<String>,
    /// Read the bearer/API key from a text file (one trailing newline is
    /// ignored). Mutually exclusive with --api-key.
    #[arg(long, env = "CAMELID_API_KEY_FILE")]
    api_key_file: Option<PathBuf>,
    /// Browser origin allowed to make cross-origin requests. Repeat the flag
    /// or use a comma-separated CAMELID_CORS_ORIGINS value. No origins are
    /// allowed by default; the embedded same-origin UI remains available.
    #[arg(
        long = "cors-origin",
        env = "CAMELID_CORS_ORIGINS",
        value_delimiter = ','
    )]
    cors_origins: Vec<String>,
    /// Explicitly permit an unauthenticated non-loopback listener. Without
    /// this acknowledgement or an API key, Camelid refuses the bind.
    #[arg(
        long,
        env = "CAMELID_ALLOW_UNAUTHENTICATED_REMOTE",
        default_value_t = false
    )]
    allow_unauthenticated_remote: bool,
    /// Expose only the authenticated read surface needed by the embedded Chat
    /// UI plus chat completions. Model mutation, Workspace, Responses, runtime
    /// controls, and every other protected route return a typed 403.
    #[arg(
        long,
        env = "CAMELID_LAN_CHAT_ONLY",
        default_value_t = false,
        conflicts_with = "allow_unauthenticated_remote"
    )]
    lan_chat_only: bool,
    /// Explicitly permit a cleartext non-loopback listener. Without this
    /// acknowledgement or TLS, Camelid refuses the bind, because the API key
    /// and every prompt would otherwise cross the network unencrypted.
    #[arg(long, env = "CAMELID_ALLOW_CLEARTEXT_REMOTE", default_value_t = false)]
    allow_cleartext_remote: bool,
    /// PEM certificate chain for HTTPS. Requires --tls-key.
    #[arg(long, env = "CAMELID_TLS_CERT", requires = "tls_key")]
    tls_cert: Option<PathBuf>,
    /// PEM private key for HTTPS. Requires --tls-cert.
    #[arg(long, env = "CAMELID_TLS_KEY", requires = "tls_cert")]
    tls_key: Option<PathBuf>,
    /// Maximum decoded HTTP request body.
    #[arg(
        long,
        env = "CAMELID_MAX_REQUEST_BODY_BYTES",
        default_value_t = 16 * 1024 * 1024
    )]
    max_request_body_bytes: usize,
    /// Maximum number of prompt tokens accepted by generation endpoints.
    #[arg(long, env = "CAMELID_MAX_PROMPT_TOKENS", default_value_t = 131_072)]
    max_prompt_tokens: usize,
    /// Maximum max_tokens allowance accepted per generation request.
    #[arg(long, env = "CAMELID_MAX_GENERATION_TOKENS", default_value_t = 8_192)]
    max_generation_tokens: u32,
    /// Maximum model download size accepted by the catalog installer.
    #[arg(
        long,
        env = "CAMELID_MAX_DOWNLOAD_BYTES",
        default_value_t = 64 * 1024 * 1024 * 1024
    )]
    max_download_bytes: u64,
}

impl ServerPolicyArgs {
    fn from_env() -> Self {
        fn parsed<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::var(name)
                .ok()
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(default)
        }
        fn enabled(name: &str) -> bool {
            std::env::var(name)
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        }
        Self {
            api_key: std::env::var("CAMELID_API_KEY").ok(),
            api_key_file: std::env::var_os("CAMELID_API_KEY_FILE").map(PathBuf::from),
            cors_origins: std::env::var("CAMELID_CORS_ORIGINS")
                .ok()
                .map(|origins| {
                    origins
                        .split(',')
                        .map(str::trim)
                        .filter(|origin| !origin.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            allow_unauthenticated_remote: enabled("CAMELID_ALLOW_UNAUTHENTICATED_REMOTE"),
            lan_chat_only: enabled("CAMELID_LAN_CHAT_ONLY"),
            allow_cleartext_remote: enabled("CAMELID_ALLOW_CLEARTEXT_REMOTE"),
            tls_cert: std::env::var_os("CAMELID_TLS_CERT").map(PathBuf::from),
            tls_key: std::env::var_os("CAMELID_TLS_KEY").map(PathBuf::from),
            max_request_body_bytes: parsed("CAMELID_MAX_REQUEST_BODY_BYTES", 16 * 1024 * 1024),
            max_prompt_tokens: parsed("CAMELID_MAX_PROMPT_TOKENS", 131_072),
            max_generation_tokens: parsed("CAMELID_MAX_GENERATION_TOKENS", 8_192),
            max_download_bytes: parsed("CAMELID_MAX_DOWNLOAD_BYTES", 64 * 1024 * 1024 * 1024),
        }
    }

    fn into_serve_options(self) -> api::ServeOptions {
        api::ServeOptions {
            api_key: self.api_key,
            api_key_file: self.api_key_file,
            cors_origins: self.cors_origins,
            allow_unauthenticated_remote: self.allow_unauthenticated_remote,
            allow_cleartext_remote: self.allow_cleartext_remote,
            tls_cert: self.tls_cert,
            tls_key: self.tls_key,
            max_request_body_bytes: self.max_request_body_bytes,
            max_prompt_tokens: self.max_prompt_tokens,
            max_generation_tokens: self.max_generation_tokens,
            max_download_bytes: self.max_download_bytes,
            api_surface: if self.lan_chat_only {
                api::ApiSurface::LanChatOnly
            } else {
                api::ApiSurface::Full
            },
        }
    }
}

#[derive(Clone, Debug, Args)]
struct RemoteChatArgs {
    /// Existing authenticated LAN Chat listener to publish privately. Tailscale
    /// Serve supports a local HTTP proxy only on 127.0.0.1.
    #[arg(long, default_value = "127.0.0.1:8181")]
    backend: SocketAddr,
    /// Absolute path to the Tailscale CLI. Normally discovered from the standard
    /// install location or PATH.
    #[arg(long, env = "CAMELID_TAILSCALE_BIN")]
    tailscale_bin: Option<PathBuf>,
}

impl RemoteChatArgs {
    fn into_options(self) -> camelid::remote_chat::RemoteChatOptions {
        camelid::remote_chat::RemoteChatOptions {
            backend: self.backend,
            tailscale_bin: self.tailscale_bin,
        }
    }
}

/// Find the effective default GGUF when `serve` starts without an explicit
/// `--model`. The configured models directory wins (this is where the desktop
/// sidecar stores downloads), followed by the historical shipped/CWD layouts.
/// Within each directory an explicit saved preference wins; otherwise the first
/// local GGUF is the zero-configuration default.
fn auto_select_model(configured_models_dir: Option<&std::path::Path>) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(dir) = configured_models_dir {
        dirs.push(dir.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
            dirs.push(parent.to_path_buf());
        }
    }
    dirs.push(PathBuf::from("models"));
    dirs.push(PathBuf::from("."));
    for dir in dirs {
        if let Some(choice) = camelid::model_default::effective_default_model(&dir) {
            return Some(choice.path);
        }
    }
    None
}

#[cfg(test)]
mod auto_select_model_tests {
    use super::auto_select_model;

    #[test]
    fn configured_desktop_models_directory_wins_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("first.gguf"), []).unwrap();
        std::fs::write(dir.path().join("preferred.gguf"), []).unwrap();
        camelid::model_default::set_default_model(dir.path(), "preferred.gguf").unwrap();

        assert_eq!(
            auto_select_model(Some(dir.path())),
            Some(dir.path().join("preferred.gguf"))
        );
    }
}

/// Windows + CUDA: make the NVIDIA runtime DLLs (NVRTC etc.) loadable without the
/// user having to add the CUDA `bin` directory to PATH. Looks in two places, in
/// priority order: (1) the running exe's OWN directory, so a self-contained
/// download that ships the NVRTC redistributable DLLs beside `camelid.exe` runs
/// on the GPU with only the NVIDIA *driver* installed (no CUDA toolkit); and
/// (2) an installed toolkit (via `CUDA_PATH*` or the standard install root). The
/// matching `bin`/exe dirs are prepended to the process PATH before any GPU code
/// runs. No-op if neither is present (the engine then falls back to CPU).
#[cfg(all(windows, feature = "cuda"))]
fn ensure_cuda_runtime_on_path() {
    use std::path::{Path, PathBuf};

    let mut candidates: Vec<PathBuf> = Vec::new();
    // The exe's own directory goes FIRST so a shipped, version-matched NVRTC pair
    // (staged by scripts/package-windows-cuda.ps1) wins over any — possibly
    // mismatched — system-installed toolkit. Windows already searches the exe dir
    // for `LoadLibrary`, but adding it to PATH explicitly is robust against
    // altered DLL search policies and makes the self-contained path intentional.
    // Only add it when NVRTC is actually present, to avoid polluting PATH (e.g. a
    // dev build under target/debug, where the DLLs are not staged).
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        if dir_has_nvrtc(&dir) {
            candidates.push(dir);
        }
    }
    for (key, value) in std::env::vars_os() {
        let key = key.to_string_lossy();
        if key == "CUDA_PATH" || key.starts_with("CUDA_PATH_V") {
            let bin = Path::new(&value).join("bin");
            if bin.is_dir() {
                candidates.push(bin);
            }
        }
    }
    let root = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut versions: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        versions.sort();
        for version in versions.into_iter().rev() {
            let bin = version.join("bin");
            if bin.is_dir() {
                candidates.push(bin);
            }
        }
    }
    if candidates.is_empty() {
        return;
    }
    let current = std::env::var_os("PATH").unwrap_or_default();
    let current_lower = current.to_string_lossy().to_lowercase();
    let mut prefix = std::ffi::OsString::new();
    for bin in &candidates {
        if !current_lower.contains(&bin.to_string_lossy().to_lowercase()) {
            prefix.push(bin);
            prefix.push(";");
        }
    }
    if prefix.is_empty() {
        return;
    }
    prefix.push(current);
    std::env::set_var("PATH", prefix);
}

/// Whether `dir` contains an NVRTC runtime DLL (`nvrtc64_*.dll`). Used to decide
/// if the exe's own directory carries a shipped, self-contained CUDA runtime.
#[cfg(all(windows, feature = "cuda"))]
fn dir_has_nvrtc(dir: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .file_name()
            .to_string_lossy()
            .to_ascii_lowercase()
            .starts_with("nvrtc64_")
    })
}

#[cfg(not(all(windows, feature = "cuda")))]
fn ensure_cuda_runtime_on_path() {}

/// Stop cudarc's missing-library panics from printing, without silencing anything
/// else.
///
/// cudarc's lazy loaders panic when a CUDA library cannot be dlopen'd. We already
/// catch those and fall back to the CPU path, so on a host with an NVIDIA driver
/// but no CUDA toolkit the fallback is *working as designed* — but the default
/// panic hook still prints the panic plus a `RUST_BACKTRACE` hint on the way
/// through, so a healthy CPU-only start reads like a crash. That population is
/// now large: CUDA is compiled into the default x86_64 Linux build, and NVRTC
/// ships with the toolkit rather than the driver.
///
/// Installed ONCE, and it filters by panic origin rather than swapping the hook
/// around each call: a scoped swap would be process-global for its duration and
/// could swallow an unrelated thread's panic message (model loads run while the
/// server handles other requests). Every non-cudarc panic reaches the previous
/// hook untouched.
fn quiet_cudarc_loader_panics() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let from_cudarc = info
                .location()
                .is_some_and(|loc| loc.file().replace('\\', "/").contains("/cudarc-"));
            if !from_cudarc {
                previous(info);
            }
        }));
    });
}

// Optimus / Enduro hint variables, exported from the exe via build.rs. A laptop's
// hybrid-graphics driver reads these at process start and routes the process to
// the discrete NVIDIA / AMD GPU rather than the integrated Intel one. Without this
// Windows assigns the process to the iGPU by default, so Task Manager shows the
// Intel GPU "busy" even though CUDA compute runs (and can only run) on the NVIDIA
// card — the source of the "it's on Intel" confusion.
#[cfg(windows)]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;
#[cfg(windows)]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

/// Tell Windows to run this executable on the high-performance (discrete NVIDIA)
/// GPU — the same setting as Settings → System → Display → Graphics → set the app
/// to "High performance". Writing it (HKCU, no admin needed) makes Windows and
/// Task Manager attribute the app to the NVIDIA GPU instead of the integrated
/// Intel one. Idempotent and best-effort; failures are ignored.
#[cfg(windows)]
fn pin_to_high_performance_gpu() {
    use std::os::windows::process::CommandExt;
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let exe = exe.to_string_lossy().to_string();
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = std::process::Command::new("reg")
        .args([
            "add",
            r"HKCU\Software\Microsoft\DirectX\UserGpuPreferences",
            "/v",
            &exe,
            "/t",
            "REG_SZ",
            "/d",
            "GpuPreference=2;",
            "/f",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
}

#[cfg(not(windows))]
fn pin_to_high_performance_gpu() {}

/// `camelid gait <action>` — GAIT cache maintenance subcommands.
#[derive(Debug, Subcommand)]
enum AgentAction {
    /// Run one goal to completion with no human present, print the final answer
    /// to stdout, and exit 0 (answered) / 1 (failed or blocked) / 3
    /// (inconclusive: step-capped, aborted, or no longer making progress).
    ///
    /// Progress narrates on stderr so stdout carries only the answer. With no
    /// operator to confirm anything, every approval-gated tool is DENIED unless
    /// --today-is-a-good-day-to-die (alias --yolo) is passed, and that flag is
    /// refused under CAMELID_PRODUCTION.
    Exec {
        /// The goal. Omit to read it from stdin.
        goal: Option<String>,
        /// GGUF to drive. Must be a tool-capable supported row.
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8231")]
        addr: SocketAddr,
        /// Sandbox root for the agent's file tools (default: cwd).
        #[arg(long)]
        workdir: Option<PathBuf>,
        #[arg(long, default_value_t = 25)]
        max_steps: usize,
        #[arg(long, default_value_t = 1024)]
        max_tokens: u32,
        /// Auto-approve write/network tools (exec tools still gated). Refused
        /// under CAMELID_PRODUCTION.
        #[arg(long, default_value_t = false)]
        auto_approve: bool,
        /// UNATTENDED: auto-approve everything including exec tools — today is
        /// a good day to die. Refused under CAMELID_PRODUCTION. (`--yolo` is a
        /// compatible alias.)
        #[arg(
            long = "today-is-a-good-day-to-die",
            visible_alias = "yolo",
            default_value_t = false
        )]
        yolo: bool,
        #[arg(long, default_value_t = false)]
        allow_net: bool,
        #[arg(long, default_value_t = false)]
        allow_fs: bool,
        #[arg(long, default_value_t = false)]
        allow_mcp: bool,
        /// Trust and immediately execute this named server from the workspace's
        /// camelid.mcp.json. Repeat for each reviewed server. Has no effect
        /// unless --allow-mcp is also present.
        #[arg(long = "trust-mcp-server", action = clap::ArgAction::Append)]
        trust_mcp_server: Vec<String>,
        #[arg(long, default_value = "sandboxed")]
        shell_sandbox: String,
        #[arg(long, default_value_t = 30)]
        shell_timeout: u64,
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum WorkspaceAction {
    /// Ask one grounded, read-only question. Use --thread to continue a saved conversation.
    Ask {
        /// Local folder that confines all Workspace file tools.
        workspace: PathBuf,
        /// Question or analysis request.
        goal: String,
        /// Resume this durable Workspace conversation.
        #[arg(long)]
        thread: Option<String>,
        #[arg(long, default_value_t = 12)]
        max_steps: usize,
        #[arg(long, default_value_t = 512)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
    },
    /// List durable conversations for a local folder and the active model.
    Threads {
        #[arg(default_value = ".")]
        workspace: PathBuf,
    },
    /// Print a durable conversation transcript.
    Show {
        thread: String,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
    },
    /// Compact a durable conversation, or restore its previous compaction state.
    Compact {
        thread: String,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        #[arg(long, default_value_t = false)]
        undo: bool,
    },
    /// Permanently delete one durable conversation.
    Delete {
        thread: String,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
    },
}

#[cfg(test)]
mod workspace_command_tests {
    use super::*;

    fn on_cli_test_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("workspace-cli-parse-test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(test)
            .expect("spawn Workspace CLI parse test")
            .join()
            .expect("Workspace CLI parse test panicked");
    }

    #[test]
    fn workspace_ask_parses_durable_resume_and_limits() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "workspace",
                "ask",
                ".",
                "inspect README.md",
                "--thread",
                "workspace-123",
                "--max-steps",
                "8",
                "--max-tokens",
                "256",
            ])
            .unwrap();
            match cli.command {
                Some(Command::Workspace {
                    action:
                        WorkspaceAction::Ask {
                            workspace,
                            goal,
                            thread,
                            max_steps,
                            max_tokens,
                            ..
                        },
                    ..
                }) => {
                    assert_eq!(workspace, PathBuf::from("."));
                    assert_eq!(goal, "inspect README.md");
                    assert_eq!(thread.as_deref(), Some("workspace-123"));
                    assert_eq!(max_steps, 8);
                    assert_eq!(max_tokens, 256);
                }
                other => panic!("expected workspace ask, got {other:?}"),
            }
        });
    }

    #[test]
    fn workspace_threads_uses_current_directory_and_json_is_global() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from(["camelid", "workspace", "threads", "--json"]).unwrap();
            match cli.command {
                Some(Command::Workspace {
                    json: true,
                    action: WorkspaceAction::Threads { workspace },
                    ..
                }) => assert_eq!(workspace, PathBuf::from(".")),
                other => panic!("expected workspace threads, got {other:?}"),
            }
        });
    }

    #[test]
    fn workspace_compaction_undo_is_explicit() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "workspace",
                "compact",
                "workspace-123",
                "--workspace",
                "project",
                "--undo",
            ])
            .unwrap();
            match cli.command {
                Some(Command::Workspace {
                    action:
                        WorkspaceAction::Compact {
                            thread,
                            workspace,
                            undo: true,
                        },
                    ..
                }) => {
                    assert_eq!(thread, "workspace-123");
                    assert_eq!(workspace, PathBuf::from("project"));
                }
                other => panic!("expected workspace compact --undo, got {other:?}"),
            }
        });
    }
}

#[derive(Debug, Subcommand)]
enum GaitAction {
    /// Clear the GAIT cache (profiles, quarantine, in-progress markers, and the
    /// DISABLE kill-file) under %LOCALAPPDATA%\Camelid\gait, fully reverting to
    /// the baseline path.
    Reset,
}

/// GPU acceleration mode for `serve` (`--gpu`, env `CAMELID_GPU`). Seeds the runtime
/// GPU switches at startup for headless/agent use where there is no UI to click; the
/// UI toggle (`POST /api/runtime/gpu`) can still flip state live afterwards.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum GpuMode {
    /// Use an available CUDA or Metal GPU; fall back to CPU otherwise (default).
    /// Leaves the master switch to lazy-seed from platform capability and the
    /// CUDA-only hybrid Q8 matmul from CAMELID_CUDA_Q8.
    Auto,
    /// Force the GPU path on (still no-ops at runtime without a device).
    On,
    /// Force the CPU reference path; never use the GPU even if one is present.
    Off,
}

impl GpuMode {
    /// Resolve from `CAMELID_GPU` for the double-click launch path, which bypasses clap
    /// arg parsing (so the `env` on the `--gpu` arg does not apply there). Values match
    /// the clap value names (`auto`/`on`/`off`, case-insensitive); an unset or
    /// unrecognised value falls back to `Auto`, identical to the clap default.
    fn from_env() -> Self {
        match std::env::var("CAMELID_GPU") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "on" => GpuMode::On,
                "off" => GpuMode::Off,
                _ => GpuMode::Auto,
            },
            Err(_) => GpuMode::Auto,
        }
    }
}

/// Resolve the explicit startup value for the shared GPU switch. `None` keeps
/// auto-detection lazy. Deterministic mode is authoritative over even `--gpu on`.
fn resolved_gpu_switch(gpu: GpuMode, deterministic: bool) -> Option<bool> {
    if deterministic {
        return Some(false);
    }
    match gpu {
        GpuMode::Auto => None,
        GpuMode::On => Some(true),
        GpuMode::Off => Some(false),
    }
}

/// Read the metadata source used by the tokenizer parity CLI.
///
/// The ordinary path deliberately remains `read_metadata`, whose bounds checks
/// use the physical file length. `--declared-len` is the explicit, narrow escape
/// hatch for an immutable HTTP Range prefix: the shared GGUF prefix parser still
/// reads every metadata value and tensor descriptor and checks tensor ranges
/// against the source artifact's pinned full length. Requiring the physical
/// prefix to cover the aligned data-start boundary makes partial metadata or a
/// truncated descriptor block fail closed before tokenizer construction.
fn read_tokenizer_gguf(
    model: &std::path::Path,
    declared_len: Option<u64>,
) -> anyhow::Result<camelid::gguf::GgufFile> {
    let Some(declared_len) = declared_len else {
        return Ok(read_metadata(model)?);
    };

    let prefix_len = std::fs::metadata(model)?.len();
    anyhow::ensure!(
        prefix_len <= declared_len,
        "GGUF prefix is {prefix_len} bytes, larger than declared artifact length {declared_len}"
    );
    let gguf = read_metadata_with_len(model, declared_len)?;
    anyhow::ensure!(
        prefix_len >= gguf.data_start_offset,
        "GGUF prefix ends before aligned tensor data start: prefix has {prefix_len} bytes, header requires {}",
        gguf.data_start_offset
    );
    Ok(gguf)
}

/// Parse the placement mode used by a resident or request-sending fabric.
fn route_mode(raw: &str) -> anyhow::Result<camelid::fabric::RouteMode> {
    match raw {
        "throughput" => Ok(camelid::fabric::RouteMode::Throughput),
        "completion-time" => Ok(camelid::fabric::RouteMode::CompletionTime),
        "affinity" => Ok(camelid::fabric::RouteMode::Affinity),
        other => anyhow::bail!(
            "unknown mode `{other}`; expected `throughput`, `completion-time` or `affinity`"
        ),
    }
}

/// A one-shot command owns no reusable service history, so it cannot honestly
/// offer the resident completion-time policy.
fn stateless_route_mode(raw: &str) -> anyhow::Result<camelid::fabric::RouteMode> {
    match route_mode(raw)? {
        camelid::fabric::RouteMode::CompletionTime => anyhow::bail!(
            "mode `completion-time` learns across completed requests and is available only to the resident `fabric serve` command"
        ),
        mode => Ok(mode),
    }
}

/// Build a fabric from whichever way the operator named its nodes.
///
/// Every `fabric` subcommand takes both forms, so the file an operator already
/// maintains for `fabric serve` is the same one they can inspect with
/// `fabric status`. Clap makes the two mutually exclusive and requires one, so
/// exactly one arrives populated here.
///
/// A file that cannot be read is an error rather than an empty fabric: on the
/// CLI that is a message instead of a table of nothing, and on `serve` it stops
/// the proxy before it announces an address.
fn fabric_from(
    nodes: Vec<String>,
    nodes_file: Option<PathBuf>,
) -> anyhow::Result<camelid::fabric::Fabric> {
    match nodes_file {
        Some(path) => Ok(camelid::fabric::Fabric::from_node_file(path)?),
        None => Ok(camelid::fabric::Fabric::new(
            camelid::fabric::parse_fabric(&nodes).map_err(|error| anyhow::anyhow!("{error}"))?,
        )),
    }
}

fn configure_node_transport(
    fabric: camelid::fabric::Fabric,
    transport: &NodeTransportArgs,
) -> anyhow::Result<camelid::fabric::Fabric> {
    Ok(fabric.with_node_transport(
        transport.node_tls_ca.as_deref(),
        transport.allow_cleartext_node_transport,
    )?)
}

/// Resolve the token the fabric authenticates to its nodes with.
///
/// Deliberately not `#[arg(env = "CAMELID_API_KEY")]`: clap prints an env var's
/// current value in `--help`, which would put the token on the terminal.
fn fabric_bearer(flag: Option<String>) -> Option<String> {
    resolve_bearer(flag, std::env::var("CAMELID_API_KEY").ok())
}

/// Choose between an explicit flag and the environment. Pure, so the precedence
/// is tested without a process-wide variable.
///
/// The flag wins; otherwise `CAMELID_API_KEY`, the same variable the server
/// reads its own key from, so a shell configured for one node needs no second
/// setting. Trimmed and emptiness-checked the way the server treats its own key,
/// so a value that arrived with a trailing newline still matches, and an empty
/// one reads as "no token" rather than becoming a bare `Bearer `.
fn resolve_bearer(flag: Option<String>, from_env: Option<String>) -> Option<String> {
    flag.or(from_env)
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

/// Actions over a fabric of independent Camelid nodes.
///
/// A fabric places whole requests on nodes that each own a complete model and
/// session, so nothing crosses the network inside the token loop. This is the
/// throughput complement to `serve-distributed`, which shards one model's layers
/// across machines and is slower than a single node by construction.
#[derive(Debug, Clone, Args)]
struct NodeTransportArgs {
    /// PEM CA bundle that authenticates every node. Node hostnames and IP
    /// literals are verified against the certificate SAN before any bearer or
    /// request bytes are sent.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with = "allow_cleartext_node_transport"
    )]
    node_tls_ca: Option<PathBuf>,
    /// Permit direct cleartext transport to a node that resolves outside
    /// loopback. Without this acknowledgement, plaintext is limited to local
    /// nodes and operator-owned tunnels.
    #[arg(
        long,
        env = "CAMELID_ALLOW_CLEARTEXT_NODE_TRANSPORT",
        default_value_t = false
    )]
    allow_cleartext_node_transport: bool,
}

#[derive(Debug, Subcommand)]
enum FabricAction {
    /// Probe every node once and report what the fabric can serve.
    Status {
        /// A node, as `LABEL=HOST[:PORT]`. Repeat for each node.
        #[arg(
            long = "node",
            required_unless_present = "nodes_file",
            conflicts_with = "nodes_file",
            value_name = "LABEL=HOST[:PORT]"
        )]
        nodes: Vec<String>,
        /// Read the nodes from the same file `fabric serve` takes, one
        /// `LABEL=HOST[:PORT]` per line.
        #[arg(long, value_name = "PATH")]
        nodes_file: Option<PathBuf>,
        /// Bearer token for nodes started with an API key. Falls back to
        /// CAMELID_API_KEY. Prefer the variable on a shared machine, so the
        /// secret is not in the process command line.
        #[arg(long, value_name = "TOKEN")]
        bearer: Option<String>,
        #[command(flatten)]
        transport: NodeTransportArgs,
        /// Per-node probe budget. Nodes are probed concurrently, so this bounds
        /// the whole command, not each node in turn.
        #[arg(long, default_value_t = 2000)]
        timeout_ms: u64,
        /// Emit the observation as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Show which node a request would be placed on, without sending it.
    ///
    /// Placement is deterministic, so this dry run predicts the real decision.
    Route {
        #[arg(
            long = "node",
            required_unless_present = "nodes_file",
            conflicts_with = "nodes_file",
            value_name = "LABEL=HOST[:PORT]"
        )]
        nodes: Vec<String>,
        /// Read the nodes from the same file `fabric serve` takes.
        #[arg(long, value_name = "PATH")]
        nodes_file: Option<PathBuf>,
        /// `throughput` spreads independent requests; `affinity` keeps a session
        /// on the node whose prefix and KV cache are already warm. The learned
        /// `completion-time` mode is unavailable to this stateless dry run.
        #[arg(long, default_value = "throughput")]
        mode: String,
        /// Only consider nodes serving this exact model id.
        #[arg(long)]
        model: Option<String>,
        /// Label of the node that served this session previously.
        #[arg(long)]
        sticky: Option<String>,
        /// Bearer token for nodes started with an API key. Falls back to
        /// CAMELID_API_KEY.
        #[arg(long, value_name = "TOKEN")]
        bearer: Option<String>,
        #[command(flatten)]
        transport: NodeTransportArgs,
        #[arg(long, default_value_t = 2000)]
        timeout_ms: u64,
        #[arg(long)]
        json: bool,
    },
    /// Send one chat request through the fabric and print the answer.
    ///
    /// Placement, then forward: the request crosses the network once, not once
    /// per token.
    Run {
        #[arg(
            long = "node",
            required_unless_present = "nodes_file",
            conflicts_with = "nodes_file",
            value_name = "LABEL=HOST[:PORT]"
        )]
        nodes: Vec<String>,
        /// Read the nodes from the same file `fabric serve` takes.
        #[arg(long, value_name = "PATH")]
        nodes_file: Option<PathBuf>,
        /// The user message to send.
        #[arg(long)]
        prompt: String,
        /// `throughput` ranks queue depth; `affinity` prefers the requested
        /// sticky node. Use resident `fabric serve` for learned placement.
        #[arg(long, default_value = "throughput")]
        mode: String,
        /// Require a node serving this exact model id. When omitted, the fabric
        /// picks a node and uses whatever that node has loaded.
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        sticky: Option<String>,
        /// Bearer token for nodes started with an API key. Falls back to
        /// CAMELID_API_KEY. Without it, a node that requires a key observes as
        /// ready and then answers this request with 401.
        #[arg(long, value_name = "TOKEN")]
        bearer: Option<String>,
        #[command(flatten)]
        transport: NodeTransportArgs,
        #[arg(long, default_value_t = 64)]
        max_tokens: u32,
        /// Per-node health probe budget.
        #[arg(long, default_value_t = 2000)]
        timeout_ms: u64,
        /// Budget for the generation itself, which can legitimately take minutes.
        #[arg(long, default_value_t = 300)]
        forward_timeout_s: u64,
        #[arg(long)]
        json: bool,
    },
    /// Run a resident HTTP proxy in front of the fabric.
    ///
    /// Every request to `/v1/chat/completions` is placed and forwarded through
    /// the same placement `fabric run` uses, so a client can point at one
    /// address instead of the operator invoking the CLI per request. A request
    /// asking for `stream: true` is relayed as server-sent events, so a stock
    /// OpenAI client works against this address unchanged.
    ///
    /// Client authentication is separate from the bearer the proxy sends to
    /// its nodes. The proxy binds loopback by default; a routable address is
    /// refused unless it is both authenticated and encrypted, or whichever of
    /// those it is missing is acknowledged explicitly.
    Serve {
        #[arg(
            long = "node",
            required_unless_present = "nodes_file",
            conflicts_with = "nodes_file",
            value_name = "LABEL=HOST[:PORT]"
        )]
        nodes: Vec<String>,
        /// Read the nodes from a file instead, one `LABEL=HOST[:PORT]` per
        /// line, and re-read it as it changes.
        ///
        /// Blank lines and whole-line `#` comments are ignored. Editing the
        /// file adds or removes a machine without restarting the proxy, so
        /// doing so does not drop the requests already in flight.
        #[arg(long, value_name = "PATH")]
        nodes_file: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1:8282")]
        addr: SocketAddr,
        /// Default placement mode. `completion-time` learns successful service
        /// time per node, model and route; it falls back to queue depth until
        /// every candidate is warm. A sticky header still requests affinity.
        #[arg(long, default_value = "throughput")]
        mode: String,
        /// Bearer token for nodes started with an API key. Falls back to
        /// CAMELID_API_KEY. Without it, such a node observes as ready and then
        /// answers every forwarded request with 401.
        #[arg(long, value_name = "TOKEN")]
        bearer: Option<String>,
        #[command(flatten)]
        transport: NodeTransportArgs,
        /// Require this key from clients of the proxy. This is the opposite
        /// direction to --bearer, and deliberately does NOT read CAMELID_API_KEY:
        /// on this command that variable already names the token sent to nodes,
        /// so one value would silently configure both directions.
        #[arg(long, value_name = "KEY", conflicts_with = "api_key_file")]
        api_key: Option<String>,
        /// Read the client key from a text file (surrounding whitespace is
        /// ignored). Prefer this on a shared machine, so the secret is not in
        /// the process command line.
        #[arg(long, value_name = "PATH")]
        api_key_file: Option<PathBuf>,
        /// Per-node health probe budget.
        #[arg(long, default_value_t = 2000)]
        timeout_ms: u64,
        /// How long a fabric observation may be reused before the nodes are
        /// probed again.
        ///
        /// Without a bound this probes every node on every request, and one
        /// node that black-holes adds the whole probe budget to all of them.
        /// Set 0 to probe on every request.
        #[arg(long, default_value_t = 500)]
        observation_max_age_ms: u64,
        /// How many nodes one request may be sent to.
        ///
        /// A request placed on a node that has gone since it was last observed
        /// is placed again on another node that serves the same model, as long
        /// as that node was never reached, so it cannot have started the work.
        /// Set 1 to fail the request instead.
        #[arg(long, default_value_t = camelid::fabric::DEFAULT_MAX_FORWARD_ATTEMPTS)]
        max_forward_attempts: usize,
        /// Budget for the generation itself, which can legitimately take
        /// minutes. On a streaming request it bounds the wait for the node's
        /// response head and any later gap in which the node sends nothing;
        /// every event resets the second, so a long stream is never cut short.
        #[arg(long, default_value_t = 300)]
        forward_timeout_s: u64,
        /// Explicitly permit an unauthenticated non-loopback listener. Without
        /// this acknowledgement the proxy refuses the bind, because anything
        /// that can reach it can drive every node in the fabric.
        #[arg(
            long,
            env = "CAMELID_ALLOW_UNAUTHENTICATED_REMOTE",
            default_value_t = false
        )]
        allow_unauthenticated_remote: bool,
        /// PEM certificate chain this proxy presents to its clients.
        /// Requires --tls-key.
        #[arg(long, value_name = "PATH")]
        tls_cert: Option<PathBuf>,
        /// PEM private key for --tls-cert.
        #[arg(long, value_name = "PATH")]
        tls_key: Option<PathBuf>,
        /// Explicitly permit a cleartext non-loopback listener. Without this
        /// acknowledgement the proxy refuses the bind, because the client key
        /// and every prompt would cross the network unencrypted.
        #[arg(long, env = "CAMELID_ALLOW_CLEARTEXT_REMOTE", default_value_t = false)]
        allow_cleartext_remote: bool,
        /// JSON file naming the clients this proxy serves, of the form
        /// {"clients":[{"name":"...","key":"..."}]}.
        ///
        /// Each client gets its own key, and the name is what the access log
        /// records. Removing an entry revokes that client without a restart
        /// and without disturbing the others.
        #[arg(long, value_name = "PATH", conflicts_with_all = ["api_key", "api_key_file"])]
        client_keys: Option<PathBuf>,
    },
}

#[cfg(test)]
mod fabric_command_tests {
    use super::*;

    /// The parsed `Cli` is large enough to want more than a test thread's
    /// default stack, the same reason `workspace_command_tests` does this.
    fn on_cli_test_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("fabric-cli-parse-test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(test)
            .expect("spawn fabric CLI parse test")
            .join()
            .expect("fabric CLI parse test panicked");
    }

    #[test]
    fn an_explicit_flag_beats_the_environment() {
        assert_eq!(
            resolve_bearer(Some("from-flag".into()), Some("from-env".into())),
            Some("from-flag".to_string())
        );
    }

    #[test]
    fn the_environment_is_the_fallback_not_the_only_source() {
        assert_eq!(
            resolve_bearer(None, Some("from-env".into())),
            Some("from-env".to_string())
        );
        assert_eq!(resolve_bearer(None, None), None);
    }

    #[test]
    fn a_token_is_trimmed_the_way_the_server_trims_its_own_key() {
        // A key exported from a file commonly arrives with a trailing newline;
        // sending that verbatim would never match the server's trimmed key.
        assert_eq!(
            resolve_bearer(None, Some("  s3cret\n".into())),
            Some("s3cret".to_string())
        );
    }

    #[test]
    fn an_empty_or_blank_token_reads_as_no_token() {
        // `CAMELID_API_KEY=` is set-but-empty; treating it as a token would send
        // a bare `Bearer ` that the server rejects for a confusing reason.
        assert_eq!(resolve_bearer(None, Some(String::new())), None);
        assert_eq!(resolve_bearer(None, Some("   ".into())), None);
        assert_eq!(resolve_bearer(Some("  ".into()), None), None);
    }

    #[test]
    fn completion_time_is_available_only_to_the_resident_command() {
        assert_eq!(
            route_mode("completion-time").expect("resident mode parses"),
            camelid::fabric::RouteMode::CompletionTime
        );
        assert!(stateless_route_mode("completion-time").is_err());

        on_cli_test_stack(|| {
            let argv = [
                "camelid",
                "fabric",
                "serve",
                "--node",
                "a=127.0.0.1",
                "--mode",
                "completion-time",
            ];
            Cli::try_parse_from(argv).expect("resident mode must parse");
        });
    }

    #[test]
    fn every_fabric_subcommand_accepts_a_bearer() {
        // `run` and `serve` are the ones that 401 without it, but `status` and
        // `route` must take it too or they would keep predicting what the
        // forwarding paths cannot do.
        on_cli_test_stack(|| {
            for argv in [
                vec!["camelid", "fabric", "status"],
                vec!["camelid", "fabric", "route"],
                vec!["camelid", "fabric", "run", "--prompt", "hi"],
                vec!["camelid", "fabric", "serve"],
            ] {
                let mut argv = argv;
                argv.extend(["--node", "a=127.0.0.1", "--bearer", "s3cret"]);
                let cli = Cli::try_parse_from(&argv).expect("parses");
                let bearer = match cli.command {
                    Some(Command::Fabric { action }) => match action {
                        FabricAction::Status { bearer, .. }
                        | FabricAction::Route { bearer, .. }
                        | FabricAction::Run { bearer, .. }
                        | FabricAction::Serve { bearer, .. } => bearer,
                    },
                    other => panic!("expected a fabric command, got {other:?}"),
                };
                assert_eq!(bearer.as_deref(), Some("s3cret"), "{argv:?}");
            }
        });
    }

    #[test]
    fn every_fabric_subcommand_has_the_same_node_transport_contract() {
        on_cli_test_stack(|| {
            for argv in [
                vec!["camelid", "fabric", "status"],
                vec!["camelid", "fabric", "route"],
                vec!["camelid", "fabric", "run", "--prompt", "hi"],
                vec!["camelid", "fabric", "serve"],
            ] {
                let mut tls = argv.clone();
                tls.extend(["--node", "a=node.example", "--node-tls-ca", "node-ca"]);
                let cli = Cli::try_parse_from(&tls).expect("TLS mode parses");
                let transport = match cli.command {
                    Some(Command::Fabric { action }) => match action {
                        FabricAction::Status { transport, .. }
                        | FabricAction::Route { transport, .. }
                        | FabricAction::Run { transport, .. }
                        | FabricAction::Serve { transport, .. } => transport,
                    },
                    other => panic!("expected a fabric command, got {other:?}"),
                };
                assert_eq!(
                    transport.node_tls_ca.as_deref(),
                    Some(std::path::Path::new("node-ca")),
                    "{tls:?}"
                );
                assert!(!transport.allow_cleartext_node_transport, "{tls:?}");

                let mut contradictory = argv;
                contradictory.extend([
                    "--node",
                    "a=node.example",
                    "--node-tls-ca",
                    "node-ca",
                    "--allow-cleartext-node-transport",
                ]);
                Cli::try_parse_from(&contradictory)
                    .expect_err("TLS and cleartext acknowledgement conflict");
            }
        });
    }

    /// The node file is the set an operator maintains for `fabric serve`.
    /// Every subcommand has to read it, or inspecting the fabric means retyping
    /// by hand the machines the proxy already knows about — and the two answers
    /// can then disagree.
    #[test]
    fn every_fabric_subcommand_accepts_a_node_file() {
        on_cli_test_stack(|| {
            for argv in [
                vec!["camelid", "fabric", "status"],
                vec!["camelid", "fabric", "route"],
                vec!["camelid", "fabric", "run", "--prompt", "hi"],
                vec!["camelid", "fabric", "serve"],
            ] {
                let mut argv = argv;
                argv.extend(["--nodes-file", "fabric.nodes"]);
                let cli = Cli::try_parse_from(&argv).expect("parses");
                let nodes_file = match cli.command {
                    Some(Command::Fabric { action }) => match action {
                        FabricAction::Status { nodes_file, .. }
                        | FabricAction::Route { nodes_file, .. }
                        | FabricAction::Run { nodes_file, .. }
                        | FabricAction::Serve { nodes_file, .. } => nodes_file,
                    },
                    other => panic!("expected a fabric command, got {other:?}"),
                };
                assert_eq!(
                    nodes_file.as_deref(),
                    Some(std::path::Path::new("fabric.nodes")),
                    "{argv:?}"
                );
            }
        });
    }

    /// Two answers to "which machines" must not be configurable at once: the
    /// file is the one that can change while the proxy runs, so a stray
    /// `--node` silently winning would leave an operator editing a file that
    /// nothing reads.
    #[test]
    fn naming_nodes_twice_is_refused_and_naming_them_not_at_all_is_too() {
        on_cli_test_stack(|| {
            for subcommand in [
                vec!["status"],
                vec!["route"],
                vec!["run", "--prompt", "hi"],
                vec!["serve"],
            ] {
                let both: Vec<&str> = ["camelid", "fabric"]
                    .into_iter()
                    .chain(subcommand.iter().copied())
                    .chain(["--node", "a=127.0.0.1", "--nodes-file", "fabric.nodes"])
                    .collect();
                Cli::try_parse_from(&both)
                    .err()
                    .unwrap_or_else(|| panic!("{both:?} named its nodes twice and was accepted"));

                let neither: Vec<&str> = ["camelid", "fabric"]
                    .into_iter()
                    .chain(subcommand.iter().copied())
                    .collect();
                Cli::try_parse_from(&neither).err().unwrap_or_else(|| {
                    panic!("{neither:?} named no nodes at all and was accepted")
                });
            }
        });
    }

    /// Two answers to "who may call" must not be configurable at once. A key
    /// set is the one that can revoke, so silently letting a stray `--api-key`
    /// win would leave an operator believing a client had been cut off.
    #[test]
    fn a_client_key_set_cannot_be_combined_with_a_single_key() {
        on_cli_test_stack(|| {
            for conflicting in [
                vec!["--api-key", "s3cret"],
                vec!["--api-key-file", "some.key"],
            ] {
                let mut argv = vec![
                    "camelid",
                    "fabric",
                    "serve",
                    "--node",
                    "a=127.0.0.1",
                    "--client-keys",
                    "clients.json",
                ];
                argv.extend(conflicting.iter().copied());
                Cli::try_parse_from(&argv)
                    .err()
                    .unwrap_or_else(|| panic!("{argv:?} was accepted"));
            }

            // Each on its own still parses, so the guard is the combination.
            for argv in [
                vec![
                    "camelid",
                    "fabric",
                    "serve",
                    "--node",
                    "a=127.0.0.1",
                    "--client-keys",
                    "clients.json",
                ],
                vec![
                    "camelid",
                    "fabric",
                    "serve",
                    "--node",
                    "a=127.0.0.1",
                    "--api-key",
                    "s3cret",
                ],
            ] {
                Cli::try_parse_from(&argv).unwrap_or_else(|_| panic!("{argv:?} must parse"));
            }
        });
    }

    /// The node set has one source. Allowing both would leave "which machines
    /// am I actually placing on" with two answers, and the file is the one
    /// that can change underneath.
    #[test]
    fn the_nodes_come_from_a_file_or_from_flags_but_never_both() {
        on_cli_test_stack(|| {
            Cli::try_parse_from([
                "camelid",
                "fabric",
                "serve",
                "--node",
                "a=127.0.0.1",
                "--nodes-file",
                "nodes.txt",
            ])
            .expect_err("--node with --nodes-file was accepted");

            // Either alone parses, so the guard is the combination — and
            // --node is no longer required now that a file can supply them.
            for argv in [
                vec!["camelid", "fabric", "serve", "--node", "a=127.0.0.1"],
                vec!["camelid", "fabric", "serve", "--nodes-file", "nodes.txt"],
            ] {
                Cli::try_parse_from(&argv).unwrap_or_else(|_| panic!("{argv:?} must parse"));
            }

            // Neither is still a refusal: a proxy with no nodes serves nothing.
            Cli::try_parse_from(["camelid", "fabric", "serve"])
                .expect_err("serve with no nodes at all was accepted");
        });
    }
}

#[derive(Debug, Subcommand)]
enum RemoteChatAction {
    /// Publish the verified loopback listener through private tailnet-only HTTPS.
    Start {
        #[command(flatten)]
        remote: RemoteChatArgs,
    },
    /// Show the private URL, local backend readiness, and transport state.
    Status {
        #[command(flatten)]
        remote: RemoteChatArgs,
        /// Emit a machine-readable status object.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Remove only the exact Camelid root mapping from Tailscale Serve.
    Stop {
        #[command(flatten)]
        remote: RemoteChatArgs,
    },
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create or display the local credential used by authenticated LAN Chat.
    LanKey {
        /// Replace the existing key, immediately invalidating browsers that
        /// still hold it. Without this flag the existing key is reused.
        #[arg(long, default_value_t = false)]
        rotate: bool,
    },
    /// Reach authenticated Chat from another network through private Tailscale HTTPS.
    RemoteChat {
        #[command(subcommand)]
        action: RemoteChatAction,
    },
    /// Start the local HTTP API server.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8181", env = "CAMELID_ADDR")]
        addr: SocketAddr,
        /// Load a GGUF model at startup and auto-select the safest validated execution plan.
        #[arg(long, env = "CAMELID_MODEL")]
        model: Option<PathBuf>,
        /// Override Rayon worker threads for the inference server.
        #[arg(long, env = "CAMELID_THREADS")]
        threads: Option<usize>,
        /// Override the linear-output parallelization threshold used by hot-path CPU kernels.
        #[arg(long, env = "CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS")]
        parallel_linear_min_outputs: Option<usize>,
        /// Override the minimum matrix size before macOS Accelerate BLAS is used.
        ///
        /// On macOS, Camelid defaults to using Accelerate only for larger dense linear rows.
        #[arg(long, env = "CAMELID_APPLE_ACCELERATE_MIN_ELEMENTS")]
        apple_accelerate_min_elements: Option<usize>,
        /// Enable the experimental Metal dense linear-row path on macOS.
        #[arg(long, env = "CAMELID_METAL_LINEAR", default_value_t = false)]
        metal_linear: bool,
        /// Enable the experimental Metal Q8_0 encoded row-dot path on macOS.
        #[arg(long, env = "CAMELID_METAL_Q8", default_value_t = false)]
        metal_q8: bool,
        /// Log the current acceleration/runtime discovery state at startup.
        #[arg(long, default_value_t = true)]
        log_acceleration: bool,
        /// Lossless greedy speculative decoding mode: "ngram" (prompt lookup,
        /// no extra weights) or "draft" (a smaller same-tokenizer model
        /// drafts; requires --spec-draft-model). Default off. A serving
        /// optimization only — it makes no support claim for any lane.
        #[arg(long, env = "CAMELID_SPEC_DECODE")]
        spec_decode: Option<String>,
        /// Draft model GGUF for --spec-decode draft (must share the target's
        /// exact token mapping).
        #[arg(long, env = "CAMELID_SPEC_DRAFT_MODEL")]
        spec_draft_model: Option<PathBuf>,
        /// Draft tokens proposed per speculation round (default: 5).
        #[arg(long, env = "CAMELID_SPEC_DRAFT_TOKENS")]
        spec_draft_tokens: Option<usize>,
        /// Gemma 4 MoE only: serve routed experts from this v2 `.cghost`
        /// artifact while retaining tokenizer, router, attention, and shared
        /// weights on the source GGUF path.
        #[arg(long, env = "CAMELID_GEMMA4_GHOST_CGHOST")]
        cghost: Option<PathBuf>,
        /// Gemma 4 Ghost-MoE only: model-global routed-expert cache ceiling in
        /// MiB. 1024 retains a little more than one Q4_0 token working set.
        #[arg(long, env = "CAMELID_GEMMA4_GHOST_CACHE_MIB", default_value_t = 1024)]
        expert_cache_mib: usize,
        /// Gemma 4 Ghost-MoE only: bypass the OS page cache for `.cghost`
        /// reads. This enforces a stricter memory ceiling but gives up buffered
        /// reuse, so normal buffered positioned reads are the default.
        #[arg(
            long,
            env = "CAMELID_GEMMA4_GHOST_STRICT_CACHE",
            default_value_t = false
        )]
        ghost_strict_cache: bool,
        /// Do not open the web UI in a browser on startup. By default, when run
        /// interactively, `serve` opens the chat surface automatically.
        #[arg(long, env = "CAMELID_NO_OPEN", default_value_t = false)]
        no_open: bool,
        /// Opt into deterministic inference: pin the forward pass to the order-stable
        /// CPU path (the whole Metal/GPU fast stack is forced off) so the supported
        /// TinyLlama 1.1B Q8_0 lane is bit-exact and reduction-order-stable across runs.
        /// Slower than the default GPU path; the default path is unchanged. Reduction
        /// order follows the llama.cpp reference Q8_0 layout (see DECISIONS.md §D9).
        #[arg(long, env = "CAMELID_DETERMINISTIC", default_value_t = false)]
        deterministic: bool,
        /// Default Qwen3/gemma4 thinking mode ON for chat requests that don't set
        /// it themselves. Opt-in and NOT parity-locked: thinking mode is supported
        /// only as a leading-trace lane (the first tokens match the llama.cpp
        /// reference before a benign f32 near-tie); the parity-locked exact-row
        /// mode remains thinking-DISABLED. A client that sends
        /// `camelid_enable_thinking` explicitly always wins over this default.
        #[arg(long, env = "CAMELID_ENABLE_THINKING", default_value_t = false)]
        enable_thinking: bool,
        /// Directory holding local GGUF models: scanned by the Models page
        /// (`/api/models/local`), the catalog download target, and the fallback
        /// base for RELATIVE model paths sent to the load endpoints (absolute
        /// paths, and relative paths that exist against the working directory,
        /// are used as given). Defaults to the first existing of
        /// `<exe dir>/models` or `./models` — the shipped layout — falling back
        /// to `./models`.
        #[arg(long, env = "CAMELID_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// GPU acceleration mode. `auto` (default) uses the GPU when a CUDA device is
        /// present, `on` forces it, `off` pins the CPU reference path. Seeds the runtime
        /// switches at startup for headless/agent runs with no UI; `on`/`off` override the
        /// env seed, and the Settings toggle can still flip state live after startup.
        #[arg(long = "gpu", value_enum, default_value_t = GpuMode::Auto, env = "CAMELID_GPU")]
        gpu: GpuMode,
        /// KV cache quantization format: "f16" (default, unquantized), "q8_0"
        /// (50% memory savings), or "q4_0" (75% memory savings).
        #[arg(long, env = "CAMELID_KV_QUANT", default_value_t = KvCacheQuantization::F16)]
        kv_quant: KvCacheQuantization,
        #[command(flatten)]
        server: ServerPolicyArgs,
    },
    /// Interactive terminal chat REPL over the local Camelid API.
    ///
    /// Attaches to (or spawns) a `camelid serve`, opens a supported-model picker,
    /// and streams `/v1/chat/completions` live. Switch models in-session with
    /// `/models`.
    Chat {
        /// Load this GGUF at startup (same semantics as `serve --model`). Omit to
        /// open the supported-model picker.
        #[arg(long, env = "CAMELID_MODEL")]
        model: Option<PathBuf>,
        /// Server to attach to, or spawn on if nothing is listening there.
        #[arg(long, default_value = "127.0.0.1:8181", env = "CAMELID_ADDR")]
        addr: SocketAddr,
        /// Initial system prompt.
        #[arg(long)]
        system: Option<String>,
        /// Maximum tokens to generate per turn.
        #[arg(long, default_value_t = 512)]
        max_tokens: u32,
        /// Sampling temperature (0 = greedy/deterministic).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Nucleus sampling top-p (omit to leave unset).
        #[arg(long)]
        top_p: Option<f32>,
        /// Top-k sampling (omit to leave unset).
        #[arg(long)]
        top_k: Option<u32>,
        /// Sampling seed (omit for the engine default).
        #[arg(long)]
        seed: Option<u64>,
        /// Print the full response after completion instead of streaming.
        #[arg(long, default_value_t = false)]
        no_stream: bool,
        /// Force the inline line REPL instead of the full-screen TUI.
        #[arg(long, default_value_t = false)]
        plain: bool,
        /// Directory holding downloaded GGUFs (picker availability + pull target).
        #[arg(long, env = "CAMELID_MODELS_DIR")]
        models_dir: Option<PathBuf>,
        /// Enter agent mode: a sandboxed tool-calling loop (requires a
        /// tool-capable supported model and `--model`).
        #[arg(long, default_value_t = false)]
        agent: bool,
        /// Sandbox root for agent file/shell tools (default: current directory).
        #[arg(long)]
        workdir: Option<PathBuf>,
        /// Max agent steps (tool-call rounds) per goal.
        #[arg(long, default_value_t = 25)]
        max_steps: usize,
        /// Agent: run write/network tools WITHOUT prompting (exec tools stay
        /// gated; sandbox still enforced). Prints a warning; not recommended.
        #[arg(long, default_value_t = false)]
        auto_approve: bool,
        /// Agent: UNATTENDED — auto-approve EVERYTHING including exec tools
        /// (shell, GUI input, run_windows_command, spawn_subagent) so the agent
        /// runs a whole task without prompting — today is a good day to die.
        /// Bounded by --max-steps and /stop. Refused under CAMELID_PRODUCTION.
        /// Powerful + dangerous; opt-in. (`--yolo` is a compatible alias.)
        #[arg(
            long = "today-is-a-good-day-to-die",
            visible_alias = "yolo",
            default_value_t = false
        )]
        yolo: bool,
        /// Agent: offer the network tool (`http_fetch`). Off by default.
        #[arg(long, default_value_t = false)]
        allow_net: bool,
        /// Agent: let the file tools read/write anywhere on disk (computer
        /// control), not just under --workdir. Still approval-gated. Off by
        /// default (file tools are confined to the workspace root).
        #[arg(long, default_value_t = false)]
        allow_fs: bool,
        /// Agent: load MCP servers declared in `camelid.mcp.json` at the
        /// workspace root and offer their tools. Third-party code: every MCP
        /// tool is approval-gated like a shell command, its output is treated
        /// as untrusted data, and the whole feature is refused under
        /// CAMELID_PRODUCTION. Off by default.
        #[arg(long, default_value_t = false)]
        allow_mcp: bool,
        /// Agent: trust and immediately execute this named server command from
        /// the workspace's camelid.mcp.json during startup. Repeat for each
        /// reviewed server. Requires --allow-mcp.
        #[arg(long = "trust-mcp-server", action = clap::ArgAction::Append)]
        trust_mcp_server: Vec<String>,
        /// Agent: shell-command timeout in seconds.
        #[arg(long, default_value_t = 30)]
        shell_timeout: u64,
        /// Render Qwen3/gemma4 thinking mode: the model emits its own
        /// `<think>…</think>` reasoning before answering. Opt-in and NOT
        /// parity-locked — supported only as a leading-trace lane (see
        /// `--enable-thinking` on `serve`). Default off keeps the parity-locked
        /// thinking-DISABLED rendering.
        #[arg(long, env = "CAMELID_ENABLE_THINKING", default_value_t = false)]
        enable_thinking: bool,
        /// Agent: POST audit events (agent.tool_call / agent.tool_result) as JSON
        /// to this URL. Delivery is async + non-blocking (drops on backpressure).
        /// Unset → no audit. No endpoint is built in.
        #[arg(long, env = "CAMELID_AUDIT_WEBHOOK")]
        audit_webhook: Option<String>,
        /// Agent: run_shell confinement — `disabled` (tool not offered),
        /// `sandboxed` (default; seccomp+uid-drop on Linux, fails closed where
        /// unenforceable), or `unrestricted` (cwd-pinned + timed only).
        #[arg(long, default_value = "sandboxed")]
        shell_sandbox: String,
    },
    /// Tool-capability promotion harness: decide whether a model drives a clean
    /// tool-call round-trip (PASS / FAIL / INCONCLUSIVE) and emit a receipt. A
    /// contended box that can't load in time yields INCONCLUSIVE, never FAIL.
    AgentEval {
        /// GGUF to evaluate.
        #[arg(long)]
        model: PathBuf,
        /// Server to attach to / spawn on.
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Seconds to wait for the model to load before reporting INCONCLUSIVE.
        #[arg(long, default_value_t = 90)]
        load_timeout: u64,
        /// Max agent steps per case.
        #[arg(long, default_value_t = 6)]
        max_steps: usize,
        /// Max tokens per model turn.
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-eval")]
        receipt_dir: PathBuf,
    },
    /// Phase-1 Windows system-control gate: exercise run_windows_command +
    /// inspect_system under the sandbox/approval contract and emit a sealed
    /// receipt (PASS / FAIL / INCONCLUSIVE). Rung-1 — promotes nothing.
    AgentSyscapEval {
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-syscap")]
        receipt_dir: PathBuf,
    },
    /// Internal: run ONE scoped subagent task described by a task file and write
    /// its result file. Spawned by the spawn_subagent tool; not for direct use.
    #[command(name = "__subagent", hide = true)]
    Subagent {
        /// Path to the task_<id>.json written by the parent.
        #[arg(long)]
        task_file: PathBuf,
    },
    /// Phase-2 subagent-orchestration gate: spawn -> run -> collect a canned
    /// subagent plus caps/depth/reaping checks, emitting a sealed receipt
    /// (PASS / FAIL / INCONCLUSIVE). Rung-2 (stub) — promotes nothing.
    AgentOrchestrationEval {
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-orchestration")]
        receipt_dir: PathBuf,
        /// Optional GGUF: run the rung-3 REAL-model round-trip instead of the
        /// canned rung-2 mechanics battery.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Server to attach to / spawn on (rung-3).
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Seconds to wait for the model to load before reporting INCONCLUSIVE.
        #[arg(long, default_value_t = 120)]
        load_timeout: u64,
    },
    /// Rung-4: measure concurrent vs sequential subagent wall-clock (I/O-bound;
    /// add --model for the inference-bound workload) and emit a sealed receipt.
    AgentOrchestrationBench {
        /// Directory for the receipt artifact.
        #[arg(long, default_value = "qa/agent-orchestration")]
        receipt_dir: PathBuf,
        /// Optional GGUF: also measure the inference-bound workload.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Server to attach to / spawn on.
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Seconds to wait for the model to load.
        #[arg(long, default_value_t = 120)]
        load_timeout: u64,
    },
    /// Start the distributed HTTP API server or TCP Worker.
    ServeDistributed {
        /// Mode to run: coordinator or worker
        #[arg(long, default_value = "coordinator")]
        role: String,
        /// Address to listen on (worker TCP listener or coordinator HTTP server)
        #[arg(long, default_value = "127.0.0.1:8181")]
        addr: SocketAddr,
        /// Address of the worker TCP listener (required for coordinator)
        #[arg(long)]
        worker_addr: Option<String>,
        /// Partition range of layers to evaluate on this node (e.g. 0..16 or 16..32)
        #[arg(long)]
        layer_range: String,
        /// Load a GGUF model at startup
        #[arg(long, env = "CAMELID_MODEL")]
        model: PathBuf,
        /// Override Rayon worker threads
        #[arg(long, env = "CAMELID_THREADS")]
        threads: Option<usize>,
        /// Shared secret the coordinator must present to this worker. Required before a
        /// worker will bind a non-loopback address, unless the risk is explicitly
        /// acknowledged with --allow-unauthenticated-remote. Prefer the file option on
        /// shared machines so the secret is not present in the process command line.
        #[arg(
            long,
            env = "CAMELID_DISTRIBUTED_TOKEN",
            conflicts_with = "distributed_token_file"
        )]
        distributed_token: Option<String>,
        /// Read the distributed shared secret from a text file.
        #[arg(long, env = "CAMELID_DISTRIBUTED_TOKEN_FILE")]
        distributed_token_file: Option<PathBuf>,
        #[command(flatten)]
        server: ServerPolicyArgs,
    },
    /// Benchmark raw TCP latency and bandwidth between Coordinator and Worker.
    #[command(hide = true)]
    BenchNetwork {
        /// Mode to run: coordinator or worker
        #[arg(long, default_value = "coordinator")]
        role: String,
        /// Address to bind to or connect to
        #[arg(long, default_value = "127.0.0.1:8182")]
        addr: String,
        /// Number of round-trips to perform for latency test
        #[arg(long, default_value_t = 1000)]
        ping_count: usize,
        /// Payload size in bytes for the latency test (default: 16KB hidden state size)
        #[arg(long, default_value_t = 16384)]
        payload_size: usize,
        /// Amount of megabytes to stream for throughput testing (default: 100 MB)
        #[arg(long, default_value_t = 100)]
        bandwidth_mb: usize,
    },
    /// Inspect and route across a fabric of independent Camelid nodes.
    Fabric {
        #[command(subcommand)]
        action: FabricAction,
    },
    /// Inspect GGUF metadata and tensor descriptors.
    Inspect { path: PathBuf },
    /// Inspect a ranged GGUF header prefix while validating tensor bounds against
    /// the source artifact's declared full length. The emitted path is redacted.
    InspectPrefix {
        path: PathBuf,
        #[arg(long)]
        declared_len: u64,
    },
    /// Inspect a GGUF file or local Hugging Face SafeTensors directory as a
    /// model source. This reports sidecar/header readiness only; SafeTensors
    /// generation stays disabled until its independent runtime gates pass.
    InspectSource { path: PathBuf },
    /// Runnable-lane smoke-admission for a single GGUF: admit -> load -> greedy
    /// forward sanity -> coherence, on oracle-qualified combos only. Prints a
    /// RUNNABLE receipt (lane=runnable, never copper; attests deterministic
    /// execution, NOT parity) to stdout on pass; exits non-zero on refusal/failure.
    RunnableSmoke { path: PathBuf },
    /// Tokenize text through the model's tokenizer (parity-harness utility,
    /// mirrors llama.cpp's `llama-tokenize`). Input is either `--prompt` or
    /// `--file` (a JSON array of strings; exact bytes preserved). Prints one
    /// JSON object per input: {"ids":[...],"decoded":"..."} where `decoded`
    /// is the decode round-trip of the encoded ids (specials retained).
    Tokenize {
        /// GGUF model (tokenizer metadata source).
        #[arg(long)]
        model: PathBuf,
        /// True full artifact length when `--model` is an immutable ranged GGUF
        /// header prefix. Without this flag, tokenization remains stat-based and
        /// truncated files fail closed exactly as before.
        #[arg(long)]
        declared_len: Option<u64>,
        /// Single prompt to tokenize.
        #[arg(long, short = 'p', conflicts_with = "file")]
        prompt: Option<String>,
        /// JSON file: array of strings to tokenize.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Parse special tokens (e.g. <|im_start|>) into their single IDs.
        #[arg(long)]
        parse_special: bool,
        /// Do not add BOS/EOS even if the model's metadata asks for them.
        #[arg(long)]
        no_add_special: bool,
    },
    /// Layer offloading — Phase 1: print the planned VRAM/host layer split for a
    /// model (no weights loaded, no compute). `--budget-mb` forces a small VRAM
    /// budget to demonstrate partial offload; `--arch <name>` plans a known
    /// architecture without its GGUF file.
    PlanOffload {
        /// GGUF model to plan (reads real per-layer tensor sizes).
        model: Option<PathBuf>,
        /// Known architecture to plan without a file (e.g. "llama-8b").
        #[arg(long)]
        arch: Option<String>,
        /// Override detected free VRAM (MiB) to force a split.
        #[arg(long)]
        budget_mb: Option<u64>,
        /// KV-cache context length to reserve (default 4096).
        #[arg(long)]
        context: Option<u64>,
        /// Safety margin in MiB (default 256).
        #[arg(long)]
        safety_mb: Option<u64>,
    },
    /// Download a supported model (a known-good Q8_0 GGUF) into ./models.
    ///
    /// Run with no argument to list the catalog. Accepts a catalog id or a
    /// fragment of the name, e.g. `camelid pull 3b_instruct_q8`.
    //
    // The id here is a plain literal because clap reads doc attributes
    // textually and silently drops anything that is not already a string
    // literal, so it cannot be spliced in from a constant. It is kept honest by
    // `catalog::tests::every_pull_example_in_the_source_resolves`, which scans
    // for this pattern and requires every id it finds to resolve.
    Pull {
        /// Catalog id or name fragment to download. Omit to list all models.
        model: Option<String>,
        /// Directory to download into (default: ./models).
        #[arg(long, env = "CAMELID_MODELS_DIR")]
        models_dir: Option<PathBuf>,
    },
    /// Generate text with a Gemma 4 model (correctness-first runtime).
    Gemma4Generate {
        path: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
        /// BASALT forced-decode harness (basalt_eval_protocol.md §5.1):
        /// teacher-force the token ids in this file (newline-separated decimal
        /// ids, or one JSON array). Each step feeds the forced token as the
        /// next input REGARDLESS of the model's argmax, while the per-step
        /// argmax id + logit are still recorded (stdout JSON). Ignores
        /// --max-tokens and stop tokens: the list length defines the step count.
        #[arg(long)]
        force_tokens: Option<PathBuf>,
        /// Write each step's FULL logit vector into this directory as raw
        /// little-endian f32 `step_<i>.bin` files, plus a `meta.json` (vocab
        /// size, step count, prompt info, per-step top-32 (id, logit)). Works
        /// with or without --force-tokens.
        #[arg(long)]
        dump_step_logits: Option<PathBuf>,
    },
    /// Load-amortized BASALT eval-pack runner (`basalt_eval_protocol.md` §5.1):
    /// load a gemma4 model ONCE and run every prompt in the given packs. This is
    /// the load-once form of the per-prompt `Gemma4Generate` harness, for the G3(b)
    /// teacher-forced top-1 agreement metric when iterating over many quant rows.
    ///
    /// Without `--score`: greedy-generate each prompt and write
    /// `<baseline_dir>/<prompt_id>.txt` (newline token ids) — the Q8_0 reference.
    /// With `--score`: teacher-force each prompt's reference ids from
    /// `<baseline_dir>` through THIS model and report teacher-forced top-1
    /// agreement (overall + per prompt). CPU path; no engine math change.
    Gemma4EvalPack {
        path: PathBuf,
        /// Pack JSON file(s), e.g. qa/gemma4/prompt_packs/basic_v1.json.
        #[arg(long = "pack", required = true, num_args = 1..)]
        packs: Vec<PathBuf>,
        /// Directory holding (score mode) or receiving (baseline mode) the
        /// per-prompt reference token-id files `<prompt_id>.txt`.
        #[arg(long)]
        baseline_dir: PathBuf,
        /// Score this model's teacher-forced agreement against the reference ids
        /// in `--baseline-dir` instead of generating them.
        #[arg(long)]
        score: bool,
    },
    /// Generate with the CUDA-resident Gemma 4 lane (dev harness for the SSER build).
    #[cfg(feature = "cuda")]
    Gemma4CudaGenerate {
        path: PathBuf,
        /// Optional sparse routed-expert payload paired with a Ghost-MoE shadow GGUF.
        #[arg(long)]
        cghost: Option<PathBuf>,
        /// Host expert-cache budget (mapped CUDA reads bypass this on the normal path).
        #[arg(long, default_value_t = 64)]
        expert_cache_mib: usize,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
        /// Hugging Face directory of the Gemma 4 MTP assistant (the drafter). When set,
        /// decode runs lossless speculative rounds: the drafter proposes, the target
        /// verifies the whole batch in ONE weight pass, and only the target's own argmax
        /// is ever emitted -- so this changes speed, never output.
        #[arg(long)]
        mtp_assistant: Option<PathBuf>,
        /// Maximum draft tokens proposed per verify round. The default-off
        /// `CAMELID_GEMMA4_MTP_WIDTH_SCHEDULE` instead names verifier widths
        /// (current target row plus drafts), each bounded by this value plus one.
        /// Default 7: on the frozen H40 fixture K=7 accepted every draft
        /// (alpha 7.00) and beat both K=6 and K=8 paired
        /// (BASELINE-2026-08-24 §16); K=8's extra row hit the known
        /// content-boundary miss and paid union bytes for nothing.
        #[arg(long, default_value_t = 7)]
        mtp_draft_k: usize,
        /// OpenAI-shaped chat request to run instead of `--prompt`, templated through the
        /// SAME renderer `/v1/chat/completions` uses. This is how an offline throughput
        /// number stays comparable to a served one: a hand-built prompt string differs
        /// from the served prompt by exactly the template, and then the token ids do not
        /// match anyone's. The request must ask for greedy decoding; `max_tokens` in the
        /// file wins over `--max-tokens`.
        #[arg(long)]
        request_json: Option<PathBuf>,
        /// JSON array of the token ids this run is expected to emit. Turns the run into a
        /// gate: the receipt records `exact_match` and the first divergence, and the
        /// process exits non-zero on a mismatch. A tok/s number without this is not a
        /// result.
        #[arg(long)]
        expect_token_ids: Option<PathBuf>,
        /// Write a machine-readable receipt (ids, walls, alpha, expert misses, and the
        /// per-round draft-vs-target trace) to this path.
        #[arg(long)]
        receipt: Option<PathBuf>,
    },
    /// Offline, exact-target repack of dense Gemma 4 Q4_0 projections into the
    /// native scale-plane/quant-plane sidecar consumed by the Metal runtime.
    /// The source GGUF is never modified and an existing output is refused.
    Gemma4Q4Repack {
        path: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate text with a Gemma 4 model on the GPU (resident decode; macOS/Metal).
    Gemma4GenerateGpu {
        path: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
    },
    /// Qualify and benchmark lossless Gemma 4 12B MTP speculative decode on
    /// Metal. The exact official assistant drafts 1/3/7/15 tokens for target
    /// verifier widths 2/4/8/16; every run must reproduce ordered K1 token IDs.
    /// For an explicitly configured W16 run, the default-off strict selector
    /// `CAMELID_GEMMA4_MTP_W16_WARMUP8=1` starts at W8 and promotes only after
    /// a complete W8 acceptance round.
    /// `CAMELID_GEMMA4_MTP_W16_ONESHOT_W8_PAD16=1` is a mutually-exclusive
    /// experiment: run one W8 bootstrap, stay W16 regardless of its acceptance,
    /// and admit a causally padded physical-W16 tail when it replaces two tails.
    Gemma4Mtp12Gpu {
        path: PathBuf,
        /// Exact official 12B assistant `model.safetensors` file.
        #[arg(long)]
        assistant: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 96)]
        max_tokens: usize,
        /// Target verifier widths. The anchor row is included, so these map to
        /// assistant draft counts 1/3/7/15. The default remains the established
        /// 2/4/8 sweep; pass 16 explicitly to exercise the opt-in K16 Q4 lane.
        #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
        widths: Vec<usize>,
    },
    /// Qualify and time the strict K<=16 ordered-Q4 Gemma 4 target verifier.
    /// This is a developer harness: it first requires whole-output K=1 parity
    /// with the established GPU lane, then teacher-forces one fixed sequence
    /// through each requested width and requires every target prediction to
    /// equal the ordered K=1 result.
    Gemma4VerifyGpu {
        path: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 96)]
        max_tokens: usize,
        /// Verifier row widths. Only 1,2,4,8,16 are admitted; each timed batch
        /// is full-width so rows/s cannot be inflated by a scalar tail. The
        /// default remains the established K<=8 sweep.
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
        widths: Vec<usize>,
    },
    /// Chat with a DiffusionGemma model: render the chat template, run the
    /// bit-exact multi-canvas block-autoregressive denoise loop, detokenize.
    /// CPU-only and slow (each denoise step is a full bidirectional forward);
    /// experimental — see the DiffusionGemma lane recon.
    DiffusionGemmaChat {
        path: PathBuf,
        #[arg(long, default_value = "Hello")]
        prompt: String,
        /// Max blocks (each block denoises one canvas_length window, then
        /// commits to the prefix). The answer stops earlier on an end token,
        /// a repetition loop, or the ubatch budget.
        #[arg(long, default_value_t = 4)]
        max_blocks: i32,
        /// Entropy-Bound sampler seed (reference default 0).
        #[arg(long, default_value_t = 0)]
        seed: u32,
        /// Max ubatch (the whole [prefix | canvas] must fit in one ubatch).
        #[arg(long, default_value_t = 1100)]
        max_ubatch: i32,
        /// Override the EB sampler's max denoise steps per block (reference
        /// default 48, with adaptive early stop). Lower it (e.g. 1-2) for a
        /// fast correctness signal — each step is a full bidirectional forward.
        #[arg(long)]
        max_steps: Option<i32>,
    },
    /// Serve the TAIL layers of a Gemma 4 model as a distributed worker
    /// (layer sharding over TCP; pair with gemma4-master on the other Mac).
    Gemma4Worker {
        path: PathBuf,
        #[arg(long, default_value = "0.0.0.0:5005")]
        addr: String,
        /// First (global) layer this worker owns; it owns through the final
        /// layer plus the output head. Must not split the shared-KV block.
        #[arg(long)]
        first_layer: usize,
    },
    /// Run the HEAD layers of a Gemma 4 model and drive a distributed worker
    /// for the tail (greedy decode; distributed layer sharding, not shared memory).
    Gemma4Master {
        path: PathBuf,
        #[arg(long)]
        worker_addr: String,
        /// Layers [0, split) run locally; [split, block_count) on the worker.
        #[arg(long)]
        split: usize,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 24)]
        max_tokens: usize,
    },
    /// Dump focused tensor descriptor, raw block, and f32 dequantization diagnostics.
    #[command(hide = true)]
    TensorDump {
        path: PathBuf,
        /// Tensor name to dump. Repeat to override the TinyLlama parity default set.
        #[arg(long = "tensor")]
        tensors: Vec<String>,
        /// Number of decoded f32 values to include from tensor start and max-absolute window.
        #[arg(long, default_value_t = 8)]
        window: usize,
        /// Row index to sample for each 2D tensor using the dump's runtime shape.
        #[arg(long = "row")]
        rows: Vec<usize>,
        /// Token id to sample as a logical token-major row for embedding-shaped tensors.
        #[arg(long = "token")]
        tokens: Vec<usize>,
        /// LLaMA layer index whose Q/K/V/O and FFN tensors should be included in the dump.
        #[arg(long = "layer")]
        layers: Vec<usize>,
    },
    /// Run a deterministic release-mode microbenchmark for dense matmul/FFN hot loops.
    #[command(hide = true)]
    BenchDenseHotloops {
        /// LLaMA hidden width for the synthetic single-row input.
        #[arg(long, default_value_t = 2048)]
        hidden: usize,
        /// LLaMA feed-forward width for synthetic gate/up/down projections.
        #[arg(long, default_value_t = 5632)]
        ffn: usize,
        /// Measured iterations after warmup.
        #[arg(long, default_value_t = 20)]
        repeats: usize,
        /// Unreported warmup iterations.
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Override Rayon worker threads for this benchmark. Defaults to RAYON_NUM_THREADS/Rayon.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Hidden: decode zero-alloc gate — decode real tokens through a loaded
    /// model under the counting global allocator and report steady-state heap
    /// allocations per token. Requires `--features alloc-gate`.
    #[cfg(feature = "alloc-gate")]
    #[command(hide = true)]
    BenchAllocGate {
        /// GGUF model to decode.
        #[arg(long)]
        model: std::path::PathBuf,
        /// Unmeasured tokens to warm pools, binding cells, and KV growth.
        #[arg(long, default_value_t = 8)]
        warmup: usize,
        /// Measured steady-state tokens.
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        /// Skip the final norm + logits projection (attribution mode).
        #[arg(long, default_value_t = false)]
        skip_logits: bool,
        /// Print backtraces for the first few >=1MiB steady-state allocations.
        #[arg(long, default_value_t = false)]
        trace_big: bool,
        /// Fail (non-zero exit) if allocations per token exceed this.
        #[arg(long)]
        max_per_token: Option<f64>,
    },
    /// Hidden: micro-benchmark rayon fork-join region overhead on the global
    /// pool (hot = back-to-back regions; cold = workers idle between regions).
    #[command(hide = true)]
    BenchRayonRegion {
        /// Measured regions per point.
        #[arg(long, default_value_t = 10_000)]
        iterations: usize,
        /// Idle time between regions in microseconds (0 = hot).
        #[arg(long, default_value_t = 0)]
        idle_us: u64,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Hidden: hot-cache micro-benchmark of the attention f32 dot kernels
    /// (legacy scalar chain vs canonical blocked scalar vs blocked AVX2/FMA).
    #[command(hide = true)]
    BenchAttnDot {
        /// Vector lengths to measure (defaults cover the real head dims).
        #[arg(long = "len", default_values_t = [64usize, 128])]
        lens: Vec<usize>,
        /// Measured iterations per variant.
        #[arg(long, default_value_t = 2_000_000)]
        repeats: usize,
        /// Unreported warmup iterations per variant.
        #[arg(long, default_value_t = 100_000)]
        warmup: usize,
    },
    /// Load one GGUF Q8_0 tensor as retained blocks and benchmark bounded row dequantization/dot rows.
    #[command(hide = true)]
    BenchQ8Blocks {
        /// GGUF model path.
        path: PathBuf,
        /// Q8_0 tensor name to load as block-only data.
        #[arg(long, default_value = "blk.0.ffn_gate.weight")]
        tensor: String,
        /// Reinterpret a rank-2 tensor by swapping its logical rows/cols before benchmarking.
        ///
        /// This mirrors Camelid's guarded rectangular linear/output-projection layout path for
        /// tensors whose GGUF descriptor dimensions are stored token/input-major but the lazy
        /// Q8 hot path consumes contiguous logical output rows.
        #[arg(long)]
        swap_rank2_shape: bool,
        /// Row index to dequantize. Repeat for multiple rows.
        #[arg(long = "row")]
        rows: Vec<usize>,
        /// Measured iterations after warmup.
        #[arg(long, default_value_t = 20)]
        repeats: usize,
        /// Unreported warmup iterations.
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Also benchmark the lazy all-row Q8_0 dot helper that returns a dense f32 output vector.
        #[arg(long)]
        all_rows_dot: bool,
        /// Also benchmark the rank-2 single-input-row Q8_0 lazy-linear adapter shape.
        #[arg(long)]
        single_input_row_dot: bool,
    },
    /// Start a distributed pipeline worker node.
    DistributeWorker {
        /// GGUF model path.
        path: PathBuf,
        /// Listen address for incoming master/worker connection.
        #[arg(long, default_value = "0.0.0.0:5005")]
        addr: SocketAddr,
        /// Target forward address (next node in the pipeline).
        #[arg(long)]
        forward_addr: Option<SocketAddr>,
        /// Range of layers to own and execute, e.g., "16..32" or "24..56".
        #[arg(long)]
        layers: String,
        /// Master address to send token feedback to when we are the final node.
        #[arg(long)]
        master_addr: Option<SocketAddr>,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// EXPERIMENTAL ghost mesh: stream this node's layer shard per token from a
        /// `.cghost` file (double-buffered) instead of holding it resident. Only the
        /// embedding/output ends stay in RAM; the shard's disk window overlaps the other
        /// node's compute.
        #[arg(long)]
        cghost: Option<PathBuf>,
    },
    /// Start a distributed pipeline master node.
    DistributeMaster {
        /// GGUF model path.
        path: PathBuf,
        /// Worker address to send activation streams to.
        #[arg(long)]
        worker_addr: SocketAddr,
        /// Range of layers to own and execute, e.g., "0..16" or "0..24".
        #[arg(long)]
        layers: String,
        /// Listen address for token feedback or final results from the last node in the pipeline.
        #[arg(long, default_value = "0.0.0.0:5006")]
        addr: SocketAddr,
        /// Prompt to execute.
        #[arg(long, default_value = "Write a quick Rust hello-world function:")]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// EXPERIMENTAL ghost mesh: stream this node's layer shard per token from a
        /// `.cghost` file (double-buffered) instead of holding it resident. Only the
        /// embedding/output ends stay in RAM; the shard's disk window overlaps the other
        /// node's compute.
        #[arg(long)]
        cghost: Option<PathBuf>,
    },
    /// Single-node generation microbenchmark. Loads a GGUF model once, generates
    /// from a prompt, and emits one JSON metrics object per measured iteration
    /// (load/prefill/TTFT/decode timings, decode tok/s, peak RSS). For runtime
    /// comparison harnesses.
    #[command(hide = true)]
    BenchGenerate {
        /// GGUF model path.
        model: PathBuf,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Maximum tokens to generate per iteration.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy/argmax, deterministic).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Number of measured iterations (one JSON object per iteration).
        #[arg(long, default_value_t = 1)]
        iterations: usize,
        /// Run one unmeasured warmup generation before the measured iterations.
        #[arg(long, default_value_t = false)]
        warmup: bool,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// Accepted for compatibility; JSON is always emitted to stdout.
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Opt into deterministic inference: pin generation to the order-stable CPU
        /// forward pass (Metal/GPU fast stack forced off) so the supported TinyLlama
        /// 1.1B Q8_0 lane is bit-exact across runs, thread counts, and processes.
        /// Slower than the default GPU path; the default path is unchanged.
        #[arg(long, env = "CAMELID_DETERMINISTIC", default_value_t = false)]
        deterministic: bool,
    },
    /// Hidden: end-to-end Prism Qwen3-VL image encode plus Qwen3.5 Metal/CUDA decode.
    #[command(hide = true)]
    BenchGenerateVision {
        /// Qwen3.5 Bonsai GGUF model path.
        model: PathBuf,
        /// Matching Prism qwen3vl_merger GGUF.
        #[arg(long)]
        mmproj: PathBuf,
        /// PNG or JPEG image.
        #[arg(long)]
        image: PathBuf,
        /// Text placed after the image.
        #[arg(long, default_value = "Describe this image briefly.")]
        prompt: String,
        /// Maximum generated tokens.
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// Minimum merged image-token budget.
        #[arg(long, default_value_t = 1)]
        image_min_tokens: usize,
        /// Maximum merged image-token budget.
        #[arg(long, default_value_t = 1024)]
        image_max_tokens: usize,
    },
    /// Hidden: in-process INTERLEAVED owner-microkernel prefill sweep. Loads the model ONCE, then
    /// rotates owner configs (off / avx2 / vnni4x4 / vnni4x8) round-by-round so every config shares
    /// the same thermal/clock state, enabling drift-cancelling PAIRED comparison (the fix for the
    /// noise that made v3 inconclusive). The owner flag is read from env per linear call, so no
    /// reload is needed between configs. Emits one JSON line per (round, config) to stdout.
    #[command(hide = true)]
    BenchOwnerSweep {
        /// GGUF model path.
        model: PathBuf,
        /// Which owner to sweep: `q8` or `kquant` (for a Q4_K/Q6_K model).
        /// Inner kernels this CPU cannot run are skipped rather than measured.
        #[arg(long, default_value = "q8")]
        lane: String,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Tokens to generate per measurement (prefill dominates; keep small).
        #[arg(long, default_value_t = 1)]
        max_tokens: usize,
        /// Measured interleaved rounds (median + paired stats taken across rounds).
        #[arg(long, default_value_t = 10)]
        rounds: usize,
        /// Leading rounds discarded as warmup (reach steady thermal state).
        #[arg(long, default_value_t = 2)]
        warmup_rounds: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// GAIT: run the parity-gated calibration tournament for this model on this
    /// machine. Times the supported execution profiles, disqualifies any whose
    /// greedy output diverges, picks the fastest parity-clean one that beats the
    /// baseline by a margin (else fails closed to baseline), and persists a
    /// `camelid.gait-receipt/v1`. Writes only a receipt; changes no decode path.
    /// The receipt is consumed later only when `CAMELID_GAIT` is set.
    #[command(hide = true)]
    GaitCalibrate {
        /// GGUF model path.
        model: PathBuf,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Tokens to generate per trial (greedy/deterministic).
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Measured interleaved rounds per variant (median is taken). More rounds
        /// reject thermal/clock noise at the cost of calibration time.
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        /// Leading rounds discarded as warmup.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// GAIT internal: run ONE calibration candidate trial in isolation and print
    /// its TrialResult as a single JSON line. Spawned as a child process by the
    /// calibration supervisor (§1.4 crash isolation); not for direct use.
    #[command(hide = true)]
    GaitTrial {
        /// GGUF model path.
        model: PathBuf,
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Profile label: auto | safe | experimental | debug.
        #[arg(long)]
        profile: String,
        /// Apply the compute-pool EcoQoS opt-out for this trial.
        #[arg(long, default_value_t = false)]
        eco_qos: bool,
        #[arg(long)]
        threads: Option<usize>,
        /// §5 groups_per_chunk tiling overrides (pass all three together).
        #[arg(long)]
        gpc_attn: Option<usize>,
        #[arg(long)]
        gpc_ffn: Option<usize>,
        #[arg(long)]
        gpc_matmul: Option<usize>,
    },
    /// Headless agent runs (e.g. `camelid agent exec "fix the failing test"`).
    Agent {
        #[command(subcommand)]
        action: AgentAction,
    },
    /// Grounded, durable, read-only conversations with a local folder.
    Workspace {
        /// Address of an already-running loopback Camelid server.
        #[arg(
            long,
            global = true,
            default_value = "127.0.0.1:8181",
            env = "CAMELID_ADDR"
        )]
        addr: SocketAddr,
        /// Emit compact JSON (JSON Lines for `ask`).
        #[arg(long, global = true, default_value_t = false)]
        json: bool,
        /// Maximum time to wait for one `ask` event stream.
        #[arg(long, global = true, default_value_t = 1800)]
        timeout_seconds: u64,
        #[command(subcommand)]
        action: WorkspaceAction,
    },
    /// GAIT cache maintenance (e.g. `camelid gait reset`).
    #[command(hide = true)]
    Gait {
        #[command(subcommand)]
        action: GaitAction,
    },
    /// SPEC_RECHECK measurement harness: run lossless greedy speculative decode and a plain
    /// greedy baseline back-to-back on one prompt, and emit a single JSON record with the
    /// per-run economics (acceptance rate, draft/verify latency split, f_draft, S_sync) plus
    /// the lossless verdict (the spec token stream's first divergence vs Camelid plain greedy).
    /// Default off; moves no support ledger; reuses the existing drafters + GPU verify.
    BenchSpeculative {
        /// Target GGUF model path (the model whose output must be reproduced exactly).
        model: PathBuf,
        /// Drafter: "ngram" (prompt lookup, no draft model) or "draft" (a smaller
        /// same-tokenizer model; requires --draft-model).
        #[arg(long, default_value = "ngram")]
        drafter: String,
        /// Draft model GGUF for --drafter draft. Must share the target's token mapping.
        #[arg(long)]
        draft_model: Option<PathBuf>,
        /// Drafted tokens per round (γ). Capped at MAX_VERIFY_K - 1 = 15 by the verify path.
        #[arg(long)]
        draft_tokens: Option<usize>,
        /// Force the draft model onto the CPU forward path (Path 3 in SPEC_RECHECK). Default
        /// leaves the draft GPU-resident (its own drafter cache), falling back to CPU only if
        /// it does not fit in VRAM.
        #[arg(long, default_value_t = false)]
        cpu_draft: bool,
        /// Build the coexistence target + resident draft once (drafter/reserve set BEFORE the
        /// target builds) and measure only the speculative run; the plain reference reuses that
        /// same resident target. Avoids an in-process full-size target rebuild whose VRAM the
        /// cudarc pool will not release back to the sizing probe. The plain tps reported here is
        /// the same-config (coexistence) target, not a full-resident-target baseline.
        #[arg(long, default_value_t = false)]
        spec_only: bool,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Workload label recorded in the JSON (e.g. "code", "json", "extraction").
        #[arg(long, default_value = "unlabeled")]
        workload: String,
        /// Maximum tokens to generate (fixed per workload for a reproducible matrix).
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Run one unmeasured warmup pair before the measured run.
        #[arg(long, default_value_t = false)]
        warmup: bool,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Benchmark-only Llama 3.2 3B EAGLE-3 speculative decode. The learned head
    /// and target verifier run on resident Metal, with target output authoritative.
    /// This is learned EAGLE-3 speculation, not a native target MTP head.
    BenchEagle3 {
        /// Exact Llama-3.2-3B-Instruct target GGUF.
        model: PathBuf,
        /// Directory containing the pinned EAGLE-3 config.json and model.safetensors.
        #[arg(long)]
        eagle3: PathBuf,
        /// Top-1 draft-chain length per verify round.
        #[arg(long, default_value_t = 4)]
        draft_tokens: usize,
        /// Enable dynamic EAGLE tree verification with this node budget (including the root).
        /// Omit to preserve the existing top-1 linear-chain path.
        #[arg(long)]
        tree_nodes: Option<usize>,
        /// Retained draft-head alternatives per expanded tree parent (1..=8).
        #[arg(long, default_value_t = 4)]
        tree_topk: usize,
        /// Learned-head parent expansions per tree round, including the root expansion.
        #[arg(long, default_value_t = 4)]
        tree_expansions: usize,
        /// Try a model-free suffix chain first, falling back to the dynamic EAGLE tree
        /// when history contains no usable suffix continuation. Benchmark-only and
        /// valid only with --tree-nodes.
        #[arg(long, default_value_t = false, requires = "tree_nodes")]
        suffix_first: bool,
        /// Read the prompt from this UTF-8 file. Takes precedence over --prompt.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Inline prompt text (used when --prompt-file is absent).
        #[arg(long)]
        prompt: Option<String>,
        /// Render the supplied text as one user turn through the same no-tools
        /// chat-template path used by /v1/chat/completions.
        #[arg(long, default_value_t = false)]
        chat: bool,
        /// Workload label recorded in the JSON receipt.
        #[arg(long, default_value = "unlabeled")]
        workload: String,
        /// Maximum generated tokens, including the first target anchor.
        #[arg(long, default_value_t = 96)]
        max_tokens: usize,
        /// Override Rayon worker threads for the target verifier.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// EXPERIMENTAL ghost (layer-streaming) mode: execute a model one transformer block at
    /// a time, streaming each block's weights from a layer-contiguous `.cghost` file
    /// (see the `repack-ghost` tool) and holding only a one-layer working window plus the
    /// embedding/output ends in RAM. Trades throughput for a strict memory ceiling.
    /// Double-buffered prefetch by default; `--sync-stream` forces the v1 serial read.
    GhostRun {
        /// GGUF model path (metadata, tokenizer, and resident embedding/output ends).
        model: PathBuf,
        /// Layer-contiguous .cghost file produced by `repack-ghost` from the same model.
        #[arg(long)]
        cghost: PathBuf,
        /// Prompt to execute (greedy decode).
        #[arg(long, default_value = "Write a quick Rust hello-world function:")]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// Override Rayon worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// Disable the double-buffered prefetch worker and read each layer synchronously
        /// on the critical path (the v1 behavior; useful for A/B comparison).
        #[arg(long, default_value_t = false)]
        sync_stream: bool,
        /// Stage-split streaming (WRAITH Phase 2): run read and decode on separate threads so
        /// layer N+1's disk read overlaps layer N's dequant. Parity-identical; biggest win in
        /// the cold-NVMe regime where read and decode are comparable. Overrides the default
        /// single-worker prefetch. Mutually exclusive with `--sync-stream`.
        #[arg(long, default_value_t = false)]
        stage_split: bool,
        /// Stage-split read-ahead: how many layers the reader may run ahead of the decoder
        /// (raw-buffer pool = read_ahead + 1 layer spans; folds into the memory ceiling).
        #[arg(long, default_value_t = 2)]
        read_ahead: usize,
        /// Speculative decode (WRAITH Phase 3): draft L tokens with a resident zero-weight
        /// n-gram, verify all L+1 in ONE streamed sweep, accept the greedy-identical prefix.
        /// Lossless — accepted output is byte-identical to non-spec greedy. Amortizes the fixed
        /// per-layer disk read across the accepted tokens; biggest win on repetitive text.
        #[arg(long, default_value_t = false)]
        spec: bool,
        /// Speculative draft length L (n-gram tokens proposed per verify sweep). Capped at 7.
        #[arg(long, default_value_t = 5)]
        draft_len: usize,
        /// Ghost-MoE only: global routed-expert cache ceiling in MiB. The
        /// budget is shared by every layer; 0 keeps no expert after its use.
        #[arg(long, default_value_t = 1024)]
        expert_cache_mib: usize,
        /// Strict memory ceiling mode: bypass the OS page cache for `.cghost` reads so
        /// streamed pages never accumulate (macOS F_NOCACHE; Windows FILE_FLAG_NO_BUFFERING).
        /// Leave off when the model fits in RAM and you want throughput (the cache is a free
        /// win there); turn ON to measure true cold-disk streaming cost even on a box where
        /// the model would otherwise cache.
        #[arg(long, default_value_t = false)]
        evict_page_cache: bool,
    },
    /// Verify one exact GGUF by replaying a built-in, reference-anchored,
    /// deterministic request. Emits a digest-sealed report. A pass proves one
    /// request for one exact file; it is not a broad support claim.
    Verify {
        /// GGUF model path. Verification abstains when no exact-hash profile exists.
        model: PathBuf,
        /// Report output path. Defaults to `<model-stem>.verify.json`.
        #[arg(long, value_name = "PATH")]
        output: Option<PathBuf>,
        /// Override Rayon worker threads for the deterministic replay.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Verify a receipt. For a parity receipt: self-digest, lane identity, an
    /// in-process Camelid re-run, and a llama.cpp reference re-run (requires
    /// `--gguf`). For a sealed agent-family receipt (syscap / orchestration /
    /// bench): a self-contained tamper-evidence + honest-scope check, no GGUF.
    /// A verified receipt changes no support claim.
    VerifyReceipt {
        /// Path to the receipt JSON file.
        receipt: PathBuf,
        /// The exact GGUF the receipt names (its SHA-256 must match). Required
        /// for a parity receipt; agent-family receipts need no GGUF.
        #[arg(long)]
        gguf: Option<PathBuf>,
        /// llama-server binary for the reference re-run (path or name in PATH).
        #[arg(long, default_value = "llama-server")]
        llama_server: String,
        /// Run only the self half (digest, lane identity, Camelid re-run);
        /// honest for verifiers without llama.cpp, but full parity is NOT
        /// asserted.
        #[arg(long, conflicts_with = "reference_only")]
        self_only: bool,
        /// Run only the reference half (digest, lane identity, llama.cpp
        /// re-run); skips the in-process Camelid re-run.
        #[arg(long)]
        reference_only: bool,
        /// Context size passed to llama-server (-c).
        #[arg(long, default_value_t = 2048)]
        llama_ctx: u32,
        /// Port for the temporary llama-server instance.
        #[arg(long, default_value_t = 8189)]
        llama_port: u16,
        /// KV cache type passed to llama-server for K (-ctk). Omit to use
        /// llama.cpp's default.
        #[arg(
            long,
            value_parser = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        )]
        llama_cache_type_k: Option<String>,
        /// KV cache type passed to llama-server for V (-ctv). Omit to use
        /// llama.cpp's default.
        #[arg(
            long,
            value_parser = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        )]
        llama_cache_type_v: Option<String>,
        /// Flash-attention mode passed to llama-server (-fa). Omit to use
        /// llama.cpp's default.
        #[arg(long, value_parser = ["on", "off", "auto"])]
        llama_flash_attn: Option<String>,
        /// Disable llama.cpp tensor repacking for the reference re-run.
        #[arg(long)]
        llama_no_repack: bool,
        /// Override Rayon worker threads for the Camelid re-run.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Audit the self-digest of every sealed receipt under a directory tree
    /// (parity / distributed / agent families). A corrupted or hand-edited
    /// committed receipt fails; unsealed and non-receipt JSON is skipped. This is
    /// the mechanical companion to review for the receipt evidence base.
    VerifyReceipts {
        /// Directory walked recursively for `*.json` receipts.
        #[arg(default_value = "qa")]
        dir: PathBuf,
    },
    /// Recompute and stamp `receipt_id` on a receipt body. Emitters (e.g. the
    /// chat-parity harness) delegate sealing here so canonical serialization
    /// and digesting live in exactly one implementation.
    #[command(hide = true)]
    SealReceipt {
        /// Receipt JSON to seal (the existing receipt_id value is ignored).
        #[arg(long, value_name = "PATH")]
        input: PathBuf,
        /// Output path; defaults to sealing in place.
        #[arg(long, value_name = "PATH")]
        output: Option<PathBuf>,
    },
}

/// Counts reported by the CUDA Gemma 4 development benchmark.
///
/// These are deliberately separate: the timed runtime currently exposes one
/// duration per decode forward, while its returned ID list contains tokens
/// emitted from both prefill logits and decode logits. Treating either count as
/// the other can overstate throughput (and hides generation-budget bugs).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4CudaBenchmarkCounts {
    requested_new_tokens: usize,
    emitted_non_stop_tokens: usize,
    timed_decode_forwards: usize,
}

#[cfg(feature = "cuda")]
impl Gemma4CudaBenchmarkCounts {
    fn new(
        requested_new_tokens: usize,
        emitted_non_stop_tokens: usize,
        timed_decode_forwards: usize,
    ) -> Self {
        Self {
            requested_new_tokens,
            emitted_non_stop_tokens,
            timed_decode_forwards,
        }
    }

    fn emitted_budget_is_respected(self) -> bool {
        self.emitted_non_stop_tokens <= self.requested_new_tokens
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gemma4_cuda_benchmark_tests {
    use super::Gemma4CudaBenchmarkCounts;

    #[test]
    fn accounting_does_not_conflate_emitted_tokens_with_decode_forwards() {
        let counts = Gemma4CudaBenchmarkCounts::new(8, 9, 8);
        assert_eq!(counts.requested_new_tokens, 8);
        assert_eq!(counts.emitted_non_stop_tokens, 9);
        assert_eq!(counts.timed_decode_forwards, 8);
        assert!(!counts.emitted_budget_is_respected());
    }

    #[test]
    fn early_stop_can_emit_fewer_tokens_than_requested() {
        let counts = Gemma4CudaBenchmarkCounts::new(8, 3, 3);
        assert!(counts.emitted_budget_is_respected());
    }
}

/// The subset of an OpenAI chat request the offline gemma4 harness honors.
///
/// Deliberately strict. This exists so that a local `gemma4-cuda-generate` run and a
/// served `/v1/chat/completions` run decode the SAME ids from the SAME prompt bytes.
/// Anything in the request that would break that equivalence is rejected rather than
/// quietly ignored: a receipt that silently dropped `temperature: 0.7` would report a
/// comparable-looking number for an incomparable run.
#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct Gemma4HarnessRequest {
    messages: Vec<Gemma4HarnessMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    camelid_enable_thinking: Option<bool>,
}

#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct Gemma4HarnessMessage {
    role: String,
    /// Plain-string content only. The OpenAI content-parts form is a served-lane
    /// feature; a fixture using it would reach the model through a different path than
    /// this one, which is exactly what this harness exists to rule out.
    content: String,
}

/// Render a harness request into `(prompt, max_tokens)` through the serve lane's own
/// template.
///
/// Errors on anything non-greedy. This lane implements argmax decoding only, so running
/// a sampling request here would silently measure a different decode than the one the
/// fixture describes.
#[cfg(feature = "cuda")]
fn gemma4_harness_request(
    path: &std::path::Path,
    fallback_max_tokens: usize,
) -> anyhow::Result<(String, usize)> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| anyhow::anyhow!("reading {}: {error}", path.display()))?;
    let request: Gemma4HarnessRequest = serde_json::from_str(&raw)
        .map_err(|error| anyhow::anyhow!("parsing {}: {error}", path.display()))?;
    anyhow::ensure!(
        !request.messages.is_empty(),
        "{} has no messages",
        path.display()
    );
    if let Some(temperature) = request.temperature {
        anyhow::ensure!(
            temperature == 0.0,
            "{} asks for temperature {temperature}; this lane decodes greedily only, so \
             the measured run would not be the one the request describes",
            path.display()
        );
    }
    if let Some(top_k) = request.top_k {
        anyhow::ensure!(
            top_k == 1,
            "{} asks for top_k {top_k}; this lane decodes greedily only",
            path.display()
        );
    }
    let messages: Vec<camelid::api::ChatMessage> = request
        .messages
        .into_iter()
        .map(|message| camelid::api::ChatMessage {
            role: message.role,
            content: message.content,
            image_urls: Vec::new(),
            unsupported_content_parts: Vec::new(),
        })
        .collect();
    let prompt = camelid::api::render_gemma4_chat_prompt(
        &messages,
        request.camelid_enable_thinking.unwrap_or(false),
    );
    Ok((
        prompt,
        request.max_tokens.unwrap_or(fallback_max_tokens).max(1),
    ))
}

/// Read a JSON array of expected token ids.
#[cfg(feature = "cuda")]
fn gemma4_harness_expected(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| anyhow::anyhow!("reading {}: {error}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|error| anyhow::anyhow!("parsing {}: {error}", path.display()))
}

#[cfg(feature = "cuda")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// One measured gemma4 harness run, and the receipt it writes.
#[cfg(feature = "cuda")]
struct Gemma4HarnessRun<'a> {
    request_path: Option<&'a std::path::Path>,
    model: &'a std::path::Path,
    cghost: Option<&'a std::path::Path>,
    prompt: &'a str,
    prompt_tokens: usize,
    max_tokens: usize,
    /// `None` for the plain lane, `Some(k)` for a speculative run.
    draft_k: Option<usize>,
    ids: &'a [u32],
    text: &'a str,
    expected: Option<&'a [u32]>,
    load_secs: f64,
    /// Whole generate call, prefill included — the number a user feels.
    generate_secs: f64,
    /// Generate minus prefill. Reported separately because the two lanes are only
    /// comparable on decode: MTP and plain share a prefill, so folding it in dilutes
    /// exactly the difference under test.
    decode_only_secs: f64,
    stats: Option<&'a camelid::gemma4_runtime::Gemma4MtpStats>,
    /// `(hits, misses, resident, capacity)` from the expert cache, lifetime since load.
    sser: Option<(u64, u64, usize, usize)>,
    /// `(hits, misses)` as of the end of prefill. Lifetime totals are dominated by the
    /// prompt sweep -- a layer-major prefill touches most of the 12 GiB payload once --
    /// so decode traffic is only visible as the difference.
    sser_prefill: (u64, u64),
    /// Host expert tier `(hits, storage reads)`, lifetime and at the prefill boundary.
    tier: (u64, u64),
    tier_prefill: (u64, u64),
    tier_eviction_policy: Option<&'static str>,
    tier_residency_prefill: Option<camelid::gemma4_runtime::Gemma4SserHostResidency>,
    tier_residency_end: Option<camelid::gemma4_runtime::Gemma4SserHostResidency>,
    /// Throughput over the second half of the decode forwards, past the arena warm-up.
    ///
    /// Layer-major prefill leaves the expert arena holding the LAST layer's union, so
    /// decode has to re-accumulate a cross-layer working set. On a 48-token request that
    /// warm-up covers half the run, which makes the whole-run average say more about the
    /// handoff from prefill than about how fast the lane decodes.
    steady_tokens_per_second: Option<f64>,
}

#[cfg(feature = "cuda")]
impl Gemma4HarnessRun<'_> {
    /// `(exact_match, first_divergence)` against the expectation, or `(true, None)` when
    /// the run was not gated. A length difference counts as a divergence at the shorter
    /// length, so a truncated run cannot pass by being a prefix.
    fn verdict(&self) -> (bool, Option<usize>) {
        let Some(expected) = self.expected else {
            return (true, None);
        };
        let divergence = self
            .ids
            .iter()
            .zip(expected)
            .position(|(got, want)| got != want)
            .or_else(|| {
                (self.ids.len() != expected.len()).then_some(self.ids.len().min(expected.len()))
            });
        (divergence.is_none(), divergence)
    }

    fn receipt(&self) -> serde_json::Value {
        let (exact_match, divergence) = self.verdict();
        let kwide_rounds = self.stats.map_or(0, |stats| {
            stats.trace.iter().filter(|round| round.kwide).count() as u64
        });
        let cuda_assistant_rounds = self.stats.map_or(0, |stats| {
            stats
                .trace
                .iter()
                .filter(|round| round.cuda_assistant)
                .count() as u64
        });
        let rounds = self.stats.map(|stats| {
            stats
                .trace
                .iter()
                .map(|round| {
                    serde_json::json!({
                        "position": round.position,
                        "bonus_token": round.bonus_token,
                        "requested_verify_k": round.requested_verify_k,
                        "verifier_k": round.verifier_k,
                        "budget_truncated": round.budget_truncated,
                        "drafts": round.drafts,
                        "target": round.target,
                        "accepted": round.accepted,
                        "assistant_ms": round.assistant_ns as f64 / 1e6,
                        "verify_ms": round.verify_ns as f64 / 1e6,
                        "misses": round.misses,
                        // Backward-compatible v1 alias. This was historically described as
                        // storage, but it has always counted VRAM-arena misses.
                        "miss_mib": round.misses as f64 * 3.19,
                        "arena_miss_mib": round.misses as f64 * 3.19,
                        // A VRAM miss served by the pinned host tier never reaches disk.
                        // Keep actual storage traffic separate so the per-round trace can
                        // answer whether verification is I/O- or compute-bound.
                        "storage_reads": round.storage_reads,
                        "storage_mib": round.storage_reads as f64 * 3.19,
                        "kwide": round.kwide,
                        "cuda_assistant": round.cuda_assistant,
                    })
                })
                .collect::<Vec<_>>()
        });
        serde_json::json!({
            "schema": "camelid.gemma4.harness.v1",
            "unix_time": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default(),
            "lane": if self.draft_k.is_some() { "mtp" } else { "plain" },
            "request_json": self.request_path.map(|p| p.display().to_string()),
            "model": self.model.display().to_string(),
            "cghost": self.cghost.map(|p| p.display().to_string()),
            // The prompt hash is the proof that templating matched: two runs with the
            // same hash decoded the same bytes, whatever produced them.
            "prompt_sha256": sha256_hex(self.prompt.as_bytes()),
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "draft_k": self.draft_k,
            "token_ids": self.ids,
            "text": self.text,
            "expected_token_ids": self.expected,
            "exact_match": exact_match,
            "first_divergence": divergence,
            "load_secs": self.load_secs,
            "generate_secs": self.generate_secs,
            "decode_only_secs": self.decode_only_secs,
            "end_to_end_tokens_per_second": self.ids.len() as f64 / self.generate_secs.max(1e-9),
            "decode_tokens_per_second": self.ids.len() as f64 / self.decode_only_secs.max(1e-9),
            "steady_tokens_per_second": self.steady_tokens_per_second,
            "mtp": self.stats.map(|stats| serde_json::json!({
                "rounds": stats.rounds,
                "drafted": stats.drafted,
                "accepted": stats.accepted,
                "alpha": stats.alpha(),
                "acceptance_rate": stats.acceptance_rate(),
                "prefill_ms": stats.prefill_ns as f64 / 1e6,
                "assistant_ms": stats.assistant_ns as f64 / 1e6,
                "verify_ms": stats.verify_ns as f64 / 1e6,
                "verify_ms_per_round": if stats.rounds > 0 {
                    stats.verify_ns as f64 / 1e6 / stats.rounds as f64
                } else {
                    0.0
                },
                "kwide_rounds": kwide_rounds,
                "cuda_assistant_rounds": cuda_assistant_rounds,
                "cuda_assistant": stats.cuda_assistant,
                "cpu_assistant_loaded": !stats.cuda_assistant,
            })),
            "expert_cache": self.expert_cache_json(),
            "rounds": rounds,
        })
    }

    /// Expert-cache and host-tier counters for the receipt.
    ///
    /// Split out of `receipt` because that whole value is one `json!`, which expands
    /// recursively -- adding the tier fields inline pushed it past the macro's recursion
    /// limit. Every derived figure is also computed as a local first: a braced block at a
    /// `json!` value position is parsed as a nested JSON OBJECT, not as a Rust block, so
    /// inlining the arithmetic there fails with a confusing "missing tokens" error.
    fn expert_cache_json(&self) -> Option<serde_json::Value> {
        let ratio = |hits: u64, misses: u64| -> f64 {
            if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            }
        };
        let residency_json =
            |snapshot: Option<camelid::gemma4_runtime::Gemma4SserHostResidency>| {
                snapshot.map(|snapshot| {
                    serde_json::json!({
                        "sser": snapshot.sser_records,
                        "host": snapshot.host_records,
                        "intersection": snapshot.intersection_records,
                        "host_only": snapshot.host_only_records,
                        "union": snapshot.union_records,
                    })
                })
            };
        self.sser.map(|(hits, misses, resident, capacity)| {
            let emitted = self.ids.len().max(1) as f64;
            let rounds = self.stats.map(|stats| stats.rounds).unwrap_or(0);
            let (prefill_hits, prefill_misses) = self.sser_prefill;
            let decode_misses = misses.saturating_sub(prefill_misses);
            let decode_hits = hits.saturating_sub(prefill_hits);
            // A layer-major prefill streams most of the payload once, so its tier lookups
            // are first touches that cannot hit. Mixing them into a lifetime rate makes a
            // tier decode never benefits from look merely mediocre instead of absent.
            let tier_decode_hits = self.tier.0.saturating_sub(self.tier_prefill.0);
            let tier_decode_reads = self.tier.1.saturating_sub(self.tier_prefill.1);
            // With no tier every arena miss is a disk read; with one, only the tier's own
            // storage reads are.
            let decode_storage_reads = if self.tier.0 + self.tier.1 > 0 {
                tier_decode_reads
            } else {
                decode_misses
            };
            // Per ROUND is incomparable across lanes, because a round commits one token
            // plain and `1 + alpha` speculatively. Per committed token is the figure that
            // answers "did batching change what we read?", so it is emitted for both lanes
            // and the round figure only where a round exists.
            let misses_per_round = if rounds > 0 {
                serde_json::json!(misses as f64 / rounds as f64)
            } else {
                serde_json::Value::Null
            };
            serde_json::json!({
                "hits": hits,
                "misses": misses,
                "prefill_hits": prefill_hits,
                "prefill_misses": prefill_misses,
                "decode_hits": decode_hits,
                "decode_misses": decode_misses,
                "decode_misses_per_token": decode_misses as f64 / emitted,
                "decode_hit_rate": ratio(decode_hits, decode_misses),
                // A VRAM-arena miss is not necessarily a disk read: with a host tier in
                // front of storage, 30.7% of them were served from RAM on the measured
                // run. Report the arena miss and the actual storage read separately --
                // conflating them overstated decode storage by 1.5 GiB.
                "decode_storage_reads": decode_storage_reads,
                "decode_storage_mib": decode_storage_reads as f64 * 3.19,
                "tier_hits": self.tier.0,
                "tier_storage_reads": self.tier.1,
                "tier_decode_hits": tier_decode_hits,
                "tier_decode_storage_reads": tier_decode_reads,
                "tier_decode_hit_rate": ratio(tier_decode_hits, tier_decode_reads),
                "host_tier_eviction_policy": self.tier_eviction_policy,
                "residency_prefill": residency_json(self.tier_residency_prefill),
                "residency_end": residency_json(self.tier_residency_end),
                "misses_per_committed_token": misses as f64 / emitted,
                "misses_per_round": misses_per_round,
                "resident_experts": resident,
                "capacity": capacity,
            })
        })
    }

    /// Write the receipt (when asked) and report the gate to stderr. Returns the
    /// exact-match verdict so the caller can set the process exit status: a harness that
    /// prints FAIL and exits 0 is a harness nobody's CI will believe.
    fn finish(&self, path: Option<&std::path::Path>) -> anyhow::Result<bool> {
        let (exact_match, divergence) = self.verdict();
        if let Some(path) = path {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_vec_pretty(&self.receipt())?)?;
            eprintln!("[gemma4-harness] receipt -> {}", path.display());
        }
        if let Some(expected) = self.expected {
            if exact_match {
                eprintln!(
                    "[gemma4-harness] PASS exact {}/{} token ids",
                    self.ids.len(),
                    expected.len()
                );
            } else {
                let at = divergence.unwrap_or(0);
                eprintln!(
                    "[gemma4-harness] FAIL: {} ids vs {} expected; first divergence at {at} \
                     (got {:?}, expected {:?})",
                    self.ids.len(),
                    expected.len(),
                    self.ids.get(at),
                    expected.get(at),
                );
            }
        }
        Ok(exact_match)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Both before ANY CUDA probe: the hook must be in place before cudarc's lazy
    // loaders can panic, or the first (caught, handled) miss still prints.
    quiet_cudarc_loader_panics();
    // Installed AFTER the cudarc filter so this is the OUTER hook: every panic is
    // recorded to disk first, then the filter decides what reaches the console.
    // A packaged install that panics used to leave nothing behind anywhere — the
    // console window was the only sink. See src/diagnostics.rs.
    camelid::diagnostics::install_panic_hook();
    // Make the GPU runtime discoverable before anything probes for a device, so
    // the shipped app needs no PATH setup (no-op off Windows / without CUDA).
    ensure_cuda_runtime_on_path();
    pin_to_high_performance_gpu();

    // No subcommand (a bare double-click of the exe) launches the open-and-use app.
    let command = Cli::parse().command.unwrap_or_else(default_launch_command);

    // §4 safe-boot: a gait/substrate that crashed or wedged the host on the
    // previous run left an `.applying` marker; detect it now — before anything is
    // applied — quarantine that profile, and boot the proven baseline so a crash
    // can never loop. Inert unless the CAMELID_GAIT gate is on, so the default
    // path is byte-identical to today.
    if camelid::gait::gait_enabled() {
        if let Some(dir) = camelid::gait::gait_dir() {
            let _ = camelid::gait::sentinel::reconcile_on_startup(&dir);
        }
    }
    // Deterministic mode opts out of the GPU fast stack entirely; otherwise the CLI
    // defaults to the measured-fastest Metal configuration. Branch before any env is set
    // so the deterministic path never even arms the GPU defaults.
    if command_requests_deterministic(&command) {
        apply_deterministic_mode();
    } else {
        apply_default_fast_stack();
    }

    match command {
        Command::RemoteChat { action } => match action {
            RemoteChatAction::Start { remote } => {
                let status = camelid::remote_chat::start(remote.into_options())?;
                println!("Remote Chat is available within your Tailscale network:");
                println!("  {}", status.url);
                println!("Camelid remains on http://{}.", status.backend);
                println!("The LAN Chat API key is still required in every browser.");
                println!("Tailscale Funnel was not enabled.");
            }
            RemoteChatAction::Status { remote, json } => {
                let status = camelid::remote_chat::status(remote.into_options())?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&status)?);
                } else {
                    println!("Remote Chat URL: {}", status.url);
                    println!("Backend: http://{}", status.backend);
                    println!(
                        "Backend policy: {}",
                        if status.backend_ready {
                            "ready (authenticated LAN Chat only)"
                        } else {
                            "not ready"
                        }
                    );
                    println!("Tailscale transport: {}", status.transport);
                }
            }
            RemoteChatAction::Stop { remote } => {
                let status = camelid::remote_chat::stop(remote.into_options())?;
                println!("Camelid's remote Chat mapping is stopped.");
                if status.transport != camelid::remote_chat::TransportState::Inactive {
                    println!(
                        "Other Tailscale configuration remains on HTTPS port 443 ({}).",
                        status.transport
                    );
                }
            }
        },
        Command::LanKey { rotate } => {
            let credential = camelid::lan_key::provision(rotate)?;
            println!(
                "LAN Chat key {} at {}",
                if credential.created() {
                    "created"
                } else {
                    "loaded"
                },
                credential.path().display()
            );
            println!("\n{}\n", credential.secret());
            println!("Treat this key like a password. Share it directly with the phone user.");
            println!("Never put it in a URL, screenshot, issue, or chat message.");
            println!(
                "\nTrusted LAN:\n  {}",
                lan_chat_serve_command(credential.path())
            );
            println!(
                "\nPrivate access from another network:\n  camelid serve --lan-chat-only --api-key-file \"{}\" --addr 127.0.0.1:8181 --model <MODEL.gguf>\n  camelid remote-chat start",
                credential.path().display()
            );
        }
        Command::Serve {
            addr,
            model,
            threads,
            parallel_linear_min_outputs,
            apple_accelerate_min_elements,
            metal_linear,
            metal_q8,
            log_acceleration,
            spec_decode,
            spec_draft_model,
            spec_draft_tokens,
            cghost,
            expert_cache_mib,
            ghost_strict_cache,
            no_open,
            deterministic,
            enable_thinking,
            models_dir,
            gpu,
            kv_quant,
            server,
        } => {
            std::env::set_var("CAMELID_KV_QUANT", kv_quant.to_string());
            configure_rayon_threads(threads)?;
            camelid::capability::HardwareProfile::detect().log();
            // In deterministic mode the engine fails every Metal gate closed (see
            // `apply_deterministic_mode`); don't also arm the Metal tuning env or the
            // GPU fast-load nocopy default, which would only be contradictory no-ops.
            apply_runtime_tuning_env(
                parallel_linear_min_outputs,
                apple_accelerate_min_elements,
                metal_linear && !deterministic,
                metal_q8 && !deterministic,
            );
            apply_spec_decode_env(spec_decode, spec_draft_model, spec_draft_tokens);
            if let Some(cghost) = cghost {
                std::env::set_var("CAMELID_GEMMA4_GHOST_CGHOST", cghost);
                std::env::set_var(
                    "CAMELID_GEMMA4_GHOST_CACHE_MIB",
                    expert_cache_mib.to_string(),
                );
                std::env::set_var(
                    "CAMELID_GEMMA4_GHOST_STRICT_CACHE",
                    ghost_strict_cache.to_string(),
                );
            }
            if !deterministic {
                apply_serve_nocopy_default();
            }
            if log_acceleration {
                log_acceleration_state();
            }
            // Seed the runtime GPU switches from --gpu / CAMELID_GPU before any model
            // load, for headless/agent runs with no Settings UI to click. `auto` (default)
            // leaves the atomics uninitialised so they lazy-seed exactly as today
            // (master acceleration from CUDA-or-Metal capability, CUDA hybrid Q8 from
            // CAMELID_CUDA_Q8). `on`/`off` are authoritative at startup and override the env seed; the UI
            // `POST /api/runtime/gpu` can still flip state live afterwards. Deterministic
            // mode and CAMELID_CUDA_RESIDENT_DECODE=0 still force the GPU off at their own
            // call sites regardless of this seed.
            if let Some(enabled) = resolved_gpu_switch(gpu, deterministic) {
                camelid::cuda::set_gpu_accel_enabled(enabled);
                camelid::cuda::set_runtime_enabled(enabled);
            }
            let gpu_info = camelid::cuda::gpu_acceleration_info();
            tracing::info!(
                gpu_mode = ?gpu,
                gpu_backend = gpu_info.backend,
                gpu_available = gpu_info.available,
                gpu_enabled = camelid::cuda::gpu_accel_enabled(),
                "camelid gpu mode resolved"
            );
            #[cfg(target_os = "macos")]
            unsafe {
                pthread_set_qos_class_self_np(0x09, 0); // QOS_CLASS_BACKGROUND (forces network I/O onto E-cores)
            }
            // Open-and-use launch: if no model was named, load the user's saved
            // default from the configured model library. With no saved choice,
            // the first local GGUF is the zero-configuration default.
            let model = match model {
                Some(path) => Some(api::StartupModel::explicit(path)),
                None => {
                    auto_select_model(models_dir.as_deref()).map(api::StartupModel::auto_selected)
                }
            };
            // Open the browser only when run interactively and not opted out.
            let open_ui = !no_open && std::io::IsTerminal::is_terminal(&std::io::stdout());
            // Journal the run. A `session_start` with no matching `session_exit`
            // means the process did not leave through the failure path. That
            // includes an ordinary Ctrl-C as well as an external kill — `serve`
            // installs no signal handler — so read it as "did not fail on the
            // way out", never as proof of a kill. See src/diagnostics.rs.
            camelid::diagnostics::record_session_start(VERSION, &addr.to_string());
            eprintln!(
                "  Diagnostics log: {}",
                camelid::diagnostics::log_path().display()
            );
            let served = api::serve(
                addr,
                threads,
                model,
                open_ui,
                enable_thinking,
                models_dir,
                server.into_serve_options(),
            )
            .await;
            match &served {
                Ok(()) => camelid::diagnostics::record_session_exit("ok", None),
                Err(err) => {
                    camelid::diagnostics::record_session_exit("error", Some(&err.to_string()));
                }
            }
            served?
        }
        Command::Chat {
            model,
            addr,
            system,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            no_stream,
            plain,
            models_dir,
            agent,
            workdir,
            max_steps,
            auto_approve,
            yolo,
            allow_net,
            allow_fs,
            allow_mcp,
            trust_mcp_server,
            shell_timeout,
            enable_thinking,
            audit_webhook,
            shell_sandbox,
        } => {
            let code = chat::run_chat(chat::ChatOptions {
                model,
                addr,
                system,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                no_stream,
                plain,
                models_dir: models_dir.unwrap_or_else(|| PathBuf::from("models")),
                exec_goal: None,
                agent,
                workdir,
                max_steps,
                auto_approve,
                yolo,
                allow_net,
                allow_fs,
                allow_mcp,
                trust_mcp_servers: trust_mcp_server,
                shell_timeout,
                enable_thinking,
                audit_webhook,
                shell_sandbox,
            })?;
            if code != 0 {
                std::process::exit(code);
            }
        }
        Command::AgentEval {
            model,
            addr,
            load_timeout,
            max_steps,
            max_tokens,
            receipt_dir,
        } => {
            let code = chat::run_agent_eval(chat::AgentEvalOptions {
                model,
                addr,
                load_timeout,
                max_steps,
                max_tokens,
                receipt_dir,
            })?;
            std::process::exit(code);
        }
        Command::AgentSyscapEval { receipt_dir } => {
            let code = chat::run_agent_syscap_eval(chat::AgentSyscapOptions { receipt_dir })?;
            std::process::exit(code);
        }
        Command::Subagent { task_file } => {
            let code = chat::run_subagent_worker(&task_file)?;
            std::process::exit(code);
        }
        Command::AgentOrchestrationEval {
            receipt_dir,
            model,
            addr,
            load_timeout,
        } => {
            let code = chat::run_agent_orchestration_eval(chat::AgentOrchestrationOptions {
                receipt_dir,
                model,
                addr,
                load_timeout,
            })?;
            std::process::exit(code);
        }
        Command::AgentOrchestrationBench {
            receipt_dir,
            model,
            addr,
            load_timeout,
        } => {
            let code = chat::run_agent_orchestration_bench(chat::AgentOrchestrationBenchOptions {
                receipt_dir,
                model,
                addr,
                load_timeout,
            })?;
            std::process::exit(code);
        }
        Command::ServeDistributed {
            role,
            addr,
            worker_addr,
            layer_range,
            model,
            threads,
            distributed_token,
            distributed_token_file,
            server,
        } => {
            configure_rayon_threads(threads)?;
            let distributed_token = camelid::distributed::resolve_distributed_token(
                distributed_token,
                distributed_token_file.as_deref(),
            )
            .map_err(anyhow::Error::msg)?;

            let parts: Vec<&str> = layer_range.split("..").collect();
            anyhow::ensure!(
                parts.len() == 2,
                "Layer range must be in format START..END (e.g. 0..16)"
            );
            let layer_start = parts[0].parse::<usize>()?;
            let layer_end = parts[1].parse::<usize>()?;
            anyhow::ensure!(
                layer_start < layer_end,
                "layer_start must be less than layer_end"
            );

            let _ = camelid::distributed::DISTRIBUTED_RANGE.set((layer_start, layer_end));

            if role == "coordinator" {
                let worker_addr_str = worker_addr.ok_or_else(|| {
                    anyhow::anyhow!("--worker-addr is required in coordinator mode")
                })?;

                let gguf = camelid::gguf::read_metadata(&model)?;
                let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
                camelid::distributed::PipelineRole::Coordinator.validate_layer_range(
                    layer_start,
                    layer_end,
                    config.block_count as usize,
                )?;
                let identity = camelid::distributed::NodeIdentity::for_model(
                    &model,
                    config.block_count,
                    config.embedding_length,
                    (layer_end as u32)..(config.block_count),
                )?
                .with_token(distributed_token.clone());
                let distributed_model_sha256 = identity.model_sha256.clone();

                tracing::info!(worker_addr = %worker_addr_str, "Coordinator connecting to worker");
                let client =
                    camelid::distributed::DistributedClient::connect(&worker_addr_str, &identity)?;
                camelid::distributed::DISTRIBUTED_CLIENT
                    .set(client)
                    .map_err(|_| anyhow::anyhow!("Failed to set global distributed client lock"))?;
                tracing::info!("Coordinator connected to worker successfully");

                #[cfg(target_os = "macos")]
                unsafe {
                    pthread_set_qos_class_self_np(0x09, 0); // QOS_CLASS_BACKGROUND (forces network I/O onto E-cores)
                }
                api::serve(
                    addr,
                    threads,
                    Some(api::StartupModel::distributed(
                        model,
                        distributed_model_sha256,
                    )),
                    false,
                    false,
                    None,
                    server.into_serve_options(),
                )
                .await?
            } else if role == "worker" {
                let gguf = camelid::gguf::read_metadata(&model)?;
                ensure_arch_has_direct_dense_session(
                    &gguf,
                    DenseLaneWindowedForward::CpuDenseOnly,
                )?;
                let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
                camelid::distributed::PipelineRole::Worker.validate_layer_range(
                    layer_start,
                    layer_end,
                    config.block_count as usize,
                )?;
                let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
                let store = camelid::tensor::TensorStore::open(&model, &gguf);

                tracing::info!(
                    "Worker loading partitioned weights (layers {}..{})",
                    layer_start,
                    layer_end
                );
                let (load_embedding, load_output) =
                    camelid::distributed::PipelineRole::Worker.tensor_ownership();
                let weights = camelid::inference::LlamaLoadedWeights::load_distributed(
                    &store,
                    &binding,
                    layer_start,
                    layer_end,
                    load_embedding,
                    load_output,
                )?;

                tracing::info!("Worker weights loaded successfully. Initializing session.");
                let identity = camelid::distributed::NodeIdentity::for_model(
                    &model,
                    config.block_count,
                    config.embedding_length,
                    (layer_start as u32)..(layer_end as u32),
                )?;
                let session = camelid::inference::LlamaInferenceSession::new(config, weights)?;

                let addr_str = addr.to_string();
                #[cfg(target_os = "macos")]
                unsafe {
                    pthread_set_qos_class_self_np(0x09, 0); // QOS_CLASS_BACKGROUND (forces network I/O onto E-cores)
                }
                camelid::distributed::run_worker_loop(
                    &addr_str,
                    session,
                    identity,
                    distributed_token,
                    server.allow_unauthenticated_remote,
                )?;
            } else {
                anyhow::bail!("Invalid role: {role}. Must be 'coordinator' or 'worker'");
            }
        }
        Command::BenchNetwork {
            role,
            addr,
            ping_count,
            payload_size,
            bandwidth_mb,
        } => {
            if role == "coordinator" {
                camelid::distributed::run_network_benchmark_coordinator(
                    &addr,
                    ping_count,
                    payload_size,
                    bandwidth_mb,
                )?;
            } else if role == "worker" {
                camelid::distributed::run_network_benchmark_worker(&addr)?;
            } else {
                anyhow::bail!("Invalid role: {role}. Must be 'coordinator' or 'worker'");
            }
        }
        Command::Fabric { action } => match action {
            FabricAction::Status {
                nodes,
                nodes_file,
                bearer,
                transport,
                timeout_ms,
                json,
            } => {
                let bearer = fabric_bearer(bearer);
                let fabric = configure_node_transport(fabric_from(nodes, nodes_file)?, &transport)?
                    .with_timeout(std::time::Duration::from_millis(timeout_ms))
                    .with_bearer(bearer.as_deref());
                let snapshots = fabric.observe();
                if json {
                    println!("{}", serde_json::to_string_pretty(&snapshots)?);
                } else {
                    print!("{}", camelid::fabric::render_status(&snapshots));
                }
            }
            FabricAction::Route {
                nodes,
                nodes_file,
                mode,
                model,
                sticky,
                bearer,
                transport,
                timeout_ms,
                json,
            } => {
                let bearer = fabric_bearer(bearer);
                let mode = stateless_route_mode(&mode)?;
                let fabric = configure_node_transport(fabric_from(nodes, nodes_file)?, &transport)?
                    .with_timeout(std::time::Duration::from_millis(timeout_ms))
                    .with_bearer(bearer.as_deref());
                let snapshots = fabric.observe();
                let request = camelid::fabric::RouteRequest::new(mode)
                    .with_model(model.as_deref())
                    .with_sticky(sticky.as_deref());

                // A fabric that cannot place the request is a failure the caller
                // must see in the exit code, not only in the text.
                let decision = camelid::fabric::route(&snapshots, &request)
                    .map_err(|error| anyhow::anyhow!("{error}"))?;

                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "label": decision.label,
                            "reason": decision.reason.as_str(),
                            "affinity_lost": decision.affinity_lost,
                        }))?
                    );
                } else {
                    println!("route -> {} ({})", decision.label, decision.reason.as_str());
                    if let Some(previous) = &decision.affinity_lost {
                        println!(
                            "note: affinity to `{previous}` could not be honoured; \
                             this request re-prefills on a cold node"
                        );
                    }
                }
            }
            FabricAction::Run {
                nodes,
                nodes_file,
                prompt,
                mode,
                model,
                sticky,
                bearer,
                transport,
                max_tokens,
                timeout_ms,
                forward_timeout_s,
                json,
            } => {
                let bearer = fabric_bearer(bearer);
                let mode = stateless_route_mode(&mode)?;
                let fabric = configure_node_transport(fabric_from(nodes, nodes_file)?, &transport)?
                    .with_timeout(std::time::Duration::from_millis(timeout_ms))
                    .with_bearer(bearer.as_deref());

                let request = camelid::fabric::RouteRequest::new(mode)
                    .with_model(model.as_deref())
                    .with_sticky(sticky.as_deref());
                // Held until the forward below returns: the node is busy for
                // exactly that long.
                let placement = fabric
                    .place(&request)
                    .map_err(|error| anyhow::anyhow!("{error}"))?;
                let decision = placement.decision();
                let chosen = placement.node();

                // The request must name a model. Prefer the operator's choice,
                // otherwise use whatever the chosen node has loaded.
                let model_id = match model {
                    Some(model) => model,
                    None => chosen
                        .active_model_id()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "node `{}` reports no active model; pass --model",
                                decision.label
                            )
                        })?
                        .to_string(),
                };

                let body = camelid::fabric::forward::chat_request(&model_id, &prompt, max_tokens);
                let answer = fabric
                    .forward_to(
                        &chosen.spec,
                        "/v1/chat/completions",
                        &body,
                        std::time::Duration::from_secs(forward_timeout_s),
                        // One request per process: this ends when the process does,
                        // so there is no client that can leave while it runs.
                        &camelid::fabric::Cancel::never(),
                    )
                    .map_err(|error| anyhow::anyhow!("{error}"))?;

                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "node": answer.label,
                            "reason": decision.reason.as_str(),
                            "affinity_lost": decision.affinity_lost,
                            "model": model_id,
                            "status": answer.status,
                            "elapsed_ms": answer.elapsed.as_millis() as u64,
                            "body": answer.body,
                        }))?
                    );
                } else {
                    println!(
                        "[{} · {} · {} · {} ms]",
                        answer.label,
                        decision.reason.as_str(),
                        model_id,
                        answer.elapsed.as_millis()
                    );
                    match camelid::fabric::forward::completion_text(&answer.body) {
                        Some(text) => println!("{text}"),
                        None => println!("(no completion text; HTTP {})", answer.status),
                    }
                }

                if !answer.is_success() {
                    let detail = camelid::fabric::forward::error_message(&answer.body)
                        .unwrap_or("no message");
                    anyhow::bail!(
                        "node `{}` answered HTTP {}: {detail}",
                        answer.label,
                        answer.status
                    );
                }
            }
            FabricAction::Serve {
                nodes,
                nodes_file,
                addr,
                mode,
                bearer,
                transport,
                api_key,
                api_key_file,
                timeout_ms,
                observation_max_age_ms,
                max_forward_attempts,
                forward_timeout_s,
                allow_unauthenticated_remote,
                tls_cert,
                tls_key,
                allow_cleartext_remote,
                client_keys,
            } => {
                let mode = route_mode(&mode)?;
                let bearer = fabric_bearer(bearer);
                let auth = match client_keys {
                    Some(path) => camelid::fabric::server::ClientAuth::from_key_file(path)?,
                    None => camelid::fabric::server::ClientAuth::resolve(api_key, api_key_file)?,
                };
                // Loaded here, before anything is bound, announced or probed:
                // a certificate that cannot be read is a refusal, and a refusal
                // must not arrive after the operator has been told the proxy is
                // listening on an https address.
                let tls = camelid::fabric::server::ProxyTls::resolve(tls_cert, tls_key).await?;
                // Same reason: a node file that cannot be read is a refusal,
                // and it has to arrive before the listening line.
                let fabric = configure_node_transport(fabric_from(nodes, nodes_file)?, &transport)?
                    .with_timeout(std::time::Duration::from_millis(timeout_ms))
                    .with_bearer(bearer.as_deref())
                    .with_max_observation_age(std::time::Duration::from_millis(
                        observation_max_age_ms,
                    ))
                    .with_max_forward_attempts(max_forward_attempts);

                // Bind before announcing, so a refused or already-taken address
                // never prints a listening line it did not earn.
                let listener = camelid::fabric::server::bind(
                    addr,
                    &auth,
                    tls.as_ref(),
                    camelid::fabric::server::RemoteAcknowledgements {
                        unauthenticated: allow_unauthenticated_remote,
                        cleartext: allow_cleartext_remote,
                    },
                )
                .await?;
                let bound = listener.local_addr()?;
                let scheme = if tls.is_some() { "https" } else { "http" };
                println!("fabric serve listening on {scheme}://{bound}");
                println!("node transport: {}", fabric.node_transport_description());
                // A key set is the only thing here an operator can get subtly
                // wrong without being told: a file that parsed but named one
                // client when they meant three looks exactly like success.
                if auth.is_reloadable() {
                    println!(
                        "serving {} named clients; editing the key file revokes \
                         a client without a restart",
                        auth.client_count()
                    );
                }
                // The startup report below names the nodes, but not that the
                // set can still change — which is the one thing an operator
                // cannot tell by looking at it.
                if fabric.is_reloadable() {
                    println!(
                        "watching the node file; editing it adds or removes a \
                         machine without a restart"
                    );
                }
                // Nothing is being served yet, so this probe costs no request, and
                // it is the only chance to tell the operator about a node that is
                // not there before a client discovers it for them.
                print!("{}", camelid::fabric::startup_report(&fabric.observe()));
                camelid::fabric::server::serve_on(
                    listener,
                    fabric,
                    camelid::fabric::server::ServeConfig {
                        mode,
                        forward_timeout: std::time::Duration::from_secs(forward_timeout_s),
                        auth,
                        tls,
                        bound,
                    },
                )
                .await?;
            }
        },
        Command::Inspect { path } => {
            let gguf = read_metadata(path)?;
            println!("{}", serde_json::to_string_pretty(&gguf)?);
        }
        Command::InspectPrefix { path, declared_len } => {
            let prefix_len = std::fs::metadata(&path)?.len();
            anyhow::ensure!(
                prefix_len <= declared_len,
                "GGUF prefix is {prefix_len} bytes, larger than declared artifact length {declared_len}"
            );
            let mut gguf = read_metadata_with_len(&path, declared_len)?;
            gguf.path = PathBuf::from("<remote-gguf-prefix>");
            println!("{}", serde_json::to_string_pretty(&gguf)?);
        }
        Command::InspectSource { path } => {
            let inspection = inspect_model_source(path)?;
            println!("{}", serde_json::to_string_pretty(&inspection)?);
        }
        Command::Tokenize {
            model,
            declared_len,
            prompt,
            file,
            parse_special,
            no_add_special,
        } => {
            // Deep-recursion headroom (large vocab BPE build): dedicated big-stack
            // thread so the harness behaves identically in debug and release.
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || -> anyhow::Result<()> {
                    let gguf = read_tokenizer_gguf(&model, declared_len)?;
                    let tokenizer = Tokenizer::from_gguf(&gguf)?;
                    let inputs: Vec<String> = if let Some(p) = prompt {
                        vec![p]
                    } else if let Some(f) = file {
                        serde_json::from_str(&std::fs::read_to_string(f)?)?
                    } else {
                        anyhow::bail!("tokenize: provide --prompt or --file");
                    };
                    for text in &inputs {
                        let ids = tokenizer.encode(text, !no_add_special, parse_special)?;
                        let decoded = tokenizer.decode(&ids, false)?;
                        println!("{}", serde_json::json!({ "ids": ids, "decoded": decoded }));
                    }
                    Ok(())
                })?
                .join()
                .map_err(|_| anyhow::anyhow!("tokenize worker panicked"))??;
        }
        Command::RunnableSmoke { path } => {
            let path_str = path.to_string_lossy();
            match camelid::runnable::smoke_admit(&path_str) {
                Ok(report) => {
                    eprintln!(
                        "smoke-admission PASSED: {}/{}/{:?}",
                        report.architecture, report.quant, report.tokenizer
                    );
                    eprintln!(
                        "  prompt_tokens={} logits=[{:.1}, {:.1}]",
                        report.prompt_token_count, report.logit_min, report.logit_max
                    );
                    eprintln!("  greedy: {:?}", report.generated_text);
                    eprintln!(
                        "  (runnable receipt below — attests deterministic execution, not parity)"
                    );
                    // The runnable receipt (lane=runnable, never copper) to stdout.
                    println!("{}", serde_json::to_string_pretty(&report.receipt)?);
                }
                Err(err) => {
                    eprintln!("smoke-admission REFUSED/FAILED: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::PlanOffload {
            model,
            arch,
            budget_mb,
            context,
            safety_mb,
        } => {
            let profile = camelid::capability::HardwareProfile::detect();
            profile.log();
            let free_vram = match budget_mb {
                Some(mb) => {
                    println!("[offload] forced VRAM budget: {mb} MiB");
                    mb * 1024 * 1024
                }
                None => {
                    anyhow::ensure!(
                        profile.cuda_available,
                        "no CUDA device — offloading is a no-op; the CPU backend already \
                         holds all weights in system RAM"
                    );
                    profile.cuda_vram_free_bytes
                }
            };
            let context = context.unwrap_or(4096);
            let safety_mb = safety_mb.unwrap_or(256);
            let (config, plan) = if let Some(path) = model {
                let gguf = read_metadata(&path)?;
                let config = LlamaModelConfig::from_gguf(&gguf)?;
                let plan = camelid::offload::OffloadPlan::from_gguf(
                    &gguf, &config, free_vram, context, safety_mb,
                );
                (config, plan)
            } else if let Some(arch) = arch {
                let config = known_arch_config(&arch)?;
                let plan = camelid::offload::OffloadPlan::from_dims(
                    &config, free_vram, context, safety_mb,
                );
                (config, plan)
            } else {
                anyhow::bail!("provide a model path or --arch <name>");
            };
            let head_dim = config
                .attention_key_length
                .unwrap_or(config.embedding_length / config.attention_head_count.max(1));
            println!(
                "model: layers={} hidden={} ffn={} heads={} kv_heads={} head_dim={} vocab={:?} | KV reserved at context={}",
                config.block_count,
                config.embedding_length,
                config.feed_forward_length,
                config.attention_head_count,
                config.attention_head_count_kv,
                head_dim,
                config.vocab_size,
                context,
            );
            println!("{}", plan.describe());
            let map: String = plan
                .layer_resident
                .iter()
                .map(|&r| if r { 'V' } else { 'H' })
                .collect();
            println!("[offload] layer map (V=VRAM, H=host): {map}");
        }
        Command::Pull { model, models_dir } => {
            let dir = models_dir.unwrap_or_else(|| PathBuf::from("models"));
            camelid::catalog::run_pull(model.as_deref(), &dir)?;
        }
        Command::Gemma4Generate {
            path,
            prompt,
            max_tokens,
            force_tokens,
            dump_step_logits,
        } => {
            eprintln!("[gemma4] loading {}...", path.display());
            let t0 = std::time::Instant::now();
            let runtime = camelid::gemma4_runtime::Gemma4Runtime::load(&path)?;
            if force_tokens.is_none() && dump_step_logits.is_none() {
                // Default arm — byte-identical behavior to before the BASALT
                // harness flags existed.
                eprintln!(
                    "[gemma4] loaded in {:.1}s; generating {max_tokens} tokens...",
                    t0.elapsed().as_secs_f32()
                );
                let t1 = std::time::Instant::now();
                let (out, ids) = runtime.generate_greedy(&prompt, max_tokens)?;
                let gen = t1.elapsed().as_secs_f32();
                eprintln!(
                    "[gemma4] generated in {gen:.1}s ({:.2} tok/s)",
                    ids.len() as f32 / gen
                );
                eprintln!("[gemma4] token_ids: {ids:?}");
                println!("{prompt}{out}");
            } else {
                // BASALT Phase 3 harness surface (basalt_eval_protocol.md §5.1):
                // forced decode and/or per-step full-logit dumps. NO engine math
                // changes — both modes drive the same step loop as generate_greedy.
                eprintln!("[gemma4] loaded in {:.1}s", t0.elapsed().as_secs_f32());
                let forced: Option<Vec<u32>> = match &force_tokens {
                    Some(p) => {
                        let text = std::fs::read_to_string(p)?;
                        let ids = parse_forced_tokens(&text)
                            .map_err(|e| anyhow::anyhow!("--force-tokens {}: {e}", p.display()))?;
                        // Vocab bound known here (post-load): refuse out-of-range
                        // ids before any decode step runs.
                        validate_forced_token_vocab(&ids, runtime.vocab_size())
                            .map_err(|e| anyhow::anyhow!("--force-tokens {}: {e}", p.display()))?;
                        eprintln!(
                            "[gemma4] teacher-forcing {} tokens from {}",
                            ids.len(),
                            p.display()
                        );
                        Some(ids)
                    }
                    None => None,
                };
                let dump_dir = dump_step_logits.clone();
                if let Some(dir) = &dump_dir {
                    // Refuse mixing this run's step_<i>.bin dumps into a
                    // directory that already has contents.
                    ensure_dump_dir_empty(dir).map_err(|e| anyhow::anyhow!(e))?;
                    std::fs::create_dir_all(dir)?;
                }

                let mut records: Vec<Gemma4StepRecord> = Vec::new();
                let mut vocab_size = 0usize;
                let mut dump_err: Option<std::io::Error> = None;
                let t1 = std::time::Instant::now();

                let (mode, prompt_token_ids, greedy_out) = match &forced {
                    Some(ids) => {
                        let ptoks = runtime.forced_decode(&prompt, ids, |i, logits| {
                            vocab_size = logits.len();
                            let (argmax_id, argmax_logit) = greedy_argmax(logits);
                            records.push(Gemma4StepRecord {
                                step: i,
                                forced_id: Some(ids[i]),
                                argmax_id,
                                argmax_logit,
                                top32: top_n_logits(logits, 32),
                            });
                            if let Some(dir) = &dump_dir {
                                if dump_err.is_none() {
                                    if let Err(e) = write_step_logits(dir, i, logits) {
                                        dump_err = Some(e);
                                    }
                                }
                            }
                        })?;
                        ("forced", ptoks, None)
                    }
                    None => {
                        // --dump-step-logits alone: observed greedy decode
                        // (token-identical to generate_greedy on the plain loop).
                        let ptoks = runtime.tokenizer().encode(&prompt, true, true)?;
                        let (out, ids) = runtime.generate_greedy_observed(
                            &prompt,
                            max_tokens,
                            |i, logits| {
                                vocab_size = logits.len();
                                let (argmax_id, argmax_logit) = greedy_argmax(logits);
                                records.push(Gemma4StepRecord {
                                    step: i,
                                    forced_id: None,
                                    argmax_id,
                                    argmax_logit,
                                    top32: top_n_logits(logits, 32),
                                });
                                if let Some(dir) = &dump_dir {
                                    if dump_err.is_none() {
                                        if let Err(e) = write_step_logits(dir, i, logits) {
                                            dump_err = Some(e);
                                        }
                                    }
                                }
                            },
                        )?;
                        ("greedy", ptoks, Some((out, ids)))
                    }
                };
                if let Some(e) = dump_err {
                    return Err(anyhow::anyhow!(
                        "--dump-step-logits write failed in {}: {e}",
                        dump_dir
                            .as_ref()
                            .expect("dump dir set when dump_err set")
                            .display()
                    ));
                }
                let gen = t1.elapsed().as_secs_f32();
                eprintln!(
                    "[gemma4] {} steps in {gen:.1}s ({:.2} steps/s)",
                    records.len(),
                    records.len() as f32 / gen.max(1e-9)
                );

                let meta = Gemma4StepMeta {
                    protocol: "basalt_eval_protocol.md §5.1/§5.3",
                    mode,
                    model: path.display().to_string(),
                    prompt: prompt.clone(),
                    prompt_token_ids,
                    vocab_size,
                    step_count: records.len(),
                    logits_dtype: "f32_le",
                    logits_file_pattern: "step_<i>.bin",
                    steps: records,
                };
                if let Some(dir) = &dump_dir {
                    let meta_path = dir.join("meta.json");
                    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
                    eprintln!(
                        "[gemma4] wrote {} step_<i>.bin dumps + meta.json to {}",
                        meta.step_count,
                        dir.display()
                    );
                }
                match (mode, &greedy_out) {
                    // Forced mode: stdout is the machine-readable step record.
                    ("forced", _) => println!("{}", serde_json::to_string_pretty(&meta)?),
                    // Greedy+dump mode: stdout keeps the default arm's shape.
                    (_, Some((out, ids))) => {
                        eprintln!("[gemma4] token_ids: {ids:?}");
                        println!("{prompt}{out}");
                    }
                    _ => unreachable!("greedy mode always carries generate output"),
                }
            }
        }
        Command::Gemma4EvalPack {
            path,
            packs,
            baseline_dir,
            score,
        } => {
            #[derive(serde::Deserialize)]
            struct PackPrompt {
                id: String,
                text: String,
                #[serde(default)]
                max_new_tokens: usize,
            }
            #[derive(serde::Deserialize)]
            struct Pack {
                prompts: Vec<PackPrompt>,
            }
            let mut prompts: Vec<PackPrompt> = Vec::new();
            for p in &packs {
                let txt = std::fs::read_to_string(p)
                    .map_err(|e| anyhow::anyhow!("--pack {}: {e}", p.display()))?;
                let parsed: Pack = serde_json::from_str(&txt)
                    .map_err(|e| anyhow::anyhow!("--pack {}: {e}", p.display()))?;
                prompts.extend(parsed.prompts);
            }
            eprintln!(
                "[gemma4-eval] loading {} ({} prompts, mode={})...",
                path.display(),
                prompts.len(),
                if score { "score" } else { "baseline" }
            );
            let t0 = std::time::Instant::now();
            let runtime = camelid::gemma4_runtime::Gemma4Runtime::load(&path)?;
            eprintln!("[gemma4-eval] loaded in {:.1}s", t0.elapsed().as_secs_f32());
            if !score {
                std::fs::create_dir_all(&baseline_dir)?;
            }
            let mut total = 0usize;
            let mut agree = 0usize;
            let t1 = std::time::Instant::now();
            for pr in &prompts {
                let f = baseline_dir.join(format!("{}.txt", pr.id));
                if score {
                    let text = std::fs::read_to_string(&f)
                        .map_err(|e| anyhow::anyhow!("baseline {}: {e}", f.display()))?;
                    let ids = parse_forced_tokens(&text)
                        .map_err(|e| anyhow::anyhow!("baseline {}: {e}", f.display()))?;
                    validate_forced_token_vocab(&ids, runtime.vocab_size())
                        .map_err(|e| anyhow::anyhow!("baseline {}: {e}", f.display()))?;
                    let mut m = 0usize;
                    runtime.forced_decode(&pr.text, &ids, |i, logits| {
                        let (argmax_id, _) = greedy_argmax(logits);
                        if argmax_id == ids[i] {
                            m += 1;
                        }
                    })?;
                    total += ids.len();
                    agree += m;
                    eprintln!(
                        "[gemma4-eval]   {:<16} {:>3}/{:<3} = {:.1}%",
                        pr.id,
                        m,
                        ids.len(),
                        100.0 * m as f64 / ids.len().max(1) as f64
                    );
                } else {
                    let (_out, ids) = runtime.generate_greedy(&pr.text, pr.max_new_tokens)?;
                    let body = ids
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    std::fs::write(&f, body)?;
                    total += ids.len();
                    eprintln!(
                        "[gemma4-eval]   {:<16} {} tokens -> {}",
                        pr.id,
                        ids.len(),
                        f.display()
                    );
                }
            }
            let secs = t1.elapsed().as_secs_f32();
            if score {
                let pct = 100.0 * agree as f64 / total.max(1) as f64;
                eprintln!(
                    "[gemma4-eval] TEACHER-FORCED TOP-1 AGREEMENT: {agree}/{total} = {pct:.1}% ({secs:.1}s)"
                );
                println!(
                    "{{\"model\":{:?},\"agreement_pct\":{:.1},\"agree\":{},\"total\":{}}}",
                    path.display(),
                    pct,
                    agree,
                    total
                );
            } else {
                eprintln!(
                    "[gemma4-eval] baseline: {total} tokens across {} prompts ({secs:.1}s)",
                    prompts.len()
                );
                println!(
                    "{{\"model\":{:?},\"baseline_total\":{}}}",
                    path.display(),
                    total
                );
            }
        }
        #[cfg(feature = "cuda")]
        Command::Gemma4CudaGenerate {
            path,
            cghost,
            expert_cache_mib,
            prompt,
            max_tokens,
            mtp_assistant,
            mtp_draft_k,
            request_json,
            expect_token_ids,
            receipt,
        } => {
            // Resolve the fixture BEFORE the model load. A typo in the request file
            // should cost a parse error, not ten seconds of resident load first.
            let (prompt, max_tokens) = match request_json.as_deref() {
                Some(request) => gemma4_harness_request(request, max_tokens)?,
                None => (prompt, max_tokens),
            };
            let expected = expect_token_ids
                .as_deref()
                .map(gemma4_harness_expected)
                .transpose()?;
            // The MTP lane's receipted profile becomes the default the moment the
            // lane is selected — BEFORE the model load, where every gate resolves.
            // An explicitly set variable always wins, so receipted arms reproduce
            // byte-for-byte, and the plain lane (no --mtp-assistant) is untouched.
            if mtp_assistant.is_some() {
                let applied =
                    camelid::gemma4_runtime::gemma4_mtp_profile_defaults_to_apply(|key| {
                        std::env::var_os(key).is_some()
                    });
                if !applied.is_empty() {
                    for (key, value) in &applied {
                        std::env::set_var(key, value);
                    }
                    eprintln!(
                        "[gemma4-mtp] promoted profile: {} default(s) applied ({})",
                        applied.len(),
                        applied
                            .iter()
                            .map(|(key, value)| format!("{key}={value}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
            }
            eprintln!("[gemma4-cuda] loading resident {}...", path.display());
            let t0 = std::time::Instant::now();
            let mut runtime = match cghost.as_deref() {
                Some(cghost) => camelid::gemma4_runtime::Gemma4CudaResident::load_ghost_moe(
                    &path,
                    cghost,
                    expert_cache_mib,
                    false,
                    4096,
                )?,
                None => camelid::gemma4_runtime::Gemma4CudaResident::load(&path, 4096)?,
            };
            let load_secs = t0.elapsed().as_secs_f64();
            eprintln!(
                "[gemma4-cuda] resident loaded in {load_secs:.1}s; generating {max_tokens} tokens..."
            );
            let prompt_tokens = runtime.tokenizer().encode(&prompt, true, true)?.len();
            let t1 = std::time::Instant::now();
            if let Some(assistant_dir) = mtp_assistant.as_deref() {
                let t_load = std::time::Instant::now();
                let cuda_assistant = runtime.try_load_mtp_cuda_assistant(assistant_dir)?;
                let t_gen = std::time::Instant::now();
                let (out, ids, stats) = if cuda_assistant {
                    eprintln!(
                        "[gemma4-mtp-cuda] assistant loaded in {:.1}s ({} resident bytes, cpu_assistant_loaded=0); drafting K={mtp_draft_k}",
                        t_load.elapsed().as_secs_f64(),
                        runtime.mtp_cuda_assistant_resident_bytes().unwrap_or(0),
                    );
                    runtime.generate_greedy_mtp_cuda(&prompt, max_tokens, mtp_draft_k)?
                } else {
                    eprintln!(
                        "[gemma4-mtp] loading CPU MTP assistant from {}...",
                        assistant_dir.display()
                    );
                    let assistant = camelid::gemma4_mtp::Gemma4MtpAssistant::load(assistant_dir)?;
                    eprintln!(
                        "[gemma4-mtp] CPU assistant loaded in {:.1}s (vocab {}); drafting K={mtp_draft_k}",
                        t_load.elapsed().as_secs_f64(),
                        assistant.vocab_size()
                    );
                    runtime.generate_greedy_mtp(&prompt, max_tokens, &assistant, mtp_draft_k)?
                };
                let wall = t_gen.elapsed().as_secs_f64();
                eprintln!(
                    "[gemma4-mtp] generated {} tokens in {:.3}s = {:.2} tok/s",
                    ids.len(),
                    wall,
                    ids.len() as f64 / wall.max(1e-9)
                );
                eprintln!(
                    "[gemma4-mtp] rounds {} | drafted {} | accepted {} | alpha {:.2} | acceptance {:.1}%",
                    stats.rounds,
                    stats.drafted,
                    stats.accepted,
                    stats.alpha(),
                    stats.acceptance_rate() * 100.0
                );
                eprintln!(
                    "[gemma4-mtp] prefill {:.0} ms | assistant {:.0} ms | verify {:.0} ms ({:.0} ms/round)",
                    stats.prefill_ns as f64 / 1e6,
                    stats.assistant_ns as f64 / 1e6,
                    stats.verify_ns as f64 / 1e6,
                    if stats.rounds > 0 {
                        stats.verify_ns as f64 / 1e6 / stats.rounds as f64
                    } else {
                        0.0
                    },
                );
                eprintln!("[gemma4-mtp] token_ids: {ids:?}");
                println!("{out}");
                let run = Gemma4HarnessRun {
                    request_path: request_json.as_deref(),
                    model: &path,
                    cghost: cghost.as_deref(),
                    prompt: &prompt,
                    prompt_tokens,
                    max_tokens,
                    draft_k: Some(mtp_draft_k),
                    ids: &ids,
                    text: &out,
                    expected: expected.as_deref(),
                    load_secs,
                    generate_secs: wall,
                    decode_only_secs: (wall - stats.prefill_ns as f64 / 1e9).max(0.0),
                    stats: Some(&stats),
                    sser: runtime.sser_stats(),
                    sser_prefill: runtime.sser_prefill_mark(),
                    tier: runtime.host_tier_counters(),
                    tier_prefill: runtime.host_tier_prefill_mark(),
                    tier_eviction_policy: runtime.host_tier_eviction_policy(),
                    tier_residency_prefill: runtime.sser_host_prefill_residency(),
                    tier_residency_end: runtime.sser_host_residency(),
                    // Speculative rounds are not per-token forwards, so there is no
                    // second-half token rate to quote here.
                    steady_tokens_per_second: None,
                };
                if !run.finish(receipt.as_deref())? {
                    std::process::exit(1);
                }
                return Ok(());
            }
            let (out, ids, per_token) = runtime.generate_greedy_timed(&prompt, max_tokens)?;
            let gen = t1.elapsed().as_secs_f64();
            let counts = Gemma4CudaBenchmarkCounts::new(max_tokens, ids.len(), per_token.len());
            let emitted_rate = if gen > 0.0 {
                counts.emitted_non_stop_tokens as f64 / gen
            } else {
                0.0
            };
            eprintln!(
                "[gemma4-cuda] generated in {gen:.3}s ({emitted_rate:.2} emitted tok/s end-to-end, incl. prefill)"
            );
            eprintln!(
                "[gemma4-cuda] accounting: requested_new_tokens={}, emitted_non_stop_tokens={}, timed_decode_forwards={}",
                counts.requested_new_tokens,
                counts.emitted_non_stop_tokens,
                counts.timed_decode_forwards,
            );
            if !counts.emitted_budget_is_respected() {
                eprintln!(
                    "[gemma4-cuda] WARNING: runtime emitted {} non-stop tokens for a {}-token request; decode rates below count timed forwards, not returned IDs",
                    counts.emitted_non_stop_tokens, counts.requested_new_tokens,
                );
            }
            // Warm-up curve: forward/s over successive 8-forward windows of decode-only
            // wall time (excludes prefill). This uses exactly the operations represented by
            // `per_token`, rather than assuming every returned ID owns a decode duration.
            if !per_token.is_empty() {
                let win = 8usize;
                eprint!(
                    "[gemma4-cuda] decode warm-up curve (forwards/s per {win}-forward window):"
                );
                let mut i = 0;
                while i < per_token.len() {
                    let end = (i + win).min(per_token.len());
                    let secs: f64 = per_token[i..end].iter().sum();
                    let n = (end - i) as f64;
                    eprint!(
                        " [{}-{}]={:.2}",
                        i,
                        end - 1,
                        if secs > 0.0 { n / secs } else { 0.0 }
                    );
                    i = end;
                }
                eprintln!();
                // Steady-state: decode-forward throughput over the second half of the
                // measured forwards (past warm-up) — the honest cache-warm rate.
                let half = per_token.len() / 2;
                let steady: f64 = per_token[half..].iter().sum();
                let sn = (per_token.len() - half) as f64;
                let decode_all: f64 = per_token.iter().sum();
                eprintln!(
                    "[gemma4-cuda] decode-only: {:.2} forwards/s all, {:.2} forwards/s steady (2nd half, {} forwards; {:.3}s timed decode wall)",
                    per_token.len() as f64 / decode_all.max(1e-9),
                    sn / steady.max(1e-9),
                    per_token.len() - half,
                    decode_all,
                );
            }
            // Read once: `sser_stats` also emits the SSER profile when it is enabled, and
            // a second call would print the whole breakdown twice.
            let sser = runtime.sser_stats();
            if let Some((hits, misses, resident, cap)) = sser {
                let total = hits + misses;
                eprintln!(
                    "[gemma4-cuda] SSER cache (lifetime since load): {hits} hits / {misses} misses = {:.1}% hit-rate; {resident}/{cap} experts resident",
                    if total > 0 { 100.0 * hits as f64 / total as f64 } else { 0.0 }
                );
                if let Some(streamed) = runtime.sser_streamed_misses() {
                    eprintln!(
                        "[gemma4-cuda] SSER frequency admission: {streamed} misses streamed without cache admission"
                    );
                }
            }
            eprintln!("[gemma4-cuda] token_ids: {ids:?}");
            println!("{prompt}{out}");
            let run = Gemma4HarnessRun {
                request_path: request_json.as_deref(),
                model: &path,
                cghost: cghost.as_deref(),
                prompt: &prompt,
                prompt_tokens,
                max_tokens,
                draft_k: None,
                ids: &ids,
                text: &out,
                expected: expected.as_deref(),
                load_secs,
                generate_secs: gen,
                // `per_token` is the decode-forward wall, so this excludes prefill by
                // construction rather than by subtracting an estimate.
                decode_only_secs: per_token.iter().sum(),
                stats: None,
                sser,
                sser_prefill: runtime.sser_prefill_mark(),
                tier: runtime.host_tier_counters(),
                tier_prefill: runtime.host_tier_prefill_mark(),
                tier_eviction_policy: runtime.host_tier_eviction_policy(),
                tier_residency_prefill: runtime.sser_host_prefill_residency(),
                tier_residency_end: runtime.sser_host_residency(),
                steady_tokens_per_second: {
                    let half = per_token.len() / 2;
                    let tail: f64 = per_token[half..].iter().sum();
                    ((per_token.len() - half) as f64 / tail.max(1e-9)).into()
                },
            };
            if !run.finish(receipt.as_deref())? {
                std::process::exit(1);
            }
        }
        Command::Gemma4Q4Repack { path, output } => {
            eprintln!(
                "[gemma4-q4-repack] validating and repacking {} -> {}",
                path.display(),
                output.display()
            );
            let manifest =
                camelid::gemma4_q4_sidecar::repack_exact_gemma4_q4_sidecar(&path, &output)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Gemma4GenerateGpu {
            path,
            prompt,
            max_tokens,
        } => {
            #[cfg(target_os = "macos")]
            {
                let max_positions = 512.max(max_tokens + 64);
                eprintln!("[gemma4-gpu] loading {} (resident)...", path.display());
                let t0 = std::time::Instant::now();
                let runtime =
                    camelid::gemma4_runtime::Gemma4GpuRuntime::load(&path, max_positions)?;
                eprintln!(
                    "[gemma4-gpu] loaded in {:.1}s; generating {max_tokens} tokens...",
                    t0.elapsed().as_secs_f32()
                );
                let t1 = std::time::Instant::now();
                let (out, ids) = runtime.generate_greedy(&prompt, max_tokens)?;
                let gen = t1.elapsed().as_secs_f32();
                eprintln!(
                    "[gemma4-gpu] generated in {gen:.1}s ({:.2} tok/s)",
                    ids.len() as f32 / gen
                );
                eprintln!("[gemma4-gpu] token_ids: {ids:?}");
                println!("{prompt}{out}");
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (&path, &prompt, max_tokens);
                return Err(camelid::BackendError::UnsupportedModelArchitecture(
                    "gemma4 GPU runtime requires macOS/Metal".into(),
                )
                .into());
            }
        }
        Command::Gemma4Mtp12Gpu {
            path,
            assistant,
            prompt,
            max_tokens,
            mut widths,
        } => {
            #[cfg(target_os = "macos")]
            {
                widths.sort_unstable();
                widths.dedup();
                if widths.is_empty()
                    || widths
                        .iter()
                        .any(|width| !matches!(width, 2 | 4 | 8 | 16))
                {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "Gemma 4 MTP12 verifier widths must be a non-empty subset of 2,4,8,16; got {widths:?}"
                    ))
                    .into());
                }
                let max_width = *widths.last().expect("non-empty MTP12 widths");
                if max_tokens == 0 {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(
                        "Gemma 4 MTP12 qualification requires max_tokens > 0".into(),
                    )
                    .into());
                }
                for forbidden_env in [
                    "CAMELID_GEMMA4_DENSE_ORDERED_Q4",
                    "CAMELID_GEMMA4_VERIFY_TRACE",
                    "CAMELID_GEMMA4_METAL_HEAD_TIMING",
                    "CAMELID_GEMMA4_GPU_TIMING",
                ] {
                    if std::env::var(forbidden_env)
                        .ok()
                        .as_deref()
                        .is_some_and(|value| {
                            value == "1" || value.eq_ignore_ascii_case("true")
                        })
                    {
                        return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                            "Gemma 4 MTP12 benchmark requires {forbidden_env} to be unset"
                        ))
                        .into());
                    }
                }

                let max_positions = 512usize;
                let conservative_positions = prompt
                    .len()
                    .saturating_add(max_tokens)
                    .saturating_add(max_width);
                if conservative_positions > max_positions {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "Gemma 4 MTP12 benchmark is capped at {max_positions} KV rows; UTF-8-byte upper bound is {conservative_positions}"
                    ))
                    .into());
                }
                let model_bytes = std::fs::metadata(&path)?.len();
                let assistant_bytes = std::fs::metadata(&assistant)?.len();

                eprintln!(
                    "[gemma4-mtp12] loading target {} with KV capacity {max_positions}...",
                    path.display()
                );
                let load_started = std::time::Instant::now();
                let runtime =
                    camelid::gemma4_runtime::Gemma4GpuRuntime::load(&path, max_positions)?;
                let target_load_us = load_started.elapsed().as_micros();
                eprintln!("[gemma4-mtp12] admitting exact target SHA-256...");
                let target_identity_us = runtime.admit_mtp12_target_identity()?;
                let prompt_tokens = runtime.tokenizer().encode(&prompt, true, true)?;
                let required_positions = prompt_tokens
                    .len()
                    .saturating_add(max_tokens)
                    .saturating_add(max_width);
                if required_positions > max_positions {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "Gemma 4 MTP12 needs {required_positions} KV positions, capacity is {max_positions}"
                    ))
                    .into());
                }

                eprintln!(
                    "[gemma4-mtp12] qualifying established vs ordered K1 target output..."
                );
                let qualification = runtime.qualify_ordered_q4_k1(&prompt, max_tokens)?;
                if qualification.token_ids.len() < max_width {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "Gemma 4 MTP12 benchmark needs at least {max_width} qualified outputs; got {}",
                        qualification.token_ids.len(),
                    ))
                    .into());
                }
                let qualification_decode_forwards_per_s = if qualification.decode_us == 0 {
                    0.0
                } else {
                    qualification.decode_forward_count as f64 * 1_000_000.0
                        / qualification.decode_us as f64
                };

                eprintln!(
                    "[gemma4-mtp12] loading exact assistant {}...",
                    assistant.display()
                );
                let assistant_load_started = std::time::Instant::now();
                let mut drafter =
                    camelid::metal::Gemma4Mtp12AssistantMetal::load(&assistant)?;
                let assistant_load_us = assistant_load_started.elapsed().as_micros();
                let mut runs = Vec::with_capacity(widths.len());
                let mut exact_all = true;

                for width in widths.iter().copied() {
                    eprintln!(
                        "[gemma4-mtp12] lossless decode W={width} / drafts={}...",
                        width - 1
                    );
                    let wall_started = std::time::Instant::now();
                    let generation = runtime.generate_greedy_mtp12_ordered_q4(
                        &mut drafter,
                        &prompt,
                        max_tokens,
                        width,
                    )?;
                    let generation_wall_us = wall_started.elapsed().as_micros();
                    let first_id_divergence = qualification
                        .token_ids
                        .iter()
                        .zip(&generation.token_ids)
                        .position(|(left, right)| left != right)
                        .or_else(|| {
                            (qualification.token_ids.len() != generation.token_ids.len()).then_some(
                                qualification
                                    .token_ids
                                    .len()
                                    .min(generation.token_ids.len()),
                            )
                        });
                    let ids_exact = first_id_divergence.is_none()
                        && generation.token_ids.len() == qualification.token_ids.len();
                    let text_exact = generation.text == qualification.text;
                    let exact = ids_exact && text_exact;
                    exact_all &= exact;
                    let decode_outputs = generation.token_ids.len().saturating_sub(1);
                    let decode_output_tok_s = if generation.stats.decode_us == 0 {
                        0.0
                    } else {
                        decode_outputs as f64 * 1_000_000.0
                            / generation.stats.decode_us as f64
                    };
                    let end_to_end_tok_s = if generation_wall_us == 0 {
                        0.0
                    } else {
                        generation.token_ids.len() as f64 * 1_000_000.0
                            / generation_wall_us as f64
                    };
                    let qualified_generation_us = generation_wall_us
                        .saturating_sub(generation.stats.target_identity_us);
                    let qualified_end_to_end_tok_s = if qualified_generation_us == 0 {
                        0.0
                    } else {
                        generation.token_ids.len() as f64 * 1_000_000.0
                            / qualified_generation_us as f64
                    };
                    eprintln!(
                        "[gemma4-mtp12] W={width}: {decode_output_tok_s:.3} decode tok/s, alpha={:.3}, accepted={}/{}, exact={exact}",
                        generation.stats.alpha(),
                        generation.stats.accepted_drafts,
                        generation.stats.drafted,
                    );
                    runs.push(serde_json::json!({
                        "configured_verify_width": width,
                        "cold_generation_wall_us": generation_wall_us,
                        "qualified_generation_wall_us": qualified_generation_us,
                        "decode_output_count": decode_outputs,
                        "decode_output_tok_s": decode_output_tok_s,
                        "cold_end_to_end_output_tok_s": end_to_end_tok_s,
                        "qualified_end_to_end_output_tok_s": qualified_end_to_end_tok_s,
                        "first_id_divergence_vs_ordered_k1": first_id_divergence.map(|index| index as i64).unwrap_or(-1),
                        "ids_exact": ids_exact,
                        "text_exact": text_exact,
                        "exact": exact,
                        "target_model_sha256": generation.target_model_sha256,
                        "assistant_model_sha256": generation.assistant_model_sha256,
                        "assistant_source_path": generation.assistant_source_path,
                        "assistant_resident_ledger": generation.assistant_resident_ledger,
                        "stats": generation.stats,
                        "token_ids": generation.token_ids,
                        "text": generation.text
                    }));
                }

                let receipt = serde_json::json!({
                    "schema": "camelid.gemma4_mtp12_metal_lossless_sweep.v1",
                    "camelid_version": VERSION,
                    "source_commit": option_env!("CAMELID_GIT_COMMIT"),
                    "model_path": path,
                    "model_bytes": model_bytes,
                    "assistant_path": assistant,
                    "assistant_bytes": assistant_bytes,
                    "prompt": prompt,
                    "prompt_tokens": prompt_tokens.len(),
                    "max_tokens": max_tokens,
                    "max_positions": max_positions,
                    "target_load_us": target_load_us,
                    "target_identity_us": target_identity_us,
                    "assistant_load_us": assistant_load_us,
                    "environment": {
                        "CAMELID_GEMMA4_MTP_W16_WARMUP8": std::env::var("CAMELID_GEMMA4_MTP_W16_WARMUP8").ok(),
                        "CAMELID_GEMMA4_MTP_W16_ONESHOT_W8_PAD16": std::env::var("CAMELID_GEMMA4_MTP_W16_ONESHOT_W8_PAD16").ok(),
                        "CAMELID_GEMMA4_Q4_NATIVE_SIDECAR": std::env::var_os("CAMELID_GEMMA4_Q4_NATIVE_SIDECAR").map(|value| value.to_string_lossy().into_owned()),
                        "CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT": std::env::var("CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT").ok(),
                        "CAMELID_GEMMA4_Q4_MMA": std::env::var("CAMELID_GEMMA4_Q4_MMA").ok(),
                        "CAMELID_GEMMA4_Q4_ROW_OPS": std::env::var("CAMELID_GEMMA4_Q4_ROW_OPS").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_ROWMAJOR": std::env::var("CAMELID_GEMMA4_Q4_MMA_ROWMAJOR").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_FRAGMENT": std::env::var("CAMELID_GEMMA4_Q4_MMA_FRAGMENT").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT": std::env::var("CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT_K16": std::env::var("CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT_K16").ok(),
                        "CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT_K16": std::env::var("CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT_K16").ok(),
                        "CAMELID_GEMMA4_Q4_DIRECT_TG": std::env::var("CAMELID_GEMMA4_Q4_DIRECT_TG").ok(),
                        "CAMELID_GEMMA4_DENSE_METAL_Q6K_HEAD": std::env::var("CAMELID_GEMMA4_DENSE_METAL_Q6K_HEAD").ok(),
                        "CAMELID_GEMMA4_DENSE_ORDERED_Q4": std::env::var("CAMELID_GEMMA4_DENSE_ORDERED_Q4").ok(),
                        "CAMELID_GEMMA4_DENSE_ATTN_ROWS": std::env::var("CAMELID_GEMMA4_DENSE_ATTN_ROWS").ok(),
                        "CAMELID_GEMMA4_VERIFY_TRACE": std::env::var("CAMELID_GEMMA4_VERIFY_TRACE").ok(),
                        "CAMELID_GEMMA4_METAL_HEAD_TIMING": std::env::var("CAMELID_GEMMA4_METAL_HEAD_TIMING").ok(),
                        "CAMELID_GEMMA4_GPU_TIMING": std::env::var("CAMELID_GEMMA4_GPU_TIMING").ok()
                    },
                    "dense_attention_rows_contract": {
                        "selector": "CAMELID_GEMMA4_DENSE_ATTN_ROWS",
                        "explicit_request_admission": "all_48_dense_layers_per_target_batch_or_fail_closed",
                        "default_when_unset": "per_row_split3"
                    },
                    "ordered_k1_qualification": {
                        "exact_vs_established": true,
                        "prefill_us": qualification.prefill_us,
                        "decode_us": qualification.decode_us,
                        "decode_forward_count": qualification.decode_forward_count,
                        "decode_target_forwards_per_s": qualification_decode_forwards_per_s,
                        "token_ids": qualification.token_ids,
                        "text": qualification.text
                    },
                    "runs": runs,
                    "exact_all_widths": exact_all,
                    "timing_scope": "Qualified generation wall includes ordered prompt replay plus lossless decode; decode_output_tok_s excludes prompt and first output produced by prefill; one-time target identity hash and assistant load are separately accounted"
                });
                println!("{}", serde_json::to_string_pretty(&receipt)?);
                if !exact_all {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(
                        "Gemma 4 MTP12 output diverged from qualified ordered K1".into(),
                    )
                    .into());
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (&path, &assistant, &prompt, max_tokens, &widths);
                return Err(camelid::BackendError::UnsupportedModelArchitecture(
                    "Gemma 4 MTP12 Metal runtime requires macOS".into(),
                )
                .into());
            }
        }
        Command::Gemma4VerifyGpu {
            path,
            prompt,
            max_tokens,
            mut widths,
        } => {
            #[cfg(target_os = "macos")]
            {
                let established_ordered_env = std::env::var("CAMELID_GEMMA4_DENSE_ORDERED_Q4")
                    .ok();
                if established_ordered_env
                    .as_deref()
                    .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(
                        "gemma4 verifier qualification requires the production established baseline; unset CAMELID_GEMMA4_DENSE_ORDERED_Q4"
                            .into(),
                    )
                    .into());
                }
                for timing_env in [
                    "CAMELID_GEMMA4_VERIFY_TRACE",
                    "CAMELID_GEMMA4_METAL_HEAD_TIMING",
                    "CAMELID_GEMMA4_GPU_TIMING",
                ] {
                    if std::env::var(timing_env).ok().as_deref().is_some_and(|value| {
                        value == "1" || value.eq_ignore_ascii_case("true")
                    }) {
                        return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                            "gemma4 verifier timing receipt requires {timing_env} to be unset"
                        ))
                        .into());
                    }
                }
                if max_tokens == 0 {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(
                        "gemma4 verifier qualification requires max_tokens > 0".into(),
                    )
                    .into());
                }
                widths.sort_unstable();
                widths.dedup();
                if widths.is_empty()
                    || widths
                        .iter()
                        .any(|width| !matches!(width, 1 | 2 | 4 | 8 | 16))
                {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "Gemma 4 ordered verifier widths must be a non-empty subset of 1,2,4,8,16; got {widths:?}"
                    ))
                    .into());
                }
                if !widths.contains(&1) {
                    widths.insert(0, 1);
                }
                let max_width = *widths.last().expect("non-empty verifier widths");
                // UTF-8 bytes are a conservative upper bound for the tokenizer's
                // byte-fallback pieces. Keep this Mini2 harness deliberately
                // short-context: never turn a long raw prompt into an accidental
                // multi-GiB KV allocation on a 16 GB machine.
                let max_positions = 512usize;
                let conservative_positions = prompt
                    .len()
                    .saturating_add(max_tokens)
                    .saturating_add(max_width);
                if conservative_positions > max_positions {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "gemma4 verifier is capped at {max_positions} KV rows; UTF-8-byte upper bound is {conservative_positions}"
                    ))
                    .into());
                }
                let model_bytes = std::fs::metadata(&path)?.len();
                let model_sha256 = camelid::receipt::sha256_file_hex_cached(&path)?;
                eprintln!(
                    "[gemma4-verify] loading {} with KV capacity {max_positions}...",
                    path.display()
                );
                let load_started = std::time::Instant::now();
                let runtime =
                    camelid::gemma4_runtime::Gemma4GpuRuntime::load(&path, max_positions)?;
                let prompt_tokens = runtime.tokenizer().encode(&prompt, true, true)?;
                let required_positions = prompt_tokens
                    .len()
                    .saturating_add(max_tokens)
                    .saturating_add(max_width);
                if required_positions > max_positions {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "Gemma 4 verifier needs {required_positions} KV positions for prompt={} + max_new={max_tokens} + K={max_width}, but capacity is {max_positions}",
                        prompt_tokens.len(),
                    ))
                    .into());
                }
                let load_us = load_started.elapsed().as_micros();

                eprintln!(
                    "[gemma4-verify] whole-target established-vs-ordered K1 qualification..."
                );
                let qualification = runtime.qualify_ordered_q4_k1(&prompt, max_tokens)?;
                let qualification_decode_forwards_per_s = if qualification.decode_us == 0 {
                    0.0
                } else {
                    qualification.decode_forward_count as f64 * 1_000_000.0
                        / qualification.decode_us as f64
                };
                eprintln!(
                    "[gemma4-verify] K1 IDs exact: {} outputs; ordered prefill {:.3}s, decode {:.3}s ({qualification_decode_forwards_per_s:.3} target forwards/s)",
                    qualification.token_ids.len(),
                    qualification.prefill_us as f64 / 1_000_000.0,
                    qualification.decode_us as f64 / 1_000_000.0,
                );

                let timed_rows = qualification.token_ids.len() / max_width * max_width;
                if timed_rows == 0 {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "Gemma 4 verifier generated sequence has {} rows, fewer than K={max_width}",
                        qualification.token_ids.len(),
                    ))
                    .into());
                }
                let teacher_tokens = &qualification.token_ids[..timed_rows];

                let run_width = |width: usize,
                                 tokens: &[u32]|
                 -> anyhow::Result<(Vec<u32>, Vec<u32>, u128)> {
                    if !tokens.len().is_multiple_of(width) {
                        return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                            "teacher-forced row count {} is not divisible by K={width}",
                            tokens.len(),
                        ))
                        .into());
                    }
                    // Replay the prompt into this width's fresh ordered cache
                    // outside the clock. Only post-prompt candidate rows are
                    // part of the target-verifier throughput receipt.
                    let prefill = runtime.prefill_ordered_q4(&prompt)?;
                    if prefill.first_greedy_id != tokens[0] {
                        return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                            "ordered prefill predicted {}, but the qualified teacher trajectory starts with {}",
                            prefill.first_greedy_id, tokens[0],
                        ))
                        .into());
                    }
                    let started = std::time::Instant::now();
                    let mut predictions = Vec::with_capacity(tokens.len());
                    let mut hidden_bits =
                        Vec::with_capacity(tokens.len().saturating_mul(3_840));
                    let mut position = prefill.prompt_token_count;
                    for chunk in tokens.chunks_exact(width) {
                        let batch = runtime.verify_consecutive_greedy(chunk, position)?;
                        predictions.extend_from_slice(&batch.greedy_ids);
                        hidden_bits.extend(
                            batch
                                .final_hidden
                                .iter()
                                .flatten()
                                .map(|value| value.to_bits()),
                        );
                        position = runtime.commit_verifier_prefix(batch.ticket, width)?;
                    }
                    Ok((predictions, hidden_bits, started.elapsed().as_micros()))
                };

                // Exercise the transactional invariant used by real speculative
                // decode, not merely the full-commit throughput path. Use the
                // requested maximum physical width so a W16 receipt really writes
                // sixteen cache rows. Zero/full pin the boundary cases; 1, 7 and
                // 15 span a large tail, the K8-fragment boundary, and a one-row tail.
                let rejected_tail_width = max_width;
                let rejected_tail_prefixes = gemma4_rejected_tail_commit_prefixes(
                    rejected_tail_width,
                )
                .expect("validated verifier width has an overwrite-prefix plan");
                if qualification.token_ids.len() < rejected_tail_width {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(
                        format!(
                            "Gemma 4 rejected-tail W{rejected_tail_width} gate requires at least {rejected_tail_width} qualified output tokens"
                        ),
                    )
                    .into());
                }
                let tail_a = &qualification.token_ids[..rejected_tail_width];
                let stop_ids = runtime.stop_token_ids();
                let mut candidate_pool: Vec<u32> = prompt_tokens
                    .iter()
                    .chain(&qualification.token_ids)
                    .copied()
                    .filter(|token| !stop_ids.contains(token))
                    .collect();
                candidate_pool.sort_unstable();
                candidate_pool.dedup();
                if candidate_pool.len() < 2 {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(
                        "Gemma 4 rejected-tail gate needs two distinct non-stop candidate tokens"
                            .into(),
                    )
                    .into());
                }
                let mut rejected_tail_runs = Vec::with_capacity(rejected_tail_prefixes.len());
                for committed in rejected_tail_prefixes.iter().copied() {
                    let overlap = rejected_tail_width - committed;
                    let mut tail_b = Vec::with_capacity(rejected_tail_width);
                    for row in 0..rejected_tail_width {
                        let stale = (row < overlap).then(|| tail_a[committed + row]);
                        let replacement = candidate_pool
                            .iter()
                            .copied()
                            .find(|token| Some(*token) != stale)
                            .expect("two-token pool always has a non-stale replacement");
                        tail_b.push(replacement);
                    }

                    let experiment_prefill = runtime.prefill_ordered_q4(&prompt)?;
                    if experiment_prefill.first_greedy_id != tail_a[0] {
                        return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                            "rejected-tail experiment prefill predicted {}, expected {}",
                            experiment_prefill.first_greedy_id, tail_a[0],
                        ))
                        .into());
                    }
                    let start = experiment_prefill.prompt_token_count;
                    let a_batch = runtime.verify_consecutive_greedy(tail_a, start)?;
                    let a_ids = a_batch.greedy_ids.clone();
                    let a_hidden_bits: Vec<u32> = a_batch
                        .final_hidden
                        .iter()
                        .take(committed)
                        .flatten()
                        .map(|value| value.to_bits())
                        .collect();
                    if committed == 0 {
                        runtime.rollback_verifier_batch(a_batch.ticket)?;
                    } else {
                        runtime.commit_verifier_prefix(a_batch.ticket, committed)?;
                    }
                    let b_batch = runtime.verify_consecutive_greedy(&tail_b, start + committed)?;
                    let b_ids = b_batch.greedy_ids.clone();
                    let b_hidden_bits: Vec<u32> = b_batch
                        .final_hidden
                        .iter()
                        .flatten()
                        .map(|value| value.to_bits())
                        .collect();
                    runtime.commit_verifier_prefix(b_batch.ticket, rejected_tail_width)?;

                    let reference_prefill = runtime.prefill_ordered_q4(&prompt)?;
                    if reference_prefill.first_greedy_id != tail_a[0] {
                        return Err(camelid::BackendError::UnsupportedModelArchitecture(
                            "rejected-tail reference prefill left the qualified trajectory".into(),
                        )
                        .into());
                    }
                    let mut reference_ids =
                        Vec::with_capacity(committed + rejected_tail_width);
                    let mut reference_hidden_bits = Vec::with_capacity(
                        (committed + rejected_tail_width).saturating_mul(3_840),
                    );
                    let mut reference_position = reference_prefill.prompt_token_count;
                    for &token in tail_a[..committed].iter().chain(&tail_b) {
                        let (prediction, hidden) =
                            runtime.forward_greedy_ordered_q4(token, reference_position)?;
                        reference_ids.push(prediction);
                        reference_hidden_bits.extend(hidden.iter().map(|value| value.to_bits()));
                        reference_position += 1;
                    }

                    let mut experiment_ids = a_ids[..committed].to_vec();
                    experiment_ids.extend_from_slice(&b_ids);
                    let mut experiment_hidden_bits = a_hidden_bits;
                    experiment_hidden_bits.extend_from_slice(&b_hidden_bits);
                    let first_id_divergence = experiment_ids
                        .iter()
                        .zip(&reference_ids)
                        .position(|(left, right)| left != right);
                    let first_hidden_divergence = experiment_hidden_bits
                        .iter()
                        .zip(&reference_hidden_bits)
                        .position(|(left, right)| left != right);
                    let ids_exact = first_id_divergence.is_none()
                        && experiment_ids.len() == reference_ids.len();
                    let hidden_bit_exact = first_hidden_divergence.is_none()
                        && experiment_hidden_bits.len() == reference_hidden_bits.len();
                    if !ids_exact || !hidden_bit_exact {
                        return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                            "Gemma 4 rejected-tail W{rejected_tail_width}/commit-{committed} gate diverged: id={first_id_divergence:?}, hidden_scalar={first_hidden_divergence:?}"
                        ))
                        .into());
                    }
                    rejected_tail_runs.push(serde_json::json!({
                        "physical_width": rejected_tail_width,
                        "committed_prefix": committed,
                        "rejected_rows_overwritten": overlap,
                        "overwrite_tokens": tail_b,
                        "first_id_divergence_vs_fresh_k1": -1,
                        "first_hidden_scalar_divergence_vs_fresh_k1": -1,
                        "ids_exact": true,
                        "hidden_bit_exact": true,
                        "exact": true
                    }));
                }

                // Allocate scratch, compile lazy pipelines, and touch every target
                // weight before each timed width. Reset makes the warm row invisible.
                let _ = run_width(1, &teacher_tokens[..1])?;
                let (reference, reference_hidden_bits, reference_us) =
                    run_width(1, teacher_tokens)?;
                if let Some(divergence) = reference
                    .iter()
                    .take(timed_rows.saturating_sub(1))
                    .zip(teacher_tokens.iter().skip(1))
                    .position(|(prediction, teacher)| prediction != teacher)
                {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(format!(
                        "ordered K1 teacher trajectory diverged after decode row {divergence}: predicted={} teacher={}",
                        reference[divergence], teacher_tokens[divergence + 1],
                    ))
                    .into());
                }
                let reference_rows_per_s = timed_rows as f64 * 1_000_000.0 / reference_us as f64;
                let mut runs = Vec::with_capacity(widths.len());
                runs.push(serde_json::json!({
                    "width": 1,
                    "batches": timed_rows,
                    "wall_us": reference_us,
                    "target_rows_per_s": reference_rows_per_s,
                    "speedup_vs_k1": 1.0,
                    "first_id_divergence_vs_k1": -1,
                    "first_hidden_scalar_divergence_vs_k1": -1,
                    "ids_exact": true,
                    "hidden_bit_exact": true,
                    "exact": true
                }));

                let mut exact_all = true;
                for &width in widths.iter().filter(|&&width| width != 1) {
                    let _ = run_width(width, &teacher_tokens[..width])?;
                    let (predictions, hidden_bits, wall_us) =
                        run_width(width, teacher_tokens)?;
                    let first_id_divergence = reference
                        .iter()
                        .zip(&predictions)
                        .position(|(left, right)| left != right);
                    let ids_exact = first_id_divergence.is_none()
                        && predictions.len() == reference.len();
                    let first_hidden_divergence = reference_hidden_bits
                        .iter()
                        .zip(&hidden_bits)
                        .position(|(left, right)| left != right);
                    let hidden_bit_exact = first_hidden_divergence.is_none()
                        && hidden_bits.len() == reference_hidden_bits.len();
                    let exact = ids_exact && hidden_bit_exact;
                    exact_all &= exact;
                    let rows_per_s = timed_rows as f64 * 1_000_000.0 / wall_us as f64;
                    let speedup = reference_us as f64 / wall_us as f64;
                    eprintln!(
                        "[gemma4-verify] K={width}: {rows_per_s:.3} target rows/s, {speedup:.3}x K1, ids_exact={ids_exact}, hidden_bit_exact={hidden_bit_exact}"
                    );
                    runs.push(serde_json::json!({
                        "width": width,
                        "batches": timed_rows / width,
                        "wall_us": wall_us,
                        "target_rows_per_s": rows_per_s,
                        "speedup_vs_k1": speedup,
                        "first_id_divergence_vs_k1": first_id_divergence.map(|index| index as i64).unwrap_or(-1),
                        "first_hidden_scalar_divergence_vs_k1": first_hidden_divergence.map(|index| index as i64).unwrap_or(-1),
                        "ids_exact": ids_exact,
                        "hidden_bit_exact": hidden_bit_exact,
                        "exact": exact
                    }));
                }

                let receipt = serde_json::json!({
                    "schema": "camelid.gemma4_ordered_q4_target_sweep.v1",
                    "model_path": path,
                    "model_bytes": model_bytes,
                    "model_sha256": model_sha256,
                    "camelid_version": VERSION,
                    "source_commit": option_env!("CAMELID_GIT_COMMIT"),
                    "prompt": prompt,
                    "prompt_tokens": prompt_tokens.len(),
                    "generated_tokens": qualification.token_ids.len(),
                    "candidate_rows_available": qualification.token_ids.len(),
                    "timed_rows": timed_rows,
                    "max_positions": max_positions,
                    "load_us": load_us,
                    "environment": {
                        "CAMELID_GEMMA4_Q4_NATIVE_SIDECAR": std::env::var_os("CAMELID_GEMMA4_Q4_NATIVE_SIDECAR").map(|value| value.to_string_lossy().into_owned()),
                        "CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT": std::env::var("CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT").ok(),
                        "CAMELID_GEMMA4_Q4_MMA": std::env::var("CAMELID_GEMMA4_Q4_MMA").ok(),
                        "CAMELID_GEMMA4_Q4_ROW_OPS": std::env::var("CAMELID_GEMMA4_Q4_ROW_OPS").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_ROWMAJOR": std::env::var("CAMELID_GEMMA4_Q4_MMA_ROWMAJOR").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_FRAGMENT": std::env::var("CAMELID_GEMMA4_Q4_MMA_FRAGMENT").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT": std::env::var("CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT").ok(),
                        "CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT_K16": std::env::var("CAMELID_GEMMA4_Q4_MMA_REGISTER_FRAGMENT_K16").ok(),
                        "CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT_K16": std::env::var("CAMELID_GEMMA4_Q4_NATIVE_REGISTER_FRAGMENT_K16").ok(),
                        "CAMELID_GEMMA4_DENSE_ORDERED_Q4": established_ordered_env,
                        "CAMELID_GEMMA4_DENSE_METAL_Q6K_HEAD": std::env::var("CAMELID_GEMMA4_DENSE_METAL_Q6K_HEAD").ok(),
                        "CAMELID_GEMMA4_DENSE_ATTN_ROWS": std::env::var("CAMELID_GEMMA4_DENSE_ATTN_ROWS").ok(),
                        "CAMELID_GEMMA4_Q4_DIRECT_TG": std::env::var("CAMELID_GEMMA4_Q4_DIRECT_TG").ok(),
                        "CAMELID_GEMMA4_VERIFY_TRACE": std::env::var("CAMELID_GEMMA4_VERIFY_TRACE").ok(),
                        "CAMELID_GEMMA4_METAL_HEAD_TIMING": std::env::var("CAMELID_GEMMA4_METAL_HEAD_TIMING").ok(),
                        "CAMELID_GEMMA4_GPU_TIMING": std::env::var("CAMELID_GEMMA4_GPU_TIMING").ok()
                    },
                    "dense_attention_rows_contract": {
                        "selector": "CAMELID_GEMMA4_DENSE_ATTN_ROWS",
                        "explicit_request_admission": "all_48_dense_layers_per_target_batch_or_fail_closed",
                        "default_when_unset": "per_row_split3"
                    },
                    "ordered_k1_qualification": {
                        "exact_vs_established": true,
                        "prefill_us": qualification.prefill_us,
                        "decode_us": qualification.decode_us,
                        "decode_forward_count": qualification.decode_forward_count,
                        "decode_target_forwards_per_s": qualification_decode_forwards_per_s,
                        "token_ids": qualification.token_ids
                    },
                    "runs": runs,
                    "rejected_tail_overwrite_gate": {
                        "physical_width": rejected_tail_width,
                        "commit_prefixes": rejected_tail_prefixes,
                        "includes_zero_rollback": true,
                        "includes_full_commit": true,
                        "exact_all_commit_prefixes": true,
                        "runs": rejected_tail_runs
                    },
                    "exact_all_widths": exact_all,
                    "timing_scope": "Warm target-only teacher-forced decode-row verifier wall; full-width batches; prompt replay/assistant excluded"
                });
                println!("{}", serde_json::to_string_pretty(&receipt)?);
                if !exact_all {
                    return Err(camelid::BackendError::UnsupportedModelArchitecture(
                        "Gemma 4 K-wide verifier diverged from ordered K1".into(),
                    )
                    .into());
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (&path, &prompt, max_tokens, &widths);
                return Err(camelid::BackendError::UnsupportedModelArchitecture(
                    "gemma4 ordered verifier requires macOS/Metal".into(),
                )
                .into());
            }
        }
        Command::DiffusionGemmaChat {
            path,
            prompt,
            max_blocks,
            seed,
            max_ubatch,
            max_steps,
        } => {
            use camelid::diffusion_gemma::chat::DgChat;
            use camelid::diffusion_gemma::DgEbParams;
            eprintln!("[dg] loading {} (CPU, lazy mmap)...", path.display());
            let t0 = std::time::Instant::now();
            let chat = DgChat::load(&path)?;
            eprintln!(
                "[dg] loaded in {:.1}s; canvas_length={}; denoising (CPU — minutes per step)...",
                t0.elapsed().as_secs_f32(),
                chat.canvas_length()
            );
            let defaults = DgEbParams::default();
            let params = DgEbParams {
                seed,
                max_steps: max_steps.map(|m| m.max(1)).unwrap_or(defaults.max_steps),
                ..defaults
            };
            eprintln!(
                "[dg] max_steps={} max_blocks={}",
                params.max_steps, max_blocks
            );
            let t1 = std::time::Instant::now();
            // CAMELID_DG_LIVE=1: print the forming answer after every denoise
            // step — the whole draft exists from step 0 and refines in place.
            let live = std::env::var("CAMELID_DG_LIVE").as_deref() == Ok("1");
            let (text, stop, ids) = chat.generate_live(
                &prompt,
                &params,
                max_blocks,
                max_ubatch,
                |b, step, draft| {
                    if live {
                        let one_line = draft.replace('\n', " ");
                        let preview: String = one_line.chars().take(160).collect();
                        eprintln!(
                            "[dg-live b{b} s{step} {:.0}s] {preview}",
                            t1.elapsed().as_secs_f32()
                        );
                    }
                },
                |b, committed| {
                    eprintln!(
                        "[dg] block {b}: committed {} tokens ({:.0}s)",
                        committed.len(),
                        t1.elapsed().as_secs_f32()
                    );
                },
            )?;
            eprintln!(
                "[dg] done in {:.1}s (stop: {stop}, {} tokens)",
                t1.elapsed().as_secs_f32(),
                ids.len()
            );
            println!("{text}");
        }
        Command::Gemma4Worker {
            path,
            addr,
            first_layer,
        } => {
            // Blocks forever serving sessions; honest claim: distributed layer
            // sharding (memory headroom), not shared memory.
            let gguf = camelid::gguf::read_metadata(&path)?;
            let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
            let block_count = config.block_count as usize;
            camelid::gemma4_distributed::run_worker(&path, &addr, first_layer..block_count)?;
        }
        Command::Gemma4Master {
            path,
            worker_addr,
            split,
            prompt,
            max_tokens,
        } => {
            eprintln!(
                "[gemma4-master] layers 0..{split} local, {split}.. on {worker_addr}; loading..."
            );
            let t0 = std::time::Instant::now();
            let (out, ids, stats) = camelid::gemma4_distributed::run_master(
                &path,
                &worker_addr,
                split,
                &prompt,
                max_tokens,
                false,
            )?;
            eprintln!(
                "[gemma4-master] done in {:.1}s; stats: {}",
                t0.elapsed().as_secs_f32(),
                serde_json::to_string(&stats)?
            );
            eprintln!("[gemma4-master] token_ids: {ids:?}");
            println!("{prompt}{out}");
        }
        Command::TensorDump {
            path,
            tensors,
            window,
            rows,
            tokens,
            layers,
        } => {
            let gguf = read_metadata(&path)?;
            let store = TensorStore::open(&path, &gguf);
            let names = tensor_dump_names(tensors, layers);
            let mut dumps = Vec::with_capacity(names.len());
            for name in names {
                dumps.push(dump_tensor(&store, &name, window, &rows, &tokens)?);
            }
            let dump = TensorDumpFile {
                path: path.display().to_string(),
                tensors: dumps,
            };
            println!("{}", serde_json::to_string_pretty(&dump)?);
        }
        Command::BenchDenseHotloops {
            hidden,
            ffn,
            repeats,
            warmup,
            threads,
        } => {
            configure_rayon_threads(threads)?;
            let report = bench_dense_hotloops(hidden, ffn, repeats, warmup)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        #[cfg(feature = "alloc-gate")]
        Command::BenchAllocGate {
            model,
            warmup,
            tokens,
            skip_logits,
            trace_big,
            max_per_token,
        } => {
            // The alloc gate runs a real dense decode; refuse runnable-lane-only
            // archs before the library loads weights (metadata read is cheap).
            ensure_arch_has_direct_dense_session(
                &read_metadata(&model)?,
                DenseLaneWindowedForward::ViaSessionDecode,
            )?;
            let report = camelid::alloc_gate::run_decode_alloc_gate(
                &model,
                warmup,
                tokens,
                !skip_logits,
                trace_big,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if let Some(max_per_token) = max_per_token {
                let per_token = report["allocations_per_token"]
                    .as_f64()
                    .expect("report always carries allocations_per_token");
                if per_token > max_per_token {
                    anyhow::bail!(
                        "decode alloc gate FAILED: {per_token} allocations/token exceeds the \
                         allowed {max_per_token}"
                    );
                }
            }
        }
        Command::BenchRayonRegion {
            iterations,
            idle_us,
            threads,
        } => {
            configure_rayon_threads(threads)?;
            let us_per_region = camelid::inference::rayon_region_microbench(iterations, idle_us);
            let record = serde_json::json!({
                "schema": "camelid.bench-rayon-region/v1",
                "threads": rayon::current_num_threads(),
                "iterations": iterations,
                "idle_us_between": idle_us,
                "us_per_region": us_per_region,
            });
            println!("{}", serde_json::to_string(&record)?);
        }
        Command::BenchAttnDot {
            lens,
            repeats,
            warmup,
        } => {
            for len in lens {
                for (variant, ns_per_call) in
                    camelid::inference::attn_f32_dot_microbench(len, repeats, warmup)
                {
                    let record = serde_json::json!({
                        "schema": "camelid.bench-attn-dot/v1",
                        "len": len,
                        "variant": variant,
                        "ns_per_call": ns_per_call,
                    });
                    println!("{}", serde_json::to_string(&record)?);
                }
            }
        }
        Command::BenchQ8Blocks {
            path,
            tensor,
            rows,
            repeats,
            warmup,
            swap_rank2_shape,
            all_rows_dot,
            single_input_row_dot,
        } => {
            let report = bench_q8_blocks(Q8BlockBenchOptions {
                path: &path,
                tensor_name: &tensor,
                rows,
                repeats,
                warmup,
                swap_rank2_shape,
                all_rows_dot,
                single_input_row_dot,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::DistributeWorker {
            path,
            addr,
            forward_addr,
            layers,
            master_addr,
            threads,
            cghost,
        } => {
            run_distribute_worker(
                path,
                addr,
                forward_addr,
                layers,
                master_addr,
                threads,
                cghost,
            )
            .await?;
        }
        Command::DistributeMaster {
            path,
            worker_addr,
            layers,
            addr,
            prompt,
            max_tokens,
            threads,
            cghost,
        } => {
            run_distribute_master(
                path,
                worker_addr,
                layers,
                addr,
                prompt,
                max_tokens,
                threads,
                cghost,
            )
            .await?;
        }
        Command::BenchGenerate {
            model,
            prompt_file,
            prompt,
            max_tokens,
            temperature,
            iterations,
            warmup,
            threads,
            json: _,
            deterministic,
        } => {
            // Match `serve`'s fast-load plan so benchmark startup/RSS measures
            // the production Metal path rather than eagerly materializing Q8.
            if !deterministic {
                apply_serve_nocopy_default();
            }
            run_bench_generate(
                model,
                prompt_file,
                prompt,
                max_tokens,
                temperature,
                iterations,
                warmup,
                threads,
            )?;
        }
        Command::BenchGenerateVision {
            model,
            mmproj,
            image,
            prompt,
            max_tokens,
            image_min_tokens,
            image_max_tokens,
        } => {
            apply_serve_nocopy_default();
            run_bench_generate_vision(
                model,
                mmproj,
                image,
                prompt,
                max_tokens,
                image_min_tokens,
                image_max_tokens,
            )?;
        }
        Command::BenchOwnerSweep {
            model,
            lane,
            prompt_file,
            prompt,
            max_tokens,
            rounds,
            warmup_rounds,
            threads,
        } => {
            run_bench_owner_sweep(
                model,
                lane,
                prompt_file,
                prompt,
                max_tokens,
                rounds,
                warmup_rounds,
                threads,
            )?;
        }
        Command::GaitCalibrate {
            model,
            prompt_file,
            prompt,
            max_tokens,
            rounds,
            warmup,
            threads,
        } => {
            run_gait_calibrate(
                model,
                prompt_file,
                prompt,
                max_tokens,
                rounds,
                warmup,
                threads,
            )?;
        }
        Command::GaitTrial {
            model,
            prompt_file,
            prompt,
            max_tokens,
            profile,
            eco_qos,
            threads,
            gpc_attn,
            gpc_ffn,
            gpc_matmul,
        } => {
            run_gait_trial(
                model,
                prompt_file,
                prompt,
                max_tokens,
                profile,
                eco_qos,
                threads,
                gpc_attn,
                gpc_ffn,
                gpc_matmul,
            )?;
        }
        Command::Workspace {
            addr,
            json,
            timeout_seconds,
            action,
        } => {
            let action = match action {
                WorkspaceAction::Ask {
                    workspace,
                    goal,
                    thread,
                    max_steps,
                    max_tokens,
                    temperature,
                } => chat::WorkspaceCliAction::Ask {
                    workspace,
                    goal,
                    thread_id: thread,
                    max_steps,
                    max_tokens,
                    temperature,
                },
                WorkspaceAction::Threads { workspace } => {
                    chat::WorkspaceCliAction::Threads { workspace }
                }
                WorkspaceAction::Show { thread, workspace } => chat::WorkspaceCliAction::Show {
                    workspace,
                    thread_id: thread,
                },
                WorkspaceAction::Compact {
                    thread,
                    workspace,
                    undo,
                } => chat::WorkspaceCliAction::Compact {
                    workspace,
                    thread_id: thread,
                    undo,
                },
                WorkspaceAction::Delete { thread, workspace } => chat::WorkspaceCliAction::Delete {
                    workspace,
                    thread_id: thread,
                },
            };
            let code = chat::run_workspace_cli(chat::WorkspaceCliOptions {
                addr,
                json,
                timeout: std::time::Duration::from_secs(timeout_seconds),
                action,
            })?;
            std::process::exit(code);
        }
        Command::Agent { action } => match action {
            AgentAction::Exec {
                goal,
                model,
                addr,
                workdir,
                max_steps,
                max_tokens,
                auto_approve,
                yolo,
                allow_net,
                allow_fs,
                allow_mcp,
                trust_mcp_server,
                shell_sandbox,
                shell_timeout,
                models_dir,
            } => {
                // The goal may come from stdin so a caller can pipe a long or
                // generated prompt in without shell quoting.
                let goal = match goal {
                    Some(g) => g,
                    None => {
                        let mut buf = String::new();
                        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
                        buf
                    }
                };
                if goal.trim().is_empty() {
                    eprintln!("agent exec needs a goal (as an argument or on stdin)");
                    std::process::exit(1);
                }
                let code = chat::run_chat(chat::ChatOptions {
                    model: Some(model),
                    addr,
                    system: None,
                    max_tokens,
                    temperature: 0.0,
                    top_p: None,
                    top_k: None,
                    seed: None,
                    no_stream: true,
                    models_dir: models_dir.unwrap_or_else(|| PathBuf::from("models")),
                    plain: true,
                    agent: true,
                    workdir,
                    max_steps,
                    auto_approve,
                    yolo,
                    allow_net,
                    allow_fs,
                    allow_mcp,
                    trust_mcp_servers: trust_mcp_server,
                    shell_timeout,
                    enable_thinking: false,
                    audit_webhook: None,
                    shell_sandbox,
                    exec_goal: Some(goal),
                })?;
                std::process::exit(code);
            }
        },
        Command::Gait { action } => match action {
            GaitAction::Reset => run_gait_reset()?,
        },
        Command::BenchSpeculative {
            model,
            drafter,
            draft_model,
            draft_tokens,
            cpu_draft,
            spec_only,
            prompt_file,
            prompt,
            workload,
            max_tokens,
            warmup,
            threads,
        } => {
            run_bench_speculative(
                model,
                drafter,
                draft_model,
                draft_tokens,
                cpu_draft,
                spec_only,
                prompt_file,
                prompt,
                workload,
                max_tokens,
                warmup,
                threads,
            )?;
        }
        Command::BenchEagle3 {
            model,
            eagle3,
            draft_tokens,
            tree_nodes,
            tree_topk,
            tree_expansions,
            suffix_first,
            prompt_file,
            prompt,
            chat,
            workload,
            max_tokens,
            threads,
        } => {
            run_bench_eagle3(
                model,
                eagle3,
                draft_tokens,
                tree_nodes,
                tree_topk,
                tree_expansions,
                suffix_first,
                prompt_file,
                prompt,
                chat,
                workload,
                max_tokens,
                threads,
            )?;
        }
        Command::GhostRun {
            model,
            cghost,
            prompt,
            max_tokens,
            threads,
            sync_stream,
            stage_split,
            read_ahead,
            spec,
            draft_len,
            expert_cache_mib,
            evict_page_cache,
        } => {
            run_ghost(
                model,
                cghost,
                prompt,
                max_tokens,
                threads,
                sync_stream,
                stage_split,
                read_ahead,
                spec,
                draft_len,
                expert_cache_mib,
                evict_page_cache,
            )?;
        }
        Command::Verify {
            model,
            output,
            threads,
        } => {
            configure_rayon_threads(threads)?;
            let report = camelid::verify::run(&model, threads)
                .await
                .map_err(anyhow::Error::msg)?;
            let output = output.unwrap_or_else(|| camelid::verify::default_report_path(&model));
            camelid::verify::write_report(&output, &report).map_err(anyhow::Error::msg)?;
            println!(
                "{} model={} report_id={} output={}",
                match report.outcome {
                    camelid::verify::VerificationOutcome::Verified => "VERIFIED",
                    camelid::verify::VerificationOutcome::NotVerified => "NOT VERIFIED",
                    camelid::verify::VerificationOutcome::NoProfile => "NO PROFILE",
                },
                report.model.gguf_filename,
                report.report_id,
                output.display()
            );
            println!("{}", report.detail);
            std::process::exit(report.outcome.exit_code());
        }
        Command::VerifyReceipt {
            receipt,
            gguf,
            llama_server,
            self_only,
            reference_only,
            llama_ctx,
            llama_port,
            llama_cache_type_k,
            llama_cache_type_v,
            llama_flash_attn,
            llama_no_repack,
            threads,
        } => {
            // Route a sealed agent-family receipt to its self-contained verifier
            // (no model, no GGUF). Any doubt about the schema falls through to the
            // parity path below, which is left unchanged.
            let is_agent = std::fs::read_to_string(&receipt)
                .ok()
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
                .and_then(|value| {
                    value
                        .get("schema")
                        .and_then(|schema| schema.as_str())
                        .map(camelid::receipt::agent::is_agent_schema)
                })
                .unwrap_or(false);
            if is_agent {
                let outcome = camelid::receipt::agent::run(&receipt);
                std::process::exit(outcome.exit_code());
            }

            configure_rayon_threads(threads)?;
            let gguf = gguf.ok_or_else(|| {
                anyhow::anyhow!(
                    "parity verification requires the exact GGUF via --gguf; agent-family \
                     receipts (syscap / orchestration / bench) need no GGUF"
                )
            })?;
            let mode = if self_only {
                camelid::receipt::verify::VerifyMode::SelfOnly
            } else if reference_only {
                camelid::receipt::verify::VerifyMode::ReferenceOnly
            } else {
                camelid::receipt::verify::VerifyMode::Full
            };
            let outcome = camelid::receipt::verify::run(camelid::receipt::verify::VerifyOptions {
                receipt_path: receipt,
                gguf,
                llama_server,
                mode,
                llama_ctx,
                llama_port,
                llama_cache_type_k,
                llama_cache_type_v,
                llama_flash_attn,
                llama_no_repack,
                threads,
            })
            .await;
            std::process::exit(outcome.exit_code());
        }
        Command::VerifyReceipts { dir } => {
            std::process::exit(camelid::receipt::audit::run(&dir));
        }
        Command::SealReceipt { input, output } => {
            let raw = std::fs::read_to_string(&input)?;
            let mut receipt: camelid::receipt::ParityReceipt = serde_json::from_str(&raw)?;
            anyhow::ensure!(
                receipt.schema == camelid::receipt::RECEIPT_SCHEMA_V1,
                "unknown receipt schema {:?} (expected {:?})",
                receipt.schema,
                camelid::receipt::RECEIPT_SCHEMA_V1
            );
            receipt.seal()?;
            let out_path = output.unwrap_or(input);
            let mut serialized = serde_json::to_string_pretty(&receipt)?;
            serialized.push('\n');
            std::fs::write(&out_path, serialized)?;
            println!(
                "sealed receipt_id={} -> {}",
                receipt.receipt_id,
                out_path.display()
            );
        }
    }

    // §4 safe-boot: an orderly exit — clear any in-progress gait marker so a
    // healthy run is never mistaken for a crash on the next launch. (No-op unless
    // a gait was applied this process.)
    if let Some(dir) = camelid::gait::gait_dir() {
        camelid::gait::sentinel::clean_shutdown(&dir);
    }
    Ok(())
}

/// How ghost mode gets each layer's weights off disk. `range` is the node's pipeline shard
/// (the whole model on a single node); streaming cycles over it chunk after chunk.
struct GhostStreamer {
    range: std::ops::Range<usize>,
    kind: GhostStreamerKind,
}

enum GhostStreamerKind {
    /// v1: the read + decode happens on the critical path, before each layer's forward.
    Sync { ghost: Arc<GhostFile>, buf: Vec<u8> },
    /// v2 double-buffered: a background worker reads + decodes layer N+1 while layer N's
    /// forward runs; the reported time is only the STALL waiting for the handoff. The
    /// rendezvous handoff bounds the weight working set to two layer windows.
    Prefetched { prefetcher: GhostPrefetcher },
    /// v3 stage-split (`--stage-split`): read and decode run on SEPARATE threads, so the read
    /// of layer N+1 overlaps the dequant of layer N (v2's single worker serializes them).
    Pipelined { prefetcher: GhostPipelinePrefetcher },
}

impl GhostStreamer {
    fn new_sync(ghost: Arc<GhostFile>, range: std::ops::Range<usize>) -> Self {
        Self {
            range,
            kind: GhostStreamerKind::Sync {
                buf: Vec::with_capacity(ghost.max_layer_span() as usize),
                ghost,
            },
        }
    }

    fn new_prefetched(ghost: Arc<GhostFile>, range: std::ops::Range<usize>) -> Self {
        Self {
            range,
            kind: GhostStreamerKind::Prefetched {
                prefetcher: GhostPrefetcher::spawn(ghost),
            },
        }
    }

    fn new_pipelined(
        ghost: Arc<GhostFile>,
        range: std::ops::Range<usize>,
        read_ahead: usize,
    ) -> Self {
        Self {
            range,
            kind: GhostStreamerKind::Pipelined {
                prefetcher: GhostPipelinePrefetcher::spawn(ghost, read_ahead),
            },
        }
    }

    /// Queue the first chunk's layer reads (prefetched / stage-split modes; no-op for sync).
    fn prime(&self) -> anyhow::Result<()> {
        match &self.kind {
            GhostStreamerKind::Prefetched { prefetcher } => {
                for layer_idx in self.range.clone() {
                    prefetcher.request(layer_idx)?;
                }
            }
            GhostStreamerKind::Pipelined { prefetcher } => {
                for layer_idx in self.range.clone() {
                    prefetcher.request(layer_idx)?;
                }
            }
            GhostStreamerKind::Sync { .. } => {}
        }
        Ok(())
    }

    /// Produce layer `layer_idx`'s decoded weights: (weights, bytes streamed, blocked µs).
    /// On the chunk's last layer the prefetched mode queues the ENTIRE next chunk first, so
    /// the worker is already rewinding to the shard's first layer for the next token while
    /// this layer's forward runs — on a mesh node that disk window overlaps the OTHER
    /// node's compute and the network hops. The trailing chunk queued after the final token
    /// is never consumed — the worker reads at most one extra layer, blocks on the
    /// rendezvous, and is released by Drop.
    /// Returns `(weights, bytes, blocked_us, read_us, decode_us)`. `blocked_us` is the stall
    /// charged to the streaming path (the whole critical-path read+decode in sync mode; only
    /// the handoff wait in prefetched mode). `read_us`/`decode_us` are the worker's actual
    /// I/O-vs-dequant split (Phase-0 attribution), independent of how much of it overlapped.
    fn fetch(
        &mut self,
        layer_idx: usize,
        last_in_chunk: bool,
    ) -> anyhow::Result<(LlamaLayerWeights, u64, u128, u128, u128)> {
        let range = self.range.clone();
        match &mut self.kind {
            GhostStreamerKind::Sync { ghost, buf } => {
                let started = Instant::now();
                let (layer, span, read_us, decode_us) = ghost.read_layer(layer_idx, buf)?;
                Ok((
                    layer,
                    span,
                    started.elapsed().as_micros(),
                    read_us,
                    decode_us,
                ))
            }
            GhostStreamerKind::Prefetched { prefetcher } => {
                if last_in_chunk {
                    for next_idx in range {
                        prefetcher.request(next_idx)?;
                    }
                }
                let started = Instant::now();
                let prefetched = prefetcher.next()?;
                anyhow::ensure!(
                    prefetched.layer_idx == layer_idx,
                    "prefetcher returned layer {} but layer {layer_idx} was expected",
                    prefetched.layer_idx
                );
                Ok((
                    prefetched.weights,
                    prefetched.bytes,
                    started.elapsed().as_micros(),
                    prefetched.read_us,
                    prefetched.decode_us,
                ))
            }
            GhostStreamerKind::Pipelined { prefetcher } => {
                if last_in_chunk {
                    for next_idx in range {
                        prefetcher.request(next_idx)?;
                    }
                }
                let started = Instant::now();
                let prefetched = prefetcher.next()?;
                anyhow::ensure!(
                    prefetched.layer_idx == layer_idx,
                    "stage-split returned layer {} but layer {layer_idx} was expected",
                    prefetched.layer_idx
                );
                Ok((
                    prefetched.weights,
                    prefetched.bytes,
                    started.elapsed().as_micros(),
                    prefetched.read_us,
                    prefetched.decode_us,
                ))
            }
        }
    }
}

/// Build the ghost-mesh streaming context for a pipeline node: open the node's `.cghost`
/// shard, spawn the double-buffered prefetcher over the node's layer range, and prime the
/// first chunk. Returns None when the node runs the resident path. While this node waits on
/// the network (the other node computing), its prefetch worker is already streaming the
/// next token's layers — the disk window overlaps the peer's compute.
fn make_ghost_node_ctx(
    session: &LlamaInferenceSession,
    cghost: Option<&std::path::Path>,
    layer_range: std::ops::Range<usize>,
) -> anyhow::Result<Option<(GhostStreamer, LlamaLayerWeights)>> {
    let Some(path) = cghost else { return Ok(None) };
    let ghost = Arc::new(GhostFile::open(path)?);
    let n_layers = session.weights.layers.len();
    anyhow::ensure!(
        ghost.index.block_count == n_layers,
        ".cghost block_count {} does not match model block_count {n_layers}",
        ghost.index.block_count
    );
    let placeholder = session.weights.layers[0].clone();
    let streamer = GhostStreamer::new_prefetched(Arc::clone(&ghost), layer_range.clone());
    streamer.prime()?;
    println!(
        "[ghost] mesh node streams layers {:?} from {:?} ({:.1} MiB window, double-buffered)",
        layer_range,
        path,
        ghost.max_layer_span() as f64 / (1024.0 * 1024.0),
    );
    Ok(Some((streamer, placeholder)))
}

/// Ghost mode: run every transformer layer of one chunk (prefill or a single decoded
/// token), streaming each layer's weights from the `.cghost` file and dropping them right
/// after the layer's forward — the weight working window is one layer (sync) or two
/// (prefetched). Returns the chunk's output hidden state plus (bytes streamed, time blocked
/// on streaming, forward time).
fn ghost_stream_layers(
    session: &mut LlamaInferenceSession,
    streamer: &mut GhostStreamer,
    placeholder: &LlamaLayerWeights,
    hidden: CpuTensor,
    pos: usize,
    seq_len: usize,
    log_layers: bool,
) -> anyhow::Result<(CpuTensor, u64, u128, u128, u128, u128)> {
    let range = streamer.range.clone();
    let mut hidden = hidden;
    let mut bytes_total = 0u64;
    let mut wait_us_total = 0u128;
    let mut forward_us_total = 0u128;
    let mut read_us_total = 0u128;
    let mut decode_us_total = 0u128;
    for layer_idx in range.clone() {
        let (layer, span, wait_us, read_us, decode_us) =
            streamer.fetch(layer_idx, layer_idx + 1 == range.end)?;
        Arc::make_mut(&mut session.weights).layers[layer_idx] = layer;
        let forward_started = Instant::now();
        hidden = session.ghost_forward_one_layer(&hidden, layer_idx, pos, seq_len)?;
        let forward_us = forward_started.elapsed().as_micros();
        // Drop the streamed weights immediately; the window never accumulates.
        Arc::make_mut(&mut session.weights).layers[layer_idx] = placeholder.clone();
        bytes_total += span;
        wait_us_total += wait_us;
        forward_us_total += forward_us;
        read_us_total += read_us;
        decode_us_total += decode_us;
        if log_layers {
            // read/decode is the worker's true I/O-vs-dequant split; wait is the main
            // thread's stall (only the unhidden remainder after prefetch overlap).
            eprintln!(
                "[ghost] layer {layer_idx:>3}: wait {:7.1} ms | read {:7.1} decode {:7.1} \
                 ({:6.1} MiB) | forward {:7.1} ms",
                wait_us as f64 / 1000.0,
                read_us as f64 / 1000.0,
                decode_us as f64 / 1000.0,
                span as f64 / (1024.0 * 1024.0),
                forward_us as f64 / 1000.0,
            );
        }
    }
    session.ghost_advance_position(seq_len);
    Ok((
        hidden,
        bytes_total,
        wait_us_total,
        forward_us_total,
        read_us_total,
        decode_us_total,
    ))
}

/// EXPERIMENTAL ghost (layer-streaming) mode: greedy generation with the model executed one
/// transformer block at a time from a `.cghost` file. RAM holds the embedding/output ends +
/// KV cache + the streaming window (one layer sync, two prefetched); everything else stays
/// on disk.
#[allow(clippy::too_many_arguments)]
fn run_ghost(
    model: PathBuf,
    cghost: PathBuf,
    prompt: String,
    max_tokens: usize,
    threads: Option<usize>,
    sync_stream: bool,
    stage_split: bool,
    read_ahead: usize,
    spec: bool,
    draft_len: usize,
    expert_cache_mib: usize,
    evict_page_cache: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !(sync_stream && stage_split),
        "--sync-stream and --stage-split are mutually exclusive"
    );
    configure_rayon_threads(threads)?;
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("[ghost] loading GGUF metadata from {:?}...", model);
    let gguf = read_metadata(&model)?;
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
    if config.gemma4.is_some() && config.moe.is_some() {
        anyhow::ensure!(
            !stage_split && !spec,
            "Ghost-MoE does not yet support --stage-split or --spec; routed experts are fetched after each layer's router decision"
        );
        return run_ghost_moe(
            &model,
            &cghost,
            &prompt,
            max_tokens,
            expert_cache_mib,
            // Buffered reads are the throughput default. Callers that need the
            // strictest physical-memory accounting opt into F_NOCACHE explicitly
            // with --evict-page-cache, matching the serve lane's policy.
            evict_page_cache,
        );
    }
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::CpuDenseOnly)?;
    let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let ghost = Arc::new(GhostFile::open_with_options(&cghost, evict_page_cache)?);
    let n_layers = config.block_count as usize;
    anyhow::ensure!(
        ghost.index.block_count == n_layers,
        ".cghost block_count {} does not match model block_count {n_layers}",
        ghost.index.block_count
    );

    // Resident ends only (embedding + output projection); every transformer layer is a
    // placeholder that ghost_stream_layers swaps real weights into, one at a time.
    let load_started = Instant::now();
    let weights = LlamaLoadedWeights::load_distributed(&store, &binding, 0, 0, true, true)?;
    let mut session = LlamaInferenceSession::new(config.clone(), Arc::new(weights))?;
    let placeholder = session.weights.layers[0].clone();
    let mut streamer = if sync_stream {
        GhostStreamer::new_sync(Arc::clone(&ghost), 0..n_layers)
    } else if stage_split {
        GhostStreamer::new_pipelined(Arc::clone(&ghost), 0..n_layers, read_ahead)
    } else {
        GhostStreamer::new_prefetched(Arc::clone(&ghost), 0..n_layers)
    };
    let mode_label = if sync_stream {
        "sync".to_string()
    } else if stage_split {
        format!("stage-split (read\u{2016}decode, read-ahead {read_ahead})")
    } else {
        "double-buffered prefetch".to_string()
    };
    println!(
        "[ghost] resident ends loaded in {:.1}s; {} layers x {:.1} MiB max streaming window \
         ({}, page cache {}); footprint {:.2} GiB",
        load_started.elapsed().as_secs_f64(),
        n_layers,
        ghost.max_layer_span() as f64 / (1024.0 * 1024.0),
        mode_label,
        if evict_page_cache { "bypassed" } else { "on" },
        gib(phys_footprint_bytes()),
    );

    let token_ids = tokenizer.encode(&prompt, true, false)?;
    println!("[ghost] prompt tokens: {:?}", token_ids);
    let mut pos = 0usize;

    let prefill_started = Instant::now();
    streamer.prime()?;
    let hidden = session
        .weights
        .token_embedding
        .embedding_lookup(&token_ids, "token_embedding_ghost")?;
    let (mut hidden, bytes, wait_us, forward_us, read_us, dec_us) = ghost_stream_layers(
        &mut session,
        &mut streamer,
        &placeholder,
        hidden,
        pos,
        token_ids.len(),
        true,
    )?;
    pos += token_ids.len();
    println!(
        "[ghost] prefill: {:.1}s ({:.2} GiB streamed, blocked {:.1}s | read {:.1}s decode \
         {:.1}s | forward {:.1}s); footprint {:.2} GiB",
        prefill_started.elapsed().as_secs_f64(),
        gib(bytes),
        wait_us as f64 / 1_000_000.0,
        read_us as f64 / 1_000_000.0,
        dec_us as f64 / 1_000_000.0,
        forward_us as f64 / 1_000_000.0,
        gib(phys_footprint_bytes()),
    );

    if spec {
        ghost_spec_decode(
            &mut session,
            &mut streamer,
            &placeholder,
            &tokenizer,
            hidden,
            &token_ids,
            max_tokens,
            draft_len,
        )?;
    } else {
        let mut generated: Vec<u32> = Vec::new();
        let mut decode_us_total: u128 = 0;
        for step in 0..max_tokens {
            let logits = session.forward_final_norm_and_logits(&hidden)?;
            let vocab = logits.dim(1)?;
            let rows = logits.dim(0)?;
            let last_row_start = (rows - 1) * vocab;
            let last_row = CpuTensor::from_f32(
                "ghost_last_logits",
                vec![1, vocab],
                logits.data[last_row_start..last_row_start + vocab].to_vec(),
            )?;
            let token = LlamaSampler::Greedy.sample(&last_row)?;
            generated.push(token);
            print!("{}", tokenizer.decode(&[token], true)?);
            std::io::stdout().flush()?;
            if tokenizer.special.eos == Some(token) || tokenizer.special.eot == Some(token) {
                break;
            }
            if step + 1 == max_tokens {
                break;
            }
            let token_started = Instant::now();
            let embedding = session
                .weights
                .token_embedding
                .embedding_lookup(&[token], "token_embedding_ghost")?;
            let (next_hidden, bytes, wait_us, forward_us, read_us, dec_us) = ghost_stream_layers(
                &mut session,
                &mut streamer,
                &placeholder,
                embedding,
                pos,
                1,
                false,
            )?;
            hidden = next_hidden;
            pos += 1;
            let token_us = token_started.elapsed().as_micros();
            decode_us_total += token_us;
            eprintln!(
                "[ghost] token {:>3}: {:6.0} ms ({:.2} GiB streamed, blocked {:5.0} ms | read \
                 {:5.0} decode {:5.0} | forward {:5.0} ms)",
                step + 1,
                token_us as f64 / 1000.0,
                gib(bytes),
                wait_us as f64 / 1000.0,
                read_us as f64 / 1000.0,
                dec_us as f64 / 1000.0,
                forward_us as f64 / 1000.0,
            );
        }
        println!();

        let streamed_tokens = generated.len().saturating_sub(1);
        if streamed_tokens > 0 {
            println!(
                "[ghost] decode: {} tokens in {:.1}s = {:.3} tok/s",
                streamed_tokens,
                decode_us_total as f64 / 1_000_000.0,
                streamed_tokens as f64 / (decode_us_total as f64 / 1_000_000.0),
            );
        }
    }
    println!(
        "[ghost] final footprint {:.2} GiB, peak RSS {:.2} GiB",
        gib(phys_footprint_bytes()),
        gib(peak_rss_bytes()),
    );
    Ok(())
}

fn run_ghost_moe(
    model: &std::path::Path,
    cghost: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    expert_cache_mib: usize,
    evict_page_cache: bool,
) -> anyhow::Result<()> {
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let loaded = Instant::now();
    let runtime = camelid::gemma4_runtime::Gemma4Runtime::load_ghost_moe(
        model,
        cghost,
        expert_cache_mib,
        evict_page_cache,
    )?;
    println!(
        "[ghost-moe] shared Gemma 4 core loaded in {:.1}s; host expert cache {} MiB; {} expert reads; footprint {:.2} GiB",
        loaded.elapsed().as_secs_f64(),
        expert_cache_mib,
        if evict_page_cache {
            "strict no-page-cache"
        } else {
            "buffered page-cache"
        },
        gib(phys_footprint_bytes())
    );
    let started = Instant::now();
    let (text, ids) = runtime.generate_greedy_streaming(prompt, max_tokens, |delta| {
        print!("{delta}");
        let _ = std::io::stdout().flush();
    })?;
    println!();
    let secs = started.elapsed().as_secs_f64();
    let stats = runtime.ghost_moe_cache_stats().unwrap_or_default();
    let lookups = stats.hits + stats.misses;
    let hit_rate = if lookups == 0 {
        0.0
    } else {
        100.0 * stats.hits as f64 / lookups as f64
    };
    println!(
        "[ghost-moe] {} tokens in {:.1}s = {:.2} tok/s; host cache {} hits / {} misses ({:.1}%), {} evictions, {:.2} GiB host-cache read",
        ids.len(),
        secs,
        ids.len() as f64 / secs.max(f64::EPSILON),
        stats.hits,
        stats.misses,
        hit_rate,
        stats.evictions,
        gib(stats.bytes_read)
    );
    println!(
        "[ghost-moe] host cache resident: {} experts, {:.1}/{:.1} MiB; final footprint {:.2} GiB, peak RSS {:.2} GiB; token_ids={ids:?}",
        stats.resident_experts,
        stats.resident_bytes as f64 / (1024.0 * 1024.0),
        stats.budget_bytes as f64 / (1024.0 * 1024.0),
        gib(phys_footprint_bytes()),
        gib(peak_rss_bytes())
    );
    // Keep the returned final string live through diagnostics so the optimizer
    // cannot elide detokenization in benchmark builds.
    let _ = text;
    Ok(())
}

/// WRAITH Phase-3 speculative ghost decode. Draft L tokens with a resident zero-weight n-gram,
/// verify `[anchor, draft_1..draft_L]` in ONE streamed sweep (each layer read once, applied to
/// all L+1 positions), accept the greedy-identical prefix, then roll the KV cache position back
/// over the rejected tail. Because a single causal `position` cursor bounds every attention
/// read, that rollback alone makes the rejected slots unreadable — no buffer truncation — so the
/// accepted stream is byte-identical to non-spec ghost greedy. A ghost sweep's cost is dominated
/// by the fixed per-layer disk read, so amortizing it across `1 + accepted` committed tokens is
/// the win. An EMA auto-disable drops drafting to single-token sweeps when acceptance collapses
/// (novel text) and re-probes periodically, so spec never badly regresses a non-repetitive load.
#[allow(clippy::too_many_arguments)]
fn ghost_spec_decode(
    session: &mut LlamaInferenceSession,
    streamer: &mut GhostStreamer,
    placeholder: &LlamaLayerWeights,
    tokenizer: &Tokenizer,
    prefill_hidden: CpuTensor,
    prompt_ids: &[u32],
    max_tokens: usize,
    draft_len: usize,
) -> anyhow::Result<()> {
    use camelid::inference::speculative::{accepted_draft_prefix, NGramDrafter};

    let draft_len = draft_len.min(15); // verify-batch width cap (MAX_VERIFY_K - 1)
    let drafter = NGramDrafter::default();
    let mut history: Vec<u32> = prompt_ids.to_vec();
    let is_stop = |t: u32| tokenizer.special.eos == Some(t) || tokenizer.special.eot == Some(t);

    // Per-row greedy argmax via the SAME sampler the non-spec path uses, so tie-breaking (and
    // therefore the accepted token stream) is identical.
    let greedy_rows = |logits: &CpuTensor| -> anyhow::Result<Vec<u32>> {
        let rows = logits.dim(0)?;
        let vocab = logits.dim(1)?;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * vocab;
            let row = CpuTensor::from_f32(
                "ghost_spec_logits",
                vec![1, vocab],
                logits.data[start..start + vocab].to_vec(),
            )?;
            out.push(LlamaSampler::Greedy.sample(&row)?);
        }
        Ok(out)
    };

    // The first token comes free from the prefill hidden (no sweep), exactly as non-spec does:
    // argmax the LAST prefill row only (not all N prompt rows).
    let ttft = {
        let logits = session.forward_final_norm_and_logits(&prefill_hidden)?;
        let vocab = logits.dim(1)?;
        let rows = logits.dim(0)?;
        let last = (rows - 1) * vocab;
        let row = CpuTensor::from_f32(
            "ghost_spec_ttft",
            vec![1, vocab],
            logits.data[last..last + vocab].to_vec(),
        )?;
        LlamaSampler::Greedy.sample(&row)?
    };
    let mut generated: Vec<u32> = vec![ttft];
    print!("{}", tokenizer.decode(&[ttft], true)?);
    std::io::stdout().flush()?;
    history.push(ttft);
    let mut current = ttft;

    let decode_started = Instant::now();
    let mut sweeps = 0usize;
    let mut drafted_total = 0usize;
    let mut accepted_total = 0usize;
    // Rounds where at least one drafted token was REJECTED — i.e. the KV rollback discarded a
    // non-empty rejected tail. If rejected KV leaked, parity vs non-spec would break; a run with
    // rejected_rounds > 0 that stays byte-identical is the rejected-KV isolation proof.
    let mut rejected_rounds = 0usize;
    let mut ema_accepted = draft_len as f64; // optimistic start; drives auto-disable
    let mut since_probe = 0usize;
    let mut auto_disabled_ever = false;

    'outer: while generated.len() < max_tokens && !is_stop(current) {
        // Auto-disable: when recent acceptance collapses, stop drafting (single-token sweeps)
        // and re-probe every 64 rounds so a return to repetitive text is picked back up.
        let drafting_on = ema_accepted >= 0.5 || since_probe >= 64;
        if drafting_on {
            since_probe = 0;
        } else {
            since_probe += 1;
            auto_disabled_ever = true;
        }
        let room = max_tokens - generated.len();
        let budget = if drafting_on {
            draft_len.min(room.saturating_sub(1))
        } else {
            0
        };
        let drafts = if budget > 0 {
            drafter.draft(&history, budget)
        } else {
            Vec::new()
        };
        drafted_total += drafts.len();

        // Verify batch = [anchor, draft_1..draft_L]; ONE streamed sweep over all layers writes
        // KV for positions [base, base+len) and yields per-position greedy predictions.
        let base = session.kv_position();
        let mut batch = Vec::with_capacity(1 + drafts.len());
        batch.push(current);
        batch.extend_from_slice(&drafts);
        let embedding = session
            .weights
            .token_embedding
            .embedding_lookup(&batch, "token_embedding_ghost_spec")?;
        let (rows_hidden, _bytes, _wait, _fwd, _read, _dec) = ghost_stream_layers(
            session,
            streamer,
            placeholder,
            embedding,
            base,
            batch.len(),
            false,
        )?;
        sweeps += 1;
        let logits = session.forward_final_norm_and_logits(&rows_hidden)?;
        let predictions = greedy_rows(&logits)?;
        let accepted = accepted_draft_prefix(&drafts, &predictions);
        accepted_total += accepted;
        if accepted < drafts.len() {
            rejected_rounds += 1; // a non-empty rejected tail was rolled back this round
        }
        ema_accepted = 0.85 * ema_accepted + 0.15 * accepted as f64;

        // Commit the anchor (position `base`) + `accepted` drafts; roll the position back over
        // the rest so rejected KV at [base+1+accepted .. base+len) is causally unreadable.
        session.rollback_to_position(base + 1 + accepted)?;

        // Emit predictions[0..=accepted] = the accepted drafts plus one correction token.
        for &token in &predictions[..=accepted] {
            if generated.len() >= max_tokens {
                break;
            }
            generated.push(token);
            history.push(token);
            print!("{}", tokenizer.decode(&[token], true)?);
            std::io::stdout().flush()?;
            current = token;
            if is_stop(token) {
                break 'outer;
            }
        }
    }
    println!();

    let secs = decode_started.elapsed().as_secs_f64();
    let decoded = generated.len().saturating_sub(1); // the TTFT was free
    let accept_rate = if drafted_total > 0 {
        accepted_total as f64 / drafted_total as f64
    } else {
        0.0
    };
    println!(
        "[ghost] spec decode: {decoded} tokens in {secs:.1}s = {:.3} tok/s | {sweeps} sweeps \
         ({:.2} tok/sweep) | draft_len {draft_len}, drafted {drafted_total}, accepted \
         {accepted_total} ({:.0}%), rejected-tail rounds {rejected_rounds}, mean {:.2}/round, \
         auto-disable {}",
        if secs > 0.0 {
            decoded as f64 / secs
        } else {
            0.0
        },
        if sweeps > 0 {
            decoded as f64 / sweeps as f64
        } else {
            0.0
        },
        accept_rate * 100.0,
        if sweeps > 0 {
            accepted_total as f64 / sweeps as f64
        } else {
            0.0
        },
        if auto_disabled_ever {
            "fired"
        } else {
            "not fired"
        },
    );
    Ok(())
}

/// One JSON metrics record per measured generation iteration (stdout, JSONL).
#[derive(Serialize)]
struct BenchGenerateRecord {
    runtime: &'static str,
    commit: String,
    model: String,
    quantization: String,
    iteration: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    load_ms: f64,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    tokens_per_second: f64,
    peak_memory_bytes: u64,
    /// GPU layer-offload split for this run (Phase 4 honest labeling). `None` on the
    /// CPU path; `source == "none"` means fully resident on the GPU. A non-zero
    /// `layers_offloaded` means a tok/s number here is a capacity-mode result, not a
    /// fully-resident one — the field carries the split + measured PCIe so it can't be
    /// read as native.
    #[serde(skip_serializing_if = "Option::is_none")]
    offload: Option<camelid::offload::OffloadRunStatus>,
    output_text: String,
    output_token_ids: Vec<u32>,
}

/// Commit label carried by benchmark JSONL records. An explicit runtime override
/// remains useful for externally orchestrated comparisons, but ordinary runs
/// must identify the binary that actually produced the record.
fn benchmark_commit() -> String {
    let runtime_override = std::env::var("CAMELID_COMMIT").ok();
    resolve_benchmark_commit(
        runtime_override.as_deref(),
        &camelid::receipt::camelid_commit(),
    )
}

fn resolve_benchmark_commit(runtime_override: Option<&str>, embedded_commit: &str) -> String {
    runtime_override
        .map(str::trim)
        .filter(|commit| !commit.is_empty())
        .unwrap_or(embedded_commit)
        .to_string()
}

#[cfg(test)]
mod benchmark_commit_tests {
    use super::resolve_benchmark_commit;

    #[test]
    fn benchmark_commit_defaults_to_the_binarys_embedded_commit() {
        assert_eq!(
            resolve_benchmark_commit(None, "0123456789abcdef"),
            "0123456789abcdef"
        );
        assert_eq!(
            resolve_benchmark_commit(Some(" \t"), "0123456789abcdef"),
            "0123456789abcdef"
        );
    }

    #[test]
    fn benchmark_commit_honors_a_nonempty_runtime_override() {
        assert_eq!(
            resolve_benchmark_commit(Some("  campaign-label  "), "0123456789abcdef"),
            "campaign-label"
        );
    }
}

/// §5: set (or clear) the three managed x86 Q8 `groups_per_chunk` env knobs for a
/// trial. `None` clears them so the trial measures the profile's default tiling;
/// `Some` pins the search candidate's values. Must be called AFTER
/// `PlannerEnv::apply` (these keys are managed) so the override is authoritative.
fn apply_groups_per_chunk(gpc: Option<camelid::gait::calibrate::GroupsPerChunk>) {
    const ATTN: &str = "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK";
    const FFN: &str = "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK";
    const MATMUL: &str = "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK";
    match gpc {
        Some(g) => {
            std::env::set_var(ATTN, g.attn_qkv_decode.to_string());
            std::env::set_var(FFN, g.ffn_gate_up_decode.to_string());
            std::env::set_var(MATMUL, g.packed_rows4_matmul.to_string());
        }
        None => {
            std::env::remove_var(ATTN);
            std::env::remove_var(FFN);
            std::env::remove_var(MATMUL);
        }
    }
}

fn gait_profile_env_value(profile: &camelid::execution_plan::ExecutionProfile) -> &'static str {
    use camelid::execution_plan::ExecutionProfile::*;
    match profile {
        Auto => "auto",
        Safe => "safe",
        Experimental => "experimental",
        Debug => "debug",
    }
}

/// Time one candidate: select its profile for the planner, reload weights (the
/// Q8 repack / kernel choice happens at load time, so each candidate needs its
/// own load), run one unmeasured warmup, then a measured greedy decode. The
/// parity token is the SHA-256 of the greedy output token ids — a candidate that
/// changes the output is disqualified by the tournament.
fn gait_profile_trial(
    model: &std::path::Path,
    threads: Option<usize>,
    prompt_token_ids: &[u32],
    max_tokens: usize,
    candidate: &camelid::gait::calibrate::Candidate,
) -> anyhow::Result<camelid::gait::calibrate::TrialResult> {
    std::env::set_var(
        "CAMELID_PROFILE",
        gait_profile_env_value(&candidate.profile),
    );
    // Apply this candidate's Windows scheduling substrate before timing, so the
    // measured decode reflects it. §1.2-scoped to the compute pool (the Rayon
    // workers + this thread), matching what production applies.
    let eco_status = camelid::gait::substrate::set_compute_pool_eco_qos(candidate.eco_qos_opt_out);
    if candidate.eco_qos_opt_out && eco_status != camelid::gait::substrate::EcoQosStatus::OptedOut {
        eprintln!(
            "[gait]   {} eco_qos opt-out unavailable -> {eco_status:?}",
            candidate.label
        );
    }

    let gguf = read_metadata(model)?;
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::ViaSessionDecode)?;
    // Apply this candidate's plan before loading weights, exactly as bench-generate does.
    let plan = camelid::execution_plan::plan_for_model(model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan.env_updates);
    // §5: apply this candidate's groups_per_chunk tiling AFTER the planner's
    // env_updates — the gpc knobs are MANAGED_ENV_KEYS, so PlannerEnv::apply would
    // otherwise clear/overwrite them. Applying here lets the search override win.
    apply_groups_per_chunk(candidate.groups_per_chunk);

    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);
    let sampler = LlamaSampler::Greedy;

    let _ = generate_run(
        &config,
        &weights,
        &tokenizer,
        prompt_token_ids,
        &sampler,
        max_tokens,
    )?;
    camelid::inference::reset_stage_timings();
    let run = generate_run(
        &config,
        &weights,
        &tokenizer,
        prompt_token_ids,
        &sampler,
        max_tokens,
    )?;

    let decode_tokens = run.generated.len().saturating_sub(1);
    let tokens_per_s = if run.decode_ms > 0.0 && decode_tokens > 0 {
        decode_tokens as f64 / (run.decode_ms / 1000.0)
    } else {
        0.0
    };
    let mut id_bytes = Vec::with_capacity(run.generated.len() * 4);
    for id in &run.generated {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    let parity_token = camelid::receipt::sha256_hex(&id_bytes);
    Ok(camelid::gait::calibrate::TrialResult {
        tokens_per_s,
        parity_token,
    })
}

fn run_gait_calibrate(
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    rounds: usize,
    warmup: usize,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    use camelid::execution_plan::ExecutionProfile;
    use camelid::gait::calibrate::{
        calibrate_and_store, default_store_dir, Candidate, TournamentConfig,
    };

    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    anyhow::ensure!(rounds >= 1, "--rounds must be at least 1");
    // Calibration must measure the candidates we choose — never let a previously
    // cached gait receipt override the candidate's profile mid-trial.
    std::env::remove_var("CAMELID_GAIT");
    // §1.2: calibrate under the same core-reserve cap production will run with, so
    // the measured tok/s reflects the host-safe thread budget. The gate is off
    // here, so configure_rayon_threads won't apply the cap itself.
    configure_rayon_threads(host_safe_thread_count(threads))?;
    camelid::capability::HardwareProfile::detect().log();

    anyhow::ensure!(
        prompt_file.is_some() || prompt.is_some(),
        "provide --prompt-file <path> or --prompt <text>"
    );

    // The gguf is needed for the fingerprint, memory measurement, and roofline
    // numerator; each candidate's prompt encoding + decode happens in its own
    // child trial (§1.4 crash isolation), so the parent does not load weights.
    let gguf = read_metadata(&model)?;

    // Baseline = today's behavior (Auto profile, OS-managed throttling). The
    // candidates vary the EcoQoS substrate (and profile) so the tournament
    // measures whether disabling throttling helps on this machine, parity-held.
    let baseline = Candidate {
        label: "auto".to_string(),
        profile: ExecutionProfile::Auto,
        eco_qos_opt_out: false,
        groups_per_chunk: None,
    };
    // §5 bounded local search: the EcoQoS substrate dimension (auto+ecoqos) plus
    // the experimental kernel under each groups_per_chunk neighbor. Every
    // candidate is parity-gated + crash-isolated; the tournament fails closed to
    // baseline if none beats it by margin (the honest outcome on a
    // memory-bandwidth-bound box, where the tiling knob is expected to be flat).
    let mut candidates = vec![Candidate {
        label: "auto+ecoqos".to_string(),
        profile: ExecutionProfile::Auto,
        eco_qos_opt_out: true,
        groups_per_chunk: None,
    }];
    for gpc in camelid::gait::calibrate::groups_per_chunk_neighbors() {
        candidates.push(Candidate {
            label: format!(
                "exp+gpc[{},{},{}]",
                gpc.attn_qkv_decode, gpc.ffn_gate_up_decode, gpc.packed_rows4_matmul
            ),
            profile: ExecutionProfile::Experimental,
            eco_qos_opt_out: false,
            groups_per_chunk: Some(gpc),
        });
    }

    let store_dir = default_store_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve the gait store directory"))?;

    let config = TournamentConfig {
        rounds,
        warmup_rounds: warmup,
        ..TournamentConfig::default()
    };
    eprintln!(
        "[gait] calibrating {} candidates (+baseline), {} measured rounds (+{} warmup), interleaved, on {} ...",
        candidates.len(),
        config.rounds,
        config.warmup_rounds,
        model.display()
    );
    // §1.1 host-safety: do not launch the calibration allocation campaign (each
    // child trial loads multi-GB weights) if the box is already low on free RAM.
    #[cfg(windows)]
    if let Some((total, avail)) = camelid::gait::host_ram_status() {
        if !camelid::gait::ram_headroom_ok(total, avail) {
            let floor = camelid::gait::ram_headroom_floor(total);
            eprintln!(
                "[gait] insufficient free RAM (avail {:.1} GiB < floor {:.1} GiB) -> skipping calibration; baseline serves",
                avail as f64 / 1e9,
                floor as f64 / 1e9
            );
            return Ok(());
        }
    }

    // §1.4 crash isolation: each candidate runs in a supervised CHILD PROCESS, so
    // a candidate that segfaults or hangs cannot take down this process. The
    // per-candidate timeout is min(3x the baseline's wall time, the absolute
    // ceiling); the baseline (timed first) gets the full ceiling.
    let exe = std::env::current_exe()
        .map_err(|err| anyhow::anyhow!("cannot resolve current exe for child trials: {err}"))?;
    let baseline_label = baseline.label.clone();
    let mut baseline_wall: Option<std::time::Duration> = None;
    let trial = |candidate: &Candidate| -> Option<camelid::gait::calibrate::TrialResult> {
        let timeout = match baseline_wall {
            Some(bw) => bw.mul_f64(3.0).min(CAL_TRIAL_CEILING),
            None => CAL_TRIAL_CEILING,
        };
        let started = std::time::Instant::now();
        let result = run_trial_in_child(
            &exe,
            &model,
            &prompt_file,
            &prompt,
            max_tokens,
            candidate,
            threads,
            timeout,
        );
        if candidate.label == baseline_label && baseline_wall.is_none() {
            baseline_wall = Some(started.elapsed());
        }
        match &result {
            Some(r) => eprintln!(
                "[gait] {:<13} {:>7.2} tok/s  parity {}",
                candidate.label,
                r.tokens_per_s,
                &r.parity_token[..12.min(r.parity_token.len())]
            ),
            None => eprintln!(
                "[gait] {:<13} disqualified (timeout/crash/parse)",
                candidate.label
            ),
        }
        result
    };

    let (outcome, path) =
        calibrate_and_store(&store_dir, &gguf, &baseline, &candidates, &config, trial);

    println!("{}", serde_json::to_string_pretty(&outcome)?);
    match path {
        Some(p) => eprintln!(
            "[gait] selected {:?} :: {} :: receipt {}",
            outcome.selected_profile,
            outcome.reason,
            p.display()
        ),
        None => eprintln!(
            "[gait] selected {:?} :: {} :: receipt NOT stored (store write failed)",
            outcome.selected_profile, outcome.reason
        ),
    }
    Ok(())
}

/// `camelid gait reset` — clear the entire GAIT cache, reverting fully to the
/// baseline path (§1.3). Best-effort and idempotent: a missing cache is not an
/// error. Deleting the folder is the documented manual revert; this is the
/// in-CLI equivalent.
fn run_gait_reset() -> anyhow::Result<()> {
    match camelid::gait::gait_dir() {
        Some(dir) if dir.exists() => {
            std::fs::remove_dir_all(&dir)?;
            println!("gait: cleared cache at {}", dir.display());
        }
        Some(dir) => {
            println!("gait: nothing to clear ({} does not exist)", dir.display());
        }
        None => println!("gait: no cache directory could be resolved"),
    }
    Ok(())
}

/// Per-candidate absolute timeout ceiling for a child trial (§1.4). The live
/// per-candidate budget is `min(3x the baseline's wall time, this ceiling)`.
const CAL_TRIAL_CEILING: std::time::Duration = std::time::Duration::from_secs(180);

/// `camelid gait-trial` (internal): run ONE candidate trial in this isolated
/// child process and print its TrialResult as a single JSON line to stdout. The
/// crash/hang isolation lives in the PARENT supervisor ([`run_trial_in_child`]);
/// here we just run the trial and report.
#[allow(clippy::too_many_arguments)]
fn run_gait_trial(
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    profile: String,
    eco_qos: bool,
    threads: Option<usize>,
    gpc_attn: Option<usize>,
    gpc_ffn: Option<usize>,
    gpc_matmul: Option<usize>,
) -> anyhow::Result<()> {
    use camelid::execution_plan::ExecutionProfile;
    use camelid::gait::calibrate::{Candidate, GroupsPerChunk};

    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    std::env::remove_var("CAMELID_GAIT");
    configure_rayon_threads(host_safe_thread_count(threads))?;

    let profile_label = profile.clone();
    let profile = match profile.to_ascii_lowercase().as_str() {
        "auto" => ExecutionProfile::Auto,
        "safe" => ExecutionProfile::Safe,
        "experimental" => ExecutionProfile::Experimental,
        "debug" => ExecutionProfile::Debug,
        other => anyhow::bail!("unknown profile {other:?} (want auto|safe|experimental|debug)"),
    };

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };
    let gguf = read_metadata(&model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    anyhow::ensure!(
        !prompt_token_ids.is_empty(),
        "prompt encoded to zero tokens"
    );

    // §5: the three gpc knobs travel together — all present, or none.
    let groups_per_chunk = match (gpc_attn, gpc_ffn, gpc_matmul) {
        (Some(a), Some(f), Some(m)) => Some(GroupsPerChunk {
            attn_qkv_decode: a,
            ffn_gate_up_decode: f,
            packed_rows4_matmul: m,
        }),
        (None, None, None) => None,
        _ => anyhow::bail!(
            "groups_per_chunk override requires all three of --gpc-attn / --gpc-ffn / --gpc-matmul"
        ),
    };
    let candidate = Candidate {
        label: profile_label,
        profile,
        eco_qos_opt_out: eco_qos,
        groups_per_chunk,
    };
    let result = gait_profile_trial(&model, threads, &prompt_token_ids, max_tokens, &candidate)?;
    // The ONLY stdout line — the JSON result the parent supervisor parses.
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

/// Run ONE candidate trial in a supervised child process, returning its result or
/// `None` if it timed out, crashed, exited non-zero, or its output could not be
/// parsed (the candidate is disqualified upstream). This is the §1.4 crash-
/// isolation boundary: a segfaulting/hanging candidate kernel cannot take down
/// the calibrating (or, in production, serving) process.
#[allow(clippy::too_many_arguments)]
fn run_trial_in_child(
    exe: &std::path::Path,
    model: &std::path::Path,
    prompt_file: &Option<PathBuf>,
    prompt: &Option<String>,
    max_tokens: usize,
    candidate: &camelid::gait::calibrate::Candidate,
    threads: Option<usize>,
    timeout: std::time::Duration,
) -> Option<camelid::gait::calibrate::TrialResult> {
    use camelid::gait::calibrate::{supervise, TrialResult, WatchdogOutcome};
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(exe);
    cmd.arg("gait-trial")
        .arg(model)
        .arg("--profile")
        .arg(gait_profile_env_value(&candidate.profile))
        .arg("--max-tokens")
        .arg(max_tokens.to_string());
    if candidate.eco_qos_opt_out {
        cmd.arg("--eco-qos");
    }
    if let Some(t) = threads {
        cmd.arg("--threads").arg(t.to_string());
    }
    if let Some(g) = candidate.groups_per_chunk {
        cmd.arg("--gpc-attn")
            .arg(g.attn_qkv_decode.to_string())
            .arg("--gpc-ffn")
            .arg(g.ffn_gate_up_decode.to_string())
            .arg("--gpc-matmul")
            .arg(g.packed_rows4_matmul.to_string());
    }
    match (prompt_file, prompt) {
        (Some(path), _) => {
            cmd.arg("--prompt-file").arg(path);
        }
        (None, Some(text)) => {
            cmd.arg("--prompt").arg(text);
        }
        (None, None) => {}
    }
    // CPU calibration; keep the child off the GPU and out of the gait selector.
    cmd.env("CUDA_VISIBLE_DEVICES", "-1");
    cmd.env_remove("CAMELID_GAIT");
    // stdout carries the JSON result; stderr is inherited so the child's logs show.
    cmd.stdout(Stdio::piped());

    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            eprintln!("[gait] {:<13} child spawn failed: {err}", candidate.label);
            return None;
        }
    };

    match supervise(child, timeout, std::time::Duration::from_millis(100)) {
        WatchdogOutcome::Completed(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            // The result is the last parseable JSON line (robust to stray stdout).
            stdout
                .lines()
                .rev()
                .find_map(|line| serde_json::from_str::<TrialResult>(line.trim()).ok())
        }
        WatchdogOutcome::Completed(out) => {
            eprintln!(
                "[gait] {:<13} child exited {} -> disqualified",
                candidate.label, out.status
            );
            None
        }
        WatchdogOutcome::TimedOut => {
            eprintln!(
                "[gait] {:<13} TIMED OUT after {timeout:?} -> disqualified",
                candidate.label
            );
            None
        }
        WatchdogOutcome::Errored => {
            eprintln!(
                "[gait] {:<13} child supervision error -> disqualified",
                candidate.label
            );
            None
        }
    }
}

struct GenerationRun {
    generated: Vec<u32>,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
}

/// One full single-node generation with a fresh KV cache (weights are reused).
fn generate_run(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    sampler: &LlamaSampler,
    max_tokens: usize,
) -> anyhow::Result<GenerationRun> {
    let mut session = LlamaInferenceSession::new(config.clone(), weights.clone())?;
    let mut history: Vec<u32> = prompt_tokens.to_vec();
    let mut input: Vec<u32> = prompt_tokens.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    // Prefill + first token: this whole span is time-to-first-token.
    let ttft_start = Instant::now();
    let step = session.generate_next_token_with_history_diagnostics(
        &input,
        sampler.clone(),
        &history,
        false,
        None,
    )?;
    let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_ms = step.prefill_timings.total as f64 / 1000.0; // microseconds -> ms
    let first = step.next_token_id;
    generated.push(first);
    history.push(first);
    let mut finished = tokenizer.special.eog.contains(&first);
    input.clear();
    input.push(first);

    // Decode the remaining tokens (pure decode throughput).
    // CAMELID_DECODE_TIME=1: split per-token wall into forward / sample / other.
    let time_decode = std::env::var_os("CAMELID_DECODE_TIME").is_some();
    // Phase 0 instrumentation: per-stage decode time sinks (CPU path). Aggregates the
    // already-collected per-layer timings across all decode steps. GPU resident decode
    // runs as one fused call so these stay ~0 there; meaningful on the CPU forward.
    let stage_timings = std::env::var_os("CAMELID_STAGE_TIMINGS").is_some();
    let mut stage_us: std::collections::BTreeMap<&'static str, u128> =
        std::collections::BTreeMap::new();
    let (mut fwd_us, mut sample_us, mut steps, mut wall_us) = (0u128, 0u128, 0u64, 0u128);
    let (mut emb_us, mut layers_us) = (0u128, 0u128);
    let greedy = matches!(sampler, LlamaSampler::Greedy)
        && std::env::var_os("CAMELID_NO_GPU_SAMPLE").is_none();
    // CAMELID_SPEC_NGRAM=<max_draft>: greedy GPU speculative decoding (lossless).
    let spec_draft = std::env::var("CAMELID_SPEC_NGRAM")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0);
    // Adaptive drafting: an EMA of how many drafts get accepted per round tunes the
    // n-gram length. Start conservative (precise 4-gram, which rarely drafts on
    // non-repetitive text so it isn't slowed) and only loosen to an aggressive
    // 2-gram once repetition is proven by a high acceptance rate.
    let mut spec_ema = 0.5f32;
    let decode_start = Instant::now();
    while !finished && generated.len() < max_tokens {
        let step_started = Instant::now();
        // Greedy speculative decoding: one batched verify can emit several tokens.
        // Falls through to the single-token path when no draft / engine not ready.
        if greedy {
            if let Some(nd) = spec_draft {
                let ngram = if spec_ema >= 2.0 {
                    2
                } else if spec_ema >= 0.9 {
                    3
                } else {
                    4
                };
                if let Some(toks) =
                    session.generate_next_tokens_speculative(input[0], &history, nd, ngram)?
                {
                    // accepted drafts = tokens emitted minus the always-present bonus.
                    let accepted = toks.len().saturating_sub(1) as f32;
                    spec_ema = 0.7 * spec_ema + 0.3 * accepted;
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        steps += 1;
                    }
                    for t in toks {
                        if generated.len() >= max_tokens {
                            break;
                        }
                        generated.push(t);
                        history.push(t);
                        if tokenizer.special.eog.contains(&t) {
                            finished = true;
                            break;
                        }
                    }
                    input.clear();
                    input.push(*generated.last().expect("at least one token"));
                    continue;
                }
            }
        }
        // Greedy decode rides the resident fast lane (GPU argmax + embedding gather,
        // next graph pre-released); anything else takes the general sampling path.
        let next = if greedy {
            match session.generate_next_token_greedy_resident(input[0])? {
                Some((id, forward_us)) => {
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += forward_us;
                        steps += 1;
                    }
                    id
                }
                None => {
                    let step = session.generate_next_token_with_history_diagnostics(
                        &input,
                        sampler.clone(),
                        &history,
                        false,
                        None,
                    )?;
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += step.timings.total;
                        sample_us += step.sample;
                        emb_us += step.timings.embedding;
                        layers_us += step.timings.layers_total;
                        steps += 1;
                    }
                    if stage_timings {
                        accumulate_stage_timings(&mut stage_us, &step.timings);
                    }
                    step.next_token_id
                }
            }
        } else {
            // Temperature-only sampling rides the GPU Gumbel-max fast lane (no host
            // logits copy, no CPU sort); top-k / top-p / penalties fall through to
            // the CPU sampler.
            let gpu_sampled = match &sampler {
                LlamaSampler::Sampling(cfg) => {
                    session.generate_next_token_sampled_resident(input[0], cfg)?
                }
                LlamaSampler::Greedy => None,
            };
            match gpu_sampled {
                Some((id, forward_us)) => {
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += forward_us;
                        steps += 1;
                    }
                    id
                }
                None => {
                    let step = session.generate_next_token_with_history_diagnostics(
                        &input,
                        sampler.clone(),
                        &history,
                        false,
                        None,
                    )?;
                    if time_decode {
                        wall_us += step_started.elapsed().as_micros();
                        fwd_us += step.timings.total;
                        sample_us += step.sample;
                        emb_us += step.timings.embedding;
                        layers_us += step.timings.layers_total;
                        steps += 1;
                    }
                    if stage_timings {
                        accumulate_stage_timings(&mut stage_us, &step.timings);
                    }
                    step.next_token_id
                }
            }
        };
        generated.push(next);
        history.push(next);
        finished = tokenizer.special.eog.contains(&next);
        input.clear();
        input.push(next);
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    if stage_timings && !stage_us.is_empty() {
        let total: u128 = stage_us.values().sum();
        let mut ranked: Vec<(&&str, &u128)> = stage_us.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1));
        eprintln!("[stage-timings] per-decode-step CPU breakdown (sum of all layers), total {:.2} ms/token over {} steps:", total as f64 / generated.len().max(1) as f64 / 1000.0, generated.len());
        for (name, us) in ranked {
            eprintln!(
                "  {:>18}  {:6.2}%  {:8.3} ms/token",
                name,
                *us as f64 / total as f64 * 100.0,
                *us as f64 / generated.len().max(1) as f64 / 1000.0,
            );
        }
    }
    if time_decode && steps > 0 {
        eprintln!(
            "[decode-time] per token: step wall {:.2}ms | forward {:.2}ms (embed {:.3} layers {:.2}) | sample {:.2}ms | in-step other {:.2}ms | loop other {:.2}ms",
            wall_us as f64 / steps as f64 / 1000.0,
            fwd_us as f64 / steps as f64 / 1000.0,
            emb_us as f64 / steps as f64 / 1000.0,
            layers_us as f64 / steps as f64 / 1000.0,
            sample_us as f64 / steps as f64 / 1000.0,
            (wall_us - fwd_us - sample_us) as f64 / steps as f64 / 1000.0,
            (decode_start.elapsed().as_micros() - wall_us) as f64 / steps as f64 / 1000.0,
        );
    }

    Ok(GenerationRun {
        generated,
        prefill_ms,
        ttft_ms,
        decode_ms,
    })
}

/// Phase 0 instrumentation: fold one forward step's per-stage timings into a
/// running per-stage accumulator, so a decode run can report where CPU time goes.
fn accumulate_stage_timings(
    acc: &mut std::collections::BTreeMap<&'static str, u128>,
    t: &LlamaForwardTimings,
) {
    *acc.entry("embedding").or_default() += t.embedding;
    *acc.entry("final_norm").or_default() += t.final_norm;
    *acc.entry("logits(output_proj)").or_default() += t.logits;
    for l in &t.layers {
        *acc.entry("attn_norm").or_default() += l.attention_norm;
        *acc.entry("attn_q_proj").or_default() += l.attention_q;
        *acc.entry("attn_k_proj").or_default() += l.attention_k;
        *acc.entry("attn_v_proj").or_default() += l.attention_v;
        *acc.entry("attn_rope").or_default() += l.attention_rope;
        *acc.entry("kv_write").or_default() += l.kv_cache_write;
        *acc.entry("attn_context").or_default() += l.attention_context;
        *acc.entry("attn_out_proj").or_default() += l.attention_output;
        *acc.entry("attn_residual").or_default() += l.attention_residual;
        *acc.entry("ffn_norm").or_default() += l.ffn_norm;
        *acc.entry("ffn_gate").or_default() += l.ffn_gate;
        *acc.entry("ffn_up").or_default() += l.ffn_up;
        *acc.entry("ffn_activation").or_default() += l.ffn_activation;
        *acc.entry("ffn_down").or_default() += l.ffn_down;
        *acc.entry("ffn_residual").or_default() += l.ffn_residual;
    }
}

/// A LlamaModelConfig for a known architecture, so `plan-offload --arch` can size
/// a model whose GGUF isn't on disk. Only the fields the offload planner reads
/// (dims, vocab, quant) are meaningful; the rest take neutral defaults.
fn known_arch_config(arch: &str) -> anyhow::Result<LlamaModelConfig> {
    // (block_count, hidden, ffn, heads, kv_heads, vocab, context)
    let (
        block_count,
        embedding_length,
        feed_forward_length,
        heads,
        kv_heads,
        vocab,
        context_length,
    ) = match arch.to_lowercase().as_str() {
        "llama-8b" | "llama3-8b" | "llama3.1-8b" | "8b" => (32, 4096, 14336, 32, 8, 128256, 131072),
        other => anyhow::bail!("unknown --arch {other:?}; known: llama-8b"),
    };
    Ok(LlamaModelConfig {
        architecture: "llama".to_string(),
        context_length,
        embedding_length,
        block_count,
        feed_forward_length,
        attention_head_count: heads,
        attention_head_count_kv: kv_heads,
        kv_quant: camelid::model::KvCacheQuantization::F16,
        rope_dimension_count: None,
        rope_freq_base: None,
        rope_scaling_type: None,
        rope_scaling_factor: None,
        rope_scaling_original_context_length: None,
        rope_scaling_low_freq_factor: None,
        rope_scaling_high_freq_factor: None,
        rms_norm_epsilon: 1e-5,
        vocab_size: Some(vocab),
        file_type: Some(7), // Q8_0
        attention_key_length: None,
        rope_neox_pairing: false,
        no_rope_layer_step: None,
        moe: None,
        gemma3: None,
        gemma4: None,
        qwen35: None,
        lfm2: None,
        logit_scale: None,
        mla: None,
    })
}

/// Hardened owner-microkernel prefill measurement: load ONCE, then measure all configs INTERLEAVED
/// within each round so per-round paired deltas cancel slow thermal/clock drift. Emits raw
/// per-(round, config) JSONL; paired stats + significance are computed downstream.
#[allow(clippy::too_many_arguments)]
fn run_bench_owner_sweep(
    model: PathBuf,
    lane: String,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    rounds: usize,
    warmup_rounds: usize,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Lane {
        Q8,
        KQuant,
    }
    let lane = match lane.trim().to_ascii_lowercase().as_str() {
        "q8" => Lane::Q8,
        "kquant" | "q4_k" | "k" => Lane::KQuant,
        other => anyhow::bail!("--lane must be `q8` or `kquant`, got `{other}`"),
    };
    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    anyhow::ensure!(rounds >= 1, "--rounds must be at least 1");
    // The sweep mutates owner env keys between configs in-process; without
    // this bypass the process-lifetime runtime-plan caches would freeze the
    // first config and every later one would silently measure it (fake null).
    // Must be set before the first inference resolves the plan.
    std::env::set_var("CAMELID_BENCH_UNCACHED_RUNTIME_PLAN", "1");
    configure_rayon_threads(threads)?;
    camelid::capability::HardwareProfile::detect().log();

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    // Load once. The owner is selected at runtime (env read per linear call), so a single load
    // serves every config; the PackedRows4 repack the owner consumes is built at load regardless.
    let gguf = read_metadata(&model)?;
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::ViaSessionDecode)?;
    let plan_outcome = camelid::execution_plan::plan_for_model(&model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);

    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    let prompt_tokens = prompt_token_ids.len();
    anyhow::ensure!(prompt_tokens >= 1, "prompt encoded to zero tokens");
    let sampler = LlamaSampler::Greedy;

    // Owner keys cleared before each config so "off" is the true default path.
    let owner_keys: &[&str] = match lane {
        Lane::Q8 => &[
            "CAMELID_X86_Q8_MATMUL_OWNER",
            "CAMELID_X86_Q8_MATMUL_OWNER_AVX2",
            "CAMELID_X86_Q8_MATMUL_OWNER_VNNI",
            "CAMELID_X86_Q8_MATMUL_OWNER_4X8",
        ],
        Lane::KQuant => &[
            "CAMELID_X86_KQUANT_MATMUL_OWNER",
            "CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI",
            "CAMELID_X86_KQUANT_MATMUL_OWNER_AVXVNNI256",
            "CAMELID_X86_KQUANT_MATMUL_OWNER_AVX512VNNI",
            "CAMELID_X86_KQUANT_MATMUL_OWNER_REPACK8",
        ],
    };
    // (label, owner_expected_to_fire, env). "off" is EXPLICIT since D15 made
    // the owner default-on for win-x86_64 — an empty env would measure the
    // default (owner on), not the baseline.
    type SweepConfig<'a> = (&'a str, bool, &'a [(&'a str, &'a str)]);
    #[cfg(target_arch = "x86_64")]
    let avx512_vnni = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni");
    #[cfg(not(target_arch = "x86_64"))]
    let avx512_vnni = false;
    #[cfg(target_arch = "x86_64")]
    let avx_vnni = std::arch::is_x86_feature_detected!("avxvnni");
    #[cfg(not(target_arch = "x86_64"))]
    let avx_vnni = false;
    // The folded 512-bit K-quant inner needs no VNNI, so it is reachable on a
    // strictly wider set than `avx512_vnni` — measure it wherever it can run.
    #[cfg(target_arch = "x86_64")]
    let avx512_bw = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let avx512_bw = false;

    // Lane A (Q8).
    let vnni4x4: SweepConfig = (
        "owner_vnni4x4",
        true,
        &[
            ("CAMELID_X86_Q8_MATMUL_OWNER", "all"),
            ("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", "1"),
            ("CAMELID_X86_Q8_MATMUL_OWNER_4X8", "0"),
        ],
    );
    let vnni4x8: SweepConfig = (
        "owner_vnni4x8",
        true,
        &[
            ("CAMELID_X86_Q8_MATMUL_OWNER", "all"),
            ("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", "1"),
            ("CAMELID_X86_Q8_MATMUL_OWNER_4X8", "1"),
        ],
    );
    // Same env as vnni4x4, but on a host without AVX-512 that env selects the 256-bit inner. The
    // label has to say which kernel actually ran: calling it "vnni4x4" here would be a fake null,
    // since the 512-bit arms silently degrade to the AVX2 inner on these parts.
    let avxvnni256: SweepConfig = (
        "owner_avxvnni256",
        true,
        &[
            ("CAMELID_X86_Q8_MATMUL_OWNER", "all"),
            ("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", "1"),
            ("CAMELID_X86_Q8_MATMUL_OWNER_4X8", "0"),
        ],
    );
    let mut q8_configs: Vec<SweepConfig> = vec![
        ("off", false, &[("CAMELID_X86_Q8_MATMUL_OWNER", "off")]),
        (
            "owner_avx2",
            true,
            &[
                ("CAMELID_X86_Q8_MATMUL_OWNER", "all"),
                ("CAMELID_X86_Q8_MATMUL_OWNER_VNNI", "0"),
            ],
        ),
    ];
    if avx512_vnni {
        q8_configs.push(vnni4x4);
        q8_configs.push(vnni4x8);
    } else if avx_vnni {
        q8_configs.push(avxvnni256);
    }

    // Lane B. The folded 512-bit inner needs avx512f+bw; the legacy dpbusd
    // inner and the 8-row repack additionally need avx512vnni; the 256-bit
    // inner needs only `vpdpbusd`. Every arm pins its own sub-flag, so which
    // kernel runs is a property of the arm rather than of whichever ran first —
    // the label is still picked from the host's capabilities, and kernels this
    // CPU cannot reach are not measured.
    const KQ_OWNER: &str = "CAMELID_X86_KQUANT_MATMUL_OWNER";
    const KQ_VNNI: &str = "CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI";
    const KQ_AVXVNNI256: &str = "CAMELID_X86_KQUANT_MATMUL_OWNER_AVXVNNI256";
    const KQ_A512VNNI: &str = "CAMELID_X86_KQUANT_MATMUL_OWNER_AVX512VNNI";
    const KQ_REPACK8: &str = "CAMELID_X86_KQUANT_MATMUL_OWNER_REPACK8";
    let kq_off: SweepConfig = ("off", false, &[(KQ_OWNER, "0")]);
    let kq_avx2: SweepConfig = ("owner_avx2", true, &[(KQ_OWNER, "1"), (KQ_VNNI, "0")]);
    let kq_a512fold: SweepConfig = ("owner_avx512fold", true, &[(KQ_OWNER, "1"), (KQ_VNNI, "1")]);
    let kq_vnni512: SweepConfig = (
        "owner_vnni512",
        true,
        &[(KQ_OWNER, "1"), (KQ_VNNI, "1"), (KQ_A512VNNI, "1")],
    );
    let kq_avxvnni: SweepConfig = (
        "owner_avxvnni256",
        true,
        &[(KQ_OWNER, "1"), (KQ_VNNI, "1"), (KQ_AVXVNNI256, "1")],
    );
    // Pins the dpbusd single-row inner too: repack8 only covers
    // floor(out_dim/8)*8 rows, so the ragged tail and any non-repacked tensor
    // fall to the single-row path. Without this the arm would be a mix of
    // repack8 and the folded inner, and would not be comparable with the
    // pre-fold repack8 receipts its label refers to.
    let kq_repack8: SweepConfig = (
        "owner_vnni512_repack8",
        true,
        &[
            (KQ_OWNER, "1"),
            (KQ_VNNI, "1"),
            (KQ_A512VNNI, "1"),
            (KQ_REPACK8, "1"),
        ],
    );
    let mut kquant_configs: Vec<SweepConfig> = vec![kq_off, kq_avx2];
    if avx512_bw {
        kquant_configs.push(kq_a512fold);
    }
    if avx512_vnni {
        kquant_configs.push(kq_vnni512);
        kquant_configs.push(kq_repack8);
    } else if avx_vnni && !avx512_bw {
        kquant_configs.push(kq_avxvnni);
    }
    let configs: Vec<SweepConfig> = match lane {
        Lane::Q8 => q8_configs,
        Lane::KQuant => kquant_configs,
    };
    let configs = configs.as_slice();
    let apply = |envs: &[(&str, &str)]| {
        for k in owner_keys {
            std::env::remove_var(k);
        }
        for (k, v) in envs {
            std::env::set_var(k, v);
        }
    };

    let model_label = model.display().to_string();
    let commit = benchmark_commit();
    let total_rounds = warmup_rounds + rounds;
    eprintln!(
        "[bench-owner-sweep] lane={} {prompt_tokens} prompt tokens, {} configs, {warmup_rounds} warmup + {rounds} measured rounds interleaved",
        match lane {
            Lane::Q8 => "q8",
            Lane::KQuant => "kquant",
        },
        configs.len()
    );
    // Both lanes are capability-aware now, so this applies to either one.
    // Capability disclosure. The folded 512-bit inner exists only in Lane B, so
    // say so only when Lane B is what is being measured — otherwise a Q8 run
    // reports a kernel it never touches.
    if !avx512_vnni {
        if avx512_bw && matches!(lane, Lane::KQuant) {
            eprintln!(
                "[bench-owner-sweep] avx512f/bw without vnni: measuring the folded 512-bit inner; the dpbusd and repack8 arms are unreachable here"
            );
        } else if avx_vnni {
            eprintln!(
                "[bench-owner-sweep] no AVX-512 at all: measuring the 256-bit AVX-VNNI inner in place of the 512-bit arms"
            );
        } else {
            eprintln!("[bench-owner-sweep] no vpdpbusd at all: only the AVX2 inner is measured");
        }
    }
    for round in 0..total_rounds {
        let measured = round >= warmup_rounds;
        for (label, owner_expected, envs) in configs {
            apply(envs);
            camelid::inference::reset_stage_timings();
            camelid::inference::reset_q8_schedule_telemetry();
            let run = generate_run(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                &sampler,
                max_tokens,
            )?;
            // Engaged-check: an owner-on config that never dispatched the
            // owner arm (e.g. env mutation swallowed by a cached plan, or the
            // planner disabled the repack) would measure a fake null. Applies
            // to warmup rounds too — fail fast.
            let telemetry_snapshot = camelid::inference::snapshot_q8_schedule_telemetry();
            let owner_taken = match lane {
                Lane::Q8 => telemetry_snapshot.matmul_owner_prefill_taken,
                Lane::KQuant => telemetry_snapshot.kquant_owner_prefill_taken,
            };
            anyhow::ensure!(
                *owner_expected == (owner_taken > 0),
                "engaged-check failed for config '{label}': owner_expected={owner_expected} \
                 but owner_prefill_taken={owner_taken} — the sweep would mint a fake receipt"
            );
            if !measured {
                continue;
            }
            let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
            let prefill_tok_s = if run.prefill_ms > 0.0 {
                prompt_tokens as f64 / (run.prefill_ms / 1000.0)
            } else {
                0.0
            };
            let decode_tokens = run.generated.len().saturating_sub(1);
            let decode_tok_s = if run.decode_ms > 0.0 && decode_tokens > 0 {
                decode_tokens as f64 / (run.decode_ms / 1000.0)
            } else {
                0.0
            };
            let rec = serde_json::json!({
                "schema": "camelid.bench-owner-sweep/v2",
                "round": round - warmup_rounds,
                "config": label,
                "model": model_label,
                "commit": commit,
                "prompt_tokens": prompt_tokens,
                "prefill_ms": r3(run.prefill_ms),
                "prefill_tok_s": r3(prefill_tok_s),
                "decode_tok_s": r3(decode_tok_s),
                "owner_prefill_taken": owner_taken,
                "q8_avxvnni256_taken": telemetry_snapshot.matmul_owner_avxvnni_taken,
                "kquant_vnni512_taken": telemetry_snapshot.kquant_owner_vnni_taken,
                "kquant_avx512fold_taken": telemetry_snapshot.kquant_owner_avx512fold_taken,
                "kquant_avxvnni256_taken": telemetry_snapshot.kquant_owner_avxvnni_taken,
            });
            println!("{}", serde_json::to_string(&rec)?);
        }
    }
    for k in owner_keys {
        std::env::remove_var(k);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_bench_generate(
    model: PathBuf,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    iterations: usize,
    warmup: bool,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    anyhow::ensure!(iterations >= 1, "--iterations must be at least 1");
    configure_rayon_threads(threads)?;
    camelid::capability::HardwareProfile::detect().log();

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    // Load the model once; this cost is measured separately from generation.
    let load_start = Instant::now();
    let gguf = read_metadata(&model)?;
    if camelid::model::file_requires_runnable_bridge(&gguf) {
        return run_bench_generate_runnable(
            model,
            prompt_text,
            max_tokens,
            temperature,
            iterations,
            warmup,
            gguf,
            load_start,
        );
    }
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::ViaSessionDecode)?;
    // Apply the model's execution plan (as serve/chat do) BEFORE loading weights so the
    // CPU Q8 runtime repack + packed-rows4 fast path is selected at load time. Without
    // this, bench-generate measures the unplanned safe (scalar) path.
    let plan_outcome = camelid::execution_plan::plan_for_model(&model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    let prompt_tokens = prompt_token_ids.len();
    anyhow::ensure!(prompt_tokens >= 1, "prompt encoded to zero tokens");

    let sampler = if temperature <= 0.0 {
        LlamaSampler::Greedy
    } else {
        LlamaSampler::Sampling(SamplingConfig {
            temperature,
            ..Default::default()
        })
    };

    let commit = benchmark_commit();
    let quantization = camelid::receipt::quantization_label(&gguf);
    let model_label = model.display().to_string();

    if warmup {
        eprintln!("[bench-generate] warmup iteration (unmeasured)...");
        let _ = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
    }

    // Drop any warmup/prefill contributions so the dump reflects only measured decode.
    camelid::inference::reset_stage_timings();
    let stdout = std::io::stdout();
    for iteration in 0..iterations {
        let run = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
        let generated_tokens = run.generated.len();
        let decode_tokens = generated_tokens.saturating_sub(1);
        let tokens_per_second = if run.decode_ms > 0.0 && decode_tokens > 0 {
            decode_tokens as f64 / (run.decode_ms / 1000.0)
        } else {
            0.0
        };
        let output_text = tokenizer.decode(&run.generated, true).unwrap_or_default();
        let record = BenchGenerateRecord {
            runtime: "camelid",
            commit: commit.clone(),
            model: model_label.clone(),
            quantization: quantization.clone(),
            iteration,
            prompt_tokens,
            generated_tokens,
            load_ms,
            prefill_ms: run.prefill_ms,
            ttft_ms: run.ttft_ms,
            decode_ms: run.decode_ms,
            tokens_per_second,
            peak_memory_bytes: peak_rss_bytes(),
            offload: camelid::offload::offload_run_status(),
            output_text,
            output_token_ids: run.generated,
        };
        {
            let mut handle = stdout.lock();
            writeln!(handle, "{}", serde_json::to_string(&record)?)?;
            handle.flush()?;
        }
        eprintln!(
            "[bench-generate] iter {} | prompt {} tok | gen {} tok | ttft {:.1} ms | decode {:.1} ms | {:.2} tok/s | peak {:.2} GB",
            iteration,
            prompt_tokens,
            generated_tokens,
            record.ttft_ms,
            record.decode_ms,
            record.tokens_per_second,
            record.peak_memory_bytes as f64 / 1.073_741_824e9,
        );
    }
    // Per-stage CPU decode profile (no-op unless CAMELID_STAGE_TIMINGS=1).
    camelid::inference::dump_stage_timings();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_bench_generate_runnable(
    model: PathBuf,
    prompt_text: String,
    max_tokens: usize,
    temperature: f32,
    iterations: usize,
    warmup: bool,
    gguf: camelid::gguf::GgufFile,
    load_start: Instant,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        temperature <= 0.0,
        "the runnable benchmark bridge currently supports greedy generation only; use --temperature 0"
    );
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let runnable = camelid::runnable::RunnableModel::load(&model.to_string_lossy())?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    let prompt_tokens = prompt_token_ids.len();
    anyhow::ensure!(prompt_tokens >= 1, "prompt encoded to zero tokens");

    eprintln!(
        "[bench-generate] architecture '{}' uses the runnable hybrid graph",
        gguf.architecture().unwrap_or("unknown")
    );
    if warmup {
        eprintln!("[bench-generate] warmup iteration (unmeasured)...");
        let _ = runnable.generate(&prompt_token_ids, max_tokens)?;
    }

    let commit = benchmark_commit();
    let quantization = camelid::receipt::quantization_label(&gguf);
    let model_label = model.display().to_string();
    let stdout = std::io::stdout();
    for iteration in 0..iterations {
        let started = Instant::now();
        let mut first_token_ms = None;
        let generated = runnable.generate_stopping_streaming(
            &prompt_token_ids,
            max_tokens,
            &[],
            &mut |_| {
                if first_token_ms.is_none() {
                    first_token_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                }
            },
        )?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = first_token_ms.unwrap_or(elapsed_ms);
        let decode_ms = (elapsed_ms - ttft_ms).max(0.0);
        let generated_tokens = generated.len();
        let decode_tokens = generated_tokens.saturating_sub(1);
        let tokens_per_second = if decode_ms > 0.0 && decode_tokens > 0 {
            decode_tokens as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let output_text = tokenizer.decode(&generated, true).unwrap_or_default();
        let record = BenchGenerateRecord {
            runtime: "camelid",
            commit: commit.clone(),
            model: model_label.clone(),
            quantization: quantization.clone(),
            iteration,
            prompt_tokens,
            generated_tokens,
            load_ms,
            prefill_ms: ttft_ms,
            ttft_ms,
            decode_ms,
            tokens_per_second,
            peak_memory_bytes: peak_rss_bytes(),
            offload: camelid::offload::offload_run_status(),
            output_text,
            output_token_ids: generated,
        };
        {
            let mut handle = stdout.lock();
            writeln!(handle, "{}", serde_json::to_string(&record)?)?;
            handle.flush()?;
        }
        eprintln!(
            "[bench-generate] iter {} | prompt {} tok | gen {} tok | ttft {:.1} ms | decode {:.1} ms | {:.2} tok/s | peak {:.2} GB",
            iteration,
            prompt_tokens,
            generated_tokens,
            record.ttft_ms,
            record.decode_ms,
            record.tokens_per_second,
            record.peak_memory_bytes as f64 / 1.073_741_824e9,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_bench_generate_vision(
    model: PathBuf,
    mmproj: PathBuf,
    image: PathBuf,
    prompt: String,
    max_tokens: usize,
    image_min_tokens: usize,
    image_max_tokens: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        image_min_tokens > 0 && image_max_tokens >= image_min_tokens,
        "image token bounds must satisfy 0 < min <= max"
    );
    let total_started = Instant::now();
    let gguf = read_metadata(&model)?;
    anyhow::ensure!(
        gguf.architecture() == Some("qwen35"),
        "Prism vision currently requires a qwen35 Bonsai model"
    );
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let load_started = Instant::now();
    let runnable = camelid::runnable::RunnableModel::load(&model.to_string_lossy())?;
    let projector = camelid::runnable::PrismVisionProjector::load(&mmproj)?;
    anyhow::ensure!(
        projector.projection_dim() == 5120,
        "mmproj output width {} is not the Bonsai 27B width 5120",
        projector.projection_dim()
    );
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let vision_started = Instant::now();
    let image_embedding = projector.encode_image(&image, image_min_tokens, image_max_tokens)?;
    let vision_ms = vision_started.elapsed().as_secs_f64() * 1000.0;
    // Single-user-turn rendering of the model's embedded Jinja chat template.
    // The image-pad token is replaced by the projected embedding grid between
    // these two text chunks.
    let prefix = tokenizer.encode("<|im_start|>user\n<|vision_start|>", true, true)?;
    let suffix = tokenizer.encode(
        &format!("<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
        false,
        true,
    )?;
    anyhow::ensure!(
        !prefix.is_empty() && !suffix.is_empty(),
        "vision marker prompt tokenized to an empty sequence"
    );
    let stop: Vec<u32> = tokenizer.special.eog.iter().copied().collect();

    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let generated = runnable.generate_vision_stopping_streaming(
        &prefix,
        &image_embedding,
        &suffix,
        max_tokens,
        &stop,
        &mut |_| {
            if first_token_ms.is_none() {
                first_token_ms = Some(decode_started.elapsed().as_secs_f64() * 1000.0);
            }
        },
    )?;
    let generation_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let output_text = tokenizer.decode(&generated, true)?;
    let record = serde_json::json!({
        "runtime": "camelid-prism-gpu",
        "model": model,
        "mmproj": mmproj,
        "image": image,
        "image_grid": [image_embedding.grid_width, image_embedding.grid_height],
        "image_tokens": image_embedding.embeddings.len(),
        "text_prompt_tokens": prefix.len() + suffix.len(),
        "generated_tokens": generated.len(),
        "load_ms": load_ms,
        "vision_encode_ms": vision_ms,
        "ttft_ms": first_token_ms.unwrap_or(generation_ms),
        "generation_ms": generation_ms,
        "total_ms": total_started.elapsed().as_secs_f64() * 1000.0,
        "peak_memory_bytes": peak_rss_bytes(),
        "output_text": output_text,
        "output_token_ids": generated,
    });
    println!("{}", serde_json::to_string(&record)?);
    eprintln!(
        "[bench-vision] {}x{} image tokens={} | vision {:.1} ms | generation {:.1} ms | peak {:.2} GB",
        image_embedding.grid_width,
        image_embedding.grid_height,
        image_embedding.embeddings.len(),
        vision_ms,
        generation_ms,
        peak_rss_bytes() as f64 / 1.073_741_824e9,
    );
    Ok(())
}

/// One full speculative generation, instrumented for SPEC_RECHECK economics. Mirrors the
/// server's accept/verify/rollback loop (`api::generate`): a normal greedy first step seeds
/// the resident engine, then each round drafts ≤γ tokens, verifies them in ONE batched
/// forward (`verify_drafts_gpu` on the resident GPU, else the CPU chunk verify), accepts the
/// longest confirmed prefix plus the target's own next token, and rolls the rest back. Every
/// emitted token is the target's greedy argmax — lossless by construction. The draft and
/// verify spans are timed separately so f_draft = draft / (draft + verify) is observable.
struct SpeculativeRun {
    generated: Vec<u32>,
    ttft_ms: f64,
    decode_ms: f64,
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
    draft_us: u128,
    /// SPECULATIVE VERIFY time only: the batched verify calls (GPU tree, GPU linear, CPU
    /// chunk) plus failed verify attempts. It must NOT accumulate plain single-token step
    /// time — those go to `normal_step_us`. Conflating the two made `verify_ms` read as
    /// ~100% of `spec_decode_ms` and silently charged plain decode to verify overhead
    /// (BARCHAN Phase 0, amendment A4).
    verify_us: u128,
    /// Plain single-token step time, for the `normal_steps` below. Kept separate from
    /// `verify_us` so the per-round verify cost curve is attributable.
    normal_step_us: u128,
    /// Single-token plain steps taken when the drafter proposed nothing (no n-gram match).
    normal_steps: u64,
    gpu_verify_rounds: u64,
    cpu_verify_rounds: u64,
}

/// Flatten a [`TokenTree`]'s PRIMARY chain (first-child path from the root):
/// for a `branch = 1` drafter the tree IS this chain; for a branching tree it
/// is the drafter's highest-ranked continuation (children are emitted in
/// frequency order). Used by the CPU verify arm, which is strictly
/// linear-causal (no ancestor-masked chunk attention).
fn spec_tree_primary_chain(tree: &camelid::inference::spec_tree::TokenTree) -> Vec<u32> {
    let mut chain = Vec::new();
    let mut current: i32 = 0;
    loop {
        let mut next = None;
        for i in (current as usize + 1)..tree.tokens.len() {
            if tree.parent[i] == current {
                next = Some(i);
                break;
            }
        }
        match next {
            Some(i) => {
                chain.push(tree.tokens[i]);
                current = i as i32;
            }
            None => break,
        }
    }
    chain
}

fn generate_run_speculative(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    drafter: &mut SpeculativeDrafter,
    draft_tokens: usize,
) -> anyhow::Result<SpeculativeRun> {
    let mut session = LlamaInferenceSession::new(config.clone(), weights.clone())?;
    // Keep the target on the resident GPU path so verify_drafts_gpu engages (this mirrors
    // the server with CAMELID_SPEC_GPU on); resident decode is the default when a CUDA
    // device is present, so this is the natural state, asserted explicitly here.
    session.set_resident_paths_disabled(false);
    // The TARGET must not pre-commit encode-ahead graphs in the speculative lane: its next
    // GPU work after a single-token step is a batched VERIFY, never the pre-encoded
    // single-token graph, so the pending graph is always stale waste — and worse, it sits
    // COMMITTED-BUT-UNGATED at the head of Metal's shared serial queue, where a coexisting
    // draft model's next command buffer queues behind it (measured: a multi-second stall on
    // the drafter's first step after the target's TTFT). The drafter's own session keeps
    // encode-ahead ON — its sequential greedy steps are exactly what the pipeline is for,
    // and its pending graphs are pre-signaled (already draining), so they never clog.
    session.set_resident_encode_ahead_enabled(false);

    let mut history: Vec<u32> = prompt_tokens.to_vec();
    let mut input: Vec<u32> = prompt_tokens.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    // TTFT span: prefill + first token. This normal step also seeds the resident engine.
    let ttft_start = Instant::now();
    let step = session.generate_next_token_with_history_diagnostics(
        &input,
        LlamaSampler::Greedy,
        &history,
        false,
        None,
    )?;
    let ttft_ms = ttft_start.elapsed().as_secs_f64() * 1000.0;
    let first = step.next_token_id;
    generated.push(first);
    history.push(first);
    let mut finished = tokenizer.special.eog.contains(&first);
    input.clear();
    input.push(first);

    let mut run = SpeculativeRun {
        generated: Vec::new(),
        ttft_ms,
        decode_ms: 0.0,
        rounds: 0,
        drafted: 0,
        accepted_drafts: 0,
        draft_us: 0,
        verify_us: 0,
        normal_step_us: 0,
        normal_steps: 0,
        gpu_verify_rounds: 0,
        cpu_verify_rounds: 0,
    };

    // Tree-speculation lane (CAMELID_SPEC_TREE): a branching drafter proposes a tree of
    // candidate continuations and one batched forward (`verify_tree_gpu`) confirms whichever
    // branch the model takes. Lossless (every emitted token is the target's greedy argmax).
    // Default off; falls back to the linear path per-round if the GPU engine isn't ready.
    let spec_tree = std::env::var_os("CAMELID_SPEC_TREE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let mut tree_drafter = camelid::inference::suffix_decoding::SuffixDecodingDrafter::default();
    // ACCEPTANCE-GATED DRAFTING (CAMELID_SPEC_TREE lane only).
    //
    // The suffix drafter only PROPOSES; `verify_tree_gpu` is the exact greedy gate, so any
    // budget we pick here stays lossless. The PROBLEM it solves: a wide tree costs a batched
    // verify + KV compaction every round regardless of how many tokens land. On low-acceptance
    // workloads (prose) most branches reject, so the wide-tree overhead exceeds the ~1 token it
    // commits and the spec lane runs SLOWER than plain decode. On high-acceptance workloads
    // (repetitive) a wide tree commits many tokens per weight read and wins big.
    //
    // Fix: gate the tree budget by RECENT acceptance (a run-length latch over accepted DRAFT tokens
    // per round; the +1 bonus is free and excluded). MEASURED full-tree acceptance on this box
    // (3B Q8, RTX 3060 6GB) cleanly separates the workloads by their net speedup:
    //   repetitive ~2.6 accepted/round -> S_sync ~1.28x (clear win)
    //   code       ~1.2               -> ~0.87x (regress)
    //   json       ~0.9               -> ~0.83x (regress)
    //   prose      ~0.5               -> ~0.76x (regress)
    // The batched-verify + KV-compaction per round only pays off at HIGH acceptance, so the policy
    // is binary: speculate (full tree every round) on the repetitive stream, SKIP (plain decode,
    // ~1.0x) on everything else. Two design points make it robust:
    //  (1) RUN-LENGTH latch (see below) keeps a speculating stream latched ON through isolated low
    //      rounds — only a RUN of consecutive non-productive rounds turns it off — so repetitive's
    //      bursty per-round variance doesn't bleed away the win via stray skips.
    //  (2) Acceptance is always measured on the SAME full tree the latch uses (warm-up and the
    //      periodic re-probe draw the full tree, never a throttled one) — a smaller probe tree would
    //      CAP how many drafts can be accepted and under-read repetitive's true ~2.6 into code's
    //      range, collapsing the gate into never speculating.
    // Thresholds are deliberately simple and collected here so they are easy to find and tune.
    //
    // The latch is RUN-LENGTH based, not a noisy per-round EWMA threshold: real-text acceptance is
    // bursty (a repetitive list still has occasional 0-accept rounds), so an EWMA Schmitt-trigger
    // flips the win off mid-stream. Instead:
    //   - While speculating, draw the full tree EVERY round (identical to the ungated path). Stay
    //     latched ON until EXIT_RUN *consecutive* rounds each accept fewer than PRODUCTIVE_DRAFTS
    //     drafts. One good round resets the run, so repetitive (which keeps landing multi-token
    //     accepts) never trips the exit; prose/code (which consistently accept ~0-1) trip it fast.
    //   - While latched OFF, SKIP (plain decode, ~1.0x). Every LOW_REPROBE skips, spend ONE
    //     full-tree probe; if it lands >= ENTER_DRAFTS accepted, re-latch ON (a stream that turned
    //     repetitive recovers). The probe is rare, so a novel stream pays ~1 wasted verify / 64 tok.
    // The latch itself now lives in `speculative::SpecLatch` (STAMPEDE P5.2) so
    // the GPU-verified and CPU-verified rounds — and, staged, the serve loop —
    // drive ONE policy. The measured constants (2/4/2/1/64) are its defaults.
    // Escape hatch for A/B measurement: CAMELID_SPEC_TREE_GATE=0 forces the OLD ungated policy
    // (full tree every round, never skip) so the gated-vs-ungated S_sync can be measured from the
    // SAME binary. Default ON (gated). The gate only changes which budget the drafter PROPOSES;
    // losslessness is the verify's job either way.
    let gate_enabled = std::env::var_os("CAMELID_SPEC_TREE_GATE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut latch = camelid::inference::speculative::SpecLatch::default();
    // STAMPEDE Phase 5 (P5.1): when the resident GPU verify is unavailable
    // (CPU-only box, CUDA hidden, resident decode off), verify the drafted
    // chain on the CPU via the batched chunk forward + KV rollback — the same
    // shipped pattern the linear lane below uses. Kill-switch:
    // CAMELID_SPEC_CPU_VERIFY=0 restores the old skip-to-plain behavior.
    let cpu_verify_allowed = std::env::var_os("CAMELID_SPEC_CPU_VERIFY")
        .map(|v| v != "0")
        .unwrap_or(true);
    // One-way ratchet: after the first CPU-verified round the session is
    // pinned off the resident paths (the chunk-verify rollback requires
    // CPU-authoritative KV, and `rollback_to_position` drops the resident
    // engine anyway — never alternate modes mid-run).
    let mut cpu_verify_pinned = false;

    let decode_start = Instant::now();
    while !finished && generated.len() < max_tokens {
        let remaining = max_tokens.saturating_sub(generated.len());
        let context_room = session.remaining_context();
        let budget = draft_tokens
            .min(remaining.saturating_sub(1))
            .min(context_room.saturating_sub(1));

        // Tree round: draft a branching tree and verify it in one batched forward.
        if spec_tree && budget > 0 && context_room > 0 {
            use camelid::inference::spec_tree::{TreeDrafter, TREE_MAX_NODES};
            let anchor = input[0];

            // Choose this round's tree budget from the run-length latch (the gate). Returns None to
            // SKIP speculation: take a plain greedy step instead. The FULL tree is the original
            // ungated budget: a gamma-deep chain (the suffix drafter may branch within the node
            // cap). Every band that speculates draws this same full tree so acceptance is measured
            // at the size the latched-ON band actually uses (a throttled probe would under-read it).
            let full_tree = ((budget + 1).min(TREE_MAX_NODES), budget);
            // Warm-up / latched-ON / re-probe rounds all draw the SAME full
            // tree (acceptance must be measured at the size the latched-ON
            // band uses); latched-OFF rounds skip speculation entirely.
            let chosen_budget: Option<(usize, usize)> = if !gate_enabled {
                // Ungated baseline (A/B): the original always-full-tree policy.
                Some(full_tree)
            } else if latch.should_speculate() {
                Some(full_tree)
            } else {
                None
            };

            if std::env::var_os("CAMELID_SPEC_TREE_TRACE").is_some() {
                eprintln!(
                    "[spec-tree] round_seen={} spec={} nonprod_run={} skips={} budget={} -> {:?}",
                    latch.rounds_done(),
                    latch.speculating(),
                    latch.nonproductive_run(),
                    latch.consecutive_skips(),
                    budget,
                    chosen_budget
                );
            }
            let Some((max_nodes, max_depth)) = chosen_budget else {
                // Latched OFF: one plain resident greedy step (no speculation this round). Recovery
                // is handled by the periodic full-tree re-probe; no speculation cost is paid here.
                latch.note_skip();
                let step_started = Instant::now();
                let next = match session.generate_next_token_greedy_resident(input[0])? {
                    Some((id, _us)) => id,
                    None => {
                        session
                            .generate_next_token_with_history_diagnostics(
                                &input,
                                LlamaSampler::Greedy,
                                &history,
                                false,
                                None,
                            )?
                            .next_token_id
                    }
                };
                run.normal_step_us += step_started.elapsed().as_micros();
                run.normal_steps += 1;
                generated.push(next);
                history.push(next);
                if tokenizer.special.eog.contains(&next) {
                    finished = true;
                }
                input.clear();
                input.push(*generated.last().expect("just pushed a token"));
                continue;
            };

            let draft_started = Instant::now();
            let tree = tree_drafter.draft_tree(&history, anchor, max_nodes, max_depth);
            run.draft_us += draft_started.elapsed().as_micros();
            if tree.nodes() > 1 {
                let verify_started = Instant::now();
                let gpu_emitted = if cpu_verify_pinned {
                    None
                } else {
                    session.verify_tree_gpu(&tree)?
                };
                if let Some(emitted) = gpu_emitted {
                    run.verify_us += verify_started.elapsed().as_micros();
                    // A verified round drives the run-length latch. accepted_drafts = emitted minus
                    // the guaranteed +1 bonus.
                    let accepted_drafts = (emitted.len() as u64).saturating_sub(1) as u32;
                    latch.note_verified(accepted_drafts);
                    run.gpu_verify_rounds += 1;
                    run.rounds += 1;
                    run.drafted += (tree.nodes() - 1) as u64;
                    run.accepted_drafts += (emitted.len() as u64).saturating_sub(1);
                    for token in emitted {
                        if generated.len() >= max_tokens {
                            break;
                        }
                        generated.push(token);
                        history.push(token);
                        if tokenizer.special.eog.contains(&token) {
                            finished = true;
                            break;
                        }
                    }
                    input.clear();
                    input.push(*generated.last().expect("a tree round emits >=1 token"));
                    continue;
                }
                // STAMPEDE Phase 5 (P5.1): resident GPU verify unavailable —
                // verify the tree's PRIMARY CHAIN on the CPU via the batched
                // chunk forward + KV rollback (the linear lane's shipped
                // pattern). Lossless: every emitted token is the target's own
                // greedy argmax given the accepted prefix.
                if cpu_verify_allowed {
                    let chain = spec_tree_primary_chain(&tree);
                    if !chain.is_empty() {
                        if !cpu_verify_pinned {
                            // One-way ratchet: pin the session off the resident
                            // paths (rollback requires CPU-authoritative KV) and
                            // switch the drafter to linear chains — deeper
                            // proposals within the same node budget, and the
                            // primary-chain flatten becomes exact.
                            cpu_verify_pinned = true;
                            session.set_resident_paths_disabled(true);
                            tree_drafter.branch = 1;
                        }
                        let base_position = session.kv_position();
                        let mut batch = Vec::with_capacity(1 + chain.len());
                        batch.push(anchor);
                        batch.extend_from_slice(&chain);
                        let (predictions, verify_timings) =
                            session.forward_greedy_verify_chunk(&batch)?;
                        // Small-M verify economics profiling (STAMPEDE P5
                        // follow-up): component split of the chunk forward.
                        if std::env::var_os("CAMELID_SPEC_VERIFY_TIMINGS").is_some() {
                            let mut sums = [0u128; 15];
                            for l in &verify_timings.layers {
                                for (slot, v) in sums.iter_mut().zip([
                                    l.attention_norm,
                                    l.attention_q,
                                    l.attention_k,
                                    l.attention_v,
                                    l.attention_rope,
                                    l.kv_cache_write,
                                    l.attention_context,
                                    l.attention_output,
                                    l.attention_residual,
                                    l.ffn_norm,
                                    l.ffn_gate,
                                    l.ffn_up,
                                    l.ffn_activation,
                                    l.ffn_down,
                                    l.ffn_residual,
                                ]) {
                                    *slot += v;
                                }
                            }
                            eprintln!(
                                "[spec-verify] rows={} layers_us={} logits_us={} | anorm={} q={} k={} v={} rope={} kvw={} actx={} aout={} ares={} fnorm={} gate={} up={} act={} down={} fres={}",
                                batch.len(),
                                verify_timings.layers_total,
                                verify_timings.logits,
                                sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6],
                                sums[7], sums[8], sums[9], sums[10], sums[11], sums[12], sums[13],
                                sums[14]
                            );
                        }
                        let accepted = accepted_draft_prefix(&chain, &predictions);
                        session.rollback_to_position(base_position + 1 + accepted)?;
                        run.verify_us += verify_started.elapsed().as_micros();
                        latch.note_verified(accepted as u32);
                        run.cpu_verify_rounds += 1;
                        run.rounds += 1;
                        run.drafted += chain.len() as u64;
                        run.accepted_drafts += accepted as u64;
                        for token in &predictions[..=accepted] {
                            if generated.len() >= max_tokens {
                                break;
                            }
                            generated.push(*token);
                            history.push(*token);
                            if tokenizer.special.eog.contains(token) {
                                finished = true;
                                break;
                            }
                        }
                        input.clear();
                        input.push(*generated.last().expect("a verify round emits >=1 token"));
                        continue;
                    }
                }
                // Engine not ready and CPU verify disabled/empty-chain: fall through to a plain
                // step. Don't score this as a low-acceptance round — it's an engine-readiness
                // miss, not a drafter miss — so leave the latch untouched.
                run.verify_us += verify_started.elapsed().as_micros();
            } else {
                // The drafter found NO recurrence (anchor-only tree). This is TRANSIENT and common
                // early in a stream (the recurrence hasn't built up yet), so it must NOT crash the
                // EWMA into the LOW band before the stream's true acceptance is ever observed. Treat
                // it exactly like the ungated path does: take a cheap plain step and try again next
                // round. The suffix scan that produced it is O(window) and cheap, and crucially NO
                // batched verify or KV compaction ran (the expensive part) — so on purely novel
                // text this costs essentially the same as plain decode. Leave the EWMA untouched.
            }
            // No usable tree this round → one plain resident greedy step.
            let step_started = Instant::now();
            let next = match session.generate_next_token_greedy_resident(input[0])? {
                Some((id, _us)) => id,
                None => {
                    session
                        .generate_next_token_with_history_diagnostics(
                            &input,
                            LlamaSampler::Greedy,
                            &history,
                            false,
                            None,
                        )?
                        .next_token_id
                }
            };
            run.normal_step_us += step_started.elapsed().as_micros();
            run.normal_steps += 1;
            generated.push(next);
            history.push(next);
            if tokenizer.special.eog.contains(&next) {
                finished = true;
            }
            input.clear();
            input.push(*generated.last().expect("just pushed a token"));
            continue;
        }

        let draft_started = Instant::now();
        let drafts = if budget > 0 && context_room > 0 {
            drafter.draft(&history, budget)?
        } else {
            Vec::new()
        };
        run.draft_us += draft_started.elapsed().as_micros();

        if drafts.is_empty() {
            // No draft proposed → one plain resident greedy step (same path the plain
            // baseline takes), so a config that never drafts measures as plain decode.
            let step_started = Instant::now();
            let next = match session.generate_next_token_greedy_resident(input[0])? {
                Some((id, _us)) => id,
                None => {
                    session
                        .generate_next_token_with_history_diagnostics(
                            &input,
                            LlamaSampler::Greedy,
                            &history,
                            false,
                            None,
                        )?
                        .next_token_id
                }
            };
            run.normal_step_us += step_started.elapsed().as_micros();
            run.normal_steps += 1;
            generated.push(next);
            history.push(next);
            if tokenizer.special.eog.contains(&next) {
                finished = true;
            }
            input.clear();
            input.push(*generated.last().expect("just pushed a token"));
            continue;
        }

        // Verify all drafts in one batched forward: GPU resident when ready, else the CPU
        // chunk verify with an explicit KV rollback. Both are lossless — the emitted tokens
        // are the target's own greedy argmax given the accepted prefix.
        let verify_started = Instant::now();
        let emitted: Vec<u32> = match session.verify_drafts_gpu(input[0], &drafts)? {
            Some(accepted) => {
                run.gpu_verify_rounds += 1;
                run.rounds += 1;
                run.drafted += drafts.len() as u64;
                run.accepted_drafts += (accepted.len() as u64).saturating_sub(1);
                accepted
            }
            None => {
                let base_position = session.kv_position();
                let mut batch = Vec::with_capacity(1 + drafts.len());
                batch.push(input[0]);
                batch.extend_from_slice(&drafts);
                let (predictions, _timings) = session.forward_greedy_verify_chunk(&batch)?;
                let accepted = accepted_draft_prefix(&drafts, &predictions);
                session.rollback_to_position(base_position + 1 + accepted)?;
                run.cpu_verify_rounds += 1;
                run.rounds += 1;
                run.drafted += drafts.len() as u64;
                run.accepted_drafts += accepted as u64;
                predictions[..=accepted].to_vec()
            }
        };
        run.verify_us += verify_started.elapsed().as_micros();

        for token in emitted {
            if generated.len() >= max_tokens {
                break;
            }
            generated.push(token);
            history.push(token);
            if tokenizer.special.eog.contains(&token) {
                finished = true;
                break;
            }
        }
        input.clear();
        input.push(*generated.last().expect("a round emits at least one token"));
    }
    run.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    run.generated = generated;
    Ok(run)
}

#[derive(Serialize)]
struct BenchSpeculativeRecord {
    runtime: &'static str,
    commit: String,
    camelid_version: String,
    binary_sha256: String,
    workload: String,
    input_sha256: String,
    prompt_sha256: String,
    prompt_format: &'static str,
    add_bos: bool,
    add_eos: bool,
    parse_special: bool,
    model: String,
    model_sha256: String,
    tokenizer_metadata_sha256: Option<String>,
    draft_model: Option<String>,
    draft_model_sha256: Option<String>,
    quantization: String,
    drafter: String,
    cpu_draft: bool,
    /// True when --spec-only: the plain fields below are the coexistence-config target (draft
    /// resident), NOT a full-resident-target baseline. S_sync here is speculation efficiency on
    /// the same target; the full-resident denominator comes from the n-gram/bench-generate runs.
    spec_only: bool,
    draft_tokens: usize,
    prompt_tokens: usize,
    max_tokens: usize,

    // Plain greedy baseline (this run, same target, same machine).
    plain_generated_tokens: usize,
    plain_ttft_ms: f64,
    plain_decode_ms: f64,
    plain_tokens_per_second: f64,

    // Speculative run.
    spec_generated_tokens: usize,
    spec_ttft_ms: f64,
    spec_decode_ms: f64,
    spec_tokens_per_second: f64,

    // Economics.
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
    accept_rate: f64,
    mean_accepted_tokens_per_round: f64,
    draft_ms: f64,
    /// SPECULATIVE VERIFY time only — batched verify calls plus failed verify attempts.
    /// Plain single-token step time is reported separately in `normal_step_ms`; before
    /// BARCHAN Phase 0 the two were summed here, which made this field read as ~100% of
    /// `spec_decode_ms` and made any per-round verify cost derived from it wrong.
    verify_ms: f64,
    /// Plain single-token step time for the `normal_steps` rounds (drafter proposed
    /// nothing, latch skipped, or the engine was not ready). Not speculation overhead.
    normal_step_ms: f64,
    /// draft / (draft + verify): the fraction of round time spent drafting. The Phase-4
    /// decision gate turns on this — ~0 for n-gram (nothing to hide with concurrency).
    f_draft: f64,
    /// Synchronous speedup over this machine's own plain greedy decode (spec t/s ÷ plain t/s).
    s_sync: f64,
    normal_steps: u64,
    gpu_verify_rounds: u64,
    cpu_verify_rounds: u64,

    // Lossless gate (intra-Camelid: spec stream vs this run's plain greedy stream).
    first_divergent_generated_token_index: i64,
    lossless: bool,
    plain_token_ids: Vec<u32>,
    spec_token_ids: Vec<u32>,

    metal_device: Option<String>,
    host_isa: String,
    effective_env: BTreeMap<String, Option<String>>,
    planner_env_updates: BTreeMap<String, Option<String>>,
    execution_plan: camelid::execution_plan::ExecutionPlan,

    peak_memory_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    offload: Option<camelid::offload::OffloadRunStatus>,
}

fn speculative_effective_env() -> BTreeMap<String, Option<String>> {
    const EXTRA_KEYS: &[&str] = &[
        "CAMELID_SPEC_GPU",
        "CAMELID_SPEC_DECODE",
        "CAMELID_SPEC_TREE_GATE",
        "CAMELID_SPEC_NGRAM_MIN",
        "CAMELID_METAL_ATTN_SPLITK",
        "CAMELID_METAL_KV_DTYPE",
        "CAMELID_KQUANT_MC_GEMV",
    ];
    let mut values = eagle3_effective_env();
    values.extend(
        EXTRA_KEYS
            .iter()
            .map(|key| ((*key).to_string(), std::env::var(key).ok())),
    );
    values
}

/// Load a draft GGUF and wrap it as a `ModelDrafter`. Mirrors the target load path so the
/// draft rides the same execution plan; the drafter routes to its own resident cache.
fn load_model_drafter(
    path: &std::path::Path,
    target_tokenizer: &Tokenizer,
    cpu_draft: bool,
    threads: Option<usize>,
) -> anyhow::Result<SpeculativeDrafter> {
    let gguf = read_metadata(path)?;
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::CpuDenseOnly)?;
    let plan_outcome = camelid::execution_plan::plan_for_model(path, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(path, &gguf);
    let draft_tokenizer = Tokenizer::from_gguf(&gguf)?;
    // Drafted token ids must mean the same text in the target vocabulary. Lossless either
    // way (the verify is authoritative), but a mismatched vocab silently drives accept to ~0.
    anyhow::ensure!(
        draft_tokenizer.model == target_tokenizer.model,
        "draft model tokenizer ({:?}) differs from target ({:?}); drafted ids would not share \
         the target vocabulary",
        draft_tokenizer.model,
        target_tokenizer.model
    );
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);
    let mut session = LlamaInferenceSession::new(config, weights)?;
    if cpu_draft {
        // Path 3 (SPEC_RECHECK): force the draft onto the CPU forward (the previously
        // "blocked" configuration). Otherwise the draft stays GPU-resident by default.
        session.set_resident_paths_disabled(true);
    }
    Ok(SpeculativeDrafter::Model(Box::new(ModelDrafter::new(
        session,
    ))))
}

#[allow(clippy::too_many_arguments)]
fn run_bench_speculative(
    model: PathBuf,
    drafter_kind: String,
    draft_model: Option<PathBuf>,
    draft_tokens: Option<usize>,
    cpu_draft: bool,
    spec_only: bool,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    workload: String,
    max_tokens: usize,
    warmup: bool,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    anyhow::ensure!(max_tokens >= 1, "--max-tokens must be at least 1");
    configure_rayon_threads(threads)?;
    camelid::capability::HardwareProfile::detect().log();

    let prompt_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    let current_exe = std::env::current_exe()?;
    let binary_sha256 = camelid::receipt::sha256_file_hex(&current_exe)
        .map_err(|error| anyhow::anyhow!("hashing benchmark binary: {error}"))?;
    let model_sha256 = camelid::receipt::sha256_file_hex_cached(&model)
        .map_err(|error| anyhow::anyhow!("hashing target {}: {error}", model.display()))?;
    let draft_model_sha256 = draft_model
        .as_deref()
        .map(camelid::receipt::sha256_file_hex_cached)
        .transpose()
        .map_err(|error| anyhow::anyhow!("hashing draft model: {error}"))?;

    // Load the target exactly as bench-generate does (execution plan applied before weights).
    let gguf = read_metadata(&model)?;
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::CpuDenseOnly)?;
    let plan_outcome = camelid::execution_plan::plan_for_model(&model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let planner_env_updates = plan_outcome
        .env_updates
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.map(str::to_string)))
        .collect();
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);

    let prompt_token_ids = tokenizer.encode(&prompt_text, true, false)?;
    let prompt_tokens = prompt_token_ids.len();
    anyhow::ensure!(prompt_tokens >= 1, "prompt encoded to zero tokens");

    let gamma = draft_tokens.unwrap_or(match drafter_kind.as_str() {
        "draft" => DEFAULT_MODEL_DRAFT_TOKENS,
        _ => DEFAULT_NGRAM_DRAFT_TOKENS,
    });

    let build_drafter = || -> anyhow::Result<SpeculativeDrafter> {
        match drafter_kind.as_str() {
            "ngram" => Ok(SpeculativeDrafter::NGram(NGramDrafter::default())),
            // Suffix drafting flattened to a chain: fills the verify window the
            // n-gram drafter leaves mostly empty, without paying the tree
            // verify's per-round cost.
            "suffix" => Ok(SpeculativeDrafter::Suffix(Box::new(
                camelid::inference::suffix_decoding::SuffixDecodingDrafter::default(),
            ))),
            "draft" => {
                let path = draft_model.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("--drafter draft requires --draft-model <gguf>")
                })?;
                load_model_drafter(path, &tokenizer, cpu_draft, threads)
            }
            other => anyhow::bail!(
                "unknown --drafter {other:?}; expected \"ngram\", \"suffix\" or \"draft\""
            ),
        }
    };

    let sampler = LlamaSampler::Greedy;

    // Two orderings. Default: the plain baseline runs FIRST on a full-resident target (the truest
    // S_sync denominator), then the drafter is added. spec_only: the drafter (and its coexistence
    // reserve) is established BEFORE any target build, so the target builds once under the
    // coexistence budget and the draft stays GPU-resident; the plain reference then reuses that
    // same resident target (so its tps is the coexistence-config target, flagged in the record).
    let (plain, spec, draft_stats) = if spec_only {
        if warmup {
            eprintln!("[bench-speculative] warmup (unmeasured, spec-only)...");
            let mut w = build_drafter()?;
            let _ = generate_run_speculative(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                max_tokens,
                &mut w,
                gamma,
            )?;
        }
        camelid::inference::reset_stage_timings();
        let mut drafter = build_drafter()?;
        let spec = generate_run_speculative(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            max_tokens,
            &mut drafter,
            gamma,
        )?;
        let draft_stats = drafter.take_forward_stats();
        // Plain reference reuses the resident coexistence target engine (no rebuild).
        let plain = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
        (plain, spec, draft_stats)
    } else {
        if warmup {
            eprintln!("[bench-speculative] warmup (unmeasured)...");
            let _ = generate_run(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                &sampler,
                max_tokens,
            )?;
            let mut warm = build_drafter()?;
            let _ = generate_run_speculative(
                &config,
                &weights,
                &tokenizer,
                &prompt_token_ids,
                max_tokens,
                &mut warm,
                gamma,
            )?;
        }
        camelid::inference::reset_stage_timings();
        // Single-model baseline: clear any reserve a warmup drafter set so the denominator is a
        // full-resident target, not one that left room for a draft.
        camelid::inference::set_spec_coexist_reserve(0);
        let plain = generate_run(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            &sampler,
            max_tokens,
        )?;
        let mut drafter = build_drafter()?;
        let spec = generate_run_speculative(
            &config,
            &weights,
            &tokenizer,
            &prompt_token_ids,
            max_tokens,
            &mut drafter,
            gamma,
        )?;
        let draft_stats = drafter.take_forward_stats();
        (plain, spec, draft_stats)
    };
    let (draft_fwd_us, draft_resident_steps, draft_cpu_steps, draft_max_step_us) = draft_stats;
    let plain_decode_tokens = plain.generated.len().saturating_sub(1);
    let plain_tps = if plain.decode_ms > 0.0 && plain_decode_tokens > 0 {
        plain_decode_tokens as f64 / (plain.decode_ms / 1000.0)
    } else {
        0.0
    };
    let spec_decode_tokens = spec.generated.len().saturating_sub(1);
    let spec_tps = if spec.decode_ms > 0.0 && spec_decode_tokens > 0 {
        spec_decode_tokens as f64 / (spec.decode_ms / 1000.0)
    } else {
        0.0
    };
    // Draft-decode profiling: the GPU forward time of the draft steps vs the wall-clock draft
    // time tells whether the draft cost is in the forward kernels or in sync/overhead around them.
    // Mean AND max/steady: a lazily-paid one-time cost (engine build, first-touch paging) lands
    // in ONE step, and a bare mean smears it into what reads as uniform per-step slowness.
    if draft_resident_steps + draft_cpu_steps > 0 {
        let steady_ms = if draft_resident_steps > 1 {
            (draft_fwd_us.saturating_sub(draft_max_step_us)) as f64
                / 1000.0
                / (draft_resident_steps - 1) as f64
        } else {
            draft_fwd_us as f64 / 1000.0
        };
        eprintln!(
            "[draft-profile] resident steps {} ({:.1} ms/step GPU forward; max {:.1} ms, steady {:.1} ms/step) | \
             cpu-fallback steps {} | wall draft {:.1} ms total = {:.1} ms/step | GPU-forward fraction {:.0}%",
            draft_resident_steps,
            if draft_resident_steps > 0 {
                draft_fwd_us as f64 / 1000.0 / draft_resident_steps as f64
            } else {
                0.0
            },
            draft_max_step_us as f64 / 1000.0,
            steady_ms,
            draft_cpu_steps,
            spec.draft_us as f64 / 1000.0,
            if draft_resident_steps + draft_cpu_steps > 0 {
                spec.draft_us as f64 / 1000.0 / (draft_resident_steps + draft_cpu_steps) as f64
            } else {
                0.0
            },
            if spec.draft_us > 0 {
                draft_fwd_us as f64 / spec.draft_us as f64 * 100.0
            } else {
                0.0
            },
        );
    }

    // Lossless gate: first index where the spec stream diverges from plain greedy (-1 if the
    // two streams are identical). A positive cell with any divergence is a correctness bug.
    let first_divergent = first_divergence(&spec.generated, &plain.generated);

    let accept_rate = if spec.drafted > 0 {
        spec.accepted_drafts as f64 / spec.drafted as f64
    } else {
        0.0
    };
    // Each verify round emits accepted drafts + 1 bonus token.
    let mean_accepted_tokens_per_round = if spec.rounds > 0 {
        (spec.accepted_drafts + spec.rounds) as f64 / spec.rounds as f64
    } else {
        0.0
    };
    let draft_ms = spec.draft_us as f64 / 1000.0;
    let verify_ms = spec.verify_us as f64 / 1000.0;
    let normal_step_ms = spec.normal_step_us as f64 / 1000.0;
    let f_draft = if draft_ms + verify_ms > 0.0 {
        draft_ms / (draft_ms + verify_ms)
    } else {
        0.0
    };
    let s_sync = if plain_tps > 0.0 {
        spec_tps / plain_tps
    } else {
        0.0
    };

    let record = BenchSpeculativeRecord {
        runtime: "camelid",
        commit: benchmark_commit(),
        camelid_version: camelid::receipt::camelid_version(),
        binary_sha256,
        workload,
        input_sha256: camelid::receipt::sha256_hex(prompt_text.as_bytes()),
        prompt_sha256: camelid::receipt::sha256_hex(prompt_text.as_bytes()),
        prompt_format: "raw_completion_bos_no_eos",
        add_bos: true,
        add_eos: false,
        parse_special: false,
        model: model.display().to_string(),
        model_sha256,
        tokenizer_metadata_sha256: camelid::receipt::tokenizer_metadata_sha256(&gguf),
        draft_model: draft_model.as_ref().map(|p| p.display().to_string()),
        draft_model_sha256,
        quantization: camelid::receipt::quantization_label(&gguf),
        drafter: drafter_kind,
        cpu_draft,
        spec_only,
        draft_tokens: gamma,
        prompt_tokens,
        max_tokens,
        plain_generated_tokens: plain.generated.len(),
        plain_ttft_ms: plain.ttft_ms,
        plain_decode_ms: plain.decode_ms,
        plain_tokens_per_second: plain_tps,
        spec_generated_tokens: spec.generated.len(),
        spec_ttft_ms: spec.ttft_ms,
        spec_decode_ms: spec.decode_ms,
        spec_tokens_per_second: spec_tps,
        rounds: spec.rounds,
        drafted: spec.drafted,
        accepted_drafts: spec.accepted_drafts,
        accept_rate,
        mean_accepted_tokens_per_round,
        draft_ms,
        verify_ms,
        normal_step_ms,
        f_draft,
        s_sync,
        normal_steps: spec.normal_steps,
        gpu_verify_rounds: spec.gpu_verify_rounds,
        cpu_verify_rounds: spec.cpu_verify_rounds,
        first_divergent_generated_token_index: first_divergent,
        lossless: first_divergent < 0,
        plain_token_ids: plain.generated,
        spec_token_ids: spec.generated,
        metal_device: camelid::metal::detect_metal_device().device_name,
        host_isa: camelid::receipt::host_isa_marker(),
        effective_env: speculative_effective_env(),
        planner_env_updates,
        execution_plan: plan_outcome.plan,
        peak_memory_bytes: peak_rss_bytes(),
        offload: camelid::offload::offload_run_status(),
    };

    let stdout = std::io::stdout();
    {
        let mut handle = stdout.lock();
        writeln!(handle, "{}", serde_json::to_string(&record)?)?;
        handle.flush()?;
    }
    eprintln!(
        "[bench-speculative] {} | {} γ={}{} | accept {:.1}% | tok/round {:.2} | f_draft {:.3} | \
         draft {:.1} ms/tok | plain {:.2} t/s → spec {:.2} t/s | S_sync {:.2}x | {} | gpu/cpu verify {}/{} | drafted {} rounds {}",
        record.workload,
        record.drafter,
        record.draft_tokens,
        if record.spec_only { " spec-only" } else { "" },
        record.accept_rate * 100.0,
        record.mean_accepted_tokens_per_round,
        record.f_draft,
        if record.drafted > 0 { record.draft_ms / record.drafted as f64 } else { 0.0 },
        record.plain_tokens_per_second,
        record.spec_tokens_per_second,
        record.s_sync,
        if record.lossless {
            "LOSSLESS ✓".to_string()
        } else {
            format!("DIVERGED @ {}", record.first_divergent_generated_token_index)
        },
        record.gpu_verify_rounds,
        record.cpu_verify_rounds,
        record.drafted,
        record.rounds,
    );
    Ok(())
}

#[derive(Default)]
struct Eagle3BenchRun {
    generated: Vec<u32>,
    drafted_token_ids: Vec<u32>,
    ttft_ms: f64,
    decode_ms: f64,
    head_upload_ms: f64,
    head_seed_ms: f64,
    bootstrap_capture_ms: f64,
    draft_us: u128,
    verify_us: u128,
    head_update_us: u128,
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
    verify_nodes: u64,
    resident_verify_rounds: u64,
    cpu_verify_rounds: u64,
    resident_normal_steps: u64,
    suffix_rounds: u64,
    suffix_offered: u64,
    suffix_emitted_tokens: u64,
    suffix_head_catchups: u64,
    suffix_head_catchup_rows: u64,
    suffix_head_discarded_rows: u64,
    suffix_head_buffer_us: u128,
    dynamic_tree_rounds: u64,
    dynamic_tree_offered: u64,
    dynamic_tree_emitted_tokens: u64,
    materialized_head_forwards: u64,
    dynamic_tree_max_depth_sum: u64,
}

/// Flatten the first deepest suffix-tree branch, bounded by the verify depth.
/// Suffix children are inserted in descending frequency order, so choosing the
/// earliest node at a tied depth preserves the drafter's deterministic ranking.
fn deepest_suffix_chain(
    tree: &camelid::inference::spec_tree::TokenTree,
    max_depth: usize,
) -> Vec<u32> {
    let mut leaf = 0usize;
    for (node, &depth) in tree.depth.iter().enumerate().skip(1) {
        let depth = depth as usize;
        if depth <= max_depth && depth > tree.depth[leaf] as usize {
            leaf = node;
        }
    }
    if leaf == 0 {
        return Vec::new();
    }
    tree.path_to(leaf)
        .into_iter()
        .skip(1)
        .map(|node| tree.tokens[node])
        .collect()
}

fn run_plain_resident_greedy(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> anyhow::Result<Eagle3BenchRun> {
    let mut session = LlamaInferenceSession::new(config.clone(), Arc::clone(weights))?;
    // Populate the shared resident weight cache outside the measured decode span. This is a
    // model-load cost, not a per-token cost, and the EAGLE run below reuses the same cache.
    let _ = session.prewarm_resident_weights();
    let ttft_started = Instant::now();
    let first = session
        .generate_next_token_with_history_diagnostics(
            prompt_tokens,
            LlamaSampler::Greedy,
            prompt_tokens,
            false,
            None,
        )?
        .next_token_id;
    let ttft_ms = ttft_started.elapsed().as_secs_f64() * 1000.0;
    let mut generated = vec![first];
    let mut resident_normal_steps = 0;
    let decode_started = Instant::now();
    while generated.len() < max_tokens && !tokenizer.special.eog.contains(generated.last().unwrap())
    {
        let anchor = *generated.last().expect("generated is seeded");
        let next = match session.generate_next_token_greedy_resident(anchor)? {
            Some((token, _)) => {
                resident_normal_steps += 1;
                token
            }
            None => anyhow::bail!(
                "the Llama-3.2 3B target did not enter the resident Metal decode lane"
            ),
        };
        generated.push(next);
    }
    Ok(Eagle3BenchRun {
        generated,
        ttft_ms,
        decode_ms: decode_started.elapsed().as_secs_f64() * 1000.0,
        resident_normal_steps,
        ..Eagle3BenchRun::default()
    })
}

fn run_eagle3_resident_greedy(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    draft_tokens: usize,
    tree_nodes: Option<usize>,
    tree_topk: usize,
    tree_expansions: usize,
    suffix_first: bool,
    checkpoint: camelid::eagle3::Eagle3DraftModel,
) -> anyhow::Result<Eagle3BenchRun> {
    use camelid::eagle3::TARGET_LAYER_INPUT_IDS;
    use camelid::eagle3_runtime::{
        Eagle3AuthoritativeCatchup, Eagle3Drafter, Eagle3DynamicFrontierConfig,
    };
    use camelid::inference::spec_tree::TreeDrafter;
    use camelid::inference::suffix_decoding::SuffixDecodingDrafter;

    let mut session = LlamaInferenceSession::new(config.clone(), Arc::clone(weights))?;
    let _ = session.prewarm_resident_weights();
    // EAGLE alternates target and head command buffers on Metal's shared serial queue.
    // Do not leave a pre-committed target graph waiting ahead of a head update.
    session.set_resident_encode_ahead_enabled(false);
    let ttft_started = Instant::now();
    let prompt = session
        .forward_greedy_resident_prefill_with_layer_inputs(prompt_tokens, &TARGET_LAYER_INPUT_IDS)?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident Metal prompt prefill with EAGLE-3 activation capture is unavailable"
            )
        })?;
    let first = *prompt
        .predictions
        .last()
        .ok_or_else(|| anyhow::anyhow!("EAGLE-3 target prompt produced no prediction"))?;
    let ttft_ms = ttft_started.elapsed().as_secs_f64() * 1000.0;

    let head_capacity = prompt_tokens
        .len()
        .checked_add(max_tokens)
        .and_then(|n| n.checked_add(draft_tokens + 1))
        .ok_or_else(|| anyhow::anyhow!("EAGLE-3 cache capacity overflow"))?;
    let head_upload_started = Instant::now();
    let mut drafter = Eagle3Drafter::new(checkpoint, head_capacity)?;
    let head_upload_ms = head_upload_started.elapsed().as_secs_f64() * 1000.0;
    let mut run = Eagle3BenchRun {
        generated: vec![first],
        ttft_ms,
        bootstrap_capture_ms: ttft_ms,
        head_upload_ms,
        ..Eagle3BenchRun::default()
    };
    let seed_started = Instant::now();
    drafter.seed_prompt(weights, prompt_tokens, first, &prompt.layer_inputs)?;
    run.head_seed_ms = seed_started.elapsed().as_secs_f64() * 1000.0;
    let tree_lattice_nodes = tree_nodes
        .map(|node_budget| {
            tree_topk
                .checked_mul(tree_expansions)
                .and_then(|nodes| nodes.checked_add(1))
                .map(|explored| explored.max(node_budget))
                .ok_or_else(|| anyhow::anyhow!("EAGLE-3 dynamic lattice budget overflow"))
        })
        .transpose()?;
    let mut suffix_drafter = suffix_first.then(SuffixDecodingDrafter::default);
    let mut pending_suffix_head = Eagle3AuthoritativeCatchup::default();
    let mut suffix_history = suffix_first.then(|| {
        let mut history = Vec::with_capacity(prompt_tokens.len() + max_tokens);
        history.extend_from_slice(prompt_tokens);
        history.push(first);
        history
    });

    let decode_started = Instant::now();
    while run.generated.len() < max_tokens
        && !tokenizer
            .special
            .eog
            .contains(run.generated.last().expect("generated is seeded"))
    {
        let remaining = max_tokens - run.generated.len();
        let context_room = session.remaining_context();
        if context_room == 0 {
            break;
        }
        let budget = draft_tokens
            .min(remaining.saturating_sub(1))
            .min(context_room.saturating_sub(1));

        // A final single token has no successor to draft. Keep it on the resident target;
        // no head update is needed once the requested output length is reached.
        if budget == 0 {
            let anchor = *run.generated.last().expect("generated is seeded");
            let next = session
                .generate_next_token_greedy_resident(anchor)?
                .ok_or_else(|| anyhow::anyhow!("resident Metal target became unavailable"))?
                .0;
            run.resident_normal_steps += 1;
            run.generated.push(next);
            if let Some(history) = suffix_history.as_mut() {
                history.push(next);
            }
            continue;
        }

        let anchor = *run.generated.last().expect("generated is seeded");
        let target_before = session.kv_position();
        let (emitted, offered, verify_nodes) = if let Some(node_budget) = tree_nodes {
            let round_node_budget = node_budget.min(context_room);
            let suffix_drafts = if let (Some(suffix), Some(history)) =
                (suffix_drafter.as_mut(), suffix_history.as_ref())
            {
                let draft_started = Instant::now();
                let suffix_depth = budget.min(round_node_budget.saturating_sub(1));
                let tree = suffix.draft_tree(history, anchor, round_node_budget, suffix_depth);
                let drafts = deepest_suffix_chain(&tree, suffix_depth);
                run.draft_us += draft_started.elapsed().as_micros();
                drafts
            } else {
                Vec::new()
            };

            if !suffix_drafts.is_empty() {
                run.drafted_token_ids.extend_from_slice(&suffix_drafts);
                let verify_started = Instant::now();
                let verified = session
                    .verify_drafts_metal_with_layer_inputs(
                        anchor,
                        &suffix_drafts,
                        &TARGET_LAYER_INPUT_IDS,
                    )?
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "resident Metal suffix-first verification became unavailable at position {target_before}"
                        )
                    })?;
                run.verify_us += verify_started.elapsed().as_micros();
                anyhow::ensure!(
                    verified.predictions.len() == suffix_drafts.len() + 1,
                    "suffix-first target returned {} predictions for {} draft tokens",
                    verified.predictions.len(),
                    suffix_drafts.len()
                );
                let accepted = accepted_draft_prefix(&suffix_drafts, &verified.predictions);
                let emitted = verified.predictions[..=accepted].to_vec();
                // Suffix proposals do not read the learned head. Keep the authoritative
                // token/capture pairs in order, but defer all EAGLE work until a dynamic-tree
                // round actually needs a current stable seed. An all-suffix completion never
                // pays this update at all.
                let buffer_started = Instant::now();
                pending_suffix_head.push(&verified.layer_inputs, &emitted)?;
                run.suffix_head_buffer_us += buffer_started.elapsed().as_micros();
                let offered = suffix_drafts.len();
                run.suffix_rounds += 1;
                run.suffix_offered += offered as u64;
                run.suffix_emitted_tokens += emitted.len() as u64;
                (emitted, offered, offered + 1)
            } else {
                if !pending_suffix_head.is_empty() {
                    let update_started = Instant::now();
                    let catchup_rows =
                        drafter.accept_authoritative_catchup(weights, &mut pending_suffix_head)?;
                    run.head_update_us += update_started.elapsed().as_micros();
                    run.suffix_head_catchups += 1;
                    run.suffix_head_catchup_rows += catchup_rows as u64;
                }
                anyhow::ensure!(
                    drafter.filled() == session.kv_position(),
                    "EAGLE-3 catch-up did not reach target watermark: head={} target={}",
                    drafter.filled(),
                    session.kv_position()
                );
                let draft_started = Instant::now();
                let frontier = drafter.draft_dynamic_frontier(
                    weights,
                    anchor,
                    Eagle3DynamicFrontierConfig {
                        max_verify_nodes: round_node_budget,
                        max_lattice_nodes: tree_lattice_nodes.expect("tree budget is present"),
                        max_depth: budget,
                        candidates_per_parent: tree_topk,
                        max_head_expansions: tree_expansions,
                    },
                )?;
                let materialized_head_forwards = frontier.materialized_head_forwards();
                let forest = frontier.finish()?;
                let actual_nodes = forest.scored.tree.nodes();
                let actual_max_depth = forest.scored.tree.max_depth();
                anyhow::ensure!(
                    (2..=round_node_budget).contains(&actual_nodes),
                    "dynamic EAGLE forest produced {actual_nodes} rows for round node budget {round_node_budget}"
                );
                run.drafted_token_ids
                    .extend_from_slice(&forest.scored.tree.tokens[1..]);
                run.draft_us += draft_started.elapsed().as_micros();

                let verify_started = Instant::now();
                let verified = session
                    .verify_tree_metal_with_layer_inputs(
                        &forest.scored.tree,
                        &TARGET_LAYER_INPUT_IDS,
                    )?
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "resident Metal EAGLE-3 tree verification became unavailable at position {target_before}"
                        )
                    })?;
                run.verify_us += verify_started.elapsed().as_micros();
                anyhow::ensure!(
                    verified.predictions.len() == actual_nodes,
                    "EAGLE-3 tree target returned {} predictions for {actual_nodes} rows",
                    verified.predictions.len()
                );
                let acceptance = forest.accept_target_predictions(&verified.predictions)?;
                anyhow::ensure!(
                    acceptance.capture_rows.len() == acceptance.emitted_tokens.len(),
                    "EAGLE-3 tree emitted/capture path lengths diverged: {}/{}",
                    acceptance.emitted_tokens.len(),
                    acceptance.capture_rows.len()
                );
                let update_started = Instant::now();
                drafter.accept_authoritative_forest(
                    weights,
                    &verified.layer_inputs,
                    &acceptance,
                )?;
                run.head_update_us += update_started.elapsed().as_micros();
                let emitted_count = acceptance.emitted_tokens.len();
                let offered = actual_nodes.saturating_sub(1);
                run.dynamic_tree_rounds += 1;
                run.dynamic_tree_offered += offered as u64;
                run.dynamic_tree_emitted_tokens += emitted_count as u64;
                run.materialized_head_forwards += materialized_head_forwards as u64;
                run.dynamic_tree_max_depth_sum += actual_max_depth as u64;
                (acceptance.emitted_tokens, offered, actual_nodes)
            }
        } else {
            // Existing top-1 chain path: kept independent of every dynamic-tree option.
            let draft_started = Instant::now();
            let drafts = drafter.draft(weights, budget)?;
            run.drafted_token_ids.extend_from_slice(&drafts);
            run.draft_us += draft_started.elapsed().as_micros();
            let verify_started = Instant::now();
            let verified = session
                .verify_drafts_metal_with_layer_inputs(
                    anchor,
                    &drafts,
                    &TARGET_LAYER_INPUT_IDS,
                )?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "resident Metal EAGLE-3 verification became unavailable at position {target_before}"
                    )
                })?;
            run.verify_us += verify_started.elapsed().as_micros();
            let accepted = accepted_draft_prefix(&drafts, &verified.predictions);
            let emitted = verified.predictions[..=accepted].to_vec();
            let update_started = Instant::now();
            drafter.accept_authoritative(weights, &verified.layer_inputs, &emitted)?;
            run.head_update_us += update_started.elapsed().as_micros();
            let offered = drafts.len();
            (emitted, offered, offered + 1)
        };

        anyhow::ensure!(
            session.kv_position() == target_before + emitted.len(),
            "EAGLE-3 target watermark advanced {} rows for {} emitted tokens",
            session.kv_position().saturating_sub(target_before),
            emitted.len()
        );
        run.resident_verify_rounds += 1;
        let effective_head_filled = pending_suffix_head.effective_filled(drafter.filled())?;
        anyhow::ensure!(
            effective_head_filled == session.kv_position(),
            "EAGLE-3/target cache watermarks diverged: materialized_head={} pending_head={} target={}",
            drafter.filled(),
            pending_suffix_head.pending_rows(),
            session.kv_position()
        );

        run.rounds += 1;
        run.drafted += offered as u64;
        run.accepted_drafts += emitted.len().saturating_sub(1) as u64;
        run.verify_nodes += verify_nodes as u64;
        let generated_before = run.generated.len();
        for token in emitted {
            if run.generated.len() >= max_tokens {
                break;
            }
            run.generated.push(token);
            if tokenizer.special.eog.contains(&token) {
                break;
            }
        }
        if let Some(history) = suffix_history.as_mut() {
            history.extend_from_slice(&run.generated[generated_before..]);
        }
    }
    // Once generation has stopped, no consumer can observe the learned head again. Deliberately
    // drop a final suffix-only streak instead of paying a useless catch-up in the epilogue.
    run.suffix_head_discarded_rows = pending_suffix_head.pending_rows() as u64;
    run.decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    anyhow::ensure!(
        run.cpu_verify_rounds == 0 && run.resident_verify_rounds == run.rounds,
        "EAGLE-3 benchmark left the resident verifier: resident={} cpu={} rounds={}",
        run.resident_verify_rounds,
        run.cpu_verify_rounds,
        run.rounds
    );

    Ok(run)
}

#[derive(Serialize)]
struct BenchEagle3Record {
    runtime: &'static str,
    commit: String,
    camelid_version: String,
    binary_sha256: String,
    workload: String,
    input_sha256: String,
    prompt_sha256: String,
    prompt_format: &'static str,
    add_bos: bool,
    add_eos: bool,
    parse_special: bool,
    model: String,
    model_sha256: String,
    tokenizer_metadata_sha256: Option<String>,
    eagle3: String,
    eagle3_sha256: String,
    eagle3_revision: &'static str,
    quantization: String,
    prompt_tokens: usize,
    max_tokens: usize,
    draft_tokens: usize,
    draft_mode: &'static str,
    tree_node_budget: Option<usize>,
    tree_topk: Option<usize>,
    tree_expansions: Option<usize>,
    plain_generated_tokens: usize,
    eagle3_generated_tokens: usize,
    head_load_ms: f64,
    head_upload_ms: f64,
    head_seed_ms: f64,
    bootstrap_capture_ms: f64,
    plain_ttft_ms: f64,
    plain_decode_ms: f64,
    plain_tokens_per_second: f64,
    eagle3_ttft_ms: f64,
    eagle3_decode_ms: f64,
    eagle3_tokens_per_second: f64,
    plain_request_tokens_per_second: f64,
    eagle3_warm_head_request_tokens_per_second: f64,
    eagle3_head_cold_tokens_per_second: f64,
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
    accept_rate: f64,
    mean_emitted_tokens_per_round: f64,
    mean_verify_nodes_per_round: f64,
    draft_ms: f64,
    verify_ms: f64,
    head_update_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_rounds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_offered: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_emitted_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_head_catchups: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_head_catchup_rows: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_head_discarded_rows: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_head_buffer_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_tree_rounds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_tree_offered: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_tree_emitted_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    materialized_head_forwards: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_materialized_head_forwards_per_dynamic_round: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_dynamic_tree_max_depth: Option<f64>,
    resident_verify_rounds: u64,
    cpu_verify_rounds: u64,
    resident_normal_steps: u64,
    speedup: f64,
    first_divergent_generated_token_index: i64,
    lossless: bool,
    plain_token_ids: Vec<u32>,
    eagle3_token_ids: Vec<u32>,
    eagle3_drafted_token_ids: Vec<u32>,
    metal_device: Option<String>,
    host_isa: String,
    effective_env: BTreeMap<String, Option<String>>,
    planner_env_updates: BTreeMap<String, Option<String>>,
    execution_plan: camelid::execution_plan::ExecutionPlan,
    peak_memory_bytes: u64,
}

fn eagle3_effective_env() -> BTreeMap<String, Option<String>> {
    const KEYS: &[&str] = &[
        "CAMELID_EAGLE3_FULL_AUTHORITATIVE",
        "CAMELID_EAGLE3_LM_HEAD_Q8",
        "CAMELID_EAGLE3_LM_HEAD_ROWS",
        "CAMELID_METAL_LINEAR",
        "CAMELID_METAL_Q8",
        "CAMELID_METAL_RESIDENT_DECODE",
        "CAMELID_METAL_RESIDENT_PREFILL",
        "CAMELID_METAL_ATTN2",
        "CAMELID_METAL_ATTN_BATCH_K",
        "CAMELID_METAL_KV_DTYPE",
        "CAMELID_METAL_WIRE",
        "CAMELID_METAL_WIRE_NSG8",
        "CAMELID_METAL_F32Y",
        "CAMELID_METAL_NOCOPY",
        "CAMELID_METAL_KQUANT",
        "CAMELID_KQUANT_V2",
        "CAMELID_KQUANT_V3",
        "CAMELID_KQUANT_V4",
        "CAMELID_KQUANT_V4_TRACE",
        "CAMELID_KQUANT_MMA",
        "CAMELID_SPEC_TREE",
    ];
    KEYS.iter()
        .map(|key| ((*key).to_string(), std::env::var(key).ok()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_bench_eagle3(
    model: PathBuf,
    eagle3_dir: PathBuf,
    draft_tokens: usize,
    tree_nodes: Option<usize>,
    tree_topk: usize,
    tree_expansions: usize,
    suffix_first: bool,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
    chat: bool,
    workload: String,
    max_tokens: usize,
    threads: Option<usize>,
) -> anyhow::Result<()> {
    const PINNED_TARGET_SHA256: &str =
        "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff";
    const PINNED_EAGLE3_SHA256: &str =
        "c0713251464a9b6b5fcf9fb229587bbe59b6fd1521027aef32101d11b9ebbdaf";
    const PINNED_EAGLE3_REVISION: &str = "02d343789b502a3edfe351bdd4537a44affb98cd";
    anyhow::ensure!(max_tokens >= 2, "--max-tokens must be at least 2");
    anyhow::ensure!(
        (1..=15).contains(&draft_tokens),
        "--draft-tokens must be in 1..=15"
    );
    if let Some(nodes) = tree_nodes {
        anyhow::ensure!(
            (2..=camelid::inference::spec_tree::TREE_MAX_NODES).contains(&nodes),
            "--tree-nodes must be in 2..={}",
            camelid::inference::spec_tree::TREE_MAX_NODES
        );
    }
    anyhow::ensure!(
        !suffix_first || tree_nodes.is_some(),
        "--suffix-first requires --tree-nodes"
    );
    anyhow::ensure!(
        (1..=camelid::metal::EAGLE3_TOP_K_CANDIDATES).contains(&tree_topk),
        "--tree-topk must be in 1..={}",
        camelid::metal::EAGLE3_TOP_K_CANDIDATES
    );
    anyhow::ensure!(
        (1..=camelid::inference::spec_tree::TREE_MAX_NODES).contains(&tree_expansions),
        "--tree-expansions must be in 1..={} (including the root)",
        camelid::inference::spec_tree::TREE_MAX_NODES
    );
    configure_rayon_threads(threads)?;
    let input_text = match (&prompt_file, &prompt) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(text)) => text.clone(),
        (None, None) => anyhow::bail!("provide --prompt-file <path> or --prompt <text>"),
    };

    let model_sha256 = camelid::receipt::sha256_file_hex_cached(&model)
        .map_err(|error| anyhow::anyhow!("hashing target {}: {error}", model.display()))?;
    anyhow::ensure!(
        model_sha256 == PINNED_TARGET_SHA256,
        "the learned head is pinned to target SHA-256 {PINNED_TARGET_SHA256}, got {model_sha256}"
    );
    let eagle3_weights = eagle3_dir.join("model.safetensors");
    let eagle3_sha256 = camelid::receipt::sha256_file_hex(&eagle3_weights).map_err(|error| {
        anyhow::anyhow!(
            "hashing EAGLE-3 checkpoint {}: {error}",
            eagle3_weights.display()
        )
    })?;
    anyhow::ensure!(
        eagle3_sha256 == PINNED_EAGLE3_SHA256,
        "expected pinned EAGLE-3 SHA-256 {PINNED_EAGLE3_SHA256}, got {eagle3_sha256}"
    );
    let current_exe = std::env::current_exe()?;
    let binary_sha256 = camelid::receipt::sha256_file_hex(&current_exe)
        .map_err(|error| anyhow::anyhow!("hashing benchmark binary: {error}"))?;

    let gguf = read_metadata(&model)?;
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::CpuDenseOnly)?;
    let plan_outcome = camelid::execution_plan::plan_for_model(&model, &gguf, threads);
    camelid::execution_plan::PlannerEnv::capture().apply(&plan_outcome.env_updates);
    let planner_env_updates = plan_outcome
        .env_updates
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.map(str::to_string)))
        .collect();
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    anyhow::ensure!(
        config.architecture == "llama"
            && config.embedding_length == 3_072
            && config.block_count == 28
            && config.feed_forward_length == 8_192
            && config.attention_head_count == 24
            && config.attention_head_count_kv == 8
            && config.vocab_size == Some(128_256),
        "the pinned EAGLE-3 head requires the exact Llama-3.2-3B target geometry; got arch={} hidden={} layers={} ffn={} heads={}/{} vocab={:?}",
        config.architecture,
        config.embedding_length,
        config.block_count,
        config.feed_forward_length,
        config.attention_head_count,
        config.attention_head_count_kv,
        config.vocab_size,
    );
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None)?);
    let (prompt_text, prompt_format, add_special, parse_special) = if chat {
        let (rendered, add_special, parse_special) =
            camelid::api::render_single_user_chat_prompt_for_benchmark(&input_text, &tokenizer)
                .map_err(|error| anyhow::anyhow!("rendering benchmark chat prompt: {error}"))?;
        (
            rendered,
            "served_single_user_chat_template",
            add_special,
            parse_special,
        )
    } else {
        (input_text.clone(), "raw_completion_bos_no_eos", true, false)
    };
    let prompt_token_ids = tokenizer.encode(&prompt_text, add_special, parse_special)?;
    anyhow::ensure!(
        prompt_token_ids.len() >= 3,
        "resident EAGLE-3 capture requires at least three encoded prompt tokens, got {}",
        prompt_token_ids.len()
    );
    anyhow::ensure!(
        prompt_token_ids.len() + max_tokens <= config.context_length as usize,
        "prompt plus generation exceeds target context"
    );

    eprintln!("[bench-eagle3] plain resident Metal target lane...");
    let plain =
        run_plain_resident_greedy(&config, &weights, &tokenizer, &prompt_token_ids, max_tokens)?;
    eprintln!("[bench-eagle3] loading strict EAGLE-3 checkpoint...");
    let head_load_started = Instant::now();
    let checkpoint = camelid::eagle3::Eagle3DraftModel::load(&eagle3_dir)?;
    let head_load_ms = head_load_started.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[bench-eagle3] learned recurrent draft + resident target verify...");
    let eagle = run_eagle3_resident_greedy(
        &config,
        &weights,
        &tokenizer,
        &prompt_token_ids,
        max_tokens,
        draft_tokens,
        tree_nodes,
        tree_topk,
        tree_expansions,
        suffix_first,
        checkpoint,
    )?;

    let decode_tps = |run: &Eagle3BenchRun| {
        let tokens = run.generated.len().saturating_sub(1);
        if tokens == 0 || run.decode_ms <= 0.0 {
            0.0
        } else {
            tokens as f64 / (run.decode_ms / 1000.0)
        }
    };
    let plain_tps = decode_tps(&plain);
    let eagle_tps = decode_tps(&eagle);
    let lossless = plain.generated == eagle.generated;
    let mut first_divergent = first_divergence(&plain.generated, &eagle.generated);
    if !lossless && first_divergent < 0 {
        first_divergent = plain.generated.len().min(eagle.generated.len()) as i64;
    }
    let accept_rate = if eagle.drafted == 0 {
        0.0
    } else {
        eagle.accepted_drafts as f64 / eagle.drafted as f64
    };
    let mean_emitted = if eagle.rounds == 0 {
        0.0
    } else {
        (eagle.accepted_drafts + eagle.rounds) as f64 / eagle.rounds as f64
    };
    let mean_verify_nodes = if eagle.rounds == 0 {
        0.0
    } else {
        eagle.verify_nodes as f64 / eagle.rounds as f64
    };
    let mean_materialized_head_forwards = if eagle.dynamic_tree_rounds == 0 {
        0.0
    } else {
        eagle.materialized_head_forwards as f64 / eagle.dynamic_tree_rounds as f64
    };
    let mean_dynamic_tree_max_depth = if eagle.dynamic_tree_rounds == 0 {
        0.0
    } else {
        eagle.dynamic_tree_max_depth_sum as f64 / eagle.dynamic_tree_rounds as f64
    };
    let record = BenchEagle3Record {
        runtime: "camelid-eagle3-resident-metal",
        commit: benchmark_commit(),
        camelid_version: camelid::receipt::camelid_version(),
        binary_sha256,
        workload,
        input_sha256: camelid::receipt::sha256_hex(input_text.as_bytes()),
        prompt_sha256: camelid::receipt::sha256_hex(prompt_text.as_bytes()),
        prompt_format,
        add_bos: add_special,
        add_eos: false,
        parse_special,
        model: model.display().to_string(),
        model_sha256,
        tokenizer_metadata_sha256: camelid::receipt::tokenizer_metadata_sha256(&gguf),
        eagle3: eagle3_dir.display().to_string(),
        eagle3_sha256,
        eagle3_revision: PINNED_EAGLE3_REVISION,
        quantization: camelid::receipt::quantization_label(&gguf),
        prompt_tokens: prompt_token_ids.len(),
        max_tokens,
        draft_tokens,
        draft_mode: if suffix_first {
            "suffix_then_dynamic_tree"
        } else if tree_nodes.is_some() {
            "dynamic_tree"
        } else {
            "linear_top1"
        },
        tree_node_budget: tree_nodes,
        tree_topk: tree_nodes.map(|_| tree_topk),
        tree_expansions: tree_nodes.map(|_| tree_expansions),
        plain_generated_tokens: plain.generated.len(),
        eagle3_generated_tokens: eagle.generated.len(),
        head_load_ms,
        head_upload_ms: eagle.head_upload_ms,
        head_seed_ms: eagle.head_seed_ms,
        bootstrap_capture_ms: eagle.bootstrap_capture_ms,
        plain_ttft_ms: plain.ttft_ms,
        plain_decode_ms: plain.decode_ms,
        plain_tokens_per_second: plain_tps,
        eagle3_ttft_ms: eagle.ttft_ms,
        eagle3_decode_ms: eagle.decode_ms,
        eagle3_tokens_per_second: eagle_tps,
        plain_request_tokens_per_second: plain.generated.len() as f64
            / ((plain.ttft_ms + plain.decode_ms) / 1000.0),
        eagle3_warm_head_request_tokens_per_second: eagle.generated.len() as f64
            / ((eagle.ttft_ms + eagle.head_seed_ms + eagle.decode_ms) / 1000.0),
        eagle3_head_cold_tokens_per_second: eagle.generated.len() as f64
            / ((head_load_ms
                + eagle.head_upload_ms
                + eagle.ttft_ms
                + eagle.head_seed_ms
                + eagle.decode_ms)
                / 1000.0),
        rounds: eagle.rounds,
        drafted: eagle.drafted,
        accepted_drafts: eagle.accepted_drafts,
        accept_rate,
        mean_emitted_tokens_per_round: mean_emitted,
        mean_verify_nodes_per_round: mean_verify_nodes,
        draft_ms: eagle.draft_us as f64 / 1000.0,
        verify_ms: eagle.verify_us as f64 / 1000.0,
        head_update_ms: eagle.head_update_us as f64 / 1000.0,
        suffix_rounds: suffix_first.then_some(eagle.suffix_rounds),
        suffix_offered: suffix_first.then_some(eagle.suffix_offered),
        suffix_emitted_tokens: suffix_first.then_some(eagle.suffix_emitted_tokens),
        suffix_head_catchups: suffix_first.then_some(eagle.suffix_head_catchups),
        suffix_head_catchup_rows: suffix_first.then_some(eagle.suffix_head_catchup_rows),
        suffix_head_discarded_rows: suffix_first.then_some(eagle.suffix_head_discarded_rows),
        suffix_head_buffer_ms: suffix_first.then_some(eagle.suffix_head_buffer_us as f64 / 1000.0),
        dynamic_tree_rounds: suffix_first.then_some(eagle.dynamic_tree_rounds),
        dynamic_tree_offered: suffix_first.then_some(eagle.dynamic_tree_offered),
        dynamic_tree_emitted_tokens: suffix_first.then_some(eagle.dynamic_tree_emitted_tokens),
        materialized_head_forwards: tree_nodes.map(|_| eagle.materialized_head_forwards),
        mean_materialized_head_forwards_per_dynamic_round: tree_nodes
            .map(|_| mean_materialized_head_forwards),
        mean_dynamic_tree_max_depth: tree_nodes.map(|_| mean_dynamic_tree_max_depth),
        resident_verify_rounds: eagle.resident_verify_rounds,
        cpu_verify_rounds: eagle.cpu_verify_rounds,
        resident_normal_steps: eagle.resident_normal_steps,
        speedup: if plain_tps > 0.0 {
            eagle_tps / plain_tps
        } else {
            0.0
        },
        first_divergent_generated_token_index: first_divergent,
        lossless,
        plain_token_ids: plain.generated,
        eagle3_token_ids: eagle.generated,
        eagle3_drafted_token_ids: eagle.drafted_token_ids,
        metal_device: camelid::metal::detect_metal_device().device_name,
        host_isa: camelid::receipt::host_isa_marker(),
        effective_env: eagle3_effective_env(),
        planner_env_updates,
        execution_plan: plan_outcome.plan,
        peak_memory_bytes: peak_rss_bytes(),
    };
    println!("{}", serde_json::to_string(&record)?);
    eprintln!(
        "[bench-eagle3] mode={} γ={} nodes/round {:.2} accept {:.1}% emitted/round {:.2} | plain {:.2} → EAGLE-3 {:.2} tok/s ({:.2}x) | {}",
        record.draft_mode,
        record.draft_tokens,
        record.mean_verify_nodes_per_round,
        100.0 * record.accept_rate,
        record.mean_emitted_tokens_per_round,
        record.plain_tokens_per_second,
        record.eagle3_tokens_per_second,
        record.speedup,
        if record.lossless { "LOSSLESS ✓" } else { "DIVERGED" },
    );
    anyhow::ensure!(
        record.lossless,
        "EAGLE-3 output diverged from the resident plain target at generated token {}",
        record.first_divergent_generated_token_index
    );
    Ok(())
}

/// First index at which `a` and `b` differ, or `-1` if one is a prefix of the other AND they
/// are the same length (i.e. identical). Differing lengths count as a divergence at the
/// shorter length — for the lossless gate, two greedy streams must be byte-identical.
fn first_divergence(a: &[u32], b: &[u32]) -> i64 {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return i as i64;
        }
    }
    if a.len() == b.len() {
        -1
    } else {
        n as i64
    }
}

/// Fail closed BEFORE weights load on every CLI lane that would construct a
/// direct dense `LlamaInferenceSession` for an architecture the runnable
/// bridge must serve ON THIS HOST. qwen35 / gemma2: always (qwen35's hybrid
/// layers do not fit the dense tensor map; gemma2's sandwich norms are still
/// silently dropped at bind). gemma3: capability-aware since the Metal
/// campaign's Phase 3b flip — on a resident-capable macOS host the direct
/// dense session runs the Metal-resident windowed forward correctly, so it is
/// allowed through; where the resident lane cannot serve (no Metal device,
/// `CAMELID_METAL_RESIDENT_DECODE=0` / `--deterministic`, CUDA-resident), the
/// CPU dense forward would be the only engine left and it has no
/// sliding-window mask, so fail here and point the user at `camelid serve`
/// (whose router falls back to the runnable bridge). The forward-dispatch
/// guard (hazard H4) backstops this with a typed error either way. Mirrors
/// serve's `is_runnable_serve_file` via the shared
/// `camelid::model::file_requires_runnable_bridge` predicate — which is
/// quant-aware since Phase 3c, so a non-Q8_0 gemma3 is refused here on every
/// host (it has no resident lane anywhere; hazard H5).
///
/// `lane` says whether THIS command's forward can reach the windowed resident
/// lane at all — see [`DenseLaneWindowedForward`]. A `CpuDenseOnly` lane
/// refuses a windowed arch regardless of host capability (Phase 3c finding
/// F2): accepting it would only defer the failure to the H4 choke point,
/// after a multi-gigabyte weight load, with an error naming a per-layer
/// dispatch instead of the command the user ran.
fn ensure_arch_has_direct_dense_session(
    gguf: &camelid::gguf::GgufFile,
    lane: DenseLaneWindowedForward,
) -> anyhow::Result<()> {
    let arch = gguf.architecture().unwrap_or_default();
    // Phase 3c finding F2: the Phase 3b flip made this guard capability-aware
    // for gemma3, which opened EVERY lane it protects — including the ones
    // that walk the CPU dense layer loop directly and can therefore never run
    // a windowed forward, on any host. Those lanes must refuse before weights
    // load, with a lane-accurate error, rather than accepting the model and
    // dying at the H4 choke point 26 layers into the first token.
    if lane == DenseLaneWindowedForward::CpuDenseOnly
        && camelid::model::arch_string_has_windowed_attention(arch)
    {
        anyhow::bail!(
            "architecture '{arch}' has per-layer sliding-window attention, and this command's \
             forward walks the CPU dense layer loop directly (distributed shards, ghost \
             probes, activation replay, speculative verify, alloc probes). There is no \
             windowed CPU dense forward on any host — the sliding-window mask, GeGLU, \
             dual-theta RoPE and sandwich norms exist only in the Metal-resident forward, \
             which this lane cannot reach — so it fails closed instead of emitting wrong \
             output. Serve this architecture with `camelid serve` (Metal-resident on macOS \
             with a Q8_0 row, otherwise the runnable bridge)."
        );
    }
    if camelid::model::file_requires_runnable_bridge(gguf) {
        anyhow::bail!(
            "architecture '{arch}' is served only through the runnable lane for this file on \
             this host; this command's direct dense-session path would run an incomplete \
             forward (missing norm/rope/window application) and emit wrong output, so it \
             fails closed. Use `camelid serve` (or `camelid chat`), which routes this \
             architecture to its correct runtime."
        );
    }
    Ok(())
}

/// Whether a CLI lane's forward can reach the Metal-resident windowed lane —
/// the ONLY engine in this repo with a correct gemma3 forward (Phase 2/3a
/// parity receipts, GEMMA3_METAL_CONDUCTOR.md §9b/§10b).
///
/// Phase 3c finding F2. Before the Phase 3b flip, `is_runnable_only_arch`
/// closed every lane below to gemma3 unconditionally. The flip replaced that
/// with a host-capability predicate, which is right for the lanes that decode
/// through the session (they inherit the resident lane on a capable host) and
/// wrong for the lanes that don't (they gained an admission they cannot honor).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DenseLaneWindowedForward {
    /// The lane generates through `generate_next_token_with_history_diagnostics`,
    /// which routes a windowed arch onto the single-token resident decode
    /// (hazard H2). Capability-aware admission is correct here.
    ViaSessionDecode,
    /// The lane walks the CPU dense layer loop directly. No windowed forward
    /// is reachable, so a windowed arch is refused on every host.
    CpuDenseOnly,
}

/// The measured-fastest Metal configuration is on by default for the CLI: Q8_0/Q4_K/Q6_K
/// weights stay in wire format, NSG=8 GEMV dispatch, f32-activation GEMV chain, tiled
/// decode attention, and one-command-buffer GPU prefill. Each remains overridable by
/// setting its variable to 0. Library defaults are unchanged: this runs only in the CLI
/// entry, so embedders retain conservative policy unless they opt in.
fn apply_default_fast_stack() {
    for key in [
        "CAMELID_METAL_RESIDENT_DECODE",
        "CAMELID_METAL_F32Y",
        "CAMELID_METAL_WIRE",
        "CAMELID_METAL_WIRE_NSG8",
        "CAMELID_METAL_ATTN2",
        "CAMELID_METAL_RESIDENT_PREFILL",
        "CAMELID_METAL_MM",
    ] {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, "1");
        }
    }
    // K-quant Metal kernels are macOS-only. Runtime tensor/dimension admission is still
    // authoritative, so Q5/other unsupported mixes transparently keep their CPU route.
    // An explicit value, including 0, is never overwritten.
    if cfg!(target_os = "macos") && std::env::var_os("CAMELID_METAL_KQUANT").is_none() {
        std::env::set_var("CAMELID_METAL_KQUANT", "1");
    }
}

/// True when the parsed subcommand opted into deterministic inference (`--deterministic`).
/// Only `serve` and `bench-generate` expose the flag today (the supported single-node
/// generate/serve path); every other subcommand keeps the default fast stack.
fn command_requests_deterministic(command: &Command) -> bool {
    matches!(
        command,
        Command::Serve {
            deterministic: true,
            ..
        } | Command::BenchGenerate {
            deterministic: true,
            ..
        }
    )
}

/// Pin the process to the opt-in deterministic CPU forward pass. Sets
/// `CAMELID_DETERMINISTIC=1` — which the engine reads (`inference::deterministic_mode_enabled`)
/// to fail every Metal/GPU dispatch gate closed to its order-stable CPU equivalent — then
/// forces the whole Metal fast stack off and disables GPU sampling so greedy decode also
/// stays on the CPU path. The result is bit-exact, reduction-order-stable logits for the
/// supported TinyLlama 1.1B Q8_0 lane. Only the CLI entry calls this; library defaults and
/// the default (GPU) fast path are byte-for-byte unchanged. The pinned reduction order
/// mirrors the llama.cpp reference block-wise Q8_0 dot layout the parity contract is gated
/// against (see DECISIONS.md §D9 and `qa/determinism/determinism-baseline-*.md`).
fn apply_deterministic_mode() {
    std::env::set_var("CAMELID_DETERMINISTIC", "1");
    for key in [
        "CAMELID_METAL_RESIDENT_DECODE",
        "CAMELID_METAL_F32Y",
        "CAMELID_METAL_WIRE",
        "CAMELID_METAL_WIRE_NSG8",
        "CAMELID_METAL_ATTN2",
        "CAMELID_METAL_RESIDENT_PREFILL",
        "CAMELID_METAL_MM",
        "CAMELID_METAL_LINEAR",
        "CAMELID_METAL_Q8",
        "CAMELID_METAL_Q8_RETAINED",
        "CAMELID_HYBRID_Q8_RETAINED",
        "CAMELID_METAL_NOCOPY",
        "CAMELID_METAL_KQUANT",
        "CAMELID_GEMMA4_GHOST_METAL_HEAD",
        "CAMELID_GEMMA4_GHOST_METAL_SLOTS",
        "CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST",
        "CAMELID_GEMMA4_GHOST_METAL_COMMON",
        "CAMELID_GEMMA4_GHOST_METAL",
        "CAMELID_GEMMA4_GHOST_CUDA",
        "CAMELID_GEMMA4_GHOST_CUDA_CACHE",
    ] {
        std::env::set_var(key, "0");
    }
    camelid::cuda::set_gpu_accel_enabled(false);
    camelid::cuda::set_runtime_enabled(false);
    std::env::set_var("CAMELID_METAL_KV_DTYPE", "f32");
    std::env::set_var("CAMELID_NO_GPU_SAMPLE", "1");
    eprintln!(
        "[deterministic] pinned to the order-stable CPU forward pass (Metal/GPU stack off). \
         Reduction order follows the llama.cpp reference block-wise Q8_0 layout; see DECISIONS.md \u{a7}D9."
    );
}

/// Default the single-node serve/benchmark path to fast-load
/// (`CAMELID_METAL_NOCOPY`): Q8_0/Q4_K/Q6_K weights map straight into page-aligned
/// wire pages the GPU reads in place. Gated to exactly the configuration that can
/// consume wire pages:
/// macOS, the resident decode path active, and the wire kernel stack on. This is
/// why callers apply it after command-specific mode selection instead of from
/// `apply_default_fast_stack` — speculative
/// decoding disables resident decode (its CPU repack plan needs the materialized
/// blocks), any wire-off override falls back to the block path, and the
/// distributed nodes (whose CPU forward needs `q8_0_blocks`) never run this arm.
/// Opt out with CAMELID_METAL_NOCOPY=0.
fn apply_serve_nocopy_default() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let on = |key: &str| std::env::var(key).map(|v| v == "1").unwrap_or(false);
    if should_default_serve_nocopy(
        std::env::var_os("CAMELID_METAL_NOCOPY").is_some(),
        on("CAMELID_METAL_RESIDENT_DECODE"),
        on("CAMELID_METAL_WIRE"),
        on("CAMELID_METAL_F32Y"),
    ) {
        std::env::set_var("CAMELID_METAL_NOCOPY", "1");
    }
}

/// Pure decision for [`apply_serve_nocopy_default`]: default fast-load on only when
/// the user has not set the flag either way AND the wire-resident stack that can
/// consume wire pages is active. Speculative decoding turns resident decode off, so
/// `resident == false` keeps NOCOPY off; an explicit `=0` sets `already_set` and is
/// honored.
fn should_default_serve_nocopy(already_set: bool, resident: bool, wire: bool, f32y: bool) -> bool {
    !already_set && resident && wire && f32y
}

/// Hard residency gate for pipeline nodes: every owned Q8_0 linear must hold plain
/// RAM-resident blocks, and the process memory footprint must account for them. Panics with
/// a per-tensor trace otherwise — a node is NEVER allowed to silently fall back to streaming
/// weights from disk per token (~100x slower decode, and it disqualifies the GPU-resident path).
fn assert_q8_0_weight_residency(weights: &LlamaLoadedWeights, node: &str) {
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let report: Q8ResidencyReport = weights.q8_0_residency_report();
    if !report.violations.is_empty() {
        eprintln!("[{node}] Q8_0 residency violations:");
        for violation in &report.violations {
            eprintln!("  - {violation}");
        }
        panic!(
            "[{node}] {} Q8_0 tensor(s) are NOT RAM-resident plain blocks; refusing to run",
            report.violations.len()
        );
    }
    // The retained blocks must show up in this process's physical footprint. The threshold
    // derives from the node's actual owned shard (a fixed floor would false-fail small
    // models and sharded splits); 90% slack covers allocator/OS accounting noise. A node
    // that silently fell back to disk streaming sits at a few hundred MB and misses this by
    // a wide margin. Footprint (not RSS) is the metric: macOS compresses untouched pages
    // under memory pressure, which drops them out of RSS while they are still materialized.
    let footprint = phys_footprint_bytes();
    let min_footprint = report.resident_block_bytes / 10 * 9;
    if footprint < min_footprint {
        panic!(
            "[{node}] memory footprint {:.2} GiB < required {:.2} GiB for {} retained Q8_0 \
             tensors ({:.2} GiB of blocks) — weights did not actually materialize in RAM",
            gib(footprint),
            gib(min_footprint),
            report.resident_tensors,
            gib(report.resident_block_bytes)
        );
    }
    println!(
        "[{node}] Q8_0 residency OK: {} tensors, {:.2} GiB retained blocks, footprint {:.2} GiB",
        report.resident_tensors,
        gib(report.resident_block_bytes),
        gib(footprint)
    );
}

/// Current physical memory footprint of this process in bytes — the metric Activity Monitor
/// and `/usr/bin/time -l`'s "memory footprint" report. Unlike RSS it includes pages the OS
/// compressed under memory pressure, so freshly-materialized weights are counted even on a
/// loaded machine. Falls back to peak RSS where unavailable.
fn phys_footprint_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        let ret = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V2,
                &mut info as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
            )
        };
        if ret == 0 && info.ri_phys_footprint > 0 {
            return info.ri_phys_footprint;
        }
    }
    peak_rss_bytes()
}

/// Peak resident set size of this process in bytes. macOS `getrusage` reports
/// bytes; other Unix reports kilobytes (scaled here).
#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return 0;
    }
    let max = usage.ru_maxrss.max(0) as u64;
    #[cfg(target_os = "macos")]
    {
        max
    }
    #[cfg(not(target_os = "macos"))]
    {
        max * 1024
    }
}

/// Peak resident set size of this process in bytes. Windows exposes the peak
/// working set directly via `GetProcessMemoryInfo`.
#[cfg(windows)]
fn peak_rss_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: GetCurrentProcess returns a valid pseudo-handle; `counters` is
    // sized via its `cb` field per the API contract.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return 0;
    }
    counters.PeakWorkingSetSize as u64
}

fn connect_with_retry(addr: SocketAddr) -> TcpStream {
    println!("Connecting to downstream {}...", addr);
    let start = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                stream.set_nodelay(true).unwrap();
                println!("Connected successfully to {}!", addr);
                return stream;
            }
            Err(e) => {
                // Pipeline nodes bind their sockets only after loading their weight
                // shard, which can take minutes for large models (especially when one
                // node streams from slower storage). Keep retrying well past that.
                if start.elapsed().as_secs() > 600 {
                    panic!("Failed to connect to {} after 600 seconds: {}", addr, e);
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
}

fn accept_connection(listener: &TcpListener) -> TcpStream {
    let (stream, client_addr) = listener.accept().unwrap();
    stream.set_nodelay(true).unwrap();
    println!("Accepted connection from upstream/client: {}", client_addr);
    stream
}

fn parse_layers_range(layers_str: &str) -> anyhow::Result<std::ops::Range<usize>> {
    let parts: Vec<&str> = layers_str.split("..").collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!(
            "Invalid layers range format: {}",
            layers_str
        ));
    }
    let start = parts[0].parse::<usize>()?;
    let end = parts[1].parse::<usize>()?;
    Ok(start..end)
}

#[allow(clippy::too_many_arguments)]
async fn run_distribute_worker(
    path: PathBuf,
    addr: SocketAddr,
    forward_addr: Option<SocketAddr>,
    layers: String,
    master_addr: Option<SocketAddr>,
    threads: Option<usize>,
    cghost: Option<PathBuf>,
) -> anyhow::Result<()> {
    configure_rayon_threads(threads)?;

    println!("Loading GGUF metadata from {:?}...", path);
    let gguf = read_metadata(&path)?;
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::CpuDenseOnly)?;
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
    let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&path, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf).ok();

    let layer_range = parse_layers_range(&layers)?;
    println!("Initializing worker session for layers {:?}", layer_range);

    let weights = Arc::new(if cghost.is_some() {
        // Ghost mesh: only the output ends stay resident (this is the LAST node when it has
        // no forward_addr); the layer shard streams from the .cghost per token.
        LlamaLoadedWeights::load_distributed(&store, &binding, 0, 0, false, true)?
    } else {
        LlamaLoadedWeights::load(&store, &binding, Some(layer_range.clone()))?
    });
    let mut session = LlamaInferenceSession::new(config.clone(), weights)?;
    assert_q8_0_weight_residency(&session.weights, "dist-worker");
    let mut ghost_ctx = make_ghost_node_ctx(&session, cghost.as_deref(), layer_range.clone())?;

    let listener = TcpListener::bind(addr)?;
    println!("Worker listening on {}...", addr);

    let mut downstream_stream = if let Some(faddr) = forward_addr {
        Some(connect_with_retry(faddr))
    } else {
        master_addr.map(connect_with_retry)
    };

    let mut client_stream = accept_connection(&listener);

    println!("Cluster worker execution loop active!");
    let trace = std::env::var_os("CAMELID_DISTRIBUTED_TRACE").is_some();
    let mut activations = Vec::new();

    loop {
        let idle_started = Instant::now();
        let header = match recv_activation_packet(&mut client_stream, &mut activations) {
            Ok(h) => h,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    println!("Upstream connection closed. Exiting worker loop.");
                    break;
                }
                return Err(e.into());
            }
        };

        let hidden_dim = config.embedding_length as usize;
        if activations.is_empty() || activations.len() % hidden_dim != 0 {
            return Err(anyhow::anyhow!(
                "Invalid activation packet size: {}",
                activations.len()
            ));
        }
        let rows = activations.len() / hidden_dim;
        let idle_us = idle_started.elapsed().as_micros();
        let hidden =
            CpuTensor::from_f32("activations", vec![rows, hidden_dim], activations.clone())?;

        let forward_started = Instant::now();
        let out_hidden = if let Some((streamer, placeholder)) = ghost_ctx.as_mut() {
            let (out, _bytes, _wait_us, _forward_us, _read_us, _decode_us) = ghost_stream_layers(
                &mut session,
                streamer,
                placeholder,
                hidden,
                header.pos as usize,
                header.seq_len as usize,
                false,
            )?;
            out
        } else {
            session.forward_layer_range_from_hidden(
                &hidden,
                header.pos as usize,
                header.seq_len as usize,
            )?
        };
        let forward_us = forward_started.elapsed().as_micros();
        let tail_started = Instant::now();

        if let Some(ref mut ds) = downstream_stream {
            if forward_addr.is_some() {
                send_activation_packet(ds, header.pos, header.seq_len, &out_hidden.data)?;
            } else {
                let logits = session.forward_final_norm_and_logits(&out_hidden)?;
                let vocab_size = logits.dim(1)?;
                let last_row_start = (header.seq_len as usize - 1) * vocab_size;
                let last_row_data =
                    logits.data[last_row_start..last_row_start + vocab_size].to_vec();
                let last_row_logits =
                    CpuTensor::from_f32("last_row_logits", vec![1, vocab_size], last_row_data)?;
                let token_id = LlamaSampler::Greedy.sample(&last_row_logits)?;

                let is_finished = tokenizer.as_ref().is_some_and(|tok| {
                    tok.special.eos == Some(token_id) || tok.special.eot == Some(token_id)
                });

                send_token_feedback(ds, token_id, is_finished)?;
            }
        }
        if trace {
            eprintln!(
                "[dist-worker] pos={} rows={} idle={}us forward={}us logits_send={}us",
                header.pos,
                rows,
                idle_us,
                forward_us,
                tail_started.elapsed().as_micros()
            );
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_distribute_master(
    path: PathBuf,
    worker_addr: SocketAddr,
    layers: String,
    addr: SocketAddr,
    prompt: String,
    max_tokens: usize,
    threads: Option<usize>,
    cghost: Option<PathBuf>,
) -> anyhow::Result<()> {
    configure_rayon_threads(threads)?;

    println!("Loading GGUF metadata from {:?}...", path);
    let gguf = read_metadata(&path)?;
    ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::CpuDenseOnly)?;
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf)?;
    let binding = camelid::model::LlamaTensorBinding::bind(&gguf, &config)?;
    let store = TensorStore::open(&path, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let layer_range = parse_layers_range(&layers)?;
    println!("Initializing master session for layers {:?}", layer_range);

    let weights = Arc::new(if cghost.is_some() {
        // Ghost mesh: only the token embedding stays resident (the master is the FIRST
        // node); the layer shard streams from the .cghost per token.
        LlamaLoadedWeights::load_distributed(&store, &binding, 0, 0, true, false)?
    } else {
        LlamaLoadedWeights::load(&store, &binding, Some(layer_range.clone()))?
    });
    let mut session = LlamaInferenceSession::new(config.clone(), weights)?;
    assert_q8_0_weight_residency(&session.weights, "dist-master");
    let mut ghost_ctx = make_ghost_node_ctx(&session, cghost.as_deref(), layer_range.clone())?;

    let listener = TcpListener::bind(addr)?;
    println!("Master listening for feedback on {}...", addr);

    let mut downstream_stream = connect_with_retry(worker_addr);
    let mut feedback_stream = accept_connection(&listener);

    println!("Tokenizing prompt: {:?}", prompt);
    let token_ids = tokenizer.encode(&prompt, true, false)?;
    println!("Encoded prompt: {:?}", token_ids);

    let mut pos = 0usize;
    let mut seq_len = token_ids.len();

    let hidden = session
        .weights
        .token_embedding
        .embedding_lookup(&token_ids, "token_embedding_prefill")?;
    let out_hidden = if let Some((streamer, placeholder)) = ghost_ctx.as_mut() {
        ghost_stream_layers(
            &mut session,
            streamer,
            placeholder,
            hidden,
            pos,
            seq_len,
            false,
        )?
        .0
    } else {
        session.forward_layer_range_from_hidden(&hidden, pos, seq_len)?
    };

    send_activation_packet(
        &mut downstream_stream,
        pos as u32,
        seq_len as u32,
        &out_hidden.data,
    )?;

    let feedback = recv_token_feedback(&mut feedback_stream)?;
    let mut current_token = feedback.token_id;
    let mut is_finished = feedback.is_finished;

    print!("{}", tokenizer.decode(&[current_token], true)?);
    std::io::stdout().flush()?;

    pos += seq_len;
    seq_len = 1;

    let trace = std::env::var_os("CAMELID_DISTRIBUTED_TRACE").is_some();
    let decode_start = Instant::now();
    let mut generated = 1;
    while !is_finished && generated < max_tokens {
        let compute_started = Instant::now();
        let hidden = session
            .weights
            .token_embedding
            .embedding_lookup(&[current_token], "token_embedding")?;
        let out_hidden = if let Some((streamer, placeholder)) = ghost_ctx.as_mut() {
            ghost_stream_layers(
                &mut session,
                streamer,
                placeholder,
                hidden,
                pos,
                seq_len,
                false,
            )?
            .0
        } else {
            session.forward_layer_range_from_hidden(&hidden, pos, seq_len)?
        };
        let compute_us = compute_started.elapsed().as_micros();
        let send_started = Instant::now();
        send_activation_packet(
            &mut downstream_stream,
            pos as u32,
            seq_len as u32,
            &out_hidden.data,
        )?;
        let send_us = send_started.elapsed().as_micros();
        let wait_started = Instant::now();
        let feedback = recv_token_feedback(&mut feedback_stream)?;
        if trace {
            eprintln!(
                "[dist-master] pos={pos} compute={compute_us}us send={send_us}us wait={}us",
                wait_started.elapsed().as_micros()
            );
        }
        current_token = feedback.token_id;
        is_finished = feedback.is_finished;

        print!("{}", tokenizer.decode(&[current_token], true)?);
        std::io::stdout().flush()?;

        pos += 1;
        generated += 1;
    }
    println!();

    let decode_secs = decode_start.elapsed().as_secs_f64();
    let decode_tokens = generated.saturating_sub(1);
    if decode_tokens > 0 && decode_secs > 0.0 {
        println!(
            "[distributed] decode: {} tokens in {:.2}s = {:.2} tok/s",
            decode_tokens,
            decode_secs,
            decode_tokens as f64 / decode_secs
        );
    }

    Ok(())
}

fn tensor_dump_names(tensors: Vec<String>, layers: Vec<usize>) -> Vec<String> {
    let mut names = if tensors.is_empty() {
        default_tensor_dump_names()
    } else {
        tensors
    };

    for layer in layers {
        names.extend(layer_tensor_dump_names(layer));
    }
    dedup_preserving_order(names)
}

fn default_tensor_dump_names() -> Vec<String> {
    let mut names = vec!["token_embd.weight".to_string(), "output.weight".to_string()];
    names.extend(layer_tensor_dump_names(0));
    names
}

fn layer_tensor_dump_names(layer: usize) -> Vec<String> {
    [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
    .into_iter()
    .map(|suffix| format!("blk.{layer}.{suffix}"))
    .collect()
}

fn dedup_preserving_order(names: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

#[derive(Debug, Serialize)]
struct DenseHotloopBenchReport {
    hidden: usize,
    ffn: usize,
    repeats: usize,
    warmup: usize,
    rayon_threads: usize,
    checksum: f32,
    avg_ms: DenseHotloopBenchTimings,
    min_ms: DenseHotloopBenchTimings,
    max_ms: DenseHotloopBenchTimings,
}

#[derive(Debug, Serialize, Clone, Copy)]
struct DenseHotloopBenchTimings {
    gate: f64,
    up: f64,
    activation: f64,
    down: f64,
    total: f64,
}

#[derive(Debug, Serialize)]
struct Q8BlockBenchDeterminismReport {
    execution: &'static str,
    parallel_kernel_default: bool,
    serial_vs_parallel_delta_target: f32,
    serial_vs_parallel_delta_fail_threshold: f32,
}

#[derive(Debug, Serialize)]
struct Q8BlockBenchReport {
    path: String,
    tensor: String,
    shape: Vec<usize>,
    storage_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    swap_rank2_shape: bool,
    tensor_n_bytes: u64,
    tensor_mib: f64,
    element_count: usize,
    block_count: usize,
    f32_materialized_mib: f64,
    retained_q8_payload_mib: f64,
    dot_input_f32_mib: f64,
    all_rows_output_f32_mib: Option<f64>,
    single_input_row_output_f32_mib: Option<f64>,
    determinism: Q8BlockBenchDeterminismReport,
    rows: Vec<usize>,
    row_len: usize,
    repeats: usize,
    warmup: usize,
    metadata_load_ms: f64,
    block_load_ms: f64,
    checksum: f32,
    avg_dequant_ms: f64,
    min_dequant_ms: f64,
    max_dequant_ms: f64,
    dot_checksum: f32,
    avg_dot_ms: f64,
    min_dot_ms: f64,
    max_dot_ms: f64,
    all_rows_dot: bool,
    all_rows_dot_checksum: Option<f32>,
    avg_all_rows_dot_ms: Option<f64>,
    min_all_rows_dot_ms: Option<f64>,
    max_all_rows_dot_ms: Option<f64>,
    single_input_row_dot: bool,
    single_input_row_dot_checksum: Option<f32>,
    avg_single_input_row_dot_ms: Option<f64>,
    min_single_input_row_dot_ms: Option<f64>,
    max_single_input_row_dot_ms: Option<f64>,
    dot_input_pattern: &'static str,
    notes: Vec<&'static str>,
}

struct Q8BlockBenchOptions<'a> {
    path: &'a PathBuf,
    tensor_name: &'a str,
    rows: Vec<usize>,
    repeats: usize,
    warmup: usize,
    swap_rank2_shape: bool,
    all_rows_dot: bool,
    single_input_row_dot: bool,
}

fn bench_q8_blocks(options: Q8BlockBenchOptions<'_>) -> anyhow::Result<Q8BlockBenchReport> {
    let Q8BlockBenchOptions {
        path,
        tensor_name,
        rows,
        repeats,
        warmup,
        swap_rank2_shape,
        all_rows_dot,
        single_input_row_dot,
    } = options;

    anyhow::ensure!(repeats > 0, "--repeats must be greater than zero");

    let started = Instant::now();
    let gguf = read_metadata(path)?;
    let metadata_load_ms = elapsed_ms(started);
    let store = TensorStore::open(path, &gguf);
    let desc = store.descriptor(tensor_name)?.clone();

    anyhow::ensure!(
        desc.tensor_type == GgufTensorType::Q8_0,
        "tensor {tensor_name} has storage type {:?}; bench-q8-blocks requires Q8_0",
        desc.tensor_type
    );

    let started = Instant::now();
    let mut tensor = store.load_q8_0_blocks(tensor_name)?;
    let block_load_ms = elapsed_ms(started);
    let storage_shape = tensor.shape.dims.clone();
    anyhow::ensure!(
        tensor.shape.dims.len() == 2,
        "bench-q8-blocks expects a rank-2 tensor, got {:?}",
        tensor.shape.dims
    );
    if swap_rank2_shape {
        tensor.shape.dims.swap(0, 1);
    }
    let row_count = tensor.shape.dims[0];
    let row_len = tensor.shape.dims[1];
    let rows = if rows.is_empty() { vec![0] } else { rows };
    for row in &rows {
        anyhow::ensure!(
            *row < row_count,
            "row {row} out of range for tensor {tensor_name} with {row_count} rows"
        );
    }

    let dot_input = bench_values(row_len, 0.00019);
    let single_input = if single_input_row_dot {
        Some(CpuTensor::from_f32(
            "bench_single_input",
            vec![1, row_len],
            dot_input.clone(),
        )?)
    } else {
        None
    };

    for _ in 0..warmup {
        let _ = dequantize_q8_rows_once(&tensor, &rows)?;
        let _ = dot_q8_rows_once(&tensor, &rows, &dot_input)?;
        if all_rows_dot {
            let _ = dot_q8_all_rows_once(&tensor, &dot_input)?;
        }
        if let Some(input) = &single_input {
            let _ = dot_q8_single_input_row_once(&tensor, input)?;
        }
    }

    let mut checksum = 0.0;
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        checksum += dequantize_q8_rows_once(&tensor, &rows)?;
        timings.push(elapsed_ms(started));
    }

    let mut dot_checksum = 0.0;
    let mut dot_timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        dot_checksum += dot_q8_rows_once(&tensor, &rows, &dot_input)?;
        dot_timings.push(elapsed_ms(started));
    }

    let (all_rows_dot_checksum, all_rows_dot_timings) = if all_rows_dot {
        let mut all_rows_checksum = 0.0;
        let mut timings = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = Instant::now();
            all_rows_checksum += dot_q8_all_rows_once(&tensor, &dot_input)?;
            timings.push(elapsed_ms(started));
        }
        (Some(all_rows_checksum), Some(timings))
    } else {
        (None, None)
    };

    let (single_input_row_dot_checksum, single_input_row_dot_timings) =
        if let Some(input) = &single_input {
            let mut single_input_checksum = 0.0;
            let mut timings = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let started = Instant::now();
                single_input_checksum += dot_q8_single_input_row_once(&tensor, input)?;
                timings.push(elapsed_ms(started));
            }
            (Some(single_input_checksum), Some(timings))
        } else {
            (None, None)
        };

    let element_count = tensor.element_count()?;
    let dot_input_f32_mib =
        bytes_to_mib(dot_input.len() as f64 * std::mem::size_of::<f32>() as f64);
    let output_vector_mib = bytes_to_mib(row_count as f64 * std::mem::size_of::<f32>() as f64);
    let all_rows_output_f32_mib = all_rows_dot.then_some(output_vector_mib);
    let single_input_row_output_f32_mib = single_input_row_dot.then_some(output_vector_mib);
    Ok(Q8BlockBenchReport {
        path: path.display().to_string(),
        tensor: tensor_name.to_string(),
        shape: tensor.shape.dims.clone(),
        storage_shape,
        logical_shape: tensor.shape.dims.clone(),
        swap_rank2_shape,
        tensor_n_bytes: desc.n_bytes,
        tensor_mib: bytes_to_mib(desc.n_bytes as f64),
        element_count,
        block_count: tensor.blocks.len(),
        f32_materialized_mib: bytes_to_mib(tensor.byte_size_if_f32_materialized()? as f64),
        retained_q8_payload_mib: bytes_to_mib(desc.n_bytes as f64),
        dot_input_f32_mib,
        all_rows_output_f32_mib,
        single_input_row_output_f32_mib,
        determinism: Q8BlockBenchDeterminismReport {
            execution: "serial_only_q8_0_block_rows",
            parallel_kernel_default: false,
            serial_vs_parallel_delta_target: 0.0,
            serial_vs_parallel_delta_fail_threshold: 1e-7,
        },
        rows,
        row_len,
        repeats,
        warmup,
        metadata_load_ms,
        block_load_ms,
        checksum,
        avg_dequant_ms: average_f64(&timings),
        min_dequant_ms: timings.iter().copied().fold(f64::INFINITY, f64::min),
        max_dequant_ms: timings.iter().copied().fold(0.0, f64::max),
        dot_checksum,
        avg_dot_ms: average_f64(&dot_timings),
        min_dot_ms: dot_timings.iter().copied().fold(f64::INFINITY, f64::min),
        max_dot_ms: dot_timings.iter().copied().fold(0.0, f64::max),
        all_rows_dot,
        all_rows_dot_checksum,
        avg_all_rows_dot_ms: all_rows_dot_timings.as_ref().map(|timings| average_f64(timings)),
        min_all_rows_dot_ms: all_rows_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(f64::INFINITY, f64::min)),
        max_all_rows_dot_ms: all_rows_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(0.0, f64::max)),
        single_input_row_dot,
        single_input_row_dot_checksum,
        avg_single_input_row_dot_ms: single_input_row_dot_timings
            .as_ref()
            .map(|timings| average_f64(timings)),
        min_single_input_row_dot_ms: single_input_row_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(f64::INFINITY, f64::min)),
        max_single_input_row_dot_ms: single_input_row_dot_timings
            .as_ref()
            .map(|timings| timings.iter().copied().fold(0.0, f64::max)),
        dot_input_pattern: "deterministic bench_values(row_len, 0.00019)",
        notes: vec![
            "Loads only the selected Q8_0 tensor payload as retained blocks, not full model f32 weights.",
            "Reports the bounded f32 activation input and optional output-vector sizes so memory pressure evidence distinguishes scratch/output buffers from avoided full f32 weight materialization.",
            "Benchmarks serial bounded row dequantization, row dot products, optional all-row dot output, and optional single-input-row lazy-linear adapter output; this is groundwork evidence for lazy/on-demand Q8_0 execution, not a generation-support claim.",
            "When swap_rank2_shape is true, the benchmark reinterprets rank-2 rows/cols without transposing payload bytes, matching the current guarded runtime layout path for selected rectangular LLaMA tensors.",
            "Determinism fields intentionally record that this bench path is serial-only today; any future parallel Q8 kernel must add serial-vs-parallel evidence targeting zero delta and failing above 1e-7 unless guarded off by default.",
        ],
    })
}

fn dequantize_q8_rows_once(tensor: &Q8_0TensorBlocks, rows: &[usize]) -> anyhow::Result<f32> {
    let mut checksum = 0.0;
    for row in rows {
        let values = tensor.dequantize_row(*row)?;
        checksum += values.iter().copied().sum::<f32>();
    }
    Ok(checksum)
}

fn dot_q8_rows_once(
    tensor: &Q8_0TensorBlocks,
    rows: &[usize],
    input: &[f32],
) -> anyhow::Result<f32> {
    let mut checksum = 0.0;
    for row in rows {
        checksum += tensor.dot_row_f32(*row, input)?;
    }
    Ok(checksum)
}

fn dot_q8_all_rows_once(tensor: &Q8_0TensorBlocks, input: &[f32]) -> anyhow::Result<f32> {
    let output = tensor.dot_all_rows_f32(input, "bench_all_rows_dot")?;
    Ok(output.data.iter().copied().sum::<f32>())
}

fn dot_q8_single_input_row_once(
    tensor: &Q8_0TensorBlocks,
    input: &CpuTensor,
) -> anyhow::Result<f32> {
    let output = tensor.dot_single_input_row_f32(input, "bench_single_input_row_dot")?;
    Ok(output.data.iter().copied().sum::<f32>())
}

fn bytes_to_mib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0)
}

fn average_f64(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn bench_dense_hotloops(
    hidden: usize,
    ffn: usize,
    repeats: usize,
    warmup: usize,
) -> anyhow::Result<DenseHotloopBenchReport> {
    anyhow::ensure!(hidden > 0, "--hidden must be greater than zero");
    anyhow::ensure!(ffn > 0, "--ffn must be greater than zero");
    anyhow::ensure!(repeats > 0, "--repeats must be greater than zero");

    let input = CpuTensor::from_f32("bench_input", vec![1, hidden], bench_values(hidden, 0.001))?;
    let gate = CpuTensor::from_f32(
        "bench_gate",
        vec![hidden, ffn],
        bench_values(hidden * ffn, 0.0003),
    )?;
    let up = CpuTensor::from_f32(
        "bench_up",
        vec![hidden, ffn],
        bench_values(hidden * ffn, 0.0005),
    )?;
    let down = CpuTensor::from_f32(
        "bench_down",
        vec![ffn, hidden],
        bench_values(ffn * hidden, 0.0007),
    )?;

    for _ in 0..warmup {
        let _ = run_dense_hotloop_once(&input, &gate, &up, &down)?;
    }

    let mut checksum = 0.0;
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let measured = run_dense_hotloop_once(&input, &gate, &up, &down)?;
        checksum += measured.checksum;
        timings.push(measured.timings);
    }

    Ok(DenseHotloopBenchReport {
        hidden,
        ffn,
        repeats,
        warmup,
        rayon_threads: rayon::current_num_threads(),
        checksum,
        avg_ms: average_timings(&timings),
        min_ms: min_timings(&timings),
        max_ms: max_timings(&timings),
    })
}

#[derive(Debug)]
struct DenseHotloopMeasurement {
    timings: DenseHotloopBenchTimings,
    checksum: f32,
}

fn run_dense_hotloop_once(
    input: &CpuTensor,
    gate: &CpuTensor,
    up: &CpuTensor,
    down: &CpuTensor,
) -> anyhow::Result<DenseHotloopMeasurement> {
    let total_started = Instant::now();

    let started = Instant::now();
    let gate_out = input.matmul(gate, "bench_gate_out")?;
    let gate_ms = elapsed_ms(started);

    let started = Instant::now();
    let up_out = input.matmul(up, "bench_up_out")?;
    let up_ms = elapsed_ms(started);

    let started = Instant::now();
    let activation = gate_out.silu_mul(&up_out, "bench_activation")?;
    let activation_ms = elapsed_ms(started);

    let started = Instant::now();
    let down_out = activation.matmul(down, "bench_down_out")?;
    let down_ms = elapsed_ms(started);

    Ok(DenseHotloopMeasurement {
        timings: DenseHotloopBenchTimings {
            gate: gate_ms,
            up: up_ms,
            activation: activation_ms,
            down: down_ms,
            total: elapsed_ms(total_started),
        },
        checksum: down_out.data.iter().copied().sum(),
    })
}

fn bench_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| (((idx % 97) as f32) - 48.0) * scale)
        .collect()
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn apply_runtime_tuning_env(
    parallel_linear_min_outputs: Option<usize>,
    apple_accelerate_min_elements: Option<usize>,
    metal_linear: bool,
    metal_q8: bool,
) {
    if let Some(value) = parallel_linear_min_outputs.filter(|value| *value > 0) {
        std::env::set_var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS", value.to_string());
    }
    if let Some(value) = apple_accelerate_min_elements.filter(|value| *value > 0) {
        std::env::set_var("CAMELID_APPLE_ACCELERATE_MIN_ELEMENTS", value.to_string());
    }
    if metal_linear {
        std::env::set_var("CAMELID_METAL_LINEAR", "1");
    }
    if metal_q8 {
        std::env::set_var("CAMELID_METAL_Q8", "1");
    }
}

fn apply_spec_decode_env(
    spec_decode: Option<String>,
    spec_draft_model: Option<PathBuf>,
    spec_draft_tokens: Option<usize>,
) {
    let mode = spec_decode.filter(|mode| {
        let trimmed = mode.trim();
        !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("off")
    });
    if let Some(mode) = mode {
        std::env::set_var("CAMELID_SPEC_DECODE", mode);
        // GPU speculative verify (CAMELID_SPEC_GPU=1) runs the batched `verify_batch` on the
        // target's resident engine, which owns the weights — so keep the Metal resident paths
        // ON for it. Without GPU verify the CPU chunk verify needs CPU-resident packed Q8
        // weights, but the Metal-resident plan deliberately keeps CPU-side weights file-backed
        // (the GPU owns the resident copy), so each verify round would pay a file-speed weight
        // pass — fall back to the validated CPU repack plan in that case only.
        // A Metal host defaults to GPU verify. Leaving it unset used to select the
        // CPU plan silently, so a spec-decode run on this host measured the repack
        // path while reporting itself as speculative. An explicit CAMELID_SPEC_GPU
        // still wins in both directions; the auto-arm only fills in the unset case,
        // and only when there is actually a resident lane for the batched verify to
        // run on (see `should_auto_arm_spec_gpu` for the full precondition list).
        let spec_gpu_var = std::env::var("CAMELID_SPEC_GPU").ok();
        // `var_os().is_none()` is this file's explicitly-set-vs-defaulted test
        // (`apply_default_fast_stack`, `apply_serve_nocopy_default`): an explicit
        // value, INCLUDING `0`, is the operator's call and is never overwritten.
        let spec_gpu_set = std::env::var_os("CAMELID_SPEC_GPU").is_some();
        let spec_gpu_truthy = matches!(
            spec_gpu_var.as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        // `apply_default_fast_stack` has already run (it is applied before the
        // subcommand match), so this reads "1" unless the operator opted out or
        // `apply_deterministic_mode` forced the whole Metal stack off.
        let resident_decode_armed = std::env::var("CAMELID_METAL_RESIDENT_DECODE")
            .map(|value| value == "1")
            .unwrap_or(false);
        let deterministic = std::env::var("CAMELID_DETERMINISTIC")
            .map(|value| value == "1")
            .unwrap_or(false);
        // One probe, once, at startup; `log_acceleration_state` performs the same one
        // moments later. False on every non-macOS build (the `detect_metal_device`
        // stub), which is what keeps CPU-only and CUDA hosts on today's behavior.
        let metal_device_available = detect_metal_device().available;
        let spec_gpu = if should_auto_arm_spec_gpu(
            spec_gpu_set,
            resident_decode_armed,
            deterministic,
            metal_device_available,
        ) {
            std::env::set_var("CAMELID_SPEC_GPU", "1");
            // Printed, not only traced: `RUST_LOG` is unset on a stock install, and this
            // line decides which execution plan every request on this server takes.
            eprintln!(
                "[spec] Metal host: defaulting CAMELID_SPEC_GPU=1, so drafts are verified by \
                 the batched GPU verify and the resident decode/prefill lanes stay on. Set \
                 CAMELID_SPEC_GPU=0 to force the CPU verify plan instead."
            );
            tracing::info!(
                "speculative decoding on a Metal host: defaulting CAMELID_SPEC_GPU=1 \
                 (set CAMELID_SPEC_GPU=0 to force the CPU verify plan)"
            );
            true
        } else {
            spec_gpu_truthy
        };
        if !spec_gpu {
            std::env::set_var("CAMELID_METAL_RESIDENT_DECODE", "0");
            std::env::set_var("CAMELID_METAL_RESIDENT_PREFILL", "0");
            // Printed, not only traced: this demotion is server-wide and expensive
            // enough that no benchmark should be able to take it without seeing it.
            // Only when there WAS a lane to lose.
            if resident_decode_armed && metal_device_available {
                eprintln!(
                    "[spec] speculative decoding is running the CPU verify plan \
                     (CAMELID_SPEC_GPU is set to a non-enabling value), so the Metal \
                     resident decode and prefill lanes are now OFF for every request on \
                     this server, not just speculative ones. Unset CAMELID_SPEC_GPU (or \
                     set it to 1) to keep the resident lane."
                );
            }
            tracing::info!(
                "speculative decoding enabled; selecting the CPU execution plan \
                 (Metal resident paths disabled server-wide)"
            );
        } else {
            tracing::info!(
                "speculative decoding enabled with GPU verify (CAMELID_SPEC_GPU=1); \
                 keeping the resident decode engine for the batched verify"
            );
        }
    }
    if let Some(path) = spec_draft_model {
        std::env::set_var("CAMELID_SPEC_DRAFT_MODEL", path);
    }
    if let Some(tokens) = spec_draft_tokens.filter(|tokens| *tokens > 0) {
        std::env::set_var("CAMELID_SPEC_DRAFT_TOKENS", tokens.to_string());
    }
}

/// Pure decision for the `CAMELID_SPEC_GPU` auto-arm in [`apply_spec_decode_env`].
///
/// Speculative decoding without GPU verify switches `CAMELID_METAL_RESIDENT_DECODE` and
/// `CAMELID_METAL_RESIDENT_PREFILL` off SERVER-WIDE — for every request on the process, not
/// just the speculative ones — because the CPU chunk verify needs materialized CPU-side
/// blocks that the Metal-resident plan deliberately does not keep. Selecting that plan merely
/// because a variable was UNSET meant `--spec-decode ngram` silently benchmarked the CPU plan
/// on the exact host whose fast lane is being measured. A Metal host arms GPU verify instead,
/// but only when every precondition holds:
///
/// * `spec_gpu_set` — an explicit `CAMELID_SPEC_GPU`, INCLUDING `0`, is the operator's call
///   and is never overwritten in either direction. Same `var_os().is_none()` test
///   `apply_default_fast_stack` and `apply_serve_nocopy_default` use.
/// * `resident_decode_armed` — with `CAMELID_METAL_RESIDENT_DECODE=0` there is no resident
///   engine for `verify_batch` to run on: `verify_drafts_metal` checks that same flag and
///   returns `Ok(None)` every round, so arming the flag would advertise a lane that cannot
///   engage. Keep today's CPU plan there.
/// * `deterministic` — the deterministic lane fails every GPU gate closed by contract
///   (DECISIONS.md §D9). It must never be handed a GPU verify arm, even a dormant one.
/// * `metal_device_available` — no device, no resident lane. False on every non-macOS build
///   (the `detect_metal_device` stub), which is what keeps CPU-only and CUDA hosts exactly as
///   they are: their resident lane is gated per-request by `set_resident_paths_disabled`,
///   not by these Metal variables.
///
/// Losslessness is not a factor in this decision. Both verify arms emit the target's own
/// greedy argmax: `verify_batch` is bit-identical to `k` single-token decodes, and
/// `verify_drafts_gpu` returns `Ok(None)` into the lossless CPU chunk verify whenever the
/// engine is not materialized exactly at the current KV position.
fn should_auto_arm_spec_gpu(
    spec_gpu_set: bool,
    resident_decode_armed: bool,
    deterministic: bool,
    metal_device_available: bool,
) -> bool {
    !spec_gpu_set && resident_decode_armed && !deterministic && metal_device_available
}

fn log_acceleration_state() {
    let metal = detect_metal_device();
    tracing::info!(
        rayon_threads = rayon::current_num_threads(),
        parallel_linear_min_outputs = std::env::var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS")
            .ok()
            .as_deref()
            .unwrap_or("default"),
        apple_accelerate_min_elements = std::env::var("CAMELID_APPLE_ACCELERATE_MIN_ELEMENTS")
            .ok()
            .as_deref()
            .unwrap_or("default(262144 on macOS)"),
        apple_accelerate = cfg!(target_os = "macos"),
        metal_linear = std::env::var("CAMELID_METAL_LINEAR")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        metal_q8 = std::env::var("CAMELID_METAL_Q8")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        metal_q8_retained = std::env::var("CAMELID_METAL_Q8_RETAINED")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        hybrid_q8_retained = std::env::var("CAMELID_HYBRID_Q8_RETAINED")
            .ok()
            .as_deref()
            .unwrap_or("off"),
        hybrid_q8_gpu_percent = std::env::var("CAMELID_HYBRID_Q8_GPU_PERCENT")
            .ok()
            .as_deref()
            .unwrap_or("default(10)"),
        metal_available = metal.available,
        metal_device = metal.device_name.as_deref().unwrap_or("none"),
        metal_note = metal.note.as_deref().unwrap_or(""),
        "camelid acceleration state"
    );
    // Probe CUDA at startup so the selected GPU (index, name, compute capability,
    // VRAM) is logged at launch via cuda::init_backend, and surface availability
    // here. A present CUDA device is always the discrete NVIDIA GPU — the Intel
    // iGPU is not CUDA-capable and is never enumerated.
    let cuda = camelid::cuda::detect_cuda_device();
    tracing::info!(
        cuda_available = cuda.available,
        cuda_device = cuda.device_name.as_deref().unwrap_or("none"),
        cuda_reason = cuda.reason.as_deref().unwrap_or(""),
        "camelid cuda state"
    );
}

// Physical-core detection moved into the library (the decode thread-policy
// default needs it too); this alias keeps the call sites below unchanged.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
use camelid::inference::windows_physical_core_count;

/// §1.2 core-cap: clamp a resolved compute-thread count so the OS always keeps
/// its reserve (see [`camelid::gait::compute_thread_budget`]). Windows x86-64
/// only — the GAIT substrate's scope; elsewhere `requested` is returned
/// unchanged. A `None` request resolves to the full safe budget.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn host_safe_thread_count(requested: Option<usize>) -> Option<usize> {
    let phys = windows_physical_core_count()?;
    let budget = camelid::gait::compute_thread_budget(phys);
    Some(
        requested
            .map(|r| r.min(budget.threads))
            .unwrap_or(budget.threads),
    )
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
fn host_safe_thread_count(requested: Option<usize>) -> Option<usize> {
    requested
}

fn configure_rayon_threads(threads: Option<usize>) -> anyhow::Result<()> {
    if let Some(t) = threads {
        anyhow::ensure!(t > 0, "--threads must be greater than zero");
    }
    // When the caller did not pin a thread count, default the Windows pool to the
    // physical core count (SMT siblings hurt compute-bound decode). Other targets
    // keep their existing defaults.
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    let resolved = threads.or_else(windows_physical_core_count);
    #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
    let resolved = threads;

    // §1.2 host-safety: when GAIT is engaged, cap the pool so the OS keeps a core
    // reserve. Gated on the bring-up flag so the default path is byte-identical;
    // when GAIT becomes the baseline the cap becomes unconditional.
    let resolved = if camelid::gait::gait_enabled() {
        host_safe_thread_count(resolved)
    } else {
        resolved
    };

    #[cfg(target_os = "macos")]
    let should_configure = true;
    #[cfg(not(target_os = "macos"))]
    let should_configure = resolved.is_some();

    if should_configure {
        let mut builder = ThreadPoolBuilder::new();
        if let Some(t) = resolved {
            builder = builder.num_threads(t);
        }
        #[cfg(target_os = "macos")]
        {
            builder = builder.start_handler(|_| {
                unsafe {
                    pthread_set_qos_class_self_np(0x21, 0); // QOS_CLASS_USER_INTERACTIVE (forces P-cores)
                }
            });
        }
        builder
            .build_global()
            .map_err(|err| anyhow::anyhow!("failed to configure Rayon thread pool: {err}"))?;
    }
    Ok(())
}

fn average_timings(timings: &[DenseHotloopBenchTimings]) -> DenseHotloopBenchTimings {
    let mut total = DenseHotloopBenchTimings::zero();
    for timing in timings {
        total.add_assign(*timing);
    }
    total.scale(1.0 / timings.len() as f64)
}

fn min_timings(timings: &[DenseHotloopBenchTimings]) -> DenseHotloopBenchTimings {
    timings.iter().copied().fold(
        DenseHotloopBenchTimings::infinity(),
        DenseHotloopBenchTimings::min,
    )
}

fn max_timings(timings: &[DenseHotloopBenchTimings]) -> DenseHotloopBenchTimings {
    timings.iter().copied().fold(
        DenseHotloopBenchTimings::zero(),
        DenseHotloopBenchTimings::max,
    )
}

impl DenseHotloopBenchTimings {
    fn zero() -> Self {
        Self {
            gate: 0.0,
            up: 0.0,
            activation: 0.0,
            down: 0.0,
            total: 0.0,
        }
    }

    fn infinity() -> Self {
        Self {
            gate: f64::INFINITY,
            up: f64::INFINITY,
            activation: f64::INFINITY,
            down: f64::INFINITY,
            total: f64::INFINITY,
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.gate += other.gate;
        self.up += other.up;
        self.activation += other.activation;
        self.down += other.down;
        self.total += other.total;
    }

    fn scale(self, scale: f64) -> Self {
        Self {
            gate: self.gate * scale,
            up: self.up * scale,
            activation: self.activation * scale,
            down: self.down * scale,
            total: self.total * scale,
        }
    }

    fn min(self, other: Self) -> Self {
        Self {
            gate: self.gate.min(other.gate),
            up: self.up.min(other.up),
            activation: self.activation.min(other.activation),
            down: self.down.min(other.down),
            total: self.total.min(other.total),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            gate: self.gate.max(other.gate),
            up: self.up.max(other.up),
            activation: self.activation.max(other.activation),
            down: self.down.max(other.down),
            total: self.total.max(other.total),
        }
    }
}

#[derive(Debug, Serialize)]
struct TensorDumpFile {
    path: String,
    tensors: Vec<TensorDump>,
}

#[derive(Debug, Serialize)]
struct TensorDump {
    name: String,
    descriptor: TensorDescriptorDump,
    q8_0: Option<Q8Dump>,
    decoded: DecodedTensorDump,
}

#[derive(Debug, Serialize)]
struct TensorDescriptorDump {
    gguf_dimensions: Vec<u64>,
    gguf_dimension_strides: Vec<u64>,
    runtime_shape: Vec<usize>,
    runtime_row_major_strides: Vec<usize>,
    tensor_type: GgufTensorType,
    absolute_offset: u64,
    relative_offset: u64,
    n_bytes: u64,
    element_count: usize,
    block_count: Option<usize>,
    storage_block_size: u64,
    storage_type_size_bytes: u64,
    storage_row_values: u64,
    storage_row_count: u64,
    storage_row_stride_values: u64,
    storage_row_size_bytes: u64,
    storage_row_stride_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Q8Dump {
    block_count: usize,
    scale: NumberStats,
    first_scales: Vec<f32>,
    first_block_quants: Vec<i8>,
    max_abs_scale_block: usize,
    max_abs_scale_block_quants: Vec<i8>,
}

#[derive(Debug, Serialize)]
struct DecodedTensorDump {
    stats: NumberStats,
    first_values: Vec<f32>,
    max_abs_window_start: usize,
    max_abs_window: Vec<f32>,
    rows: Vec<RowDump>,
    logical_token_rows: Vec<LogicalTokenRowDump>,
    descriptor_token_columns: Vec<LogicalTokenRowDump>,
}

#[derive(Debug, Serialize)]
struct RowDump {
    row: usize,
    start: usize,
    len: usize,
    first_values: Vec<f32>,
    max_abs_window_start: usize,
    max_abs_window: Vec<f32>,
    q8_0_blocks: Vec<Q8BlockDump>,
    q8_0_value_checks: Vec<Q8ValueCheckDump>,
}

#[derive(Debug, Serialize)]
struct LogicalTokenRowDump {
    token_id: usize,
    start: usize,
    stride: usize,
    len: usize,
    source_layout: &'static str,
    first_values: Vec<f32>,
    max_abs_window_start: usize,
    max_abs_window: Vec<f32>,
    q8_0_blocks: Vec<Q8BlockDump>,
    q8_0_value_checks: Vec<Q8ValueCheckDump>,
}

#[derive(Debug, Serialize)]
struct Q8BlockDump {
    block: usize,
    value_start: usize,
    scale: f32,
    quant_values: Vec<i8>,
    dequantized_values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct Q8ValueCheckDump {
    element_index: usize,
    block: usize,
    block_offset: usize,
    scale: f32,
    quant_value: i8,
    dequantized: f32,
    decoded: f32,
    absolute_delta: f32,
}

#[derive(Debug, Serialize)]
struct NumberStats {
    min: f32,
    max: f32,
    mean: f64,
    rms: f64,
    max_abs: f32,
    max_abs_index: usize,
}

fn dump_tensor(
    store: &TensorStore,
    name: &str,
    window: usize,
    rows: &[usize],
    tokens: &[usize],
) -> anyhow::Result<TensorDump> {
    let desc = store.descriptor(name)?.clone();
    let tensor = store.load_cpu_f32(name)?;
    let bytes = store.tensor_bytes(name)?;
    let element_count = tensor.shape.element_count()?;
    let block_count = desc.tensor_type.layout().and_then(|(block_size, _)| {
        if block_size > 1 {
            usize::try_from(block_size)
                .ok()
                .map(|size| element_count / size)
        } else {
            None
        }
    });
    let row_dumps = dump_rows(
        &tensor.data,
        &tensor.shape.dims,
        &desc.tensor_type,
        &bytes,
        rows,
        window,
    )?;
    let logical_token_rows = dump_logical_token_rows(
        name,
        &tensor.data,
        &tensor.shape.dims,
        &desc.tensor_type,
        &bytes,
        tokens,
        window,
    )?;
    let descriptor_token_columns = dump_descriptor_token_columns(
        name,
        &tensor.data,
        &tensor.shape.dims,
        &desc.tensor_type,
        &bytes,
        tokens,
        window,
    )?;
    let storage = tensor_storage_layout(&desc.dimensions, desc.tensor_type)?;
    Ok(TensorDump {
        name: name.to_string(),
        descriptor: TensorDescriptorDump {
            gguf_dimension_strides: gguf_dimension_strides(&desc.dimensions),
            gguf_dimensions: desc.dimensions,
            runtime_row_major_strides: row_major_strides(&tensor.shape.dims),
            runtime_shape: tensor.shape.dims.clone(),
            tensor_type: desc.tensor_type,
            absolute_offset: desc.absolute_offset,
            relative_offset: desc.relative_offset,
            n_bytes: desc.n_bytes,
            element_count,
            block_count,
            storage_block_size: storage.block_size,
            storage_type_size_bytes: storage.type_size_bytes,
            storage_row_values: storage.row_values,
            storage_row_count: storage.row_count,
            storage_row_stride_values: storage.row_stride_values,
            storage_row_size_bytes: storage.row_size_bytes,
            storage_row_stride_bytes: storage.row_stride_bytes,
        },
        q8_0: match desc.tensor_type {
            GgufTensorType::Q8_0 => Some(dump_q8_0(&bytes, window)?),
            _ => None,
        },
        decoded: DecodedTensorDump {
            stats: number_stats(&tensor.data),
            first_values: tensor.data.iter().copied().take(window).collect(),
            max_abs_window_start: max_abs_window_start(&tensor.data, window),
            max_abs_window: window_around_max_abs(&tensor.data, window),
            rows: row_dumps,
            logical_token_rows,
            descriptor_token_columns,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TensorStorageLayoutDump {
    block_size: u64,
    type_size_bytes: u64,
    row_values: u64,
    row_count: u64,
    row_stride_values: u64,
    row_size_bytes: u64,
    row_stride_bytes: u64,
}

fn tensor_storage_layout(
    dimensions: &[u64],
    tensor_type: GgufTensorType,
) -> anyhow::Result<TensorStorageLayoutDump> {
    let (block_size, type_size_bytes) = tensor_type
        .layout()
        .ok_or_else(|| anyhow::anyhow!("unsupported tensor type {tensor_type:?}"))?;
    let row_values = *dimensions.first().unwrap_or(&1);
    if !row_values.is_multiple_of(block_size) {
        anyhow::bail!(
            "first tensor dimension {row_values} is not divisible by block size {block_size}"
        );
    }
    let row_count = dimensions.iter().skip(1).try_fold(1u64, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| anyhow::anyhow!("tensor storage row-count overflow"))
    })?;
    let row_size_bytes = row_values
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size_bytes))
        .ok_or_else(|| anyhow::anyhow!("tensor storage row-size overflow"))?;
    Ok(TensorStorageLayoutDump {
        block_size,
        type_size_bytes,
        row_values,
        row_count,
        row_stride_values: row_values,
        row_size_bytes,
        row_stride_bytes: row_size_bytes,
    })
}

fn dump_rows(
    values: &[f32],
    shape: &[usize],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    rows: &[usize],
    window: usize,
) -> anyhow::Result<Vec<RowDump>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    if shape.len() != 2 {
        anyhow::bail!("--row requires 2D tensors, got shape {shape:?}");
    }
    let row_count = shape[0];
    let row_len = shape[1];
    let mut dumps = Vec::with_capacity(rows.len());
    for row in rows {
        if *row >= row_count {
            anyhow::bail!("row {row} out of range for shape {shape:?}");
        }
        let start = row * row_len;
        let slice = &values[start..start + row_len];
        let max_abs_offset = max_abs_window_start(slice, window);
        let q8_value_indices = sampled_q8_indices(start, row_len, 1, max_abs_offset, window);
        dumps.push(RowDump {
            row: *row,
            start,
            len: row_len,
            first_values: slice.iter().copied().take(window).collect(),
            max_abs_window_start: start + max_abs_offset,
            max_abs_window: window_around_max_abs(slice, window),
            q8_0_blocks: dump_q8_0_blocks_for_range(tensor_type, bytes, start, row_len, window)?,
            q8_0_value_checks: dump_q8_0_value_checks(
                tensor_type,
                bytes,
                values,
                q8_value_indices,
            )?,
        });
    }
    Ok(dumps)
}

fn dump_logical_token_rows(
    name: &str,
    values: &[f32],
    shape: &[usize],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    tokens: &[usize],
    window: usize,
) -> anyhow::Result<Vec<LogicalTokenRowDump>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    if shape.len() != 2 {
        anyhow::bail!("--token requires 2D tensors, got {name} shape {shape:?}");
    }
    let Some(layout) = logical_token_row_layout(name, shape) else {
        return Ok(Vec::new());
    };
    dump_token_rows_for_layout(values, tensor_type, bytes, tokens, window, layout)
}

fn dump_descriptor_token_columns(
    name: &str,
    values: &[f32],
    shape: &[usize],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    tokens: &[usize],
    window: usize,
) -> anyhow::Result<Vec<LogicalTokenRowDump>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let Some(layout) = descriptor_token_column_layout(name, shape) else {
        return Ok(Vec::new());
    };
    dump_token_rows_for_layout(values, tensor_type, bytes, tokens, window, layout)
}

fn dump_token_rows_for_layout(
    values: &[f32],
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    tokens: &[usize],
    window: usize,
    layout: LogicalTokenRowLayout,
) -> anyhow::Result<Vec<LogicalTokenRowDump>> {
    let mut dumps = Vec::with_capacity(tokens.len());
    for token_id in tokens {
        if *token_id >= layout.vocab_size {
            anyhow::bail!(
                "token {token_id} out of range for logical vocab size {}",
                layout.vocab_size
            );
        }
        let start = layout.start_for_token(*token_id);
        let row_values = strided_values(
            values,
            start,
            layout.embedding_width,
            layout.component_stride,
        );
        let max_abs_offset = max_abs_window_start(&row_values, window);
        let q8_value_indices = sampled_q8_indices(
            start,
            layout.embedding_width,
            layout.component_stride,
            max_abs_offset,
            window,
        );
        dumps.push(LogicalTokenRowDump {
            token_id: *token_id,
            start,
            stride: layout.component_stride,
            len: layout.embedding_width,
            source_layout: layout.source_layout,
            first_values: row_values.iter().copied().take(window).collect(),
            max_abs_window_start: start + max_abs_offset * layout.component_stride,
            max_abs_window: row_values
                .iter()
                .copied()
                .skip(max_abs_offset)
                .take(window)
                .collect(),
            q8_0_blocks: dump_q8_0_blocks_for_strided_row(
                tensor_type,
                bytes,
                start,
                layout.embedding_width,
                layout.component_stride,
                max_abs_offset,
                window,
            )?,
            q8_0_value_checks: dump_q8_0_value_checks(
                tensor_type,
                bytes,
                values,
                q8_value_indices,
            )?,
        });
    }
    Ok(dumps)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogicalTokenRowLayout {
    vocab_size: usize,
    embedding_width: usize,
    token_start_stride: usize,
    component_stride: usize,
    source_layout: &'static str,
}

impl LogicalTokenRowLayout {
    fn start_for_token(self, token_id: usize) -> usize {
        token_id * self.token_start_stride
    }
}

fn logical_token_row_layout(name: &str, shape: &[usize]) -> Option<LogicalTokenRowLayout> {
    match name {
        "token_embd.weight" if shape[0] < shape[1] => Some(LogicalTokenRowLayout {
            vocab_size: shape[1],
            embedding_width: shape[0],
            token_start_stride: shape[0],
            component_stride: 1,
            source_layout: "gguf_token_major_shape_reinterpreted",
        }),
        "token_embd.weight" => Some(LogicalTokenRowLayout {
            vocab_size: shape[0],
            embedding_width: shape[1],
            token_start_stride: shape[1],
            component_stride: 1,
            source_layout: "runtime_token_major",
        }),
        "output.weight" if shape[0] < shape[1] => Some(LogicalTokenRowLayout {
            vocab_size: shape[1],
            embedding_width: shape[0],
            token_start_stride: shape[0],
            component_stride: 1,
            source_layout: "gguf_output_token_major_shape_reinterpreted",
        }),
        "output.weight" => Some(LogicalTokenRowLayout {
            vocab_size: shape[0],
            embedding_width: shape[1],
            token_start_stride: shape[1],
            component_stride: 1,
            source_layout: "token_major_output_row",
        }),
        _ => None,
    }
}

fn descriptor_token_column_layout(name: &str, shape: &[usize]) -> Option<LogicalTokenRowLayout> {
    match name {
        "output.weight" if shape.len() == 2 && shape[0] < shape[1] => Some(LogicalTokenRowLayout {
            vocab_size: shape[1],
            embedding_width: shape[0],
            token_start_stride: 1,
            component_stride: shape[1],
            source_layout: "descriptor_output_column",
        }),
        _ => None,
    }
}

fn strided_values(values: &[f32], start: usize, len: usize, stride: usize) -> Vec<f32> {
    (0..len).map(|idx| values[start + idx * stride]).collect()
}

fn gguf_dimension_strides(dims: &[u64]) -> Vec<u64> {
    let mut stride = 1u64;
    let mut strides = Vec::with_capacity(dims.len());
    for dim in dims {
        strides.push(stride);
        stride = stride.saturating_mul(*dim);
    }
    strides
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; dims.len()];
    let mut stride = 1usize;
    for idx in (0..dims.len()).rev() {
        strides[idx] = stride;
        stride = stride.saturating_mul(dims[idx]);
    }
    strides
}

fn dump_q8_0(bytes: &[u8], window: usize) -> anyhow::Result<Q8Dump> {
    const BLOCK_BYTES: usize = 34;
    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        anyhow::bail!(
            "q8_0 byte length {} is not divisible by {BLOCK_BYTES}",
            bytes.len()
        );
    }
    let mut scales = Vec::with_capacity(bytes.len() / BLOCK_BYTES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        scales.push(f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]])));
    }
    let max_abs_scale_block = number_stats(&scales).max_abs_index;
    let first_block_quants = block_quants(bytes, 0, window);
    let max_abs_scale_block_quants = block_quants(bytes, max_abs_scale_block, window);
    Ok(Q8Dump {
        block_count: scales.len(),
        scale: number_stats(&scales),
        first_scales: scales.iter().copied().take(window).collect(),
        first_block_quants,
        max_abs_scale_block,
        max_abs_scale_block_quants,
    })
}

fn block_quants(bytes: &[u8], block_idx: usize, window: usize) -> Vec<i8> {
    const BLOCK_BYTES: usize = 34;
    let start = block_idx * BLOCK_BYTES + 2;
    bytes[start..start + 32]
        .iter()
        .copied()
        .map(|value| value as i8)
        .take(window)
        .collect()
}

fn dump_q8_0_blocks_for_range(
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    start: usize,
    len: usize,
    window: usize,
) -> anyhow::Result<Vec<Q8BlockDump>> {
    if *tensor_type != GgufTensorType::Q8_0 || len == 0 {
        return Ok(Vec::new());
    }
    dump_q8_0_blocks(bytes, [start, start + len - 1], window)
}

fn dump_q8_0_blocks_for_strided_row(
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    start: usize,
    len: usize,
    stride: usize,
    max_abs_offset: usize,
    window: usize,
) -> anyhow::Result<Vec<Q8BlockDump>> {
    if *tensor_type != GgufTensorType::Q8_0 || len == 0 {
        return Ok(Vec::new());
    }
    let first_indices = (0..len.min(window)).map(|offset| start + offset * stride);
    let max_window_end = len.min(max_abs_offset.saturating_add(window));
    let max_indices = (max_abs_offset..max_window_end).map(|offset| start + offset * stride);
    dump_q8_0_blocks(bytes, first_indices.chain(max_indices), window)
}

fn dump_q8_0_blocks(
    bytes: &[u8],
    indices: impl IntoIterator<Item = usize>,
    window: usize,
) -> anyhow::Result<Vec<Q8BlockDump>> {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let mut blocks = Vec::new();
    for index in indices {
        let block = index / BLOCK_VALUES;
        if blocks.iter().any(|dump: &Q8BlockDump| dump.block == block) {
            continue;
        }
        let byte_start = block * BLOCK_BYTES;
        if byte_start + BLOCK_BYTES > bytes.len() {
            anyhow::bail!(
                "q8_0 block {block} exceeds tensor byte length {}",
                bytes.len()
            );
        }
        let scale = f16_bits_to_f32(u16::from_le_bytes([
            bytes[byte_start],
            bytes[byte_start + 1],
        ]));
        let quant_values = bytes[byte_start + 2..byte_start + BLOCK_BYTES]
            .iter()
            .copied()
            .map(|value| value as i8)
            .take(window)
            .collect::<Vec<_>>();
        blocks.push(Q8BlockDump {
            block,
            value_start: block * BLOCK_VALUES,
            scale,
            dequantized_values: quant_values
                .iter()
                .map(|value| scale * f32::from(*value))
                .collect(),
            quant_values,
        });
    }
    Ok(blocks)
}

fn sampled_q8_indices(
    start: usize,
    len: usize,
    stride: usize,
    max_abs_offset: usize,
    window: usize,
) -> Vec<usize> {
    if len == 0 || window == 0 {
        return Vec::new();
    }
    let first_indices = (0..len.min(window)).map(|offset| start + offset * stride);
    let max_window_end = len.min(max_abs_offset.saturating_add(window));
    let max_indices = (max_abs_offset..max_window_end).map(|offset| start + offset * stride);
    dedup_usize_preserving_order(first_indices.chain(max_indices).collect())
}

fn dedup_usize_preserving_order(values: Vec<usize>) -> Vec<usize> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if !out.contains(&value) {
            out.push(value);
        }
    }
    out
}

fn dump_q8_0_value_checks(
    tensor_type: &GgufTensorType,
    bytes: &[u8],
    values: &[f32],
    indices: Vec<usize>,
) -> anyhow::Result<Vec<Q8ValueCheckDump>> {
    if *tensor_type != GgufTensorType::Q8_0 {
        return Ok(Vec::new());
    }
    let mut checks = Vec::with_capacity(indices.len());
    for element_index in indices {
        checks.push(q8_0_value_check(bytes, values, element_index)?);
    }
    Ok(checks)
}

fn q8_0_value_check(
    bytes: &[u8],
    values: &[f32],
    element_index: usize,
) -> anyhow::Result<Q8ValueCheckDump> {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if element_index >= values.len() {
        anyhow::bail!(
            "q8_0 value index {element_index} exceeds decoded tensor length {}",
            values.len()
        );
    }
    let block = element_index / BLOCK_VALUES;
    let block_offset = element_index % BLOCK_VALUES;
    let byte_start = block * BLOCK_BYTES;
    if byte_start + BLOCK_BYTES > bytes.len() {
        anyhow::bail!(
            "q8_0 block {block} for value index {element_index} exceeds tensor byte length {}",
            bytes.len()
        );
    }
    let scale = f16_bits_to_f32(u16::from_le_bytes([
        bytes[byte_start],
        bytes[byte_start + 1],
    ]));
    let quant_value = bytes[byte_start + 2 + block_offset] as i8;
    let dequantized = scale * f32::from(quant_value);
    let decoded = values[element_index];
    Ok(Q8ValueCheckDump {
        element_index,
        block,
        block_offset,
        scale,
        quant_value,
        dequantized,
        decoded,
        absolute_delta: (decoded - dequantized).abs(),
    })
}

fn number_stats(values: &[f32]) -> NumberStats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut square_sum = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_abs_index = 0usize;
    for (idx, value) in values.iter().copied().enumerate() {
        min = min.min(value);
        max = max.max(value);
        sum += f64::from(value);
        square_sum += f64::from(value) * f64::from(value);
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_index = idx;
        }
    }
    let len = values.len() as f64;
    NumberStats {
        min,
        max,
        mean: sum / len,
        rms: (square_sum / len).sqrt(),
        max_abs,
        max_abs_index,
    }
}

fn max_abs_window_start(values: &[f32], window: usize) -> usize {
    if values.is_empty() || window == 0 {
        return 0;
    }
    let max_idx = number_stats(values).max_abs_index;
    max_idx.min(values.len().saturating_sub(window))
}

fn window_around_max_abs(values: &[f32], window: usize) -> Vec<f32> {
    let start = max_abs_window_start(values, window);
    values.iter().copied().skip(start).take(window).collect()
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = u32::try_from(e + 127).expect("subnormal f16 exponent in range");
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

// ---------------------------------------------------------------------------
// BASALT Phase 3 forced-decode harness helpers (basalt_eval_protocol.md §5.1) —
// `camelid gemma4-generate --force-tokens/--dump-step-logits` records. Pure
// CLI-side plumbing: no engine math lives here.
// ---------------------------------------------------------------------------

/// One recorded step of the §5.1 harness: the argmax (id, logit) of the step's
/// full logit vector plus its top-32 excerpt (§5.3 bundle convention), and the
/// teacher-forced token when in forced mode.
#[derive(Debug, Serialize)]
struct Gemma4StepRecord {
    step: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    forced_id: Option<u32>,
    argmax_id: u32,
    argmax_logit: f32,
    /// Top-32 (token id, logit) pairs, logit-descending (ties: lower id first).
    top32: Vec<(u32, f32)>,
}

/// The harness `meta.json` (and forced-mode stdout) document.
#[derive(Debug, Serialize)]
struct Gemma4StepMeta {
    protocol: &'static str,
    /// "forced" (--force-tokens) or "greedy" (--dump-step-logits alone).
    mode: &'static str,
    model: String,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    vocab_size: usize,
    step_count: usize,
    logits_dtype: &'static str,
    logits_file_pattern: &'static str,
    steps: Vec<Gemma4StepRecord>,
}

/// Parse a `--force-tokens` file: either one JSON array of token ids
/// (`[5, 6, 7]`) or newline-separated decimal ids (blank lines, CR, and a BOM
/// tolerated). Empty files are an error — a forced decode with zero steps is
/// always a harness mistake.
fn parse_forced_tokens(text: &str) -> Result<Vec<u32>, String> {
    let trimmed = text.trim_start_matches('\u{feff}').trim();
    if trimmed.is_empty() {
        return Err("forced-token file is empty".into());
    }
    if trimmed.starts_with('[') {
        let ids = serde_json::from_str::<Vec<u32>>(trimmed)
            .map_err(|e| format!("forced-token JSON parse failed: {e}"))?;
        // The JSON branch must not bypass the emptiness guard above: `[]`
        // parses fine but a zero-step forced decode is always a harness mistake.
        if ids.is_empty() {
            return Err("forced-token list is empty".into());
        }
        return Ok(ids);
    }
    trimmed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.parse::<u32>()
                .map_err(|e| format!("bad forced token id {l:?}: {e}"))
        })
        .collect()
}

/// Refuse forced token ids outside the model's vocab (BASALT Amendment 3 review
/// fix): an out-of-range id would panic (or silently mis-embed) deep inside the
/// forward step, so it is validated at the CLI call site — the first point where
/// the vocab size is known post-load. Names the offending id and step.
fn validate_forced_token_vocab(ids: &[u32], vocab: usize) -> Result<(), String> {
    match ids.iter().enumerate().find(|(_, &id)| id as usize >= vocab) {
        Some((step, &id)) => Err(format!(
            "forced token id {id} at step {step} is out of range for this model's \
             vocab size {vocab}"
        )),
        None => Ok(()),
    }
}

/// Refuse a non-empty existing `--dump-step-logits` directory (BASALT
/// Amendment 3 review fix): `step_<i>.bin` files from a previous run would
/// silently mix with this run's dumps and corrupt the §5.3 exact-KL input.
/// A missing directory is fine (created after this check); an existing empty
/// directory is fine; anything else is a named error listing the offending dir.
fn ensure_dump_dir_empty(dir: &std::path::Path) -> Result<(), String> {
    match std::fs::read_dir(dir) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                Err(format!(
                    "--dump-step-logits directory {} already exists and is not empty; \
                     refusing to mix step dumps with pre-existing files (pass a fresh \
                     or empty directory)",
                    dir.display()
                ))
            } else {
                Ok(())
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        // Exists but is not a listable directory (e.g. a plain file).
        Err(e) => Err(format!(
            "--dump-step-logits {} is not usable as a dump directory: {e}",
            dir.display()
        )),
    }
}

/// Per-step argmax (id, logit) with `Gemma4Runtime::generate_greedy`'s EXACT
/// tie convention (`max_by` + `partial_cmp`: the last of equal maxima wins), so
/// the recorded argmax is the token the greedy decoder would emit at this step.
fn greedy_argmax(logits: &[f32]) -> (u32, f32) {
    let (i, v) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .expect("non-empty logits");
    (i as u32, *v)
}

/// Top-`n` (id, logit) pairs by logit descending, ties broken by lower id —
/// deterministic (`total_cmp`) and independent of the argmax tie convention
/// above (the two can differ on an exact tie; the argmax field is authoritative
/// for greedy-parity questions).
fn top_n_logits(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let take = n.min(logits.len());
    if take == 0 {
        return Vec::new();
    }
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| {
        logits[*b as usize]
            .total_cmp(&logits[*a as usize])
            .then(a.cmp(b))
    };
    if take < ids.len() {
        ids.select_nth_unstable_by(take - 1, cmp);
        ids.truncate(take);
    }
    ids.sort_unstable_by(cmp);
    ids.into_iter().map(|i| (i, logits[i as usize])).collect()
}

/// Write one step's FULL logit vector as raw little-endian f32 bytes
/// (`step_<i>.bin` — the §5.3 exact-KL input; dumps are temporary, the bundle
/// keeps the meta.json top-32 excerpts).
fn write_step_logits(dir: &std::path::Path, step: usize, logits: &[f32]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(logits.len() * 4);
    for v in logits {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(format!("step_{step}.bin")), buf)
}

#[cfg(test)]
mod windowed_arch_cli_lane_tests {
    use super::*;
    use camelid::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};

    fn gguf_for(arch: &str, quant: GgufTensorType) -> GgufFile {
        GgufFile {
            path: std::path::PathBuf::from("/models/synthetic.gguf"),
            version: 3,
            tensor_count: 0,
            metadata_count: 1,
            alignment: 32,
            data_start_offset: 0,
            metadata: std::collections::BTreeMap::from([(
                "general.architecture".to_string(),
                GgufMetadataValue::String(arch.to_string()),
            )]),
            tensors: [
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_down.weight",
            ]
            .into_iter()
            .map(|name| GgufTensorDescriptor {
                name: name.to_string(),
                dimensions: vec![32, 32],
                tensor_type: quant,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 0,
            })
            .collect(),
        }
    }

    /// Phase 3c finding F2: the Phase 3b flip made this guard capability-aware
    /// for gemma3, which opened EVERY CLI lane it protects — including
    /// distribute master/worker, ghost and the speculative benches, whose
    /// forwards walk the CPU dense layer loop directly and can therefore never
    /// run a windowed arch on any host. Those lanes must refuse before weights
    /// load, naming `camelid serve`, rather than accepting the model and dying
    /// at the H4 choke point after a multi-gigabyte load.
    ///
    /// Host-independent: `CpuDenseOnly` never consults the capability probe.
    #[test]
    fn cpu_dense_only_cli_lanes_refuse_a_windowed_arch_on_every_host() {
        let gguf = gguf_for("gemma3", GgufTensorType::Q8_0);
        let err =
            ensure_arch_has_direct_dense_session(&gguf, DenseLaneWindowedForward::CpuDenseOnly)
                .expect_err("a CPU-dense-only lane must refuse a windowed arch");
        let message = err.to_string();
        assert!(
            message.contains("sliding-window attention"),
            "the refusal must name the hazard: {message}"
        );
        assert!(
            message.contains("camelid serve"),
            "the refusal must be actionable — name the lane that serves it: {message}"
        );

        // Control: a dense arch stays admitted on the very same lane, so the
        // refusal is caused by the window schedule and not by the lane class.
        ensure_arch_has_direct_dense_session(
            &gguf_for("llama", GgufTensorType::Q8_0),
            DenseLaneWindowedForward::CpuDenseOnly,
        )
        .expect("a dense arch must still be admitted on a CPU-dense-only lane");
    }

    /// Phase 3c finding F3 at the CLI: a non-Q8_0 gemma3 has no resident lane
    /// on ANY host (hazard H5), so even a session-decode lane must refuse it
    /// and point at `camelid serve`, whose router falls back to the bridge.
    #[test]
    fn a_kquant_windowed_row_is_refused_even_on_a_session_decode_lane() {
        let err = ensure_arch_has_direct_dense_session(
            &gguf_for("gemma3", GgufTensorType::Q4K),
            DenseLaneWindowedForward::ViaSessionDecode,
        )
        .expect_err("a non-Q8_0 windowed row has no direct dense session anywhere");
        assert!(
            err.to_string().contains("camelid serve"),
            "the refusal must point at the lane that serves it: {err}"
        );
    }

    /// qwen35 / gemma2 / bitnet-b1.58 stay refused on every lane class — the flip did not
    /// touch them, and this is the causality control for the lane split.
    #[test]
    fn runnable_only_archs_stay_refused_on_both_lane_classes() {
        for arch in ["qwen35", "gemma2", "bitnet-b1.58"] {
            for lane in [
                DenseLaneWindowedForward::CpuDenseOnly,
                DenseLaneWindowedForward::ViaSessionDecode,
            ] {
                assert!(
                    ensure_arch_has_direct_dense_session(
                        &gguf_for(arch, GgufTensorType::Q8_0),
                        lane
                    )
                    .is_err(),
                    "{arch} must be refused on every CLI lane class"
                );
            }
        }
    }
}

#[cfg(test)]
mod basalt_forced_decode_tests {
    use super::*;

    fn on_cli_test_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("cli-parse-test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(test)
            .expect("spawn CLI parse test")
            .join()
            .expect("CLI parse test panicked");
    }

    #[test]
    fn gemma4_generate_parses_forced_decode_flags() {
        on_cli_test_stack(|| {
            let cli = Cli::try_parse_from([
                "camelid",
                "gemma4-generate",
                "model.gguf",
                "--force-tokens",
                "toks.txt",
                "--dump-step-logits",
                "dumps",
                "--max-tokens",
                "8",
            ])
            .expect("parse");
            match cli.command {
                Some(Command::Gemma4Generate {
                    path,
                    max_tokens,
                    force_tokens,
                    dump_step_logits,
                    ..
                }) => {
                    assert_eq!(path, PathBuf::from("model.gguf"));
                    assert_eq!(max_tokens, 8);
                    assert_eq!(force_tokens, Some(PathBuf::from("toks.txt")));
                    assert_eq!(dump_step_logits, Some(PathBuf::from("dumps")));
                }
                other => panic!("expected Gemma4Generate, got {other:?}"),
            }
        });
    }

    #[test]
    fn gemma4_generate_harness_flags_default_off() {
        on_cli_test_stack(|| {
            let cli =
                Cli::try_parse_from(["camelid", "gemma4-generate", "model.gguf"]).expect("parse");
            match cli.command {
                Some(Command::Gemma4Generate {
                    prompt,
                    max_tokens,
                    force_tokens,
                    dump_step_logits,
                    ..
                }) => {
                    // Default behavior unchanged: no harness flags, prior defaults intact.
                    assert_eq!(force_tokens, None);
                    assert_eq!(dump_step_logits, None);
                    assert_eq!(prompt, "The capital of France is");
                    assert_eq!(max_tokens, 24);
                }
                other => panic!("expected Gemma4Generate, got {other:?}"),
            }
        });
    }

    #[test]
    fn forced_token_file_parses_newline_and_json_forms() {
        assert_eq!(parse_forced_tokens("5\n6\n7\n").unwrap(), vec![5, 6, 7]);
        assert_eq!(parse_forced_tokens("[5, 6, 7]").unwrap(), vec![5, 6, 7]);
        // CRLF + blank lines + BOM tolerated (Windows-authored token files).
        assert_eq!(
            parse_forced_tokens("\u{feff}5\r\n\r\n6\r\n").unwrap(),
            vec![5, 6]
        );
        assert!(parse_forced_tokens("").is_err());
        assert!(parse_forced_tokens("   \n  ").is_err());
        assert!(parse_forced_tokens("notanid").is_err());
        assert!(parse_forced_tokens("[1, -2]").is_err());
        // Review fix: the JSON branch must not bypass the emptiness guard.
        for empty_json in ["[]", "[ ]"] {
            match parse_forced_tokens(empty_json) {
                Err(e) => assert_eq!(e, "forced-token list is empty", "input {empty_json:?}"),
                Ok(ids) => panic!("empty JSON list {empty_json:?} must error, got {ids:?}"),
            }
        }
    }

    #[test]
    fn forced_token_vocab_validation_names_the_offending_id() {
        // In-range ids (including vocab-1) pass.
        validate_forced_token_vocab(&[0, 5, 261_143], 261_144).expect("in-range ids admit");
        validate_forced_token_vocab(&[], 16).expect("empty list is vacuously in range");
        // First offending id + its step are named.
        let err = validate_forced_token_vocab(&[3, 16, 2], 16).expect_err("16 >= vocab 16");
        assert!(err.contains("forced token id 16"), "{err}");
        assert!(err.contains("at step 1"), "{err}");
        assert!(err.contains("vocab size 16"), "{err}");
        // Boundary: id == vocab is out of range (ids are 0-based).
        assert!(validate_forced_token_vocab(&[8], 8).is_err());
    }

    #[test]
    fn dump_dir_check_refuses_non_empty_existing_directory() {
        let root = tempfile::tempdir().expect("tempdir");

        // Nonexistent path: fine (created later by create_dir_all).
        let fresh = root.path().join("fresh-dumps");
        ensure_dump_dir_empty(&fresh).expect("missing dir is usable");

        // Existing but empty: fine.
        let empty = root.path().join("empty-dumps");
        std::fs::create_dir(&empty).expect("mkdir");
        ensure_dump_dir_empty(&empty).expect("empty dir is usable");

        // Existing with contents: named refusal listing the offending dir.
        let dirty = root.path().join("dirty-dumps");
        std::fs::create_dir(&dirty).expect("mkdir");
        std::fs::write(dirty.join("step_0.bin"), b"stale").expect("write");
        let err = ensure_dump_dir_empty(&dirty).expect_err("non-empty dir must refuse");
        assert!(err.contains("already exists and is not empty"), "{err}");
        assert!(err.contains(&dirty.display().to_string()), "{err}");

        // A plain file at the path is also a named error, not a panic.
        let file_path = root.path().join("not-a-dir");
        std::fs::write(&file_path, b"x").expect("write");
        let err = ensure_dump_dir_empty(&file_path).expect_err("file path must refuse");
        assert!(err.contains("not usable as a dump directory"), "{err}");
    }

    #[test]
    fn step_record_helpers_are_deterministic() {
        let logits = [0.5f32, 2.5, -1.0, 2.5, 0.0];
        // generate_greedy's max_by(partial_cmp) keeps the LAST of equal maxima.
        assert_eq!(greedy_argmax(&logits), (3, 2.5));
        // top-n orders logit-descending with lower-id-first ties.
        assert_eq!(top_n_logits(&logits, 3), vec![(1, 2.5), (3, 2.5), (0, 0.5)]);
        // n larger than vocab clamps.
        assert_eq!(top_n_logits(&logits, 64).len(), 5);
        assert_eq!(top_n_logits(&[], 32), Vec::new());
    }
}

#[cfg(test)]
mod tensor_dump_tests {
    use super::*;

    #[test]
    fn tensor_dump_layer_selection_extends_defaults_without_duplicates() {
        let names = tensor_dump_names(Vec::new(), vec![0, 2]);

        assert_eq!(names[0], "token_embd.weight");
        assert_eq!(names[1], "output.weight");
        assert!(names.contains(&"blk.0.attn_q.weight".to_string()));
        assert!(names.contains(&"blk.2.attn_q.weight".to_string()));
        assert!(names.contains(&"blk.2.ffn_down.weight".to_string()));
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "blk.0.attn_q.weight")
                .count(),
            1
        );
    }

    #[test]
    fn tensor_dump_layer_selection_extends_explicit_tensors() {
        let names = tensor_dump_names(vec!["output.weight".to_string()], vec![2]);

        assert_eq!(names[0], "output.weight");
        assert!(!names.contains(&"token_embd.weight".to_string()));
        assert_eq!(names[1], "blk.2.attn_q.weight");
        assert_eq!(
            names.last().map(String::as_str),
            Some("blk.2.ffn_down.weight")
        );
    }

    #[test]
    fn logical_token_row_layout_reports_embedding_and_output_strides() {
        assert_eq!(
            logical_token_row_layout("token_embd.weight", &[4, 10]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "gguf_token_major_shape_reinterpreted",
            })
        );
        assert_eq!(
            logical_token_row_layout("token_embd.weight", &[10, 4]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "runtime_token_major",
            })
        );
        assert_eq!(
            logical_token_row_layout("output.weight", &[4, 10]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "gguf_output_token_major_shape_reinterpreted",
            })
        );
        assert_eq!(
            descriptor_token_column_layout("output.weight", &[4, 10]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 1,
                component_stride: 10,
                source_layout: "descriptor_output_column",
            })
        );
        assert_eq!(
            logical_token_row_layout("output.weight", &[10, 4]),
            Some(LogicalTokenRowLayout {
                vocab_size: 10,
                embedding_width: 4,
                token_start_stride: 4,
                component_stride: 1,
                source_layout: "token_major_output_row",
            })
        );
    }

    #[test]
    fn serve_nocopy_default_only_with_active_wire_resident_stack() {
        // Default on: fresh (unset) + full wire-resident stack.
        assert!(should_default_serve_nocopy(false, true, true, true));
        // User set it either way (incl. an explicit =0): never override.
        assert!(!should_default_serve_nocopy(true, true, true, true));
        // Speculative decoding on the CPU verify plan (an explicit non-enabling
        // CAMELID_SPEC_GPU) turns resident decode off -> stay off, since that plan
        // needs materialized blocks, not wire pages. With GPU verify armed (the
        // Metal default) resident stays on and this arm does fire.
        assert!(!should_default_serve_nocopy(false, false, true, true));
        // Any wire-stack component off -> the wire kernels can't consume pages.
        assert!(!should_default_serve_nocopy(false, true, false, true));
        assert!(!should_default_serve_nocopy(false, true, true, false));
    }

    /// Phase 0 FIX A. Asking for `--spec-decode ngram` on a Mac used to switch
    /// `CAMELID_METAL_RESIDENT_DECODE`/`_PREFILL` off SERVER-WIDE just because
    /// `CAMELID_SPEC_GPU` was unset, and said so only through a `tracing::info!` that a
    /// stock install (no `RUST_LOG`) never prints — so every speculative measurement
    /// silently timed the CPU repack plan. The auto-arm closes that, without ever
    /// overriding an operator who has spoken.
    #[test]
    fn spec_gpu_auto_arms_on_metal_but_never_overrides_an_explicit_opt_out() {
        // The regression: unset flag, stock fast stack, real Metal device.
        assert!(should_auto_arm_spec_gpu(false, true, false, true));
        // An explicit CAMELID_SPEC_GPU is the operator's call in BOTH directions --
        // including `0`, which must keep selecting the CPU repack plan.
        assert!(!should_auto_arm_spec_gpu(true, true, false, true));
        // No resident decode lane to preserve: `verify_drafts_metal` would decline every
        // round, so don't advertise GPU verify.
        assert!(!should_auto_arm_spec_gpu(false, false, false, true));
        // Deterministic mode fails every GPU gate closed by contract (DECISIONS.md D9).
        assert!(!should_auto_arm_spec_gpu(false, true, true, true));
        // No Metal device -> no resident lane. False on every non-macOS build, which is
        // what keeps CPU-only and CUDA hosts byte-identical to today.
        assert!(!should_auto_arm_spec_gpu(false, true, false, false));
    }

    #[test]
    fn tensor_dump_reports_gguf_and_runtime_strides() {
        assert_eq!(gguf_dimension_strides(&[4, 10, 3]), vec![1, 4, 40]);
        assert_eq!(row_major_strides(&[4, 10, 3]), vec![30, 3, 1]);
    }

    #[test]
    fn tensor_dump_reports_q8_0_storage_row_size_and_stride() {
        let storage = tensor_storage_layout(&[2048, 32000], GgufTensorType::Q8_0)
            .expect("q8 output storage layout");

        assert_eq!(storage.block_size, 32);
        assert_eq!(storage.type_size_bytes, 34);
        assert_eq!(storage.row_values, 2048);
        assert_eq!(storage.row_count, 32000);
        assert_eq!(storage.row_stride_values, 2048);
        assert_eq!(storage.row_size_bytes, 2176);
        assert_eq!(storage.row_stride_bytes, 2176);
        assert_eq!(storage.row_size_bytes * storage.row_count, 69_632_000);
    }

    #[test]
    fn dump_logical_token_rows_samples_prompt_embedding_rows() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let rows = dump_logical_token_rows(
            "token_embd.weight",
            &values,
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[0, 2],
            2,
        )
        .expect("logical token rows");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].token_id, 0);
        assert_eq!(rows[0].start, 0);
        assert_eq!(rows[0].stride, 1);
        assert_eq!(rows[0].len, 3);
        assert_eq!(rows[0].first_values, vec![0.0, 1.0]);
        assert_eq!(rows[1].token_id, 2);
        assert_eq!(rows[1].start, 6);
        assert_eq!(rows[1].first_values, vec![6.0, 7.0]);
        assert!(rows[0].q8_0_blocks.is_empty());
    }

    #[test]
    fn dump_logical_token_rows_samples_output_weight_token_vectors() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let rows = dump_logical_token_rows(
            "output.weight",
            &values,
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[1],
            3,
        )
        .expect("output token rows");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].token_id, 1);
        assert_eq!(rows[0].start, 3);
        assert_eq!(rows[0].stride, 1);
        assert_eq!(rows[0].len, 3);
        assert_eq!(
            rows[0].source_layout,
            "gguf_output_token_major_shape_reinterpreted"
        );
        assert_eq!(rows[0].first_values, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn dump_descriptor_token_columns_samples_output_weight_descriptor_columns() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let rows = dump_descriptor_token_columns(
            "output.weight",
            &values,
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[1],
            3,
        )
        .expect("output descriptor token columns");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].token_id, 1);
        assert_eq!(rows[0].start, 1);
        assert_eq!(rows[0].stride, 4);
        assert_eq!(rows[0].len, 3);
        assert_eq!(rows[0].source_layout, "descriptor_output_column");
        assert_eq!(rows[0].first_values, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn dump_rows_reports_q8_value_checks_for_contiguous_rows() {
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..4 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
            for offset in 0..32 {
                let quant = block as i8 + offset as i8;
                bytes.push(quant as u8);
                values.push(f32::from(quant));
            }
        }

        let rows = dump_rows(&values, &[2, 64], &GgufTensorType::Q8_0, &bytes, &[1], 2)
            .expect("q8 row dump");

        let row = &rows[0];
        assert_eq!(row.row, 1);
        assert_eq!(row.start, 64);
        assert_eq!(row.first_values, vec![2.0, 3.0]);
        assert_eq!(row.max_abs_window_start, 126);
        assert_eq!(row.max_abs_window, vec![33.0, 34.0]);
        assert_eq!(row.q8_0_value_checks.len(), 4);
        assert_eq!(row.q8_0_value_checks[0].element_index, 64);
        assert_eq!(row.q8_0_value_checks[0].block, 2);
        assert_eq!(row.q8_0_value_checks[0].block_offset, 0);
        assert_eq!(row.q8_0_value_checks[0].quant_value, 2);
        assert_eq!(row.q8_0_value_checks[0].decoded, 2.0);
        assert_eq!(row.q8_0_value_checks[0].absolute_delta, 0.0);
        assert_eq!(row.q8_0_value_checks[3].element_index, 127);
        assert_eq!(row.q8_0_value_checks[3].block, 3);
        assert_eq!(row.q8_0_value_checks[3].block_offset, 31);
        assert_eq!(row.q8_0_value_checks[3].dequantized, 34.0);
    }

    #[test]
    fn dump_logical_token_rows_reports_q8_value_checks_for_token_major_output_rows() {
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..8 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
            for offset in 0..32 {
                let quant = block as i8 + offset as i8;
                bytes.push(quant as u8);
                values.push(f32::from(quant));
            }
        }

        let rows = dump_logical_token_rows(
            "output.weight",
            &values,
            &[4, 64],
            &GgufTensorType::Q8_0,
            &bytes,
            &[1],
            2,
        )
        .expect("q8 output token row");

        let row = &rows[0];
        assert_eq!(row.start, 4);
        assert_eq!(row.stride, 1);
        assert_eq!(row.first_values, vec![4.0, 5.0]);
        assert_eq!(row.max_abs_window_start, 6);
        assert_eq!(row.max_abs_window, vec![6.0, 7.0]);
        assert_eq!(row.q8_0_blocks.len(), 1);
        assert_eq!(row.q8_0_blocks[0].block, 0);
        assert_eq!(row.q8_0_blocks[0].value_start, 0);
        assert_eq!(row.q8_0_blocks[0].quant_values, vec![0, 1]);
        assert_eq!(row.q8_0_blocks[0].dequantized_values, vec![0.0, 1.0]);
        assert_eq!(row.q8_0_value_checks.len(), 4);
        assert_eq!(row.q8_0_value_checks[0].element_index, 4);
        assert_eq!(row.q8_0_value_checks[0].block, 0);
        assert_eq!(row.q8_0_value_checks[0].block_offset, 4);
        assert_eq!(row.q8_0_value_checks[0].quant_value, 4);
        assert_eq!(row.q8_0_value_checks[0].dequantized, 4.0);
        assert_eq!(row.q8_0_value_checks[0].decoded, 4.0);
        assert_eq!(row.q8_0_value_checks[0].absolute_delta, 0.0);
        assert_eq!(row.q8_0_value_checks[3].element_index, 7);
        assert_eq!(row.q8_0_value_checks[3].block, 0);
        assert_eq!(row.q8_0_value_checks[3].block_offset, 7);
        assert_eq!(row.q8_0_value_checks[3].quant_value, 7);
    }

    #[test]
    fn dump_descriptor_token_columns_reports_strided_q8_value_checks() {
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..8 {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
            for offset in 0..32 {
                let quant = block as i8 + offset as i8;
                bytes.push(quant as u8);
                values.push(f32::from(quant));
            }
        }

        let rows = dump_descriptor_token_columns(
            "output.weight",
            &values,
            &[4, 64],
            &GgufTensorType::Q8_0,
            &bytes,
            &[1],
            2,
        )
        .expect("q8 output descriptor token column");

        let row = &rows[0];
        assert_eq!(row.start, 1);
        assert_eq!(row.stride, 64);
        assert_eq!(row.first_values, vec![1.0, 3.0]);
        assert_eq!(row.max_abs_window_start, 129);
        assert_eq!(row.max_abs_window, vec![5.0, 7.0]);
        assert_eq!(row.q8_0_blocks.len(), 4);
        assert_eq!(row.q8_0_blocks[0].block, 0);
        assert_eq!(row.q8_0_blocks[0].value_start, 0);
        assert_eq!(row.q8_0_blocks[0].quant_values, vec![0, 1]);
        assert_eq!(row.q8_0_blocks[0].dequantized_values, vec![0.0, 1.0]);
        assert_eq!(row.q8_0_value_checks.len(), 4);
        assert_eq!(row.q8_0_value_checks[0].element_index, 1);
        assert_eq!(row.q8_0_value_checks[0].block, 0);
        assert_eq!(row.q8_0_value_checks[0].block_offset, 1);
        assert_eq!(row.q8_0_value_checks[0].quant_value, 1);
        assert_eq!(row.q8_0_value_checks[0].dequantized, 1.0);
        assert_eq!(row.q8_0_value_checks[0].decoded, 1.0);
        assert_eq!(row.q8_0_value_checks[0].absolute_delta, 0.0);
        assert_eq!(row.q8_0_value_checks[3].element_index, 193);
        assert_eq!(row.q8_0_value_checks[3].block, 6);
        assert_eq!(row.q8_0_value_checks[3].block_offset, 1);
        assert_eq!(row.q8_0_value_checks[3].quant_value, 7);
    }

    #[test]
    fn dump_logical_token_rows_rejects_out_of_range_tokens() {
        let err = dump_logical_token_rows(
            "token_embd.weight",
            &[0.0; 12],
            &[3, 4],
            &GgufTensorType::F32,
            &[],
            &[4],
            2,
        )
        .expect_err("token should be out of range");
        assert!(err.to_string().contains("token 4 out of range"));
    }
}
