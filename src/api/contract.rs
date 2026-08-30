//! Typed API capability contract.
//!
//! `/api/capabilities` is generated from this registry, and the API vertical
//! slice drives the listed routes/modes as executable conformance probes.  Keep
//! implementation claims here rather than embedding free-form feature rows in
//! the response builder.

use super::SupportItem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SupportStatus {
    Partial,
    Supported,
    SupportedCurrentGate,
    SupportedCurrentGateNonStreaming,
    SupportedExactModelRow,
    Unsupported,
}

impl SupportStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Partial => "partial",
            Self::Supported => "supported",
            Self::SupportedCurrentGate => "supported_current_gate",
            Self::SupportedCurrentGateNonStreaming => "supported_current_gate_nonstreaming",
            Self::SupportedExactModelRow => "supported_exact_model_row",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HttpMethod {
    Delete,
    Get,
    Post,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RouteContract {
    pub method: HttpMethod,
    pub path: &'static str,
}

const fn get(path: &'static str) -> RouteContract {
    RouteContract {
        method: HttpMethod::Get,
        path,
    }
}

const fn delete(path: &'static str) -> RouteContract {
    RouteContract {
        method: HttpMethod::Delete,
        path,
    }
}

const fn post(path: &'static str) -> RouteContract {
    RouteContract {
        method: HttpMethod::Post,
        path,
    }
}

/// One feature row plus the route surface that conformance tests must drive.
///
/// `supported_modes` and `unsupported_modes` are deliberately machine-readable:
/// they prevent a broad status word from hiding an important boundary such as
/// "streaming supported, stateful chaining unsupported".
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct ApiConformanceCase {
    pub id: &'static str,
    pub status: SupportStatus,
    pub notes: &'static str,
    pub routes: &'static [RouteContract],
    pub supported_modes: &'static [&'static str],
    pub unsupported_modes: &'static [&'static str],
}

pub(super) const API_CONFORMANCE_CASES: &[ApiConformanceCase] = &[
    ApiConformanceCase {
        id: "camelid_verify",
        status: SupportStatus::Partial,
        notes: "CLI `camelid verify <gguf>` plus GET/POST /api/models/verify replay one pinned deterministic request for a built-in exact-GGUF-hash profile and emit a digest-sealed report. The initial profile covers only the tracked Llama 3.2 1B Instruct Q8_0 Windows artifact. A pass proves that one request on those exact bytes; it is not a digital signature, broad certification, support promotion, or performance claim.",
        routes: &[get("/api/models/verify"), post("/api/models/verify")],
        supported_modes: &["pinned_profile_replay"],
        unsupported_modes: &["arbitrary_model_certification"],
    },
    ApiConformanceCase {
        id: "openai_chat_completions",
        status: SupportStatus::SupportedCurrentGate,
        notes: "POST /v1/chat/completions supports non-streaming and SSE streaming for loaded supported dense GGUF models, including stateless function-tool round trips on tool-capable rows. The exact 27B Prism/Qwen3.5 rows additionally accept one OpenAI image_url content part containing a local PNG/JPEG data URL when their sibling mmproj is resident on macOS Metal or Windows CUDA; /v1/health advertises vision_ready.",
        routes: &[post("/v1/chat/completions")],
        supported_modes: &[
            "nonstreaming",
            "streaming",
            "function_tool_roundtrip",
            "prism_single_image_data_url_metal_or_windows_cuda",
        ],
        unsupported_modes: &[
            "multiple_images",
            "remote_image_urls",
            "audio_or_video_input",
            "vision_tool_combination",
        ],
    },
    ApiConformanceCase {
        id: "openai_responses",
        status: SupportStatus::SupportedCurrentGate,
        notes: "POST /v1/responses supports stateless generation by default plus opt-in local SQLite durability through store:true, previous_response_id, and /v1/conversations. Canonical message/function-call/function-call-output items survive restarts; stored Responses and conversation items can be retrieved or deleted; conversation mutation and idempotent requests are serialized. The same evidence-gated generation core supports text/JSON formats, SSE events, usage, and disconnect cancellation. State is local-only and bounded to 512 items/8 MiB; background mode, hosted built-in tools, multimodal input, cross-process request serialization, and remote OpenAI state remain unsupported.",
        routes: &[
            post("/v1/responses"),
            get("/v1/responses/:id"),
            delete("/v1/responses/:id"),
            post("/v1/conversations"),
            get("/v1/conversations/:id"),
            post("/v1/conversations/:id"),
            delete("/v1/conversations/:id"),
            get("/v1/conversations/:id/items"),
            post("/v1/conversations/:id/items"),
            get("/v1/conversations/:id/items/:item_id"),
            delete("/v1/conversations/:id/items/:item_id"),
        ],
        supported_modes: &[
            "stateless_nonstreaming",
            "stateless_streaming",
            "durable_nonstreaming",
            "durable_streaming",
            "previous_response_id",
            "conversations",
            "response_retrieval_and_deletion",
            "idempotency_key",
            "function_tools",
            "manual_function_call_continuation",
        ],
        unsupported_modes: &[
            "background",
            "hosted_tools",
            "multimodal_input",
            "remote_state",
            "cross_process_request_serialization",
        ],
    },
    ApiConformanceCase {
        id: "streaming_tool_calls",
        status: SupportStatus::SupportedCurrentGate,
        notes: "Tool-capable chat streams emit OpenAI tool_calls deltas with stable call ids, function names, JSON argument strings, and finish_reason=tool_calls without leaking the model's raw tool-call envelope. The Responses adapter maps the same generation to response.output_item and response.function_call_arguments events. Models without a certified tool template remain fail-closed.",
        routes: &[post("/v1/chat/completions"), post("/v1/responses")],
        supported_modes: &["chat_completions_sse", "responses_sse"],
        unsupported_modes: &["uncertified_model_tool_templates"],
    },
    ApiConformanceCase {
        id: "llguidance_structured_outputs",
        status: SupportStatus::SupportedCurrentGateNonStreaming,
        notes: "Non-streaming /v1/chat/completions and stateless /v1/responses constrained decoding through LLGuidance 1.7.6 for JSON object, supported JSON Schema, and LLGuidance/Lark CFG requests. Request forms are mutually exclusive, unsupported schemas/grammars fail closed before generation, and token masks use the loaded tokenizer's exact byte vocabulary. Streaming and raw-completions constrained generation remain typed unsupported.",
        routes: &[post("/v1/chat/completions"), post("/v1/responses")],
        supported_modes: &["chat_nonstreaming", "responses_nonstreaming"],
        unsupported_modes: &["streaming", "raw_completions"],
    },
    ApiConformanceCase {
        id: "openai_embeddings",
        status: SupportStatus::SupportedExactModelRow,
        notes: "POST /v1/embeddings plus /embedding and /embeddings aliases accept one string or a bounded string batch, return finite unit-normalized float vectors and exact tokenizer usage, and support Nomic Matryoshka dimensions. The current gate is the exact Nomic Embed Text v1.5 Q8_0 catalog row; base64 encoding, token-id input, arbitrary encoder families/quants, and GPU execution are unsupported.",
        routes: &[
            post("/v1/embeddings"),
            post("/embedding"),
            post("/embeddings"),
        ],
        supported_modes: &["string", "bounded_string_batch", "float_encoding"],
        unsupported_modes: &["base64_encoding", "token_id_input", "arbitrary_encoder"],
    },
    ApiConformanceCase {
        id: "embedding_similarity_reranking",
        status: SupportStatus::SupportedExactModelRow,
        notes: "POST /rerank, /reranking, /v1/rerank, and /v1/reranking rank bounded string/object documents by cosine similarity using Nomic search_query/search_document prefixes, stable score ordering, optional top_n, and optional returned documents. This is bi-encoder embedding-similarity reranking, not a classifier-head cross-encoder claim.",
        routes: &[
            post("/rerank"),
            post("/reranking"),
            post("/v1/rerank"),
            post("/v1/reranking"),
        ],
        supported_modes: &["string_documents", "object_documents", "top_n"],
        unsupported_modes: &["cross_encoder"],
    },
    ApiConformanceCase {
        id: "web_workspace",
        status: SupportStatus::SupportedCurrentGate,
        notes: "Loopback WebUI only: durable conversations over exactly read_file/list_dir/literal-content search inside one canonical workspace root; allow_writes=true is rejected. When the exact supported Nomic embedding row is also loaded, each session lazily builds a bounded read-only in-memory source index and injects semantically relevant, explicitly untrusted excerpts before model execution; absence/failure degrades to the existing lexical path. Evidence-first extension inventories are derived from successful list_dir observations. Exact prompt-plus-generation budgeting, SQLite/FTS5 retrieval, reversible automatic compaction, turn-scoped cancellation, a 90-second model-step deadline, and model-transition exclusion fail closed. Generation remains available only to supported exact rows with tool_capable=true. No write, shell, network, GUI, subagent, unattended, neighboring-model, persistent vector database, broad retrieval-quality, portability, or throughput claim.",
        routes: &[post("/api/agent/workspace/sessions")],
        supported_modes: &["read_only_tools", "durable_threads"],
        unsupported_modes: &["writes", "shell", "network", "subagents"],
    },
    ApiConformanceCase {
        id: "web_ui_research",
        status: SupportStatus::Supported,
        notes: "POST /api/web/research is a bounded pre-generation research helper for the server-policy-protected embedded Chat UI. A deterministic prompt classifier can combine embedded public HTTP(S) reads with a supplemental Brave HTML search (DuckDuckGo Lite and Bing HTML fallbacks), then returns explicitly untrusted, path-labelled chunks and warning strings for the browser to fit into an ordinary chat request. GitHub repository roots and explicit tree/blob refs preserve requested provenance; API quota failures fall back to bounded public raw/HTML reads. CAMELID_WEB_GITHUB_TOKEN is optional, accepted only from the operator environment, sent only on Camelid-managed repository metadata/tree/README requests to exact HTTPS api.github.com hops, and never permits private-repository ingestion. A short bounded in-memory TTL/ETag cache stores fetched responses only, never generated answers. Targets with credentials, non-default ports, loopback/private/link-local/reserved addresses, unsafe redirects or HTTPS downgrades, binary/oversized responses, and excessive URLs/redirects fail closed. Shared admission caps concurrent and abandoned jobs; one total research deadline covers DNS resolution, redirects, and every curl hop, with a separate process-wide cap on resolver calls that outlive their caller. Provider and partial-fetch failures return a structured HTTP-200 response so local generation can continue. Configured Camelid API credentials are never forwarded, and this path does not advertise tools to or require tool support from the model.",
        routes: &[post("/api/web/research")],
        supported_modes: &[
            "deterministic_auto_classification",
            "direct_public_url_fetch",
            "brave_html_search_with_ddg_lite_and_bing_fallbacks",
            "combined_link_and_supplemental_search",
            "branch_aware_github_public_fallback",
            "fetched_source_ttl_etag_cache",
            "typed_nonblocking_failure",
        ],
        unsupported_modes: &[
            "private_network_targets",
            "credential_forwarding",
            "model_tool_calling",
        ],
    },
    ApiConformanceCase {
        id: "stream_options.include_usage",
        status: SupportStatus::SupportedCurrentGate,
        notes: "Chat-completions streaming only: stream_options.include_usage:true appends one terminal chunk with choices:[] and a usage object {prompt_tokens, completion_tokens, total_tokens} identical to the non-streaming endpoint's counts, then [DONE]. Omitting it is byte-identical to the prior baseline. Malformed/other stream_options shapes and subfields are tolerated and ignored (no error), matching the llama-server acd79d6 oracle; no other stream_options subfield is supported. Evidence: qa/evidence-bundles/stream-options-include-usage-20260623/.",
        routes: &[post("/v1/chat/completions")],
        supported_modes: &["include_usage"],
        unsupported_modes: &["other_stream_options"],
    },
    ApiConformanceCase {
        id: "tokenizer_encode_decode",
        status: SupportStatus::SupportedCurrentGate,
        notes: "Loaded-model tokenizer APIs for supported tokenizer families.",
        routes: &[
            post("/api/models/tokenizer/encode"),
            post("/api/models/tokenizer/decode"),
        ],
        supported_modes: &["encode", "decode"],
        unsupported_modes: &["unloaded_model"],
    },
    ApiConformanceCase {
        id: "llama_server_tokenizer_aliases",
        status: SupportStatus::Partial,
        notes: "POST /tokenize and POST /detokenize are bounded loaded-model tokenizer aliases that return token ids/text, with /tokenize with_pieces=true exposing id/piece objects for supported tokenizer lanes. Arbitrary tokenizer kwargs and broader tokenizer parity remain unsupported.",
        routes: &[post("/tokenize"), post("/detokenize")],
        supported_modes: &["tokenize", "detokenize", "pieces"],
        unsupported_modes: &["arbitrary_tokenizer_kwargs"],
    },
    ApiConformanceCase {
        id: "llama_server_models",
        status: SupportStatus::Partial,
        notes: "GET /models returns a privacy-safe read-only list of currently loaded Camelid models with redacted paths and text-only architecture metadata. POST /models/load is a narrow local-path alias over Camelid's stable /api/models/load path and returns a redacted compatibility response. Router-mode query params such as reload/autoload/model selection, cache listing, POST /models/unload, multimodal metadata, and full llama-server model-management parity remain unsupported.",
        routes: &[get("/models"), post("/models/load"), post("/models/unload")],
        supported_modes: &["list", "local_path_load"],
        unsupported_modes: &["unload", "router_mode", "multimodal_metadata"],
    },
    ApiConformanceCase {
        id: "llama_server_props",
        status: SupportStatus::Partial,
        notes: "GET /props returns read-only public server properties, default generation settings, explicit fail-closed chat_template_caps, chat-template metadata when a model is loaded, and Camelid readiness notes. Local model paths are intentionally redacted, router-mode model/autoload query params and POST /props are unsupported, and this does not imply slot lifecycle, native /completion streaming, generic embedding-model compatibility, or full llama-server WebUI parity.",
        routes: &[get("/props"), post("/props")],
        supported_modes: &["read"],
        unsupported_modes: &["write", "router_mode"],
    },
    ApiConformanceCase {
        id: "llama_server_slots",
        status: SupportStatus::Partial,
        notes: "GET /slots returns one read-only, privacy-safe entry per admissible cooperative streaming slot, matching /props total_slots, with generation readiness, queue depth, decoded-token progress, elapsed/stalled time, and fail_on_no_slot=1 handling that refuses only when every slot is busy. Per-slot task identity and per-slot progress are engine-wide values repeated on the busy entries, declared as per_slot_task_identity and per_slot_progress. Router-mode model/autoload query params, POST /slots, slot save/restore/erase actions, prompt-cache metadata, and cancellation actions remain unsupported.",
        routes: &[get("/slots"), post("/slots")],
        supported_modes: &["read"],
        unsupported_modes: &["write", "cache_actions", "cancellation"],
    },
    ApiConformanceCase {
        id: "production_server_hardening",
        status: SupportStatus::Supported,
        notes: "Startup-resolved bearer/X-API-Key authentication with key-file support, fail-closed non-loopback binding, explicit CORS origin allowlists, optional rustls TLS, request/prompt/generation/download ceilings, and a Prometheus /metrics surface for HTTP/generation/token/cache/queue/slot/RSS/VRAM telemetry. Anonymous same-origin loopback remains the default.",
        routes: &[get("/metrics")],
        supported_modes: &["auth", "cors", "tls", "limits", "metrics"],
        unsupported_modes: &[],
    },
    ApiConformanceCase {
        id: "llama_server_apply_template",
        status: SupportStatus::Partial,
        notes: "POST /apply-template renders loaded-model chat messages to a prompt string without inference. It is scoped to Camelid's supported tokenizer/template renderers and returns typed unsupported errors for unknown request fields or unsupported templates.",
        routes: &[post("/apply-template")],
        supported_modes: &["supported_templates"],
        unsupported_modes: &["unknown_fields", "unsupported_templates"],
    },
    ApiConformanceCase {
        id: "llama_server_completion",
        status: SupportStatus::Partial,
        notes: "POST /completion accepts a narrow non-streaming text-generation subset: text prompts, token-id prompt arrays, n_predict/max_tokens, supported sampler fields, and stop sequences are mapped onto Camelid's existing generation path. Native stream=true chunks, slot selection, cache_prompt controls, llama-server timings shape, rich token probabilities, and full llama-server generation parity remain unsupported.",
        routes: &[post("/completion")],
        supported_modes: &["nonstreaming", "text_prompt", "token_id_prompt"],
        unsupported_modes: &["streaming", "slot_selection", "cache_prompt"],
    },
    ApiConformanceCase {
        id: "fail_closed_native_compatibility_routes",
        status: SupportStatus::Unsupported,
        notes: "Native /infill, /v1/messages, POST /models/unload, POST /slots, and slot cache actions return typed not_implemented errors until real route semantics and backend support exist. Unsupported /models/load router-mode fields and /completion modes remain typed parameter errors. Embedding and reranking routes are separately supported only for the exact evidence-gated Nomic row.",
        routes: &[
            post("/infill"),
            post("/v1/messages"),
            post("/models/unload"),
            post("/slots"),
        ],
        supported_modes: &[],
        unsupported_modes: &["all"],
    },
    ApiConformanceCase {
        id: "multi_choice_generation",
        status: SupportStatus::SupportedCurrentGateNonStreaming,
        notes: "POST /v1/chat/completions and /v1/completions support 1..=8 independent, reproducibly seeded choices in non-streaming mode. Streaming multi-choice, receipts, and logprobs with n>1 remain typed unsupported.",
        routes: &[post("/v1/chat/completions"), post("/v1/completions")],
        supported_modes: &["nonstreaming_1_to_8"],
        unsupported_modes: &["streaming", "receipts", "logprobs"],
    },
    ApiConformanceCase {
        id: "rich_logprobs",
        status: SupportStatus::SupportedCurrentGateNonStreaming,
        notes: "POST /v1/chat/completions supports OpenAI-shaped logprobs/top_logprobs and POST /v1/completions supports legacy parallel-array logprobs for one non-streaming choice. Streaming and n>1 logprobs remain typed unsupported.",
        routes: &[post("/v1/chat/completions"), post("/v1/completions")],
        supported_modes: &["chat_nonstreaming", "completions_nonstreaming"],
        unsupported_modes: &["streaming", "multi_choice"],
    },
];

pub(super) fn api_feature_contract() -> Vec<SupportItem> {
    API_CONFORMANCE_CASES
        .iter()
        .map(|case| SupportItem {
            id: case.id,
            status: case.status.as_str(),
            notes: case.notes,
        })
        .collect()
}

pub(super) fn api_conformance_contract() -> Vec<serde_json::Value> {
    API_CONFORMANCE_CASES
        .iter()
        .map(|case| {
            serde_json::json!({
                "id": case.id,
                "status": case.status.as_str(),
                "routes": case.routes.iter().map(|route| {
                    serde_json::json!({
                        "method": match route.method {
                            HttpMethod::Delete => "DELETE",
                            HttpMethod::Get => "GET",
                            HttpMethod::Post => "POST",
                        },
                        "path": route.path,
                    })
                }).collect::<Vec<_>>(),
                "supported_modes": case.supported_modes,
                "unsupported_modes": case.unsupported_modes,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn conformance_registry_has_unique_ids_and_well_formed_routes() {
        let mut ids = HashSet::new();
        for case in API_CONFORMANCE_CASES {
            assert!(ids.insert(case.id), "duplicate API feature id {}", case.id);
            assert!(!case.routes.is_empty(), "{} has no route probes", case.id);
            for route in case.routes {
                assert!(route.path.starts_with('/'), "invalid route {}", route.path);
            }
            if case.status == SupportStatus::Unsupported {
                assert!(case.supported_modes.is_empty());
                assert!(!case.unsupported_modes.is_empty());
            } else {
                assert!(
                    !case.supported_modes.is_empty(),
                    "{} advertises support without a supported mode",
                    case.id
                );
            }
        }
    }

    #[test]
    fn generated_rows_are_a_lossless_view_of_the_registry() {
        let rows = api_feature_contract();
        assert_eq!(rows.len(), API_CONFORMANCE_CASES.len());
        for (row, case) in rows.iter().zip(API_CONFORMANCE_CASES) {
            assert_eq!(row.id, case.id);
            assert_eq!(row.status, case.status.as_str());
            assert_eq!(row.notes, case.notes);
        }
        let descriptors = api_conformance_contract();
        assert_eq!(descriptors.len(), API_CONFORMANCE_CASES.len());
        assert_eq!(
            descriptors[0]["supported_modes"].as_array().unwrap().len(),
            API_CONFORMANCE_CASES[0].supported_modes.len()
        );
    }
}
