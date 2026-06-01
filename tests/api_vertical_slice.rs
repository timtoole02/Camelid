use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use serde_json::{json, Value};
use tower::ServiceExt;

#[tokio::test]
async fn health_reports_not_generation_ready() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["ok"], true);
    assert_eq!(body["engine"], "camelid");
    assert_eq!(body["loaded_now"], false);
    assert_eq!(body["generation_ready"], false);
}

#[tokio::test]
async fn capabilities_public_contract_omits_local_private_paths() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let serialized = body.to_string();
    for forbidden in [
        "/Users/",
        "/home/",
        "file://",
        "file:\\",
        "/Volumes/",
        "/private/tmp/",
        "C:\\Users\\",
        "C:/Users/",
        "\\Users\\",
    ] {
        assert!(
            !serialized.contains(forbidden),
            "/api/capabilities must not expose local/private path marker {forbidden:?}"
        );
    }
}

#[tokio::test]
async fn props_reports_public_fail_closed_llama_server_shape() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/props")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["total_slots"], 1);
    assert_eq!(body["model_path"], Value::Null);
    assert_eq!(body["model_id"], Value::Null);
    assert_eq!(body["chat_template"], Value::Null);
    assert_eq!(body["default_generation_settings"]["n_ctx"], 0);
    assert_eq!(
        body["default_generation_settings"]["next_token"]["has_next_token"],
        false
    );
    assert_eq!(
        body["camelid"]["compatibility"],
        "partial_llama_server_props_read_only"
    );
    assert_eq!(body["camelid"]["generation_ready"], false);
    assert_eq!(body["camelid"]["model_path_redacted"], true);
    assert!(body["camelid"]["unsupported"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "native_completion"));

    let serialized = body.to_string();
    for forbidden in [
        "/Users/",
        "/home/",
        "file://",
        "file:\\",
        "/Volumes/",
        "/private/tmp/",
        "C:\\Users\\",
        "C:/Users/",
        "\\Users\\",
    ] {
        assert!(
            !serialized.contains(forbidden),
            "/props must not expose local/private path marker {forbidden:?}"
        );
    }
}

#[tokio::test]
async fn models_reports_public_loaded_model_list_shape() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["data"], json!([]));
    assert_eq!(
        body["camelid"]["compatibility"],
        "partial_llama_server_models_read_only"
    );
    assert_eq!(body["camelid"]["scope"], "loaded_models_only");
    assert!(body["camelid"]["unsupported"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "models_load"));
    assert!(body["camelid"]["unsupported"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "router_model_cache_listing"));

    let serialized = body.to_string();
    for forbidden in [
        "/Users/",
        "/home/",
        "file://",
        "file:\\",
        "/Volumes/",
        "/private/tmp/",
        "C:\\Users\\",
        "C:/Users/",
        "\\Users\\",
    ] {
        assert!(
            !serialized.contains(forbidden),
            "/models must not expose local/private path marker {forbidden:?}"
        );
    }
}

#[tokio::test]
async fn llama_server_read_only_routes_reject_router_mode_query_params() {
    for uri in [
        "/models?reload=1",
        "/props?model=tiny&autoload=true",
        "/slots?model=tiny",
    ] {
        let app = camelid::api::router();
        let response = app
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["code"], "unsupported_parameter", "{uri}");
        assert_eq!(body["error"]["param"], "query", "{uri}");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("active-model read-only discovery"));
    }
}

#[tokio::test]
async fn native_compatibility_routes_fail_closed_with_typed_errors() {
    let cases = [
        (
            "POST",
            "/models/load",
            "unsupported_llama_server_models_load",
            "model",
        ),
        (
            "POST",
            "/models/unload",
            "unsupported_llama_server_models_unload",
            "model",
        ),
        ("POST", "/props", "unsupported_llama_server_props", "props"),
        ("POST", "/slots", "unsupported_llama_server_slots", "slots"),
        (
            "GET",
            "/metrics",
            "unsupported_llama_server_metrics",
            "metrics",
        ),
        (
            "POST",
            "/infill",
            "unsupported_llama_server_infill",
            "input",
        ),
        ("POST", "/embedding", "unsupported_embeddings", "input"),
        ("POST", "/embeddings", "unsupported_embeddings", "input"),
        ("POST", "/v1/embeddings", "unsupported_embeddings", "input"),
        ("POST", "/rerank", "unsupported_reranking", "input"),
        ("POST", "/reranking", "unsupported_reranking", "input"),
        ("POST", "/v1/rerank", "unsupported_reranking", "input"),
        ("POST", "/v1/reranking", "unsupported_reranking", "input"),
        ("POST", "/v1/responses", "unsupported_responses", "input"),
        ("POST", "/v1/messages", "unsupported_messages", "input"),
    ];

    for (method, uri, code, param) in cases {
        let app = camelid::api::router();
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{uri}");
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["type"], "not_implemented", "{uri}");
        assert_eq!(body["error"]["code"], code, "{uri}");
        assert_eq!(body["error"]["param"], param, "{uri}");
    }
}

#[tokio::test]
async fn slots_reports_public_fail_closed_llama_server_shape() {
    let app = camelid::api::router();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/slots")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let slots = body.as_array().unwrap();
    assert_eq!(slots.len(), 1);
    let slot = &slots[0];
    assert_eq!(slot["id"], 0);
    assert_eq!(slot["id_task"], -1);
    assert_eq!(slot["n_ctx"], 0);
    assert_eq!(slot["speculative"], false);
    assert_eq!(slot["is_processing"], false);
    assert_eq!(slot["params"]["n_predict"], -1);
    assert_eq!(slot["params"]["stream"], true);
    assert_eq!(slot["next_token"]["has_next_token"], false);
    assert_eq!(
        slot["camelid"]["compatibility"],
        "partial_llama_server_slots_read_only"
    );
    assert_eq!(slot["camelid"]["generation_ready"], false);
    assert_eq!(slot["camelid"]["status"], "unavailable");
    assert!(slot["camelid"]["unsupported"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "slot_cache_save_restore_erase"));

    let serialized = body.to_string();
    for forbidden in [
        "/Users/",
        "/home/",
        "file://",
        "file:\\",
        "/Volumes/",
        "/private/tmp/",
        "C:\\Users\\",
        "C:/Users/",
        "\\Users\\",
    ] {
        assert!(
            !serialized.contains(forbidden),
            "/slots must not expose local/private path marker {forbidden:?}"
        );
    }

    let response = app
        .oneshot(
            Request::builder()
                .uri("/slots?fail_on_no_slot=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "runtime_unavailable");
    assert_eq!(body["error"]["code"], "no_available_slot");
    assert_eq!(body["error"]["param"], "fail_on_no_slot");
}

#[tokio::test]
async fn capabilities_report_support_contract_and_planned_lanes() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(
        body["support_contract"]["current_gate"],
        "Current exact-row support: TinyLlama Q8_0 current gate; Llama 3.2 1B Instruct Q8_0 has checked bounded 512/1024/2048/4096/8192 packs; Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with canonical Ubuntu main-lane API/WebUI refresh at source head e9f926ed1a65 plus checked bounded 512/1024/2048 packs; and Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048 packs where row-specific PASS artifacts exist. Mistral 7B Instruct v0.3 Q8_0 has evidence-only bring-up with checked tokenizer/template, parity, and bounded 512/1024/2048/4096/8192 context artifacts, but support remains fail-closed until API/WebUI support-surface proof is synchronized. Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf has bounded one-token backend MoE runtime evidence only; later 5-token/API/WebUI/RSS promotion-candidate artifacts are superseded by Gate 9A 50-token divergence and a longer-continuation hang, so broad/API/WebUI/frontend readiness remains unsupported. These are exact bounded lanes only; no model-native/larger context beyond the checked packs, arbitrary-template behavior, production throughput, portability, neighboring-row, or broad-family support is implied."
    );
    let q8 = body["supported_quantization"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "Q8_0")
        .unwrap();
    assert_eq!(q8["status"], "supported_current_gate");
    let q8_notes = q8["notes"].as_str().unwrap();
    assert!(q8_notes.contains(
        "exact Llama 3.2 1B Instruct Q8_0 now has checked bounded 512/1024/2048/4096/8192-context packs"
    ));
    assert!(q8_notes.contains(
        "exact Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with canonical Ubuntu main-lane API/WebUI refresh at source head e9f926ed1a65 plus checked bounded 512/1024/2048-context packs"
    ));
    assert!(q8_notes.contains(
        "exact Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048-context packs"
    ));
    assert!(q8_notes.contains("where row-specific PASS artifacts exist"));
    assert!(!q8_notes.contains("8B 1024/2048 remain red"));
    assert!(q8_notes.contains("exact bounded-pack lanes only"));
    assert!(!q8_notes.contains("conditional"));
    assert!(!q8_notes.contains("gated"));
    assert!(body["planned_quantization"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "Q4_K_M/Q5_K_M"));
    let llama_bpe_family = body["supported_model_families"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "llama_bpe_decoder_exact_1b_3b_8b_q8_0")
        .unwrap();
    assert_eq!(
        llama_bpe_family["status"],
        "supported_exact_row_smoke_lanes"
    );
    let llama_bpe_notes = llama_bpe_family["notes"].as_str().unwrap();
    assert!(llama_bpe_notes.contains(
        "exact Llama 3.2 1B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048/4096/8192-context packs"
    ));
    assert!(llama_bpe_notes.contains(
        "exact Llama 3.2 3B Instruct Q8_0 has supported_exact_row_smoke canonical Ubuntu main-lane API/WebUI evidence at source head e9f926ed1a65 plus checked bounded 512/1024/2048-context packs"
    ));
    assert!(llama_bpe_notes.contains(
        "exact Llama 3 8B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048-context packs"
    ));
    assert!(llama_bpe_notes
        .contains("published source/runtime-head 8B 1024/2048 PASS bundle at 8e26be0a73c0"));
    assert!(!llama_bpe_notes.contains("8B 1024/2048 current-head bundle"));
    assert!(!llama_bpe_notes.contains("8B 1024/2048 remain red"));
    assert!(llama_bpe_notes.contains("Broader 50-token"));
    assert!(!llama_bpe_notes.contains("conditional"));
    assert!(!llama_bpe_notes.contains("gated"));
    assert!(body["planned_model_families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "mistral"
            && item["status"] == "active_validation_unsupported"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("v0.1 support remains fail-closed")));
    assert!(body["planned_model_families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "mixtral_moe"
            && item["status"] == "active_validation_partial_runtime"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("bounded one-token exact-row MoE runtime evidence")));
    for id in ["qwen25", "gemma2"] {
        assert!(body["planned_model_families"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["id"] == id && item["status"] == "planned_exact_row_candidate"));
    }
    assert!(body["api_features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "multi_choice_generation" && item["status"] == "unsupported"));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "llama_server_tokenizer_aliases"
            && item["status"] == "partial"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("POST /tokenize and POST /detokenize")
            && item["notes"].as_str().unwrap().contains("with_pieces=true")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "llama_server_models"
            && item["status"] == "partial"
            && item["notes"].as_str().unwrap().contains("GET /models")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("currently loaded Camelid models")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("POST /models/load")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "llama_server_props"
            && item["status"] == "partial"
            && item["notes"].as_str().unwrap().contains("GET /props")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("Local model paths are intentionally redacted")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "llama_server_slots"
            && item["status"] == "partial"
            && item["notes"].as_str().unwrap().contains("GET /slots")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("fail_on_no_slot=1")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("slot save/restore/erase")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "llama_server_apply_template"
            && item["status"] == "partial"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("without inference")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "llama_server_completion"
            && item["status"] == "partial"
            && item["notes"].as_str().unwrap().contains("POST /completion")
            && item["notes"].as_str().unwrap().contains("n_predict")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("token-id prompt arrays")
            && item["notes"].as_str().unwrap().contains("non-streaming")
            && item["notes"].as_str().unwrap().contains("stream=true")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "fail_closed_native_compatibility_routes"
            && item["status"] == "unsupported"
            && item["notes"].as_str().unwrap().contains("/infill")
            && item["notes"].as_str().unwrap().contains("/metrics")
            && item["notes"].as_str().unwrap().contains("/v1/embeddings")
            && item["notes"].as_str().unwrap().contains("/v1/responses")
            && item["notes"].as_str().unwrap().contains("/v1/messages")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("POST /models/load")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("POST /models/unload")
            && item["notes"].as_str().unwrap().contains("POST /slots")
    }));
    let compatibility = body["model_compatibility"].as_array().unwrap();
    let tinyllama = compatibility
        .iter()
        .find(|item| item["id"] == "tinyllama_1_1b_chat_q8_0")
        .unwrap();
    assert_eq!(tinyllama["status"], "supported_current_gate");
    assert_eq!(tinyllama["metadata_parses"], "validated");
    assert_eq!(tinyllama["generation_runs"], "validated");
    assert_eq!(tinyllama["parity_audited"], "validated");
    assert_eq!(
        tinyllama["tested_context"],
        "short_50_token_gate_plus_bounded_512_context_pack"
    );
    assert_eq!(tinyllama["chat_template_renderer"], "tinyllama-marker");
    assert_eq!(
        tinyllama["chat_template_shape_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        tinyllama["chat_template_shape_pack_id"],
        "tinyllama-chat-template-shapes-v1"
    );
    assert_eq!(
        tinyllama["bounded_context_512_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        tinyllama["bounded_context_512_pack_id"],
        "tinyllama-context-512-smoke-v1"
    );
    assert_eq!(tinyllama["bounded_context_window"], 512);
    assert_eq!(tinyllama["bounded_context_1024_pack"], "not_promoted");
    assert_eq!(tinyllama["bounded_context_1024_pack_id"], "not_selected");
    assert_eq!(tinyllama["bounded_context_1024_window"], 1024);
    assert_eq!(tinyllama["bounded_context_2048_pack"], "not_promoted");
    assert_eq!(tinyllama["bounded_context_2048_pack_id"], "not_selected");
    assert_eq!(tinyllama["bounded_context_2048_window"], 2048);
    assert_eq!(tinyllama["bounded_context_4096_pack"], "not_promoted");
    assert_eq!(tinyllama["bounded_context_4096_pack_id"], "not_selected");
    assert_eq!(tinyllama["bounded_context_4096_window"], 4096);
    assert_eq!(tinyllama["latest_checked_bucket"], "direct_chat_smoke");
    assert_eq!(tinyllama["latest_checked_result"], "pass");
    assert_eq!(tinyllama["latest_checked_output"], "Certainly! Here");
    let llama32_1b = compatibility
        .iter()
        .find(|item| item["id"] == "llama32_1b_instruct_q8_0")
        .unwrap();
    assert_eq!(llama32_1b["status"], "supported_exact_row_smoke");
    assert_eq!(llama32_1b["metadata_parses"], "validated");
    assert_eq!(
        llama32_1b["generation_runs"],
        "api_completion_and_chat_smoke_validated"
    );
    assert_eq!(llama32_1b["frontend_load_path_verified"], "validated");
    assert_eq!(
        llama32_1b["tested_context"],
        "short_api_webui_smoke_plus_first_512_second_1024_third_2048_fourth_4096_and_fifth_8192_context_packs"
    );
    assert_eq!(
        llama32_1b["chat_template_renderer"],
        "metadata_jinja_supported_for_exact_row"
    );
    assert_eq!(
        llama32_1b["chat_template_shape_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        llama32_1b["chat_template_shape_pack_id"],
        "llama3-chat-template-shapes-v1"
    );
    assert_eq!(
        llama32_1b["bounded_context_512_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        llama32_1b["bounded_context_512_pack_id"],
        "llama3-context-512-smoke-v1"
    );
    assert_eq!(llama32_1b["bounded_context_window"], 512);
    assert_eq!(
        llama32_1b["bounded_context_1024_pack"],
        "validated_second_pack"
    );
    assert_eq!(
        llama32_1b["bounded_context_1024_pack_id"],
        "llama3-context-1024-smoke-v1"
    );
    assert_eq!(llama32_1b["bounded_context_1024_window"], 1024);
    assert_eq!(
        llama32_1b["bounded_context_2048_pack"],
        "validated_third_pack"
    );
    assert_eq!(
        llama32_1b["bounded_context_2048_pack_id"],
        "llama3-context-2048-smoke-v1"
    );
    assert_eq!(llama32_1b["bounded_context_2048_window"], 2048);
    assert_eq!(
        llama32_1b["bounded_context_4096_pack"],
        "validated_fourth_pack"
    );
    assert_eq!(
        llama32_1b["bounded_context_4096_pack_id"],
        "llama3-context-4096-smoke-v1"
    );
    assert_eq!(llama32_1b["bounded_context_4096_window"], 4096);
    assert_eq!(
        llama32_1b["bounded_context_8192_pack"],
        "validated_fifth_pack"
    );
    assert_eq!(
        llama32_1b["bounded_context_8192_pack_id"],
        "llama3-context-8192-smoke-v1"
    );
    assert_eq!(llama32_1b["bounded_context_8192_window"], 8192);
    assert_eq!(
        llama32_1b["latest_checked_bucket"],
        "llama3-context-8192-smoke-v1"
    );
    assert_eq!(llama32_1b["latest_checked_result"], "pass");
    assert_eq!(llama32_1b["latest_checked_output"], "CMLD-819");
    assert!(llama32_1b["evidence"]
        .as_str()
        .unwrap()
        .contains("fifth bounded 8192-context parity on current head"));
    let llama32_3b = compatibility
        .iter()
        .find(|item| item["id"] == "llama32_3b_instruct_q8_0")
        .unwrap();
    assert_eq!(llama32_3b["status"], "supported_exact_row_smoke");
    assert_eq!(
        llama32_3b["generation_runs"],
        "api_completion_and_chat_smoke_plus_five_prompt_api_smoke"
    );
    assert_eq!(llama32_3b["frontend_load_path_verified"], "validated");
    assert_eq!(
        llama32_3b["tested_context"],
        "short_api_webui_smoke_with_broader_prompt_pack_parity_plus_first_512_second_1024_and_third_2048_context_packs"
    );
    assert_eq!(
        llama32_3b["chat_template_renderer"],
        "metadata_jinja_supported_for_exact_row"
    );
    assert_eq!(
        llama32_3b["chat_template_shape_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        llama32_3b["chat_template_shape_pack_id"],
        "llama3-chat-template-shapes-v1"
    );
    assert_eq!(
        llama32_3b["bounded_context_512_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        llama32_3b["bounded_context_512_pack_id"],
        "llama3-context-512-smoke-v1"
    );
    assert_eq!(llama32_3b["bounded_context_window"], 512);
    assert_eq!(
        llama32_3b["bounded_context_1024_pack"],
        "validated_second_pack"
    );
    assert_eq!(
        llama32_3b["bounded_context_1024_pack_id"],
        "llama3-context-1024-smoke-v1"
    );
    assert_eq!(llama32_3b["bounded_context_1024_window"], 1024);
    assert_eq!(
        llama32_3b["bounded_context_2048_pack"],
        "validated_third_pack"
    );
    assert_eq!(
        llama32_3b["bounded_context_2048_pack_id"],
        "llama3-context-2048-smoke-v1"
    );
    assert_eq!(llama32_3b["bounded_context_2048_window"], 2048);
    assert_eq!(llama32_3b["bounded_context_4096_pack"], "not_promoted");
    assert_eq!(llama32_3b["bounded_context_4096_pack_id"], "not_selected");
    assert_eq!(llama32_3b["bounded_context_4096_window"], 4096);
    assert_eq!(
        llama32_3b["latest_checked_bucket"],
        "llama3-context-2048-smoke-v1"
    );
    assert_eq!(llama32_3b["latest_checked_result"], "pass");
    assert_eq!(llama32_3b["latest_checked_output"], "CMLD-204");
    let llama3 = compatibility
        .iter()
        .find(|item| item["id"] == "llama3_8b_instruct_q8_0")
        .unwrap();
    assert_eq!(llama3["status"], "supported_exact_row_smoke");
    assert_eq!(
        llama3["metadata_parses"],
        "real_artifact_inspected_and_config_guarded"
    );
    assert_eq!(
        llama3["tokenizer_works"],
        "validated_for_compact_llama3_bpe"
    );
    assert_eq!(
        llama3["generation_runs"],
        "api_completion_and_chat_smoke_validated"
    );
    assert_eq!(llama3["frontend_load_path_verified"], "validated");
    assert_eq!(
        llama3["parity_audited"],
        "compact_50_token_plus_broader_50_token_prompt_pack_match"
    );
    assert_eq!(
        llama3["performance_measured"],
        "bounded_ubuntu_backend_memory_gate_plus_lazy_q8_hotpath_costs"
    );
    assert_eq!(
        llama3["tested_context"],
        "short_api_webui_smoke_with_broader_50_token_plus_checked_512_1024_2048_context_packs"
    );
    assert_eq!(llama3["chat_template_renderer"], "compact");
    assert_eq!(llama3["chat_template_shape_pack"], "validated_compact_pack");
    assert_eq!(
        llama3["chat_template_shape_pack_id"],
        "llama3-chat-template-shapes-v1"
    );
    assert_eq!(llama3["bounded_context_512_pack"], "validated_first_pack");
    assert_eq!(
        llama3["bounded_context_512_pack_id"],
        "llama3-context-512-smoke-v1"
    );
    assert_eq!(llama3["bounded_context_window"], 512);
    assert_eq!(llama3["bounded_context_1024_pack"], "validated_second_pack");
    assert_eq!(
        llama3["bounded_context_1024_pack_id"],
        "llama3-context-1024-smoke-v1"
    );
    assert_eq!(llama3["bounded_context_1024_window"], 1024);
    assert_eq!(llama3["bounded_context_2048_pack"], "validated_third_pack");
    assert_eq!(
        llama3["bounded_context_2048_pack_id"],
        "llama3-context-2048-smoke-v1"
    );
    assert_eq!(llama3["bounded_context_2048_window"], 2048);
    assert_eq!(llama3["bounded_context_4096_pack"], "not_promoted");
    assert_eq!(llama3["bounded_context_4096_pack_id"], "not_selected");
    assert_eq!(llama3["bounded_context_4096_window"], 4096);
    assert_eq!(
        llama3["latest_checked_bucket"],
        "llama3-context-2048-smoke-v1"
    );
    assert_eq!(llama3["latest_checked_result"], "pass");
    assert_eq!(llama3["latest_checked_output"], "CMLD-204");
    let llama3_evidence = llama3["evidence"].as_str().unwrap();
    assert!(llama3_evidence.contains("checked 512/1024/2048-context packs"));
    assert!(llama3_evidence.contains("published source/runtime-head 1024/2048 pass"));
    assert!(llama3_evidence.contains("retained-block lazy-Q8 hot-path cost probes"));
    let llama3_next_step = llama3["next_step"].as_str().unwrap();
    assert!(llama3_next_step.contains("checked 512/1024/2048 context support"));
    assert!(llama3_next_step.contains("before any wider 8B claim"));
    let mistral = compatibility
        .iter()
        .find(|item| item["id"] == "mistral_7b_instruct_v0_3_q8_0")
        .unwrap();
    assert_eq!(mistral["status"], "active_validation_unsupported");
    assert_eq!(mistral["support_scope"], "bringup_exact_row_unsupported");
    assert_eq!(
        mistral["full_support_status"],
        "blocked_unsupported_bringup"
    );
    assert_eq!(mistral["metadata_parses"], "validated");
    assert_eq!(mistral["tokenizer_works"], "validated");
    assert_eq!(mistral["tensors_load"], "validated");
    assert_eq!(
        mistral["generation_runs"],
        "evidence_only_completion_and_chat_smoke_plus_broader_50_token_api_smoke"
    );
    assert_eq!(
        mistral["frontend_load_path_verified"],
        "fail_closed_pending_support_surface_sync"
    );
    assert_eq!(
        mistral["tested_context"],
        "tokenizer_template_1tok_bounded_and_checked_512_1024_2048_4096_8192_context_packs"
    );
    assert_eq!(mistral["chat_template_renderer"], "mistral_instruct");
    assert_eq!(
        mistral["chat_template_shape_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        mistral["chat_template_shape_pack_id"],
        "mistral-instruct-v0.3-chat-template-pack-v1"
    );
    assert_eq!(
        mistral["bounded_context_512_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        mistral["bounded_context_512_pack_id"],
        "mistral-context-512-smoke-v1"
    );
    assert_eq!(
        mistral["latest_checked_bucket"],
        "current_head_api_webui_rss_fail_closed"
    );
    assert_eq!(mistral["latest_checked_result"], "pass");
    assert_eq!(mistral["bounded_context_8192_pack"], "validated_fifth_pack");
    assert_eq!(
        mistral["bounded_context_8192_pack_id"],
        "mistral-context-8192-max-ladder-v1"
    );
    let mistral_evidence = mistral["evidence"].as_str().unwrap();
    assert!(mistral_evidence.contains("Mistral v0.3 active validation only"));
    assert!(mistral_evidence.contains("support-promotion evidence remains fail-closed"));
    let mistral_next_step = mistral["next_step"].as_str().unwrap();
    assert!(mistral_next_step.contains("produce a support-promotion manifest"));
    let mixtral = compatibility
        .iter()
        .find(|item| item["id"] == "mixtral_8x7b_instruct_v0_1_q8_0")
        .unwrap();
    assert_eq!(mixtral["status"], "active_validation_partial_runtime");
    assert_eq!(
        mixtral["support_scope"],
        "exact_row_bounded_moe_runtime_only"
    );
    assert_eq!(
        mixtral["generation_runs"],
        "bounded_one_token_runtime_smoke_observed"
    );
    assert_eq!(
        mixtral["frontend_load_path_verified"],
        "fail_closed_partial_runtime_only"
    );
    assert_eq!(
        mixtral["latest_checked_bucket"],
        "mixtral_8x7b_q8_gate9a_50tok_divergence_20260511"
    );
    assert_eq!(
        mixtral["latest_checked_result"],
        "blocked_later_generation_divergence"
    );
    assert_eq!(
        mixtral["latest_checked_output"],
        "qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/README.md"
    );
    assert!(mixtral["evidence"]
        .as_str()
        .unwrap()
        .contains("llama.expert_count=8"));
    assert!(mixtral["evidence"]
        .as_str()
        .unwrap()
        .contains("Gate 9A 50-token evidence diverged at generated token index 9"));
    assert!(mixtral["evidence"]
        .as_str()
        .unwrap()
        .contains("backend HTTP hang"));
    assert!(mixtral["evidence"]
        .as_str()
        .unwrap()
        .contains("No broad Mixtral"));

    for (id, filename) in [
        ("qwen25_7b_instruct_q8_0", "Qwen2.5-7B-Instruct-Q8_0.gguf"),
        ("gemma2_9b_it_q8_0", "gemma-2-9b-it-Q8_0.gguf"),
    ] {
        let row = compatibility.iter().find(|item| item["id"] == id).unwrap();
        assert_eq!(row["status"], "planned_exact_row_candidate");
        assert_eq!(row["generation_runs"], "not_started");
        assert_eq!(row["frontend_load_path_verified"], "fail_closed_planned");
        assert_eq!(row["latest_checked_result"], "planning_only");
        assert!(row["evidence"].as_str().unwrap().contains(filename));
    }
    let planned_quant = compatibility
        .iter()
        .find(|item| item["id"] == "llama_spm_q4_k_q5_k")
        .unwrap();
    assert_eq!(planned_quant["status"], "planned_phase_10");
    assert_eq!(planned_quant["tensors_load"], "unsupported_typed_error");
    assert_eq!(planned_quant["generation_runs"], "blocked_until_dequant");
    assert_eq!(planned_quant["chat_template_renderer"], "not_selected");
    assert_eq!(planned_quant["chat_template_shape_pack"], "not_started");
    assert_eq!(planned_quant["chat_template_shape_pack_id"], "not_selected");
    assert_eq!(planned_quant["bounded_context_512_pack"], "not_started");
    assert_eq!(planned_quant["bounded_context_512_pack_id"], "not_selected");
    assert_eq!(planned_quant["bounded_context_window"], 512);
    assert_eq!(planned_quant["bounded_context_1024_pack"], "not_started");
    assert_eq!(
        planned_quant["bounded_context_1024_pack_id"],
        "not_selected"
    );
    assert_eq!(planned_quant["bounded_context_1024_window"], 1024);
    assert_eq!(planned_quant["latest_checked_bucket"], "not_selected");
    assert_eq!(planned_quant["latest_checked_result"], "not_started");
    assert_eq!(planned_quant["latest_checked_output"], "not_applicable");
}

#[tokio::test]
async fn chat_completion_validates_generation_input_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[],"stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "missing_generation_input");
}

#[tokio::test]
async fn completion_accepts_prompt_token_ids_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","camelid_prompt_token_ids":[1,2,3],"stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn llama_server_completion_maps_n_predict_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/completion")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","n_predict":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn llama_server_completion_accepts_token_id_prompt_arrays_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/completion")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":[1,2,3],"n_predict":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn llama_server_completion_rejects_empty_token_id_prompt_arrays_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/completion")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"tiny","prompt":[],"n_predict":1}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "empty_prompt_tokens");
    assert_eq!(body["error"]["param"], "prompt");
}

#[tokio::test]
async fn llama_server_completion_rejects_streaming_shape_until_implemented() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/completion")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","n_predict":1,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "stream");
}

#[tokio::test]
async fn llama_server_completion_rejects_invalid_n_predict_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/completion")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","n_predict":0}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_n_predict");
    assert_eq!(body["error"]["param"], "n_predict");
}

#[tokio::test]
async fn llama_server_completion_rejects_cache_controls_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/completion")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","n_predict":1,"cache_prompt":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "cache_prompt");
}

#[tokio::test]
async fn completion_rejects_ambiguous_prompt_and_prompt_token_ids() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","camelid_prompt_token_ids":[1,2,3],"stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "ambiguous_generation_input");
}

#[tokio::test]
async fn chat_completion_requires_loaded_model_after_valid_input() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn streaming_chat_completion_validates_input_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[],"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "missing_generation_input");
}

#[tokio::test]
async fn chat_completion_rejects_empty_message_role_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"   ","content":"hello"}],"max_tokens":1,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response.headers()["content-type"].to_str().unwrap(),
        "application/json"
    );
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_message_role");
    assert_eq!(body["error"]["param"], "messages");
}

#[tokio::test]
async fn chat_completion_rejects_empty_message_content_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":""}],"max_tokens":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_message_content");
    assert_eq!(body["error"]["param"], "messages");
}

#[tokio::test]
async fn completion_validates_sampling_parameters_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"temperature":-1.0}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
}

#[tokio::test]
async fn streaming_completion_validates_top_p_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"stream":true,"top_p":1.5}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
}

#[tokio::test]
async fn completion_validates_presence_penalty_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"presence_penalty":2.5}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
}

#[tokio::test]
async fn streaming_chat_completion_validates_frequency_penalty_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true,"frequency_penalty":-2.5}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
}

#[tokio::test]
async fn chat_completion_validates_logit_bias_token_ids_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"logit_bias":{"not-a-token":1.0}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
    assert_eq!(body["error"]["param"], "logit_bias");
}

#[tokio::test]
async fn completion_validates_logit_bias_values_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"logit_bias":{"0":101.0}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
    assert_eq!(body["error"]["param"], "logit_bias");
}

#[tokio::test]
async fn completion_rejects_unsupported_best_of_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"best_of":2}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "best_of");
}

#[tokio::test]
async fn completion_rejects_invalid_zero_choice_fields_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"n":0}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_parameter");
    assert_eq!(body["error"]["param"], "n");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"best_of":0}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_parameter");
    assert_eq!(body["error"]["param"], "best_of");
}

#[tokio::test]
async fn streaming_completion_rejects_unsupported_multiple_choices_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"stream":true,"n":2}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "n");
}

#[tokio::test]
async fn chat_completion_rejects_unsupported_multiple_choices_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"n":2}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "n");
}

#[tokio::test]
async fn completion_rejects_unsupported_logprobs_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"logprobs":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "logprobs");
}

#[tokio::test]
async fn chat_completion_rejects_unsupported_logprobs_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"logprobs":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "logprobs");
}

#[tokio::test]
async fn chat_completion_rejects_unsupported_top_logprobs_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"logprobs":false,"top_logprobs":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "top_logprobs");
}

#[tokio::test]
async fn chat_completion_rejects_top_logprobs_without_logprobs_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"top_logprobs":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "top_logprobs");
}

#[tokio::test]
async fn chat_completion_rejects_tool_fields_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"tools":[]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "tools");
}

#[tokio::test]
async fn chat_completion_rejects_response_format_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"response_format":{"type":"json_object"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "response_format");
}

#[tokio::test]
async fn completion_rejects_llama_server_sampler_fields_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"min_p":0.05}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "min_p");
}

#[tokio::test]
async fn completion_returns_typed_error_for_malformed_logprobs_payload() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"logprobs":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "malformed_json");
}

#[tokio::test]
async fn single_choice_defaults_preserve_loaded_model_preflight_order() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"n":1,"best_of":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn chat_single_choice_defaults_preserve_loaded_model_preflight_order() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"n":1,"logprobs":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn completion_validates_stop_sequences_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"stop":["a","b","c","d","e"]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_stop");
    assert_eq!(body["error"]["param"], "stop");
}

#[tokio::test]
async fn valid_advanced_sampling_fields_preserve_loaded_model_preflight_order() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"presence_penalty":0.25,"frequency_penalty":0.5,"logit_bias":{"0":-1.0}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn v1_models_supports_openai_style_model_retrieve() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/tiny-tokenizer")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["id"], "tiny-tokenizer");
    assert_eq!(body["object"], "model");
    assert_eq!(body["owned_by"], "camelid");
    assert_eq!(body["created"], 0);
    assert_eq!(body["meta"], Value::Null);
}

#[tokio::test]
async fn v1_model_retrieve_rejects_unloaded_or_unknown_model() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/missing-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["param"], "model");
}

#[tokio::test]
async fn v1_model_retrieve_reports_loaded_dense_model_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/tiny-generation")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["id"], "tiny-generation");
    assert_eq!(body["object"], "model");
    assert_eq!(body["owned_by"], "camelid");
    assert_eq!(body["created"], 0);
    assert_eq!(body["meta"]["n_vocab"], 4);
    assert_eq!(body["meta"]["n_ctx_train"], 64);
    assert_eq!(body["meta"]["n_embd"], 4);
    assert_eq!(body["meta"]["n_params"], 164);
    assert_eq!(body["meta"]["file_type"], 0);
    assert!(body["meta"]["size"].as_u64().unwrap() > 0);

    let serialized = body.to_string();
    for forbidden in [
        "/Users/",
        "/home/",
        "file://",
        "file:\\",
        "/Volumes/",
        "/private/tmp/",
        "C:\\Users\\",
        "C:/Users/",
        "\\Users\\",
    ] {
        assert!(
            !serialized.contains(forbidden),
            "/v1/models/:model meta must not expose local/private path marker {forbidden:?}"
        );
    }
}

#[tokio::test]
async fn load_model_reports_tokenizer_summary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["tokenizer"]["status"], "available");
    assert_eq!(body["tokenizer"]["model"], "llama_spm");
    assert_eq!(body["tokenizer"]["token_count"], 7);
    assert_eq!(body["tokenizer"]["byte_token_count"], 1);
    assert_eq!(body["tokenizer"]["special"]["bos"], 1);
    assert_eq!(body["tokenizer"]["special"]["eos"], 2);
    assert_eq!(body["tokenizer"]["config"]["add_bos"], true);
    assert_eq!(body["tokenizer"]["config"]["add_eos"], false);
}

#[tokio::test]
async fn generation_session_endpoint_preflights_tokenizer_then_reports_missing_dense_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/generation/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-tokenizer","prompt":"hello","max_tokens":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_model_architecture");
}

#[tokio::test]
async fn generation_session_without_max_tokens_uses_remaining_context() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation-default-max.gguf");
    write_generation_gguf_with_options(
        &path,
        GenerationFixtureOptions {
            context_length: 64,
            include_tokenizer: true,
            truncate_payload: false,
        },
    );

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation-default-max"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/generation/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation-default-max","prompt":"hello"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let prompt_tokens = body["prompt_token_count"].as_u64().unwrap();
    let max_tokens = body["max_tokens"].as_u64().unwrap();
    assert_eq!(prompt_tokens + max_tokens, 64);
    assert!(max_tokens > 16);
}

#[tokio::test]
async fn public_chat_completion_without_max_tokens_uses_demo_safe_default_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation-public-default-max.gguf");
    write_generation_gguf_with_options(
        &path,
        GenerationFixtureOptions {
            context_length: 1024,
            include_tokenizer: true,
            truncate_payload: false,
        },
    );

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation-public-default-max"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation-public-default-max","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["usage"]["completion_tokens"], 800);
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn public_completion_without_max_tokens_uses_remaining_context_when_below_demo_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("tiny-generation-public-short-context-default-max.gguf");
    write_generation_gguf_with_options(
        &path,
        GenerationFixtureOptions {
            context_length: 64,
            include_tokenizer: true,
            truncate_payload: false,
        },
    );

    let app = camelid::api::router();
    let load_body =
        serde_json::json!({"path": path, "id": "tiny-generation-public-short-context-default-max"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation-public-short-context-default-max","prompt":"hello","stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(prompt_tokens + completion_tokens, 64);
    assert!(completion_tokens < 800);
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn tokenizer_endpoint_returns_current_model_tokenizer_summary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/tokenizer")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["model"], "llama_spm");
    assert_eq!(body["token_count"], 7);
    assert_eq!(
        body["special"]["eog"].as_array().unwrap(),
        &[serde_json::json!(2)]
    );
}

#[tokio::test]
async fn tokenizer_endpoint_reports_unsupported_tokenizer_honestly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "wordpiece", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "unsupported-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/tokenizer")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_tokenizer");
}

#[tokio::test]
async fn tokenizer_endpoint_requires_loaded_model() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/tokenizer")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn tokenizer_encode_decode_endpoints_use_loaded_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/tokenizer/encode")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(
        body["tokens"].as_array().unwrap(),
        &[serde_json::json!(1), serde_json::json!(3)]
    );
    assert_eq!(body["token_count"], 2);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/tokenizer/decode")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"tokens":[3,5]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["text"], " hello!");
    assert_eq!(body["token_count"], 2);
}

#[tokio::test]
async fn llama_server_tokenize_detokenize_aliases_use_loaded_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"content":" hello!"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(
        body["tokens"].as_array().unwrap(),
        &[
            serde_json::json!(6),
            serde_json::json!(4),
            serde_json::json!(5)
        ]
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"content":" hello!","with_pieces":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(
        body["tokens"].as_array().unwrap(),
        &[
            serde_json::json!({"id":6,"piece":" "}),
            serde_json::json!({"id":4,"piece":"hello"}),
            serde_json::json!({"id":5,"piece":"!"})
        ]
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/detokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"tokens":[3,5]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["content"], " hello!");
}

#[tokio::test]
async fn llama_server_tokenize_alias_requires_loaded_model_for_piece_metadata() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"content":"hello","with_pieces":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn llama_server_tokenize_alias_rejects_unknown_fields_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"content":"hello","pieces":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "request");
}

#[tokio::test]
async fn llama_server_apply_template_renders_loaded_model_chat_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf_with_chat_template(
        &path,
        "llama",
        true,
        false,
        true,
        "<|system|>{{ system }}<|user|>{{ user }}<|assistant|>{{ assistant }}",
    );

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/apply-template")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["prompt"], "<|user|>\nhello</s>\n<|assistant|>\n");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/apply-template")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"messages":[{"role":"user","content":"hello"}],"template":"custom"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "request");
}

#[tokio::test]
async fn tokenizer_encode_requires_loaded_model() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/tokenizer/encode")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn tokenizer_decode_reports_unsupported_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "wordpiece", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "unsupported-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/tokenizer/decode")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"tokens":[1,2]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_tokenizer");
}

#[tokio::test]
async fn tokenizer_endpoints_return_typed_malformed_request_errors() {
    let app = camelid::api::router();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/tokenizer/encode")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "missing_tokenizer_text");
    assert_eq!(body["error"]["param"], "text");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/tokenizer/decode")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"tokens":"not-a-list"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "malformed_json");
}

#[tokio::test]
async fn tokenizer_decode_reports_out_of_range_token_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let body = serde_json::json!({"path": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/tokenizer/decode")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"tokens":[999]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "token_decode_failed");
    assert_eq!(body["error"]["param"], "tokens");
}

#[tokio::test]
async fn chat_completion_generates_one_decoded_token_from_loaded_dense_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let load_status = response.status();
    let load_body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        load_status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&load_body)
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":false,"camelid_dense_diagnostics":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "tiny-generation");
    assert_eq!(body["choices"].as_array().unwrap().len(), 1);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "<unk>");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert!(body["choices"][0].get("logprobs").is_none());
    assert_eq!(body["usage"]["completion_tokens"], 1);
    assert!(body["camelid"]["prompt_token_ids"]
        .as_array()
        .is_some_and(|tokens| !tokens.is_empty()));
    assert_eq!(body["camelid"]["generated_token_ids"], json!([0]));
    let dense_metadata = &body["camelid"]["dense_metadata"];
    assert_eq!(dense_metadata["embedding_length"], 4);
    assert_eq!(dense_metadata["attention_head_count"], 2);
    assert_eq!(dense_metadata["attention_head_count_kv"], 1);
    assert_eq!(dense_metadata["head_dim"], 2);
    assert_eq!(dense_metadata["rope_dimension_count"], 2);
    assert_eq!(dense_metadata["rope_freq_base"], 10_000.0);
    assert_eq!(dense_metadata["rope_pairing"], "adjacent_even_odd");
    assert_eq!(dense_metadata["rope_direction"], "forward");
    assert_eq!(dense_metadata["rope_position_mode"], "zero_based");
    assert_eq!(dense_metadata["attention_score_scale"], "head_dim");
    assert_eq!(dense_metadata["ffn_gate_up_order"], "gate_up");
    assert_eq!(dense_metadata["rms_norm_epsilon"], 1e-6);
    assert_eq!(dense_metadata["rms_norm_effective_epsilon"], 1e-6);
    assert_eq!(
        dense_metadata["square_linear_diagnostic_layout"],
        "transposed"
    );
    assert_eq!(dense_metadata["token_embedding_shape"], json!([4, 4]));
    assert_eq!(dense_metadata["output_shape"], json!([4, 4]));
    assert_eq!(dense_metadata["output_is_tied_embedding"], false);
    assert_eq!(dense_metadata["output_projection_layout"], "input_output");
    assert_eq!(
        dense_metadata["output_projection_diagnostic_layout"],
        "token_major"
    );
    assert_eq!(dense_metadata["zero_attention_delta"], "none");
    assert_eq!(dense_metadata["zero_ffn_delta"], "none");
    let orientations = &dense_metadata["projection_orientations"];
    assert_eq!(orientations["attention_q"]["shape"], json!([4, 4]));
    assert_eq!(orientations["attention_q"]["input_width"], 4);
    assert_eq!(orientations["attention_q"]["output_width"], 4);
    assert_eq!(
        orientations["attention_q"]["descriptor_layout"],
        "input_output"
    );
    assert_eq!(
        orientations["attention_q"]["runtime_interpretation"],
        "rhs_transposed"
    );
    assert_eq!(
        orientations["attention_q"]["square_diagnostic_applies"],
        true
    );
    assert_eq!(
        orientations["ffn_down"]["descriptor_layout"],
        "input_output"
    );
    let top_logits = body["camelid"]["top_logits"].as_array().unwrap();
    assert!(!top_logits.is_empty());
    assert_eq!(top_logits[0]["token_id"], 0);
    assert!(top_logits[0]["logit"].is_number());
    assert!(top_logits[0]["probability"].is_number());
    assert_eq!(top_logits[0]["rank"], 1);
    assert_eq!(top_logits[0]["selected"], false);
    assert_eq!(top_logits[0]["text"], "<unk>");
    let output_projection = body["camelid"]["output_projection"].as_array().unwrap();
    assert_eq!(output_projection.len(), top_logits.len());
    assert_eq!(output_projection[0]["token_id"], top_logits[0]["token_id"]);
    assert_eq!(output_projection[0]["layout"], "output_input");
    assert!(output_projection[0]["reported_logit"].is_number());
    assert!(output_projection[0]["reconstructed_logit"].is_number());
    assert!(output_projection[0]["absolute_delta"].as_f64().unwrap() < 1e-4);
    assert!(output_projection[0]["output_row_rms"].is_number());
    assert!(output_projection[0]["cosine_similarity"].is_number());
    assert!(output_projection[0]["output_norm_first_values"].is_array());
    assert!(output_projection[0]["output_row_first_values"].is_array());
    assert!(output_projection[0]["component_products_first_values"].is_array());
    assert!(output_projection[0]["component_products_max_abs_window_start"].is_number());
    assert!(output_projection[0]["component_products_max_abs_window"].is_array());
    assert!(output_projection[0]["max_abs_component_index"].is_number());
    assert!(output_projection[0]["max_abs_component"].is_number());
    assert!(output_projection[0]["positive_component_sum"].is_number());
    assert!(output_projection[0]["negative_component_sum"].is_number());
    assert!(output_projection[0]["top_positive_components"].is_array());
    assert!(output_projection[0]["top_negative_components"].is_array());
    if let Some(component) = output_projection[0]["top_positive_components"]
        .as_array()
        .and_then(|items| items.first())
    {
        assert!(component["index"].is_number());
        assert!(component["final_hidden_value"].is_number());
        assert!(component["output_norm_weight_value"].is_number());
        assert!(component["output_norm_scale"].is_number());
        assert!(component["reconstructed_output_norm_value"].is_number());
        assert!(
            component["output_norm_reconstruction_delta"]
                .as_f64()
                .unwrap()
                < 1e-5
        );
        assert!(component["output_norm_value"].is_number());
        assert!(component["output_row_value"].is_number());
        assert!(component["component"].is_number());
    }
    let dense = &body["camelid"]["dense"];
    assert!(dense["embedding"]["rms"].is_number());
    assert!(dense["final_hidden"]["mean"].is_number());
    assert!(dense["final_norm"]["hidden_mean_square"].is_number());
    assert!(dense["final_norm"]["scale"].is_number());
    assert!(dense["final_norm"]["hidden_first_values"].is_array());
    assert!(dense["final_norm"]["weight_first_values"].is_array());
    assert!(dense["final_norm"]["reconstructed_first_values"].is_array());
    assert!(dense["final_norm"]["reported_first_values"].is_array());
    assert!(dense["final_norm"]["reported_max_abs_index"].is_number());
    assert!(dense["final_norm"]["reported_max_abs"].is_number());
    assert!(dense["final_norm"]["reported_max_abs_window_start"].is_number());
    assert!(dense["final_norm"]["reported_max_abs_window"].is_array());
    assert!(dense["final_norm"]["reconstructed_reported_max_abs_window"].is_array());
    assert!(dense["final_norm"]["max_abs_delta"].as_f64().unwrap() < 1e-5);
    assert!(dense["output_norm"]["rms"].is_number());
    assert!(dense["logits"]["max"].is_number());
    assert!(dense["logits"]["max_index"].is_number());
    assert!(dense["logits"]["max_abs"].is_number());
    assert!(dense["logits"]["max_abs_index"].is_number());
    assert_eq!(dense["layers"].as_array().unwrap().len(), 1);
    let residual_flow = &dense["layers"][0]["residual_flow"];
    assert!(residual_flow["attention_input"]["checkpoint"]["first_values"].is_array());
    assert!(residual_flow["attention_delta"]["input_rms"].is_number());
    assert!(residual_flow["attention_delta"]["delta_rms"].is_number());
    assert!(residual_flow["attention_delta"]["reported_rms"].is_number());
    assert!(residual_flow["attention_delta"]["delta_to_input_rms_ratio"].is_number());
    assert!(residual_flow["attention_delta"]["delta_input_cosine_similarity"].is_number());
    assert!(residual_flow["attention_delta"]["input_first_values"].is_array());
    assert!(residual_flow["attention_delta"]["delta_first_values"].is_array());
    assert!(residual_flow["attention_delta"]["reconstructed_first_values"].is_array());
    assert!(residual_flow["attention_delta"]["reported_first_values"].is_array());
    assert!(
        residual_flow["attention_delta"]["max_abs_delta"]
            .as_f64()
            .unwrap()
            < 1e-6
    );
    assert!(residual_flow["ffn_input"]["checkpoint"]["first_values"].is_array());
    assert!(residual_flow["ffn_delta"]["input_rms"].is_number());
    assert!(residual_flow["ffn_delta"]["delta_rms"].is_number());
    assert!(residual_flow["ffn_delta"]["reported_rms"].is_number());
    assert!(residual_flow["ffn_delta"]["delta_to_input_rms_ratio"].is_number());
    assert!(residual_flow["ffn_delta"]["delta_input_cosine_similarity"].is_number());
    assert!(residual_flow["ffn_delta"]["input_first_values"].is_array());
    assert!(residual_flow["ffn_delta"]["delta_first_values"].is_array());
    assert!(residual_flow["ffn_delta"]["reconstructed_first_values"].is_array());
    assert!(residual_flow["ffn_delta"]["reported_first_values"].is_array());
    assert!(
        residual_flow["ffn_delta"]["max_abs_delta"]
            .as_f64()
            .unwrap()
            < 1e-6
    );
    assert!(dense["layers"][0]["attention_norm_reconstruction"]["input_mean_square"].is_number());
    assert!(dense["layers"][0]["attention_norm_reconstruction"]["scale"].is_number());
    assert!(dense["layers"][0]["attention_norm_reconstruction"]["input_first_values"].is_array());
    assert!(dense["layers"][0]["attention_norm_reconstruction"]["weight_first_values"].is_array());
    assert!(
        dense["layers"][0]["attention_norm_reconstruction"]["reconstructed_first_values"]
            .is_array()
    );
    assert!(
        dense["layers"][0]["attention_norm_reconstruction"]["reported_first_values"].is_array()
    );
    assert!(
        dense["layers"][0]["attention_norm_reconstruction"]["reported_max_abs_index"].is_number()
    );
    assert!(dense["layers"][0]["attention_norm_reconstruction"]["reported_max_abs"].is_number());
    assert!(
        dense["layers"][0]["attention_norm_reconstruction"]["reported_max_abs_window_start"]
            .is_number()
    );
    assert!(
        dense["layers"][0]["attention_norm_reconstruction"]["reported_max_abs_window"].is_array()
    );
    assert!(dense["layers"][0]["attention_norm_reconstruction"]
        ["reconstructed_reported_max_abs_window"]
        .is_array());
    assert!(
        dense["layers"][0]["attention_norm_reconstruction"]["max_abs_delta"]
            .as_f64()
            .unwrap()
            < 1e-6
    );
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["input_mean_square"].is_number());
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["scale"].is_number());
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["input_first_values"].is_array());
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["weight_first_values"].is_array());
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["reconstructed_first_values"].is_array());
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["reported_first_values"].is_array());
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["reported_max_abs_index"].is_number());
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["reported_max_abs"].is_number());
    assert!(
        dense["layers"][0]["ffn_norm_reconstruction"]["reported_max_abs_window_start"].is_number()
    );
    assert!(dense["layers"][0]["ffn_norm_reconstruction"]["reported_max_abs_window"].is_array());
    assert!(
        dense["layers"][0]["ffn_norm_reconstruction"]["reconstructed_reported_max_abs_window"]
            .is_array()
    );
    assert!(
        dense["layers"][0]["ffn_norm_reconstruction"]["max_abs_delta"]
            .as_f64()
            .unwrap()
            < 1e-6
    );
    assert!(dense["layers"][0]["attention_q_rope"]["rms"].is_number());
    assert!(dense["layers"][0]["attention_q_rope"]["min_index"].is_number());
    assert_eq!(
        dense["layers"][0]["attention_q_rope"]["checkpoint"]["shape"],
        json!([1, 4])
    );
    assert!(dense["layers"][0]["attention_q_rope"]["checkpoint"]["first_values"].is_array());
    assert!(dense["layers"][0]["attention_q_rope"]["checkpoint"]["max_abs_window"].is_array());
    assert_eq!(
        dense["layers"][0]["attention_q_rope_reconstruction"]["role"],
        "attention_q"
    );
    assert_eq!(
        dense["layers"][0]["attention_q_rope_reconstruction"]["pairing"],
        "adjacent_even_odd"
    );
    assert_eq!(
        dense["layers"][0]["attention_q_rope_reconstruction"]["direction"],
        "forward"
    );
    assert!(dense["layers"][0]["attention_q_rope_reconstruction"]["input_first_values"].is_array());
    assert!(
        dense["layers"][0]["attention_q_rope_reconstruction"]["reconstructed_first_values"]
            .is_array()
    );
    assert!(
        dense["layers"][0]["attention_q_rope_reconstruction"]["reported_first_values"].is_array()
    );
    assert!(
        dense["layers"][0]["attention_q_rope_reconstruction"]["max_abs_delta"]
            .as_f64()
            .unwrap()
            < 1e-6
    );
    assert_eq!(
        dense["layers"][0]["attention_k_rope_reconstruction"]["role"],
        "attention_k"
    );
    assert!(
        dense["layers"][0]["attention_k_rope_reconstruction"]["max_abs_delta"]
            .as_f64()
            .unwrap()
            < 1e-6
    );
    let attention_trace = &dense["layers"][0]["attention_trace"];
    assert!(attention_trace["scale"].is_number());
    let position_count = attention_trace["position_count"].as_u64().unwrap();
    assert!(position_count >= 1);
    assert_eq!(attention_trace["head_dim"], 2);
    assert_eq!(attention_trace["heads"].as_array().unwrap().len(), 2);
    assert_eq!(attention_trace["heads"][0]["attention_head"], 0);
    assert_eq!(attention_trace["heads"][0]["kv_head"], 0);
    assert!(attention_trace["heads"][0]["probability_sum"].is_number());
    assert!(attention_trace["heads"][0]["probability_entropy"].is_number());
    assert!(attention_trace["heads"][0]["probability_rms"].is_number());
    assert!(attention_trace["heads"][0]["query_first_values"].is_array());
    assert!(attention_trace["heads"][0]["context_first_values"].is_array());
    assert!(attention_trace["heads"][0]["top_probability_positions"].is_array());
    assert!(attention_trace["heads"][0]["top_probability_positions"][0]["position"].is_number());
    assert!(attention_trace["heads"][0]["top_probability_positions"][0]["score"].is_number());
    assert!(attention_trace["heads"][0]["top_probability_positions"][0]["probability"].is_number());
    assert!(
        attention_trace["heads"][0]["top_probability_positions"][0]["key_first_values"].is_array()
    );
    assert!(
        attention_trace["heads"][0]["top_probability_positions"][0]["value_first_values"]
            .is_array()
    );
    assert_eq!(
        attention_trace["heads"][0]["positions"]
            .as_array()
            .unwrap()
            .len(),
        position_count.min(8) as usize
    );
    assert_eq!(attention_trace["heads"][0]["positions"][0]["position"], 0);
    assert!(attention_trace["heads"][0]["positions"][0]["score"].is_number());
    assert!(attention_trace["heads"][0]["positions"][0]["reconstructed_score"].is_number());
    assert!(attention_trace["heads"][0]["positions"][0]["score_reconstruction_delta"].is_number());
    assert!(attention_trace["heads"][0]["positions"][0]["probability"].is_number());
    assert!(attention_trace["heads"][0]["positions"][0]["key_first_values"].is_array());
    assert!(attention_trace["heads"][0]["positions"][0]["qk_products_first_values"].is_array());
    assert!(
        attention_trace["heads"][0]["positions"][0]["qk_products_max_abs_window_start"].is_number()
    );
    assert!(attention_trace["heads"][0]["positions"][0]["qk_products_max_abs_window"].is_array());
    assert!(attention_trace["heads"][0]["positions"][0]["value_first_values"].is_array());
    assert!(dense["layers"][0]["ffn_gate"]["checkpoint"]["first_values"].is_array());
    assert!(dense["layers"][0]["ffn_up"]["checkpoint"]["first_values"].is_array());
    assert!(dense["layers"][0]["ffn_activation"]["max"].is_number());
    assert!(
        dense["layers"][0]["ffn_activation_reconstruction"]["max_abs_delta"]
            .as_f64()
            .unwrap()
            < 1e-6
    );
    assert!(
        dense["layers"][0]["ffn_activation_reconstruction"]["reported_max_abs_window"].is_array()
    );
    assert_eq!(body["camelid"]["timings_ms"]["weight_cache_hit"], false);
    assert!(body["camelid"]["timings_ms"]["generation"]["forward_total"].is_number());
    assert!(body["camelid"]["timings_ms"]["generation"]["logits"].is_number());
    assert_eq!(
        body["camelid"]["timings_ms"]["prompt_evaluation"]["prompt_token_count"],
        body["usage"]["prompt_tokens"]
    );
    assert!(
        body["camelid"]["timings_ms"]["prompt_evaluation"]["first_token"]["forward_total"]
            .is_number()
    );
    assert_eq!(
        body["camelid"]["timings_ms"]["prompt_evaluation"]["first_token_layers"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        body["camelid"]["timings_ms"]["layers"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(body["camelid"]["timings_ms"]["layers"][0]["attention_q"].is_number());
    assert!(body["camelid"]["timings_ms"]["layers"][0]["ffn_down"].is_number());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"stream":false,"camelid_dense_diagnostic_generated_index":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["usage"]["completion_tokens"], 2);
    assert_eq!(body["camelid"]["dense_diagnostic_generated_index"], 1);
    assert_eq!(
        body["camelid"]["dense"]["layers"].as_array().unwrap().len(),
        1
    );
    assert!(!body["camelid"]["output_projection"]
        .as_array()
        .unwrap()
        .is_empty());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["camelid"]["timings_ms"]["weight_cache_hit"], true);
}

#[tokio::test]
async fn chat_completion_streams_openai_compatible_sse_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body_bytes.to_vec()).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(content_type.starts_with("text/event-stream"));
    assert!(body.contains("data: {\"id\":\"chatcmpl-"));
    assert!(body.contains("\"object\":\"chat.completion.chunk\""));
    assert!(body.contains("\"delta\":{\"role\":\"assistant\"}"));
    assert!(body.contains("\"delta\":{\"content\":\"<unk>\"}"));
    assert!(body.contains("\"finish_reason\":\"length\""));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn completion_endpoint_generates_multiple_greedy_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let load_status = response.status();
    let load_body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        load_status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&load_body)
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","prompt":"hello","max_tokens":2,"stream":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"].as_array().unwrap().len(), 1);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["text"], "<unk><unk>");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert!(body["choices"][0].get("logprobs").is_none());
    assert_eq!(body["usage"]["completion_tokens"], 2);
    assert!(body["camelid"]["prompt_token_ids"]
        .as_array()
        .is_some_and(|tokens| !tokens.is_empty()));
    assert_eq!(body["camelid"]["generated_token_ids"], json!([0, 0]));
    assert!(body["camelid"]["timings_ms"]["generation"]["forward_total"].is_number());
    assert!(body["camelid"]["timings_ms"]["layers"].is_array());
}

#[tokio::test]
async fn llama_server_completion_generates_non_streaming_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/completion")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","prompt":"hello","n_predict":2}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["content"], "<unk><unk>");
    assert_eq!(body["model"], "tiny-generation");
    assert_eq!(body["stop"], false);
    assert_eq!(body["stopped_limit"], true);
    assert_eq!(body["tokens_predicted"], 2);
    assert!(body["tokens_evaluated"].as_u64().unwrap() > 0);
    assert_eq!(
        body["camelid"]["compatibility"],
        "partial_llama_server_completion_non_streaming"
    );
    assert_eq!(body["camelid"]["finish_reason"], "length");
    assert!(body["camelid"]["unsupported"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "streaming_completion_shape"));
}

#[tokio::test]
async fn completion_endpoint_honors_stop_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","prompt":"hello","max_tokens":2,"stream":false,"stop":"<unk>"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["choices"][0]["text"], "");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn completion_rejects_invalid_stop_sequence_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"stop":[]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_stop");
    assert_eq!(body["error"]["param"], "stop");
}

#[tokio::test]
async fn completion_rejects_empty_stop_string_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"stop":""}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_stop");
    assert_eq!(body["error"]["param"], "stop");
}

#[tokio::test]
async fn chat_completion_returns_typed_error_for_malformed_stop_payload() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stop":{"bad":"shape"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "malformed_json");
    assert_eq!(body["error"]["type"], "invalid_request");
}

#[tokio::test]
async fn completion_endpoint_truncates_stop_sequence_after_partial_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","prompt":"hello","max_tokens":2,"stream":false,"stop":">"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["choices"][0]["text"], "<unk");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn streaming_completion_honors_stop_sequence_finish_reason() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","prompt":"hello","max_tokens":2,"stream":true,"stop":"<unk>"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body_bytes.to_vec()).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(content_type.starts_with("text/event-stream"));
    assert!(body.contains("\"finish_reason\":\"stop\""));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn completion_endpoint_streams_openai_compatible_sse_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","prompt":"hello","max_tokens":1,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body_bytes.to_vec()).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(content_type.starts_with("text/event-stream"));
    assert!(body.contains("data: {\"id\":\"cmpl-"));
    assert!(body.contains("\"object\":\"text_completion\""));
    assert!(body.contains("\"text\":\"<unk>\""));
    assert!(body.contains("\"finish_reason\":\"length\""));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn streaming_completion_accepts_advanced_sampling_controls() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-generation.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-generation"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-generation","prompt":"hello","max_tokens":1,"stream":true,"presence_penalty":0.25,"frequency_penalty":0.5,"logit_bias":{"0":1.0}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body_bytes.to_vec()).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(content_type.starts_with("text/event-stream"));
    assert!(body.contains("\"object\":\"text_completion\""));
    assert!(body.contains("\"text\":\"<unk>\""));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn streaming_completion_rejects_context_overflow_before_loading_weights() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-short-context.gguf");
    write_generation_gguf_with_options(
        &path,
        GenerationFixtureOptions {
            context_length: 2,
            include_tokenizer: true,
            truncate_payload: false,
        },
    );

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-short-context"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-short-context","prompt":"hello","max_tokens":1,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "context_length_exceeded");
    assert_eq!(body["error"]["param"], "max_tokens");
}

#[tokio::test]
async fn completion_reports_missing_tokenizer_for_dense_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-no-tokenizer.gguf");
    write_generation_gguf_with_options(
        &path,
        GenerationFixtureOptions {
            context_length: 64,
            include_tokenizer: false,
            truncate_payload: false,
        },
    );

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-no-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body_bytes)
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-no-tokenizer","prompt":"hello","max_tokens":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "tokenizer_not_available");
}

#[tokio::test]
async fn load_model_rejects_truncated_weight_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-truncated-weights.gguf");
    write_generation_gguf_with_options(
        &path,
        GenerationFixtureOptions {
            context_length: 64,
            include_tokenizer: true,
            truncate_payload: true,
        },
    );

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-truncated-weights"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_model");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("data extends beyond end of file"));
}

#[derive(Clone, Copy)]
struct GenerationFixtureOptions {
    context_length: u32,
    include_tokenizer: bool,
    truncate_payload: bool,
}

fn write_generation_gguf(path: &std::path::Path) {
    write_generation_gguf_with_options(
        path,
        GenerationFixtureOptions {
            context_length: 64,
            include_tokenizer: true,
            truncate_payload: false,
        },
    );
}

fn write_generation_gguf_with_options(path: &std::path::Path, options: GenerationFixtureOptions) {
    let tensors: Vec<(&str, Vec<i64>)> = vec![
        ("token_embd.weight", vec![4, 4]),
        ("output_norm.weight", vec![4]),
        ("output.weight", vec![4, 4]),
        ("blk.0.attn_norm.weight", vec![4]),
        ("blk.0.attn_q.weight", vec![4, 4]),
        ("blk.0.attn_k.weight", vec![4, 2]),
        ("blk.0.attn_v.weight", vec![4, 2]),
        ("blk.0.attn_output.weight", vec![4, 4]),
        ("blk.0.ffn_norm.weight", vec![4]),
        ("blk.0.ffn_gate.weight", vec![4, 6]),
        ("blk.0.ffn_up.weight", vec![4, 6]),
        ("blk.0.ffn_down.weight", vec![6, 4]),
    ];
    let tokens = ["<unk>", "<s>", "</s>", "▁hello"];
    let scores = [0.0, 0.0, 0.0, 10.0];
    let token_types = [2, 3, 3, 1];

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, tensors.len() as i64);
    push_i64(&mut b, if options.include_tokenizer { 21 } else { 12 });

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_u32(&mut b, "general.file_type", 0);
    push_kv_u32(&mut b, "llama.context_length", options.context_length);
    push_kv_u32(&mut b, "llama.embedding_length", 4);
    push_kv_u32(&mut b, "llama.block_count", 1);
    push_kv_u32(&mut b, "llama.feed_forward_length", 6);
    push_kv_u32(&mut b, "llama.attention.head_count", 2);
    push_kv_u32(&mut b, "llama.attention.head_count_kv", 1);
    push_kv_u32(&mut b, "llama.rope.dimension_count", 2);
    push_kv_f32(&mut b, "llama.rope.freq_base", 10_000.0);
    push_kv_f32(&mut b, "llama.attention.layer_norm_rms_epsilon", 1e-6);
    push_kv_u32(&mut b, "llama.vocab_size", 4);
    if options.include_tokenizer {
        push_kv_string(&mut b, "tokenizer.ggml.model", "llama");
        push_kv_array_strings(&mut b, "tokenizer.ggml.tokens", &tokens);
        push_kv_array_f32(&mut b, "tokenizer.ggml.scores", &scores);
        push_kv_array_i32(&mut b, "tokenizer.ggml.token_type", &token_types);
        push_kv_u32(&mut b, "tokenizer.ggml.bos_token_id", 1);
        push_kv_u32(&mut b, "tokenizer.ggml.eos_token_id", 2);
        push_kv_bool(&mut b, "tokenizer.ggml.add_bos_token", true);
        push_kv_bool(&mut b, "tokenizer.ggml.add_eos_token", false);
        push_kv_bool(&mut b, "tokenizer.ggml.add_space_prefix", true);
    }

    let mut relative_offset = 0u64;
    for (name, dims) in &tensors {
        push_string(&mut b, name);
        push_u32(&mut b, dims.len() as u32);
        for dim in dims {
            push_i64(&mut b, *dim);
        }
        push_i32(&mut b, 0);
        push_u64(&mut b, relative_offset);
        relative_offset += dims.iter().product::<i64>() as u64 * 4;
        while !relative_offset.is_multiple_of(32) {
            relative_offset += 1;
        }
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    let payload_bytes = if options.truncate_payload {
        relative_offset.saturating_sub(4) as usize
    } else {
        relative_offset as usize
    };
    b.extend(vec![0u8; payload_bytes]);
    std::fs::write(path, b).unwrap();
}

fn write_tokenizer_gguf(
    path: &std::path::Path,
    model: &str,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
) {
    write_tokenizer_gguf_with_optional_chat_template(
        path,
        model,
        add_bos,
        add_eos,
        add_space_prefix,
        None,
    );
}

fn write_tokenizer_gguf_with_chat_template(
    path: &std::path::Path,
    model: &str,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    chat_template: &str,
) {
    write_tokenizer_gguf_with_optional_chat_template(
        path,
        model,
        add_bos,
        add_eos,
        add_space_prefix,
        Some(chat_template),
    );
}

fn write_tokenizer_gguf_with_optional_chat_template(
    path: &std::path::Path,
    model: &str,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    chat_template: Option<&str>,
) {
    let tokens = ["<unk>", "<s>", "</s>", "▁hello", "hello", "<0x21>", "▁"];
    let scores = [0.0, 0.0, 0.0, 10.0, 2.0, 0.0, 1.0];
    let token_types = [2, 3, 3, 1, 1, 6, 1];

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, 0);
    push_i64(&mut b, if chat_template.is_some() { 11 } else { 10 });

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_string(&mut b, "tokenizer.ggml.model", model);
    push_kv_array_strings(&mut b, "tokenizer.ggml.tokens", &tokens);
    push_kv_array_f32(&mut b, "tokenizer.ggml.scores", &scores);
    push_kv_array_i32(&mut b, "tokenizer.ggml.token_type", &token_types);
    push_kv_u32(&mut b, "tokenizer.ggml.bos_token_id", 1);
    push_kv_u32(&mut b, "tokenizer.ggml.eos_token_id", 2);
    push_kv_bool(&mut b, "tokenizer.ggml.add_bos_token", add_bos);
    push_kv_bool(&mut b, "tokenizer.ggml.add_eos_token", add_eos);
    push_kv_bool(&mut b, "tokenizer.ggml.add_space_prefix", add_space_prefix);
    if let Some(chat_template) = chat_template {
        push_kv_string(&mut b, "tokenizer.chat_template", chat_template);
    }

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    std::fs::write(path, b).unwrap();
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

fn push_kv_f32(b: &mut Vec<u8>, key: &str, value: f32) {
    push_string(b, key);
    push_i32(b, 6);
    b.extend_from_slice(&value.to_le_bytes());
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
