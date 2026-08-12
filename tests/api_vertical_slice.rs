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
async fn verify_without_an_active_model_fails_closed() {
    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .uri("/api/models/verify")
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
    let verify = body["api_features"]
        .as_array()
        .and_then(|features| {
            features
                .iter()
                .find(|feature| feature["id"] == "camelid_verify")
        })
        .expect("capabilities must advertise the bounded Camelid Verify surface");
    assert_eq!(verify["status"], "partial");
    assert!(verify["notes"]
        .as_str()
        .unwrap()
        .contains("exact-GGUF-hash"));
    assert!(verify["notes"]
        .as_str()
        .unwrap()
        .contains("not a digital signature"));
    assert!(verify["notes"]
        .as_str()
        .unwrap()
        .contains("support promotion"));
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
    // On a box with a usable CUDA device the engine truthfully advertises ONE
    // slot (GPU-resident streams run exclusive), while this test asserts the
    // llama-server default shape. Pin the CPU lane; never removed, because
    // tests in this binary run in parallel threads and no test here wants the
    // GPU lane. CI runners have no GPU and are unaffected.
    std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
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
    // `total_slots` is the length of the `/slots` array, which is one entry per
    // admissible cooperative streaming slot (see `slots_reports_public_...`).
    assert_eq!(
        body["total_slots"],
        camelid::runtime_config::DEFAULT_CONTINUOUS_BATCH_SLOTS
    );
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
    assert!(!body["camelid"]["unsupported"]
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
async fn llama_server_models_load_alias_loads_local_model_with_redacted_response() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.gguf");
    write_tokenizer_gguf(&path, "llama", true, false, true);

    let app = camelid::api::router();
    let load_body = serde_json::json!({"model": path, "id": "tiny-tokenizer"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/models/load")
                .header("content-type", "application/json")
                .body(Body::from(load_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["data"]["id"], "tiny-tokenizer");
    assert_eq!(body["data"]["path"], Value::Null);
    assert_eq!(body["data"]["status"]["value"], "loaded");
    assert_eq!(body["data"]["camelid"]["model_path_redacted"], true);
    assert_eq!(
        body["camelid"]["compatibility"],
        "partial_llama_server_models_load_local_path"
    );
    assert_eq!(body["camelid"]["model_path_redacted"], true);
    assert!(body["camelid"]["unsupported"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "models_autoload"));
    assert!(!body.to_string().contains(path.to_str().unwrap()));

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
    assert_eq!(body["data"][0]["id"], "tiny-tokenizer");
    assert_eq!(body["data"][0]["path"], Value::Null);
}

#[tokio::test]
async fn llama_server_models_load_alias_rejects_router_mode_fields() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/models/load")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"tiny.gguf","autoload":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "body");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("autoload"));
}

#[tokio::test]
async fn llama_server_models_load_alias_rejects_ambiguous_model_and_path() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/models/load")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"a.gguf","path":"b.gguf"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "ambiguous_model_path");
    assert_eq!(body["error"]["param"], "model");
}

#[tokio::test]
async fn llama_server_models_load_alias_redacts_failed_local_path() {
    let private_path = "C:/Users/example/private/missing.gguf";
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/models/load")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"model": private_path}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let serialized = body.to_string();
    assert_eq!(body["error"]["param"], "model");
    assert!(!serialized.contains(private_path));
    assert!(!serialized.contains("C:/Users/"));
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("failed to load requested local GGUF path"));
}

#[tokio::test]
async fn llama_server_models_load_alias_redacts_relative_path_on_every_platform() {
    // Relative on Windows AND Unix, so it walks the models-dir resolver's
    // attempted-locations error (which names paths) on every CI platform —
    // the alias's redaction must scrub all of it, not just the top-level path.
    let private_relative = "example-private/missing-model.gguf";
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/models/load")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"model": private_relative}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let serialized = body.to_string();
    assert_eq!(body["error"]["param"], "model");
    assert!(!serialized.contains(private_relative));
    assert!(!serialized.contains("example-private"));
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("failed to load requested local GGUF path"));
}

#[tokio::test]
async fn llama_server_models_load_alias_requires_model_or_path() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/models/load")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "missing_model_path");
    assert_eq!(body["error"]["param"], "model");
}

#[tokio::test]
async fn native_compatibility_routes_fail_closed_with_typed_errors() {
    let cases = [
        (
            "POST",
            "/models/unload",
            "unsupported_llama_server_models_unload",
            "model",
        ),
        ("POST", "/props", "unsupported_llama_server_props", "props"),
        ("POST", "/slots", "unsupported_llama_server_slots", "slots"),
        (
            "POST",
            "/infill",
            "unsupported_llama_server_infill",
            "input",
        ),
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
async fn responses_route_accepts_stateless_text_and_reaches_the_runtime_gate() {
    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"not-loaded","input":"hello","store":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_ne!(body["error"]["code"], "unsupported_responses");
    assert!(
        body["error"]["code"] == "model_not_loaded",
        "stateless Responses input should pass request conversion and stop at the model gate: {body}"
    );
}

#[tokio::test]
async fn responses_route_resolves_stateful_ids_and_keeps_hosted_features_typed_unsupported() {
    let temp = tempfile::tempdir().unwrap();
    let missing_previous = camelid::api::router_with_state(
        camelid::api::AppState::default()
            .with_responses_store_path(temp.path().join("responses.sqlite3")),
    )
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"input":"hello","previous_response_id":"resp_prior"}"#,
            ))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(missing_previous.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_slice(
        &to_bytes(missing_previous.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["error"]["code"], "resource_not_found");
    assert_eq!(body["error"]["param"], "previous_response_id");

    let cases = [
        (r#"{"input":"hello","background":true}"#, "background"),
        (
            r#"{"input":"hello","tools":[{"type":"web_search"}]}"#,
            "tools",
        ),
    ];
    for (request_body, param) in cases {
        let response = camelid::api::router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["code"], "unsupported_parameter");
        assert_eq!(body["error"]["param"], param);
    }
}

#[tokio::test]
async fn conversations_crud_survives_router_restart_and_preserves_tool_items() {
    let temp = tempfile::tempdir().unwrap();
    let database = temp.path().join("responses.sqlite3");
    let app = camelid::api::router_with_state(
        camelid::api::AppState::default().with_responses_store_path(database.clone()),
    );
    let created = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/conversations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "metadata":{"owner":"vertical-test"},
                        "items":[{
                            "type":"function_call",
                            "call_id":"call_1",
                            "name":"weather",
                            "arguments":"{\"city\":\"Paris\"}"
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::OK);
    let created: Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    let conversation_id = created["id"].as_str().unwrap().to_string();

    let added = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/conversations/{conversation_id}/items"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"items":[{
                        "type":"function_call_output",
                        "call_id":"call_1",
                        "output":"{\"temp\":25}"
                    }]})
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(added.status(), StatusCode::OK);

    // A new state/router instance must observe the same on-disk rows.
    let restarted = camelid::api::router_with_state(
        camelid::api::AppState::default().with_responses_store_path(database),
    );
    let listed = restarted
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/v1/conversations/{conversation_id}/items?order=asc"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(listed.status(), StatusCode::OK);
    let listed: Value =
        serde_json::from_slice(&to_bytes(listed.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(listed["data"].as_array().unwrap().len(), 2);
    assert_eq!(listed["data"][0]["type"], "function_call");
    assert_eq!(listed["data"][0]["call_id"], "call_1");
    assert_eq!(listed["data"][1]["type"], "function_call_output");
    assert_eq!(listed["data"][1]["call_id"], "call_1");
    let item_id = listed["data"][0]["id"].as_str().unwrap();

    let deleted_item = restarted
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!(
                    "/v1/conversations/{conversation_id}/items/{item_id}"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted_item.status(), StatusCode::OK);

    let deleted = restarted
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/conversations/{conversation_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_route_accepts_assistant_tool_calls_and_tool_results_for_the_next_turn() {
    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "model":"not-loaded",
                        "messages":[
                            {"role":"user","content":"weather?"},
                            {"role":"assistant","content":null,"tool_calls":[{
                                "id":"call_1",
                                "type":"function",
                                "function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}
                            }]},
                            {"role":"tool","tool_call_id":"call_1","content":"{\"temp\":25}"}
                        ],
                        "tools":[{
                            "type":"function",
                            "function":{"name":"weather","parameters":{"type":"object"}}
                        }]
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_ne!(body["error"]["code"], "malformed_json");
    assert_ne!(body["error"]["code"], "invalid_message_content");
}

#[tokio::test]
async fn chat_tools_fail_closed_without_template_but_tool_choice_none_still_generates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-tool-choice.gguf");
    write_generation_gguf(&path);

    let app = camelid::api::router();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"path": path, "id": "tiny-tool-choice"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let request_body = |tool_choice: Option<&str>| {
        let mut body = json!({
            "model": "tiny-tool-choice",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": false,
            "tools": [{
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}}
            }]
        });
        if let Some(choice) = tool_choice {
            body["tool_choice"] = json!(choice);
        }
        body.to_string()
    };

    // The tiny fixture has no chat template, so a request that wants tool
    // calls fails closed instead of silently rendering without tools.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(request_body(None)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_chat_template");

    // `tool_choice: "none"` opts out of tool calling (OpenAI semantics), so
    // the same request must render as plain chat and generate.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(request_body(Some("none"))))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "<unk>");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert!(body["choices"][0]["message"].get("tool_calls").is_none());
}

#[tokio::test]
async fn metrics_exposes_prometheus_runtime_counters() {
    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let body = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(body.contains("# TYPE camelid_prompt_tokens_total counter"));
    assert!(body.contains("# TYPE camelid_engine_queue_depth gauge"));
    assert!(body.contains("# TYPE camelid_process_resident_memory_bytes gauge"));
}

#[tokio::test]
async fn slots_reports_public_fail_closed_llama_server_shape() {
    // See props_reports_public_fail_closed_llama_server_shape: pin the CPU
    // lane so slot count matches the llama-server default shape on CUDA boxes.
    std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
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
    // One entry per admissible streaming slot, so this length is what `/props`
    // reports as `total_slots` and what `fail_on_no_slot` arbitrates against.
    assert_eq!(
        slots.len(),
        camelid::runtime_config::DEFAULT_CONTINUOUS_BATCH_SLOTS
    );
    for (index, slot) in slots.iter().enumerate() {
        assert_eq!(slot["id"], index);
        assert_eq!(slot["is_processing"], false);
        assert_eq!(slot["id_task"], -1);
    }
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
        "Current exact-row support: TinyLlama Q8_0 current gate; Llama 3.2 1B Instruct Q8_0 has checked bounded 512/1024/2048/4096/8192 packs; Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with the anchored checked bounded 512/1024/2048/4096/8192 raw-decode context ladder on the current canonical GGUF (prior-upload Ubuntu API/WebUI refresh at source head e9f926ed1a65 retained as historical evidence); and Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048 packs where row-specific PASS artifacts exist. Mistral 7B Instruct v0.3 Q8_0 is supported_exact_row_smoke: checked tokenizer/template, parity (including GPU-vs-CPU greedy continuations on the exact row), bounded 512/1024/2048/4096/8192 context artifacts, and a support-promotion API/WebUI smoke bundle. LFM2.5-2.6B Q8_0 is hash-pinned supported_exact_row_smoke only on the Windows x86_64 runnable-CPU lane and one Mac16,10 / Apple M4 / macOS 26.5 arm64 resident-Metal lane after tokenizer/template and 96/96 short greedy parity, native runnable-chat receipts, a checked exact 512-token chat bucket, and the clean d31e5cb0 API/WebUI/SSE smoke; tools, sampling beyond greedy, adjacent files, broader context, CUDA, Linux, other Apple hardware or macOS revisions, and broad portability remain unclaimed. Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf has bounded one-token backend MoE runtime evidence only; later 5-token/API/WebUI/RSS promotion-candidate artifacts are superseded by Gate 9A 50-token divergence and a longer-continuation hang, so broad/API/WebUI/frontend readiness remains unsupported. The dense Qwen3 Q8_0 ChatML rows (0.6B/1.7B/4B/8B Instruct, thinking disabled) are supported_exact_row_smoke: qwen2 BPE pre-tokenizer + ChatML renderer, per-head QK-norm + NEOX RoPE, and token+text parity vs llama.cpp at 1/5/50 on macOS/Ubuntu and on Windows x86_64 CPU (cpu_reference + the x86_q8 AVX2 runtime-repack path, bit-identical), and additionally on Windows CUDA: the 0.6B/1.7B/4B rows fully VRAM-resident and the 8B row via the VRAM+host-RAM offload split (RTX 3060 Laptop 6 GB, driver 576.83, CUDA 12.9; GPU decode+single-shot prefill token+text identical to cpu_reference/llama.cpp at 1/5/50); 1.7B additionally has GPU-resident decode+prefill and a 15,373-token single-shot prefill lane on macOS, and thinking-mode is opt-in (leading-trace parity only). The 4B row additionally carries checked bounded-context packs 512/1024/2048/4096/8192, the 1.7B row 512/1024/2048/4096, and the 0.6B row 512/2048/4096/8192 (fully-GPU-resident raw-decode greedy parity vs llama.cpp acd79d603 at 50 tokens; the 1.7B 8192 and 0.6B 1024 buckets are held as documented benign near-ties). These are exact bounded lanes only; no model-native/larger context beyond the checked packs, arbitrary-template behavior, production throughput, portability, neighboring-row, or broad-family support is implied. Seven hash-pinned Prism ML Bonsai Q1_0, Prism Q2_0, and PQ2_0 artifacts are supported_exact_row_smoke on macOS Apple Silicon Metal and Windows x86_64 CUDA after paired text and vision validation; the claim is exact-file and limited to those two GPU platforms, with broader qwen3/qwen35 or quant support, bounded/model-native context, and production throughput still unclaimed. The seven files are mixed-arch: the 4B and 8B rows declare general.architecture=qwen3 (dense), only the 27B rows declare qwen35 (hybrid)."
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
        "exact Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with the anchored checked bounded 512/1024/2048/4096/8192-context raw-decode ladder on the current canonical GGUF (prior-upload Ubuntu API/WebUI refresh at source head e9f926ed1a65 retained as historical evidence)"
    ));
    assert!(q8_notes.contains(
        "exact Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048-context packs"
    ));
    assert!(q8_notes.contains("where row-specific PASS artifacts exist"));
    assert!(!q8_notes.contains("8B 1024/2048 remain red"));
    assert!(q8_notes.contains("exact LFM2.5-2.6B Q8_0 adds tokenizer/template"));
    assert!(q8_notes.contains("exact bounded-pack or checked-smoke lanes only"));
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
        "exact Llama 3.2 3B Instruct Q8_0 has supported_exact_row_smoke standing on the anchored checked bounded 512/1024/2048/4096/8192-context raw-decode ladder for the current canonical GGUF (prior-upload Ubuntu main-lane API/WebUI evidence at source head e9f926ed1a65 retained as historical)"
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
    assert!(body["supported_model_families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "mistral_instruct_exact_7b_v0_3_q8_0"
            && item["status"] == "supported_exact_row_smoke_lane"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("support-promotion API/WebUI smoke bundle")));
    assert!(!body["planned_model_families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "mistral"));
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
    assert!(body["planned_model_families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "qwen25" && item["status"] == "planned_exact_row_candidate"));
    assert!(body["planned_model_families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == "gemma2"
            && item["status"] == "active_validation_api_webui_pass_pending_context"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("verified runnable")));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "multi_choice_generation"
            && item["status"] == "supported_current_gate_nonstreaming"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("1..=8 independent")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "rich_logprobs"
            && item["status"] == "supported_current_gate_nonstreaming"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("OpenAI-shaped logprobs")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "openai_responses"
            && item["status"] == "supported_current_gate"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("opt-in local SQLite durability")
    }));
    assert!(body["api_features"].as_array().unwrap().iter().any(|item| {
        item["id"] == "streaming_tool_calls"
            && item["status"] == "supported_current_gate"
            && item["notes"].as_str().unwrap().contains("without leaking")
    }));
    let api_feature_ids = body["api_features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    let api_conformance = body["api_conformance"].as_array().unwrap();
    let conformance_ids = api_conformance
        .iter()
        .map(|item| item["id"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        api_feature_ids, conformance_ids,
        "compact feature rows and executable conformance descriptors must be projections of one registry"
    );
    let responses_contract = api_conformance
        .iter()
        .find(|item| item["id"] == "openai_responses")
        .unwrap();
    assert!(responses_contract["routes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|route| route["method"] == "POST" && route["path"] == "/v1/responses"));
    assert!(responses_contract["supported_modes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|mode| mode == "stateless_streaming"));
    assert!(responses_contract["supported_modes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|mode| mode == "previous_response_id"));
    assert!(responses_contract["routes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|route| route["method"] == "POST" && route["path"] == "/v1/conversations"));
    assert!(responses_contract["routes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|route| route["method"] == "DELETE" && route["path"] == "/v1/responses/:id"));
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
                .contains("POST /models/load is a narrow local-path alias")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("currently loaded Camelid models")
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
        item["id"] == "production_server_hardening"
            && item["status"] == "supported"
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("Prometheus /metrics")
            && item["notes"]
                .as_str()
                .unwrap()
                .contains("fail-closed non-loopback")
    }));
    let api_features = body["api_features"].as_array().unwrap();
    assert!(api_features.iter().any(|item| {
        item["id"] == "openai_embeddings"
            && item["status"] == "supported_exact_model_row"
            && item["notes"].as_str().unwrap().contains("Nomic")
    }));
    assert!(api_features.iter().any(|item| {
        item["id"] == "embedding_similarity_reranking"
            && item["status"] == "supported_exact_model_row"
            && item["notes"].as_str().unwrap().contains("bi-encoder")
    }));
    assert!(api_features.iter().any(|item| {
        let notes = item["notes"].as_str().unwrap();
        item["id"] == "fail_closed_native_compatibility_routes"
            && item["status"] == "unsupported"
            && notes.contains("/infill")
            && notes.contains("/v1/messages")
            && notes.contains("Unsupported /models/load router-mode fields")
            && notes.contains("POST /models/unload")
            && notes.contains("POST /slots")
            && !notes.contains("/v1/embeddings")
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
        "short_api_webui_smoke_with_broader_prompt_pack_parity_plus_anchored_raw_decode_512_1024_2048_4096_8192_context_ladder"
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
        "validated_anchored_raw_decode_ladder"
    );
    assert_eq!(
        llama32_3b["bounded_context_512_pack_id"],
        "llama32-3b-anchored-raw-ladder-v1"
    );
    assert_eq!(llama32_3b["bounded_context_window"], 512);
    assert_eq!(
        llama32_3b["bounded_context_1024_pack"],
        "validated_anchored_raw_decode_ladder"
    );
    assert_eq!(
        llama32_3b["bounded_context_1024_pack_id"],
        "llama32-3b-anchored-raw-ladder-v1"
    );
    assert_eq!(llama32_3b["bounded_context_1024_window"], 1024);
    assert_eq!(
        llama32_3b["bounded_context_2048_pack"],
        "validated_anchored_raw_decode_ladder"
    );
    assert_eq!(
        llama32_3b["bounded_context_2048_pack_id"],
        "llama32-3b-anchored-raw-ladder-v1"
    );
    assert_eq!(llama32_3b["bounded_context_2048_window"], 2048);
    assert_eq!(
        llama32_3b["bounded_context_4096_pack"],
        "validated_anchored_raw_decode_ladder"
    );
    assert_eq!(
        llama32_3b["bounded_context_4096_pack_id"],
        "llama32-3b-anchored-raw-ladder-v1"
    );
    assert_eq!(llama32_3b["bounded_context_4096_window"], 4096);
    assert_eq!(
        llama32_3b["bounded_context_8192_pack"],
        "validated_anchored_raw_decode_ladder"
    );
    assert_eq!(
        llama32_3b["bounded_context_8192_pack_id"],
        "llama32-3b-anchored-raw-ladder-v1"
    );
    assert_eq!(llama32_3b["bounded_context_8192_window"], 8192);
    assert_eq!(
        llama32_3b["latest_checked_bucket"],
        "llama32-3b-anchored-raw-ladder-v1"
    );
    assert_eq!(llama32_3b["latest_checked_result"], "pass");
    assert_eq!(
        llama32_3b["latest_checked_output"],
        "50/50 greedy tokens identical on all five buckets"
    );
    let llama32_3b_evidence = llama32_3b["evidence"].as_str().unwrap();
    assert!(llama32_3b_evidence.contains("sha256 f34112a1"));
    assert!(llama32_3b_evidence.contains("PRIOR upload"));
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
    assert_eq!(mistral["status"], "supported_exact_row_smoke");
    assert_eq!(mistral["support_scope"], "exact_row_smoke_only");
    assert_eq!(
        mistral["full_support_status"],
        "blocked_pending_normalized_full_support"
    );
    assert_eq!(mistral["metadata_parses"], "validated");
    assert_eq!(mistral["tokenizer_works"], "validated");
    assert_eq!(mistral["tensors_load"], "validated");
    assert_eq!(
        mistral["generation_runs"],
        "api_completion_and_chat_smoke_plus_broader_50_token_api_smoke"
    );
    assert_eq!(mistral["frontend_load_path_verified"], "validated");
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
        "support_promotion_api_webui_smoke"
    );
    assert_eq!(mistral["latest_checked_result"], "pass");
    assert_eq!(mistral["bounded_context_8192_pack"], "validated_fifth_pack");
    assert_eq!(
        mistral["bounded_context_8192_pack_id"],
        "mistral-context-8192-max-ladder-v1"
    );
    let mistral_evidence = mistral["evidence"].as_str().unwrap();
    assert!(mistral_evidence.contains("GPU-vs-CPU greedy continuations match token-for-token"));
    assert!(mistral_evidence.contains("support-promotion API/WebUI smoke bundle"));
    let mistral_next_step = mistral["next_step"].as_str().unwrap();
    assert!(mistral_next_step.contains("repeat the current-head promotion smoke"));
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
        .contains("separates forward progress from a stalled engine"));
    assert!(mixtral["evidence"]
        .as_str()
        .unwrap()
        .contains("No broad Mixtral"));

    // Qwen3-4B Q4_K_M promoted to supported exact-row smoke (GPU-resident CUDA lane).
    let qwen3_4b_q4km = compatibility
        .iter()
        .find(|item| item["id"] == "qwen3_4b_q4_k_m")
        .unwrap();
    assert_eq!(qwen3_4b_q4km["status"], "supported_exact_row_smoke");
    assert_eq!(qwen3_4b_q4km["family"], "qwen3");
    assert_eq!(qwen3_4b_q4km["quantization"], "Q4_K_M");
    // tool_capable earned via a committed agent-eval PASS receipt (Qwen3's native
    // tool template holds up at Q4_K_M; the Llama-3.2-3B-Q4_K_M sibling FAILs the battery).
    assert_eq!(qwen3_4b_q4km["tool_capable"], true);
    assert!(qwen3_4b_q4km["evidence"]
        .as_str()
        .unwrap()
        .contains("Qwen3-4B-Q4_K_M-1783378260-PASS.json"));
    assert_eq!(qwen3_4b_q4km["latest_checked_result"], "pass");
    assert!(qwen3_4b_q4km["evidence"]
        .as_str()
        .unwrap()
        .contains("all_pass=true"));
    assert!(qwen3_4b_q4km["evidence"]
        .as_str()
        .unwrap()
        .contains("qwen3-4b-q4_k_m-windows-cuda-resident-parity"));

    // Qwen3-4B Q8_0 bounded-context ladder 512/1024/2048 promoted (GPU-resident raw-decode
    // parity vs llama.cpp acd79d603). 4096/8192 now promoted too: on this 6 GiB card both
    // buckets exceed the 2090-position VRAM resident KV cap, so they decode via a
    // GPU-resident prefix (<2090) plus a CPU-fallback tail; the token stream is still
    // token-AND-text-identical to llama.cpp at 50 generated tokens.
    let qwen3_4b_q8 = compatibility
        .iter()
        .find(|item| item["id"] == "qwen3_4b_instruct_q8_0")
        .unwrap();
    assert_eq!(
        qwen3_4b_q8["bounded_context_512_pack"],
        "validated_bounded_pack"
    );
    assert_eq!(
        qwen3_4b_q8["bounded_context_1024_pack"],
        "validated_second_pack"
    );
    assert_eq!(
        qwen3_4b_q8["bounded_context_2048_pack"],
        "validated_third_pack"
    );
    assert_eq!(
        qwen3_4b_q8["bounded_context_4096_pack"],
        "validated_fourth_pack"
    );
    assert_eq!(
        qwen3_4b_q8["bounded_context_8192_pack"],
        "validated_fifth_pack"
    );
    assert!(qwen3_4b_q8["evidence"]
        .as_str()
        .unwrap()
        .contains("qwen3-4b-q8-context-512-2048"));
    assert!(qwen3_4b_q8["evidence"]
        .as_str()
        .unwrap()
        .contains("qwen3-4b-q8-context-4096-8192"));

    // Llama-3.2-3B K-quant rows promoted (GPU-resident raw-decode parity). Ids are the
    // normalized-filename form so the frontend exact-row matcher resolves them (the
    // llama-bpe branch only knows the Q8_0 quant and would quant-mismatch).
    for (id, quant) in [
        ("llama_3_2_3b_instruct_q4_k_m", "Q4_K_M"),
        ("llama_3_2_3b_instruct_q5_k_m", "Q5_K_M"),
    ] {
        let row = compatibility
            .iter()
            .find(|item| item["id"] == id)
            .unwrap_or_else(|| panic!("{id} row must be advertised"));
        assert_eq!(row["status"], "supported_exact_row_smoke");
        assert_eq!(row["family"], "llama_bpe_decoder");
        assert_eq!(row["quantization"], quant);
        assert_eq!(row["tool_capable"], false);
        assert_eq!(row["latest_checked_result"], "pass");
    }

    let qwen25_7b = compatibility
        .iter()
        .find(|item| item["id"] == "qwen25_7b_instruct_q8_0")
        .unwrap();
    assert_eq!(qwen25_7b["status"], "planned_exact_row_candidate");
    assert_eq!(qwen25_7b["generation_runs"], "not_started");
    assert_eq!(
        qwen25_7b["frontend_load_path_verified"],
        "fail_closed_planned"
    );
    assert_eq!(qwen25_7b["latest_checked_result"], "planning_only");
    assert!(qwen25_7b["evidence"]
        .as_str()
        .unwrap()
        .contains("Qwen2.5-7B-Instruct-Q8_0.gguf"));

    let gemma2_9b = compatibility
        .iter()
        .find(|item| item["id"] == "gemma2_9b_it_q8_0")
        .unwrap();
    assert_eq!(
        gemma2_9b["status"],
        "active_validation_api_webui_pass_pending_context"
    );
    assert_eq!(
        gemma2_9b["generation_runs"],
        "validated_deterministic_greedy"
    );
    assert_eq!(
        gemma2_9b["frontend_load_path_verified"],
        "validated_guarded_api_webui_smoke"
    );
    assert_eq!(
        gemma2_9b["latest_checked_result"],
        "active_validation_api_webui_pass_pending_context"
    );
    assert!(gemma2_9b["evidence"]
        .as_str()
        .unwrap()
        .contains("phase2-runtime/gemma2_9b_it_q8_0.json"));
    let planned_quant = compatibility
        .iter()
        .find(|item| item["id"] == "llama_spm_q4_k_q5_k")
        .unwrap();
    assert_eq!(
        planned_quant["status"],
        "planned_beyond_named_certified_rows"
    );
    assert_eq!(
        planned_quant["tensors_load"],
        "cpu_f32_dequant_and_wire_kernels_implemented_engine_fact_no_family_claim"
    );
    assert_eq!(
        planned_quant["generation_runs"],
        "certified_on_named_exact_rows_only_otherwise_unverified_experimental_lane"
    );
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

    // n>1 is now supported (independent choices), but not together with streaming.
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "n");
}

#[tokio::test]
async fn chat_completion_rejects_n_above_cap_before_runtime() {
    // n>1 is now supported (independent choices) up to MAX_N_CHOICES; only n above
    // the cap is rejected, before any runtime/model load.
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"n":9}"#,
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
}

#[tokio::test]
async fn completion_rejects_logprobs_above_cap_before_runtime() {
    // completions logprobs is now supported; only a value above the cap is rejected.
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"logprobs":25}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_parameter");
    assert_eq!(body["error"]["param"], "logprobs");
}

#[tokio::test]
async fn chat_completion_rejects_logprobs_with_stream() {
    // chat logprobs is now supported, but not together with streaming.
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"logprobs":true,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "logprobs");
}

#[tokio::test]
async fn chat_completion_rejects_top_logprobs_above_cap_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"logprobs":true,"top_logprobs":25}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_parameter");
    assert_eq!(body["error"]["param"], "top_logprobs");
}

#[tokio::test]
async fn chat_completion_rejects_top_logprobs_without_logprobs_before_runtime() {
    // top_logprobs still requires logprobs:true (now a typed validation error).
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
    assert_eq!(body["error"]["code"], "invalid_request_parameter");
    assert_eq!(body["error"]["param"], "top_logprobs");
}

#[tokio::test]
async fn chat_completion_accepts_tools_but_rejects_other_tool_fields() {
    // `tools` is now an accepted field (rendered into the model's own chat
    // template for agent mode), so it is NOT rejected as an unsupported
    // parameter — the request falls through to model resolution instead.
    let response = camelid::api::router()
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
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_ne!(body["error"]["code"], "unsupported_parameter");

    // tool_choice and parallel_tool_calls are now accepted too; only the
    // camelid-specific parse_tool_calls control remains unsupported.
    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"parse_tool_calls":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "parse_tool_calls");
}

#[tokio::test]
async fn chat_completion_rejects_invalid_json_schema_before_runtime() {
    // LLGuidance supports scalar roots and a broad JSON Schema surface. A schema
    // it cannot compile still fails before runtime with a typed, named error.
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"camel"}}}}"#,
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
async fn embedding_routes_validate_payload_and_require_a_loaded_encoder() {
    for uri in ["/embedding", "/embeddings", "/v1/embeddings"] {
        let response = camelid::api::router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{uri}");
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["code"], "model_not_loaded", "{uri}");
        assert_eq!(body["error"]["param"], "model", "{uri}");
    }

    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"input":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "empty_embedding_input");

    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"input":"hello","encoding_format":"base64"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_embedding_encoding");
}

#[tokio::test]
async fn reranking_routes_validate_payload_and_require_a_loaded_encoder() {
    for uri in ["/rerank", "/reranking", "/v1/rerank", "/v1/reranking"] {
        let response = camelid::api::router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"query":"camel","documents":["camelid","database"]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{uri}");
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["code"], "model_not_loaded", "{uri}");
    }

    let response = camelid::api::router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/rerank")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"query":"camel","documents":["one"],"top_n":2}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_rerank_top_n");
}

#[tokio::test]
async fn chat_completion_rejects_invalid_llguidance_grammar_before_runtime() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"grammar":"start: ("}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "grammar");
}

#[tokio::test]
async fn chat_completion_rejects_ambiguous_constraint_fields() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"response_format":{"type":"json_object"},"grammar":"start: \"ok\""}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "response_format");
}

#[tokio::test]
async fn chat_completion_rejects_json_schema_without_schema_payload() {
    // Reinstated alongside the out-of-subset case above (the json_schema PR
    // replaced this case rather than extending it): a bare
    // {"type":"json_schema"} with no json_schema.schema payload is a malformed
    // request, 400 before runtime with the param named.
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"response_format":{"type":"json_schema"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "response_format");
}

#[tokio::test]
async fn stream_true_with_json_object_is_rejected() {
    // Constrained decoding is non-streaming only: stream:true + json_object is
    // a typed 400 before runtime, never a silently unconstrained stream.
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true,"response_format":{"type":"json_object"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "response_format");
}

#[tokio::test]
async fn stream_true_with_json_schema_is_rejected() {
    // Same invariant for a supported json_schema: the constraint compiles, but
    // streaming under a constraint is rejected before runtime.
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true,"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer"}},"required":["a"]}}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "response_format");
}

#[tokio::test]
async fn completion_rejects_llama_server_sampler_fields_before_runtime() {
    let app = camelid::api::router();
    // min_p and repeat_penalty are now supported sampler fields; mirostat remains an
    // unsupported llama-server sampler control and is still rejected before runtime.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny","prompt":"hello","max_tokens":1,"mirostat":1}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "mirostat");
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
    assert_eq!(body["tokenizer"]["token_count"], 14);
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
    assert_eq!(body["token_count"], 14);
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
    // `add_space_prefix=true` unconditionally prepends a space, so the already
    // space-led " hello!" normalizes to `▁▁hello!`. `▁▁` is not in this vocab, so
    // the first `▁` stands alone (6) and the second merges into `▁hello` (3) —
    // the reference merges the dummy prefix into the following word rather than
    // leaving it as a bare piece. The previous `6,6,4,5` came from the old
    // longest-match encoder plus its `▁▁` deferral, not from the reference.
    assert_eq!(
        body["tokens"].as_array().unwrap(),
        &[
            serde_json::json!(6),
            serde_json::json!(3),
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
            // Same `▁▁hello!` segmentation as the id-only assertion above.
            serde_json::json!({"id":6,"piece":" "}),
            serde_json::json!({"id":3,"piece":" hello"}),
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
        .clone()
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

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/detokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"tokens":[4],"remove_special":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "request");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("/detokenize unsupported request field(s): remove_special"));
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
async fn responses_adapter_generates_nonstreaming_and_streaming_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-responses.gguf");
    let database = dir.path().join("responses.sqlite3");
    write_generation_gguf(&path);

    let app = camelid::api::router_with_state(
        camelid::api::AppState::default().with_responses_store_path(database.clone()),
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"path": path, "id": "tiny-responses"}).to_string(),
                ))
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
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-responses","input":"hello","max_output_tokens":1,"stream":false,"store":true,"metadata":{"test":"durable"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["model"], "tiny-responses");
    assert_eq!(body["status"], "incomplete");
    assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "<unk>");
    assert_eq!(body["usage"]["output_tokens"], 1);
    assert_eq!(body["store"], true);
    assert_eq!(body["metadata"]["test"], "durable");
    let stored_response_id = body["id"].as_str().unwrap().to_string();

    let retrieved = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/responses/{stored_response_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(retrieved.status(), StatusCode::OK);
    let retrieved: Value =
        serde_json::from_slice(&to_bytes(retrieved.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(retrieved["id"], stored_response_id);

    let continuation = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"tiny-responses",
                        "input":"continue",
                        "previous_response_id":stored_response_id,
                        "max_output_tokens":1,
                        "store":true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(continuation.status(), StatusCode::OK);
    let continuation: Value = serde_json::from_slice(
        &to_bytes(continuation.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(continuation["previous_response_id"], stored_response_id);

    let idempotent_body =
        r#"{"model":"tiny-responses","input":"once","max_output_tokens":1,"store":true}"#;
    let mut idempotent_ids = Vec::new();
    for _ in 0..2 {
        let replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .header("idempotency-key", "responses-vertical-test")
                    .body(Body::from(idempotent_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        let replay: Value =
            serde_json::from_slice(&to_bytes(replay.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        idempotent_ids.push(replay["id"].as_str().unwrap().to_string());
    }
    assert_eq!(idempotent_ids[0], idempotent_ids[1]);
    let idempotency_conflict = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .header("idempotency-key", "responses-vertical-test")
                .body(Body::from(
                    r#"{"model":"tiny-responses","input":"different","max_output_tokens":1,"store":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(idempotency_conflict.status(), StatusCode::CONFLICT);

    let conversation = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/conversations")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    let conversation: Value = serde_json::from_slice(
        &to_bytes(conversation.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let conversation_id = conversation["id"].as_str().unwrap();
    let conversation_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"tiny-responses",
                        "input":"remember this",
                        "conversation":conversation_id,
                        "max_output_tokens":1,
                        "store":false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_response.status(), StatusCode::OK);
    let conversation_response: Value = serde_json::from_slice(
        &to_bytes(conversation_response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(conversation_response["conversation"]["id"], conversation_id);
    let conversation_items = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/v1/conversations/{conversation_id}/items?order=asc"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let conversation_items: Value = serde_json::from_slice(
        &to_bytes(conversation_items.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(conversation_items["data"].as_array().unwrap().len(), 2);
    assert_eq!(conversation_items["data"][0]["role"], "user");
    assert_eq!(conversation_items["data"][1]["role"], "assistant");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-responses","input":"hello","max_output_tokens":1,"stream":true,"store":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let stream = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    for event_type in [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.incomplete",
    ] {
        assert!(
            stream.contains(&format!("\"type\":\"{event_type}\"")),
            "missing {event_type} in stream: {stream}"
        );
    }
    assert!(stream.contains("\"output_tokens\":1"));

    let terminal_response = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .find(|event| {
            matches!(
                event["type"].as_str(),
                Some("response.completed" | "response.incomplete")
            )
        })
        .unwrap();
    let streamed_response_id = terminal_response["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let restarted = camelid::api::router_with_state(
        camelid::api::AppState::default().with_responses_store_path(database),
    );
    let retrieved_after_restart = restarted
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/responses/{streamed_response_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(retrieved_after_restart.status(), StatusCode::OK);

    let deleted = restarted
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/responses/{streamed_response_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::OK);
    let missing = restarted
        .oneshot(
            Request::builder()
                .uri(format!("/v1/responses/{streamed_response_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
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
    // What this test is for is that presence_penalty / frequency_penalty /
    // logit_bias are ACCEPTED and still stream a well-formed completion. The
    // emitted token is incidental and fixture-specific: this vocab cannot build
    // `▁hello` by pairwise merge, so the prompt is a run of byte-fallback <unk>s,
    // and penalising that repeat (plus the logit_bias) makes the sampler pick EOS
    // immediately — hence an empty delta and finish_reason=stop. The sibling
    // `streaming_completion_*` test covers the unpenalised <unk> emission.
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn streaming_completion_rejects_when_prompt_fills_context_before_loading_weights() {
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
    // The prompt ("hello" -> [BOS, hello]) is exactly the 2-token context, so it
    // fills the window with no room to generate at all: rejected with param
    // "prompt". (An over-large max_tokens on a NON-filling prompt is CLAMPED, not
    // rejected -- see completion_clamps_over_limit_max_tokens_to_context below.)
    assert_eq!(body["error"]["param"], "prompt");
}

#[tokio::test]
async fn completion_clamps_over_limit_max_tokens_to_context() {
    // A response limit larger than the room left in the context is an upper bound,
    // not a demand: the request must succeed and generate up to the remaining room,
    // never reject. Regression guard for the v0.2.2 auto-fit fix.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-clamp.gguf");
    write_generation_gguf_with_options(
        &path,
        GenerationFixtureOptions {
            // 16, not 8: this fixture's 4-token vocab holds `▁hello` but none of the
            // single-step pieces SPM needs to build it, so `"hello"` byte-falls-back
            // (and `▁` is 3 UTF-8 bytes) to a 9-token prompt. A longest-match encoder
            // could jump straight to `▁hello`; the reference algorithm cannot. The
            // clamp path under test needs a context with room left over, which is
            // what this widens — the assertion below is unchanged.
            context_length: 16,
            include_tokenizer: true,
            truncate_payload: false,
        },
    );

    let app = camelid::api::router();
    let load_body = serde_json::json!({"path": path, "id": "tiny-clamp"});
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

    // Prompt "hello" -> [BOS, hello] = 2 tokens in an 8-token context, leaving room
    // for 6. max_tokens is absurdly large; it must be clamped to the room, not
    // rejected as it was before the auto-fit fix.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"tiny-clamp","prompt":"hello","max_tokens":1000000}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "an over-limit max_tokens must clamp and generate, not reject: {body}"
    );
    // 9-token prompt in a 16-token context leaves room for 7.
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(
        (1..=7).contains(&completion_tokens),
        "generation must be clamped to the room left in the context (<=7), got {completion_tokens}: {body}"
    );
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
    assert_eq!(body["error"]["code"], "invalid_gguf");
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
    // SPM reaches a token only through successive pairwise merges whose every
    // intermediate is itself in the vocab, so the single-step pieces are required
    // for `▁hello`/`hello` to be reachable at all. Appended (never reordered) so
    // the ids these tests assert stay put. Mirrors tests/tokenizer.rs.
    let tokens = [
        "<unk>", "<s>", "</s>", "▁hello", "hello", "<0x21>", "▁", "▁h", "▁he", "▁hel", "▁hell",
        "he", "hel", "hell",
    ];
    let scores = [
        0.0, 0.0, 0.0, 10.0, 2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 2.5, 2.6, 2.7,
    ];
    let token_types = [2, 3, 3, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1];

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
