//! Real-artifact evidence gate for the exact admitted Nomic embedding row.
//!
//! The fixture is intentionally not committed. Download the SHA-pinned model to
//! `target/embedding-fixtures/nomic-embed-text-v1.5.Q8_0.gguf`, then run:
//! `cargo test --test embedding_real_model -- --ignored --nocapture`.

use std::path::PathBuf;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use camelid::embedding::{cosine_similarity, NomicBertRuntime};
use camelid::gguf::read_metadata;
use camelid::tokenizer::Tokenizer;
use serde_json::{json, Value};
use tower::ServiceExt;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("embedding-fixtures")
        .join("nomic-embed-text-v1.5.Q8_0.gguf")
}

async fn post_json(app: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    (status, body)
}

#[test]
#[ignore = "requires the SHA-pinned 146 MB real GGUF fixture"]
fn nomic_wordpiece_tokenization_matches_llama_cpp_oracle() {
    let gguf = read_metadata(fixture_path()).expect("read exact fixture");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("load BERT WordPiece tokenizer");
    let text = "search_query: Which animals store fat in their humps?";
    for id in [100, 3945, 1035, 23032, 1024, 2029, 29638, 29627] {
        println!("token[{id}]={:?}", tokenizer.token_text(Some(id)));
    }
    assert_eq!(
        tokenizer.encode(text, true, false).unwrap(),
        vec![
            101, 3945, 1035, 23032, 1024, 2029, 4176, 3573, 6638, 1999, 2037, 14910, 4523, 1029,
            102,
        ]
    );
}

#[test]
#[ignore = "requires the SHA-pinned 146 MB real GGUF fixture"]
fn nomic_q8_real_artifact_is_deterministic_normalized_and_semantic() {
    let path = fixture_path();
    assert!(
        path.is_file(),
        "download the exact evidence fixture to {}",
        path.display()
    );

    let load_started = Instant::now();
    let runtime = NomicBertRuntime::load(&path).expect("load exact Nomic-BERT Q8 runtime");
    println!(
        "load_ms={} config={:?}",
        load_started.elapsed().as_millis(),
        runtime.config()
    );
    assert_eq!(runtime.config().architecture, "nomic-bert");
    assert_eq!(runtime.config().embedding_length, 768);
    assert_eq!(runtime.config().block_count, 12);
    assert_eq!(runtime.config().head_count, 12);
    assert_eq!(runtime.config().pooling.as_str(), "mean");

    let prompts = [
        "search_query: Which animals store fat in their humps?",
        "search_document: Camels and other camelids use their humps to store fat.",
        "search_document: A database index accelerates structured record lookups.",
        "search_query: Which animals store fat in their humps?",
    ];
    for prompt in &prompts[..3] {
        println!(
            "tokens={:?}",
            runtime
                .tokenizer()
                .encode(prompt, true, false)
                .expect("tokenize probe")
        );
    }
    for id in [100, 3945, 1035, 23032, 1024, 2029, 29638, 29627] {
        println!("token[{id}]={:?}", runtime.tokenizer().token_text(Some(id)));
    }
    let run_started = Instant::now();
    let embeddings = runtime
        .embed_batch(
            &prompts
                .iter()
                .map(|text| text.to_string())
                .collect::<Vec<_>>(),
            None,
        )
        .expect("embed semantic probe pack");
    println!(
        "embed_ms={} vectors={} dimensions={}",
        run_started.elapsed().as_millis(),
        embeddings.len(),
        embeddings[0].len()
    );

    for embedding in &embeddings {
        assert_eq!(embedding.len(), 768);
        assert!(embedding.iter().all(|value| value.is_finite()));
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "L2 norm was {norm}");
    }
    assert_eq!(
        embeddings[0], embeddings[3],
        "same input must be bit-deterministic"
    );
    let relevant = cosine_similarity(&embeddings[0], &embeddings[1]).unwrap();
    let irrelevant = cosine_similarity(&embeddings[0], &embeddings[2]).unwrap();
    println!("relevant={relevant:.6} irrelevant={irrelevant:.6}");
    assert!(
        relevant > irrelevant + 0.15,
        "semantic ordering margin was too small: relevant={relevant}, irrelevant={irrelevant}"
    );

    let oracle_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("llama-reference")
        .join("oracle-embeddings.json");
    if oracle_path.is_file() {
        let oracle_raw = std::fs::read_to_string(&oracle_path).unwrap();
        let oracle: serde_json::Value =
            serde_json::from_str(oracle_raw.trim_start_matches('\u{feff}')).unwrap();
        for (index, camelid) in embeddings.iter().take(3).enumerate() {
            let reference = oracle["data"][index]["embedding"]
                .as_array()
                .expect("llama.cpp oracle embedding")
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            let parity = cosine_similarity(camelid, &reference).unwrap();
            let max_abs_delta = camelid
                .iter()
                .zip(&reference)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            println!("oracle_vector[{index}] cosine={parity:.9} max_abs_delta={max_abs_delta:.9}");
            assert!(
                parity > 0.999_7,
                "Camelid/reference vector cosine was {parity}"
            );
            assert!(
                max_abs_delta < 0.003,
                "Camelid/reference max absolute vector delta was {max_abs_delta}"
            );
        }
    }

    let short = runtime
        .embed("search_query: camel humps", Some(256))
        .expect("Matryoshka 256-d embedding");
    assert_eq!(short.len(), 256);
    let short_norm = short.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((short_norm - 1.0).abs() < 1e-4);
}

#[tokio::test]
#[ignore = "requires the SHA-pinned 146 MB real GGUF fixture"]
async fn nomic_real_artifact_loads_through_http_and_serves_embeddings_and_rerank() {
    let path = fixture_path();
    let model_id = "nomic-embed-text-v1.5.Q8_0.gguf";
    let app = camelid::api::router();
    let (status, loaded) = post_json(
        app.clone(),
        "/api/models/load",
        json!({
            "path": path,
            "id": model_id,
            "replace": false,
            "set_active": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{loaded}");

    let (status, response) = post_json(
        app.clone(),
        "/v1/embeddings",
        json!({
            "model": model_id,
            "input": [
                "search_query: Which animals store fat in their humps?",
                "search_document: Camels and other camelids use their humps to store fat."
            ],
            "dimensions": 256
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["object"], "list");
    assert_eq!(response["model"], model_id);
    assert_eq!(
        response["data"][0]["embedding"].as_array().unwrap().len(),
        256
    );
    assert_eq!(response["usage"]["prompt_tokens"], 35);

    let (status, response) = post_json(
        app,
        "/v1/rerank",
        json!({
            "model": model_id,
            "query": "Which animals store fat in their humps?",
            "documents": [
                "A database index accelerates structured record lookups.",
                {"text": "Camels and other camelids use their humps to store fat."}
            ],
            "top_n": 2,
            "return_documents": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["results"][0]["index"], 1);
    assert!(
        response["results"][0]["relevance_score"].as_f64().unwrap()
            > response["results"][1]["relevance_score"].as_f64().unwrap()
    );
}
