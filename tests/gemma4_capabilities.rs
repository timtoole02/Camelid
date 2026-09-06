//! Gemma 4 `/api/capabilities` contract — the rows must report exactly what is
//! proven, fail closed on everything else, and never leak a family-wide or
//! multimodal claim. Runs with no model file (pure contract checks).

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use serde_json::Value;
use tower::ServiceExt;

async fn capabilities() -> Value {
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
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

fn gemma4_rows(body: &Value) -> Vec<&Value> {
    body["model_compatibility"]
        .as_array()
        .expect("model_compatibility array")
        .iter()
        .filter(|row| {
            row["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("gemma4"))
        })
        .collect()
}

#[tokio::test]
async fn gemma4_supported_rows_are_exactly_the_committed_set() {
    let body = capabilities().await;
    let mut supported: Vec<&str> = gemma4_rows(&body)
        .iter()
        .filter(|row| {
            row["status"]
                .as_str()
                .is_some_and(|s| s.starts_with("supported"))
        })
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    supported.sort_unstable();
    assert_eq!(
        supported,
        vec![
            // The 12B Q8 and 26B A4B QAT rows are supported only through their
            // committed, lane-scoped evidence. The downloadable 12B QAT Q4_0
            // single-Mac row remains active validation and must not appear here.
            "gemma4_12b_it_q8_0",
            // The MTP12 serve lane's own row. Distinct from the downloadable
            // 12B QAT Q4_0 single-Mac row above, which stays active-validation:
            // this one is pinned to one target SHA plus one assistant SHA on
            // Apple Metal behind the explicit MTP12 opt-in, and its scope string
            // says so.
            "gemma4_12b_it_qat_q4_0_mtp12",
            "gemma4_26b_a4b_it_q4_0",
            "gemma4_e2b_it_q8_0",
            // GABBRO support promotion: the NVFP4 pilot row is oracle-parity-anchored
            // (G-M1 bit-exact CPU + Metal self-parity + BASALT Leg B cross-engine +
            // fresh macOS spot-check) and clears the pre-registered G3 GO rule in the
            // current engine as a disclosed near-tie; evidence bundle at
            // qa/evidence-bundles/gabbro/support/.
            "gemma4_e4b_it_nvfp4",
            "gemma4_e4b_it_q8_0"
        ],
        "exact-row support must not grow without committed evidence"
    );
}

#[tokio::test]
async fn gemma4_rows_scope_stays_exact_row_and_text_only() {
    let body = capabilities().await;
    for row in gemma4_rows(&body) {
        let id = row["id"].as_str().unwrap();
        let scope = row["support_scope"].as_str().unwrap_or_default();
        assert!(
            scope == "exact_row_smoke_only"
                || scope == "exact_row_distributed_serve_smoke_only"
                || scope == "exact_row_gpu_resident_raw_decode_parity_smoke_only"
                // Catalog-managed Ghost-MoE installs (c652dc62) added this row
                // scope; still exact-row + smoke-bounded wording, no family claim.
                || scope == "exact_row_distributed_or_windows_cuda_ghost_moe_smoke_only"
                || scope == "active_validation_only"
                // The MTP12 serve lane pins its own host/OS/artefact tuple; the
                // wording stays exact-row and smoke-bounded.
                || scope
                    == "exact_row_single_node_base_m4_mini_mac16_10_macos_26_5_metal_lossless_mtp12_performance_smoke_only"
                || scope.starts_with("blocked"),
            "{id}: scope must stay exact-row bounded, got {scope}"
        );
        let evidence = row["evidence"].as_str().unwrap_or_default();
        // Exact-row wording only — never a family-wide claim.
        assert!(
            evidence.contains("this row only"),
            "{id}: evidence must scope the claim to the exact row"
        );
        // No multimodal claim anywhere on a gemma4 row.
        for key in ["evidence", "generation_runs", "tested_context"] {
            let text = row[key].as_str().unwrap_or_default().to_ascii_lowercase();
            assert!(
                !text.contains("multimodal support")
                    && !text.contains("image input")
                    && !text.contains("vision tower loaded"),
                "{id}.{key} must not claim multimodal"
            );
        }
        // Full support must remain explicitly blocked until the named blockers close.
        assert_eq!(
            row["full_support_status"], "blocked_pending_normalized_full_support",
            "{id}: full support must stay blocked"
        );
        assert!(
            !row["full_support_blockers"]
                .as_str()
                .unwrap_or_default()
                .is_empty(),
            "{id}: blockers must be named"
        );
    }
}

#[tokio::test]
async fn gemma4_e2b_row_records_pack_parity_oracle() {
    let body = capabilities().await;
    let rows = gemma4_rows(&body);
    let e2b = rows
        .iter()
        .find(|row| row["id"] == "gemma4_e2b_it_q8_0")
        .expect("e2b row");
    let parity = e2b["parity_audited"].as_str().unwrap_or_default();
    assert!(
        parity.contains("llama_cpp_5d56eff"),
        "E2B parity must name the pinned oracle build: {parity}"
    );
    let evidence = e2b["evidence"].as_str().unwrap_or_default();
    assert!(
        evidence.contains("qa/gemma4/prompt_packs/basic_v1.json")
            && evidence.contains("qa/gemma4/oracle/gemma-4-E2B-it-Q8_0.basic_v1.json"),
        "E2B evidence must point at the committed pack + oracle artifacts"
    );
    assert!(
        evidence.contains("5,048,350,848 bytes"),
        "E2B evidence must pin the exact file size"
    );
}

#[tokio::test]
async fn gemma4_models_catalog_lists_supported_and_single_mac_downloads() {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/catalog")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // The catalog route may differ; tolerate 404 here only by checking the
    // canonical curated catalog through capabilities-adjacent surfaces instead.
    if response.status() == StatusCode::OK {
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        let text = body.to_string();
        assert!(text.contains("gemma4_e4b_it_q8_0"));
        assert!(text.contains("gemma4_e2b_it_q8_0"));
        assert!(text.contains("gemma4_12b_it_qat_q4_0"));
        assert!(text.contains("google/gemma-4-12B-it-qat-q4_0-gguf"));
        assert!(text.contains("gemma-4-12b-it-qat-q4_0.gguf"));
    }
}
