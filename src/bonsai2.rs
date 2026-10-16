//! Official Bonsai 2 compact packs and their distinct Q8 vision companions.
//! Publisher identities, not a blanket parity or GPU-support claim.
use crate::api::{CatalogCompanionArtifact, CatalogItem};

pub(crate) struct Model {
    pub catalog: CatalogItem,
    pub sha256: &'static str,
    pub projector: CatalogCompanionArtifact,
    pub projector_sha256: &'static str,
}

pub(crate) const MODELS: &[Model] = &[
    Model {
        catalog: CatalogItem {
            catalog_id: "ternary_bonsai_2_2b_ptq1_0",
            name: "Bonsai 2 2B PTQ1_0",
            repo_id: "prism-ml/Ternary-Bonsai-2-2B-gguf",
            filename: "Ternary-Bonsai-2-2B-PTQ1_0.gguf",
            size_bytes: 429183456,
            downloads: 0,
            likes: 0,
            quant: "PTQ1_0",
            architecture: "qwen35",
            license: "apache-2.0",
            task_tags: &["general", "reasoning"],
        },
        sha256: "36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa",
        projector: CatalogCompanionArtifact {
            role: "vision_projector",
            repo_id: "prism-ml/Ternary-Bonsai-2-2B-gguf",
            filename: "Ternary-Bonsai-2-2B-mmproj-Q8_0.gguf",
            size_bytes: 364664352,
        },
        projector_sha256: "ac9934e8d7966d0d62442c202ec8fdad99a045dea79366c1a4a246a46f5c0533",
    },
    Model {
        catalog: CatalogItem {
            catalog_id: "ternary_bonsai_2_4b_ptq1_0",
            name: "Bonsai 2 4B PTQ1_0",
            repo_id: "prism-ml/Ternary-Bonsai-2-4B-gguf",
            filename: "Ternary-Bonsai-2-4B-PTQ1_0.gguf",
            size_bytes: 949549920,
            downloads: 0,
            likes: 0,
            quant: "PTQ1_0",
            architecture: "qwen35",
            license: "apache-2.0",
            task_tags: &["general", "reasoning", "coding"],
        },
        sha256: "56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60",
        projector: CatalogCompanionArtifact {
            role: "vision_projector",
            repo_id: "prism-ml/Ternary-Bonsai-2-4B-gguf",
            filename: "Ternary-Bonsai-2-4B-mmproj-Q8_0.gguf",
            size_bytes: 366894624,
        },
        projector_sha256: "db939d5f9b7014137810205e91d05e5f54afd72b0b3af9d33bd7d82f7906a55d",
    },
    Model {
        catalog: CatalogItem {
            catalog_id: "ternary_bonsai_2_9b_ptq1_0",
            name: "Bonsai 2 9B PTQ1_0",
            repo_id: "prism-ml/Ternary-Bonsai-2-9B-gguf",
            filename: "Ternary-Bonsai-2-9B-PTQ1_0.gguf",
            size_bytes: 1997486976,
            downloads: 0,
            likes: 0,
            quant: "PTQ1_0",
            architecture: "qwen35",
            license: "apache-2.0",
            task_tags: &["general", "reasoning", "coding"],
        },
        sha256: "3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b",
        projector: CatalogCompanionArtifact {
            role: "vision_projector",
            repo_id: "prism-ml/Ternary-Bonsai-2-9B-gguf",
            filename: "Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf",
            size_bytes: 624229664,
        },
        projector_sha256: "e0cb6f6988cea323ef798cd77d019bbb819af75582f034eaab0e7232314953db",
    },
    Model {
        catalog: CatalogItem {
            catalog_id: "ternary_bonsai_2_27b_ptq1_0",
            name: "Bonsai 2 27B PTQ1_0",
            repo_id: "prism-ml/Ternary-Bonsai-2-27B-gguf",
            filename: "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            size_bytes: 5946648928,
            downloads: 0,
            likes: 0,
            quant: "PTQ1_0",
            architecture: "qwen35",
            license: "apache-2.0",
            task_tags: &["general", "reasoning", "coding"],
        },
        sha256: "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
        projector: CatalogCompanionArtifact {
            role: "vision_projector",
            repo_id: "prism-ml/Ternary-Bonsai-2-27B-gguf",
            filename: "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
            size_bytes: 629246976,
        },
        projector_sha256: "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903",
    },
];

pub(crate) fn identity_matches(filename: &str, sha256: &str) -> bool {
    expected_sha256(filename).is_some_and(|expected| expected.eq_ignore_ascii_case(sha256))
}

pub(crate) fn expected_sha256(filename: &str) -> Option<&'static str> {
    MODELS
        .iter()
        .find(|m| m.catalog.filename == filename)
        .map(|m| m.sha256)
}

pub(crate) fn host_eligible(operating_system: &str, architecture: &str) -> bool {
    operating_system == "windows" && architecture == "x86_64"
}

pub(crate) fn supported_filename(filename: &str) -> bool {
    MODELS.iter().any(|model| {
        model.catalog.filename == filename && compatibility(model).status.starts_with("supported")
    })
}

pub(crate) fn compatibility(model: &Model) -> crate::api::ModelCompatibilityTarget {
    crate::api::ModelCompatibilityTarget {
        id: model.catalog.catalog_id, family: "bonsai2_qwen35", quantization: "PTQ1_0",
        status: "supported_windows_runtime", tool_capable: true,
        support_scope: "exact_row_windows_x86_64_text_single_image_and_tools",
        full_support_status: "supported_windows_runtime",
        full_support_blockers: "",
        metadata_parses: "PTQ1_0_and_Hadamard_v1_v2",
        tokenizer_works: "qwen2_bpe", tensors_load: "packed_weights_no_full_f32_expansion",
        generation_runs: "windows_cpu_text_single_image_and_tools",
        parity_audited: "see_bonsai2_evidence", performance_measured: "see_bonsai2_evidence",
        frontend_load_path_verified: "catalog_with_matching_Q8_0_vision_companion",
        frontend_readiness_gate: "loaded_now and generation_ready; images additionally require vision_ready and the exact projector digest",
        tested_context: "1044_prompt_tokens_context_recall", chat_template_renderer: "qwen35_chatml",
        chat_template_shape_pack: "not_promoted", chat_template_shape_pack_id: "",
        bounded_context_512_pack: "not_checked", bounded_context_512_pack_id: "", bounded_context_window: 0,
        bounded_context_1024_pack: "not_checked", bounded_context_1024_pack_id: "", bounded_context_1024_window: 0,
        bounded_context_2048_pack: "not_checked", bounded_context_2048_pack_id: "", bounded_context_2048_window: 0,
        bounded_context_4096_pack: "not_checked", bounded_context_4096_pack_id: "", bounded_context_4096_window: 0,
        bounded_context_8192_pack: "not_checked", bounded_context_8192_pack_id: "", bounded_context_8192_window: 0,
        latest_checked_bucket: "windows_api_gui", latest_checked_result: "pass",
        latest_checked_output: "qa/bonsai2/manifest.json",
        evidence: "qa/bonsai2/manifest.json; publisher model/projector hashes in src/bonsai2.rs",
        next_step: "qualify Apple Silicon, resident GPU language and extended context separately",
    }
}
