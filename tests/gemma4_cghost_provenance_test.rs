use camelid::gguf::read_metadata;
use camelid::ghost::{write_cghost_moe, GhostFile};
use camelid::model::{Gemma4Binding, LlamaModelConfig};
use camelid::tensor::TensorStore;
use std::path::PathBuf;

#[test]
fn test_force_repack_cghost_with_sha256_provenance() {
    let model_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from("/Users/timtoole/models/gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() {
        eprintln!("Model not found: {:?}", model_path);
        return;
    }

    println!("Opening GGUF model metadata: {:?}", model_path);
    let gguf = read_metadata(&model_path).expect("read gguf metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("metadata");
    let binding = Gemma4Binding::bind(&gguf, &config).expect("bind gemma4");
    let store = TensorStore::open(&model_path, &gguf);

    let moe = config.moe.as_ref().expect("has moe");
    let n_experts = moe.expert_count as usize;
    println!(
        "GGUF loaded: architecture={}, n_experts={}",
        config.architecture, n_experts
    );

    let mut needs_repack = true;
    if cghost_path.is_file() {
        if let Ok(ghost) = GhostFile::open(&cghost_path) {
            // Only accept if source_identity is present AND matches
            if ghost.has_sampled_source_identity() {
                if ghost
                    .validate_moe_source_identity(&model_path, &binding, n_experts)
                    .is_ok()
                {
                    println!(
                        "[PROVENANCE OK] .cghost has full SHA-256 identity matching source GGUF"
                    );
                    needs_repack = false;
                }
            } else {
                println!("[LEGACY DETECTED] .cghost is legacy v2 without cryptographic source identity. Repacking...");
            }
        }
    }

    if needs_repack {
        println!(
            ">>> Starting fresh .cghost repack directly from {:?}",
            model_path
        );
        let temp_cghost = cghost_path.with_extension("cghost.repack.part");
        let source_name = model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let t0 = std::time::Instant::now();
        write_cghost_moe(&store, &binding, &config, &source_name, &temp_cghost, None)
            .expect("write_cghost_moe");
        println!(
            "Repack write completed in {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        println!("Validating newly created .cghost cryptographic SHA-256 identity...");
        let ghost = GhostFile::open(&temp_cghost).expect("open new ghost");
        assert!(
            ghost.has_sampled_source_identity(),
            "new ghost must embed source identity"
        );
        ghost
            .validate_moe_source_identity(&model_path, &binding, n_experts)
            .expect("validate new ghost identity");

        std::fs::rename(&temp_cghost, &cghost_path).expect("rename to final cghost");
        println!(
            "[SUCCESS] Installed fresh verified .cghost at {:?}",
            cghost_path
        );
    }
}
